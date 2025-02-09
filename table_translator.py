from typing import List, Dict, Union, Optional
import logging
from datetime import datetime
from google.cloud import translate_v2 as translate
from google.cloud.exceptions import GoogleCloudError
import pandas as pd
import json
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class TableTranslator:
    """
    A class for translating table data while preserving structure.
    Created by: kaxm23
    Created on: 2025-02-09 08:26:56 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 source_lang: str = 'en',
                 target_lang: str = 'ar',
                 max_workers: int = 5,
                 requests_per_minute: int = 60,
                 log_level: int = logging.INFO):
        """
        Initialize the table translator.
        
        Args:
            api_key: Google Cloud API key
            source_lang: Source language code
            target_lang: Target language code
            max_workers: Maximum number of concurrent translation workers
            requests_per_minute: Maximum API requests per minute
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize translation client
        try:
            self.client = translate.Client(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Failed to initialize translation client: {str(e)}")
            raise
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_workers = max_workers
        self.requests_per_minute = requests_per_minute
        
        # Statistics tracking
        self.stats = {
            'cells_translated': 0,
            'characters_translated': 0,
            'failed_translations': 0,
            'processing_time': None,
            'tables_processed': 0
        }

    @sleep_and_retry
    @limits(calls=60, period=60)
    def _translate_cell(self, 
                       cell_content: Union[str, int, float],
                       preserve_format: bool = True) -> str:
        """
        Translate a single cell with rate limiting.
        
        Args:
            cell_content: Content to translate
            preserve_format: Whether to preserve number formatting
            
        Returns:
            str: Translated content
        """
        try:
            # Handle non-string content
            if not isinstance(cell_content, str):
                if preserve_format:
                    return cell_content
                cell_content = str(cell_content)
            
            # Skip empty cells
            if not cell_content.strip():
                return cell_content
            
            # Translate content
            result = self.client.translate(
                cell_content,
                target_language=self.target_lang,
                source_language=self.source_lang
            )
            
            # Update statistics
            self.stats['cells_translated'] += 1
            self.stats['characters_translated'] += len(cell_content)
            
            return result['translatedText']
            
        except Exception as e:
            self.stats['failed_translations'] += 1
            self.logger.error(f"Cell translation failed: {str(e)}")
            return f"[Translation Error: {str(e)}]"

    def _translate_table(self,
                        table: Union[List[List], pd.DataFrame, Dict],
                        preserve_format: bool = True,
                        show_progress: bool = True) -> Union[List[List], pd.DataFrame, Dict]:
        """
        Translate a single table.
        
        Args:
            table: Table to translate
            preserve_format: Whether to preserve number formatting
            show_progress: Whether to show progress bar
            
        Returns:
            Translated table in the same format as input
        """
        try:
            # Convert table to DataFrame if needed
            if isinstance(table, list):
                df = pd.DataFrame(table)
            elif isinstance(table, dict):
                df = pd.DataFrame.from_dict(table)
            else:
                df = table.copy()
            
            # Prepare cell translation tasks
            translation_tasks = []
            cell_positions = []
            
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    cell_content = df.iloc[i, j]
                    if pd.notna(cell_content):  # Skip NaN/None values
                        translation_tasks.append(cell_content)
                        cell_positions.append((i, j))
            
            # Translate cells in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                if show_progress:
                    translated_cells = list(tqdm(
                        executor.map(
                            lambda x: self._translate_cell(x, preserve_format),
                            translation_tasks
                        ),
                        total=len(translation_tasks),
                        desc="Translating cells"
                    ))
                else:
                    translated_cells = list(executor.map(
                        lambda x: self._translate_cell(x, preserve_format),
                        translation_tasks
                    ))
            
            # Update table with translations
            for (i, j), translated_content in zip(cell_positions, translated_cells):
                df.iloc[i, j] = translated_content
            
            # Convert back to original format
            if isinstance(table, list):
                return df.values.tolist()
            elif isinstance(table, dict):
                return df.to_dict()
            else:
                return df
            
        except Exception as e:
            self.logger.error(f"Table translation failed: {str(e)}")
            raise

    def translate_tables(self,
                        tables: List[Dict],
                        preserve_format: bool = True,
                        show_progress: bool = True) -> Dict:
        """
        Translate multiple tables while preserving structure.
        
        Args:
            tables: List of table dictionaries
            preserve_format: Whether to preserve number formatting
            show_progress: Whether to show progress bar
            
        Returns:
            Dict: Translation results and statistics
        """
        start_time = datetime.now()
        
        result = {
            'status': 'success',
            'translated_tables': [],
            'statistics': {},
            'error': None,
            'metadata': {
                'source_language': self.source_lang,
                'target_language': self.target_lang,
                'translator': 'kaxm23',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        }
        
        try:
            # Process each table
            for table_info in tables:
                translated_table = self._translate_table(
                    table_info['data'],
                    preserve_format,
                    show_progress
                )
                
                # Update table info with translation
                translated_info = table_info.copy()
                translated_info['data'] = translated_table
                translated_info['translated'] = True
                translated_info['source_language'] = self.source_lang
                translated_info['target_language'] = self.target_lang
                
                result['translated_tables'].append(translated_info)
                self.stats['tables_processed'] += 1
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"Translation failed: {str(e)}")
            return result
            
        finally:
            # Calculate processing time and update statistics
            self.stats['processing_time'] = \
                (datetime.now() - start_time).total_seconds()
            
            result['statistics'] = {
                'tables_processed': self.stats['tables_processed'],
                'cells_translated': self.stats['cells_translated'],
                'characters_translated': self.stats['characters_translated'],
                'failed_translations': self.stats['failed_translations'],
                'processing_time': self.stats['processing_time'],
                'translation_rate': (
                    self.stats['characters_translated'] / self.stats['processing_time']
                    if self.stats['processing_time'] > 0 else 0
                )
            }
        
        return result

    def save_translated_tables(self,
                             translation_result: Dict,
                             output_path: str,
                             format: str = 'json') -> None:
        """
        Save translated tables to file.
        
        Args:
            translation_result: Translation results dictionary
            output_path: Path to save the output
            format: Output format ('json', 'csv', or 'excel')
        """
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(translation_result, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                # Save each table as separate CSV file
                base_path = Path(output_path).stem
                for i, table_info in enumerate(translation_result['translated_tables'], 1):
                    if isinstance(table_info['data'], pd.DataFrame):
                        df = table_info['data']
                    else:
                        df = pd.DataFrame(table_info['data'])
                    
                    csv_path = f"{base_path}_table_{i}.csv"
                    df.to_csv(csv_path, index=False)
                    
            elif format == 'excel':
                # Save all tables to different sheets in Excel
                writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
                
                # Add metadata sheet
                pd.DataFrame([translation_result['metadata']]).to_excel(
                    writer,
                    sheet_name='Metadata',
                    index=False
                )
                
                # Add statistics sheet
                pd.DataFrame([translation_result['statistics']]).to_excel(
                    writer,
                    sheet_name='Statistics',
                    index=False
                )
                
                # Add tables
                for i, table_info in enumerate(translation_result['translated_tables'], 1):
                    if isinstance(table_info['data'], pd.DataFrame):
                        df = table_info['data']
                    else:
                        df = pd.DataFrame(table_info['data'])
                    
                    sheet_name = f"Table_{i}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                writer.save()
                
            else:
                raise ValueError(f"Unsupported output format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to save translated tables: {str(e)}")
            raise