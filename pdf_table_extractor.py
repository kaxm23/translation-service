import pdfplumber
import pandas as pd
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import json

class PDFTableExtractor:
    """
    A class for extracting tables from PDF documents using pdfplumber.
    Created by: kaxm23
    Created on: 2025-02-09 08:24:52 UTC
    """
    
    def __init__(self, 
                 table_settings: Optional[Dict] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the PDF table extractor.
        
        Args:
            table_settings (Dict, optional): Custom table extraction settings
            log_level (int): Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Default table extraction settings
        self.table_settings = table_settings or {
            'vertical_strategy': 'text',
            'horizontal_strategy': 'text',
            'intersection_x_tolerance': 3,
            'intersection_y_tolerance': 3,
            'snap_x_tolerance': 3,
            'snap_y_tolerance': 3,
            'join_tolerance': 3,
            'edge_min_length': 3,
            'min_cells': 4
        }
        
        # Statistics tracking
        self.stats = {
            'total_pages': 0,
            'pages_with_tables': 0,
            'total_tables': 0,
            'total_cells': 0,
            'empty_cells': 0,
            'processing_time': None,
            'tables_by_page': {}
        }

    def extract_tables(self,
                      pdf_path: str,
                      pages: Optional[List[int]] = None,
                      output_format: str = 'list') -> Dict:
        """
        Extract tables from PDF.
        
        Args:
            pdf_path (str): Path to PDF file
            pages (List[int], optional): Specific pages to process
            output_format (str): Output format ('list', 'dict', or 'dataframe')
            
        Returns:
            Dict: Dictionary containing extracted tables and metadata
        """
        start_time = datetime.now()
        
        result = {
            'status': 'success',
            'tables': [],
            'metadata': {},
            'statistics': {},
            'error': None
        }
        
        try:
            # Validate PDF file
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Open PDF
            with pdfplumber.open(pdf_path) as pdf:
                self.stats['total_pages'] = len(pdf.pages)
                
                # Get metadata
                result['metadata'] = {
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                    'producer': pdf.metadata.get('Producer', ''),
                    'total_pages': len(pdf.pages),
                    'processed_by': 'kaxm23',
                    'processed_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                }
                
                # Determine pages to process
                pages_to_process = pages if pages is not None else range(len(pdf.pages))
                
                # Extract tables from each page
                for page_num in pages_to_process:
                    try:
                        page_tables = self._extract_page_tables(
                            pdf.pages[page_num],
                            page_num,
                            output_format
                        )
                        
                        if page_tables:
                            result['tables'].extend(page_tables)
                            self.stats['pages_with_tables'] += 1
                            self.stats['tables_by_page'][page_num] = len(page_tables)
                            
                    except Exception as e:
                        self.logger.error(f"Error processing page {page_num}: {str(e)}")
                        continue
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"Error processing PDF: {str(e)}")
            return result
            
        finally:
            # Calculate processing time and update statistics
            self.stats['processing_time'] = \
                (datetime.now() - start_time).total_seconds()
            
            result['statistics'] = self._get_statistics()
        
        return result

    def _extract_page_tables(self,
                           page: 'pdfplumber.Page',
                           page_num: int,
                           output_format: str) -> List:
        """
        Extract tables from a single page.
        
        Args:
            page: PDF page object
            page_num: Page number
            output_format: Desired output format
            
        Returns:
            List: List of extracted tables
        """
        page_tables = []
        
        # Extract tables using settings
        tables = page.extract_tables(table_settings=self.table_settings)
        
        for table_num, table in enumerate(tables, 1):
            if not table:
                continue
                
            self.stats['total_tables'] += 1
            
            # Process table based on output format
            if output_format == 'dataframe':
                # Convert to DataFrame
                df = pd.DataFrame(table)
                # Use first row as header if it contains strings
                if df.iloc[0].apply(lambda x: isinstance(x, str)).all():
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                processed_table = df
                
            elif output_format == 'dict':
                # Convert to dictionary with header row as keys
                headers = [str(cell).strip() if cell else f'Column_{i}'
                         for i, cell in enumerate(table[0])]
                processed_table = []
                for row in table[1:]:
                    row_dict = {}
                    for i, cell in enumerate(row):
                        key = headers[i]
                        row_dict[key] = cell.strip() if isinstance(cell, str) else cell
                    processed_table.append(row_dict)
                    
            else:  # 'list' format
                processed_table = table
            
            # Count cells and empty cells
            total_cells = sum(len(row) for row in table)
            empty_cells = sum(
                1 for row in table for cell in row 
                if cell is None or str(cell).strip() == ''
            )
            
            self.stats['total_cells'] += total_cells
            self.stats['empty_cells'] += empty_cells
            
            # Add table metadata
            table_info = {
                'page_number': page_num,
                'table_number': table_num,
                'rows': len(table),
                'columns': len(table[0]) if table else 0,
                'data': processed_table
            }
            
            page_tables.append(table_info)
            
        return page_tables

    def _get_statistics(self) -> Dict:
        """
        Get extraction statistics.
        
        Returns:
            Dict: Extraction statistics
        """
        return {
            'total_pages': self.stats['total_pages'],
            'pages_with_tables': self.stats['pages_with_tables'],
            'total_tables': self.stats['total_tables'],
            'total_cells': self.stats['total_cells'],
            'empty_cells': self.stats['empty_cells'],
            'cell_fill_rate': (
                (self.stats['total_cells'] - self.stats['empty_cells']) /
                self.stats['total_cells'] if self.stats['total_cells'] > 0 else 0
            ),
            'tables_by_page': self.stats['tables_by_page'],
            'processing_time': self.stats['processing_time']
        }

    def save_tables(self,
                   extraction_result: Dict,
                   output_path: str,
                   format: str = 'json') -> None:
        """
        Save extracted tables to file.
        
        Args:
            extraction_result: Dictionary containing extracted tables
            output_path: Path to save the output
            format: Output format ('json', 'csv', or 'excel')
        """
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(extraction_result, f, indent=2, ensure_ascii=False)
                    
            elif format == 'csv':
                # Save each table as separate CSV file
                base_path = Path(output_path).stem
                for i, table_info in enumerate(extraction_result['tables'], 1):
                    if isinstance(table_info['data'], pd.DataFrame):
                        df = table_info['data']
                    else:
                        df = pd.DataFrame(table_info['data'])
                    
                    csv_path = f"{base_path}_table_{i}.csv"
                    df.to_csv(csv_path, index=False)
                    
            elif format == 'excel':
                # Save all tables to different sheets in Excel
                writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
                
                for i, table_info in enumerate(extraction_result['tables'], 1):
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
            self.logger.error(f"Failed to save tables: {str(e)}")
            raise