from fpdf import FPDF, XPos, YPos
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

class ArabicPDFWriter:
    """
    A class to create PDF files with proper Arabic text rendering.
    Created by: kaxm23
    Created on: 2025-02-09 08:12:28 UTC
    """
    
    def __init__(self,
                 font_size: int = 12,
                 margin: int = 20,
                 log_level: int = logging.INFO):
        """
        Initialize the PDF writer with Arabic support.
        
        Args:
            font_size (int): Default font size (default: 12)
            margin (int): Page margin in mm (default: 20)
            log_level (int): Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.font_size = font_size
        self.margin = margin
        
        # Initialize PDF settings
        self.pdf = FPDF()
        self._setup_pdf()
        
        # Track statistics
        self.stats = {
            'pages_created': 0,
            'characters_written': 0,
            'processing_start': None,
            'processing_end': None
        }

    def _setup_pdf(self):
        """Configure PDF settings for Arabic support."""
        try:
            # Add Arabic font
            self.pdf.add_font('Arial', '', 'arial.ttf', uni=True)
            self.pdf.add_font('Arial', 'B', 'arialbd.ttf', uni=True)
            
            # Set default font
            self.pdf.set_font('Arial', '', self.font_size)
            
            # Set margins
            self.pdf.set_margins(self.margin, self.margin, self.margin)
            
            # Set auto page break
            self.pdf.set_auto_page_break(True, margin=self.margin)
            
        except Exception as e:
            self.logger.error(f"Failed to setup PDF: {str(e)}")
            raise

    def _prepare_arabic_text(self, text: str) -> str:
        """
        Prepare Arabic text for PDF rendering.
        
        Args:
            text (str): Text to prepare
            
        Returns:
            str: Prepared text
        """
        try:
            reshaped_text = reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            self.logger.warning(f"Error preparing Arabic text: {str(e)}")
            return text

    def create_translated_pdf(self,
                            translations: List[Dict[str, str]],
                            output_path: str,
                            metadata: Optional[Dict] = None,
                            include_original: bool = True,
                            page_numbers: bool = True) -> None:
        """
        Create a PDF file with translated text.
        
        Args:
            translations (List[Dict[str, str]]): List of translation dictionaries
            output_path (str): Path to save the PDF file
            metadata (Dict, optional): PDF metadata
            include_original (bool): Include original text (default: True)
            page_numbers (bool): Include page numbers (default: True)
        """
        self.stats['processing_start'] = datetime.now()
        
        try:
            # Set metadata if provided
            if metadata:
                self.pdf.set_title(metadata.get('title', 'Translated Document'))
                self.pdf.set_author(metadata.get('author', 'PDF Translator'))
                self.pdf.set_creator(metadata.get('creator', f'ArabicPDFWriter by {metadata.get("translator", "Unknown")}'))
                self.pdf.set_subject(metadata.get('subject', 'Translated Document'))
            
            # Process each translation
            for page_num, translation in enumerate(translations, 1):
                self._add_translation_page(
                    translation,
                    page_num,
                    include_original,
                    page_numbers
                )
                
                self.stats['pages_created'] += 1
                self.stats['characters_written'] += len(translation['translated'])
                if include_original:
                    self.stats['characters_written'] += len(translation['original'])
            
            # Save the PDF
            self.pdf.output(output_path)
            
            self.stats['processing_end'] = datetime.now()
            self._log_statistics()
            
        except Exception as e:
            self.logger.error(f"Failed to create PDF: {str(e)}")
            raise

    def _add_translation_page(self,
                            translation: Dict[str, str],
                            page_num: int,
                            include_original: bool,
                            page_numbers: bool) -> None:
        """
        Add a page with translated text to the PDF.
        
        Args:
            translation (Dict[str, str]): Translation dictionary
            page_num (int): Page number
            include_original (bool): Include original text
            page_numbers (bool): Include page numbers
        """
        self.pdf.add_page()
        
        # Set up initial position
        y_position = self.margin
        
        # Add page number if enabled
        if page_numbers:
            self.pdf.set_font('Arial', 'B', 8)
            self.pdf.set_y(-15)
            self.pdf.cell(0, 10, f'Page {page_num}', align='C')
            self.pdf.set_y(y_position)
        
        # Add original text if enabled
        if include_original:
            self.pdf.set_font('Arial', 'B', self.font_size)
            self.pdf.cell(0, 10, 'Original Text:', ln=True)
            self.pdf.set_font('Arial', '', self.font_size)
            
            # Write original text with word wrapping
            self.pdf.multi_cell(0, 10, translation['original'])
            self.pdf.ln(10)
        
        # Add translated text
        self.pdf.set_font('Arial', 'B', self.font_size)
        self.pdf.cell(0, 10, 'Arabic Translation:', ln=True)
        self.pdf.set_font('Arial', '', self.font_size)
        
        # Prepare and write Arabic text
        arabic_text = self._prepare_arabic_text(translation['translated'])
        self.pdf.multi_cell(0, 10, arabic_text)

    def _log_statistics(self) -> None:
        """Log PDF creation statistics."""
        if self.stats['processing_start'] and self.stats['processing_end']:
            processing_time = (self.stats['processing_end'] - self.stats['processing_start']).total_seconds()
            
            self.logger.info(f"""
PDF Creation Statistics:
----------------------
Pages Created: {self.stats['pages_created']}
Characters Written: {self.stats['characters_written']:,}
Processing Time: {processing_time:.2f} seconds
Average Speed: {self.stats['characters_written'] / processing_time:.2f} characters/second
            """)

    def save_statistics(self, output_path: str) -> None:
        """
        Save statistics to a JSON file.
        
        Args:
            output_path (str): Path to save the statistics
        """
        try:
            stats_dict = {
                **self.stats,
                'processing_start': self.stats['processing_start'].isoformat() if self.stats['processing_start'] else None,
                'processing_end': self.stats['processing_end'].isoformat() if self.stats['processing_end'] else None
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Failed to save statistics: {str(e)}")
            raise