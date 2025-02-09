import fitz
import json
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

class TextStyle(Enum):
    NORMAL = "normal"
    BOLD = "bold"
    ITALIC = "italic"
    BOLD_ITALIC = "bold-italic"
    HEADING1 = "h1"
    HEADING2 = "h2"
    HEADING3 = "h3"

@dataclass
class TextBlock:
    """Represents a block of text with formatting."""
    text: str
    style: TextStyle
    font_size: float
    font_name: str
    color: Tuple[float, float, float]  # RGB values
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    page_number: int

class FormattedPDFExtractor:
    """
    Enhanced PDF text extractor that preserves formatting.
    Created by: kaxm23
    Created on: 2025-02-09 08:21:26 UTC
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize the formatted PDF extractor."""
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Font characteristics for style detection
        self.heading_sizes = {
            TextStyle.HEADING1: 20,
            TextStyle.HEADING2: 16,
            TextStyle.HEADING3: 14
        }
        
        # Stats tracking
        self.stats = {
            'total_pages': 0,
            'total_blocks': 0,
            'styles_found': set(),
            'fonts_found': set(),
            'processing_time': None
        }

    def _determine_text_style(self, 
                            font_name: str, 
                            flags: int, 
                            font_size: float) -> TextStyle:
        """
        Determine text style based on font properties.
        
        Args:
            font_name: Name of the font
            flags: Font flags from PyMuPDF
            font_size: Size of the font
            
        Returns:
            TextStyle: Determined text style
        """
        # Check for headings based on font size
        for heading_style, min_size in self.heading_sizes.items():
            if font_size >= min_size:
                return heading_style
        
        # Check for bold and italic
        is_bold = bool(flags & 2**4)  # Check bold bit
        is_italic = bool(flags & 2**1)  # Check italic bit
        
        if is_bold and is_italic:
            return TextStyle.BOLD_ITALIC
        elif is_bold:
            return TextStyle.BOLD
        elif is_italic:
            return TextStyle.ITALIC
        else:
            return TextStyle.NORMAL

    def _extract_block_properties(self, 
                                block: fitz.TextBlock, 
                                page_number: int) -> TextBlock:
        """
        Extract properties from a text block.
        
        Args:
            block: PyMuPDF text block
            page_number: Current page number
            
        Returns:
            TextBlock: Structured text block with formatting
        """
        # Get the most common font properties in the block
        spans = block['spans']
        if not spans:
            return None
            
        # Analyze spans for dominant properties
        fonts = {}
        sizes = {}
        flags = {}
        colors = {}
        
        for span in spans:
            font = span['font']
            size = span['size']
            flag = span['flags']
            color = span['color']
            
            fonts[font] = fonts.get(font, 0) + 1
            sizes[size] = sizes.get(size, 0) + 1
            flags[flag] = flags.get(flag, 0) + 1
            colors[color] = colors.get(color, 0) + 1
        
        # Get dominant properties
        dominant_font = max(fonts.items(), key=lambda x: x[1])[0]
        dominant_size = max(sizes.items(), key=lambda x: x[1])[0]
        dominant_flags = max(flags.items(), key=lambda x: x[1])[0]
        dominant_color = max(colors.items(), key=lambda x: x[1])[0]
        
        # Determine text style
        style = self._determine_text_style(
            dominant_font, 
            dominant_flags, 
            dominant_size
        )
        
        # Update statistics
        self.stats['styles_found'].add(style)
        self.stats['fonts_found'].add(dominant_font)
        self.stats['total_blocks'] += 1
        
        return TextBlock(
            text=block['text'],
            style=style,
            font_size=dominant_size,
            font_name=dominant_font,
            color=dominant_color,
            bbox=block['bbox'],
            page_number=page_number
        )

    def extract_formatted_text(self, 
                             pdf_path: str,
                             pages: Optional[List[int]] = None) -> Dict:
        """
        Extract formatted text from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Optional list of page numbers to extract
            
        Returns:
            Dict: Structured text content with formatting
        """
        start_time = datetime.now()
        
        result = {
            'metadata': {},
            'content': [],
            'statistics': {},
            'status': 'success',
            'error': None
        }
        
        try:
            # Validate PDF file
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            self.stats['total_pages'] = doc.page_count
            
            # Extract metadata
            result['metadata'] = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'format': doc.metadata.get('format', ''),
                'encryption': doc.metadata.get('encryption', None),
                'total_pages': doc.page_count
            }
            
            # Determine pages to process
            pages_to_process = pages if pages is not None else range(doc.page_count)
            
            # Process each page
            for page_num in pages_to_process:
                try:
                    page = doc[page_num]
                    
                    # Extract text blocks with formatting
                    blocks = page.get_text("dict")["blocks"]
                    
                    # Process each block
                    page_content = []
                    for block in blocks:
                        if block.get('type') == 0:  # Text block
                            formatted_block = self._extract_block_properties(
                                block, 
                                page_num
                            )
                            if formatted_block:
                                page_content.append(asdict(formatted_block))
                    
                    # Add page content to result
                    result['content'].append({
                        'page_number': page_num,
                        'blocks': page_content
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num}: {str(e)}")
                    result['content'].append({
                        'page_number': page_num,
                        'error': str(e)
                    })
            
            # Close the document
            doc.close()
            
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"Error processing PDF: {str(e)}")
            return result
            
        finally:
            # Calculate processing time and update statistics
            self.stats['processing_time'] = \
                (datetime.now() - start_time).total_seconds()
            
            result['statistics'] = {
                'total_pages': self.stats['total_pages'],
                'total_blocks': self.stats['total_blocks'],
                'styles_found': [s.value for s in self.stats['styles_found']],
                'fonts_found': list(self.stats['fonts_found']),
                'processing_time': self.stats['processing_time']
            }
        
        return result

    def save_formatted_text(self, 
                          extracted_data: Dict,
                          output_path: str,
                          format: str = 'json') -> None:
        """
        Save extracted formatted text to file.
        
        Args:
            extracted_data: Dictionary containing extracted text and formatting
            output_path: Path to save the output
            format: Output format ('json' or 'html')
        """
        try:
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                    
            elif format == 'html':
                html_content = self._convert_to_html(extracted_data)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                    
            else:
                raise ValueError(f"Unsupported output format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to save formatted text: {str(e)}")
            raise

    def _convert_to_html(self, extracted_data: Dict) -> str:
        """
        Convert extracted data to HTML format.
        
        Args:
            extracted_data: Dictionary containing extracted text and formatting
            
        Returns:
            str: HTML representation of the formatted text
        """
        html = ['<!DOCTYPE html><html><head><meta charset="UTF-8">']
        html.append('<style>')
        html.append("""
            body { font-family: Arial, sans-serif; line-height: 1.6; }
            .block { margin-bottom: 1em; }
            .bold { font-weight: bold; }
            .italic { font-style: italic; }
            .bold-italic { font-weight: bold; font-style: italic; }
            h1 { font-size: 24px; }
            h2 { font-size: 20px; }
            h3 { font-size: 16px; }
        """)
        html.append('</style></head><body>')
        
        # Add metadata
        if extracted_data['metadata']:
            html.append('<div class="metadata">')
            html.append('<h2>Document Metadata</h2>')
            for key, value in extracted_data['metadata'].items():
                if value:
                    html.append(f'<p><strong>{key}:</strong> {value}</p>')
            html.append('</div><hr>')
        
        # Add content
        for page in extracted_data['content']:
            html.append(f'<div class="page" id="page-{page["page_number"]+1}">')
            html.append(f'<h2>Page {page["page_number"]+1}</h2>')
            
            for block in page['blocks']:
                style_class = block['style'].lower()
                font_size = block['font_size']
                color = f'rgb({",".join(map(str, block["color"]))})'
                
                if style_class.startswith('h'):
                    html.append(f'<{style_class}>{block["text"]}</{style_class}>')
                else:
                    html.append(
                        f'<div class="block {style_class}" '
                        f'style="font-size: {font_size}px; color: {color}">'
                        f'{block["text"]}</div>'
                    )
            
            html.append('</div>')
        
        html.append('</body></html>')
        return '\n'.join(html)