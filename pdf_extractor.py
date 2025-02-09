import fitz  # PyMuPDF
from datetime import datetime
import logging
from typing import List, Optional, Dict
from pathlib import Path

class PDFExtractor:
    """
    A class to handle PDF text extraction using PyMuPDF.
    Created by: kaxm23
    Created on: 2025-02-09 08:08:37 UTC
    """
    
    def __init__(self, log_level: int = logging.INFO):
        """
        Initialize the PDF extractor with logging configuration.
        
        Args:
            log_level: Logging level (default: logging.INFO)
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text(self, pdf_path: str, 
                    pages: Optional[List[int]] = None,
                    include_metadata: bool = False) -> Dict:
        """
        Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path (str): Path to the PDF file
            pages (List[int], optional): List of specific page numbers to extract (0-based).
                                       If None, extracts all pages.
            include_metadata (bool): Whether to include PDF metadata in the output

        Returns:
            Dict: Dictionary containing:
                - 'text': Dictionary of page numbers and their text content
                - 'metadata': PDF metadata (if include_metadata is True)
                - 'status': Success or error status
                - 'error': Error message if any

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            RuntimeError: If there's an error processing the PDF
        """
        result = {
            'text': {},
            'metadata': {},
            'status': 'success',
            'error': None
        }

        try:
            # Check if file exists
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Open the PDF file
            self.logger.info(f"Opening PDF file: {pdf_path}")
            pdf_document = fitz.open(pdf_path)

            # Get total number of pages
            total_pages = pdf_document.page_count

            # Extract metadata if requested
            if include_metadata:
                result['metadata'] = {
                    'title': pdf_document.metadata.get('title', ''),
                    'author': pdf_document.metadata.get('author', ''),
                    'subject': pdf_document.metadata.get('subject', ''),
                    'keywords': pdf_document.metadata.get('keywords', ''),
                    'creator': pdf_document.metadata.get('creator', ''),
                    'producer': pdf_document.metadata.get('producer', ''),
                    'page_count': total_pages
                }

            # Determine which pages to process
            if pages is None:
                pages_to_process = range(total_pages)
            else:
                # Validate page numbers
                pages_to_process = [p for p in pages if 0 <= p < total_pages]
                if len(pages_to_process) != len(pages):
                    self.logger.warning("Some requested pages were out of range and will be skipped")

            # Extract text from each page
            for page_num in pages_to_process:
                try:
                    page = pdf_document[page_num]
                    text = page.get_text()
                    
                    # Store non-empty text
                    if text.strip():
                        result['text'][page_num] = text
                    else:
                        self.logger.warning(f"Page {page_num + 1} appears to be empty or contains no extractable text")
                
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    result['text'][page_num] = f"Error extracting text: {str(e)}"

            # Close the PDF
            pdf_document.close()

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

        return result

    @staticmethod
    def save_extracted_text(extracted_data: Dict, output_path: str) -> None:
        """
        Save extracted text to a file.

        Args:
            extracted_data (Dict): The dictionary returned by extract_text()
            output_path (str): Path where to save the text file

        Returns:
            None
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write metadata if available
            if extracted_data.get('metadata'):
                f.write("=== Document Metadata ===\n")
                for key, value in extracted_data['metadata'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n=== Document Content ===\n\n")

            # Write text content
            for page_num, text in sorted(extracted_data['text'].items()):
                f.write(f"--- Page {page_num + 1} ---\n")
                f.write(text)
                f.write("\n\n")