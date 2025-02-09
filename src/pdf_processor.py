import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display

class PDFProcessor:
    def __init__(self):
        """Initialize the PDF processor"""
        self.text_content = []

    def extract_text(self, pdf_path):
        """
        Extract text from PDF using both direct extraction and OCR
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            list: List of extracted text pages
        """
        try:
            # First attempt: Direct PDF text extraction
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text.strip():  # If text was successfully extracted
                        # Handle Arabic text direction
                        reshaped_text = arabic_reshaper.reshape(text)
                        bidi_text = get_display(reshaped_text)
                        self.text_content.append(bidi_text)
                    else:  # If no text was extracted, try OCR
                        self._process_page_with_ocr(page)
                        
            return self.text_content
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return []

    def _process_page_with_ocr(self, page):
        """
        Process a single page using OCR when direct extraction fails
        
        Args:
            page: PDF page object
        """
        # Convert PDF page to image
        images = convert_from_path(page)
        
        for image in images:
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR results
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Perform OCR with Arabic language support
            text = pytesseract.image_to_string(preprocessed, lang='ara')
            
            # Handle Arabic text direction
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            
            self.text_content.append(bidi_text)