from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image, ImageDraw, ImageEnhance
import pypdf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF
from io import BytesIO
import tempfile
from tqdm import tqdm

class SmartPDFExtractor:
    """
    Smart PDF text extraction using AI OCR and traditional methods.
    Created by: kaxm23
    Created on: 2025-02-09 08:49:10 UTC
    """
    
    def __init__(self,
                 model_name: str = "microsoft/trocr-large-handwritten",
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 dpi: int = 300,
                 log_level: int = logging.INFO):
        """
        Initialize the smart PDF extractor.
        
        Args:
            model_name: Name of the TrOCR model
            device: Device to use (cuda/cpu)
            confidence_threshold: Minimum confidence score
            dpi: DPI for image extraction
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set parameters
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"DPI: {self.dpi}")
        
        # Load AI model
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'pdfs_processed': 0,
            'pages_processed': 0,
            'ocr_pages': 0,
            'traditional_pages': 0,
            'total_words': 0,
            'processing_time': 0,
            'last_processed': None
        }

    def _load_model(self) -> None:
        """Load the TrOCR model and processor."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            start_time = datetime.now()
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def extract_text_from_pdf(self,
                            pdf_path: Union[str, Path],
                            output_path: Optional[Union[str, Path]] = None,
                            save_images: bool = False,
                            image_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Extract text from PDF using both traditional and AI methods.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Path to save extracted text
            save_images: Whether to save extracted images
            image_dir: Directory to save images
            
        Returns:
            Dict: Extraction results and statistics
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing PDF: {pdf_path}")
            pdf_path = Path(pdf_path)
            
            # Create image directory if needed
            if save_images and image_dir:
                image_dir = Path(image_dir)
                image_dir.mkdir(parents=True, exist_ok=True)
            
            # Process PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            results = []
            page_types = []
            
            # Process each page
            for page_num in tqdm(range(total_pages), desc="Processing pages"):
                page_result = self._process_page(
                    doc[page_num],
                    page_num,
                    save_images,
                    image_dir
                )
                
                results.append(page_result)
                page_types.append(page_result['extraction_method'])
            
            # Combine results
            combined_text = self._combine_results(results)
            
            # Calculate statistics
            process_time = (datetime.now() - start_time).total_seconds()
            stats = self._calculate_statistics(
                results,
                page_types,
                process_time
            )
            
            # Save results if requested
            if output_path:
                self._save_results(
                    combined_text,
                    results,
                    stats,
                    output_path
                )
            
            return {
                'text': combined_text,
                'page_results': results,
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process PDF: {str(e)}")
            raise

    def _process_page(self,
                     page: fitz.Page,
                     page_num: int,
                     save_images: bool,
                     image_dir: Optional[Path]) -> Dict:
        """Process a single PDF page."""
        try:
            # Try traditional text extraction first
            traditional_text = page.get_text()
            
            # Check if page needs OCR
            if self._needs_ocr(traditional_text, page):
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Save image if requested
                if save_images and image_dir:
                    img_path = image_dir / f"page_{page_num + 1}.png"
                    img.save(img_path)
                
                # Perform OCR
                ocr_result = self._perform_ocr(img)
                
                return {
                    'page_number': page_num + 1,
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence'],
                    'extraction_method': 'ocr',
                    'word_count': len(ocr_result['text'].split())
                }
            else:
                return {
                    'page_number': page_num + 1,
                    'text': traditional_text,
                    'confidence': 1.0,
                    'extraction_method': 'traditional',
                    'word_count': len(traditional_text.split())
                }
                
        except Exception as e:
            self.logger.error(f"Failed to process page {page_num + 1}: {str(e)}")
            raise

    def _needs_ocr(self, text: str, page: fitz.Page) -> bool:
        """Determine if page needs OCR."""
        # Check if traditional extraction yielded any text
        if not text.strip():
            return True
        
        # Check for images on page
        image_list = page.get_images(full=True)
        if len(image_list) > 0 and len(text.split()) < 50:
            return True
        
        # Check text density
        words_per_area = len(text.split()) / (page.rect.width * page.rect.height)
        if words_per_area < 0.001:  # Threshold can be adjusted
            return True
        
        return False

    def _perform_ocr(self, image: Image.Image) -> Dict:
        """Perform OCR on an image."""
        try:
            # Preprocess image
            image = self._preprocess_image(image)
            
            # Prepare image for model
            pixel_values = self.processor(
                image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_length=512
                )
            
            # Decode text
            predicted_text = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence
            confidence = float(torch.mean(torch.stack(outputs.scores)).item())
            
            return {
                'text': predicted_text,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"OCR failed: {str(e)}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for OCR."""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def _combine_results(self, results: List[Dict]) -> str:
        """Combine results from all pages."""
        return "\n\n".join(
            f"[Page {r['page_number']}]\n{r['text']}"
            for r in results
        )

    def _calculate_statistics(self,
                            results: List[Dict],
                            page_types: List[str],
                            process_time: float) -> Dict:
        """Calculate extraction statistics."""
        ocr_pages = page_types.count('ocr')
        traditional_pages = page_types.count('traditional')
        total_words = sum(r['word_count'] for r in results)
        
        # Update global statistics
        self.stats['pdfs_processed'] += 1
        self.stats['pages_processed'] += len(results)
        self.stats['ocr_pages'] += ocr_pages
        self.stats['traditional_pages'] += traditional_pages
        self.stats['total_words'] += total_words
        self.stats['processing_time'] += process_time
        self.stats['last_processed'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )
        
        return {
            'total_pages': len(results),
            'ocr_pages': ocr_pages,
            'traditional_pages': traditional_pages,
            'total_words': total_words,
            'processing_time': process_time,
            'words_per_second': total_words / process_time,
            'extraction_methods': {
                'ocr': ocr_pages,
                'traditional': traditional_pages
            }
        }

    def _save_results(self,
                     text: str,
                     results: List[Dict],
                     stats: Dict,
                     output_path: Union[str, Path]) -> None:
        """Save extraction results."""
        try:
            output_path = Path(output_path)
            
            # Save main text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save detailed results
            detailed_path = output_path.parent / f"{output_path.stem}_detailed.txt"
            with open(detailed_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Page {result['page_number']}\n")
                    f.write(f"Method: {result['extraction_method']}\n")
                    f.write(f"Confidence: {result['confidence']:.2f}\n")
                    f.write(f"Word Count: {result['word_count']}\n")
                    f.write(f"{'='*50}\n\n")
                    f.write(result['text'])
                    f.write("\n")
                
                f.write("\n\nStatistics:\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            
            self.logger.info(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    def get_statistics(self) -> Dict:
        """
        Get extraction statistics.
        
        Returns:
            Dict: Extraction statistics
        """
        return {
            'pdfs_processed': self.stats['pdfs_processed'],
            'pages_processed': self.stats['pages_processed'],
            'ocr_pages': self.stats['ocr_pages'],
            'traditional_pages': self.stats['traditional_pages'],
            'total_words': self.stats['total_words'],
            'average_time_per_page': (
                self.stats['processing_time'] / self.stats['pages_processed']
                if self.stats['pages_processed'] > 0 else 0
            ),
            'ocr_percentage': (
                100 * self.stats['ocr_pages'] / self.stats['pages_processed']
                if self.stats['pages_processed'] > 0 else 0
            ),
            'device': self.device,
            'model': self.model_name,
            'last_processed': self.stats['last_processed'],
            'processed_by': 'kaxm23'
        }

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cleared")