from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class TrOCRProcessor:
    """
    OCR processing using TrOCR model.
    Created by: kaxm23
    Created on: 2025-02-09 08:46:35 UTC
    """
    
    def __init__(self,
                 model_name: str = "microsoft/trocr-large-handwritten",
                 device: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize the OCR processor.
        
        Args:
            model_name: Name of the TrOCR model
            device: Device to use (cuda/cpu)
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Set parameters
        self.model_name = model_name
        
        # Load model and processor
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'images_processed': 0,
            'characters_extracted': 0,
            'processing_time': 0,
            'last_processed': None
        }

    def _load_model(self) -> None:
        """Load the OCR model and processor."""
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

    def process_image(self,
                     image_path: Union[str, Path, Image.Image],
                     preprocess: bool = True) -> str:
        """
        Process single image with OCR.
        
        Args:
            image_path: Path to image or PIL Image
            preprocess: Whether to preprocess image
            
        Returns:
            str: Extracted text
        """
        start_time = datetime.now()
        
        try:
            # Load and preprocess image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path
            
            if preprocess:
                image = self._preprocess_image(image)
            
            # Prepare image for model
            pixel_values = self.processor(
                image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Decode text
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Update statistics
            process_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(
                images=1,
                characters=len(generated_text),
                process_time=process_time
            )
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Failed to process image: {str(e)}")
            raise

    def process_batch(self,
                     image_paths: List[Union[str, Path, Image.Image]],
                     batch_size: int = 8,
                     show_progress: bool = True,
                     preprocess: bool = True) -> List[str]:
        """
        Process batch of images with OCR.
        
        Args:
            image_paths: List of image paths or PIL Images
            batch_size: Batch size for processing
            show_progress: Show progress bar
            preprocess: Whether to preprocess images
            
        Returns:
            List[str]: List of extracted texts
        """
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                
                if show_progress:
                    self.logger.info(f"Processing batch {i//batch_size + 1}")
                
                # Process each image in batch
                batch_results = [
                    self.process_image(image, preprocess)
                    for image in (tqdm(batch) if show_progress else batch)
                ]
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Add more preprocessing steps as needed
            # For example:
            # - Resize
            # - Enhance contrast
            # - Remove noise
            # - Deskew
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def _update_statistics(self,
                          images: int,
                          characters: int,
                          process_time: float) -> None:
        """Update OCR statistics."""
        self.stats['images_processed'] += images
        self.stats['characters_extracted'] += characters
        self.stats['processing_time'] += process_time
        self.stats['last_processed'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get OCR statistics.
        
        Returns:
            Dict: OCR statistics
        """
        return {
            'images_processed': self.stats['images_processed'],
            'characters_extracted': self.stats['characters_extracted'],
            'average_time_per_image': (
                self.stats['processing_time'] / self.stats['images_processed']
                if self.stats['images_processed'] > 0 else 0
            ),
            'average_chars_per_image': (
                self.stats['characters_extracted'] / self.stats['images_processed']
                if self.stats['images_processed'] > 0 else 0
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