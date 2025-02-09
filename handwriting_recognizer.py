from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

class HandwritingRecognizer:
    """
    Handwritten text recognition using TrOCR.
    Created by: kaxm23
    Created on: 2025-02-09 08:47:35 UTC
    """
    
    def __init__(self,
                 model_name: str = "microsoft/trocr-large-handwritten",
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 log_level: int = logging.INFO):
        """
        Initialize the handwriting recognizer.
        
        Args:
            model_name: Name of the TrOCR model
            device: Device to use (cuda/cpu)
            confidence_threshold: Minimum confidence score
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
        
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        
        # Load model and processor
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'images_processed': 0,
            'words_recognized': 0,
            'total_confidence': 0,
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

    def preprocess_handwriting(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better handwriting recognition.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to grayscale
            image = ImageOps.grayscale(image)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Convert back to RGB
            image = image.convert('RGB')
            
            # Resize if needed (maintaining aspect ratio)
            max_size = 1000
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {str(e)}")
            raise

    def recognize_handwriting(self,
                            image_path: Union[str, Path, Image.Image],
                            preprocess: bool = True) -> Dict:
        """
        Recognize handwritten text in an image.
        
        Args:
            image_path: Path to image or PIL Image
            preprocess: Whether to preprocess image
            
        Returns:
            Dict: Recognition results and confidence scores
        """
        start_time = datetime.now()
        
        try:
            # Load image
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path
            
            # Preprocess image
            if preprocess:
                image = self.preprocess_handwriting(image)
            
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
                    max_length=128
                )
            
            # Decode text
            predicted_text = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )[0]
            
            # Calculate confidence score
            confidence = float(torch.mean(torch.stack(outputs.scores)).item())
            
            # Update statistics
            process_time = (datetime.now() - start_time).total_seconds()
            word_count = len(predicted_text.split())
            
            self._update_statistics(
                images=1,
                words=word_count,
                confidence=confidence,
                process_time=process_time
            )
            
            return {
                'text': predicted_text,
                'confidence': confidence,
                'processing_time': process_time,
                'word_count': word_count,
                'above_threshold': confidence >= self.confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Recognition failed: {str(e)}")
            raise

    def recognize_batch(self,
                       image_paths: List[Union[str, Path, Image.Image]],
                       batch_size: int = 8,
                       show_progress: bool = True,
                       preprocess: bool = True) -> List[Dict]:
        """
        Recognize handwriting in batch of images.
        
        Args:
            image_paths: List of image paths or PIL Images
            batch_size: Batch size for processing
            show_progress: Show progress bar
            preprocess: Whether to preprocess images
            
        Returns:
            List[Dict]: List of recognition results
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
                    self.recognize_handwriting(image, preprocess)
                    for image in (tqdm(batch) if show_progress else batch)
                ]
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            raise

    def _update_statistics(self,
                          images: int,
                          words: int,
                          confidence: float,
                          process_time: float) -> None:
        """Update recognition statistics."""
        self.stats['images_processed'] += images
        self.stats['words_recognized'] += words
        self.stats['total_confidence'] += confidence
        self.stats['processing_time'] += process_time
        self.stats['last_processed'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get recognition statistics.
        
        Returns:
            Dict: Recognition statistics
        """
        return {
            'images_processed': self.stats['images_processed'],
            'words_recognized': self.stats['words_recognized'],
            'average_confidence': (
                self.stats['total_confidence'] / self.stats['images_processed']
                if self.stats['images_processed'] > 0 else 0
            ),
            'average_time_per_image': (
                self.stats['processing_time'] / self.stats['images_processed']
                if self.stats['images_processed'] > 0 else 0
            ),
            'average_words_per_image': (
                self.stats['words_recognized'] / self.stats['images_processed']
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