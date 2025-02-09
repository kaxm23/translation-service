from transformers import MarianMTModel, MarianTokenizer
import torch
import logging
from typing import Dict, List, Union, Optional
from datetime import datetime
from tqdm import tqdm
import numpy as np

class EnglishArabicTranslator:
    """
    English to Arabic translator using MarianMT.
    Created by: kaxm23
    Created on: 2025-02-09 08:40:13 UTC
    """
    
    def __init__(self,
                 model_name: str = "Helsinki-NLP/opus-mt-en-ar",
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 log_level: int = logging.INFO):
        """
        Initialize the translator.
        
        Args:
            model_name: Name of the MarianMT model
            device: Device to use (cuda/cpu)
            batch_size: Default batch size for translation
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
        self.batch_size = batch_size
        self.model_name = model_name
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'total_translations': 0,
            'total_characters': 0,
            'total_time': 0,
            'average_time_per_char': 0,
            'last_translation': None
        }

    def _load_model(self) -> None:
        """Load the translation model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            start_time = datetime.now()
            
            # Load tokenizer
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def translate(self,
                 text: Union[str, List[str]],
                 batch_size: Optional[int] = None,
                 show_progress: bool = True,
                 max_length: int = 512) -> Union[str, List[str]]:
        """
        Translate English text to Arabic.
        
        Args:
            text: Text or list of texts to translate
            batch_size: Batch size for processing
            show_progress: Show progress bar
            max_length: Maximum sequence length
            
        Returns:
            Union[str, List[str]]: Translated text(s)
        """
        start_time = datetime.now()
        
        try:
            # Handle single text input
            if isinstance(text, str):
                return self._translate_single(text, max_length)
            
            # Handle batch translation
            batch_size = batch_size or self.batch_size
            return self._translate_batch(text, batch_size, show_progress, max_length)
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise
            
        finally:
            # Update statistics
            process_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(text, process_time)

    def _translate_single(self,
                         text: str,
                         max_length: int) -> str:
        """Translate a single text."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            translated = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
        
        # Decode translation
        result = self.tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )
        
        return result

    def _translate_batch(self,
                        texts: List[str],
                        batch_size: int,
                        show_progress: bool,
                        max_length: int) -> List[str]:
        """Translate a batch of texts."""
        results = []
        
        # Create batches
        batches = [
            texts[i:i + batch_size]
            for i in range(0, len(texts), batch_size)
        ]
        
        # Process batches
        iterator = tqdm(batches) if show_progress else batches
        
        for batch in iterator:
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translations
            with torch.no_grad():
                translated = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
            
            # Decode translations
            batch_results = [
                self.tokenizer.decode(t, skip_special_tokens=True)
                for t in translated
            ]
            
            results.extend(batch_results)
        
        return results

    def _update_statistics(self,
                          text: Union[str, List[str]],
                          process_time: float) -> None:
        """Update translation statistics."""
        # Calculate text length
        if isinstance(text, str):
            char_count = len(text)
            translation_count = 1
        else:
            char_count = sum(len(t) for t in text)
            translation_count = len(text)
        
        # Update stats
        self.stats['total_translations'] += translation_count
        self.stats['total_characters'] += char_count
        self.stats['total_time'] += process_time
        self.stats['average_time_per_char'] = (
            self.stats['total_time'] / self.stats['total_characters']
            if self.stats['total_characters'] > 0 else 0
        )
        self.stats['last_translation'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get translation statistics.
        
        Returns:
            Dict: Translation statistics
        """
        return {
            'total_translations': self.stats['total_translations'],
            'total_characters': self.stats['total_characters'],
            'total_time': f"{self.stats['total_time']:.2f}s",
            'average_time_per_char': f"{self.stats['average_time_per_char']*1000:.2f}ms",
            'device': self.device,
            'model': self.model_name,
            'last_translation': self.stats['last_translation'],
            'processed_by': 'kaxm23'
        }

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cleared")