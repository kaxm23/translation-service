from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List, Dict
import logging
from datetime import datetime

class MarianTranslator:
    """
    Neural Machine Translation using MarianMT models.
    Created by: kaxm23
    Created on: 2025-02-09 08:39:24 UTC
    """
    
    def __init__(self,
                 source_lang: str = "en",
                 target_lang: str = "ar",
                 device: str = None,
                 log_level: int = logging.INFO):
        """
        Initialize the translator.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            device: Device to use (cuda/cpu)
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set up device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        try:
            model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.logger.info(f"Model loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Statistics tracking
        self.stats = {
            'translations': 0,
            'characters_processed': 0,
            'processing_time': 0
        }

    def translate(self,
                 text: str,
                 batch_size: int = 8,
                 max_length: int = 512) -> str:
        """
        Translate text using MarianMT.
        
        Args:
            text: Text to translate
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            str: Translated text
        """
        try:
            # Tokenize input text
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Generate translation
            translated = self.model.generate(
                **encoded,
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True
            )
            
            # Decode translation
            translated_text = self.tokenizer.decode(
                translated[0],
                skip_special_tokens=True
            )
            
            # Update statistics
            self.stats['translations'] += 1
            self.stats['characters_processed'] += len(text)
            
            return translated_text
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    def translate_batch(self,
                       texts: List[str],
                       batch_size: int = 8,
                       max_length: int = 512) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            List[str]: List of translated texts
        """
        translations = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Tokenize batch
                encoded = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Generate translations
                translated = self.model.generate(
                    **encoded,
                    max_length=max_length,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
                
                # Decode translations
                batch_translations = [
                    self.tokenizer.decode(t, skip_special_tokens=True)
                    for t in translated
                ]
                
                translations.extend(batch_translations)
                
                # Update statistics
                self.stats['translations'] += len(batch)
                self.stats['characters_processed'] += sum(len(t) for t in batch)
                
        except Exception as e:
            self.logger.error(f"Batch translation failed: {str(e)}")
            raise
            
        return translations

    def get_statistics(self) -> Dict:
        """
        Get translation statistics.
        
        Returns:
            Dict: Translation statistics
        """
        return {
            'total_translations': self.stats['translations'],
            'total_characters': self.stats['characters_processed'],
            'device_used': self.device,
            'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        }