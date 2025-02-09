import openai
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm_asyncio

class GPTTranslator:
    """
    Text translation using GPT-4 API.
    Created by: kaxm23
    Created on: 2025-02-09 08:55:37 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 max_tokens: int = 1000,
                 log_level: int = logging.INFO):
        """
        Initialize GPT translator.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4)
            temperature: Temperature for translation
            max_tokens: Maximum tokens per request
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set up OpenAI client
        openai.api_key = api_key
        
        # Set parameters
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Language codes mapping
        self.language_codes = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'tr': 'Turkish',
            'pl': 'Polish'
        }
        
        # Initialize statistics
        self.stats = {
            'translations_made': 0,
            'tokens_used': 0,
            'total_cost': 0,
            'last_translation': None
        }
        
        self.logger.info(f"GPT Translator initialized with model: {model}")

    async def translate(self,
                       text: str,
                       source_lang: str,
                       target_lang: str,
                       preserve_formatting: bool = True,
                       tone: Optional[str] = None) -> Dict:
        """
        Translate text using GPT-4.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            preserve_formatting: Preserve original formatting
            tone: Translation tone (formal/informal/technical)
            
        Returns:
            Dict: Translation results and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Validate language codes
            if source_lang not in self.language_codes:
                raise ValueError(f"Invalid source language code: {source_lang}")
            if target_lang not in self.language_codes:
                raise ValueError(f"Invalid target language code: {target_lang}")
            
            # Prepare system prompt
            system_prompt = self._create_system_prompt(
                source_lang,
                target_lang,
                preserve_formatting,
                tone
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract translation
            translated_text = response.choices[0].message.content
            
            # Calculate metadata
            process_time = (datetime.utcnow() - start_time).total_seconds()
            token_count = response.usage.total_tokens
            
            # Update statistics
            self._update_statistics(token_count, process_time)
            
            return {
                'translated_text': translated_text,
                'source_lang': self.language_codes[source_lang],
                'target_lang': self.language_codes[target_lang],
                'tokens_used': token_count,
                'processing_time': process_time,
                'model': self.model,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    async def translate_batch(self,
                            texts: List[str],
                            source_lang: str,
                            target_lang: str,
                            batch_size: int = 5,
                            **kwargs) -> List[Dict]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Number of concurrent translations
            **kwargs: Additional arguments for translate()
            
        Returns:
            List[Dict]: List of translation results
        """
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Create tasks for concurrent translation
                tasks = [
                    self.translate(
                        text,
                        source_lang,
                        target_lang,
                        **kwargs
                    )
                    for text in batch
                ]
                
                # Execute batch
                batch_results = await tqdm_asyncio.gather(
                    *tasks,
                    desc=f"Processing batch {i//batch_size + 1}"
                )
                
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch translation failed: {str(e)}")
            raise

    def _create_system_prompt(self,
                            source_lang: str,
                            target_lang: str,
                            preserve_formatting: bool,
                            tone: Optional[str]) -> str:
        """Create system prompt for translation."""
        prompt = (
            f"You are a professional translator. Translate the following text "
            f"from {self.language_codes[source_lang]} to {self.language_codes[target_lang]}. "
        )
        
        if preserve_formatting:
            prompt += "Preserve the original text formatting, including line breaks and spacing. "
        
        if tone:
            prompt += f"Use a {tone} tone in the translation. "
        
        prompt += (
            "Maintain accuracy while ensuring natural-sounding translations. "
            "Do not include explanations or notes, only provide the translation."
        )
        
        return prompt

    def _update_statistics(self,
                          tokens: int,
                          process_time: float) -> None:
        """Update usage statistics."""
        self.stats['translations_made'] += 1
        self.stats['tokens_used'] += tokens
        
        # Calculate approximate cost (varies by model)
        cost_per_token = 0.0001  # Example rate, adjust based on model
        self.stats['total_cost'] += tokens * cost_per_token
        
        self.stats['last_translation'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Dict: Usage statistics
        """
        return {
            'translations_made': self.stats['translations_made'],
            'tokens_used': self.stats['tokens_used'],
            'total_cost': f"${self.stats['total_cost']:.4f}",
            'average_tokens_per_translation': (
                self.stats['tokens_used'] / self.stats['translations_made']
                if self.stats['translations_made'] > 0 else 0
            ),
            'model': self.model,
            'last_translation': self.stats['last_translation'],
            'processed_by': 'kaxm23'
        }

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages.
        
        Returns:
            Dict[str, str]: Language codes and names
        """
        return self.language_codes