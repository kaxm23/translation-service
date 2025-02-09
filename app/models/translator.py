from openai import AsyncOpenAI
from typing import Dict, List, Optional
import logging
from datetime import datetime
from app.core.config import Settings

class GPTTranslator:
    """
    GPT-4 translation service.
    Created by: kaxm23
    Created on: 2025-02-09 08:57:02 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 temperature: float = 0.3,
                 log_level: int = logging.INFO):
        """Initialize GPT translator."""
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Set parameters
        self.model = model
        self.temperature = temperature
        
        # Track usage
        self.stats = {
            'requests': 0,
            'tokens': 0,
            'cost': 0.0
        }

    async def translate(self,
                       text: str,
                       source_lang: str,
                       target_lang: str,
                       **kwargs) -> Dict:
        """Translate text using GPT-4."""
        try:
            # Create system prompt
            system_prompt = (
                f"You are a professional translator. Translate the following text "
                f"from {source_lang} to {target_lang}. Preserve formatting and "
                f"maintain the original meaning. Only return the translation, no "
                f"explanations or additional text."
            )
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=self.temperature,
                **kwargs
            )
            
            # Update stats
            self.stats['requests'] += 1
            self.stats['tokens'] += response.usage.total_tokens
            self.stats['cost'] += self._calculate_cost(response.usage.total_tokens)
            
            return {
                'translation': response.choices[0].message.content,
                'tokens': response.usage.total_tokens,
                'model': self.model
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    def _calculate_cost(self, tokens: int) -> float:
        """Calculate cost based on token usage."""
        # Adjust rates based on your model
        return tokens * 0.0001  # Example rate
