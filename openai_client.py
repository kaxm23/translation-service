import openai
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime

class OpenAIClient:
    """
    OpenAI API client wrapper.
    Created by: kaxm23
    Created on: 2025-02-09 08:53:44 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 150,
                 log_level: int = logging.INFO):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
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
        
        # Initialize statistics
        self.stats = {
            'requests_made': 0,
            'tokens_used': 0,
            'total_cost': 0,
            'last_request': None
        }
        
        self.logger.info(f"OpenAI client initialized with model: {model}")

    async def generate_text(self,
                          prompt: str,
                          temperature: Optional[float] = None,
                          max_tokens: Optional[int] = None) -> Dict:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Dict: Generation results and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }
            
            # Make API call
            response = await openai.ChatCompletion.acreate(**params)
            
            # Extract response
            generated_text = response.choices[0].message.content
            
            # Calculate metadata
            process_time = (datetime.utcnow() - start_time).total_seconds()
            token_count = response.usage.total_tokens
            
            # Update statistics
            self._update_statistics(token_count, process_time)
            
            return {
                'text': generated_text,
                'tokens_used': token_count,
                'processing_time': process_time,
                'model': self.model,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise

    async def generate_batch(self,
                           prompts: List[str],
                           **kwargs) -> List[Dict]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments for generate_text()
            
        Returns:
            List[Dict]: List of generation results
        """
        try:
            results = []
            
            for prompt in prompts:
                result = await self.generate_text(prompt, **kwargs)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            raise

    def _update_statistics(self,
                          tokens: int,
                          process_time: float) -> None:
        """Update usage statistics."""
        self.stats['requests_made'] += 1
        self.stats['tokens_used'] += tokens
        
        # Calculate approximate cost (varies by model)
        cost_per_token = 0.0001  # Example rate, adjust based on model
        self.stats['total_cost'] += tokens * cost_per_token
        
        self.stats['last_request'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Dict: Usage statistics
        """
        return {
            'requests_made': self.stats['requests_made'],
            'tokens_used': self.stats['tokens_used'],
            'total_cost': f"${self.stats['total_cost']:.4f}",
            'average_tokens_per_request': (
                self.stats['tokens_used'] / self.stats['requests_made']
                if self.stats['requests_made'] > 0 else 0
            ),
            'model': self.model,
            'last_request': self.stats['last_request'],
            'processed_by': 'kaxm23'
        }