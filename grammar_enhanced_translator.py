import torch
from gramformer import Gramformer
from typing import Dict, List, Optional
import logging
from datetime import datetime
import openai
import asyncio

class GrammarEnhancedTranslator:
    """
    Translation with grammar pre-correction using Gramformer.
    Created by: kaxm23
    Created on: 2025-02-09 09:07:46 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 grammar_model: str = "prithivida/grammar_error_correcter_v1",
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.8,
                 log_level: int = logging.INFO):
        """
        Initialize translator with grammar correction.
        
        Args:
            api_key: OpenAI API key
            model: Translation model
            grammar_model: Gramformer model name
            device: Device to use (cuda/cpu)
            confidence_threshold: Minimum confidence for grammar corrections
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
        self.model = model
        self.confidence_threshold = confidence_threshold
        openai.api_key = api_key
        
        # Initialize Gramformer
        self._init_grammar_corrector(grammar_model)
        
        # Initialize statistics
        self.stats = {
            'translations': 0,
            'grammar_corrections': 0,
            'tokens_used': 0,
            'total_cost': 0.0,
            'last_processed': "2025-02-09 09:07:46"
        }
        
        self.logger.info(
            f"Initialized with model: {model}, device: {self.device}"
        )

    def _init_grammar_corrector(self, model_name: str):
        """Initialize Gramformer model."""
        try:
            self.logger.info("Initializing Gramformer...")
            self.grammar_corrector = Gramformer(
                models=1,  # 1 for grammar correction
                use_gpu=self.device == "cuda"
            )
            self.logger.info("Gramformer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gramformer: {str(e)}")
            raise

    async def translate_with_grammar_check(self,
                                         text: str,
                                         source_lang: str,
                                         target_lang: str,
                                         show_corrections: bool = False,
                                         **kwargs) -> Dict:
        """
        Translate text with grammar correction.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            show_corrections: Show grammar corrections
            **kwargs: Additional translation parameters
            
        Returns:
            Dict: Translation results and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Step 1: Grammar correction (for English source text)
            if source_lang.lower() in ['en', 'en-us', 'en-gb']:
                grammar_result = await self._correct_grammar(text)
                corrected_text = grammar_result['corrected_text']
                grammar_changes = grammar_result['changes']
            else:
                corrected_text = text
                grammar_changes = []
            
            # Step 2: Translation
            translation_result = await self._translate_text(
                corrected_text,
                source_lang,
                target_lang,
                **kwargs
            )
            
            # Calculate processing time
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._update_statistics(
                translation_result['tokens'],
                len(grammar_changes)
            )
            
            result = {
                'original_text': text,
                'corrected_text': corrected_text if show_corrections else None,
                'translated_text': translation_result['translation'],
                'grammar_changes': grammar_changes if show_corrections else None,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'confidence_score': translation_result.get('confidence', 1.0),
                'tokens_used': translation_result['tokens'],
                'processing_time': process_time,
                'model': self.model,
                'timestamp': "2025-02-09 09:07:46",
                'processed_by': 'kaxm23'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Translation with grammar check failed: {str(e)}")
            raise

    async def _correct_grammar(self, text: str) -> Dict:
        """
        Correct grammar using Gramformer.
        
        Args:
            text: Text to correct
            
        Returns:
            Dict: Correction results
        """
        try:
            # Get corrections
            corrections = list(self.grammar_corrector.correct(text))
            
            if not corrections:
                return {
                    'corrected_text': text,
                    'changes': []
                }
            
            # Use the best correction
            corrected_text = corrections[0]
            
            # Identify changes
            changes = self._identify_grammar_changes(text, corrected_text)
            
            return {
                'corrected_text': corrected_text,
                'changes': changes
            }
            
        except Exception as e:
            self.logger.error(f"Grammar correction failed: {str(e)}")
            raise

    def _identify_grammar_changes(self,
                                original: str,
                                corrected: str) -> List[Dict]:
        """
        Identify grammar changes made.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            List[Dict]: List of changes made
        """
        changes = []
        
        # Split into words
        original_words = original.split()
        corrected_words = corrected.split()
        
        # Simple word-level diff
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig != corr:
                changes.append({
                    'position': i,
                    'original': orig,
                    'correction': corr,
                    'type': 'word_replacement'
                })
        
        # Check for length differences
        if len(original_words) != len(corrected_words):
            changes.append({
                'type': 'structure_change',
                'description': 'Sentence structure was modified'
            })
        
        return changes

    async def _translate_text(self,
                            text: str,
                            source_lang: str,
                            target_lang: str,
                            **kwargs) -> Dict:
        """
        Translate text using OpenAI API.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            **kwargs: Additional parameters
            
        Returns:
            Dict: Translation results
        """
        try:
            # Create system prompt
            system_prompt = (
                f"You are a professional translator. Translate the following text "
                f"from {source_lang} to {target_lang}. Maintain the original "
                f"meaning and tone while ensuring natural language flow."
            )
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                **kwargs
            )
            
            return {
                'translation': response.choices[0].message.content,
                'tokens': response.usage.total_tokens,
                'confidence': response.choices[0].finish_reason == "stop"
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    async def translate_batch(self,
                            texts: List[str],
                            source_lang: str,
                            target_lang: str,
                            **kwargs) -> List[Dict]:
        """
        Translate multiple texts with grammar correction.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language
            **kwargs: Additional parameters
            
        Returns:
            List[Dict]: List of translation results
        """
        try:
            tasks = [
                self.translate_with_grammar_check(
                    text,
                    source_lang,
                    target_lang,
                    **kwargs
                )
                for text in texts
            ]
            
            results = await asyncio.gather(*tasks)
            return results
            
        except Exception as e:
            self.logger.error(f"Batch translation failed: {str(e)}")
            raise

    def _update_statistics(self,
                         tokens: int,
                         grammar_changes: int) -> None:
        """Update usage statistics."""
        self.stats['translations'] += 1
        self.stats['grammar_corrections'] += grammar_changes
        self.stats['tokens_used'] += tokens
        
        # Calculate cost (adjust rate as needed)
        cost_per_token = 0.00003
        self.stats['total_cost'] += tokens * cost_per_token
        
        self.stats['last_processed'] = "2025-02-09 09:07:46"

    def get_statistics(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Dict: Usage statistics
        """
        return {
            'translations_completed': self.stats['translations'],
            'grammar_corrections_made': self.stats['grammar_corrections'],
            'total_tokens_used': self.stats['tokens_used'],
            'total_cost': f"${self.stats['total_cost']:.4f}",
            'average_corrections_per_text': (
                self.stats['grammar_corrections'] / self.stats['translations']
                if self.stats['translations'] > 0 else 0
            ),
            'model': self.model,
            'device': self.device,
            'last_processed': self.stats['last_processed'],
            'processed_by': 'kaxm23'
        }