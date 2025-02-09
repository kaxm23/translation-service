import language_tool_python
from typing import Dict, List, Optional
import logging
from datetime import datetime

class LanguageToolChecker:
    """
    AI-based grammar and translation error detection using LanguageTool.
    Created by: kaxm23
    Created on: 2025-02-09 09:12:02 UTC
    """
    
    def __init__(self,
                 language: str = 'en-US',
                 log_level: int = logging.INFO):
        """
        Initialize LanguageTool checker.
        
        Args:
            language: Default language for checking
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize LanguageTool
        try:
            self.tool = language_tool_python.LanguageTool(language)
            self.logger.info(f"LanguageTool initialized for {language}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LanguageTool: {str(e)}")
            raise
        
        # Initialize statistics
        self.stats = {
            'checks_performed': 0,
            'errors_found': 0,
            'texts_processed': 0,
            'last_check': None
        }

    async def check_text(self,
                        text: str,
                        language: Optional[str] = None) -> Dict:
        """
        Check text for grammar and style errors.
        
        Args:
            text: Text to check
            language: Language of the text (optional)
            
        Returns:
            Dict: Check results and suggestions
        """
        try:
            start_time = datetime.utcnow()
            
            # Set language if provided
            if language:
                self.tool.language = language
            
            # Get matches (errors and suggestions)
            matches = self.tool.check(text)
            
            # Process matches
            errors = []
            for match in matches:
                error = {
                    'message': match.message,
                    'context': match.context,
                    'offset': match.offset,
                    'length': match.length,
                    'rule_id': match.ruleId,
                    'category': match.category,
                    'replacements': match.replacements
                }
                errors.append(error)
            
            # Calculate processing time
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._update_statistics(len(errors))
            
            return {
                'text': text,
                'language': self.tool.language,
                'error_count': len(errors),
                'errors': errors,
                'suggestions': self.tool.correct(text),
                'processing_time': process_time,
                'timestamp': "2025-02-09 09:12:02",
                'processed_by': 'kaxm23'
            }
            
        except Exception as e:
            self.logger.error(f"Text check failed: {str(e)}")
            raise

    async def check_batch(self,
                         texts: List[str],
                         language: Optional[str] = None) -> List[Dict]:
        """
        Check multiple texts for errors.
        
        Args:
            texts: List of texts to check
            language: Language of the texts (optional)
            
        Returns:
            List[Dict]: List of check results
        """
        try:
            results = []
            
            for text in texts:
                result = await self.check_text(text, language)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch check failed: {str(e)}")
            raise

    async def compare_translations(self,
                                 source_text: str,
                                 translated_text: str,
                                 source_lang: str,
                                 target_lang: str) -> Dict:
        """
        Compare source and translated text for potential errors.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Dict: Comparison results
        """
        try:
            # Check source text
            source_result = await self.check_text(source_text, source_lang)
            
            # Check translated text
            target_result = await self.check_text(translated_text, target_lang)
            
            return {
                'source': {
                    'text': source_text,
                    'language': source_lang,
                    'errors': source_result['errors'],
                    'suggestions': source_result['suggestions']
                },
                'translation': {
                    'text': translated_text,
                    'language': target_lang,
                    'errors': target_result['errors'],
                    'suggestions': target_result['suggestions']
                },
                'comparison': {
                    'source_error_count': len(source_result['errors']),
                    'translation_error_count': len(target_result['errors']),
                    'quality_score': self._calculate_quality_score(
                        source_result['errors'],
                        target_result['errors']
                    )
                },
                'timestamp': "2025-02-09 09:12:02",
                'processed_by': 'kaxm23'
            }
            
        except Exception as e:
            self.logger.error(f"Translation comparison failed: {str(e)}")
            raise

    def _calculate_quality_score(self,
                               source_errors: List[Dict],
                               target_errors: List[Dict]) -> float:
        """Calculate translation quality score."""
        # Simple scoring based on error counts
        source_count = len(source_errors)
        target_count = len(target_errors)
        
        if source_count == 0 and target_count == 0:
            return 1.0
        elif source_count == 0:
            return max(0.0, 1.0 - (target_count * 0.1))
        else:
            ratio = target_count / source_count
            return max(0.0, 1.0 - (ratio * 0.5))

    def _update_statistics(self, error_count: int) -> None:
        """Update usage statistics."""
        self.stats['checks_performed'] += 1
        self.stats['errors_found'] += error_count
        self.stats['texts_processed'] += 1
        self.stats['last_check'] = "2025-02-09 09:12:02"

    def get_statistics(self) -> Dict:
        """
        Get usage statistics.
        
        Returns:
            Dict: Usage statistics
        """
        return {
            'checks_performed': self.stats['checks_performed'],
            'errors_found': self.stats['errors_found'],
            'texts_processed': self.stats['texts_processed'],
            'average_errors_per_text': (
                self.stats['errors_found'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0 else 0
            ),
            'language': self.tool.language,
            'last_check': self.stats['last_check'],
            'processed_by': 'kaxm23'
        }

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            List[str]: Supported languages
        """
        return self.tool.languages