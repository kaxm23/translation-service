from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

# Set seed for consistent results
DetectorFactory.seed = 0

@dataclass
class LanguageDetectionResult:
    """Represents the result of language detection."""
    primary_language: str
    confidence: float
    all_languages: List[Dict[str, float]]
    text_sample: str
    timestamp: str
    detection_method: str
    processed_by: str = "kaxm23"

class LanguageDetector:
    """
    A class for detecting text language with confidence scores.
    Created by: kaxm23
    Created on: 2025-02-09 08:23:43 UTC
    """
    
    # ISO language codes mapping
    LANGUAGE_CODES = {
        'ar': 'Arabic',
        'en': 'English',
        'fr': 'French',
        'es': 'Spanish',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh-cn': 'Chinese (Simplified)',
        'zh-tw': 'Chinese (Traditional)',
        'hi': 'Hindi',
        'ur': 'Urdu',
        'fa': 'Persian',
        'tr': 'Turkish'
    }
    
    def __init__(self, 
                 min_text_length: int = 10,
                 confidence_threshold: float = 0.5,
                 log_level: int = logging.INFO):
        """
        Initialize the language detector.
        
        Args:
            min_text_length (int): Minimum text length for reliable detection
            confidence_threshold (float): Minimum confidence score threshold
            log_level (int): Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.min_text_length = min_text_length
        self.confidence_threshold = confidence_threshold
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'language_distribution': {},
            'average_confidence': 0.0
        }

    def detect_language(self, 
                       text: str,
                       return_all: bool = False) -> Union[LanguageDetectionResult, str]:
        """
        Detect the language of the given text with confidence scores.
        
        Args:
            text (str): Text to analyze
            return_all (bool): Whether to return full detection result or just language code
            
        Returns:
            Union[LanguageDetectionResult, str]: Detection result or language code
        """
        self.stats['total_detections'] += 1
        
        try:
            # Validate input
            if not text or len(text.strip()) < self.min_text_length:
                raise ValueError(
                    f"Text too short. Minimum length: {self.min_text_length}"
                )
            
            # Get all language probabilities
            language_probabilities = detect_langs(text)
            
            # Sort by probability
            language_probabilities.sort(key=lambda x: x.prob, reverse=True)
            
            # Get primary language and confidence
            primary_lang = language_probabilities[0]
            
            # Update statistics
            self.stats['successful_detections'] += 1
            self.stats['language_distribution'][primary_lang.lang] = \
                self.stats['language_distribution'].get(primary_lang.lang, 0) + 1
            self.stats['average_confidence'] = (
                (self.stats['average_confidence'] * (self.stats['successful_detections'] - 1) +
                 primary_lang.prob) / self.stats['successful_detections']
            )
            
            # Create result object
            result = LanguageDetectionResult(
                primary_language=primary_lang.lang,
                confidence=primary_lang.prob,
                all_languages=[
                    {'lang': lang.lang, 'confidence': lang.prob}
                    for lang in language_probabilities
                ],
                text_sample=text[:100] + '...' if len(text) > 100 else text,
                timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                detection_method='langdetect'
            )
            
            # Log detection
            self.logger.info(
                f"Language detected: {self._get_language_name(result.primary_language)} "
                f"(confidence: {result.confidence:.2f})"
            )
            
            return result if return_all else result.primary_language
            
        except LangDetectException as e:
            self.stats['failed_detections'] += 1
            self.logger.error(f"Language detection failed: {str(e)}")
            raise
        
        except Exception as e:
            self.stats['failed_detections'] += 1
            self.logger.error(f"Unexpected error in language detection: {str(e)}")
            raise

    def _get_language_name(self, lang_code: str) -> str:
        """
        Get full language name from ISO code.
        
        Args:
            lang_code (str): ISO language code
            
        Returns:
            str: Full language name
        """
        return self.LANGUAGE_CODES.get(lang_code, lang_code)

    def get_statistics(self) -> Dict:
        """
        Get language detection statistics.
        
        Returns:
            Dict: Detection statistics
        """
        return {
            'total_detections': self.stats['total_detections'],
            'successful_detections': self.stats['successful_detections'],
            'failed_detections': self.stats['failed_detections'],
            'success_rate': (
                self.stats['successful_detections'] / self.stats['total_detections']
                if self.stats['total_detections'] > 0 else 0
            ),
            'language_distribution': {
                self._get_language_name(lang): count
                for lang, count in self.stats['language_distribution'].items()
            },
            'average_confidence': self.stats['average_confidence']
        }

    def save_detection_result(self, 
                            result: LanguageDetectionResult,
                            output_path: str) -> None:
        """
        Save detection result to file.
        
        Args:
            result (LanguageDetectionResult): Detection result
            output_path (str): Path to save the result
        """
        try:
            # Convert result to dictionary
            result_dict = {
                'primary_language': {
                    'code': result.primary_language,
                    'name': self._get_language_name(result.primary_language)
                },
                'confidence': result.confidence,
                'all_languages': [
                    {
                        'code': lang['lang'],
                        'name': self._get_language_name(lang['lang']),
                        'confidence': lang['confidence']
                    }
                    for lang in result.all_languages
                ],
                'text_sample': result.text_sample,
                'timestamp': result.timestamp,
                'detection_method': result.detection_method,
                'processed_by': result.processed_by
            }
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save detection result: {str(e)}")
            raise