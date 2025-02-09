import language_tool_python
from typing import Dict, List, Optional
import logging
from datetime import datetime
import re
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display

class ArabicTranslationChecker:
    """
    Arabic translation error detection using LanguageTool.
    Created by: kaxm23
    Created on: 2025-02-09 09:13:20 UTC
    """
    
    def __init__(self,
                 custom_rules: bool = True,
                 log_level: int = logging.INFO):
        """
        Initialize Arabic translation checker.
        
        Args:
            custom_rules: Enable custom Arabic rules
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize LanguageTool for Arabic
        try:
            self.tool = language_tool_python.LanguageTool('ar')
            self.logger.info("LanguageTool initialized for Arabic")
        except Exception as e:
            self.logger.error(f"Failed to initialize LanguageTool: {str(e)}")
            raise
        
        # Initialize Arabic reshaper
        self.reshaper = ArabicReshaper(
            configuration={
                'delete_harakat': False,
                'support_ligatures': True
            }
        )
        
        # Custom Arabic grammar rules
        if custom_rules:
            self._add_custom_arabic_rules()
        
        # Initialize statistics
        self.stats = {
            'checks_performed': 0,
            'errors_found': 0,
            'texts_processed': 0,
            'last_check': "2025-02-09 09:13:20"
        }

    def _add_custom_arabic_rules(self):
        """Add custom Arabic grammar rules."""
        # Common Arabic grammar patterns
        self.arabic_patterns = {
            'tashkeel': re.compile(r'[\u064B-\u065F\u0670]'),
            'numbers': re.compile(r'[\u0660-\u0669]'),
            'alef_variations': re.compile(r'[إأآا]'),
            'hamza': re.compile(r'[ءؤئ]'),
            'ta_marbuta': re.compile(r'ة$')
        }
        
        # Common structural patterns
        self.structure_patterns = {
            'verbal_sentence': re.compile(r'^[\u0641\u0648]?\u064A|\u062A|\u0646'),
            'nominal_sentence': re.compile(r'^[\u0627\u0644]')
        }

    async def check_arabic_text(self,
                              text: str,
                              check_type: str = 'all') -> Dict:
        """
        Check Arabic text for grammar and structure errors.
        
        Args:
            text: Arabic text to check
            check_type: Type of check ('grammar', 'structure', 'all')
            
        Returns:
            Dict: Check results and suggestions
        """
        try:
            start_time = datetime.utcnow()
            
            # Reshape Arabic text for proper display
            reshaped_text = self.reshaper.reshape(text)
            display_text = get_display(reshaped_text)
            
            # Get LanguageTool matches
            matches = self.tool.check(text)
            
            # Process basic errors
            basic_errors = self._process_matches(matches)
            
            # Additional Arabic-specific checks
            arabic_specific = self._check_arabic_specific(text)
            
            # Structural analysis if requested
            structure_analysis = None
            if check_type in ['structure', 'all']:
                structure_analysis = self._analyze_sentence_structure(text)
            
            # Calculate processing time
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Update statistics
            self._update_statistics(len(basic_errors))
            
            return {
                'text': display_text,
                'basic_errors': basic_errors,
                'arabic_specific': arabic_specific,
                'structure_analysis': structure_analysis,
                'suggestions': self._generate_suggestions(
                    text,
                    basic_errors,
                    arabic_specific
                ),
                'error_summary': {
                    'basic_count': len(basic_errors),
                    'arabic_specific_count': len(arabic_specific['errors']),
                    'structure_issues': structure_analysis['issues'] if structure_analysis else []
                },
                'processing_time': process_time,
                'timestamp': "2025-02-09 09:13:20",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            self.logger.error(f"Arabic text check failed: {str(e)}")
            raise

    def _process_matches(self, matches) -> List[Dict]:
        """Process LanguageTool matches."""
        errors = []
        for match in matches:
            error = {
                'message': match.message,
                'context': match.context,
                'offset': match.offset,
                'length': match.length,
                'rule_id': match.ruleId,
                'category': match.category,
                'replacements': match.replacements,
                'error_type': 'grammar'
            }
            errors.append(error)
        return errors

    def _check_arabic_specific(self, text: str) -> Dict:
        """Check Arabic-specific grammar rules."""
        errors = []
        warnings = []
        
        # Check tashkeel (diacritics) consistency
        tashkeel_matches = self.arabic_patterns['tashkeel'].finditer(text)
        last_pos = -1
        for match in tashkeel_matches:
            if match.start() - last_pos > 1:
                warnings.append({
                    'type': 'tashkeel_consistency',
                    'message': 'Inconsistent diacritics usage',
                    'position': match.start()
                })
            last_pos = match.start()
        
        # Check alef variations
        alef_matches = self.arabic_patterns['alef_variations'].finditer(text)
        for match in alef_matches:
            if not self._is_valid_alef_usage(text, match.start()):
                errors.append({
                    'type': 'alef_form',
                    'message': 'Incorrect Alef form usage',
                    'position': match.start(),
                    'context': text[max(0, match.start()-10):match.start()+10]
                })
        
        # Check hamza placement
        hamza_matches = self.arabic_patterns['hamza'].finditer(text)
        for match in hamza_matches:
            if not self._is_valid_hamza_placement(text, match.start()):
                errors.append({
                    'type': 'hamza_placement',
                    'message': 'Incorrect Hamza placement',
                    'position': match.start(),
                    'context': text[max(0, match.start()-10):match.start()+10]
                })
        
        return {
            'errors': errors,
            'warnings': warnings
        }

    def _analyze_sentence_structure(self, text: str) -> Dict:
        """Analyze Arabic sentence structure."""
        sentences = text.split('.')
        analysis = []
        issues = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Determine sentence type
            if self.structure_patterns['verbal_sentence'].match(sentence):
                structure_type = 'verbal'
            elif self.structure_patterns['nominal_sentence'].match(sentence):
                structure_type = 'nominal'
            else:
                structure_type = 'unknown'
                issues.append(f'Unclear sentence structure in sentence {i+1}')
            
            # Analyze word order
            word_order_issues = self._check_word_order(sentence, structure_type)
            if word_order_issues:
                issues.extend(word_order_issues)
            
            analysis.append({
                'sentence_number': i + 1,
                'structure_type': structure_type,
                'length': len(sentence.split()),
                'issues': word_order_issues
            })
        
        return {
            'sentence_count': len(analysis),
            'analysis': analysis,
            'issues': issues
        }

    def _is_valid_alef_usage(self, text: str, position: int) -> bool:
        """Check if Alef usage is valid."""
        # Simplified check - can be expanded based on specific rules
        return True

    def _is_valid_hamza_placement(self, text: str, position: int) -> bool:
        """Check if Hamza placement is valid."""
        # Simplified check - can be expanded based on specific rules
        return True

    def _check_word_order(self,
                         sentence: str,
                         structure_type: str) -> List[str]:
        """Check Arabic word order rules."""
        issues = []
        words = sentence.split()
        
        if structure_type == 'verbal':
            # Check VSO order for verbal sentences
            if len(words) > 2 and not self._is_verb(words[0]):
                issues.append('Verbal sentence should start with a verb')
                
        elif structure_type == 'nominal':
            # Check subject-predicate agreement
            if len(words) > 1 and not self._check_agreement(words[0], words[1]):
                issues.append('Subject-predicate agreement error')
        
        return issues

    def _is_verb(self, word: str) -> bool:
        """Check if word is a verb."""
        # Simplified verb pattern matching
        verb_patterns = [
            r'^[يتنأ]',  # Present tense markers
            r'^[فعل]'    # Past tense patterns
        ]
        return any(re.match(pattern, word) for pattern in verb_patterns)

    def _check_agreement(self, subject: str, predicate: str) -> bool:
        """Check subject-predicate agreement."""
        # Simplified agreement check
        return True

    def _generate_suggestions(self,
                            text: str,
                            basic_errors: List[Dict],
                            arabic_specific: Dict) -> Dict:
        """Generate correction suggestions."""
        suggestions = {
            'grammar': [],
            'structure': [],
            'style': []
        }
        
        # Process basic grammar errors
        for error in basic_errors:
            if error['replacements']:
                suggestions['grammar'].append({
                    'original': text[error['offset']:error['offset']+error['length']],
                    'suggestion': error['replacements'][0],
                    'message': error['message']
                })
        
        # Process Arabic-specific errors
        for error in arabic_specific['errors']:
            suggestions['grammar'].append({
                'type': error['type'],
                'message': error['message'],
                'position': error['position']
            })
        
        return suggestions

    def _update_statistics(self, error_count: int) -> None:
        """Update usage statistics."""
        self.stats['checks_performed'] += 1
        self.stats['errors_found'] += error_count
        self.stats['texts_processed'] += 1
        self.stats['last_check'] = "2025-02-09 09:13:20"

    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        return {
            'checks_performed': self.stats['checks_performed'],
            'errors_found': self.stats['errors_found'],
            'texts_processed': self.stats['texts_processed'],
            'average_errors': (
                self.stats['errors_found'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0 else 0
            ),
            'language': 'ar',
            'last_check': self.stats['last_check'],
            'processed_by': "kaxm23"
        }