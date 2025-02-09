from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration

class GrammarCorrector:
    """
    Grammar correction using Gramformer or LanguageTool.
    Created by: kaxm23
    Created on: 2025-02-09 09:06:47 UTC
    """
    
    def __init__(self,
                 tool: str = "gramformer",
                 model_name: str = "prithivida/grammar_error_correcter_v1",
                 device: Optional[str] = None,
                 log_level: int = logging.INFO):
        """
        Initialize grammar corrector.
        
        Args:
            tool: Tool to use ('gramformer' or 'languagetool')
            model_name: Model name for Gramformer
            device: Device to use (cuda/cpu)
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Set parameters
        self.tool = tool.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tool
        if self.tool == "gramformer":
            self._init_gramformer(model_name)
        elif self.tool == "languagetool":
            self._init_languagetool()
        else:
            raise ValueError(f"Unsupported tool: {tool}")
        
        # Initialize statistics
        self.stats = {
            'corrections_made': 0,
            'texts_processed': 0,
            'total_time': 0,
            'last_correction': None
        }
        
        self.logger.info(f"Grammar corrector initialized using {tool}")

    def _init_gramformer(self, model_name: str):
        """Initialize Gramformer."""
        try:
            from gramformer import Gramformer
            self.model = Gramformer(
                models=1,  # 1 for grammar correction
                use_gpu=self.device == "cuda"
            )
            self.logger.info("Gramformer initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gramformer: {str(e)}")
            raise

    def _init_languagetool(self):
        """Initialize LanguageTool."""
        try:
            import language_tool_python
            self.model = language_tool_python.LanguageTool('en-US')
            self.logger.info("LanguageTool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LanguageTool: {str(e)}")
            raise

    async def correct_text(self,
                          text: str,
                          return_all: bool = False,
                          **kwargs) -> Dict:
        """
        Correct grammar in text.
        
        Args:
            text: Text to correct
            return_all: Return all corrections (Gramformer only)
            **kwargs: Additional arguments for specific tools
            
        Returns:
            Dict: Correction results and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            if self.tool == "gramformer":
                corrections = list(self.model.correct(text, **kwargs))
                corrected_text = corrections[0] if corrections else text
                all_corrections = corrections if return_all else None
                
            else:  # LanguageTool
                matches = self.model.check(text)
                corrected_text = self.model.correct(text)
                all_corrections = [
                    {
                        'message': match.message,
                        'replacements': match.replacements,
                        'offset': match.offset,
                        'length': match.length,
                        'rule': match.ruleId
                    }
                    for match in matches
                ] if return_all else None
            
            # Calculate metrics
            process_time = (datetime.utcnow() - start_time).total_seconds()
            changes_made = self._count_differences(text, corrected_text)
            
            # Update statistics
            self._update_statistics(changes_made, process_time)
            
            return {
                'original_text': text,
                'corrected_text': corrected_text,
                'changes_made': changes_made,
                'all_corrections': all_corrections,
                'processing_time': process_time,
                'tool_used': self.tool,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'processed_by': 'kaxm23'
            }
            
        except Exception as e:
            self.logger.error(f"Correction failed: {str(e)}")
            raise

    async def correct_batch(self,
                          texts: List[str],
                          **kwargs) -> List[Dict]:
        """
        Correct grammar in multiple texts.
        
        Args:
            texts: List of texts to correct
            **kwargs: Additional arguments for correct_text()
            
        Returns:
            List[Dict]: List of correction results
        """
        try:
            results = []
            
            for text in texts:
                result = await self.correct_text(text, **kwargs)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch correction failed: {str(e)}")
            raise

    def _count_differences(self,
                         original: str,
                         corrected: str) -> int:
        """Count number of corrections made."""
        if original == corrected:
            return 0
            
        # Simple word-based difference count
        original_words = set(original.split())
        corrected_words = set(corrected.split())
        return len(original_words.symmetric_difference(corrected_words))

    def _update_statistics(self,
                         changes: int,
                         process_time: float) -> None:
        """Update correction statistics."""
        self.stats['corrections_made'] += changes
        self.stats['texts_processed'] += 1
        self.stats['total_time'] += process_time
        self.stats['last_correction'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get correction statistics.
        
        Returns:
            Dict: Correction statistics
        """
        return {
            'corrections_made': self.stats['corrections_made'],
            'texts_processed': self.stats['texts_processed'],
            'average_corrections': (
                self.stats['corrections_made'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0 else 0
            ),
            'average_time': (
                self.stats['total_time'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0 else 0
            ),
            'tool_used': self.tool,
            'device': self.device,
            'last_correction': self.stats['last_correction'],
            'processed_by': 'kaxm23'
        }