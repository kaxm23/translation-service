from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from typing import List, Dict, Optional
import logging
from datetime import datetime
from tqdm import tqdm

class TextSummarizer:
    """
    Text summarization using BART model.
    Created by: kaxm23
    Created on: 2025-02-09 08:42:31 UTC
    """
    
    def __init__(self,
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None,
                 max_length: int = 1024,
                 min_length: int = 40,
                 log_level: int = logging.INFO):
        """
        Initialize the summarizer.
        
        Args:
            model_name: Name of the BART model
            device: Device to use (cuda/cpu)
            max_length: Maximum summary length
            min_length: Minimum summary length
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
        self.model_name = model_name
        self.max_length = max_length
        self.min_length = min_length
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'texts_processed': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'processing_time': 0,
            'last_summary': None
        }

    def _load_model(self) -> None:
        """Load the summarization model and tokenizer."""
        try:
            self.logger.info(f"Loading model: {self.model_name}")
            start_time = datetime.now()
            
            # Load tokenizer and model
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            load_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def summarize(self,
                 text: str,
                 max_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 early_stopping: bool = True) -> str:
        """
        Summarize a single text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            early_stopping: Whether to stop early
            
        Returns:
            str: Summarized text
        """
        start_time = datetime.now()
        
        try:
            # Set lengths
            max_length = max_length or self.max_length
            min_length = min_length or self.min_length
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            # Update statistics
            self._update_statistics(
                input_tokens=len(inputs["input_ids"][0]),
                output_tokens=len(summary_ids[0]),
                process_time=(datetime.now() - start_time).total_seconds()
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            raise

    def summarize_batch(self,
                       texts: List[str],
                       batch_size: int = 8,
                       show_progress: bool = True,
                       **kwargs) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            batch_size: Batch size for processing
            show_progress: Show progress bar
            **kwargs: Additional arguments for summarize()
            
        Returns:
            List[str]: List of summarized texts
        """
        summaries = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if show_progress:
                    self.logger.info(f"Processing batch {i//batch_size + 1}")
                
                # Process each text in batch
                batch_summaries = [
                    self.summarize(text, **kwargs)
                    for text in (tqdm(batch) if show_progress else batch)
                ]
                
                summaries.extend(batch_summaries)
            
            return summaries
            
        except Exception as e:
            self.logger.error(f"Batch summarization failed: {str(e)}")
            raise

    def _update_statistics(self,
                          input_tokens: int,
                          output_tokens: int,
                          process_time: float) -> None:
        """Update summarization statistics."""
        self.stats['texts_processed'] += 1
        self.stats['total_input_tokens'] += input_tokens
        self.stats['total_output_tokens'] += output_tokens
        self.stats['processing_time'] += process_time
        self.stats['last_summary'] = datetime.utcnow().strftime(
            '%Y-%m-%d %H:%M:%S UTC'
        )

    def get_statistics(self) -> Dict:
        """
        Get summarization statistics.
        
        Returns:
            Dict: Summarization statistics
        """
        return {
            'texts_processed': self.stats['texts_processed'],
            'total_input_tokens': self.stats['total_input_tokens'],
            'total_output_tokens': self.stats['total_output_tokens'],
            'average_compression_ratio': (
                self.stats['total_output_tokens'] / self.stats['total_input_tokens']
                if self.stats['total_input_tokens'] > 0 else 0
            ),
            'average_time_per_text': (
                self.stats['processing_time'] / self.stats['texts_processed']
                if self.stats['texts_processed'] > 0 else 0
            ),
            'device': self.device,
            'model': self.model_name,
            'last_summary': self.stats['last_summary'],
            'processed_by': 'kaxm23'
        }

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            self.logger.info("GPU memory cleared")