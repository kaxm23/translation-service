from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import pypdf
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

class PDFSummarizer:
    """
    PDF text summarization using BART model.
    Created by: kaxm23
    Created on: 2025-02-09 08:44:17 UTC
    """
    
    def __init__(self,
                 model_name: str = "facebook/bart-large-cnn",
                 device: Optional[str] = None,
                 chunk_size: int = 1024,
                 overlap_size: int = 100,
                 log_level: int = logging.INFO):
        """
        Initialize the PDF summarizer.
        
        Args:
            model_name: Name of the BART model
            device: Device to use (cuda/cpu)
            chunk_size: Maximum tokens per chunk
            overlap_size: Overlap size between chunks
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
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize statistics
        self.stats = {
            'pdfs_processed': 0,
            'pages_processed': 0,
            'chunks_processed': 0,
            'total_tokens': 0,
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

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple[str, int]: Extracted text and page count
        """
        try:
            self.logger.info(f"Extracting text from: {pdf_path}")
            
            with open(pdf_path, 'rb') as file:
                # Create PDF reader
                pdf = pypdf.PdfReader(file)
                page_count = len(pdf.pages)
                
                # Extract text from all pages
                text = []
                for page in tqdm(pdf.pages, desc="Extracting pages"):
                    text.append(page.extract_text())
                
                full_text = ' '.join(text)
                self.logger.info(f"Extracted {page_count} pages")
                
                return full_text, page_count
                
        except Exception as e:
            self.logger.error(f"Failed to extract text: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List[str]: List of text chunks
        """
        try:
            # Tokenize full text
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            # Create chunks with overlap
            for i in range(0, len(tokens), self.chunk_size - self.overlap_size):
                chunk = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk, skip_special_tokens=True)
                chunks.append(chunk_text)
            
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to chunk text: {str(e)}")
            raise

    def summarize_chunk(self,
                       chunk: str,
                       max_length: int = 150,
                       min_length: int = 40,
                       num_beams: int = 4) -> str:
        """
        Summarize a single text chunk.
        
        Args:
            chunk: Text chunk to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Summarized text
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                chunk,
                max_length=self.chunk_size,
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
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to summarize chunk: {str(e)}")
            raise

    def combine_summaries(self, summaries: List[str]) -> str:
        """
        Combine chunk summaries into final summary.
        
        Args:
            summaries: List of chunk summaries
            
        Returns:
            str: Combined summary
        """
        try:
            # Join summaries
            combined = ' '.join(summaries)
            
            # Create final summary
            final_summary = self.summarize_chunk(
                combined,
                max_length=250,
                min_length=100
            )
            
            return final_summary
            
        except Exception as e:
            self.logger.error(f"Failed to combine summaries: {str(e)}")
            raise

    def summarize_pdf(self,
                     pdf_path: str,
                     save_summary: bool = True,
                     output_dir: Optional[str] = None) -> Dict:
        """
        Summarize PDF document.
        
        Args:
            pdf_path: Path to PDF file
            save_summary: Whether to save summary to file
            output_dir: Directory to save summary
            
        Returns:
            Dict: Summary and statistics
        """
        start_time = datetime.now()
        
        try:
            # Extract text from PDF
            text, page_count = self.extract_text_from_pdf(pdf_path)
            
            # Split into chunks
            chunks = self.chunk_text(text)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in tqdm(chunks, desc="Summarizing chunks"):
                summary = self.summarize_chunk(chunk)
                chunk_summaries.append(summary)
            
            # Combine summaries
            final_summary = self.combine_summaries(chunk_summaries)
            
            # Update statistics
            process_time = (datetime.now() - start_time).total_seconds()
            self._update_statistics(
                pages=page_count,
                chunks=len(chunks),
                tokens=len(self.tokenizer.encode(text)),
                process_time=process_time
            )
            
            # Save summary if requested
            if save_summary:
                self._save_summary(
                    final_summary,
                    pdf_path,
                    output_dir
                )
            
            return {
                'summary': final_summary,
                'statistics': {
                    'pages': page_count,
                    'chunks': len(chunks),
                    'processing_time': process_time,
                    'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to summarize PDF: {str(e)}")
            raise

    def _save_summary(self,
                     summary: str,
                     pdf_path: str,
                     output_dir: Optional[str] = None) -> None:
        """Save summary to file."""
        try:
            # Create output directory
            output_dir = output_dir or Path(pdf_path).parent
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Create output path
            pdf_name = Path(pdf_path).stem
            output_path = Path(output_dir) / f"{pdf_name}_summary.txt"
            
            # Save summary
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
                
            self.logger.info(f"Summary saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save summary: {str(e)}")
            raise

    def _update_statistics(self,
                          pages: int,
                          chunks: int,
                          tokens: int,
                          process_time: float) -> None:
        """Update summarization statistics."""
        self.stats['pdfs_processed'] += 1
        self.stats['pages_processed'] += pages
        self.stats['chunks_processed'] += chunks
        self.stats['total_tokens'] += tokens
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
            'pdfs_processed': self.stats['pdfs_processed'],
            'pages_processed': self.stats['pages_processed'],
            'chunks_processed': self.stats['chunks_processed'],
            'total_tokens': self.stats['total_tokens'],
            'average_time_per_pdf': (
                self.stats['processing_time'] / self.stats['pdfs_processed']
                if self.stats['pdfs_processed'] > 0 else 0
            ),
            'average_tokens_per_page': (
                self.stats['total_tokens'] / self.stats['pages_processed']
                if self.stats['pages_processed'] > 0 else 0
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