import re
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

class ChatExtractor:
    """
    Extract chat-style messages from PDFs using regex.
    Created by: kaxm23
    Created on: 2025-02-09 09:17:30 UTC
    """
    
    def __init__(self,
                 custom_patterns: Optional[Dict[str, str]] = None,
                 log_level: int = logging.INFO):
        """
        Initialize chat extractor.
        
        Args:
            custom_patterns: Additional regex patterns
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Common chat patterns
        self.patterns = {
            'timestamp': r'(?P<timestamp>\d{1,2}[/.:-]\d{1,2}[/.:-]\d{2,4}|\d{1,2}:\d{2}(?:\s*(?:AM|PM))?)',
            'username': r'(?P<username>[@\w\s]+?)[:>]',
            'message': r'(?P<message>[^<>]+?)(?=\n|$)',
            'system_message': r'\*\*(?P<system_message>.*?)\*\*',
            'emoji': r'(?P<emoji>[😀-🙏🌀-🗿]|:[a-z_]+:)',
            'mention': r'@(?P<mention>\w+)',
            'url': r'(?P<url>https?://\S+)',
            'reaction': r'(?P<reaction>[👍👎❤️😂😮😡]+)',
            'file': r'(?P<file>📎\s*[\w\s.-]+\.\w+)',
        }
        
        # Add custom patterns
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        # Compiled regex patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.MULTILINE | re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        
        # Combined chat message pattern
        self.chat_pattern = re.compile(
            rf"{self.patterns['timestamp']}\s*"
            rf"{self.patterns['username']}\s*"
            rf"{self.patterns['message']}",
            re.MULTILINE | re.IGNORECASE
        )
        
        # Initialize statistics
        self.stats = {
            'files_processed': 0,
            'messages_extracted': 0,
            'total_pages': 0,
            'last_processed': "2025-02-09 09:17:30"
        }

    async def extract_from_pdf(self,
                             pdf_path: str,
                             output_format: str = 'dict',
                             page_range: Optional[Tuple[int, int]] = None) -> Dict:
        """
        Extract chat messages from PDF.
        
        Args:
            pdf_path: Path to PDF file
            output_format: Output format ('dict', 'dataframe', 'json')
            page_range: Optional page range to process
            
        Returns:
            Dict: Extraction results
        """
        try:
            start_time = datetime.utcnow()
            
            # Open PDF
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            
            # Set page range
            start_page = page_range[0] if page_range else 0
            end_page = min(page_range[1], total_pages) if page_range else total_pages
            
            # Extract text and messages
            messages = []
            page_stats = []
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                text = page.get_text()
                
                # Extract messages from page
                page_messages = self._extract_messages(text)
                messages.extend(page_messages)
                
                # Collect page statistics
                page_stats.append({
                    'page_number': page_num + 1,
                    'message_count': len(page_messages),
                    'has_system_messages': any(
                        msg.get('type') == 'system'
                        for msg in page_messages
                    )
                })
            
            # Process messages
            processed_messages = self._process_messages(messages)
            
            # Format output
            result = self._format_output(
                processed_messages,
                output_format
            )
            
            # Update statistics
            self._update_statistics(
                1,
                len(messages),
                total_pages
            )
            
            return {
                'messages': result,
                'metadata': {
                    'filename': Path(pdf_path).name,
                    'total_pages': total_pages,
                    'processed_pages': end_page - start_page,
                    'message_count': len(messages),
                    'page_statistics': page_stats,
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'timestamp': "2025-02-09 09:17:30",
                    'processed_by': "kaxm23"
                }
            }
            
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {str(e)}")
            raise
        
        finally:
            if 'doc' in locals():
                doc.close()

    def _extract_messages(self, text: str) -> List[Dict]:
        """Extract messages from text."""
        messages = []
        
        # Find chat messages
        for match in self.chat_pattern.finditer(text):
            message_dict = match.groupdict()
            
            # Extract additional elements
            message_dict.update(self._extract_message_elements(
                message_dict['message']
            ))
            
            # Determine message type
            message_dict['type'] = self._determine_message_type(
                message_dict
            )
            
            messages.append(message_dict)
        
        # Find system messages
        for match in self.compiled_patterns['system_message'].finditer(text):
            messages.append({
                'type': 'system',
                'message': match.group('system_message'),
                'timestamp': None,
                'username': 'SYSTEM'
            })
        
        return messages

    def _extract_message_elements(self, message: str) -> Dict:
        """Extract additional message elements."""
        elements = {
            'emojis': [],
            'mentions': [],
            'urls': [],
            'reactions': [],
            'files': []
        }
        
        # Extract emojis
        emojis = self.compiled_patterns['emoji'].finditer(message)
        elements['emojis'] = [m.group('emoji') for m in emojis]
        
        # Extract mentions
        mentions = self.compiled_patterns['mention'].finditer(message)
        elements['mentions'] = [m.group('mention') for m in mentions]
        
        # Extract URLs
        urls = self.compiled_patterns['url'].finditer(message)
        elements['urls'] = [m.group('url') for m in urls]
        
        # Extract reactions
        reactions = self.compiled_patterns['reaction'].finditer(message)
        elements['reactions'] = [m.group('reaction') for m in reactions]
        
        # Extract files
        files = self.compiled_patterns['file'].finditer(message)
        elements['files'] = [m.group('file') for m in files]
        
        return elements

    def _determine_message_type(self, message_dict: Dict) -> str:
        """Determine message type."""
        if message_dict.get('system_message'):
            return 'system'
        elif message_dict.get('files'):
            return 'file'
        elif message_dict.get('urls'):
            return 'link'
        elif message_dict.get('reactions'):
            return 'reaction'
        else:
            return 'message'

    def _process_messages(self, messages: List[Dict]) -> List[Dict]:
        """Process and clean extracted messages."""
        processed = []
        
        for msg in messages:
            # Clean timestamp
            if msg.get('timestamp'):
                msg['timestamp'] = self._normalize_timestamp(
                    msg['timestamp']
                )
            
            # Clean username
            if msg.get('username'):
                msg['username'] = msg['username'].strip(': >')
            
            # Clean message
            if msg.get('message'):
                msg['message'] = msg['message'].strip()
            
            # Add metadata
            msg['processed_at'] = "2025-02-09 09:17:30"
            msg['processor'] = "kaxm23"
            
            processed.append(msg)
        
        return processed

    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize timestamp format."""
        try:
            # Common timestamp patterns
            patterns = [
                '%d/%m/%y %H:%M',
                '%d-%m-%y %H:%M',
                '%H:%M',
                '%I:%M %p'
            ]
            
            for pattern in patterns:
                try:
                    dt = datetime.strptime(timestamp, pattern)
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
            
            return timestamp
            
        except Exception:
            return timestamp

    def _format_output(self,
                      messages: List[Dict],
                      format_type: str) -> object:
        """Format output in requested format."""
        if format_type == 'dataframe':
            return pd.DataFrame(messages)
        elif format_type == 'json':
            return pd.DataFrame(messages).to_json(orient='records')
        else:
            return messages

    def _update_statistics(self,
                         files: int,
                         messages: int,
                         pages: int) -> None:
        """Update extraction statistics."""
        self.stats['files_processed'] += files
        self.stats['messages_extracted'] += messages
        self.stats['total_pages'] += pages
        self.stats['last_processed'] = "2025-02-09 09:17:30"

    def get_statistics(self) -> Dict:
        """Get extraction statistics."""
        return {
            'files_processed': self.stats['files_processed'],
            'messages_extracted': self.stats['messages_extracted'],
            'total_pages': self.stats['total_pages'],
            'average_messages_per_page': (
                self.stats['messages_extracted'] / self.stats['total_pages']
                if self.stats['total_pages'] > 0 else 0
            ),
            'last_processed': self.stats['last_processed'],
            'processed_by': "kaxm23"
        }

    def add_custom_pattern(self,
                          name: str,
                          pattern: str) -> None:
        """Add custom regex pattern."""
        try:
            # Validate pattern
            re.compile(pattern)
            
            # Add pattern
            self.patterns[name] = pattern
            self.compiled_patterns[name] = re.compile(
                pattern,
                re.MULTILINE | re.IGNORECASE
            )
            
            self.logger.info(f"Added custom pattern: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to add pattern: {str(e)}")
            raise