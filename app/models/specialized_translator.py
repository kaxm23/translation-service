from openai import AsyncOpenAI
from typing import Dict, List, Optional, Literal
import logging
from datetime import datetime
from pydantic import BaseModel

class DocumentType(BaseModel):
    """Document type configuration."""
    name: Literal['legal', 'medical', 'academic', 'general']
    style_prompt: str
    temperature: float
    key_terms_file: Optional[str] = None

class SpecializedTranslator:
    """
    Specialized GPT-4 translation for different document types.
    Created by: kaxm23
    Created on: 2025-02-09 08:59:58 UTC
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-4",
                 log_level: int = logging.INFO):
        """Initialize specialized translator."""
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
        # Define document types and their configurations
        self.document_types = {
            'legal': DocumentType(
                name='legal',
                style_prompt=(
                    "You are a legal translator with expertise in international law "
                    "and legal terminology. Maintain formal legal language and precise "
                    "terminology. Preserve any legal citations and formatting. Ensure "
                    "exact translation of legal terms and maintain the authoritative tone "
                    "of legal documents."
                ),
                temperature=0.1,
                key_terms_file="legal_terms.json"
            ),
            'medical': DocumentType(
                name='medical',
                style_prompt=(
                    "You are a medical translator with expertise in healthcare and "
                    "medical terminology. Use standardized medical terms and maintain "
                    "clinical accuracy. Preserve medical abbreviations where appropriate. "
                    "Ensure precise translation of anatomical terms, procedures, and "
                    "medical conditions."
                ),
                temperature=0.2,
                key_terms_file="medical_terms.json"
            ),
            'academic': DocumentType(
                name='academic',
                style_prompt=(
                    "You are an academic translator with expertise in scholarly writing. "
                    "Maintain formal academic tone and discipline-specific terminology. "
                    "Preserve citations and references. Ensure accurate translation of "
                    "technical terms and maintain the scholarly style."
                ),
                temperature=0.3,
                key_terms_file="academic_terms.json"
            ),
            'general': DocumentType(
                name='general',
                style_prompt=(
                    "You are a professional translator. Maintain natural language flow "
                    "while preserving the original meaning and tone. Adapt idioms and "
                    "cultural references appropriately for the target language."
                ),
                temperature=0.7
            )
        }
        
        # Load specialized terminology
        self._load_terminology()
        
        # Initialize statistics
        self.stats = {
            'requests': 0,
            'tokens': 0,
            'cost': 0.0,
            'document_types': {t: 0 for t in self.document_types.keys()}
        }

    def _load_terminology(self):
        """Load specialized terminology for different document types."""
        self.terminology = {}
        
        for doc_type, config in self.document_types.items():
            if config.key_terms_file:
                try:
                    with open(f"app/data/{config.key_terms_file}", 'r', encoding='utf-8') as f:
                        self.terminology[doc_type] = json.load(f)
                        self.logger.info(f"Loaded terminology for {doc_type}")
                except FileNotFoundError:
                    self.logger.warning(f"Terminology file not found for {doc_type}")
                    self.terminology[doc_type] = {}

    async def translate(self,
                       text: str,
                       source_lang: str,
                       target_lang: str,
                       document_type: str = 'general',
                       context: Optional[Dict] = None) -> Dict:
        """
        Translate text with specialized handling based on document type.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            document_type: Type of document (legal/medical/academic/general)
            context: Additional context for translation
            
        Returns:
            Dict: Translation results and metadata
        """
        try:
            start_time = datetime.utcnow()
            
            # Validate document type
            if document_type not in self.document_types:
                raise ValueError(f"Unsupported document type: {document_type}")
            
            # Get document type configuration
            doc_config = self.document_types[document_type]
            
            # Prepare system prompt
            system_prompt = self._create_system_prompt(
                doc_config,
                source_lang,
                target_lang,
                context
            )
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Add terminology if available
            if document_type in self.terminology and self.terminology[document_type]:
                messages.append({
                    "role": "system",
                    "content": "Use these specialized terms: " + 
                             json.dumps(self.terminology[document_type])
                })
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=doc_config.temperature,
                max_tokens=2000
            )
            
            # Extract translation
            translation = response.choices[0].message.content
            
            # Update statistics
            process_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_statistics(
                document_type,
                response.usage.total_tokens,
                process_time
            )
            
            return {
                'translation': translation,
                'document_type': document_type,
                'model': self.model,
                'tokens': response.usage.total_tokens,
                'processing_time': process_time,
                'confidence': self._calculate_confidence(response),
                'terminology_used': bool(self.terminology.get(document_type)),
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
                'processed_by': 'kaxm23'
            }
            
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            raise

    def _create_system_prompt(self,
                            doc_config: DocumentType,
                            source_lang: str,
                            target_lang: str,
                            context: Optional[Dict]) -> str:
        """Create specialized system prompt."""
        prompt = (
            f"{doc_config.style_prompt}\n\n"
            f"Translate from {source_lang} to {target_lang}."
        )
        
        if context:
            if 'audience' in context:
                prompt += f"\nTarget audience: {context['audience']}"
            if 'purpose' in context:
                prompt += f"\nDocument purpose: {context['purpose']}"
            if 'specialty' in context:
                prompt += f"\nSpecialty area: {context['specialty']}"
        
        return prompt

    def _calculate_confidence(self, response) -> float:
        """Calculate confidence score for translation."""
        # Implement confidence calculation based on model output
        # This is a simplified example
        return 0.95 if response.choices[0].finish_reason == "stop" else 0.7

    def _update_statistics(self,
                         document_type: str,
                         tokens: int,
                         process_time: float) -> None:
        """Update translation statistics."""
        self.stats['requests'] += 1
        self.stats['tokens'] += tokens
        self.stats['document_types'][document_type] += 1
        
        # Calculate cost based on model and tokens
        cost_per_token = 0.00003  # Adjust based on your model
        self.stats['cost'] += tokens * cost_per_token

    def get_statistics(self) -> Dict:
        """Get translation statistics."""
        return {
            'total_requests': self.stats['requests'],
            'total_tokens': self.stats['tokens'],
            'total_cost': f"${self.stats['cost']:.4f}",
            'document_type_distribution': {
                k: v/self.stats['requests'] if self.stats['requests'] > 0 else 0
                for k, v in self.stats['document_types'].items()
            },
            'average_tokens_per_request': (
                self.stats['tokens'] / self.stats['requests']
                if self.stats['requests'] > 0 else 0
            ),
            'model': self.model,
            'supported_types': list(self.document_types.keys()),
            'last_updated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
            'processed_by': 'kaxm23'
        }