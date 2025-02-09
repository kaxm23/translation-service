from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime
import openai
from app.core.config import Settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()
settings = Settings()

class AlternativeStyle(BaseModel):
    """Alternative translation style configuration."""
    name: str
    description: str
    tone: str
    formality_level: str
    temperature: float

class AlternativeTranslation(BaseModel):
    """Alternative translation result."""
    style: str
    text: str
    tone: str
    formality_level: str
    confidence_score: float
    explanation: str

class TranslationRequest(BaseModel):
    """Translation request with alternatives."""
    text: str = Field(..., example="The proposal was well received.")
    source_lang: str = Field(..., example="en")
    target_lang: str = Field(..., example="es")
    document_type: Literal["legal", "medical", "academic", "general"] = "general"
    required_styles: Optional[List[str]] = Field(
        None,
        example=["formal", "casual", "technical"]
    )

class TranslationResponse(BaseModel):
    """Enhanced translation response with alternatives."""
    original_text: str
    source_lang: str
    target_lang: str
    document_type: str
    primary_translation: str
    alternatives: List[AlternativeTranslation]
    tokens_used: int
    processing_time: float
    timestamp: str = "2025-02-09 09:16:15"
    processed_by: str = "kaxm23"

class AlternativeTranslator:
    """
    Translation service with alternatives.
    Created by: kaxm23
    Created on: 2025-02-09 09:16:15
    """
    
    def __init__(self):
        """Initialize alternative translator."""
        self.translation_styles = {
            "formal": AlternativeStyle(
                name="formal",
                description="Formal and professional language",
                tone="professional",
                formality_level="high",
                temperature=0.3
            ),
            "casual": AlternativeStyle(
                name="casual",
                description="Casual and conversational language",
                tone="friendly",
                formality_level="low",
                temperature=0.7
            ),
            "technical": AlternativeStyle(
                name="technical",
                description="Technical and precise language",
                tone="technical",
                formality_level="high",
                temperature=0.2
            ),
            "literary": AlternativeStyle(
                name="literary",
                description="Literary and expressive language",
                tone="artistic",
                formality_level="medium",
                temperature=0.8
            ),
            "simplified": AlternativeStyle(
                name="simplified",
                description="Simple and clear language",
                tone="straightforward",
                formality_level="low",
                temperature=0.4
            )
        }
        
        self.document_prompts = {
            "legal": "Maintain legal terminology and formal structure.",
            "medical": "Use precise medical terms and clinical language.",
            "academic": "Employ scholarly language and technical precision.",
            "general": "Use natural and appropriate language."
        }
        
        self.stats = {
            'translations': 0,
            'alternatives_generated': 0,
            'tokens_used': 0,
            'last_translation': "2025-02-09 09:16:15"
        }

    async def translate_with_alternatives(self,
                                       text: str,
                                       source_lang: str,
                                       target_lang: str,
                                       document_type: str,
                                       required_styles: Optional[List[str]] = None) -> Dict:
        """Generate primary translation and alternatives."""
        try:
            start_time = datetime.utcnow()
            total_tokens = 0
            
            # Select styles
            styles_to_use = (
                required_styles if required_styles
                else list(self.translation_styles.keys())[:3]
            )
            
            # Generate translations
            translations = []
            for style_name in styles_to_use:
                style = self.translation_styles[style_name]
                
                # Create specialized prompt
                system_prompt = self._create_translation_prompt(
                    source_lang,
                    target_lang,
                    document_type,
                    style
                )
                
                # Generate translation
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text}
                    ],
                    temperature=style.temperature,
                    n=1
                )
                
                translation = response.choices[0].message.content
                total_tokens += response.usage.total_tokens
                
                # Get explanation
                explanation = await self._generate_style_explanation(
                    text,
                    translation,
                    style,
                    source_lang,
                    target_lang
                )
                
                translations.append(
                    AlternativeTranslation(
                        style=style.name,
                        text=translation,
                        tone=style.tone,
                        formality_level=style.formality_level,
                        confidence_score=self._calculate_confidence(response),
                        explanation=explanation
                    )
                )
            
            # Update statistics
            self._update_statistics(len(translations), total_tokens)
            
            return {
                'original_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'document_type': document_type,
                'primary_translation': translations[0].text,
                'alternatives': translations,
                'tokens_used': total_tokens,
                'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                'timestamp': "2025-02-09 09:16:15",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Alternative translation generation failed: {str(e)}"
            )

    def _create_translation_prompt(self,
                                 source_lang: str,
                                 target_lang: str,
                                 document_type: str,
                                 style: AlternativeStyle) -> str:
        """Create specialized translation prompt."""
        return (
            f"You are a professional translator specializing in {style.tone} translations. "
            f"Translate from {source_lang} to {target_lang} using {style.description}. "
            f"{self.document_prompts[document_type]} "
            f"Maintain a {style.formality_level} formality level. "
            f"Focus on natural expression while preserving the original meaning."
        )

    async def _generate_style_explanation(self,
                                        source_text: str,
                                        translated_text: str,
                                        style: AlternativeStyle,
                                        source_lang: str,
                                        target_lang: str) -> str:
        """Generate explanation of translation choices."""
        try:
            prompt = (
                f"Explain the key stylistic choices in this {style.name} translation:\n"
                f"Source ({source_lang}): {source_text}\n"
                f"Translation ({target_lang}): {translated_text}\n"
                f"Focus on tone ({style.tone}) and formality level ({style.formality_level})."
            )
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a translation style analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Style analysis unavailable: {str(e)}"

    def _calculate_confidence(self, response) -> float:
        """Calculate confidence score for translation."""
        # Base confidence on response characteristics
        base_score = 0.85
        
        if response.choices[0].finish_reason == "stop":
            base_score += 0.1
        
        # Add additional factors here
        
        return min(1.0, base_score)

    def _update_statistics(self, num_alternatives: int, tokens: int):
        """Update usage statistics."""
        self.stats['translations'] += 1
        self.stats['alternatives_generated'] += num_alternatives
        self.stats['tokens_used'] += tokens
        self.stats['last_translation'] = "2025-02-09 09:16:15"

    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        return {
            'total_translations': self.stats['translations'],
            'total_alternatives': self.stats['alternatives_generated'],
            'average_alternatives': (
                self.stats['alternatives_generated'] / self.stats['translations']
                if self.stats['translations'] > 0 else 0
            ),
            'total_tokens': self.stats['tokens_used'],
            'last_translation': self.stats['last_translation'],
            'available_styles': list(self.translation_styles.keys()),
            'processed_by': "kaxm23"
        }

translator_service = AlternativeTranslator()

@router.post("/translate/alternatives/",
             response_model=TranslationResponse,
             description="Translate text with multiple style alternatives")
async def get_alternative_translations(
    request: TranslationRequest,
    current_user: User = Depends(get_current_user)
) -> TranslationResponse:
    """
    Translate text with multiple style alternatives.
    """
    result = await translator_service.translate_with_alternatives(
        request.text,
        request.source_lang,
        request.target_lang,
        request.document_type,
        request.required_styles
    )
    
    return TranslationResponse(**result)

@router.get("/styles/",
            description="Get available translation styles")
async def get_available_styles(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get available translation styles."""
    return {
        'styles': [
            {
                'name': style.name,
                'description': style.description,
                'tone': style.tone,
                'formality_level': style.formality_level
            }
            for style in translator_service.translation_styles.values()
        ],
        'timestamp': "2025-02-09 09:16:15",
        'processed_by': "kaxm23"
    }

@router.get("/stats/",
            description="Get translation service statistics")
async def get_translation_stats(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get translation service statistics."""
    return translator_service.get_statistics()