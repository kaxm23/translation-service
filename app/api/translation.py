from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime
import torch
from gramformer import Gramformer
import openai
from app.core.config import Settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()
settings = Settings()

class GrammarAwareTranslator:
    """
    Translation service with grammar pre-correction.
    Created by: kaxm23
    Created on: 2025-02-09 09:09:07 UTC
    """
    
    def __init__(self):
        """Initialize translation service."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._init_grammar_corrector()
        openai.api_key = settings.OPENAI_API_KEY
        
        self.stats = {
            'translations': 0,
            'grammar_corrections': 0,
            'tokens': 0,
            'cost': 0.0
        }

    def _init_grammar_corrector(self):
        """Initialize Gramformer."""
        try:
            self.grammar_corrector = Gramformer(
                models=1,
                use_gpu=self.device == "cuda"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gramformer: {str(e)}")

translator_service = GrammarAwareTranslator()

class TextCorrection(BaseModel):
    """Grammar correction details."""
    original_text: str
    corrected_text: str
    changes_made: List[Dict[str, str]]
    confidence: float

class TranslationRequest(BaseModel):
    """Translation request model."""
    text: str = Field(..., example="I has went to the store yesterday.")
    source_lang: str = Field(..., example="en")
    target_lang: str = Field(..., example="es")
    document_type: Literal["legal", "medical", "academic", "general"] = Field(
        "general",
        example="general"
    )
    check_grammar: bool = Field(True, example=True)
    show_corrections: bool = Field(False, example=True)
    preserve_formatting: bool = Field(True, example=True)

class TranslationResponse(BaseModel):
    """Translation response model."""
    original_text: str
    grammar_check: Optional[TextCorrection]
    translated_text: str
    source_lang: str
    target_lang: str
    document_type: str
    confidence_score: float
    tokens_used: int
    processing_time: float
    processed_by: str = "kaxm23"
    timestamp: str = "2025-02-09 09:09:07"

async def correct_grammar(text: str) -> TextCorrection:
    """
    Correct grammar using Gramformer.
    
    Args:
        text: Text to correct
        
    Returns:
        TextCorrection: Grammar correction results
    """
    try:
        # Get corrections
        corrections = list(translator_service.grammar_corrector.correct(text))
        
        if not corrections:
            return TextCorrection(
                original_text=text,
                corrected_text=text,
                changes_made=[],
                confidence=1.0
            )
        
        # Use best correction
        corrected_text = corrections[0]
        
        # Identify changes
        changes = []
        original_words = text.split()
        corrected_words = corrected_text.split()
        
        for i, (orig, corr) in enumerate(zip(original_words, corrected_words)):
            if orig != corr:
                changes.append({
                    'position': i,
                    'original': orig,
                    'correction': corr,
                    'type': 'word_replacement'
                })
        
        # Calculate confidence
        confidence = 0.95 if changes else 1.0
        
        return TextCorrection(
            original_text=text,
            corrected_text=corrected_text,
            changes_made=changes,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Grammar correction failed: {str(e)}"
        )

async def translate_text(text: str,
                        source_lang: str,
                        target_lang: str,
                        document_type: str,
                        **kwargs) -> Dict:
    """
    Translate text using GPT-4.
    
    Args:
        text: Text to translate
        source_lang: Source language
        target_lang: Target language
        document_type: Type of document
        **kwargs: Additional parameters
        
    Returns:
        Dict: Translation results
    """
    try:
        # Create system prompt based on document type
        system_prompt = get_translation_prompt(
            document_type,
            source_lang,
            target_lang
        )
        
        # Make API call
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            **kwargs
        )
        
        return {
            'translation': response.choices[0].message.content,
            'tokens': response.usage.total_tokens,
            'confidence': 0.95 if response.choices[0].finish_reason == "stop" else 0.7
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

def get_translation_prompt(document_type: str,
                         source_lang: str,
                         target_lang: str) -> str:
    """Get specialized translation prompt."""
    base_prompt = (
        f"You are a professional translator. Translate from {source_lang} "
        f"to {target_lang}. "
    )
    
    if document_type == "legal":
        base_prompt += (
            "Maintain legal terminology accuracy and formal tone. "
            "Preserve any legal citations and formatting."
        )
    elif document_type == "medical":
        base_prompt += (
            "Use precise medical terminology and maintain clinical accuracy. "
            "Preserve medical abbreviations where appropriate."
        )
    elif document_type == "academic":
        base_prompt += (
            "Maintain academic tone and technical precision. "
            "Preserve citations and references."
        )
    else:
        base_prompt += (
            "Maintain natural language flow while preserving the original "
            "meaning and tone."
        )
    
    return base_prompt

@router.post("/translate/",
            response_model=TranslationResponse,
            description="Translate text with optional grammar correction")
async def translate_with_grammar(
    request: TranslationRequest,
    current_user: User = Depends(get_current_user)
) -> TranslationResponse:
    """
    Translate text with optional grammar correction.
    """
    try:
        start_time = datetime.utcnow()
        
        # Step 1: Grammar correction (if enabled and source is English)
        grammar_result = None
        text_to_translate = request.text
        
        if (request.check_grammar and 
            request.source_lang.lower() in ['en', 'en-us', 'en-gb']):
            grammar_result = await correct_grammar(request.text)
            text_to_translate = grammar_result.corrected_text
        
        # Step 2: Translation
        translation_result = await translate_text(
            text_to_translate,
            request.source_lang,
            request.target_lang,
            request.document_type,
            temperature=0.3 if request.document_type in ['legal', 'medical'] else 0.7
        )
        
        # Calculate processing time
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update statistics
        translator_service.stats['translations'] += 1
        translator_service.stats['tokens'] += translation_result['tokens']
        translator_service.stats['grammar_corrections'] += (
            len(grammar_result.changes_made) if grammar_result else 0
        )
        
        return TranslationResponse(
            original_text=request.text,
            grammar_check=grammar_result if request.show_corrections else None,
            translated_text=translation_result['translation'],
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            document_type=request.document_type,
            confidence_score=translation_result['confidence'],
            tokens_used=translation_result['tokens'],
            processing_time=process_time,
            processed_by="kaxm23",
            timestamp="2025-02-09 09:09:07"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation process failed: {str(e)}"
        )

@router.post("/translate/batch/",
             response_model=List[TranslationResponse],
             description="Batch translate texts with grammar correction")
async def translate_batch(
    texts: List[TranslationRequest],
    current_user: User = Depends(get_current_user)
) -> List[TranslationResponse]:
    """
    Batch translate texts with grammar correction.
    """
    try:
        tasks = [
            translate_with_grammar(request, current_user)
            for request in texts
        ]
        
        results = await asyncio.gather(*tasks)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch translation failed: {str(e)}"
        )

@router.get("/stats",
            description="Get translation service statistics")
async def get_stats(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """
    Get translation service statistics.
    """
    stats = translator_service.stats
    
    return {
        'translations_completed': stats['translations'],
        'grammar_corrections_made': stats['grammar_corrections'],
        'total_tokens_used': stats['tokens'],
        'total_cost': f"${stats['cost']:.4f}",
        'average_corrections_per_text': (
            stats['grammar_corrections'] / stats['translations']
            if stats['translations'] > 0 else 0
        ),
        'model': "gpt-4",
        'device': translator_service.device,
        'last_update': "2025-02-09 09:09:07",
        'processed_by': "kaxm23"
    }