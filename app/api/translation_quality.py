from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime
import language_tool_python
import openai
from app.core.config import Settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()
settings = Settings()

class QualityReport(BaseModel):
    """Translation quality report model."""
    source_errors: List[Dict]
    target_errors: List[Dict]
    quality_score: float
    suggestions: List[Dict]
    timestamp: str = "2025-02-09 09:14:54"
    processed_by: str = "kaxm23"

class TranslationRequest(BaseModel):
    """Translation request with quality check options."""
    text: str = Field(..., example="The text to translate")
    source_lang: str = Field(..., example="en")
    target_lang: str = Field(..., example="es")
    document_type: Literal["legal", "medical", "academic", "general"] = "general"
    quality_check: bool = Field(True, example=True)
    error_threshold: float = Field(0.8, example=0.8)

class TranslationResponse(BaseModel):
    """Enhanced translation response with quality report."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    document_type: str
    quality_report: Optional[QualityReport]
    confidence_score: float
    tokens_used: int
    processing_time: float
    timestamp: str = "2025-02-09 09:14:54"
    processed_by: str = "kaxm23"

class QualityChecker:
    """Translation quality checker."""
    
    def __init__(self):
        """Initialize quality checker."""
        self.source_checker = language_tool_python.LanguageTool('en-US')
        self.target_checkers = {}
        self.stats = {
            'checks_performed': 0,
            'total_errors': 0,
            'average_quality': 0.0
        }

    def get_checker(self, language: str) -> language_tool_python.LanguageTool:
        """Get or create language checker."""
        if language not in self.target_checkers:
            self.target_checkers[language] = language_tool_python.LanguageTool(language)
        return self.target_checkers[language]

    async def check_quality(self,
                          source_text: str,
                          translated_text: str,
                          source_lang: str,
                          target_lang: str) -> QualityReport:
        """Check translation quality."""
        try:
            # Check source text
            source_checker = self.source_checker if source_lang.startswith('en') else self.get_checker(source_lang)
            source_errors = source_checker.check(source_text)
            
            # Check translated text
            target_checker = self.get_checker(target_lang)
            target_errors = target_checker.check(translated_text)
            
            # Process errors
            processed_source_errors = self._process_errors(source_errors)
            processed_target_errors = self._process_errors(target_errors)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                processed_source_errors,
                processed_target_errors,
                source_text,
                translated_text
            )
            
            # Generate suggestions
            suggestions = self._generate_suggestions(
                processed_source_errors,
                processed_target_errors,
                target_checker.correct(translated_text)
            )
            
            # Update statistics
            self._update_stats(quality_score, len(processed_target_errors))
            
            return QualityReport(
                source_errors=processed_source_errors,
                target_errors=processed_target_errors,
                quality_score=quality_score,
                suggestions=suggestions,
                timestamp="2025-02-09 09:14:54",
                processed_by="kaxm23"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Quality check failed: {str(e)}"
            )

    def _process_errors(self, errors: List) -> List[Dict]:
        """Process and categorize errors."""
        processed = []
        for error in errors:
            processed.append({
                'message': error.message,
                'context': error.context,
                'category': error.category,
                'type': self._categorize_error(error),
                'replacements': error.replacements[:3] if error.replacements else [],
                'offset': error.offset,
                'length': error.length
            })
        return processed

    def _categorize_error(self, error) -> str:
        """Categorize error type."""
        if 'grammar' in error.category.lower():
            return 'grammar'
        elif 'style' in error.category.lower():
            return 'style'
        elif 'spelling' in error.category.lower():
            return 'spelling'
        else:
            return 'other'

    def _calculate_quality_score(self,
                               source_errors: List[Dict],
                               target_errors: List[Dict],
                               source_text: str,
                               target_text: str) -> float:
        """Calculate translation quality score."""
        # Base score
        score = 1.0
        
        # Penalize based on target errors
        error_penalty = len(target_errors) * 0.05
        score -= min(error_penalty, 0.5)
        
        # Penalize based on relative error increase
        if len(source_errors) > 0:
            error_ratio = len(target_errors) / len(source_errors)
            if error_ratio > 1:
                score -= min((error_ratio - 1) * 0.1, 0.3)
        
        # Length ratio check
        length_ratio = len(target_text) / len(source_text)
        if length_ratio < 0.5 or length_ratio > 2.0:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _generate_suggestions(self,
                            source_errors: List[Dict],
                            target_errors: List[Dict],
                            corrected_text: str) -> List[Dict]:
        """Generate improvement suggestions."""
        suggestions = []
        
        # Group errors by type
        error_types = {}
        for error in target_errors:
            error_type = error['type']
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        # Generate suggestions for each error type
        for error_type, errors in error_types.items():
            if len(errors) > 0:
                suggestions.append({
                    'type': error_type,
                    'count': len(errors),
                    'message': f"Found {len(errors)} {error_type} issues",
                    'examples': errors[:3],
                    'priority': self._get_priority(error_type, len(errors))
                })
        
        return sorted(suggestions, key=lambda x: x['priority'])

    def _get_priority(self, error_type: str, count: int) -> int:
        """Get suggestion priority."""
        priorities = {
            'grammar': 1,
            'spelling': 2,
            'style': 3,
            'other': 4
        }
        base_priority = priorities.get(error_type, 5)
        return base_priority - (count > 5)

    def _update_stats(self, quality_score: float, error_count: int):
        """Update quality check statistics."""
        self.stats['checks_performed'] += 1
        self.stats['total_errors'] += error_count
        current_avg = self.stats['average_quality']
        self.stats['average_quality'] = (
            (current_avg * (self.stats['checks_performed'] - 1) + quality_score) /
            self.stats['checks_performed']
        )

quality_checker = QualityChecker()

@router.post("/translate/",
            response_model=TranslationResponse,
            description="Translate text with quality check")
async def translate_with_quality_check(
    request: TranslationRequest,
    current_user: User = Depends(get_current_user)
) -> TranslationResponse:
    """
    Translate text and check quality.
    """
    try:
        start_time = datetime.utcnow()
        
        # Translate text
        translation_result = await translate_text(
            request.text,
            request.source_lang,
            request.target_lang,
            request.document_type
        )
        
        # Quality check if requested
        quality_report = None
        if request.quality_check:
            quality_report = await quality_checker.check_quality(
                request.text,
                translation_result['translation'],
                request.source_lang,
                request.target_lang
            )
            
            # Check against threshold
            if quality_report.quality_score < request.error_threshold:
                # Attempt improvement
                improved_translation = await improve_translation(
                    request.text,
                    translation_result['translation'],
                    quality_report,
                    request.source_lang,
                    request.target_lang
                )
                translation_result['translation'] = improved_translation
                
                # Re-check quality
                quality_report = await quality_checker.check_quality(
                    request.text,
                    improved_translation,
                    request.source_lang,
                    request.target_lang
                )
        
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        return TranslationResponse(
            original_text=request.text,
            translated_text=translation_result['translation'],
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            document_type=request.document_type,
            quality_report=quality_report,
            confidence_score=translation_result['confidence'],
            tokens_used=translation_result['tokens'],
            processing_time=process_time,
            timestamp="2025-02-09 09:14:54",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation process failed: {str(e)}"
        )

async def improve_translation(
    source_text: str,
    translated_text: str,
    quality_report: QualityReport,
    source_lang: str,
    target_lang: str
) -> str:
    """Improve translation based on quality report."""
    try:
        # Create improvement prompt
        prompt = (
            f"Improve this translation while fixing the following issues:\n\n"
            f"Source ({source_lang}): {source_text}\n"
            f"Translation ({target_lang}): {translated_text}\n\n"
            "Issues to address:\n"
        )
        
        for error in quality_report.target_errors[:5]:
            prompt += f"- {error['message']}\n"
        
        # Get improved translation
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a translation improvement specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation improvement failed: {str(e)}"
        )

@router.get("/quality-stats",
            description="Get quality check statistics")
async def get_quality_stats(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get quality checking statistics."""
    return {
        'checks_performed': quality_checker.stats['checks_performed'],
        'total_errors': quality_checker.stats['total_errors'],
        'average_quality': quality_checker.stats['average_quality'],
        'timestamp': "2025-02-09 09:14:54",
        'processed_by': "kaxm23"
    }