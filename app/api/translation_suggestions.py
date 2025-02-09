from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime

from app.models.user_auth import User
from app.database.session import get_async_session
from app.core.security import get_current_user
from app.services.translation_analyzer import TranslationAnalyzer

router = APIRouter()

class ImprovementSuggestion(BaseModel):
    """Improvement suggestion model."""
    suggestion_text: str
    category: str
    confidence: float
    examples: List[Dict]
    implementation_steps: List[str]
    impact_level: str
    timestamp: str = "2025-02-09 09:55:40"
    processed_by: str = "kaxm23"

class SuggestionResponse(BaseModel):
    """Suggestion response model."""
    suggestions: List[ImprovementSuggestion]
    analysis_summary: Dict
    metadata: Dict
    timestamp: str = "2025-02-09 09:55:40"
    processed_by: str = "kaxm23"

@router.get("/translations/suggestions",
           response_model=SuggestionResponse,
           description="Get AI-generated translation improvement suggestions")
async def get_translation_suggestions(
    source_lang: str,
    target_lang: str,
    category: Optional[str] = Query(
        None,
        description="Filter suggestions by category (grammar, vocabulary, style, context)"
    ),
    min_confidence: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for suggestions"
    ),
    time_range_days: int = Query(
        30,
        ge=1,
        le=90,
        description="Analysis time range in days"
    ),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> SuggestionResponse:
    """
    Get AI-generated translation improvement suggestions.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        category: Filter suggestions by category
        min_confidence: Minimum confidence score
        time_range_days: Analysis time range in days
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        SuggestionResponse: Translation improvement suggestions
    """
    try:
        # Initialize analyzer
        analyzer = TranslationAnalyzer(session)
        analyzer.analysis_window_days = time_range_days
        
        # Get analysis results
        analysis_results = await analyzer.analyze_translation_errors(
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        if analysis_results['status'] != 'success':
            raise HTTPException(
                status_code=404,
                detail=analysis_results['message']
            )
        
        # Process suggestions
        suggestions = await _process_suggestions(
            analysis_results=analysis_results,
            category=category,
            min_confidence=min_confidence
        )
        
        # Prepare response
        return SuggestionResponse(
            suggestions=suggestions,
            analysis_summary={
                'total_patterns_analyzed': len(
                    analysis_results['analysis']['error_patterns']
                ),
                'confidence_level': analysis_results['analysis']['confidence_level'],
                'language_pair': f"{source_lang}-{target_lang}",
                'time_range': f"{time_range_days} days",
                'categories_found': list(
                    analysis_results['statistics']['common_errors']['categories'].keys()
                )
            },
            metadata={
                'user_id': current_user.id,
                'request_time': datetime.utcnow().isoformat(),
                'source_lang': source_lang,
                'target_lang': target_lang,
                'category_filter': category,
                'min_confidence': min_confidence
            },
            timestamp="2025-02-09 09:55:40",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )

async def _process_suggestions(
    analysis_results: Dict,
    category: Optional[str] = None,
    min_confidence: float = 0.7
) -> List[ImprovementSuggestion]:
    """Process and filter improvement suggestions."""
    suggestions = []
    
    for pattern in analysis_results['analysis']['error_patterns']:
        # Skip if category doesn't match filter
        if category and pattern['category'].lower() != category.lower():
            continue
        
        # Get matching suggestions
        pattern_suggestions = [
            s for s in analysis_results['analysis']['suggestions']
            if _matches_pattern(s, pattern)
        ]
        
        for suggestion in pattern_suggestions:
            confidence = _calculate_suggestion_confidence(
                pattern=pattern,
                suggestion=suggestion,
                base_confidence=analysis_results['analysis']['confidence_level']
            )
            
            # Skip if confidence is too low
            if confidence < min_confidence:
                continue
            
            suggestions.append(ImprovementSuggestion(
                suggestion_text=suggestion['text'],
                category=pattern['category'],
                confidence=confidence,
                examples=[{
                    'error_pattern': pattern['description'],
                    'example': pattern['examples'],
                    'severity': pattern['severity']
                }],
                implementation_steps=_parse_implementation_steps(
                    suggestion['implementation']
                ),
                impact_level=_get_impact_level(pattern['severity']),
                timestamp="2025-02-09 09:55:40",
                processed_by="kaxm23"
            ))
    
    return suggestions

def _matches_pattern(suggestion: Dict, pattern: Dict) -> bool:
    """Check if suggestion matches error pattern."""
    # Look for pattern keywords in suggestion
    pattern_keywords = set(pattern['description'].lower().split())
    suggestion_text = suggestion['text'].lower()
    
    return any(keyword in suggestion_text for keyword in pattern_keywords)

def _calculate_suggestion_confidence(
    pattern: Dict,
    suggestion: Dict,
    base_confidence: float
) -> float:
    """Calculate confidence score for suggestion."""
    confidence = base_confidence
    
    # Adjust based on pattern severity
    severity_factor = (6 - pattern['severity']) / 5  # Higher severity = lower confidence
    confidence *= severity_factor
    
    # Adjust based on suggestion specificity
    if len(suggestion.get('implementation', '').split()) > 50:
        confidence *= 1.1  # Bonus for detailed implementation
    
    # Adjust based on impact assessment
    if 'critical' in suggestion.get('impact', '').lower():
        confidence *= 0.9  # Penalty for critical impact
    
    return min(max(confidence, 0.0), 1.0)

def _parse_implementation_steps(implementation: str) -> List[str]:
    """Parse implementation steps from suggestion."""
    steps = []
    current_step = []
    
    for line in implementation.split('\n'):
        line = line.strip()
        if line:
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '*')):
                if current_step:
                    steps.append(' '.join(current_step))
                current_step = [line.lstrip('123456789.-* ')]
            else:
                current_step.append(line)
    
    if current_step:
        steps.append(' '.join(current_step))
    
    return steps or ["Implementation details not provided"]

def _get_impact_level(severity: int) -> str:
    """Get impact level from severity score."""
    if severity >= 5:
        return "Critical"
    elif severity >= 4:
        return "High"
    elif severity >= 3:
        return "Medium"
    return "Low"