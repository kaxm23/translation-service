from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import Optional, Dict
from datetime import datetime

from app.models.user_auth import User
from app.models.translation_stats import UserTranslationStats
from app.database.session import get_async_session
from app.core.security import get_current_user

router = APIRouter()

class TranslationStatsResponse(BaseModel):
    """Translation statistics response model."""
    total_translations: int
    total_characters: int
    avg_confidence_score: float
    last_translation_at: Optional[str]
    daily_average: float
    language_pairs: Dict[str, int]
    timestamp: str = "2025-02-09 10:00:03"
    processed_by: str = "kaxm23"

@router.get("/users/me/translation-stats",
           response_model=TranslationStatsResponse,
           description="Get current user's translation statistics")
async def get_user_translation_stats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> TranslationStatsResponse:
    """
    Get user's translation statistics.
    
    Args:
        current_user: Current user
        session: Database session
        
    Returns:
        TranslationStatsResponse: Translation statistics
    """
    try:
        # Get user stats
        result = await session.execute(
            select(UserTranslationStats)
            .filter_by(user_id=current_user.id)
        )
        stats = result.scalar_one_or_none()
        
        if not stats:
            return TranslationStatsResponse(
                total_translations=0,
                total_characters=0,
                avg_confidence_score=0.0,
                last_translation_at=None,
                daily_average=0.0,
                language_pairs={},
                timestamp="2025-02-09 10:00:03",
                processed_by="kaxm23"
            )
        
        # Get language pair statistics
        lang_pairs_result = await session.execute(
            """
            SELECT 
                source_lang || '-' || target_lang as pair,
                COUNT(*) as count
            FROM translation_history
            WHERE user_id = :user_id
            GROUP BY source_lang, target_lang
            """,
            {'user_id': current_user.id}
        )
        language_pairs = dict(lang_pairs_result.fetchall())
        
        # Calculate daily average
        if stats.last_translation_at and stats.created_at:
            days_active = (
                stats.last_translation_at - stats.created_at
            ).days or 1
            daily_average = stats.total_translations / days_active
        else:
            daily_average = 0.0
        
        return TranslationStatsResponse(
            total_translations=stats.total_translations,
            total_characters=stats.total_characters,
            avg_confidence_score=stats.avg_confidence_score,
            last_translation_at=stats.last_translation_at.isoformat()
                if stats.last_translation_at else None,
            daily_average=daily_average,
            language_pairs=language_pairs,
            timestamp="2025-02-09 10:00:03",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch translation statistics: {str(e)}"
        )