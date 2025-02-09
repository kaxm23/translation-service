from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

from app.models.translation_rating import TranslationRating, RatingType
from app.models.user_auth import User
from app.database.session import get_async_session
from app.core.security import get_current_user

router = APIRouter()

class RatingCreate(BaseModel):
    """Rating creation request model."""
    rating: RatingType
    feedback: Optional[str] = None
    timestamp: str = "2025-02-09 09:43:13"
    processed_by: str = "kaxm23"

class RatingResponse(BaseModel):
    """Rating response model."""
    id: int
    translation_id: int
    user_id: int
    rating: str
    feedback: Optional[str]
    created_at: str
    timestamp: str = "2025-02-09 09:43:13"
    processed_by: str = "kaxm23"

class RatingStats(BaseModel):
    """Rating statistics model."""
    total_ratings: int
    thumbs_up: int
    thumbs_down: int
    feedback_count: int
    rating_ratio: float
    timestamp: str = "2025-02-09 09:43:13"
    processed_by: str = "kaxm23"

@router.post("/translations/{translation_id}/rate",
            response_model=RatingResponse,
            description="Rate a translation")
async def rate_translation(
    translation_id: int,
    rating: RatingCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> RatingResponse:
    """
    Rate a translation.
    
    Args:
        translation_id: Translation ID
        rating: Rating data
        current_user: Current user
        session: Database session
        
    Returns:
        RatingResponse: Created rating
    """
    try:
        # Check if translation exists
        translation_result = await session.execute(
            select(TranslationHistory).filter_by(id=translation_id)
        )
        translation = translation_result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=404,
                detail="Translation not found"
            )
        
        # Check if user has already rated
        existing_rating = await session.execute(
            select(TranslationRating).filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        
        if existing_rating.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail="You have already rated this translation"
            )
        
        # Create rating
        new_rating = TranslationRating(
            translation_id=translation_id,
            user_id=current_user.id,
            rating=rating.rating,
            feedback=rating.feedback
        )
        
        session.add(new_rating)
        await session.commit()
        await session.refresh(new_rating)
        
        return RatingResponse(
            id=new_rating.id,
            translation_id=new_rating.translation_id,
            user_id=new_rating.user_id,
            rating=new_rating.rating.value,
            feedback=new_rating.feedback,
            created_at=new_rating.created_at.isoformat(),
            timestamp="2025-02-09 09:43:13",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create rating: {str(e)}"
        )

@router.put("/translations/{translation_id}/rate",
           response_model=RatingResponse,
           description="Update translation rating")
async def update_rating(
    translation_id: int,
    rating: RatingCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> RatingResponse:
    """
    Update translation rating.
    
    Args:
        translation_id: Translation ID
        rating: Updated rating data
        current_user: Current user
        session: Database session
        
    Returns:
        RatingResponse: Updated rating
    """
    try:
        # Get existing rating
        result = await session.execute(
            select(TranslationRating).filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        existing_rating = result.scalar_one_or_none()
        
        if not existing_rating:
            raise HTTPException(
                status_code=404,
                detail="Rating not found"
            )
        
        # Update rating
        existing_rating.rating = rating.rating
        existing_rating.feedback = rating.feedback
        existing_rating.updated_at = datetime.utcnow()
        
        await session.commit()
        await session.refresh(existing_rating)
        
        return RatingResponse(
            id=existing_rating.id,
            translation_id=existing_rating.translation_id,
            user_id=existing_rating.user_id,
            rating=existing_rating.rating.value,
            feedback=existing_rating.feedback,
            created_at=existing_rating.created_at.isoformat(),
            timestamp="2025-02-09 09:43:13",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update rating: {str(e)}"
        )

@router.get("/translations/{translation_id}/ratings",
           response_model=List[RatingResponse],
           description="Get translation ratings")
async def get_translation_ratings(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> List[RatingResponse]:
    """
    Get translation ratings.
    
    Args:
        translation_id: Translation ID
        current_user: Current user
        session: Database session
        
    Returns:
        List[RatingResponse]: List of ratings
    """
    try:
        result = await session.execute(
            select(TranslationRating)
            .filter_by(translation_id=translation_id)
            .order_by(TranslationRating.created_at.desc())
        )
        ratings = result.scalars().all()
        
        return [
            RatingResponse(
                id=rating.id,
                translation_id=rating.translation_id,
                user_id=rating.user_id,
                rating=rating.rating.value,
                feedback=rating.feedback,
                created_at=rating.created_at.isoformat(),
                timestamp="2025-02-09 09:43:13",
                processed_by="kaxm23"
            )
            for rating in ratings
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch ratings: {str(e)}"
        )

@router.get("/translations/{translation_id}/rating-stats",
           response_model=RatingStats,
           description="Get translation rating statistics")
async def get_rating_stats(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> RatingStats:
    """
    Get translation rating statistics.
    
    Args:
        translation_id: Translation ID
        current_user: Current user
        session: Database session
        
    Returns:
        RatingStats: Rating statistics
    """
    try:
        # Get total ratings
        total_result = await session.execute(
            select(func.count()).select_from(TranslationRating)
            .filter_by(translation_id=translation_id)
        )
        total_ratings = total_result.scalar()
        
        # Get thumbs up count
        thumbs_up_result = await session.execute(
            select(func.count()).select_from(TranslationRating)
            .filter_by(
                translation_id=translation_id,
                rating=RatingType.THUMBS_UP
            )
        )
        thumbs_up = thumbs_up_result.scalar()
        
        # Get thumbs down count
        thumbs_down_result = await session.execute(
            select(func.count()).select_from(TranslationRating)
            .filter_by(
                translation_id=translation_id,
                rating=RatingType.THUMBS_DOWN
            )
        )
        thumbs_down = thumbs_down_result.scalar()
        
        # Get feedback count
        feedback_result = await session.execute(
            select(func.count()).select_from(TranslationRating)
            .filter(
                TranslationRating.translation_id == translation_id,
                TranslationRating.feedback.isnot(None)
            )
        )
        feedback_count = feedback_result.scalar()
        
        # Calculate rating ratio
        rating_ratio = (
            thumbs_up / total_ratings
            if total_ratings > 0
            else 0
        )
        
        return RatingStats(
            total_ratings=total_ratings,
            thumbs_up=thumbs_up,
            thumbs_down=thumbs_down,
            feedback_count=feedback_count,
            rating_ratio=rating_ratio,
            timestamp="2025-02-09 09:43:13",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch rating statistics: {str(e)}"
        )