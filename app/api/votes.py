from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import enum

from app.models.user_auth import User
from app.database.session import get_async_session
from app.core.security import get_current_user

router = APIRouter()

class VoteType(str, enum.Enum):
    """Vote type enumeration."""
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"

class VoteCreate(BaseModel):
    """Vote creation request model."""
    vote_type: VoteType
    timestamp: str = "2025-02-09 10:04:26"
    processed_by: str = "kaxm23"

class VoteResponse(BaseModel):
    """Vote response model."""
    translation_id: int
    user_id: int
    vote_type: VoteType
    upvotes: int
    downvotes: int
    score: int
    timestamp: str = "2025-02-09 10:04:26"
    processed_by: str = "kaxm23"

class VoteStats(BaseModel):
    """Vote statistics model."""
    total_votes: int
    upvote_ratio: float
    user_vote_count: int
    timestamp: str = "2025-02-09 10:04:26"
    processed_by: str = "kaxm23"

@router.post("/translations/{translation_id}/vote",
            response_model=VoteResponse,
            description="Vote on a translation")
async def vote_translation(
    translation_id: int,
    vote: VoteCreate,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> VoteResponse:
    """
    Vote on a translation.
    
    Args:
        translation_id: Translation ID
        vote: Vote data
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        VoteResponse: Updated vote information
    """
    try:
        # Check if translation exists
        translation_result = await session.execute(
            select(TranslationHistory)
            .filter_by(id=translation_id)
        )
        translation = translation_result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=404,
                detail="Translation not found"
            )
        
        # Check if user has already voted
        existing_vote = await session.execute(
            select(TranslationVote)
            .filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        vote_record = existing_vote.scalar_one_or_none()
        
        if vote_record:
            # Update existing vote
            if vote_record.vote_type != vote.vote_type:
                # Change vote type
                if vote.vote_type == VoteType.UPVOTE:
                    translation.upvotes += 1
                    translation.downvotes -= 1
                else:
                    translation.upvotes -= 1
                    translation.downvotes += 1
                
                vote_record.vote_type = vote.vote_type
                vote_record.updated_at = datetime.utcnow()
        else:
            # Create new vote
            new_vote = TranslationVote(
                translation_id=translation_id,
                user_id=current_user.id,
                vote_type=vote.vote_type,
                created_at=datetime.utcnow()
            )
            
            session.add(new_vote)
            
            # Update vote counts
            if vote.vote_type == VoteType.UPVOTE:
                translation.upvotes += 1
            else:
                translation.downvotes += 1
        
        # Update translation metadata
        if not translation.metadata:
            translation.metadata = {}
        
        translation.metadata['last_vote'] = {
            'timestamp': "2025-02-09 10:04:26",
            'processed_by': "kaxm23",
            'vote_type': vote.vote_type
        }
        
        await session.commit()
        
        return VoteResponse(
            translation_id=translation_id,
            user_id=current_user.id,
            vote_type=vote.vote_type,
            upvotes=translation.upvotes,
            downvotes=translation.downvotes,
            score=translation.upvotes - translation.downvotes,
            timestamp="2025-02-09 10:04:26",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process vote: {str(e)}"
        )

@router.delete("/translations/{translation_id}/vote",
               description="Remove vote from translation")
async def remove_vote(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Remove vote from translation.
    
    Args:
        translation_id: Translation ID
        current_user: Current authenticated user
        session: Database session
    """
    try:
        # Get vote
        vote_result = await session.execute(
            select(TranslationVote)
            .filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        vote = vote_result.scalar_one_or_none()
        
        if not vote:
            raise HTTPException(
                status_code=404,
                detail="Vote not found"
            )
        
        # Get translation
        translation_result = await session.execute(
            select(TranslationHistory)
            .filter_by(id=translation_id)
        )
        translation = translation_result.scalar_one_or_none()
        
        # Update vote counts
        if vote.vote_type == VoteType.UPVOTE:
            translation.upvotes -= 1
        else:
            translation.downvotes -= 1
        
        # Remove vote
        await session.delete(vote)
        
        # Update translation metadata
        if not translation.metadata:
            translation.metadata = {}
        
        translation.metadata['vote_removed'] = {
            'timestamp': "2025-02-09 10:04:26",
            'processed_by': "kaxm23"
        }
        
        await session.commit()
        
        return {"message": "Vote removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove vote: {str(e)}"
        )

@router.get("/translations/{translation_id}/votes",
            response_model=VoteStats,
            description="Get translation vote statistics")
async def get_vote_stats(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> VoteStats:
    """
    Get translation vote statistics.
    
    Args:
        translation_id: Translation ID
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        VoteStats: Vote statistics
    """
    try:
        # Get translation
        translation_result = await session.execute(
            select(TranslationHistory)
            .filter_by(id=translation_id)
        )
        translation = translation_result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=404,
                detail="Translation not found"
            )
        
        # Get user's vote count
        user_votes_result = await session.execute(
            select(TranslationVote)
            .filter_by(user_id=current_user.id)
        )
        user_votes = user_votes_result.scalars().all()
        
        # Calculate statistics
        total_votes = translation.upvotes + translation.downvotes
        upvote_ratio = (
            translation.upvotes / total_votes
            if total_votes > 0
            else 0.0
        )
        
        return VoteStats(
            total_votes=total_votes,
            upvote_ratio=upvote_ratio,
            user_vote_count=len(user_votes),
            timestamp="2025-02-09 10:04:26",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get vote statistics: {str(e)}"
        )

@router.get("/translations/votes/top",
            response_model=List[VoteResponse],
            description="Get top voted translations")
async def get_top_voted_translations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> List[VoteResponse]:
    """
    Get top voted translations.
    
    Args:
        limit: Number of translations to return
        offset: Number of translations to skip
        current_user: Current authenticated user
        session: Database session
        
    Returns:
        List[VoteResponse]: List of top voted translations
    """
    try:
        # Get translations ordered by vote score
        result = await session.execute(
            select(TranslationHistory)
            .order_by(
                (TranslationHistory.upvotes - TranslationHistory.downvotes).desc()
            )
            .offset(offset)
            .limit(limit)
        )
        translations = result.scalars().all()
        
        return [
            VoteResponse(
                translation_id=t.id,
                user_id=t.user_id,
                vote_type=None,  # No specific vote type for listing
                upvotes=t.upvotes,
                downvotes=t.downvotes,
                score=t.upvotes - t.downvotes,
                timestamp="2025-02-09 10:04:26",
                processed_by="kaxm23"
            )
            for t in translations
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get top translations: {str(e)}"
        )