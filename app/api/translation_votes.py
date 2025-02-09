from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel
from typing import Dict
from datetime import datetime

from app.models.translation_history import TranslationHistory
from app.models.translation_vote import TranslationVote, VoteType
from app.models.user_auth import User
from app.database.session import get_async_session
from app.core.security import get_current_user

router = APIRouter()

class VoteRequest(BaseModel):
    """Vote request model."""
    vote_type: VoteType
    timestamp: str = "2025-02-09 10:03:07"
    processed_by: str = "kaxm23"

class VoteResponse(BaseModel):
    """Vote response model."""
    translation_id: int
    vote_type: str
    vote_score: int
    upvotes: int
    downvotes: int
    vote_ratio: float
    timestamp: str = "2025-02-09 10:03:07"
    processed_by: str = "kaxm23"

@router.post("/translations/{translation_id}/vote",
            response_model=VoteResponse,
            description="Vote on a translation")
async def vote_translation(
    translation_id: int,
    vote: VoteRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> VoteResponse:
    """
    Vote on a translation.
    
    Args:
        translation_id: Translation ID
        vote: Vote data
        current_user: Current user
        session: Database session
        
    Returns:
        VoteResponse: Updated vote information
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
        
        # Check existing vote
        existing_vote_result = await session.execute(
            select(TranslationVote)
            .filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        existing_vote = existing_vote_result.scalar_one_or_none()
        
        if existing_vote:
            if existing_vote.vote_type == vote.vote_type:
                # Remove vote if same type
                if vote.vote_type == VoteType.UPVOTE:
                    translation.upvotes -= 1
                else:
                    translation.downvotes -= 1
                await session.delete(existing_vote)
            else:
                # Change vote type
                if vote.vote_type == VoteType.UPVOTE:
                    translation.upvotes += 1
                    translation.downvotes -= 1
                else:
                    translation.upvotes -= 1
                    translation.downvotes += 1
                existing_vote.vote_type = vote.vote_type
                existing_vote.updated_at = datetime.utcnow()
        else:
            # Create new vote
            new_vote = TranslationVote(
                translation_id=translation_id,
                user_id=current_user.id,
                vote_type=vote.vote_type
            )
            session.add(new_vote)
            
            if vote.vote_type == VoteType.UPVOTE:
                translation.upvotes += 1
            else:
                translation.downvotes += 1
        
        await session.commit()
        
        return VoteResponse(
            translation_id=translation_id,
            vote_type=vote.vote_type.value,
            vote_score=translation.vote_score,
            upvotes=translation.upvotes,
            downvotes=translation.downvotes,
            vote_ratio=translation.vote_ratio,
            timestamp="2025-02-09 10:03:07",
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

@router.get("/translations/{translation_id}/votes",
           response_model=VoteResponse,
           description="Get translation vote information")
async def get_translation_votes(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> VoteResponse:
    """
    Get translation vote information.
    
    Args:
        translation_id: Translation ID
        current_user: Current user
        session: Database session
        
    Returns:
        VoteResponse: Vote information
    """
    try:
        # Get translation
        result = await session.execute(
            select(TranslationHistory)
            .filter_by(id=translation_id)
        )
        translation = result.scalar_one_or_none()
        
        if not translation:
            raise HTTPException(
                status_code=404,
                detail="Translation not found"
            )
        
        # Get user's vote if exists
        vote_result = await session.execute(
            select(TranslationVote)
            .filter_by(
                translation_id=translation_id,
                user_id=current_user.id
            )
        )
        user_vote = vote_result.scalar_one_or_none()
        
        return VoteResponse(
            translation_id=translation_id,
            vote_type=user_vote.vote_type.value if user_vote else None,
            vote_score=translation.vote_score,
            upvotes=translation.upvotes,
            downvotes=translation.downvotes,
            vote_ratio=translation.vote_ratio,
            timestamp="2025-02-09 10:03:07",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get vote information: {str(e)}"
        )