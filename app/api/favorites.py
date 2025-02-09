from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.models.favorite_translation import FavoriteTranslation
from app.models.translation_history import TranslationHistory
from app.models.user_auth import User
from app.database.session import get_async_session
from app.core.security import get_current_user

router = APIRouter()

class FavoriteResponse(BaseModel):
    """Favorite translation response model."""
    id: int
    translation_id: int
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    created_at: str
    timestamp: str = "2025-02-09 09:47:47"
    processed_by: str = "kaxm23"

class FavoriteStats(BaseModel):
    """Favorite translation statistics model."""
    total_favorites: int
    language_pairs: List[dict]
    recent_favorites: List[dict]
    most_used_languages: dict
    timestamp: str = "2025-02-09 09:47:47"
    processed_by: str = "kaxm23"

@router.post("/translations/{translation_id}/favorite",
            response_model=FavoriteResponse,
            description="Mark translation as favorite")
async def add_favorite(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> FavoriteResponse:
    """
    Mark translation as favorite.
    
    Args:
        translation_id: Translation ID
        current_user: Current user
        session: Database session
        
    Returns:
        FavoriteResponse: Favorite translation details
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
        
        # Check if already favorited
        existing_result = await session.execute(
            select(FavoriteTranslation).filter_by(
                user_id=current_user.id,
                translation_id=translation_id
            )
        )
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail="Translation already in favorites"
            )
        
        # Create favorite
        favorite = FavoriteTranslation(
            user_id=current_user.id,
            translation_id=translation_id
        )
        
        session.add(favorite)
        await session.commit()
        await session.refresh(favorite)
        
        return FavoriteResponse(
            id=favorite.id,
            translation_id=translation.id,
            source_text=translation.source_text,
            translated_text=translation.translated_text,
            source_lang=translation.source_lang,
            target_lang=translation.target_lang,
            created_at=favorite.created_at.isoformat(),
            timestamp="2025-02-09 09:47:47",
            processed_by="kaxm23"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add favorite: {str(e)}"
        )

@router.delete("/translations/{translation_id}/favorite",
               description="Remove translation from favorites")
async def remove_favorite(
    translation_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Remove translation from favorites.
    
    Args:
        translation_id: Translation ID
        current_user: Current user
        session: Database session
    """
    try:
        # Get favorite
        result = await session.execute(
            select(FavoriteTranslation).filter_by(
                user_id=current_user.id,
                translation_id=translation_id
            )
        )
        favorite = result.scalar_one_or_none()
        
        if not favorite:
            raise HTTPException(
                status_code=404,
                detail="Favorite not found"
            )
        
        # Remove favorite
        await session.delete(favorite)
        await session.commit()
        
        return {"message": "Favorite removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to remove favorite: {str(e)}"
        )

@router.get("/translations/favorites",
            response_model=List[FavoriteResponse],
            description="Get user's favorite translations")
async def get_favorites(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> List[FavoriteResponse]:
    """
    Get user's favorite translations.
    
    Args:
        page: Page number
        page_size: Items per page
        source_lang: Filter by source language
        target_lang: Filter by target language
        current_user: Current user
        session: Database session
        
    Returns:
        List[FavoriteResponse]: List of favorite translations
    """
    try:
        # Build query
        query = (
            select(FavoriteTranslation, TranslationHistory)
            .join(TranslationHistory)
            .filter(FavoriteTranslation.user_id == current_user.id)
            .order_by(desc(FavoriteTranslation.created_at))
        )
        
        # Apply language filters
        if source_lang:
            query = query.filter(
                TranslationHistory.source_lang == source_lang
            )
        if target_lang:
            query = query.filter(
                TranslationHistory.target_lang == target_lang
            )
        
        # Apply pagination
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        # Execute query
        result = await session.execute(query)
        favorites = result.all()
        
        return [
            FavoriteResponse(
                id=favorite.id,
                translation_id=translation.id,
                source_text=translation.source_text,
                translated_text=translation.translated_text,
                source_lang=translation.source_lang,
                target_lang=translation.target_lang,
                created_at=favorite.created_at.isoformat(),
                timestamp="2025-02-09 09:47:47",
                processed_by="kaxm23"
            )
            for favorite, translation in favorites
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch favorites: {str(e)}"
        )

@router.get("/translations/favorites/stats",
            response_model=FavoriteStats,
            description="Get favorite translation statistics")
async def get_favorite_stats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> FavoriteStats:
    """
    Get favorite translation statistics.
    
    Args:
        current_user: Current user
        session: Database session
        
    Returns:
        FavoriteStats: Favorite translation statistics
    """
    try:
        # Get total favorites
        total_result = await session.execute(
            select(FavoriteTranslation)
            .filter_by(user_id=current_user.id)
        )
        favorites = total_result.scalars().all()
        total_favorites = len(favorites)
        
        # Get language pairs
        language_pairs = {}
        source_langs = {}
        target_langs = {}
        
        for favorite in favorites:
            translation = await session.get(
                TranslationHistory,
                favorite.translation_id
            )
            
            if translation:
                pair = f"{translation.source_lang}-{translation.target_lang}"
                language_pairs[pair] = language_pairs.get(pair, 0) + 1
                
                source_langs[translation.source_lang] = (
                    source_langs.get(translation.source_lang, 0) + 1
                )
                target_langs[translation.target_lang] = (
                    target_langs.get(translation.target_lang, 0) + 1
                )
        
        # Get recent favorites
        recent_result = await session.execute(
            select(FavoriteTranslation, TranslationHistory)
            .join(TranslationHistory)
            .filter(FavoriteTranslation.user_id == current_user.id)
            .order_by(desc(FavoriteTranslation.created_at))
            .limit(5)
        )
        recent_favorites = [
            {
                "id": f.id,
                "translation_id": t.id,
                "source_text": t.source_text[:50],
                "created_at": f.created_at.isoformat()
            }
            for f, t in recent_result
        ]
        
        return FavoriteStats(
            total_favorites=total_favorites,
            language_pairs=[
                {"pair": k, "count": v}
                for k, v in language_pairs.items()
            ],
            recent_favorites=recent_favorites,
            most_used_languages={
                "source": dict(sorted(
                    source_langs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]),
                "target": dict(sorted(
                    target_langs.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5])
            },
            timestamp="2025-02-09 09:47:47",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch statistics: {str(e)}"
        )