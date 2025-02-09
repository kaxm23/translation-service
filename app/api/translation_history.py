from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from app.database.session import get_async_session
from app.core.security import get_current_user
from app.models.user import User
from app.models.translation_history import TranslationHistory, TranslationType, TranslationStatus

router = APIRouter()

class TranslationHistoryResponse(BaseModel):
    """Translation history response model."""
    id: int
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    translation_type: str
    status: str
    confidence_score: Optional[float]
    processing_time: Optional[float]
    cost: Optional[float]
    created_at: str
    file_path: Optional[str] = None
    metadata: Optional[Dict] = None
    timestamp: str = "2025-02-09 09:30:07"
    processed_by: str = "kaxm23"

class TranslationHistoryFilter(BaseModel):
    """Translation history filter options."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    translation_type: Optional[TranslationType] = None
    status: Optional[TranslationStatus] = None
    min_confidence: Optional[float] = None
    has_file: Optional[bool] = None

class TranslationHistoryStats(BaseModel):
    """Translation history statistics."""
    total_translations: int
    total_words: int
    total_cost: float
    average_confidence: float
    language_pairs: List[Dict[str, str]]
    translation_types: Dict[str, int]
    timestamp: str = "2025-02-09 09:30:07"
    processed_by: str = "kaxm23"

class HistoryManager:
    """
    Translation history manager.
    Created by: kaxm23
    Created on: 2025-02-09 09:30:07 UTC
    """
    
    def __init__(self):
        """Initialize history manager."""
        self.cache = {}
        self.stats = {
            'queries_executed': 0,
            'cache_hits': 0,
            'total_records': 0
        }

    async def get_user_translations(self,
                                  session: AsyncSession,
                                  user_id: str,
                                  filters: TranslationHistoryFilter,
                                  page: int = 1,
                                  page_size: int = 20,
                                  sort_by: str = "created_at",
                                  sort_order: str = "desc") -> Dict:
        """
        Get user's translation history.
        
        Args:
            session: Database session
            user_id: User identifier
            filters: Filter options
            page: Page number
            page_size: Items per page
            sort_by: Sort field
            sort_order: Sort direction
            
        Returns:
            Dict: Translation history
        """
        try:
            # Build query
            query = select(TranslationHistory).where(
                TranslationHistory.user_id == user_id
            )
            
            # Apply filters
            query = self._apply_filters(query, filters)
            
            # Apply sorting
            query = self._apply_sorting(query, sort_by, sort_order)
            
            # Get total count
            count_query = select(func.count()).select_from(query)
            total_count = await session.scalar(count_query)
            
            # Apply pagination
            query = query.offset((page - 1) * page_size).limit(page_size)
            
            # Execute query
            result = await session.execute(query)
            translations = result.scalars().all()
            
            # Update statistics
            self.stats['queries_executed'] += 1
            self.stats['total_records'] = total_count
            
            return {
                'items': [
                    TranslationHistoryResponse(
                        id=t.id,
                        source_text=t.source_text,
                        translated_text=t.translated_text,
                        source_lang=t.source_lang,
                        target_lang=t.target_lang,
                        translation_type=t.translation_type.value,
                        status=t.status.value,
                        confidence_score=t.confidence_score,
                        processing_time=t.processing_time,
                        cost=t.cost,
                        created_at=t.created_at.isoformat(),
                        file_path=t.file_path,
                        metadata=t.metadata,
                        timestamp="2025-02-09 09:30:07",
                        processed_by="kaxm23"
                    ).dict()
                    for t in translations
                ],
                'pagination': {
                    'page': page,
                    'page_size': page_size,
                    'total_items': total_count,
                    'total_pages': (total_count + page_size - 1) // page_size
                },
                'filters': filters.dict(exclude_none=True),
                'timestamp': "2025-02-09 09:30:07",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch translation history: {str(e)}"
            )

    def _apply_filters(self, query, filters: TranslationHistoryFilter):
        """Apply filters to query."""
        if filters.start_date:
            query = query.where(
                TranslationHistory.created_at >= filters.start_date
            )
        
        if filters.end_date:
            query = query.where(
                TranslationHistory.created_at <= filters.end_date
            )
        
        if filters.source_lang:
            query = query.where(
                TranslationHistory.source_lang == filters.source_lang
            )
        
        if filters.target_lang:
            query = query.where(
                TranslationHistory.target_lang == filters.target_lang
            )
        
        if filters.translation_type:
            query = query.where(
                TranslationHistory.translation_type == filters.translation_type
            )
        
        if filters.status:
            query = query.where(
                TranslationHistory.status == filters.status
            )
        
        if filters.min_confidence:
            query = query.where(
                TranslationHistory.confidence_score >= filters.min_confidence
            )
        
        if filters.has_file is not None:
            if filters.has_file:
                query = query.where(
                    TranslationHistory.file_path.isnot(None)
                )
            else:
                query = query.where(
                    TranslationHistory.file_path.is_(None)
                )
        
        return query

    def _apply_sorting(self,
                      query,
                      sort_by: str,
                      sort_order: str):
        """Apply sorting to query."""
        column = getattr(TranslationHistory, sort_by)
        if sort_order.lower() == "desc":
            column = desc(column)
        return query.order_by(column)

    async def get_user_stats(self,
                           session: AsyncSession,
                           user_id: str,
                           time_range: Optional[int] = None) -> TranslationHistoryStats:
        """Get user's translation statistics."""
        try:
            # Build base query
            query = select(TranslationHistory).where(
                TranslationHistory.user_id == user_id
            )
            
            # Apply time range filter
            if time_range:
                start_date = datetime.utcnow() - timedelta(days=time_range)
                query = query.where(
                    TranslationHistory.created_at >= start_date
                )
            
            # Execute query
            result = await session.execute(query)
            translations = result.scalars().all()
            
            # Calculate statistics
            total_translations = len(translations)
            total_words = sum(
                len(t.source_text.split())
                for t in translations
            )
            total_cost = sum(
                t.cost or 0
                for t in translations
            )
            
            # Calculate average confidence
            confidences = [
                t.confidence_score
                for t in translations
                if t.confidence_score is not None
            ]
            avg_confidence = (
                sum(confidences) / len(confidences)
                if confidences else 0
            )
            
            # Get language pairs
            language_pairs = []
            seen_pairs = set()
            for t in translations:
                pair = (t.source_lang, t.target_lang)
                if pair not in seen_pairs:
                    language_pairs.append({
                        'source': t.source_lang,
                        'target': t.target_lang
                    })
                    seen_pairs.add(pair)
            
            # Count translation types
            type_counts = {}
            for t in translations:
                type_name = t.translation_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            return TranslationHistoryStats(
                total_translations=total_translations,
                total_words=total_words,
                total_cost=total_cost,
                average_confidence=avg_confidence,
                language_pairs=language_pairs,
                translation_types=type_counts,
                timestamp="2025-02-09 09:30:07",
                processed_by="kaxm23"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch user statistics: {str(e)}"
            )

history_manager = HistoryManager()

@router.get("/translations/history/",
            response_model=Dict,
            description="Get user's translation history")
async def get_translation_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
    translation_type: Optional[TranslationType] = None,
    status: Optional[TranslationStatus] = None,
    min_confidence: Optional[float] = Query(None, ge=0, le=1),
    has_file: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> Dict:
    """
    Get user's translation history with filtering and pagination.
    """
    filters = TranslationHistoryFilter(
        start_date=start_date,
        end_date=end_date,
        source_lang=source_lang,
        target_lang=target_lang,
        translation_type=translation_type,
        status=status,
        min_confidence=min_confidence,
        has_file=has_file
    )
    
    return await history_manager.get_user_translations(
        session=session,
        user_id=current_user.id,
        filters=filters,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order
    )

@router.get("/translations/stats/",
            response_model=TranslationHistoryStats,
            description="Get user's translation statistics")
async def get_user_translation_stats(
    time_range: Optional[int] = Query(None, description="Time range in days"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> TranslationHistoryStats:
    """
    Get user's translation statistics.
    """
    return await history_manager.get_user_stats(
        session=session,
        user_id=current_user.id,
        time_range=time_range
    )