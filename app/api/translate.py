from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

from app.core.security import get_current_user
from app.database.session import get_async_session
from app.services.translation import translate_text
from app.api.translation_notification import (
    TranslationNotification,
    send_translation_notification
)

router = APIRouter()

class TranslationRequest(BaseModel):
    """Translation request model."""
    text: str
    source_lang: str
    target_lang: str
    notify_email: Optional[bool] = True
    timestamp: str = "2025-02-09 09:58:39"
    processed_by: str = "kaxm23"

class TranslationResponse(BaseModel):
    """Translation response model."""
    translation_id: int
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence_score: float
    completion_time: str
    notification_sent: bool
    timestamp: str = "2025-02-09 09:58:39"
    processed_by: str = "kaxm23"

@router.post("/translate",
            response_model=TranslationResponse,
            description="Translate text with email notification")
async def translate(
    request: TranslationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> TranslationResponse:
    """
    Translate text and send email notification.
    
    Args:
        request: Translation request
        background_tasks: Background tasks
        current_user: Current user
        session: Database session
        
    Returns:
        TranslationResponse: Translation result
    """
    try:
        # Perform translation
        translation_result = await translate_text(
            text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            user_id=current_user.id,
            session=session
        )
        
        # Record completion time
        completion_time = datetime.utcnow().isoformat()
        
        # Prepare notification if requested
        notification_sent = False
        if request.notify_email and current_user.email:
            notification = TranslationNotification(
                translation_id=translation_result['id'],
                source_text=request.text,
                translated_text=translation_result['translated_text'],
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                confidence_score=translation_result['confidence_score'],
                completion_time=completion_time,
                user_email=current_user.email,
                timestamp="2025-02-09 09:58:39",
                processed_by="kaxm23"
            )
            
            # Send notification in background
            background_tasks.add_task(
                send_translation_notification,
                notification
            )
            notification_sent = True
        
        return TranslationResponse(
            translation_id=translation_result['id'],
            source_text=request.text,
            translated_text=translation_result['translated_text'],
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            confidence_score=translation_result['confidence_score'],
            completion_time=completion_time,
            notification_sent=notification_sent,
            timestamp="2025-02-09 09:58:39",
            processed_by="kaxm23"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )