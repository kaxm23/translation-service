from fastapi import APIRouter, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr
from typing import Dict, Optional
from datetime import datetime

from app.core.email import EmailSchema, send_email
from app.models.user_auth import User

class TranslationNotification(BaseModel):
    """Translation notification model."""
    translation_id: int
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence_score: float
    completion_time: str
    user_email: EmailStr
    timestamp: str = "2025-02-09 09:58:39"
    processed_by: str = "kaxm23"

async def send_translation_notification(
    notification: TranslationNotification
) -> Dict:
    """Send translation completion notification."""
    try:
        email = EmailSchema(
            recipients=[notification.user_email],
            subject="Translation Complete",
            template_name="translation_complete.html",
            template_data={
                "translation_id": notification.translation_id,
                "source_text": notification.source_text[:100] + "..." if len(notification.source_text) > 100 else notification.source_text,
                "translated_text": notification.translated_text[:100] + "..." if len(notification.translated_text) > 100 else notification.translated_text,
                "source_lang": notification.source_lang,
                "target_lang": notification.target_lang,
                "confidence_score": f"{notification.confidence_score:.2%}",
                "completion_time": notification.completion_time,
                "timestamp": "2025-02-09 09:58:39",
                "processed_by": "kaxm23"
            }
        )
        
        return await send_email(email)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send notification: {str(e)}"
        )