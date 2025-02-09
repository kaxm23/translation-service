from fastapi import APIRouter, HTTPException, Depends, WebSocket
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Literal
from datetime import datetime
import asyncio
import json
import openai
from app.core.config import Settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()
settings = Settings()

class ChatMessage(BaseModel):
    """Chat message model."""
    message_id: str
    timestamp: str = "2025-02-09 09:19:33"
    username: str = "kaxm23"
    message: str
    message_type: Literal["text", "system", "emoji", "file", "url"] = "text"
    source_lang: str
    target_lang: str
    metadata: Optional[Dict] = None

class TranslatedMessage(BaseModel):
    """Translated chat message model."""
    original_message: ChatMessage
    translated_text: str
    detected_elements: Dict
    confidence_score: float
    processing_time: float
    timestamp: str = "2025-02-09 09:19:33"
    processed_by: str = "kaxm23"

class ChatTranslator:
    """
    Real-time chat translation service.
    Created by: kaxm23
    Created on: 2025-02-09 09:19:33
    """
    
    def __init__(self):
        """Initialize chat translator."""
        self.active_sessions = {}
        self.message_cache = {}
        self.stats = {
            'messages_processed': 0,
            'total_tokens': 0,
            'active_users': 0
        }
        
        # Chat element patterns
        self.patterns = {
            'emoji': r'[😀-🙏🌀-🗿]|:[a-z_]+:',
            'mention': r'@[\w]+',
            'url': r'https?://\S+',
            'formatting': r'[*_~`]+.+?[*_~`]+',
            'command': r'/\w+'
        }

    async def translate_message(self,
                              message: ChatMessage) -> TranslatedMessage:
        """
        Translate chat message.
        
        Args:
            message: Chat message to translate
            
        Returns:
            TranslatedMessage: Translated message
        """
        try:
            start_time = datetime.utcnow()
            
            # Detect message elements
            elements = await self._detect_elements(message.message)
            
            # Create translation prompt
            prompt = self._create_chat_prompt(
                message,
                elements
            )
            
            # Translate message
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message.message}
                ],
                temperature=0.7 if elements['has_emoji'] else 0.3
            )
            
            translated_text = response.choices[0].message.content
            
            # Update cache
            self._update_message_cache(
                message.message_id,
                translated_text
            )
            
            # Calculate processing time
            process_time = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            # Update statistics
            self._update_statistics(response.usage.total_tokens)
            
            return TranslatedMessage(
                original_message=message,
                translated_text=translated_text,
                detected_elements=elements,
                confidence_score=self._calculate_confidence(response),
                processing_time=process_time,
                timestamp="2025-02-09 09:19:33",
                processed_by="kaxm23"
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Translation failed: {str(e)}"
            )

    async def _detect_elements(self, text: str) -> Dict:
        """Detect chat message elements."""
        import re
        
        elements = {
            'emojis': [],
            'mentions': [],
            'urls': [],
            'formatting': [],
            'commands': [],
            'has_emoji': False
        }
        
        # Detect emojis
        emojis = re.findall(self.patterns['emoji'], text)
        if emojis:
            elements['emojis'] = emojis
            elements['has_emoji'] = True
        
        # Detect mentions
        elements['mentions'] = re.findall(self.patterns['mention'], text)
        
        # Detect URLs
        elements['urls'] = re.findall(self.patterns['url'], text)
        
        # Detect formatting
        elements['formatting'] = re.findall(self.patterns['formatting'], text)
        
        # Detect commands
        elements['commands'] = re.findall(self.patterns['command'], text)
        
        return elements

    def _create_chat_prompt(self,
                          message: ChatMessage,
                          elements: Dict) -> str:
        """Create chat-specific translation prompt."""
        prompt = (
            f"Translate this chat message from {message.source_lang} "
            f"to {message.target_lang}. "
        )
        
        if elements['has_emoji']:
            prompt += "Preserve emojis and their contextual meaning. "
        
        if elements['mentions']:
            prompt += "Maintain @mentions without translation. "
        
        if elements['formatting']:
            prompt += "Preserve text formatting markers. "
        
        if message.message_type == "system":
            prompt += "This is a system message, maintain formal tone. "
        
        prompt += (
            "Ensure the translation maintains the casual, "
            "conversational style of chat messages while preserving "
            "the original meaning."
        )
        
        return prompt

    def _update_message_cache(self,
                            message_id: str,
                            translation: str) -> None:
        """Update message cache."""
        self.message_cache[message_id] = {
            'translation': translation,
            'timestamp': "2025-02-09 09:19:33",
            'ttl': 3600  # 1 hour cache
        }

    def _calculate_confidence(self, response) -> float:
        """Calculate translation confidence score."""
        base_score = 0.85
        
        if response.choices[0].finish_reason == "stop":
            base_score += 0.1
        
        return min(1.0, base_score)

    def _update_statistics(self, tokens: int) -> None:
        """Update usage statistics."""
        self.stats['messages_processed'] += 1
        self.stats['total_tokens'] += tokens

    def get_statistics(self) -> Dict:
        """Get translation statistics."""
        return {
            'messages_processed': self.stats['messages_processed'],
            'total_tokens': self.stats['total_tokens'],
            'active_users': self.stats['active_users'],
            'average_tokens': (
                self.stats['total_tokens'] / self.stats['messages_processed']
                if self.stats['messages_processed'] > 0 else 0
            ),
            'timestamp': "2025-02-09 09:19:33",
            'processed_by': "kaxm23"
        }

chat_translator = ChatTranslator()

@router.websocket("/ws/translate/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live translation."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = ChatMessage(**json.loads(data))
            
            # Translate message
            translation = await chat_translator.translate_message(message)
            
            # Send translation
            await websocket.send_text(translation.json())
            
    except Exception as e:
        await websocket.close(code=1001, reason=str(e))

@router.post("/translate/chat/",
             response_model=TranslatedMessage,
             description="Translate chat message")
async def translate_chat_message(
    message: ChatMessage,
    current_user: User = Depends(get_current_user)
) -> TranslatedMessage:
    """
    Translate chat message via REST API.
    """
    return await chat_translator.translate_message(message)

@router.post("/translate/chat/batch/",
             response_model=List[TranslatedMessage],
             description="Batch translate chat messages")
async def translate_chat_messages(
    messages: List[ChatMessage],
    current_user: User = Depends(get_current_user)
) -> List[TranslatedMessage]:
    """
    Batch translate chat messages.
    """
    try:
        tasks = [
            chat_translator.translate_message(message)
            for message in messages
        ]
        
        results = await asyncio.gather(*tasks)
        return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch translation failed: {str(e)}"
        )

@router.get("/stats/chat/",
            description="Get chat translation statistics")
async def get_chat_stats(
    current_user: User = Depends(get_current_user)
) -> Dict:
    """Get chat translation statistics."""
    return chat_translator.get_statistics()