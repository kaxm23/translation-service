from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import aiofiles
import hashlib
import os
from datetime import datetime
import fitz  # PyMuPDF
from app.models.translation_history import TranslationHistory, TranslationType
from app.database.session import get_async_session
from app.core.config import Settings
from app.core.security import get_current_user
from app.models.user import User

router = APIRouter()
settings = Settings()

class PDFTranslationRequest(BaseModel):
    """PDF translation request model."""
    source_lang: str = Field(..., example="en")
    target_lang: str = Field(..., example="es")
    save_history: bool = Field(True, example=True)
    quality_check: bool = Field(True, example=True)

class PDFTranslationResponse(BaseModel):
    """PDF translation response model."""
    original_file_id: str
    translated_file_id: str
    pages_translated: int
    total_words: int
    processing_time: float
    source_lang: str
    target_lang: str
    translation_quality: float
    file_urls: Dict[str, str]
    history_id: Optional[int]
    timestamp: str = "2025-02-09 09:27:48"
    processed_by: str = "kaxm23"

class PDFTranslator:
    """
    PDF translation service with database storage.
    Created by: kaxm23
    Created on: 2025-02-09 09:27:48
    """
    
    def __init__(self):
        """Initialize PDF translator."""
        self.upload_dir = "uploads/pdfs"
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            'pdfs_processed': 0,
            'pages_translated': 0,
            'total_words': 0
        }

    async def translate_pdf(self,
                          file: UploadFile,
                          source_lang: str,
                          target_lang: str,
                          user: User,
                          session: AsyncSession,
                          save_history: bool = True) -> Dict:
        """
        Translate PDF and store in database.
        
        Args:
            file: PDF file
            source_lang: Source language
            target_lang: Target language
            user: Current user
            session: Database session
            save_history: Save to history
            
        Returns:
            Dict: Translation results
        """
        try:
            start_time = datetime.utcnow()
            
            # Generate file IDs
            original_file_id = self._generate_file_id(file.filename)
            translated_file_id = f"{original_file_id}_translated"
            
            # Save original PDF
            original_path = os.path.join(
                self.upload_dir,
                f"{original_file_id}.pdf"
            )
            translated_path = os.path.join(
                self.upload_dir,
                f"{translated_file_id}.pdf"
            )
            
            await self._save_upload(file, original_path)
            
            # Process PDF
            doc = fitz.open(original_path)
            translated_doc = fitz.open()
            
            total_words = 0
            translated_text = []
            
            # Translate each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Count words
                words = len(text.split())
                total_words += words
                
                # Translate text
                translated_page = await self._translate_text(
                    text,
                    source_lang,
                    target_lang
                )
                translated_text.append(translated_page)
                
                # Create new PDF page
                new_page = translated_doc.new_page(
                    width=page.rect.width,
                    height=page.rect.height
                )
                
                # Add translated text
                new_page.insert_text(
                    point=(50, 50),
                    text=translated_page,
                    fontname="helv",
                    fontsize=11
                )
            
            # Save translated PDF
            translated_doc.save(translated_path)
            
            # Calculate processing time
            process_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create history record if requested
            history_id = None
            if save_history:
                history_record = TranslationHistory(
                    user_id=user.id,
                    username=user.username,
                    source_text="\n".join([doc[i].get_text() for i in range(len(doc))]),
                    translated_text="\n".join(translated_text),
                    source_lang=source_lang,
                    target_lang=target_lang,
                    file_path=original_path,
                    translation_type=TranslationType.FILE,
                    metadata={
                        "original_file_id": original_file_id,
                        "translated_file_id": translated_file_id,
                        "pages": len(doc),
                        "total_words": total_words,
                        "file_type": "pdf",
                        "timestamp": "2025-02-09 09:27:48"
                    }
                )
                
                # Mark as completed
                history_record.mark_completed(
                    tokens=total_words * 1.5,  # Estimate
                    processing_time=process_time,
                    confidence=0.95,
                    cost=total_words * 0.00002  # Example rate
                )
                
                session.add(history_record)
                await session.commit()
                await session.refresh(history_record)
                history_id = history_record.id
            
            # Update statistics
            self._update_stats(len(doc), total_words)
            
            # Clean up
            doc.close()
            translated_doc.close()
            
            return {
                'original_file_id': original_file_id,
                'translated_file_id': translated_file_id,
                'pages_translated': len(doc),
                'total_words': total_words,
                'processing_time': process_time,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_quality': 0.95,  # Example score
                'file_urls': {
                    'original': f"/pdfs/{original_file_id}.pdf",
                    'translated': f"/pdfs/{translated_file_id}.pdf"
                },
                'history_id': history_id,
                'timestamp': "2025-02-09 09:27:48",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            # Clean up on error
            for path in [original_path, translated_path]:
                if os.path.exists(path):
                    os.remove(path)
            raise HTTPException(
                status_code=500,
                detail=f"PDF translation failed: {str(e)}"
            )

    async def _translate_text(self,
                            text: str,
                            source_lang: str,
                            target_lang: str) -> str:
        """Translate text using GPT-4."""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": (
                        f"Translate the following text from {source_lang} "
                        f"to {target_lang}, maintaining formatting and structure."
                    )},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Translation failed: {str(e)}"
            )

    async def _save_upload(self,
                          file: UploadFile,
                          path: str):
        """Save uploaded file."""
        async with aiofiles.open(path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

    def _generate_file_id(self, filename: str) -> str:
        """Generate unique file ID."""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        name_hash = hashlib.md5(filename.encode()).hexdigest()[:10]
        return f"{timestamp}_{name_hash}"

    def _update_stats(self,
                     pages: int,
                     words: int):
        """Update translation statistics."""
        self.stats['pdfs_processed'] += 1
        self.stats['pages_translated'] += pages
        self.stats['total_words'] += words

    def get_statistics(self) -> Dict:
        """Get translation statistics."""
        return {
            'pdfs_processed': self.stats['pdfs_processed'],
            'pages_translated': self.stats['pages_translated'],
            'total_words': self.stats['total_words'],
            'average_words_per_page': (
                self.stats['total_words'] / self.stats['pages_translated']
                if self.stats['pages_translated'] > 0 else 0
            ),
            'timestamp': "2025-02-09 09:27:48",
            'processed_by': "kaxm23"
        }

pdf_translator = PDFTranslator()

@router.post("/translate/pdf/",
             response_model=PDFTranslationResponse)
async def translate_pdf(
    file: UploadFile = File(...),
    request: PDFTranslationRequest = Depends(),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
) -> PDFTranslationResponse:
    """
    Translate PDF document.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    result = await pdf_translator.translate_pdf(
        file=file,
        source_lang=request.source_lang,
        target_lang=request.target_lang,
        user=current_user,
        session=session,
        save_history=request.save_history
    )
    
    return PDFTranslationResponse(**result)

@router.get("/pdf/{file_id}",
            description="Get PDF file by ID")
async def get_pdf(
    file_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get PDF file by ID."""
    file_path = os.path.join(pdf_translator.upload_dir, f"{file_id}.pdf")
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="PDF file not found"
        )
    
    return FileResponse(
        file_path,
        media_type='application/pdf',
        filename=f"{file_id}.pdf"
    )

@router.get("/pdf/history/{history_id}",
            description="Get PDF translation history")
async def get_pdf_history(
    history_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get PDF translation history."""
    record = await session.get(TranslationHistory, history_id)
    
    if not record:
        raise HTTPException(
            status_code=404,
            detail="History record not found"
        )
    
    if record.user_id != current_user.id:
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )
    
    return record.to_dict()

@router.get("/pdf/stats",
            description="Get PDF translation statistics")
async def get_pdf_stats(
    current_user: User = Depends(get_current_user)
):
    """Get PDF translation statistics."""
    return pdf_translator.get_statistics()