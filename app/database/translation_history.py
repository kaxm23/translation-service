from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import logging
from datetime import datetime

# Initialize Base
Base = declarative_base()

class TranslationRecord(Base):
    """Translation history record model."""
    __tablename__ = 'translation_history'
    
    id = Column(Integer, primary_key=True)
    source_text = Column(String, nullable=False)
    translated_text = Column(String, nullable=False)
    source_lang = Column(String(5), nullable=False)
    target_lang = Column(String(5), nullable=False)
    document_type = Column(String(50))
    user_id = Column(String(50), nullable=False)
    tokens_used = Column(Integer)
    cost = Column(Float)
    confidence_score = Column(Float)
    processing_time = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class TranslationError(Base):
    """Translation error record model."""
    __tablename__ = 'translation_errors'
    
    id = Column(Integer, primary_key=True)
    translation_id = Column(Integer, ForeignKey('translation_history.id'))
    error_type = Column(String(50))
    error_message = Column(String)
    error_details = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    translation = relationship("TranslationRecord")

class TranslationHistoryManager:
    """
    Translation history manager using SQLite and SQLAlchemy.
    Created by: kaxm23
    Created on: 2025-02-09 09:22:12 UTC
    """
    
    def __init__(self,
                 database_url: str = "sqlite+aiosqlite:///translations.db",
                 log_level: int = logging.INFO):
        """
        Initialize translation history manager.
        
        Args:
            database_url: SQLite database URL
            log_level: Logging level
        """
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.engine = create_async_engine(
            database_url,
            echo=True
        )
        
        # Initialize session maker
        self.async_session = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Initialize statistics
        self.stats = {
            'records_added': 0,
            'errors_logged': 0,
            'queries_executed': 0
        }

    async def init_db(self):
        """Initialize database."""
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    @asynccontextmanager
    async def get_session(self):
        """Get database session."""
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def add_translation(self,
                            source_text: str,
                            translated_text: str,
                            source_lang: str,
                            target_lang: str,
                            user_id: str,
                            document_type: Optional[str] = None,
                            tokens_used: Optional[int] = None,
                            cost: Optional[float] = None,
                            confidence_score: Optional[float] = None,
                            processing_time: Optional[float] = None,
                            metadata: Optional[Dict] = None) -> int:
        """
        Add translation record to history.
        
        Args:
            source_text: Original text
            translated_text: Translated text
            source_lang: Source language
            target_lang: Target language
            user_id: User identifier
            document_type: Type of document
            tokens_used: Number of tokens used
            cost: Translation cost
            confidence_score: Translation confidence
            processing_time: Processing time
            metadata: Additional metadata
            
        Returns:
            int: Record ID
        """
        try:
            record = TranslationRecord(
                source_text=source_text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                document_type=document_type,
                user_id=user_id,
                tokens_used=tokens_used,
                cost=cost,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata=metadata,
                created_at=datetime.utcnow()
            )
            
            async with self.get_session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)
                
                self.stats['records_added'] += 1
                return record.id
                
        except Exception as e:
            self.logger.error(f"Failed to add translation record: {str(e)}")
            raise

    async def log_error(self,
                       translation_id: int,
                       error_type: str,
                       error_message: str,
                       error_details: Optional[Dict] = None):
        """
        Log translation error.
        
        Args:
            translation_id: Translation record ID
            error_type: Type of error
            error_message: Error message
            error_details: Additional error details
        """
        try:
            error = TranslationError(
                translation_id=translation_id,
                error_type=error_type,
                error_message=error_message,
                error_details=error_details,
                created_at=datetime.utcnow()
            )
            
            async with self.get_session() as session:
                session.add(error)
                await session.commit()
                
                self.stats['errors_logged'] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")
            raise

    async def get_user_history(self,
                             user_id: str,
                             limit: int = 100,
                             offset: int = 0) -> List[Dict]:
        """
        Get user's translation history.
        
        Args:
            user_id: User identifier
            limit: Maximum records to return
            offset: Record offset
            
        Returns:
            List[Dict]: Translation records
        """
        try:
            async with self.get_session() as session:
                query = (
                    session.query(TranslationRecord)
                    .filter(TranslationRecord.user_id == user_id)
                    .order_by(TranslationRecord.created_at.desc())
                    .offset(offset)
                    .limit(limit)
                )
                
                result = await session.execute(query)
                records = result.scalars().all()
                
                self.stats['queries_executed'] += 1
                
                return [
                    {
                        'id': record.id,
                        'source_text': record.source_text,
                        'translated_text': record.translated_text,
                        'source_lang': record.source_lang,
                        'target_lang': record.target_lang,
                        'document_type': record.document_type,
                        'tokens_used': record.tokens_used,
                        'cost': record.cost,
                        'confidence_score': record.confidence_score,
                        'processing_time': record.processing_time,
                        'metadata': record.metadata,
                        'created_at': record.created_at.isoformat(),
                        'user_id': "kaxm23"
                    }
                    for record in records
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get user history: {str(e)}")
            raise

    async def get_statistics(self,
                           user_id: Optional[str] = None) -> Dict:
        """
        Get translation statistics.
        
        Args:
            user_id: Optional user identifier for user-specific stats
            
        Returns:
            Dict: Translation statistics
        """
        try:
            async with self.get_session() as session:
                # Base query
                query = session.query(TranslationRecord)
                
                if user_id:
                    query = query.filter(TranslationRecord.user_id == user_id)
                
                # Get counts
                total_count = await session.execute(query.count())
                total_tokens = await session.execute(
                    query.with_entities(
                        func.sum(TranslationRecord.tokens_used)
                    )
                )
                total_cost = await session.execute(
                    query.with_entities(
                        func.sum(TranslationRecord.cost)
                    )
                )
                
                self.stats['queries_executed'] += 3
                
                return {
                    'total_translations': total_count.scalar(),
                    'total_tokens': total_tokens.scalar() or 0,
                    'total_cost': f"${total_cost.scalar() or 0:.2f}",
                    'records_added': self.stats['records_added'],
                    'errors_logged': self.stats['errors_logged'],
                    'queries_executed': self.stats['queries_executed'],
                    'timestamp': "2025-02-09 09:22:12",
                    'user_id': user_id or "all",
                    'processed_by': "kaxm23"
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            raise

    async def cleanup_old_records(self,
                                days: int = 30):
        """
        Clean up old translation records.
        
        Args:
            days: Age of records to delete in days
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            async with self.get_session() as session:
                # Delete old errors
                await session.execute(
                    delete(TranslationError)
                    .where(TranslationError.created_at < cutoff_date)
                )
                
                # Delete old translations
                await session.execute(
                    delete(TranslationRecord)
                    .where(TranslationRecord.created_at < cutoff_date)
                )
                
                await session.commit()
                
                self.logger.info(f"Cleaned up records older than {days} days")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup records: {str(e)}")
            raise