from sqlalchemy import Column, Integer, DateTime, ForeignKey, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserTranslationStats(Base):
    """
    User translation statistics model.
    Created by: kaxm23
    Created on: 2025-02-09 10:00:03 UTC
    """
    __tablename__ = 'user_translation_stats'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    total_translations = Column(Integer, default=0)
    total_characters = Column(Integer, default=0)
    avg_confidence_score = Column(Float, default=0.0)
    last_translation_at = Column(DateTime(timezone=True))
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now()
    )

    # Relationships
    user = relationship("User", back_populates="translation_stats")

    # Indexes
    __table_args__ = (
        Index('idx_user_stats', 'user_id', unique=True),
    )

    def update_stats(self, char_count: int, confidence_score: float):
        """Update translation statistics."""
        self.total_translations += 1
        self.total_characters += char_count
        
        # Update rolling average confidence score
        self.avg_confidence_score = (
            (self.avg_confidence_score * (self.total_translations - 1) +
             confidence_score) / self.total_translations
        )
        
        self.last_translation_at = func.now()