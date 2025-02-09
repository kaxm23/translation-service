from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    """
    User model with translation statistics.
    Created by: kaxm23
    Created on: 2025-02-09 10:00:03 UTC
    """
    __tablename__ = 'users'

    # ... existing User model fields ...

    # Add relationships
    translations = relationship("TranslationHistory", back_populates="user")
    translation_stats = relationship(
        "UserTranslationStats",
        back_populates="user",
        uselist=False
    )

    def get_translation_stats(self) -> dict:
        """Get user's translation statistics."""
        if not self.translation_stats:
            return {
                'total_translations': 0,
                'total_characters': 0,
                'avg_confidence_score': 0.0,
                'last_translation_at': None,
                'timestamp': "2025-02-09 10:00:03",
                'processed_by': "kaxm23"
            }
        
        return {
            'total_translations': self.translation_stats.total_translations,
            'total_characters': self.translation_stats.total_characters,
            'avg_confidence_score': self.translation_stats.avg_confidence_score,
            'last_translation_at': self.translation_stats.last_translation_at,
            'timestamp': "2025-02-09 10:00:03",
            'processed_by': "kaxm23"
        }