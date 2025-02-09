from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()

class RatingType(enum.Enum):
    """Translation rating type."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"

class TranslationRating(Base):
    """
    Translation rating model.
    Created by: kaxm23
    Created on: 2025-02-09 09:43:13
    """
    __tablename__ = 'translation_ratings'

    id = Column(Integer, primary_key=True)
    translation_id = Column(Integer, ForeignKey('translation_history.id'), nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    rating = Column(Enum(RatingType), nullable=False)
    feedback = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    translation = relationship("TranslationHistory", back_populates="ratings")
    user = relationship("User", back_populates="translation_ratings")