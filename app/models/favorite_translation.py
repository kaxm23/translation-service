from sqlalchemy import Column, Integer, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class FavoriteTranslation(Base):
    """
    Favorite translation model.
    Created by: kaxm23
    Created on: 2025-02-09 09:47:47
    """
    __tablename__ = 'favorite_translations'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    translation_id = Column(
        Integer,
        ForeignKey('translation_history.id'),
        nullable=False
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    user = relationship("User", back_populates="favorites")
    translation = relationship("TranslationHistory", back_populates="favorites")
    
    # Ensure unique favorites per user
    __table_args__ = (
        UniqueConstraint('user_id', 'translation_id', name='unique_favorite'),
    )