from sqlalchemy import Column, Integer, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TranslationHistory(Base):
    """
    Translation history model with voting.
    Created by: kaxm23
    Created on: 2025-02-09 10:03:07 UTC
    """
    __tablename__ = 'translation_history'

    # ... existing fields ...

    # Add vote counts
    upvotes = Column(Integer, default=0)
    downvotes = Column(Integer, default=0)

    # Relationships
    votes = relationship("TranslationVote", back_populates="translation")

    @property
    def vote_score(self) -> int:
        """Get net vote score."""
        return self.upvotes - self.downvotes

    @property
    def vote_ratio(self) -> float:
        """Get vote ratio (upvotes/total)."""
        total = self.upvotes + self.downvotes
        return self.upvotes / total if total > 0 else 0