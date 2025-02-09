from sqlalchemy import Column, Integer, DateTime, ForeignKey, Enum, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class VoteType(enum.Enum):
    """Translation vote type."""
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"

class TranslationVote(Base):
    """
    Translation vote model.
    Created by: kaxm23
    Created on: 2025-02-09 10:03:07 UTC
    """
    __tablename__ = 'translation_votes'

    id = Column(Integer, primary_key=True)
    translation_id = Column(
        Integer,
        ForeignKey('translation_history.id'),
        nullable=False
    )
    user_id = Column(
        Integer,
        ForeignKey('users.id'),
        nullable=False
    )
    vote_type = Column(Enum(VoteType), nullable=False)
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
    translation = relationship("TranslationHistory", back_populates="votes")
    user = relationship("User", back_populates="translation_votes")

    # Ensure one vote per user per translation
    __table_args__ = (
        UniqueConstraint(
            'user_id',
            'translation_id',
            name='unique_user_translation_vote'
        ),
    )