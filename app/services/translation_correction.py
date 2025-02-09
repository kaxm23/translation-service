from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Tuple
import openai
from datetime import datetime, timedelta
import asyncio
import json

from app.models.translation_history import TranslationHistory
from app.models.translation_rating import TranslationRating, RatingType

class TranslationCorrector:
    """
    Auto-correction service for low-rated translations using GPT-4.
    Created by: kaxm23
    Created on: 2025-02-09 10:01:51 UTC
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize corrector."""
        self.session = session
        self.model = "gpt-4"
        self.min_confidence = 0.7
        self.correction_threshold = 0.3  # 30% or lower rating triggers correction
        self.feedback_window_days = 30
        self.max_retries = 2
        self.stats = {
            'corrections_made': 0,
            'quality_improved': 0,
            'timestamp': "2025-02-09 10:01:51",
            'processed_by': "kaxm23"
        }

    async def detect_and_correct_translations(self) -> Dict:
        """
        Detect and auto-correct low-rated translations.
        
        Returns:
            Dict: Correction results
        """
        try:
            # Get low-rated translations
            problematic_translations = await self._get_low_rated_translations()
            
            if not problematic_translations:
                return {
                    'status': 'no_corrections_needed',
                    'message': 'No low-rated translations found',
                    'timestamp': "2025-02-09 10:01:51",
                    'processed_by': "kaxm23"
                }
            
            corrections = []
            for translation in problematic_translations:
                # Generate correction
                correction = await self._generate_correction(translation)
                
                if correction:
                    # Apply correction
                    await self._apply_correction(translation['id'], correction)
                    corrections.append({
                        'translation_id': translation['id'],
                        'original': translation['translated_text'],
                        'corrected': correction['text'],
                        'confidence': correction['confidence'],
                        'improvements': correction['improvements']
                    })
            
            return {
                'status': 'success',
                'corrections_made': len(corrections),
                'corrections': corrections,
                'stats': self.stats,
                'timestamp': "2025-02-09 10:01:51",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            raise CorrectionError(f"Correction process failed: {str(e)}")

    async def _get_low_rated_translations(self) -> List[Dict]:
        """Get translations with low ratings."""
        cutoff_date = datetime.utcnow() - timedelta(
            days=self.feedback_window_days
        )
        
        # Query translations with low ratings
        query = (
            select(TranslationHistory, TranslationRating)
            .join(TranslationRating)
            .filter(
                and_(
                    TranslationHistory.created_at >= cutoff_date,
                    TranslationRating.rating == RatingType.THUMBS_DOWN
                )
            )
            .order_by(desc(TranslationHistory.created_at))
        )
        
        result = await self.session.execute(query)
        translations = []
        
        for trans, rating in result:
            # Calculate rating ratio
            ratings_query = await self.session.execute(
                select(TranslationRating)
                .filter_by(translation_id=trans.id)
            )
            all_ratings = ratings_query.scalars().all()
            
            if all_ratings:
                rating_ratio = sum(
                    1 for r in all_ratings
                    if r.rating == RatingType.THUMBS_UP
                ) / len(all_ratings)
                
                # Check if below threshold
                if rating_ratio <= self.correction_threshold:
                    translations.append({
                        'id': trans.id,
                        'source_text': trans.source_text,
                        'translated_text': trans.translated_text,
                        'source_lang': trans.source_lang,
                        'target_lang': trans.target_lang,
                        'confidence_score': trans.confidence_score,
                        'rating_ratio': rating_ratio,
                        'feedback': [r.feedback for r in all_ratings if r.feedback]
                    })
        
        return translations

    async def _generate_correction(self, translation: Dict) -> Optional[Dict]:
        """Generate correction using GPT-4."""
        try:
            # Prepare correction prompt
            prompt = self._build_correction_prompt(translation)
            
            # Get GPT-4 correction
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a translation improvement specialist. Analyze "
                        "and correct the translation while maintaining context "
                        "and natural language flow."
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse correction
            correction = self._parse_correction_response(
                response.choices[0].message.content
            )
            
            # Validate correction
            if await self._validate_correction(
                original=translation['translated_text'],
                correction=correction['text'],
                target_lang=translation['target_lang']
            ):
                return correction
            
            return None
            
        except Exception as e:
            print(f"Correction generation failed: {str(e)}")
            return None

    def _build_correction_prompt(self, translation: Dict) -> str:
        """Build correction prompt."""
        prompt = (
            f"Please improve this translation from {translation['source_lang']} "
            f"to {translation['target_lang']}.\n\n"
            f"Source text: {translation['source_text']}\n"
            f"Current translation: {translation['translated_text']}\n\n"
        )
        
        if translation['feedback']:
            prompt += (
                "User feedback:\n" +
                "\n".join(f"- {feedback}" for feedback in translation['feedback'])
                + "\n\n"
            )
        
        prompt += (
            "Please provide:\n"
            "1. Corrected translation\n"
            "2. List of specific improvements made\n"
            "3. Confidence score (0-1)\n"
            "4. Explanation of changes\n"
        )
        
        return prompt

    def _parse_correction_response(self, response: str) -> Dict:
        """Parse GPT-4 correction response."""
        lines = response.split('\n')
        correction = {
            'text': '',
            'improvements': [],
            'confidence': 0.0,
            'explanation': '',
            'timestamp': "2025-02-09 10:01:51",
            'processed_by': "kaxm23"
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Corrected translation:'):
                current_section = 'text'
                continue
            elif line.startswith('Improvements:'):
                current_section = 'improvements'
                continue
            elif line.startswith('Confidence:'):
                correction['confidence'] = float(
                    line.split(':')[1].strip().rstrip('%')
                ) / 100
                continue
            elif line.startswith('Explanation:'):
                current_section = 'explanation'
                continue
            
            if current_section == 'text':
                correction['text'] += line + ' '
            elif current_section == 'improvements':
                if line.startswith('-'):
                    correction['improvements'].append(line[1:].strip())
            elif current_section == 'explanation':
                correction['explanation'] += line + ' '
        
        correction['text'] = correction['text'].strip()
        correction['explanation'] = correction['explanation'].strip()
        
        return correction

    async def _validate_correction(self,
                                 original: str,
                                 correction: str,
                                 target_lang: str) -> bool:
        """Validate correction quality."""
        if not correction or correction == original:
            return False
        
        # Additional validation could include:
        # 1. Length comparison
        if len(correction) < len(original) * 0.5:
            return False
        
        # 2. Language detection
        # Ensure correction is in target language
        # This would require a language detection service
        
        # 3. Content preservation
        # Check if key terms are preserved
        
        return True

    async def _apply_correction(self,
                              translation_id: int,
                              correction: Dict) -> None:
        """Apply correction to translation."""
        try:
            # Update translation
            query = (
                select(TranslationHistory)
                .filter_by(id=translation_id)
            )
            result = await self.session.execute(query)
            translation = result.scalar_one_or_none()
            
            if translation:
                # Store original in metadata
                if not translation.metadata:
                    translation.metadata = {}
                
                translation.metadata['correction_history'] = {
                    'original_text': translation.translated_text,
                    'correction_date': "2025-02-09 10:01:51",
                    'processed_by': "kaxm23",
                    'improvements': correction['improvements'],
                    'explanation': correction['explanation']
                }
                
                # Update translation
                translation.translated_text = correction['text']
                translation.confidence_score = correction['confidence']
                translation.updated_at = datetime.utcnow()
                
                await self.session.commit()
                
                # Update stats
                self.stats['corrections_made'] += 1
                if correction['confidence'] > translation.confidence_score:
                    self.stats['quality_improved'] += 1
                
        except Exception as e:
            await self.session.rollback()
            raise CorrectionError(f"Failed to apply correction: {str(e)}")

class CorrectionError(Exception):
    """Translation correction error."""
    pass