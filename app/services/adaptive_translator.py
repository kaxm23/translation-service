from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Tuple
import openai
from datetime import datetime, timedelta
from app.models.translation_history import TranslationHistory
from app.models.translation_rating import TranslationRating, RatingType

class AdaptiveTranslator:
    """
    Adaptive translation service using GPT-4 with feedback learning.
    Created by: kaxm23
    Created on: 2025-02-09 09:46:00 UTC
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize translator."""
        self.session = session
        self.model = "gpt-4"
        self.max_history_samples = 5
        self.feedback_window_days = 30
        self.min_confidence_threshold = 0.85
        self.stats = {
            'translations_improved': 0,
            'feedback_incorporated': 0,
            'confidence_adjustments': 0,
            'timestamp': "2025-02-09 09:46:00",
            'processed_by': "kaxm23"
        }

    async def translate(self,
                       text: str,
                       source_lang: str,
                       target_lang: str,
                       user_id: str) -> Dict:
        """
        Translate text with adaptive improvements.
        
        Args:
            text: Text to translate
            source_lang: Source language
            target_lang: Target language
            user_id: User identifier
            
        Returns:
            Dict: Translation result
        """
        try:
            # Get learning context
            learning_context = await self._build_learning_context(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                user_id=user_id
            )
            
            # Prepare prompt with learning context
            prompt = self._build_adaptive_prompt(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                learning_context=learning_context
            )
            
            # Get translation
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                temperature=0.3,
                max_tokens=1000,
                n=1,
                stop=None,
                presence_penalty=0.0,
                frequency_penalty=0.0
            )
            
            translated_text = response.choices[0].message.content
            
            # Apply post-processing based on learned patterns
            translated_text = await self._post_process_translation(
                text=translated_text,
                learning_context=learning_context
            )
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence(
                text=text,
                translated_text=translated_text,
                learning_context=learning_context
            )
            
            return {
                'text': text,
                'translated_text': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'model': self.model,
                'confidence_score': confidence_score,
                'learning_context': {
                    'similar_translations': len(learning_context['similar']),
                    'negative_patterns': len(learning_context['negative']),
                    'positive_patterns': len(learning_context['positive'])
                },
                'timestamp': "2025-02-09 09:46:00",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            raise TranslationError(f"Translation failed: {str(e)}")

    async def _build_learning_context(self,
                                    text: str,
                                    source_lang: str,
                                    target_lang: str,
                                    user_id: str) -> Dict:
        """Build learning context from past translations."""
        context = {
            'similar': [],
            'positive': [],
            'negative': [],
            'common_mistakes': [],
            'style_preferences': {}
        }
        
        try:
            # Get similar past translations
            similar_translations = await self._get_similar_translations(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Analyze ratings and feedback
            for trans in similar_translations:
                ratings = await self._get_translation_ratings(trans.id)
                
                if ratings:
                    avg_rating = sum(
                        1 if r.rating == RatingType.THUMBS_UP else 0
                        for r in ratings
                    ) / len(ratings)
                    
                    if avg_rating > 0.7:
                        context['positive'].append({
                            'source': trans.source_text,
                            'translation': trans.translated_text,
                            'confidence': trans.confidence_score
                        })
                    elif avg_rating < 0.3:
                        context['negative'].append({
                            'source': trans.source_text,
                            'translation': trans.translated_text,
                            'feedback': [r.feedback for r in ratings if r.feedback]
                        })
            
            # Extract common mistakes
            context['common_mistakes'] = await self._analyze_common_mistakes(
                context['negative']
            )
            
            # Get user style preferences
            context['style_preferences'] = await self._get_user_preferences(
                user_id
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to build learning context: {str(e)}")
            return context

    async def _get_similar_translations(self,
                                      text: str,
                                      source_lang: str,
                                      target_lang: str) -> List[TranslationHistory]:
        """Get similar past translations."""
        cutoff_date = datetime.utcnow() - timedelta(
            days=self.feedback_window_days
        )
        
        query = (
            select(TranslationHistory)
            .filter(
                TranslationHistory.source_lang == source_lang,
                TranslationHistory.target_lang == target_lang,
                TranslationHistory.created_at >= cutoff_date
            )
            .order_by(desc(TranslationHistory.confidence_score))
            .limit(self.max_history_samples)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()

    async def _get_translation_ratings(self,
                                     translation_id: int) -> List[TranslationRating]:
        """Get ratings for a translation."""
        query = (
            select(TranslationRating)
            .filter(TranslationRating.translation_id == translation_id)
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()

    async def _analyze_common_mistakes(self,
                                     negative_examples: List[Dict]) -> List[Dict]:
        """Analyze common mistakes from negative examples."""
        mistakes = []
        
        for example in negative_examples:
            if example.get('feedback'):
                for feedback in example['feedback']:
                    # Extract specific error patterns from feedback
                    error_pattern = await self._extract_error_pattern(
                        source=example['source'],
                        translation=example['translation'],
                        feedback=feedback
                    )
                    
                    if error_pattern:
                        mistakes.append(error_pattern)
        
        return mistakes

    async def _get_user_preferences(self, user_id: str) -> Dict:
        """Get user's translation preferences."""
        query = (
            select(TranslationHistory)
            .filter(
                TranslationHistory.user_id == user_id,
                TranslationHistory.confidence_score >= self.min_confidence_threshold
            )
            .order_by(desc(TranslationHistory.created_at))
            .limit(10)
        )
        
        result = await self.session.execute(query)
        translations = result.scalars().all()
        
        preferences = {
            'formality_level': await self._analyze_formality(translations),
            'terminology': await self._extract_terminology(translations),
            'style_markers': await self._identify_style_markers(translations),
            'timestamp': "2025-02-09 09:46:00",
            'processed_by': "kaxm23"
        }
        
        return preferences

    def _build_adaptive_prompt(self,
                             text: str,
                             source_lang: str,
                             target_lang: str,
                             learning_context: Dict) -> Dict:
        """Build adaptive translation prompt."""
        # System prompt with learning context
        system_prompt = (
            f"You are an adaptive translation system that learns from past "
            f"feedback and mistakes. Translate from {source_lang} to "
            f"{target_lang} while considering:\n\n"
            
            "1. Common mistakes to avoid:\n"
            f"{self._format_mistakes(learning_context['common_mistakes'])}\n\n"
            
            "2. Style preferences:\n"
            f"{self._format_preferences(learning_context['style_preferences'])}\n\n"
            
            "3. Positive examples:\n"
            f"{self._format_examples(learning_context['positive'])}\n\n"
            
            "4. Translation guidelines:\n"
            "- Maintain consistent terminology\n"
            "- Preserve formatting and structure\n"
            "- Adapt cultural context appropriately\n"
            "- Consider user's preferred style\n"
        )
        
        # User prompt with text to translate
        user_prompt = (
            f"Please translate the following text from {source_lang} to "
            f"{target_lang}, applying the learned patterns and avoiding "
            f"known mistakes:\n\n{text}"
        )
        
        return {
            'system': system_prompt,
            'user': user_prompt
        }

    async def _post_process_translation(self,
                                      text: str,
                                      learning_context: Dict) -> str:
        """Post-process translation based on learning context."""
        processed_text = text
        
        # Apply terminology preferences
        if learning_context['style_preferences'].get('terminology'):
            processed_text = self._apply_terminology(
                text=processed_text,
                terminology=learning_context['style_preferences']['terminology']
            )
        
        # Apply style markers
        if learning_context['style_preferences'].get('style_markers'):
            processed_text = self._apply_style_markers(
                text=processed_text,
                style_markers=learning_context['style_preferences']['style_markers']
            )
        
        # Fix common mistakes
        for mistake in learning_context['common_mistakes']:
            processed_text = self._fix_mistake_pattern(
                text=processed_text,
                pattern=mistake
            )
        
        return processed_text

    async def _calculate_confidence(self,
                                  text: str,
                                  translated_text: str,
                                  learning_context: Dict) -> float:
        """Calculate confidence score for translation."""
        base_confidence = 0.85  # Base confidence score
        
        # Adjust based on similar translations
        if learning_context['similar']:
            similarity_score = await self._calculate_similarity(
                text=translated_text,
                references=[t['translation'] for t in learning_context['positive']]
            )
            base_confidence += similarity_score * 0.1
        
        # Adjust based on mistake patterns
        if learning_context['common_mistakes']:
            mistake_penalty = len(learning_context['common_mistakes']) * 0.02
            base_confidence -= mistake_penalty
        
        # Adjust based on style compliance
        style_compliance = await self._check_style_compliance(
            text=translated_text,
            preferences=learning_context['style_preferences']
        )
        base_confidence += style_compliance * 0.05
        
        return min(max(base_confidence, 0), 1)  # Ensure score is between 0 and 1

    def _format_mistakes(self, mistakes: List[Dict]) -> str:
        """Format common mistakes for prompt."""
        if not mistakes:
            return "No common mistakes identified."
        
        formatted = []
        for mistake in mistakes:
            formatted.append(
                f"- Avoid: {mistake['pattern']}\n"
                f"  Correct: {mistake['correction']}"
            )
        
        return "\n".join(formatted)

    def _format_preferences(self, preferences: Dict) -> str:
        """Format style preferences for prompt."""
        if not preferences:
            return "No specific style preferences."
        
        formatted = []
        for key, value in preferences.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)

    def _format_examples(self, examples: List[Dict]) -> str:
        """Format positive examples for prompt."""
        if not examples:
            return "No positive examples available."
        
        formatted = []
        for example in examples[:3]:  # Limit to 3 examples
            formatted.append(
                f"Source: {example['source']}\n"
                f"Translation: {example['translation']}"
            )
        
        return "\n".join(formatted)

class TranslationError(Exception):
    """Translation error exception."""
    pass