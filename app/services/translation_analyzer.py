from sqlalchemy import select, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Tuple
import openai
from datetime import datetime, timedelta
import json
import asyncio

from app.models.translation_history import TranslationHistory
from app.models.translation_rating import TranslationRating, RatingType

class TranslationAnalyzer:
    """
    Translation error analyzer with GPT-4 suggestions.
    Created by: kaxm23
    Created on: 2025-02-09 09:53:01 UTC
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize analyzer."""
        self.session = session
        self.model = "gpt-4"
        self.analysis_window_days = 90
        self.min_samples = 10
        self.max_samples = 100
        self.confidence_threshold = 0.7
        
        self.stats = {
            'patterns_analyzed': 0,
            'suggestions_generated': 0,
            'improvements_tracked': 0,
            'timestamp': "2025-02-09 09:53:01",
            'processed_by': "kaxm23"
        }

    async def analyze_translation_errors(self,
                                      source_lang: str,
                                      target_lang: str) -> Dict:
        """
        Analyze translation errors and generate improvement suggestions.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Dict: Analysis results and suggestions
        """
        try:
            # Get translation history with errors
            translations = await self._get_problematic_translations(
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            if not translations:
                return {
                    'status': 'no_data',
                    'message': 'Insufficient data for analysis',
                    'timestamp': "2025-02-09 09:53:01",
                    'processed_by': "kaxm23"
                }
            
            # Extract error patterns
            error_patterns = await self._extract_error_patterns(translations)
            
            # Generate improvement suggestions
            suggestions = await self._generate_suggestions(
                error_patterns=error_patterns,
                source_lang=source_lang,
                target_lang=target_lang
            )
            
            # Update statistics
            self.stats['patterns_analyzed'] += len(error_patterns)
            self.stats['suggestions_generated'] += len(suggestions)
            
            return {
                'status': 'success',
                'analysis': {
                    'total_samples': len(translations),
                    'error_patterns': error_patterns,
                    'suggestions': suggestions,
                    'language_pair': f"{source_lang}-{target_lang}",
                    'confidence_level': await self._calculate_confidence(
                        patterns=error_patterns
                    )
                },
                'statistics': {
                    'common_errors': await self._get_error_statistics(
                        error_patterns
                    ),
                    'improvement_trends': await self._get_improvement_trends(
                        translations
                    ),
                    'impact_assessment': await self._assess_error_impact(
                        error_patterns
                    )
                },
                'timestamp': "2025-02-09 09:53:01",
                'processed_by': "kaxm23"
            }
            
        except Exception as e:
            raise AnalysisError(f"Analysis failed: {str(e)}")

    async def _get_problematic_translations(self,
                                          source_lang: str,
                                          target_lang: str) -> List[Dict]:
        """Get translations with negative feedback."""
        cutoff_date = datetime.utcnow() - timedelta(
            days=self.analysis_window_days
        )
        
        # Query translations with negative ratings
        query = (
            select(TranslationHistory, TranslationRating)
            .join(TranslationRating)
            .filter(
                and_(
                    TranslationHistory.source_lang == source_lang,
                    TranslationHistory.target_lang == target_lang,
                    TranslationHistory.created_at >= cutoff_date,
                    TranslationRating.rating == RatingType.THUMBS_DOWN
                )
            )
            .order_by(desc(TranslationHistory.created_at))
            .limit(self.max_samples)
        )
        
        result = await self.session.execute(query)
        translations = []
        
        for trans, rating in result:
            translations.append({
                'id': trans.id,
                'source_text': trans.source_text,
                'translated_text': trans.translated_text,
                'feedback': rating.feedback,
                'confidence_score': trans.confidence_score,
                'created_at': trans.created_at.isoformat(),
                'metadata': trans.metadata
            })
        
        return translations

    async def _extract_error_patterns(self,
                                    translations: List[Dict]) -> List[Dict]:
        """Extract error patterns from translations."""
        try:
            # Prepare analysis prompt
            analysis_prompt = (
                "Analyze the following translations with negative feedback and "
                "identify common error patterns. For each pattern, provide:\n"
                "1. Pattern description\n"
                "2. Error category (grammar, vocabulary, context, style)\n"
                "3. Severity level (1-5)\n"
                "4. Example occurrences\n\n"
                "Translations to analyze:\n"
            )
            
            for t in translations[:5]:  # Analyze top 5 examples
                analysis_prompt += (
                    f"\nSource: {t['source_text']}\n"
                    f"Translation: {t['translated_text']}\n"
                    f"Feedback: {t['feedback']}\n"
                )
            
            # Get GPT-4 analysis
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a translation quality analyst. Identify and "
                        "categorize translation error patterns."
                    )},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse patterns from response
            patterns = self._parse_error_patterns(
                response.choices[0].message.content
            )
            
            return patterns
            
        except Exception as e:
            raise AnalysisError(f"Pattern extraction failed: {str(e)}")

    async def _generate_suggestions(self,
                                  error_patterns: List[Dict],
                                  source_lang: str,
                                  target_lang: str) -> List[Dict]:
        """Generate improvement suggestions using GPT-4."""
        try:
            # Prepare suggestion prompt
            suggestion_prompt = (
                f"Based on the following error patterns in {source_lang} to "
                f"{target_lang} translations, provide specific improvement "
                f"suggestions:\n\n"
            )
            
            for pattern in error_patterns:
                suggestion_prompt += (
                    f"Pattern: {pattern['description']}\n"
                    f"Category: {pattern['category']}\n"
                    f"Severity: {pattern['severity']}\n"
                    f"Examples: {pattern['examples']}\n\n"
                )
            
            # Get GPT-4 suggestions
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "You are a translation improvement advisor. Provide "
                        "specific, actionable suggestions to improve translation "
                        "quality."
                    )},
                    {"role": "user", "content": suggestion_prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            # Parse suggestions
            suggestions = self._parse_suggestions(
                response.choices[0].message.content
            )
            
            return suggestions
            
        except Exception as e:
            raise AnalysisError(f"Suggestion generation failed: {str(e)}")

    def _parse_error_patterns(self, analysis_text: str) -> List[Dict]:
        """Parse error patterns from GPT-4 response."""
        patterns = []
        current_pattern = {}
        
        for line in analysis_text.split('\n'):
            if line.startswith('Pattern:'):
                if current_pattern:
                    patterns.append(current_pattern)
                current_pattern = {'description': line[8:].strip()}
            elif line.startswith('Category:'):
                current_pattern['category'] = line[9:].strip()
            elif line.startswith('Severity:'):
                current_pattern['severity'] = int(line[9:].strip())
            elif line.startswith('Examples:'):
                current_pattern['examples'] = line[9:].strip()
        
        if current_pattern:
            patterns.append(current_pattern)
        
        return patterns

    def _parse_suggestions(self, suggestion_text: str) -> List[Dict]:
        """Parse improvement suggestions from GPT-4 response."""
        suggestions = []
        current_suggestion = {}
        
        for line in suggestion_text.split('\n'):
            if line.startswith('Suggestion:'):
                if current_suggestion:
                    suggestions.append(current_suggestion)
                current_suggestion = {'text': line[11:].strip()}
            elif line.startswith('Implementation:'):
                current_suggestion['implementation'] = line[15:].strip()
            elif line.startswith('Impact:'):
                current_suggestion['impact'] = line[7:].strip()
        
        if current_suggestion:
            suggestions.append(current_suggestion)
        
        return suggestions

    async def _calculate_confidence(self,
                                  patterns: List[Dict]) -> float:
        """Calculate confidence score for analysis."""
        if not patterns:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of patterns (more patterns = lower confidence)
        # 2. Pattern severity (higher severity = lower confidence)
        # 3. Pattern consistency (consistent patterns = higher confidence)
        
        base_confidence = 0.9  # Start with high confidence
        
        # Adjust for number of patterns
        pattern_penalty = len(patterns) * 0.05
        base_confidence -= min(pattern_penalty, 0.3)
        
        # Adjust for severity
        avg_severity = sum(p['severity'] for p in patterns) / len(patterns)
        severity_penalty = avg_severity * 0.1
        base_confidence -= severity_penalty
        
        return max(min(base_confidence, 1.0), 0.0)

    async def _get_error_statistics(self,
                                  patterns: List[Dict]) -> Dict:
        """Calculate error pattern statistics."""
        stats = {
            'categories': {},
            'severity_distribution': {
                1: 0, 2: 0, 3: 0, 4: 0, 5: 0
            },
            'total_patterns': len(patterns)
        }
        
        for pattern in patterns:
            # Count categories
            category = pattern['category']
            stats['categories'][category] = (
                stats['categories'].get(category, 0) + 1
            )
            
            # Count severity levels
            stats['severity_distribution'][pattern['severity']] += 1
        
        return stats

    async def _get_improvement_trends(self,
                                    translations: List[Dict]) -> Dict:
        """Analyze improvement trends over time."""
        trends = {
            'weekly_improvements': {},
            'confidence_trend': [],
            'common_fixes': []
        }
        
        # Group by week
        for trans in translations:
            week = datetime.fromisoformat(
                trans['created_at']
            ).strftime('%Y-%W')
            
            if week not in trends['weekly_improvements']:
                trends['weekly_improvements'][week] = {
                    'total': 0,
                    'improved': 0
                }
            
            trends['weekly_improvements'][week]['total'] += 1
            if trans['confidence_score'] > self.confidence_threshold:
                trends['weekly_improvements'][week]['improved'] += 1
        
        return trends

    async def _assess_error_impact(self,
                                 patterns: List[Dict]) -> Dict:
        """Assess impact of error patterns."""
        impact = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for pattern in patterns:
            if pattern['severity'] >= 5:
                impact['critical'].append(pattern)
            elif pattern['severity'] >= 4:
                impact['high'].append(pattern)
            elif pattern['severity'] >= 3:
                impact['medium'].append(pattern)
            else:
                impact['low'].append(pattern)
        
        return impact

class AnalysisError(Exception):
    """Analysis error exception."""
    pass