import aiohttp
import asyncio
from typing import Dict
from datetime import datetime

async def get_translation_suggestions(
    source_lang: str = "en",
    target_lang: str = "es",
    category: str = None,
    min_confidence: float = 0.7,
    time_range_days: int = 30,
    token: str = None
) -> Dict:
    """Get translation improvement suggestions."""
    async with aiohttp.ClientSession() as session:
        # Prepare query parameters
        params = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "min_confidence": min_confidence,
            "time_range_days": time_range_days
        }
        
        if category:
            params["category"] = category
        
        # Make request
        async with session.get(
            "http://localhost:8000/api/v1/translations/suggestions",
            params=params,
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            data = await response.json()
            
            if response.status == 200:
                print("\nTranslation Improvement Suggestions:")
                print(f"Language Pair: {data['analysis_summary']['language_pair']}")
                print(f"Time Range: {data['analysis_summary']['time_range']}")
                print(f"Confidence Level: {data['analysis_summary']['confidence_level']:.2f}")
                
                print("\nSuggestions:")
                for i, suggestion in enumerate(data['suggestions'], 1):
                    print(f"\n{i}. {suggestion['suggestion_text']}")
                    print(f"Category: {suggestion['category']}")
                    print(f"Confidence: {suggestion['confidence']:.2f}")
                    print(f"Impact Level: {suggestion['impact_level']}")
                    
                    print("\nImplementation Steps:")
                    for step in suggestion['implementation_steps']:
                        print(f"- {step}")
                    
                    print("\nExamples:")
                    for example in suggestion['examples']:
                        print(f"Pattern: {example['error_pattern']}")
                        print(f"Example: {example['example']}")
                
                return data
            else:
                print(f"Error: {data.get('detail', 'Unknown error')}")
                return None

async def main():
    # Replace with actual token
    token = "your-access-token"
    
    # Get suggestions for English to Spanish translations
    await get_translation_suggestions(
        source_lang="en",
        target_lang="es",
        category="grammar",
        min_confidence=0.8,
        time_range_days=30,
        token=token
    )

if __name__ == "__main__":
    asyncio.run(main())