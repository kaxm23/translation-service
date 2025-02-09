import aiohttp
import asyncio
import json

async def test_alternative_translations():
    async with aiohttp.ClientSession() as session:
        # Translation with alternatives
        request_data = {
            "text": "The proposal was well received by the board.",
            "source_lang": "en",
            "target_lang": "es",
            "document_type": "general",
            "required_styles": ["formal", "casual", "technical"]
        }
        
        async with session.post(
            "http://localhost:8000/api/v1/translate/alternatives/",
            json=request_data,
            headers={"Authorization": "Bearer your-token"}
        ) as response:
            result = await response.json()
            
            print("\nTranslation Results:")
            print(f"Original: {result['original_text']}")
            print(f"\nPrimary Translation: {result['primary_translation']}")
            
            print("\nAlternative Translations:")
            for alt in result['alternatives']:
                print(f"\nStyle: {alt['style']}")
                print(f"Translation: {alt['text']}")
                print(f"Tone: {alt['tone']}")
                print(f"Formality: {alt['formality_level']}")
                print(f"Confidence: {alt['confidence_score']}")
                print(f"Explanation: {alt['explanation']}")
            
            print(f"\nTokens Used: {result['tokens_used']}")
            print(f"Processing Time: {result['processing_time']:.2f}s")

if __name__ == "__main__":
    asyncio.run(test_alternative_translations())