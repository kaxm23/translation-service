import aiohttp
import asyncio
from typing import Dict, List

async def vote_on_translation(
    translation_id: int,
    vote_type: str,
    token: str
) -> Dict:
    """Vote on a translation."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://localhost:8000/api/v1/translations/{translation_id}/vote",
            json={
                "vote_type": vote_type,
                "timestamp": "2025-02-09 10:04:26",
                "processed_by": "kaxm23"
            },
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                print("\nVote submitted successfully:")
                print(f"Translation ID: {result['translation_id']}")
                print(f"Vote Type: {result['vote_type']}")
                print(f"Current Score: {result['score']}")
                print(f"Upvotes: {result['upvotes']}")
                print(f"Downvotes: {result['downvotes']}")
            else:
                print(f"Error: {result.get('detail', 'Unknown error')}")
            
            return result

async def remove_translation_vote(
    translation_id: int,
    token: str
) -> Dict:
    """Remove vote from translation."""
    async with aiohttp.ClientSession() as session:
        async with session.delete(
            f"http://localhost:8000/api/v1/translations/{translation_id}/vote",
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                print("\nVote removed successfully")
            else:
                print(f"Error: {result.get('detail', 'Unknown error')}")
            
            return result

async def get_translation_stats(
    translation_id: int,
    token: str
) -> Dict:
    """Get translation vote statistics."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"http://localhost:8000/api/v1/translations/{translation_id}/votes",
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                print("\nVote Statistics:")
                print(f"Total Votes: {result['total_votes']}")
                print(f"Upvote Ratio: {result['upvote_ratio']:.2%}")
                print(f"User Vote Count: {result['user_vote_count']}")
            else:
                print(f"Error: {result.get('detail', 'Unknown error')}")
            
            return result

async def get_top_translations(
    limit: int = 10,
    offset: int = 0,
    token: str = None
) -> List[Dict]:
    """Get top voted translations."""
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:8000/api/v1/translations/votes/top",
            params={"limit": limit, "offset": offset},
            headers={"Authorization": f"Bearer {token}"}
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                print("\nTop Translations:")
                for translation in result:
                    print(
                        f"\nID: {translation['translation_id']}, "
                        f"Score: {translation['score']}"
                    )
            else:
                print(f"Error: {result.get('detail', 'Unknown error')}")
            
            return result

async def main():
    # Replace with actual token
    token = "your-access-token"
    
    # Vote on a translation
    await vote_on_translation(
        translation_id=1,
        vote_type="upvote",
        token=token
    )
    
    # Get vote statistics
    await get_translation_stats(
        translation_id=1,
        token=token
    )
    
    # Get top translations
    await get_top_translations(
        limit=5,
        token=token
    )
    
    # Remove vote
    await remove_translation_vote(
        translation_id=1,
        token=token
    )

if __name__ == "__main__":
    asyncio.run(main())