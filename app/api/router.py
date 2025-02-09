from fastapi import APIRouter
from app.api import (
    translate,
    votes,
    favorites,
    translation_stats
)

api_router = APIRouter()

# Include all routers
api_router.include_router(
    translate.router,
    tags=["translations"]
)
api_router.include_router(
    votes.router,
    tags=["votes"]
)
api_router.include_router(
    favorites.router,
    tags=["favorites"]
)
api_router.include_router(
    translation_stats.router,
    tags=["statistics"]
)