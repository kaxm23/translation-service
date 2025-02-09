from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core import config

app = FastAPI(
    title=config.settings.PROJECT_NAME,
    version=config.settings.VERSION,
    openapi_url=f"{config.settings.API_V1_STR}/openapi.json"
)

# Set CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(
    api_router,
    prefix=config.settings.API_V1_STR
)