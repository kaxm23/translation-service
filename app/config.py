from pydantic import BaseSettings
from typing import Optional
from datetime import datetime

class Settings(BaseSettings):
    """Application settings."""
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Translation Service"
    VERSION: str = "1.0.0"
    CREATED_AT: str = "2025-02-09 10:06:13"
    PROCESSED_BY: str = "kaxm23"
    
    # Database Settings
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: Optional[str] = None
    
    # JWT Settings
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Email Settings
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: str
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_TLS: bool = True
    MAIL_SSL: bool = False
    
    # OpenAI Settings
    OPENAI_API_KEY: str
    GPT_MODEL: str = "gpt-4"
    
    # Redis Settings (for caching)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()