from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "GPT Translation API"
    VERSION: str = "1.0.0"
    
    # OpenAI settings
    OPENAI_API_KEY: str
    GPT_MODEL: str = "gpt-4"
    DEFAULT_TEMPERATURE: float = 0.3
    
    # Performance settings
    MAX_BATCH_SIZE: int = 10
    REQUEST_TIMEOUT: int = 30
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()