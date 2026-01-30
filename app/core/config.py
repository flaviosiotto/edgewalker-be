from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Edgewalker API"
    VERSION: str = "0.1.0"
    API_ROOT_PATH: str = "/api"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = "sqlite:///./app.db"

    SECRET_KEY: str = "change-me"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost"
    ]

    LOG_LEVEL: str = "INFO"
    
    # Market data directory (parquet files)
    MARKETDATA_DIR: str = "/data/raw"

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
