from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Edgewalker API"
    VERSION: str = "0.1.0"
    API_ROOT_PATH: str = "/api"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = "sqlite:///./app.db"

    SECRET_KEY: str = "change-me"
    ALGORITHM: str = "RS256"
    JWT_ISSUER: str = "edgewalker-backend"
    ACCESS_TOKEN_AUDIENCE: str = "edgewalker-ui"
    REFRESH_TOKEN_AUDIENCE: str = "edgewalker-refresh"
    N8N_TOKEN_AUDIENCE: str = "edgewalker-n8n"
    RUNNER_TOKEN_AUDIENCE: str = "edgewalker-runner"
    DELEGATED_TOKEN_EXPIRE_MINUTES: int = 480
    JWT_PRIVATE_KEY: Optional[str] = None
    JWT_PUBLIC_KEY: Optional[str] = None
    JWT_PRIVATE_KEY_PATH: Optional[str] = None
    JWT_PUBLIC_KEY_PATH: Optional[str] = None
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30
    PASSWORD_RESET_DEBUG_RETURN_TOKEN: bool = False
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost"
    ]

    LOG_LEVEL: str = "INFO"
    
    # Sync settings
    SYNC_POLL_INTERVAL_SECONDS: int = 60  # How often to check if sources need sync
    SYNC_STARTUP_ENABLED: bool = True      # Run sync at startup

    def _read_secret_material(
        self,
        inline_value: Optional[str],
        path_value: Optional[str],
        label: str,
    ) -> str:
        if inline_value:
            return inline_value.replace("\\n", "\n")

        if path_value:
            return Path(path_value).read_text(encoding="utf-8")

        raise ValueError(f"{label} is required when using {self.ALGORITHM}")

    @property
    def jwt_signing_key(self) -> str:
        if self.ALGORITHM.upper().startswith("HS"):
            return self.SECRET_KEY

        return self._read_secret_material(
            self.JWT_PRIVATE_KEY,
            self.JWT_PRIVATE_KEY_PATH,
            "JWT private key",
        )

    @property
    def jwt_verifying_key(self) -> str:
        if self.ALGORITHM.upper().startswith("HS"):
            return self.SECRET_KEY

        if self.JWT_PUBLIC_KEY or self.JWT_PUBLIC_KEY_PATH:
            return self._read_secret_material(
                self.JWT_PUBLIC_KEY,
                self.JWT_PUBLIC_KEY_PATH,
                "JWT public key",
            )

        return self.jwt_signing_key

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
