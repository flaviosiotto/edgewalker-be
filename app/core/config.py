from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Edgewalker API"
    VERSION: str = "0.1.0"
    API_ROOT_PATH: str = "/api"
    API_V1_STR: str = "/api/v1"

    CLIENT_PORTAL_ACCESS_BASE_URL: str = ""
    CLIENT_PORTAL_PROXY_BRIDGE_TOKEN: str = ""
    CLIENT_PORTAL_LAUNCH_TTL_SECONDS: int = 900
    # Path-based browser routing: when enabled, the launch flow redirects the
    # browser straight to the per-connection container under
    # ``CLIENT_PORTAL_ROUTING_BASE_URL/<prefix>/<connection_id>`` instead of
    # proxying the login through this backend.
    CLIENT_PORTAL_PATH_ROUTING_ENABLED: bool = False
    CLIENT_PORTAL_ROUTING_BASE_URL: str = ""
    CLIENT_PORTAL_PATH_PREFIX_BASE: str = "/ib-access"

    DATABASE_URL: str = "sqlite:///./app.db"
    # Connection pool sizing. Sync `def` endpoints run in FastAPI's threadpool
    # (default 40) and each pins a DB connection for the whole request, so the
    # pool must be able to cover that concurrency plus the background loops
    # (health/auth-reconcile/live-runner monitor). pool_size + max_overflow is
    # the hard ceiling of connections opened toward Postgres per worker process.
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800

    SECRET_KEY: str = "change-me"
    ALGORITHM: str = "RS256"
    JWT_ISSUER: str = "edgewalker-backend"
    ACCESS_TOKEN_AUDIENCE: str = "edgewalker-ui"
    REFRESH_TOKEN_AUDIENCE: str = "edgewalker-refresh"
    N8N_TOKEN_AUDIENCE: str = "edgewalker-n8n"
    N8N_WEBHOOK_JWT_SHARED_SECRET: Optional[str] = None
    N8N_WEBHOOK_JWT_ISSUER: Optional[str] = None
    N8N_WEBHOOK_JWT_AUDIENCE: Optional[str] = None
    N8N_WEBHOOK_JWT_EXPIRE_MINUTES: int = 15
    RUNNER_TOKEN_AUDIENCE: str = "edgewalker-runner"
    AGENT_TOKEN_AUDIENCE: str = "edgewalker-agent"
    DELEGATED_TOKEN_EXPIRE_MINUTES: int = 480
    AGENT_CALLBACK_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_PRIVATE_KEY: Optional[str] = None
    JWT_PUBLIC_KEY: Optional[str] = None
    JWT_PRIVATE_KEY_PATH: Optional[str] = None
    JWT_PUBLIC_KEY_PATH: Optional[str] = None
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 30
    PASSWORD_RESET_DEBUG_RETURN_TOKEN: bool = False
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Outbound email. With EMAIL_ENABLED off the message is logged instead of
    # delivered, which is how local dev runs without an SMTP server.
    EMAIL_ENABLED: bool = False
    SMTP_HOST: str = "mailpit"
    SMTP_PORT: int = 1025
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    # Left unset these auto-negotiate: aiosmtplib upgrades the connection when
    # the server advertises STARTTLS. Forcing them to False would silently send
    # credentials in the clear against a provider that supports encryption.
    SMTP_STARTTLS: Optional[bool] = None
    SMTP_TLS: Optional[bool] = None
    SMTP_TIMEOUT_SECONDS: int = 15
    SMTP_MAX_ATTEMPTS: int = 3
    EMAIL_FROM_ADDRESS: str = "no-reply@edgewalker.tech"
    EMAIL_FROM_NAME: str = "Edgewalker"

    # Public origin of the SPA, used to build the links embedded in emails.
    FRONTEND_BASE_URL: str = "http://localhost:5173"

    # REDIS_URL wins when set, matching how connection_manager, live_runner and
    # backtest_runner already resolve Redis. Deployments hand us a full URL with
    # credentials and a generated hostname, so host/port alone are not enough.
    REDIS_URL: str = ""
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_USERNAME: str = ""
    REDIS_PASSWORD: str = ""

    # Fixed-window caps for the unauthenticated password reset endpoint, applied
    # per identifier and per client IP.
    PASSWORD_RESET_MAX_PER_IDENTIFIER_PER_HOUR: int = 5
    PASSWORD_RESET_MAX_PER_IP_PER_HOUR: int = 20

    # First-run admin seeding. Applied only when the user table is empty, which
    # replaces the previous "first user to hit POST /users/ becomes admin" rule.
    BOOTSTRAP_ADMIN_EMAIL: Optional[str] = None
    BOOTSTRAP_ADMIN_USERNAME: str = "admin"
    BOOTSTRAP_ADMIN_PASSWORD: Optional[str] = None

    # Who may end up with a usable account:
    #   closed             -> only allowlisted emails may register at all
    #   family_and_friends -> allowlisted go straight through, everyone else
    #                         lands in the administrator approval queue
    #   open               -> anyone becomes active once their email is verified
    # Google OAuth (Authorization Code + PKCE, mediated by this backend).
    # The redirect URI must match a value registered in the Google console
    # character for character, or Google refuses the exchange.
    GOOGLE_OAUTH_CLIENT_ID: Optional[str] = None
    GOOGLE_OAUTH_CLIENT_SECRET: Optional[str] = None
    GOOGLE_OAUTH_REDIRECT_URI: str = ""
    OAUTH_STATE_TTL_SECONDS: int = 600
    OAUTH_EXCHANGE_TTL_SECONDS: int = 120
    OAUTH_HTTP_TIMEOUT_SECONDS: int = 15

    REGISTRATION_MODE: str = "family_and_friends"
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 48
    REGISTRATION_MAX_PER_IP_PER_HOUR: int = 10
    EMAIL_VERIFICATION_MAX_RESENDS_PER_HOUR: int = 5

    BACKEND_CORS_ORIGINS: List[str] = []

    LOG_LEVEL: str = "INFO"
    LIVE_RECONCILIATION_ENABLED: bool = False
    CTRADER_OAUTH_CLIENT_ID: str = ""
    CTRADER_OAUTH_CLIENT_SECRET: str = ""
    
    # Sync settings
    SYNC_POLL_INTERVAL_SECONDS: int = 60  # How often to check if sources need sync
    SYNC_STARTUP_ENABLED: bool = True      # Run sync at startup

    @field_validator("SMTP_STARTTLS", "SMTP_TLS", mode="before")
    @classmethod
    def _blank_is_unset(cls, value):
        """Treat an empty env var as "not configured".

        docker-compose renders an unset optional variable as the empty string,
        which pydantic would otherwise reject as an invalid boolean and take the
        whole process down at startup.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

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

    @property
    def google_oauth_enabled(self) -> bool:
        return bool(
            self.GOOGLE_OAUTH_CLIENT_ID
            and self.GOOGLE_OAUTH_CLIENT_SECRET
            and self.GOOGLE_OAUTH_REDIRECT_URI
        )

    @property
    def password_reset_debug_token_enabled(self) -> bool:
        """Return the raw reset token in the API response.

        Only honoured while email delivery is off: once mail actually goes out
        the token must never travel back to an unauthenticated caller, since
        that turns the endpoint into an account takeover primitive.
        """
        return self.PASSWORD_RESET_DEBUG_RETURN_TOKEN and not self.EMAIL_ENABLED

    @property
    def n8n_webhook_jwt_issuer(self) -> str:
        return self.N8N_WEBHOOK_JWT_ISSUER or self.JWT_ISSUER

    @property
    def n8n_webhook_jwt_audience(self) -> str:
        return self.N8N_WEBHOOK_JWT_AUDIENCE or self.N8N_TOKEN_AUDIENCE

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
