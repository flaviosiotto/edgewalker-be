from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import logging

from app.core.config import settings
from app.db.database import create_db_and_tables
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.healthcheck import router as system_router
from app.api.agents import router as agents_router
from app.api.strategies import router as strategies_router
from app.api.marketdata import router as marketdata_router
from app.api.datasources import router as datasources_router
from app.services.sync_manager import start_sync_manager, stop_sync_manager
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    
    # Start background sync manager
    if settings.SYNC_STARTUP_ENABLED:
        await start_sync_manager()
        logging.getLogger(__name__).info("Started background sync manager")
    
    yield
    
    # Cleanup
    await stop_sync_manager()


_level_name = (settings.LOG_LEVEL or "INFO").upper().strip()
_valid = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
logging.basicConfig(
    level=_valid.get(_level_name, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logging.getLogger(__name__).info(f"Logging initialized with level {_level_name}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    root_path=settings.API_ROOT_PATH,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(system_router)
app.include_router(agents_router)
app.include_router(strategies_router)
app.include_router(marketdata_router)
app.include_router(datasources_router)
