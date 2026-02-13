from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from app.core.config import settings
from app.db.database import create_db_and_tables
from app.api.auth import router as auth_router
from app.api.users import router as users_router
from app.api.healthcheck import router as system_router
from app.api.agents import router as agents_router
from app.api.strategies import router as strategies_router
from app.api.marketdata import router as marketdata_router
from app.api.live import router as live_router
from app.api.connections import router as connections_router
from app.services.connection_manager import start_connection_manager, stop_connection_manager
from app.services.market_price_cache import price_cache
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    
    # Start connection manager (broker connect/disconnect lifecycle)
    await start_connection_manager()
    logging.getLogger(__name__).info("Started connection manager")
    
    # Start market price cache (Redis Pub/Sub → last-price cache for PnL)
    await price_cache.start()
    
    yield
    
    # Cleanup
    await price_cache.stop()
    await stop_connection_manager()


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
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
    lifespan=lifespan
)

_swagger_ui_dir = Path(os.environ.get("SWAGGER_UI_DIR", "/opt/swagger-ui"))
_has_local_swagger_ui = _swagger_ui_dir.exists() and _swagger_ui_dir.is_dir()
if _has_local_swagger_ui:
    app.mount(
        "/static/swagger-ui",
        StaticFiles(directory=str(_swagger_ui_dir)),
        name="swagger-ui",
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html(request: Request):
    root_path = request.scope.get("root_path", "")
    openapi_url = f"{root_path}{app.openapi_url}"
    if _has_local_swagger_ui:
        return get_swagger_ui_html(
            openapi_url=openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url=f"{root_path}/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url=f"{root_path}/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url=f"{root_path}/static/swagger-ui/favicon-32x32.png",
        )

    # Fallback to FastAPI defaults (Swagger UI v5 CDN). Useful for local dev.
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title=f"{app.title} - Swagger UI",
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
app.include_router(live_router)
app.include_router(connections_router)
