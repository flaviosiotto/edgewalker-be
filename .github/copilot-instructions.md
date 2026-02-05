# Copilot Instructions for EDGEWALKER-BE

## Stack
FastAPI + SQLModel + PostgreSQL (pgvector) + Pydantic v2

## Architecture
```
app/
├── main.py          # FastAPI app, lifespan, CORS, routers
├── api/             # Route handlers (thin layer, delegates to services)
├── services/        # Business logic (CRUD, n8n webhooks, sync)
├── models/          # SQLModel ORM models
├── schemas/         # Pydantic request/response schemas
├── db/              # Database session, migrations
├── core/            # Settings, security utils
└── utils/           # Helpers
```

## Key Patterns

### Router → Service Separation
API routes in `api/*.py` are thin – they validate input and delegate to `services/*.py`:
```python
# api/strategies.py
@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(payload: StrategyCreate, session: Session = Depends(get_session)):
    return create_strategy(session, payload)  # service function
```

### Database
- SQLModel (SQLAlchemy + Pydantic hybrid)
- Session via `Depends(get_session)` – auto-managed
- Models in `models/` define tables; schemas in `schemas/` define API contracts
- Migrations: manual SQL files in `migrations/` (no Alembic)

### N8N Integration
Backend triggers n8n workflows via webhooks:
- `N8N_WEBHOOK_BASE_URL`, `N8N_API_KEY` in env
- Agents store `n8n_webhook` URL for their workflow
- `trigger_rule_agent()` in `services/strategy_service.py` POSTs to n8n

### Real-time WebSocket
- `api/ws_marketdata.py` – WebSocket endpoint for UI streaming
- Subscribes to Redis Pub/Sub channels (`live:ticks:*`, `live:bars:*`)
- Fans out to connected WebSocket clients

### Background Sync
- `services/sync_manager.py` – polls datasources for updates
- Controlled by `SYNC_STARTUP_ENABLED`, `SYNC_POLL_INTERVAL_SECONDS`

## Development
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --root-path /api

# With Docker (via devops stack)
./start.sh up -d backend
```

## Environment Variables
Key settings in `core/config.py` (Pydantic BaseSettings):
- `DATABASE_URL` – PostgreSQL connection string
- `SECRET_KEY`, `ALGORITHM` – JWT auth
- `MARKETDATA_DIR` – Path to partitioned OHLCV data (read from edgewalker)
- `N8N_*` – n8n integration settings

## API Structure
- `POST /auth/token` – OAuth2 password flow
- `GET/POST /strategies/` – CRUD strategies
- `POST /strategies/{id}/backtests/{bid}/run` – Trigger backtest via n8n
- `GET /datasources/` – List available market data
- `WS /ws/marketdata` – Real-time streaming

## Conventions
- Use `HTTPException` for error responses
- Service functions return models/raise exceptions (no Response objects)
- Timestamps in UTC, timezone-aware
- Root path `/api` stripped by Traefik – internal routes start at `/`
