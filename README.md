# Edgewalker FastAPI Boilerplate

Starter FastAPI + SQLModel con auth JWT, utenti e healthcheck.

## Requisiti
- Python 3.11+

## Setup
1. Copia le variabili ambiente:
   - `.env.example` → `.env`
2. Crea l'ambiente virtuale e installa le dipendenze:
   - `pip install -r requirements.txt`
3. Avvia il server:
   - `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --root-path /api`

## Endpoints
- `POST /auth/token` (OAuth2 password flow)
- `POST /auth/refresh`
- `POST /auth/logout`
- `POST /users/` (registrazione)
- `GET /users/me`
- `GET /health`
- `GET /version`
