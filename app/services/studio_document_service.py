"""Documenti degli Studi legati a una strategia, per i turni chat.

Il runner allega i documenti as-of nei turni ask_agent che origina lui; per
i MESSAGGI DELL'UTENTE il webhook verso n8n parte dal backend, che qui legge
da studio-svc l'ultimo documento pubblicato per ciascuno Studio legato alla
strategia (metadata.studio_documents, stesso contratto del runner). In una
chat live/design l'as-of è "adesso", quindi basta l'asse published.

Auth: service token HS256 (INTERNAL_TOKEN_SECRET condiviso, scope
studios:read) — stesso contratto di edgewalker_platform.auth.service_token,
qui replicato con jose per non aggiungere il kit alle dipendenze del
backend. Best-effort per costruzione: qualsiasi errore = niente documenti,
la chat parte comunque.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from jose import jwt

logger = logging.getLogger(__name__)


def _studio_svc_url() -> str:
    return os.getenv("STUDIO_SVC_URL", "http://studio-svc:8080").rstrip("/")


def strategy_studio_bindings(definition: Any) -> list[dict[str, Any]]:
    """Estrae strategy.studios [{id, slug, name}] dalla definition (opaca)."""
    if not isinstance(definition, dict):
        return []
    strat = definition.get("strategy")
    strat = strat if isinstance(strat, dict) else definition
    studios = strat.get("studios")
    if not isinstance(studios, list):
        return []
    return [b for b in studios if isinstance(b, dict) and b.get("id")]


def _service_token() -> str | None:
    secret = os.getenv("INTERNAL_TOKEN_SECRET", "")
    if not secret:
        return None
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "v": 1,
            "iss": "backend",
            "aud": "studio-svc",
            "sub": "service",
            "scope": ["studios:read"],
            "iat": now,
            "exp": now + timedelta(seconds=300),
        },
        secret,
        algorithm="HS256",
    )


def fetch_latest_studio_documents(
    *, user_id: int, bindings: list[dict[str, Any]], timeout: float = 5.0
) -> list[dict[str, Any]]:
    """Ultimo documento pubblicato per ciascuno Studio legato (o [])."""
    if not bindings:
        return []
    token = _service_token()
    if token is None:
        logger.warning("studio documents: INTERNAL_TOKEN_SECRET assente, salto")
        return []
    try:
        response = httpx.get(
            f"{_studio_svc_url()}/internal/documents",
            params={
                "user_id": user_id,
                "studio_ids": ",".join(str(b["id"]) for b in bindings),
                "axis": "published",
                "include_content": True,
                "limit": 100,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        response.raise_for_status()
        rows = response.json()
    except Exception as exc:  # noqa: BLE001 - best-effort, mai bloccare la chat
        logger.warning("studio documents fetch saltato (%s)", exc)
        return []
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:  # ordinati per published asc: l'ultimo per slug vince
        slug = row.get("slug")
        if slug:
            latest[slug] = row
    return [
        {
            "slug": row.get("slug"),
            "studio_id": row.get("studio_id"),
            "run_id": row.get("run_id"),
            "ref_date": row.get("ref_date"),
            "published_at": row.get("published_at"),
            "title": row.get("title"),
            "document": row.get("document"),
        }
        for row in latest.values()
    ]
