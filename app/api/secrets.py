"""Secrets di piattaforma per-utente (gestione: Settings → Secrets).

Write-only per la UI: i valori si salvano e si eliminano, mai si rileggono
da qui. Chi li consuma sono i runtime — oggi gli Studi: studio-svc legge la
stessa tabella (chiave Fernet condivisa SECRETS_ENCRYPTION_KEY) per l'env
dei run schedulati e per ew_studio.get_secret nel Lab.
"""

import re
from datetime import datetime, timezone

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from app.core.config import settings
from app.db.database import get_session
from app.models.user_secret import UserSecret
from app.schemas.secrets import SecretRead, SecretWrite
from app.utils.auth_utils import AuthPrincipal, get_current_active_principal

router = APIRouter(prefix="/secrets", tags=["Secrets"])

# Stile env-var: nei run degli Studi il nome finisce nell'environment.
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MAX_SECRET_BYTES = 4 * 1024


def _require_interactive_session(principal: AuthPrincipal) -> None:
    # Un PAT non deve poter scrivere credenziali che poi altri runtime
    # eseguono: gestione riservata alla sessione UI (come /pats e /lab).
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Secrets can only be managed from an interactive login session",
        )


def _fernet() -> Fernet:
    if not settings.SECRETS_ENCRYPTION_KEY:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="SECRETS_ENCRYPTION_KEY non configurata: secrets non disponibili",
        )
    return Fernet(settings.SECRETS_ENCRYPTION_KEY.encode())


@router.get("/", response_model=list[SecretRead])
def list_secrets(
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    rows = session.exec(
        select(UserSecret)
        .where(UserSecret.user_id == principal.user.id)
        .order_by(UserSecret.name)
    ).all()
    return [SecretRead(name=r.name, created_at=r.created_at, updated_at=r.updated_at)
            for r in rows]


@router.put("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def put_secret(
    name: str,
    payload: SecretWrite,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    if not _NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=422,
            detail="Nome secret non valido: maiuscole/cifre/underscore, es. OPENROUTER_API_KEY",
        )
    if len(payload.value.encode("utf-8")) > _MAX_SECRET_BYTES:
        raise HTTPException(status_code=413, detail="Valore secret troppo grande")
    encrypted = _fernet().encrypt(payload.value.encode("utf-8")).decode("ascii")
    secret = session.exec(
        select(UserSecret)
        .where(UserSecret.user_id == principal.user.id, UserSecret.name == name)
    ).first()
    if secret is None:
        secret = UserSecret(
            user_id=principal.user.id, name=name, value_encrypted=encrypted
        )
    else:
        secret.value_encrypted = encrypted
        secret.updated_at = datetime.now(timezone.utc)
    session.add(secret)
    session.commit()


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_secret(
    name: str,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    secret = session.exec(
        select(UserSecret)
        .where(UserSecret.user_id == principal.user.id, UserSecret.name == name)
    ).first()
    if secret is None:
        raise HTTPException(status_code=404, detail="Secret non trovato")
    session.delete(secret)
    session.commit()
