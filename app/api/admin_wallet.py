"""Admin console: platform credit (wallet) and provider token-cost report.

Prefix ``/admin`` (admin-only). Settings, top-up packs (mirrored on the
payment provider through ``sync_packs``), overview, ledger search, per-user
wallet with manual adjustments, and the provider cost report
(``docs/credito-piattaforma-studio.md`` §5b).
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from sqlmodel import Session, select

from app.api.billing import serialize_wallet
from app.core.config import settings
from app.db.database import get_session
from app.models.user import User
from app.models.wallet import CreditPack, WalletLedgerKind
from app.schemas.auth import MessageResponse
from app.schemas.billing import (
    AdminUserWalletRead,
    AdminWalletAdjustRequest,
    AiCostReport,
    CreditPackInput,
    CreditPackRead,
    PackSyncRow,
    PlatformCreditSettingsRead,
    PlatformCreditSettingsUpdate,
    WalletLedgerPage,
    WalletLedgerRead,
    WalletSummaryRead,
)
from app.services import wallet_service
from app.services.ai_cost_report import GROUP_KEYS, build_cost_report
from app.services.billing.billing_service import log_event
from app.utils.auth_utils import get_current_admin_user

router = APIRouter(prefix="/admin", tags=["Admin billing"], dependencies=[Depends(get_current_admin_user)])


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get_user_or_404(session: Session, user_id: int) -> User:
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Utente non trovato")
    return user


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@router.get("/credit/settings", response_model=PlatformCreditSettingsRead)
def read_credit_settings(session: Session = Depends(get_session)):
    return PlatformCreditSettingsRead.model_validate(wallet_service.get_settings(session), from_attributes=True)


@router.put("/credit/settings", response_model=PlatformCreditSettingsRead)
def update_credit_settings(
    payload: PlatformCreditSettingsUpdate,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    row = wallet_service.get_settings(session)
    for field in ("enabled", "price_per_ai_credit_cents", "low_balance_cents", "min_topup_cents"):
        value = getattr(payload, field)
        if value is not None:
            setattr(row, field, value)
    if payload.clear_display_fx:
        row.display_fx_eur_per_usd = None
    elif payload.display_fx_eur_per_usd is not None:
        row.display_fx_eur_per_usd = payload.display_fx_eur_per_usd
    row.updated_at = _utcnow()
    row.updated_by = admin.id
    session.add(row)
    session.commit()
    session.refresh(row)
    return PlatformCreditSettingsRead.model_validate(row, from_attributes=True)


# ---------------------------------------------------------------------------
# Packs
# ---------------------------------------------------------------------------


@router.get("/credit/packs", response_model=list[CreditPackRead])
def list_packs_endpoint(session: Session = Depends(get_session)):
    return [CreditPackRead.model_validate(p, from_attributes=True) for p in wallet_service.list_packs(session, active_only=False)]


@router.post("/credit/packs", response_model=CreditPackRead, status_code=status.HTTP_201_CREATED)
def create_pack_endpoint(payload: CreditPackInput, session: Session = Depends(get_session)):
    if payload.credit_cents < payload.amount_cents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il credito accreditato non puo' essere inferiore al prezzo")
    pack = CreditPack(**payload.model_dump())
    pack.currency = pack.currency.upper()
    session.add(pack)
    session.commit()
    session.refresh(pack)
    return CreditPackRead.model_validate(pack, from_attributes=True)


@router.patch("/credit/packs/{pack_id}", response_model=CreditPackRead)
def update_pack_endpoint(pack_id: int, payload: CreditPackInput, session: Session = Depends(get_session)):
    pack = session.get(CreditPack, pack_id)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pacchetto non trovato")
    if payload.credit_cents < payload.amount_cents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il credito accreditato non puo' essere inferiore al prezzo")
    for field, value in payload.model_dump().items():
        setattr(pack, field, value.upper() if field == "currency" else value)
    pack.updated_at = _utcnow()
    session.add(pack)
    session.commit()
    session.refresh(pack)
    return CreditPackRead.model_validate(pack, from_attributes=True)


@router.delete("/credit/packs/{pack_id}", response_model=MessageResponse)
def delete_pack_endpoint(pack_id: int, session: Session = Depends(get_session)):
    """Packs already sold are kept (top-ups reference them): they are only deactivated."""
    pack = session.get(CreditPack, pack_id)
    if pack is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pacchetto non trovato")
    from app.models.wallet import WalletTopup

    sold = session.exec(select(WalletTopup.id).where(WalletTopup.pack_id == pack_id)).first()
    if sold is not None:
        pack.is_active = False
        pack.updated_at = _utcnow()
        session.add(pack)
        session.commit()
        return MessageResponse(message="Pacchetto gia' venduto: disattivato")
    session.delete(pack)
    session.commit()
    return MessageResponse(message="Pacchetto eliminato")


@router.post("/credit/packs/sync", response_model=list[PackSyncRow])
def sync_packs_endpoint(session: Session = Depends(get_session)):
    if not settings.BILLING_ENABLED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pagamenti disabilitati (BILLING_ENABLED=false)")
    from app.services.billing.checkout_service import sync_packs

    return [PackSyncRow(**row) for row in sync_packs(session)]


# ---------------------------------------------------------------------------
# Overview & ledger
# ---------------------------------------------------------------------------


@router.get("/credit/summary", response_model=WalletSummaryRead)
def credit_summary_endpoint(session: Session = Depends(get_session), days: int = Query(default=30, ge=1, le=365)):
    return WalletSummaryRead(**wallet_service.summary(session, days=days))


def _ledger_page(session: Session, rows, total: int) -> WalletLedgerPage:
    user_ids = {r.user_id for r in rows}
    emails = {
        u.id: u.email for u in session.exec(select(User).where(User.id.in_(list(user_ids)))).all()
    } if user_ids else {}
    items = []
    for r in rows:
        item = WalletLedgerRead.model_validate(r, from_attributes=True)
        item.email = emails.get(r.user_id)
        items.append(item)
    return WalletLedgerPage(items=items, total=total)


@router.get("/credit/ledger", response_model=WalletLedgerPage)
def credit_ledger_endpoint(
    session: Session = Depends(get_session),
    user_id: Optional[int] = Query(default=None),
    kind: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    rows, total = wallet_service.list_ledger(
        session, user_id=user_id, kind=kind, since=since, until=until, limit=limit, offset=offset
    )
    return _ledger_page(session, rows, total)


@router.get("/credit/ledger.csv", include_in_schema=False)
def credit_ledger_csv_endpoint(
    session: Session = Depends(get_session),
    user_id: Optional[int] = Query(default=None),
    kind: Optional[str] = Query(default=None),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
):
    rows, _ = wallet_service.list_ledger(session, user_id=user_id, kind=kind, since=since, until=until, limit=10000)
    page = _ledger_page(session, rows, len(rows))
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(["id", "created_at", "email", "kind", "amount_cents", "balance_after_cents", "ai_credits", "note"])
    for item in page.items:
        writer.writerow([item.id, item.created_at.isoformat(), item.email or "", item.kind, item.amount_cents,
                         item.balance_after_cents, item.ai_credits or "", item.note or ""])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=credito-movimenti.csv"})


# ---------------------------------------------------------------------------
# Per-user wallet
# ---------------------------------------------------------------------------


@router.get("/users/{user_id}/wallet", response_model=AdminUserWalletRead)
def read_user_wallet_endpoint(user_id: int, session: Session = Depends(get_session)):
    user = _get_user_or_404(session, user_id)
    rows, _ = wallet_service.list_ledger(session, user_id=user.id, limit=100)
    return AdminUserWalletRead(
        wallet=serialize_wallet(session, user.id, with_packs=False),
        ledger=[WalletLedgerRead.model_validate(r, from_attributes=True) for r in rows],
    )


@router.post("/users/{user_id}/wallet/adjust", response_model=AdminUserWalletRead)
def adjust_user_wallet_endpoint(
    user_id: int,
    payload: AdminWalletAdjustRequest,
    background_tasks: BackgroundTasks,
    admin: Annotated[User, Depends(get_current_admin_user)],
    session: Session = Depends(get_session),
):
    """Manual credit/charge with an audit note; never takes the balance below zero."""
    user = _get_user_or_404(session, user_id)
    entry = wallet_service.adjust(
        session,
        user_id=user.id,
        amount_cents=payload.amount_cents,
        kind=WalletLedgerKind.REFUND if payload.kind == "refund" else WalletLedgerKind.ADMIN_ADJUST,
        note=payload.note,
        actor_user_id=admin.id,
    )
    log_event(
        session, user_id=user.id, type="wallet_adjusted_by_admin",
        payload={"amount_cents": payload.amount_cents, "note": payload.note, "balance_after_cents": entry.balance_after_cents},
        actor_user_id=admin.id,
    )
    session.commit()
    if payload.notify:
        try:
            from app.services.email_service import queue_email
            from app.services.email_templates import wallet_adjusted_email

            subject, text_body, html_body = wallet_adjusted_email(
                display_name=user.display_name,
                amount_cents=payload.amount_cents,
                balance_cents=entry.balance_after_cents,
                currency=wallet_service.get_settings(session).currency,
                note=payload.note,
            )
            queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
        except Exception:  # noqa: BLE001
            pass
    if payload.amount_cents > 0:
        wallet = wallet_service.get_wallet(session, user.id)
        if wallet is not None and wallet.low_notified_at is not None:
            wallet.low_notified_at = None
            session.add(wallet)
            session.commit()
    return read_user_wallet_endpoint(user_id, session)


# ---------------------------------------------------------------------------
# Provider token cost report
# ---------------------------------------------------------------------------


@router.get("/ai-costs", response_model=AiCostReport)
def ai_costs_endpoint(
    session: Session = Depends(get_session),
    group_by: str = Query(default="model"),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    user_id: Optional[int] = Query(default=None),
    plan_code: Optional[str] = Query(default=None),
    context: Optional[str] = Query(default=None, description="design | live | backtest | runner"),
    estimated: Optional[bool] = Query(default=None),
):
    if group_by not in GROUP_KEYS:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"group_by non valido: {', '.join(GROUP_KEYS)}")
    until = until or _utcnow()
    since = since or (until - timedelta(days=30))
    return build_cost_report(
        session, group_by=group_by, since=since, until=until, provider=provider, model=model,
        user_id=user_id, plan_code=plan_code, context=context, estimated=estimated,
    )


@router.get("/ai-costs.csv", include_in_schema=False)
def ai_costs_csv_endpoint(
    session: Session = Depends(get_session),
    group_by: str = Query(default="model"),
    since: Optional[datetime] = Query(default=None),
    until: Optional[datetime] = Query(default=None),
    provider: Optional[str] = Query(default=None),
    model: Optional[str] = Query(default=None),
    user_id: Optional[int] = Query(default=None),
    plan_code: Optional[str] = Query(default=None),
    context: Optional[str] = Query(default=None),
    estimated: Optional[bool] = Query(default=None),
):
    report = ai_costs_endpoint(session, group_by, since, until, provider, model, user_id, plan_code, context, estimated)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    fields = ["group", "calls", "calls_with_cost", "tokens_input", "tokens_output", "tokens_cached", "tokens_reasoning",
              "cost", "cost_currency", "avg_cost_per_call", "cost_per_1k_tokens", "cached_share", "credits",
              "credits_per_call", "wallet_cents"]
    writer.writerow(fields)
    for row in [*report.rows, report.total]:
        writer.writerow([getattr(row, f) if getattr(row, f) is not None else "" for f in fields])
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=costi-token-{group_by}.csv"})
