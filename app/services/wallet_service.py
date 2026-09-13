"""Platform credit (prepaid wallet) — docs/credito-piattaforma-studio.md.

Balance and ledger live in the database only. Every movement goes through
:func:`adjust`, which locks the wallet row, writes the ledger entry with the
balance after it and keeps ``user_wallet.balance_cents`` in sync.

Order of consumption for AI turns (``entitlement_service``): plan credits of
the period first, then the wallet (if enabled globally and by the user), then
a 402. The overage charge of one turn is computed by
:func:`charge_ai_overage` and is idempotent per ``ai_credit_ledger`` row (the
estimate → real-token replacement re-runs it and adjusts the difference).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from fastapi import BackgroundTasks, HTTPException, status
from sqlalchemy import func
from sqlmodel import Session, select

from app.models.billing import AiCreditLedger
from app.models.user import User
from app.models.wallet import (
    CreditPack,
    PlatformCreditSettings,
    TopupStatus,
    UserWallet,
    WalletLedger,
    WalletLedgerKind,
    WalletTopup,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def get_settings(session: Session) -> PlatformCreditSettings:
    row = session.get(PlatformCreditSettings, 1)
    if row is None:
        row = PlatformCreditSettings(id=1)
        session.add(row)
        session.commit()
        session.refresh(row)
    return row


def wallet_enabled(session: Session) -> bool:
    try:
        return bool(get_settings(session).enabled)
    except Exception:  # noqa: BLE001 - table missing on a DB that predates 059
        session.rollback()
        return False


def credits_to_cents(settings_row: PlatformCreditSettings, credits: Decimal) -> int:
    """Cents charged for ``credits`` AI credits at the configured price
    (rounded half-up to the cent; never negative)."""
    if credits <= 0:
        return 0
    cents = (Decimal(credits) * Decimal(settings_row.price_per_ai_credit_cents)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return max(0, int(cents))


# ---------------------------------------------------------------------------
# Wallet & ledger
# ---------------------------------------------------------------------------


def get_wallet(session: Session, user_id: int, *, for_update: bool = False) -> Optional[UserWallet]:
    stmt = select(UserWallet).where(UserWallet.user_id == user_id)
    if for_update:
        stmt = stmt.with_for_update()
    return session.exec(stmt).first()


def get_or_create_wallet(session: Session, user_id: int, *, for_update: bool = False) -> UserWallet:
    wallet = get_wallet(session, user_id, for_update=for_update)
    if wallet is None:
        wallet = UserWallet(user_id=user_id, currency=get_settings(session).currency)
        session.add(wallet)
        session.flush()
        if for_update:
            wallet = get_wallet(session, user_id, for_update=True) or wallet
    return wallet


def balance_cents(session: Session, user_id: int) -> int:
    wallet = get_wallet(session, user_id)
    return int(wallet.balance_cents) if wallet else 0


def wallet_usable_for_ai(session: Session, user_id: int) -> bool:
    """Globally enabled, not disabled by the user and with a positive balance."""
    if not wallet_enabled(session):
        return False
    wallet = get_wallet(session, user_id)
    return wallet is not None and wallet.auto_use_for_ai and int(wallet.balance_cents) > 0


def adjust(
    session: Session,
    *,
    user_id: int,
    amount_cents: int,
    kind: WalletLedgerKind | str,
    note: str | None = None,
    actor_user_id: int | None = None,
    ai_credits: Decimal | None = None,
    ai_ledger_id: int | None = None,
    topup_id: int | None = None,
    allow_negative: bool = False,
    commit: bool = True,
) -> WalletLedger:
    """One wallet movement (positive = credit, negative = charge). Locks the
    wallet row; refuses to take the balance below zero unless
    ``allow_negative`` (only the AI overage of a single turn may do that)."""
    if amount_cents == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Importo nullo")
    wallet = get_or_create_wallet(session, user_id, for_update=True)
    new_balance = int(wallet.balance_cents) + int(amount_cents)
    if new_balance < 0 and not allow_negative:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "wallet_insufficient",
                "message": f"Credito insufficiente: saldo {wallet.balance_cents / 100:.2f} {wallet.currency}",
            },
        )
    wallet.balance_cents = new_balance
    wallet.updated_at = _utcnow()
    entry = WalletLedger(
        user_id=user_id,
        amount_cents=int(amount_cents),
        balance_after_cents=new_balance,
        kind=kind.value if isinstance(kind, WalletLedgerKind) else str(kind),
        ai_credits=ai_credits,
        ai_ledger_id=ai_ledger_id,
        topup_id=topup_id,
        note=note,
        actor_user_id=actor_user_id,
    )
    session.add(wallet)
    session.add(entry)
    if commit:
        session.commit()
        session.refresh(entry)
    else:
        session.flush()
    return entry


def charge_ai_overage(
    session: Session,
    *,
    user_id: int,
    ai_ledger: AiCreditLedger,
    credits_over: Decimal,
    commit: bool = True,
) -> int:
    """Charge (or re-charge) the wallet for the credits of one turn that
    exceed the period allowance. Idempotent per ledger row: a second call
    with a different ``credits_over`` (estimate replaced by real tokens)
    adjusts the existing movement by the difference. Returns the cents now
    charged for this turn (0 when nothing is due)."""
    settings_row = get_settings(session)
    target_cents = credits_to_cents(settings_row, credits_over) if settings_row.enabled else 0
    existing = session.exec(select(WalletLedger).where(WalletLedger.ai_ledger_id == ai_ledger.id)).first()
    if existing is None and target_cents > 0:
        # The wallet pays only when it was usable for the turn: a user who
        # never topped up keeps the small last-turn overshoot of the plan
        # for free (as before the wallet existed) instead of a negative balance.
        wallet = get_wallet(session, user_id)
        if wallet is None or not wallet.auto_use_for_ai or int(wallet.balance_cents) <= 0:
            return 0
    already = -int(existing.amount_cents) if existing is not None else 0
    delta = target_cents - already
    if delta == 0:
        return already
    if existing is None:
        adjust(
            session,
            user_id=user_id,
            amount_cents=-delta,
            kind=WalletLedgerKind.AI_OVERAGE,
            ai_credits=credits_over,
            ai_ledger_id=ai_ledger.id,
            note=f"Eccedenza crediti AI ({credits_over:.3f} crediti)",
            allow_negative=True,
            commit=commit,
        )
        return target_cents
    # Re-charge: move the balance by the difference and rewrite the movement so
    # the ledger keeps one row per turn.
    wallet = get_or_create_wallet(session, user_id, for_update=True)
    wallet.balance_cents = int(wallet.balance_cents) - delta
    wallet.updated_at = _utcnow()
    existing.amount_cents = -target_cents
    existing.balance_after_cents = int(existing.balance_after_cents) - delta
    existing.ai_credits = credits_over
    existing.note = f"Eccedenza crediti AI ({credits_over:.3f} crediti, rettifica)"
    session.add(wallet)
    session.add(existing)
    if commit:
        session.commit()
    else:
        session.flush()
    return target_cents


def notify_balance(
    session: Session, user_id: int, *, background_tasks: BackgroundTasks | None, ai_period_key: date | None = None
) -> None:
    """Low-balance / exhausted emails, at most once per threshold crossing
    (low: reset when the balance goes back above; exhausted: once per AI
    period)."""
    settings_row = get_settings(session)
    if not settings_row.enabled:
        return
    wallet = get_wallet(session, user_id)
    if wallet is None:
        return
    balance = int(wallet.balance_cents)
    kind: str | None = None
    now = _utcnow()
    if balance <= 0:
        if ai_period_key is not None and wallet.exhausted_notified_period != ai_period_key:
            wallet.exhausted_notified_period = ai_period_key
            kind = "exhausted"
    elif balance <= settings_row.low_balance_cents:
        if wallet.low_notified_at is None:
            wallet.low_notified_at = now
            kind = "low"
    else:
        if wallet.low_notified_at is not None:
            wallet.low_notified_at = None
            session.add(wallet)
            session.commit()
    if kind is None:
        return
    session.add(wallet)
    session.commit()
    user = session.get(User, user_id)
    if user is None:
        return
    try:
        from app.services.email_service import queue_email
        from app.services.email_templates import wallet_exhausted_email, wallet_low_email

        builder = wallet_exhausted_email if kind == "exhausted" else wallet_low_email
        subject, text_body, html_body = builder(
            display_name=user.display_name, balance_cents=balance, currency=wallet.currency
        )
        queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
    except Exception:  # noqa: BLE001 - a mail failure must not break accounting
        logger.exception("Wallet %s notification failed for user %s", kind, user_id)


# ---------------------------------------------------------------------------
# Top-ups
# ---------------------------------------------------------------------------


def list_packs(session: Session, *, active_only: bool = True) -> list[CreditPack]:
    stmt = select(CreditPack)
    if active_only:
        stmt = stmt.where(CreditPack.is_active == True)  # noqa: E712
    return list(session.exec(stmt.order_by(CreditPack.sort_order, CreditPack.id)).all())


def open_topup(
    session: Session, *, user_id: int, pack: CreditPack, provider: str, checkout_external_id: str
) -> WalletTopup:
    topup = WalletTopup(
        user_id=user_id,
        pack_id=pack.id,
        amount_cents=pack.amount_cents,
        credit_cents=pack.credit_cents,
        currency=pack.currency,
        provider=provider,
        checkout_external_id=checkout_external_id,
    )
    session.add(topup)
    session.commit()
    session.refresh(topup)
    return topup


def settle_topup(
    session: Session,
    *,
    provider: str,
    checkout_external_id: str,
    payment_external_id: str | None,
    background_tasks: BackgroundTasks | None = None,
) -> Optional[WalletTopup]:
    """Mark a top-up paid and credit the wallet. Idempotent: a second delivery
    of the same checkout returns ``None``."""
    topup = session.exec(
        select(WalletTopup)
        .where(WalletTopup.provider == provider)
        .where(WalletTopup.checkout_external_id == checkout_external_id)
        .with_for_update()
    ).first()
    if topup is None:
        logger.warning("Top-up settlement for unknown checkout %s/%s", provider, checkout_external_id)
        return None
    if topup.status == TopupStatus.PAID.value:
        return None
    topup.status = TopupStatus.PAID.value
    topup.payment_external_id = payment_external_id
    topup.paid_at = _utcnow()
    session.add(topup)
    session.flush()
    adjust(
        session,
        user_id=topup.user_id,
        amount_cents=topup.credit_cents,
        kind=WalletLedgerKind.TOPUP,
        topup_id=topup.id,
        note=f"Ricarica {topup.amount_cents / 100:.2f} {topup.currency}",
        commit=True,
    )
    user = session.get(User, topup.user_id)
    if user is not None:
        try:
            from app.core.config import settings as app_settings
            from app.services.email_service import queue_email
            from app.services.email_templates import wallet_topup_email

            subject, text_body, html_body = wallet_topup_email(
                display_name=user.display_name,
                amount_cents=topup.amount_cents,
                credit_cents=topup.credit_cents,
                balance_cents=balance_cents(session, user.id),
                currency=topup.currency,
            )
            queue_email(background_tasks, to_address=user.email, subject=subject, text_body=text_body, html_body=html_body)
            if app_settings.BILLING_ADMIN_NOTIFY_EMAIL:
                queue_email(
                    background_tasks,
                    to_address=app_settings.BILLING_ADMIN_NOTIFY_EMAIL,
                    subject=f"[admin] {subject} ({user.email})",
                    text_body=text_body,
                    html_body=html_body,
                )
        except Exception:  # noqa: BLE001
            logger.exception("Top-up email failed for user %s", topup.user_id)
    # The balance went up: allow the low-balance email to fire again later.
    wallet = get_wallet(session, topup.user_id)
    if wallet is not None and wallet.low_notified_at is not None and int(wallet.balance_cents) > get_settings(session).low_balance_cents:
        wallet.low_notified_at = None
        session.add(wallet)
        session.commit()
    return topup


# ---------------------------------------------------------------------------
# Read models
# ---------------------------------------------------------------------------


@dataclass
class WalletView:
    enabled: bool
    currency: str
    balance_cents: int
    auto_use_for_ai: bool
    price_per_ai_credit_cents: Decimal
    low_balance_cents: int
    min_topup_cents: int


def wallet_view(session: Session, user_id: int) -> WalletView:
    settings_row = get_settings(session)
    wallet = get_wallet(session, user_id)
    return WalletView(
        enabled=bool(settings_row.enabled),
        currency=wallet.currency if wallet else settings_row.currency,
        balance_cents=int(wallet.balance_cents) if wallet else 0,
        auto_use_for_ai=wallet.auto_use_for_ai if wallet else True,
        price_per_ai_credit_cents=Decimal(settings_row.price_per_ai_credit_cents),
        low_balance_cents=int(settings_row.low_balance_cents),
        min_topup_cents=int(settings_row.min_topup_cents),
    )


def list_ledger(
    session: Session,
    *,
    user_id: int | None = None,
    kind: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[WalletLedger], int]:
    stmt = select(WalletLedger)
    count_stmt = select(func.count(WalletLedger.id))
    if user_id is not None:
        stmt = stmt.where(WalletLedger.user_id == user_id)
        count_stmt = count_stmt.where(WalletLedger.user_id == user_id)
    if kind:
        stmt = stmt.where(WalletLedger.kind == kind)
        count_stmt = count_stmt.where(WalletLedger.kind == kind)
    if since is not None:
        stmt = stmt.where(WalletLedger.created_at >= since)
        count_stmt = count_stmt.where(WalletLedger.created_at >= since)
    if until is not None:
        stmt = stmt.where(WalletLedger.created_at < until)
        count_stmt = count_stmt.where(WalletLedger.created_at < until)
    total = int(session.exec(count_stmt).one())
    rows = list(session.exec(stmt.order_by(WalletLedger.created_at.desc(), WalletLedger.id.desc()).offset(offset).limit(limit)).all())
    return rows, total


def balances_by_user(session: Session) -> dict[int, int]:
    return {int(uid): int(bal) for uid, bal in session.exec(select(UserWallet.user_id, UserWallet.balance_cents)).all()}


def summary(session: Session, *, days: int = 30) -> dict[str, Any]:
    """Console overview: outstanding balance (a liability), top-ups and
    consumption over the last ``days``, users under the low threshold."""
    settings_row = get_settings(session)
    since = _utcnow() - timedelta(days=days)
    outstanding = int(session.exec(select(func.coalesce(func.sum(UserWallet.balance_cents), 0))).one())
    wallets_with_balance = int(
        session.exec(select(func.count(UserWallet.user_id)).where(UserWallet.balance_cents > 0)).one()
    )
    low = int(
        session.exec(
            select(func.count(UserWallet.user_id))
            .where(UserWallet.balance_cents > 0)
            .where(UserWallet.balance_cents <= settings_row.low_balance_cents)
        ).one()
    )

    def _sum(kind: str, sign: int) -> tuple[int, int]:
        total, count = session.exec(
            select(func.coalesce(func.sum(WalletLedger.amount_cents), 0), func.count(WalletLedger.id))
            .where(WalletLedger.kind == kind)
            .where(WalletLedger.created_at >= since)
        ).one()
        return int(total) * sign, int(count)

    topups_cents, topups_count = _sum(WalletLedgerKind.TOPUP.value, 1)
    overage_cents, overage_count = _sum(WalletLedgerKind.AI_OVERAGE.value, -1)
    adjust_cents, adjust_count = _sum(WalletLedgerKind.ADMIN_ADJUST.value, 1)
    overage_credits = session.exec(
        select(func.coalesce(func.sum(WalletLedger.ai_credits), 0))
        .where(WalletLedger.kind == WalletLedgerKind.AI_OVERAGE.value)
        .where(WalletLedger.created_at >= since)
    ).one()
    return {
        "currency": settings_row.currency,
        "days": days,
        "outstanding_cents": outstanding,
        "wallets_with_balance": wallets_with_balance,
        "users_low_balance": low,
        "topups_cents": topups_cents,
        "topups_count": topups_count,
        "overage_cents": overage_cents,
        "overage_count": overage_count,
        "overage_credits": float(overage_credits or 0),
        "admin_adjust_cents": adjust_cents,
        "admin_adjust_count": adjust_count,
    }
