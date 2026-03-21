"""Live alert service.

CRUD operations for persistent live alerts keyed by StrategyLive.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, select

from app.models.strategy import LiveAlert
from app.schemas.live_trading import LiveAlertCreate, LiveAlertUpdate


def create_live_alert(
    session: Session,
    strategy_live_id: int,
    payload: LiveAlertCreate,
) -> LiveAlert:
    alert = LiveAlert(
        strategy_live_id=strategy_live_id,
        request_id=payload.request_id,
        name=payload.name,
        trigger_type=payload.trigger_type,
        trigger=payload.trigger,
        recipient=payload.recipient,
        fire_mode=payload.fire_mode,
        enabled=payload.enabled,
        status=payload.status,
        message=payload.message,
        expires_at=payload.expires_at,
        target=payload.target,
        metadata_json=payload.metadata,
    )
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def get_live_alert(session: Session, alert_id: int) -> LiveAlert | None:
    return session.get(LiveAlert, alert_id)


def list_live_alerts(
    session: Session,
    strategy_live_id: int,
    *,
    enabled: bool | None = None,
    status: str | None = None,
) -> list[LiveAlert]:
    stmt = (
        select(LiveAlert)
        .where(LiveAlert.strategy_live_id == strategy_live_id)
        .order_by(LiveAlert.created_at.desc())  # type: ignore[union-attr]
    )
    if enabled is not None:
        stmt = stmt.where(LiveAlert.enabled == enabled)
    if status:
        stmt = stmt.where(LiveAlert.status == status)
    return list(session.exec(stmt).all())


def update_live_alert(
    session: Session,
    alert_id: int,
    payload: LiveAlertUpdate,
) -> LiveAlert | None:
    alert = session.get(LiveAlert, alert_id)
    if alert is None:
        return None

    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        if key == "metadata":
            key = "metadata_json"
        setattr(alert, key, value)
    alert.updated_at = datetime.now(timezone.utc)

    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def delete_live_alert(session: Session, alert_id: int) -> bool:
    alert = session.get(LiveAlert, alert_id)
    if alert is None:
        return False
    session.delete(alert)
    session.commit()
    return True