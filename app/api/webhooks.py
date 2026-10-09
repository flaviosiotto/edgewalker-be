"""Outbound webhooks: endpoints, event catalogue, delivery log (migr. 067).

Managed from an interactive session only (like /pats and /secrets): a leaked
PAT must not be able to point the user's events at an attacker's URL.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.webhook import (
    WebhookDeliveryRead,
    WebhookEventSpec,
    WebhookSubscriptionCreate,
    WebhookSubscriptionCreated,
    WebhookSubscriptionRead,
    WebhookSubscriptionUpdate,
)
from app.services import webhook_service
from app.utils.auth_utils import AuthPrincipal, get_current_active_principal

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def _require_interactive_session(principal: AuthPrincipal) -> None:
    if principal.claims.get("purpose") != "ui_auth":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Webhooks can only be managed from an interactive login session",
        )


def _delivery_read(row) -> WebhookDeliveryRead:
    return WebhookDeliveryRead(
        id=row.id,
        subscription_id=row.subscription_id,
        event_id=str(row.event_id),
        event_type=row.event_type,
        status=row.status,
        attempts=row.attempts,
        max_attempts=row.max_attempts,
        next_attempt_at=row.next_attempt_at if row.status == "pending" else None,
        last_status_code=row.last_status_code,
        last_error=row.last_error,
        last_attempt_at=row.last_attempt_at,
        delivered_at=row.delivered_at,
        created_at=row.created_at,
        payload=row.payload,
    )


@router.get("/events", response_model=list[WebhookEventSpec])
def list_event_types(principal: AuthPrincipal = Depends(get_current_active_principal)):
    """The event catalogue (readable with a PAT too: an agent may want to know
    what it can subscribe its operator to)."""
    return [WebhookEventSpec(**e) for e in webhook_service.event_catalog() if e["type"] != "ping"]


@router.get("/", response_model=list[WebhookSubscriptionRead])
def list_webhooks(
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    return webhook_service.list_subscriptions(session, principal.user.id)


@router.post("/", response_model=WebhookSubscriptionCreated, status_code=status.HTTP_201_CREATED)
def create_webhook(
    payload: WebhookSubscriptionCreate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Register an endpoint. The signing secret is returned once, here."""
    _require_interactive_session(principal)
    row, secret = webhook_service.create_subscription(
        session,
        user_id=principal.user.id,
        name=payload.name,
        url=str(payload.url),
        events=payload.events,
        agent_id=payload.agent_id,
        active=payload.active,
    )
    return WebhookSubscriptionCreated(**WebhookSubscriptionRead.model_validate(row).model_dump(), secret=secret)


@router.patch("/{subscription_id}", response_model=WebhookSubscriptionCreated)
def update_webhook(
    subscription_id: int,
    payload: WebhookSubscriptionUpdate,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Edit an endpoint; ``rotate_secret`` returns a new secret (once)."""
    _require_interactive_session(principal)
    row, secret = webhook_service.update_subscription(
        session,
        user_id=principal.user.id,
        subscription_id=subscription_id,
        name=payload.name,
        url=str(payload.url) if payload.url is not None else None,
        events=payload.events,
        agent_id=payload.agent_id,
        agent_id_set="agent_id" in payload.model_fields_set,
        active=payload.active,
        rotate_secret=payload.rotate_secret,
    )
    return WebhookSubscriptionCreated(**WebhookSubscriptionRead.model_validate(row).model_dump(), secret=secret or "")


@router.delete("/{subscription_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_webhook(
    subscription_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    webhook_service.delete_subscription(session, principal.user.id, subscription_id)
    return None


@router.post("/{subscription_id}/test", response_model=WebhookDeliveryRead, status_code=status.HTTP_202_ACCEPTED)
def test_webhook(
    subscription_id: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    """Enqueue a ``ping`` delivery; the dispatcher sends it within seconds."""
    _require_interactive_session(principal)
    return _delivery_read(webhook_service.enqueue_ping(session, principal.user.id, subscription_id))


@router.get("/{subscription_id}/deliveries", response_model=list[WebhookDeliveryRead])
def list_webhook_deliveries(
    subscription_id: int,
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_principal),
):
    _require_interactive_session(principal)
    rows = webhook_service.list_deliveries(session, principal.user.id, subscription_id, limit=limit)
    return [_delivery_read(r) for r in rows]
