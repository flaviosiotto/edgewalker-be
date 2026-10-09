"""The "ask first" queue (migr. 068, agent bridge F3, decisions D6/D7/D15).

A hosted agent whose tool policy says ``ask`` for a trading tool does not
execute it: agent-svc records an *action request* here (with the arguments
the model produced and its one-line rationale), the request shows up in the
chat as a system row with Approve / Reject, is pushed to the user's webhooks
as ``agent.action.requested`` and can be decided from the UI, from the agent
page or with the MCP tool ``decide_action_request`` (a PAT with the ``trade``
scope — the user's manual channel, never an external agent's PAT).

On approval the backend executes the action itself through the same command
services the API endpoints use, attributed to the agent (actor ``via=agent``)
with the approver recorded in the order's ``extra``. Requests expire after a
TTL (markets move): a sweeper marks them ``expired``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder
from sqlalchemy import text
from sqlmodel import Session, select

from edgewalker_platform.agent_tools import ASKABLE_TOOLS

from app.core.actor import Actor, reset_current_actor, set_current_actor
from app.models.agent import Agent, Chat
from app.models.agent_model import AgentActionRequest
from app.models.strategy import StrategyLive
from app.schemas.agent_model import ActionRequestCreate, ActionRequestRead
from app.services.agent_service import get_agent, require_hosted_agent
from app.services.webhook_service import emit_event

logger = logging.getLogger(__name__)

DEFAULT_TTL = timedelta(minutes=15)
MAX_TTL = timedelta(hours=24)
SWEEP_INTERVAL_S = 30.0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def summarize(tool_name: str, args: dict[str, Any]) -> str:
    """One line a human reads in the chat row, the approval list and the webhook."""
    a = args or {}
    if tool_name == "place_order":
        parts = [f"{_fmt(a.get('side')).upper()} {_fmt(a.get('quantity'))} {_fmt(a.get('symbol'))} {_fmt(a.get('order_type') or 'market')}"]
        if a.get("limit_price"):
            parts.append(f"@ {_fmt(a['limit_price'])}")
        if a.get("stop_price"):
            parts.append(f"stop {_fmt(a['stop_price'])}")
        legs = []
        if a.get("take_profit_price"):
            legs.append(f"TP {_fmt(a['take_profit_price'])}")
        if a.get("stop_loss_price"):
            legs.append(f"SL {_fmt(a['stop_loss_price'])}")
        if legs:
            parts.append("(" + ", ".join(legs) + ")")
        return "Ordine: " + " ".join(p for p in parts if p)
    if tool_name == "close_position":
        return f"Chiudi posizione #{_fmt(a.get('position_id'))} qty {_fmt(a.get('quantity'))} {_fmt(a.get('symbol'))}".rstrip()
    if tool_name == "cancel_order":
        return f"Annulla ordine #{_fmt(a.get('order_id'))}"
    if tool_name == "modify_position_protection":
        legs = []
        if a.get("clear_take_profit"):
            legs.append("rimuovi TP")
        elif a.get("take_profit_price"):
            legs.append(f"TP {_fmt(a['take_profit_price'])}")
        if a.get("clear_stop_loss"):
            legs.append("rimuovi SL")
        elif a.get("stop_loss_price"):
            legs.append(f"SL {_fmt(a['stop_loss_price'])}")
        return f"Protezioni posizione #{_fmt(a.get('position_id'))}: " + (", ".join(legs) or "nessuna modifica")
    return f"{tool_name} {json.dumps(a, ensure_ascii=False, default=str)[:200]}"


def to_read(row: AgentActionRequest, *, agent_name: Optional[str] = None) -> ActionRequestRead:
    return ActionRequestRead(
        id=int(row.id or 0),
        agent_id=row.agent_id,
        agent_name=agent_name,
        chat_id=row.chat_id,
        strategy_live_id=row.strategy_live_id,
        account_id=row.account_id,
        tool_name=row.tool_name,
        args=dict(row.args or {}),
        rationale=row.rationale,
        status=row.status,  # type: ignore[arg-type]
        expires_at=row.expires_at,
        decided_at=row.decided_at,
        decided_by=row.decided_by,
        result=row.result,
        error=row.error,
        created_at=row.created_at,
        summary=summarize(row.tool_name, row.args or {}),
    )


# ── chat rows ───────────────────────────────────────────────────────────────


def _chat_meta(row: AgentActionRequest) -> dict[str, Any]:
    return {
        "action_request": {
            "id": row.id,
            "status": row.status,
            "tool_name": row.tool_name,
            "summary": summarize(row.tool_name, row.args or {}),
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "agent_id": row.agent_id,
        }
    }


def _chat_session_id(session: Session, chat_id: Optional[int]) -> Optional[str]:
    if chat_id is None:
        return None
    chat = session.get(Chat, chat_id)
    if chat is None:
        return None
    return chat.n8n_session_id or str(chat.id)


def _append_system_row(session: Session, row: AgentActionRequest, text_: str) -> Optional[int]:
    """A ``system`` row in the chat of the request (the FE shows it in the
    timeline; the agent sees it as a notification in its next turn)."""
    from app.services.chat_service import persist_asker_message

    session_id = _chat_session_id(session, row.chat_id)
    if session_id is None:
        return None
    try:
        return persist_asker_message(
            session,
            session_id=session_id,
            text=text_,
            sender_kind="system",
            message_type="system",
            metadata=_chat_meta(row),
        )
    except Exception:  # noqa: BLE001 - the queue must work even when the chat row fails
        logger.exception("action request %s: chat row failed", row.id)
        return None


def _patch_request_row(session: Session, row: AgentActionRequest) -> None:
    """Keep the original request row's status in sync, so the chat shows the
    buttons only while the request is pending (also after a reload)."""
    if not row.chat_row_id:
        return
    try:
        session.execute(
            text(
                "UPDATE n8n_chat_histories SET message = jsonb_set(message, '{metadata,action_request,status}', to_jsonb(CAST(:st AS text)), true) "
                "WHERE id = :id"
            ),
            {"st": row.status, "id": row.chat_row_id},
        )
    except Exception:  # noqa: BLE001
        logger.exception("action request %s: chat row patch failed", row.id)


# ── create / list ───────────────────────────────────────────────────────────


def create_request(session: Session, *, user_id: int, agent_id: int, payload: ActionRequestCreate) -> AgentActionRequest:
    agent = require_hosted_agent(get_agent(session, agent_id, user_id), role="author of an action request")
    tool = str(payload.tool_name or "").strip()
    if tool not in ASKABLE_TOOLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{tool}' cannot be queued for approval; askable tools: {', '.join(sorted(ASKABLE_TOOLS))}",
        )
    account_id = payload.account_id
    live_id = payload.strategy_live_id
    if live_id is not None:
        live = session.get(StrategyLive, live_id)
        if live is None or live.strategy is None or live.strategy.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Live session not found")
        if account_id is None:
            account_id = live.account_id
    if account_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="account_id (or strategy_live_id) is required")
    if payload.chat_id is not None:
        chat = session.get(Chat, payload.chat_id)
        if chat is None or chat.user_id != user_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
    ttl = DEFAULT_TTL if payload.ttl_minutes is None else min(timedelta(minutes=payload.ttl_minutes), MAX_TTL)
    now = _utcnow()
    row = AgentActionRequest(
        agent_id=agent.id_agent,
        user_id=user_id,
        chat_id=payload.chat_id,
        strategy_live_id=live_id,
        account_id=account_id,
        tool_name=tool,
        args=dict(payload.args or {}),
        rationale=(payload.rationale or "").strip() or None,
        status="pending",
        expires_at=now + ttl,
        created_at=now,
    )
    session.add(row)
    session.flush()
    summary = summarize(row.tool_name, row.args)
    text_ = f"{agent.agent_name} chiede l'approvazione: {summary}"
    if row.rationale:
        text_ += f"\nMotivo: {row.rationale}"
    text_ += f"\nScade alle {row.expires_at.astimezone(timezone.utc).strftime('%H:%M')} UTC."
    row.chat_row_id = _append_system_row(session, row, text_)
    emit_event(
        session,
        user_id=user_id,
        event_type="agent.action.requested",
        data={
            "request_id": row.id,
            "agent_id": agent.id_agent,
            "agent_name": agent.agent_name,
            "tool_name": row.tool_name,
            "args": row.args,
            "rationale": row.rationale,
            "summary": summary,
            "chat_id": row.chat_id,
            "live_id": row.strategy_live_id,
            "account_id": row.account_id,
            "expires_at": row.expires_at,
        },
        links={"request": f"/agents/{agent.id_agent}?request={row.id}"},
        dedupe_key=f"action:{row.id}:requested",
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_requests(
    session: Session,
    *,
    user_id: int,
    agent_id: Optional[int] = None,
    status_filter: Optional[str] = None,
    live_id: Optional[int] = None,
    chat_id: Optional[int] = None,
    limit: int = 50,
) -> list[AgentActionRequest]:
    stmt = select(AgentActionRequest).where(AgentActionRequest.user_id == user_id)
    if agent_id is not None:
        stmt = stmt.where(AgentActionRequest.agent_id == agent_id)
    if status_filter:
        stmt = stmt.where(AgentActionRequest.status == status_filter)
    if live_id is not None:
        stmt = stmt.where(AgentActionRequest.strategy_live_id == live_id)
    if chat_id is not None:
        stmt = stmt.where(AgentActionRequest.chat_id == chat_id)
    stmt = stmt.order_by(AgentActionRequest.id.desc()).limit(max(1, min(int(limit), 200)))  # type: ignore[union-attr]
    return list(session.exec(stmt).all())


def get_request(session: Session, *, user_id: int, request_id: int) -> AgentActionRequest:
    row = session.get(AgentActionRequest, request_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action request not found")
    return row


# ── decisions ───────────────────────────────────────────────────────────────


def _emit_decided(session: Session, row: AgentActionRequest, agent: Agent | None) -> None:
    emit_event(
        session,
        user_id=row.user_id,
        event_type="agent.action.decided",
        data={
            "request_id": row.id,
            "agent_id": row.agent_id,
            "agent_name": agent.agent_name if agent else None,
            "tool_name": row.tool_name,
            "summary": summarize(row.tool_name, row.args or {}),
            "status": row.status,
            "decided_by": row.decided_by,
            "result": row.result,
            "error": row.error,
        },
        dedupe_key=f"action:{row.id}:{row.status}",
    )


def _expire_row(session: Session, row: AgentActionRequest, agent: Agent | None) -> None:
    row.status = "expired"
    row.decided_at = _utcnow()
    session.add(row)
    _patch_request_row(session, row)
    _append_system_row(session, row, f"Richiesta scaduta senza decisione: {summarize(row.tool_name, row.args or {})}")
    _emit_decided(session, row, agent)


async def decide(
    session: Session,
    *,
    user_id: int,
    request_id: int,
    approve: bool,
    decided_by: dict[str, Any],
    note: Optional[str] = None,
) -> AgentActionRequest:
    row = get_request(session, user_id=user_id, request_id=request_id)
    agent = session.get(Agent, row.agent_id)
    if row.status != "pending":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Request already {row.status}")
    if row.expires_at <= _utcnow():
        _expire_row(session, row, agent)
        session.commit()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Request expired")
    row.decided_at = _utcnow()
    row.decided_by = {**decided_by, **({"note": note} if note else {})}
    summary = summarize(row.tool_name, row.args or {})
    if not approve:
        row.status = "rejected"
        session.add(row)
        _patch_request_row(session, row)
        _append_system_row(session, row, f"Richiesta RIFIUTATA dal trader: {summary}" + (f" — {note}" if note else ""))
        _emit_decided(session, row, agent)
        session.commit()
        session.refresh(row)
        return row

    try:
        result = await _execute(session, row, agent)
        row.status = "approved"
        row.result = result
        row.error = None
    except HTTPException as exc:
        row.status = "failed"
        row.error = str(exc.detail)
    except Exception as exc:  # noqa: BLE001 - the decision is recorded whatever the broker said
        logger.exception("action request %s: execution failed", row.id)
        row.status = "failed"
        row.error = str(exc)[:1000]
    session.add(row)
    _patch_request_row(session, row)
    if row.status == "approved":
        _append_system_row(session, row, f"Richiesta APPROVATA ed eseguita: {summary}")
    else:
        _append_system_row(session, row, f"Richiesta approvata ma l'esecuzione e FALLITA: {summary} — {row.error}")
    _emit_decided(session, row, agent)
    session.commit()
    session.refresh(row)
    return row


async def _execute(session: Session, row: AgentActionRequest, agent: Agent | None) -> dict[str, Any]:
    from app.services.connection_service import get_account

    if row.account_id is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The request has no account")
    account = get_account(session, row.account_id, row.user_id)
    if account is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Account not found")
    args = dict(row.args or {})
    extra = {
        "actor": Actor(
            via="agent",
            user_id=row.user_id,
            agent_id=row.agent_id,
            agent_name=agent.agent_name if agent else None,
            agent_kind=(agent.kind if agent else None) or "hosted",
        ).as_dict(),
        "approval": {"request_id": row.id, "decided_by": row.decided_by},
    }
    token = set_current_actor(
        Actor(via="agent", user_id=row.user_id, agent_id=row.agent_id, agent_name=agent.agent_name if agent else None)
    )
    try:
        if row.tool_name == "place_order":
            from app.services.order_command_service import place_account_order

            result = await place_account_order(
                session,
                account,
                symbol=str(args.get("symbol") or ""),
                side=str(args.get("side") or "").lower(),
                order_type=str(args.get("order_type") or "market").lower(),
                quantity=float(args.get("quantity") or 0),
                limit_price=_opt_float(args.get("limit_price")),
                stop_price=_opt_float(args.get("stop_price")),
                take_profit_price=_opt_float(args.get("take_profit_price")),
                stop_loss_price=_opt_float(args.get("stop_loss_price")),
                strategy_live_id=row.strategy_live_id,
                extra=extra,
            )
        elif row.tool_name == "cancel_order":
            from app.services.order_command_service import cancel_account_order

            result = await cancel_account_order(session, account, str(args.get("order_id") or ""))
        elif row.tool_name == "close_position":
            from app.services.position_command_service import close_account_position

            result = await close_account_position(
                session,
                account,
                str(args.get("position_id") or ""),
                quantity=_opt_float(args.get("quantity")),
                symbol=args.get("symbol") or None,
                reason=str(args.get("reason") or "agent_close_approved"),
                extra=extra,
            )
        elif row.tool_name == "modify_position_protection":
            from app.services.position_command_service import amend_account_position_protection

            result = await amend_account_position_protection(
                session,
                account,
                str(args.get("position_id") or ""),
                take_profit_price=_opt_float(args.get("take_profit_price")),
                stop_loss_price=_opt_float(args.get("stop_loss_price")),
                clear_take_profit=bool(args.get("clear_take_profit")),
                clear_stop_loss=bool(args.get("clear_stop_loss")),
            )
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"No executor for {row.tool_name}")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    finally:
        reset_current_actor(token)
    encoded = jsonable_encoder(result)
    return encoded if isinstance(encoded, dict) else {"result": encoded}


def _opt_float(value: Any) -> Optional[float]:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ── expiry sweeper ─────────────────────────────────────────────────────────


def expire_pending(session: Session) -> int:
    rows = session.exec(
        select(AgentActionRequest).where(
            AgentActionRequest.status == "pending", AgentActionRequest.expires_at <= _utcnow()
        )
    ).all()
    for row in rows:
        _expire_row(session, row, session.get(Agent, row.agent_id))
    if rows:
        session.commit()
    return len(rows)


async def sweeper_loop(stop: asyncio.Event) -> None:
    from app.db.database import get_session_context

    logger.info("action request sweeper started (every %.0fs)", SWEEP_INTERVAL_S)

    def _tick() -> int:
        with get_session_context() as session:
            return expire_pending(session)

    while not stop.is_set():
        try:
            n = await asyncio.to_thread(_tick)
            if n:
                logger.info("action request sweeper: %d expired", n)
        except Exception:  # noqa: BLE001
            logger.exception("action request sweeper tick failed")
        try:
            await asyncio.wait_for(stop.wait(), timeout=SWEEP_INTERVAL_S)
        except asyncio.TimeoutError:
            continue
