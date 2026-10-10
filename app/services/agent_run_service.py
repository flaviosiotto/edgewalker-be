"""Runs: EdgeWalker as an agent (migr. 069, bridge F4, decisions D12-D14).

A run is a task handed to a HOSTED agent from outside:

* ``POST /agents/{id}/runs`` (REST, any orchestrator, optional callback);
* the Paperclip ``http`` adapter (``POST /agents/{id}/paperclip`` → 202; the
  adapter is fire-and-forget, so the outcome goes back through the Paperclip
  API as a comment + status on the issue the agent was woken for);
* A2A ``message/send`` on ``/a2a/agents/{id}`` (``tasks/get`` to poll).

It is executed as ONE turn in a chat of the agent — its own chat for
``scope=agent`` (briefing, creation, comparison, memory), a design chat of
the strategy for ``scope=strategy`` (the agent may edit the strategy) —
through the ordinary chat path, so the turn has the same context, tools,
policy, credits and audit trail as a message typed by the user. The task
never emits orders: trading stays in the strategies (D13). The answer, the
usage and the credits of the turn are copied on the run and, when a
callback is configured, pushed to the caller.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.models.agent import Agent, Chat
from app.models.agent_model import AgentRun
from app.models.strategy import Strategy
from app.schemas.agent_run import PaperclipHeartbeat, RunCreate, RunRead
from app.schemas.chat import ChatCreate
from app.services.agent_service import get_agent, require_hosted_agent
from app.services.webhook_service import _encrypt, _fernet, emit_event

logger = logging.getLogger(__name__)

CALLBACK_TIMEOUT_S = 15.0
RUN_STUCK_AFTER = timedelta(minutes=20)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _decrypt(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        return _fernet().decrypt(value.encode("ascii")).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def to_read(run: AgentRun, *, agent_name: Optional[str] = None) -> RunRead:
    return RunRead(
        id=int(run.id or 0),
        agent_id=run.agent_id,
        agent_name=agent_name,
        source=run.source,  # type: ignore[arg-type]
        external_run_id=run.external_run_id,
        scope=run.scope,  # type: ignore[arg-type]
        strategy_id=run.strategy_id,
        chat_id=run.chat_id,
        task=run.task,
        context=dict(run.context or {}),
        status=run.status,  # type: ignore[arg-type]
        result=run.result,
        error=run.error,
        usage=run.usage,
        cost_credits=float(run.cost_credits) if run.cost_credits is not None else None,
        callback_url=run.callback_url,
        callback_status=run.callback_status,
        callback_error=run.callback_error,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


# ── budget ──────────────────────────────────────────────────────────────────


def credits_spent_this_month(session: Session, agent_id: int) -> float:
    start = _utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    total = session.exec(
        select(func.coalesce(func.sum(AgentRun.cost_credits), 0)).where(
            AgentRun.agent_id == agent_id, AgentRun.created_at >= start
        )
    ).one()
    return float(total or 0)


def check_budget(session: Session, agent: Agent) -> None:
    cap = getattr(agent, "budget_credits_month", None)
    if cap is None:
        return
    spent = credits_spent_this_month(session, agent.id_agent)
    if spent >= cap:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=f"Monthly run budget of agent '{agent.agent_name}' exhausted ({spent:.1f}/{cap} credits)",
        )


# ── chat resolution ─────────────────────────────────────────────────────────


def _agent_chat(session: Session, agent: Agent) -> Chat:
    chat = session.exec(
        select(Chat)
        .where(Chat.id_agent == agent.id_agent, Chat.user_id == agent.user_id)
        .where(Chat.strategy_id.is_(None), Chat.live_id.is_(None), Chat.backtest_id.is_(None))  # type: ignore[union-attr]
        .order_by(Chat.id)
    ).first()
    if chat is not None:
        return chat
    chat = Chat(
        user_id=agent.user_id,
        id_agent=agent.id_agent,
        nome=f"{agent.agent_name} Default",
        descrizione="Chat predefinita",
        chat_type=Chat.ChatType.USER,
        created_at=datetime.now(),
    )
    session.add(chat)
    session.commit()
    session.refresh(chat)
    return chat


def _strategy_chat(session: Session, agent: Agent, strategy_id: int) -> Chat:
    strategy = session.get(Strategy, strategy_id)
    if strategy is None or strategy.user_id != agent.user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    chat = session.exec(
        select(Chat)
        .where(Chat.strategy_id == strategy_id, Chat.user_id == agent.user_id, Chat.id_agent == agent.id_agent)
        .order_by(Chat.id)
    ).first()
    if chat is not None:
        return chat
    from app.services.strategy_service import create_strategy_chat

    return create_strategy_chat(
        session,
        strategy_id,
        ChatCreate(nome=f"{agent.agent_name} · run", id_agent=agent.id_agent, chat_type=Chat.ChatType.STRATEGY),
        agent.user_id,
    )


# ── create / read ───────────────────────────────────────────────────────────


def create_run(session: Session, *, user_id: int, agent_id: int, payload: RunCreate, created_by: Optional[dict[str, Any]] = None) -> AgentRun:
    agent = require_hosted_agent(get_agent(session, agent_id, user_id), role="target of a run")
    check_budget(session, agent)
    if payload.scope == "strategy":
        if payload.strategy_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="strategy_id is required for scope=strategy")
        chat = _strategy_chat(session, agent, payload.strategy_id)
    else:
        chat = _agent_chat(session, agent)
    if payload.callback_url and not payload.callback_url.lower().startswith(("https://", "http://")):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="callback_url must be an http(s) URL")
    run = AgentRun(
        agent_id=agent.id_agent,
        user_id=user_id,
        source=payload.source,
        external_run_id=payload.external_run_id,
        scope=payload.scope,
        strategy_id=payload.strategy_id if payload.scope == "strategy" else None,
        chat_id=chat.id,
        task=payload.task.strip(),
        context=dict(payload.context or {}),
        status="queued",
        callback_url=payload.callback_url,
        callback_auth=_encrypt(payload.callback_auth) if payload.callback_auth else None,
        created_by=created_by,
        created_at=_utcnow(),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def get_run(session: Session, *, user_id: int, run_id: int) -> AgentRun:
    run = session.get(AgentRun, run_id)
    if run is None or run.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run


def list_runs(session: Session, *, user_id: int, agent_id: Optional[int] = None, status_filter: Optional[str] = None, limit: int = 50) -> list[AgentRun]:
    stmt = select(AgentRun).where(AgentRun.user_id == user_id)
    if agent_id is not None:
        stmt = stmt.where(AgentRun.agent_id == agent_id)
    if status_filter:
        stmt = stmt.where(AgentRun.status == status_filter)
    stmt = stmt.order_by(AgentRun.id.desc()).limit(max(1, min(int(limit), 200)))  # type: ignore[union-attr]
    return list(session.exec(stmt).all())


# ── execution ───────────────────────────────────────────────────────────────


def _task_message(run: AgentRun) -> str:
    """The text the agent receives: the task, then the caller's context."""
    lines = [run.task.strip()]
    ctx = {k: v for k, v in (run.context or {}).items() if v not in (None, "", [], {})}
    if ctx:
        lines.append("")
        lines.append(f"[Contesto del task #{run.id} da {run.source}" + (f", riferimento {run.external_run_id}" if run.external_run_id else "") + "]")
        for key, value in ctx.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _answer_row(session: Session, chat: Chat, request_id: str) -> Optional[dict[str, Any]]:
    """The agent's answer row of this request (the last ``ai`` row without
    tool calls whose turn usage carries our request id)."""
    row = session.execute(
        text(
            "SELECT message FROM n8n_chat_histories WHERE session_id = :sid "
            "AND message->>'type' = 'ai' "
            "AND (jsonb_typeof(message->'tool_calls') IS DISTINCT FROM 'array' OR jsonb_array_length(message->'tool_calls') = 0) "
            "AND message->'response_metadata'->'edgewalker'->'turn_usage'->>'correlation_id' = :rid "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"sid": chat.n8n_session_id or str(chat.id), "rid": request_id},
    ).first()
    if row is None:
        return None
    return row[0] if isinstance(row[0], dict) else None


def _latest_ai_row_after(session: Session, chat: Chat, after_row_id: int) -> Optional[dict[str, Any]]:
    row = session.execute(
        text(
            "SELECT message FROM n8n_chat_histories WHERE session_id = :sid AND id > :after "
            "AND message->>'type' = 'ai' "
            "AND (jsonb_typeof(message->'tool_calls') IS DISTINCT FROM 'array' OR jsonb_array_length(message->'tool_calls') = 0) "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"sid": chat.n8n_session_id or str(chat.id), "after": after_row_id},
    ).first()
    return row[0] if row is not None and isinstance(row[0], dict) else None


def _content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def execute_run(session: Session, run_id: int) -> AgentRun:
    """Run the task as a chat turn and record the outcome. Blocking (the
    turn takes seconds to minutes): called from a background task."""
    from app.services.chat_service import send_chat_message

    run = session.get(AgentRun, run_id)
    if run is None or run.status != "queued":
        return run  # type: ignore[return-value]
    agent = session.get(Agent, run.agent_id)
    chat = session.get(Chat, run.chat_id) if run.chat_id else None
    if chat is None:
        run.status = "failed"
        run.error = "chat of the run no longer exists"
        run.finished_at = _utcnow()
        session.add(run)
        session.commit()
        return run
    run.status = "running"
    run.started_at = _utcnow()
    session.add(run)
    session.commit()
    last_row_id = session.execute(
        text("SELECT COALESCE(MAX(id), 0) FROM n8n_chat_histories WHERE session_id = :sid"),
        {"sid": chat.n8n_session_id or str(chat.id)},
    ).scalar() or 0
    label = {"paperclip": "Paperclip", "a2a": "A2A", "mcp": "MCP"}.get(run.source, "API")
    try:
        response = send_chat_message(
            session,
            chat_id=chat.id,
            user_id=run.user_id,
            text=_task_message(run),
            metadata={"run_id": run.id, "run_source": run.source, "external_run_id": run.external_run_id},
            sender_kind="external_agent",
            sender_label=f"{label} · run #{run.id}",
        )
        request_id = response.request_id
    except HTTPException as exc:
        return _finish(session, run, agent, status_="failed", error=str(exc.detail))
    except Exception as exc:  # noqa: BLE001
        logger.exception("run %s: dispatch failed", run.id)
        return _finish(session, run, agent, status_="failed", error=str(exc)[:1000])
    # send_chat_message closes the session before the webhook call: re-fetch.
    run = session.get(AgentRun, run_id)  # type: ignore[assignment]
    chat = session.get(Chat, chat.id)  # type: ignore[assignment]
    run.request_id = request_id
    answer = (_answer_row(session, chat, request_id) if request_id else None) or _latest_ai_row_after(session, chat, int(last_row_id))
    if answer is None:
        return _finish(session, run, agent, status_="failed", error="the agent produced no answer")
    usage = ((answer.get("response_metadata") or {}).get("edgewalker") or {}).get("turn_usage")
    usage = usage if isinstance(usage, dict) else None
    credits = usage.get("credits") if usage else None
    return _finish(
        session, run, agent, status_="succeeded", result=_content_text(answer).strip() or None,
        usage=usage, cost_credits=credits,
    )


def _finish(
    session: Session,
    run: AgentRun,
    agent: Agent | None,
    *,
    status_: str,
    result: Optional[str] = None,
    error: Optional[str] = None,
    usage: Optional[dict[str, Any]] = None,
    cost_credits: Any = None,
) -> AgentRun:
    run.status = status_
    run.result = result
    run.error = error
    run.usage = usage
    try:
        run.cost_credits = Decimal(str(cost_credits)) if cost_credits is not None else None
    except Exception:  # noqa: BLE001
        run.cost_credits = None
    run.finished_at = _utcnow()
    session.add(run)
    emit_event(
        session,
        user_id=run.user_id,
        event_type="agent.run.completed",
        data={
            "run_id": run.id,
            "agent_id": run.agent_id,
            "agent_name": agent.agent_name if agent else None,
            "source": run.source,
            "external_run_id": run.external_run_id,
            "scope": run.scope,
            "strategy_id": run.strategy_id,
            "status": run.status,
            "result": run.result,
            "usage": run.usage,
            "cost_credits": float(run.cost_credits) if run.cost_credits is not None else None,
            "error": run.error,
        },
        dedupe_key=f"run:{run.id}:{run.status}",
    )
    session.commit()
    session.refresh(run)
    if run.callback_url:
        _callback(session, run, agent)
    return run


# ── callbacks ───────────────────────────────────────────────────────────────


def callback_body(run: AgentRun, agent: Agent | None) -> dict[str, Any]:
    """What is POSTed to a generic ``callback_url`` (source api / mcp / a2a)."""
    return {
        "run_id": run.id,
        "external_run_id": run.external_run_id,
        "agent_id": run.agent_id,
        "agent_name": agent.agent_name if agent else None,
        "status": run.status,
        "result": run.result,
        "error": run.error,
        "usage": run.usage,
        "cost_credits": float(run.cost_credits) if run.cost_credits is not None else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _callback(session: Session, run: AgentRun, agent: Agent | None) -> None:
    if run.source == "paperclip":
        _paperclip_report(session, run, agent)
        return
    headers = {"Content-Type": "application/json"}
    auth = _decrypt(run.callback_auth)
    if auth:
        headers["Authorization"] = auth if auth.lower().startswith(("bearer ", "basic ")) else f"Bearer {auth}"
    try:
        with httpx.Client(timeout=CALLBACK_TIMEOUT_S, follow_redirects=False) as client:
            response = client.post(run.callback_url, json=callback_body(run, agent), headers=headers)
        if 200 <= response.status_code < 300:
            run.callback_status = "delivered"
            run.callback_error = None
        else:
            run.callback_status = "failed"
            run.callback_error = f"HTTP {response.status_code}: {response.text[:300]}"
    except httpx.HTTPError as exc:
        run.callback_status = "failed"
        run.callback_error = str(exc)[:500]
    session.add(run)
    session.commit()


# ── Paperclip ───────────────────────────────────────────────────────────────
#
# The Paperclip ``http`` adapter is fire-and-forget: it POSTs one request per
# heartbeat and a 2xx answer closes its run as succeeded, whatever we reply.
# There is no completion callback; an external agent reports through the
# Paperclip API with its own agent key: read the issue it was woken for,
# comment on it and move its status (docs.paperclip.ing/reference/api/issues).

PAPERCLIP_TIMEOUT_S = 8.0
PAPERCLIP_COMMENTS = 6
PAPERCLIP_CHECKOUT_STATUSES = ["todo", "backlog", "blocked", "in_review", "in_progress"]
PAPERCLIP_DEFAULT_DONE_STATUS = "done"


def _paperclip_headers(api_key: Optional[str], run_id: Optional[str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
    if run_id:
        headers["X-Paperclip-Run-Id"] = run_id
    return headers


def fetch_paperclip_issue(api_url: str, api_key: str, *, run_id: Optional[str], issue_id: str) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    """Issue (title, description, status…) and its latest comments, oldest
    first. Failures are logged and leave the run with the bare ids: a
    heartbeat must never fail because Paperclip is slow to answer."""
    base = api_url.rstrip("/")
    headers = _paperclip_headers(api_key, run_id)
    issue: Optional[dict[str, Any]] = None
    comments: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=PAPERCLIP_TIMEOUT_S, follow_redirects=False) as client:
            response = client.get(f"{base}/api/issues/{issue_id}", headers=headers)
            if response.status_code == 200 and isinstance(response.json(), dict):
                issue = response.json()
            else:
                logger.warning("paperclip: issue %s not readable (HTTP %s)", issue_id, response.status_code)
            response = client.get(f"{base}/api/issues/{issue_id}/comments", params={"order": "desc", "limit": PAPERCLIP_COMMENTS}, headers=headers)
            if response.status_code == 200:
                data = response.json()
                rows = data if isinstance(data, list) else data.get("comments") or data.get("items") or []
                comments = [c for c in rows if isinstance(c, dict) and isinstance(c.get("body"), str)]
                comments.reverse()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("paperclip: cannot read issue %s: %s", issue_id, exc)
    return issue, comments


def _issue_label(issue: Optional[dict[str, Any]], issue_id: str) -> str:
    if issue and isinstance(issue.get("identifier"), str) and issue["identifier"]:
        return issue["identifier"]
    return issue_id


def paperclip_task_text(beat: PaperclipHeartbeat, issue: Optional[dict[str, Any]] = None, comments: Optional[list[dict[str, Any]]] = None) -> str:
    """The message the agent receives: who woke it and why, the issue as
    Paperclip shows it, the latest comments, then the explicit instructions
    of the payloadTemplate (if any)."""
    explicit = (beat.task or beat.instructions or "").strip()
    reason = beat.wakeReason or "heartbeat"
    issue_id = beat.issue_ref
    refs = []
    if issue_id:
        refs.append(f"task {_issue_label(issue, issue_id)}")
    others = [i for i in beat.issueIds if i != issue_id]
    if others:
        refs.append("altri task " + ", ".join(others))
    lines = ["Richiesta da Paperclip (" + reason + ((": " + ", ".join(refs)) if refs else "") + ")."]
    if issue:
        title = (issue.get("title") or "").strip()
        description = (issue.get("description") or "").strip()
        meta = [str(issue.get(k)) for k in ("status", "priority") if issue.get(k)]
        lines.append("")
        lines.append(f"Task: {title}" if title else "Task senza titolo")
        if meta:
            lines.append("Stato/priorità: " + " · ".join(meta))
        if description:
            lines.append("Descrizione:")
            lines.append(description[:6000])
    if comments:
        lines.append("")
        lines.append("Ultimi commenti (dal più vecchio):")
        for c in comments[-PAPERCLIP_COMMENTS:]:
            who = "agente" if c.get("authorAgentId") else "umano"
            flag = " ← nuovo" if beat.commentId and c.get("id") == beat.commentId else ""
            lines.append(f"- [{who}]{flag} {str(c.get('body')).strip()[:1500]}")
    lines.append("")
    if explicit:
        lines.append(explicit)
    elif reason == "issue_assigned" or (issue and not comments):
        lines.append("Ti è stato assegnato questo task: fai ciò che chiede con gli strumenti della piattaforma e rispondi con un riepilogo di quanto fatto e dei risultati. La risposta viene pubblicata come commento sul task.")
    elif reason in ("issue_commented", "comment", "mention", "wake_comment") or comments:
        lines.append("C'è un nuovo commento sul task: rispondi nel merito usando i dati della piattaforma. La risposta viene pubblicata come commento sul task.")
    elif issue_id:
        lines.append("Riprendi il task: verifica lo stato e completa quanto manca, poi riepiloga.")
    else:
        lines.append("Fai un briefing breve: stato delle live, risultati recenti, cosa merita attenzione, cosa proponi.")
    return "\n".join(lines).strip()


def run_from_paperclip(session: Session, *, user_id: int, agent_id: int, beat: PaperclipHeartbeat, created_by: Optional[dict[str, Any]] = None) -> AgentRun:
    api_url = (beat.paperclipApiUrl or "").rstrip("/")
    api_key = beat.paperclipApiKey or None
    issue_id = beat.issue_ref
    issue, comments = (None, [])
    if api_url and api_key and issue_id:
        issue, comments = fetch_paperclip_issue(api_url, api_key, run_id=beat.runId, issue_id=issue_id)
    # the outcome goes back as a comment (+ status) on the issue: nothing to
    # report when the heartbeat is not about an issue (timer, manual wake)
    callback_url = f"{api_url}/api/issues/{issue_id}" if api_url and api_key and issue_id else None
    extra = {k: v for k, v in (beat.model_extra or {}).items() if v not in (None, "", [], {})}
    context: dict[str, Any] = {
        "paperclip": {
            "runId": beat.runId, "agentId": beat.agentId, "companyId": beat.companyId, "taskId": beat.taskId,
            "issueId": beat.issueId, "wakeReason": beat.wakeReason, "commentId": beat.commentId, "issueIds": beat.issueIds,
            "doneStatus": beat.paperclipDoneStatus or PAPERCLIP_DEFAULT_DONE_STATUS,
            "issue": {k: issue.get(k) for k in ("id", "identifier", "title", "status", "priority", "projectId") if issue and issue.get(k) is not None} if issue else None,
            **{k: v for k, v in beat.context.items() if k not in ("paperclipWorkspace", "connectionInstructions")},
        },
        **extra,
    }
    payload = RunCreate(
        task=paperclip_task_text(beat, issue, comments),
        scope=beat.scope or ("strategy" if beat.strategy_id else "agent"),
        strategy_id=beat.strategy_id,
        context=context,
        external_run_id=beat.runId,
        callback_url=callback_url,
        callback_auth=api_key,
        source="paperclip",
    )
    return create_run(session, user_id=user_id, agent_id=agent_id, payload=payload, created_by=created_by)


def paperclip_comment_body(run: AgentRun, agent: Agent | None) -> str:
    """Markdown comment posted on the Paperclip issue when the run ends."""
    name = agent.agent_name if agent else "EdgeWalker"
    if run.status == "succeeded":
        text = (run.result or "").strip() or "_(nessuna risposta)_"
    else:
        text = f"**Run EdgeWalker fallito**: {run.error or 'errore sconosciuto'}"
    usage = run.usage or {}
    bits = [f"EdgeWalker · {name} · run #{run.id}"]
    if run.cost_credits is not None:
        bits.append(f"{float(run.cost_credits):g} crediti")
    tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    if tokens_in or tokens_out:
        bits.append(f"token {int(tokens_in or 0)} in / {int(tokens_out or 0)} out")
    return text + "\n\n---\n_" + " · ".join(bits) + "_"


def _paperclip_report(session: Session, run: AgentRun, agent: Agent | None) -> None:
    """Report the outcome on the issue: checkout (idempotent for the agent
    itself), then comment + status. ``callback_url`` is the issue URL."""
    pc = (run.context or {}).get("paperclip") or {}
    headers = _paperclip_headers(_decrypt(run.callback_auth), run.external_run_id)
    done_status = pc.get("doneStatus") or PAPERCLIP_DEFAULT_DONE_STATUS
    body = paperclip_comment_body(run, agent)
    steps: list[str] = []
    try:
        with httpx.Client(timeout=PAPERCLIP_TIMEOUT_S, follow_redirects=False) as client:
            if pc.get("agentId"):
                checkout = client.post(f"{run.callback_url}/checkout", json={"agentId": pc["agentId"], "expectedStatuses": PAPERCLIP_CHECKOUT_STATUSES}, headers=headers)
                steps.append(f"checkout {checkout.status_code}")
            if run.status == "succeeded" and done_status == "comment":
                response = client.post(f"{run.callback_url}/comments", json={"body": body}, headers=headers)
                steps.append(f"comment {response.status_code}")
            else:
                new_status = done_status if run.status == "succeeded" else "blocked"
                response = client.patch(run.callback_url, json={"status": new_status, "comment": body}, headers=headers)
                steps.append(f"{new_status} {response.status_code}")
                if response.status_code >= 400:
                    # status change refused (e.g. another owner): at least leave the comment
                    response = client.post(f"{run.callback_url}/comments", json={"body": body}, headers=headers)
                    steps.append(f"comment {response.status_code}")
        if 200 <= response.status_code < 300:
            run.callback_status = "delivered"
            run.callback_error = None
        else:
            run.callback_status = "failed"
            run.callback_error = f"HTTP {response.status_code}: {response.text[:300]} ({', '.join(steps)})"
    except httpx.HTTPError as exc:
        run.callback_status = "failed"
        run.callback_error = f"{str(exc)[:400]} ({', '.join(steps)})"
    session.add(run)
    session.commit()


# ── A2A (minimal: message/send + tasks/get) ─────────────────────────────────

A2A_STATE = {"queued": "submitted", "running": "working", "succeeded": "completed", "failed": "failed", "cancelled": "canceled"}


def a2a_task(run: AgentRun) -> dict[str, Any]:
    task: dict[str, Any] = {
        "id": str(run.id),
        "contextId": f"chat-{run.chat_id}" if run.chat_id else f"agent-{run.agent_id}",
        "kind": "task",
        "status": {
            "state": A2A_STATE.get(run.status, "unknown"),
            "timestamp": (run.finished_at or run.started_at or run.created_at).isoformat(),
        },
        "metadata": {"source": run.source, "scope": run.scope, "cost_credits": float(run.cost_credits) if run.cost_credits is not None else None},
    }
    if run.status == "succeeded" and run.result:
        task["artifacts"] = [{"artifactId": f"run-{run.id}-answer", "name": "answer", "parts": [{"kind": "text", "text": run.result}]}]
    if run.status == "failed" and run.error:
        task["status"]["message"] = {"kind": "message", "role": "agent", "messageId": f"run-{run.id}-error", "parts": [{"kind": "text", "text": run.error}]}
    return task


def a2a_message_text(message: dict[str, Any]) -> str:
    parts = message.get("parts") if isinstance(message, dict) else None
    texts = []
    for part in parts or []:
        if isinstance(part, dict) and part.get("kind", "text") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return "\n".join(texts).strip()


def agent_card(agent: Agent, *, base_url: str) -> dict[str, Any]:
    """A2A Agent Card (v0.3 shape) of one hosted agent."""
    base = base_url.rstrip("/")
    return {
        "name": agent.agent_name,
        "description": agent.description or "EdgeWalker trading-desk agent: portfolio briefings, strategy design, backtests, playbooks.",
        "url": f"{base}/a2a/agents/{agent.id_agent}",
        "version": "1.0",
        "protocolVersion": "0.3.0",
        "provider": {"organization": "EdgeWalker", "url": "https://edgewalker.tech"},
        "capabilities": {"streaming": False, "pushNotifications": False, "stateTransitionHistory": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer", "description": "EdgeWalker personal access token (scope write)"}},
        "security": [{"bearer": []}],
        "skills": [
            {"id": "briefing", "name": "Portfolio briefing", "description": "Accounts, live sessions, performance, what needs attention.", "tags": ["trading", "briefing"], "examples": ["How are my live sessions doing today?"]},
            {"id": "design", "name": "Strategy design", "description": "Create a strategy from a template or from scratch; with scope=strategy, edit an existing one.", "tags": ["trading", "strategy"], "examples": ["Create a mean-reversion strategy on MNQ 5m on my IBKR demo account."]},
            {"id": "review", "name": "Backtest review", "description": "Compare strategies and backtests, read playbooks and lessons.", "tags": ["backtest", "analysis"], "examples": ["Which of my gold strategies has the smallest drawdown?"]},
        ],
        "metadata": {"edgewalker": {"agent_id": agent.id_agent, "slug": agent.slug, "kind": agent.kind, "risk_profile": agent.risk_profile}},
    }
