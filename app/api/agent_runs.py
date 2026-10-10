"""Run endpoint (bridge F4): EdgeWalker as an agent.

* ``POST /agents/{id}/runs`` — hand a task to a hosted agent (202, the run
  executes in the background; poll ``GET /runs/{id}`` or give a callback).
* ``POST /agents/{id}/paperclip`` — the Paperclip ``http`` adapter target:
  202 ``{status: accepted, executionId}``; the outcome is reported on the
  Paperclip issue (comment + status) through the Paperclip API.
* ``/a2a/agents/{id}`` — minimal A2A: Agent Card + JSON-RPC ``message/send``
  and ``tasks/get``.

Auth: UI session or a personal access token (``write`` scope for POSTs).
A token bound to an external agent may hand tasks to a hosted agent: that is
the bridge (the external collaborator asks the hosted trader), never the
other way round.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlmodel import Session

from app.core.actor import current_actor_dict
from app.db.database import get_session, get_session_context
from app.models.agent import Agent
from app.models.user import User
from app.schemas.agent_run import PaperclipHeartbeat, RunCreate, RunRead
from app.services import agent_run_service as runs
from app.services.agent_service import get_agent, require_hosted_agent
from app.utils.auth_utils import get_current_active_user

runs_router = APIRouter(tags=["Agent runs"])
a2a_router = APIRouter(prefix="/a2a", tags=["A2A"])


def _execute_in_background(run_id: int) -> None:
    with get_session_context() as session:
        try:
            runs.execute_run(session, run_id)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger(__name__).exception("run %s crashed", run_id)


def _read(session: Session, run) -> RunRead:
    agent = session.get(Agent, run.agent_id)
    return runs.to_read(run, agent_name=agent.agent_name if agent else None)


@runs_router.post("/agents/{agent_id}/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
def create_run_endpoint(
    agent_id: int,
    payload: RunCreate,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    if payload.source in ("paperclip", "a2a"):
        payload.source = "api"
    run = runs.create_run(session, user_id=current_user.id, agent_id=agent_id, payload=payload, created_by=current_actor_dict())
    background.add_task(_execute_in_background, run.id)
    return _read(session, run)


@runs_router.get("/agents/{agent_id}/runs", response_model=list[RunRead])
def list_agent_runs_endpoint(
    agent_id: int,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    get_agent(session, agent_id, current_user.id)
    return [_read(session, r) for r in runs.list_runs(session, user_id=current_user.id, agent_id=agent_id, status_filter=status_filter, limit=limit)]


@runs_router.get("/runs", response_model=list[RunRead])
def list_runs_endpoint(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return [_read(session, r) for r in runs.list_runs(session, user_id=current_user.id, status_filter=status_filter, limit=limit)]


@runs_router.get("/runs/{run_id}", response_model=RunRead)
def get_run_endpoint(
    run_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return _read(session, runs.get_run(session, user_id=current_user.id, run_id=run_id))


@runs_router.post("/agents/{agent_id}/paperclip", status_code=status.HTTP_202_ACCEPTED)
def paperclip_heartbeat_endpoint(
    agent_id: int,
    beat: PaperclipHeartbeat,
    background: BackgroundTasks,
    request: Request,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Target URL of a Paperclip ``http`` adapter (fire-and-forget: our 2xx
    closes its heartbeat run). Configure in Paperclip: ``url`` = this
    endpoint, ``headers.Authorization`` = ``Bearer <PAT with write>``,
    ``payloadTemplate`` with ``paperclipApiUrl`` and ``paperclipApiKey`` (the
    Paperclip agent's own API key: used to read the issue and to post the
    answer as a comment + status) and optionally ``paperclipDoneStatus``
    (``done`` | ``in_review`` | ``comment``), ``task``/``instructions``,
    ``scope``, ``strategy_id``. The headers ``X-Paperclip-Api-Url`` /
    ``X-Paperclip-Api-Key`` are accepted too."""
    if not beat.paperclipApiUrl:
        beat.paperclipApiUrl = request.headers.get("x-paperclip-api-url")
    if not beat.paperclipApiKey:
        beat.paperclipApiKey = request.headers.get("x-paperclip-api-key")
    run = runs.run_from_paperclip(session, user_id=current_user.id, agent_id=agent_id, beat=beat, created_by=current_actor_dict())
    background.add_task(_execute_in_background, run.id)
    return {"status": "accepted", "executionId": str(run.id), "runId": beat.runId, "reportsTo": run.callback_url}


# ── A2A ─────────────────────────────────────────────────────────────────────


def _base_url(request: Request) -> str:
    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_host:
        return f"{forwarded_proto or request.url.scheme}://{forwarded_host}"
    return str(request.base_url).rstrip("/")


@a2a_router.get("/agents/{agent_id}/agent-card.json")
@a2a_router.get("/agents/{agent_id}/.well-known/agent-card.json")
def agent_card_endpoint(
    agent_id: int,
    request: Request,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    agent = require_hosted_agent(get_agent(session, agent_id, current_user.id), role="A2A server")
    return runs.agent_card(agent, base_url=_base_url(request))


def _rpc_error(req_id: Any, code: int, message: str, http_status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}, status_code=http_status)


@a2a_router.post("/agents/{agent_id}")
async def a2a_rpc_endpoint(
    agent_id: int,
    request: Request,
    background: BackgroundTasks,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """JSON-RPC 2.0: ``message/send`` (→ a task in state submitted; the run
    executes in the background) and ``tasks/get`` (state + answer artifact)."""
    try:
        body = await request.json()
    except ValueError:
        return _rpc_error(None, -32700, "Parse error", 400)
    if not isinstance(body, dict):
        return _rpc_error(None, -32600, "Invalid Request", 400)
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params") if isinstance(body.get("params"), dict) else {}
    try:
        if method == "message/send":
            message = params.get("message") if isinstance(params.get("message"), dict) else {}
            text = runs.a2a_message_text(message)
            if not text:
                return _rpc_error(req_id, -32602, "message.parts must contain text")
            meta = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            scope = "strategy" if meta.get("strategy_id") else "agent"
            payload = RunCreate(
                task=text,
                scope=scope,  # type: ignore[arg-type]
                strategy_id=int(meta["strategy_id"]) if meta.get("strategy_id") else None,
                context={k: v for k, v in meta.items() if k != "strategy_id"},
                external_run_id=str(message.get("messageId") or "")[:128] or None,
                source="a2a",
            )
            run = runs.create_run(session, user_id=current_user.id, agent_id=agent_id, payload=payload, created_by=current_actor_dict())
            background.add_task(_execute_in_background, run.id)
            return {"jsonrpc": "2.0", "id": req_id, "result": runs.a2a_task(run)}
        if method == "tasks/get":
            try:
                run_id = int(str(params.get("id")))
            except (TypeError, ValueError):
                return _rpc_error(req_id, -32602, "params.id must be the task id")
            run = runs.get_run(session, user_id=current_user.id, run_id=run_id)
            return {"jsonrpc": "2.0", "id": req_id, "result": runs.a2a_task(run)}
        return _rpc_error(req_id, -32601, f"Method not found: {method}")
    except HTTPException as exc:
        return _rpc_error(req_id, -32000, str(exc.detail), exc.status_code if exc.status_code >= 500 else 200)
