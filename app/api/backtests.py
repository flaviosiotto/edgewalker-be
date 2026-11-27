import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.user import User
from app.models.strategy import BacktestStatus
from app.schemas.strategy import (
    AgentEvaluationWrite,
    BacktestCreateRequest,
    BacktestListPage,
    BacktestPlaybackControl,
    BacktestRead,
    BacktestRuntimeOrderRequest,
    BacktestRuntimePositionCloseRequest,
    BacktestSummary,
    BacktestTrialRead,
    LayoutConfigUpdate,
    StrategyRead,
    TradeRead,
    ChartDrawingsRead,
    ChartDrawingsPatch,
    ChartDrawingsUpdate,
)
from app.schemas.chat import ChatRead
from app.services.backtest_analysis import analyze_backtest
from app.services.strategy_service import (
    apply_trial_to_strategy,
    create_backtest,
    delete_backtest,
    get_backtest,
    get_or_create_backtest_chat,
    get_strategy,
    list_all_backtests,
    list_backtest_trials,
    list_trades,
    open_trial_as_backtest,
    retry_backtest,
    run_backtest,
    set_backtest_agent_evaluation,
    update_backtest_chart_drawings,
    update_backtest_layout,
)
from app.utils.auth_utils import (
    AuthPrincipal,
    get_current_active_or_consultative_principal,
    get_current_active_or_consultative_user,
)

logger = logging.getLogger(__name__)

# The single backtest API: every backtest is addressed by its own id, the
# owning strategy is resolved (and ownership-checked) server side. The old
# /strategies/{sid}/backtests/* routes were removed in favour of this router.
# Auth mirrors /accounts: user JWT, PAT, delegated n8n token, or the agent's
# consultative token — the in-backtest manager agent reads and trades through
# the /runtime/* endpoints with its agent_backend_consult token.
router = APIRouter(prefix="/backtests", tags=["Backtests"])

_ACTIVE_STATUSES = {BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value}


def _ledger_from_metrics(backtest, key: str) -> list:
    """Persisted ledger section (orders / fills / equity_snapshots) of a finished run.

    The coordinator keeps a run in memory only while its container lives; the
    same ledger is embedded in ``strategy_backtests.metrics`` at finalize, so a
    completed backtest can still serve its equity curve and order log.
    """
    metrics = backtest.metrics if isinstance(backtest.metrics, dict) else {}
    ledger = metrics.get("ledger") if isinstance(metrics.get("ledger"), dict) else {}
    rows = ledger.get(key) or []
    return list(rows) if isinstance(rows, list) else []


def _ledger_in_files(backtest) -> bool:
    """A finished bulk run keeps its ledger in files next to its dataset
    (coordinator ``ledger_store``): the metrics carry a reference instead."""
    metrics = backtest.metrics if isinstance(backtest.metrics, dict) else {}
    return metrics.get("ledger_store") == "file"


_LEDGER_KEYS = {"orders": "orders", "fills": "fills", "equity": "equity_snapshots", "trades": "trades"}


def _ledger_files(backtest_id: int, trial: int | None, table: str | None) -> dict:
    """Ledger files of a run or of one trial, from the coordinator. Call it
    with the DB session already closed. 404 when the files are not there."""
    from app.services.backtest_runner_service import backtest_runner_service

    try:
        payload = backtest_runner_service.get_backtest_ledger(backtest_id, trial, [table] if table else [])
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Dati della prova {trial} non disponibili: l'esplorazione è precedente al salvataggio delle prove o i file sono stati rimossi"
                if trial else "Dati del backtest non disponibili: i file sono stati rimossi"
            ),
        )
    return payload


def _trial_or_404(session: Session, backtest_id: int, trial: int, user_id: int | None):
    from app.services.strategy_service import list_backtest_trials

    row = next((t for t in list_backtest_trials(session, backtest_id, user_id) if t.trial_index == trial), None)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Trial {trial} not found")
    return row


@router.get("/", response_model=BacktestListPage)
def list_all_backtests_endpoint(
    status_filter: str | None = Query(default=None, alias="status"),
    strategy_id: int | None = Query(default=None),
    symbol: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    with_progress: bool = Query(default=True),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List backtests across all the user's strategies, newest first.

    ``status`` accepts a comma-separated list (e.g. ``running,pending``).
    Rows in an active status carry live ``phase``/``progress`` from the
    coordinator (unless ``with_progress=false``); a row stuck in ``running``
    with no coordinator state and no runner container is flagged ``stale``.
    """
    from app.services.backtest_runner_service import backtest_runner_service

    statuses = None
    if status_filter:
        statuses = [s.strip() for s in status_filter.split(",") if s.strip()]

    rows, total = list_all_backtests(
        session,
        current_user.id,
        statuses=statuses,
        strategy_id=strategy_id,
        symbol=symbol,
        limit=limit,
        offset=offset,
    )
    # Rows are plain mappings of scalar columns (no JSONB blobs, no ORM
    # entities): strategy_name / connection_id / chat_id / agent_score_pct are
    # already projected by the query.
    items = [BacktestSummary.model_validate(row) for row in rows]
    # Release the pooled DB connection before the per-row HTTP probes.
    session.close()

    if with_progress:
        for item in items:
            if item.status not in _ACTIVE_STATUSES:
                continue
            probe = backtest_runner_service.get_backtest_progress(item.id)
            if probe is not None:
                item.phase = probe.get("phase")
                progress = probe.get("progress")
                item.progress = float(progress) if progress is not None else None
            elif item.status == BacktestStatus.RUNNING.value:
                item.stale = not backtest_runner_service.is_backtest_container_running(item.id)

    return BacktestListPage(items=items, total=total, limit=limit, offset=offset)


@router.post("/", response_model=BacktestRead)
def create_backtest_endpoint(
    payload: BacktestCreateRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Create a new backtest with status=pending. Call /run to execute."""
    return create_backtest(session, payload.strategy_id, payload, current_user.id)


@router.get("/{backtest_id}", response_model=BacktestRead)
def get_backtest_endpoint(
    backtest_id: int,
    trial: int | None = Query(default=None, ge=1, description="Describe this trial of the exploration instead of the backtest"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get backtest details including status and results if completed.

    The response does not grow with the length of the run: ``metrics`` holds
    the scalar indicators and ``analysis`` the statistics derived from the
    ledger and the trades.

    ``trial=N`` describes one trial of the backtest's exploration as a
    backtest of its own: same inputs, the trial's KPI, metrics, analysis and
    definition (``config``), no exploration; ``trial`` in the response says
    which one. 404 when the trial's files are not there.
    """
    from app.services.backtest_analysis import scalar_metrics

    backtest = get_backtest(session, backtest_id, current_user.id)
    read = BacktestRead.model_validate(backtest, from_attributes=True)
    strategy = get_strategy(session, backtest.strategy_id, current_user.id)
    read.connection_id = strategy.connection_id
    bt_id, raw_metrics = backtest.id, backtest.metrics
    if trial:
        row = _trial_or_404(session, bt_id, trial, current_user.id)
        trial_info = {"index": row.trial_index, "params": dict(row.params or {}), "is_reference": bool(row.is_reference)}
        session.close()  # release the pooled DB connection before the blocking HTTP call
        files = _ledger_files(bt_id, trial, None)
        payload = _ledger_files(bt_id, trial, "equity")
        trades = _ledger_files(bt_id, trial, "trades").get("trades") or []
        summary = files.get("summary") or {}
        metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
        for name in ("return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "profit_factor", "total_trades", "equity_final", "equity_peak"):
            setattr(read, name, metrics.get(name))
        read.metrics = scalar_metrics(metrics)
        if isinstance(summary.get("definition"), dict):
            read.config = summary["definition"]
        read.analysis = analyze_backtest({"ledger": {"equity_snapshots": payload.get("equity_snapshots") or []}}, trades)
        read.exploration = None
        read.agent_evaluation = None
        read.trial = trial_info
        return read
    if backtest.status not in _ACTIVE_STATUSES:
        trades = list_trades(session, backtest_id, current_user.id)
        if _ledger_in_files(backtest):
            for row in trades:
                session.expunge(row)  # read after the session is closed
            session.close()  # release the pooled DB connection before the blocking HTTP call
            try:
                equity = _ledger_files(bt_id, None, "equity").get("equity_snapshots") or []
            except HTTPException:
                equity = []  # files removed: the page still opens, without the equity statistics
            raw_metrics = {"ledger": {"equity_snapshots": equity}}
        read.analysis = analyze_backtest(raw_metrics, trades)
    return read


@router.delete("/{backtest_id}")
def delete_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Delete a backtest, its trades and its persisted chart dataset."""
    from app.services.backtest_runner_service import backtest_runner_service

    delete_backtest(session, backtest_id, current_user.id)
    session.close()  # release the pooled DB connection before the blocking HTTP call
    backtest_runner_service.delete_backtest_dataset(backtest_id)
    return {"status": "ok"}


@router.get("/{backtest_id}/chat", response_model=ChatRead)
def get_backtest_chat_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get or create the dedicated chat for a backtest instance."""
    _ = get_backtest(session, backtest_id, current_user.id)
    return get_or_create_backtest_chat(session, backtest_id, current_user.id)


@router.post("/{backtest_id}/run", response_model=BacktestRead)
def run_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Start backtest execution.

    Spawns strategy-runner in backtest mode. The always-on strategy-backtest
    service prepares/replays data, records simulated orders, and writes results.
    """
    return run_backtest(session, backtest_id, current_user.id)


@router.post("/{backtest_id}/retry", response_model=BacktestRead)
def retry_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Repeat a finished backtest ("Riprova"): a new backtest with the same
    inputs and config snapshot, its own chat and playbook copy, started at
    once. The source row is left untouched; the new one points back to it
    (``retried_from_backtest_id``). 409 unless the source is completed,
    failed or error.
    """
    return retry_backtest(session, backtest_id, current_user.id)


# ── Parameter exploration (docs/valutazione-esplorazione-parametri.md) ──
# Trials are listed in the order of the space: no endpoint ranks them or
# names a "best" one (the platform is a tool, not a verdict).


@router.get("/{backtest_id}/trials", response_model=list[BacktestTrialRead])
def list_backtest_trials_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Trials of the backtest's parameter exploration, by ``trial_index``
    (0 = the backtest itself, on the strategy's values). Empty when the run
    has no exploration or its trials have not started yet; the progress is
    in ``exploration.state`` of ``GET /backtests/{id}``."""
    return list_backtest_trials(session, backtest_id, current_user.id)


@router.post("/{backtest_id}/exploration/stop")
def stop_backtest_exploration_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Stop the trials of the backtest's exploration; the backtest and the
    trials already done stay. 409 when no exploration of it is running."""
    from app.services.backtest_runner_service import CoordinatorConflict, backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.stop_exploration(bt_id)
    except CoordinatorConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/trials/{trial_index}/backtest", response_model=BacktestRead)
def open_trial_as_backtest_endpoint(
    backtest_id: int,
    trial_index: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """"Apri come backtest": create and start a new bulk backtest with the
    inputs of this one and the trial's values written into the strategy's
    definition (``parameters.definition_values``, ``parameters.from_trial``),
    no exploration. 404 unknown trial; 409 when the strategy no longer has
    one of the trial's parameters."""
    return open_trial_as_backtest(session, backtest_id, trial_index, current_user.id)


@router.post("/{backtest_id}/trials/{trial_index}/apply", response_model=StrategyRead)
def apply_trial_to_strategy_endpoint(
    backtest_id: int,
    trial_index: int,
    session: Session = Depends(get_session),
    principal: AuthPrincipal = Depends(get_current_active_or_consultative_principal),
    x_client_request_id: str | None = Header(default=None, alias="X-Client-Request-Id"),
):
    """"Applica alla strategia": write the trial's values into the strategy
    definition through the normal update path; returns the strategy as
    ``PATCH /strategies/{id}`` does. 409 when the strategy no longer has one
    of the trial's parameters."""
    from app.api.strategies import _publish_strategy_updated, _serialize_strategy_with_live

    strategy = apply_trial_to_strategy(session, backtest_id, trial_index, principal.user.id)
    origin = "user" if principal.claims.get("purpose") == "ui_auth" else "agent"
    _publish_strategy_updated(strategy, origin=origin, request_id=x_client_request_id)
    return _serialize_strategy_with_live(session, strategy)


@router.post("/{backtest_id}/control")
def control_backtest_playback_endpoint(
    backtest_id: int,
    payload: BacktestPlaybackControl,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Control runtime backtest playback: pause, resume, speed, or step."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    try:
        return backtest_runner_service.control_backtest_playback(
            backtest.id,
            payload.model_dump(exclude_none=True),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/stop")
def stop_backtest_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Stop a running backtest runner and request replay cancellation."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    was_active = backtest.status in (BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value)
    if was_active:
        # Classify BEFORE the cancel: the coordinator finalizes the row on
        # its CancelledError and keeps a failure_kind already set, so the
        # stop is never recorded as a service shutdown (migr. 073).
        backtest.failure_kind = "user_stop"
        backtest.error_message = "Stopped by user"
        session.add(backtest)
        session.commit()
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP/docker calls
    result = backtest_runner_service.stop_backtest(bt_id)

    # Terminal status, unless the coordinator already wrote it.
    if was_active:
        backtest = get_backtest(session, backtest_id, current_user.id)
        if backtest.status in (BacktestStatus.PENDING.value, BacktestStatus.RUNNING.value):
            backtest.status = BacktestStatus.FAILED.value
            backtest.completed_at = datetime.now(timezone.utc)
            backtest.error_message = "Stopped by user"
            backtest.failure_kind = "user_stop"
            session.add(backtest)
            session.commit()

    return result


@router.get("/{backtest_id}/status")
def get_backtest_runtime_status_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get runner container and replay-service progress for a backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    strategy = get_strategy(session, backtest.strategy_id, current_user.id)
    connection_id = strategy.connection_id
    # Release the pooled DB connection before the blocking HTTP call to the
    # backtest container: this endpoint is polled once per second per client
    # during playback, and holding the connection across a slow (up to 5s)
    # httpx call exhausts the pool.
    session.close()

    status_payload = backtest_runner_service.get_backtest_status(bt_id)
    service_status = status_payload.setdefault("service", {})
    if isinstance(service_status, dict):
        service_status.setdefault("backtest_id", bt_id)
        service_status.setdefault("stream_id", f"backtest-{bt_id}")
        service_status.setdefault("bars_stream", f"bars:backtest-{bt_id}")
        if connection_id is not None:
            service_status.setdefault("connection_id", str(connection_id))
    return status_payload


@router.get("/{backtest_id}/runtime/orders")
def get_backtest_runtime_orders_endpoint(
    backtest_id: int,
    active_only: bool = Query(default=False),
    trial: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Simulated orders: live from the coordinator, persisted ledger once
    finished (``trial``: the orders of that trial of the exploration)."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    active = backtest.status in _ACTIVE_STATUSES
    in_files = bool(trial) or (not active and _ledger_in_files(backtest))
    ledger_orders = [] if active or in_files else _ledger_from_metrics(backtest, "orders")
    session.close()  # release the pooled DB connection before the blocking HTTP call
    if in_files:
        ledger_orders = _ledger_files(bt_id, trial, "orders").get("orders") or []
        if active_only:
            ledger_orders = [
                o for o in ledger_orders
                if str(o.get("status") or "") in {"pending", "submitted", "partially_filled"}
            ]
        return {"orders": ledger_orders, "source": "ledger"}
    if not active and ledger_orders:
        # Finished run: the coordinator has forgotten it, serve the persisted ledger.
        if active_only:
            ledger_orders = [
                o for o in ledger_orders
                if str(o.get("status") or "") in {"pending", "submitted", "partially_filled"}
            ]
        return {"orders": ledger_orders, "source": "ledger"}
    try:
        payload = backtest_runner_service.list_backtest_orders(bt_id, active_only=active_only)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    payload.setdefault("source", "coordinator")
    return payload


@router.post("/{backtest_id}/runtime/orders")
def submit_backtest_runtime_order_endpoint(
    backtest_id: int,
    payload: BacktestRuntimeOrderRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Submit a manual order to the source-of-truth strategy-backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    strategy_id = backtest.strategy_id
    order_payload = payload.model_dump(exclude_none=True)
    extra = dict(order_payload.get("extra") or {})
    extra.setdefault("reason", "manual_backtest_order")
    extra.setdefault("source", "backtest_detail_ui")
    extra.setdefault("strategy_id", strategy_id)
    extra.setdefault("backtest_id", backtest.id)
    order_payload["extra"] = extra
    order_payload.setdefault(
        "order_ref",
        f"strategy-{strategy_id}:backtest-{backtest.id}:manual:{int(datetime.now(timezone.utc).timestamp() * 1000)}",
    )
    try:
        return backtest_runner_service.place_backtest_order(backtest.id, order_payload)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.delete("/{backtest_id}/runtime/orders/{order_id}")
def cancel_backtest_runtime_order_endpoint(
    backtest_id: int,
    order_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Cancel an order recorded by strategy-backtest when it is still cancellable."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    try:
        return backtest_runner_service.cancel_backtest_order(
            backtest.id,
            order_id,
            status_message="Cancelled from backtest detail UI",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/positions")
def get_backtest_runtime_positions_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Get source-of-truth simulated position recorded by strategy-backtest."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.get_backtest_position(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.post("/{backtest_id}/runtime/positions/{position_id}/close")
def close_backtest_runtime_position_endpoint(
    backtest_id: int,
    position_id: str,
    payload: BacktestRuntimePositionCloseRequest,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Close one simulated position through the strategy-backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    close_payload: dict = {
        "quantity": payload.quantity,
        "extra": {
            "reason": payload.reason or "backtest_position_close",
            "source": "backend_backtest_api",
            **(payload.extra or {}),
        },
    }
    if payload.symbol:
        close_payload["symbol"] = payload.symbol
    try:
        result = backtest_runner_service.close_backtest_position(
            backtest.id, position_id, close_payload
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    return {"venue": "backtest", "backtest_id": backtest.id, "result": result}


@router.get("/{backtest_id}/runtime/trades")
def get_backtest_runtime_trades_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List source-of-truth closed trades computed from the runtime backtest ledger."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_trades(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/equity")
def get_backtest_runtime_equity_endpoint(
    backtest_id: int,
    trial: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Equity curve: live from the coordinator, persisted ledger once
    finished (``trial``: the curve of that trial of the exploration)."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    active = backtest.status in _ACTIVE_STATUSES
    in_files = bool(trial) or (not active and _ledger_in_files(backtest))
    ledger_equity = [] if active or in_files else _ledger_from_metrics(backtest, "equity_snapshots")
    session.close()  # release the pooled DB connection before the blocking HTTP call
    if in_files:
        return {"equity": _ledger_files(bt_id, trial, "equity").get("equity_snapshots") or [], "source": "ledger"}
    if not active and ledger_equity:
        # Finished run: the coordinator has forgotten it, serve the persisted curve.
        return {"equity": ledger_equity, "source": "ledger"}
    try:
        payload = backtest_runner_service.list_backtest_equity(bt_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    payload.setdefault("source", "coordinator")
    return payload


@router.get("/{backtest_id}/ohlc-history")
def get_backtest_ohlc_history_endpoint(
    backtest_id: int,
    chart_id: str | None = Query(default=None),
    trial: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Chart dataset persisted by the coordinator at materialisation (parquet).

    Same JSON shape as ``/marketdata/ohlc-history`` (candles, indicators,
    indicator_meta, frames); ``chart_id`` selects a secondary chart;
    ``trial`` shows the chart with the indicators of that trial of the
    exploration. 404 when the run predates dataset persistence.
    """
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        payload = backtest_runner_service.get_backtest_ohlc_history(bt_id, chart_id, trial)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dati del backtest non disponibili: run precedente alla persistenza dei dataset",
        )
    return payload


@router.get("/{backtest_id}/runtime/alerts")
def get_backtest_runtime_alerts_endpoint(
    backtest_id: int,
    active_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List structured alerts tracked by the active backtest runner."""
    from app.services.backtest_runner_service import backtest_runner_service

    backtest = get_backtest(session, backtest_id, current_user.id)
    bt_id = backtest.id
    session.close()  # release the pooled DB connection before the blocking HTTP call
    try:
        return backtest_runner_service.list_backtest_alerts(bt_id, active_only=active_only)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


@router.get("/{backtest_id}/runtime/agent-calls")
def get_backtest_agent_calls_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List manager-agent invocations recorded by the runner for this backtest.

    Read from the ``agent_call`` table (written by the runner at dispatch
    time), so the history survives the end of the run. ``bar_ts`` is the
    simulated replay timestamp used to place each call on the chart.
    """
    from sqlmodel import select

    from app.models.agent_call import AgentCall

    backtest = get_backtest(session, backtest_id, current_user.id)
    calls = session.exec(
        select(AgentCall)
        .where(AgentCall.backtest_id == backtest.id)
        .order_by(AgentCall.id)
    ).all()
    return {"agent_calls": [call.model_dump(mode="json") for call in calls]}


@router.post("/{backtest_id}/agent-evaluation/request", status_code=202)
def request_backtest_agent_evaluation_endpoint(
    backtest_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Ask the run's agent to evaluate a completed backtest (Agent Score +
    Agent Hint). Returns at once: the evaluation lands on the backtest when
    the agent closes its turn. The way to get one for a bulk run, which never
    calls the agent, and for a replay run that ended without it."""
    from app.services.backtest_evaluation_service import request_backtest_agent_evaluation

    return request_backtest_agent_evaluation(session, backtest_id, current_user.id)


@router.put("/{backtest_id}/agent-evaluation")
def set_backtest_agent_evaluation_endpoint(
    backtest_id: int,
    payload: AgentEvaluationWrite,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Final evaluation of the run by its agent: six axis scores (radar), a
    diagnosis and the hints. Returns the stored evaluation with the derived
    overall ``score_pct``."""
    return set_backtest_agent_evaluation(session, backtest_id, payload, current_user.id)


@router.patch("/{backtest_id}/layout", response_model=BacktestRead)
def update_backtest_layout_endpoint(
    backtest_id: int,
    payload: LayoutConfigUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Update only the UI layout configuration for a backtest."""
    return update_backtest_layout(session, backtest_id, payload, current_user.id)


@router.get("/{backtest_id}/charts/{chart_id}/drawings", response_model=ChartDrawingsRead)
def read_backtest_chart_drawings_endpoint(
    backtest_id: int,
    chart_id: str,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Current drawings of one chart (lettura leggera per il push ``drawings_changed``)."""
    from app.services.strategy_service import read_backtest_chart_drawings
    drawings = read_backtest_chart_drawings(session, backtest_id, chart_id, current_user.id)
    return ChartDrawingsRead(chart_id=chart_id, drawings=drawings)


@router.patch("/{backtest_id}/charts/{chart_id}/drawings", response_model=ChartDrawingsRead)
def patch_backtest_chart_drawings_endpoint(
    backtest_id: int,
    chart_id: str,
    payload: ChartDrawingsPatch,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Incremental edit (deletions by id/group/author, then upsert by id): the
    agent's ``draw_on_chart`` and the FE deletion of agent drawings."""
    from app.services.strategy_service import patch_backtest_chart_drawings
    drawings = patch_backtest_chart_drawings(session, backtest_id, chart_id, payload, current_user.id)
    return ChartDrawingsRead(chart_id=chart_id, drawings=drawings)


@router.put("/{backtest_id}/charts/{chart_id}/drawings", response_model=ChartDrawingsRead)
def update_backtest_chart_drawings_endpoint(
    backtest_id: int,
    chart_id: str,
    payload: ChartDrawingsUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """Replace the user drawings of one chart on the backtest's config snapshot."""
    drawings = update_backtest_chart_drawings(session, backtest_id, chart_id, payload, current_user.id)
    return ChartDrawingsRead(chart_id=chart_id, drawings=drawings)


@router.get("/{backtest_id}/trades", response_model=list[TradeRead])
def list_trades_endpoint(
    backtest_id: int,
    trial: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_or_consultative_user),
):
    """List all trades for a backtest (``trial``: the trades of that trial of
    its exploration, read from the trial's files)."""
    if trial:
        backtest = get_backtest(session, backtest_id, current_user.id)
        bt_id = backtest.id
        session.close()  # release the pooled DB connection before the blocking HTTP call
        return _ledger_files(bt_id, trial, "trades").get("trades") or []
    return list_trades(session, backtest_id, current_user.id)
