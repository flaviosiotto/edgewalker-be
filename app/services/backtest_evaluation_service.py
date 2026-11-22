"""On-demand agent evaluation of a finished backtest (Agent Score + Agent Hint).

A replay run gets its evaluation from the runner, which sends the agent an
end-of-run analysis turn. A bulk run has no runner and never calls the agent,
and a replay run can end without one (agent turn failed, no agent at launch).
Here the trader asks for it afterwards: the backend sends the same kind of turn
(``message_type = backtest_synthesis``) to the run's chat, and the agent answers
through ``submit_backtest_evaluation`` exactly as it does at the end of a replay.

The turn is read-only on the playbook: an evaluation asked after the fact never
records or edits lessons.

The request is tracked in ``strategy_backtests.parameters.agent_evaluation_request``
(``requested_at``, ``agent_id``, ``error``) so the UI knows it is waiting and
why a request ended without an evaluation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import HTTPException, status
from sqlmodel import Session

from app.core.config import settings
from app.models.strategy import BacktestResult, BacktestStatus

logger = logging.getLogger(__name__)

REQUEST_KEY = "agent_evaluation_request"
# A request still without an answer blocks a new one for this long.
PENDING_WINDOW_SECONDS = 15 * 60


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def evaluation_request_pending(backtest: BacktestResult, *, now: datetime | None = None) -> bool:
    """A request was sent, has not failed, and no evaluation came after it."""
    request = (backtest.parameters or {}).get(REQUEST_KEY)
    if not isinstance(request, dict) or request.get("error"):
        return False
    requested_at = _parse_ts(request.get("requested_at"))
    if requested_at is None:
        return False
    evaluated_at = _parse_ts((backtest.agent_evaluation or {}).get("created_at"))
    if evaluated_at is not None and evaluated_at >= requested_at:
        return False
    return ((now or _utcnow()) - requested_at).total_seconds() < PENDING_WINDOW_SECONDS


def _fmt(value: Any, digits: int = 2, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/d"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def build_evaluation_prompt(backtest: BacktestResult) -> str:
    """The analysis turn, headline only: the numbers live in ``get_backtest``."""
    metrics = backtest.metrics if isinstance(backtest.metrics, dict) else {}
    bulk = backtest.mode == "bulk"
    trades = int(backtest.total_trades or 0)
    pnl = None
    if backtest.equity_final is not None and backtest.initial_capital is not None:
        pnl = float(backtest.equity_final) - float(backtest.initial_capital)
    headline = (
        f"Trade chiusi: {trades} (win rate {_fmt(backtest.win_rate_pct, 0)}%), "
        f"P&L netto {_fmt(pnl, 2, signed=True)}, rendimento {_fmt(backtest.return_pct, 2, signed=True)}%, "
        f"max drawdown {_fmt(backtest.max_drawdown_pct, 2)}%, profit factor {_fmt(backtest.profit_factor, 2)}"
    )
    lines = [
        f"🎓 ANALISI FINALE BACKTEST {backtest.id} — run completato su {backtest.symbol} "
        f"{backtest.timeframe}, periodo {backtest.start_date} → {backtest.end_date}. "
        "Valutazione richiesta dal trader a run concluso.",
        headline,
    ]
    if bulk:
        info = metrics.get("bulk") if isinstance(metrics.get("bulk"), dict) else {}
        holds = int(info.get("agent_hold_count") or 0)
        lines.append(
            "MODALITA MASSIVA: i trade sono stati eseguiti dalle sole rule, su barre chiuse, "
            "senza alcun tuo intervento durante il run. Stai giudicando la strategia come "
            "scritta, non tue decisioni."
            + (
                f" Le rule ask_agent sono scattate {holds} volte senza chiamarti (nessuna "
                "operazione in quei punti): tienine conto nel giudicare cosa il run NON ha provato."
                if holds
                else ""
            )
        )
    lines.append(
        f"Prima di scrivere chiama get_backtest con backtest_id={backtest.id}: metrics ha gli "
        "indicatori (rendimento, drawdown, sharpe, sortino, profit factor) e analysis le "
        "statistiche gia calcolate (distribuzione dei trade, per direzione e motivo di uscita, "
        "episodi di drawdown, esposizione, segmenti temporali, concentrazione). "
        "get_backtest_trades da i singoli trade se ti serve un dettaglio. Usa quei numeri, non "
        "ricalcolarli."
    )
    if trades < 30:
        lines.append(
            f"CAMPIONE: {trades} trade sono un campione piccolo: ogni conclusione e un IPOTESI "
            "da verificare su altri run."
        )
    lines.append(
        "Analizza il run come un trader esperto che rivede il proprio diario: ragiona sulla "
        "DISTRIBUZIONE dei risultati, non sui singoli trade.\n"
        "1) DIAGNOSI: dove nasce e dove si perde l'edge (expectancy, payoff contro win rate, "
        "concentrazione del profitto, drawdown e serie negative, stabilita' tra i segmenti, "
        "differenze per direzione, motivo di uscita, durata).\n"
        "2) ADERENZA AL CONTESTO: i trade hanno rispettato DESCRIZIONE e STUDI della strategia?"
        + (" Le rule fanno cio' che la descrizione promette?\n" if bulk else " Le deviazioni dal piano hanno aiutato o danneggiato il risultato?\n")
        + "3) LEZIONI: questa valutazione e' in sola lettura sul playbook, NON registrare ne' "
        "modificare lezioni.\n"
        "4) PROPOSTE: non modificare la strategia in questo turno; individua al massimo 5 "
        "interventi azionabili (regole, filtri, size, stop, orari), ciascuno con il numero che "
        "lo giustifica.\n"
        "5) VALUTAZIONE: come ultimo passo chiama UNA volta submit_backtest_evaluation con i sei "
        "punteggi 0-100 (edge, risk, consistency, discipline, execution, robustness), la "
        "motivazione di ogni asse, la diagnosi in summary e le proposte del punto 4 in hints: "
        "e' cio' che il trader vede come Agent Score e Agent Hint.\n"
        "Poi chiudi con un report breve in chat: Diagnosi, Proposte."
    )
    return "\n".join(lines)


def _store_request(session: Session, backtest: BacktestResult, request: dict[str, Any]) -> None:
    parameters = dict(backtest.parameters or {})
    parameters[REQUEST_KEY] = request
    backtest.parameters = parameters
    session.add(backtest)
    session.commit()


def request_backtest_agent_evaluation(
    session: Session,
    backtest_id: int,
    user_id: int | None = None,
) -> dict[str, Any]:
    """Ask the run's agent for its evaluation. Returns at once; the turn runs
    on the agent-run thread pool and the evaluation lands on the backtest."""
    from app.services.agent_run_service import _run_executor
    from app.services.entitlement_service import check_ai_budget
    from app.services.strategy_service import get_backtest, get_or_create_backtest_chat

    backtest = get_backtest(session, backtest_id, user_id)
    if backtest.status != BacktestStatus.COMPLETED.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backtest {backtest_id} is {backtest.status}: the agent evaluates a completed run",
        )
    if not backtest.total_trades:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Backtest {backtest_id} closed no trades: there is nothing to evaluate",
        )
    if evaluation_request_pending(backtest):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An agent evaluation of backtest {backtest_id} is already in progress",
        )

    # Resolves the agent (the run's own, else the strategy's manager agent)
    # and stamps it on the backtest and its chat.
    chat = get_or_create_backtest_chat(session, backtest_id, user_id)
    if chat.id_agent is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No agent configured for this strategy: set its manager agent, then ask again",
        )
    check_ai_budget(session, chat.user_id)

    backtest = get_backtest(session, backtest_id, user_id)
    request = {
        "requested_at": _utcnow().isoformat(),
        "agent_id": chat.id_agent,
        "error": None,
    }
    prompt = build_evaluation_prompt(backtest)
    _store_request(session, backtest, request)

    _run_executor().submit(
        _run_evaluation_turn,
        backtest_id=backtest_id,
        chat_id=chat.id,
        user_id=chat.user_id,
        strategy_id=backtest.strategy_id,
        prompt=prompt,
    )
    logger.info("Backtest %s: agent evaluation requested (agent=%s chat=%s)", backtest_id, chat.id_agent, chat.id)
    return {"status": "requested", "backtest_id": backtest_id, "chat_id": chat.id, **request}


def _run_evaluation_turn(*, backtest_id: int, chat_id: int, user_id: int, strategy_id: int, prompt: str) -> None:
    """The agent turn, detached from the request, in its own DB session."""
    from app.db.database import get_session_context
    from app.services.chat_service import send_chat_message

    error: str | None = None
    try:
        with get_session_context() as session:
            send_chat_message(
                session,
                chat_id=chat_id,
                user_id=user_id,
                text=prompt,
                metadata={
                    "message_type": "backtest_synthesis",
                    "event_type": "backtest_synthesis",
                    "backtest_id": backtest_id,
                    "strategy_id": strategy_id,
                    # Asked after the fact: the playbook stays as it is.
                    "lessons_enabled": False,
                    "learning_mode": "off",
                },
                sender_kind="system",
                sender_label="Valutazione richiesta",
                turn_timeout_seconds=float(settings.AGENT_RUN_TURN_TIMEOUT_SECONDS),
            )
    except HTTPException as exc:
        # 504: the turn is still running and its evaluation can still land.
        if exc.status_code != status.HTTP_504_GATEWAY_TIMEOUT:
            error = str(exc.detail)[:500]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Backtest %s: agent evaluation turn crashed", backtest_id)
        error = str(exc)[:500]

    try:
        with get_session_context() as session:
            backtest = session.get(BacktestResult, backtest_id)
            if backtest is None:
                return
            request = dict((backtest.parameters or {}).get(REQUEST_KEY) or {})
            evaluated_at = _parse_ts((backtest.agent_evaluation or {}).get("created_at"))
            requested_at = _parse_ts(request.get("requested_at"))
            answered = evaluated_at is not None and requested_at is not None and evaluated_at >= requested_at
            if error is None and not answered:
                error = "the agent closed its turn without submitting an evaluation"
            if answered:
                error = None
            request["error"] = error
            request["finished_at"] = _utcnow().isoformat()
            _store_request(session, backtest, request)
    except Exception:  # noqa: BLE001
        logger.exception("Backtest %s: could not record the evaluation request outcome", backtest_id)
    if error:
        logger.warning("Backtest %s: agent evaluation request failed: %s", backtest_id, error)
