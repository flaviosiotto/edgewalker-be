"""Provider token-cost report (docs/credito-piattaforma-studio.md §5b).

Aggregates ``ai_credit_ledger`` (one row per agent turn, with the real cost
the LLM provider reported) along one analysis axis. Not a margin: the
platform's other costs are out of scope by decision; the report compares the
provider's meter with the credits charged (credits per call, cost per call).

``provider`` is data on every row: nothing here assumes OpenRouter.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import text
from sqlmodel import Session

from app.schemas.billing import AiCostReport, AiCostRow

#: group key expressions; ``t`` = agent_turn (LEFT JOIN on correlation_id),
#: ``l`` = ai_credit_ledger, ``u`` = user, ``a`` = agent
_CONTEXT_EXPR = (
    "CASE WHEN t.strategy_live_id IS NOT NULL THEN 'live' "
    "WHEN t.backtest_id IS NOT NULL THEN 'backtest' "
    "WHEN t.turn_id IS NULL AND l.reason <> 'agent_turn' THEN l.reason "
    "ELSE 'design' END"
)
_PLAN_EXPR = "COALESCE(t.context->'execution'->>'plan_code', 'n/d')"
_TIER_EXPR = "COALESCE(t.context->'execution'->>'tier', 'n/d')"

GROUP_EXPRS: dict[str, str] = {
    "day": "to_char(date_trunc('day', l.created_at), 'YYYY-MM-DD')",
    "week": "to_char(date_trunc('week', l.created_at), 'IYYY-\"W\"IW')",
    "month": "to_char(date_trunc('month', l.created_at), 'YYYY-MM')",
    "provider": "COALESCE(l.provider, 'n/d')",
    "model": "COALESCE(l.model, 'n/d')",
    "tier": _TIER_EXPR,
    "agent": "COALESCE(a.agent_name, CASE WHEN t.agent_id IS NOT NULL THEN 'agent #' || t.agent_id ELSE 'n/d' END)",
    "user": "u.email",
    "plan": _PLAN_EXPR,
    "context": _CONTEXT_EXPR,
    "trigger": "COALESCE(t.trigger_type, l.reason)",
    "estimated": "CASE WHEN l.estimated THEN 'stima' ELSE 'reale' END",
}
GROUP_KEYS = tuple(GROUP_EXPRS)


def _dec(value: Any) -> Optional[Decimal]:
    return None if value is None else Decimal(str(value))


def _row(group: str, r: dict[str, Any]) -> AiCostRow:
    calls = int(r["calls"] or 0)
    with_cost = int(r["calls_with_cost"] or 0)
    cost = _dec(r["cost"])
    tokens_with_cost = int(r["tokens_with_cost"] or 0)
    tokens_input = int(r["tokens_input"] or 0)
    tokens_cached = int(r["tokens_cached"] or 0)
    credits = _dec(r["credits"]) or Decimal("0")
    return AiCostRow(
        group=group,
        calls=calls,
        calls_with_cost=with_cost,
        tokens_input=tokens_input,
        tokens_output=int(r["tokens_output"] or 0),
        tokens_cached=tokens_cached,
        tokens_reasoning=int(r["tokens_reasoning"] or 0),
        cost=cost,
        cost_currency=r["cost_currency"],
        avg_cost_per_call=(cost / with_cost).quantize(Decimal("0.000001")) if cost is not None and with_cost else None,
        cost_per_1k_tokens=(cost / tokens_with_cost * 1000).quantize(Decimal("0.000001")) if cost is not None and tokens_with_cost else None,
        cached_share=(tokens_cached / tokens_input) if tokens_input else None,
        credits=credits,
        credits_per_call=(credits / calls).quantize(Decimal("0.001")) if calls else None,
        wallet_cents=int(r["wallet_cents"] or 0),
    )


def build_cost_report(
    session: Session,
    *,
    group_by: str,
    since: datetime,
    until: datetime,
    provider: str | None = None,
    model: str | None = None,
    user_id: int | None = None,
    plan_code: str | None = None,
    context: str | None = None,
    estimated: bool | None = None,
) -> AiCostReport:
    expr = GROUP_EXPRS[group_by]
    where = ["l.created_at >= :since", "l.created_at < :until", "l.reason <> 'admin_grant'", "l.credits >= 0"]
    params: dict[str, Any] = {"since": since, "until": until}
    if provider:
        where.append("l.provider = :provider")
        params["provider"] = provider
    if model:
        where.append("l.model = :model")
        params["model"] = model
    if user_id is not None:
        where.append("l.user_id = :user_id")
        params["user_id"] = user_id
    if plan_code:
        where.append(f"{_PLAN_EXPR} = :plan_code")
        params["plan_code"] = plan_code
    if context:
        where.append(f"{_CONTEXT_EXPR} = :context")
        params["context"] = context
    if estimated is not None:
        where.append("l.estimated = :estimated")
        params["estimated"] = estimated
    sql = f"""
        SELECT {expr} AS grp,
               count(l.id) AS calls,
               count(l.cost) AS calls_with_cost,
               coalesce(sum(l.tokens_input), 0) AS tokens_input,
               coalesce(sum(l.tokens_output), 0) AS tokens_output,
               coalesce(sum(l.tokens_cached), 0) AS tokens_cached,
               coalesce(sum(l.tokens_reasoning), 0) AS tokens_reasoning,
               sum(l.cost) AS cost,
               max(l.cost_currency) AS cost_currency,
               coalesce(sum(CASE WHEN l.cost IS NOT NULL THEN coalesce(l.tokens_input, 0) + coalesce(l.tokens_output, 0) ELSE 0 END), 0) AS tokens_with_cost,
               coalesce(sum(l.credits), 0) AS credits,
               coalesce(sum(l.wallet_cents), 0) AS wallet_cents
        FROM ai_credit_ledger l
        LEFT JOIN agent_turn t ON t.correlation_id = l.correlation_id AND l.correlation_id IS NOT NULL
        LEFT JOIN agent a ON a.id_agent = t.agent_id
        JOIN "user" u ON u.id = l.user_id
        WHERE {" AND ".join(where)}
        GROUP BY 1
        ORDER BY cost DESC NULLS LAST, calls DESC
    """
    result = session.execute(text(sql), params).mappings().all()
    rows = [_row(str(r["grp"]), dict(r)) for r in result]
    totals: dict[str, Any] = {
        "calls": sum(int(r["calls"] or 0) for r in result),
        "calls_with_cost": sum(int(r["calls_with_cost"] or 0) for r in result),
        "tokens_input": sum(int(r["tokens_input"] or 0) for r in result),
        "tokens_output": sum(int(r["tokens_output"] or 0) for r in result),
        "tokens_cached": sum(int(r["tokens_cached"] or 0) for r in result),
        "tokens_reasoning": sum(int(r["tokens_reasoning"] or 0) for r in result),
        "cost": sum((Decimal(str(r["cost"])) for r in result if r["cost"] is not None), Decimal("0")) if any(r["cost"] is not None for r in result) else None,
        "cost_currency": next((r["cost_currency"] for r in result if r["cost_currency"]), None),
        "tokens_with_cost": sum(int(r["tokens_with_cost"] or 0) for r in result),
        "credits": sum((Decimal(str(r["credits"] or 0)) for r in result), Decimal("0")),
        "wallet_cents": sum(int(r["wallet_cents"] or 0) for r in result),
    }
    from app.services.wallet_service import get_settings

    fx = None
    try:
        fx = get_settings(session).display_fx_eur_per_usd
    except Exception:  # noqa: BLE001
        session.rollback()
    return AiCostReport(
        group_by=group_by, since=since, until=until, rows=rows, total=_row("Totale", totals),
        display_fx_eur_per_usd=fx,
    )
