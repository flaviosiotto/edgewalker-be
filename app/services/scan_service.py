"""Symbol scans: create one from a strategy, start it, read its table.

The scan itself runs in the coordinator (strategy-backtest ``/scans/{id}/run``),
which reads the market through the connection's gateway and writes
``scan_rows``. The backend owns the request and the reads
(docs/valutazione-scanner-simboli.md).

Rows are returned in the order asked, never ranked by the platform: the
default order is by symbol, and a sort key is always the caller's choice.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any

import httpx
from fastapi import HTTPException, status
from sqlalchemy import func
from sqlmodel import Session, select

from app.models.connection import Connection
from app.models.scan import Scan, ScanRow, ScanStatus
from app.schemas.scan import ScanCreate

logger = logging.getLogger(__name__)

BACKTEST_SERVICE_URL = os.getenv("BACKTEST_SERVICE_URL", "http://strategy-backtest:8080").rstrip("/")

# Sort keys that are columns of the row; anything else is looked up in
# ``market`` (``market.turnover_24h``) or ``values`` (``indicators.rsi14``).
_ROW_SORT_KEYS = {"symbol", "bar_ts", "matched_rules", "conditions_true", "conditions_total"}


def _primary_timeframe(definition: Any) -> str:
    strategy = definition.get("strategy") if isinstance(definition, dict) and isinstance(definition.get("strategy"), dict) else {}
    charts = strategy.get("charts")
    if isinstance(charts, list) and charts and isinstance(charts[0], dict) and charts[0].get("timeframe"):
        return str(charts[0]["timeframe"])
    return str(strategy.get("timeframe") or (definition.get("timeframe") if isinstance(definition, dict) else "") or "")


FREE_RULE_NAME = "filter"
FREE_CHART_ID = "main"
_FREE_NAMESPACES = {"price", "prev_price", "indicators", "prev_indicators", "session", "time"}


def free_definition(payload: ScanCreate, timeframe: str) -> dict[str, Any]:
    """The definition a free scan evaluates: one chart with the indicators
    asked and, when there are conditions, one rule that never trades.

    It is a strategy definition on purpose — the coordinator reads every scan
    the same way — and it is validated like one.
    """
    from edgewalker_platform.rules_dsl import FIELD_PATH_RE, split_chart_prefix, split_offset, validate_definition

    indicators: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in payload.indicators or []:
        kind = str(raw.get("type") or "").strip()
        name = str(raw.get("name") or "").strip() or kind.lower()
        if not kind:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Every indicator needs a type")
        if name in seen:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Two indicators are named '{name}'")
        seen.add(name)
        params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
        indicators.append({"name": name, "type": kind, "params": {k: v for k, v in params.items() if k != "ew_hash"}})
    conditions = [dict(c) for c in payload.conditions or [] if isinstance(c, dict)]
    for cond in conditions:
        if not str(cond.get("field") or "").strip() or not str(cond.get("op") or "").strip():
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Every condition needs a field and an operator")
    rules = (
        [{"name": FREE_RULE_NAME, "action": "hold", "logic": payload.logic, "conditions": conditions}]
        if conditions else []
    )
    definition = {"strategy": {
        "name": (payload.name or "").strip() or "scan",
        "symbol": "", "timeframe": timeframe, "timezone": "UTC",
        "indicators": indicators,
        "charts": [{"id": FREE_CHART_ID, "symbol": "", "timeframe": timeframe, "indicators": indicators}],
        "rules": rules,
    }}
    problems = list(validate_definition(definition))
    # What a scan cannot read is refused here rather than shown as "not evaluable":
    # in a free scan the conditions are the whole point.
    for cond in conditions:
        texts = [str(cond.get(k)) for k in ("field", "value", "upper", "lower") if isinstance(cond.get(k), str)]
        for ref in (m.group(0) for text in texts for m in FIELD_PATH_RE.finditer(text)):
            chart, rest = split_chart_prefix(split_offset(ref)[0])
            namespace = rest.split(".", 1)[0]
            if namespace not in _FREE_NAMESPACES or (chart is not None and chart != FREE_CHART_ID):
                problems.append(f"'{ref}' cannot be read in a scan: use price.*, indicators.*, session.* or time.*")
    if problems:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="; ".join(problems[:5]))
    return definition


def create_scan(session: Session, payload: ScanCreate, user_id: int) -> Scan:
    """Create a scan and start it on the coordinator.

    From a strategy of the user (its saved definition), or free: indicators
    and conditions given in the request (``free_definition``).
    """
    from app.services.indicator_ref_service import inject_user_indicator_refs
    from app.services.strategy_service import (
        _normalize_strategy_indicator_field_references,
        _strip_rule_chat_ids,
        get_strategy,
    )

    strategy = get_strategy(session, payload.strategy_id, user_id) if payload.strategy_id is not None else None
    connection_id = payload.connection_id if payload.is_free and payload.connection_id is not None else (
        strategy.connection_id if strategy is not None else None
    )
    connection = session.get(Connection, connection_id) if connection_id else None
    if connection is None or connection.user_id != user_id:
        if payload.connection_id is not None and payload.is_free:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Connection not found")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The strategy has no connection to read the market from",
        )
    if payload.is_free:
        timeframe = str(payload.timeframe or "").strip()
        config = _normalize_strategy_indicator_field_references(free_definition(payload, timeframe))
    else:
        config = _strip_rule_chat_ids(_normalize_strategy_indicator_field_references(strategy.definition))
        timeframe = str(payload.timeframe or "").strip() or _primary_timeframe(config)
    # Pin user-defined indicators to their current version, as a backtest does
    config = inject_user_indicator_refs(config, user_id)
    if not timeframe:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The strategy has no timeframe: give one")
    symbols = [s.strip().upper() for s in (payload.symbols or []) if s and s.strip()] or None

    request: dict[str, Any] = {
        "symbols": symbols,
        "min_turnover_24h": payload.min_turnover_24h,
        "max_symbols": payload.max_symbols,
        "lookback": payload.lookback,
    }
    if payload.is_free:
        # Kept as asked, so the scan can be shown and launched again as it was
        request["free"] = {
            "name": (payload.name or "").strip() or None,
            "indicators": config["strategy"]["charts"][0]["indicators"],
            "conditions": payload.conditions or [],
            "logic": payload.logic,
        }
    scan = Scan(
        user_id=user_id,
        strategy_id=None if payload.is_free else strategy.id,
        # A free scan has no strategy: its own title takes that place
        strategy_name=((payload.name or "").strip() or None) if payload.is_free else strategy.name,
        connection_id=connection.id,
        broker_type=connection.broker_type,
        timeframe=timeframe,
        request=request,
        config=config,
        status=ScanStatus.PENDING.value,
    )
    session.add(scan)
    session.commit()
    session.refresh(scan)

    try:
        _start_on_coordinator(scan)
    except RuntimeError as exc:
        scan.status = ScanStatus.FAILED.value
        scan.error_message = str(exc)
        session.add(scan)
        session.commit()
        session.refresh(scan)
    return scan


def _start_on_coordinator(scan: Scan) -> None:
    body = {
        "strategy_config": scan.config,
        "connection_id": scan.connection_id,
        "broker_type": scan.broker_type or "",
        "timeframe": scan.timeframe,
        **{k: v for k, v in (scan.request or {}).items() if k in ("symbols", "min_turnover_24h", "max_symbols", "lookback")},
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(f"{BACKTEST_SERVICE_URL}/scans/{scan.id}/run", json=body)
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail: Any = exc.response.json().get("detail")
        except Exception:  # noqa: BLE001
            detail = exc.response.text
        raise RuntimeError(f"The scan could not be started: {detail}") from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"The scan service is not reachable: {exc}") from exc


def get_scan(session: Session, scan_id: int, user_id: int) -> Scan:
    scan = session.get(Scan, scan_id)
    if scan is None or scan.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scan not found")
    return scan


def list_scans(
    session: Session, user_id: int, *, strategy_id: int | None = None, limit: int = 50, offset: int = 0,
) -> tuple[list[Scan], int]:
    """Scans of the user, newest first. ``config`` and ``plan`` stay in the DB."""
    where = [Scan.user_id == user_id]
    if strategy_id is not None:
        where.append(Scan.strategy_id == strategy_id)
    total = session.exec(select(func.count()).select_from(Scan).where(*where)).one()
    summary_columns = [
        Scan.id, Scan.strategy_id, Scan.strategy_name, Scan.connection_id, Scan.broker_type, Scan.market_type,
        Scan.timeframe, Scan.mode, Scan.status, Scan.symbols_total, Scan.symbols_done, Scan.symbols_failed,
        Scan.error_message, Scan.created_at, Scan.started_at, Scan.completed_at,
        Scan.request.op("?")("free").label("free"),
    ]
    rows = session.exec(
        select(*summary_columns).where(*where).order_by(Scan.id.desc()).limit(limit).offset(offset)
    ).mappings().all()
    return [dict(r) for r in rows], int(total)  # type: ignore[return-value]


def delete_scan(session: Session, scan_id: int, user_id: int) -> None:
    scan = get_scan(session, scan_id, user_id)
    if scan.status == ScanStatus.RUNNING.value:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The scan is still running")
    session.delete(scan)
    session.commit()


def value_keys(session: Session, scan_id: int) -> list[str]:
    """Field paths present in the rows' ``values`` (the table's measure columns)."""
    keys: set[str] = set()
    for values in session.exec(select(ScanRow.values).where(ScanRow.scan_id == scan_id)).all():
        if isinstance(values, dict):
            keys.update(str(k) for k in values)
    return sorted(keys)


def _last_true(row: ScanRow, rule: str | None = None) -> int | None:
    """Bars ago a rule was last true (0 = the last closed bar): of ``rule``, or the most recent of all."""
    found: list[int] = []
    for outcome in row.rules or []:
        if not isinstance(outcome, dict) or (rule and outcome.get("name") != rule):
            continue
        ago = outcome.get("last_true")
        if ago is None and "last_true" not in outcome and outcome.get("matched") is True:
            ago = 0  # a scan of the last bar only
        if isinstance(ago, int):
            found.append(ago)
    return min(found) if found else None


def _sort_value(row: ScanRow, key: str, rule: str | None = None) -> Any:
    if key == "last_true":
        return _last_true(row, rule)
    if key in _ROW_SORT_KEYS:
        return getattr(row, key)
    if key.startswith("market."):
        return (row.market or {}).get(key[len("market."):]) if isinstance(row.market, dict) else None
    return (row.values or {}).get(key) if isinstance(row.values, dict) else None


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _rule_matched(row: ScanRow, rule: str) -> bool:
    for outcome in row.rules or []:
        if isinstance(outcome, dict) and outcome.get("name") == rule:
            return outcome.get("matched") is True
    return False


def list_scan_rows(
    session: Session,
    scan_id: int,
    user_id: int,
    *,
    sort: str = "symbol",
    descending: bool = False,
    rule: str | None = None,
    matched_only: bool = False,
    within: int | None = None,
    symbol: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> tuple[Scan, list[ScanRow], int]:
    """A page of the scan's table.

    ``rule`` keeps the rows where that rule is true; ``matched_only`` the rows
    where at least one rule is. ``within`` widens both from "true on the last
    closed bar" to "true on one of the last ``within`` bars" (1 = the last bar),
    which is how a rule that fired and went off again is found; ``sort=last_true``
    orders by how many bars ago that was. Rows without the sort value come last in both
    directions, ties are broken by symbol. A scan has at most a thousand rows:
    they are filtered and ordered here rather than in SQL over JSONB.
    """
    scan = get_scan(session, scan_id, user_id)
    rows = list(session.exec(select(ScanRow).where(ScanRow.scan_id == scan_id)).all())
    if within:
        if rule or matched_only:
            rows = [r for r in rows if (ago := _last_true(r, rule)) is not None and ago < within]
    else:
        if rule:
            rows = [r for r in rows if _rule_matched(r, rule)]
        if matched_only:
            rows = [r for r in rows if (r.matched_rules or 0) > 0]
    if symbol:
        needle = symbol.strip().upper()
        rows = [r for r in rows if needle in r.symbol.upper() or needle in str(r.name or "").upper()]

    key = (sort or "symbol").strip()
    rows.sort(key=lambda r: r.symbol)
    if key != "symbol" or descending:
        present = [r for r in rows if not _is_missing(_sort_value(r, key, rule))]
        missing = [r for r in rows if _is_missing(_sort_value(r, key, rule))]
        try:
            present.sort(key=lambda r: _sort_value(r, key, rule), reverse=descending)
        except TypeError:
            present.sort(key=lambda r: str(_sort_value(r, key, rule)), reverse=descending)
        rows = present + missing
    total = len(rows)
    return scan, rows[offset:offset + limit], total
