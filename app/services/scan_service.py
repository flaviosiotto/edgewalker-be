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


def create_scan(session: Session, payload: ScanCreate, user_id: int) -> Scan:
    """Create a scan from a strategy of the user and start it on the coordinator."""
    from app.services.indicator_ref_service import inject_user_indicator_refs
    from app.services.strategy_service import (
        _normalize_strategy_indicator_field_references,
        _strip_rule_chat_ids,
        get_strategy,
    )

    strategy = get_strategy(session, payload.strategy_id, user_id)
    connection = session.get(Connection, strategy.connection_id) if strategy.connection_id else None
    if connection is None or connection.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The strategy has no connection to read the market from",
        )
    config = _strip_rule_chat_ids(_normalize_strategy_indicator_field_references(strategy.definition))
    # Pin user-defined indicators to their current version, as a backtest does
    config = inject_user_indicator_refs(config, user_id)
    timeframe = str(payload.timeframe or "").strip() or _primary_timeframe(config)
    if not timeframe:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The strategy has no timeframe: give one")
    symbols = [s.strip().upper() for s in (payload.symbols or []) if s and s.strip()] or None

    scan = Scan(
        user_id=user_id,
        strategy_id=strategy.id,
        strategy_name=strategy.name,
        connection_id=connection.id,
        broker_type=connection.broker_type,
        timeframe=timeframe,
        request={
            "symbols": symbols,
            "min_turnover_24h": payload.min_turnover_24h,
            "max_symbols": payload.max_symbols,
        },
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
        **(scan.request or {}),
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


def _sort_value(row: ScanRow, key: str) -> Any:
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
    symbol: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> tuple[Scan, list[ScanRow], int]:
    """A page of the scan's table.

    ``rule`` keeps the rows where that rule is true; ``matched_only`` the rows
    where at least one rule is. Rows without the sort value come last in both
    directions, ties are broken by symbol. A scan has at most a thousand rows:
    they are filtered and ordered here rather than in SQL over JSONB.
    """
    scan = get_scan(session, scan_id, user_id)
    rows = list(session.exec(select(ScanRow).where(ScanRow.scan_id == scan_id)).all())
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
        present = [r for r in rows if not _is_missing(_sort_value(r, key))]
        missing = [r for r in rows if _is_missing(_sort_value(r, key))]
        try:
            present.sort(key=lambda r: _sort_value(r, key), reverse=descending)
        except TypeError:
            present.sort(key=lambda r: str(_sort_value(r, key)), reverse=descending)
        rows = present + missing
    total = len(rows)
    return scan, rows[offset:offset + limit], total
