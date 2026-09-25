"""Contract multiplier of the instrument a backtest trades.

Money P&L is ``price difference x quantity x multiplier``: 1 for stocks, the
contract multiplier for futures (MNQ = 2 USD/point, ES = 50, MES = 5, ...).
Live, the ledger learns it from the broker fills; a backtest has no broker in
the loop, so the backend resolves it here at launch and pins it into the run
(``config.backtest.contract_multiplier`` + ``CONTRACT_MULTIPLIER`` for the
runner, which forwards it to strategy-backtest).

Sources, most trustworthy first:

1. the value already pinned in the backtest config (re-runs stay consistent);
2. the strategy definition (``strategy.extra_data.multiplier`` or the primary
   chart's ``extra_data``): the symbol picker stores the gateway's contract
   details there;
3. the symbol cache of the strategy's connection;
4. the live ``trades`` ledger of the connection's accounts for the same
   symbol: what the broker actually reported;
5. for futures, a live contract lookup on the connection's gateway;
6. 1.0, logged as a warning for futures.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from sqlmodel import Session, select

from app.models.connection import Account, Connection
from app.models.live_trading import LiveTrade
from app.models.marketdata import SymbolCache

logger = logging.getLogger(__name__)

FUTURES_ASSETS = {"future", "futures", "fut"}


def _positive(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _extra_multiplier(extra: Any) -> float | None:
    if not isinstance(extra, dict):
        return None
    return _positive(extra.get("multiplier") or extra.get("contract_multiplier"))


def multiplier_from_config(config: Any) -> float | None:
    """``config.backtest.contract_multiplier`` of a backtest config snapshot."""
    if not isinstance(config, dict):
        return None
    backtest_cfg = config.get("backtest")
    if not isinstance(backtest_cfg, dict):
        return None
    return _positive(backtest_cfg.get("contract_multiplier"))


def multiplier_from_definition(definition: Any, symbol: str | None = None) -> float | None:
    """Contract multiplier stored by the symbol picker in a strategy DSL document.

    ``symbol`` restricts the lookup to the chart trading that symbol (the
    strategy-level ``extra_data`` always refers to the primary symbol).
    """
    if not isinstance(definition, dict):
        return None
    strategy = definition.get("strategy") if isinstance(definition.get("strategy"), dict) else definition
    if not isinstance(strategy, dict):
        return None
    wanted = str(symbol or strategy.get("symbol") or "").strip().upper()
    if not symbol or str(strategy.get("symbol") or "").strip().upper() == wanted:
        found = _extra_multiplier(strategy.get("extra_data"))
        if found is not None:
            return found
    charts = strategy.get("charts")
    if isinstance(charts, list):
        for chart in charts:
            if not isinstance(chart, dict):
                continue
            if wanted and str(chart.get("symbol") or "").strip().upper() != wanted:
                continue
            found = _extra_multiplier(chart.get("extra_data"))
            if found is not None:
                return found
    return None


def multiplier_from_symbol_cache(session: Session, *, connection_id: int, symbol: str) -> float | None:
    rows = session.exec(
        select(SymbolCache).where(SymbolCache.connection_id == connection_id, SymbolCache.symbol == symbol)
    ).all()
    for row in rows:
        found = _extra_multiplier(row.extra_data)
        if found is not None:
            return found
    return None


def multiplier_from_live_trades(session: Session, *, connection_id: int, symbol: str) -> float | None:
    """Multiplier the broker reported on the most recent live trade of ``symbol``
    on any account of the connection (only values above 1 are informative: a
    1.0 may just be the ledger default)."""
    row = session.exec(
        select(LiveTrade.multiplier)
        .join(Account, Account.id == LiveTrade.account_id)
        .where(Account.connection_id == connection_id, LiveTrade.symbol == symbol, LiveTrade.multiplier > 1)
        .order_by(LiveTrade.id.desc())
        .limit(1)
    ).first()
    return _positive(row)


def _futures_root(symbol: str) -> str:
    match = re.match(r"^([A-Z0-9]+?)[FGHJKMNQUVXZ]\d{1,2}$", symbol.upper())
    return match.group(1) if match else symbol.upper()


def multiplier_from_gateway(*, connection: Connection, symbol: str) -> float | None:
    """Live contract lookup on the connection's gateway (futures only)."""
    from app.services.symbol_sync_handler import search_gateway_symbols_by_id

    try:
        results = search_gateway_symbols_by_id(
            _futures_root(symbol), int(connection.id), str(connection.broker_type or "ibkr"), asset_type="futures", limit=200
        )
    except Exception as exc:  # noqa: BLE001 - the gateway may simply be offline
        logger.info("Contract lookup for %s on connection %s skipped: %s", symbol, connection.id, exc)
        return None
    wanted = symbol.strip().upper()
    for item in results:
        if str(item.get("symbol") or "").strip().upper() != wanted:
            continue
        found = _extra_multiplier(item.get("extra_data"))
        if found is not None:
            return found
    return None


def resolve_contract_multiplier(
    session: Session,
    *,
    symbol: str,
    asset: str | None,
    connection: Connection | None,
    backtest_config: Any = None,
    strategy_definition: Any = None,
) -> tuple[float, str]:
    """Return ``(multiplier, source)`` for the instrument of a backtest."""
    symbol = str(symbol or "").strip().upper()
    is_future = str(asset or "").strip().lower() in FUTURES_ASSETS

    found = multiplier_from_config(backtest_config)
    if found is not None:
        return found, "backtest_config"
    for definition in (backtest_config, strategy_definition):
        found = multiplier_from_definition(definition, symbol)
        if found is not None:
            return found, "strategy_definition"
    if connection is not None and connection.id is not None and symbol:
        try:
            found = multiplier_from_symbol_cache(session, connection_id=int(connection.id), symbol=symbol)
        except Exception:
            logger.exception("symbol cache lookup failed for %s", symbol)
            found = None
        if found is not None:
            return found, "symbol_cache"
        try:
            found = multiplier_from_live_trades(session, connection_id=int(connection.id), symbol=symbol)
        except Exception:
            logger.exception("live trades lookup failed for %s", symbol)
            found = None
        if found is not None:
            return found, "live_trades"
        if is_future:
            found = multiplier_from_gateway(connection=connection, symbol=symbol)
            if found is not None:
                return found, "gateway"
    if is_future:
        logger.warning(
            "Contract multiplier for future %s not found (connection %s): P&L will use 1.0",
            symbol, getattr(connection, "id", None),
        )
    return 1.0, "default"
