"""Symbol catalogue of a connection, cached.

Three reads, all about *looking at* symbols (picker, watchlist), none of them
opening a market data subscription:

``get_catalog``      every instrument the connection trades, from the gateway's
                     ``/universe/snapshot`` (the scanner's capability). The
                     client filters it locally, so browsing costs no request.
``get_symbol_info``  the broker's full description of one symbol, from the
                     gateway's ``/search`` (tick size, lot, contract, expiry…).
``get_quotes``       last price and 24 h change of a few symbols: from the
                     catalogue when the venue has a bulk ticker, otherwise from
                     the last closed hourly bars.

Cache (Redis, one key per connection):

- a catalogue is served from cache while fresh (``CATALOG_FRESH_PRICED_S`` when
  it carries prices, ``CATALOG_FRESH_STATIC_S`` when it is a list of names);
- past that it is read again; if the gateway does not answer, the last copy is
  returned with ``stale: true`` and its ``as_of`` — an old list is more useful
  to a picker than an error;
- a connection whose broker has no catalogue (``supported: false``) is
  remembered for ``UNSUPPORTED_TTL_S``: its symbols are found by search only.

Redis being down only removes the cache: every read falls through to the
gateway.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Any

import httpx

from app.services.gateway_client import gateway_url_for
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

CATALOG_FRESH_PRICED_S = 20
CATALOG_FRESH_STATIC_S = 15 * 60
CATALOG_KEEP_S = 7 * 24 * 3600
UNSUPPORTED_TTL_S = 10 * 60
INFO_TTL_S = 60 * 60
SEARCH_TTL_S = 10 * 60
QUOTES_TTL_S = 120

# Bars read to quote a symbol without a ticker: last closed hourly close, and
# the close 24 bars earlier for the change.
QUOTE_TIMEFRAME = "1h"
QUOTE_BARS = 25
MAX_QUOTE_SYMBOLS = 200

_SNAPSHOT_TIMEOUT_S = 20.0
_SEARCH_TIMEOUT_S = 15.0
_BARS_TIMEOUT_S = 60.0

# Status codes that mean "this gateway cannot do it" (409 = capability missing,
# 404/405 = a gateway image older than the capability).
_UNSUPPORTED_STATUS = {404, 405, 409}

_SYMBOL_CORE_FIELDS = ("symbol", "name", "asset_type", "exchange", "currency", "con_id")


class CatalogUnavailableError(RuntimeError):
    """The gateway did not answer and there is no cached copy to fall back on."""


# ── cache ────────────────────────────────────────────────────────────────────


def _cache_get(key: str) -> Any:
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:  # noqa: BLE001 - the cache never fails a read
        logger.debug("Symbol cache read failed for %s: %s", key, exc)
        return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.set(key, json.dumps(value, separators=(",", ":"), default=str), ex=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Symbol cache write failed for %s: %s", key, exc)


def _cache_delete(key: str) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Symbol cache delete failed for %s: %s", key, exc)


def _catalog_key(connection_id: int) -> str:
    return f"symbols:catalog:{connection_id}"


# One gateway read at a time per connection: a page that opens with a cold
# cache from several tabs must not send the same snapshot request N times.
_locks: dict[int, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(connection_id: int) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(connection_id)
        if lock is None:
            lock = _locks[connection_id] = threading.Lock()
        return lock


# ── gateway ──────────────────────────────────────────────────────────────────


class _Unsupported(Exception):
    pass


def _gateway_request(
    connection_id: int,
    broker_type: str,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> Any:
    """One gateway call. ``_Unsupported`` when the gateway cannot serve it,
    ``CatalogUnavailableError`` when it does not answer."""
    url = f"{gateway_url_for(connection_id, broker_type)}{path}"
    try:
        response = httpx.request(method, url, params=params, json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        raise CatalogUnavailableError(f"Gateway of connection {connection_id} unreachable: {exc}") from exc
    if response.status_code in _UNSUPPORTED_STATUS:
        raise _Unsupported(path)
    if response.status_code >= 400:
        raise CatalogUnavailableError(
            f"Gateway of connection {connection_id} answered {response.status_code} on {path}"
        )
    return response.json()


# ── catalogue ────────────────────────────────────────────────────────────────


def _has_prices(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("last") is not None for row in rows)


def _catalog_view(entry: dict[str, Any], connection_id: int, broker_type: str, *, stale: bool) -> dict[str, Any]:
    rows = entry.get("rows") or []
    return {
        "complete": bool(entry.get("complete", True)),
        "connection_id": connection_id,
        "broker_type": entry.get("broker_type") or broker_type,
        "market_type": entry.get("market_type"),
        "supported": bool(entry.get("supported", True)),
        "has_prices": bool(entry.get("has_prices")),
        "quotes": bool(entry.get("quotes", True)),
        "as_of": entry.get("as_of"),
        "stale": stale,
        "count": len(rows),
        "rows": rows,
    }


def _is_fresh(entry: dict[str, Any]) -> bool:
    fetched_at = float(entry.get("fetched_at") or 0)
    age = time.time() - fetched_at
    if not entry.get("supported", True):
        return age < UNSUPPORTED_TTL_S
    return age < (CATALOG_FRESH_PRICED_S if entry.get("has_prices") else CATALOG_FRESH_STATIC_S)


# Brokers whose API lists no market: their catalogue is assembled from public
# listings (``app.services.symbol_listings``) and is a part of what they trade.
LISTING_BROKERS = frozenset({"ibkr"})


def _listing_entry(connection_id: int, broker_type: str) -> dict[str, Any]:
    from app.services import symbol_listings as listings

    futures = listings.ibkr_futures_rows(
        connection_id,
        lambda root: search_symbols_cached(connection_id, broker_type, root, "futures"),
        _cache_get,
        _cache_set,
    )
    # A futures row is what the gateway's search returned (contract under
    # ``extra_data``): the picker selects it as it is.
    rows = futures + listings.us_listed_rows(_cache_get, _cache_set)
    return {
        "supported": bool(rows),
        "complete": False,
        "fetched_at": time.time(),
        "as_of": int(time.time() * 1000),
        "broker_type": broker_type,
        "market_type": None,
        "has_prices": False,
        # No bulk ticker and no recent-bars read on these brokers.
        "quotes": False,
        "rows": rows,
    }


def get_catalog(connection_id: int, broker_type: str, *, refresh: bool = False) -> dict[str, Any]:
    """Every instrument of the connection. ``refresh`` skips the fresh copy
    (the stale fallback still applies)."""
    key = _catalog_key(connection_id)
    cached = _cache_get(key)
    if cached and not refresh and _is_fresh(cached):
        return _catalog_view(cached, connection_id, broker_type, stale=False)

    with _lock_for(connection_id):
        # Another request may have filled the cache while this one waited.
        cached = _cache_get(key) or cached
        if cached and not refresh and _is_fresh(cached):
            return _catalog_view(cached, connection_id, broker_type, stale=False)
        if broker_type in LISTING_BROKERS:
            entry = _listing_entry(connection_id, broker_type)
            _cache_set(key, entry, CATALOG_KEEP_S if entry["rows"] else UNSUPPORTED_TTL_S)
            return _catalog_view(entry, connection_id, broker_type, stale=False)
        try:
            payload = _gateway_request(
                connection_id, broker_type, "GET", "/universe/snapshot", timeout=_SNAPSHOT_TIMEOUT_S
            )
        except _Unsupported:
            entry = {"supported": False, "fetched_at": time.time(), "broker_type": broker_type, "rows": []}
            _cache_set(key, entry, UNSUPPORTED_TTL_S)
            return _catalog_view(entry, connection_id, broker_type, stale=False)
        except CatalogUnavailableError:
            if cached and cached.get("supported", True) and cached.get("rows"):
                return _catalog_view(cached, connection_id, broker_type, stale=True)
            raise

        rows = [row for row in (payload.get("rows") or []) if isinstance(row, dict) and row.get("symbol")]
        entry = {
            "supported": True,
            "fetched_at": time.time(),
            "as_of": payload.get("as_of") or int(time.time() * 1000),
            "broker_type": payload.get("broker_type") or broker_type,
            "market_type": payload.get("market_type"),
            "has_prices": _has_prices(rows),
            "rows": rows,
        }
        _cache_set(key, entry, CATALOG_KEEP_S)
        return _catalog_view(entry, connection_id, broker_type, stale=False)


def forget_catalog(connection_id: int) -> None:
    _cache_delete(_catalog_key(connection_id))


# ── one symbol ───────────────────────────────────────────────────────────────


def as_symbol(raw: dict[str, Any], broker_type: str) -> dict[str, Any]:
    """A gateway ``/search`` row in the shape the symbol picker selects
    (``CachedSymbol`` on the frontend): everything that is not a core field
    goes under ``extra_data``."""
    return {
        "symbol": raw.get("symbol", ""),
        "name": raw.get("name"),
        "asset_type": raw.get("asset_type", ""),
        "exchange": raw.get("exchange"),
        "currency": raw.get("currency"),
        "broker_type": broker_type,
        "con_id": raw.get("con_id"),
        "extra_data": {k: v for k, v in raw.items() if k not in _SYMBOL_CORE_FIELDS and k != "broker_type"},
    }


def search_symbols_cached(
    connection_id: int, broker_type: str, query: str, asset_type: str | None = None
) -> list[dict[str, Any]]:
    """Gateway ``/search``, remembered for ``SEARCH_TTL_S`` per query."""
    text = str(query or "").strip()
    key = f"symbols:search:{connection_id}:{(asset_type or '').lower()}:{text.upper()}"
    cached = _cache_get(key)
    if isinstance(cached, list):
        return cached
    params: dict[str, Any] = {"query": text}
    if asset_type:
        params["asset_type"] = asset_type
    try:
        payload = _gateway_request(
            connection_id, broker_type, "GET", "/search", params=params, timeout=_SEARCH_TIMEOUT_S
        )
    except _Unsupported as exc:
        raise CatalogUnavailableError(f"Gateway of connection {connection_id} has no symbol search") from exc
    results = [as_symbol(row, broker_type) for row in (payload.get("symbols") or []) if isinstance(row, dict)]
    if results:
        _cache_set(key, results, SEARCH_TTL_S)
    return results


def get_symbol_info(connection_id: int, broker_type: str, symbol: str) -> dict[str, Any] | None:
    """The broker's description of one symbol, with its catalogue row (prices)
    under ``market`` when there is one. ``None`` when the broker does not know
    the symbol."""
    wanted = str(symbol or "").strip()
    if not wanted:
        return None
    row = None
    try:
        catalog = get_catalog(connection_id, broker_type)
        row = next((r for r in catalog["rows"] if str(r.get("symbol", "")).upper() == wanted.upper()), None)
    except CatalogUnavailableError:
        pass  # the description stands on its own

    key = f"symbols:info:{connection_id}:{wanted.upper()}"
    info = _cache_get(key)
    if not isinstance(info, dict):
        if row is not None and row.get("extra_data"):
            # The catalogue row already is the broker's description (futures contracts).
            info = {k: row.get(k) for k in (*_SYMBOL_CORE_FIELDS, "extra_data")}
            info["broker_type"] = broker_type
        else:
            matches = [
                m for m in search_symbols_cached(connection_id, broker_type, wanted)
                if str(m.get("symbol", "")).upper() == wanted.upper()
            ]
            info = _best_match(matches, row)
        if info is None:
            return None
        _cache_set(key, info, INFO_TTL_S)

    market = row if row is not None and row.get("last") is not None else None
    return {**info, "market": market}


def _best_match(matches: list[dict[str, Any]], row: dict[str, Any] | None) -> dict[str, Any] | None:
    """Among the broker's exact matches, the one that agrees with the
    catalogue row. A ticker exists on several venues (``AAPL`` in USD on
    Nasdaq, and again in other currencies): currency first, then the type."""
    if not matches:
        return None
    if row is None:
        return matches[0]

    def agreement(match: dict[str, Any]) -> int:
        score = 0
        if row.get("currency") and match.get("currency") == row.get("currency"):
            score += 2
        if row.get("asset_type") and match.get("asset_type") == row.get("asset_type"):
            score += 1
        return score

    return max(matches, key=agreement)  # ties keep the broker's order


# ── quotes ───────────────────────────────────────────────────────────────────

_QUOTE_FIELDS = (
    "last", "bid", "ask", "open_24h", "high_24h", "low_24h", "change_24h_pct", "volume_24h", "turnover_24h", "ts",
)


def _quote_from_bars(bars: list[list[float]]) -> dict[str, Any] | None:
    """Last closed bar as a quote: rows are ``[ts_ms, o, h, l, c, v]``."""
    rows = [bar for bar in bars or [] if isinstance(bar, (list, tuple)) and len(bar) >= 5]
    if not rows:
        return None
    last = rows[-1]
    quote: dict[str, Any] = {"last": last[4], "ts": int(last[0]), "last_from_bar": True}
    if len(rows) >= QUOTE_BARS:
        reference = rows[-QUOTE_BARS][4]
        window = rows[-(QUOTE_BARS - 1):]
        if reference:
            quote["change_24h_pct"] = (last[4] - reference) / reference * 100.0
        quote["high_24h"] = max(bar[2] for bar in window)
        quote["low_24h"] = min(bar[3] for bar in window)
    return quote


def get_quotes(connection_id: int, broker_type: str, symbols: list[str]) -> dict[str, Any]:
    """Quotes of ``symbols`` keyed by symbol. A symbol without a quote is
    simply absent: nothing is estimated."""
    wanted: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        sym = str(raw or "").strip()
        if sym and sym.upper() not in seen:
            seen.add(sym.upper())
            wanted.append(sym)
    wanted = wanted[:MAX_QUOTE_SYMBOLS]
    empty = {"connection_id": connection_id, "supported": False, "as_of": None, "stale": False, "quotes": {}}
    if not wanted:
        return {**empty, "supported": True}

    catalog = get_catalog(connection_id, broker_type)
    if not catalog["supported"] or not catalog["quotes"]:
        return empty
    if catalog["has_prices"]:
        by_symbol = {str(row.get("symbol", "")).upper(): row for row in catalog["rows"]}
        quotes = {}
        for sym in wanted:
            row = by_symbol.get(sym.upper())
            if row is not None:
                quotes[sym] = {field: row.get(field) for field in _QUOTE_FIELDS}
        return {
            "connection_id": connection_id, "supported": True,
            "as_of": catalog["as_of"], "stale": catalog["stale"], "quotes": quotes,
        }

    # No bulk ticker on this venue: the last closed hourly bars, kept a while
    # (one history request per symbol on the connection's own budget).
    digest = hashlib.sha1("|".join(sorted(s.upper() for s in wanted)).encode()).hexdigest()[:16]
    key = f"symbols:quotes:{connection_id}:{digest}"
    cached = _cache_get(key)
    if isinstance(cached, dict):
        return cached
    try:
        payload = _gateway_request(
            connection_id, broker_type, "POST", "/universe/bars",
            body={"symbols": wanted, "timeframe": QUOTE_TIMEFRAME, "count": QUOTE_BARS},
            timeout=_BARS_TIMEOUT_S,
        )
    except _Unsupported:
        return empty
    bars = payload.get("bars") or {}
    by_upper = {str(sym).upper(): rows for sym, rows in bars.items()}
    quotes = {}
    for sym in wanted:
        quote = _quote_from_bars(by_upper.get(sym.upper()) or [])
        if quote is not None:
            quotes[sym] = quote
    result = {
        "connection_id": connection_id, "supported": True,
        "as_of": payload.get("as_of") or int(time.time() * 1000), "stale": False, "quotes": quotes,
    }
    _cache_set(key, result, QUOTES_TTL_S)
    return result
