"""Public listings behind the catalogue of brokers that have none.

IBKR's socket API answers questions about one symbol and never lists a market.
The catalogue of an IBKR connection is therefore assembled here:

- US stocks and ETFs from the Nasdaq Trader Symbol Directory, two public files
  refreshed every trading day (``nasdaqlisted.txt`` = Nasdaq, ``otherlisted.txt``
  = NYSE, NYSE Arca, NYSE American, Cboe BZX, IEX). It is market data of no
  connection in particular: one copy for the whole platform.
- futures from the roots the IBKR gateway knows how to expand into contracts
  (its ``/search``), one read per root, kept per connection.

Such a catalogue is a part of what the broker trades, not all of it: it is
returned with ``complete: false`` and the picker keeps asking the broker for
whatever the list does not hold.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

SYMBOL_DIRECTORY_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/{name}.txt"
US_LISTING_FRESH_S = 24 * 3600
US_LISTING_KEEP_S = 14 * 24 * 3600
FUTURES_FRESH_S = 24 * 3600
_FETCH_TIMEOUT_S = 20.0
_NAME_MAX = 96

# ``otherlisted.txt`` exchange codes.
_OTHER_EXCHANGES = {"A": "AMEX", "N": "NYSE", "P": "ARCA", "Z": "BATS", "V": "IEX"}

# A plain ticker, or a share class written ``BRK.B`` (IBKR spells it ``BRK B``).
# Preferreds (``$``), warrants, units and rights are left to the broker search.
_PLAIN_TICKER = re.compile(r"^[A-Z]{1,5}$")
_CLASS_TICKER = re.compile(r"^([A-Z]{1,5})\.([A-Z])$")

# Roots the IBKR gateway expands into contracts (``_KNOWN_FUTURES_ROOTS`` in
# its ``broker/ibkr/streaming.py``). A root it cannot resolve adds no row.
IBKR_FUTURES_ROOTS = (
    "ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K",
    "CL", "MCL", "NG", "GC", "MGC", "SI", "HG", "ZB", "ZN", "ZF", "ZT",
)


def _ibkr_ticker(raw: str) -> str | None:
    ticker = raw.strip().upper()
    if _PLAIN_TICKER.match(ticker):
        return ticker
    share_class = _CLASS_TICKER.match(ticker)
    return f"{share_class.group(1)} {share_class.group(2)}" if share_class else None


def _row(ticker: str, name: str, exchange: str, is_etf: bool) -> dict[str, Any]:
    return {
        "symbol": ticker,
        "name": name.strip()[:_NAME_MAX],
        "asset_type": "etf" if is_etf else "stock",
        "exchange": exchange,
        "currency": "USD",
    }


def parse_symbol_directory(nasdaq_listed: str, other_listed: str) -> list[dict[str, Any]]:
    """Rows of the two Symbol Directory files, test issues left out, one row
    per ticker, sorted by symbol."""
    rows: dict[str, dict[str, Any]] = {}

    def lines(text: str):
        for line in text.splitlines()[1:]:
            fields = line.split("|")
            if len(fields) >= 7 and not fields[0].startswith("File Creation Time"):
                yield fields

    # Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
    for fields in lines(nasdaq_listed):
        ticker = _ibkr_ticker(fields[0])
        if ticker and fields[3] != "Y":
            rows.setdefault(ticker, _row(ticker, fields[1], "NASDAQ", fields[6] == "Y"))
    # ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
    for fields in lines(other_listed):
        ticker = _ibkr_ticker(fields[0])
        if ticker and fields[6] != "Y":
            rows.setdefault(ticker, _row(ticker, fields[1], _OTHER_EXCHANGES.get(fields[2], fields[2]), fields[4] == "Y"))
    return [rows[key] for key in sorted(rows)]


def _download(name: str) -> str:
    response = httpx.get(
        SYMBOL_DIRECTORY_URL.format(name=name),
        timeout=_FETCH_TIMEOUT_S,
        headers={"User-Agent": "Mozilla/5.0 (compatible; symbol-directory-reader)"},
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.text


def us_listed_rows(cache_get: Callable[[str], Any], cache_set: Callable[[str, Any, int], None]) -> list[dict[str, Any]]:
    """US stocks and ETFs. Read once a day; when the download fails the last
    copy is used, and with no copy the list is empty (the broker search still
    works)."""
    key = "symbols:listing:us"
    cached = cache_get(key)
    if isinstance(cached, dict) and time.time() - float(cached.get("fetched_at") or 0) < US_LISTING_FRESH_S:
        return cached.get("rows") or []
    try:
        rows = parse_symbol_directory(_download("nasdaqlisted"), _download("otherlisted"))
        if not rows:
            raise ValueError("the symbol directory came back empty")
    except Exception as exc:  # noqa: BLE001 - a public file must never break the picker
        logger.warning("US symbol directory not read: %s", exc)
        return (cached.get("rows") or []) if isinstance(cached, dict) else []
    cache_set(key, {"fetched_at": time.time(), "rows": rows}, US_LISTING_KEEP_S)
    return rows


def ibkr_futures_rows(
    connection_id: int,
    search: Callable[[str], list[dict[str, Any]]],
    cache_get: Callable[[str], Any],
    cache_set: Callable[[str, Any, int], None],
) -> list[dict[str, Any]]:
    """Active contracts of the known roots, as the gateway's search describes
    them (``extra_data`` included: a row is selectable as it is). Kept a day;
    nothing is kept when the gateway does not answer, so the next read retries."""
    key = f"symbols:listing:ibkr-futures:{connection_id}"
    cached = cache_get(key)
    if isinstance(cached, list):
        return cached
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    answered = False
    for root in IBKR_FUTURES_ROOTS:
        try:
            matches = search(root)
        except Exception as exc:  # noqa: BLE001
            logger.info("IBKR futures root %s not read on connection %s: %s", root, connection_id, exc)
            if not answered:
                # The gateway is not answering at all: no point asking 18 more times.
                return []
            continue
        answered = True
        for match in matches:
            symbol = str(match.get("symbol") or "")
            if not symbol or symbol in seen or match.get("asset_type") != "futures":
                continue
            seen.add(symbol)
            rows.append({**match, "status": "live", "tradable": True})
    if rows:
        cache_set(key, rows, FUTURES_FRESH_S)
    return rows
