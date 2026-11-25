"""Symbol scans (migr. 077, docs/valutazione-scanner-simboli.md): creation from
a strategy, the start on the coordinator, the table reads and ownership.
Run: ``venv/bin/python -m pytest tests/test_scans_db.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlmodel import Session, select

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


def _tenant(session, tag: str):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == f"{tag}@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email=f"{tag}@example.com", username=f"user_{tag}", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name=f"conn-{tag}", broker_type="okx", status="connected", created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id=f"acc-{tag}", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    return user, conn, acc


@pytest.fixture
def tenant(session):
    user, conn, acc = _tenant(session, "scan")
    yield user, conn, acc
    session.delete(user)
    session.commit()


@pytest.fixture
def stranger(session):
    user, conn, acc = _tenant(session, "scan_other")
    yield user
    session.delete(user)
    session.commit()


def _strategy(session, user, acc):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    ema = {"name": "ema20", "type": "ema", "params": {"timeperiod": 20}}
    definition = {"strategy": {
        "name": "scan", "symbol": "BTC-USDT-SWAP", "asset": "crypto_futures", "timeframe": "1h", "timezone": "UTC",
        "indicators": [ema],
        "charts": [{"id": "main", "symbol": "BTC-USDT-SWAP", "asset": "crypto_futures", "timeframe": "1h", "indicators": [ema]}],
        "rules": [{"name": "above", "action": "buy", "conditions": [{"field": "price.close", "op": ">", "value": "indicators.ema20"}]}],
    }}
    return create_strategy(session, StrategyCreate(name="scan strategy", definition=definition, account_id=acc.id), user.id)


@pytest.fixture
def started(monkeypatch):
    """The coordinator call, recorded instead of sent."""
    from app.services import scan_service

    calls: list[dict] = []

    def fake(scan):
        calls.append({"id": scan.id, "config": scan.config, "timeframe": scan.timeframe, "request": dict(scan.request)})

    monkeypatch.setattr(scan_service, "_start_on_coordinator", fake)
    monkeypatch.setattr("app.services.indicator_ref_service.inject_user_indicator_refs", lambda definition, user_id: definition)
    return calls


def _rows(session, scan_id):
    from app.models.scan import ScanRow

    data = [
        ("BBB", 30.0, 500.0, 1, [{"name": "above", "matched": True, "conditions": [{"passed": True}]}]),
        ("AAA", 70.0, 900.0, 0, [{"name": "above", "matched": False, "conditions": [{"passed": False}]}]),
        ("CCC", None, None, 0, [{"name": "above", "matched": None, "conditions": [{"passed": None}]}]),
    ]
    for symbol, rsi, turnover, matched, rules in data:
        session.add(ScanRow(
            scan_id=scan_id, symbol=symbol, name=symbol.lower(), bar_ts=1,
            market={"turnover_24h": turnover}, values={"indicators.rsi": rsi} if rsi is not None else {},
            rules=rules, matched_rules=matched, conditions_true=matched, conditions_total=1,
        ))
    session.commit()


def test_migration_077_is_valid_sql_and_idempotent(app_engine):
    sql = (Path(__file__).resolve().parents[1] / "migrations" / "077_scans.sql").read_text()
    for _ in range(2):
        with app_engine.connect() as conn:
            conn.exec_driver_sql(sql)
    with app_engine.connect() as conn:
        columns = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'scan_rows'"
        ))}
    from app.models.scan import Scan, ScanRow

    assert columns == set(ScanRow.__table__.columns.keys())
    with app_engine.connect() as conn:
        scan_columns = {r[0] for r in conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'scans'"
        ))}
    assert scan_columns == set(Scan.__table__.columns.keys())


def test_create_snapshots_the_strategy_and_starts_the_coordinator(session, tenant, started):
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan

    user, conn, acc = tenant
    strategy = _strategy(session, user, acc)
    scan = create_scan(session, ScanCreate(strategy_id=strategy.id, symbols=[" btc-usdt-swap ", ""], min_turnover_24h=1e6), user.id)

    assert scan.status == "pending" and scan.mode == "on_demand"
    assert scan.timeframe == "1h", "the primary chart's timeframe when none is given"
    assert scan.connection_id == conn.id and scan.broker_type == "okx" and scan.strategy_name == "scan strategy"
    assert scan.request == {"symbols": ["BTC-USDT-SWAP"], "min_turnover_24h": 1e6, "max_symbols": 300}
    assert scan.config["strategy"]["rules"][0]["name"] == "above"
    assert started == [{"id": scan.id, "config": scan.config, "timeframe": "1h", "request": scan.request}]

    other_tf = create_scan(session, ScanCreate(strategy_id=strategy.id, timeframe="15m"), user.id)
    assert other_tf.timeframe == "15m" and other_tf.request["symbols"] is None


def test_a_coordinator_that_cannot_start_leaves_a_failed_scan(session, tenant, monkeypatch):
    from app.schemas.scan import ScanCreate
    from app.services import scan_service

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)
    monkeypatch.setattr("app.services.indicator_ref_service.inject_user_indicator_refs", lambda definition, user_id: definition)
    monkeypatch.setattr(scan_service, "BACKTEST_SERVICE_URL", "http://127.0.0.1:1")
    scan = scan_service.create_scan(session, ScanCreate(strategy_id=strategy.id), user.id)
    assert scan.status == "failed" and "not reachable" in scan.error_message


def test_rows_are_by_symbol_unless_a_sort_is_asked_and_missing_values_come_last(session, tenant, started):
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan, list_scan_rows, value_keys

    user, _, acc = tenant
    scan = create_scan(session, ScanCreate(strategy_id=_strategy(session, user, acc).id), user.id)
    _rows(session, scan.id)

    def symbols(**kwargs):
        _, rows, total = list_scan_rows(session, scan.id, user.id, **kwargs)
        return [r.symbol for r in rows], total

    assert symbols() == (["AAA", "BBB", "CCC"], 3)
    assert symbols(sort="indicators.rsi")[0] == ["BBB", "AAA", "CCC"]
    assert symbols(sort="indicators.rsi", descending=True)[0] == ["AAA", "BBB", "CCC"], "missing last in both directions"
    assert symbols(sort="market.turnover_24h", descending=True)[0] == ["AAA", "BBB", "CCC"]
    assert symbols(sort="matched_rules", descending=True)[0] == ["BBB", "AAA", "CCC"]
    assert symbols(sort="symbol", descending=True)[0] == ["CCC", "BBB", "AAA"]
    assert symbols(rule="above") == (["BBB"], 1)
    assert symbols(rule="nope") == ([], 0)
    assert symbols(matched_only=True) == (["BBB"], 1)
    assert symbols(symbol="aa") == (["AAA"], 1)
    assert symbols(limit=1, offset=1) == (["BBB"], 3), "total counts the rows before the page cut"
    assert value_keys(session, scan.id) == ["indicators.rsi"]


def test_scans_belong_to_their_user(session, tenant, stranger, started):
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan, delete_scan, get_scan, list_scan_rows, list_scans

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)
    scan = create_scan(session, ScanCreate(strategy_id=strategy.id), user.id)

    for call in (
        lambda: get_scan(session, scan.id, stranger.id),
        lambda: list_scan_rows(session, scan.id, stranger.id),
        lambda: delete_scan(session, scan.id, stranger.id),
        lambda: create_scan(session, ScanCreate(strategy_id=strategy.id), stranger.id),
    ):
        with pytest.raises(HTTPException) as err:
            call()
        assert err.value.status_code == 404
    assert list_scans(session, stranger.id) == ([], 0)

    items, total = list_scans(session, user.id, strategy_id=strategy.id)
    assert total == 1 and items[0]["id"] == scan.id and "config" not in items[0]
    assert items[0]["free"] is False


def test_delete_removes_the_rows_and_refuses_a_running_scan(session, tenant, started):
    from app.models.scan import ScanRow
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan, delete_scan

    user, _, acc = tenant
    scan = create_scan(session, ScanCreate(strategy_id=_strategy(session, user, acc).id), user.id)
    _rows(session, scan.id)
    scan.status = "running"
    session.add(scan)
    session.commit()
    with pytest.raises(HTTPException) as err:
        delete_scan(session, scan.id, user.id)
    assert err.value.status_code == 409
    scan.status = "completed"
    session.add(scan)
    session.commit()
    delete_scan(session, scan.id, user.id)
    assert session.exec(select(ScanRow).where(ScanRow.scan_id == scan.id)).all() == []


# ── Free scans: indicators and conditions without a strategy ──────────

RSI = {"name": "rsi14", "type": "rsi", "params": {"timeperiod": 14, "ew_hash": "stale"}}


def test_a_free_scan_builds_its_own_definition_on_a_connection(session, tenant, started):
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan

    user, conn, _ = tenant
    scan = create_scan(session, ScanCreate(
        connection_id=conn.id, name="Ipervenduto 1h", timeframe="1h", indicators=[RSI],
        conditions=[{"field": "indicators.rsi14", "op": "<", "value": 30}, {"field": "price.close", "op": ">", "value": "price.open"}],
        logic="or", max_symbols=50,
    ), user.id)

    assert scan.strategy_id is None and scan.strategy_name == "Ipervenduto 1h" and scan.connection_id == conn.id
    strategy = scan.config["strategy"]
    assert strategy["timeframe"] == "1h" and strategy["charts"][0]["id"] == "main"
    assert strategy["charts"][0]["indicators"] == [{"name": "rsi14", "type": "rsi", "params": {"timeperiod": 14}}], "client ew_hash dropped"
    assert strategy["rules"] == [{"name": "filter", "action": "hold", "logic": "or", "conditions": [
        {"field": "indicators.rsi14", "op": "<", "value": 30}, {"field": "price.close", "op": ">", "value": "price.open"},
    ]}]
    assert scan.request["free"]["logic"] == "or" and scan.request["free"]["name"] == "Ipervenduto 1h"
    assert set(started[-1]["request"]) >= {"symbols", "min_turnover_24h", "max_symbols", "free"}

    from app.services.scan_service import list_scans

    assert [i["free"] for i in list_scans(session, user.id)[0]] == [True]
    only_values = create_scan(session, ScanCreate(connection_id=conn.id, timeframe="15m", indicators=[RSI]), user.id)
    assert only_values.config["strategy"]["rules"] == [] and only_values.strategy_name is None


def test_a_free_scan_can_borrow_the_connection_of_a_strategy(session, tenant, started):
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan

    user, conn, acc = tenant
    strategy = _strategy(session, user, acc)
    scan = create_scan(session, ScanCreate(strategy_id=strategy.id, timeframe="4h", indicators=[RSI]), user.id)
    assert scan.connection_id == conn.id and scan.strategy_id is None, "the strategy only lends its connection"
    assert [i["name"] for i in scan.config["strategy"]["charts"][0]["indicators"]] == ["rsi14"]


def test_the_coordinator_gets_only_the_fields_it_knows(session, tenant, monkeypatch):
    from app.schemas.scan import ScanCreate
    from app.services import scan_service

    user, conn, _ = tenant
    sent: list[dict] = []

    class _Response:
        def raise_for_status(self): return None

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, json): sent.append({"url": url, **json}); return _Response()

    monkeypatch.setattr(scan_service.httpx, "Client", _Client)
    monkeypatch.setattr("app.services.indicator_ref_service.inject_user_indicator_refs", lambda definition, user_id: definition)
    scan = scan_service.create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", indicators=[RSI], symbols=["aaa"]), user.id)
    assert sent[0]["url"].endswith(f"/scans/{scan.id}/run")
    assert set(sent[0]) == {"url", "strategy_config", "connection_id", "broker_type", "timeframe", "symbols", "min_turnover_24h", "max_symbols"}
    assert sent[0]["symbols"] == ["AAA"] and sent[0]["broker_type"] == "okx"


def test_free_scans_are_refused_when_they_cannot_be_read(session, tenant, stranger, started):
    from pydantic import ValidationError

    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan

    user, conn, _ = tenant
    for bad in (
        {},                                                      # neither a strategy nor indicators
        {"indicators": [RSI], "timeframe": "1h"},                # no connection
        {"connection_id": conn.id, "indicators": [RSI]},         # no timeframe
        {"connection_id": conn.id, "indicators": [], "timeframe": "1h"},
    ):
        with pytest.raises(ValidationError):
            ScanCreate(**bad)

    def refused(**kw):
        with pytest.raises(HTTPException) as err:
            create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", **kw), user.id)
        assert err.value.status_code == 422
        return str(err.value.detail)

    assert "ema50" in refused(indicators=[RSI], conditions=[{"field": "indicators.ema50", "op": ">", "value": 1}])
    assert "cannot be read in a scan" in refused(indicators=[RSI], conditions=[{"field": "position.pl", "op": ">", "value": 0}])
    assert "cannot be read in a scan" in refused(indicators=[RSI], conditions=[{"field": "price.close", "op": ">", "value": "htf.indicators.ema"}])
    assert "named 'rsi14'" in refused(indicators=[RSI, RSI])
    assert "needs a type" in refused(indicators=[{"name": "x"}])
    assert "field and an operator" in refused(indicators=[RSI], conditions=[{"field": "price.close"}])

    with pytest.raises(HTTPException) as err:
        create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", indicators=[RSI]), stranger.id)
    assert err.value.status_code == 404
