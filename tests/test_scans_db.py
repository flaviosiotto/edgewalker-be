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
    assert scan.request == {"symbols": ["BTC-USDT-SWAP"], "min_turnover_24h": 1e6, "max_symbols": 300, "lookback": 20}
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
    assert set(sent[0]) == {"url", "strategy_config", "connection_id", "broker_type", "timeframe", "symbols", "min_turnover_24h", "max_symbols", "lookback"}
    assert sent[0]["lookback"] == 20
    assert sent[0]["symbols"] == ["AAA"] and sent[0]["broker_type"] == "okx"
    # asset types narrow the market: sent only when no symbols are given
    scan_service.create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", indicators=[RSI], symbols=["aaa"], asset_types=["Forex"]), user.id)
    assert "asset_types" not in sent[1]
    scan = scan_service.create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", indicators=[RSI], asset_types=["Index", " forex "]), user.id)
    assert sent[2]["asset_types"] == ["forex", "index"] and scan.request["asset_types"] == ["forex", "index"]


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


def test_a_rule_that_fired_and_went_off_is_found_by_how_long_ago(session, tenant, started):
    from pydantic import ValidationError

    from app.models.scan import ScanRow
    from app.schemas.scan import ScanCreate, ScanRowRead
    from app.services.scan_service import create_scan, list_scan_rows

    user, _, acc = tenant
    strategy = _strategy(session, user, acc)
    scan = create_scan(session, ScanCreate(strategy_id=strategy.id, lookback=10), user.id)
    assert scan.request["lookback"] == 10 and started[-1]["request"]["lookback"] == 10
    with pytest.raises(ValidationError):
        ScanCreate(strategy_id=strategy.id, lookback=0)

    def rule(name, last_true, true_for):
        return {"name": name, "matched": last_true == 0, "last_true": last_true, "true_for": true_for,
                "true_bars": 1 if last_true is not None else 0, "lookback": 10, "conditions": []}

    for symbol, rules in [
        ("NOW", [rule("cross", 0, 1), rule("state", None, 0)]),
        ("AGO3", [rule("cross", 3, 0), rule("state", 7, 0)]),
        ("AGO8", [rule("cross", 8, 0), rule("state", 0, 4)]),
        ("NEVER", [rule("cross", None, 0), rule("state", None, 0)]),
        ("OLD", [{"name": "cross", "matched": True, "conditions": []}]),  # a scan of the last bar only
    ]:
        matched = sum(1 for r in rules if r["matched"])
        session.add(ScanRow(scan_id=scan.id, symbol=symbol, rules=rules, matched_rules=matched))
    session.commit()

    def symbols(**kwargs):
        return [r.symbol for r in list_scan_rows(session, scan.id, user.id, **kwargs)[1]]

    assert symbols(rule="cross") == ["NOW", "OLD"], "without a window: true on the last bar"
    assert symbols(rule="cross", within=5, sort="last_true") == ["NOW", "OLD", "AGO3"]
    assert symbols(rule="cross", within=10, sort="last_true") == ["NOW", "OLD", "AGO3", "AGO8"]
    assert symbols(rule="cross", within=1) == ["NOW", "OLD"]
    assert symbols(matched_only=True, within=5, sort="last_true") == ["AGO8", "NOW", "OLD", "AGO3"], "any rule, most recent"
    assert symbols(sort="last_true")[-1] == "NEVER", "never true comes last"
    assert symbols(within=3) == ["AGO3", "AGO8", "NEVER", "NOW", "OLD"], "a window alone filters nothing"

    rows = list_scan_rows(session, scan.id, user.id)[1]
    read = {r.symbol: ScanRowRead.model_validate(r, from_attributes=True).lookback for r in rows}
    assert read["NOW"] == 10 and read["OLD"] is None


def test_params_can_be_changed_for_one_scan_without_touching_the_strategy(session, tenant, started):
    from fastapi import HTTPException

    from app.api.scans import _read
    from app.schemas.scan import ScanCreate
    from app.schemas.strategy import StrategyCreate
    from app.services.scan_service import create_scan
    from app.services.strategy_service import create_strategy, get_strategy

    user, conn, acc = tenant
    ema = {"name": "ema20", "type": "ema", "params": {"timeperiod": 20}}
    definition = {"strategy": {
        "name": "p", "symbol": "BTC-USDT-SWAP", "asset": "crypto_futures", "timeframe": "1h", "timezone": "UTC",
        "indicators": [ema], "params": {"rsi_entry": 30, "max_bars": 12},
        "charts": [{"id": "main", "symbol": "BTC-USDT-SWAP", "asset": "crypto_futures", "timeframe": "1h", "indicators": [ema]}],
        "rules": [{"name": "in", "action": "buy", "conditions": [{"field": "price.close", "op": ">", "value": "params.rsi_entry"}]}],
    }}
    strategy = create_strategy(session, StrategyCreate(name="with params", definition=definition, account_id=acc.id), user.id)

    plain = create_scan(session, ScanCreate(strategy_id=strategy.id), user.id)
    assert _read(plain).params == {"rsi_entry": 30.0, "max_bars": 12.0} and "params" not in plain.request

    scan = create_scan(session, ScanCreate(strategy_id=strategy.id, params={"rsi_entry": 25, "max_bars": 12}), user.id)
    assert scan.request["params"] == {"rsi_entry": 25.0}, "only what differs from the strategy"
    assert _read(scan).params == {"rsi_entry": 25.0, "max_bars": 12.0}
    assert started[-1]["config"]["strategy"]["params"]["rsi_entry"] == 25.0, "the coordinator reads the scan's snapshot"
    session.expire_all()
    assert get_strategy(session, strategy.id, user.id).definition["strategy"]["params"]["rsi_entry"] == 30

    with pytest.raises(HTTPException) as unknown:
        create_scan(session, ScanCreate(strategy_id=strategy.id, params={"nope": 1}), user.id)
    assert unknown.value.status_code == 422 and "rsi_entry" in unknown.value.detail
    with pytest.raises(HTTPException) as free:
        create_scan(session, ScanCreate(connection_id=conn.id, timeframe="1h", indicators=[RSI], params={"x": 1}), user.id)
    assert free.value.status_code == 422


def test_an_exit_rule_does_not_make_a_row_recent(session, tenant, started):
    from app.models.scan import ScanRow
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan, list_scan_rows

    user, _, acc = tenant
    scan = create_scan(session, ScanCreate(strategy_id=_strategy(session, user, acc).id), user.id)
    session.add(ScanRow(scan_id=scan.id, symbol="OUT", matched_rules=0, rules=[
        {"name": "in", "role": "entry", "matched": False, "last_true": None},
        {"name": "out", "role": "exit", "matched": True, "last_true": 0},
    ]))
    session.commit()
    assert list_scan_rows(session, scan.id, user.id, matched_only=True, within=20)[1] == []
    assert [r.symbol for r in list_scan_rows(session, scan.id, user.id, rule="out", within=20)[1]] == ["OUT"], "asked by name, it is read"


# ---------------------------------------------------------------------------
# Templates as the source; a strategy scanned as the template it would make
# ---------------------------------------------------------------------------

def _multi_chart_definition(*, own="BTC-USDT-SWAP", market="ETH-USDT-SWAP"):
    ema = {"name": "ema20", "type": "ema", "params": {"timeperiod": 20}}
    chart = lambda cid, symbol, tf: {"id": cid, "symbol": symbol, "asset": "crypto_futures", "timeframe": tf, "indicators": [ema]}  # noqa: E731
    return {"strategy": {
        "name": "mtf", "symbol": own, "asset": "crypto_futures", "timeframe": "15m", "timezone": "UTC",
        "indicators": [ema], "params": {"k": 2},
        "charts": [chart("main", own, "15m"), chart("htf", own, "1h"), chart("mkt", market, "1h")],
        "rules": [{"name": "in", "action": "buy", "conditions": [
            {"field": "price.close", "op": ">", "value": "htf.indicators.ema20"},
            {"field": "mkt.price.close", "op": ">", "value": "mkt.indicators.ema20"},
        ]}],
    }}


def _template(session, user_id, definition, name="tpl scan"):
    from app.models.strategy_template import StrategyTemplate
    from app.services.strategy_template_service import sanitize

    clean, meta, _ = sanitize(definition)
    row = StrategyTemplate(user_id=user_id, name=name, definition=clean, charts_meta=[m.model_dump() for m in meta])
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def test_a_template_is_scanned_on_a_connection_with_its_slots(session, tenant, stranger, started):
    from fastapi import HTTPException
    from pydantic import ValidationError

    from app.api.scans import _read
    from app.schemas.scan import ScanCreate
    from app.services.scan_service import create_scan, list_scans

    user, conn, _ = tenant
    template = _template(session, user.id, _multi_chart_definition())

    scan = create_scan(session, ScanCreate(template_id=template.id, connection_id=conn.id, charts={"mkt": "btc-usdt-swap"}, params={"k": 3}), user.id)
    charts = {c["id"]: c for c in scan.config["strategy"]["charts"]}
    assert scan.strategy_id is None and scan.strategy_name == "tpl scan" and scan.timeframe == "15m"
    assert charts["main"]["symbol"] == "" and charts["htf"]["symbol"] == "", "they follow the scanned symbol"
    assert charts["mkt"]["symbol"] == "BTC-USDT-SWAP", "a slot put on a market of its own"
    assert scan.request["template"] == {"id": template.id, "name": "tpl scan"}
    assert scan.request["charts"] == {"mkt": "BTC-USDT-SWAP"} and scan.request["params"] == {"k": 3.0}
    read = _read(scan)
    assert (read.source, read.template_id, read.free) == ("template", template.id, False)
    listed = {row["id"]: row for row in list_scans(session, user.id)[0]}
    assert listed[scan.id]["source"] == "template" and listed[scan.id]["template_id"] == template.id

    every_slot_follows = create_scan(session, ScanCreate(template_id=template.id, connection_id=conn.id), user.id)
    assert all(c["symbol"] == "" for c in every_slot_follows.config["strategy"]["charts"])

    with pytest.raises(ValidationError):
        ScanCreate(template_id=template.id)  # no connection
    with pytest.raises(ValidationError):
        ScanCreate(template_id=template.id, connection_id=conn.id, strategy_id=1)
    for bad in ({"charts": {"main": "ETH-USDT-SWAP"}}, {"charts": {"nope": "X"}}, {"timeframe": "4h"}):
        with pytest.raises(HTTPException) as refused:
            create_scan(session, ScanCreate(template_id=template.id, connection_id=conn.id, **bad), user.id)
        assert refused.value.status_code == 422
    with pytest.raises(HTTPException) as foreign:
        create_scan(session, ScanCreate(template_id=template.id, connection_id=conn.id), stranger.id)
    assert foreign.value.status_code == 404


def test_a_strategy_is_scanned_as_the_template_it_would_make(session, tenant, started):
    from app.api.scans import _read
    from app.schemas.scan import ScanCreate
    from app.schemas.strategy import StrategyCreate
    from app.services.scan_service import create_scan
    from app.services.strategy_service import create_strategy, get_strategy

    user, conn, acc = tenant
    strategy = create_strategy(session, StrategyCreate(name="mtf strategy", definition=_multi_chart_definition(), account_id=acc.id), user.id)
    scan = create_scan(session, ScanCreate(strategy_id=strategy.id), user.id)
    charts = {c["id"]: c for c in scan.config["strategy"]["charts"]}
    assert charts["main"]["symbol"] == "" and charts["htf"]["symbol"] == ""
    assert charts["mkt"]["symbol"] == "ETH-USDT-SWAP", "a chart on another market stays on it"
    assert scan.strategy_id == strategy.id and scan.connection_id == conn.id and scan.request["charts"] == {"mkt": "ETH-USDT-SWAP"}
    assert _read(scan).source == "strategy" and _read(scan).params == {"k": 2.0}
    session.expire_all()
    kept = get_strategy(session, strategy.id, user.id).definition["strategy"]["charts"]
    assert [c["symbol"] for c in kept] == ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "ETH-USDT-SWAP"], "the strategy is not touched"

    follows = create_scan(session, ScanCreate(strategy_id=strategy.id, charts={"mkt": None}), user.id)
    assert {c["id"]: c["symbol"] for c in follows.config["strategy"]["charts"]}["mkt"] == ""
