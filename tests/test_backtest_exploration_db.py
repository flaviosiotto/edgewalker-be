"""Parameter exploration of a bulk backtest (migr. 076,
docs/valutazione-esplorazione-parametri.md): the request validated and stored
at creation, the plan cap, the listing scalars, the trials, the stop proxy,
"Apri come backtest" and "Applica alla strategia".
Run: ``venv/bin/python -m pytest tests/test_backtest_exploration_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx
import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

pytest.importorskip("pgserver")


@pytest.fixture
def session(app_engine):
    with Session(app_engine, expire_on_commit=False) as s:
        yield s


@pytest.fixture
def tenant(session):
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    stale = session.exec(select(User).where(User.email == "explore@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="explore@example.com", username="user_explore", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-explore", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-explore", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _bb():
    return {"name": "bb", "type": "bbands", "params": {"timeperiod": 20, "nbdevup": 2.0}}


def _definition(*, with_sl: bool = True):
    params = {"sl_atr": 1.5} if with_sl else {}
    return {"strategy": {
        "name": "explore", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "1h", "timezone": "UTC",
        "params": params,
        "indicators": [_bb()],
        "charts": [
            {"id": "main", "symbol": "BTCUSDT", "asset": "crypto", "timeframe": "1h", "indicators": [_bb()]},
        ],
        "rules": [],
    }}


def _strategy(session, user, acc):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="EX", definition=_definition(), account_id=acc.id), user.id)


def _payload(**kw):
    from app.schemas.strategy import BacktestCreate

    base = dict(symbol="BTCUSDT", start_date=date(2020, 1, 1), end_date=date(2026, 1, 1), timeframe="1h",
                asset="crypto", mode="bulk", simulation_timeframe="1m", initial_capital=5_000.0, commission=0.001)
    base.update(kw)
    return BacktestCreate(**base)


_SPACE = {"dimensions": [
    {"chart": "main", "indicator": "bb", "param": "timeperiod", "min": 10, "max": 30, "step": 10},
    {"param": "sl_atr", "values": [1.0, 2.0]},
]}


def _wire_run(monkeypatch):
    """Capture what the coordinator would be sent; no plan check, no catalog."""
    import app.services.contract_multiplier as contract_multiplier
    import app.services.indicator_ref_service as indicator_ref_service
    from app.services import strategy_service
    from app.services.backtest_runner_service import backtest_runner_service

    calls: list[dict] = []

    def fake_bulk(**kw):
        calls.append(kw)
        return {"status": "accepted"}

    def fake_refs(definition, _user_id):
        # What the launch injects for catalog indicators: a content hash in
        # the params (a string, never an explorable number).
        import copy

        out = copy.deepcopy(definition)
        for chart in out["strategy"]["charts"]:
            for spec in chart["indicators"]:
                spec["params"]["ew_hash"] = "h-bb"
        for spec in out["strategy"]["indicators"]:
            spec["params"]["ew_hash"] = "h-bb"
        return out

    monkeypatch.setattr(backtest_runner_service, "start_bulk_backtest", fake_bulk)
    monkeypatch.setattr(strategy_service, "assert_within", lambda *a, **k: None)
    monkeypatch.setattr(contract_multiplier, "resolve_contract_multiplier", lambda *a, **k: (1.0, "default"))
    monkeypatch.setattr(indicator_ref_service, "inject_user_indicator_refs", fake_refs)
    return calls


def _add_trials(session, backtest_id: int, rows: list[tuple[int, dict]]):
    from app.models.strategy import BacktestTrial

    for index, params in rows:
        session.add(BacktestTrial(
            backtest_id=backtest_id, trial_index=index, is_reference=index == 0, params=params,
            metrics={"return_pct": float(index), "total_trades": 10 + index},
            in_sample={"return_pct": 1.0}, out_of_sample={"return_pct": -1.0}, elapsed_s=0.1,
        ))
    session.commit()


# ── Creation ─────────────────────────────────────────────────────────────────


def test_create_with_exploration_stores_the_normalised_request(session, tenant):
    from app.schemas.strategy import BacktestRead
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)

    request = bt.exploration["request"]
    assert request["dimensions"] == _SPACE["dimensions"]  # as given
    assert request["oos_fraction"] == 0.3  # defaulted
    assert request["max_trials"] == 6  # 3 x 2 within the plan cap: the planned count
    assert "state" not in bt.exploration  # the coordinator's, never the backend's
    assert BacktestRead.model_validate(bt, from_attributes=True).exploration == {"request": request}

    plain = create_backtest(session, strategy.id, _payload(), user.id)
    assert plain.exploration is None


def test_grid_beyond_the_plan_cap_is_stored_with_the_cap(session, tenant):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    big = {"oos_fraction": 0.25, "dimensions": [
        {"chart": "main", "indicator": "bb", "param": "timeperiod", "min": 2, "max": 200, "step": 1},  # 199
        {"param": "sl_atr", "min": 0.5, "max": 5.0, "step": 0.5},  # 10
    ]}
    bt = create_backtest(session, strategy.id, _payload(exploration=big), user.id)
    assert bt.exploration["request"]["max_trials"] == 1000  # default cap, the grid is 1990: not refused
    assert bt.exploration["request"]["oos_fraction"] == 0.25

    asked_less = create_backtest(session, strategy.id, _payload(exploration={**big, "max_trials": 50}), user.id)
    assert asked_less.exploration["request"]["max_trials"] == 50


def test_exploration_needs_bulk_mode(session, tenant):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    with pytest.raises(HTTPException) as exc:
        create_backtest(session, strategy.id, _payload(mode="replay", lessons_from="none", exploration=_SPACE), user.id)
    assert exc.value.status_code == 422 and "bulk" in exc.value.detail


@pytest.mark.parametrize("space, message", [
    ({"dimensions": [{"chart": "main", "indicator": "bb", "param": "nope", "values": [1, 2]}]}, "main.bb.nope"),
    ({"dimensions": [{"chart": "main", "indicator": "bb", "param": "timeperiod", "values": [10.5]}]}, "integer"),
    ({"dimensions": []}, "at least one"),
    ({"dimensions": [{"param": "sl_atr", "values": [1]}], "oos_fraction": 1.5}, "oos_fraction"),
])
def test_bad_space_is_refused_with_the_kit_message(session, tenant, space, message):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    with pytest.raises(HTTPException) as exc:
        create_backtest(session, strategy.id, _payload(exploration=space), user.id)
    assert exc.value.status_code == 422 and message in exc.value.detail


def test_definition_values_that_do_not_fit_are_refused_at_creation(session, tenant):
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    with pytest.raises(HTTPException) as exc:
        create_backtest(session, strategy.id, _payload(parameters={"definition_values": {"params.gone": 1}}), user.id)
    assert exc.value.status_code == 422 and "params.gone" in exc.value.detail


def test_explorable_parameters(session, tenant):
    from app.services.strategy_service import explorable_parameters_of

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    out = explorable_parameters_of(session, strategy.id, user.id)
    keys = {p["key"]: p for p in out["parameters"]}
    assert set(keys) == {"main.bb.timeperiod", "main.bb.nbdevup", "params.sl_atr"}
    assert keys["main.bb.timeperiod"]["value"] == 20 and keys["main.bb.timeperiod"]["integer"] is True
    assert keys["params.sl_atr"]["kind"] == "param"
    assert out["max_trials"] == 1000 and out["default_oos_fraction"] == 0.3

    with pytest.raises(HTTPException) as exc:
        explorable_parameters_of(session, strategy.id, user.id + 999_999)
    assert exc.value.status_code == 404


# ── Listing, concurrency, trials ─────────────────────────────────────────────


def test_listing_scalars_and_running_count(session, tenant):
    from app.schemas.strategy import BacktestSummary
    from app.services.entitlement_service import count_running_backtests
    from app.services.strategy_service import create_backtest, list_all_backtests

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    plain = create_backtest(session, strategy.id, _payload(), user.id)
    explored = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)

    rows, _ = list_all_backtests(session, user.id, strategy_id=strategy.id)
    by_id = {r["id"]: BacktestSummary.model_validate(r) for r in rows}
    assert not any(k.startswith("_exploration") for k in rows[0])
    assert by_id[plain.id].exploration_status is None and by_id[plain.id].exploration_trials is None
    assert by_id[explored.id].exploration_status is None and by_id[explored.id].exploration_trials == 7  # 6 + itself

    # The run is over, its trials are still running: one running backtest.
    explored.status = "completed"
    explored.exploration = {**explored.exploration, "state": {"status": "running", "done": 2, "failed": 0, "total": 4}}
    session.add(explored)
    session.commit()
    rows, _ = list_all_backtests(session, user.id, strategy_id=strategy.id)
    row = next(r for r in rows if r["id"] == explored.id)
    assert row["exploration_status"] == "running" and row["exploration_trials"] == 5
    assert count_running_backtests(session, user.id) == 1

    explored.exploration = {**explored.exploration, "state": {"status": "completed", "total": 4}}
    session.add(explored)
    session.commit()
    assert count_running_backtests(session, user.id) == 0


def test_trials_are_listed_by_index(session, tenant):
    from app.schemas.strategy import BacktestTrialRead
    from app.services.strategy_service import create_backtest, list_backtest_trials

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)
    _add_trials(session, bt.id, [(2, {"main.bb.timeperiod": 20}), (0, {"main.bb.timeperiod": 20}), (1, {"main.bb.timeperiod": 10})])

    trials = [BacktestTrialRead.model_validate(t) for t in list_backtest_trials(session, bt.id, user.id)]
    assert [t.trial_index for t in trials] == [0, 1, 2]
    assert trials[0].is_reference and not trials[1].is_reference
    assert trials[1].params == {"main.bb.timeperiod": 10} and trials[1].out_of_sample == {"return_pct": -1.0}

    with pytest.raises(HTTPException) as exc:
        list_backtest_trials(session, bt.id, user.id + 999_999)
    assert exc.value.status_code == 404


def test_trials_go_with_the_backtest(session, tenant):
    from app.models.strategy import BacktestTrial
    from app.services.strategy_service import create_backtest, delete_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)
    _add_trials(session, bt.id, [(0, {"main.bb.timeperiod": 20})])
    delete_backtest(session, bt.id, user.id)
    assert session.exec(select(BacktestTrial).where(BacktestTrial.backtest_id == bt.id)).all() == []


# ── Stop proxy ───────────────────────────────────────────────────────────────


_REAL_CLIENT = httpx.Client


def _coordinator(monkeypatch, handler):
    import app.services.backtest_runner_service as runner_module

    real_client = _REAL_CLIENT
    seen: list[str] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        return handler(request)

    monkeypatch.setattr(runner_module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(wrapped), **kw))
    return seen


def test_stop_proxies_the_coordinator_and_keeps_its_409(session, tenant, monkeypatch):
    from app.api.backtests import stop_backtest_exploration_endpoint
    from app.services.strategy_service import create_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)

    seen = _coordinator(monkeypatch, lambda r: httpx.Response(200, json={"status": "stopping", "backtest_id": bt.id}))
    assert stop_backtest_exploration_endpoint(bt.id, session=session, current_user=user) == {
        "status": "stopping", "backtest_id": bt.id}
    assert seen == [f"POST /backtests/{bt.id}/exploration/stop"]

    _coordinator(monkeypatch, lambda r: httpx.Response(409, json={"detail": f"Backtest {bt.id} has no exploration in progress"}))
    with pytest.raises(HTTPException) as exc:
        stop_backtest_exploration_endpoint(bt.id, session=session, current_user=user)
    assert exc.value.status_code == 409 and "no exploration in progress" in exc.value.detail

    _coordinator(monkeypatch, lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(HTTPException) as exc:
        stop_backtest_exploration_endpoint(bt.id, session=session, current_user=user)
    assert exc.value.status_code == 502


# ── From a trial ─────────────────────────────────────────────────────────────


def _explored_with_trial(session, user, acc):
    from app.services.strategy_service import create_backtest

    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration=_SPACE), user.id)
    bt.status = "completed"
    session.add(bt)
    session.commit()
    _add_trials(session, bt.id, [
        (0, {"main.bb.timeperiod": 20, "params.sl_atr": 1.5}),
        (3, {"main.bb.timeperiod": 30, "params.sl_atr": 2.0}),
    ])
    return strategy, bt


def test_open_trial_as_backtest(session, tenant, monkeypatch):
    from app.services import strategy_service
    from app.services.strategy_service import open_trial_as_backtest, retry_backtest

    user, conn, acc = tenant
    strategy, source = _explored_with_trial(session, user, acc)
    calls = _wire_run(monkeypatch)

    opened = open_trial_as_backtest(session, source.id, 3, user.id)

    assert opened.id != source.id and opened.status == "running"
    assert opened.mode == "bulk" and opened.exploration is None and opened.agent_id is None
    assert (opened.symbol, opened.start_date, opened.end_date, opened.timeframe, opened.simulation_timeframe) == (
        source.symbol, source.start_date, source.end_date, source.timeframe, source.simulation_timeframe)
    assert (opened.initial_capital, opened.commission, opened.source, opened.asset) == (
        source.initial_capital, source.commission, source.source, source.asset)
    assert opened.parameters["definition_values"] == {"main.bb.timeperiod": 30, "params.sl_atr": 2.0}
    assert opened.parameters["from_trial"] == {"backtest_id": source.id, "trial_index": 3}

    (call,) = calls
    sent = call["strategy_config"]["strategy"]
    assert call["backtest_id"] == opened.id and call["connection_id"] == conn.id
    for spec in (sent["charts"][0]["indicators"][0], sent["indicators"][0]):
        assert spec["params"]["timeperiod"] == 30
        assert spec["params"]["ew_hash"] == "h-bb"  # the launch injections survive the values
    assert sent["params"]["sl_atr"] == 2.0
    # The strategy itself is untouched.
    session.refresh(strategy)
    assert strategy.definition["strategy"]["params"]["sl_atr"] == 1.5

    # A retry of it runs the same values again.
    monkeypatch.setattr(strategy_service, "run_backtest", lambda sess, bid, uid=None: strategy_service.get_backtest(sess, bid, uid))
    opened.status = "completed"
    session.add(opened)
    session.commit()
    again = retry_backtest(session, opened.id, user.id)
    assert again.parameters["definition_values"] == opened.parameters["definition_values"]
    assert again.parameters["from_trial"] == opened.parameters["from_trial"]


def test_unknown_trial_is_404(session, tenant, monkeypatch):
    from app.services.strategy_service import apply_trial_to_strategy, open_trial_as_backtest

    user, _conn, acc = tenant
    _strategy_row, source = _explored_with_trial(session, user, acc)
    _wire_run(monkeypatch)
    for action in (open_trial_as_backtest, apply_trial_to_strategy):
        with pytest.raises(HTTPException) as exc:
            action(session, source.id, 42, user.id)
        assert exc.value.status_code == 404


def test_apply_trial_to_strategy(session, tenant):
    from app.services.strategy_service import apply_trial_to_strategy

    user, _conn, acc = tenant
    strategy, source = _explored_with_trial(session, user, acc)
    before = strategy.updated_at

    updated = apply_trial_to_strategy(session, source.id, 3, user.id)

    definition = updated.definition["strategy"]
    assert definition["params"]["sl_atr"] == 2.0
    assert definition["charts"][0]["indicators"][0]["params"]["timeperiod"] == 30
    assert definition["indicators"][0]["params"]["timeperiod"] == 30
    assert definition["indicators"][0]["params"]["nbdevup"] == 2.0  # the rest untouched
    assert updated.updated_at >= before


def test_after_the_strategy_changed_a_trial_still_opens_as_itself_but_no_longer_applies(session, tenant, monkeypatch):
    """The exploration ran on the definition frozen on its backtest: opening a
    trial runs THAT definition with the trial's values, whatever the strategy
    became since. Applying the values to the changed strategy is refused."""
    from app.schemas.strategy import StrategyUpdate
    from app.services.strategy_service import apply_trial_to_strategy, open_trial_as_backtest, update_strategy

    user, _conn, acc = tenant
    strategy, source = _explored_with_trial(session, user, acc)
    explored_rules = (source.config or strategy.definition)["strategy"]["rules"]
    update_strategy(session, strategy.id, StrategyUpdate(definition=_definition(with_sl=False)), user.id)
    calls = _wire_run(monkeypatch)

    opened = open_trial_as_backtest(session, source.id, 3, user.id)

    (call,) = calls
    sent = call["strategy_config"]["strategy"]
    assert call["backtest_id"] == opened.id
    assert sent["params"]["sl_atr"] == 2.0  # the parameter the strategy no longer has
    assert sent["rules"] == explored_rules
    assert sent["charts"][0]["indicators"][0]["params"]["timeperiod"] == 30

    with pytest.raises(HTTPException) as exc:
        apply_trial_to_strategy(session, source.id, 3, user.id)
    assert exc.value.status_code == 409
    assert "changed since the exploration" in exc.value.detail and "params.sl_atr" in exc.value.detail


def test_retry_keeps_the_exploration_request(session, tenant, monkeypatch):
    from app.services import strategy_service
    from app.services.strategy_service import create_backtest, retry_backtest

    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    bt = create_backtest(session, strategy.id, _payload(exploration={**_SPACE, "oos_fraction": 0.2}), user.id)
    bt.status = "completed"
    bt.exploration = {**bt.exploration, "state": {"status": "completed", "total": 6}}
    session.add(bt)
    session.commit()
    monkeypatch.setattr(strategy_service, "run_backtest", lambda sess, bid, uid=None: strategy_service.get_backtest(sess, bid, uid))

    clone = retry_backtest(session, bt.id, user.id)
    assert clone.exploration == {"request": bt.exploration["request"]}  # same request, no state


# ── Ledger in files: the page of a trial, the page of a bulk run ─────────────


def _ledger_coordinator(monkeypatch, bt_id: int, trial: int, *, present: bool = True):
    """The coordinator serving the files of trial ``trial`` (and of the run)."""
    equity = [
        {"ts": 1_700_000_000_000 + i * 3_600_000, "equity": 5_000.0 + i * 10, "position_qty": i % 2, "positions": []}
        for i in range(48)
    ]
    trades = [{
        "id": None, "backtest_id": bt_id, "strategy_id": 1, "entry_time": "2024-01-01T00:00:00+00:00",
        "exit_time": "2024-01-01T05:00:00+00:00", "direction": "long", "size": 1.0, "entry_price": 100.0,
        "exit_price": 101.0, "pnl": 1.0, "pnl_pct": 1.0, "exit_reason": "tp", "extra": {"rule": "long"},
    }]
    summary = {
        "trial_index": trial, "values": {"main.bb.timeperiod": 30, "params.sl_atr": 2.0},
        "metrics": {"return_pct": 3.0, "total_trades": 1, "sharpe_ratio": 1.2, "mode": "bulk",
                    "segments": {"in_sample": {"return_pct": 2.0}, "out_of_sample": {"return_pct": 1.0}},
                    "bulk": {"entries": 1, "agent_hold_points": [{"ts": 1}]}, "positions_summary": {"x": 1}},
        "definition": {"strategy": {"name": "trial", "params": {"sl_atr": 2.0}}},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if not present:
            return httpx.Response(404, json={"detail": "No persisted ledger"})
        path, params = request.url.path, dict(request.url.params)
        if path == f"/backtests/{bt_id}/ledger":
            tables = params.get("tables", "orders,fills,equity,trades").split(",") if "tables" in params else ["orders", "fills", "equity", "trades"]
            out = {"summary": summary}
            if "orders" in tables:
                out["orders"] = [{"id": "o1", "status": "filled"}, {"id": "o2", "status": "submitted"}]
            if "equity" in tables:
                out["equity_snapshots"] = equity
            if "trades" in tables:
                out["trades"] = trades
            return httpx.Response(200, json=out)
        if path == f"/backtests/{bt_id}/ohlc-history":
            return httpx.Response(200, json={"candles": [], "indicators": {}, "trial": params.get("trial")})
        return httpx.Response(404)

    return _coordinator(monkeypatch, handler)


def test_the_page_of_a_trial_reads_its_files(session, tenant, monkeypatch):
    from app.api.backtests import (
        get_backtest_endpoint,
        get_backtest_ohlc_history_endpoint,
        get_backtest_runtime_equity_endpoint,
        get_backtest_runtime_orders_endpoint,
        list_trades_endpoint,
    )
    from app.schemas.strategy import TradeRead

    user, _conn, acc = tenant
    _strategy_row, bt = _explored_with_trial(session, user, acc)
    seen = _ledger_coordinator(monkeypatch, bt.id, 3)

    read = get_backtest_endpoint(bt.id, trial=3, session=session, current_user=user)
    assert read.id == bt.id and read.trial == {"index": 3, "params": {"main.bb.timeperiod": 30, "params.sl_atr": 2.0}, "is_reference": False}
    assert read.return_pct == 3.0 and read.total_trades == 1 and read.sharpe_ratio == 1.2
    assert read.metrics["segments"]["out_of_sample"] == {"return_pct": 1.0}
    assert read.metrics["bulk"] == {"entries": 1}  # per-point holds stay out
    assert "positions_summary" not in read.metrics
    assert read.config == {"strategy": {"name": "trial", "params": {"sl_atr": 2.0}}}
    assert read.exploration is None and read.analysis["trades"]["trades"] == 1 and read.analysis["drawdowns"] is not None

    orders = get_backtest_runtime_orders_endpoint(bt.id, active_only=True, trial=3, session=session, current_user=user)
    assert orders == {"orders": [{"id": "o2", "status": "submitted"}], "source": "ledger"}
    equity = get_backtest_runtime_equity_endpoint(bt.id, trial=3, session=session, current_user=user)
    assert len(equity["equity"]) == 48 and equity["source"] == "ledger"
    trades = list_trades_endpoint(bt.id, trial=3, session=session, current_user=user)
    assert [TradeRead.model_validate(t).pnl for t in trades] == [1.0]
    chart = get_backtest_ohlc_history_endpoint(bt.id, chart_id=None, trial=3, session=session, current_user=user)
    assert chart["trial"] == "3"
    assert all(path.startswith(f"GET /backtests/{bt.id}/") for path in seen)

    with pytest.raises(HTTPException) as exc:
        get_backtest_endpoint(bt.id, trial=7, session=session, current_user=user)
    assert exc.value.status_code == 404  # not a trial of this exploration
    _ledger_coordinator(monkeypatch, bt.id, 3, present=False)
    with pytest.raises(HTTPException) as exc:
        get_backtest_endpoint(bt.id, trial=3, session=session, current_user=user)
    assert exc.value.status_code == 404 and "prova 3" in exc.value.detail


def test_a_bulk_run_with_its_ledger_in_files(session, tenant, monkeypatch):
    from app.api.backtests import (
        get_backtest_endpoint,
        get_backtest_runtime_equity_endpoint,
        get_backtest_runtime_orders_endpoint,
    )

    user, _conn, acc = tenant
    _strategy_row, bt = _explored_with_trial(session, user, acc)
    bt.metrics = {"return_pct": 1.5, "ledger_store": "file", "segments": {"in_sample": {}, "out_of_sample": {}}}
    session.add(bt)
    session.commit()
    seen = _ledger_coordinator(monkeypatch, bt.id, 0)

    read = get_backtest_endpoint(bt.id, trial=None, session=session, current_user=user)
    assert read.trial is None and read.metrics["ledger_store"] == "file" and "segments" in read.metrics
    assert read.analysis["drawdowns"] is not None  # equity statistics from the files
    assert get_backtest_runtime_orders_endpoint(bt.id, active_only=False, trial=None, session=session, current_user=user)["orders"][0]["id"] == "o1"
    assert len(get_backtest_runtime_equity_endpoint(bt.id, trial=None, session=session, current_user=user)["equity"]) == 48
    assert f"GET /backtests/{bt.id}/ledger" in seen

    # files removed: the page still opens, the ledger endpoints say so
    _ledger_coordinator(monkeypatch, bt.id, 0, present=False)
    assert get_backtest_endpoint(bt.id, trial=None, session=session, current_user=user).analysis is None
    with pytest.raises(HTTPException) as exc:
        get_backtest_runtime_equity_endpoint(bt.id, trial=None, session=session, current_user=user)
    assert exc.value.status_code == 404
