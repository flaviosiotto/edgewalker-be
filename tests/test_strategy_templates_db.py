"""Strategy templates — service flows against an embedded Postgres.

Covers: official file sync (upsert/remove), save from a strategy, preview,
instantiate on an account (strategy + lessons + symbol warnings), ownership
and edit rules. Run: ``venv/bin/python -m pytest tests/test_strategy_templates_db.py``.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

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
    """Two users, each with a connection + configured account."""
    from app.models.connection import Account, Connection
    from app.models.user import User

    now = datetime.now(timezone.utc)
    out = {}
    for tag in ("a", "b"):
        user = User(email=f"{tag}@example.com", username=f"user_{tag}", hashed_password="x")
        session.add(user)
        session.flush()
        conn = Connection(user_id=user.id, name=f"conn-{tag}", broker_type="binance", status="connected",
                          created_at=now, updated_at=now)
        session.add(conn)
        session.flush()
        acc = Account(connection_id=conn.id, account_id=f"acc-{tag}", currency="USDT", created_at=now, updated_at=now)
        session.add(acc)
        session.flush()
        out[tag] = (user, conn, acc)
    session.commit()
    yield out
    # Cleanup in FK order (strategies/templates cascade from user).
    for user, _, _ in out.values():
        session.delete(user)
    session.commit()


def _definition(symbol="BTCUSDT"):
    return {"strategy": {
        "name": "src", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [{"name": "ema20", "type": "EMA", "params": {"timeperiod": 20, "ew_hash": "abc"}}],
        "charts": [
            {"id": "main", "symbol": symbol, "asset": "crypto", "timeframe": "5m",
             "indicators": [{"name": "ema20", "type": "EMA", "params": {"timeperiod": 20, "ew_hash": "abc"}}]},
            {"id": "ctx", "symbol": symbol, "asset": "crypto", "timeframe": "1h", "history_depth_days": 60, "indicators": []},
        ],
        "rules": [{"name": "r1", "action": "buy", "chart_id": "main", "conditions": [{"field": "price.close", "op": "greater_than", "value": "ema20"}]}],
    }}


def _make_strategy(session, user, acc, definition):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="Sorgente", definition=definition, account_id=acc.id), user.id)


def test_sync_official_templates_upserts_and_removes(session, tmp_path):
    from app.models.strategy_template import StrategyTemplate
    from app.services.strategy_template_service import SYSTEM_TEMPLATES_DIR, sync_system_templates

    # Real files from the repo.
    n = sync_system_templates(session, SYSTEM_TEMPLATES_DIR)
    assert n >= 3
    rows = {r.key: r for r in session.exec(select(StrategyTemplate).where(StrategyTemplate.user_id == None)).all()}  # noqa: E711
    assert "ema-crossover-trend-filter" in rows
    first_id = rows["ema-crossover-trend-filter"].id

    # A changed subset: one file edited, the others gone → updated + removed.
    src = json.loads((SYSTEM_TEMPLATES_DIR / "ema-crossover-trend-filter.json").read_text())
    src["name"] = "Rinominato"
    (tmp_path / "ema-crossover-trend-filter.json").write_text(json.dumps(src), encoding="utf-8")
    assert sync_system_templates(session, tmp_path) == 1
    rows = {r.key: r for r in session.exec(select(StrategyTemplate).where(StrategyTemplate.user_id == None)).all()}  # noqa: E711
    assert list(rows) == ["ema-crossover-trend-filter"]
    assert rows["ema-crossover-trend-filter"].id == first_id and rows["ema-crossover-trend-filter"].name == "Rinominato"

    # A broken file never blocks the startup: logged, nothing changes.
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert sync_system_templates(session, tmp_path) == 0
    assert session.exec(select(StrategyTemplate).where(StrategyTemplate.key == "ema-crossover-trend-filter")).first() is not None

    # Restore the repo set for the other tests.
    sync_system_templates(session, SYSTEM_TEMPLATES_DIR)


def test_save_preview_instantiate_round_trip(session, tenant, monkeypatch):
    from app.models.agent_lesson import AgentLesson
    from app.models.strategy import Strategy
    from app.schemas.strategy_template import (
        StrategyTemplateCreate,
        StrategyTemplateInstantiate,
        StrategyTemplateSource,
        TemplateChartBinding,
    )
    from app.services import strategy_template_service as svc

    # Catalogue offline in tests: classification degrades gracefully.
    monkeypatch.setattr(svc, "classify_indicator_types", lambda keys: (set(), set(keys), False))

    user_a, conn_a, acc_a = tenant["a"]
    user_b, conn_b, acc_b = tenant["b"]
    strategy = _make_strategy(session, user_a, acc_a, _definition())
    session.add(AgentLesson(strategy_id=strategy.id, user_id=user_a.id, lesson="Non operare in apertura", confidence=0.7))
    session.add(AgentLesson(strategy_id=strategy.id, user_id=user_a.id, lesson="Vecchia", status="retired"))
    session.commit()

    payload = StrategyTemplateCreate(
        name="Il mio template", description="desc", tags=["Trend", "trend", "x"],
        source=StrategyTemplateSource(strategy_id=strategy.id), chart_labels={"main": "esecuzione"},
    )
    preview = svc.preview_template(session, payload, user_a.id)
    assert [m.id for m in preview.charts_meta] == ["main", "ctx"] and preview.rules_count == 1
    assert [l.lesson for l in preview.lessons] == ["Non operare in apertura"]  # retired ones excluded
    assert any("non verificati" in w for w in preview.warnings)

    tpl = svc.create_template(session, payload, user_a.id)
    assert tpl.user_id == user_a.id and tpl.tags == ["trend", "x"]
    assert "symbol" not in tpl.definition["strategy"] and "symbol" not in tpl.definition["strategy"]["charts"][0]
    assert tpl.definition["strategy"]["charts"][1]["timeframe"] == "1h"
    assert tpl.charts_meta[0]["label"] == "esecuzione"
    assert tpl.origin["strategy_id"] == strategy.id

    # Same name twice → 409.
    with pytest.raises(HTTPException) as exc:
        svc.create_template(session, payload, user_a.id)
    assert exc.value.status_code == 409

    # Other user: cannot see it, cannot source someone else's strategy.
    with pytest.raises(HTTPException) as exc:
        svc.get_template(session, tpl.id, user_b.id)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException):
        svc.create_template(session, payload, user_b.id)

    # Instantiate on user A's account with new markets and a changed context timeframe.
    inst = StrategyTemplateInstantiate(
        account_id=acc_a.id,
        charts={"main": TemplateChartBinding(symbol="ethusdt", asset_type="crypto"),
                "ctx": TemplateChartBinding(symbol="ETHUSDT", asset_type="crypto", timeframe="4h")},
    )
    created, warnings = svc.instantiate_template(session, tpl.id, inst, user_a.id)
    assert created.user_id == user_a.id and created.account_id == acc_a.id and created.connection_id == conn_a.id
    assert created.name == "Il mio template"
    body = created.definition["strategy"]
    assert body["symbol"] == "ETHUSDT" and body["timeframe"] == "5m" and body["name"] == "Il mio template"
    assert body["charts"][0]["symbol"] == "ETHUSDT" and body["charts"][1]["timeframe"] == "4h"
    assert "ew_hash" not in body["charts"][0]["indicators"][0]["params"]
    lessons = session.exec(select(AgentLesson).where(AgentLesson.strategy_id == created.id)).all()
    assert [l.lesson for l in lessons] == ["Non operare in apertura"] and lessons[0].source == "template"
    # No symbol cache on this connection → generic "verify" warning.
    assert any("non verificati" in w for w in warnings)

    # Second instantiation without a name → "Il mio template 2"; explicit taken name → 409.
    created2, _ = svc.instantiate_template(session, tpl.id, inst, user_a.id)
    assert created2.name == "Il mio template 2"
    with pytest.raises(HTTPException) as exc:
        svc.instantiate_template(session, tpl.id, inst.model_copy(update={"name": "Il mio template"}), user_a.id)
    assert exc.value.status_code == 409

    # Missing slot → 400 naming the chart.
    with pytest.raises(HTTPException) as exc:
        svc.instantiate_template(session, tpl.id, inst.model_copy(update={"charts": {"main": inst.charts["main"]}}), user_a.id)
    assert exc.value.status_code == 400 and "ctx" in exc.value.detail

    # Foreign account → 404 (owner check on the account).
    with pytest.raises(HTTPException):
        svc.instantiate_template(session, tpl.id, inst.model_copy(update={"account_id": acc_b.id}), user_a.id)

    # Update + delete only by the owner; official templates are read-only.
    from app.schemas.strategy_template import StrategyTemplateUpdate

    updated = svc.update_template(session, tpl.id, StrategyTemplateUpdate(name="Nuovo nome", chart_labels={"ctx": "contesto"}), user_a.id)
    assert updated.name == "Nuovo nome" and updated.charts_meta[1]["label"] == "contesto"
    with pytest.raises(HTTPException) as exc:
        svc.update_template(session, tpl.id, StrategyTemplateUpdate(name="x"), user_b.id)
    assert exc.value.status_code == 404

    official = [t for t in svc.list_templates(session, user_a.id, scope="official")]
    assert official, "official templates synced by the previous test"
    with pytest.raises(HTTPException) as exc:
        svc.update_template(session, official[0].id, StrategyTemplateUpdate(name="x"), user_a.id)
    assert exc.value.status_code == 403
    with pytest.raises(HTTPException) as exc:
        svc.delete_template(session, official[0].id, user_a.id)
    assert exc.value.status_code == 403

    # Official template instantiates for anyone.
    ema = next(t for t in official if t.key == "ema-crossover-trend-filter")
    created3, _ = svc.instantiate_template(
        session, ema.id,
        StrategyTemplateInstantiate(account_id=acc_b.id, charts={
            "main": TemplateChartBinding(symbol="BTCUSDT", asset_type="crypto"),
            "context": TemplateChartBinding(symbol="BTCUSDT", asset_type="crypto"),
        }),
        user_b.id,
    )
    assert created3.user_id == user_b.id and created3.definition["strategy"]["charts"][1]["timeframe"] == "1h"
    assert session.exec(select(AgentLesson).where(AgentLesson.strategy_id == created3.id)).all()

    # Listing scopes.
    assert {t.id for t in svc.list_templates(session, user_a.id, scope="mine")} == {tpl.id}
    assert svc.list_templates(session, user_b.id, scope="mine") == []
    all_a = svc.list_templates(session, user_a.id, scope="all")
    assert tpl.id in {t.id for t in all_a} and all_a[0].user_id == user_a.id  # mine first

    svc.delete_template(session, tpl.id, user_a.id)
    with pytest.raises(HTTPException):
        svc.get_template(session, tpl.id, user_a.id)
    # Instantiated strategies survive the template deletion.
    assert session.get(Strategy, created.id) is not None


def test_template_from_live_session_uses_frozen_definition(session, tenant, monkeypatch):
    """A live session runs a frozen copy of the design: the template starts
    from that snapshot (what the user watched), lessons come from the strategy."""
    from app.models.agent_lesson import AgentLesson
    from app.models.strategy import StrategyLive
    from app.schemas.strategy_template import StrategyTemplateCreate, StrategyTemplateSource
    from app.services import strategy_template_service as svc

    monkeypatch.setattr(svc, "classify_indicator_types", lambda keys: (set(), set(keys), False))
    user_a, conn_a, acc_a = tenant["a"]
    user_b, _, _ = tenant["b"]
    strategy = _make_strategy(session, user_a, acc_a, _definition())
    # The design moved on after the launch: the snapshot still has one rule, the design two.
    frozen = _definition(symbol="ETHUSDT")
    live = StrategyLive(strategy_id=strategy.id, status="running", symbol="ETHUSDT", timeframe="5m",
                        account_id=acc_a.id, connection_id=conn_a.id, definition=frozen)
    session.add(live)
    design = copy.deepcopy(strategy.definition)
    design["strategy"]["rules"].append({"name": "r2", "action": "sell", "chart_id": "main", "conditions": []})
    strategy.definition = design
    session.add(strategy)
    session.add(AgentLesson(strategy_id=strategy.id, user_id=user_a.id, lesson="Dal live", confidence=0.5))
    session.commit()

    payload = StrategyTemplateCreate(name="Dal live", source=StrategyTemplateSource(live_id=live.id))
    preview = svc.preview_template(session, payload, user_a.id)
    assert preview.rules_count == 1 and [l.lesson for l in preview.lessons] == ["Dal live"]

    tpl = svc.create_template(session, payload, user_a.id)
    assert tpl.origin["live_id"] == live.id and tpl.origin["strategy_id"] == strategy.id
    assert len(tpl.definition["strategy"]["rules"]) == 1
    assert "symbol" not in tpl.definition["strategy"]["charts"][0]

    # Foreign live / unknown live → 404.
    with pytest.raises(HTTPException) as exc:
        svc.preview_template(session, payload, user_b.id)
    assert exc.value.status_code == 404
    with pytest.raises(HTTPException) as exc:
        svc.preview_template(session, StrategyTemplateCreate(name="x", source=StrategyTemplateSource(live_id=10**9)), user_a.id)
    assert exc.value.status_code == 404

    # A session without a snapshot falls back to the design.
    live.definition = None
    session.add(live)
    session.commit()
    assert svc.preview_template(session, payload, user_a.id).rules_count == 2
