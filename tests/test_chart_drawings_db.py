"""Disegni sul chart con autore e gruppo (docs/valutazione-disegni-agente.md F1):
fusione per autore nel PUT del trader, PATCH incrementale (cancella per id,
gruppo, autore + upsert), conservazione dei disegni dell'agente nell'update
integrale della definition, tetti. Run: ``venv/bin/python -m pytest tests/test_chart_drawings_db.py``.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from fastapi import HTTPException
from sqlmodel import Session, select

from app.schemas.strategy import ChartDrawing, ChartDrawingsPatch, ChartDrawingsUpdate, MAX_AGENT_DRAWINGS_PER_CHART
from app.services.strategy_service import merge_chart_drawings_patch, merge_chart_drawings_put, preserve_agent_drawings


def _d(id_: str, author: str = "user", group: str | None = None, p: float = 100.0) -> dict:
    return {"id": id_, "type": "horizontal_line", "anchors": [{"t": 1_700_000_000, "p": p}],
            "style": {"color": "#2962ff", "width": 2.0, "dash": "solid"}, "props": {}, "locked": False,
            "author": author, **({"group": group} if group else {})}


def _model(id_: str, author: str = "agent", group: str | None = "similarity", p: float = 100.0) -> ChartDrawing:
    return ChartDrawing.model_validate(_d(id_, author, group, p))


# ── funzioni pure ──

def test_put_replaces_user_drawings_and_keeps_agent_ones():
    current = [_d("u1"), _d("a1", "agent", "levels")]
    merged = merge_chart_drawings_put(current, [_d("u2")])
    assert [d["id"] for d in merged] == ["u2", "a1"]
    # un disegno dell'agente incluso nel payload viene aggiornato, non duplicato
    merged = merge_chart_drawings_put(current, [_d("a1", "agent", "levels", p=101.0)])
    assert [d["id"] for d in merged] == ["a1"] and merged[0]["anchors"][0]["p"] == 101.0


def test_patch_deletes_then_upserts_in_order():
    current = [_d("u1"), _d("a1", "agent", "similarity"), _d("a2", "agent", "levels")]
    patch = ChartDrawingsPatch(delete_group="similarity", upsert=[_model("a3"), _model("u1", "user", None, p=99.0)])
    merged = merge_chart_drawings_patch(current, patch)
    ids = [d["id"] for d in merged]
    assert ids == ["u1", "a2", "a3"]
    u1 = next(d for d in merged if d["id"] == "u1")
    assert u1["anchors"][0]["p"] == 99.0 and u1["updated_at"] and u1["created_at"]
    a3 = next(d for d in merged if d["id"] == "a3")
    assert a3["author"] == "agent" and a3["group"] == "similarity"
    # cancellazione per autore e per id
    merged = merge_chart_drawings_patch(merged, ChartDrawingsPatch(delete_author="agent"))
    assert [d["id"] for d in merged] == ["u1"]
    merged = merge_chart_drawings_patch(merged, ChartDrawingsPatch(delete_ids=["u1"]))
    assert merged == []


def test_agent_cap_is_enforced():
    current = [_d(f"a{i}", "agent", "g") for i in range(MAX_AGENT_DRAWINGS_PER_CHART)]
    with pytest.raises(HTTPException) as exc:
        merge_chart_drawings_patch(current, ChartDrawingsPatch(upsert=[_model("a-extra")]))
    assert exc.value.status_code == 422 and "agente" in exc.value.detail
    # sostituendo il gruppo si rientra nel tetto
    merged = merge_chart_drawings_patch(current, ChartDrawingsPatch(delete_group="g", upsert=[_model("a-extra")]))
    assert [d["id"] for d in merged] == ["a-extra"]


def test_preserve_agent_drawings_on_full_definition_update():
    old = {"strategy": {"charts": [{"id": "main", "drawings": [_d("u1"), _d("a1", "agent", "levels")]},
                                   {"id": "chart-x", "drawings": [_d("a2", "agent")]}]}}
    new = {"strategy": {"charts": [{"id": "main", "drawings": [_d("u9")]}, {"id": "chart-x"}]}}
    out = preserve_agent_drawings(old, new)
    assert [d["id"] for d in out["strategy"]["charts"][0]["drawings"]] == ["u9", "a1"]
    assert [d["id"] for d in out["strategy"]["charts"][1]["drawings"]] == ["a2"]
    # senza disegni dell'agente la definition passa invariata
    assert preserve_agent_drawings({"strategy": {"charts": [{"id": "main"}]}}, {"strategy": {"charts": []}}) == {"strategy": {"charts": []}}


# ── con DB ──

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
    stale = session.exec(select(User).where(User.email == "draw@example.com")).first()
    if stale is not None:
        session.delete(stale)
        session.commit()
    user = User(email="draw@example.com", username="user_draw", hashed_password="x")
    session.add(user)
    session.flush()
    conn = Connection(user_id=user.id, name="conn-draw", broker_type="binance", status="connected",
                      created_at=now, updated_at=now)
    session.add(conn)
    session.flush()
    acc = Account(connection_id=conn.id, account_id="acc-draw", currency="USDT", created_at=now, updated_at=now)
    session.add(acc)
    session.commit()
    yield user, conn, acc
    session.delete(user)
    session.commit()


def _definition(symbol: str = "BTCUSDT"):
    return {"strategy": {
        "name": "draw", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "timezone": "UTC", "params": {},
        "indicators": [], "charts": [{"id": "main", "symbol": symbol, "asset": "crypto", "timeframe": "5m", "indicators": []}],
        "rules": [],
    }}


def _strategy(session, user, acc):
    from app.schemas.strategy import StrategyCreate
    from app.services.strategy_service import create_strategy

    return create_strategy(session, StrategyCreate(name="DR", definition=_definition(), account_id=acc.id), user.id)


def test_strategy_patch_put_get_and_definition_update(session, tenant, monkeypatch):
    from app.schemas.strategy import StrategyUpdate
    from app.services import strategy_service as svc

    published: list[tuple] = []
    monkeypatch.setattr(svc, "publish_drawings_changed", lambda *a: published.append(a))
    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)

    # l'agente traccia un gruppo
    out = svc.patch_strategy_chart_drawings(session, strategy.id, "main", ChartDrawingsPatch(upsert=[_model("a1"), _model("a2")]), user.id)
    assert [d["id"] for d in out] == ["a1", "a2"] and published[-1] == (user.id, "strategy", strategy.id, "main")
    # il trader salva i suoi disegni con il PUT integrale: quelli dell'agente restano
    out = svc.update_strategy_chart_drawings(session, strategy.id, "main", ChartDrawingsUpdate(drawings=[_model("u1", "user", None)]), user.id)
    assert [d["id"] for d in out] == ["u1", "a1", "a2"]
    assert [d["id"] for d in svc.read_strategy_chart_drawings(session, strategy.id, "main", user.id)] == ["u1", "a1", "a2"]
    # l'autosave della bozza manda la definition senza i disegni dell'agente: rientrano
    definition = _definition()
    definition["strategy"]["charts"][0]["drawings"] = [_d("u1")]
    updated = svc.update_strategy(session, strategy.id, StrategyUpdate(definition=definition), user.id)
    assert [d["id"] for d in updated.definition["strategy"]["charts"][0]["drawings"]] == ["u1", "a1", "a2"]
    # cancellazione esplicita di un disegno dell'agente dal FE: PATCH
    out = svc.patch_strategy_chart_drawings(session, strategy.id, "main", ChartDrawingsPatch(delete_ids=["a1"]), user.id)
    assert [d["id"] for d in out] == ["u1", "a2"]
    # chart sconosciuto -> 404
    with pytest.raises(HTTPException) as exc:
        svc.read_strategy_chart_drawings(session, strategy.id, "nope", user.id)
    assert exc.value.status_code == 404


def test_backtest_patch_and_read(session, tenant, monkeypatch):
    from app.schemas.strategy import BacktestCreate
    from app.services import strategy_service as svc

    monkeypatch.setattr(svc, "publish_drawings_changed", lambda *a: None)
    user, _conn, acc = tenant
    strategy = _strategy(session, user, acc)
    payload = BacktestCreate(
        symbol="BTCUSDT", start_date=date(2026, 1, 1), end_date=date(2026, 2, 1), lessons_from="none",
        learning_mode="per_trade", agent_timeout_s=300, agent_reasoning="deep", on_agent_timeout="fail",
        simulation_timeframe="1m", initial_capital=50_000, commission=1.5, rth=False, parameters={},
    )
    bt = svc.create_backtest(session, strategy.id, payload, user.id)
    out = svc.patch_backtest_chart_drawings(session, bt.id, "main", ChartDrawingsPatch(upsert=[_model("a1")]), user.id)
    assert [d["id"] for d in out] == ["a1"]
    assert [d["id"] for d in svc.read_backtest_chart_drawings(session, bt.id, "main", user.id)] == ["a1"]
    # il design non e' toccato
    assert svc.read_strategy_chart_drawings(session, strategy.id, "main", user.id) == []
