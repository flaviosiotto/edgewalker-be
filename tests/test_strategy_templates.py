"""Strategy templates — pure-function coverage (no database).

sanitize/instantiate are the contract: a template carries rules, indicators,
params, timezone and chart timeframes, never symbol/asset/contract/broker.
Run: ``venv/bin/python -m pytest tests/test_strategy_templates.py``.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:1/test")

from fastapi import HTTPException  # noqa: E402

from app.services import strategy_template_service as svc  # noqa: E402
from app.services.strategy_template_service import (  # noqa: E402
    SYSTEM_TEMPLATES_DIR,
    classify_indicator_types,
    instantiate,
    load_system_template_files,
    sanitize,
)

RUNTIME_SYSTEM_INDICATORS = Path(__file__).resolve().parents[2] / "edgewalker-runtime" / "indicator-svc" / "system_indicators"


def _ind(name, type_, params, overlay=True):
    return {"name": name, "type": type_, "domain": "time", "params": params, "overlay": overlay,
            "output_groups": [{"name": "default"}]}


@pytest.fixture
def real_definition():
    """Shape of a real multi-chart strategy (MNQZ6 5m/1h/1d, studios, drawings)."""
    return {"strategy": {
        "name": "MNQU6_STRAT",
        "asset": "future",
        "symbol": "MNQZ6",
        "timeframe": "5m",
        "timezone": "Europe/Rome",
        "params": {},
        "studios": [{"id": 7, "name": "LIVE BNP", "slug": "live-bnp-7"}],
        "indicators": [_ind("pivottraditional", "PIVOT", {"kind": "traditional", "ew_hash": "8d30"})],
        "charts": [
            {"id": "main", "asset": "future", "symbol": "MNQZ6", "timeframe": "5m",
             "extra_data": {"expiry": "20261218", "sec_type": "FUT", "multiplier": "2"},
             "drawings": [{"kind": "trend_line", "points": []}],
             "indicators": [
                 _ind("pivottraditional", "PIVOT", {"kind": "traditional", "ew_hash": "8d30"}),
                 _ind("my_vwap_2", "MY_VWAP_2", {"symbol": "", "ew_hash": "f57c"}),
             ]},
            {"id": "chart-h1", "asset": "future", "symbol": "MNQZ6", "timeframe": "1h",
             "extra_data": {"expiry": "20261218"}, "history_depth_days": 90,
             "indicators": [_ind("ema30", "EMA", {"input": "close", "timeperiod": 30, "ew_hash": "db66"})]},
            {"id": "chart-d1", "asset": "future", "symbol": "MNQZ6", "timeframe": "1d",
             "extra_data": {"expiry": "20261218"}, "history_depth_days": 365, "indicators": []},
        ],
        "rules": [
            {"name": "session_start", "size": 1, "logic": "and", "action": "ask_agent", "chart_id": "main",
             "agent_id": 3, "chat_id": 99,
             "prompt": "Analizza", "conditions": [{"op": "equal", "field": "time.hour", "value": "15"}]},
            {"name": "doc_rule", "size": 1, "logic": "and", "action": "buy", "chart_id": "main",
             "conditions": [{"op": "equal", "field": "studio.live-bnp-7.bias", "value": "long"}]},
        ],
    }}


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------

def test_sanitize_strips_market_and_keeps_logic(real_definition):
    original = copy.deepcopy(real_definition)
    out, metas, warnings = sanitize(real_definition, chart_labels={"main": "esecuzione", "chart-h1": "contesto"})

    assert real_definition == original, "input must not be mutated"
    body = out["strategy"]
    for key in ("symbol", "asset", "studios", "name", "sources"):
        assert key not in body
    # Logic survives: timeframe, timezone, params, rules, indicators.
    assert body["timeframe"] == "5m" and body["timezone"] == "Europe/Rome"
    assert len(body["rules"]) == 2 and len(body["charts"]) == 3

    for chart in body["charts"]:
        for key in ("symbol", "asset", "extra_data", "drawings"):
            assert key not in chart
        assert chart["timeframe"]
    assert body["charts"][1]["history_depth_days"] == 90

    # Chart slots: id, role, label, timeframe, depth, indicator names.
    assert [(m.id, m.role, m.label, m.timeframe, m.history_depth_days) for m in metas] == [
        ("main", "primary", "esecuzione", "5m", None),
        ("chart-h1", "secondary", "contesto", "1h", 90),
        ("chart-d1", "secondary", None, "1d", 365),
    ]
    assert metas[0].indicators == ["pivottraditional", "my_vwap_2"]

    # Pinned hashes, rule agent/chat ids are gone; indicator params otherwise intact.
    for lst in [body["indicators"]] + [c["indicators"] for c in body["charts"]]:
        for cfg in lst:
            assert "ew_hash" not in cfg["params"]
    assert body["charts"][0]["indicators"][1]["params"] == {"symbol": ""}
    assert "agent_id" not in body["rules"][0] and "chat_id" not in body["rules"][0]

    joined = " ".join(warnings)
    assert "disegni" in joined and "Studi" in joined and "doc_rule" in joined


def test_sanitize_is_idempotent(real_definition):
    once, metas1, _ = sanitize(real_definition)
    twice, metas2, warnings2 = sanitize(once)
    assert twice == once
    assert [m.model_dump() for m in metas1] == [m.model_dump() for m in metas2]
    # Second pass: drawings/studios already gone → only the studio-rule warning remains.
    assert warnings2 == [w for w in warnings2 if "doc_rule" in w] and len(warnings2) == 1


def test_sanitize_legacy_flat_definition_synthesises_primary_slot():
    legacy = {"symbol": "AAPL", "timeframe": "1h", "asset": "stock", "indicators": [_ind("rsi14", "RSI", {"timeperiod": 14})],
              "rules": [], "params": {}}
    out, metas, warnings = sanitize(legacy)
    assert "symbol" not in out and "asset" not in out and out["timeframe"] == "1h"
    assert [(m.id, m.role, m.timeframe) for m in metas] == [("main", "primary", "1h")]
    assert metas[0].indicators == ["rsi14"]
    assert warnings == []


def test_sanitize_rejects_non_dict():
    with pytest.raises(HTTPException) as exc:
        sanitize("not a definition")
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# instantiate
# ---------------------------------------------------------------------------

def test_instantiate_binds_every_chart_and_legacy_keys(real_definition):
    template, metas, _ = sanitize(real_definition)
    bound = instantiate(
        template,
        charts={
            "main": {"symbol": "nq", "asset_type": "future", "timeframe": None,
                     "extra_data": {"expiry": "20270319", "sec_type": "FUT"}},
            "chart-h1": {"symbol": "NQ", "asset_type": "future", "timeframe": "4h", "extra_data": None},
            "chart-d1": {"symbol": "NQ", "asset_type": "future", "timeframe": None, "extra_data": None},
        },
        name="Nuova da template",
    )
    body = bound["strategy"]
    assert body["name"] == "Nuova da template"
    assert body["symbol"] == "NQ" and body["timeframe"] == "5m" and body["asset"] == "future"
    main, h1, d1 = body["charts"]
    assert main["symbol"] == "NQ" and main["timeframe"] == "5m" and main["extra_data"]["expiry"] == "20270319"
    assert h1["timeframe"] == "4h" and "extra_data" not in h1
    assert d1["timeframe"] == "1d"
    # The template itself is untouched.
    assert "symbol" not in template["strategy"]["charts"][0]


def test_instantiate_requires_every_slot(real_definition):
    template, _, _ = sanitize(real_definition)
    with pytest.raises(HTTPException) as exc:
        instantiate(template, charts={"main": {"symbol": "NQ"}}, name="x")
    assert exc.value.status_code == 400 and "chart-h1" in exc.value.detail


def test_instantiate_legacy_flat_template():
    template, _, _ = sanitize({"symbol": "AAPL", "timeframe": "1h", "indicators": [], "rules": []})
    bound = instantiate(template, charts={"main": {"symbol": "msft", "asset_type": "stock", "timeframe": "30m"}}, name="n")
    assert bound["symbol"] == "MSFT" and bound["timeframe"] == "30m" and bound["asset"] == "stock"


def test_instantiate_normalises_catalogue_asset_type():
    """The symbol catalogue says 'futures', the definition says 'future'."""
    template, _, _ = sanitize({"symbol": "NQ", "timeframe": "5m", "indicators": [], "rules": []})
    bound = instantiate(template, charts={"main": {"symbol": "NQ", "asset_type": "Futures"}}, name="n")
    assert bound["asset"] == "future"
    bound = instantiate(template, charts={"main": {"symbol": "NQ", "asset_type": "  "}}, name="n")
    assert "asset" not in bound


# ---------------------------------------------------------------------------
# indicator classification (indicator-svc resolve with user_id=0)
# ---------------------------------------------------------------------------

def test_classify_indicator_types_uses_system_resolution(monkeypatch):
    calls = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"resolved": {"ema": {}, "atr": {}}, "missing": ["my_vwap_2"]}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        return _Resp()

    monkeypatch.setattr(svc.httpx, "post", fake_post)
    monkeypatch.setenv("INTERNAL_TOKEN_SECRET", "x")
    pytest.importorskip("edgewalker_platform.auth.service_token")
    system, custom, verified = classify_indicator_types(["EMA", "atr", "MY_VWAP_2"])
    assert verified and system == {"ema", "atr"} and custom == {"my_vwap_2"}
    assert calls["json"] == {"user_id": 0, "type_keys": ["atr", "ema", "my_vwap_2"]}


def test_classify_indicator_types_degrades_when_catalogue_down(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(svc.httpx, "post", boom)
    monkeypatch.setenv("INTERNAL_TOKEN_SECRET", "x")
    pytest.importorskip("edgewalker_platform.auth.service_token")
    system, custom, verified = classify_indicator_types(["ema"])
    assert not verified and system == set() and custom == {"ema"}
    assert classify_indicator_types([]) == (set(), set(), True)


# ---------------------------------------------------------------------------
# official template files
# ---------------------------------------------------------------------------

def test_system_template_files_are_valid():
    files = load_system_template_files(SYSTEM_TEMPLATES_DIR)
    assert len(files) >= 3
    keys = {f["key"] for f in files}
    assert {"ema-crossover-trend-filter", "rsi-pullback-in-trend", "session-open-agent-review"} <= keys
    for f in files:
        body = f["definition"]["strategy"]
        assert "symbol" not in body and "asset" not in body and "studios" not in body
        for chart in body["charts"]:
            assert "symbol" not in chart and chart["timeframe"]
        assert f["charts_meta"][0]["role"] == "primary"
        assert all(m["timeframe"] for m in f["charts_meta"])
        assert f["name"] and f["description"]
        for lesson in f["lessons"]:
            assert lesson["lesson"] and 0 <= lesson["confidence"] <= 1


def test_system_template_files_carry_author_labels():
    files = {f["key"]: f for f in load_system_template_files(SYSTEM_TEMPLATES_DIR)}
    metas = files["ema-crossover-trend-filter"]["charts_meta"]
    assert [(m["id"], m["label"], m["timeframe"]) for m in metas] == [("main", "esecuzione", "5m"), ("context", "contesto", "1h")]


@pytest.mark.skipif(not RUNTIME_SYSTEM_INDICATORS.is_dir(), reason="edgewalker-runtime not checked out next to the backend")
def test_system_templates_use_only_system_indicators():
    system_keys = {p.stem for p in RUNTIME_SYSTEM_INDICATORS.glob("*.py") if not p.name.startswith("_")}
    for f in load_system_template_files(SYSTEM_TEMPLATES_DIR):
        used = set(svc._indicator_types(f["definition"]))
        assert used <= system_keys, f"{f['key']}: non-system indicators {sorted(used - system_keys)}"


def test_system_template_file_validation_rejects_market_traces(tmp_path):
    bad = {"schema": 1, "key": "bad-one", "name": "Bad",
           "definition": {"strategy": {"symbol": "NQ", "timeframe": "5m", "charts": [], "rules": []}}}
    (tmp_path / "bad-one.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not sanitised"):
        load_system_template_files(tmp_path)

    mismatch = {"schema": 1, "key": "other", "name": "X", "definition": {"strategy": {"timeframe": "5m", "rules": []}}}
    (tmp_path / "bad-one.json").write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="key"):
        load_system_template_files(tmp_path)


def test_source_accepts_exactly_one_reference():
    from pydantic import ValidationError

    from app.schemas.strategy_template import StrategyTemplateSource

    for kwargs in ({"strategy_id": 1}, {"backtest_id": 2}, {"live_id": 3}, {"definition": {"strategy": {}}}):
        StrategyTemplateSource(**kwargs)
    with pytest.raises(ValidationError):
        StrategyTemplateSource()
    with pytest.raises(ValidationError):
        StrategyTemplateSource(strategy_id=1, live_id=3)


def test_template_fields_from_file_strict_vs_import(real_definition):
    from app.services.strategy_template_service import template_fields_from_file

    # A raw strategy definition carries market traces: strict (official) refuses,
    # import sanitises and reports what it removed.
    data = {"schema": 1, "name": "Da file", "definition": copy.deepcopy(real_definition), "tags": ["Trend"],
            "charts_meta": [{"id": "main", "label": "esecuzione"}]}
    with pytest.raises(ValueError, match="not sanitised"):
        template_fields_from_file(data, strict=True)
    fields, warnings = template_fields_from_file(data, strict=False)
    assert "symbol" not in fields["definition"]["strategy"] and fields["tags"] == ["trend"]
    assert fields["charts_meta"][0]["label"] == "esecuzione"
    assert isinstance(warnings, list)

    for bad in (None, [], {"schema": 2, "name": "x", "definition": {}}, {"schema": 1, "definition": {}},
                {"schema": 1, "name": "x", "definition": "nope"}, {"schema": 1, "name": "x", "definition": {}, "lessons": [{"nope": 1}]}):
        with pytest.raises(ValueError):
            template_fields_from_file(bad, strict=False)


def test_export_file_name_is_a_safe_slug():
    from app.services.strategy_template_service import export_file_name

    assert export_file_name("Pullback RSI / trend (v2)") == "pullback-rsi-trend-v2.edgewalker-template.json"
    assert export_file_name("///") == "template.edgewalker-template.json"
