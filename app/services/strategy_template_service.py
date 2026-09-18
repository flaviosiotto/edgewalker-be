"""Strategy templates: a strategy definition detached from any market.

Two pure functions carry the whole idea:

- :func:`sanitize` strips everything that identifies the market of origin
  (symbol, asset, contract data, multi-feed sources, Studio bindings,
  drawings, pinned indicator hashes) and keeps the strategy logic: rules,
  indicators, params, timezone and the chart timeframes (the multi-chart
  structure IS the strategy — decision of 18/09/2026).
- :func:`instantiate` rebinds a sanitised definition to the markets chosen in
  the wizard (one symbol per chart slot, timeframe confirmed or changed) and
  creates the strategy on an account through the same checks as
  ``create_strategy``.

Official templates live as JSON files in ``system_templates/`` and are
upserted at backend startup (:func:`sync_system_templates`), same pattern as
indicator-svc's ``system_indicators/``.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from app.models.agent_lesson import AgentLesson
from app.models.strategy import Strategy
from app.models.strategy_template import StrategyTemplate
from app.schemas.strategy import StrategyCreate
from app.schemas.strategy_template import (
    StrategyTemplateCreate,
    StrategyTemplateInstantiate,
    StrategyTemplateUpdate,
    TemplateChartMeta,
    TemplateLesson,
    TemplatePreview,
)
from app.services import strategy_service

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "system_templates"
TEMPLATE_FILE_SCHEMA = 1

# Chart-level fields that identify the market of origin.
_CHART_MARKET_FIELDS = ("symbol", "asset", "extra_data", "drawings")
# Strategy-level fields that identify the market of origin or the author's
# private context.
_STRATEGY_MARKET_FIELDS = ("symbol", "asset", "sources", "studios", "source", "exchange", "currency", "expiry")
# Present in every template: the DSL body (indicators may be empty).
_STUDIO_FIELD_RE = re.compile(r"\bstudio\.[A-Za-z0-9_\-]+")


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def _strategy_body(definition: Any) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Definizione non valida")
    body = definition.get("strategy")
    if not isinstance(body, dict):
        # Legacy flat definition: same keys at the top level.
        body = definition
    return body


def _indicator_names(indicators: Any) -> list[str]:
    out: list[str] = []
    if isinstance(indicators, list):
        for cfg in indicators:
            if isinstance(cfg, dict):
                name = cfg.get("name") or cfg.get("type")
                if isinstance(name, str) and name:
                    out.append(name)
    return out


def _indicator_types(definition: dict[str, Any]) -> list[str]:
    """Every indicator type key used anywhere in the definition (lower-case)."""
    body = _strategy_body(definition)
    types: set[str] = set()
    lists: list[Any] = [body.get("indicators")]
    for chart in body.get("charts") or []:
        if isinstance(chart, dict):
            lists.append(chart.get("indicators"))
    for lst in lists:
        if isinstance(lst, list):
            for cfg in lst:
                if isinstance(cfg, dict) and cfg.get("type"):
                    types.add(str(cfg["type"]).lower())
    return sorted(types)


def _rules_mention_studio(rules: Any) -> list[str]:
    """Names of the rules whose conditions reference ``studio.<slug>``."""
    names: list[str] = []
    if not isinstance(rules, list):
        return names
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        blob = json.dumps(rule.get("conditions") or [], ensure_ascii=False)
        if _STUDIO_FIELD_RE.search(blob):
            names.append(str(rule.get("name") or "?"))
    return names


def sanitize(
    definition: Any,
    *,
    chart_labels: Optional[dict[str, str]] = None,
) -> tuple[dict[str, Any], list[TemplateChartMeta], list[str]]:
    """Strip the market of origin from a definition.

    Returns ``(template_definition, charts_meta, warnings)``. Never mutates
    the input. Idempotent: sanitising a template returns it unchanged.
    """
    src = copy.deepcopy(definition)
    body = _strategy_body(src)
    warnings: list[str] = []
    labels = chart_labels or {}

    had_drawings = False
    charts_meta: list[TemplateChartMeta] = []
    charts = body.get("charts")
    if isinstance(charts, list) and charts:
        for idx, chart in enumerate(charts):
            if not isinstance(chart, dict):
                continue
            if chart.get("drawings"):
                had_drawings = True
            for field in _CHART_MARKET_FIELDS:
                chart.pop(field, None)
            chart_id = str(chart.get("id") or ("main" if idx == 0 else f"chart-{idx}"))
            chart["id"] = chart_id
            timeframe = str(chart.get("timeframe") or body.get("timeframe") or "5m")
            chart["timeframe"] = timeframe
            depth = chart.get("history_depth_days")
            charts_meta.append(
                TemplateChartMeta(
                    id=chart_id,
                    role="primary" if idx == 0 else "secondary",
                    label=(labels.get(chart_id) or None),
                    timeframe=timeframe,
                    history_depth_days=int(depth) if isinstance(depth, (int, float)) else None,
                    indicators=_indicator_names(chart.get("indicators")),
                )
            )
    else:
        # Legacy single-chart definition: synthesise the primary slot so the
        # wizard has something to bind; the runtime still reads the flat keys.
        timeframe = str(body.get("timeframe") or "5m")
        body["timeframe"] = timeframe
        charts_meta.append(
            TemplateChartMeta(
                id="main",
                role="primary",
                label=labels.get("main") or None,
                timeframe=timeframe,
                history_depth_days=None,
                indicators=_indicator_names(body.get("indicators")),
            )
        )

    if had_drawings:
        warnings.append("I disegni sui grafici non sono inclusi nel template (sono legati a prezzo e tempo del simbolo originale).")

    if body.get("studios"):
        warnings.append("I collegamenti agli Studi non sono inclusi nel template.")
    studio_rules = _rules_mention_studio(body.get("rules"))
    if studio_rules:
        warnings.append(
            "Regole legate a documenti di Studi (non valuteranno senza uno Studio collegato): "
            + ", ".join(studio_rules)
        )
    for field in _STRATEGY_MARKET_FIELDS:
        body.pop(field, None)
    body.pop("name", None)

    # Pinned indicator versions belong to the author's launch: the ref is
    # re-injected at the next launch of the instantiated strategy.
    for lst in [body.get("indicators")] + [c.get("indicators") for c in (body.get("charts") or []) if isinstance(c, dict)]:
        if isinstance(lst, list):
            for cfg in lst:
                if isinstance(cfg, dict) and isinstance(cfg.get("params"), dict):
                    cfg["params"].pop("ew_hash", None)

    # Rules never carry run-scoped chat ids nor the author's agent into a
    # template (rule.agent_id wins over the run agent: it must be the user's).
    rules = strategy_service._strip_rule_chat_ids(body.get("rules") or [])
    for rule in rules if isinstance(rules, list) else []:
        if isinstance(rule, dict):
            rule.pop("agent_id", None)
    body["rules"] = rules

    result = {"strategy": body} if "strategy" in src else src
    return result, charts_meta, warnings


def instantiate(
    template_definition: Any,
    *,
    charts: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Bind a sanitised definition to the markets chosen in the wizard.

    ``charts`` maps chart id → binding with ``symbol``, optional
    ``asset_type``, ``timeframe`` (None keeps the template's) and
    ``extra_data``. Every chart slot of the template must be bound.
    """
    out = copy.deepcopy(template_definition)
    body = _strategy_body(out)
    body["name"] = name

    def _binding(chart_id: str) -> Any:
        b = charts.get(chart_id)
        if b is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Manca il simbolo per il grafico '{chart_id}'",
            )
        return b

    def _get(b: Any, key: str) -> Any:
        return b.get(key) if isinstance(b, dict) else getattr(b, key, None)

    def _asset(b: Any) -> Any:
        # The symbol catalogue says "futures"; the definition says "future"
        # (same normalisation the design workspace applies on symbol change).
        raw = _get(b, "asset_type")
        if not isinstance(raw, str) or not raw.strip():
            return None
        value = raw.strip().lower()
        return "future" if value in ("futures", "fut") else value

    chart_list = body.get("charts")
    if isinstance(chart_list, list) and chart_list:
        for idx, chart in enumerate(chart_list):
            if not isinstance(chart, dict):
                continue
            chart_id = str(chart.get("id") or ("main" if idx == 0 else f"chart-{idx}"))
            b = _binding(chart_id)
            chart["id"] = chart_id
            chart["symbol"] = str(_get(b, "symbol")).strip().upper()
            asset = _asset(b)
            if asset:
                chart["asset"] = asset
            tf = _get(b, "timeframe")
            if tf:
                chart["timeframe"] = tf
            extra = _get(b, "extra_data")
            if extra:
                chart["extra_data"] = extra
            if idx == 0:
                body["symbol"] = chart["symbol"]
                body["timeframe"] = chart["timeframe"]
                if asset:
                    body["asset"] = asset
    else:
        b = _binding("main")
        body["symbol"] = str(_get(b, "symbol")).strip().upper()
        asset = _asset(b)
        if asset:
            body["asset"] = asset
        tf = _get(b, "timeframe")
        if tf:
            body["timeframe"] = tf
    return out


# ---------------------------------------------------------------------------
# Indicator classification (system vs custom)
# ---------------------------------------------------------------------------

def classify_indicator_types(type_keys: Iterable[str]) -> tuple[set[str], set[str], bool]:
    """Split type keys into (system, custom). Third value False when the
    catalogue could not be reached (nothing verified).

    Trick: indicator-svc resolves ``user_id`` first and falls back to the
    system row; asking for user 0 (no such user) resolves system keys only.
    """
    keys = sorted({k.lower() for k in type_keys if k})
    if not keys:
        return set(), set(), True
    base_url = os.getenv("INDICATOR_SVC_URL", "http://indicator-svc:8080").rstrip("/")
    try:
        from edgewalker_platform.auth.service_token import mint_service_token

        token = mint_service_token(issuer="backend", audience="indicator-svc", scopes=["indicators:resolve"])
        resp = httpx.post(
            f"{base_url}/internal/resolve",
            json={"user_id": 0, "type_keys": keys},
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("indicator-svc resolve unavailable for template check: %s", exc)
        return set(), set(keys), False
    system = set((payload.get("resolved") or {}).keys())
    custom = set(payload.get("missing") or [])
    return system, custom, True


# ---------------------------------------------------------------------------
# Building a template from a source
# ---------------------------------------------------------------------------

def _lessons_of_strategy(session: Session, strategy_id: int) -> list[TemplateLesson]:
    rows = session.exec(
        select(AgentLesson)
        .where(AgentLesson.strategy_id == strategy_id)
        .where(AgentLesson.status == "active")
        .order_by(AgentLesson.id)
    ).all()
    return [
        TemplateLesson(lesson=r.lesson, context=r.context, confidence=float(r.confidence or 0.5))
        for r in rows
    ]


def _resolve_source(
    session: Session, payload: StrategyTemplateCreate, user_id: int
) -> tuple[Any, list[TemplateLesson], dict[str, Any]]:
    """Definition + lessons + origin for the requested source (owner-checked)."""
    src = payload.source
    if src.strategy_id is not None:
        strategy = strategy_service.get_strategy(session, src.strategy_id, user_id)
        lessons = _lessons_of_strategy(session, strategy.id) if payload.include_lessons else []
        return strategy.definition, lessons, {"strategy_id": strategy.id}
    if src.backtest_id is not None:
        backtest = strategy_service.get_backtest(session, src.backtest_id, user_id)
        definition = backtest.config or strategy_service.get_strategy(session, backtest.strategy_id, user_id).definition
        lessons = _lessons_of_strategy(session, backtest.strategy_id) if payload.include_lessons else []
        return definition, lessons, {"strategy_id": backtest.strategy_id, "backtest_id": backtest.id}
    return src.definition, [], {}


def preview_template(session: Session, payload: StrategyTemplateCreate, user_id: int) -> TemplatePreview:
    definition, lessons, _ = _resolve_source(session, payload, user_id)
    sanitised, charts_meta, warnings = sanitize(definition, chart_labels=payload.chart_labels)
    _, custom, verified = classify_indicator_types(_indicator_types(sanitised))
    if custom and verified:
        warnings.append(
            "Indicatori personali usati dal template (funzionano solo per chi li possiede): "
            + ", ".join(sorted(custom))
        )
    elif custom and not verified:
        warnings.append("Catalogo indicatori non raggiungibile: indicatori non verificati.")
    return TemplatePreview(
        charts_meta=charts_meta,
        rules_count=len(_strategy_body(sanitised).get("rules") or []),
        lessons=lessons,
        warnings=warnings,
        custom_indicators=sorted(custom) if verified else [],
    )


def _template_name_taken(session: Session, user_id: int, name: str, *, exclude_id: int | None = None) -> bool:
    stmt = (
        select(StrategyTemplate.id)
        .where(StrategyTemplate.user_id == user_id)
        .where(StrategyTemplate.name == name)
    )
    if exclude_id is not None:
        stmt = stmt.where(StrategyTemplate.id != exclude_id)
    return session.exec(stmt).first() is not None


def create_template(session: Session, payload: StrategyTemplateCreate, user_id: int) -> StrategyTemplate:
    name = payload.name.strip()
    if _template_name_taken(session, user_id, name):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Esiste già un template con questo nome")
    definition, lessons, origin = _resolve_source(session, payload, user_id)
    sanitised, charts_meta, warnings = sanitize(definition, chart_labels=payload.chart_labels)
    now = datetime.now(timezone.utc)
    row = StrategyTemplate(
        user_id=user_id,
        name=name,
        description=(payload.description or "").strip() or None,
        tags=_clean_tags(payload.tags),
        definition=sanitised,
        lessons=[l.model_dump() for l in lessons],
        charts_meta=[c.model_dump() for c in charts_meta],
        origin={**origin, "warnings": warnings} if (origin or warnings) else None,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        logger.warning("create_template user %s name %r: %s", user_id, name, exc.orig)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Esiste già un template con questo nome") from exc
    session.refresh(row)
    return row


def _clean_tags(tags: Iterable[str] | None) -> list[str]:
    seen: list[str] = []
    for t in tags or []:
        s = str(t).strip().lower()[:32]
        if s and s not in seen:
            seen.append(s)
    return seen[:20]


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def list_templates(session: Session, user_id: int, *, scope: str = "all") -> list[StrategyTemplate]:
    stmt = select(StrategyTemplate)
    if scope == "mine":
        stmt = stmt.where(StrategyTemplate.user_id == user_id)
    elif scope == "official":
        stmt = stmt.where(StrategyTemplate.user_id == None)  # noqa: E711
    else:
        stmt = stmt.where((StrategyTemplate.user_id == user_id) | (StrategyTemplate.user_id == None))  # noqa: E711
    return list(session.exec(stmt.order_by(StrategyTemplate.user_id.is_(None), StrategyTemplate.name)).all())  # type: ignore[union-attr]


def get_template(session: Session, template_id: int, user_id: int) -> StrategyTemplate:
    """Own or official template; anything else is a 404 (no enumeration)."""
    row = session.get(StrategyTemplate, template_id)
    if row is None or (row.user_id is not None and row.user_id != user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template non trovato")
    return row


def _get_owned_template(session: Session, template_id: int, user_id: int) -> StrategyTemplate:
    row = get_template(session, template_id, user_id)
    if row.user_id is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="I template EdgeWalker non sono modificabili")
    return row


def update_template(session: Session, template_id: int, payload: StrategyTemplateUpdate, user_id: int) -> StrategyTemplate:
    row = _get_owned_template(session, template_id, user_id)
    if payload.name is not None:
        name = payload.name.strip()
        if _template_name_taken(session, user_id, name, exclude_id=row.id):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Esiste già un template con questo nome")
        row.name = name
    if payload.description is not None:
        row.description = payload.description.strip() or None
    if payload.tags is not None:
        row.tags = _clean_tags(payload.tags)
    if payload.lessons is not None:
        row.lessons = [l.model_dump() for l in payload.lessons]
    if payload.chart_labels is not None:
        metas = []
        for meta in row.charts_meta or []:
            m = dict(meta)
            if m.get("id") in payload.chart_labels:
                m["label"] = (payload.chart_labels[m["id"]] or "").strip() or None
            metas.append(m)
        row.charts_meta = metas
    row.updated_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def delete_template(session: Session, template_id: int, user_id: int) -> None:
    row = _get_owned_template(session, template_id, user_id)
    session.delete(row)
    session.commit()


# ---------------------------------------------------------------------------
# Instantiation on an account
# ---------------------------------------------------------------------------

def instantiate_template(
    session: Session,
    template_id: int,
    payload: StrategyTemplateInstantiate,
    user_id: int,
) -> tuple[Strategy, list[str]]:
    template = get_template(session, template_id, user_id)

    expected = {str(m.get("id")) for m in (template.charts_meta or [])}
    missing = sorted(expected - set(payload.charts.keys()))
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Manca il simbolo per i grafici: {', '.join(missing)}",
        )

    account = strategy_service._get_owned_account(session, payload.account_id, user_id)
    connection = strategy_service._get_owned_connection(session, account.connection_id, user_id)

    explicit = (payload.name or "").strip()
    if explicit:
        if strategy_service._strategy_name_taken(session, user_id, account.id, explicit):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Strategy name already exists on this account",
            )
        name = explicit
    else:
        name = strategy_service._unique_strategy_name(session, user_id, account.id, template.name[:60], style="number")

    definition = instantiate(
        template.definition,
        charts={cid: b.model_dump() for cid, b in payload.charts.items()},
        name=name,
    )

    strategy = strategy_service.create_strategy(
        session,
        StrategyCreate(
            name=name,
            description=template.description,
            definition=definition,
            manager_agent_id=payload.manager_agent_id,
            account_id=account.id,
        ),
        user_id,
    )

    warnings = list((template.origin or {}).get("warnings") or [])
    warnings += strategy_service._symbol_warnings(
        session,
        definition,
        source_connection_id=-1,  # the template has no connection: always check
        target_connection=connection,
    )

    if payload.include_lessons and template.lessons:
        now = datetime.now(timezone.utc)
        for item in template.lessons:
            if not isinstance(item, dict) or not item.get("lesson"):
                continue
            session.add(
                AgentLesson(
                    strategy_id=strategy.id,
                    user_id=user_id,
                    lesson=str(item["lesson"]),
                    context=item.get("context"),
                    status="active",
                    confidence=float(item.get("confidence") or 0.5),
                    source="template",
                    backtest_id=None,
                    evidence={"template_id": template.id, "template_name": template.name},
                    created_at=now,
                    updated_at=now,
                )
            )
        session.commit()

    return strategy, warnings


# ---------------------------------------------------------------------------
# Official templates (files → DB at startup)
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,62}$")


def load_system_template_files(directory: Path = SYSTEM_TEMPLATES_DIR) -> list[dict[str, Any]]:
    """Parse and validate every ``*.json`` template file. Raises ValueError
    on a malformed file (CI test) — the startup sync logs and skips instead."""
    files: list[dict[str, Any]] = []
    if not directory.is_dir():
        return files
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != TEMPLATE_FILE_SCHEMA:
            raise ValueError(f"{path.name}: schema {data.get('schema')!r} != {TEMPLATE_FILE_SCHEMA}")
        key = data.get("key") or path.stem
        if key != path.stem or not _KEY_RE.match(key):
            raise ValueError(f"{path.name}: key {key!r} must equal the file name (slug)")
        if not isinstance(data.get("name"), str) or not data["name"].strip():
            raise ValueError(f"{path.name}: name is required")
        if not isinstance(data.get("definition"), dict):
            raise ValueError(f"{path.name}: definition is required")
        sanitised, charts_meta, warnings = sanitize(data["definition"], chart_labels=data.get("chart_labels"))
        if sanitised != data["definition"]:
            raise ValueError(f"{path.name}: definition is not sanitised (symbol/asset/sources/studios/drawings present)")
        if warnings:
            raise ValueError(f"{path.name}: {'; '.join(warnings)}")
        lessons = [TemplateLesson(**l).model_dump() for l in (data.get("lessons") or [])]
        # Author-provided chart meta may add labels; ids/timeframes come from the definition.
        given = {str(m.get("id")): m for m in (data.get("charts_meta") or []) if isinstance(m, dict)}
        metas = []
        for m in charts_meta:
            d = m.model_dump()
            g = given.get(d["id"], {})
            d["label"] = g.get("label") or d["label"]
            metas.append(d)
        files.append(
            {
                "key": key,
                "name": data["name"].strip()[:80],
                "description": (data.get("description") or "").strip() or None,
                "tags": _clean_tags(data.get("tags")),
                "definition": sanitised,
                "lessons": lessons,
                "charts_meta": metas,
            }
        )
    return files


def sync_system_templates(session: Session, directory: Path = SYSTEM_TEMPLATES_DIR) -> int:
    """Upsert official templates by key; remove the ones whose file is gone.
    Never raises: a broken file is logged and skipped."""
    try:
        files = load_system_template_files(directory)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error("system templates: %s — sync skipped", exc)
        return 0
    existing = {
        row.key: row
        for row in session.exec(select(StrategyTemplate).where(StrategyTemplate.user_id == None)).all()  # noqa: E711
        if row.key
    }
    now = datetime.now(timezone.utc)
    seen: set[str] = set()
    for data in files:
        seen.add(data["key"])
        row = existing.get(data["key"])
        if row is None:
            row = StrategyTemplate(user_id=None, key=data["key"], created_at=now, **{k: v for k, v in data.items() if k != "key"})
            row.updated_at = now
            session.add(row)
            continue
        changed = False
        for field in ("name", "description", "tags", "definition", "lessons", "charts_meta"):
            if getattr(row, field) != data[field]:
                setattr(row, field, data[field])
                changed = True
        if changed:
            row.updated_at = now
            session.add(row)
    for key, row in existing.items():
        if key not in seen:
            session.delete(row)
    session.commit()
    logger.info("system templates synced: %d file(s)", len(files))
    return len(files)
