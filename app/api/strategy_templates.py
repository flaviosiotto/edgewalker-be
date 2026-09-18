"""Strategy templates: save a strategy/backtest as a market-free template,
browse own + official templates, instantiate one on an account."""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Response, status
from sqlmodel import Session

from app.db.database import get_session
from app.models.strategy_template import StrategyTemplate
from app.models.user import User
from app.schemas.strategy_template import (
    StrategyTemplateCreate,
    StrategyTemplateInstantiate,
    StrategyTemplateInstantiateResponse,
    StrategyTemplateRead,
    StrategyTemplateSummary,
    StrategyTemplateUpdate,
    TemplateChartMeta,
    TemplateLesson,
    TemplateOrigin,
    TemplatePreview,
)
from app.services.strategy_template_service import (
    create_template,
    delete_template,
    get_template,
    instantiate_template,
    list_templates,
    preview_template,
    update_template,
)
from app.utils.auth_utils import get_current_active_user

router = APIRouter(prefix="/strategy-templates", tags=["strategy-templates"])


def _rules_count(row: StrategyTemplate) -> int:
    body = row.definition.get("strategy") if isinstance(row.definition, dict) else None
    body = body if isinstance(body, dict) else (row.definition if isinstance(row.definition, dict) else {})
    rules = body.get("rules")
    return len(rules) if isinstance(rules, list) else 0


def _summary(row: StrategyTemplate) -> StrategyTemplateSummary:
    return StrategyTemplateSummary(
        id=row.id,
        official=row.official,
        key=row.key,
        name=row.name,
        description=row.description,
        tags=list(row.tags or []),
        charts_meta=[TemplateChartMeta(**m) for m in (row.charts_meta or [])],
        rules_count=_rules_count(row),
        lessons_count=len(row.lessons or []),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _read(row: StrategyTemplate) -> StrategyTemplateRead:
    origin = dict(row.origin or {})
    warnings = list(origin.pop("warnings", []) or [])
    return StrategyTemplateRead(
        **_summary(row).model_dump(),
        definition=row.definition,
        lessons=[TemplateLesson(**l) for l in (row.lessons or [])],
        origin=TemplateOrigin(**{k: v for k, v in origin.items() if k in ("strategy_id", "backtest_id")}) if origin else None,
        warnings=warnings,
    )


@router.get("", response_model=list[StrategyTemplateSummary])
def list_strategy_templates(
    scope: Literal["mine", "official", "all"] = Query("all"),
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return [_summary(r) for r in list_templates(session, current_user.id, scope=scope)]


@router.post("/preview", response_model=TemplatePreview)
def preview_strategy_template(
    payload: StrategyTemplateCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    """Dry run of ``POST /strategy-templates``: chart slots, lessons and the
    sanitisation warnings, without saving anything."""
    return preview_template(session, payload, current_user.id)


@router.post("", response_model=StrategyTemplateRead, status_code=status.HTTP_201_CREATED)
def create_strategy_template(
    payload: StrategyTemplateCreate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return _read(create_template(session, payload, current_user.id))


@router.get("/{template_id}", response_model=StrategyTemplateRead)
def get_strategy_template(
    template_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return _read(get_template(session, template_id, current_user.id))


@router.patch("/{template_id}", response_model=StrategyTemplateRead)
def update_strategy_template(
    template_id: int,
    payload: StrategyTemplateUpdate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    return _read(update_template(session, template_id, payload, current_user.id))


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_strategy_template(
    template_id: int,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    delete_template(session, template_id, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{template_id}/instantiate", response_model=StrategyTemplateInstantiateResponse, status_code=status.HTTP_201_CREATED)
def instantiate_strategy_template(
    template_id: int,
    payload: StrategyTemplateInstantiate,
    session: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),
):
    strategy, warnings = instantiate_template(session, template_id, payload, current_user.id)
    return StrategyTemplateInstantiateResponse(strategy_id=strategy.id, name=strategy.name, warnings=warnings)
