from __future__ import annotations

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlmodel import Session, select

from app.models.strategy import Strategy, BacktestResult, BacktestTrade
from app.schemas.strategy import (
    StrategyCreate,
    StrategyUpdate,
    BacktestCreate,
    TradeCreate,
)


def create_strategy(session: Session, payload: StrategyCreate) -> Strategy:
    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Strategy name is required",
        )

    existing = session.exec(select(Strategy).where(Strategy.name == name)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Strategy name already exists",
        )

    now = datetime.now(timezone.utc)
    strategy = Strategy(
        name=name,
        description=payload.description,
        definition=payload.definition,
        created_at=now,
        updated_at=now,
    )
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def list_strategies(session: Session) -> list[Strategy]:
    return list(session.exec(select(Strategy).order_by(Strategy.id.desc())).all())


def get_strategy(session: Session, strategy_id: int) -> Strategy:
    strategy = session.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Strategy not found")
    return strategy


def update_strategy(session: Session, strategy_id: int, payload: StrategyUpdate) -> Strategy:
    strategy = get_strategy(session, strategy_id)

    if payload.name is not None and payload.name != strategy.name:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Strategy name is required",
            )

        existing = session.exec(select(Strategy).where(Strategy.name == name)).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Strategy name already exists",
            )
        strategy.name = name

    if payload.description is not None:
        strategy.description = payload.description

    if payload.definition is not None:
        strategy.definition = payload.definition

    strategy.updated_at = datetime.now(timezone.utc)

    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return strategy


def delete_strategy(session: Session, strategy_id: int) -> None:
    strategy = get_strategy(session, strategy_id)
    session.delete(strategy)
    session.commit()


def create_backtest(session: Session, strategy_id: int, payload: BacktestCreate) -> BacktestResult:
    _ = get_strategy(session, strategy_id)
    backtest = BacktestResult(
        strategy_id=strategy_id,
        symbol=payload.symbol,
        start_date=payload.start_date,
        end_date=payload.end_date,
        parameters=payload.parameters,
        metrics=payload.metrics,
    )
    session.add(backtest)
    session.commit()
    session.refresh(backtest)
    return backtest


def list_backtests(session: Session, strategy_id: int) -> list[BacktestResult]:
    _ = get_strategy(session, strategy_id)
    return list(
        session.exec(
            select(BacktestResult)
            .where(BacktestResult.strategy_id == strategy_id)
            .order_by(BacktestResult.id.desc())
        ).all()
    )


def get_backtest(session: Session, backtest_id: int) -> BacktestResult:
    backtest = session.get(BacktestResult, backtest_id)
    if not backtest:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Backtest not found")
    return backtest


def create_trade(session: Session, backtest_id: int, payload: TradeCreate) -> BacktestTrade:
    backtest = get_backtest(session, backtest_id)

    trade = BacktestTrade(
        backtest_id=backtest_id,
        strategy_id=backtest.strategy_id,
        ts_open=payload.ts_open,
        ts_close=payload.ts_close,
        side=payload.side,
        quantity=payload.quantity,
        entry_price=payload.entry_price,
        exit_price=payload.exit_price,
        pnl=payload.pnl,
        fees=payload.fees,
        meta=payload.meta,
    )
    session.add(trade)
    session.commit()
    session.refresh(trade)
    return trade


def list_trades(session: Session, backtest_id: int) -> list[BacktestTrade]:
    _ = get_backtest(session, backtest_id)
    return list(
        session.exec(
            select(BacktestTrade)
            .where(BacktestTrade.backtest_id == backtest_id)
            .order_by(BacktestTrade.id.asc())
        ).all()
    )
