from fastapi import APIRouter, Depends
from sqlmodel import Session

from app.db.database import get_session
from app.schemas.strategy import (
    StrategyCreate,
    StrategyRead,
    StrategyUpdate,
    BacktestCreate,
    BacktestRead,
    TradeCreate,
    TradeRead,
)
from app.services.strategy_service import (
    create_strategy,
    delete_strategy,
    get_strategy,
    list_strategies,
    update_strategy,
    create_backtest,
    list_backtests,
    get_backtest,
    create_trade,
    list_trades,
)

router = APIRouter(prefix="/strategies", tags=["Strategies"])


@router.post("/", response_model=StrategyRead)
def create_strategy_endpoint(payload: StrategyCreate, session: Session = Depends(get_session)):
    return create_strategy(session, payload)


@router.get("/", response_model=list[StrategyRead])
def list_strategies_endpoint(session: Session = Depends(get_session)):
    return list_strategies(session)


@router.get("/{strategy_id}", response_model=StrategyRead)
def get_strategy_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    return get_strategy(session, strategy_id)


@router.patch("/{strategy_id}", response_model=StrategyRead)
def update_strategy_endpoint(
    strategy_id: int,
    payload: StrategyUpdate,
    session: Session = Depends(get_session),
):
    return update_strategy(session, strategy_id, payload)


@router.delete("/{strategy_id}")
def delete_strategy_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    delete_strategy(session, strategy_id)
    return {"status": "ok"}


@router.post("/{strategy_id}/backtests", response_model=BacktestRead)
def create_backtest_endpoint(
    strategy_id: int,
    payload: BacktestCreate,
    session: Session = Depends(get_session),
):
    return create_backtest(session, strategy_id, payload)


@router.get("/{strategy_id}/backtests", response_model=list[BacktestRead])
def list_backtests_endpoint(strategy_id: int, session: Session = Depends(get_session)):
    return list_backtests(session, strategy_id)


@router.get("/backtests/{backtest_id}", response_model=BacktestRead)
def get_backtest_endpoint(backtest_id: int, session: Session = Depends(get_session)):
    return get_backtest(session, backtest_id)


@router.post("/backtests/{backtest_id}/trades", response_model=TradeRead)
def create_trade_endpoint(
    backtest_id: int,
    payload: TradeCreate,
    session: Session = Depends(get_session),
):
    return create_trade(session, backtest_id, payload)


@router.get("/backtests/{backtest_id}/trades", response_model=list[TradeRead])
def list_trades_endpoint(backtest_id: int, session: Session = Depends(get_session)):
    return list_trades(session, backtest_id)
