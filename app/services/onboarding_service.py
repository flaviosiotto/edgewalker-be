from copy import deepcopy
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.config import settings
from app.models.agent import Agent, Chat
from app.models.connection import Account, Connection
from app.models.strategy import Strategy
from app.models.user import User
from app.services.onboarding_defaults import starter_agents
from app.services.onboarding_state import is_provisional_account, require_configured_account


def starter_definition(symbol: str = "EURUSD", asset: str = "forex", size: float = 1000) -> dict:
    return {
        "strategy": {
            "symbol": symbol,
            "asset": asset,
            "timeframe": "1h",
            "indicators": [],
            "rules": [
                {
                    "name": "Ingresso su rialzo orario",
                    "action": "buy",
                    "size": size,
                    "conditions": [
                        {"field": "position.size", "op": "equal", "value": 0},
                        {"field": "price.close", "op": "greater_than", "value": "prev_price.high"},
                    ],
                    "tp_expr": "price.close * 1.005",
                    "sl_expr": "price.close * 0.9975",
                },
            ],
        },
    }


def _prepare_bitcoin(session: Session, user_id: int, agent_id: int) -> dict:
    connection = Connection(
        user_id=user_id, name="Binance - Dati pubblici", broker_type="binance",
        config={"market_type": "spot", "testnet": False, "data_only": True, "read_only": True},
        is_active=False, sync_enabled=False,
    )
    session.add(connection)
    session.flush()
    account = Account(
        connection_id=connection.id, account_id="spot", display_name="Binance public data",
        account_type="data_only", currency="USDT", extra={"data_only": True},
    )
    session.add(account)
    session.flush()
    definition = starter_definition("BTCUSDT", "crypto", 0.001)
    definition["strategy"].update(source="binance", rth=False)
    strategy = Strategy(
        user_id=user_id, name="Primi passi - Bitcoin", account_id=account.id,
        connection_id=connection.id, manager_agent_id=agent_id,
        description="Esempio didattico BTC/USDT sui dati pubblici Binance Spot. Solo dati e backtest, senza trading reale.",
        definition=definition,
        layout_config={"extendedHours": True, "timezone": "UTC"},
    )
    session.add(strategy)
    session.flush()
    session.add(Chat(
        user_id=user_id, id_agent=agent_id, strategy_id=strategy.id, nome="Strategia Bitcoin",
        chat_type=Chat.ChatType.STRATEGY, created_at=datetime.now(timezone.utc),
    ))
    return {
        "bitcoin_strategy_id": strategy.id, "bitcoin_connection_id": connection.id,
        "bitcoin_account_id": account.id, "bitcoin_step": 0, "bitcoin_dismissed": False,
    }


def prepare_onboarding(session: Session, user_id: int) -> dict:
    user = session.exec(select(User).where(User.id == user_id).with_for_update()).one()
    if user.onboarding:
        if user.onboarding.get("status") in {"pending", "ready"} and "bitcoin_strategy_id" not in user.onboarding:
            starter = session.get(Strategy, user.onboarding.get("strategy_id"))
            if starter and starter.user_id == user_id and starter.manager_agent_id:
                user.onboarding = {**user.onboarding, **_prepare_bitcoin(session, user_id, starter.manager_agent_id)}
                session.add(user)
                session.commit()
        return dict(user.onboarding)
    if (
        user.role == "admin"
        or session.exec(select(Strategy.id).where(Strategy.user_id == user_id)).first() is not None
        or session.exec(select(Connection.id).where(Connection.user_id == user_id)).first() is not None
    ):
        user.onboarding = {"status": "existing", "dismissed": True}
        session.add(user)
        session.commit()
        return dict(user.onboarding)

    state = provision_user_workspace(session, user)
    session.commit()
    return state


def provision_user_workspace(session: Session, user: User) -> dict:
    if user.onboarding:
        return dict(user.onboarding)
    user_id = user.id
    templates = starter_agents(settings.ONBOARDING_AGENT_WEBHOOK_URL)
    agents = []
    now = datetime.now(timezone.utc)
    for template in templates:
        agent = session.exec(select(Agent).where(Agent.user_id == user_id, Agent.agent_name == template["agent_name"])).first()
        if agent is None:
            agent = Agent(
                user_id=user_id, is_default=not agents, **deepcopy(template),
            )
            session.add(agent)
            session.flush()
            session.add(Chat(user_id=user_id, id_agent=agent.id_agent, nome=f"{agent.agent_name} Default", chat_type=Chat.ChatType.USER, created_at=now))
        agents.append(agent)

    connection = Connection(
        user_id=user_id, name="cTrader - Primi passi", broker_type="ctrader",
        config={"environment": "demo", "onboarding_pending": True, "read_only": True},
        is_active=False, sync_enabled=False,
        status_message="Collega un conto cTrader per attivare la strategia Forex.",
    )
    session.add(connection)
    session.flush()
    account = Account(
        connection_id=connection.id, account_id="onboarding", display_name="Conto da collegare",
        account_type="provisional", currency="USD", extra={"onboarding_provisional": True},
    )
    session.add(account)
    session.flush()
    strategy = Strategy(
        user_id=user_id, name="Primi passi - EUR/USD", account_id=account.id,
        connection_id=connection.id, manager_agent_id=agents[0].id_agent,
        description="Esempio didattico: ingresso sul superamento del massimo orario precedente. Non costituisce una raccomandazione di investimento.",
        definition=starter_definition(),
    )
    session.add(strategy)
    session.flush()
    session.add(Chat(user_id=user_id, id_agent=agents[0].id_agent, strategy_id=strategy.id, nome="Strategia Forex", chat_type=Chat.ChatType.STRATEGY, created_at=now))
    bitcoin = _prepare_bitcoin(session, user_id, agents[0].id_agent)
    user.onboarding = {
        "status": "pending", "dismissed": False, "step": 0,
        "strategy_id": strategy.id, "connection_id": connection.id,
        "provisional_account_id": account.id,
        **bitcoin,
    }
    session.add(user)
    session.flush()
    return dict(user.onboarding)


def update_onboarding(session: Session, user_id: int, *, dismissed: bool, step: int, track: str = "forex") -> dict:
    user = session.exec(select(User).where(User.id == user_id).with_for_update()).one()
    if user.onboarding.get("status") not in {"pending", "ready"}:
        raise HTTPException(409, "La guida iniziale non e' disponibile.")
    if track not in {"forex", "bitcoin", "welcome"} or (track == "bitcoin" and not user.onboarding.get("bitcoin_strategy_id")):
        raise HTTPException(409, "Il percorso richiesto non e' disponibile.")
    prefix = f"{track}_" if track != "forex" else ""
    user.onboarding = {**user.onboarding, f"{prefix}dismissed": dismissed, f"{prefix}step": step}
    session.add(user)
    session.commit()
    return dict(user.onboarding)


def activate_onboarding(session: Session, user_id: int, account_id: int, symbol: str) -> dict:
    from app.services.symbol_sync_handler import search_gateway_symbols_by_id

    account = session.get(Account, account_id)
    connection = session.get(Connection, account.connection_id) if account else None
    if connection is None or connection.user_id != user_id:
        raise HTTPException(404, "Conto non trovato.")
    require_configured_account(account)
    if connection.broker_type != "ctrader" or connection.status not in {"connected", "degraded"}:
        raise HTTPException(409, "Collega prima il conto cTrader dalla pagina Connessioni.")
    results = search_gateway_symbols_by_id(symbol, connection.id, "ctrader", limit=100)
    selected = next((item for item in results if item.get("symbol") == symbol), None)
    if selected is None or selected.get("asset_type") != "forex":
        raise HTTPException(422, "Seleziona uno strumento Forex presente nel catalogo del conto.")

    user = session.exec(select(User).where(User.id == user_id).with_for_update()).one()
    state = dict(user.onboarding)
    if state.get("status") == "ready":
        return state
    strategy = session.get(Strategy, state.get("strategy_id")) if state.get("strategy_id") else None
    if strategy is None or strategy.user_id != user_id or state.get("status") != "pending":
        raise HTTPException(409, "La strategia iniziale non e' disponibile.")
    provisional = session.get(Account, strategy.account_id)
    if not is_provisional_account(provisional):
        raise HTTPException(409, "La strategia e' gia' associata a un conto.")
    duplicate = session.exec(select(Strategy.id).where(
        Strategy.user_id == user_id, Strategy.account_id == account.id,
        Strategy.name == strategy.name, Strategy.id != strategy.id,
    )).first()
    if duplicate is not None:
        raise HTTPException(409, "Esiste gia' una strategia con questo nome sul conto: rinomina quella iniziale.")
    definition = deepcopy(strategy.definition)
    config = definition.setdefault("strategy", {})
    config.update(symbol=symbol, asset="forex", extra_data=deepcopy(selected.get("extra_data") or {}))
    charts = config.get("charts")
    if isinstance(charts, list) and charts:
        charts[0].update(symbol=symbol, asset="forex", extra_data=deepcopy(selected.get("extra_data") or {}))
    strategy.definition = definition
    strategy.account_id = account.id
    strategy.connection_id = connection.id
    strategy.updated_at = datetime.now(timezone.utc)
    session.add(strategy)
    session.flush()
    if session.exec(select(Strategy.id).where(Strategy.account_id == provisional.id)).first() is None:
        session.delete(provisional)
    connection.config = {**connection.config, "onboarding_pending": False}
    connection.is_active = True
    connection.sync_enabled = True
    session.add(connection)
    user.onboarding = {**state, "status": "ready", "connection_id": connection.id, "step": 3, "dismissed": False}
    session.add(user)
    session.commit()
    return dict(user.onboarding)