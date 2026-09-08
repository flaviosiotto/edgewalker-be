import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://localhost/onboarding_test")

from fastapi import HTTPException
from sqlmodel import Session

from app.models.live_trading import LiveOrder
from app.models.agent import Agent, Chat
from app.models.connection import Account, Connection
from app.models.strategy import Strategy
from app.models.user import User
from app.services.onboarding_service import activate_onboarding, prepare_onboarding, starter_definition, update_onboarding
from app.services.onboarding_state import require_configured_account


def result(value):
    response = Mock()
    response.one.return_value = value
    response.first.return_value = value
    return response


class OnboardingTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock(spec=Session)
        self.user = User(id=1, email="new@example.test", username="new")

    def test_guards(self):
        require_configured_account(SimpleNamespace(extra=None))
        for account in (None, SimpleNamespace(extra={"onboarding_provisional": True})):
            with self.assertRaises(HTTPException) as raised:
                require_configured_account(account)
            self.assertEqual(raised.exception.status_code, 409)

    def test_data_only_credentials_are_not_forwarded_and_live_is_blocked(self):
        from app.services.connection_manager import _binance_env
        from app.services.live_trading_service import validate_account_for_live

        environment = _binance_env({"data_only": True, "market_type": "spot", "api_key": "test-key", "api_secret": "test-secret"})
        self.assertEqual(environment["BINANCE_API_KEY"], "")
        self.assertEqual(environment["BINANCE_API_SECRET"], "")
        account = Account(id=30, connection_id=20, account_id="spot", account_type="data_only")
        connection = Connection(id=20, user_id=1, name="Public", broker_type="binance", config={"data_only": True}, status="connected")
        self.session.get.side_effect = [account, connection]
        with self.assertRaisesRegex(ValueError, "sola lettura"):
            validate_account_for_live(self.session, 30, 1)

    def test_prepare_is_idempotent_even_after_deletion(self):
        self.user.onboarding = {"status": "pending", "strategy_id": 42, "dismissed": True, "bitcoin_strategy_id": 43}
        self.session.exec.return_value = result(self.user)
        self.assertEqual(prepare_onboarding(self.session, 1), self.user.onboarding)
        self.session.get.assert_not_called()
        self.session.add.assert_not_called()

    def test_provisional_account_blocks_backtest_creation_and_copy(self):
        from app.services.strategy_service import copy_strategy, create_backtest

        account = Account(id=30, connection_id=20, account_id="onboarding", extra={"onboarding_provisional": True})
        strategy = Strategy(id=40, user_id=1, account_id=30, name="Starter", definition=starter_definition())
        self.session.get.return_value = account
        with patch("app.services.strategy_service.get_strategy", return_value=strategy), patch(
            "app.services.strategy_service._get_owned_account", return_value=account
        ):
            for operation in (
                lambda: create_backtest(self.session, 40, SimpleNamespace(), 1),
                lambda: copy_strategy(self.session, 40, 1, target_account_id=30),
            ):
                with self.assertRaises(HTTPException) as raised:
                    operation()
                self.assertEqual(raised.exception.status_code, 409)
        self.session.commit.assert_not_called()

    def test_existing_connection_is_not_modified(self):
        self.session.exec.side_effect = [result(self.user), result(None), result(12)]
        self.assertEqual(prepare_onboarding(self.session, 1)["status"], "existing")
        self.session.get.assert_not_called()

    @patch("app.services.onboarding_service.settings.ONBOARDING_AGENT_IDS", [])
    def test_missing_templates_do_not_create_partial_resources(self):
        self.session.exec.side_effect = [result(self.user), result(None), result(None)]
        with self.assertRaises(HTTPException) as raised:
            prepare_onboarding(self.session, 1)
        self.assertEqual(raised.exception.status_code, 503)
        self.session.add.assert_not_called()
        self.session.commit.assert_not_called()

    @patch("app.services.onboarding_service.settings.ONBOARDING_AGENT_IDS", [9])
    def test_provisioning_clones_only_template_configuration(self):
        template = Agent(id_agent=9, user_id=99, agent_name="Tutor", n8n_webhook="https://example.test/webhook", is_default=True, persona={"style": "patient"})
        self.session.get.return_value = template
        self.session.exec.side_effect = [result(self.user), result(None), result(None), result(None)]
        added = []

        def add(entity):
            added.append(entity)
            if isinstance(entity, Agent):
                entity.id_agent = 20
            elif getattr(entity, "id", None) is None:
                entity.id = len(added) + 20

        self.session.add.side_effect = add
        state = prepare_onboarding(self.session, 1)
        cloned = next(entity for entity in added if isinstance(entity, Agent))
        connection = next(entity for entity in added if isinstance(entity, Connection))
        account = next(entity for entity in added if isinstance(entity, Account))
        strategy = next(entity for entity in added if isinstance(entity, Strategy))
        self.assertEqual(cloned.user_id, 1)
        self.assertIsNot(cloned.persona, template.persona)
        self.assertFalse(connection.is_active)
        self.assertFalse(connection.sync_enabled)
        self.assertNotIn("access_token", connection.config)
        self.assertTrue(account.extra["onboarding_provisional"])
        self.assertEqual(strategy.account_id, account.id)
        self.assertEqual(strategy.manager_agent_id, cloned.id_agent)
        self.assertEqual(state["strategy_id"], strategy.id)
        self.assertEqual(len([entity for entity in added if isinstance(entity, Chat)]), 3)
        bitcoin_connection = next(entity for entity in added if isinstance(entity, Connection) and entity.broker_type == "binance")
        self.assertTrue(bitcoin_connection.config["data_only"])
        self.assertTrue(bitcoin_connection.config["read_only"])
        self.assertNotIn("api_key", bitcoin_connection.config)
        bitcoin = next(entity for entity in added if isinstance(entity, Strategy) and entity.id == state["bitcoin_strategy_id"])
        self.assertEqual(bitcoin.definition["strategy"]["symbol"], "BTCUSDT")
        self.assertEqual(bitcoin.definition["strategy"]["rules"][0]["size"], 0.001)
        self.assertFalse(bitcoin.definition["strategy"]["rth"])
        self.assertTrue(bitcoin.layout_config["extendedHours"])
        self.session.commit.assert_called_once()

    def test_bitcoin_backtest_payload_accepts_crypto(self):
        from app.schemas.strategy import BacktestCreate

        payload = BacktestCreate.model_validate({
            "symbol": "BTCUSDT",
            "start_date": "2026-09-01", "end_date": "2026-09-07",
            "source": "binance", "asset": "crypto", "rth": False,
        })
        self.assertEqual(payload.asset, "crypto")
        self.assertFalse(payload.rth)

    def test_guides_have_independent_progress(self):
        self.user.onboarding = {"status": "pending", "strategy_id": 40, "bitcoin_strategy_id": 50, "step": 2, "dismissed": True}
        self.session.exec.return_value = result(self.user)
        state = update_onboarding(self.session, 1, dismissed=False, step=1, track="bitcoin")
        self.assertEqual(state["step"], 2)
        self.assertTrue(state["dismissed"])
        self.assertEqual(state["bitcoin_step"], 1)
        self.assertFalse(state["bitcoin_dismissed"])

    def test_cannot_update_guide_before_preparation(self):
        self.session.exec.return_value = result(self.user)
        with self.assertRaises(HTTPException):
            update_onboarding(self.session, 1, dismissed=True, step=2)
        self.session.commit.assert_not_called()

    def test_activation_rejects_another_users_account(self):
        account = Account(id=10, connection_id=20, account_id="123")
        connection = Connection(id=20, user_id=99, name="Other", broker_type="ctrader")
        self.session.get.side_effect = [account, connection]
        with self.assertRaises(HTTPException) as raised:
            activate_onboarding(self.session, 1, 10, "EURUSD")
        self.assertEqual(raised.exception.status_code, 404)
        self.session.commit.assert_not_called()

    @patch("app.services.symbol_sync_handler.search_gateway_symbols_by_id")
    def test_activation_rejects_disconnected_or_non_forex_account(self, search):
        account = Account(id=10, connection_id=20, account_id="123")
        connection = Connection(id=20, user_id=1, name="Demo", broker_type="ctrader", status="disconnected")
        self.session.get.side_effect = [account, connection, account, connection]
        with self.assertRaises(HTTPException) as raised:
            activate_onboarding(self.session, 1, 10, "EURUSD")
        self.assertEqual(raised.exception.status_code, 409)
        search.assert_not_called()
        connection.status = "connected"
        search.return_value = [{"symbol": "EURUSD", "asset_type": "stock"}]
        with self.assertRaises(HTTPException) as raised:
            activate_onboarding(self.session, 1, 10, "EURUSD")
        self.assertEqual(raised.exception.status_code, 422)
        self.session.commit.assert_not_called()

    @patch("app.services.symbol_sync_handler.search_gateway_symbols_by_id")
    def test_activation_preserves_rules_and_replaces_provisional_account(self, search):
        account = Account(id=10, connection_id=20, account_id="123")
        connection = Connection(id=20, user_id=1, name="Demo", broker_type="ctrader", status="connected", config={"read_only": True})
        provisional = Account(id=30, connection_id=20, account_id="onboarding", extra={"onboarding_provisional": True})
        definition = starter_definition()
        definition["strategy"]["rules"][0]["size"] = 2000
        strategy = Strategy(id=40, user_id=1, name="Personalizzata", account_id=30, connection_id=20, definition=definition)
        self.user.onboarding = {"status": "pending", "strategy_id": 40}
        self.session.get.side_effect = [account, connection, strategy, provisional]
        self.session.exec.side_effect = [result(self.user), result(None), result(None)]
        search.return_value = [{"symbol": "EURUSD.demo", "asset_type": "forex", "extra_data": {"symbol_id": 17}}]
        state = activate_onboarding(self.session, 1, 10, "EURUSD.demo")
        self.assertEqual(state["status"], "ready")
        self.assertEqual(strategy.account_id, 10)
        self.assertEqual(strategy.definition["strategy"]["rules"][0]["size"], 2000)
        self.assertEqual(strategy.definition["strategy"]["symbol"], "EURUSD.demo")
        self.assertTrue(connection.config["read_only"])
        self.session.delete.assert_called_once_with(provisional)
        self.session.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()