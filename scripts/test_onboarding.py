import os
import unittest
from datetime import datetime, timezone
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
from app.services.onboarding_service import activate_onboarding, prepare_onboarding, provision_user_workspace, starter_definition, update_onboarding
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

    def test_user_and_workspace_share_one_commit(self):
        from app.services.user_service import save_new_user

        self.session.exec.return_value = result(None)
        added = []

        def add(entity):
            added.append(entity)
            if isinstance(entity, Agent):
                entity.id_agent = len(added) + 20
            elif getattr(entity, "id", None) is None:
                entity.id = len(added) + 20

        self.session.add.side_effect = add
        save_new_user(self.session, self.user)
        self.session.commit.assert_called_once()
        self.session.rollback.assert_not_called()
        self.assertEqual(len([entity for entity in added if isinstance(entity, Agent)]), 2)
        self.assertEqual(len([entity for entity in added if isinstance(entity, Strategy)]), 2)
        self.assertEqual(self.user.onboarding["status"], "pending")

    def test_workspace_failure_rolls_back_user_creation(self):
        from app.services.user_service import save_new_user

        with patch("app.services.user_service.provision_user_workspace", side_effect=RuntimeError("write failed")):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                save_new_user(self.session, self.user)
        self.session.commit.assert_not_called()
        self.session.rollback.assert_called_once()

    def test_personas_are_independent_between_users(self):
        from app.services.onboarding_defaults import starter_agents

        first = starter_agents("/n8n/webhook/edgewalker-manager-v2")
        second = starter_agents("/n8n/webhook/edgewalker-manager-v2")
        first[0]["persona"]["style"] = "personalizzato"
        self.assertEqual(second[0]["persona"]["style"], "didattico")

    def test_two_users_have_distinct_editable_agents(self):
        sequence = 100
        created = []

        def add(entity):
            nonlocal sequence
            if isinstance(entity, Agent) and entity.id_agent is None:
                sequence += 1
                entity.id_agent = sequence
                created.append(entity)
            elif hasattr(entity, "id") and entity.id is None:
                sequence += 1
                entity.id = sequence

        self.session.add.side_effect = add
        self.session.exec.return_value = result(None)
        provision_user_workspace(self.session, self.user)
        other = User(id=2, email="other@example.test", username="other")
        provision_user_workspace(self.session, other)
        first = [agent for agent in created if agent.user_id == 1]
        second = [agent for agent in created if agent.user_id == 2]
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertTrue({agent.id_agent for agent in first}.isdisjoint(agent.id_agent for agent in second))
        first[0].agent_name = "Il mio tutor"
        first[0].persona["style"] = "personale"
        self.assertEqual(second[0].agent_name, "Tutor")
        self.assertEqual(second[0].persona["style"], "didattico")
        self.session.commit.assert_not_called()

    def test_admin_creation_uses_workspace_transaction(self):
        from app.schemas.user import UserCreate
        from app.services.user_service import create_user

        self.session.exec.return_value = result(None)
        payload = UserCreate(email="created@example.com", username="created", password="test-password", role="admin")
        with patch("app.services.user_service.get_password_hash", return_value="hash"), patch(
            "app.services.user_service.save_new_user", return_value=self.user
        ) as save:
            self.assertIs(create_user(self.session, payload), self.user)
        self.assertEqual(save.call_args.args[1].role, "admin")
        save.assert_called_once()

    def test_registration_provisions_before_verification_email(self):
        from app.schemas.auth import RegistrationRequest
        from app.services.registration_service import register_user

        self.session.exec.return_value = result(None)
        events = []
        with patch("app.services.registration_service.settings.REGISTRATION_MODE", "open"), patch(
            "app.services.registration_service.validate_password_strength"
        ), patch("app.services.registration_service.get_password_hash", return_value="hash"), patch(
            "app.services.registration_service.save_new_user", side_effect=lambda *args: events.append("workspace")
        ) as save, patch(
            "app.services.registration_service.issue_verification_email", side_effect=lambda *args: events.append("email")
        ):
            register_user(self.session, RegistrationRequest(email="created@example.com", username="created", password="test-password"))
        self.assertEqual(events, ["workspace", "email"])
        self.assertEqual(save.call_args.args[1].status, "pending_email")
        self.assertFalse(save.call_args.args[1].is_active)

    def test_google_creation_uses_workspace_transaction(self):
        from app.models.user import UserStatus
        from app.services.google_oauth_service import _create_user_from_google

        with patch("app.services.google_oauth_service.resolve_signup_outcome", return_value=UserStatus.PENDING_APPROVAL), patch(
            "app.services.google_oauth_service._unique_username", return_value="created"
        ), patch("app.services.user_service.save_new_user") as save:
            user = _create_user_from_google(self.session, {}, "created@example.com", "test-subject", datetime.now(timezone.utc))
        save.assert_called_once_with(self.session, user)
        self.assertFalse(user.is_active)

    def test_provisioned_user_is_not_recreated(self):
        self.user.onboarding = {"status": "pending", "strategy_id": 40, "bitcoin_strategy_id": 41}
        self.assertEqual(provision_user_workspace(self.session, self.user), self.user.onboarding)
        self.session.add.assert_not_called()
        self.session.exec.assert_not_called()
        self.session.commit.assert_not_called()

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

    def test_provisioning_creates_personal_agents_without_source_ids(self):
        self.session.exec.side_effect = [result(self.user), result(None), result(None), result(None), result(None)]
        added = []

        def add(entity):
            added.append(entity)
            if isinstance(entity, Agent):
                entity.id_agent = len(added) + 20
            elif getattr(entity, "id", None) is None:
                entity.id = len(added) + 20

        self.session.add.side_effect = add
        state = prepare_onboarding(self.session, 1)
        cloned = next(entity for entity in added if isinstance(entity, Agent))
        connection = next(entity for entity in added if isinstance(entity, Connection))
        account = next(entity for entity in added if isinstance(entity, Account))
        strategy = next(entity for entity in added if isinstance(entity, Strategy))
        self.assertEqual(cloned.user_id, 1)
        self.session.get.assert_not_called()
        agents = [entity for entity in added if isinstance(entity, Agent)]
        self.assertEqual(len(agents), 2)
        self.assertNotEqual(agents[0].id_agent, agents[1].id_agent)
        self.assertTrue(all(entity.user_id == 1 for entity in agents))
        self.assertFalse(connection.is_active)
        self.assertFalse(connection.sync_enabled)
        self.assertNotIn("access_token", connection.config)
        self.assertTrue(account.extra["onboarding_provisional"])
        self.assertEqual(strategy.account_id, account.id)
        self.assertEqual(strategy.manager_agent_id, cloned.id_agent)
        self.assertEqual(state["strategy_id"], strategy.id)
        self.assertEqual(len([entity for entity in added if isinstance(entity, Chat)]), 4)
        bitcoin_connection = next(entity for entity in added if isinstance(entity, Connection) and entity.broker_type == "binance")
        self.assertTrue(bitcoin_connection.config["data_only"])
        self.assertTrue(bitcoin_connection.config["read_only"])
        self.assertFalse(bitcoin_connection.is_active)
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
        state = update_onboarding(self.session, 1, dismissed=True, step=3, track="welcome")
        self.assertEqual(state["welcome_step"], 3)
        self.assertTrue(state["welcome_dismissed"])
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