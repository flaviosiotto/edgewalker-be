import ast
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.services.tws_diagnostics import redact_tws_logs, tws_phase


class TwsDiagnosticsTests(unittest.TestCase):
    def test_latest_login_event_wins(self):
        logs = {'ibc': 'Second Factor Authentication initiated\nLogin has completed'}
        self.assertEqual(tws_phase('awaiting_auth', 'running', logs)[0], 'api')
        logs['ibc'] += '\nSecond Factor Authentication initiated'
        self.assertEqual(tws_phase('awaiting_auth', 'running', logs)[0], 'two_factor')

    def test_setting_name_is_not_a_two_factor_event(self):
        logs = {'launcher': 'SecondFactorAuthenticationTimeout=180'}
        self.assertEqual(tws_phase('awaiting_auth', 'running', logs)[0], 'login')

    def test_timeout_and_terminal_states(self):
        logs = {'ibc': 'Second Factor Authentication initiated\nRe-login after second factor authentication timeout not required'}
        self.assertEqual(tws_phase('awaiting_auth', 'running', logs)[0], 'login')
        self.assertEqual(tws_phase('connected', 'running', logs)[0], 'connected')
        self.assertEqual(tws_phase('connected', 'exited', logs)[0], 'error')
        self.assertEqual(tws_phase('error', 'running', logs)[0], 'error')
        self.assertEqual(tws_phase('awaiting_auth', 'running', logs, api_ready=True)[0], 'gateway')

    def test_redacts_actual_runtime_credentials_and_labels(self):
        text = 'new-secret old-secret user-demo\nIbPassword=other-secret\n--pw = hidden\nAuthorization: Bearer abc.def'
        cleaned = redact_tws_logs(text, {'password': 'new-secret', 'username': 'user-demo'}, ['TWS_PASSWORD=old-secret'])
        for secret in ('new-secret', 'old-secret', 'user-demo', 'other-secret', 'hidden', 'abc.def'):
            self.assertNotIn(secret, cleaned)

    def test_disconnected_runtime_is_not_starting(self):
        self.assertEqual(tws_phase('disconnected', 'missing', {})[0], 'stopped')


class TwsStatusTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_is_observer_without_spawn_router_or_api_probe(self):
        source = Path('app/services/connection_manager.py').read_text()
        manager_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'ConnectionManager')
        method = next(node for node in manager_class.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'tws_auth_status')
        session = MagicMock()
        session.get.return_value = SimpleNamespace(config={}, status='awaiting_auth')
        context = MagicMock()
        context.return_value.__enter__.return_value = session
        logs = {'ibc': 'Second Factor Authentication initiated'}
        namespace = {
            'Any': object, 'asyncio': asyncio, 'get_session_context': context,
            'Connection': object, 'ConnectionStatus': SimpleNamespace(AWAITING_AUTH=SimpleNamespace(value='awaiting_auth'), CONNECTED=SimpleNamespace(value='connected')),
            'TWS_API_PROBE_CACHE_SECONDS': 8, 'read_tws_logs': lambda *args: logs,
            'tws_phase': tws_phase, 'datetime': datetime, 'timezone': timezone,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<tws_auth_status>', 'exec'), namespace)
        manager = SimpleNamespace(_get_tws_container=lambda connection_id: SimpleNamespace(status='running'), _tws_api_probe_cache={})
        result = await namespace['tws_auth_status'](manager, 7)
        self.assertEqual(result['phase'], 'two_factor')
        self.assertEqual(result['logs'], {})
        self.assertFalse(result['ready_to_connect'])
        result = await namespace['tws_auth_status'](manager, 7, include_logs=True)
        self.assertEqual(result['logs'], logs)
        session.get.return_value.status = 'connected'
        result = await namespace['tws_auth_status'](manager, 7)
        self.assertTrue(result['authenticated'])
        self.assertTrue(result['ready_to_connect'])
        self.assertEqual(result['phase'], 'connected')

    async def test_status_endpoint_checks_owner_before_reading_logs(self):
        source = Path('app/api/connections.py').read_text()
        method = next(node for node in ast.parse(source).body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'tws_auth_status_endpoint')
        method.decorator_list = []
        method.args.defaults = [ast.Constant(value=None) for _ in method.args.defaults]
        tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

        class NotFoundError(Exception):
            def __init__(self, **kwargs):
                self.status_code = kwargs['status_code']

        manager = SimpleNamespace(tws_auth_status=AsyncMock())
        namespace = {
            'Response': object, 'Session': object, 'User': object,
            'get_connection': lambda *args: None, 'HTTPException': NotFoundError,
            'get_connection_manager': lambda: manager,
        }
        exec(compile(tree, '<tws_auth_status_endpoint>', 'exec'), namespace)
        with self.assertRaises(NotFoundError) as error:
            await namespace['tws_auth_status_endpoint'](7, SimpleNamespace(headers={}), True, MagicMock(), SimpleNamespace(id=99))
        self.assertEqual(error.exception.status_code, 404)
        manager.tws_auth_status.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()