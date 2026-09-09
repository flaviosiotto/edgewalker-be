from __future__ import annotations

import json
import re
from typing import Any


TWS_LOG_READER = r"""
import json
import os
from pathlib import Path

def tail(path):
    try:
        with path.open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 65536))
            return '\n'.join(handle.read(65536).decode('utf-8', errors='replace').splitlines()[-200:])
    except OSError:
        return ''

home = Path(os.environ.get('HOME', '/tmp'))
log_dir = Path(os.environ.get('IBC_LOG_PATH', str(home / 'ibc/logs')))
try:
    candidates = sorted(log_dir.glob('ibc-*.txt'), key=lambda path: path.stat().st_mtime)
except OSError:
    candidates = []
print(json.dumps({
    'launcher': tail(Path('/tmp/ibgateway.log')),
    'ibc': tail(candidates[-1]) if candidates else '',
}))
"""


def redact_tws_logs(text: str, config: dict[str, Any], environment: list[str]) -> str:
    secrets = {
        str(value) for key, value in config.items()
        if value and any(part in key.lower() for part in ('password', 'username', 'login_id', 'login_password'))
    }
    for entry in environment:
        key, separator, value = entry.partition('=')
        if separator and value and any(part in key.lower() for part in ('password', 'username', 'login_id')):
            secrets.add(value)
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, '***')
    text = re.sub(
        r'(?im)((?:IbLoginId|IbPassword|TWS_USERNAME|TWS_PASSWORD|--user|--pw)\s*=)[^\r\n]*',
        r'\1***', text,
    )
    text = re.sub(r'(?i)(bearer\s+)[\w.\-]+', r'\1***', text)
    return text


def read_tws_logs(container: Any, config: dict[str, Any]) -> dict[str, str]:
    result = container.exec_run(['timeout', '5', 'python3', '-c', TWS_LOG_READER])
    if result.exit_code != 0:
        raise RuntimeError('TWS diagnostic reader failed')
    payload = json.loads(result.output)
    environment = container.attrs.get('Config', {}).get('Env', []) or []
    return {
        source: redact_tws_logs(str(payload.get(source) or ''), config, environment)
        for source in ('launcher', 'ibc')
    }


def tws_phase(status: str, runtime_status: str, logs: dict[str, str], *, api_ready: bool = False) -> tuple[str, str]:
    if runtime_status in {'exited', 'dead'}:
        return 'error', 'Il processo IB Gateway si e\' arrestato.'
    if status == 'connected' and runtime_status == 'running':
        return 'connected', 'Connessione Interactive Brokers pronta.'
    if status == 'error':
        return 'error', 'Connessione non riuscita. Consulta i log per i dettagli.'
    if status == 'disconnected' and runtime_status != 'running':
        return 'stopped', 'Runtime IB Gateway non avviato.'
    if status == 'connecting' or api_ready:
        return 'gateway', 'Collegamento del gateway dati e ordini.'
    if runtime_status != 'running':
        return 'starting', 'Avvio del runtime IB Gateway.'
    for line in reversed((logs.get('ibc') or logs.get('launcher') or '').splitlines()):
        if 'Login has completed' in line:
            return 'api', 'Login completato. Verifica della TWS API.'
        if 'Re-login after second factor authentication timeout' in line:
            return 'login', 'Tempo per la 2FA scaduto. Verifica il login IB Gateway.'
        if 'Second Factor Authentication initiated' in line:
            return 'two_factor', 'Autenticazione a due fattori richiesta. Approva la richiesta sul tuo dispositivo.'
    return 'login', 'Login IB Gateway in corso. In attesa di autenticazione.'