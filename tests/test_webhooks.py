"""Outbound webhooks, pure parts: signature, backoff, request building, catalogue."""
from __future__ import annotations

import json
import time
import uuid
from datetime import timedelta

# The ORM registry configures every mapper at once: import the whole
# model tree, as the app startup does, before instantiating any row.
import app.models.agent_turn  # noqa: E402,F401
import app.models.connection  # noqa: E402,F401
import app.models.live_trading  # noqa: E402,F401
import app.models.strategy  # noqa: E402,F401
import app.models.strategy_template  # noqa: E402,F401
from app.models.webhook import WebhookDelivery
from app.services import webhook_service as ws
from app.services.webhook_dispatcher import build_request


def test_signature_roundtrip_and_tamper():
    body = b'{"id":"x","type":"ping"}'
    ts = int(time.time())
    header = ws.sign_payload("whsec_abc", ts, body)
    assert header.startswith("v1=") and len(header) == 3 + 64
    assert ws.verify_signature("whsec_abc", ts, body, header)
    assert not ws.verify_signature("whsec_abc", ts, body + b" ", header)
    assert not ws.verify_signature("whsec_other", ts, body, header)
    assert not ws.verify_signature("whsec_abc", ts - 3600, body, ws.sign_payload("whsec_abc", ts - 3600, body))


def test_backoff_schedule_is_bounded():
    delays = [ws.next_backoff(n) for n in range(ws.MAX_ATTEMPTS + 2)]
    assert delays[0] == timedelta(seconds=10)
    assert delays[ws.MAX_ATTEMPTS - 1] == timedelta(seconds=43200)
    assert delays[ws.MAX_ATTEMPTS] is None and delays[ws.MAX_ATTEMPTS + 1] is None
    assert sum(d.total_seconds() for d in delays if d) < 24 * 3600


def test_build_request_headers_and_signature():
    event_id = uuid.uuid4()
    delivery = WebhookDelivery(
        id=42,
        subscription_id=1,
        event_id=event_id,
        event_type="backtest.completed",
        payload=ws.build_envelope(event_id=event_id, event_type="backtest.completed", user_id=7, data={"backtest_id": 212}),
    )
    body, headers = build_request(delivery, "whsec_s", now_ts=1700000000)
    assert headers["X-EdgeWalker-Event"] == "backtest.completed"
    assert headers["X-EdgeWalker-Delivery"] == "42"
    assert headers["X-EdgeWalker-Event-Id"] == str(event_id)
    assert headers["X-EdgeWalker-Timestamp"] == "1700000000"
    assert headers["X-EdgeWalker-Signature"] == ws.sign_payload("whsec_s", 1700000000, body)
    parsed = json.loads(body)
    assert parsed["type"] == "backtest.completed" and parsed["data"]["backtest_id"] == 212 and parsed["user_id"] == 7


def test_envelope_is_json_safe():
    from datetime import date, datetime, timezone
    from decimal import Decimal

    env = ws.build_envelope(
        event_id=uuid.uuid4(), event_type="ping", user_id=1,
        data={"when": datetime(2026, 10, 9, tzinfo=timezone.utc), "day": date(2026, 10, 9), "amount": Decimal("1.5")},
    )
    json.dumps(env)
    assert env["data"]["day"] == "2026-10-09"


def test_catalogue_has_the_first_tier_events():
    types = {e["type"] for e in ws.event_catalog()}
    assert {
        "live.status.changed", "live.alert.triggered", "live.trade.closed", "backtest.completed",
        "backtest.failed", "agent.turn.completed", "credits.exhausted", "connection.stale", "ping",
    } <= types
