#!/usr/bin/env python3
"""Smoke test del webhook Stripe senza Stripe CLI.

Firma payload sintetici con lo stesso schema di Stripe (header
``Stripe-Signature: t=<ts>,v1=<hmac-sha256(secret, "<ts>.<body>")>``) e li
invia all'endpoint ``POST /billing/webhooks/stripe`` del backend, verificando:

  1. firma errata            → 400
  2. timestamp vecchio (>5m) → 400 (tolleranza di ``construct_event``)
  3. provider sconosciuto    → 404
  4. evento valido su un abbonamento non collegato → 200, ``applied: 0``
  5. (opzionale, ``--replay FILE``) un evento reale scaricato dal dashboard
     Stripe (Developers → Webhooks → evento → payload) inviato due volte:
     la prima applica, la seconda è deduplicata su (provider, event_id).

Uso:
  scripts/stripe_webhook_smoke.py --base-url https://api.edgewalker.tech --secret whsec_...
  STRIPE_WEBHOOK_SECRET=whsec_... scripts/stripe_webhook_smoke.py --base-url http://localhost:8000
  ... --replay evt_1Abc.json

Richiede solo la libreria standard.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request


def sign(secret: str, body: bytes, timestamp: int) -> str:
    signed = f"{timestamp}.".encode() + body
    digest = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


# Cloudflare davanti a api.edgewalker.tech rifiuta lo UA di default di urllib (403).
USER_AGENT = "EdgeWalker-webhook-smoke/1.0 (+https://edgewalker.tech)"


def post(url: str, body: bytes, signature: str | None) -> tuple[int, str]:
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if signature is not None:
        headers["Stripe-Signature"] = signature
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode())


def synthetic_subscription_event(suffix: str) -> bytes:
    now = int(time.time())
    event = {
        "id": f"evt_smoke_{suffix}",
        "object": "event",
        "api_version": "2025-08-27.basil",
        "created": now,
        "livemode": False,
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": f"sub_smoke_{suffix}",
                "object": "subscription",
                "customer": f"cus_smoke_{suffix}",
                "status": "active",
                "cancel_at_period_end": False,
                "metadata": {"smoke": "true"},
                "items": {
                    "object": "list",
                    "data": [
                        {
                            "id": f"si_smoke_{suffix}",
                            "object": "subscription_item",
                            "price": {"id": f"price_smoke_{suffix}", "object": "price"},
                            "current_period_start": now,
                            "current_period_end": now + 30 * 86400,
                        }
                    ],
                },
            }
        },
    }
    return json.dumps(event, separators=(",", ":")).encode()


class Runner:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str) -> None:
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        if not ok:
            self.failures += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("EDGEWALKER_API_URL", "http://localhost:8000"),
                        help="origine del backend, senza slash finale (default: $EDGEWALKER_API_URL o localhost:8000)")
    parser.add_argument("--secret", default=os.environ.get("STRIPE_WEBHOOK_SECRET"),
                        help="signing secret dell'endpoint (default: $STRIPE_WEBHOOK_SECRET)")
    parser.add_argument("--replay", metavar="FILE",
                        help="payload JSON di un evento Stripe reale da inviare due volte (test di idempotenza)")
    args = parser.parse_args()

    if not args.secret:
        print("Serve il signing secret: --secret whsec_... oppure STRIPE_WEBHOOK_SECRET", file=sys.stderr)
        return 2

    base = args.base_url.rstrip("/")
    endpoint = f"{base}/billing/webhooks/stripe"
    runner = Runner()

    config = get_json(f"{base}/billing/config")
    print(f"backend {base}: billing enabled={config.get('enabled')} provider={config.get('provider')}")
    if not config.get("enabled") or config.get("provider") != "stripe":
        print("Billing disabilitato o provider diverso da stripe: il webhook risponde 404 per costruzione.")
        print("Imposta BILLING_ENABLED=true BILLING_PROVIDER=stripe e riavvia il backend, poi rilancia.")
        return 1

    suffix = secrets.token_hex(4)
    body = synthetic_subscription_event(suffix)
    now = int(time.time())

    code, text = post(endpoint, body, sign("whsec_wrong_" + suffix, body, now))
    runner.check("firma errata → 400", code == 400, f"HTTP {code} {text[:120]}")

    code, text = post(endpoint, body, None)
    runner.check("senza header firma → 400", code == 400, f"HTTP {code} {text[:120]}")

    code, text = post(endpoint, body, sign(args.secret, body, now - 600))
    runner.check("timestamp vecchio (10 min) → 400", code == 400, f"HTTP {code} {text[:120]}")

    code, text = post(f"{base}/billing/webhooks/paypal", body, sign(args.secret, body, now))
    runner.check("provider sconosciuto → 404", code == 404, f"HTTP {code} {text[:120]}")

    code, text = post(endpoint, body, sign(args.secret, body, now))
    applied = None
    try:
        applied = json.loads(text).get("applied")
    except (ValueError, AttributeError):
        pass
    runner.check("evento valido, abbonamento sconosciuto → 200 applied 0",
                 code == 200 and applied == 0, f"HTTP {code} {text[:120]}")

    if args.replay:
        with open(args.replay, "rb") as handle:
            replay = json.load(handle)
        # Il dashboard esporta l'evento completo; "stripe events retrieve" idem.
        replay_body = json.dumps(replay, separators=(",", ":")).encode()
        event_id = replay.get("id", "?")
        code1, text1 = post(endpoint, replay_body, sign(args.secret, replay_body, int(time.time())))
        code2, text2 = post(endpoint, replay_body, sign(args.secret, replay_body, int(time.time())))
        applied1 = applied2 = None
        try:
            applied1 = json.loads(text1).get("applied")
            applied2 = json.loads(text2).get("applied")
        except (ValueError, AttributeError):
            pass
        runner.check(f"replay {event_id} prima consegna → 200", code1 == 200, f"HTTP {code1} {text1[:120]}")
        runner.check(f"replay {event_id} seconda consegna → applied 0 (dedup)",
                     code2 == 200 and applied2 == 0, f"HTTP {code2} {text2[:120]} (prima: applied={applied1})")
        if applied1 == 0:
            print("      nota: applied=0 anche alla prima consegna: evento già ricevuto in passato,"
                  " oppure abbonamento non collegato localmente.")

    print()
    print("tutti i controlli superati" if runner.failures == 0 else f"{runner.failures} controlli falliti")
    return 0 if runner.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
