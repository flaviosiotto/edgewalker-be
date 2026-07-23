"""One-shot cTrader Open API account listing for the connection-setup UI.

The ``ctidTraderAccountId`` the gateway needs as its "Account ID" is not shown
anywhere in the broker UI — FTMO/cTrader expose only the *login* number. It only
exists as a field of ``ProtoOAGetAccountListByAccessTokenRes``. This module makes
a single, short-lived Open API round-trip (ApplicationAuth + account list) so the
frontend can offer a pick-list instead of asking the user for an opaque id.

It reuses the cTrader protobuf wheel the gateway already ships (installed
``--no-deps`` to sidestep its Twisted/protobuf pins) and deliberately does NOT
spawn a gateway container: this is a stateless request/response, closed at once.
The account list is host-independent — it returns every account the token can
see, on both the demo and live hosts — so the chosen host only decides which
endpoint we dial, never which accounts come back.
"""
from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import logging
import ssl
import struct
import sys
import types
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_DEMO_HOST = "demo.ctraderapi.com"
_LIVE_HOST = "live.ctraderapi.com"
_PORT = 5035


class CTraderAccountsError(RuntimeError):
    """cTrader rejected, or could not answer, the account-list request."""


@dataclass(frozen=True)
class _Proto:
    common: ModuleType
    messages: ModuleType
    by_payload_type: dict[int, type]
    error_types: frozenset[int]

    def extract(self, envelope: Any) -> Any:
        cls = self.by_payload_type.get(int(envelope.payloadType))
        if cls is None:
            return None
        message = cls()
        message.ParseFromString(envelope.payload)
        return message


def _load_message_module(messages_pkg: ModuleType, messages_dir: Path, name: str) -> ModuleType:
    full_name = f"ctrader_open_api.messages.{name}"
    existing = sys.modules.get(full_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(full_name, messages_dir / f"{name}.py")
    if spec is None or spec.loader is None:
        raise CTraderAccountsError(f"Unable to load cTrader protobuf module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    setattr(messages_pkg, name, module)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_proto() -> _Proto:
    """Load only the generated protobuf modules from the ctrader wheel.

    The wheel's top-level package imports the Twisted client, which we do not
    want; we mirror the gateway's approach and load the ``*_pb2`` modules alone.
    """
    try:
        dist = importlib.metadata.distribution("ctrader-open-api")
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise CTraderAccountsError(
            "cTrader support requires the ctrader-open-api wheel "
            "(pip install --no-deps ctrader-open-api==0.9.2)."
        ) from exc

    package_dir = Path(dist.locate_file("ctrader_open_api"))
    messages_dir = package_dir / "messages"

    package = sys.modules.get("ctrader_open_api")
    if package is None:
        package = types.ModuleType("ctrader_open_api")
        package.__path__ = [str(package_dir)]
        sys.modules["ctrader_open_api"] = package
    elif not hasattr(package, "__path__"):
        package.__path__ = [str(package_dir)]

    messages_pkg = sys.modules.get("ctrader_open_api.messages")
    if messages_pkg is None:
        messages_pkg = types.ModuleType("ctrader_open_api.messages")
        messages_pkg.__path__ = [str(messages_dir)]
        sys.modules["ctrader_open_api.messages"] = messages_pkg
    setattr(package, "messages", messages_pkg)

    _load_message_module(messages_pkg, messages_dir, "OpenApiCommonModelMessages_pb2")
    _load_message_module(messages_pkg, messages_dir, "OpenApiModelMessages_pb2")
    common = _load_message_module(messages_pkg, messages_dir, "OpenApiCommonMessages_pb2")
    messages = _load_message_module(messages_pkg, messages_dir, "OpenApiMessages_pb2")

    by_payload_type: dict[int, type] = {}
    for module in (common, messages):
        for attr in dir(module):
            if not attr.startswith("Proto"):
                continue
            candidate = getattr(module, attr)
            if not isinstance(candidate, type):
                continue
            try:
                payload_type = candidate().payloadType
            except Exception:
                continue
            if payload_type is not None:
                by_payload_type[int(payload_type)] = candidate

    error_types: set[int] = set()
    for module, name in ((common, "ProtoErrorRes"), (messages, "ProtoOAErrorRes")):
        cls = getattr(module, name, None)
        if cls is None:
            continue
        try:
            error_types.add(int(cls().payloadType))
        except Exception:  # pragma: no cover - defensive against wheel drift
            logger.warning("Unable to resolve payloadType for %s", name)

    return _Proto(
        common=common,
        messages=messages,
        by_payload_type=by_payload_type,
        error_types=frozenset(error_types),
    )


async def _roundtrip(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    proto: _Proto,
    payload: Any,
    *,
    timeout: float,
) -> Any:
    """Send one length-prefixed ProtoMessage and await its correlated reply."""
    client_msg_id = uuid.uuid4().hex
    envelope = proto.common.ProtoMessage(
        payloadType=payload.payloadType,
        payload=payload.SerializeToString(),
    )
    envelope.clientMsgId = client_msg_id
    data = envelope.SerializeToString()
    writer.write(struct.pack(">I", len(data)) + data)
    await writer.drain()

    heartbeat_type = int(proto.common.ProtoHeartbeatEvent().payloadType)

    async def _await_response() -> Any:
        while True:
            header = await reader.readexactly(4)
            length = struct.unpack(">I", header)[0]
            raw = await reader.readexactly(length)
            reply = proto.common.ProtoMessage()
            reply.ParseFromString(raw)
            payload_type = int(reply.payloadType)
            if payload_type == heartbeat_type:
                continue
            # Ignore anything that is not the answer to the request we just sent.
            if reply.clientMsgId and reply.clientMsgId != client_msg_id:
                continue
            extracted = proto.extract(reply)
            if payload_type in proto.error_types:
                code = str(getattr(extracted, "errorCode", "CTRADER_ERROR") or "CTRADER_ERROR")
                description = str(getattr(extracted, "description", "") or "")
                raise CTraderAccountsError(f"{code}: {description}" if description else code)
            return extracted

    return await asyncio.wait_for(_await_response(), timeout=timeout)


async def fetch_ctrader_accounts(
    *,
    access_token: str,
    environment: str,
    client_id: str,
    client_secret: str,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Return the trading accounts a cTrader access token can see.

    Each item is ``{"ctid": int, "login": int | None, "is_live": bool}``.
    Raises :class:`CTraderAccountsError` on protocol/auth failures and
    :class:`asyncio.TimeoutError` if cTrader does not answer in time.
    """
    proto = _load_proto()
    messages = proto.messages
    host = _LIVE_HOST if str(environment).lower() == "live" else _DEMO_HOST

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(
            host,
            _PORT,
            ssl=ssl.create_default_context(),
            server_hostname=host,
        ),
        timeout=timeout,
    )
    try:
        await _roundtrip(
            reader,
            writer,
            proto,
            messages.ProtoOAApplicationAuthReq(clientId=client_id, clientSecret=client_secret),
            timeout=timeout,
        )
        result = await _roundtrip(
            reader,
            writer,
            proto,
            messages.ProtoOAGetAccountListByAccessTokenReq(accessToken=access_token),
            timeout=timeout,
        )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # pragma: no cover - best-effort socket teardown
            pass

    accounts: list[dict[str, Any]] = []
    for account in result.ctidTraderAccount:
        login = int(getattr(account, "traderLogin", 0) or 0)
        accounts.append(
            {
                "ctid": int(account.ctidTraderAccountId),
                "login": login or None,
                "is_live": bool(account.isLive),
            }
        )
    return accounts
