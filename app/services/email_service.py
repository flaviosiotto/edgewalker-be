"""Outbound transactional email.

Delivery is intentionally decoupled from the request that triggers it: every
caller goes through :func:`queue_email`, which hands the send to FastAPI's
background task runner. A slow or unreachable SMTP server therefore delays the
mail, never the HTTP response that produced it.

With ``EMAIL_ENABLED`` off nothing is delivered and the rendered message is
logged instead, so local development can follow the links without an SMTP
server in the loop.
"""

import asyncio
import logging
from email.message import EmailMessage
from email.utils import formataddr
from typing import Optional

import aiosmtplib
from fastapi import BackgroundTasks

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_message(
    *,
    to_address: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = formataddr((settings.EMAIL_FROM_NAME, settings.EMAIL_FROM_ADDRESS))
    message["To"] = to_address
    message["Subject"] = subject
    message.set_content(text_body)
    if html_body:
        message.add_alternative(html_body, subtype="html")
    return message


async def send_email(
    *,
    to_address: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
) -> bool:
    """Deliver one message, retrying transient SMTP failures.

    Returns whether the message was handed to the SMTP server. Failures are
    logged rather than raised: this runs detached from the request, so there is
    nobody left to surface an exception to.
    """
    if not settings.EMAIL_ENABLED:
        logger.info(
            "Email delivery disabled; message not sent.\nTo: %s\nSubject: %s\n%s",
            to_address,
            subject,
            text_body,
        )
        return False

    message = build_message(
        to_address=to_address,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )

    # Only forward the TLS switches that were actually configured, so that
    # aiosmtplib keeps its own auto-negotiation behaviour when they are unset.
    send_kwargs: dict = {
        "hostname": settings.SMTP_HOST,
        "port": settings.SMTP_PORT,
        "username": settings.SMTP_USERNAME or None,
        "password": settings.SMTP_PASSWORD or None,
        "timeout": settings.SMTP_TIMEOUT_SECONDS,
    }
    if settings.SMTP_STARTTLS is not None:
        send_kwargs["start_tls"] = settings.SMTP_STARTTLS
    if settings.SMTP_TLS is not None:
        send_kwargs["use_tls"] = settings.SMTP_TLS

    last_error: Optional[Exception] = None
    for attempt in range(1, max(settings.SMTP_MAX_ATTEMPTS, 1) + 1):
        try:
            await aiosmtplib.send(message, **send_kwargs)
            logger.info("Sent email '%s' to %s", subject, to_address)
            return True
        except Exception as exc:  # noqa: BLE001 - background send must not escalate
            last_error = exc
            logger.warning(
                "Email send attempt %s/%s failed for %s: %s",
                attempt,
                settings.SMTP_MAX_ATTEMPTS,
                to_address,
                exc,
            )
            if attempt < settings.SMTP_MAX_ATTEMPTS:
                await asyncio.sleep(2 ** (attempt - 1))

    logger.error(
        "Giving up on email '%s' to %s after %s attempts: %s",
        subject,
        to_address,
        settings.SMTP_MAX_ATTEMPTS,
        last_error,
    )
    return False


def queue_email(
    background_tasks: Optional[BackgroundTasks],
    *,
    to_address: str,
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
) -> None:
    """Schedule a message for delivery after the response has been returned.

    ``background_tasks`` may be ``None`` for callers outside a request (startup
    hooks, background loops); those send inline on the running event loop.
    """
    if background_tasks is not None:
        background_tasks.add_task(
            send_email,
            to_address=to_address,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
        )
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    coroutine = send_email(
        to_address=to_address,
        subject=subject,
        text_body=text_body,
        html_body=html_body,
    )
    if loop is not None:
        loop.create_task(coroutine)
    else:
        asyncio.run(coroutine)
