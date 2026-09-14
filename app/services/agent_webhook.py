"""Single source of truth for the agent execution endpoint.

Since the n8n dismissal (phase 3) every agent turn — chat, rule trigger,
live lifecycle, manager notifications, runner-side dispatch — goes to
agent-svc. ``agent.n8n_webhook`` is a legacy column that is never read any
more: callers must use :func:`agent_webhook_url` instead.
"""

from app.core.config import settings


def agent_webhook_url() -> str:
    """Return the agent-svc webhook URL (``AGENT_SVC_WEBHOOK_URL``)."""
    return settings.AGENT_SVC_WEBHOOK_URL
