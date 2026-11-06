"""Push ``drawings_changed``: i disegni di un chart sono cambiati (trader da
un'altra scheda, oppure l'agente dal tool ``draw_on_chart``).

Stesso canale per utente degli altri push (``connection-events:{user_id}``,
rilanciato dall'SSE di ``api/connections.py``): il payload dice solo QUALE
chart di QUALE vista; il client rilegge ``GET .../charts/{chart_id}/drawings``.
Best effort: senza Redis non succede nulla.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from app.services.connection_events import CONNECTION_EVENTS_CHANNEL_PREFIX
from app.utils.redis_client import get_redis

logger = logging.getLogger(__name__)

DRAWINGS_CHANGED_TYPE = "drawings_changed"
DrawingsScope = Literal["strategy", "live", "backtest"]


def publish_drawings_changed(user_id: int, scope: DrawingsScope, entity_id: int, chart_id: str) -> None:
    client = get_redis()
    if client is None:
        return
    payload = {"v": 1, "type": DRAWINGS_CHANGED_TYPE, "scope": scope, "id": int(entity_id), "chart_id": chart_id}
    try:
        client.publish(f"{CONNECTION_EVENTS_CHANNEL_PREFIX}{int(user_id)}", json.dumps(payload))
    except Exception as exc:  # noqa: BLE001 - il push e' best effort
        logger.debug("drawings_changed publish failed (non-critical): %s", exc)
