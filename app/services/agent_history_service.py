"""Full-text-ish search over the chats of one agent (``search_history`` tool,
decision D8): a bounded ILIKE over the human/ai rows of the agent's chats,
newest first. No index: the rows of one agent are few thousands at most and
the tool is called by the agent a handful of times per turn at worst.
"""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlmodel import Session

from app.schemas.agent_model import HistoryHit

EXCERPT_CHARS = 320
MAX_HITS = 50


def _excerpt(content: str, query: str) -> str:
    flat = re.sub(r"\s+", " ", content or "").strip()
    if len(flat) <= EXCERPT_CHARS:
        return flat
    pos = flat.lower().find(query.lower())
    if pos < 0:
        return flat[:EXCERPT_CHARS] + "…"
    start = max(0, pos - EXCERPT_CHARS // 3)
    end = min(len(flat), start + EXCERPT_CHARS)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(flat) else ""
    return prefix + flat[start:end] + suffix


def search_history(
    session: Session,
    *,
    user_id: int,
    agent_id: int,
    query: str,
    limit: int = 20,
    chat_id: int | None = None,
) -> list[HistoryHit]:
    q = str(query or "").strip()
    if len(q) < 2:
        return []
    limit = max(1, min(int(limit or 20), MAX_HITS))
    sql = """
        SELECT h.id AS row_id, h.created_at, h.message->>'type' AS type, h.message->>'content' AS content,
               c.id AS chat_id, c.nome AS chat_name
        FROM n8n_chat_histories h
        JOIN chat c ON c.n8n_session_id = h.session_id
        WHERE c.user_id = :uid AND c.id_agent = :aid
          AND h.message->>'type' IN ('human', 'ai')
          AND (jsonb_typeof(h.message->'tool_calls') IS DISTINCT FROM 'array' OR jsonb_array_length(h.message->'tool_calls') = 0)
          AND h.message->>'content' ILIKE :pattern
    """
    params: dict[str, Any] = {"uid": user_id, "aid": agent_id, "pattern": f"%{q}%", "lim": limit}
    if chat_id is not None:
        sql += " AND c.id = :cid"
        params["cid"] = chat_id
    sql += " ORDER BY h.id DESC LIMIT :lim"
    rows = session.execute(text(sql), params).mappings().all()
    return [
        HistoryHit(
            chat_id=int(r["chat_id"]),
            chat_name=r["chat_name"],
            row_id=int(r["row_id"]),
            timestamp=r["created_at"],
            type=str(r["type"]),
            text=_excerpt(str(r["content"] or ""), q),
        )
        for r in rows
    ]
