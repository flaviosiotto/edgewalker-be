"""Inputs a skill asks for when the trader invokes it from the chat with
``/<name>``.

They are declared in the skill's own text, under a heading ``## Input``, one
per line — so the agent reads them too and an agentskills.io reader sees
plain Markdown:

    ## Input
    - backtest (backtest; facoltativo): il run da valutare, vuoto = l'ultimo
    - obiettivo (scelta: più trade | meno drawdown; facoltativo): cosa migliorare
    - modifiche (scelta: proponi | applica; default proponi): cosa farne
    - nota (testo; facoltativo): altro da tenere presente

Types: ``testo``/``text``, ``numero``/``number``, ``scelta``/``choice`` (options
separated by ``|``), ``backtest`` (one of the strategy's runs). Flags after
``;``: ``facoltativo``/``optional``, ``default <value>``.

The backend is the only interpreter of this grammar: it exposes the result as
``SkillRead.inputs``, which the frontend turns into a form and the agent
service reads from ``GET /skills/{name}``.
"""
from __future__ import annotations

import re
from typing import Any

_INPUT_HEADING_RE = re.compile(r"^#{1,6}\s*(?:input|inputs|parametri)\s*:?\s*$", re.I)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_INPUT_LINE_RE = re.compile(r"^\s*[-*]\s*`?([A-Za-z0-9_]{1,40})`?\s*\(([^)]*)\)\s*:?\s*(.*)$")
_INPUT_TYPES = {
    "testo": "text", "text": "text",
    "numero": "number", "number": "number",
    "scelta": "choice", "choice": "choice",
    "backtest": "backtest",
}
SKILL_INPUTS_MAX = 8
SKILL_INPUT_OPTIONS_MAX = 8


def parse_skill_inputs(body: str) -> list[dict[str, Any]]:
    """The inputs a skill declares under ``## Input`` (module docstring), as
    ``[{key, type, label, options, required, default}]``. Lines that do not
    follow the grammar are skipped: a skill is never refused for them."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    inside = False
    for line in (body or "").splitlines():
        if _INPUT_HEADING_RE.match(line.strip()):
            inside = True
            continue
        if not inside:
            continue
        if _HEADING_RE.match(line.strip()):
            break
        match = _INPUT_LINE_RE.match(line)
        if not match:
            continue
        key = match.group(1).lower()
        parts = [p.strip() for p in match.group(2).split(";") if p.strip()]
        if not parts or key in seen:
            continue
        kind_raw, _, options_raw = parts[0].partition(":")
        kind = _INPUT_TYPES.get(kind_raw.strip().lower())
        if kind is None:
            continue
        options = [o.strip() for o in options_raw.split("|") if o.strip()][:SKILL_INPUT_OPTIONS_MAX] if kind == "choice" else []
        if kind == "choice" and len(options) < 2:
            continue
        required, default = True, ""
        for flag in parts[1:]:
            low = flag.lower()
            if low in ("facoltativo", "optional", "opzionale"):
                required = False
            elif low.startswith("default"):
                default = flag[len("default"):].strip(" :=")
        if kind == "choice" and default and default not in options:
            default = next((o for o in options if o.lower() == default.lower()), "")
        if default:
            required = False
        seen.add(key)
        out.append({
            "key": key, "type": kind, "label": match.group(3).strip()[:200] or key,
            "options": options, "required": required, "default": default,
        })
        if len(out) >= SKILL_INPUTS_MAX:
            break
    return out
