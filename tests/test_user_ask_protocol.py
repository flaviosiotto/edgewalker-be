"""Mid-turn questions (pure protocol), skill inputs grammar, system skills."""
import json

import pytest

from app.services import user_ask
from app.services.skill_inputs import parse_skill_inputs
from app.services.system_skills import system_skills


def _ask(**kw):
    return user_ask.build_ask(session_id=621, question="  Quale   variante? ", options=["V1", "V2", "v1", ""], now=1000.0, **kw)


def test_build_ask_cleans_and_times():
    ask = _ask(timeout_seconds=120, turn_id="t1", request_id="r1")
    assert ask["question"] == "Quale variante?"
    assert ask["options"] == [{"label": "V1"}, {"label": "V2"}]
    assert ask["created_at"] == 1_000_000 and ask["expires_at"] == 1_120_000
    assert ask["session_id"] == "621" and len(ask["id"]) == 32
    assert user_ask.pending_ttl_seconds(ask, now=1000.0) == 150
    assert not user_ask.is_expired(ask, now=1119.0) and user_ask.is_expired(ask, now=1120.0)
    assert user_ask.pending_key(621) == "chat-ask:621" and user_ask.channel(621) == "chat-stream:621"
    assert user_ask.answer_key(ask["id"]) == f"chat-ask-answer:{ask['id']}"


@pytest.mark.parametrize("options", ["V1,V2", ["solo una"], ["x" * 81, "b"], list("abcdefg")])
def test_build_ask_refuses_bad_options(options):
    with pytest.raises(user_ask.AskError):
        user_ask.build_ask(session_id=1, question="q", options=options)


def test_build_ask_refuses_empty_question_and_clamps_timeout():
    with pytest.raises(user_ask.AskError):
        user_ask.build_ask(session_id=1, question="  ", options=["a", "b"])
    ask = user_ask.build_ask(session_id=1, question="q", options=["a", "b"], timeout_seconds=99999, now=0.0)
    assert ask["expires_at"] == user_ask.MAX_TIMEOUT_SECONDS * 1000


def test_build_answer_option_text_skip():
    ask = _ask()
    assert user_ask.build_answer(ask, option=1) == {"status": "answered", "answer": "V2", "option": 1}
    assert user_ask.build_answer(ask, text="  v2 ") == {"status": "answered", "answer": "V2", "option": 1}
    assert user_ask.build_answer(ask, text="nessuna delle due") == {"status": "answered", "answer": "nessuna delle due", "option": None}
    assert user_ask.build_answer(ask, skip=True) == {"status": "skipped", "answer": "", "option": None}
    for bad in ({"option": 2}, {"option": -1}, {"option": "x"}, {"option": True}, {"text": "  "}, {}):
        with pytest.raises(user_ask.AskError):
            user_ask.build_answer(ask, **bad)
    with pytest.raises(user_ask.AskError):
        user_ask.build_answer(_ask(allow_text=False), text="altro")


def test_events_and_loads_round_trip():
    ask = _ask(request_id="r1")
    event = json.loads(user_ask.ask_event(ask))
    assert event["type"] == "ask" and event["ask"]["id"] == ask["id"] and event["request_id"] == "r1"
    done = json.loads(user_ask.resolved_event(ask, status="answered", answer="V1"))
    assert done == {"v": 1, "type": "ask_resolved", "request_id": "r1", "ask_id": ask["id"], "status": "answered", "answer": "V1"}
    assert user_ask.loads(json.dumps(ask).encode()) == ask
    assert user_ask.loads(b"not json") is None and user_ask.loads(None) is None and user_ask.loads("[1]") is None


def test_parse_skill_inputs_grammar():
    body = """# Titolo

## Input
- backtest (backtest; facoltativo): il run (da valutare)
- `modo` (scelta: proponi | applica; default Applica): cosa fare
- soglia (numero): soglia minima
- rotta (scelta: una sola): scartata
- strano (colore): scartata
- backtest (testo): doppione scartato
riga libera

## Procedura
- dopo (testo): fuori dalla sezione
"""
    assert parse_skill_inputs(body) == [
        {"key": "backtest", "type": "backtest", "label": "il run (da valutare)", "options": [], "required": False, "default": ""},
        {"key": "modo", "type": "choice", "label": "cosa fare", "options": ["proponi", "applica"], "required": False, "default": "applica"},
        {"key": "soglia", "type": "number", "label": "soglia minima", "options": [], "required": True, "default": ""},
    ]
    assert parse_skill_inputs("nessuna sezione") == []


def test_backend_system_skills_extend_and_override_the_kit_ones():
    names = [s.name for s in system_skills()]
    assert names == sorted(names) and len(names) == len(set(names))
    assert {"backtest-review", "strategy-design", "morning-briefing"} <= set(names)
    skill = next(s for s in system_skills() if s.name == "backtest-review")
    # the backend's own revision of the skill, not the kit's
    assert [i["key"] for i in parse_skill_inputs(skill.body)] == ["backtest", "confronto", "obiettivo", "modifiche"]
