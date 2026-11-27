"""Schema dei disegni (``ChartDrawing``): tipi aggiunti il 10/11/2026
(``polyline``, ``fib_retracement`` con ``props.levels``) e validazioni pure,
senza DB. Run: ``venv/bin/python -m pytest tests/test_chart_drawing_schema.py``.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.strategy import DRAWING_ANCHOR_COUNTS, MAX_FIB_LEVELS, ChartDrawing


def _a(n: int) -> list[dict]:
    return [{"t": 1_700_000_000 + i * 300, "p": 100.0 + i} for i in range(n)]


def test_polyline_keeps_every_vertex_and_needs_two():
    d = ChartDrawing(id="p", type="polyline", anchors=_a(5))
    assert len(d.anchors) == 5
    with pytest.raises(ValidationError):
        ChartDrawing(id="p", type="polyline", anchors=_a(1))


def test_fib_retracement_two_anchors_and_levels():
    d = ChartDrawing(id="f", type="fib_retracement", anchors=_a(2), props={"levels": [0, 0.5, 1, 1.618], "extend_right": True})
    assert d.props["levels"] == [0, 0.5, 1, 1.618]
    assert ChartDrawing(id="f", type="fib_retracement", anchors=_a(2)).props == {}
    with pytest.raises(ValidationError):
        ChartDrawing(id="f", type="fib_retracement", anchors=_a(3))


@pytest.mark.parametrize("levels", [[0, "a"], [0, True], [float("nan")], [11.0], list(range(MAX_FIB_LEVELS + 1)), "0,0.5"])
def test_fib_levels_rejected_when_malformed(levels):
    with pytest.raises(ValidationError):
        ChartDrawing(id="f", type="fib_retracement", anchors=_a(2), props={"levels": levels})


def test_levels_only_on_fib():
    with pytest.raises(ValidationError):
        ChartDrawing(id="t", type="trend_line", anchors=_a(2), props={"levels": [0.5]})


def test_anchor_table_matches_literal():
    # Ogni tipo del Literal ha la sua riga nella tabella delle ancore (e viceversa).
    from typing import get_args
    from app.schemas.strategy import DrawingType
    assert set(get_args(DrawingType)) == set(DRAWING_ANCHOR_COUNTS)


@pytest.mark.parametrize("side,tp,sl", [("long_position", 120, 90), ("short_position", 80, 110)])
def test_position_round_trip_and_invalid_levels(side, tp, sl):
    anchors = [{"t": 100, "p": 100}, {"t": 200, "p": tp}, {"t": 200, "p": sl}]
    drawing = ChartDrawing(id="position", type=side, anchors=anchors)
    assert ChartDrawing.model_validate_json(drawing.model_dump_json()) == drawing
    invalid = [
        anchors[:2],
        [anchors[0], anchors[2], anchors[1]],
        [anchors[0], {"t": 200, "p": 100}, anchors[2]],
        [anchors[0], anchors[1], {"t": 300, "p": sl}],
        [{"t": 200, "p": 100}, anchors[1], anchors[2]],
        [anchors[0], {"t": 200, "p": float("inf")}, anchors[2]],
        [{"t": float("nan"), "p": 100}, anchors[1], anchors[2]],
    ]
    for points in invalid:
        with pytest.raises(ValidationError):
            ChartDrawing(id="position", type=side, anchors=points)
