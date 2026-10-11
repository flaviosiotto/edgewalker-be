"""Regression for the 11/10/2026 incident: the snapshot lookups of the
reconciliation must carry LIMIT 1. Without it ``.first()`` still fetches and
hydrates every snapshot of the account (355k rows, ~7 s, ~700 MB per call),
which held DB connections long enough for Postgres to kill them."""
from datetime import datetime, timezone

from app.services import performance_service as ps


class _Result:
    def first(self):
        return None


class _Session:
    def __init__(self):
        self.statements = []

    def exec(self, stmt):
        self.statements.append(stmt)
        return _Result()


def _sql(stmt) -> str:
    return str(stmt.compile(compile_kwargs={"literal_binds": True})).upper()


def test_snapshot_lookups_are_limited_to_one_row():
    session = _Session()
    at = datetime(2026, 10, 11, tzinfo=timezone.utc)
    ps._snapshot_at_or_before(session, 1, at)
    ps._snapshot_at_or_before(session, 1, None)
    ps._snapshot_at_or_after(session, 1, at)
    assert len(session.statements) == 3
    for stmt in session.statements:
        sql = _sql(stmt)
        assert "LIMIT 1" in sql, sql
        assert "ORDER BY" in sql
