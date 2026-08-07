-- Migration: personal_access_token — long-lived per-user credentials for
--            machine clients (MCP server, scripts). The raw token is shown
--            once at creation; only its SHA-256 hex digest is stored.
--            scopes is a JSONB array drawn from {read, write, trade}.
-- Date: 2026-08-07
--
-- Safe to re-run: every statement is guarded.

BEGIN;

CREATE TABLE IF NOT EXISTS personal_access_token (
    id           SERIAL PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    name         VARCHAR(120) NOT NULL,
    token_hash   VARCHAR(64) NOT NULL UNIQUE,
    token_prefix VARCHAR(16) NOT NULL,
    scopes       JSONB NOT NULL DEFAULT '["read"]',
    expires_at   TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_personal_access_token_user_id ON personal_access_token (user_id);

COMMIT;
