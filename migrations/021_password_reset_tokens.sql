-- Migration: add one-time password reset tokens
-- Date: 2026-04-29

CREATE TABLE IF NOT EXISTS password_reset_token (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    token_hash VARCHAR(128) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_password_reset_token_user_id
ON password_reset_token(user_id);

CREATE INDEX IF NOT EXISTS idx_password_reset_token_expires_at
ON password_reset_token(expires_at);

COMMENT ON TABLE password_reset_token IS 'One-time password reset tokens for self-service account recovery';
COMMENT ON COLUMN password_reset_token.token_hash IS 'SHA-256 hash of the raw reset token presented to the client';