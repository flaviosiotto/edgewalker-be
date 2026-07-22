-- Migration: self-service registration, family&friends access gating,
--            external (OAuth) identities and 2FA scaffolding
-- Date: 2026-07-22
--
-- Safe to re-run: every statement is guarded.
--
-- ORDER MATTERS: apply this BEFORE rolling out the backend build that adds the
-- new User columns. The ORM selects them on every query, so a backend started
-- against the old schema fails every authenticated request.

BEGIN;

-- ---------------------------------------------------------------------------
-- 1. user: profile fields, lifecycle status, verification/approval audit
-- ---------------------------------------------------------------------------

ALTER TABLE "user" ADD COLUMN IF NOT EXISTS first_name VARCHAR(120);
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS last_name VARCHAR(120);
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS status VARCHAR(32) NOT NULL DEFAULT 'active';
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS email_verified_at TIMESTAMPTZ;
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ;
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS approved_by INTEGER;
ALTER TABLE "user" ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ;

-- Accounts created before this migration predate the whole lifecycle: they were
-- provisioned by an administrator, so they count as verified and approved.
UPDATE "user"
SET email_verified_at = created_at
WHERE email_verified_at IS NULL;

UPDATE "user"
SET status = 'suspended'
WHERE NOT is_active
  AND status = 'active';

-- Users authenticating only through an external provider never have a local
-- password, so the column can no longer be mandatory.
ALTER TABLE "user" ALTER COLUMN hashed_password DROP NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'fk_user_approved_by'
    ) THEN
        ALTER TABLE "user"
        ADD CONSTRAINT fk_user_approved_by
        FOREIGN KEY (approved_by) REFERENCES "user"(id) ON DELETE SET NULL;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'ck_user_status'
    ) THEN
        ALTER TABLE "user"
        ADD CONSTRAINT ck_user_status CHECK (
            status IN ('pending_email', 'pending_approval', 'active', 'rejected', 'suspended')
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_user_status ON "user"(status);

COMMENT ON COLUMN "user".status IS
    'Account lifecycle. is_active stays the enforcement point and mirrors this column.';
COMMENT ON COLUMN "user".hashed_password IS
    'NULL for accounts that authenticate only through an external identity provider';

-- ---------------------------------------------------------------------------
-- 2. user_identity: external identity providers linked to a local account
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_identity (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    provider VARCHAR(32) NOT NULL,
    provider_subject VARCHAR(255) NOT NULL,
    email VARCHAR(320),
    raw_profile JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,
    CONSTRAINT uq_user_identity_provider_subject UNIQUE (provider, provider_subject)
);

CREATE INDEX IF NOT EXISTS idx_user_identity_user_id ON user_identity(user_id);

COMMENT ON TABLE user_identity IS 'External login identities (google, ...) bound to a local user';
COMMENT ON COLUMN user_identity.provider_subject IS
    'Stable provider-side user id (the OIDC "sub"). Never match on email alone.';

-- ---------------------------------------------------------------------------
-- 3. access_allowlist: pre-authorised emails for the family&friends phase
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS access_allowlist (
    id SERIAL PRIMARY KEY,
    email VARCHAR(320) NOT NULL UNIQUE,
    note TEXT,
    invited_by INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at TIMESTAMPTZ,
    consumed_by_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    CONSTRAINT ck_access_allowlist_email_lower CHECK (email = lower(email))
);

COMMENT ON TABLE access_allowlist IS
    'Emails allowed to register without administrator approval while REGISTRATION_MODE gates signup';
COMMENT ON COLUMN access_allowlist.email IS 'Always stored lowercased; matching is case-insensitive';

-- Everyone who already has an account is implicitly allowed, so that a future
-- re-registration or provider link is not stopped by the gate.
INSERT INTO access_allowlist (email, note, consumed_at, consumed_by_user_id)
SELECT lower(u.email), 'Backfilled: account existed before access gating', NOW(), u.id
FROM "user" u
ON CONFLICT (email) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 4. email_verification_token: mirrors password_reset_token
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS email_verification_token (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    token_hash VARCHAR(128) NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_email_verification_token_user_id
ON email_verification_token(user_id);

CREATE INDEX IF NOT EXISTS idx_email_verification_token_expires_at
ON email_verification_token(expires_at);

COMMENT ON COLUMN email_verification_token.token_hash IS
    'SHA-256 hash of the raw token emailed to the user';

-- ---------------------------------------------------------------------------
-- 5. Two-factor scaffolding (tables only; enrolment lands in a later phase)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS user_totp (
    user_id INTEGER PRIMARY KEY REFERENCES "user"(id) ON DELETE CASCADE,
    secret VARCHAR(255) NOT NULL,
    confirmed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE user_totp IS
    'TOTP enrolment. A row with confirmed_at IS NULL is a started-but-unverified enrolment and must not gate login.';

CREATE TABLE IF NOT EXISTS user_recovery_code (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    code_hash VARCHAR(128) NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_recovery_code_user_id ON user_recovery_code(user_id);

COMMIT;
