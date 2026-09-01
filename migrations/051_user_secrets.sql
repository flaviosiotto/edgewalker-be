-- Secrets di piattaforma per-utente (es. OPENROUTER_API_KEY): meccanismo
-- CROSS, non legato agli Studi. Gestione (UI/API) sul backend; i servizi
-- che devono consegnare i valori ai runtime (oggi studio-svc per i kernel
-- dei notebook) leggono la stessa tabella e decifrano con la chiave
-- condivisa SECRETS_ENCRYPTION_KEY (Fernet).
-- Procedura: backup -> dry-run ROLLBACK -> apply.

BEGIN;

CREATE TABLE user_secret (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value_encrypted TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_user_secret_user_name UNIQUE (user_id, name)
);

CREATE INDEX ix_user_secret_user_id ON user_secret (user_id);

COMMIT;
