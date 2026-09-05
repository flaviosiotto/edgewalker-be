-- Piani di abbonamento (studio 05/09/2026, docs/piani-abbonamento-studio.md).
--
-- Il DB e' la fonte di verita' di piani, prezzi, coupon, trial e stato
-- dell'abbonamento; il provider di pagamento (Stripe, fase 3) sta dietro
-- l'interfaccia BillingProvider e i suoi id vivono SOLO in
-- billing_external_ref, mai come colonne delle tabelle di dominio.
--
-- I limiti sono un JSONB sul piano con chiavi tipizzate nel registro
-- app/services/limits.py (null = illimitato). La vista user_effective_limits
-- e' il contratto letto da studio-svc (stessa eccezione read-only di
-- user_lookup.py) per il cap Studi.
--
-- Procedura: backup -> dry-run ROLLBACK -> apply.

BEGIN;

CREATE TABLE plan (
    id SERIAL PRIMARY KEY,
    code VARCHAR(40) NOT NULL UNIQUE,
    name VARCHAR(120) NOT NULL,
    description TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_public BOOLEAN NOT NULL DEFAULT TRUE,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    trial_days INTEGER NOT NULL DEFAULT 0,
    limits JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Un solo piano di default (quello su cui cade chi non ha abbonamento).
CREATE UNIQUE INDEX uq_plan_default ON plan (is_default) WHERE is_default;

CREATE TABLE plan_price (
    id SERIAL PRIMARY KEY,
    plan_id INTEGER NOT NULL REFERENCES plan(id) ON DELETE CASCADE,
    interval VARCHAR(16) NOT NULL CHECK (interval IN ('month', 'quarter', 'semester', 'year')),
    amount_cents INTEGER NOT NULL CHECK (amount_cents >= 0),
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_plan_price_plan_interval UNIQUE (plan_id, interval)
);

CREATE TABLE billing_external_ref (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(32) NOT NULL,
    entity_id INTEGER NOT NULL,
    provider VARCHAR(32) NOT NULL,
    external_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_billing_external_ref_entity UNIQUE (entity_type, entity_id, provider),
    CONSTRAINT uq_billing_external_ref_external UNIQUE (provider, external_id)
);

CREATE TABLE coupon (
    id SERIAL PRIMARY KEY,
    code VARCHAR(40) NOT NULL UNIQUE,
    kind VARCHAR(16) NOT NULL CHECK (kind IN ('percent', 'fixed')),
    value INTEGER NOT NULL CHECK (value >= 0),
    currency CHAR(3),
    duration VARCHAR(16) NOT NULL DEFAULT 'once' CHECK (duration IN ('once', 'repeating', 'forever')),
    duration_months INTEGER,
    applies_to_plan_ids INTEGER[],
    max_redemptions INTEGER,
    redeemed_count INTEGER NOT NULL DEFAULT 0,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    revoked_at TIMESTAMPTZ,
    note TEXT,
    created_by INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE trial_grant (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    plan_id INTEGER NOT NULL REFERENCES plan(id) ON DELETE CASCADE,
    email_hash CHAR(64) NOT NULL,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_trial_grant_user_plan UNIQUE (user_id, plan_id),
    CONSTRAINT uq_trial_grant_email_plan UNIQUE (email_hash, plan_id)
);

CREATE TABLE subscription (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    plan_id INTEGER NOT NULL REFERENCES plan(id) ON DELETE RESTRICT,
    plan_price_id INTEGER REFERENCES plan_price(id) ON DELETE SET NULL,
    status VARCHAR(16) NOT NULL CHECK (status IN ('trialing', 'active', 'past_due', 'canceled', 'expired', 'free', 'manual')),
    interval VARCHAR(16),
    current_period_start TIMESTAMPTZ,
    current_period_end TIMESTAMPTZ,
    trial_end TIMESTAMPTZ,
    cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
    ends_at TIMESTAMPTZ,
    provider VARCHAR(32) NOT NULL DEFAULT 'none',
    coupon_id INTEGER REFERENCES coupon(id) ON DELETE SET NULL,
    ending_notice_sent_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_subscription_user_id ON subscription (user_id);
CREATE INDEX ix_subscription_status ON subscription (status);
-- Una sola subscription "corrente" per utente.
CREATE UNIQUE INDEX uq_subscription_current_user ON subscription (user_id)
    WHERE status IN ('trialing', 'active', 'past_due', 'free', 'manual');

CREATE TABLE coupon_redemption (
    id SERIAL PRIMARY KEY,
    coupon_id INTEGER NOT NULL REFERENCES coupon(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    subscription_id INTEGER REFERENCES subscription(id) ON DELETE SET NULL,
    redeemed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_coupon_redemption_coupon_id ON coupon_redemption (coupon_id);

CREATE TABLE subscription_event (
    id SERIAL PRIMARY KEY,
    subscription_id INTEGER REFERENCES subscription(id) ON DELETE SET NULL,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    type VARCHAR(48) NOT NULL,
    payload JSONB,
    provider VARCHAR(32),
    provider_event_id VARCHAR(255),
    actor_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_subscription_event_provider UNIQUE (provider, provider_event_id)
);
CREATE INDEX ix_subscription_event_user_id ON subscription_event (user_id);
CREATE INDEX ix_subscription_event_subscription_id ON subscription_event (subscription_id);

CREATE TABLE ai_model_rate (
    id SERIAL PRIMARY KEY,
    model_pattern VARCHAR(120) NOT NULL UNIQUE,
    input_per_1k NUMERIC(8, 3) NOT NULL DEFAULT 1.0,
    output_per_1k NUMERIC(8, 3) NOT NULL DEFAULT 1.0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE ai_credit_period (
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    period_key DATE NOT NULL,
    period_end DATE NOT NULL,
    granted NUMERIC(12, 3),
    used NUMERIC(12, 3) NOT NULL DEFAULT 0,
    low_notified_at TIMESTAMPTZ,
    exhausted_notified_at TIMESTAMPTZ,
    PRIMARY KEY (user_id, period_key)
);

CREATE TABLE ai_credit_ledger (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    period_key DATE NOT NULL,
    credits NUMERIC(12, 3) NOT NULL,
    reason VARCHAR(32) NOT NULL,
    model VARCHAR(120),
    tokens_input INTEGER,
    tokens_output INTEGER,
    correlation_id VARCHAR(100),
    session_id VARCHAR(100),
    estimated BOOLEAN NOT NULL DEFAULT FALSE,
    actor_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_ai_credit_ledger_user_period ON ai_credit_ledger (user_id, period_key);
-- Idempotenza dei report di uso: un turno = una riga (la stima viene
-- sovrascritta dal report con i token reali, mai duplicata).
CREATE UNIQUE INDEX uq_ai_credit_ledger_turn ON ai_credit_ledger (correlation_id, session_id)
    WHERE correlation_id IS NOT NULL;

-- Contratto letto da studio-svc (cap Studi) e dal backend.
CREATE OR REPLACE VIEW user_effective_limits AS
SELECT
    u.id AS user_id,
    p.id AS plan_id,
    p.code AS plan_code,
    p.limits AS limits,
    s.id AS subscription_id,
    s.status AS status,
    s.current_period_end AS current_period_end
FROM "user" u
LEFT JOIN subscription s
    ON s.user_id = u.id
   AND s.status IN ('trialing', 'active', 'past_due', 'free', 'manual')
LEFT JOIN plan dp ON dp.is_default
JOIN plan p ON p.id = COALESCE(s.plan_id, dp.id);

-- Seed (modificabile dalla console): Free di default + Pro pubblico.
INSERT INTO plan (code, name, description, is_active, is_public, is_default, sort_order, trial_days, limits)
VALUES
    ('free', 'Free', 'Per iniziare: una strategia live, un backtest alla volta, crediti AI mensili.',
     TRUE, TRUE, TRUE, 0, 0,
     '{"strategies_max": 3, "indicators_per_strategy_max": 6, "live_concurrent_max": 1,
       "backtest_concurrent_max": 1, "ai_credits_per_period": 300, "studios_max": 2,
       "studio_runs_concurrent_max": 1}'::jsonb),
    ('pro', 'Pro', 'Per chi fa trading sistematico ogni giorno: piu'' live, piu'' backtest, piu'' crediti AI.',
     TRUE, TRUE, FALSE, 10, 14,
     '{"strategies_max": 30, "indicators_per_strategy_max": 20, "live_concurrent_max": 5,
       "backtest_concurrent_max": 3, "ai_credits_per_period": 5000, "studios_max": 25,
       "studio_runs_concurrent_max": 3}'::jsonb);

INSERT INTO plan_price (plan_id, interval, amount_cents, currency)
SELECT p.id, v.interval, v.amount_cents, 'EUR'
FROM plan p
JOIN (VALUES ('month', 2900), ('quarter', 7900), ('semester', 14900), ('year', 27900))
     AS v(interval, amount_cents) ON TRUE
WHERE p.code = 'pro';

INSERT INTO ai_model_rate (model_pattern, input_per_1k, output_per_1k) VALUES ('*', 1.0, 1.0);

-- Ogni utente esistente parte dal piano di default.
INSERT INTO subscription (user_id, plan_id, status, provider, current_period_start)
SELECT u.id, p.id, 'free', 'none', now()
FROM "user" u
JOIN plan p ON p.is_default
WHERE NOT EXISTS (
    SELECT 1 FROM subscription s
    WHERE s.user_id = u.id
      AND s.status IN ('trialing', 'active', 'past_due', 'free', 'manual')
);

COMMIT;
