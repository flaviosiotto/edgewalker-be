-- Credito piattaforma (wallet prepagato) + costo reale del provider LLM
-- (studio 13/09/2026, docs/credito-piattaforma-studio.md).
--
-- Il wallet e' un saldo in centesimi per utente con ledger dei movimenti; il
-- DB e' l'unica verita' (Stripe accredita via webhook, mai letto per il
-- saldo). Quando i crediti AI del periodo sono esauriti, l'eccedenza di un
-- turno viene addebitata al wallet al prezzo per credito deciso dall'admin.
--
-- Il costo reale del provider (OpenRouter oggi) viene salvato per turno nel
-- ledger dei crediti: nessun listino da mantenere, `provider` e' sempre un
-- dato esplicito (nessun lock-in).
--
-- Procedura: backup -> dry-run ROLLBACK -> apply.

BEGIN;

-- ---------------------------------------------------------------------------
-- Costo reale del provider per turno
-- ---------------------------------------------------------------------------
ALTER TABLE ai_credit_ledger
    ADD COLUMN IF NOT EXISTS provider VARCHAR(40),
    ADD COLUMN IF NOT EXISTS cost NUMERIC(12, 6),
    ADD COLUMN IF NOT EXISTS cost_currency CHAR(3),
    ADD COLUMN IF NOT EXISTS wallet_cents INTEGER;

COMMENT ON COLUMN ai_credit_ledger.provider IS 'Provider LLM che ha servito il turno (openrouter, ...); NULL per stime o righe amministrative';
COMMENT ON COLUMN ai_credit_ledger.cost IS 'Costo fatturato dal provider per il turno, nella sua valuta; NULL se non riportato';
COMMENT ON COLUMN ai_credit_ledger.wallet_cents IS 'Centesimi addebitati al credito piattaforma per l''eccedenza di questo turno (0/NULL = tutto dal piano)';

CREATE INDEX IF NOT EXISTS ix_ai_credit_ledger_created ON ai_credit_ledger (created_at);
CREATE INDEX IF NOT EXISTS ix_ai_credit_ledger_provider_model ON ai_credit_ledger (provider, model);

-- ---------------------------------------------------------------------------
-- Impostazioni (riga unica)
-- ---------------------------------------------------------------------------
CREATE TABLE platform_credit_settings (
    id SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    -- prezzo di 1 credito AI in centesimi (1.0000 = 1 centesimo = 100 crediti/EUR)
    price_per_ai_credit_cents NUMERIC(10, 4) NOT NULL DEFAULT 1.0,
    low_balance_cents INTEGER NOT NULL DEFAULT 100,
    min_topup_cents INTEGER NOT NULL DEFAULT 500,
    -- tasso di sola visualizzazione per il report costi (EUR per 1 USD); NULL = non convertire
    display_fx_eur_per_usd NUMERIC(10, 6),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by INTEGER REFERENCES "user"(id) ON DELETE SET NULL
);

INSERT INTO platform_credit_settings (id) VALUES (1);

-- ---------------------------------------------------------------------------
-- Pacchetti di ricarica (prodotti one-off sul provider di pagamento)
-- ---------------------------------------------------------------------------
CREATE TABLE credit_pack (
    id SERIAL PRIMARY KEY,
    name VARCHAR(80) NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),      -- prezzo pagato
    credit_cents INTEGER NOT NULL CHECK (credit_cents > 0),      -- credito accreditato (prezzo + bonus)
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO credit_pack (name, amount_cents, credit_cents, sort_order) VALUES
    ('5 EUR', 500, 500, 1),
    ('10 EUR', 1000, 1050, 2),
    ('25 EUR', 2500, 2750, 3);

-- ---------------------------------------------------------------------------
-- Wallet e movimenti
-- ---------------------------------------------------------------------------
CREATE TABLE user_wallet (
    user_id INTEGER PRIMARY KEY REFERENCES "user"(id) ON DELETE CASCADE,
    balance_cents BIGINT NOT NULL DEFAULT 0,
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    auto_use_for_ai BOOLEAN NOT NULL DEFAULT TRUE,
    low_notified_at TIMESTAMPTZ,
    exhausted_notified_period DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE wallet_topup (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    pack_id INTEGER REFERENCES credit_pack(id) ON DELETE SET NULL,
    amount_cents INTEGER NOT NULL,
    credit_cents INTEGER NOT NULL,
    currency CHAR(3) NOT NULL DEFAULT 'EUR',
    status VARCHAR(16) NOT NULL DEFAULT 'pending',   -- pending | paid | canceled
    provider VARCHAR(20) NOT NULL,
    checkout_external_id VARCHAR(120) NOT NULL UNIQUE,
    payment_external_id VARCHAR(120),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    paid_at TIMESTAMPTZ
);
CREATE INDEX ix_wallet_topup_user ON wallet_topup (user_id, created_at);

CREATE TABLE wallet_ledger (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
    amount_cents INTEGER NOT NULL,                 -- positivo = accredito, negativo = addebito
    balance_after_cents BIGINT NOT NULL,
    kind VARCHAR(20) NOT NULL,                     -- topup | admin_adjust | ai_overage | refund
    ai_credits NUMERIC(12, 3),                     -- crediti AI coperti (ai_overage)
    ai_ledger_id INTEGER REFERENCES ai_credit_ledger(id) ON DELETE SET NULL,
    topup_id INTEGER REFERENCES wallet_topup(id) ON DELETE SET NULL,
    note TEXT,
    actor_user_id INTEGER REFERENCES "user"(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_wallet_ledger_user_created ON wallet_ledger (user_id, created_at);
-- un solo addebito per turno: la rettifica stima -> reale aggiorna la riga
CREATE UNIQUE INDEX uq_wallet_ledger_ai_ledger ON wallet_ledger (ai_ledger_id) WHERE ai_ledger_id IS NOT NULL;

COMMIT;
