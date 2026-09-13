-- Fase 2 dismissione n8n (docs/valutazione-dismissione-n8n.md §10):
--
-- 1. `agent.settings` JSONB: come si comporta l'agent, scelto dall'utente
--    nel form /agents. Chiavi: reasoning (quick|balanced|deep, default
--    balanced), autonomy (propose|execute, default execute), include_chart,
--    lessons_enabled (default true). Il modello LLM NON e' fra le chiavi:
--    e' politica di piattaforma (punto 2).
-- 2. `ai_model_policy`: mappa (piano, livello di ragionamento) -> modello
--    reale, reasoning_effort del provider, iterazioni e finestra di memoria.
--    plan_code '*' = qualunque piano (fallback). Una riga INATTIVA per un
--    piano dice "questo livello non e' disponibile su quel piano": agent-svc
--    degrada (deep -> balanced -> quick). Riga assente = si usa '*'.
--    Il seed riproduce ESATTAMENTE il comportamento attuale al livello
--    balanced (gemini-3.7-flash, effort del provider, 15 iterazioni, 12 righe).
-- 3. `ai_credit_ledger.tokens_reasoning/tokens_cached`: dettaglio informativo
--    (i token di reasoning restano contati in tokens_output per la tariffa).
--
-- Additiva: il BE vecchio ignora le colonne nuove. Procedura: backup ->
-- dry-run ROLLBACK -> apply.

BEGIN;

ALTER TABLE agent
    ADD COLUMN IF NOT EXISTS settings JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS ai_model_policy (
    id SERIAL PRIMARY KEY,
    plan_code VARCHAR(40) NOT NULL DEFAULT '*',
    tier VARCHAR(16) NOT NULL CHECK (tier IN ('quick', 'balanced', 'deep')),
    provider VARCHAR(40) NOT NULL DEFAULT 'openrouter',
    model VARCHAR(120) NOT NULL,
    reasoning_effort VARCHAR(16) CHECK (reasoning_effort IS NULL OR reasoning_effort IN ('low', 'medium', 'high')),
    max_iterations INTEGER NOT NULL DEFAULT 15 CHECK (max_iterations BETWEEN 1 AND 60),
    history_window INTEGER NOT NULL DEFAULT 12 CHECK (history_window BETWEEN 0 AND 60),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_ai_model_policy_plan_tier UNIQUE (plan_code, tier)
);

INSERT INTO ai_model_policy (plan_code, tier, provider, model, reasoning_effort, max_iterations, history_window, notes)
VALUES
    ('*', 'quick',    'openrouter', 'google/gemini-3.7-flash', 'low',  10, 8,  'Risposte brevi, meno iterazioni e memoria corta.'),
    ('*', 'balanced', 'openrouter', 'google/gemini-3.7-flash', NULL,   15, 12, 'Comportamento di riferimento (= agent-svc fase 1).'),
    ('*', 'deep',     'openrouter', 'google/gemini-3.7-flash', 'high', 20, 16, 'Ragionamento esteso: piu token e piu iterazioni.')
ON CONFLICT (plan_code, tier) DO NOTHING;

ALTER TABLE ai_credit_ledger
    ADD COLUMN IF NOT EXISTS tokens_reasoning INTEGER,
    ADD COLUMN IF NOT EXISTS tokens_cached INTEGER;

COMMIT;
