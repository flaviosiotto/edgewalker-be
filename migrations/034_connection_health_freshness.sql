-- Migration 034: Connection health freshness columns
--
-- Adds probe-freshness timestamps so the backend can detect stale "connected"
-- statuses (a gateway may report connected from a cached flag even after the
-- broker session has silently died). `last_checked_at` is written on every
-- health probe; `last_ok_at` only when the probe confirmed the connection is
-- alive. The API derives an `is_stale` flag from `last_checked_at` age.

BEGIN;

ALTER TABLE public.connections
    ADD COLUMN IF NOT EXISTS last_checked_at TIMESTAMPTZ NULL,
    ADD COLUMN IF NOT EXISTS last_ok_at      TIMESTAMPTZ NULL;

COMMIT;
