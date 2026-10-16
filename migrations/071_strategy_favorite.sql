-- 071: favourite strategies (quick access).
--
-- A per-strategy flag set by the owner from the Strategies page (star on the
-- card). The header dropdown lists only favourites and the Strategies page
-- pins them in a "Favourites" section above the per-account groups. Toggling
-- it is a UI preference, not an edit of the strategy: PUT
-- /strategies/{id}/favorite does not touch updated_at/updated_by and does not
-- publish strategy_updated. Copies start unstarred.
--
-- Run BEFORE deploying the backend (create_all adds tables, never columns).
BEGIN;

ALTER TABLE strategies
    ADD COLUMN IF NOT EXISTS is_favorite BOOLEAN NOT NULL DEFAULT false;

COMMENT ON COLUMN strategies.is_favorite IS 'Starred by the owner: header dropdown + pinned section on the Strategies page';

COMMIT;
