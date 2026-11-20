-- 074: drop strategies.description.
--
-- The free-text description of a strategy left the workspace long ago but the
-- design agent kept writing it (update_strategy) and reading it back as an
-- "authoritative guide" block of its prompt: a second, invisible source of
-- instructions next to the definition, the playbook lessons and the Studi.
-- Dismissed on 21/11/2026: the API, the agent tools and the MCP tools no
-- longer carry it. Strategy TEMPLATES keep their own description (the blurb
-- shown in the catalogue): strategy_templates is not touched.
--
-- Run AFTER deploying backend, agent-svc and mcp-svc: the previous backend
-- still selects the column. The text is lost: export it first if wanted
--   \copy (SELECT id, name, description FROM strategies WHERE description IS NOT NULL) TO 'strategy_descriptions.csv' CSV HEADER
BEGIN;

ALTER TABLE strategies
    DROP COLUMN IF EXISTS description;

COMMIT;
