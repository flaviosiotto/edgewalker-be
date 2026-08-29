-- 050: neutral defaults for the agent persona (follow-up to 049).
--
-- Flavio's review (29/08/2026): the avatar library is a set of human
-- characters on a NEUTRAL background, not purple. New defaults:
--   avatar        'analyst'  (first character of the FE set)
--   accent_color  '#6c757d'  (Bootstrap secondary grey)
-- The three agents created before this change carried the 049 defaults
-- ('robot', '#6f42c1'): they are moved to the new ones. Nothing else changes.
--
-- Safe to re-run.

BEGIN;

ALTER TABLE agent ALTER COLUMN avatar       SET DEFAULT 'analyst';
ALTER TABLE agent ALTER COLUMN accent_color SET DEFAULT '#6c757d';

UPDATE agent SET avatar = 'analyst'      WHERE avatar = 'robot';
UPDATE agent SET accent_color = '#6c757d' WHERE accent_color = '#6f42c1';

COMMENT ON COLUMN agent.avatar IS
'Chiave del set di personaggi SVG del FE (analyst, quant, veteran, executive, strategist, portfolio_manager, risk_officer, senior_partner, macro, chartist, associate, advisor)';

COMMIT;
