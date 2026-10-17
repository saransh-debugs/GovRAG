-- Query path additions.
--
-- 1. The vault login reads token_refs. Rehydration only reveals tokens that ingest
--    *recorded* for a retrieved chunk. Matching token text is not enough: a document
--    author could paste "[US_SSN_<token from an exec-only doc>]" into their own document,
--    and a text match would then reveal that SSN to anyone who retrieves their chunk.
GRANT SELECT ON vault.token_refs TO govrag_vault;

-- 2. Policy dry runs get their own audit action.
ALTER TABLE audit.events DROP CONSTRAINT events_action_check;
ALTER TABLE audit.events ADD CONSTRAINT events_action_check CHECK (action IN
  ('retrieve', 'rehydrate', 'ingest', 'delete', 'denied',
   'policy_change', 'policy_rollback', 'rescan', 'shred', 'dry_run'));
