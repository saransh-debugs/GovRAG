-- Revoke a user's tokens before they expire.
--
-- A JWT is valid until exp, which is how a fired employee keeps access for up to an hour.
-- Revoking sets not_before = now(): any token for that (tenant, user) issued before then is
-- rejected. Every instance learns about it via NOTIFY in milliseconds (30 s worst case if a
-- notification is lost, same as policies).

CREATE TABLE core.revocations (
  tenant_id   text NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  subject     text NOT NULL CHECK (subject ~ '^[A-Za-z0-9_-]{1,128}$'),
  not_before  timestamptz NOT NULL,
  created_by  text NOT NULL,
  PRIMARY KEY (tenant_id, subject)
);

CREATE FUNCTION core.notify_revocation() RETURNS trigger
  LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_notify('revocations_changed', NEW.tenant_id);
  RETURN NEW;
END $$;
CREATE TRIGGER revocations_notify AFTER INSERT OR UPDATE ON core.revocations
  FOR EACH ROW EXECUTE FUNCTION core.notify_revocation();

ALTER TABLE core.revocations ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.revocations FORCE ROW LEVEL SECURITY;
-- Every request checks revocations before it knows anything else, so the service keeps
-- all of them in memory. Rows hold opaque user IDs only.
CREATE POLICY revocations_read ON core.revocations FOR SELECT TO govrag_app USING (true);
CREATE POLICY revocations_write ON core.revocations FOR INSERT TO govrag_app
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));
CREATE POLICY revocations_update ON core.revocations FOR UPDATE TO govrag_app
  USING (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()))
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));
GRANT SELECT, INSERT, UPDATE ON core.revocations TO govrag_app;

ALTER TABLE audit.events DROP CONSTRAINT events_action_check;
ALTER TABLE audit.events ADD CONSTRAINT events_action_check CHECK (action IN
  ('retrieve', 'rehydrate', 'ingest', 'delete', 'denied',
   'policy_change', 'policy_rollback', 'rescan', 'shred', 'dry_run', 'revoke'));
