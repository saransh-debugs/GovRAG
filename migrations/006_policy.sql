-- Per-tenant policies: immutable versions + a pointer to the live one.
-- Plus the job table for re-scanning old chunks when a tenant adds an extra_redact type.

CREATE SCHEMA policy;
GRANT USAGE ON SCHEMA policy TO govrag_app, govrag_ingest;

CREATE TABLE policy.versions (
  tenant_id   text NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  version     int  NOT NULL CHECK (version > 0),
  doc         jsonb NOT NULL,
  created_by  text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, version)
);

CREATE TABLE policy.active (
  tenant_id   text PRIMARY KEY REFERENCES core.tenants(id) ON DELETE CASCADE,
  version     int  NOT NULL,
  updated_by  text NOT NULL,
  updated_at  timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (tenant_id, version) REFERENCES policy.versions(tenant_id, version)
);

-- Versions never change. Rollback = point `active` at an old version.
CREATE FUNCTION policy.block_mutation() RETURNS trigger
  LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'policy.versions is immutable' USING ERRCODE = 'insufficient_privilege';
END $$;
CREATE TRIGGER versions_immutable BEFORE UPDATE OR DELETE ON policy.versions
  FOR EACH ROW EXECUTE FUNCTION policy.block_mutation();

-- Tell every app instance which tenant changed. NOTIFY is delivered on COMMIT, so a
-- listener never sees a change that later rolls back.
CREATE FUNCTION policy.notify_change() RETURNS trigger
  LANGUAGE plpgsql AS $$
BEGIN
  PERFORM pg_notify('policy_changed', NEW.tenant_id);
  RETURN NEW;
END $$;
CREATE TRIGGER active_notify AFTER INSERT OR UPDATE ON policy.active
  FOR EACH ROW EXECUTE FUNCTION policy.notify_change();

ALTER TABLE policy.versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy.versions FORCE ROW LEVEL SECURITY;
ALTER TABLE policy.active   ENABLE ROW LEVEL SECURITY;
ALTER TABLE policy.active   FORCE ROW LEVEL SECURITY;

-- Everyone in the tenant can read its policy (every request needs it).
CREATE POLICY versions_read ON policy.versions FOR SELECT TO govrag_app, govrag_ingest
  USING (tenant_id = core.req_tenant());
CREATE POLICY active_read ON policy.active FOR SELECT TO govrag_app, govrag_ingest
  USING (tenant_id = core.req_tenant());

-- Only a policy_admin of that tenant can write. The API checks the role too;
-- this checks it again in the DB, on purpose.
CREATE POLICY versions_write ON policy.versions FOR INSERT TO govrag_app
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));
CREATE POLICY active_insert ON policy.active FOR INSERT TO govrag_app
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));
CREATE POLICY active_update ON policy.active FOR UPDATE TO govrag_app
  USING (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()))
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));

GRANT SELECT ON policy.versions, policy.active TO govrag_app, govrag_ingest;
GRANT INSERT ON policy.versions TO govrag_app;
GRANT INSERT, UPDATE ON policy.active TO govrag_app;

-- Re-scan jobs. Created by the policy write (app login), worked by the ingest login.
-- lease_until: a worker that crashes mid-job loses its lease and another one picks it up.
-- Progress lives on the chunks (redaction_rev), so a resumed job skips finished documents.
-- cursor_doc / docs_done / chunks_done are for showing progress only.
CREATE TABLE core.rescan_jobs (
  tenant_id     text NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  id            uuid NOT NULL DEFAULT gen_random_uuid(),
  types         text[] NOT NULL CHECK (cardinality(types) > 0),
  target_rev    int  NOT NULL CHECK (target_rev > 0),
  status        text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done')),
  cursor_doc    uuid,
  docs_done     int  NOT NULL DEFAULT 0,
  chunks_done   int  NOT NULL DEFAULT 0,
  lease_until   timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id)
);
CREATE INDEX rescan_claim ON core.rescan_jobs (status, created_at) WHERE status <> 'done';
-- The re-scan finds work by "chunks scanned under an older policy version", which makes it
-- resumable after a crash and catches documents ingested while it was running.
CREATE INDEX chunks_redaction_rev ON core.chunks (tenant_id, redaction_rev);

ALTER TABLE core.rescan_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.rescan_jobs FORCE ROW LEVEL SECURITY;
CREATE POLICY rescan_create ON core.rescan_jobs FOR INSERT TO govrag_app
  WITH CHECK (tenant_id = core.req_tenant() AND 'policy_admin' = ANY(core.req_roles()));
CREATE POLICY rescan_read_app ON core.rescan_jobs FOR SELECT TO govrag_app
  USING (tenant_id = core.req_tenant());
-- The worker claims jobs across tenants. Job rows hold only IDs and type names.
CREATE POLICY rescan_worker ON core.rescan_jobs FOR ALL TO govrag_ingest
  USING (true) WITH CHECK (true);
GRANT SELECT, INSERT ON core.rescan_jobs TO govrag_app;
GRANT SELECT, UPDATE ON core.rescan_jobs TO govrag_ingest;
