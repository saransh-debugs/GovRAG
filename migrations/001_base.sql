-- Base: extensions, schemas, login roles, tenants, per-tenant keys.
--
-- Four logins, so a bug in one code path can only touch what that path needs:
--   govrag_owner  : owns every table, runs migrations. Never used by the app.
--   govrag_app    : query path. Reads chunks under RLS. Cannot read the vault.
--   govrag_vault  : rehydration. Reads vault ciphertext + tenant enc keys. Nothing else.
--   govrag_ingest : writes chunks + vault rows. Cannot read vault ciphertext.
-- Passwords are set by scripts/migrate.py from env, never stored in SQL.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
DECLARE r text;
BEGIN
  FOREACH r IN ARRAY ARRAY['govrag_app', 'govrag_vault', 'govrag_ingest'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
      EXECUTE format('CREATE ROLE %I LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE', r);
    END IF;
  END LOOP;
END $$;

-- Nobody gets anything by default. Every grant below is explicit.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

CREATE SCHEMA core;
CREATE SCHEMA vault;
CREATE SCHEMA audit;

GRANT USAGE ON SCHEMA core  TO govrag_app, govrag_ingest, govrag_vault;
GRANT USAGE ON SCHEMA vault TO govrag_vault, govrag_ingest;
GRANT USAGE ON SCHEMA audit TO govrag_app, govrag_ingest;
-- The <=> operator lives in public (pgvector), so the query path needs USAGE there.
GRANT USAGE ON SCHEMA public TO govrag_app, govrag_ingest;

-- Request context. Set per transaction with set_config(..., true) by govrag/db.py.
-- Unset or empty -> NULL -> every RLS predicate below is NULL -> row hidden. Fails closed.
-- Plain SQL + STABLE so the planner inlines them instead of calling a function per row.
CREATE FUNCTION core.req_tenant() RETURNS text
  LANGUAGE sql STABLE PARALLEL SAFE
  AS $$ SELECT NULLIF(current_setting('app.tenant', true), '') $$;

CREATE FUNCTION core.req_roles() RETURNS text[]
  LANGUAGE sql STABLE PARALLEL SAFE
  AS $$ SELECT NULLIF(current_setting('app.roles', true), '')::text[] $$;

GRANT EXECUTE ON FUNCTION core.req_tenant(), core.req_roles()
  TO govrag_app, govrag_ingest, govrag_vault;

CREATE TABLE core.tenants (
  id         text PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
  created_at timestamptz NOT NULL DEFAULT now()
);

-- HMAC key per tenant, wrapped (encrypted) by the master key from env.
-- Needed by the query path (to tokenize the user's question) and by ingest.
CREATE TABLE core.tenant_hmac_keys (
  tenant_id    text PRIMARY KEY REFERENCES core.tenants(id) ON DELETE CASCADE,
  wrapped_key  bytea NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);

-- Encryption keys per tenant, wrapped by the master key. Versioned for rotation.
-- Envelope encryption: rotating the master key re-wraps N tenant keys,
-- not millions of vault rows. Deleting a tenant's rows here crypto-shreds its PII.
CREATE TABLE vault.tenant_enc_keys (
  tenant_id    text NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  key_version  int  NOT NULL CHECK (key_version > 0),
  wrapped_key  bytea NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, key_version)
);

ALTER TABLE core.tenants          ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.tenants          FORCE ROW LEVEL SECURITY;
ALTER TABLE core.tenant_hmac_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.tenant_hmac_keys FORCE ROW LEVEL SECURITY;
ALTER TABLE vault.tenant_enc_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE vault.tenant_enc_keys FORCE ROW LEVEL SECURITY;

CREATE POLICY tenants_own ON core.tenants FOR SELECT
  TO govrag_app, govrag_ingest, govrag_vault
  USING (id = core.req_tenant());

CREATE POLICY hmac_own ON core.tenant_hmac_keys FOR SELECT
  TO govrag_app, govrag_ingest
  USING (tenant_id = core.req_tenant());

CREATE POLICY enc_own ON vault.tenant_enc_keys FOR SELECT
  TO govrag_vault, govrag_ingest
  USING (tenant_id = core.req_tenant());

GRANT SELECT ON core.tenants          TO govrag_app, govrag_ingest, govrag_vault;
GRANT SELECT ON core.tenant_hmac_keys TO govrag_app, govrag_ingest;
GRANT SELECT ON vault.tenant_enc_keys TO govrag_vault, govrag_ingest;
