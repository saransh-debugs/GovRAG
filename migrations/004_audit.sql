-- Durable audit log. Insert-only.
--
-- Why not just OpenTelemetry spans: spans are sampled (dropped on purpose), and the batch
-- exporter drops spans when its queue is full. A compliance record can't lose entries.
-- So the request writes here *before* returning an answer, and fails (503) if it can't.
--
-- No column holds free text. Every field is an ID, an enum, or a list of PII *types*.

CREATE TABLE audit.events (
  id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  ts                  timestamptz NOT NULL DEFAULT now(),
  tenant_id           text NOT NULL,
  request_id          uuid NOT NULL,
  user_id             text NOT NULL,
  action              text NOT NULL CHECK (action IN
                        ('retrieve', 'rehydrate', 'ingest', 'delete', 'denied',
                         'policy_change', 'policy_rollback', 'rescan', 'shred')),
  outcome             text NOT NULL CHECK (outcome IN ('ok', 'denied', 'error')),
  document_ids        uuid[] NOT NULL DEFAULT '{}',
  chunk_ids           uuid[] NOT NULL DEFAULT '{}',
  pii_types_present   text[] NOT NULL DEFAULT '{}',
  pii_types_unmasked  text[] NOT NULL DEFAULT '{}',
  policy_version      int,
  prev_policy_version int,
  model               text,
  error_code          text CHECK (error_code IS NULL OR error_code ~ '^[a-z0-9_]{1,64}$')
);
CREATE INDEX audit_tenant_ts ON audit.events (tenant_id, ts);

-- No login has UPDATE/DELETE. The trigger also stops the owner, so changing history
-- needs a deliberate DROP TRIGGER, which itself shows up in the DB logs.
CREATE FUNCTION audit.block_mutation() RETURNS trigger
  LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'audit.events is append-only' USING ERRCODE = 'insufficient_privilege';
END $$;

CREATE TRIGGER events_no_update BEFORE UPDATE OR DELETE ON audit.events
  FOR EACH ROW EXECUTE FUNCTION audit.block_mutation();
CREATE TRIGGER events_no_truncate BEFORE TRUNCATE ON audit.events
  FOR EACH STATEMENT EXECUTE FUNCTION audit.block_mutation();

ALTER TABLE audit.events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit.events FORCE ROW LEVEL SECURITY;

-- A request can only write audit rows for its own tenant.
CREATE POLICY events_insert ON audit.events FOR INSERT TO govrag_app, govrag_ingest
  WITH CHECK (tenant_id = core.req_tenant());

GRANT INSERT ON audit.events TO govrag_app, govrag_ingest;
