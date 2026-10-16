-- Token vault. token -> Fernet(original value), per tenant.
--
-- token     = "<TYPE>_<first 16 hex of HMAC(tenant_key, type || value)>", e.g. PERSON_3f9a1c2e4b7d9a01
-- value_mac = the full 32-byte HMAC. Used to detect the (~1e-8 at 1M values) case where two
--             different values share a 64-bit token. Without it, ON CONFLICT DO NOTHING would
--             silently map the second person to the first person's name.

CREATE TABLE vault.pii_tokens (
  tenant_id    text  NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  token        text  NOT NULL CHECK (token ~ '^[A-Z][A-Z0-9_]*_[0-9a-f]{16}$'),
  entity_type  text  NOT NULL,
  value_mac    bytea NOT NULL CHECK (length(value_mac) = 32),
  ciphertext   bytea NOT NULL,
  key_version  int   NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, token)
);

-- Which chunks mention which token. Deleting a document cascades here; tokens left with
-- no refs are then deleted (right to be forgotten).
CREATE TABLE vault.token_refs (
  tenant_id  text NOT NULL,
  token      text NOT NULL,
  chunk_id   uuid NOT NULL,
  PRIMARY KEY (tenant_id, token, chunk_id),
  FOREIGN KEY (tenant_id, token) REFERENCES vault.pii_tokens(tenant_id, token) ON DELETE CASCADE,
  FOREIGN KEY (tenant_id, chunk_id) REFERENCES core.chunks(tenant_id, id) ON DELETE CASCADE
);
CREATE INDEX token_refs_chunk ON vault.token_refs (tenant_id, chunk_id);

ALTER TABLE vault.pii_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE vault.pii_tokens FORCE ROW LEVEL SECURITY;
ALTER TABLE vault.token_refs ENABLE ROW LEVEL SECURITY;
ALTER TABLE vault.token_refs FORCE ROW LEVEL SECURITY;

CREATE POLICY tokens_tenant ON vault.pii_tokens FOR ALL TO govrag_vault, govrag_ingest
  USING (tenant_id = core.req_tenant()) WITH CHECK (tenant_id = core.req_tenant());
CREATE POLICY refs_tenant ON vault.token_refs FOR ALL TO govrag_vault, govrag_ingest
  USING (tenant_id = core.req_tenant()) WITH CHECK (tenant_id = core.req_tenant());

-- Vault login: read ciphertext. Nothing else.
GRANT SELECT ON vault.pii_tokens TO govrag_vault;

-- Ingest login: insert and delete, but column-level SELECT only. It can see a token
-- exists and compare value_mac for collisions. It can never SELECT ciphertext.
GRANT INSERT, DELETE ON vault.pii_tokens TO govrag_ingest;
GRANT SELECT (tenant_id, token, entity_type, value_mac) ON vault.pii_tokens TO govrag_ingest;
GRANT SELECT, INSERT, DELETE ON vault.token_refs TO govrag_ingest;
GRANT INSERT ON vault.tenant_enc_keys TO govrag_ingest;

-- govrag_app gets nothing in this schema. Not even USAGE.
