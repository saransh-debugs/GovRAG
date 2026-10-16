-- Ingest can replace a tenant's encryption keys (needed for crypto-shredding).
-- 001 only had a SELECT policy, so the INSERT grant from 003 was blocked by RLS.

CREATE POLICY enc_ingest_insert ON vault.tenant_enc_keys FOR INSERT TO govrag_ingest
  WITH CHECK (tenant_id = core.req_tenant());
CREATE POLICY enc_ingest_delete ON vault.tenant_enc_keys FOR DELETE TO govrag_ingest
  USING (tenant_id = core.req_tenant());

GRANT DELETE ON vault.tenant_enc_keys TO govrag_ingest;
