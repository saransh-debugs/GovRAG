-- Documents and chunks. Text is stored redacted only. Embeddings are of redacted text only:
-- embeddings can be inverted back to text (Morris et al. 2023), so embedding raw text
-- would leak PII into the vectors.

CREATE TABLE core.documents (
  tenant_id     text NOT NULL REFERENCES core.tenants(id) ON DELETE CASCADE,
  id            uuid NOT NULL DEFAULT gen_random_uuid(),
  title         text NOT NULL,           -- redacted, same as chunk text
  allowed_roles text[] NOT NULL CHECK (cardinality(allowed_roles) > 0),
  created_by    text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id)
);

CREATE TABLE core.chunks (
  tenant_id      text NOT NULL,
  id             uuid NOT NULL DEFAULT gen_random_uuid(),
  document_id    uuid NOT NULL,
  ord            int  NOT NULL CHECK (ord >= 0),
  text_redacted  text NOT NULL,
  embedding      vector(384) NOT NULL,
  allowed_roles  text[] NOT NULL CHECK (cardinality(allowed_roles) > 0),
  pii_types      text[] NOT NULL DEFAULT '{}',
  redaction_rev  int  NOT NULL DEFAULT 0,  -- bumped by the extra_redact re-scan job
  PRIMARY KEY (tenant_id, id),
  UNIQUE (tenant_id, document_id, ord),
  FOREIGN KEY (tenant_id, document_id)
    REFERENCES core.documents(tenant_id, id) ON DELETE CASCADE
);

-- One HNSW index across all tenants. RLS filters after the index scan, so a small tenant
-- or a rare role would get fewer than k results. The query sets hnsw.iterative_scan
-- so the scan keeps going until it has k visible rows (pgvector >= 0.8).
CREATE INDEX chunks_embedding_hnsw ON core.chunks
  USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_roles_gin ON core.chunks USING gin (allowed_roles);
CREATE INDEX chunks_doc ON core.chunks (tenant_id, document_id);

-- A chunk can never be visible to more roles than its document.
-- Ingest computes the intersection in Python; this checks it again in the DB, on purpose.
CREATE FUNCTION core.check_chunk_acl() RETURNS trigger
  LANGUAGE plpgsql AS $$
DECLARE doc_roles text[];
BEGIN
  SELECT allowed_roles INTO doc_roles
    FROM core.documents WHERE tenant_id = NEW.tenant_id AND id = NEW.document_id;
  -- IS NOT TRUE, not NOT: if the document isn't visible, doc_roles is NULL and
  -- `NOT (x <@ NULL)` is NULL, which plpgsql treats as false and would let the row in.
  IF (NEW.allowed_roles <@ doc_roles) IS NOT TRUE THEN
    RAISE EXCEPTION 'chunk allowed_roles must be a subset of document allowed_roles'
      USING ERRCODE = 'check_violation';
  END IF;
  RETURN NEW;
END $$;

CREATE TRIGGER chunks_acl BEFORE INSERT OR UPDATE OF allowed_roles, document_id
  ON core.chunks FOR EACH ROW EXECUTE FUNCTION core.check_chunk_acl();

ALTER TABLE core.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.documents FORCE ROW LEVEL SECURITY;
ALTER TABLE core.chunks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE core.chunks    FORCE ROW LEVEL SECURITY;

-- Query path: right tenant AND at least one shared role.
CREATE POLICY documents_read ON core.documents FOR SELECT TO govrag_app
  USING (tenant_id = core.req_tenant() AND allowed_roles && core.req_roles());
CREATE POLICY chunks_read ON core.chunks FOR SELECT TO govrag_app
  USING (tenant_id = core.req_tenant() AND allowed_roles && core.req_roles());

-- Ingest is a service path: it sees and writes its whole tenant, never another one.
CREATE POLICY documents_ingest ON core.documents FOR ALL TO govrag_ingest
  USING (tenant_id = core.req_tenant()) WITH CHECK (tenant_id = core.req_tenant());
CREATE POLICY chunks_ingest ON core.chunks FOR ALL TO govrag_ingest
  USING (tenant_id = core.req_tenant()) WITH CHECK (tenant_id = core.req_tenant());

GRANT SELECT ON core.documents, core.chunks TO govrag_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON core.documents, core.chunks TO govrag_ingest;
