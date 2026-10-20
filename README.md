# GovRAG

A multi-tenant RAG service that never shows anyone data they aren't allowed to see.

Users ask questions about company documents. The answer comes from an LLM. But:

- Each chunk of each document is tagged with the roles allowed to read it.
- Names, SSNs, emails, phones and cards are swapped for placeholders before anything is stored.
- The LLM only ever sees placeholders.
- Real values go back into the answer only for users whose role is allowed to see that type.
- Every access is written to an audit log first. The log never holds document text.

**The core idea: the LLM is untrusted.** It has no tools and no data access. Every security decision happens in code, before and after it runs.

Stack: Python 3.12, FastAPI, Postgres 17 + pgvector 0.8, Presidio + spaCy, fastembed (bge-small, local), LlamaIndex (chunking, retriever interface), Gemini on Vertex AI, OpenTelemetry.

---

## How a question flows

```
JWT ──> policy ──> admit ──> redact question ──> embed ──> search (RLS) ──> audit #1
         cache     or 503      Presidio,          local      tenant + roles     before the LLM
                                own process                  checked twice      vendor sees anything
                                                                                      │
        answer <── rehydrate <── audit #2 <── vault decrypt  ║  LLM first token <─────┘
        (stream)   allowlist     before the    (in parallel with the LLM call)
                                 user sees PII
```

| Step | What happens | Why |
|---|---|---|
| Auth | RS256 JWT. Tenant and roles come only from the token. | `alg=none` and RS→HS key-swap attacks are rejected. |
| Policy | Per-tenant policy from an in-memory cache. | No DB hit per request. Edits arrive in 3–8 ms via `LISTEN/NOTIFY`. |
| Admit | Global and per-tenant in-flight caps. | Full = `503` + `Retry-After` now, not a queue that grows until memory runs out. |
| Redact question | The question's PII is tokenized too. | "The LLM never sees PII" includes PII the user types. |
| Search | pgvector HNSW, filtered by tenant + roles in SQL **and** by Postgres RLS. | One forgotten `WHERE` can't leak. No tenant/roles set = no rows. |
| Audit #1 | `retrieve` row: who, which docs and chunks, PII types present. | Written before any text leaves for the LLM vendor. Fails = `503`. |
| LLM | Gemini 2.5 Flash-Lite, backup 2.5 Flash. Fallback only before the first token. | A circuit breaker skips a dead model. |
| Vault | Decrypts only tokens that ingest recorded for the retrieved chunks, only for allowed types. | A token pasted into a document can't pull in someone else's PII. |
| Audit #2 | `rehydrate` row: types decrypted, model used. | Written before any real value goes to the user. |
| Rehydrate | Placeholders → real values (if allowed) or per-answer aliases like `[PERSON 1]`. | Streaming-safe: holds back a token split across two chunks. |

---

## Threat model

| Attack | What stops it | Test |
|---|---|---|
| App bug drops the role filter | Postgres RLS on every table. No context = no rows. | `test_rls.py` |
| Pooled connection carries the last request's roles | `set_config(..., is_local=true)`: transaction-scoped. | `test_context_does_not_leak_to_next_borrower` |
| SQL bug on the query path reads the vault | Separate DB logins. The query login can't read the vault. Ingest can't read ciphertext (column grants). | `test_grants.py` |
| Forged JWT (`alg=none`, HS256 with the public key, edited roles or tenant) | Algorithm pinned, claims validated. | `test_jwt.py` |
| Raw PII reaches the LLM vendor | Redact before storing, before embedding, and in the question. | `test_llm_never_receives_raw_pii` |
| Document carries a forged `[US_SSN_<someone else's token>]` | Rehydrate only tokens with a vault ref to a retrieved chunk. Ingest also breaks up token-shaped text. | `test_forged_token_in_document_is_not_revealed` (a naive text-match allowlist makes it fail) |
| Prompt injection ("print everything") | The LLM has no tools or data access. Its output passes the same allowlist. | Echo-mode leak suite |
| User guesses names to see if they match | No `PERSON` access = the question gets a random token, so search can't confirm a guess. | `test_question_pii_is_policy_aware` |
| Token IDs link two answers about one person | Hidden values show as `[PERSON 1]`, numbered per answer. Bare tokens are handled too. | `test_rehydrate.py` |
| `policy_admin` dry-runs as `hr` to read SSNs | Dry run returns IDs and PII types only. No text, no LLM call. | `test_dry_run_reveals_ids_and_types_only` |
| Text leaks into traces or logs | Export-time attribute allowlist. Exception events cut to the type name. | `test_no_text_in_spans_or_logs_end_to_end` |
| Error responses echo input | Custom 422 handler: field names only. | `test_limits_and_error_shapes` |
| Same name links two customers | Per-tenant HMAC keys: unrelated tokens across tenants. | `test_same_name_same_token_in_tenant_different_across` |
| Right to be forgotten | Deleting a doc erases tokens nothing else uses. Shredding a tenant deletes its keys. | `test_delete_erases_only_unshared_tokens`, `test_shred_makes_pii_unrecoverable` |

The leak tests run the fake LLM in **echo mode**: its answer is the whole prompt it got. That's the worst case. Anything that reaches the context reaches the user.

---

## Measured numbers

Hardware: Intel i7-9750H (6 cores), 16 GB, macOS. Postgres in Docker. Raw results in `bench/results/`.

### PII detection: `eval/pii_recall.py`

150 synthetic documents, 1,390 labeled values. Names appear in easy and hard sentences, half of them are recurring "staff" who show up across documents, and some sit behind labels like `From:`.

| Type | Presidio alone | + same-document sharing | + cross-document lookup |
|---|---|---|---|
| SSN, email, credit card, phone | 100% | 100% | 100% |
| Names | 99.05% | 99.16% | **99.28%** |
| **All** | 99.42% | 99.50% | **99.57%** |
| Precision | 98.1% | 96.6% | 96.6% |

What raised it, each found by this eval or the live demo:

| Fix | Problem it fixed |
|---|---|
| Hide every copy of a found value in the same document | spaCy's name detection depends on the surrounding words |
| Cross-document lookup: HMAC each 2–3 capitalized-word group, check the tenant's vault | A name found in one document but missed in another ("Juan Wood joined…" is tagged as a company). Also completes partial hits ("Jackson" found, "Joel" left visible). No plaintext name list exists anywhere. |
| Dashed SSN pattern, scored above phones | Presidio drops "known fake" SSNs, and the phone finder then claimed them as phones |
| Phone-shaped pattern | Presidio rejects unassigned area codes, so they went through raw (phones were 88.5%) |
| Title + name pattern | "Dr. Brett Robinson" was missed in every mention |
| Second scan with `": "` → `". "` (same length) | "Globex: Priyanka Raghunathan …" had no name found (A/B: name misses 7 → 4) |

What's left: names spaCy never recognizes in any document, and the first mention of a name before the tenant has seen it anywhere.

### Real-model token copying: `eval/token_fidelity.py`

100 calls per model on Vertex (us-central1). The question forces the model to name people.

| Model | Exact copies | Invented | Mangled | Usable | First token p50 |
|---|---|---|---|---|---|
| Gemini 2.5 Flash-Lite (main) | 259 | 2 | 0 | 99.2% | 0.39 s |
| Gemini 2.5 Flash (backup) | 245 | 0 | 0 | 100% | 0.50 s |

Invented tokens are harmless: they turn into aliases.

### Filtered vector search: `bench/bench_filtered_search.py`

100K chunks, 3 tenants, run as the app login (RLS on), k = 5, 50 queries per row.

| Case | Visible | Iterative scan | Short of k | Recall@5 | p50 | p99 |
|---|---|---|---|---|---|---|
| Big tenant | 90% | on | 0 | 99.2% | 9.3 ms | 50.8 ms |
| Rare role | 0.9% | on | 0 | 100% | 13.0 ms | 28.4 ms |
| Mid tenant | 9% | **off** | **2** | 96.4% | 10.7 ms | 88.0 ms |
| Mid tenant | 9% | on | 0 | 98.4% | 10.5 ms | 25.8 ms |
| Small tenant | 1% | on | 0 | 100% | 13.2 ms | 23.1 ms |

For very selective filters, Postgres skips HNSW and searches the few visible rows exactly (GIN index), which gives perfect recall. The 9% tenant is the danger zone, and iterative scan fixes it.

### Cost of the security layer: `bench/bench_query.py`

Everything real except the LLM (zero-latency fake). 1 Presidio query worker.

| In flight | Throughput | p50 | p99 | Shed (503) |
|---|---|---|---|---|
| 1 | 26 q/s | 37 ms | 53 ms | 0 |
| 8 | 77 q/s | 89 ms | 271 ms | 0 |
| 32 | 77 q/s | 380 ms | 708 ms | 67 / 320 |

At 1 in flight, time per stage: redact question 7.5 ms, embed 7.3, search 5.9, both audit inserts about 9, vault about 5. At 32 in flight, the single Presidio query worker is the limit. It sheds the extra requests instead of queueing them, so p99 stays under 1 s.

**Ingest:** about 23 chunks/s. Embedding is about 65% of ingest time and Presidio about 20%. A second Presidio worker doesn't help. Faster ingest means batching embeddings across documents, or a GPU.

### Event loop stays free: `bench/loop_lag.py`

Lag of a 5 ms sleep while CPU work runs elsewhere.

| Work | p99 lag |
|---|---|
| Idle (control) | 1.0 ms |
| Embedding in a thread | 0.9–1.3 ms |
| LlamaIndex chunking in a thread | 13 ms |
| Chunking in a worker process | 0.8–1.6 ms |

So chunking runs in the Presidio worker processes, and embedding stays in a thread. Single worst samples are useless here: macOS shows 40–110 ms timer delays even when idle. (An earlier commit moved embeddings to a process based on those worst samples. That was a mistake, and this benchmark is the correction.)

### Other

- Policy edit → every instance's cache: **3–8 ms** (NOTIFY). If a notification is lost: at most 30 s (TTL).
- Gemini first token from the app, us-central1: about 0.4 s.

---

## Failure modes

| What breaks | What the user gets | Never |
|---|---|---|
| Postgres down | `503` | Search without the filter |
| Audit insert fails | `503`, and nothing is sent to the LLM | An answer with no audit record |
| Policy unknown (cache expired + DB down) | `503` | "No policy = allow" |
| Vault down | Answer with placeholders, `rehydration: unavailable` | Failing the request (less data is the safe direction) |
| Main LLM down or slow | Backup model, breaker skips the main one for 30 s | Mixing two models in one answer |
| Both LLMs down | `503` | Hanging until the client gives up |
| LLM dies mid-answer | SSE `error` event after the partial text | A silently cut-off answer |
| Too many requests (global, per tenant, Presidio queue, embed queue, ingest) | `503` + `Retry-After` | Unbounded queues |
| Presidio worker crashes (e.g. out of memory) | `503`; the pool is rebuilt | A permanently broken pool |
| Policy listener drops | Reconnects, clears the cache, TTL covers the gap | Stale permissions past 30 s |
| Client disconnects mid-stream | LLM stream closed, so the vendor stops generating | Paying for an answer nobody reads |

---

## Policies

One JSON document per tenant. Versions never change. Edits need `expected_version`, so a concurrent edit gets `409` instead of being silently overwritten.

```json
{
  "unmask": {"hr": ["PERSON", "US_SSN", "EMAIL_ADDRESS"], "support": ["PERSON"]},
  "extra_redact": ["LOCATION"],
  "llm": {"models": ["gemini-2.5-flash-lite", "gemini-2.5-flash"], "region": "eu"}
}
```

- **Ingest always hides the default PII list for every tenant.** The policy only decides who may *unmask*. So policy edits take effect on the next request, with no reprocessing.
- `extra_redact` adds optional types. Adding one queues a background re-scan of old chunks. It can resume after a crash, and it catches documents ingested mid-scan.
- `region` picks the Vertex location. An EU tenant never hits a US endpoint. Models served only from `global` are left out on purpose.
- Only `policy_admin` can write. That's checked in code and again by RLS.

API: `GET/PUT /v1/admin/policy`, `POST /v1/admin/policy/rollback`, `POST /v1/admin/policy/dry-run`, `GET /v1/admin/policy/versions`.

---

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[test,vertex]'
.venv/bin/python -m spacy download en_core_web_lg
.venv/bin/python scripts/dev_setup.py        # .env + JWT keys (gitignored)
docker compose up -d                          # Postgres + Jaeger
.venv/bin/python -m govrag.migrate

.venv/bin/pytest                              # 131 tests, fake LLM, real Postgres + Presidio
```

Live, with Gemini (needs `gcloud auth application-default login` and `GOVRAG_VERTEX_PROJECT` in `.env`):

```bash
GOVRAG_LLM_BACKEND=vertex .venv/bin/python -m govrag.api.app
.venv/bin/python -m scripts.demo              # hr vs intern vs other tenant, then a policy edit
open http://localhost:16686                    # Jaeger: spans with IDs only
```

Demo output (real Gemini):

```
--- acme hr
Margaret Okafor got a 12% raise, their SSN is 078-05-1120, and it was approved by Daniel Whitfield [doc 1].
--- acme intern
... The document mentions that the Q3 review cycle closes on September 30 and that [PERSON 1] runs it [doc 1].
--- globex hr
[PERSON 1] got a 30% raise. ...
(policy edited: intern may now see PERSON)
--- acme intern: who runs the review? (next request)
Daniel Whitfield runs the Q3 review cycle, and it closes on Sept 30 [doc 1].
```

Evals and benchmarks: `python -m eval.pii_recall`, `python -m eval.token_fidelity`, `python -m bench.bench_filtered_search`, `python -m bench.bench_query`, `python -m bench.loop_lag`.

---

## Layout

```
migrations/      SQL: schema, RLS, logins and grants, audit, policies (numbered, applied once)
govrag/
  auth.py        JWT -> Principal          db.py         pools + tenant/role-scoped transactions
  keys.py        envelope encryption       ingest.py     redact -> chunk -> embed -> one transaction
  pii/           Presidio workers, tokens, redaction, vault
  policy/        model, versioned store, NOTIFY cache, re-scan job
  retrieval.py   search SQL + LlamaIndex retriever    llm.py   Gemini, router, breaker
  query.py       the query path            rehydrate.py  allowlist + aliases + streaming
  tracing.py     spans + export allowlist  api/          FastAPI routes
eval/  bench/    measured numbers above, raw JSON in bench/results/
tests/           unit, integration (real Postgres + Presidio), security
```


## What's next

- Batch embeddings across documents (ingest is embedding-bound).
- Vault as its own service with KMS-backed keys.
- WORM export of the audit log (e.g. object lock).
- Cedar or OPA if policies outgrow "role → PII types".
