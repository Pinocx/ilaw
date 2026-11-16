# Derecho IA PRD2 Implementation Blueprint

## Purpose and scope

This blueprint reconciles `docs/PRD2` with the running repository. It defines
component boundaries, supported claims, staged interfaces, and the first safe
implementation slice. The PRD2 status is proposed; unverified infrastructure is
not treated as available.

## Current implementation facts

| Capability | Current state | Integration decision |
|---|---|---|
| Document repository and extraction | 326 processed source documents with structured JSON and Markdown | Reuse existing artifacts; never re-run OCR for RAG work |
| Candidate chunks | 368,053 derived chunks with source unit references | RAG reads the candidate index only |
| Lexical search | SQLite FTS5 trigram index, with JSONL fallback | First retrieval adapter |
| Document metadata | Versioned `document_metadata` v1 contract; 10 human-reviewed types | Preserve per-field provenance; unknown legal facts stay unknown |
| PostgreSQL | Compose provisions PostgreSQL 16 with pgvector and pg_trgm; all 368,053 candidate chunks are loaded in a derived Spanish FTS/hierarchy index | Candidate JSONL remains authoritative; PostgreSQL is a rebuildable retrieval index |
| pgvector / embeddings | pgvector extension and Ollama multilingual embedding model are available; no complete vector corpus is populated | Keep semantic search disabled until the full index finishes |
| STAIR | PostgreSQL-backed structure-aware retriever indexes hierarchy paths and legal references; live hybrid RAG smoke-tested | This implements the repository STAIR contract; no external STAIR service/API was specified |
| LLM providers | Provider-neutral gateway with OpenAI-compatible, native Ollama, Anthropic, and Gemini adapters; task routing, configured fallback, prompt versions, SQLite usage ledger | No encrypted BYOK vault, authenticated tenant isolation, live provider validation, or organization-specific fallback permissions |
| API and SaaS tenancy | No API, identity, or tenant storage in this repository | Keep interfaces tenant-ready but do not simulate authentication/isolation |

## Component boundaries

### RAG Engine

Owns query normalization, retrieval coordination, ranking policy, context
assembly, source resolution, and citation verification. It consumes candidate
repository artifacts through adapters. It does not call an LLM or change the
authoritative knowledge corpus. The default is lexical FTS5; optional weighted
reciprocal-rank fusion accepts semantic and hierarchy results with traceable
chunk payloads.

The initial lexical implementation reports its backend and heuristic score. Exact
article-number lookups route through the indexed PostgreSQL hierarchy path. Generic
queries combine SQLite lexical and PostgreSQL hierarchy rankings. It
does not label a score as confidence, infer legal validity, or treat suggested
classifications as reviewed facts. Every result includes document ID, chunk ID,
source units, available pages, classification state, and verification state.

### LLM Engine

Owns provider configuration, capability negotiation, text and structured
generation, streaming, tool-call proposals, task routing, usage callbacks,
retries, circuit breaking, and normalized provider errors. RAG and Agent depend
on a provider-neutral protocol, not SDKs. Implemented adapters are
OpenAI-compatible Chat Completions, native Ollama `/api/chat`, Anthropic Messages,
and Google Gemini generateContent. Secrets resolve from environment variables
at request time. Capabilities default to text generation only. Prompt templates
are versioned in an immutable in-memory registry; usage records can be stored in
an SQLite ledger with optional daily token limits and configured model rates.
Encrypted BYOK storage and authenticated tenant key isolation remain absent.

### Legal AI Agent

Will own bounded orchestration: query understanding, a visible plan, allowlisted
tool selection, tool argument validation, context sufficiency checks, citation
verification, and structured response assembly. It will not execute arbitrary
code, accept instructions embedded in retrieved documents, or silently answer
when evidence is insufficient.

The agent has a typed allowlist for the PRD's initial repository tools,
deterministic task classification, bounded retrieval, and source verification.
Its optional `answer()` path uses the LLM gateway for structured Spanish analysis,
separates ephemeral user facts from source evidence, checks source IDs and exact
quotes, and suppresses the analysis if citation checks fail. It always marks
legal validity unverified and recommends human review. Bounded process-local
conversation memory keeps tenant, user, and session keys separate. An opt-in
SQLite audit ledger stores hash-chained records and blocks normal update/delete
operations. Authenticated user identity and protected remote audit storage
remain unimplemented.

## Official metadata policy

1. System-verified source identity: source filename, relative path, SHA-256,
   format, and byte size copied from the registry.
2. Human-verified document type: only an explicit classification decision with
   review timestamp and source evidence.
3. Heuristic title, authority mentions, legal identifiers, and date mentions:
   candidates only. A mention does not establish title, issuing authority,
   publication date, effective date, or legal force.
4. Jurisdiction, legal status, validity intervals, amendments, and repeals stay
   unknown until supported by reviewed evidence.
5. Every promoted value requires per-field provenance. Unknown is a valid value,
   not a reason to guess.

## Initial RAG contracts and behavior

- Preserve the raw user query in every request and response.
- Deterministically extract only explicit article/law references and date text;
  do not claim full legal intent understanding without a model.
- Use the existing FTS5 search adapter by default; optional semantic and
  hierarchy adapters participate in weighted reciprocal-rank fusion.
- Allow reviewed document-type and structure filters.
- Treat legal validity as `unknown` unless metadata says otherwise with evidence.
- Build compact context by selecting whole chunks; never cut citation metadata or
  silently truncate a provision. Report omitted results when the context budget
  is reached.
- Verify source IDs against candidate JSONL; verify a quote only by exact match
  in the indexed chunk text. Verification proves repository traceability, not
  current legal validity.
- Direct cross-references may be surfaced only when explicitly present and
  parsed. Semantic relationships and historical lineage remain unavailable.
  PostgreSQL hierarchy retrieval is configured; reviewed relationship and legal
  history expansion remain unavailable.

## Retrieval evaluation

Use a versioned local JSONL dataset with query ID, query text, filters, relevant
document IDs, relevant chunk IDs, and graded relevance. Implement deterministic
Recall@K, Precision@K, MRR, nDCG@K, coverage, and citation traceability. Evaluation
must run offline and must not imply legal correctness from relevance scores.
No gold labels are fabricated; the first evaluation report must state dataset
size and label provenance.

## Sequential delivery

### Phase 1: RAG lexical MVP

1. Provider-neutral request/result contracts and query reference extraction.
2. RAG service over the candidate FTS5 adapter.
3. Structured source context and citation verification.
4. Offline retrieval metrics and a clearly labeled starter query set.
5. Regression tests for provenance, temporal uncertainty, and prompt-safe source
   treatment.

Exit condition: reproducible traceable context retrieval. PostgreSQL hierarchy
retrieval and lexical RRF are live; semantic fusion awaits a complete vector
index and gold-label evaluation.

### Phase 2: LLM gateway

1. Provider and model capability protocols.
2. Runtime-only secret resolution and tenant-aware credential interface.
3. Mock provider tests for errors, structured-output validation, retries, and
   usage capture.
4. Add a local provider adapter first; add external providers only when their
   credentials and outbound data policy are configured.

Exit condition: multiple providers pass the same provider contract tests without
provider-specific logic in RAG or Agent. Four mock-tested adapters and task
routing now exist. Live provider validation, encrypted tenant credential storage,
and organization-specific fallback permissions remain unimplemented.

### Phase 3: Legal research agent

1. Typed, allowlisted tool registry with per-tool schemas and limits.
2. Bounded deterministic research plans and execution audit events.
3. Evidence sufficiency and citation verification before response assembly.
4. Conversation state that keeps user facts separate from repository evidence.
5. Enable model-based reasoning only through the LLM gateway.

Exit condition: bounded, traceable research runs with an explicit insufficient-
evidence path and reproducible audit record. The deterministic and optional
structured-answer path exist; authenticated identity, tenant policy, and a
protected shared audit service remain unimplemented.

### Phase 4: Infrastructure adapters and release validation

Connect PostgreSQL, pgvector, STAIR, tenant identity, and private collections
only after their schemas, running services, authorization model, and migration
ownership are confirmed. Run retrieval, citation, security, and performance
benchmarks before platform integration.

## Current implementation slice

The lexical RAG service, provider gateway, and evidence-first agent are
implemented over the existing candidate index. The `legal-answer` command can
call a configured provider when explicitly run; no provider was called during
development. PostgreSQL/pgvector and PostgreSQL-backed STAIR-compatible retrieval
are live. Semantic vector indexing, external STAIR service integration, encrypted
BYOK vault, authenticated tenancy, or durable agent audit store are not configured
in this checkout.
