# Repository audit and continuation backlog

Reviewed against `docs/PRD.md` on 2026-11-14. Counts below were verified from the
local registry and generated index artifacts. This folder does not contain a Git
repository, so Git status and diff-based change tracking are unavailable here.

## Verified repository state

- The registry contains 326 unique documents. The inventory report lists 329
  supported file entries (328 PDFs and one DOCX), including exact duplicates.
- All 326 records are marked processed; status reports zero pending documents,
  zero errors, and 316 classifications requiring human review. The top-level
  extraction status was synchronized by an inventory refresh; it now agrees with
  processing status for all 326 documents.
- The registry currently stores `unknown` as the filename-rule suggestion for all
  326 records, while structured JSON contains 32 `legislation`, 25 `jurisprudence`,
  25 `legal_manual`, 13 `legal_doctrine`, five `complementary_material`, and 226
  `unknown` proposals. The index reads the structured proposal when the registry
  says `unknown`; this discrepancy needs reconciliation without promoting either
  proposal to ground truth.
- Fifty PDF records are marked OCR-required. Three currently completed records
  have earlier error events in processing history. Those records were not
  reprocessed during this work; two are represented in the pilot packet.
- Six Markdown files contain 1,326 Unicode replacement characters in total.
  Source-level comparison now confirms that all six preserve the PDF's extracted
  text layer at the affected page locations. This does not verify visual page
  fidelity; the affected pages still need visual inspection where legal accuracy
  is important.
- `validate` checked all 326 structured documents and returned zero errors. This
  checks technical structure, not extraction fidelity or legal correctness.
- The active index was not changed during this work. It contains 330,889 chunks
  and 159,132 page units.
- The rebuilt candidate at `data/review/index-candidate/` contains 326 documents,
  159,132 units, and 368,053 chunks. All candidate chunks pass `ChunkMetadata`
  validation; chunk IDs are unique; document, unit, and page references resolve.
  There are 761 chunks without page references, all from the DOCX source whose
  extraction has no reliable page boundaries.
- Ordered non-whitespace token comparison reconstructed all 336,522,573 source
  characters across all 326 documents. Coverage is 100%, with exact reconstruction
  in every document. This measures text preservation only; it does not measure
  whether structural labels are correct.
- Candidate chunk structure counts: 23,718 paragraphs; 112,024 chapters; 28,137
  case sections; 80,357 table-of-contents entries; 48,916 titles; 7,892 decisions;
  4,026 dispositions; 19,988 books; 39,758 sections; 471 commentary chunks;
  671 quoted provisions; and 2,095 article chunks. Article chunks remain
  unverified legal structure proposals pending review.
- The user confirmed classifications for ten pilot documents. The ten explicit
  human decisions are persisted in the registry; seven new labels were applied
  and three already-matching labels were left unchanged.
- The candidate contains hierarchy paths, including article ancestry, and the
  report provides coverage totals and per-document reconstruction results.
- The deterministic validator had a syntax error and called nonexistent private
  method names. These defects are corrected. Validation now checks the full source
  text instead of truncating it at 50,000 characters and does not mutate frozen
  segment objects.
- The reviewed DOCX contents sample showed that tab-separated page numbers were
  being collapsed before table-of-contents detection. Segmentation now checks the
  raw line first; the index labels the sampled `CAPÍTULO I 8` entry as
  `table_of_contents_entry` while retaining actual body headings as `chapter`.
  A regression test covers tab-separated contents entries.
- `setup.py` contained Docker Compose YAML rather than Python package metadata; it
  now contains Python package metadata, with the build backend declared in
  `pyproject.toml`. The packaging command could not be exercised because
  `setuptools` is not installed in this local Python environment; the Docker
  workflow does not install this package and remains independent of that backend.
- The previous AI segmentation test file depended on missing runtime modules
  (`ai_segmentation`, `ai_client`, `llm_client`, and `ai_config`). Those tests were
  replaced with tests for the implemented rules-based system. No AI segmentation
  provider or hybrid runtime is implemented yet.
- The current regression suite has 102 passing `unittest` tests. `compileall` passes
  for the package and tests. `pytest` is not installed or required by the revised
  tests.
- The `index` CLI now defaults to `data/review/index-candidate/`; replacing the
  active index requires the explicit `--replace-active` option.
- A retrospective review packet was generated at
  `data/review/retrospective-pilot/` for ten representative records. It includes
  source Markdown, structured units, selected candidate chunks, provenance checks,
  existing human labels, and empty fields for new reviewer decisions. No ground
  truth was created or changed.
- A reproducible quality command now writes
  `data/reports/repository-quality-report.json` and
  `data/reports/inventory-summary.md`. The report finds 100 registry-vs-structured
  proposal differences, no missing originals/artifacts, no unsupported files, no
  invalid candidate references, and no extraction-status mismatches after the
  inventory refresh. It also records all missing legal metadata and suspicious
  text per document. Its PDF text-layer comparison confirms preservation for all
  six affected documents; it cannot establish visual page fidelity.
- The `derecho-ia` container had been stopped. It was rebuilt from the local
  source and started; its CLI reports 326 processed documents and `validate`
  returns zero errors. `review-packet` completed a one-record smoke run and
  `quality-report` completed inside the container after a rebuild with the latest
  source. The inventory refresh reused cached OCR assessments and did not repeat
  extraction or OCR. The 47-test local suite and Python compilation pass.
- No embeddings or vector database were generated. The active index was not
  overwritten. No external AI or legal-source service was called, and no source
  documents were transmitted for processing.

## Roadmap status

### Phase 1 - Repository audit and baseline: partially complete

The code, data reports, schemas, status, index counts, source references, local
tests, and an operational quality report were reviewed. A retrospective packet is
available for human review. PDF text-layer checks were compared with structured
units and Markdown, but visual page fidelity and segmentation evidence have not
yet been scored against reviewer-confirmed labels.

### Phase 2 - Segmentation quality and validation: partially complete

Deterministic rules, hierarchy paths, quoted provisions, TOC entries, jurisprudence
markers, source-text coverage, and regression tests are available. Ten pilot
documents have human-confirmed classifications. Precision, recall, legal structure
accuracy, and hierarchy accuracy have not been measured on a representative
human-labelled dataset. The AI/hybrid segmentation modes in the PRD are not
implemented. A `review-evaluate` command validates reviewer CSV decisions and
reports supported classification diagnostics and review coverage without changing
the registry. It deliberately withholds segmentation precision, recall, and F1
because the packet has one selected chunk per document rather than exhaustive
human reference boundaries. Its default output is
`data/reports/segmentation-review-evaluation.json`.

A second `segment-review-packet` workflow now samples two reproducible pages from
each of the nine page-addressable PDF documents in the pilot. It creates a
source-only packet plus `page-review.csv` and `gold-segments.csv`, protecting
completed human annotations from overwrite. The `segment-evaluate` command checks
full ordered token coverage, aligns grouped chunk parts to source offsets, and can
report one-to-one segment precision/recall/F1 at token IoU 0.5, token-level
structure metrics, and matched hierarchy/reference accuracy. The current batch
contains 18 pages. The user deferred manual annotation; segmentation metrics
remain unavailable, but this does not block text search or metadata work. It is a
development set, not a held-out evaluation.

### Phase 3 - Metadata enrichment: partially complete

The `metadata-proposals` command now emits source-linked candidates for title,
document identifiers, date mentions, and authority mentions from at most three
source units (12,000 characters per unit). The pilot report has title candidates
for 317 of 326 documents, authority mentions for 44, date mentions for 29, and
document-identifier candidates for 53. It labels each as requiring review and
does not write to the registry. Jurisdiction, publication/effective dates, legal
status, document versions, and reuse/copyright status remain unfilled until
supported by explicit source evidence and review.

The versioned `document_metadata` contract now promotes only registry-backed
technical identity fields (filename, relative path, SHA-256, format, and size)
and human-reviewed document type. Suggested classifications remain pending;
heuristic title, authority, identifier, and date mentions are not official.
Jurisdiction, publication/effective dates, and legal status are not inferred.
Per-field provenance is validated in the candidate quality report.

### Phase 4 - Text search: partially complete

The `search` CLI provides deterministic all-term and exact-phrase search over
candidate chunks, with classification and structure filters, an optional
reviewed-only filter, and source page/unit references. It now uses a persistent
SQLite FTS5 trigram index under `data/review/index-candidate/search.sqlite3` when
that database matches its JSONL source files. It falls back to the JSONL scan for
short terms or missing/stale indexes. The current database indexes 368,053 chunks
with zero malformed rows and is about 2.55 GB. A full-corpus `prision preventiva`
query examined 10,500 FTS candidates and returned in 3.3 seconds on the local
machine; the earlier linear scan took about 65 to 95 seconds in Docker. Regression
tests cover accent-insensitive matching, substring and exact phrase handling,
filters, stale-index fallback, bounds, and provenance. Retrieval ranking and
relevance still require evaluation against representative human-reviewed queries;
this prototype is not an approved legal retrieval engine.

### Phases 5-7 - Embeddings, hybrid retrieval, and RAG: partially implemented

Weighted reciprocal-rank fusion, an Ollama embedding adapter, PostgreSQL pgvector,
and a PostgreSQL-backed structure-aware STAIR-compatible retriever are implemented.
Compose provisions PostgreSQL with pgvector and trigram support; all 368,053 candidate chunks are
loaded into its Spanish full-text and hierarchy indexes with `build-stair-index`.
The STAIR-compatible retriever passed a live RAG smoke query. A multilingual
embedding model is installed and its dimensions verified; the complete vector
corpus remains unpopulated because local CPU throughput makes bulk generation
impractical in this run. Semantic search remains disabled until indexing finishes.
Retrieval quality still needs reviewer-labeled legal queries.

## PRD2 - RAG, LLM Engine, and Legal AI Agent

`docs/PRD2` assumes PostgreSQL, pgvector, and STAIR. PostgreSQL/pgvector and the
repository's STAIR-compatible hierarchy adapter are now configured through
Compose; no external STAIR API was specified or integrated. The implementation
blueprint at `docs/architecture/prd2-implementation-blueprint.md` records the
remaining discrepancies,
provider/security boundaries, the official metadata policy, and sequential exit
conditions. It explicitly avoids duplicate knowledge storage or claims about
unavailable infrastructure.

The RAG layer now exists under `normative_repository/rag/`. It preserves query
text, extracts explicit references only, searches through the candidate FTS
adapter or configured hybrid retriever, assembles whole traceable chunks under a
character budget, marks retrieved text as untrusted, and verifies
document/chunk/quote traceability.
Unknown legal validity remains unknown. Related-provision expansion and legal
history return explicit unavailable results until reviewed relationship/version
data exists. The offline evaluator computes Precision@K, Recall@K, MRR, nDCG@K,
retrieval coverage, and citation traceability only against supplied labels; no
gold queries have been generated. Semantic vector population remains pending.

The provider-neutral LLM gateway now has OpenAI-compatible, Ollama-native,
Anthropic, and Gemini adapters; capability checks; task routing; prompt version
metadata; retries; circuit breaking; structured output; streaming; proposed tool
calls; and a SQLite usage ledger with optional quotas and configured pricing.
Credentials are resolved at runtime from environment variables. Mock contract
tests cover each adapter. No live LLM provider request was made. The local
embedding endpoint was smoke-tested, but no complete vector index is active. A
runtime `CredentialResolver` receives the organization and provider.
Explicit provider fallback routes are configurable and covered by contract tests.
Encrypted BYOK storage, authenticated tenant isolation, organization-specific
fallback permissions, and live provider validation remain unimplemented. Model
comparison metrics are available only for explicitly human-reviewed benchmark
records; none were fabricated.

The legal agent now has deterministic task classification, a typed allowlist for
the PRD's initial tools, bounded retrieval and source verification, explicit
insufficient-evidence handling, ephemeral conversation facts separated from
repository evidence, and optional structured analysis through the gateway. It
suppresses analysis unless each finding and exact quote resolves to retrieved
source IDs, and marks legal validity unverified. Conversation memory is bounded
and process-local. An opt-in SQLite audit ledger stores query, plan, tools,
source IDs, model metadata, verified response, and citation results in a
hash-chained append-only log. Authenticated tenant policy and reviewed legal
history/relationship indexes remain unavailable in this checkout.

### Phase 8 - Operational readiness: partially complete

Docker and CLI workflows exist. Incremental processing, complete operational
reports, recovery guarantees, release procedures, and data contracts need further
validation.

## Recommended next work

1. Evaluate candidate text search with reviewer-confirmed legal queries and tune
   relevance only against a held-out evaluation set.
2. Use the metadata proposal report to define a reviewed-field import contract;
   do not promote its heuristic candidates directly to authoritative metadata.
3. Plan migration for remaining PRD index fields, including parent-unit IDs,
   document version/reuse metadata, and neighboring-unit relationships.
4. Review the remaining 316 classification proposals and persist only
   human-approved labels as ground truth.
5. Revisit optional exhaustive segmentation annotation if measured recall/precision
   is needed before a downstream legal retrieval release.

Do not regenerate embeddings, call external AI providers, or promote candidate
artifacts to the active index until segmentation quality is reviewed and approved.
