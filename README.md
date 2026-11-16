# Derecho IA – Normative Repository

Local document preparation pipeline for Peruvian legal sources.

## Requirements

- Windows 10/11
- Docker Desktop
- Docker Compose
- Original documents in PDF, DOCX or ODT format

## 1. Add source documents

Copy the original documents into:

    data/incoming_originals/

Keep the original files unchanged. The pipeline does not delete duplicates.

## 2. Build and start the Docker service

Run from the project root:

    docker compose up -d --build derecho-ia

## 3. Generate the inventory

    docker compose exec -T derecho-ia python -m normative_repository inventory

This scans the source folder, calculates SHA-256 hashes and detects exact duplicates.

## 4. Suggest document classifications

    docker compose exec -T derecho-ia python -m normative_repository classify

Classifications are suggestions. They require human review.

## 5. Process a pilot

Start with a small sample:

    docker compose exec -T derecho-ia python -m normative_repository process --pilot --limit 10

Review the generated files before processing the full collection.

## 6. Check the results

    docker compose exec -T derecho-ia python -m normative_repository status

    docker compose exec -T derecho-ia python -m normative_repository validate

## 7. Process the remaining documents

    docker compose exec -T derecho-ia python -m normative_repository process --batch-size 10

To retry documents that previously failed:

    docker compose exec -T derecho-ia python -m normative_repository process --batch-size 10 --retry-errors

## 8. Split Markdown into chunks for indexing

    docker compose exec -T derecho-ia python -m normative_repository index

The command reads the generated files in `data/markdown/`, keeps page references,
and writes document, page-unit and chunk JSONL files to
`data/review/index-candidate/`. The active index is protected from accidental
replacement. After reviewing a candidate and approving promotion, replace the
active index explicitly with `--replace-active`. The default target chunk size is
1,800 characters; change it with `--max-chars`, for example:

    docker compose exec -T derecho-ia python -m normative_repository index --max-chars 2200

To use a different candidate directory, pass `--output-dir`. The candidate report
includes ordered non-whitespace source-text coverage and per-document
reconstruction status:

    docker compose exec -T derecho-ia python -m normative_repository index --output-dir /app/data/review/index-candidate

Prepare a retrospective, source-grounded human review packet from the candidate
index:

    docker compose exec -T derecho-ia python -m normative_repository review-packet --limit 10

The command writes `review-packet.md`, `review-labels.csv`, and `manifest.json` to
`data/review/retrospective-pilot/`. New ground-truth fields remain blank until a
human reviewer fills them.

Generate an operational quality report without rerunning extraction or OCR:

    docker compose exec -T derecho-ia python -m normative_repository quality-report

This writes `data/reports/repository-quality-report.json` and the human-readable
`data/reports/inventory-summary.md`. It reports missing metadata and artifacts,
classification proposal differences, OCR/extraction status, suspicious text,
unsupported files, and candidate-index references.

## 9. Search the candidate chunks

Search is deterministic and operates only on the candidate index. Build its
optional local SQLite FTS5 accelerator after generating candidate JSONL files:

    docker compose exec -T derecho-ia python -m normative_repository build-text-index

Then run a query:

    docker compose exec -T derecho-ia python -m normative_repository search "prisión preventiva" --limit 10

By default, every query term must occur in the chunk text or searchable metadata.
Use `--exact-phrase` to require the complete phrase. Optional filters include
`--classification`, `--structure-type`, and `--reviewed-only`:

    docker compose exec -T derecho-ia python -m normative_repository search "artículo 268" --exact-phrase --classification legislation --structure-type article --reviewed-only

Every result includes document, chunk, page, and unit references. Candidate
classifications and legal verification statuses remain explicitly marked; search
results are not legal validation.

When the FTS5 index matches the JSONL source files, searches use its trigram
index and retain the existing Spanish normalization, substring matching, filters,
ranking, and provenance checks. Queries with terms shorter than three characters,
or a missing, stale, or unreadable search database, automatically use the JSONL
scan. Rebuild the text index after rebuilding the candidate chunks. This database
is derived data under `data/review/index-candidate/`; it does not modify the active
index or create embeddings.

## 10. Retrieve source-grounded legal context

The initial RAG service uses candidate FTS5 search and returns a structured
context with document, chunk, page, unit, hierarchy, classification, and legal
verification fields:

    docker compose exec -T derecho-ia python -m normative_repository rag-context "¿Qué establece el artículo 1?" --limit 5

The service preserves the original query and extracts only explicit references.
It does not infer legal validity; unknown metadata stays unknown. Retrieved text
is marked as untrusted source content and must never be followed as instructions.
Context assembly includes whole chunks only. Chunks that do not fit the configured
budget are reported as omitted rather than silently truncated.

Verify a source citation and an optional exact quote with:

    docker compose exec -T derecho-ia python -m normative_repository verify-citation --document-id DOC_ID --chunk-id CHUNK_ID --quote "Exact source text"

This verifies traceability against the candidate index; it does not establish
that the cited law is currently in force. PostgreSQL hierarchy retrieval is
active; semantic retrieval and legal relationship graphs remain unconfigured in
this slice. LLM generation is explicit and is never triggered by ingestion.

## 11. Prepare source-linked metadata candidates

Generate a bounded report of possible document titles, legal identifiers, date
mentions, and authority mentions from the first source units:

    docker compose exec -T derecho-ia python -m normative_repository metadata-proposals --max-units 3

The report is written to `data/reports/metadata-proposals.json`. Every proposal
contains a source unit, page when available, and an evidence excerpt. These are
unverified candidates; dates are not assigned as publication or effective dates,
and the command does not change the registry or source files.

The official metadata policy is enforced in `documents.jsonl`: filename, relative
path, SHA-256, format, and size are system-verified from the registry. Document
type is official only when `reviewed_classification` has a human decision. Title,
jurisdiction, issuing authority, document number, publication/effective dates,
and legal status are not promoted from heuristic mentions. `legal_status` stays
`unknown` until human-verified evidence is recorded. The versioned
`document_metadata` object includes per-field provenance and status, and
`quality-report` validates it.

## 12. Evaluate completed human reviews

After a human reviewer fills in `data/review/retrospective-pilot/review-labels.csv`,
validate the decisions and generate an evaluation report:

    docker compose exec -T derecho-ia python -m normative_repository review-evaluate

This writes `data/reports/segmentation-review-evaluation.json`. It does not change
the registry or reviewer labels. The current packet does not contain exhaustive
reference segment boundaries, so the report deliberately leaves segmentation
precision, recall, and F1 unavailable until such human annotations exist.

To create an exhaustive page-level annotation batch from the reviewed PDF sample:

    docker compose exec -T derecho-ia python -m normative_repository segment-review-packet --pages-per-document 2 --seed 2026

The packet contains source pages only, `page-review.csv`, and a blank
`gold-segments.csv`; it does not show candidate chunks while reviewers create gold
labels. Annotators must label every non-whitespace token on a page exactly once
and mark the page complete only after checking coverage. Then run:

    docker compose exec -T derecho-ia python -m normative_repository segment-evaluate

The report measures one-to-one segment precision/recall/F1 at token IoU 0.5,
token-level structural labels, and matched-segment hierarchy/reference accuracy.
Results cover the selected development pages, not entire documents or a held-out
set. Incomplete pages are skipped; no metrics are fabricated.

RAG retrieval evaluation uses reviewer-labeled JSONL with this schema:
`docs/evaluation/retrieval-dataset-schema.md`. Run it with:

    docker compose exec -T derecho-ia python -m normative_repository rag-evaluate --dataset /app/data/review/retrieval-queries.jsonl --output /app/data/reports/rag-evaluation.json

The evaluator reports Precision@K, Recall@K, MRR, nDCG@K, retrieval coverage,
and citation traceability. It refuses empty or unlabeled datasets and does not
claim legal correctness.

Run the deterministic regression tests from the project root with:

    python -m unittest discover -s tests -p "test_*.py" -v

## Legal research agent (PRD2)

The repository now includes a provider-neutral LLM gateway and a bounded legal
research agent. Run `legal-answer` only when you want to make an explicit model
request:

    docker compose exec -T derecho-ia python -m normative_repository legal-answer "What does Article 1 provide?"

The default Compose configuration routes structured analysis to the local Ollama
endpoint. The agent retrieves source chunks, treats their text as untrusted,
requires exact quote verification, and suppresses generated analysis if source
checks fail. It always reports legal validity as unverified and recommends human
review. No model call is made during ingestion or `rag-context`.

The gateway includes OpenAI-compatible, Ollama-native, Anthropic, and Gemini
adapters. Configuration and environment variables are documented in
`docs/llm-gateway.md`. `llm-usage-report` reports saved organization usage;
`LEGAL_AUDIT_DB` enables the local agent audit ledger.

Compose runs PostgreSQL with pgvector in a persistent named Docker volume. A
random local database password is stored in the ignored `.env` file. Run
`build-stair-index` to load Spanish PostgreSQL full-text and hierarchy data from
candidate chunks. RAG fuses these structure-aware results with its SQLite lexical
index. This repository provides a STAIR-compatible adapter; it does not connect
to an external STAIR product. Exact article lookups use the PostgreSQL hierarchy
index directly, avoiding a broad scan of the local lexical index.

Semantic vectors are optional and remain disabled in the default Compose
configuration so retrieval never searches a partially populated vector table.
The locally installed Ollama model `nomic-embed-text-v2-moe` returns 768
dimensions. To populate pgvector, pass `EMBEDDING_ENDPOINT`, `EMBEDDING_MODEL`,
and `EMBEDDING_DIMENSION` to `docker compose exec` and run:

    docker compose exec -T -e EMBEDDING_ENDPOINT=http://host.docker.internal:11434 -e EMBEDDING_MODEL=nomic-embed-text-v2-moe -e EMBEDDING_DIMENSION=768 derecho-ia python -m normative_repository build-vector-index

This calls the local Ollama embedding service and can take substantial time for
the full corpus. Add those settings to the application service only after the
vector index completes; then RAG combines lexical, semantic, and hierarchy
rankings.

Human-reviewed provider comparisons use `docs/evaluation/model-benchmark-schema.md`
and `llm-evaluate`; no benchmark labels are generated automatically.

## Generated folders

- `data/reports/`: inventory and processing reports
- `data/extracted/`: extracted text in JSON
- `data/markdown/`: readable Markdown versions
- `data/structured/`: normalized document data
- `data/index/`: `documents.jsonl`, `units.jsonl`, `chunks.jsonl` and a processing report
- `data/review/`: candidate indexes, retrospective review packets and reviewer labels

## Important limitations

- Classification is preliminary and must be reviewed.
- Text extraction does not verify legal validity or current force.
- PDF page references are available when supported by the extractor.
- DOCX and ODT extraction does not preserve reliable page numbers.
- Rules-based segmentation recognizes selected legal structures. It is not a legal
  verification system, and measured text coverage does not establish segmentation
  precision or recall.
- Exact duplicate files are identified by content hash; originals are not deleted.
- pgvector is active in the Compose database. Semantic vectors remain optional
  until an embedding model is configured and `build-vector-index` is run.
- The agent is a research prototype, not a legal opinion system. Current legal
  force, conflicting authorities, and legal correctness require human review.
- Tenant authentication, encrypted BYOK storage, and private collections are not
  implemented here.
