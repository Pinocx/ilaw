"""Command-line interface for the normative repository."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .config import (
    DEFAULT_SOURCE_DIR,
    REGISTRY_FILE,
    INDEX_DIR,
    CANDIDATE_INDEX_DIR,
    RETROSPECTIVE_PILOT_DIR,
    REPORTS_DIR,
    SEGMENT_EVALUATION_DIR,
    PILOT_SIZE,
    BATCH_SIZE_DEFAULT,
)
from .registry import load_registry
from .inventory import run_inventory
from .classify import classify_all
from .pipeline import run_processing, validate_outputs
from .index import build_index_dataset
from .review_packet import build_review_packet
from .quality_report import build_quality_report
from .search import search_chunks
from .text_index import build_text_search_index
from .evaluation import apply_reviewed_classifications, evaluate_review_labels
from .segment_review import build_segment_review_packet, evaluate_segment_review_packet
from .metadata_proposals import build_metadata_proposals
from .rag.engine import MAX_CONTEXT_CHARS, retrieve_legal_context, verify_citation
from .rag.evaluation import evaluate_retrieval_dataset
from .agent import LegalResearchAgent, SqliteAuditLedger
from .llm import SqliteUsageLedger, build_gateway_from_environment, evaluate_model_benchmarks
from .rag.hybrid import OllamaEmbedder, PostgresPgVectorRetriever, PostgresStairRetriever
from .rag.hybrid import HybridRetriever


def _print_json(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _pg_connection_factory(dsn: str):
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("Install the optional pgvector extra to use PostgreSQL retrieval.") from exc

    def connect():
        try:
            return psycopg.connect(dsn)
        except Exception:
            raise RuntimeError("PostgreSQL vector store connection failed.") from None

    return connect


def _configured_hybrid_retriever(index_dir: Path) -> HybridRetriever | None:
    dsn = os.environ.get("PGVECTOR_DSN")
    if not dsn:
        return None
    connection_factory = _pg_connection_factory(dsn)
    stair = PostgresStairRetriever(connection_factory, table_name=os.environ.get("STAIR_TABLE", "legal_hierarchy_chunks"))
    embedding_values = [os.environ.get(name) for name in ("EMBEDDING_ENDPOINT", "EMBEDDING_MODEL", "EMBEDDING_DIMENSION")]
    if any(embedding_values) and not all(embedding_values):
        raise ValueError("EMBEDDING_ENDPOINT, EMBEDDING_MODEL, and EMBEDDING_DIMENSION must be configured together.")
    vector_store = None
    if all(embedding_values):
        vector_store = PostgresPgVectorRetriever(
            connection_factory,
            OllamaEmbedder(embedding_values[0], embedding_values[1]),
            dimension=int(embedding_values[2]),
            table_name=os.environ.get("PGVECTOR_TABLE", "legal_chunk_vectors"),
        )
    return HybridRetriever(index_dir=index_dir, semantic_retriever=vector_store, hierarchy_retriever=stair)


def _status() -> dict:
    registry = load_registry(REGISTRY_FILE)
    records = list(registry.get("records", {}).values())

    result = {
        "total_documents": len(records),
        "pending": 0,
        "done": 0,
        "errors": 0,
        "classification_review_required": 0,
    }

    for record in records:
        processing_status = record.get("processing", {}).get("status", "pending")

        if processing_status == "done":
            result["done"] += 1
        elif processing_status == "error":
            result["errors"] += 1
        else:
            result["pending"] += 1

        if record.get("classification_review_required", True):
            result["classification_review_required"] += 1

    return result


def _resolve_index_output_dir(
    output_dir: Path | None,
    replace_active: bool,
) -> Path:
    """Select a candidate index path unless active replacement is explicit."""
    if replace_active:
        if output_dir is not None and output_dir.resolve() != INDEX_DIR.resolve():
            raise ValueError(
                "--replace-active can only be used with the active index directory."
            )
        return INDEX_DIR

    resolved_output = output_dir or CANDIDATE_INDEX_DIR
    if resolved_output.resolve() == INDEX_DIR.resolve():
        raise ValueError(
            "Refusing to overwrite the active index without --replace-active."
        )
    return resolved_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="normative-repository",
        description="Local document preparation pipeline for Peruvian legal sources.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser(
        "inventory",
        help="Scan source documents and update the inventory.",
    )
    inventory_parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing original documents.",
    )

    subparsers.add_parser(
        "classify",
        help="Suggest document classifications.",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Extract text and generate document outputs.",
    )
    process_parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing original documents.",
    )
    process_parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Process documents that do not require OCR.",
    )

    process_parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Process only documents that require OCR.",
    )    
    process_parser.add_argument("--pilot", action="store_true")
    process_parser.add_argument("--limit", type=int, default=PILOT_SIZE)
    process_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    process_parser.add_argument("--retry-errors", action="store_true")

    subparsers.add_parser(
        "validate",
        help="Validate generated structured outputs.",
    )
    index_parser = subparsers.add_parser(
        "index",
        help="Chunk Markdown files and generate JSONL indexing data.",
    )
    index_parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Maximum target chunk size in characters (default: 1800).",
    )
    index_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated JSONL files (defaults to the review candidate directory).",
    )
    index_parser.add_argument(
        "--replace-active",
        action="store_true",
        help="Explicitly replace the active index in data/index/ after building it.",
    )
    subparsers.add_parser(
        "status",
        help="Show repository processing status.",
    )
    review_parser = subparsers.add_parser(
        "review-packet",
        help="Prepare a source-grounded packet for human review.",
    )
    review_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of representative documents to include.",
    )
    review_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to data/review/retrospective-pilot/).",
    )
    quality_parser = subparsers.add_parser(
        "quality-report",
        help="Audit existing repository artifacts without reprocessing documents.",
    )
    quality_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    quality_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    quality_parser.add_argument("--json-output", type=Path, default=None)
    quality_parser.add_argument("--summary-output", type=Path, default=None)

    metadata_proposals_parser = subparsers.add_parser(
        "metadata-proposals",
        help="Create source-linked metadata candidates without mutating the registry.",
    )
    metadata_proposals_parser.add_argument("--max-units", type=int, default=3)
    metadata_proposals_parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "metadata-proposals.json",
    )

    text_index_parser = subparsers.add_parser(
        "build-text-index",
        help="Build a persistent SQLite FTS5 index for candidate chunks.",
    )
    text_index_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)

    search_parser = subparsers.add_parser(
        "search",
        help="Search the review candidate chunks with source references.",
    )
    search_parser.add_argument("query", help="Text or phrase to search for.")
    search_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.add_argument("--classification")
    search_parser.add_argument("--structure-type")
    search_parser.add_argument("--exact-phrase", action="store_true")
    search_parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Return only chunks with a human-reviewed document classification.",
    )

    rag_parser = subparsers.add_parser(
        "rag-context",
        help="Retrieve source-traceable lexical context for a legal question.",
    )
    rag_parser.add_argument("query", help="Original legal question or search query.")
    rag_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    rag_parser.add_argument("--limit", type=int, default=10)
    rag_parser.add_argument("--classification")
    rag_parser.add_argument("--structure-type")
    rag_parser.add_argument("--reviewed-only", action="store_true")
    rag_parser.add_argument("--context-char-budget", type=int, default=12000)

    citation_parser = subparsers.add_parser(
        "verify-citation",
        help="Verify a document/chunk citation against candidate source text.",
    )
    citation_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    citation_parser.add_argument("--document-id", required=True)
    citation_parser.add_argument("--chunk-id", required=True)
    citation_parser.add_argument("--quote")

    legal_answer_parser = subparsers.add_parser(
        "legal-answer",
        help="Run bounded legal research and generate a citation-verified answer through the configured LLM.",
    )
    legal_answer_parser.add_argument("query", help="Natural-language legal question.")
    legal_answer_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    legal_answer_parser.add_argument("--limit", type=int, default=8)
    legal_answer_parser.add_argument("--context-char-budget", type=int, default=12000)
    legal_answer_parser.add_argument("--organization-id")
    legal_answer_parser.add_argument("--user-id")

    usage_report_parser = subparsers.add_parser(
        "llm-usage-report",
        help="Report recorded model usage for one organization.",
    )
    usage_report_parser.add_argument("--organization-id", required=True)
    usage_report_parser.add_argument("--database", type=Path, default=None)
    usage_report_parser.add_argument("--limit", type=int, default=1000)

    vector_index_parser = subparsers.add_parser(
        "build-vector-index",
        help="Generate derived chunk embeddings and upsert them into PostgreSQL pgvector.",
    )
    vector_index_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    vector_index_parser.add_argument("--batch-size", type=int, default=32)

    stair_index_parser = subparsers.add_parser(
        "build-stair-index",
        help="Build PostgreSQL full-text and legal hierarchy indexes from candidate chunks.",
    )
    stair_index_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    stair_index_parser.add_argument("--batch-size", type=int, default=500)

    audit_verify_parser = subparsers.add_parser(
        "legal-audit-verify",
        help="Verify the integrity of an existing legal agent audit ledger.",
    )
    audit_verify_parser.add_argument("--database", type=Path, required=True)

    llm_eval_parser = subparsers.add_parser(
        "llm-evaluate",
        help="Evaluate models against human-reviewed legal response records.",
    )
    llm_eval_parser.add_argument("--dataset", type=Path, required=True)
    llm_eval_parser.add_argument("--output", type=Path, default=None)

    rag_eval_parser = subparsers.add_parser(
        "rag-evaluate",
        help="Evaluate retrieval against a reviewer-labeled JSONL dataset.",
    )
    rag_eval_parser.add_argument("--dataset", type=Path, required=True)
    rag_eval_parser.add_argument("--index-dir", type=Path, default=CANDIDATE_INDEX_DIR)
    rag_eval_parser.add_argument("--output", type=Path, default=None)
    rag_eval_parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])

    evaluation_parser = subparsers.add_parser(
        "review-evaluate",
        help="Validate human review labels and report evidence-qualified metrics.",
    )
    evaluation_parser.add_argument(
        "--input",
        type=Path,
        default=RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
        help="Reviewer-completed labels CSV.",
    )
    evaluation_parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "segmentation-review-evaluation.json",
        help="Output path for the evaluation report JSON.",
    )

    classification_import_parser = subparsers.add_parser(
        "review-apply-classifications",
        help="Persist only explicit human-approved classification decisions.",
    )
    classification_import_parser.add_argument(
        "--input",
        type=Path,
        default=RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
    )
    classification_import_parser.add_argument(
        "--report-output",
        type=Path,
        default=REPORTS_DIR / "classification-review-import.json",
    )

    segment_packet_parser = subparsers.add_parser(
        "segment-review-packet",
        help="Create a randomized, source-only page annotation packet.",
    )
    segment_packet_parser.add_argument("--pages-per-document", type=int, default=2)
    segment_packet_parser.add_argument("--seed", type=int, default=2026)
    segment_packet_parser.add_argument(
        "--review-csv",
        type=Path,
        default=RETROSPECTIVE_PILOT_DIR / "review-labels.csv",
    )
    segment_packet_parser.add_argument(
        "--output-dir", type=Path, default=SEGMENT_EVALUATION_DIR
    )

    segment_evaluation_parser = subparsers.add_parser(
        "segment-evaluate",
        help="Evaluate completed exhaustive page segment annotations.",
    )
    segment_evaluation_parser.add_argument(
        "--packet-dir", type=Path, default=SEGMENT_EVALUATION_DIR
    )
    segment_evaluation_parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS_DIR / "segmentation-page-evaluation.json",
    )
    segment_evaluation_parser.add_argument("--matching-iou", type=float, default=0.5)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.command == "inventory":
            result = run_inventory(args.source)
            if result is not None:
                _print_json(result)

        elif args.command == "classify":
            result = classify_all()
            if result is not None:
                _print_json(result)

        elif args.command == "process":
            result = run_processing(
                source_dir=args.source,
                pilot=args.pilot,
                limit=args.limit,
                batch_size=args.batch_size,
                retry_errors=args.retry_errors,
                skip_ocr=args.skip_ocr,
                ocr_only=args.ocr_only,
            )
            _print_json(result)

        elif args.command == "validate":
            result = validate_outputs()
            _print_json(result)
            return 0 if result["valid"] else 1

        elif args.command == "index":
            output_dir = _resolve_index_output_dir(
                args.output_dir,
                args.replace_active,
            )
            _print_json(build_index_dataset(
                max_chars=args.max_chars,
                output_dir=output_dir,
            ))

        elif args.command == "status":
            _print_json(_status())

        elif args.command == "review-packet":
            _print_json(build_review_packet(
                limit=args.limit,
                output_dir=args.output_dir or RETROSPECTIVE_PILOT_DIR,
            ))

        elif args.command == "quality-report":
            _print_json(build_quality_report(
                source_dir=args.source,
                index_dir=args.index_dir,
                json_output=args.json_output,
                summary_output=args.summary_output,
            ))

        elif args.command == "metadata-proposals":
            report = build_metadata_proposals(
                output_path=args.output,
                max_units=args.max_units,
            )
            _print_json({
                key: value
                for key, value in report.items()
                if key != "documents"
            })

        elif args.command == "search":
            _print_json(search_chunks(
                args.query,
                index_dir=args.index_dir,
                limit=args.limit,
                classification=args.classification,
                structure_type=args.structure_type,
                exact_phrase=args.exact_phrase,
                reviewed_only=args.reviewed_only,
            ))

        elif args.command == "rag-context":
            _print_json(retrieve_legal_context(
                args.query,
                index_dir=args.index_dir,
                limit=args.limit,
                classification=args.classification,
                structure_type=args.structure_type,
                reviewed_only=args.reviewed_only,
                context_char_budget=args.context_char_budget,
                hybrid_retriever=_configured_hybrid_retriever(args.index_dir),
            ))

        elif args.command == "verify-citation":
            _print_json(verify_citation(
                index_dir=args.index_dir,
                document_id=args.document_id,
                chunk_id=args.chunk_id,
                quote=args.quote,
            ))

        elif args.command == "legal-answer":
            audit_database = os.environ.get("LEGAL_AUDIT_DB")
            agent = LegalResearchAgent(
                index_dir=args.index_dir,
                context_char_budget=args.context_char_budget,
                hybrid_retriever=_configured_hybrid_retriever(args.index_dir),
                audit_ledger=SqliteAuditLedger(Path(audit_database)) if audit_database else None,
            )
            usage_db = Path(os.environ.get("LLM_USAGE_DB", str(REPORTS_DIR / "llm-usage.sqlite3")))
            ledger = SqliteUsageLedger.from_environment(usage_db)
            gateway = build_gateway_from_environment(usage_ledger=ledger)
            _print_json(agent.answer(
                args.query,
                gateway,
                limit=args.limit,
                organization_id=args.organization_id,
                user_id=args.user_id,
            ).model_dump(mode="json"))

        elif args.command == "legal-audit-verify":
            if not args.database.is_file():
                raise FileNotFoundError("Audit database does not exist.")
            _print_json(SqliteAuditLedger(args.database).verify_integrity())

        elif args.command == "llm-evaluate":
            report = evaluate_model_benchmarks(args.dataset)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                report["output_json"] = str(args.output.resolve())
            _print_json(report)

        elif args.command == "llm-usage-report":
            usage_db = args.database or Path(os.environ.get("LLM_USAGE_DB", str(REPORTS_DIR / "llm-usage.sqlite3")))
            ledger = SqliteUsageLedger.from_environment(usage_db)
            _print_json(ledger.report(args.organization_id, limit=args.limit))

        elif args.command == "build-vector-index":
            dsn = os.environ.get("PGVECTOR_DSN")
            embedding_endpoint = os.environ.get("EMBEDDING_ENDPOINT")
            embedding_model = os.environ.get("EMBEDDING_MODEL")
            dimension_raw = os.environ.get("EMBEDDING_DIMENSION")
            if not all((dsn, embedding_endpoint, embedding_model, dimension_raw)):
                raise ValueError("PGVECTOR_DSN, EMBEDDING_ENDPOINT, EMBEDDING_MODEL, and EMBEDDING_DIMENSION are required.")
            dimension = int(dimension_raw)
            embedder = OllamaEmbedder(embedding_endpoint, embedding_model)
            retriever = PostgresPgVectorRetriever(
                _pg_connection_factory(dsn), embedder, dimension=dimension,
                table_name=os.environ.get("PGVECTOR_TABLE", "legal_chunk_vectors"),
            )
            retriever.ensure_schema()
            chunks_path = args.index_dir / "chunks.jsonl"

            def read_chunks():
                with chunks_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)

            count = retriever.index_chunks(read_chunks(), batch_size=args.batch_size)
            _print_json({"indexed_chunks": count, "table": retriever.table_name, "dimension": dimension})

        elif args.command == "build-stair-index":
            dsn = os.environ.get("PGVECTOR_DSN")
            if not dsn:
                raise ValueError("PGVECTOR_DSN is required.")
            retriever = PostgresStairRetriever(
                _pg_connection_factory(dsn), table_name=os.environ.get("STAIR_TABLE", "legal_hierarchy_chunks"),
            )
            retriever.ensure_schema()
            chunks_path = args.index_dir / "chunks.jsonl"
            def read_chunks():
                with chunks_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)
            count = retriever.index_chunks(read_chunks(), batch_size=args.batch_size)
            _print_json({"indexed_chunks": count, "table": retriever.table_name, "backend": "postgresql_stair"})

        elif args.command == "rag-evaluate":
            report = evaluate_retrieval_dataset(
                args.dataset,
                retrieve=lambda query, limit: retrieve_legal_context(
                    query,
                    index_dir=args.index_dir,
                    limit=limit,
                    context_char_budget=MAX_CONTEXT_CHARS,
                ),
                k_values=tuple(args.k),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                report["output_json"] = str(args.output.resolve())
            _print_json(report)

        elif args.command == "build-text-index":
            _print_json(build_text_search_index(args.index_dir))

        elif args.command == "review-evaluate":
            _print_json(evaluate_review_labels(
                input_path=args.input,
                output_path=args.output,
            ))

        elif args.command == "review-apply-classifications":
            _print_json(apply_reviewed_classifications(
                input_path=args.input,
                output_path=args.report_output,
            ))

        elif args.command == "segment-review-packet":
            _print_json(build_segment_review_packet(
                review_csv=args.review_csv,
                output_dir=args.output_dir,
                pages_per_document=args.pages_per_document,
                seed=args.seed,
            ))

        elif args.command == "segment-evaluate":
            _print_json(evaluate_segment_review_packet(
                packet_dir=args.packet_dir,
                output_path=args.output,
                matching_threshold=args.matching_iou,
            ))

        return 0

    except Exception as exc:
        logging.exception("Command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
