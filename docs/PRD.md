# Derecho IA
# Product Requirements Document (PRD)
## Peruvian Legal Normative Repository

**Document version:** 1.0  
**Project:** Derecho IA  
**Repository:** `derecho-ia-normativa-pe`  
**Status:** Active Development  
**Initial jurisdiction:** Republic of Peru  
**Primary language:** Spanish  
**Technical documentation and source code language:** English

---

# 1. Executive Summary

The Peruvian Legal Normative Repository is the foundational document-processing, structuring, classification, segmentation, and indexing system for the Derecho IA ecosystem.

Its purpose is to transform heterogeneous legal documents into a structured, traceable, searchable, and machine-readable legal knowledge repository.

The repository will serve as the authoritative internal document source for future AI-powered legal research, retrieval-augmented generation (RAG), hierarchical retrieval (STAIR), legal analysis, and document-assistance capabilities.

The repository must preserve the relationship between every processed fragment and its original source document, including page references and structural context.

The system must prioritize legal accuracy, source traceability, data integrity, reproducibility, and extensibility.

It is not intended to generate legal opinions or replace professional legal judgment. Its responsibility is to provide reliable, structured legal information for downstream applications.

# 2. Product Vision

Build a maintainable and extensible legal knowledge repository that can:

1. Ingest legal documents from multiple sources and formats.
2. Preserve original documents without modification.
3. Extract text from digital and scanned documents.
4. Classify documents according to their legal nature.
5. Identify and structure legal provisions and other meaningful document sections.
6. Generate traceable, context-aware chunks.
7. Support human and AI-assisted quality validation.
8. Provide reliable textual and semantic retrieval.
9. Support future RAG and STAIR architectures.
10. Expand beyond Peruvian law to additional jurisdictions.

The repository must be independent of any particular AI provider, vector database, or future user interface.

# 3. Scope

## 3.1 In Scope

The repository includes:

- Document ingestion.
- Original document preservation.
- File inventory and metadata extraction.
- Text extraction.
- OCR processing.
- Document classification.
- Legal document type identification.
- Structural segmentation.
- AI-assisted segmentation.
- Deterministic structural validation.
- Chunk generation.
- Source traceability.
- Document and chunk metadata.
- Quality assurance and validation.
- Text-based indexing.
- Embedding generation.
- Vector indexing.
- Hybrid retrieval.
- Retrieval evaluation.
- Export of structured data for downstream applications.

## 3.2 Out of Scope

The following capabilities belong to the future Derecho IA platform and are not part of this repository's core responsibilities:

- User registration and authentication.
- Organization and tenant management.
- Subscription management.
- Billing and payment processing.
- Legal case management.
- Client management.
- User-facing AI chat interface.
- Legal document drafting interface.
- Legal opinion generation.
- Lawyer collaboration tools.
- User-specific document workspaces.

The repository may expose APIs or structured exports to support these future capabilities, but it must not depend on them.

# 4. Target Document Types

The repository must support different legal and legal-related document categories.

## 4.1 Normative Documents

Examples:

- Constitution.
- Laws.
- Legislative decrees.
- Emergency decrees.
- Supreme decrees.
- Ministerial resolutions.
- Administrative resolutions.
- Regulations.
- Codes.
- Statutory amendments.
- Repealing provisions.
- Transitional and complementary provisions.
- International treaties and agreements.

## 4.2 Jurisprudence

Examples:

- Constitutional Court judgments.
- Supreme Court judgments.
- Supreme Court cassation decisions.
- Judicial resolutions.
- Plenary agreements.
- Binding precedents.
- Administrative tribunal decisions.
- Other relevant judicial and administrative decisions.

## 4.3 Legal Doctrine

Examples:

- Legal books.
- Academic papers.
- Legal commentaries.
- Treatises.
- Professional publications.
- Legal manuals.
- Research papers.

## 4.4 Supporting Legal Material

Examples:

- Legal indexes.
- Bibliographies.
- Tables of contents.
- Legal compilations.
- Annotations.
- Historical records.
- Explanatory documents.

The system must distinguish actual legal provisions from references to provisions, commentary, quoted legal text, and table-of-contents entries.

# 5. System Architecture

The repository must use a modular processing pipeline.

## 5.1 High-Level Pipeline

Original Documents
→ Inventory
→ Text Extraction
→ OCR (when required)
→ Document Classification
→ Structural Analysis
→ Legal Segmentation
→ Deterministic Validation
→ Chunk Generation
→ Metadata Enrichment
→ Indexing
→ Embeddings
→ Hybrid Retrieval
→ Retrieval Evaluation

Each stage must be independently executable, testable, and reproducible.

A failure in one stage must not require reprocessing completed previous stages unnecessarily.

## 5.2 Architecture Principles

- Preserve original documents.
- Avoid destructive transformations.
- Separate source data from derived data.
- Separate extraction, classification, segmentation, and indexing.
- Use stable identifiers.
- Maintain document-level and fragment-level provenance.
- Make processing idempotent whenever possible.
- Support incremental processing.
- Avoid unnecessary OCR repetition.
- Avoid unnecessary embedding regeneration.
- Maintain processing history and audit logs.
- Use configurable AI providers.
- Keep deterministic validation independent of AI output.

# 6. Technical Environment

The current development environment is based on:

| Component | Current configuration |
|---|---|
| Operating system | Windows |
| Development environment | Docker Desktop + WSL2 |
| Project directory | `D:\Docker\ILaw` |
| Docker Compose service | `derecho-ia` |
| Docker image | `ilaw-derecho-ia` |
| Python | 3.12 |
| Container base | Python slim |
| OCR | OCRmyPDF + Tesseract |
| PDF extraction | pypdf |
| Data serialization | JSONL |
| Structured data | JSON |
| Main language | Python |

The agent must inspect the actual repository and Docker configuration before making changes. This specification does not authorize replacing existing components simply because an alternative technology is available.

# 7. Data Organization

The repository must maintain a clear separation between original, intermediate, structured, and indexed data.

Expected logical organization:

```text
/app/data/
│
├── incoming_originals/
│   └── Original documents
│
├── inventory/
│   └── Inventory and source metadata
│
├── extracted/
│   └── Extracted text and extraction metadata
│
├── structured/
│   └── Structured document representations
│
├── processing/
│   └── Intermediate processing artifacts
│
├── index/
│   ├── documents.jsonl
│   ├── units.jsonl
│   ├── chunks.jsonl
│   └── index-report.json
│
├── embeddings/
│   └── Embedding artifacts
│
├── validation/
│   ├── validation-reports/
│   └── evaluation-datasets/
│
└── logs/
    └── Processing and error logs
```

This is the intended logical organization. The implementation must first inspect the existing directory structure and preserve compatible paths.

# 8. Original Document Management

Original documents are immutable source material.

Requirements:

1. Never overwrite original files.
2. Generate a stable document identifier.
3. Calculate a cryptographic hash, preferably SHA-256.
4. Detect duplicate files.
5. Preserve original filenames.
6. Record file format, size, and timestamps where available.
7. Preserve source URL and acquisition metadata when available.
8. Track processing status.
9. Maintain processing history.
10. Allow new documents to be added without reprocessing previously completed documents.

A change in processing logic must not automatically trigger OCR or extraction again when existing artifacts remain valid.

# 9. Text Extraction and OCR

## 9.1 Extraction

The system must extract text from supported digital documents while preserving page boundaries whenever possible.

Requirements:

- Preserve page numbering.
- Preserve paragraph order.
- Preserve Unicode characters.
- Handle Spanish legal terminology.
- Detect extraction failures.
- Identify potentially corrupted text.
- Record extraction method and version.
- Preserve references to original pages.

## 9.2 OCR

OCR must be executed only when necessary.

Requirements:

- Detect scanned or image-based pages.
- Support Spanish OCR.
- Record OCR execution status.
- Record OCR engine and configuration.
- Maintain the original document unchanged.
- Support separate OCR-only processing.
- Support skipping OCR when extracted text already exists.
- Support retrying failed OCR jobs.
- Avoid repeated OCR detection when a reliable cached result exists.

OCR processing must be independent from later segmentation improvements.

Changing the segmentation engine must never trigger OCR again.

# 10. Document Classification

Each document must receive a structured classification.

Minimum metadata:

```json
{
  "document_id": "stable-document-id",
  "document_title": "Document title",
  "document_type": "legislation",
  "jurisdiction": "PE",
  "issuing_authority": null,
  "document_number": null,
  "publication_date": null,
  "effective_date": null,
  "legal_status": "unknown",
  "classification_status": "pending_review",
  "classification_confidence": 0.0,
  "source_file": "original-document.pdf"
}
```

The schema is illustrative and must be reconciled with existing data structures before implementation.

Supported document types must include:

- Constitution.
- Legislation.
- Regulation.
- Administrative decision.
- Judicial decision.
- Jurisprudence.
- Legal doctrine.
- Commentary.
- Compilation.
- Other.
- Unknown.

Classification must distinguish a document's actual nature from its subject matter.

For example, a book commenting on the Criminal Code must not be classified as legislation merely because it contains article references.

AI-generated classifications must be identified as suggestions until verified.

# 11. Legal Segmentation Engine

## 11.1 Objective

The segmentation engine must identify meaningful legal and documentary structures rather than simply dividing text by character count or token count.

The system must recognize:

- Books.
- Parts.
- Titles.
- Chapters.
- Sections.
- Subsections.
- Articles.
- Paragraphs.
- Numerals.
- Literals.
- Provisions.
- Final provisions.
- Transitional provisions.
- Complementary provisions.
- Judicial decision sections.
- Legal reasoning.
- Legal commentary.
- Bibliographies.
- Tables of contents.
- Other relevant structures.

The hierarchy must reflect the actual source document.

## 11.2 Hybrid Segmentation Architecture

The preferred architecture is hybrid:

1. Rules-based pre-processing.
2. AI-assisted structural analysis.
3. Deterministic validation.
4. Source-based chunk construction.
5. Human review when required.

AI is responsible for interpreting document structure and context.

Python is responsible for verifying source boundaries, references, hierarchy, integrity, and traceability.

The existing rules-based segmentation engine must remain available as a fallback.

## 11.3 Segmentation Modes

The engine must support:

- `rules`
- `ai`
- `hybrid`

The selected mode must be configurable.

## 11.4 Important Legal Distinctions

The engine must distinguish at least:

- `legal_provision`: An actual provision in the source.
- `legal_reference`: A reference to a provision elsewhere.
- `quoted_provision`: A provision quoted within another document.
- `commentary`: Author interpretation or explanation.
- `table_of_contents_entry`: An index entry.
- `bibliographic_reference`: A bibliographic citation.
- `unclassified`: Content that cannot be reliably categorized.

An article reference in a book or table of contents must not automatically become a legal article segment.

## 11.5 Source Anchoring

The AI must not be trusted to calculate exact character offsets.

The system must:

1. Assign stable identifiers to source lines or blocks.
2. Ask the AI to identify source block identifiers.
3. Support multiple source spans per segment.
4. Allow textual anchors for partial blocks.
5. Calculate exact character offsets deterministically in Python.

Offsets must use a clearly documented convention, preferably half-open intervals `[start, end)`.

Offsets must refer to the original extracted text representation, not a silently normalized version.

## 11.6 Segmentation Confidence

Each segment should contain:

- Segmentation method.
- Confidence score.
- Validation status.
- Source spans.
- Parent structure.
- Structural type.
- Review status.

Confidence scores must not be interpreted as proof of legal correctness.

# 12. Chunk Generation

Chunks must be generated from validated source segments.

Requirements:

- Preserve legal structure.
- Preserve source order.
- Preserve parent-child relationships.
- Preserve page references.
- Avoid duplicated content.
- Avoid silently omitted content.
- Avoid combining unrelated legal provisions.
- Support long provisions split into multiple chunks.
- Maintain the relationship between chunks and their source units.
- Maintain stable identifiers when the underlying content has not changed.

A chunk may represent an article, section, paragraph, legal reasoning segment, commentary passage, or another meaningful unit.

For long segments, chunking must preserve the original legal hierarchy and indicate part numbers.

Example:

```json
{
  "chunk_id": "document-id:chunk-000001",
  "document_id": "document-id",
  "structure_type": "article",
  "structure_reference": "Article 1",
  "parent_reference": "Title I / Chapter I",
  "part_number": 1,
  "part_count": 1,
  "source_pages": [1],
  "source_unit_ids": ["unit-000001"],
  "chunking_method": "adaptive_rules_v1",
  "segmentation_confidence": 0.95,
  "legal_verification_status": "not_verified",
  "text": "Original source text..."
}
```

This is an illustrative target, not a mandate to rename existing fields without migration planning.

# 13. Data Integrity and Traceability

This is one of the most important requirements.

Every chunk must be traceable to:

1. Its original document.
2. Its source page or pages.
3. Its source unit or units.
4. Its structural parent.
5. Its extraction history.
6. Its segmentation method.
7. Its validation status.

The system must support reconstruction of the original source context surrounding a chunk.

## 13.1 Source Text Coverage

The system must calculate source-text coverage.

Coverage must be calculated using non-whitespace characters from the source text.

Every meaningful source fragment must either:

- Belong to a validated segment, or
- Be explicitly identified as unclassified, noise, or another documented category with a reason.

No significant content may be silently discarded.

The system must identify:

- Missing source text.
- Duplicate spans.
- Overlapping spans where not permitted.
- Invalid offsets.
- Invalid source references.
- Incorrect hierarchy.
- Broken document relationships.

# 14. Validation and Quality Assurance

Validation must be performed at several levels.

## 14.1 Technical Validation

- JSON schema validity.
- Stable identifiers.
- Valid source references.
- Valid page references.
- Valid source offsets.
- Duplicate detection.
- Hierarchy consistency.
- Chunk integrity.
- Source coverage.
- Processing status consistency.

## 14.2 Structural Validation

- Articles must be actual provisions, not index references.
- Parent structures must be valid.
- Child structures must be consistent.
- Document structure must follow source order.
- Commentary must not be confused with legislation.
- Quoted provisions must remain distinguishable from actual provisions.

## 14.3 Legal Verification

The system must distinguish technical correctness from legal correctness.

A technically valid segment is not necessarily legally verified.

Supported statuses should include:

- `not_verified`
- `pending_review`
- `verified`
- `rejected`
- `needs_correction`

Only an authorized human reviewer may mark legal content as verified.

## 14.4 Ground Truth and Evaluation

The project must maintain a human-confirmed evaluation dataset.

The evaluation process must include:

- A small, representative pilot dataset.
- Documents from different legal categories.
- Development examples.
- A separate held-out evaluation set.
- Human-confirmed reference labels.
- Baseline metrics from the existing segmentation engine.

The AI development agent must not create or modify ground-truth labels autonomously.

Held-out examples must not be used for iterative tuning.

Where feasible, the evaluation process must prevent the development agent from seeing individual held-out failures until final evaluation.

The evaluator should provide aggregate results during development.

# 15. AI Provider Architecture

The repository must not be permanently coupled to a specific AI provider.

The architecture must support:

- Local AI models.
- Ollama.
- OpenAI-compatible endpoints.
- Cloud AI providers.
- Future providers through adapters.

The development agent's AI model and the runtime segmentation model are separate concepts.

The development agent may use whichever AI model is available in its development environment.

The repository's runtime AI provider must be configured independently.

No paid cloud AI service may be activated without explicit authorization.

If Ollama is selected, the system must inspect:

- Available models.
- Context window configuration.
- Actual request context limits.
- Output token limits.
- Truncation risks.
- Timeout and retry behavior.

The system must not assume that a model supports a context size simply because a configuration value was requested.

AI responses must be validated against explicit schemas.

Invalid or incomplete AI responses must not be silently accepted.

# 16. Indexing and Retrieval

## 16.1 Current Structured Index

The current index is based on JSONL artifacts:

- `documents.jsonl`
- `units.jsonl`
- `chunks.jsonl`
- `index-report.json`

The actual schema and current index contents must be inspected before any changes.

These files are derived representations, not authoritative legal sources.

## 16.2 Text Search

The repository must support exact and full-text search.

Examples:

- Search by article number.
- Search by law number.
- Search by document title.
- Search for exact phrases.
- Search by legal concept.
- Search by issuing authority.
- Search by date.
- Search by document category.

## 16.3 Semantic Search

The repository must eventually support embeddings and vector search.

The embedding model must be configurable and replaceable.

Embedding metadata must include:

- Model name.
- Model version.
- Embedding dimensions.
- Generation date.
- Source chunk identifier.
- Text hash.

Changing the embedding model must not require OCR or document segmentation again.

## 16.4 Hybrid Retrieval

The final retrieval system must combine:

- Exact textual search.
- Full-text search.
- Semantic similarity.
- Legal metadata filters.
- Structural relationships.

The system should support reranking and query expansion where appropriate.

# 17. Hierarchical Retrieval (STAIR)

The repository must be designed to support hierarchical retrieval.

The retrieval engine should be able to move through levels of legal context.

Example:

1. Identify the relevant document.
2. Identify the relevant book, title, or chapter.
3. Retrieve the relevant article or section.
4. Retrieve the relevant paragraph or fragment.
5. Expand context when necessary.

The system must preserve parent-child relationships so that retrieval can return both a precise fragment and its broader legal context.

STAIR must not be reduced to ordinary vector similarity search.

# 18. Legal Source Provenance

Every document should record available source information.

Metadata should include:

- Source institution.
- Source URL.
- Acquisition date.
- Original filename.
- Document identifier.
- Publication information.
- Issuing authority.
- Jurisdiction.
- Source reliability or authority category.

The repository must distinguish:

- Official source documents.
- Official reproductions.
- Secondary compilations.
- Academic or professional publications.
- Unverified external documents.

The system must not claim that a document is an official legal source unless that status is supported by evidence.

Where possible, legal documents should be linked to their official publication or issuing authority.

# 19. Incremental Processing

The repository must support incremental processing.

When a new document is added:

1. Detect the new file.
2. Calculate its hash.
3. Check for duplicates.
4. Add it to inventory.
5. Process only the required stages.
6. Generate structured data.
7. Validate it.
8. Update relevant indexes.

Previously completed documents must not be unnecessarily reprocessed.

A segmentation engine update should permit rebuilding structured segments and chunks without repeating extraction or OCR.

An embedding model update should permit regenerating embeddings without repeating OCR or segmentation.

The system must maintain clear dependency and invalidation rules between processing stages.

# 20. Reporting and Monitoring

The repository must provide operational reports.

Minimum statistics:

- Total documents.
- Documents by category.
- Documents by processing status.
- Documents requiring OCR.
- Extraction failures.
- Classification failures.
- Segmentation failures.
- Total units.
- Total chunks.
- Chunks by structure type.
- Chunks requiring review.
- Source coverage.
- Duplicate chunks.
- Invalid references.
- Hierarchy errors.
- Processing duration.

The `status` command must provide an accurate overview of repository health.

The system should support:

```text
inventory
classify
process
validate
index
status
```

Existing CLI commands must be inspected and preserved where appropriate.

# 21. Security and Data Governance

The repository must:

- Preserve original source material.
- Maintain processing logs.
- Avoid embedding secrets in source code.
- Use environment variables for credentials.
- Avoid sending documents to cloud AI providers without explicit configuration and authorization.
- Record which AI provider processed a document.
- Support access restrictions for future APIs.
- Maintain source provenance.
- Distinguish public legal documents from restricted or licensed material.

Copyright and licensing restrictions must be considered when importing and distributing legal books, commentaries, and academic publications.

Public accessibility does not automatically imply unrestricted redistribution rights.

# 22. Testing Strategy

Testing must include:

## Unit Tests

- Document identification.
- Classification.
- Structure detection.
- Article detection.
- Source anchoring.
- Parent-child relationships.
- Chunk generation.
- Duplicate detection.
- Coverage calculation.
- Metadata validation.

## Integration Tests

- Complete processing pipeline.
- Incremental document ingestion.
- Rebuilding chunks without OCR.
- Rebuilding indexes.
- Provider failures.
- Invalid AI responses.
- Recovery after processing interruptions.

## Regression Tests

Maintain representative documents for:

- Legislation.
- Codes.
- Regulations.
- Judicial decisions.
- Legal books.
- Commentaries.
- Mixed documents.
- Documents with extensive indexes.
- Scanned documents.
- Documents with complex hierarchies.

Every segmentation correction must be tested against representative documents to ensure that previously working cases do not regress.

# 23. Performance and Reliability

The system should:

- Process documents incrementally.
- Support batch processing.
- Support recoverable jobs.
- Avoid unnecessary memory consumption.
- Log processing duration.
- Support configurable AI timeouts.
- Handle large documents.
- Avoid silent failures.
- Provide actionable error messages.

Performance optimization must not compromise legal structure or source traceability.

# 24. Development Roadmap

The implementation must follow these phases.

## Phase 1 — Repository Audit and Baseline

- Inspect the existing codebase.
- Inspect data directories.
- Inspect document, unit, and chunk schemas.
- Verify the current document inventory.
- Verify processing statuses.
- Inspect segmentation implementation.
- Inspect current index generation.
- Run existing tests.
- Establish baseline quality metrics.

Deliverable: Repository audit and baseline report.

## Phase 2 — Segmentation Quality and Validation

- Audit representative chunks.
- Validate document classification.
- Validate article detection.
- Validate structural hierarchy.
- Validate source references.
- Implement or improve AI-assisted segmentation.
- Maintain rules-based fallback.
- Establish human-confirmed ground truth.
- Evaluate segmentation quality.

Deliverable: Validated segmentation engine and quality report.

## Phase 3 — Metadata Enrichment

- Normalize document metadata.
- Identify legal references.
- Improve legal document classification.
- Add available authority and publication information.
- Track legal verification status.

Deliverable: Consistent structured legal metadata.

## Phase 4 — Text Search

- Implement a textual search index.
- Support exact article and law-number queries.
- Support full-text search.
- Support metadata filters.
- Test retrieval against representative legal questions.

Deliverable: Working legal textual search engine.

## Phase 5 — Embeddings and Vector Search

- Select an appropriate embedding model.
- Generate embeddings from validated chunks.
- Store vectors and metadata.
- Implement semantic search.
- Evaluate retrieval quality.

Deliverable: Semantic search capability.

## Phase 6 — Hybrid Retrieval and STAIR

- Combine text and semantic retrieval.
- Implement hierarchical retrieval.
- Retrieve parent context.
- Implement reranking where useful.
- Evaluate relevance and source accuracy.

Deliverable: Hybrid legal retrieval engine.

## Phase 7 — RAG Integration

- Define retrieval context format.
- Define AI provider interface.
- Integrate configurable language models.
- Enforce source-based answers.
- Require citations.
- Handle insufficient evidence.
- Test hallucination and citation accuracy.

Deliverable: Repository-ready RAG interface for Derecho IA.

## Phase 8 — Operational Readiness

- Document APIs and data contracts.
- Improve monitoring.
- Validate incremental processing.
- Validate backups and recovery.
- Establish release and versioning procedures.

Deliverable: Stable repository ready for integration with the Derecho IA platform.

# 25. Acceptance Criteria

The repository must satisfy the following criteria before being considered ready for downstream AI integration.

## Document Integrity

- Original documents remain unchanged.
- Every processed document has a stable identifier.
- Duplicate detection works.
- Processing history is available.

## Extraction

- Page references are preserved where possible.
- OCR status is recorded.
- Extraction failures are reported.
- Segmentation changes do not trigger OCR unnecessarily.

## Segmentation

- Actual legal provisions are distinguished from references and commentary.
- Structural hierarchy is consistent.
- Chunks are traceable to source documents.
- No significant text is silently omitted.
- Invalid source references are rejected.
- Ambiguous segments can be flagged for human review.

## Retrieval

- Exact legal references can be searched.
- Full-text retrieval works.
- Semantic retrieval works after embedding implementation.
- Parent context can be retrieved.
- Search results include source references.

## AI Integration

- AI providers are configurable.
- Invalid AI output is rejected.
- Responses can be traced to retrieved chunks.
- Insufficient evidence can be reported.
- No paid provider is activated without authorization.

## Quality Evaluation

The initial segmentation pilot should target:

- Precision: at least 95%.
- Recall: at least 90%.
- F1 score: at least 92%.
- Source-text coverage: at least 99%.
- Zero critical hierarchy errors.
- Zero invalid source references.
- Zero unexplained content loss.

These are pilot acceptance targets, not claims about the current repository. Results must be measured on a human-confirmed dataset.

# 26. Current Project Status and Continuation Instructions

The project has already undergone document ingestion, text extraction, OCR processing, structural processing, and chunk generation.

The latest known working environment is:

```text
Project: D:\Docker\ILaw
Docker service: derecho-ia
Docker image: ilaw-derecho-ia

Source documents:
  /app/data/incoming_originals/

Structured documents:
  /app/data/structured/

Index:
  /app/data/index/

Index artifacts:
  documents.jsonl
  units.jsonl
  chunks.jsonl
  index-report.json
```

The project has previously processed approximately 326 documents. The exact current counts and status must be verified by inspecting the repository.

The previous segmentation implementation used a rules-based engine. An AI-assisted hybrid segmentation architecture has been specified as the preferred direction.

The repository has also undergone corrections to article detection and structural hierarchy. These changes must be preserved and regression-tested.

The current chunk index is a derived artifact and must not be treated as a legally verified corpus.

## Mandatory Continuation Rules

The development agent must:

1. Inspect the current repository before proposing changes.
2. Read existing documentation and code.
3. Inspect actual data schemas and processing reports.
4. Verify current document and chunk counts.
5. Identify completed and pending roadmap items.
6. Preserve all existing working functionality.
7. Never repeat completed OCR without a demonstrated need.
8. Never modify original source documents.
9. Never overwrite the active index without explicit approval.
10. Never generate embeddings before segmentation quality has been reviewed and approved.
11. Never invent current project status or test results.
12. Avoid unnecessary changes to existing modules.
13. Use English for source code, comments, identifiers, technical documentation, and prompts.
14. Explain findings and recommendations to the user in Spanish.
15. Provide reproducible PowerShell and Docker commands for Windows.

## Immediate Next Task

The first task is NOT to implement a new search engine or generate embeddings.

The agent must perform a read-only audit of the existing repository and answer:

1. What is the actual current state of the repository?
2. How many documents, units, and chunks exist?
3. What are the current document and chunk classifications?
4. What proportion of chunks has valid source references?
5. Are article segments correctly distinguished from references and table-of-contents entries?
6. Are parent-child relationships correct?
7. Is source-text coverage measurable with the current data?
8. What validation tools and tests already exist?
9. Which roadmap phases are complete, partially complete, or pending?
10. What is the smallest safe next implementation step?

The agent must produce a report in Spanish, with evidence from the actual repository.

It must not modify any files during this initial audit.

It must wait for explicit user approval before beginning implementation.

# 27. Long-Term Product Direction

The repository is the knowledge foundation of the Derecho IA platform.

Future applications should be able to consume its information through stable APIs or data contracts.

The repository must remain independently maintainable and reusable.

The long-term architecture should support:

- Multiple jurisdictions.
- Multiple languages.
- Multiple legal sources.
- Multiple AI providers.
- Multiple embedding models.
- Multiple downstream applications.
- Versioned legal documents.
- Historical legal versions.
- Temporal legal validity.
- Cross-reference analysis.
- Legal concept relationships.
- Explainable and traceable retrieval.

The fundamental design principle is:

**Every legal answer generated by a downstream system must be traceable to the source documents and specific source passages used to produce it.**

The repository must prioritize legal reliability over the apparent sophistication of the AI model.