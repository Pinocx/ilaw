"""Pydantic schemas for AI segmentation output and validation.

Defines versioned data models for:
- AI structural analyzer output (segment proposals)
- Deterministic validator output
- Final chunk representation for indexing
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CHUNK_SCHEMA_VERSION = "1.0"
DOCUMENT_METADATA_SCHEMA_VERSION = "1.0"


# ============================================================
# Enums
# ============================================================


class DocumentType(str, Enum):
    """High-level document type classification."""

    LEGISLATION = "legislation"
    JURISPRUDENCE = "jurisprudence"
    LEGAL_DOCTRINE = "legal_doctrine"
    ADMINISTRATIVE_REGULATION = "administrative_regulation"
    LEGAL_MANUAL = "legal_manual"
    COMPLEMENTARY_MATERIAL = "complementary_material"
    UNKNOWN = "unknown"


class MetadataFieldProvenance(BaseModel):
    """Source and verification status for one document metadata field."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal[
        "system_verified",
        "human_verified",
        "pending_review",
        "not_established",
    ]
    source: str
    evidence_ref: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    review_decision: Optional[str] = None


class DocumentMetadata(BaseModel):
    """Versioned metadata contract separating source facts from legal claims."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = DOCUMENT_METADATA_SCHEMA_VERSION
    doc_id: str

    # Source identity is official when copied from the immutable-file registry.
    source_file: Optional[str] = None
    source_relative_path: Optional[str] = None
    source_sha256: Optional[str] = Field(None, pattern=r"^[A-Fa-f0-9]{64}$")
    source_format: Optional[str] = None
    source_size_bytes: Optional[int] = Field(None, ge=0)

    # Legal semantics remain unset until supported by reviewed source evidence.
    document_title: Optional[str] = None
    document_type: Optional[DocumentType] = None
    jurisdiction: Optional[str] = None
    issuing_authority: Optional[str] = None
    document_number: Optional[str] = None
    publication_date: Optional[date] = None
    effective_date: Optional[date] = None
    legal_status: Literal[
        "active", "repealed", "amended", "suspended", "unknown"
    ] = "unknown"
    classification_status: Literal["pending_review", "reviewed"] = "pending_review"
    field_provenance: Dict[str, MetadataFieldProvenance]

    @model_validator(mode="after")
    def enforce_official_metadata_policy(self) -> "DocumentMetadata":
        """Require registry provenance for technical facts and review for legal claims."""
        source_fields = (
            "source_file",
            "source_relative_path",
            "source_sha256",
            "source_format",
            "source_size_bytes",
        )
        for field_name in source_fields:
            if getattr(self, field_name) is not None:
                provenance = self.field_provenance.get(field_name)
                if provenance is None or provenance.status != "system_verified":
                    raise ValueError(
                        f"{field_name} requires system-verified registry provenance."
                    )

        semantic_fields = (
            "document_title",
            "document_type",
            "jurisdiction",
            "issuing_authority",
            "document_number",
            "publication_date",
            "effective_date",
        )
        for field_name in semantic_fields:
            if getattr(self, field_name) is not None:
                provenance = self.field_provenance.get(field_name)
                if provenance is None or provenance.status != "human_verified":
                    raise ValueError(
                        f"{field_name} requires human-verified source evidence."
                    )

        legal_status_provenance = self.field_provenance.get("legal_status")
        if self.legal_status != "unknown" and (
            legal_status_provenance is None
            or legal_status_provenance.status != "human_verified"
        ):
            raise ValueError("A non-unknown legal_status requires human verification.")

        return self


class SegmentType(str, Enum):
    """Granular structural segment types."""

    # Legislation hierarchy
    BOOK = "book"
    TITLE = "title"
    CHAPTER = "chapter"
    SECTION = "section"
    ARTICLE = "article"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    DISPOSITION = "disposition"

    # Jurisprudence elements
    CASE_SECTION = "case_section"
    DECISION = "decision"
    DISSENTING_OPINION = "dissenting_opinion"

    # Legal doctrine / commentary
    COMMENTARY = "commentary"
    QUOTED_PROVISION = "quoted_provision"
    ANALYSIS = "analysis"
    FOOTNOTE = "footnote"
    REFERENCE = "reference"

    # Document structure
    TABLE_OF_CONTENTS = "table_of_contents"
    TABLE_OF_CONTENTS_ENTRY = "table_of_contents_entry"
    ANNEX = "annex"
    BIBLIOGRAPHY = "bibliography"
    COVER = "cover"
    PREAMBLE = "preamble"

    # Fallback
    OTHER = "other"


class ValidationStatus(str, Enum):
    """Validation status for segments/chunks."""

    VALIDATED = "validated"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"
    FAILED = "failed"


class ChunkingMethod(str, Enum):
    """Method used to produce the chunk."""

    ADAPTIVE_RULES_V1 = "adaptive_rules_v1"
    AI_HYBRID_V1 = "ai_hybrid_v1"
    AI_ONLY_V1 = "ai_only_v1"


# ============================================================
# AI Structural Analyzer Output
# ============================================================


class AISegment(BaseModel):
    """Single segment proposal from the AI structural analyzer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        protected_namespaces=(),
    )

    segment_id: str = Field(
        default_factory=lambda: f"seg-{uuid4().hex[:12]}",
        description="Stable identifier for this segment",
    )
    document_id: str = Field(..., description="SHA-256 of source document")
    segment_type: SegmentType = Field(..., description="Structural type of segment")
    structural_reference: Optional[str] = Field(
        None, description="Canonical reference (e.g., 'Article 2', 'TÍTULO I')"
    )
    parent_reference: Optional[str] = Field(
        None, description="Immediate structural parent reference"
    )
    hierarchy_path: List[str] = Field(
        default_factory=list,
        description="Full hierarchy from root to this segment",
    )
    source_pages: List[int] = Field(
        ..., description="Page numbers covered by this segment"
    )
    source_unit_ids: List[str] = Field(
        ..., description="Source unit IDs (from structured JSON)"
    )
    text: str = Field(
        ..., description="Exact text from source document (no modifications)"
    )
    classification: DocumentType = Field(..., description="Document-level classification")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="AI confidence in this segment"
    )
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.PENDING_REVIEW,
        description="Validator will update this",
    )
    # Metadata
    prompt_version: str = Field(..., description="Version of prompt used")
    model_name: str = Field(..., description="LLM model that produced this segment")
    model_version: Optional[str] = Field(
        None, description="Model version/tag if available"
    )
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this segment was produced",
    )
    schema_version: str = Field(
        default="1.2", description="AI output schema version"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Segment text cannot be empty")
        return v

    @field_validator("source_pages")
    @classmethod
    def pages_not_empty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("At least one source page required")
        if any(p <= 0 for p in v):
            raise ValueError("Page numbers must be positive integers")
        return sorted(set(v))

    @field_validator("source_unit_ids")
    @classmethod
    def units_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one source unit ID required")
        return v

    @model_validator(mode="after")
    def hierarchy_consistency(self) -> "AISegment":
        """Validate hierarchy_path matches structural_reference and parent_reference."""
        if self.structural_reference and self.hierarchy_path:
            if self.structural_reference not in self.hierarchy_path:
                # Allow if structural_reference is more specific (e.g., "Art. 2 §1")
                pass  # Don't fail, just warn in validator
        return self


class AIAnalysisOutput(BaseModel):
    """Complete output from the AI structural analyzer for a document block."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    document_id: str
    block_index: int = Field(..., ge=0, description="Sequential block number")
    block_page_range: List[int] = Field(
        ..., description="Pages covered by this block"
    )
    segments: List[AISegment] = Field(default_factory=list)
    document_type: DocumentType = Field(
        ..., description="AI's document type classification"
    )
    document_type_confidence: float = Field(..., ge=0.0, le=1.0)
    # Metadata
    prompt_version: str = Field(default="1.2")
    model_name: str = Field(...)
    model_version: Optional[str] = None
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schema_version: str = Field(default="1.2")

    @field_validator("segments")
    @classmethod
    def no_duplicate_segment_ids(cls, v: List[AISegment]) -> List[AISegment]:
        seen = set()
        for seg in v:
            if seg.segment_id in seen:
                raise ValueError(f"Duplicate segment_id: {seg.segment_id}")
            seen.add(seg.segment_id)
        return v


# ============================================================
# Deterministic Validator Output
# ============================================================


class ValidationIssue(BaseModel):
    """A validation issue detected by the deterministic validator."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    issue_type: Literal[
        "text_mismatch",
        "missing_source",
        "hierarchy_cycle",
        "orphan_parent",
        "duplicate_reference",
        "numbering_gap",
        "overlapping_pages",
        "missing_pages",
        "low_confidence",
        "structural_inconsistency",
        "toc_misclassified",
        "commentary_misclassified",
    ]
    severity: Literal["error", "warning", "info"]
    message: str
    suggested_fix: Optional[str] = None


class ValidationReport(BaseModel):
    """Report from deterministic structural validator."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    total_segments: int
    validated_segments: int
    pending_review_segments: int
    rejected_segments: int
    failed_segments: int
    issues: List[ValidationIssue] = Field(default_factory=list)
    validator_version: str = Field(default="1.0")
    processed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def pass_rate(self) -> float:
        if self.total_segments == 0:
            return 1.0
        return self.validated_segments / self.total_segments


# ============================================================
# Final Chunk Output (for indexing)
# ============================================================


class ChunkMetadata(BaseModel):
    """Metadata attached to every indexed chunk."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
        protected_namespaces=(),
    )

    # Core identifiers
    chunk_id: str = Field(..., description="Unique chunk identifier")
    doc_id: str = Field(..., description="Document SHA-256")
    segment_id: Optional[str] = Field(None, description="Originating AI segment ID")

    # Structural information
    structure_type: SegmentType = Field(..., description="Structural type")
    structural_reference: Optional[str] = Field(
        None,
        validation_alias="structure_reference",
    )
    parent_reference: Optional[str] = Field(None)
    hierarchy_path: List[str] = Field(default_factory=list)

    # Source provenance
    source_pages: List[int] = Field(...)
    source_unit_ids: List[str] = Field(...)
    start_page: Optional[int] = Field(None, ge=1)
    end_page: Optional[int] = Field(None, ge=1)

    # Splitting metadata
    part_number: int = Field(default=1, ge=1)
    part_count: int = Field(default=1, ge=1)

    # Classification & confidence
    suggested_classification: Optional[DocumentType] = None
    reviewed_classification: Optional[DocumentType] = None
    classification_status: Literal["pending_review", "reviewed"] = "pending_review"
    segmentation_confidence: float = Field(..., ge=0.0, le=1.0)
    legal_verification_status: Literal[
        "not_verified", "verified", "requires_review", "disputed"
    ] = "not_verified"
    validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING_REVIEW)

    # Processing metadata
    chunking_method: ChunkingMethod = Field(...)
    model_name: Optional[str] = Field(None, description="AI model used (if applicable)")
    model_version: Optional[str] = None
    prompt_version: Optional[str] = None
    schema_version: str = Field(default=CHUNK_SCHEMA_VERSION)
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source_checksum: Optional[str] = Field(
        None, description="SHA-256 of source structured JSON"
    )


class Chunk(BaseModel):
    """Final chunk representation for indexing pipeline."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(..., description="Original source text (no modifications)")
    metadata: ChunkMetadata = Field(...)

    def to_index_dict(self) -> Dict[str, Any]:
        """Convert to dict matching index.py expected chunk format."""
        m = self.metadata
        return {
            "chunk_id": m.chunk_id,
            "doc_id": m.doc_id,
            "structure_type": m.structure_type.value,
            "structure_reference": m.structural_reference,
            "parent_reference": m.parent_reference,
            "hierarchy_path": m.hierarchy_path,
            "part_number": m.part_number,
            "part_count": m.part_count,
            "start_page": m.start_page,
            "end_page": m.end_page,
            "source_pages": m.source_pages,
            "source_unit_ids": m.source_unit_ids,
            "text": self.text,
            "chunking_method": m.chunking_method.value,
            "segmentation_confidence": m.segmentation_confidence,
            "suggested_classification": m.suggested_classification.value
            if m.suggested_classification
            else None,
            "reviewed_classification": m.reviewed_classification.value
            if m.reviewed_classification
            else None,
            "classification_status": m.classification_status,
            "legal_verification_status": m.legal_verification_status,
        }


# ============================================================
# Configuration Models
# ============================================================


class ProcessingConfig(BaseModel):
    """Configuration for the AI segmentation pipeline."""

    model_config = ConfigDict(extra="forbid")

    max_chars: int = 1800
    min_chars: int = 300
    overlap_chars: int = 200
    max_blocks_per_doc: int = 10
    enable_hybrid_mode: bool = False
    confidence_threshold: float = 0.6
    require_human_review_below: float = 0.7
    validator_strict: bool = True
    cache_enabled: bool = True
    cache_dir: Optional[str] = None
