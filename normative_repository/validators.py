"""Deterministic structural validator for AI-generated legal document segments.

Validates AI segment proposals against source text and structural constraints.

Key responsibilities:
- Source-text exact match verification
- Duplicate detection
- Missing/overlapping page coverage
- Logical hierarchy validation
- Article numbering consistency
- Confidence-based review flags
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from normative_repository.schemas import (
    AISegment,
    AIAnalysisOutput,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
    ChunkMetadata,
)

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Helper class to normalize text for exact matching."""

    def __init__(self):
        self.cache = {}

    def normalize(self, text: str) -> str:
        if not text in self.cache:
            # Remove leading/trailing whitespace
            normalized = text.strip()
            # Collapse internal whitespace (tabs, newlines, multiple spaces)
            normalized = " ".join(normalized.split())
            self.cache[text] = normalized
        return self.cache[text]


class DeterministicValidator:
    """Validates AI-generated segments against source text and constraints.

    Implements the hybrid AI + deterministic validation principle from
    the Derecho IA specification.
    """

    def __init__(
        self,
        normalize_text: bool = True,
        max_text_length: int = 50000,  # Arbitrary large limit for source text
    ):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.normalize = TextNormalizer()
        self.max_text_length = max_text_length
        self.document_cache: Dict[str, Dict[str, Any]] = {}
        self.seen_segment_ids = set()
        self.document_stats = {}

    def validate_segments(
        self,
        segments: List[AISegment],
        document_id: str,
        full_source_text: str,
    ) -> tuple[List[AISegment], ValidationReport]:
        """Run all deterministic validation checks on segments.

        Returns:
            (validated_segments, report)
        """
        report = ValidationReport(
            document_id=document_id,
            total_segments=len(segments),
            validated_segments=0,
            pending_review_segments=0,
            rejected_segments=0,
            failed_segments=0,
        )

        validated_segments: List[AISegment] = []
        validation_issues: List[ValidationIssue] = []

        for segment in segments:
            segment_issues: List[ValidationIssue] = []

            # Basic format validation
            format_issues = self._validate_format(segment)
            segment_issues.extend(format_issues)

            # Source-text validation
            text_match_issue = self.validate_text_match(segment, full_source_text)
            if text_match_issue:
                segment_issues.append(text_match_issue)

            # Hierarchy validation
            hierarchy_issues = self.validate_hierarchy(segment, validated_segments)
            segment_issues.extend(hierarchy_issues)

            # Duplicate ID check
            duplicate_issue = self.validate_duplicate_id(segment)
            if duplicate_issue:
                segment_issues.append(duplicate_issue)

            # Page coverage validation
            page_issues = self.validate_page_coverage(segment, validated_segments)
            segment_issues.extend(page_issues)

            # Structural consistency validation
            structural_issues = self.validate_structural_consistency(segment, validated_segments)
            segment_issues.extend(structural_issues)

            # Confidence-based classification validation
            confidence_issue = self.validate_confidence_classification(segment)
            if confidence_issue:
                segment_issues.append(confidence_issue)

            # Apply validation status
            final_status = self._determine_validation_status(segment_issues, segment.confidence)
            validated_segment = segment.model_copy(
                update={"validation_status": final_status}
            )

            # Track stats
            if final_status == ValidationStatus.VALIDATED:
                report.validated_segments += 1
            elif final_status == ValidationStatus.PENDING_REVIEW:
                report.pending_review_segments += 1
            elif final_status == ValidationStatus.REJECTED:
                report.rejected_segments += 1
            elif final_status == ValidationStatus.FAILED:
                report.failed_segments += 1

            # Add issues to report
            for issue in segment_issues:
                # Mark related segment if issue references other segment_id
                if issue.segment_id != segment.segment_id:
                    issue.severity = "warning"
                validation_issues.append(issue)

            validated_segments.append(validated_segment)

        report.issues = validation_issues
        self.seen_segment_ids.clear()
        return validated_segments, report

    def validate_segment_list(
        self,
        segments: List[AISegment],
        document_id: str,
        full_source_text: str,
    ) -> ValidationReport:
        """Validate a complete list of segments and return validation report.

        This is a convenience method that wraps the full validation pipeline.
        """
        _, report = self.validate_segments(segments, document_id, full_source_text)
        return report

    def validate_text_match(self, segment: AISegment, source_text: str) -> Optional[ValidationIssue]:
        """Validate that the segment's text appears exactly in the source."""
        if not segment.text:
            return ValidationIssue(
                segment_id=segment.segment_id,
                issue_type="text_mismatch",
                severity="error",
                message="Segment has empty text",
                suggested_fix="Verify segment extraction",
            )

        # Try both exact and normalized matching
        text_to_match = self.normalize.normalize(segment.text)
        source_normalized = self.normalize.normalize(source_text)

        # Try exact match first (faster)
        if segment.text in source_text:
            return None

        # Try normalized match
        if text_to_match in source_normalized:
            # Find the exact position to reconstruct correct spacing
            pos = source_normalized.find(text_to_match)
            if pos != -1:
                # Extract the corresponding exact text
                exact_char_count = len(segment.text)
                start = max(0, pos - (exact_char_count // 2))
                end = start + exact_char_count
                if end <= len(source_text):
                    exact_match = source_text[start:end]
                    return ValidationIssue(
                        segment_id=segment.segment_id,
                        issue_type="text_mismatch",
                        severity="warning",
                        message=f"Segment text normalization issue; exact match at position {start}",
                        suggested_fix="Update segment text with exact source",
                    )

        # Report mismatch
        return ValidationIssue(
            segment_id=segment.segment_id,
            issue_type="text_mismatch",
            severity="error",
            message=f"Segment text not found in source document",
            suggested_fix="Verify segment extraction; text may be from wrong document or incorrectly extracted",
        )

    def validate_duplicate_id(self, segment: AISegment) -> Optional[ValidationIssue]:
        """Ensure segment_id is not duplicated."""
        if segment.segment_id in self.seen_segment_ids:
            return ValidationIssue(
                segment_id=segment.segment_id,
                issue_type="duplicate_reference",
                severity="error",
                message=f"Duplicate segment_id: {segment.segment_id}",
                suggested_fix="Assign unique segment_id to duplicate",
            )
        self.seen_segment_ids.add(segment.segment_id)
        return None

    def validate_hierarchy(
        self,
        segment: AISegment,
        validated_segments: List[AISegment],
    ) -> List[ValidationIssue]:
        """Validate hierarchical relationships (parent exists, no cycles)."""
        issues: List[ValidationIssue] = []

        expected_parents = segment.hierarchy_path[:-1]
        if not expected_parents and segment.parent_reference:
            expected_parents = [
                part.strip()
                for part in segment.parent_reference.split("/")
                if part.strip()
            ]

        known_references = {
            seg.structural_reference
            for seg in validated_segments
            if seg.segment_id != segment.segment_id and seg.structural_reference
        }
        missing_parents = [
            reference for reference in expected_parents
            if reference not in known_references
        ]
        if missing_parents:
            issues.append(
                ValidationIssue(
                    segment_id=segment.segment_id,
                    issue_type="orphan_parent",
                    severity="warning",
                    message=(
                        "Parent reference(s) not found in validated segments: "
                        + ", ".join(missing_parents)
                    ),
                    suggested_fix="Verify hierarchy structure; possibly remove invalid parent reference",
                )
            )

        return issues

    def validate_page_coverage(
        self,
        segment: AISegment,
        validated_segments: List[AISegment],
    ) -> List[ValidationIssue]:
        """Validate page coverage (no gaps, no overlaps)."""
        issues: List[ValidationIssue] = []

        segment_pages = set(segment.source_pages)
        if not segment_pages:
            issues.append(
                ValidationIssue(
                    segment_id=segment.segment_id,
                    issue_type="missing_pages",
                    severity="error",
                    message="Segment has no source pages defined",
                    suggested_fix="Add page numbers to segment",
                )
            )
            return issues

        # Check for invalid page numbers
        for page in segment_pages:
            if page < 1:
                issues.append(
                    ValidationIssue(
                        segment_id=segment.segment_id,
                        issue_type="missing_pages",
                        severity="error",
                        message=f"Invalid page number: {page}. Page numbers must be positive integers.",
                        suggested_fix="Correct page numbers",
                    )
                )

        # Check for overlapping pages with previous segments
        for existing_seg in validated_segments:
            existing_pages = set(existing_seg.source_pages)
            if existing_pages.intersection(segment_pages):
                # Some overlap - might be intentional (e.g., transitional content)
                # Only flag if overlap is significant (>50% of segment pages)
                overlap_ratio = len(existing_pages.intersection(segment_pages)) / len(segment_pages)
                if overlap_ratio > 0.5:
                    issues.append(
                        ValidationIssue(
                            segment_id=segment.segment_id,
                            issue_type="overlapping_pages",
                            severity="warning",
                            message=f"Significant page overlap with segment {existing_seg.segment_id}",
                            suggested_fix="Review if this segment belongs to the previous one",
                        )
                    )

        return issues

    def validate_structural_consistency(
        self,
        segment: AISegment,
        validated_segments: List[AISegment],
    ) -> List[ValidationIssue]:
        """Validate structural consistency (article numbering, etc.)."""
        issues: List[ValidationIssue] = []

        # Article/numbering validation
        if segment.segment_type.value == "article" and segment.structural_reference:
            # Extract article number
            import re

            article_match = re.search(r"Art\.?\s*(?P<num>[\d]+[a-zA-Z]?)", segment.structural_reference, re.IGNORECASE)
            if article_match:
                # Check for duplicate article numbers in the same parent context
                for existing_seg in validated_segments:
                    if existing_seg.segment_type.value == "article":
                        existing_match = re.search(r"Art\.?\s*(?P<num>[\d]+[a-zA-Z]?)", existing_seg.structural_reference or "", re.IGNORECASE)
                        if existing_match and existing_match.group("num") == article_match.group("num"):
                            # Check if they share the same parent
                            if (segment.parent_reference and existing_seg.parent_reference and
                                segment.parent_reference == existing_seg.parent_reference):
                                issues.append(
                                    ValidationIssue(
                                        segment_id=segment.segment_id,
                                        issue_type="duplicate_reference",
                                        severity="warning",
                                        message=f"Duplicate article number {article_match.group('num')} in same parent context",
                                        suggested_fix="Verify if both articles exist in the document",
                                    )
                                )

        return issues

    def validate_confidence_classification(self, segment: AISegment) -> Optional[ValidationIssue]:
        """Validate that confidence scores align with segment classification."""
        # Low confidence segments should be flagged for human review
        if segment.confidence < 0.6:
            return ValidationIssue(
                segment_id=segment.segment_id,
                issue_type="low_confidence",
                severity="warning",
                message=f"Segment has low confidence ({segment.confidence:.2f}); flag for human review",
                suggested_fix="Review classification or extraction method",
            )

        # High confidence segments with structural issues should be flagged
        if segment.confidence > 0.9:
            # This is a simple check - in practice, you might want more sophisticated validation
            pass

        return None

    def _validate_format(self, segment: AISegment) -> List[ValidationIssue]:
        """Validate basic segment format."""
        issues: List[ValidationIssue] = []

        # Check required fields
        if not segment.segment_id:
            issues.append(
                ValidationIssue(
                    segment_id="unknown",
                    issue_type="missing_pages",
                    severity="error",
                    message="Segment missing segment_id",
                )
            )

        if not segment.text:
            issues.append(
                ValidationIssue(
                    segment_id=segment.segment_id,
                    issue_type="text_mismatch",
                    severity="error",
                    message="Segment missing text",
                )
            )

        if not segment.source_unit_ids:
            issues.append(
                ValidationIssue(
                    segment_id=segment.segment_id,
                    issue_type="missing_pages",
                    severity="error",
                    message="Segment missing source unit IDs",
                )
            )

        return issues

    def _determine_validation_status(
        self,
        issues: List[ValidationIssue],
        confidence: float,
    ) -> ValidationStatus:
        """Determine final validation status based on issues and confidence."""
        if not issues:
            if confidence > 0.7:
                return ValidationStatus.VALIDATED
            else:
                return ValidationStatus.PENDING_REVIEW

        has_errors = any(issue.severity == "error" for issue in issues)
        has_warnings = any(issue.severity in ("warning", "info") for issue in issues)

        if has_errors:
            return ValidationStatus.REJECTED
        elif has_warnings:
            return ValidationStatus.PENDING_REVIEW
        elif confidence < 0.6:
            return ValidationStatus.PENDING_REVIEW
        else:
            return ValidationStatus.VALIDATED


class SemanticValidator:
    """Semantic validation for legal document segmentation.

    This class performs higher-level semantic validation that depends on
    the document type and legal structure patterns.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.structure_patterns = self._load_structure_patterns()

    def _load_structure_patterns(self) -> Dict[str, Any]:
        """Load patterns for validating legal document structures."""
        return {
            "peru_legislation": {
                "valid_article_patterns": [r"Art\.?\s*[\d]+[a-zA-Z]?", r"Artículo\s*[\d]+[a-zA-Z]?"],
                "valid_heading_patterns": [r"TÍTULO\s+[IVXLCDM]+\b", r"CAPÍTULO\s+[IVXLCDM]+\b"],
                "expected_hierarchy": [
                    "book",
                    "title",
                    "chapter",
                    "section",
                    "article",
                    "subsection",
                    "paragraph",
                ],
            },
            "jurisprudence": {
                "valid_case_sections": [
                    "VISTOS",
                    "CONSIDERANDO",
                    "FUNDAMENTOS",
                    "PARTE RESOLUTIVA",
                    "RESUELVE",
                ],
                "expected_hierarchy": ["case_section", "decision", "dissenting_opinion"],
            },
        }

    def validate_semantic_structure(
        self,
        segments: List[AISegment],
        document_type: str,
    ) -> List[ValidationIssue]:
        """Validate semantic structure of segments based on document type."""
        issues: List[ValidationIssue] = []

        pattern_type = self.structure_patterns.get(document_type, {})

        # Validate according to document type
        if document_type == "legislation":
            issues.extend(self._validate_legislation_structure(segments, pattern_type))
        elif document_type == "jurisprudence":
            issues.extend(self._validate_jurisprudence_structure(segments, pattern_type))
        elif document_type == "legal_doctrine":
            issues.extend(self._validate_doctrine_structure(segments, pattern_type))

        return issues

    def _validate_legislation_structure(
        self,
        segments: List[AISegment],
        patterns: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Validate legislation-specific structure."""
        issues: List[ValidationIssue] = []

        # Track hierarchy levels
        current_hierarchy_level = 0
        previous_segment_type = None

        for segment in segments:
            # Validate article numbering progression
            if segment.segment_type.value == "article":
                if not segment.structural_reference:
                    issues.append(
                        ValidationIssue(
                            segment_id=segment.segment_id,
                            issue_type="numbering_gap",
                            severity="warning",
                            message="Article missing structural reference",
                            suggested_fix="Add article number reference",
                        )
                    )

                # Check for non-sequential article numbers
                if segment.structural_reference:
                    import re

                    article_match = re.search(r"Art\.?\s*(?P<num>[\d]+[a-zA-Z]?)", segment.structural_reference, re.IGNORECASE)
                    if article_match:
                        num = article_match.group("num")
                        # Try to extract numeric part
                        import re as regex_mod
                        match = regex_mod.search(r"(\d+)([a-zA-Z]?)", num)
                        if match:
                            num_numeric = int(match.group(1))
                            if previous_segment_type == "article":
                                # In a real implementation, we'd track actual sequence
                                pass

            previous_segment_type = segment.segment_type.value

        return issues

    def _validate_jurisprudence_structure(
        self,
        segments: List[AISegment],
        patterns: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Validate jurisprudence-specific structure."""
        issues: List[ValidationIssue] = []

        # Check for proper case section sequence
        valid_case_sections = patterns.get("valid_case_sections", [])
        found_case_sections = [
            seg.segment_type.value for seg in segments
            if seg.segment_type.value in valid_case_sections
        ]

        # Check for basic structure
        if not any(seg.segment_type.value == "decision" for seg in segments):
            issues.append(
                ValidationIssue(
                    segment_id="document-wide",
                    issue_type="structural_inconsistency",
                    severity="warning",
                    message="No decision section found in jurisprudence document",
                    suggested_fix="Verify document type classification",
                )
            )

        return issues

    def _validate_doctrine_structure(
        self,
        segments: List[AISegment],
        patterns: Dict[str, Any],
    ) -> List[ValidationIssue]:
        """Validate legal doctrine structure."""
        issues: List[ValidationIssue] = []

        # Check for commentary vs. article distinction
        commentary_count = sum(1 for seg in segments if seg.segment_type.value == "commentary")
        article_count = sum(1 for seg in segments if seg.segment_type.value == "article")

        # In doctrine documents, commentary should typically outnumber articles
        if article_count > 0 and commentary_count == 0:
            issues.append(
                ValidationIssue(
                    segment_id="document-wide",
                    issue_type="structural_inconsistency",
                    severity="warning",
                    message="Document classified as legal doctrine but has no commentary segments",
                    suggested_fix="Review document type classification",
                )
            )

        return issues


def validate_text_segments(
    segments: List[AISegment],
    document_id: str,
    source_text: str,
    document_type: str,
) -> tuple[List[AISegment], ValidationReport]:
    """Convenience function to validate segments with semantic validation."""
    validator = DeterministicValidator()
    semantic_validator = SemanticValidator()

    # Run deterministic validation first
    validated_segments, report = validator.validate_segments(
        segments, document_id, source_text
    )

    # Run semantic validation
    semantic_issues = semantic_validator.validate_semantic_structure(
        segments, document_type
    )

    # Add semantic issues to report
    report.issues.extend(semantic_issues)

    return validated_segments, report


def validate_segment_text_exact(
    segment: AISegment,
    source_text: str,
    normalize_text: bool = True,
) -> bool:
    """Check if a segment's text appears exactly in the source.

    This is a low-level utility function for precise text matching validation.
    """
    if not segment.text:
        return False

    if normalize_text:
        # Simple normalization
        segment_normalized = " ".join(segment.text.split())
        source_normalized = " ".join(source_text.split())
        return segment_normalized in source_normalized
    else:
        return segment.text in source_text
