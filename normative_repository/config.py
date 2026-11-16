"""Central application configuration."""

from pathlib import Path
import os


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(
    os.getenv("DATA_DIR", PROJECT_ROOT / "data")
).resolve()

DEFAULT_SOURCE_DIR = Path(
    os.getenv("SOURCE_DIR", DATA_DIR / "incoming_originals")
).resolve()

REPORTS_DIR = DATA_DIR / "reports"
EXTRACTED_DIR = DATA_DIR / "extracted"
MARKDOWN_DIR = DATA_DIR / "markdown"
STRUCTURED_DIR = DATA_DIR / "structured"
INDEX_DIR = DATA_DIR / "index"
CANDIDATE_INDEX_DIR = DATA_DIR / "review" / "index-candidate"
RETROSPECTIVE_PILOT_DIR = DATA_DIR / "review" / "retrospective-pilot"
SEGMENT_EVALUATION_DIR = DATA_DIR / "review" / "segmentation-evaluation-pilot"

REGISTRY_FILE = DATA_DIR / "registry.json"


# Processing configuration
OCR_ENABLED = os.getenv(
    "OCR_ENABLED", "true"
).lower() in {"1", "true", "yes"}

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "spa")

PILOT_SIZE = int(os.getenv("PILOT_SIZE", "10"))

BATCH_SIZE_DEFAULT = 25


# Supported document formats
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".odt",
}


# Preliminary document categories
CATEGORIES = (
    "legislation",
    "jurisprudence",
    "administrative_regulation",
    "legal_doctrine",
    "legal_manual",
    "complementary_material",
    "unknown",
)


# Create required directories
for directory in (
    DATA_DIR,
    REPORTS_DIR,
    EXTRACTED_DIR,
    MARKDOWN_DIR,
    STRUCTURED_DIR,
    INDEX_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
