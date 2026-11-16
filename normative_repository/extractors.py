
"""Local document text extraction."""

from __future__ import annotations

import shutil
import subprocess
import tempfile

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from pypdf import PdfReader
from docx import Document
from odf.opendocument import load as load_odt
from odf import text as odf_text
from odf import teletype

class OCRRequiredError(Exception):
    """Raised when a document requires OCR but OCR is being skipped."""


class OCRNotRequiredError(Exception):
    """Raised when a document does not require OCR but only OCR is requested."""


def extract_pdf_text(path: Path) -> list[str]:
    """Extract text page by page from PDF."""

    reader = PdfReader(str(path))

    return [
        (page.extract_text() or "").strip()
        for page in reader.pages
    ]

def assess_pdf_ocr(path: Path) -> dict[str, Any]:
    """Assess whether a PDF requires OCR without running OCR."""

    pages = extract_pdf_text(path)

    average_chars = (
        sum(len(page) for page in pages) / len(pages)
        if pages
        else 0
    )

    required = (
        not pages
        or average_chars < 40
    )

    return {
        "required": required,
        "status": "done",
        "method": "pypdf_text_density_v1",
        "page_count": len(pages),
        "average_chars_per_page": round(average_chars, 2),
        "assessed_at": datetime.now(timezone.utc).isoformat(),
    }

def run_ocr(
    source: Path,
    language: str = "spa",
) -> Path:
    """Run OCRmyPDF and return the generated PDF path."""

    temp_dir = Path(
        tempfile.mkdtemp(prefix="legalrepo-ocr-")
    )

    output = temp_dir / "ocr.pdf"

    command = [
        "ocrmypdf",
        "--skip-text",
        "--language",
        language,
        "--output-type",
        "pdf",
        str(source),
        str(output),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0 or not output.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

        raise RuntimeError(
            f"OCRmyPDF failed: {result.stderr[-1500:]}"
        )

    return output


def run_poppler_tesseract_ocr(
    source: Path,
    language: str = "spa",
) -> tuple[list[str], int]:
    """Fallback OCR using Poppler and Tesseract."""

    reader = PdfReader(str(source))
    page_count = len(reader.pages)
    extracted_pages: list[str] = []
    empty_pages: list[int] = []

    with tempfile.TemporaryDirectory(
        prefix="legalrepo-fallback-"
    ) as temp_dir:

        temp_path = Path(temp_dir)

        for page_number in range(1, page_count + 1):

            output_prefix = temp_path / f"page-{page_number}"
            image_path = output_prefix.with_suffix(".png")

            render_command = [
                "pdftoppm",
                "-f",
                str(page_number),
                "-l",
                str(page_number),
                "-r",
                "200",
                "-png",
                "-singlefile",
                str(source),
                str(output_prefix),
            ]

            render_result = subprocess.run(
                render_command,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if render_result.returncode != 0 or not image_path.exists():
                raise RuntimeError(
                    f"Poppler failed on page {page_number}: "
                    f"{render_result.stderr[-1000:]}"
                )

            tesseract_command = [
                "tesseract",
                str(image_path),
                "stdout",
                "-l",
                language,
                "--psm",
                "3",
            ]

            ocr_result = subprocess.run(
                tesseract_command,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if ocr_result.returncode != 0:
                raise RuntimeError(
                    f"Tesseract failed on page {page_number}: "
                    f"{ocr_result.stderr[-1000:]}"
                )

            page_text = ocr_result.stdout.strip()
            extracted_pages.append(page_text)

            if not page_text:
                empty_pages.append(page_number)

    if not any(page.strip() for page in extracted_pages):
        raise RuntimeError(
            "Fallback OCR completed but produced no readable text."
        )

    return extracted_pages, len(empty_pages)


# def extract_pdf(
#     path: Path,
#     ocr_enabled: bool = True,
#     language: str = "spa",
# ) -> tuple[list[str], dict[str, Any]]:

#     pages = extract_pdf_text(path)

#     average_chars = (
#         sum(len(page) for page in pages) / len(pages)
#         if pages
#         else 0
#     )

#     likely_scanned = (
#         not pages
#         or average_chars < 40
#     )

#     metadata = {
#         "method": "pypdf",
#         "ocr_used": False,
#         "page_count": len(pages),
#     }

#     if likely_scanned and ocr_enabled:

#         ocr_path: Path | None = None

#         try:
#             ocr_path = run_ocr(
#                 path,
#                 language,
#             )

#             pages = extract_pdf_text(ocr_path)

#             metadata.update({
#                 "method": "ocrmypdf+pypdf",
#                 "ocr_used": True,
#             })

#         except Exception as ocr_error:

#             # Remove temporary OCRmyPDF output if it exists.
#             if ocr_path is not None:
#                 shutil.rmtree(
#                     ocr_path.parent,
#                     ignore_errors=True,
#                 )

#             # Fall back to Poppler + Tesseract.
#             pages, empty_pages = run_poppler_tesseract_ocr(
#                 path,
#                 language,
#             )

#             metadata.update({
#                 "method": "poppler+tesseract",
#                 "ocr_used": True,
#                 "ocr_fallback": True,
#                 "ocrmypdf_error": str(ocr_error)[-1500:],
#                 "empty_ocr_pages": empty_pages,
#             })

#         finally:
#             if ocr_path is not None:
#                 shutil.rmtree(
#                     ocr_path.parent,
#                     ignore_errors=True,
#                 )

#     return pages, metadata
def extract_pdf(
    path: Path,
    ocr_enabled: bool = True,
    language: str = "spa",
    ocr_mode: str = "auto",
    ocr_required: bool | None = None,
) -> tuple[list[str], dict[str, Any]]:

    pages = extract_pdf_text(path)

    if ocr_required is None:
        average_chars = (
            sum(len(page) for page in pages) / len(pages)
            if pages
            else 0
        )

        likely_scanned = (
            not pages
            or average_chars < 40
        )
    else:
        likely_scanned = ocr_required

    # Decide whether this document belongs to the requested processing mode.
    if ocr_mode == "skip" and likely_scanned:
        raise OCRRequiredError(
            "Document requires OCR and was deferred."
        )

    if ocr_mode == "only" and not likely_scanned:
        raise OCRNotRequiredError(
            "Document does not require OCR and was deferred."
        )

    metadata = {
        "method": "pypdf",
        "ocr_used": False,
        "page_count": len(pages),
    }

    if likely_scanned and ocr_enabled:

        ocr_path: Path | None = None

        try:
            ocr_path = run_ocr(
                path,
                language,
            )

            pages = extract_pdf_text(ocr_path)

            metadata.update({
                "method": "ocrmypdf+pypdf",
                "ocr_used": True,
            })

        except Exception as ocr_error:

            if ocr_path is not None:
                shutil.rmtree(
                    ocr_path.parent,
                    ignore_errors=True,
                )

            pages, empty_pages = run_poppler_tesseract_ocr(
                path,
                language,
            )

            metadata.update({
                "method": "poppler+tesseract",
                "ocr_used": True,
                "ocr_fallback": True,
                "ocrmypdf_error": str(ocr_error)[-1500:],
                "empty_ocr_pages": empty_pages,
            })

        finally:
            if ocr_path is not None:
                shutil.rmtree(
                    ocr_path.parent,
                    ignore_errors=True,
                )

    return pages, metadata

def extract_docx(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:

    document = Document(str(path))

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [text], {
        "method": "python-docx",
        "ocr_used": False,
        "page_count": None,
    }


def extract_odt(
    path: Path,
) -> tuple[list[str], dict[str, Any]]:

    document = load_odt(str(path))

    paragraphs = document.getElementsByType(
        odf_text.P
    )

    text = "\n".join(
        teletype.extractText(paragraph)
        for paragraph in paragraphs
    ).strip()

    return [text], {
        "method": "odfpy",
        "ocr_used": False,
        "page_count": None,
    }

def extract_document(
    path: Path,
    ocr_enabled: bool = True,
    language: str = "spa",
    ocr_mode: str = "auto",
    ocr_required: bool | None = None,
) -> tuple[list[str], dict[str, Any]]:

    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(
            path,
            ocr_enabled,
            language,
            ocr_mode,
            ocr_required,
        )

    if ocr_mode == "only":
        raise OCRNotRequiredError(
            "This document format does not require OCR."
        )

    if extension == ".docx":
        return extract_docx(path)

    if extension == ".odt":
        return extract_odt(path)

    raise ValueError(
        f"Unsupported document format: {extension}"
    )

# def extract_document(
#     path: Path,
#     ocr_enabled: bool = True,
#     language: str = "spa",
# ) -> tuple[list[str], dict[str, Any]]:

#     extension = path.suffix.lower()

#     if extension == ".pdf":
#         return extract_pdf(
#             path,
#             ocr_enabled,
#             language,
#         )

#     if extension == ".docx":
#         return extract_docx(path)

#     if extension == ".odt":
#         return extract_odt(path)

#     raise ValueError(
#         f"Unsupported document format: {extension}"
#     )
