"""Preliminary document classification."""

from __future__ import annotations

import json
import re
import unicodedata

from pathlib import Path

from .config import REPORTS_DIR
from .registry import load_registry, save_registry


RULES = {
    "jurisprudence": [
        r"jurisprudencia",
        r"sentencia",
        r"casacion",
        r"expediente",
        r"tribunal constitucional",
    ],

    "administrative_regulation": [
        r"reglamento",
        r"directiva",
        r"resolucion ministerial",
        r"resolucion administrativa",
    ],

    "legal_manual": [
        r"manual",
        r"guia practica",
        r"guia de",
    ],

    "legal_doctrine": [
        r"comentarios? al",
        r"tratado",
        r"doctrina",
        r"ensayos?",
        r"estudio[s]? sobre",
    ],

    "legislation": [
        r"\bley\b",
        r"decreto legislativo",
        r"decreto supremo",
        r"codigo",
        r"constitucion",
        r"ley organica",
    ],

    "complementary_material": [
        r"formato",
        r"plantilla",
        r"formulario",
        r"compendio",
    ],
}


def normalize(value: str) -> str:
    """Normalize accents and whitespace."""

    value = value.lower()

    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", value)


def classify_name(
    filename: str,
) -> tuple[str, float, list[str]]:
    """Suggest a category based on filename."""

    name = normalize(Path(filename).stem)

    scores = {
        category: sum(
            bool(re.search(pattern, name))
            for pattern in patterns
        )
        for category, patterns in RULES.items()
    }

    best_score = max(scores.values(), default=0)

    matches = [
        category
        for category, score in scores.items()
        if score == best_score and score > 0
    ]

    if not matches:
        return "unknown", 0.0, []

    priority = [
        "jurisprudence",
        "legal_manual",
        "legal_doctrine",
        "administrative_regulation",
        "legislation",
        "complementary_material",
    ]

    chosen = next(
        category
        for category in priority
        if category in matches
    )

    confidence = (
        0.65
        if len(matches) == 1
        else 0.40
    )

    return chosen, confidence, matches


def classify_all() -> dict:
    """Classify all documents in the registry."""

    registry = load_registry()

    report = []

    for doc_id, meta in registry["records"].items():

        category, confidence, matches = classify_name(
            meta.get("filename", "")
        )

        meta["suggested_classification"] = category
        meta["classification_confidence"] = confidence
        meta["classification_matches"] = matches

        if (
            not meta.get("reviewed_classification")
            and (category == "unknown" or confidence < 0.5)
        ):

            meta.setdefault(
                "review",
                {},
            )["classification"] = "required"

        report.append({
            "doc_id": doc_id,
            "filename": meta.get("filename"),
            "suggested_classification": category,
            "reviewed_classification": meta.get("reviewed_classification"),
            "classification_review_status": (
                "reviewed" if meta.get("reviewed_classification") else "pending_review"
            ),
            "confidence": confidence,
            "matches": matches,
        })

    save_registry(registry)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        REPORTS_DIR / "classification-report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "classified": len(report),
        "report": str(report_path),
    }
