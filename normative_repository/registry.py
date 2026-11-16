"""Central document registry."""

from __future__ import annotations

import json
import os
import tempfile

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REGISTRY_FILE


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry(path: Path = REGISTRY_FILE) -> dict[str, Any]:
    """Load the document registry."""

    if not path.exists():
        return {
            "schema_version": 1,
            "updated_at": now_iso(),
            "records": {},
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read registry {path}: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError("Invalid registry structure.")

    if not isinstance(data.get("records"), dict):
        raise RuntimeError("Registry has no valid records dictionary.")

    return data


def save_registry(
    registry: dict[str, Any],
    path: Path = REGISTRY_FILE,
) -> None:
    """Save registry atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    registry["updated_at"] = now_iso()

    fd, temp_name = tempfile.mkstemp(
        prefix=path.name,
        suffix=".tmp",
        dir=path.parent,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                registry,
                stream,
                ensure_ascii=False,
                indent=2,
            )
            stream.write("\n")

        os.replace(temp_name, path)

    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def upsert_record(
    registry: dict[str, Any],
    record: dict[str, Any],
) -> None:
    """Insert or update a document without losing review history."""

    doc_id = record["doc_id"]

    current = registry.setdefault("records", {}).get(doc_id, {})

    merged = {
        **current,
        **record,
    }

    merged.setdefault(
        "review",
        {
            "classification": "pending",
            "legal_verification": "pending",
        },
    )

    merged.setdefault(
        "processing",
        {
            "status": "pending",
            "history": [],
        },
    )

    registry["records"][doc_id] = merged


def add_history(
    meta: dict[str, Any],
    stage: str,
    status: str,
    details: str = "",
) -> None:
    """Append a processing event."""

    processing = meta.setdefault(
        "processing",
        {
            "status": "pending",
            "history": [],
        },
    )

    processing.setdefault("history", []).append({
        "stage": stage,
        "status": status,
        "details": details,
        "at": now_iso(),
    })

    if status in {"done", "error", "pending"}:
        processing["status"] = status