"""Append-only, hash-chained local audit events for agent executions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SqliteAuditLedger:
    """Persist audit payloads with append-only triggers and tamper-evident chaining."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS agent_audit (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    organization_id TEXT,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS agent_audit_org_time ON agent_audit(organization_id, created_at)")
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS agent_audit_no_update
                BEFORE UPDATE ON agent_audit BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END
            """)
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS agent_audit_no_delete
                BEFORE DELETE ON agent_audit BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END
            """)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def append(self, *, event_type: str, run_id: str, organization_id: str | None, payload: dict[str, Any]) -> str:
        if not event_type or len(event_type) > 128 or not run_id or len(run_id) > 256:
            raise ValueError("Audit event type and run ID are required and bounded.")
        created_at = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT event_hash FROM agent_audit ORDER BY event_id DESC LIMIT 1").fetchone()
            previous_hash = row["event_hash"] if row else "0" * 64
            digest_input = "\n".join((previous_hash, created_at, organization_id or "", run_id, event_type, payload_json))
            event_hash = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
            connection.execute("""
                INSERT INTO agent_audit(
                    created_at, organization_id, run_id, event_type, payload_json, previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (created_at, organization_id, run_id, event_type, payload_json, previous_hash, event_hash))
        return event_hash

    def verify_integrity(self) -> dict[str, Any]:
        """Check event ordering, payload hashes, and chain linkage."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM agent_audit ORDER BY event_id").fetchall()
        previous_hash = "0" * 64
        for row in rows:
            expected_input = "\n".join((
                previous_hash, row["created_at"], row["organization_id"] or "",
                row["run_id"], row["event_type"], row["payload_json"],
            ))
            actual_hash = hashlib.sha256(expected_input.encode("utf-8")).hexdigest()
            if row["previous_hash"] != previous_hash or row["event_hash"] != actual_hash:
                return {"valid": False, "checked_events": rows.index(row), "event_id": row["event_id"]}
            previous_hash = actual_hash
        return {"valid": True, "checked_events": len(rows), "head_hash": previous_hash}

    def records(self, *, organization_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000.")
        with self._connect() as connection:
            if organization_id is None:
                rows = connection.execute("SELECT * FROM agent_audit ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = connection.execute("SELECT * FROM agent_audit WHERE organization_id = ? ORDER BY event_id DESC LIMIT ?", (organization_id, limit)).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload_json"])}
            for row in rows
        ]
