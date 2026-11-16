"""Local token-usage ledger with optional per-organization daily limits."""

from __future__ import annotations

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Mapping

from .models import GenerationRequest, LLMError, UsageRecord


class SqliteUsageLedger:
    """Persist usage metadata without storing prompts, responses, or credentials."""

    def __init__(
        self,
        database_path: Path,
        *,
        daily_token_limits: Mapping[str, int] | None = None,
        model_rates_per_million: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    ) -> None:
        self.database_path = database_path
        self.daily_token_limits = dict(daily_token_limits or {})
        self.model_rates = dict(model_rates_per_million or {})
        if any(value < 1 for value in self.daily_token_limits.values()):
            raise ValueError("Daily token limits must be positive integers.")
        if any(input_rate < 0 or output_rate < 0 for input_rate, output_rate in self.model_rates.values()):
            raise ValueError("Model token rates cannot be negative.")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS llm_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    organization_id TEXT,
                    user_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_id TEXT,
                    prompt_version TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost REAL,
                    duration_ms INTEGER NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS llm_reservations (
                    organization_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    reserved_tokens INTEGER NOT NULL,
                    usage_day TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (organization_id, request_id)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS llm_usage_org_time ON llm_usage(organization_id, timestamp)")

    @classmethod
    def from_environment(cls, database_path: Path, environment: Mapping[str, str] | None = None) -> "SqliteUsageLedger":
        """Load optional quota and per-model price policies from JSON environment values."""
        environment = os.environ if environment is None else environment
        try:
            limits = json.loads(environment.get("LLM_DAILY_TOKEN_LIMITS", "{}"))
            raw_rates = json.loads(environment.get("LLM_MODEL_RATES_USD_PER_MILLION", "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM quota and rate settings must be valid JSON.") from exc
        if not isinstance(limits, dict) or not isinstance(raw_rates, dict):
            raise ValueError("LLM quota and rate settings must be JSON objects.")
        rates: dict[tuple[str, str], tuple[float, float]] = {}
        for key, values in raw_rates.items():
            if not isinstance(key, str) or "/" not in key or not isinstance(values, list) or len(values) != 2:
                raise ValueError("Each model rate must use provider/model and contain input/output prices.")
            rates[tuple(key.split("/", 1))] = (float(values[0]), float(values[1]))
        return cls(database_path, daily_token_limits=limits, model_rates_per_million=rates)

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

    @staticmethod
    def _utc_day() -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def authorize(self, request: GenerationRequest) -> None:
        """Reserve bounded output tokens before dispatch when an org quota exists."""
        organization_id = request.organization_id
        if not organization_id:
            return
        limit = self.daily_token_limits.get(organization_id)
        if limit is None:
            return
        day = self._utc_day()
        reserved_tokens = request.max_output_tokens + sum(
            len(message.content.encode("utf-8")) + 4 for message in request.messages
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            expiry = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
            connection.execute("DELETE FROM llm_reservations WHERE created_at < ?", (expiry,))
            used = connection.execute(
                "SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage WHERE organization_id = ? AND substr(timestamp, 1, 10) = ?",
                (organization_id, day),
            ).fetchone()[0]
            reserved = connection.execute(
                "SELECT COALESCE(SUM(reserved_tokens), 0) FROM llm_reservations WHERE organization_id = ? AND usage_day = ?",
                (organization_id, day),
            ).fetchone()[0]
            if int(used) + int(reserved) + reserved_tokens > limit:
                raise LLMError("Organization token quota would be exceeded.", code="quota_exceeded")
            connection.execute(
                "INSERT OR REPLACE INTO llm_reservations(organization_id, request_id, reserved_tokens, usage_day, created_at) VALUES (?, ?, ?, ?, ?)",
                (organization_id, request.request_id, reserved_tokens, day, datetime.now(timezone.utc).isoformat()),
            )

    def __call__(self, record: UsageRecord) -> None:
        """Record a normalized usage event and release any matching reservation."""
        estimated_cost = record.estimated_cost
        rates = self.model_rates.get((record.provider, record.model))
        if estimated_cost is None and rates and record.input_tokens is not None and record.output_tokens is not None:
            estimated_cost = (record.input_tokens * rates[0] + record.output_tokens * rates[1]) / 1_000_000
        saved = record.model_copy(update={"estimated_cost": estimated_cost})
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("""
                INSERT INTO llm_usage(
                    request_id, organization_id, user_id, provider, model,
                    prompt_id, prompt_version, input_tokens, output_tokens,
                    total_tokens, estimated_cost, duration_ms, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                saved.request_id, saved.organization_id, saved.user_id, saved.provider, saved.model,
                saved.prompt_id, saved.prompt_version, saved.input_tokens, saved.output_tokens,
                saved.total_tokens, saved.estimated_cost, saved.duration_ms, saved.timestamp.isoformat(),
            ))
            if saved.organization_id:
                connection.execute(
                    "DELETE FROM llm_reservations WHERE organization_id = ? AND request_id = ?",
                    (saved.organization_id, saved.request_id),
                )

    def release(self, request: GenerationRequest) -> None:
        """Release a failed request's reserved quota."""
        if not request.organization_id:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM llm_reservations WHERE organization_id = ? AND request_id = ?",
                (request.organization_id, request.request_id),
            )

    def report(self, organization_id: str, *, limit: int = 1000) -> dict[str, object]:
        """Return usage metadata for one organization, without prompts or outputs."""
        if not organization_id or not 1 <= limit <= 10000:
            raise ValueError("organization_id is required and limit must be between 1 and 10000.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_usage WHERE organization_id = ? ORDER BY usage_id DESC LIMIT ?",
                (organization_id, limit),
            ).fetchall()
        return {
            "organization_id": organization_id,
            "records": [dict(row) for row in rows],
            "record_count": len(rows),
        }
