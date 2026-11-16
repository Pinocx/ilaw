"""Tests for organization usage accounting and token quotas."""

from __future__ import annotations

import unittest
from pathlib import Path

from normative_repository.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMCapability,
    LLMGateway,
    LLMMessage,
    ModelConfig,
    SqliteUsageLedger,
    TokenUsage,
)
from normative_repository.llm.models import LLMError


class SuccessfulProvider:
    def __init__(self, config):
        self.config = config

    def generate(self, request):
        return GenerationResponse(
            request_id=request.request_id, provider=self.config.provider, model=self.config.model,
            text="done", usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5), duration_ms=1,
        )

    def stream(self, request):
        yield "done"


class UsageLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_path = Path(__file__).parent / "fixtures" / "llm-usage-test.sqlite3"
        self.database_path.unlink(missing_ok=True)
        self.addCleanup(self.database_path.unlink, missing_ok=True)

    def test_records_usage_cost_and_prompt_version_without_content(self) -> None:
        ledger = SqliteUsageLedger(
            self.database_path,
            model_rates_per_million={("mock", "model"): (1.0, 2.0)},
        )
        config = ModelConfig(
            provider="mock", model="model", endpoint="http://localhost/v1",
            capabilities=frozenset({LLMCapability.TEXT_GENERATION}), max_retries=0,
        )
        gateway = LLMGateway(
            {"mock": SuccessfulProvider(config)}, default_provider="mock",
            usage_callback=ledger, usage_guard=ledger.authorize, usage_release=ledger.release,
        )
        gateway.generate(GenerationRequest(
            messages=[LLMMessage(role="user", content="private prompt")],
            organization_id="org-1", user_id="user-1", prompt_id="research", prompt_version="v3",
        ))
        report = ledger.report("org-1")
        record = report["records"][0]
        self.assertEqual(record["total_tokens"], 5)
        self.assertAlmostEqual(record["estimated_cost"], 0.000007)
        self.assertEqual(record["prompt_version"], "v3")
        self.assertNotIn("private prompt", str(record))

    def test_quota_reserves_before_call_and_releases_on_failure(self) -> None:
        ledger = SqliteUsageLedger(self.database_path, daily_token_limits={"org": 20})
        request = GenerationRequest(
            messages=[LLMMessage(role="user", content="query")],
            max_output_tokens=10, organization_id="org",
        )
        ledger.authorize(request)
        with self.assertRaises(LLMError) as caught:
            ledger.authorize(request.model_copy(update={"request_id": "another"}))
        self.assertEqual(caught.exception.code, "quota_exceeded")
        ledger.release(request)
        ledger.authorize(request.model_copy(update={"request_id": "retry"}))


if __name__ == "__main__":
    unittest.main()
