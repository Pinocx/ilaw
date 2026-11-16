"""Contract tests for the provider-neutral LLM gateway."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from pydantic import BaseModel

from normative_repository.llm import (
    GenerationRequest,
    GenerationResponse,
    LLMCapability,
    LLMGateway,
    LLMMessage,
    StreamEvent,
    ModelConfig,
    OpenAICompatibleProvider,
    OllamaNativeProvider,
    AnthropicProvider,
    GeminiProvider,
    TokenUsage,
    ToolCall,
    build_gateway_from_environment,
)
from normative_repository.llm.models import CapabilityNotSupportedError, LLMError


def make_config(*capabilities: LLMCapability, **overrides: object) -> ModelConfig:
    values = {
        "provider": "mock",
        "model": "mock-model",
        "endpoint": "http://localhost:11434/v1",
        "capabilities": frozenset(capabilities),
        "max_retries": 0,
        "circuit_failure_threshold": 2,
        "circuit_reset_seconds": 10,
    }
    values.update(overrides)
    return ModelConfig(**values)


class FakeProvider:
    def __init__(self, config: ModelConfig, responses: list[object]) -> None:
        self.config = config
        self.responses = responses
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def stream(self, request: GenerationRequest):
        yield "hello"


def response(text: str = "ok", **kwargs: object) -> GenerationResponse:
    return GenerationResponse(
        request_id="request-1", provider="mock", model="mock-model", text=text,
        usage=TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        duration_ms=1, **kwargs,
    )


class Answer(BaseModel):
    answer: str


class LLMGatewayTests(unittest.TestCase):
    @staticmethod
    def stream_text(events):
        return "".join(event.text_delta if isinstance(event, StreamEvent) else event for event in events)
    def test_records_usage_without_prompt_or_cost_guess(self) -> None:
        records = []
        config = make_config(LLMCapability.TEXT_GENERATION)
        provider = FakeProvider(config, [response()])
        gateway = LLMGateway({"mock": provider}, default_provider="mock", usage_callback=records.append)
        result = gateway.generate(GenerationRequest(messages=[LLMMessage(role="user", content="question")]))
        self.assertEqual(result.text, "ok")
        self.assertEqual(records[0].total_tokens, 5)
        self.assertIsNone(records[0].estimated_cost)
        self.assertFalse(hasattr(records[0], "prompt"))

    def test_rejects_unsupported_capability_before_provider_call(self) -> None:
        provider = FakeProvider(make_config(LLMCapability.TEXT_GENERATION), [response()])
        gateway = LLMGateway({"mock": provider}, default_provider="mock")
        request = GenerationRequest(messages=[LLMMessage(role="user", content="x")], tools=[])
        with self.assertRaises(CapabilityNotSupportedError):
            gateway.generate_structured(request, Answer)
        self.assertEqual(provider.calls, 0)

    def test_retries_retryable_errors(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, max_retries=1)
        provider = FakeProvider(config, [LLMError("transient", retryable=True), response()])
        delays = []
        gateway = LLMGateway({"mock": provider}, default_provider="mock", sleeper=delays.append)
        result = gateway.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.text, "ok")
        self.assertEqual(provider.calls, 2)
        self.assertEqual(delays, [0.25])

    def test_streaming_records_provider_token_usage(self) -> None:
        class StreamingProvider(FakeProvider):
            def stream(self, request):
                yield StreamEvent(text_delta="hello")
                yield StreamEvent(usage=TokenUsage(input_tokens=6, output_tokens=2, total_tokens=8))

        records = []
        config = make_config(LLMCapability.STREAMING, LLMCapability.TEXT_GENERATION)
        provider = StreamingProvider(config, [])
        gateway = LLMGateway({"mock": provider}, default_provider="mock", usage_callback=records.append)
        result = "".join(gateway.stream(GenerationRequest(messages=[LLMMessage(role="user", content="x")])) )
        self.assertEqual(result, "hello")
        self.assertEqual(records[0].total_tokens, 8)

    def test_circuit_breaker_opens_after_threshold(self) -> None:
        now = [0.0]
        config = make_config(LLMCapability.TEXT_GENERATION, circuit_failure_threshold=1)
        provider = FakeProvider(config, [LLMError("down")])
        gateway = LLMGateway({"mock": provider}, default_provider="mock", clock=lambda: now[0])
        request = GenerationRequest(messages=[LLMMessage(role="user", content="x")])
        with self.assertRaises(LLMError):
            gateway.generate(request)
        with self.assertRaises(LLMError) as caught:
            gateway.generate(request)
        self.assertEqual(caught.exception.code, "circuit_open")
        now[0] = 11
        provider.responses.append(response())
        self.assertEqual(gateway.generate(request).text, "ok")

    def test_structured_output_validates_and_retries_once(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, LLMCapability.STRUCTURED_OUTPUT)
        provider = FakeProvider(config, [response("bad"), response('{"answer":"ok"}')])
        gateway = LLMGateway({"mock": provider}, default_provider="mock")
        result = gateway.generate_structured(
            GenerationRequest(messages=[LLMMessage(role="user", content="x")]), Answer,
        )
        self.assertEqual(result.response.answer, "ok")
        self.assertEqual(result.validation_attempts, 2)
        self.assertEqual(provider.calls, 2)

    def test_tool_calls_are_returned_as_proposals(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, LLMCapability.TOOL_CALLS)
        call = ToolCall(call_id="c1", name="lookup", arguments={"q": "x"})
        provider = FakeProvider(config, [response(tool_calls=[call])])
        gateway = LLMGateway({"mock": provider}, default_provider="mock")
        request = GenerationRequest(messages=[LLMMessage(role="user", content="x")], tools=[])
        self.assertEqual(gateway.generate(request).tool_calls[0].name, "lookup")

    def test_factory_empty_environment_does_not_read_process_environment(self) -> None:
        with patch.dict(os.environ, {"LLM_ENDPOINT": "http://secret-host/v1", "LLM_PRIMARY_MODEL": "secret"}):
            with self.assertRaises(ValueError):
                build_gateway_from_environment({})

    def test_factory_does_not_store_api_key_value_and_defaults_to_text(self) -> None:
        gateway = build_gateway_from_environment({
            "LLM_ENDPOINT": "http://localhost:11434/v1", "LLM_PRIMARY_MODEL": "local",
            "LLM_API_KEY_ENV": "MODEL_KEY", "MODEL_KEY": "never-store-this",
        })
        config = gateway._providers["primary"].config
        self.assertEqual(config.api_key_env, "MODEL_KEY")
        self.assertNotIn("never-store-this", repr(config))
        self.assertEqual(config.capabilities, {LLMCapability.TEXT_GENERATION})

    def test_openai_compatible_adapter_normalizes_usage_and_tool_calls(self) -> None:
        provider = OpenAICompatibleProvider(make_config(LLMCapability.TEXT_GENERATION))
        payload = {
            "id": "provider-1", "model": "mock-model", "choices": [{
                "finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                    "id": "call-1", "function": {"name": "lookup", "arguments": '{"q":"law"}'},
                }]},
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }
        with patch("normative_repository.llm.providers.openai_compatible.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            result = provider.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.usage.total_tokens, 9)
        self.assertEqual(result.tool_calls[0].arguments, {"q": "law"})
        self.assertEqual(result.provider_request_id, "provider-1")

    def test_ollama_native_adapter_normalizes_completion_and_usage(self) -> None:
        provider = OllamaNativeProvider(make_config(LLMCapability.TEXT_GENERATION, provider="ollama"))
        payload = {
            "model": "llama-local", "done": True, "prompt_eval_count": 6,
            "eval_count": 3, "message": {"content": "answer", "tool_calls": [{
                "function": {"name": "lookup", "arguments": {"q": "law"}},
            }]},
        }
        with patch("normative_repository.llm.providers.ollama_native.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            result = provider.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.usage.total_tokens, 9)
        self.assertEqual(result.tool_calls[0].call_id, "ollama-0")
        self.assertEqual(result.model, "llama-local")

    def test_factory_selects_native_ollama_adapter(self) -> None:
        gateway = build_gateway_from_environment({
            "LLM_ENDPOINT": "http://localhost:11434", "LLM_PRIMARY_MODEL": "local",
            "LLM_ADAPTER": "ollama_native",
        })
        self.assertIsInstance(gateway._providers["primary"], OllamaNativeProvider)

    def test_factory_loads_explicit_fallback_routes(self) -> None:
        gateway = build_gateway_from_environment({
            "LLM_ENDPOINT": "http://localhost:11434/v1",
            "LLM_PRIMARY_MODEL": "primary",
            "LLM_SECONDARY_MODEL": "secondary",
            "LLM_FALLBACK_ROUTES": '{"primary":"secondary"}',
        })
        self.assertEqual(gateway._fallback_routes, {"primary": "secondary"})

    def test_factory_selects_anthropic_and_gemini_adapters(self) -> None:
        for adapter, adapter_type, provider in (
            ("anthropic", AnthropicProvider, "anthropic"),
            ("gemini", GeminiProvider, "gemini"),
        ):
            gateway = build_gateway_from_environment({
                "LLM_ENDPOINT": "https://api.example.test", "LLM_PRIMARY_MODEL": "model",
                "LLM_ADAPTER": adapter, "LLM_PROVIDER": provider, "LLM_API_KEY_ENV": "PROVIDER_KEY",
            })
            self.assertIsInstance(gateway._providers["primary"], adapter_type)

    def test_factory_routes_tasks_to_secondary_model(self) -> None:
        gateway = build_gateway_from_environment({
            "LLM_ENDPOINT": "http://localhost:11434/v1",
            "LLM_PRIMARY_MODEL": "fast-model",
            "LLM_SECONDARY_MODEL": "reasoning-model",
            "LLM_TASK_ROUTES": '{"legal_analysis":"secondary"}',
        })
        self.assertEqual(gateway._provider_name("legal_analysis"), "secondary")
        self.assertEqual(gateway._provider_name("summarization"), "primary")
        self.assertEqual(gateway._providers["secondary"].config.model, "reasoning-model")
        self.assertEqual(gateway._providers["secondary"].config.provider, "ollama")

    def test_gateway_uses_configured_provider_fallback_after_failure(self) -> None:
        primary_config = make_config(LLMCapability.TEXT_GENERATION, provider="primary-provider", max_retries=0)
        secondary_config = make_config(LLMCapability.TEXT_GENERATION, provider="secondary-provider", max_retries=0)
        primary = FakeProvider(primary_config, [LLMError("primary unavailable")])

        class SuccessProvider(FakeProvider):
            def generate(self, request):
                self.calls += 1
                return GenerationResponse(
                    request_id=request.request_id, provider=self.config.provider,
                    model=self.config.model, text="fallback", duration_ms=1,
                )

        secondary = SuccessProvider(secondary_config, [])
        gateway = LLMGateway(
            {"primary": primary, "secondary": secondary},
            default_provider="primary", fallback_routes={"primary": "secondary"},
        )
        result = gateway.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.text, "fallback")
        self.assertEqual(result.provider, "secondary-provider")

    def test_gateway_rejects_cyclic_fallback_routes(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION)
        providers = {"a": FakeProvider(config, []), "b": FakeProvider(config, [])}
        with self.assertRaises(ValueError):
            LLMGateway(providers, default_provider="a", fallback_routes={"a": "b", "b": "a"})

    def test_anthropic_adapter_normalizes_text_usage_and_tool_call(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, LLMCapability.TOOL_CALLS, provider="anthropic", api_key_env="ANTHROPIC_TEST_KEY")
        provider = AnthropicProvider(config)
        raw = {"id": "msg-1", "model": "claude-test", "stop_reason": "tool_use", "content": [
            {"type": "text", "text": "Found it."},
            {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"query": "law"}},
        ], "usage": {"input_tokens": 8, "output_tokens": 4}}
        with patch.dict(os.environ, {"ANTHROPIC_TEST_KEY": "runtime-secret"}), patch(
            "normative_repository.llm.providers.anthropic.urlopen"
        ) as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(raw).encode()
            result = provider.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.text, "Found it.")
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(result.tool_calls[0].arguments, {"query": "law"})

    def test_gemini_adapter_normalizes_text_usage_and_tool_call(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, LLMCapability.TOOL_CALLS, provider="gemini", api_key_env="GEMINI_TEST_KEY")
        provider = GeminiProvider(config)
        raw = {"responseId": "response-1", "candidates": [{
            "finishReason": "STOP", "content": {"parts": [
                {"text": "Found it."}, {"functionCall": {"name": "lookup", "args": {"query": "law"}}},
            ]},
        }], "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12}}
        with patch.dict(os.environ, {"GEMINI_TEST_KEY": "runtime-secret"}), patch(
            "normative_repository.llm.providers.gemini.urlopen"
        ) as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(raw).encode()
            result = provider.generate(GenerationRequest(messages=[LLMMessage(role="user", content="x")]))
        self.assertEqual(result.text, "Found it.")
        self.assertEqual(result.usage.total_tokens, 12)
        self.assertEqual(result.tool_calls[0].arguments, {"query": "law"})

    def test_all_adapters_stream_text_deltas(self) -> None:
        request = GenerationRequest(messages=[LLMMessage(role="user", content="x")])
        openai = OpenAICompatibleProvider(make_config(LLMCapability.STREAMING))
        ollama = OllamaNativeProvider(make_config(LLMCapability.STREAMING))
        anthropic = AnthropicProvider(make_config(LLMCapability.STREAMING, provider="anthropic", api_key_env="ANTHROPIC_STREAM_KEY"))
        gemini = GeminiProvider(make_config(LLMCapability.STREAMING, provider="gemini", api_key_env="GEMINI_STREAM_KEY"))
        with patch("normative_repository.llm.providers.openai_compatible.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.__iter__.return_value = iter([
                b'data: {"choices":[{"delta":{"content":"hello"}}]}\n', b"data: [DONE]\n",
            ])
            self.assertEqual(self.stream_text(openai.stream(request)), "hello")
        with patch("normative_repository.llm.providers.ollama_native.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.__iter__.return_value = iter([
                b'{"message":{"content":"hello"},"done":false}\n', b'{"message":{"content":" world"},"done":true}\n',
            ])
            self.assertEqual(self.stream_text(ollama.stream(request)), "hello world")
        with patch.dict(os.environ, {"ANTHROPIC_STREAM_KEY": "runtime-secret"}), patch(
            "normative_repository.llm.providers.anthropic.urlopen"
        ) as urlopen:
            urlopen.return_value.__iter__.return_value = iter([
                b'data: {"type":"content_block_delta","delta":{"text":"hello"}}\n',
            ])
            self.assertEqual(self.stream_text(anthropic.stream(request)), "hello")
        with patch.dict(os.environ, {"GEMINI_STREAM_KEY": "runtime-secret"}), patch(
            "normative_repository.llm.providers.gemini.urlopen"
        ) as urlopen:
            urlopen.return_value.__iter__.return_value = iter([
                b'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}\n',
            ])
            self.assertEqual(self.stream_text(gemini.stream(request)), "hello")

    def test_api_key_is_resolved_only_at_request_time(self) -> None:
        config = make_config(LLMCapability.TEXT_GENERATION, api_key_env="TEST_LLM_SECRET")
        provider = OpenAICompatibleProvider(config)
        with patch.dict(os.environ, {"TEST_LLM_SECRET": "runtime-secret"}):
            self.assertEqual(provider._headers()["Authorization"], "Bearer runtime-secret")
        self.assertNotIn("runtime-secret", repr(config))

    def test_injected_credential_resolver_receives_organization_identity(self) -> None:
        seen = []

        class TenantResolver:
            def resolve(self, *, organization_id, provider, credential_ref):
                seen.append((organization_id, provider, credential_ref))
                return {"org-a": "tenant-key-a"}.get(organization_id)

        provider = OpenAICompatibleProvider(
            make_config(LLMCapability.TEXT_GENERATION), credential_resolver=TenantResolver(),
        )
        request = GenerationRequest(
            messages=[LLMMessage(role="user", content="x")], organization_id="org-a",
        )
        self.assertEqual(provider._headers(request)["Authorization"], "Bearer tenant-key-a")
        self.assertEqual(seen, [("org-a", "mock", None)])
        with self.assertRaises(LLMError):
            provider._headers(request.model_copy(update={"organization_id": "org-b"}))


if __name__ == "__main__":
    unittest.main()
