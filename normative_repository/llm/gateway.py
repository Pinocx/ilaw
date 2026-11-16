"""Capability-aware LLM routing, retries, structured validation, and usage capture."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .models import (
    CapabilityNotSupportedError,
    CircuitOpenError,
    GenerationRequest,
    GenerationResponse,
    LLMCapability,
    LLMError,
    LLMMessage,
    ModelConfig,
    StructuredGenerationResponse,
    StreamEvent,
    TokenUsage,
    UsageRecord,
)
from .protocol import LLMProvider


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
UsageCallback = Callable[[UsageRecord], None]


class LLMGateway:
    """Route requests by task to capability-checked provider adapters."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        *,
        default_provider: str,
        task_routes: dict[str, str] | None = None,
        fallback_routes: dict[str, str] | None = None,
        usage_callback: UsageCallback | None = None,
        usage_guard: Callable[[GenerationRequest], None] | None = None,
        usage_release: Callable[[GenerationRequest], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if default_provider not in providers:
            raise ValueError("default_provider must reference a registered provider.")
        if any(provider_name not in providers for provider_name in (task_routes or {}).values()):
            raise ValueError("Every task route must reference a registered provider.")
        if any(source not in providers or target not in providers for source, target in (fallback_routes or {}).items()):
            raise ValueError("Every fallback route must reference registered providers.")
        for source in (fallback_routes or {}):
            visited: set[str] = set()
            current = source
            while current in (fallback_routes or {}):
                if current in visited:
                    raise ValueError("Fallback routes cannot contain cycles.")
                visited.add(current)
                current = fallback_routes[current]
        self._providers = dict(providers)
        self._default_provider = default_provider
        self._task_routes = dict(task_routes or {})
        self._fallback_routes = dict(fallback_routes or {})
        self._usage_callback = usage_callback
        self._usage_guard = usage_guard
        self._usage_release = usage_release
        self._clock = clock
        self._sleeper = sleeper
        self._failures: dict[str, int] = {name: 0 for name in providers}
        self._circuit_open_until: dict[str, float] = {}

    def _provider_name(self, task: str | None) -> str:
        return self._task_routes.get(task or "", self._default_provider)

    def _provider(self, task: str | None) -> tuple[str, LLMProvider, ModelConfig]:
        name = self._provider_name(task)
        provider = self._providers[name]
        config = provider.config
        open_until = self._circuit_open_until.get(name, 0.0)
        if open_until > self._clock():
            raise CircuitOpenError(name)
        if open_until:
            self._circuit_open_until.pop(name, None)
            self._failures[name] = 0
        return name, provider, config

    def _provider_chain(self, task: str | None) -> list[str]:
        name = self._provider_name(task)
        chain = [name]
        while name in self._fallback_routes:
            name = self._fallback_routes[name]
            chain.append(name)
        return chain

    @staticmethod
    def _require_capability(config: ModelConfig, capability: LLMCapability) -> None:
        if capability not in config.capabilities:
            raise CapabilityNotSupportedError(capability, config.provider)

    def _record_usage(self, request: GenerationRequest, response: GenerationResponse) -> None:
        if self._usage_callback is None:
            return
        usage = response.usage
        record = UsageRecord(
            request_id=response.request_id,
            organization_id=request.organization_id,
            user_id=request.user_id,
            provider=response.provider,
            model=response.model,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=None,
            duration_ms=response.duration_ms,
        )
        try:
            self._usage_callback(record)
        except Exception:
            return

    def generate(
        self,
        request: GenerationRequest,
        *,
        task: str | None = None,
    ) -> GenerationResponse:
        """Generate text or return proposed tool calls without executing them."""
        chain = self._provider_chain(task)
        required = [LLMCapability.TEXT_GENERATION]
        if request.response_schema is not None:
            required.append(LLMCapability.STRUCTURED_OUTPUT)
        if request.tools:
            required.append(LLMCapability.TOOL_CALLS)
        capable_names = [
            name for name in chain
            if all(capability in self._providers[name].config.capabilities for capability in required)
        ]
        if not capable_names:
            config = self._providers[chain[0]].config
            for capability in required:
                self._require_capability(config, capability)
        if self._usage_guard:
            self._usage_guard(request)
        last_error: LLMError | None = None
        for name in chain:
            provider = self._providers[name]
            config = provider.config
            open_until = self._circuit_open_until.get(name, 0.0)
            if open_until > self._clock():
                last_error = CircuitOpenError(name)
                continue
            if open_until:
                self._circuit_open_until.pop(name, None)
                self._failures[name] = 0
            if name not in capable_names:
                continue
            attempts = 0
            provider_error: LLMError | None = None
            while True:
                try:
                    response = provider.generate(request)
                    break
                except LLMError as exc:
                    attempts += 1
                    if exc.retryable and attempts <= config.max_retries:
                        self._sleeper(min(0.25 * (2 ** (attempts - 1)), 2.0))
                        continue
                    self._register_failure(name, config)
                    provider_error = exc
                    break
                except Exception as exc:
                    self._register_failure(name, config)
                    provider_error = LLMError(
                        "The configured provider adapter failed unexpectedly.",
                        code="provider_adapter_error",
                    )
                    break
            if provider_error is not None:
                last_error = provider_error
                continue
            self._failures[name] = 0
            self._circuit_open_until.pop(name, None)
            if response.tool_calls and LLMCapability.TOOL_CALLS not in config.capabilities:
                last_error = CapabilityNotSupportedError(LLMCapability.TOOL_CALLS, config.provider)
                continue
            self._record_usage(request, response)
            self._release_usage(request)
            return response
        self._release_usage(request)
        if last_error:
            raise last_error
        raise LLMError("No configured provider could fulfill the request.", code="provider_unavailable")

    async def generate_async(
        self,
        request: GenerationRequest,
        *,
        task: str | None = None,
    ) -> GenerationResponse:
        """Run the synchronous adapter without blocking the event loop."""
        return await asyncio.to_thread(self.generate, request, task=task)

    def generate_structured(
        self,
        request: GenerationRequest,
        response_model: type[ResponseModel],
        *,
        task: str | None = None,
    ) -> StructuredGenerationResponse:
        """Generate and validate JSON, retrying malformed output a bounded number of times."""
        name = next((
            candidate for candidate in self._provider_chain(task)
            if {LLMCapability.TEXT_GENERATION, LLMCapability.STRUCTURED_OUTPUT}
            <= self._providers[candidate].config.capabilities
        ), None)
        if name is None:
            config = self._providers[self._provider_name(task)].config
            self._require_capability(config, LLMCapability.TEXT_GENERATION)
            self._require_capability(config, LLMCapability.STRUCTURED_OUTPUT)
        config = self._providers[name].config
        current_request = request.model_copy(update={
            "response_schema": response_model.model_json_schema(),
        })

        for validation_attempt in range(1, config.structured_output_retries + 2):
            response = self.generate(current_request, task=task)
            try:
                value = response_model.model_validate_json(response.text)
                return StructuredGenerationResponse(
                    response=value,
                    generation=response,
                    validation_attempts=validation_attempt,
                )
            except (ValidationError, ValueError, json.JSONDecodeError):
                if validation_attempt > config.structured_output_retries:
                    raise LLMError(
                        "The provider output did not match the required response schema.",
                        code="structured_output_invalid",
                    ) from None
                current_request = current_request.model_copy(update={
                    "messages": [
                        *current_request.messages,
                        LLMMessage(
                            role="system",
                            content=(
                                "Your previous response failed JSON schema validation. "
                                "Return only one valid JSON object matching the required schema."
                            ),
                        ),
                    ],
                })
        raise LLMError("Structured output validation did not complete.", code="structured_output_invalid")

    def stream(
        self,
        request: GenerationRequest,
        *,
        task: str | None = None,
    ) -> Iterator[str]:
        """Stream text; retrying after streaming begins is intentionally disabled."""
        chain = self._provider_chain(task)
        selected = None
        for candidate in chain:
            candidate_config = self._providers[candidate].config
            if LLMCapability.STREAMING not in candidate_config.capabilities:
                continue
            if self._circuit_open_until.get(candidate, 0.0) > self._clock():
                continue
            selected = (candidate, self._providers[candidate], candidate_config)
            break
        if selected is None:
            self._require_capability(self._providers[chain[0]].config, LLMCapability.STREAMING)
            raise CircuitOpenError(chain[0])
        name, provider, config = selected
        if self._usage_guard:
            self._usage_guard(request)
        started = self._clock()
        usage = TokenUsage()
        try:
            for event in provider.stream(request):
                if isinstance(event, StreamEvent):
                    if event.usage is not None:
                        usage = event.usage
                    if event.text_delta:
                        yield event.text_delta
                else:
                    yield event
            self._failures[name] = 0
            self._circuit_open_until.pop(name, None)
            self._record_usage(request, GenerationResponse(
                request_id=request.request_id,
                provider=config.provider,
                model=config.model,
                text="",
                usage=usage,
                duration_ms=max(0, round((self._clock() - started) * 1000)),
                finish_reason="stream_complete",
            ))
        except LLMError:
            self._register_failure(name, config)
            raise
        except Exception as exc:
            self._register_failure(name, config)
            raise LLMError("The provider stream failed unexpectedly.", code="provider_stream_error") from exc
        finally:
            self._release_usage(request)

    def _register_failure(self, name: str, config: ModelConfig) -> None:
        self._failures[name] = self._failures.get(name, 0) + 1
        if self._failures[name] >= config.circuit_failure_threshold:
            self._circuit_open_until[name] = self._clock() + config.circuit_reset_seconds

    def _release_usage(self, request: GenerationRequest) -> None:
        if self._usage_release:
            try:
                self._usage_release(request)
            except Exception:
                return
