"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import GenerationRequest, GenerationResponse, LLMError, ModelConfig, StreamEvent, TokenUsage, ToolCall
from ..credentials import CredentialResolver, resolve_credential


class AnthropicProvider:
    """Provider adapter for the Anthropic Messages API."""

    def __init__(self, config: ModelConfig, credential_resolver: CredentialResolver | None = None) -> None:
        self.config = config
        self.credential_resolver = credential_resolver

    def _headers(self, request: GenerationRequest, *, streaming: bool = False) -> dict[str, str]:
        api_key = resolve_credential(
            self.credential_resolver,
            organization_id=request.organization_id,
            provider=self.config.provider,
            credential_ref=self.config.api_key_env,
        )
        if not api_key:
            raise LLMError("Anthropic API credential is unavailable.", code="credential_unavailable")
        return {
            "Content-Type": "application/json", "Accept": "text/event-stream" if streaming else "application/json",
            "x-api-key": api_key, "anthropic-version": "2023-06-01",
        }

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        system = "\n".join(message.content for message in request.messages if message.role == "system")
        messages = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": message.content,
                    }],
                })
            else:
                messages.append({"role": message.role, "content": message.content})
        payload: dict[str, Any] = {
            "model": self.config.model, "messages": messages,
            "max_tokens": request.max_output_tokens, "temperature": request.temperature, "stream": stream,
        }
        if system:
            payload["system"] = system
        if request.response_schema:
            payload["system"] = (payload.get("system", "") + "\nReturn only JSON conforming to this schema:\n" + json.dumps(request.response_schema))
        if request.tools:
            payload["tools"] = [{
                "name": tool.name, "description": tool.description, "input_schema": tool.parameters,
            } for tool in request.tools]
            choice = {"auto": "auto", "none": "none", "required": "any"}[request.tool_choice]
            payload["tool_choice"] = {"type": choice}
        return payload

    def _call(self, request: GenerationRequest, *, stream: bool = False) -> Any:
        http_request = Request(
            self.config.endpoint.rstrip("/") + "/v1/messages",
            data=json.dumps(self._payload(request, stream=stream), ensure_ascii=False).encode(),
            headers=self._headers(request, streaming=stream), method="POST",
        )
        try:
            if stream:
                return urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds)
            with urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMError("Anthropic returned an HTTP error.", code="rate_limited" if exc.code == 429 else "http_error", retryable=exc.code == 429 or exc.code >= 500, status_code=exc.code) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError("Anthropic request timed out.", code="timeout", retryable=True) from exc
        except URLError as exc:
            raise LLMError("Anthropic could not be reached.", code="connection_error", retryable=True) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError("Anthropic returned invalid JSON.", code="invalid_provider_response") from exc

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter()
        raw = self._call(request)
        blocks = raw.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        calls = [ToolCall(call_id=block.get("id", ""), name=block.get("name", ""), arguments=block.get("input") or {}) for block in blocks if block.get("type") == "tool_use"]
        usage = raw.get("usage") or {}
        return GenerationResponse(
            request_id=request.request_id, provider=self.config.provider, model=raw.get("model") or self.config.model,
            text=text, tool_calls=calls,
            usage=TokenUsage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), total_tokens=(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) if usage.get("input_tokens") is not None and usage.get("output_tokens") is not None else None),
            duration_ms=max(0, round((time.perf_counter() - started) * 1000)), finish_reason=raw.get("stop_reason"), provider_request_id=raw.get("id"),
        )

    def stream(self, request: GenerationRequest) -> Iterator[str | StreamEvent]:
        response = self._call(request, stream=True)
        input_tokens = None
        output_tokens = None
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError as exc:
                    raise LLMError("Anthropic stream contained invalid JSON.", code="invalid_provider_response") from exc
                if event.get("type") == "message_start":
                    input_tokens = ((event.get("message") or {}).get("usage") or {}).get("input_tokens")
                if event.get("type") == "message_delta":
                    output_tokens = (event.get("usage") or {}).get("output_tokens")
                if event.get("type") == "content_block_delta":
                    text = (event.get("delta") or {}).get("text")
                    if text:
                        yield StreamEvent(text_delta=text)
            yield StreamEvent(usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None else None,
            ))
        finally:
            response.close()
