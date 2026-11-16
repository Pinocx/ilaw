"""Adapter for local or remote OpenAI-compatible chat-completions endpoints."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import (
    GenerationRequest,
    GenerationResponse,
    LLMError,
    ModelConfig,
    TokenUsage,
    StreamEvent,
    ToolCall,
)
from ..credentials import CredentialResolver, resolve_credential


class OpenAICompatibleProvider:
    """HTTP adapter for Ollama and other compatible chat-completions servers."""

    def __init__(self, config: ModelConfig, credential_resolver: CredentialResolver | None = None) -> None:
        self.config = config
        self.credential_resolver = credential_resolver

    def _headers(self, generation_request: GenerationRequest | None = None) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key_env or self.credential_resolver:
            api_key = resolve_credential(
                self.credential_resolver,
                organization_id=generation_request.organization_id if generation_request else None,
                provider=self.config.provider,
                credential_ref=self.config.api_key_env,
            )
            if not api_key:
                raise LLMError(
                    "Required provider credential is unavailable.",
                    code="credential_unavailable",
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": stream,
        }
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = request.tool_choice
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _post(self, payload: dict[str, Any], timeout: float, generation_request: GenerationRequest) -> Any:
        endpoint = f"{self.config.endpoint}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(endpoint, data=body, headers=self._headers(generation_request), method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code <= 599
            raise LLMError(
                "The language model provider returned an HTTP error.",
                code="rate_limited" if exc.code == 429 else "http_error",
                retryable=retryable,
                status_code=exc.code,
            ) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError(
                "The language model provider request timed out.",
                code="timeout",
                retryable=True,
            ) from exc
        except URLError as exc:
            reason = exc.reason
            timed_out = isinstance(reason, (TimeoutError, socket.timeout))
            raise LLMError(
                "The language model provider could not be reached.",
                code="timeout" if timed_out else "connection_error",
                retryable=True,
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(
                "The language model provider returned an invalid JSON response.",
                code="invalid_provider_response",
            ) from exc

    @staticmethod
    def _message_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        raise LLMError("The provider returned an unsupported message content format.", code="invalid_provider_response")

    @staticmethod
    def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError(
                    "The provider returned invalid JSON tool arguments.",
                    code="invalid_tool_arguments",
                ) from exc
            if not isinstance(arguments, dict):
                raise LLMError(
                    "Tool arguments must be a JSON object.",
                    code="invalid_tool_arguments",
                )
            calls.append(ToolCall(
                call_id=str(item.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=arguments,
            ))
        return calls

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text or proposed tool calls; never execute tools."""
        started = time.perf_counter()
        payload = self._payload(request, stream=False)
        response = self._post(payload, request.timeout_seconds or self.config.timeout_seconds, request)
        try:
            choice = response["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                "The provider response is missing a chat completion choice.",
                code="invalid_provider_response",
            ) from exc
        usage_payload = response.get("usage") or {}
        usage = TokenUsage(
            input_tokens=usage_payload.get("prompt_tokens"),
            output_tokens=usage_payload.get("completion_tokens"),
            total_tokens=usage_payload.get("total_tokens"),
        )
        return GenerationResponse(
            request_id=request.request_id,
            provider=self.config.provider,
            model=response.get("model") or self.config.model,
            text=self._message_text(message.get("content")),
            tool_calls=self._tool_calls(message),
            usage=usage,
            duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
            finish_reason=choice.get("finish_reason"),
            provider_request_id=response.get("id"),
        )

    def stream(self, request: GenerationRequest) -> Iterator[str | StreamEvent]:
        """Yield server-sent text deltas and authoritative usage when supplied."""
        payload = self._payload(request, stream=True)
        endpoint = f"{self.config.endpoint}/chat/completions"
        http_request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={**self._headers(request), "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    event = line[5:].strip()
                    if event == "[DONE]":
                        return
                    try:
                        payload = json.loads(event)
                    except json.JSONDecodeError as exc:
                        raise LLMError(
                            "The provider stream contained an invalid JSON event.",
                            code="invalid_provider_response",
                        ) from exc
                    choices = payload.get("choices") or []
                    usage_payload = payload.get("usage") or {}
                    if usage_payload:
                        yield StreamEvent(usage=TokenUsage(
                            input_tokens=usage_payload.get("prompt_tokens"),
                            output_tokens=usage_payload.get("completion_tokens"),
                            total_tokens=usage_payload.get("total_tokens"),
                        ))
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield StreamEvent(text_delta=content)
        except HTTPError as exc:
            raise LLMError(
                "The language model provider returned an HTTP error while streaming.",
                code="rate_limited" if exc.code == 429 else "http_error",
                retryable=exc.code == 429 or 500 <= exc.code <= 599,
                status_code=exc.code,
            ) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError("The language model provider stream timed out.", code="timeout", retryable=True) from exc
        except URLError as exc:
            raise LLMError("The language model provider stream could not be reached.", code="connection_error", retryable=True) from exc
