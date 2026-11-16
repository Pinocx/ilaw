"""Native Ollama chat API adapter."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..models import GenerationRequest, GenerationResponse, LLMError, ModelConfig, StreamEvent, TokenUsage, ToolCall


class OllamaNativeProvider:
    """Provider adapter for Ollama's native ``/api/chat`` endpoint."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": stream,
            "options": {"temperature": request.temperature, "num_predict": request.max_output_tokens},
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        if request.tools:
            payload["tools"] = [
                {"type": "function", "function": tool.model_dump()}
                for tool in request.tools
            ]
            payload["tool_choice"] = request.tool_choice
        return payload

    def _request(self, payload: dict[str, Any], timeout: float) -> Any:
        request = Request(
            f"{self.config.endpoint}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMError(
                "The Ollama provider returned an HTTP error.",
                code="http_error", retryable=exc.code == 429 or 500 <= exc.code <= 599,
                status_code=exc.code,
            ) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError("The Ollama request timed out.", code="timeout", retryable=True) from exc
        except URLError as exc:
            raise LLMError("The Ollama provider could not be reached.", code="connection_error", retryable=True) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError("The Ollama provider returned invalid JSON.", code="invalid_provider_response") from exc

    @staticmethod
    def _tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = item.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise LLMError("Ollama returned invalid tool arguments.", code="invalid_tool_arguments") from exc
            if not isinstance(arguments, dict):
                raise LLMError("Tool arguments must be a JSON object.", code="invalid_tool_arguments")
            calls.append(ToolCall(
                call_id=str(item.get("id") or f"ollama-{index}"),
                name=str(function.get("name") or ""),
                arguments=arguments,
            ))
        return calls

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter()
        raw = self._request(self._payload(request, stream=False), request.timeout_seconds or self.config.timeout_seconds)
        message = raw.get("message") or {}
        usage = TokenUsage(
            input_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
            total_tokens=(raw.get("prompt_eval_count", 0) + raw.get("eval_count", 0))
            if raw.get("prompt_eval_count") is not None and raw.get("eval_count") is not None else None,
        )
        return GenerationResponse(
            request_id=request.request_id, provider=self.config.provider,
            model=raw.get("model") or self.config.model, text=message.get("content") or "",
            tool_calls=self._tool_calls(message), usage=usage,
            duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
            finish_reason="stop" if raw.get("done") else None,
        )

    def stream(self, request: GenerationRequest) -> Iterator[str | StreamEvent]:
        payload = self._payload(request, stream=True)
        http_request = Request(
            f"{self.config.endpoint}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        event = json.loads(raw_line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise LLMError("The Ollama stream contained invalid JSON.", code="invalid_provider_response") from exc
                    content = (event.get("message") or {}).get("content")
                    if isinstance(content, str) and content:
                        yield StreamEvent(text_delta=content)
                    if event.get("done"):
                        prompt_tokens = event.get("prompt_eval_count")
                        output_tokens = event.get("eval_count")
                        yield StreamEvent(usage=TokenUsage(
                            input_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            total_tokens=prompt_tokens + output_tokens
                            if prompt_tokens is not None and output_tokens is not None else None,
                        ))
                        return
        except HTTPError as exc:
            raise LLMError("The Ollama stream returned an HTTP error.", code="http_error", retryable=exc.code == 429 or 500 <= exc.code <= 599, status_code=exc.code) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError("The Ollama stream timed out.", code="timeout", retryable=True) from exc
        except URLError as exc:
            raise LLMError("The Ollama stream could not be reached.", code="connection_error", retryable=True) from exc
