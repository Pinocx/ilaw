"""Google Gemini generateContent API adapter."""

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


class GeminiProvider:
    """Provider adapter for Google's Gemini generateContent API."""

    def __init__(self, config: ModelConfig, credential_resolver: CredentialResolver | None = None) -> None:
        self.config = config
        self.credential_resolver = credential_resolver

    def _key(self, request: GenerationRequest) -> str:
        key = resolve_credential(
            self.credential_resolver,
            organization_id=request.organization_id,
            provider=self.config.provider,
            credential_ref=self.config.api_key_env,
        )
        if not key:
            raise LLMError("Gemini API credential is unavailable.", code="credential_unavailable")
        return key

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        system = [part for message in request.messages if message.role == "system" for part in [{"text": message.content}]]
        contents = []
        for message in request.messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                parts = [{"functionResponse": {
                    "name": message.name or message.tool_call_id or "tool",
                    "response": {"content": message.content},
                }}]
                role = "user"
            else:
                parts = [{"text": message.content}]
                role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": parts})
        generation_config: dict[str, Any] = {
            "temperature": request.temperature, "maxOutputTokens": request.max_output_tokens,
        }
        if request.response_schema:
            generation_config.update({"responseMimeType": "application/json", "responseSchema": request.response_schema})
        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
        if system:
            payload["systemInstruction"] = {"parts": system}
        if request.tools:
            payload["tools"] = [{"functionDeclarations": [{
                "name": tool.name, "description": tool.description, "parameters": tool.parameters,
            } for tool in request.tools]}]
            mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}[request.tool_choice]
            payload["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
        return payload

    def _request(self, request: GenerationRequest, *, streaming: bool = False) -> Any:
        operation = "streamGenerateContent?alt=sse" if streaming else "generateContent"
        endpoint = f"{self.config.endpoint.rstrip('/')}/models/{self.config.model}:{operation}"
        http_request = Request(
            endpoint, data=json.dumps(self._payload(request), ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream" if streaming else "application/json", "x-goog-api-key": self._key(request)},
            method="POST",
        )
        try:
            if streaming:
                return urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds)
            with urlopen(http_request, timeout=request.timeout_seconds or self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise LLMError("Gemini returned an HTTP error.", code="rate_limited" if exc.code == 429 else "http_error", retryable=exc.code == 429 or exc.code >= 500, status_code=exc.code) from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError("Gemini request timed out.", code="timeout", retryable=True) from exc
        except URLError as exc:
            raise LLMError("Gemini could not be reached.", code="connection_error", retryable=True) from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError("Gemini returned invalid JSON.", code="invalid_provider_response") from exc

    @staticmethod
    def _normalize(raw: dict[str, Any], request: GenerationRequest, config: ModelConfig, duration_ms: int) -> GenerationResponse:
        candidates = raw.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini response did not include a candidate.", code="invalid_provider_response")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))
        calls = [ToolCall(
            call_id=str(index), name=(part.get("functionCall") or {}).get("name", ""),
            arguments=(part.get("functionCall") or {}).get("args") or {},
        ) for index, part in enumerate(parts) if part.get("functionCall")]
        usage = raw.get("usageMetadata") or {}
        prompt_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        return GenerationResponse(
            request_id=request.request_id, provider=config.provider, model=config.model, text=text, tool_calls=calls,
            usage=TokenUsage(input_tokens=prompt_tokens, output_tokens=output_tokens,
                             total_tokens=usage.get("totalTokenCount") or (prompt_tokens + output_tokens if prompt_tokens is not None and output_tokens is not None else None)),
            duration_ms=duration_ms, finish_reason=candidate.get("finishReason"), provider_request_id=raw.get("responseId"),
        )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        started = time.perf_counter()
        raw = self._request(request)
        return self._normalize(raw, request, self.config, max(0, round((time.perf_counter() - started) * 1000)))

    def stream(self, request: GenerationRequest) -> Iterator[str | StreamEvent]:
        response = self._request(request, streaming=True)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError as exc:
                    raise LLMError("Gemini stream contained invalid JSON.", code="invalid_provider_response") from exc
                candidates = event.get("candidates") or []
                text_delta = ""
                if candidates:
                    for part in (candidates[0].get("content") or {}).get("parts", []):
                        if isinstance(part.get("text"), str) and part["text"]:
                            text_delta += part["text"]
                usage_payload = event.get("usageMetadata") or {}
                usage = None
                if usage_payload:
                    usage = TokenUsage(
                        input_tokens=usage_payload.get("promptTokenCount"),
                        output_tokens=usage_payload.get("candidatesTokenCount"),
                        total_tokens=usage_payload.get("totalTokenCount"),
                    )
                if text_delta or usage is not None:
                    yield StreamEvent(text_delta=text_delta, usage=usage)
        finally:
            response.close()
