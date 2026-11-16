"""Provider-neutral language model data contracts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMCapability(str, Enum):
    """Capabilities a configured model/provider adapter can guarantee."""

    TEXT_GENERATION = "text_generation"
    STRUCTURED_OUTPUT = "structured_output"
    STREAMING = "streaming"
    TOOL_CALLS = "tool_calls"
    TOKEN_USAGE = "token_usage"


class LLMMessage(BaseModel):
    """A role-tagged conversation message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    name: str | None = None


class ToolDefinition(BaseModel):
    """A JSON-schema description for a tool; it is not permission to execute it."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1, max_length=2000)
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A provider-proposed tool invocation awaiting application authorization."""

    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: dict[str, Any]


class GenerationRequest(BaseModel):
    """Provider-neutral generation request with bounded parameters."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: f"llm-{uuid4().hex}")
    messages: list[LLMMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)
    timeout_seconds: float | None = Field(default=None, gt=0, le=600)
    response_schema: dict[str, Any] | None = None
    tools: list[ToolDefinition] = Field(default_factory=list, max_length=32)
    tool_choice: Literal["auto", "none", "required"] = "auto"
    prompt_id: str | None = Field(default=None, max_length=128)
    prompt_version: str | None = Field(default=None, max_length=64)
    organization_id: str | None = None
    user_id: str | None = None

    @field_validator("tools")
    @classmethod
    def tool_names_are_unique(cls, tools: list[ToolDefinition]) -> list[ToolDefinition]:
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("Tool names must be unique within one generation request.")
        return tools


class TokenUsage(BaseModel):
    """Provider-reported token counts; missing values remain unknown."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class StreamEvent(BaseModel):
    """Provider stream delta and optional authoritative token usage update."""

    model_config = ConfigDict(extra="forbid")

    text_delta: str = ""
    usage: TokenUsage | None = None


class GenerationResponse(BaseModel):
    """Normalized completion result without provider secrets."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    provider: str
    model: str
    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_ms: int = Field(ge=0)
    finish_reason: str | None = None
    provider_request_id: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StructuredGenerationResponse(BaseModel):
    """Validated structured model output paired with provider usage metadata."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    response: Any
    generation: GenerationResponse
    validation_attempts: int = Field(ge=1)


class UsageRecord(BaseModel):
    """Auditable usage record; no raw prompt or credential is persisted here."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    organization_id: str | None = None
    user_id: str | None = None
    provider: str
    model: str
    prompt_id: str | None = None
    prompt_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    duration_ms: int = Field(ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelConfig(BaseModel):
    """Runtime model settings; api_key_env stores a variable name, not a secret."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=256)
    endpoint: str = Field(min_length=1, max_length=2048)
    api_key_env: str | None = None
    capabilities: frozenset[LLMCapability]
    timeout_seconds: float = Field(default=60.0, gt=0, le=600)
    max_retries: int = Field(default=2, ge=0, le=5)
    structured_output_retries: int = Field(default=1, ge=0, le=2)
    circuit_failure_threshold: int = Field(default=5, ge=1, le=20)
    circuit_reset_seconds: float = Field(default=30.0, gt=0, le=600)

    @field_validator("endpoint")
    @classmethod
    def endpoint_has_safe_http_scheme(cls, endpoint: str) -> str:
        from urllib.parse import urlsplit

        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Provider endpoint must be an absolute HTTP or HTTPS URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Provider endpoint must not contain credentials, query data, or a fragment.")
        return endpoint.rstrip("/")


class LLMError(Exception):
    """Base error for provider-neutral LLM failures."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_error",
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code


class CapabilityNotSupportedError(LLMError):
    """Raised before dispatch when the selected provider lacks a capability."""

    def __init__(self, capability: LLMCapability, provider: str) -> None:
        super().__init__(
            f"Provider '{provider}' does not support capability '{capability.value}'.",
            code="capability_not_supported",
        )


class CircuitOpenError(LLMError):
    """Raised when a provider circuit breaker is open."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            f"Provider '{provider}' is temporarily unavailable after repeated failures.",
            code="circuit_open",
            retryable=True,
        )
