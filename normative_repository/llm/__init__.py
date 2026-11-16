"""Provider-independent language model gateway."""

from .gateway import LLMGateway
from .models import (
    GenerationRequest,
    GenerationResponse,
    LLMCapability,
    LLMMessage,
    ModelConfig,
    TokenUsage,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    UsageRecord,
)
from .providers.openai_compatible import OpenAICompatibleProvider
from .providers.ollama_native import OllamaNativeProvider
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .factory import build_gateway_from_environment
from .prompts import PromptRegistry, PromptTemplate, default_prompt_registry
from .usage import SqliteUsageLedger
from .credentials import CredentialResolver, EnvironmentCredentialResolver
from .evaluation import evaluate_model_benchmarks

__all__ = [
    "GenerationRequest",
    "GenerationResponse",
    "LLMCapability",
    "LLMGateway",
    "LLMMessage",
    "ModelConfig",
    "OpenAICompatibleProvider",
    "OllamaNativeProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "TokenUsage",
    "StreamEvent",
    "ToolCall",
    "ToolDefinition",
    "UsageRecord",
    "build_gateway_from_environment",
    "PromptRegistry",
    "PromptTemplate",
    "default_prompt_registry",
    "SqliteUsageLedger",
    "CredentialResolver",
    "EnvironmentCredentialResolver",
    "evaluate_model_benchmarks",
]
