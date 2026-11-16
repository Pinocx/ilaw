"""Build a task-routed model gateway from runtime environment configuration."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from .gateway import LLMGateway
from .models import LLMCapability, ModelConfig
from .credentials import CredentialResolver
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.ollama_native import OllamaNativeProvider
from .providers.openai_compatible import OpenAICompatibleProvider


DEFAULT_CAPABILITIES = frozenset({LLMCapability.TEXT_GENERATION})
ADAPTERS = {
    "openai_compatible": OpenAICompatibleProvider,
    "ollama_native": OllamaNativeProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def build_gateway_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    usage_ledger: object | None = None,
    credential_resolver: CredentialResolver | None = None,
) -> LLMGateway:
    """Construct configured model routes without reading or exposing key values."""
    if environment is None:
        environment = os.environ

    def capabilities(raw: str | None) -> frozenset[LLMCapability]:
        if not raw:
            return DEFAULT_CAPABILITIES
        try:
            return frozenset(LLMCapability(value.strip()) for value in raw.split(",") if value.strip())
        except ValueError as exc:
            raise ValueError("LLM capabilities contain an unknown capability.") from exc

    def make_provider(*, secondary: bool = False):
        prefix = "LLM_SECONDARY_" if secondary else "LLM_"
        endpoint = environment.get(prefix + "ENDPOINT") or environment.get("LLM_ENDPOINT")
        model = environment.get(prefix + "MODEL") if secondary else environment.get("LLM_PRIMARY_MODEL")
        if not endpoint or not model:
            raise ValueError("LLM endpoint and model must be configured for every route.")
        adapter = (environment.get(prefix + "ADAPTER") or environment.get("LLM_ADAPTER", "openai_compatible")).strip().lower()
        provider_name = environment.get(prefix + "PROVIDER") or environment.get("LLM_PROVIDER", "ollama")
        api_key_env = environment.get(prefix + "API_KEY_ENV") or (environment.get("LLM_API_KEY_ENV") if not secondary else None)
        config = ModelConfig(
            provider=provider_name,
            model=model,
            endpoint=endpoint,
            api_key_env=api_key_env,
            capabilities=capabilities(environment.get(prefix + "CAPABILITIES") or environment.get("LLM_CAPABILITIES")),
            timeout_seconds=float(environment.get("LLM_TIMEOUT_SECONDS", "60")),
            max_retries=int(environment.get("LLM_MAX_RETRIES", "2")),
            structured_output_retries=int(environment.get("LLM_STRUCTURED_RETRIES", "1")),
        )
        provider_type = ADAPTERS.get(adapter)
        if provider_type is None:
            raise ValueError("Unsupported LLM adapter.")
        if adapter == "ollama_native" and api_key_env:
            raise ValueError("The native Ollama adapter does not accept remote API key configuration.")
        if adapter == "ollama_native":
            return provider_type(config)
        return provider_type(config, credential_resolver=credential_resolver)

    providers = {"primary": make_provider()}
    if environment.get("LLM_SECONDARY_MODEL"):
        providers["secondary"] = make_provider(secondary=True)
    try:
        routes = json.loads(environment.get("LLM_TASK_ROUTES", "{}"))
        fallbacks = json.loads(environment.get("LLM_FALLBACK_ROUTES", "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("LLM task and fallback routes must be valid JSON objects.") from exc
    if not isinstance(routes, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in routes.items()):
        raise ValueError("LLM_TASK_ROUTES must map task names to provider aliases.")
    if any(value not in providers for value in routes.values()):
        raise ValueError("LLM_TASK_ROUTES references an unconfigured provider alias.")
    if not isinstance(fallbacks, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in fallbacks.items()):
        raise ValueError("LLM_FALLBACK_ROUTES must map provider aliases to provider aliases.")
    if any(key not in providers or value not in providers for key, value in fallbacks.items()):
        raise ValueError("LLM_FALLBACK_ROUTES references an unconfigured provider alias.")
    return LLMGateway(
        providers,
        default_provider="primary",
        task_routes=routes,
        fallback_routes=fallbacks,
        usage_callback=usage_ledger if callable(usage_ledger) else None,
        usage_guard=getattr(usage_ledger, "authorize", None),
        usage_release=getattr(usage_ledger, "release", None),
    )
