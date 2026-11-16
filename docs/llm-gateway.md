# LLM Gateway

The gateway is provider-neutral and is not invoked by repository ingestion or
RAG automatically. Applications must construct it explicitly and pass a bounded
`GenerationRequest`.

## Runtime configuration

Required environment variables:

- `LLM_ENDPOINT`: provider base URL, without credentials, query string, or fragment.
- `LLM_PRIMARY_MODEL`: model name.

Optional variables:

- `LLM_PROVIDER`: normalized provider label (defaults to `ollama`).
- `LLM_ADAPTER`: `openai_compatible` (default), `ollama_native`, `anthropic`, or `gemini`.
- `LLM_API_KEY_ENV`: name of an environment variable holding the API key for
  compatible, Anthropic, or Gemini adapters. Native Ollama rejects API keys.
- `LLM_CAPABILITIES`: comma-separated capabilities. If omitted, only
  `text_generation` is enabled.
- `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, `LLM_STRUCTURED_RETRIES`.
- `LLM_SECONDARY_MODEL` and optional `LLM_SECONDARY_ENDPOINT`,
  `LLM_SECONDARY_ADAPTER`, `LLM_SECONDARY_PROVIDER`, `LLM_SECONDARY_API_KEY_ENV`.
- `LLM_TASK_ROUTES`: JSON mapping task names to `primary` or `secondary`.
- `LLM_FALLBACK_ROUTES`: optional JSON map from provider alias to an explicitly
  allowed fallback alias. Cyclic routes are rejected.
- `LLM_DAILY_TOKEN_LIMITS`: optional JSON map from organization IDs to daily
  token limits.
- `LLM_MODEL_RATES_USD_PER_MILLION`: optional JSON map such as
  `{"ollama/qwen2.5:14b":[0,0]}` for input and output prices.
- `LLM_USAGE_DB`: SQLite usage-ledger path (defaults to
  `data/reports/llm-usage.sqlite3` in the CLI).
- `LEGAL_AUDIT_DB`: opt-in SQLite legal-agent audit ledger. It stores queries,
  source IDs, verification results, and verified generated analysis.

The compatible adapter sends requests to `<LLM_ENDPOINT>/chat/completions`; the
native Ollama adapter sends requests to `<LLM_ENDPOINT>/api/chat`. For example,
use `http://ollama:11434/v1` with `openai_compatible` or
`http://ollama:11434` with `ollama_native`. Anthropic uses the configured base
URL and `/v1/messages`; Gemini uses the configured API base and model-specific
`generateContent` route. Set `LLM_PROVIDER` to the matching provider label for
pricing and usage attribution. Do not put secrets in the endpoint.

## Python use

```python
from normative_repository.llm import (
    GenerationRequest,
    LLMMessage,
    build_gateway_from_environment,
)

gateway = build_gateway_from_environment()
result = gateway.generate(GenerationRequest(messages=[
    LLMMessage(role="user", content="Summarize the supplied evidence."),
]))
```

Structured output and streaming require the corresponding capability to be
explicitly enabled in `LLM_CAPABILITIES`. Tool calls are returned as proposals;
the gateway never executes them. Usage callbacks receive token and latency
metadata without prompts or inferred cost. The local SQLite usage ledger
supports organization attribution, optional token quotas, and configured
per-model cost rates. Prompt templates are versioned in an immutable in-memory
registry. The `legal-answer` CLI command performs retrieval and structured
answer generation; `llm-usage-report --organization-id ...` reports usage.

BYOK credentials must be supplied via environment-variable names backed by the
deployment secret manager. Encrypted tenant credential storage, user
authentication, and organization-level authorization are not implemented in
this repository. Applications may inject a `CredentialResolver`; it receives
the organization identity and provider so an authenticated deployment can
resolve isolated tenant keys without changing adapters.

When `LEGAL_AUDIT_DB` is configured, agent events are appended to a local
SQLite ledger with a hash chain and update/delete prevention triggers. The
`legal-audit-verify --database ...` command checks chain integrity. This is
tamper-evident local storage, not a substitute for an access-controlled,
externally protected audit service. Keep the ledger path private and apply the
deployment's retention policy.
