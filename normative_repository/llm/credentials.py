"""Credential resolution interfaces for runtime and tenant secret managers."""

from __future__ import annotations

import os
from typing import Protocol


class CredentialResolver(Protocol):
    """Resolve a provider key using the authenticated organization context."""

    def resolve(self, *, organization_id: str | None, provider: str, credential_ref: str | None) -> str | None:
        """Return a credential or ``None``; implementations must enforce tenant isolation."""
        ...


class EnvironmentCredentialResolver:
    """Resolve process-scoped keys by environment variable name."""

    def resolve(self, *, organization_id: str | None, provider: str, credential_ref: str | None) -> str | None:
        del organization_id, provider
        return os.environ.get(credential_ref) if credential_ref else None


def resolve_credential(
    resolver: CredentialResolver | None,
    *,
    organization_id: str | None,
    provider: str,
    credential_ref: str | None,
) -> str | None:
    """Use the injected tenant resolver as authoritative, otherwise environment lookup."""
    if resolver is not None:
        return resolver.resolve(
            organization_id=organization_id, provider=provider, credential_ref=credential_ref,
        )
    return EnvironmentCredentialResolver().resolve(
        organization_id=organization_id, provider=provider, credential_ref=credential_ref,
    )
