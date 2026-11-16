"""Provider protocol consumed by the model-selection gateway."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .models import GenerationRequest, GenerationResponse, ModelConfig, StreamEvent


class LLMProvider(Protocol):
    """Synchronous provider contract; async wrappers can use ``asyncio.to_thread``."""

    config: ModelConfig

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate one normalized response."""
        ...

    def stream(self, request: GenerationRequest) -> Iterator[str | StreamEvent]:
        """Yield text deltas from one response."""
        ...
