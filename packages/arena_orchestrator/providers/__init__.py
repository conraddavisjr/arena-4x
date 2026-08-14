"""Provider registry.

The import is deferred into `build()` on purpose. Naming a provider in a roster
is what installs the cost of its SDK - a match played entirely on Anthropic must
not fail to start because `google-genai` is absent or because an unrelated
vendor's package raises at import time. The engine test suite depends on this
too: it imports nothing from here, and the orchestrator extra is not a
precondition for running it.
"""

from __future__ import annotations

from typing import Any

from .base import (
    FatalProviderError,
    LLMClient,
    Malformed,
    Overloaded,
    ProviderError,
    RateLimited,
    Refused,
    Timeout,
    Turn,
    Usage,
)

__all__ = [
    "FatalProviderError",
    "LLMClient",
    "Malformed",
    "Overloaded",
    "PROVIDERS",
    "ProviderError",
    "RateLimited",
    "Refused",
    "Timeout",
    "Turn",
    "Usage",
    "build",
]

PROVIDERS = ("anthropic", "openai", "google", "xai", "scripted")


def build(provider: str, model: str | None = None, **options: Any) -> LLMClient:
    """Construct one client by name.

    Raises `ValueError` for an unknown name rather than falling back to a
    default, because silently seating the wrong vendor in a four-way comparison
    would invalidate the whole match without anything looking wrong.
    """
    if provider == "anthropic":
        from .anthropic_provider import AnthropicClient

        return AnthropicClient(**_with_model(model, "claude-opus-5"), **options)
    if provider == "openai":
        from .openai_provider import OpenAIClient

        return OpenAIClient(**_with_model(model, "gpt-5.6"), **options)
    if provider == "xai":
        from .openai_provider import XAIClient

        return XAIClient(**_with_model(model, "grok-4"), **options)
    if provider == "google":
        from .google_provider import GoogleClient

        return GoogleClient(**_with_model(model, "gemini-3.6-pro"), **options)
    if provider == "scripted":
        from .scripted import ScriptedClient

        return ScriptedClient(**options)
    raise ValueError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")


def _with_model(model: str | None, default: str) -> dict[str, str]:
    return {"model": model or default}
