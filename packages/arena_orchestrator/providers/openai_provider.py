"""OpenAI, and xAI riding on the same SDK.

Two surfaces, not one, and the difference is the whole reason this file has two
classes rather than a base URL parameter:

- **OpenAI** has moved to the Responses API. Structured output is
  `text={"format": {"type": "json_schema", "name": ..., "strict": True,
  "schema": ...}}` on `client.responses.create`, and usage comes back as
  `input_tokens` / `output_tokens` with reasoning broken out under
  `output_tokens_details`.
- **xAI** is OpenAI-*compatible*, which means the older chat-completions shape:
  `response_format={"type": "json_schema", "json_schema": {...}}` against
  `https://api.x.ai/v1`, with usage as `prompt_tokens` / `completion_tokens`.

Pointing the Responses API at xAI's base URL would fail, and pretending xAI is
"OpenAI with a different host" is exactly the assumption that would produce a
confusing 404 in the middle of a flagship run. Both share error translation and
nothing else.

Neither vendor currently returns reasoning text, only a count of reasoning
tokens, so `Turn.thinking` is None for both. The reasoning this lab studies
comes from the `reasoning` block the action schema requires in the body, which
is why that block is required rather than optional.
"""

from __future__ import annotations

import time
from typing import Any

from .base import (
    FatalProviderError,
    Malformed,
    Overloaded,
    ProviderError,
    RateLimited,
    Refused,
    Timeout,
    Turn,
    Usage,
)

XAI_BASE_URL = "https://api.x.ai/v1"
SCHEMA_NAME = "arena_action"


class OpenAIClient:
    """The Responses API."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5.6",
        *,
        api_key: str | None = None,
        max_output_tokens: int = 8_000,
        reasoning_effort: str | None = "high",
        timeout: float = 180.0,
    ):
        import openai

        self._sdk = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "max_output_tokens": self._max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": SCHEMA_NAME,
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        if self._reasoning_effort:
            request["reasoning"] = {"effort": self._reasoning_effort}

        started = time.monotonic()
        try:
            response = await self._client.responses.create(**request)
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self._sdk, self.name) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        refusal = _refusal_of(response)
        if refusal:
            raise Refused(refusal, provider=self.name)

        text = getattr(response, "output_text", "") or ""
        if not text:
            raise Malformed("empty response body", provider=self.name)
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            raise Malformed(f"response incomplete: {reason}", provider=self.name)

        return Turn(
            text=text,
            usage=_responses_usage(getattr(response, "usage", None)),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=getattr(response, "status", None),
        )

    async def aclose(self) -> None:
        await self._client.close()


class XAIClient:
    """Grok, over the OpenAI-compatible chat-completions surface."""

    name = "xai"

    def __init__(
        self,
        model: str = "grok-4",
        *,
        api_key: str | None = None,
        base_url: str = XAI_BASE_URL,
        max_tokens: int = 8_000,
        timeout: float = 180.0,
    ):
        import openai

        self._sdk = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model
        self._max_tokens = max_tokens

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        started = time.monotonic()
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": schema},
                },
            )
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self._sdk, self.name) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        if not response.choices:
            raise Malformed("no choices returned", provider=self.name)
        choice = response.choices[0]
        if getattr(choice.message, "refusal", None):
            raise Refused(choice.message.refusal, provider=self.name)
        if choice.finish_reason == "length":
            raise Malformed("response truncated at max_tokens", provider=self.name)

        text = choice.message.content or ""
        if not text:
            raise Malformed("empty message content", provider=self.name)

        return Turn(
            text=text,
            usage=_completions_usage(getattr(response, "usage", None)),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=choice.finish_reason,
        )

    async def aclose(self) -> None:
        await self._client.close()


# ---------------------------------------------------------------------------
# Shared translation
# ---------------------------------------------------------------------------


def _refusal_of(response: Any) -> str | None:
    """A refusal on the Responses API arrives as a content block, not a status."""
    for item in getattr(response, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "refusal":
                return getattr(block, "refusal", "refused")
    return None


def _field(raw: Any, name: str) -> int:
    return int(getattr(raw, name, 0) or 0)


def _responses_usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    details = getattr(raw, "input_tokens_details", None)
    return Usage(
        input_tokens=_field(raw, "input_tokens"),
        output_tokens=_field(raw, "output_tokens"),
        cache_read_tokens=_field(details, "cached_tokens"),
        # Reported for the efficiency stats only. These tokens are already
        # inside output_tokens, so adding them to it would bill them twice.
        reasoning_tokens=_field(getattr(raw, "output_tokens_details", None), "reasoning_tokens"),
    )


def _completions_usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    details = getattr(raw, "prompt_tokens_details", None)
    cached = _field(details, "cached_tokens")
    return Usage(
        # prompt_tokens is inclusive of the cached portion on this surface, so
        # the cached count is subtracted out rather than added alongside - it is
        # priced separately and would otherwise be charged at the full rate too.
        input_tokens=max(0, _field(raw, "prompt_tokens") - cached),
        output_tokens=_field(raw, "completion_tokens"),
        cache_read_tokens=cached,
        reasoning_tokens=_field(
            getattr(raw, "completion_tokens_details", None), "reasoning_tokens"
        ),
    )


def _translate(error: Exception, sdk: Any, provider: str) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, sdk.RateLimitError):
        headers = getattr(getattr(error, "response", None), "headers", None) or {}
        try:
            retry_after = float(headers["retry-after"])
        except (KeyError, TypeError, ValueError):
            retry_after = None
        return RateLimited(str(error), provider=provider, retry_after=retry_after)
    if isinstance(error, sdk.APITimeoutError):
        return Timeout(str(error), provider=provider)
    if isinstance(error, sdk.APIConnectionError):
        return Overloaded(str(error), provider=provider)
    if isinstance(error, sdk.APIStatusError):
        status = getattr(error, "status_code", 0)
        if status >= 500:
            return Overloaded(str(error), provider=provider)
        return FatalProviderError(str(error), provider=provider)
    return FatalProviderError(str(error), provider=provider)
