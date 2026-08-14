"""Google Gemini.

The odd one out. Google's current surface is `client.interactions.create`, with
the schema under a `response_format` dict that looks superficially like
OpenAI's and is not: the discriminator is `{"type": "text", "mime_type":
"application/json", "schema": ...}` rather than a `json_schema` type, the body
comes back on `output_text`, and there is no `choices` array to unwrap.

Worth stating because the plan for this project was written against
`generate_content` with `response_schema` and `response_mime_type`, and that is
no longer the call. Anything here that reads like an over-defensive `getattr`
is guarding the fields this vendor renames most often.
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


class GoogleClient:
    name = "google"

    def __init__(
        self,
        model: str = "gemini-3.6-pro",
        *,
        api_key: str | None = None,
        max_output_tokens: int = 8_000,
        timeout: float = 180.0,
    ):
        from google import genai

        self._sdk = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        started = time.monotonic()
        try:
            response = await self._client.aio.interactions.create(
                model=self.model,
                # The system prompt is the stable prefix here as everywhere; this
                # vendor caches implicitly on prefix match rather than taking an
                # explicit breakpoint, so keeping it first is the whole trick.
                instructions=system,
                input=user,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                max_output_tokens=self._max_output_tokens,
            )
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self.name) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        finish = _finish_reason(response)
        if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
            raise Refused(f"blocked: {finish}", provider=self.name)
        if finish == "MAX_TOKENS":
            raise Malformed("response truncated at max_output_tokens", provider=self.name)

        text = getattr(response, "output_text", "") or ""
        if not text:
            raise Malformed(f"empty response body (finish={finish})", provider=self.name)

        raw_usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        return Turn(
            text=text,
            usage=_usage(raw_usage),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=finish,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


def _finish_reason(response: Any) -> str | None:
    direct = getattr(response, "finish_reason", None)
    if direct:
        return str(direct)
    for candidate in getattr(response, "candidates", None) or []:
        if reason := getattr(candidate, "finish_reason", None):
            return str(reason)
    return None


def _field(raw: Any, name: str) -> int:
    return int(getattr(raw, name, 0) or 0)


def _usage(raw: Any) -> Usage:
    if raw is None:
        return Usage()
    cached = _field(raw, "cached_content_token_count")
    prompt = _field(raw, "prompt_token_count") or _field(raw, "input_tokens")
    return Usage(
        # Like the chat-completions surface, the prompt count includes the
        # cached portion, which prices at a tenth - so it is split out rather
        # than counted twice at the full rate.
        input_tokens=max(0, prompt - cached),
        output_tokens=_field(raw, "candidates_token_count") or _field(raw, "output_tokens"),
        cache_read_tokens=cached,
        reasoning_tokens=_field(raw, "thoughts_token_count"),
    )


def _translate(error: Exception, provider: str) -> ProviderError:
    """Classify by status code, because this SDK raises one error type.

    `google.genai.errors.APIError` carries the status rather than splitting into
    a class per failure, so unlike the other three adapters there is nothing to
    match on but the number.
    """
    if isinstance(error, ProviderError):
        return error
    status = getattr(error, "code", None) or getattr(error, "status_code", 0)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = 0
    message = str(error)
    if status == 429:
        return RateLimited(message, provider=provider)
    if status >= 500:
        return Overloaded(message, provider=provider)
    if status:
        return FatalProviderError(message, provider=provider)
    if isinstance(error, TimeoutError):
        return Timeout(message, provider=provider)
    # No status at all is usually a socket problem rather than a rejection.
    if isinstance(error, ConnectionError | OSError):
        return Overloaded(message, provider=provider)
    return FatalProviderError(message, provider=provider)
