"""Google Gemini.

The odd one out, and the one this project got wrong twice.

The design was written against `generate_content` with `response_schema`; that
is no longer the call. The replacement was then written from a documentation
summary and was wrong in four separate ways, all of which were caught by
introspecting the installed SDK rather than by reading about it. Recorded here
because every one of them would have failed silently or late:

  - The system prompt is `system_instruction`, not `instructions`. Passing the
    wrong name would have sent the entire rules reference as *nothing* - the
    SDK takes `**body`, so a misspelled parameter is not an error, it is an
    omission, and the agent would have played with no rules at all.
  - `response_format` is a *list* of per-modality formats, and each entry has
    to be a plain dict using the wire aliases - the SDK's own
    `TextResponseFormat` object fails to unmarshal, and the schema key is
    `schema` on the wire even though the Python field is called `jsonSchema`.
    Sending `jsonSchema` is accepted and silently ignored: the model answers in
    prose, every turn fails to parse, and every agent passes.
  - `max_output_tokens` lives inside `generation_config`.

The response side had two more. There is no `finish_reason` and no `candidates`
array - completion is reported as `status`, and a refusal shows up as a status
plus an `errors` list. And the usage counter is `total_output_tokens`; the
`candidates_token_count` name belongs to the old `generate_content` response, so
reading it would have priced every Gemini turn's output at zero and quietly
undercounted the budget for a whole match.
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
        # There is no Gemini pro above 3.1, and `gemini-3.6-pro` 404s. The
        # registry default was fixed and this one was missed, so anyone
        # constructing the client directly still got the dead id.
        model: str = "gemini-3.1-pro-preview",
        *,
        api_key: str | None = None,
        max_output_tokens: int = 8_000,
        # Just under the orchestrator's per-turn deadline. If the HTTP client
        # gave up first the turn would be recorded as a transport timeout rather
        # than as the model taking too long, which are different diagnoses.
        timeout: float = 400.0,
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
                # The stable prefix, as everywhere. This vendor caches
                # implicitly on prefix match rather than taking an explicit
                # breakpoint, so keeping it here and keeping it byte-identical
                # is the whole trick.
                system_instruction=system,
                input=user,
                # A *list* of per-modality formats, using **wire** names. Two
                # traps here, both of which produce a 200 rather than an error:
                #
                #   - The SDK's own `TextResponseFormat` fails to unmarshal when
                #     passed as an object, so this has to be a plain dict.
                #   - That dict is sent as written, so it needs the serialisation
                #     aliases rather than the Python field names. The field is
                #     `jsonSchema`; its alias, and the only spelling the API
                #     honours, is `schema`. Sending `jsonSchema` is accepted and
                #     silently ignored - the model then answers in prose, every
                #     turn fails to parse, and every agent passes.
                response_format=[{"mime_type": "application/json", "schema": schema}],
                generation_config={"max_output_tokens": self._max_output_tokens},
            )
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self.name) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        status = str(getattr(response, "status", "") or "")
        if status in {"failed", "cancelled"}:
            # A refusal arrives as a status plus an errors list rather than as
            # a distinguished stop reason.
            errors = "; ".join(
                str(getattr(e, "message", e)) for e in (getattr(response, "errors", None) or [])
            )
            raise Refused(f"{status}: {errors or 'no detail given'}", provider=self.name)
        if status in {"incomplete", "budget_exceeded"}:
            raise Malformed(f"response {status}", provider=self.name)

        text = getattr(response, "output_text", "") or ""
        if not text:
            raise Malformed(f"empty response body (status={status})", provider=self.name)

        return Turn(
            text=text,
            usage=_usage(getattr(response, "usage", None)),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=status,
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            await close()


def _field(raw: Any, name: str) -> int:
    return int(getattr(raw, name, 0) or 0)


def _usage(raw: Any) -> Usage:
    """Read the Interactions usage counters.

    These are `total_*` names on this surface. The `prompt_token_count` /
    `candidates_token_count` pair belongs to the older `generate_content`
    response, and reading those here silently returned zero for every field -
    which prices a whole match at nothing and leaves the budget halt disarmed.
    """
    if raw is None:
        return Usage()
    cached = _field(raw, "total_cached_tokens")
    return Usage(
        # The input count is inclusive of the cached portion, which prices at a
        # tenth, so it is split out rather than charged at the full rate twice.
        input_tokens=max(0, _field(raw, "total_input_tokens") - cached),
        output_tokens=_field(raw, "total_output_tokens"),
        cache_read_tokens=cached,
        reasoning_tokens=_field(raw, "total_thought_tokens"),
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
