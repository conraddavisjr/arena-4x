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

Both return reasoning text, and neither does so by default. OpenAI needs
`reasoning.summary` requested or the reasoning items come back with an empty
summary list - billed, spent, discarded. xAI hangs a `reasoning_content` field
off the message, which is a vendor extension to a shared schema and therefore
absent from the SDK's typed model, so it is read by name.

That is separate from the `reasoning` block the action schema requires in the
body. The block is the account a model writes knowing it will be read and handed
back next turn; the trace is the deliberation behind it. Both are worth keeping
and they are not the same evidence.
"""

from __future__ import annotations

import os
import time
from typing import Any

from .base import (
    FatalProviderError,
    Malformed,
    OutOfCredits,
    Overloaded,
    ProviderError,
    RateLimited,
    Refused,
    Timeout,
    Turn,
    Usage,
    unstalled,
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
        # Reasoning tokens are billed against this cap, not on top of it, so a
        # budget sized for the action payload alone is consumed entirely by
        # thinking and the response comes back incomplete with nothing in it.
        # Measured: gpt-5.4-mini at effort=high spent all 8k on reasoning for
        # every turn of a shakeout. Sized for thinking plus the payload; unused
        # headroom costs nothing, since billing is on tokens produced.
        max_output_tokens: int = 32_000,
        reasoning_effort: str | None = "medium",
        # The transport backstop, for a connection that never opens. A stream
        # that opens and then dies is caught by the gap below, far sooner.
        timeout: float = 400.0,
        # How long the stream may say nothing before it is treated as dead.
        stall_gap_s: float = 90.0,
    ):
        import openai

        self._sdk = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self._max_output_tokens = max_output_tokens
        self._reasoning_effort = reasoning_effort
        self._stall_gap_s = stall_gap_s

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
            # `summary` has to be asked for. Without it the reasoning items come
            # back with an empty `summary` list and the trace is billed, spent
            # and discarded - which is what was happening: the tokens showed up
            # in `reasoning_tokens` and the thinking behind them went nowhere.
            # It is a summary rather than the raw chain, because the raw chain is
            # not offered on this surface at all.
            request["reasoning"] = {"effort": self._reasoning_effort, "summary": "auto"}

        started = time.monotonic()
        try:
            # Streamed for the stall guard, not for the tokens. This seat has the
            # highest median latency of the four and it has hung at the turn
            # deadline as well, which is the pairing a total-duration cap handles
            # worst: any value low enough to catch the hang truncates the normal
            # case. Silence between events separates them.
            #
            # The response is taken from the events, not from
            # `get_final_response()`. That helper raises "Didn't receive a
            # `response.completed` event" whenever a stream ends any other way -
            # and the most common other way is `response.incomplete`, which is
            # what this model does when reasoning exhausts the output cap.
            #
            # Non-streaming `create()` returned that as an ordinary object with
            # `status == "incomplete"`, handled below as a retryable `Malformed`
            # carrying a diagnosis. Streaming turned the same condition into a
            # fatal error, which then tripped the circuit breaker: measured on a
            # live 51-turn run, this seat lost 21 of its turns - 41% - to a
            # condition it had been recovering from for weeks.
            #
            # Every terminal event carries the response, so keeping the last one
            # seen restores the old behaviour exactly while keeping the stall
            # guard.
            final = None
            async with self._client.responses.stream(**request) as stream:
                async for event in unstalled(stream, gap_s=self._stall_gap_s, provider=self.name):
                    final = getattr(event, "response", None) or final
            if final is None:
                raise Malformed("stream ended with no response object", provider=self.name)
            response = final
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self._sdk, self.name) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        refusal = _refusal_of(response)
        if refusal:
            raise Refused(refusal, provider=self.name)

        # Status before content, always. An incomplete response has empty text,
        # so checking text first reports "empty response body" - which is true,
        # useless, and sent me looking at the schema when the actual cause was
        # reasoning tokens eating the whole output budget.
        if getattr(response, "status", None) == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", "unknown")
            spent = getattr(getattr(response, "usage", None), "output_tokens", 0)
            raise Malformed(
                f"response incomplete: {reason} (output_tokens={spent}, "
                f"cap={self._max_output_tokens}); reasoning is billed against this cap",
                provider=self.name,
            )
        text = getattr(response, "output_text", "") or ""
        if not text:
            raise Malformed("empty response body", provider=self.name)

        return Turn(
            text=text,
            usage=_responses_usage(getattr(response, "usage", None)),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=getattr(response, "status", None),
            thinking=_reasoning_of(response),
            effort=self._reasoning_effort,
            effort_sent=self._reasoning_effort,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _reasoning_of(response: Any) -> str | None:
    """The reasoning summaries, joined, or None.

    Reasoning arrives as its own item type in `output` rather than as a field on
    the response, and each item carries a list of summary parts. Read defensively
    because a model may return no reasoning item at all, and because whether the
    summaries are populated depends on the request having asked for them.
    """
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for chunk in getattr(item, "summary", None) or []:
            text = getattr(chunk, "text", "")
            if text:
                parts.append(text)
    return "\n\n".join(parts) or None


class XAIClient:
    """Grok, over the OpenAI-compatible chat-completions surface."""

    name = "xai"

    def __init__(
        self,
        model: str = "grok-4.6",
        *,
        api_key: str | None = None,
        base_url: str = XAI_BASE_URL,
        max_tokens: int = 8_000,
        timeout: float = 400.0,
        # Was sent nothing at all, so this seat ran on whatever the vendor
        # defaults to while two other seats were explicitly set to `high`.
        reasoning_effort: str | None = "medium",
    ):
        import openai

        self._sdk = openai
        # The key has to be resolved here rather than left to the SDK. Riding
        # OpenAI's client means its fallback is `OPENAI_API_KEY`, so an unset
        # key sends the *OpenAI* credential to api.x.ai - which xAI answers,
        # entirely correctly, with "Incorrect API key provided". That message
        # sent me looking at billing and model ids for an hour.
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("XAI_API_KEY", ""),
            base_url=base_url,
            timeout=timeout,
        )
        self.model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        started = time.monotonic()
        extra = {"reasoning_effort": self._reasoning_effort} if self._reasoning_effort else {}
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=self._max_tokens,
                **extra,
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
            # xAI returns its trace as `reasoning_content` on the message. That
            # field is not in the OpenAI SDK's typed model - it is a vendor
            # extension riding a shared schema - so it is read by name and
            # tolerated absent rather than declared.
            thinking=getattr(choice.message, "reasoning_content", None) or None,
            effort=self._reasoning_effort,
            effort_sent=self._reasoning_effort,
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
    cached = _field(details, "cached_tokens")
    return Usage(
        # Inclusive of the cached portion here, exactly as on the completions
        # surface below, so the cached count is subtracted out rather than left
        # to be counted twice. It was not, and the arithmetic is unkind: the
        # pricer charges `input * 1.0 + cache_read * 0.1`, so every cached token
        # on this seat was billed at 1.1x instead of 0.1x - eleven times over.
        #
        # `Usage` states the contract in its docstring and the sibling function
        # forty lines down states it again in a comment. It still went wrong,
        # because nothing failed: the number was plausible, the seat was the
        # expensive one anyway, and the only visible symptom was a cache-rate
        # column whose four seats were not measuring the same thing.
        input_tokens=max(0, _field(raw, "input_tokens") - cached),
        output_tokens=_field(raw, "output_tokens"),
        cache_read_tokens=cached,
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
    # Before anything else, including the 429 path: an exhausted account
    # often arrives *as* a rate limit, and retrying it burns the ladder on a
    # condition that cannot improve.
    if OutOfCredits.matches(str(error)):
        return OutOfCredits(str(error), provider=provider)
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
