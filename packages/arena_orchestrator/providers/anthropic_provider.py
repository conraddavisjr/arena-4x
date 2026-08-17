"""Anthropic.

Four things here are specific to this vendor and easy to get wrong:

1. **Sampling parameters are rejected.** `temperature`, `top_p` and `top_k`
   return a 400 on Opus 5. A generic provider interface that carries a
   temperature and passes it along will fail every request, so this adapter
   never accepts one - behavioural variety comes from the prompt.

2. **Thinking is adaptive, and silent by default.** `budget_tokens` is removed
   on Opus 5 and 400s if sent. Thinking is on by default, but the *text* is
   omitted unless `display` is set - which matters more here than in most
   applications, because reading the model's reasoning is the entire experiment.

3. **A refusal is an HTTP 200.** `stop_reason == "refusal"` arrives with a
   normal response whose content array may be empty. Reading `content[0]` before
   checking the stop reason turns a refusal into an IndexError, and then into a
   crash report about a bug that is not there.

4. **Caching is a strict prefix match, and the breakpoint placement matters
   more than it looks.** One changed byte in the system block invalidates
   everything after it. The convenient top-level `cache_control` caches the
   *last* cacheable block - which is the user turn, carrying an observation that
   differs every turn - so it wrote a fresh entry on every call and never read
   one. The breakpoint belongs on the system block, which is the only part of
   the request that is byte-identical turn to turn.

The SDK is imported inside `__init__` rather than at module scope, so a repo
without the orchestrator extra installed can still import the package.
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
    unstalled,
)

# Above this, non-streaming requests hit the SDK's HTTP timeout. The action
# payload plus adaptive thinking sits well under it, but a match is long and
# the ceiling is cheap to respect.
STREAM_ABOVE_MAX_TOKENS = 16_000

# Adaptive thinking is a 4.6-and-later feature. Sending it to an older model is
# a 400, not a graceful downgrade - which is how the shakeout roster lost every
# turn on `claude-haiku-4-5` while the flagship roster was fine. Listed as
# families that *do* support it, so an unrecognised model degrades to no
# thinking rather than to a hard failure.
ADAPTIVE_THINKING = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-fable-5",
    "claude-mythos-5",
)


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        api_key: str | None = None,
        # Caps thinking *and* response text together, so this is sized for both.
        # Above STREAM_ABOVE_MAX_TOKENS the adapter switches to streaming, which
        # is why that ceiling exists.
        max_tokens: int = 32_000,
        effort: str = "high",
        thinking_display: str | None = "summarized",
        # The transport backstop. A stalled stream is now caught by `stall_gap_s`
        # in a fraction of this, so this only fires on something the stream layer
        # cannot see - a connection that never opens at all.
        timeout: float = 400.0,
        # How long the stream may say nothing before it is treated as dead. Well
        # above the gap between tokens on any of these models and well below any
        # plausible total, which is the property that lets it tell a model
        # thinking from a socket that has stopped talking.
        stall_gap_s: float = 90.0,
    ):
        import anthropic

        self._sdk = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self._max_tokens = max_tokens
        self._effort = effort
        self._thinking_display = thinking_display
        self._stall_gap_s = stall_gap_s

    @property
    def _supports_adaptive(self) -> bool:
        return self.model.startswith(ADAPTIVE_THINKING)

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            # The breakpoint goes *on the system block*, not at the top level.
            #
            # Top-level `cache_control` caches the last cacheable block, and the
            # last block is the user turn - which carries the observation and is
            # different every turn. So the cache key included the board: every
            # turn wrote a fresh 5,033-token entry at 1.25x and never read one.
            # Measured side by side: top-level wrote on every call, a system
            # breakpoint wrote once and then read 5,015 tokens on each call
            # after. That is 12.5x the input cost of the prefix, forever, and it
            # shows up as nothing except a slightly larger bill.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            # `format` constrains the body to the action schema; `effort` is the
            # reasoning dial. Both live under output_config - the old top-level
            # `output_format` is deprecated API-wide.
            "output_config": output_config,
        }
        # `effort` travels with adaptive thinking; both are 4.6-and-later.
        if self._supports_adaptive:
            thinking: dict[str, Any] = {"type": "adaptive"}
            if self._thinking_display:
                thinking["display"] = self._thinking_display
            request["thinking"] = thinking
            output_config["effort"] = self._effort

        started = time.monotonic()
        try:
            if self._max_tokens > STREAM_ABOVE_MAX_TOKENS:
                async with self._client.messages.stream(**request) as stream:
                    # Drained through the stall guard rather than awaited whole.
                    # `get_final_message()` on its own is a single await that
                    # cannot tell "still arriving" from "stopped arriving", and
                    # a stream that dies mid-message then costs the entire turn
                    # deadline with nothing to show for it. Consuming the events
                    # first gives the guard something to measure; the accumulated
                    # message is still assembled by the SDK, so usage, refusals
                    # and stop reasons are read exactly as before.
                    async for _ in unstalled(stream, gap_s=self._stall_gap_s, provider=self.name):
                        pass
                    message = await stream.get_final_message()
            else:
                message = await self._client.messages.create(**request)
        except Exception as error:  # noqa: BLE001 - translated below, never swallowed
            raise _translate(error, self._sdk) from error
        latency_ms = int((time.monotonic() - started) * 1000)

        # Before touching content: a refusal can carry an empty content array.
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise Refused(f"model refused: {details}", provider=self.name)

        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        reasoning = "\n".join(
            b.thinking for b in message.content if getattr(b, "type", None) == "thinking"
        )
        if not text:
            raise Malformed(
                f"no text content (stop_reason={message.stop_reason})", provider=self.name
            )
        if message.stop_reason == "max_tokens":
            raise Malformed("response truncated at max_tokens", provider=self.name)

        return Turn(
            text=text,
            usage=_usage(message.usage),
            model=self.model,
            latency_ms=latency_ms,
            stop_reason=message.stop_reason,
            thinking=reasoning or None,
        )

    async def aclose(self) -> None:
        await self._client.close()


def _field(raw: Any, name: str) -> int:
    return int(getattr(raw, name, 0) or 0)


def _usage(raw: Any) -> Usage:
    return Usage(
        input_tokens=_field(raw, "input_tokens"),
        output_tokens=_field(raw, "output_tokens"),
        cache_read_tokens=_field(raw, "cache_read_input_tokens"),
        cache_write_tokens=_field(raw, "cache_creation_input_tokens"),
    )


def retry_after_of(error: Exception) -> float | None:
    """The server's own hint about when to come back, if it gave one."""
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers["retry-after"])
    except (KeyError, TypeError, ValueError):
        return None


def _translate(error: Exception, sdk: Any) -> ProviderError:
    """Turn an SDK exception into one the retry policy understands.

    Matched by SDK class rather than by status code, because the SDK is what
    knows which status it mapped from - and the status alone is ambiguous
    anyway, since a 429 from the rate limiter and a 400 from a bad schema both
    look like "the request failed" from out here.
    """
    if isinstance(error, ProviderError):
        return error
    name = "anthropic"
    if isinstance(error, sdk.RateLimitError):
        return RateLimited(str(error), provider=name, retry_after=retry_after_of(error))
    if isinstance(error, sdk.APITimeoutError):
        return Timeout(str(error), provider=name)
    if isinstance(error, sdk.APIConnectionError):
        return Overloaded(str(error), provider=name)
    if isinstance(error, sdk.APIStatusError):
        # 529 is Anthropic's "overloaded"; anything else in the 5xx range is the
        # fleet rather than the request. A 4xx is ours to fix and never retried.
        status = getattr(error, "status_code", 0)
        if status >= 500:
            return Overloaded(str(error), provider=name)
        return FatalProviderError(str(error), provider=name)
    return FatalProviderError(str(error), provider=name)
