"""The seam between the match loop and four different vendors.

Everything above this module talks about *a model taking a turn*. Everything
below it deals in the specifics of one vendor's HTTP API. The whole point is
that the orchestrator never learns which is which.

That seam has to be genuinely narrow, because the four APIs agree on almost
nothing. Asking each for the same thing - "return an object matching this JSON
schema" - currently means four unrelated call shapes:

    Anthropic   messages.create(output_config={"format": {...}})
    OpenAI      responses.create(text={"format": {...}})
    Google      interactions.create(response_format={...})
    xAI         chat.completions.create(response_format={"json_schema": {...}})

and four unrelated usage payloads. They also disagree about what a refusal is,
which errors are worth retrying, and whether reasoning is something you can
read. A `Turn` is the small set of facts the match actually needs, and each
adapter's job is to produce one - or to raise one of the errors below.

**No SDK is imported here, or anywhere except inside the adapter that needs
it.** A missing xAI key must not break an Anthropic-only run, and installing
the orchestrator extra must not be a precondition for running the engine tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Usage:
    """What one request cost, in tokens.

    Cache reads and writes are separate fields rather than folded into
    `input_tokens` because they bill at different multipliers, and because a
    cache read count of zero across repeated turns is the signal that something
    has crept into the supposedly-stable prefix. Folding them in would hide the
    one number worth watching.

    Providers that report neither simply leave both at zero, which prices
    correctly and reads honestly.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    # OpenAI bills reasoning tokens as output and also reports them separately.
    # Kept for the efficiency stats, not for pricing - adding it to the output
    # count would charge for the same tokens twice.
    reasoning_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True, slots=True)
class Turn:
    """One model response, in the terms the match cares about.

    `text` is the raw body, not a parsed action: parsing and validation belong
    to the caller, so that a schema violation is recorded against the model that
    produced it rather than disappearing inside an adapter.
    """

    text: str
    usage: Usage
    model: str
    latency_ms: int
    stop_reason: str | None = None
    # Summarised reasoning, where the vendor exposes it at all. Anthropic needs
    # `thinking.display` set to return any; OpenAI and Google currently return
    # none, so this is None for them and the reasoning the lab actually studies
    # is the `reasoning` block the schema requires in the response body.
    thinking: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """One model, ready to take turns.

    `system` is byte-identical across every turn of a match, which is what makes
    it cacheable; adapters that support explicit cache breakpoints put one at
    the end of it. Anything that varies - the turn number, the board, the
    agent's own dossier - belongs in `user`.
    """

    name: str
    model: str

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        """Return the model's response, or raise a `ProviderError`."""
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP client."""
        ...


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
#
# The retry policy has to decide what to do without importing four SDKs to
# catch four exception hierarchies, so every adapter translates into these.
# The distinction that matters is not what went wrong but what to do next, so
# each error carries that as a property rather than leaving the policy to
# pattern-match on types it might not know about.


class ProviderError(Exception):
    """Base class. Never raised directly."""

    retryable: bool = False

    def __init__(self, message: str, *, provider: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after


class RateLimited(ProviderError):
    """429. `retry_after` is the server's hint, in seconds, when it gives one."""

    retryable = True


class Overloaded(ProviderError):
    """5xx, including Anthropic's 529. The request was fine; the fleet was not."""

    retryable = True


class Timeout(ProviderError):
    retryable = True


class Malformed(ProviderError):
    """A 200 whose body is not usable: truncated mid-JSON, or empty content.

    Retryable because a second attempt often succeeds, but the caller is
    expected to give up quickly rather than burn a budget on it.
    """

    retryable = True


class Refused(ProviderError):
    """The model declined. Not a failure of the request, so not retryable.

    Anthropic signals this with HTTP 200 and `stop_reason == "refusal"`, which
    is worth stating out loud: code that reads `content[0]` before checking the
    stop reason raises IndexError on an empty content array and reports a
    refusal as a crash.
    """


class FatalProviderError(ProviderError):
    """Bad key, bad model id, malformed request. Retrying cannot help."""
