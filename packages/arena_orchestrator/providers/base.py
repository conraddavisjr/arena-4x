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

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


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
    # Summarised reasoning, where the vendor exposes it at all, and the vendors
    # differ sharply. Anthropic returns thinking blocks once `thinking.display`
    # is set; OpenAI returns reasoning summaries only if the request asks for
    # them; xAI hangs a `reasoning_content` field off the message as an
    # extension to a shared schema; Google's interactions surface bills thought
    # tokens and exposes no thought text at all.
    #
    # So None here means "this vendor did not offer one", never "the model did
    # not think". Distinct from the `reasoning` block the action schema requires,
    # which is the account a model writes knowing it will be read - this is the
    # deliberation behind that account.
    thinking: str | None = None
    # What this seat was actually asked for, as the adapter sent it. Recorded
    # because nothing recorded it, and so "were all four asked the same question
    # this turn" was unanswerable from the artifacts - which is exactly how one
    # seat came to play every match of this project with reasoning off while the
    # seat beside it was set to `high`.
    #
    # Two fields, because the vendors do not share an instrument. `effort` is the
    # match-level intent every seat gets; `effort_sent` is the vendor form, which
    # for a pre-4.6 Anthropic model is a token budget rather than an enum. Under
    # self-selected effort this stops being configuration and becomes a move, and
    # then the difference between the two is worth being able to read.
    effort: str | None = None
    effort_sent: str | None = None
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


# How hard every seat is told to think. One value for the match, mapped by each
# adapter onto whatever dial its vendor exposes.
#
# This exists because the four seats were not being asked the same question.
# Anthropic and OpenAI were sent `effort: high`; xAI and Google were sent
# nothing and ran on vendor defaults; and `claude-haiku-4-5`, being pre-4.6,
# had adaptive thinking skipped entirely and so received no reasoning
# instruction at all. Measured over 51 live turns the spread was 608k output
# tokens on one seat against 35k on another, with two seats producing no
# reasoning trace whatsoever. Some of that is the models differing. An unknown
# amount of it was us, and in a four-way comparison that is the part that
# invalidates the result.
#
# **Parity here is intent, not identity, and the difference matters.** An
# effort enum, a thinking-token budget and a thinking level are not the same
# instrument, and no mapping makes "high" mean the same thing to four vendors.
# What this guarantees is that every seat is asked deliberately and that the
# asking is recorded, so the residue is visible in the journal instead of
# hiding in four different defaults.
EFFORTS = ("low", "medium", "high")

# Anthropic before 4.6 takes a token budget rather than an enum. Sized well
# under the 32k output cap, because thinking is billed against it and a budget
# that leaves no room for the answer produces a truncated response instead.
THINKING_BUDGET = {"low": 2_000, "medium": 6_000, "high": 12_000}


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


class Stalled(Timeout):
    """A streaming response went silent mid-flight.

    Distinguished from `Timeout` because the two say different things and the
    difference is the whole point. A `Timeout` means "this took longer than we
    allow", which on a model that thinks for two minutes is often a statement
    about our patience rather than about the vendor. A `Stalled` means the
    connection stopped producing bytes while we were still reading it, which no
    amount of patience fixes and which a retry usually does.

    Two matches lost turns to a total deadline that could not tell these apart:
    a seat whose median latency is 23 seconds sat at the 420-second cap twice in
    a row, and it was recorded exactly as a slow model would have been.
    """

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


class OutOfCredits(FatalProviderError):
    """The account cannot pay. Distinguished because it ends the *match*.

    Every other failure here is survivable by design: an agent that cannot
    answer passes its turn, plays badly, and the match continues - which is the
    right response to a vendor having a bad ten minutes.

    An exhausted account is not that. It will not recover inside the run, so the
    "keep going" policy turns one problem into a slow corruption of the result.
    Measured: a 300-turn baseline ran 29 further turns after one seat's credits
    ran out, with that civ holding its cities and issuing no orders, on its way
    to producing a four-way comparison missing a fourth. It cost about six hours
    and would have cost eighteen dollars more.

    So this halts the match the way the budget cap does: on a coherent board,
    scored, with a reason recorded. One lost turn instead of two hundred and
    sixty.
    """

    # Recognised by message, because none of the four vendors gives this its own
    # status code or error type - it arrives as a 400 or a 429 whose body
    # happens to mention money. Matching on prose is fragile and is the only
    # option; a miss costs the old behaviour rather than a crash.
    # Every phrase here was observed on a live account, not guessed. Google's
    # wording was missed by the first version of this list and arrived as a
    # *429* - so it was classified `RateLimited`, which is retryable, which
    # means the ladder would have spun on a condition that cannot improve and
    # the match would have limped instead of halting. Found by the model-action
    # suite rather than by a run, which is the cheap way round.
    #
    # Deliberately specific. Matching "quota" or "billing" alone would catch
    # ordinary rate limits and halt a multi-day match over a bad ten minutes,
    # which is the failure this class exists to avoid causing.
    MARKERS = (
        "no credits remaining",
        "credit balance is too low",
        "insufficient_quota",
        "insufficient credits",
        "credits are depleted",
        "prepayment credits",
        "exceeded your current quota",
        "billing hard limit",
        "quota exceeded",
    )

    @classmethod
    def matches(cls, message: str) -> bool:
        low = message.lower()
        return any(marker in low for marker in cls.MARKERS)


async def unstalled(
    events: AsyncIterator[T],
    *,
    gap_s: float,
    provider: str = "",
) -> AsyncIterator[T]:
    """Pass a stream through, raising `Stalled` if it goes quiet for `gap_s`.

    This exists because a total-duration timeout is the wrong instrument for
    the job it was doing. A cap on how long a turn may take cannot distinguish
    a model thinking hard from a connection that died with the socket open, and
    the two need opposite responses: wait, and give up immediately.

    Both matches so far lost turns to that confusion. The seat that hung has a
    median latency of 23 seconds and sat at the 420-second cap twice in a row -
    obviously a stall, recorded identically to a slow model, and its score came
    in last as a result. Meanwhile the *fix* for the previous version of this
    problem had been to raise the cap from 180s to 420s, because at 180s a seat
    thinking legitimately for two minutes lost five turns out of seven. Every
    setting was wrong for one case or the other.

    Time between events answers both at once. A model streaming tokens is alive
    however long it takes; a stream silent for a minute and a half is not going
    to speak again. So a long thinker is never truncated and a hang is caught in
    a fraction of the time, which finally leaves room inside the turn for the
    retry that a hang actually needs.

    Wraps the iterator rather than replacing it, so an adapter keeps its own
    handling of usage, refusals and stop reasons untouched - the parts most
    likely to break if streaming were rebuilt around this.
    """
    iterator = events.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(iterator.__anext__(), timeout=gap_s)
        except StopAsyncIteration:
            return
        except TimeoutError as error:
            raise Stalled(f"stream produced nothing for {gap_s:.0f}s", provider=provider) from error
