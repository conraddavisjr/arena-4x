"""Keeping a multi-day match alive across four vendors' bad days.

A flagship run is four providers times three hundred turns, unattended, over
days. Something will 429, something will 529, and at some point one vendor will
be down for an hour. None of that may end the match: an agent that cannot
answer passes its turn and plays badly, which is a result. A crash is not.

Three mechanisms, deliberately separate because they fail differently:

- A **token bucket** throttles ahead of the vendor's limit, so the common case
  is waiting 200ms rather than being rejected and backing off for seconds.
- **Retry with backoff** handles the transient rejections that get through.
- A **circuit breaker** handles the case retry cannot: a provider that is down
  rather than busy, where continuing to retry every turn burns wall-clock for
  nothing.

Everything takes its clock, sleep and randomness as arguments. That is not
ceremony - it is what lets the tests assert that a breaker opens after five
failures and half-opens sixty seconds later without the suite taking sixty
seconds, and it keeps the whole module free of real time.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from .providers.base import Malformed, ProviderError

T = TypeVar("T")

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TokenBucket:
    """Requests and tokens per minute, refilled continuously.

    Two buckets rather than one, because the vendors enforce two limits and a
    match hits them at different times: the request limit binds when four
    agents fire at once at the top of a turn, the token limit binds later in a
    match when observations have grown.

    Continuous refill rather than a fixed window, because a window lets four
    agents pass together at the boundary and then stall for the rest of it -
    which is exactly the burst that trips the vendor's own limiter.
    """

    def __init__(
        self,
        requests_per_minute: float,
        tokens_per_minute: float,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleeper = asyncio.sleep,
    ):
        self._rpm = requests_per_minute
        self._tpm = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._requests = float(requests_per_minute)
        self._tokens = float(tokens_per_minute)
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._requests = min(self._rpm, self._requests + elapsed * self._rpm / 60)
        self._tokens = min(self._tpm, self._tokens + elapsed * self._tpm / 60)

    def _wait_for(self, tokens: int) -> float:
        """Seconds until both buckets can cover this request."""
        need_requests = max(0.0, 1 - self._requests)
        need_tokens = max(0.0, tokens - self._tokens)
        return max(
            need_requests * 60 / self._rpm if self._rpm else 0.0,
            need_tokens * 60 / self._tpm if self._tpm else 0.0,
        )

    async def acquire(self, tokens: int = 0) -> None:
        """Block until this request fits, then charge it.

        A request larger than the whole token bucket would otherwise wait
        forever, so it is allowed through once the bucket is full and left to
        the vendor to reject - a permanent hang is worse than a 429.
        """
        async with self._lock:
            tokens = min(tokens, int(self._tpm))
            while True:
                self._refill()
                delay = self._wait_for(tokens)
                if delay <= 0:
                    self._requests -= 1
                    self._tokens -= tokens
                    return
                await self._sleep(delay)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How hard to try before giving the turn up.

    `malformed_attempts` is deliberately much lower than `attempts`. A 429 is
    the vendor asking us to wait and will clear; a truncated response is the
    model failing at the task, and re-rolling it four more times spends real
    money to arrive at the same place.
    """

    attempts: int = 5
    malformed_attempts: int = 2
    base_delay: float = 1.0
    max_delay: float = 30.0

    def limit(self, error: ProviderError) -> int:
        return self.malformed_attempts if isinstance(error, Malformed) else self.attempts


DEFAULT_RETRY = RetryPolicy()


async def with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
    sleep: Sleeper = asyncio.sleep,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[ProviderError, int, float], None] | None = None,
) -> T:
    """Run `call`, retrying only what is worth retrying.

    Non-`ProviderError` exceptions propagate untouched: those are bugs in our
    own code, and retrying a bug five times only makes it five times harder to
    find in the log.

    Backoff uses full jitter - a uniform draw from `[0, delay]` rather than
    `delay` plus a wobble. With four agents rejected at the same instant by the
    same limiter, anything less spreads them badly and they collide again on
    the next attempt.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except ProviderError as error:
            attempt += 1
            if not error.retryable or attempt >= policy.limit(error):
                raise
            window = min(policy.max_delay, policy.base_delay * 2 ** (attempt - 1))
            # A server that told us when to come back knows better than we do.
            delay = error.retry_after if error.retry_after is not None else jitter() * window
            if on_retry:
                on_retry(error, attempt, delay)
            await sleep(delay)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class BreakerOpen(ProviderError):
    """The breaker is open, so the request was never sent."""

    retryable = False


class CircuitBreaker:
    """Stop calling a provider that is down, and probe until it is back.

    Retry handles a provider that is busy. This handles one that is *broken*,
    where every turn would otherwise spend its full retry ladder before passing
    anyway - four agents times five attempts times thirty seconds of backoff is
    most of an hour of wall clock spent learning nothing.

    Half-open lets exactly one request through as a probe. Letting several
    through is how a recovering provider gets knocked over again.
    """

    def __init__(
        self,
        *,
        threshold: int = 5,
        cooldown: float = 60.0,
        clock: Clock = time.monotonic,
    ):
        self._threshold = threshold
        self._cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self._cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        state = self.state
        if state == "closed":
            return True
        if state == "half_open" and not self._probing:
            self._probing = True
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None
        self._probing = False

    def record_failure(self) -> None:
        self._probing = False
        if self._opened_at is not None:
            # The probe failed. Start the cooldown again rather than letting a
            # dead provider be probed on every single turn.
            self._opened_at = self._clock()
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = self._clock()

    async def call(self, fn: Callable[[], Awaitable[T]], *, provider: str = "") -> T:
        if not self.allow():
            raise BreakerOpen(
                f"{provider or 'provider'} circuit is open after {self._threshold} "
                f"consecutive failures",
                provider=provider,
            )
        try:
            result = await fn()
        except ProviderError:
            self.record_failure()
            raise
        self.record_success()
        return result
