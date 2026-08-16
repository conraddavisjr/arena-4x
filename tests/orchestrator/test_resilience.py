"""Rate limiting, retry and circuit breaking.

None of this can be tested against a live vendor - you cannot ask a provider to
have an outage on demand, and a test that waits out a sixty-second cooldown is a
test nobody runs. So every mechanism takes its clock, sleep and randomness as
arguments, and these tests drive all three.

What is actually being asserted is *behaviour under failure*, not the arithmetic
of a backoff curve: a bug in a retry policy shows up as a match that dies at 3am
on day two, and by then the only evidence is the absence of turns.
"""

from __future__ import annotations

import pytest

from arena_orchestrator.providers.base import (
    FatalProviderError,
    Malformed,
    Overloaded,
    ProviderError,
    RateLimited,
)
from arena_orchestrator.resilience import (
    BreakerOpen,
    CircuitBreaker,
    RetryPolicy,
    TokenBucket,
    with_retry,
)


class FakeTime:
    """A clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


async def test_a_transient_failure_is_retried_and_succeeds() -> None:
    clock = FakeTime()
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Overloaded("529")
        return "ok"

    result = await with_retry(call, sleep=clock.sleep, jitter=lambda: 1.0)
    assert result == "ok"
    assert attempts == 3


async def test_a_fatal_error_is_not_retried() -> None:
    """A bad API key does not get better on the fifth attempt, and retrying it
    four more times only delays the operator finding out."""
    clock = FakeTime()
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        raise FatalProviderError("401 unauthorized")

    with pytest.raises(FatalProviderError):
        await with_retry(call, sleep=clock.sleep)
    assert attempts == 1
    assert clock.slept == []


async def test_a_bug_in_our_own_code_is_not_retried() -> None:
    """Only ProviderError is transport failure. Anything else is ours, and
    retrying it five times makes it five times harder to find in the log."""
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("off-by-one")

    with pytest.raises(ValueError):
        await with_retry(call, sleep=FakeTime().sleep)
    assert attempts == 1


async def test_backoff_grows_and_is_capped() -> None:
    clock = FakeTime()

    async def call() -> str:
        raise Overloaded("529")

    with pytest.raises(Overloaded):
        await with_retry(
            call,
            policy=RetryPolicy(attempts=6, base_delay=1.0, max_delay=8.0),
            sleep=clock.sleep,
            jitter=lambda: 1.0,  # full jitter at its maximum draw
        )
    assert clock.slept == [1.0, 2.0, 4.0, 8.0, 8.0]


async def test_the_servers_retry_after_wins_over_our_curve() -> None:
    """A server that told us when to come back knows better than we do."""
    clock = FakeTime()

    async def call() -> str:
        raise RateLimited("429", retry_after=12.5)

    with pytest.raises(RateLimited):
        await with_retry(call, policy=RetryPolicy(attempts=2), sleep=clock.sleep)
    assert clock.slept == [12.5]


async def test_full_jitter_can_draw_the_whole_window() -> None:
    """Full jitter draws from [0, window]. With four agents rejected by the same
    limiter in the same instant, a fixed delay would collide them again."""
    clock = FakeTime()
    draws = iter([0.0, 0.5, 1.0, 0.25])

    async def call() -> str:
        raise Overloaded("529")

    with pytest.raises(Overloaded):
        await with_retry(
            call,
            policy=RetryPolicy(attempts=5, base_delay=4.0, max_delay=100.0),
            sleep=clock.sleep,
            jitter=lambda: next(draws),
        )
    assert clock.slept == [0.0, 4.0, 16.0, 8.0]


async def test_a_malformed_body_gives_up_far_sooner_than_a_rate_limit() -> None:
    """A 429 clears. A truncated response is the model failing at the task, and
    re-rolling it four more times spends real money to arrive nowhere."""
    clock = FakeTime()
    attempts = 0

    async def call() -> str:
        nonlocal attempts
        attempts += 1
        raise Malformed("truncated mid-JSON")

    with pytest.raises(Malformed):
        await with_retry(
            call,
            policy=RetryPolicy(attempts=5, malformed_attempts=2),
            sleep=clock.sleep,
            jitter=lambda: 1.0,
        )
    assert attempts == 2


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


async def test_the_breaker_opens_after_the_threshold_and_stops_calling() -> None:
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=3, cooldown=60, clock=clock)
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise Overloaded("529")

    for _ in range(3):
        with pytest.raises(Overloaded):
            await breaker.call(call, provider="anthropic")
    assert breaker.state == "open"

    # The fourth attempt never reaches the provider.
    with pytest.raises(BreakerOpen):
        await breaker.call(call, provider="anthropic")
    assert calls == 3


async def test_the_breaker_half_opens_and_recovers() -> None:
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=2, cooldown=60, clock=clock)

    async def fail() -> str:
        raise Overloaded("529")

    async def succeed() -> str:
        return "ok"

    for _ in range(2):
        with pytest.raises(Overloaded):
            await breaker.call(fail)
    assert breaker.state == "open"

    clock.now += 60
    assert breaker.state == "half_open"
    assert await breaker.call(succeed) == "ok"
    assert breaker.state == "closed"


async def test_only_one_probe_goes_through_while_half_open() -> None:
    """Releasing the whole backlog at a recovering provider is how it gets
    knocked over a second time."""
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=1, cooldown=30, clock=clock)

    async def fail() -> str:
        raise Overloaded("529")

    with pytest.raises(Overloaded):
        await breaker.call(fail)
    clock.now += 30
    assert breaker.allow() is True
    assert breaker.allow() is False


async def test_a_failed_probe_restarts_the_cooldown() -> None:
    """Otherwise a dead provider gets probed on every single turn for the rest
    of the match."""
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=1, cooldown=30, clock=clock)

    async def fail() -> str:
        raise Overloaded("529")

    with pytest.raises(Overloaded):
        await breaker.call(fail)
    clock.now += 30
    with pytest.raises(Overloaded):
        await breaker.call(fail)
    assert breaker.state == "open"


async def test_a_success_clears_accumulated_failures() -> None:
    """Failures have to be *consecutive*. Three scattered 529s over two hundred
    turns are a normal day, not an outage."""
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=3, cooldown=60, clock=clock)

    async def fail() -> str:
        raise Overloaded("529")

    async def succeed() -> str:
        return "ok"

    for _ in range(2):
        with pytest.raises(Overloaded):
            await breaker.call(fail)
    await breaker.call(succeed)
    for _ in range(2):
        with pytest.raises(Overloaded):
            await breaker.call(fail)
    assert breaker.state == "closed"


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


async def test_the_bucket_lets_a_burst_through_then_throttles() -> None:
    clock = FakeTime()
    bucket = TokenBucket(60, 1_000_000, clock=clock, sleep=clock.sleep)

    for _ in range(60):
        await bucket.acquire(0)
    assert clock.slept == []

    await bucket.acquire(0)
    assert clock.slept == [pytest.approx(1.0)]


async def test_the_token_limit_binds_independently_of_the_request_limit() -> None:
    """Late in a match the observation grows and the token limit starts binding
    first, which is why these are two buckets and not one."""
    clock = FakeTime()
    bucket = TokenBucket(1000, 60_000, clock=clock, sleep=clock.sleep)

    await bucket.acquire(60_000)
    assert clock.slept == []
    await bucket.acquire(30_000)
    assert clock.slept == [pytest.approx(30.0)]


async def test_refill_is_continuous_rather_than_windowed() -> None:
    """A fixed window lets four agents through together at the boundary and then
    stalls them - which is precisely the burst the vendor's limiter punishes."""
    clock = FakeTime()
    bucket = TokenBucket(60, 1_000_000, clock=clock, sleep=clock.sleep)

    for _ in range(60):
        await bucket.acquire(0)
    clock.now += 10  # ten seconds at 60/min is ten requests back
    for _ in range(10):
        await bucket.acquire(0)
    assert clock.slept == []


async def test_an_oversized_request_is_not_a_permanent_hang() -> None:
    """A request bigger than the whole bucket can never fit. Waiting forever is
    worse than letting the vendor reject it."""
    clock = FakeTime()
    bucket = TokenBucket(1000, 1000, clock=clock, sleep=clock.sleep)
    await bucket.acquire(50_000)
    assert clock.slept == []


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


async def test_retry_and_breaker_compose_into_a_pass_rather_than_a_crash() -> None:
    """The property the whole match depends on: an agent that cannot answer
    passes its turn and plays badly. It never takes the match down with it."""
    clock = FakeTime()
    breaker = CircuitBreaker(threshold=2, cooldown=60, clock=clock)

    async def dead() -> str:
        raise Overloaded("provider is down")

    async def take_turn() -> str | None:
        try:
            return await breaker.call(
                lambda: with_retry(
                    dead,
                    policy=RetryPolicy(attempts=3),
                    sleep=clock.sleep,
                    jitter=lambda: 1.0,
                )
            )
        except ProviderError:
            return None  # pass_turn

    assert await take_turn() is None
    assert await take_turn() is None
    assert breaker.state == "open"
    # Now costing nothing: the breaker short-circuits before any HTTP happens.
    before = len(clock.slept)
    assert await take_turn() is None
    assert len(clock.slept) == before


async def test_the_bucket_records_how_long_it_made_callers_wait() -> None:
    """Throttling is invisible otherwise. It produces no errors and no retries -
    it just makes a match take longer, and "the run is slow" is not a diagnosis
    you can act on. The accumulator is what the loop journals per turn."""
    clock = FakeTime()
    bucket = TokenBucket(60, 1_000_000, clock=clock, sleep=clock.sleep)

    for _ in range(60):
        await bucket.acquire(0)
    assert bucket.waited_s == 0.0

    await bucket.acquire(0)
    await bucket.acquire(0)
    assert bucket.waited_s == pytest.approx(2.0)
    assert bucket.waited_s == pytest.approx(sum(clock.slept))
