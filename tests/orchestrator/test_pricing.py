"""The rate card, and the reasons it went wrong.

Eight of ten rates in this file were incorrect for weeks - `grok-4.3` by six
times, `gpt-5.6` by four - so every match reported roughly half its true cost and
the $75 safety halt was in practice a $150 halt. Nothing caught it, because
nothing could: the numbers were internally consistent, the arithmetic was right,
and the only thing wrong was the data.

These tests cannot verify a price against a vendor - that needs a human reading a
pricing page, and `make prices` is the prompt to do it. What they can do is
refuse to let the card rot silently again, and refuse to let a rate be added
without saying where it came from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arena_orchestrator.pricing import RATES, STALE_AFTER_DAYS, UnknownModel, cost_of
from arena_orchestrator.providers.base import Usage


def rostered_models() -> set[str]:
    """Every model any roster can actually seat."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_match

    return {model for roster in run_match.DEFAULT_MODELS.values() for model in roster.values()}


def test_every_rostered_model_is_priced() -> None:
    """An unpriced model would spend against a meter that never moves."""
    missing = sorted(rostered_models() - set(RATES))
    assert not missing, f"these can be seated but not priced: {missing}"


def test_no_rate_is_stale() -> None:
    """The failure this file exists to prevent, restated as a test.

    Vendor pricing moves in both directions - Sonnet 5's introductory $2/$10
    became permanent while this project was running, which made the card wrong
    by being too *high*. A rate nobody has re-read is not a fact, and after
    `STALE_AFTER_DAYS` this suite stops pretending otherwise.
    """
    stale = {m: r.age_days for m, r in RATES.items() if r.age_days > STALE_AFTER_DAYS}
    assert not stale, (
        f"unverified for more than {STALE_AFTER_DAYS} days: {stale}. "
        f"Re-read the vendor pages, update the rate and its `checked` date."
    )


def test_every_rate_says_where_it_came_from() -> None:
    """Provenance per entry, because one blanket comment did not work.

    The previous card carried a note saying most of its rates were unverified
    and should be re-checked. That note was accurate and was ignored - by me,
    while quoting the numbers to two decimal places. Per-entry provenance makes
    "did anyone actually check this one" answerable rather than a matter of
    remembering to read the header.
    """
    for model, rate in RATES.items():
        assert rate.source.startswith("https://"), f"{model} cites no source"
        assert rate.checked, f"{model} has no check date"


def test_an_unpriced_model_raises_rather_than_costing_nothing() -> None:
    with pytest.raises(UnknownModel) as caught:
        cost_of("claude-imaginary-9", Usage(input_tokens=1000))
    assert "budget meter that never moves" in str(caught.value)


def test_cache_reads_are_a_tenth_and_writes_are_1_25x() -> None:
    """Confirmed against vendor documentation rather than assumed.

    Anthropic publishes 0.1x reads and 1.25x five-minute writes; OpenAI's cached
    input is exactly a tenth of its input rate. This is the multiplier that makes
    a byte-stable system prefix worth the trouble, so it is worth pinning.
    """
    priced = cost_of("claude-haiku-4-5", Usage(cache_read_tokens=1_000_000))
    assert priced == pytest.approx(0.10), "a cache read should be a tenth of $1.00 input"
    written = cost_of("claude-haiku-4-5", Usage(cache_write_tokens=1_000_000))
    assert written == pytest.approx(1.25)


def test_the_seat_that_was_six_times_wrong() -> None:
    """A regression pin on the single worst entry.

    `grok-4.3` was carried at $0.20/$0.50 against a real $1.25/$2.50, which made
    that seat look like the cheapest on the board by a distance and skewed a
    roster recommendation built on it. Pinned so a future edit that reintroduces
    a plausible-looking cheap rate has to argue with a test.
    """
    rate = RATES["grok-4.3"]
    assert (rate.input, rate.output) == (1.25, 2.50)
    assert "200k" in rate.tier_note, "the tier break above 200k prompt tokens must stay recorded"


# ---------------------------------------------------------------------------
# Credit exhaustion, recognised by wording
# ---------------------------------------------------------------------------


def test_every_observed_credit_message_is_recognised() -> None:
    """Each of these was seen on a live account, and one was missed.

    Google reports an exhausted account as a **429**, so before its wording was
    added it classified as `RateLimited` - retryable - and a match would have
    spun the retry ladder on a condition that cannot improve rather than
    halting. The preflight blocked the run anyway because any failure blocks,
    but that was luck rather than design: mid-run there is no preflight.

    Found by the model-action suite rather than by a 300-turn match, which is
    the cheap way round and the reason that suite exists.
    """
    from arena_orchestrator.providers.base import OutOfCredits

    seen = [
        "You have no credits remaining. Add credits to continue using the API",  # OpenAI
        "Your credit balance is too low to access the Anthropic API",  # Anthropic
        "Your prepayment credits are depleted. Please go to AI Studio",  # Google, as a 429
    ]
    for message in seen:
        assert OutOfCredits.matches(message), f"not recognised: {message[:40]}"


def test_an_ordinary_rate_limit_is_not_mistaken_for_bankruptcy() -> None:
    """The expensive direction of this mistake.

    Matching "quota" or "billing" loosely would catch routine throttling and end
    a multi-day match over a bad ten minutes - causing exactly the failure this
    class exists to prevent, in the other direction.
    """
    from arena_orchestrator.providers.base import OutOfCredits

    for message in (
        "Rate limit reached for requests. Please try again in 20s",
        "429 Too Many Requests",
        "overloaded_error: the fleet is busy",
    ):
        assert not OutOfCredits.matches(message), f"false positive: {message}"
