"""Pricing, the safety halt, and the in-fiction allowance.

The one property worth protecting above all others: an unpriced model must
never cost zero. A silent zero lets a misconfigured roster run to the turn cap
while the budget meter sits at $0.00, which is the exact failure the cap exists
to prevent - and it would only be discovered on the invoice.
"""

from __future__ import annotations

import pytest

from arena_orchestrator.budget import Allowance, Ledger
from arena_orchestrator.pricing import RATES, Rate, UnknownModel, cost_of
from arena_orchestrator.providers.base import Usage


def test_an_unpriced_model_raises_rather_than_costing_nothing() -> None:
    with pytest.raises(UnknownModel) as caught:
        cost_of("gpt-9-turbo-ultra", Usage(input_tokens=1_000_000))
    assert "no rate card" in str(caught.value)


def test_every_model_in_the_rate_card_prices() -> None:
    for model in RATES:
        assert cost_of(model, Usage(input_tokens=1000, output_tokens=1000)) > 0


def test_cached_reads_are_a_tenth_of_fresh_input() -> None:
    """The whole reason the system prefix is kept byte-stable. At 250 turns the
    difference between a hit and a miss on a 6k block is most of the input
    bill."""
    rate = Rate(input=5.0, output=25.0)
    fresh = rate.cost(Usage(input_tokens=1_000_000))
    cached = rate.cost(Usage(cache_read_tokens=1_000_000))
    assert fresh == pytest.approx(5.0)
    assert cached == pytest.approx(0.5)


def test_cache_writes_cost_more_than_fresh_input() -> None:
    rate = Rate(input=5.0, output=25.0)
    assert rate.cost(Usage(cache_write_tokens=1_000_000)) == pytest.approx(6.25)


def test_reasoning_tokens_are_not_billed_twice() -> None:
    """OpenAI reports reasoning tokens separately *and* inside the output count.
    Adding them would charge for the same tokens twice and halt the match
    early."""
    rate = Rate(input=1.0, output=10.0)
    plain = rate.cost(Usage(output_tokens=1000))
    with_reasoning = rate.cost(Usage(output_tokens=1000, reasoning_tokens=800))
    assert plain == with_reasoning


# ---------------------------------------------------------------------------
# The dollar cap
# ---------------------------------------------------------------------------


def test_the_ledger_totals_what_was_actually_reported() -> None:
    ledger = Ledger(cap_usd=75.0)
    ledger.charge("p1", "claude-opus-5", Usage(input_tokens=6000, output_tokens=1500))
    ledger.charge("p2", "claude-opus-5", Usage(input_tokens=6000, output_tokens=1500))
    # 6k at $5/M plus 1.5k at $25/M is $0.0675 a turn.
    assert ledger.spent_usd == pytest.approx(0.135)
    assert ledger.by_agent["p1"] == pytest.approx(0.0675)
    assert ledger.requests == 2


def test_the_cap_halts_the_match() -> None:
    ledger = Ledger(cap_usd=0.10)
    assert not ledger.exhausted
    ledger.charge("p1", "claude-opus-5", Usage(output_tokens=5000))
    assert ledger.exhausted
    assert ledger.remaining_usd == 0.0


def test_a_misconfigured_roster_cannot_spend_silently() -> None:
    """The halt is only as good as the meter behind it."""
    ledger = Ledger(cap_usd=75.0)
    with pytest.raises(UnknownModel):
        ledger.charge("p1", "not-a-model", Usage(output_tokens=100_000))
    assert ledger.spent_usd == 0.0


def test_usage_accumulates_per_agent_across_turns() -> None:
    ledger = Ledger(cap_usd=75.0)
    for _ in range(3):
        ledger.charge("p1", "claude-haiku-4-5", Usage(input_tokens=100, output_tokens=50))
    assert ledger.usage_by_agent["p1"].output_tokens == 150
    assert ledger.usage_by_agent["p1"].total == 450


def test_score_per_100k_compares_a_cheap_model_against_an_expensive_one() -> None:
    """The closest thing this experiment has to a headline result."""
    ledger = Ledger(cap_usd=75.0)
    ledger.charge("p1", "claude-opus-5", Usage(input_tokens=100_000))
    assert ledger.score_per_100k("p1", 400) == pytest.approx(400.0)
    assert ledger.score_per_100k("p9", 400) == 0.0  # never played, no divide by zero


# ---------------------------------------------------------------------------
# The in-fiction allowance
# ---------------------------------------------------------------------------


def test_the_allowance_counts_output_only() -> None:
    """Input is dominated by the observation, which the engine writes and the
    agent cannot control. Charging for it would measure the size of the board
    rather than the agent's restraint."""
    allowance = Allowance(per_agent_tokens=1000)
    allowance.charge("p1", Usage(input_tokens=50_000, output_tokens=200))
    assert allowance.remaining("p1") == 800


def test_running_out_is_an_in_game_consequence() -> None:
    allowance = Allowance(per_agent_tokens=100)
    allowance.charge("p1", Usage(output_tokens=140))
    assert allowance.exhausted("p1")
    assert allowance.remaining("p1") == 0  # never negative
    assert not allowance.exhausted("p2")


def test_the_allowance_is_symmetric_across_vendors() -> None:
    """Deliberately tokens, not dollars: per-token prices differ by more than an
    order of magnitude across these four, so a shared dollar allowance would
    partly measure who got the cheaper contract rather than who reasons
    better."""
    allowance = Allowance(per_agent_tokens=400_000)
    for player in ("p1", "p2", "p3", "p4"):
        allowance.charge(player, Usage(output_tokens=1500))
    assert len({allowance.remaining(p) for p in ("p1", "p2", "p3", "p4")}) == 1


def test_the_observation_block_shape() -> None:
    allowance = Allowance(per_agent_tokens=400_000)
    allowance.charge("p1", Usage(output_tokens=71_400))
    block = allowance.observation_block("p1", turn=47, turn_limit=300)
    assert block == {
        "tokens_allowance": 400_000,
        "tokens_spent": 71_400,
        "tokens_remaining": 328_600,
        "match_pct_elapsed": 15.7,
    }
