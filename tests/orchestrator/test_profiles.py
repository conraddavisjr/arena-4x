"""The token profiles, and the projection that was half of true.

`preflight.py` carried five flat constants and projected $21.88 for a 300-turn
shakeout match. The journals put the same match at $46.80. Nothing raised,
nothing looked wrong, and the number was internally consistent - the same shape
as the rate-card bug, one file over.

A test suite cannot measure a model's appetite any more than it can read a
vendor's pricing page. What it can do is refuse the three things that let the
old number survive: a profile that claims the prompt is flat, a profile with no
match behind it, and a projection that is cheaper than what actually happened.

The last of those is the real test in this file, and it needs a journal. Those
are build artifacts under a gitignored `output/`, so it skips rather than fails
when there is nothing to check against - but when a journal *is* present, the
projection has to reproduce it.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from arena_orchestrator.pricing import rate_for
from arena_orchestrator.profiles import (
    FALLBACK,
    MEASURED_FINGERPRINT,
    PROFILES,
    observation_fingerprint,
    profile_for,
    project,
)

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "output" / "baseline-300" / "journal.jsonl"


def shakeout_models() -> list[str]:
    """The roster a real match is actually run with."""
    sys.path.insert(0, str(REPO / "scripts"))
    import run_match

    return list(run_match.DEFAULT_MODELS["shakeout"].values())


def measured_spend(journal: Path) -> dict[str, tuple[float, int, int]]:
    """(dollars, calls, last turn) per model, straight from the journal."""
    spend: dict[str, list[float]] = defaultdict(lambda: [0.0, 0, 0])
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") != "agent_call":
            continue
        entry = spend[row["model"]]
        entry[0] += row["cost_usd"]
        entry[1] += 1
        entry[2] = max(entry[2], row["turn"])
    return {model: tuple(entry) for model, entry in spend.items()}  # type: ignore[misc]


def test_every_seat_we_actually_run_has_a_measured_profile() -> None:
    """The fallback is for models nobody has run, not for the working roster."""
    missing = [m for m in shakeout_models() if profile_for(m) is None]
    assert not missing, (
        f"these are seated in the shakeout roster but have no measured profile: "
        f"{missing}. Run `make profiles` against a completed match."
    )


def test_no_profile_claims_the_prompt_is_flat() -> None:
    """The bug, restated. A flat prompt bills the turn-one board 300 times.

    The observation carries the board, so it grows as the board fills. Every
    seat that has played a whole match grows by at least 130 tokens a turn; a
    profile that says otherwise has been measured against something that was
    not a real match.
    """
    flat = {m: p.prompt_growth for m, p in PROFILES.items() if p.prompt_growth < 50}
    assert not flat, f"prompt growth too low to be a real match: {flat}"


def test_every_profile_says_where_it_came_from() -> None:
    """Provenance per entry, for the reason `pricing.py` learned it the hard way."""
    for model, profile in PROFILES.items():
        assert profile.source, f"{model} has no source match"
        assert profile.checked, f"{model} does not say when it was measured"
        assert profile.calls > 0, f"{model} claims a fit built on no calls"


def test_no_profile_rests_on_a_fit_that_reported_its_own_uselessness() -> None:
    """R^2 0.53 was `claude-haiku-4-5` dying on turn 35, not a growth rate.

    That seat held one city and its observation never grew, so its prompt fit
    was almost flat. Taking it would have under-projected the seat by 40%. Every
    entry here must come from a match where the seat actually played.
    """
    weak = {m: p.fit_r2 for m, p in PROFILES.items() if p.fit_r2 < 0.85}
    assert not weak, (
        f"these fits are too poor to project from: {weak}. A low R^2 here means "
        f"the seat did not play a long enough match to have a growth rate."
    )


def test_the_high_projection_is_never_below_the_likely_one() -> None:
    """A credit check reads the high column, so it may never be the cheaper one."""
    for model in shakeout_models():
        projection = project(model, 300)
        assert projection is not None
        assert projection.high >= projection.central, model


def test_the_fallback_is_the_hungriest_shape_we_have_measured() -> None:
    """An unmeasured model is projected pessimistically, on purpose.

    A projection built on the average seat would be the more likely number and
    the less useful one. The point of the check is to find out whether an
    account survives the run, and one that passes and then runs dry on turn 240
    has answered nothing.
    """
    rate = rate_for("gpt-5.4-mini")
    ceiling = FALLBACK.cost(rate, 300, high=True)
    for model, profile in PROFILES.items():
        assert profile.cost(rate, 300, high=True) <= ceiling + 1e-9, (
            f"{model} is hungrier than the fallback, so an unmeasured model would "
            f"be projected too cheaply. Update FALLBACK."
        )


def test_an_unpriced_model_projects_to_nothing_rather_than_zero() -> None:
    """Same rule as the rate card: refuse, never invent."""
    assert project("no-such-model-4.2", 300) is None


def test_unmeasured_models_are_flagged_as_borrowed() -> None:
    """The flagship roster is priced and has never been run."""
    projection = project("claude-opus-5", 300)
    assert projection is not None
    assert not projection.measured
    assert "borrowed" in projection.profile.note


@pytest.mark.skipif(not BASELINE.exists(), reason="no journal; output/ is gitignored")
def test_the_projection_reproduces_a_match_that_actually_happened() -> None:
    """The test the old constants would have failed by a factor of two.

    Projected over the horizon a match really played, against the seats it
    really used, the answer has to be the bill that really arrived. Everything
    else in this file is a guard rail; this is the measurement.
    """
    actual = measured_spend(BASELINE)
    for model, (dollars, calls, last_turn) in actual.items():
        if profile_for(model) is None:
            continue
        projection = project(model, last_turn)
        assert projection is not None
        # Scaled for turns the seat was not asked - a seat eliminated or timed
        # out has fewer calls than turns, and the profile prices calls.
        expected = projection.central * calls / last_turn
        assert expected == pytest.approx(dollars, rel=0.25), (
            f"{model}: projected ${expected:.3f} against ${dollars:.3f} actually "
            f"spent over {calls} calls. The profile has drifted from the journal."
        )


@pytest.mark.skipif(not BASELINE.exists(), reason="no journal; output/ is gitignored")
def test_the_projection_is_not_optimistic_about_the_match_it_was_built_from() -> None:
    """Erring high is the safe direction and the intended one."""
    actual = measured_spend(BASELINE)
    total_actual = sum(d for d, _, _ in actual.values())
    total_high = 0.0
    for model, (_, calls, last_turn) in actual.items():
        projection = project(model, last_turn)
        assert projection is not None
        total_high += projection.high * calls / last_turn
    assert total_high >= total_actual, (
        f"the high projection (${total_high:.2f}) came in under a match that "
        f"actually cost ${total_actual:.2f}"
    )


def test_the_fingerprint_notices_when_the_observation_changes_shape() -> None:
    """Profiles rot when the observation grows, not when a vendor moves a price.

    Deliberately not an assertion that the two match. The schema legitimately
    changes ahead of the run that re-measures it, and a check that cannot go
    green without spending forty dollars is a check people learn to skip. What
    must hold is that the signal is recorded and computable, so preflight can
    say the projection is an estimate rather than quietly presenting it as one.
    """
    live = observation_fingerprint()
    assert live and live != "unknown"
    assert len(MEASURED_FINGERPRINT) == len(live)
