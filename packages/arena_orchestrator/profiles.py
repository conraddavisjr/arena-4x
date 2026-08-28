"""How many tokens a seat actually spends, as a function of the turn number.

The companion to `pricing.py`. That file answers "what does a token cost"; this
one answers "how many tokens will there be", and a projection needs both. It
existed as five constants inside `preflight.py`:

    PER_TURN = {"input": 2_000, "output": 4_000, "cached": 6_500}

and multiplying that by the turn count projected **$21.88** for a 300-turn
shakeout match. Measured against the journals, the same match is **$46.80**.

Two independent errors, both in the same direction:

**Output was averaged across seats that differ by seven times.** `gpt-5.4-mini`
emits about 9,200 output tokens a call and `gemini-3.6-flash` about 1,270. One
figure of 4,000 charges the cheap seat too much and the expensive seat - which
was 57% of the entire bill - less than half of what it costs.

**The prompt was treated as a constant, and it is a ramp.** The observation
carries the board, so it grows as the board fills: `gpt-5.4-mini`'s prompt went
from 8k tokens on turn 1 to 30k by turn 128, climbing about 200 tokens a turn
with an R^2 of 0.997. A flat profile multiplied by 300 bills the turn-one board
three hundred times, and the error compounds exactly where the risk is - late in
a long run, which is the only part a 128-turn match never reached.

This is the same shape as the rate-card bug in `pricing.py`: nothing raised,
nothing looked wrong, and the number was about half of true. So the same
defences apply. Every profile carries the match it was measured from, the number
of calls behind it and the fit quality, because a growth rate with no provenance
is a guess wearing three significant figures. Re-derive with `make profiles`
after any completed match.

**The projection is a range, not a number.** `output` is the central estimate
and `output_high` is the worst per-call mean any completed match has shown,
which for `gpt-5.4-mini` is more than twice its central figure. A credit check
wants the ceiling, because the question is not "what will this probably cost"
but "can the account cover it if it goes badly".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .pricing import Rate, rate_for

# Where the numbers came from. Named rather than inlined so an entry cites a run
# the way a rate cites a vendor page.
BASELINE = "baseline-300 (128 turns, one call per seat per turn)"
SHAKEOUT = "shakeout-300 (51 turns)"

# `shakeout-300` ran before `8cc9608` normalised reasoning effort across seats,
# so its dollar figures are not comparable and its output counts are not used.
# Prompt growth is exempt from that: the observation's size is a fact about how
# many cities and units are on the board, which no effort setting changes. That
# is the only figure taken from it, and only for the one seat whose baseline fit
# is unusable.
MEASURED = "2026-08-18"


@dataclass(frozen=True, slots=True)
class Profile:
    """One seat's token appetite, fitted against a completed match.

    The prompt is modelled as a line and the output as a constant, which is what
    the data supports rather than what is convenient. Prompt growth fits with an
    R^2 between 0.92 and 0.997 across every seat measured. Output fits a line
    with an R^2 between 0.004 and 0.375, which is another way of saying it does
    not trend at all - it is the model's own verbosity, and the board getting
    bigger does not make a model wordier.

    Cache is held as a fraction of the prompt rather than an absolute count for
    the same reason: what is cached is a slice of the observation, so it scales
    with the observation.
    """

    prompt_base: float
    prompt_growth: float
    output: float
    output_high: float
    cache_read_frac: float
    cache_write_frac: float
    source: str
    checked: str
    calls: int
    fit_r2: float
    note: str = ""

    def tokens_at(self, turn: int) -> tuple[float, float, float, float]:
        """(input, output, cache_read, cache_write) for one call on this turn."""
        prompt = max(0.0, self.prompt_base + self.prompt_growth * turn)
        read = prompt * self.cache_read_frac
        write = prompt * self.cache_write_frac
        return prompt - read - write, self.output, read, write

    def cost(self, rate: Rate, turns: int, *, high: bool = False) -> float:
        """Dollars for turns 1..turns, summed rather than averaged.

        Summed because the prompt is a ramp: the mean turn and the mean cost
        coincide only for a straight line through the origin, and this line has
        an intercept. At 300 turns the difference is small; the habit of not
        assuming it away is not.
        """
        output = self.output_high if high else self.output
        total = 0.0
        for turn in range(1, turns + 1):
            inp, _, read, write = self.tokens_at(turn)
            total += (
                inp * rate.input
                + output * rate.output
                + read * rate.input * rate.cache_read_multiplier
                + write * rate.input * rate.cache_write_multiplier
            ) / 1_000_000
        return total


PROFILES: dict[str, Profile] = {
    # -- Anthropic -----------------------------------------------------------
    # The one entry built from two matches, and the reason the derive script
    # prints R^2. In `baseline-300` this seat was eliminated on turn 35 holding
    # a single city, so its prompt barely grew: 61 tokens a turn at R^2 0.53,
    # which is the fit reporting its own uselessness. `shakeout-300` ran it 51
    # turns as a going concern and gives 142 a turn at R^2 0.92, in line with
    # the three seats that played whole matches. Taking the 61 would have
    # under-projected this seat by 40%.
    "claude-haiku-4-5": Profile(
        prompt_base=7980,
        prompt_growth=141.8,
        output=3416,
        output_high=3684,
        cache_read_frac=0.405,
        cache_write_frac=0.034,
        source=f"{SHAKEOUT} for growth, {BASELINE} for output",
        checked=MEASURED,
        calls=86,
        fit_r2=0.923,
        note=(
            "Both measurements predate e98541a, which took this seat's extended "
            "thinking away in exchange for order enforcement. Output is "
            "therefore an over-estimate, by an unknown amount, until a match "
            "runs on the current dialect."
        ),
    ),
    # -- OpenAI --------------------------------------------------------------
    # The seat that decides whether a run fits in its budget: 57% of the bill in
    # `baseline-300`, and the widest spread between matches of any seat. Its
    # central and high output differ by 2.2x, which is most of the gap between
    # the $47 and $62 ends of a 300-turn projection.
    "gpt-5.4-mini": Profile(
        prompt_base=6134,
        prompt_growth=200.1,
        output=9197,
        output_high=20260,
        cache_read_frac=0.186,
        cache_write_frac=0.0,
        source=BASELINE,
        checked=MEASURED,
        calls=128,
        fit_r2=0.997,
        note="output_high is the per-call mean from shakeout-300, 2.2x the central figure",
    ),
    # -- Google --------------------------------------------------------------
    # Reasons at length and answers briefly: 3,000 reasoning tokens behind 1,270
    # output. Almost no cache read, which is worth a look before a long run -
    # at 48k prompt tokens by turn 300 the misses are not free.
    "gemini-3.6-flash": Profile(
        prompt_base=4386,
        prompt_growth=144.4,
        output=1270,
        output_high=1270,
        cache_read_frac=0.040,
        cache_write_frac=0.0,
        source=BASELINE,
        checked=MEASURED,
        calls=128,
        fit_r2=0.953,
        note="no completed match has shown a higher per-call mean, so high == central",
    ),
    # -- xAI -----------------------------------------------------------------
    "grok-4.3": Profile(
        prompt_base=5934,
        prompt_growth=134.2,
        output=1871,
        output_high=1871,
        cache_read_frac=0.169,
        cache_write_frac=0.0,
        source=BASELINE,
        checked=MEASURED,
        calls=128,
        fit_r2=0.976,
        note="no completed match has shown a higher per-call mean, so high == central",
    ),
}

# Every model in the flagship roster has a rate and no profile: it is priced and
# has never been run. Projecting it needs a shape from somewhere, and the choice
# of which shape is a judgement about which way to be wrong.
#
# This takes the hungriest measured seat. A projection built on the average seat
# would be the more likely number and the less useful one - the reason to run
# this at all is to find out whether an account can cover the run, and a check
# that passes and then runs out of credit on turn 240 has answered nothing. It
# is labelled as unmeasured wherever it is used, because a fallback that reads
# like a measurement is how the old constants survived as long as they did.
FALLBACK = Profile(
    prompt_base=6134,
    prompt_growth=200.1,
    output=9197,
    output_high=20260,
    cache_read_frac=0.186,
    cache_write_frac=0.0,
    source=f"fallback: the shape of gpt-5.4-mini, the hungriest seat in {BASELINE}",
    checked=MEASURED,
    calls=0,
    fit_r2=0.997,
    note="this model has never been run; the shape is borrowed, only the rate is its own",
)


def observation_fingerprint() -> str:
    """A short hash of the observation schema, as a staleness signal.

    Profiles go stale when the observation changes shape, not when a vendor
    changes a price - `9c679b1` added the science race and named rivals to every
    observation, and every token of that lands in the prompt on every turn for
    every seat. So the thing to watch is the schema file, and the honest way to
    watch it is to record what it looked like when the numbers were taken.

    Deliberately a warning rather than a test failure. The schema legitimately
    changes ahead of the run that re-measures it, and a check that cannot be
    made green without spending forty dollars is a check people learn to skip.
    """
    schema = Path(__file__).resolve().parents[2] / "schemas" / "observation.schema.json"
    if not schema.exists():
        return "unknown"
    return hashlib.sha256(schema.read_bytes()).hexdigest()[:12]


# What `observation_fingerprint()` returned when the profiles above were taken,
# which is the schema as of 6564296 - the commit the measured matches ran on.
# If the live value differs, the observation has changed shape since and every
# projection here is an estimate of unknown tightness. It already does differ:
# 9c679b1 added the science race and named rivals, taking the schema from 20,055
# bytes to 21,708, so the current numbers are a lower bound.
MEASURED_FINGERPRINT = "60dbef5ab7cc"


@dataclass(frozen=True, slots=True)
class Projection:
    """A cost range for one seat over a whole match, and how much to trust it."""

    model: str
    central: float
    high: float
    measured: bool
    profile: Profile

    @property
    def basis(self) -> str:
        return self.profile.source


def profile_for(model: str) -> Profile | None:
    """The measured profile for a model, or None if it has never been run."""
    return PROFILES.get(model)


def project(model: str, turns: int) -> Projection | None:
    """Projected spend for one seat over `turns` turns.

    Returns None for a model with no rate card, which is the one case that must
    not be guessed at: `pricing.rate_for` raises there for the same reason, and
    a projection that invented a price would be worse than no projection.
    """
    try:
        rate = rate_for(model)
    except Exception:
        return None
    profile = profile_for(model)
    measured = profile is not None
    profile = profile or FALLBACK
    return Projection(
        model=model,
        central=profile.cost(rate, turns),
        high=profile.cost(rate, turns, high=True),
        measured=measured,
        profile=profile,
    )
