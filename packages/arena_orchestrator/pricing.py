"""What a match costs, in dollars.

This is the accountant behind the safety halt, so its one hard rule is that it
never guesses. A model with no rate card raises rather than costing zero: a
silent zero would let an unpriced model run to the turn cap while the budget
meter sat at $0.00, which is the exact failure the cap exists to prevent.

**Eight of ten rates in this file were wrong, and the file said so.** An earlier
version carried one blanket comment - "the others are the figures the roster was
costed against and should be re-checked before a flagship run" - and that was
true, accurate, and ignored, including by the person who then quoted the numbers
to two decimal places in a roster recommendation. `grok-4.3` was six times low.
`gpt-5.6` was four times low. Every match reported roughly **half** its real
cost, which meant the $75 safety halt was in practice a $150 halt: the one
number the cap exists to enforce was the one being mis-measured.

So provenance is now per entry rather than per file. Each rate carries the URL it
came from and the date it was read, because "which of these did somebody actually
check" turned out to be the question that mattered, and a single paragraph at the
top could not answer it.

Rates are per million tokens. Re-check with `make prices`; `test_pricing.py`
fails once any entry is older than `STALE_AFTER_DAYS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from .providers.base import Usage

# Vendor pricing pages, kept as constants so an entry cites a source rather than
# asserting one.
ANTHROPIC = "https://platform.claude.com/docs/en/about-claude/pricing"
OPENAI = "https://developers.openai.com/api/docs/models/"
XAI = "https://docs.x.ai/developers/pricing"
GOOGLE = "https://ai.google.dev/gemini-api/docs/pricing"

# How long a rate may go unverified before the suite refuses it. Vendor pricing
# moves - Sonnet 5's introductory $2/$10 became its standard price during this
# project - and a card nobody re-reads is a budget meter nobody can trust.
STALE_AFTER_DAYS = 90


@dataclass(frozen=True, slots=True)
class Rate:
    """Dollars per million tokens, plus the two cache multipliers.

    Cache reads are a tenth of the input rate and cache writes 1.25x, confirmed
    on all four vendors rather than assumed: Anthropic publishes 0.1x reads and
    1.25x five-minute writes, and OpenAI's cached input is exactly a tenth of
    its input rate. That is why the system prefix is worth keeping byte-stable -
    at 300 turns the difference between a hit and a miss on a 6k block is most
    of the input bill.

    `source` and `checked` exist because the previous version of this file had
    neither, and so could not distinguish a rate somebody had verified from one
    somebody had typed. Both are required: a rate with no provenance is a guess
    wearing a decimal point.
    """

    input: float
    output: float
    source: str = ""
    checked: str = ""
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25
    # Several vendors charge more above a prompt-length threshold: xAI doubles
    # above 200k, Google's pro tier steps up above 200k, OpenAI's 5.6 above
    # 272k. Our observations run near 10k so the base tier always applies, and
    # this is recorded rather than modelled - but it is recorded, so the day a
    # match approaches one the number is here instead of being rediscovered.
    tier_note: str = ""

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.input * self.cache_read_multiplier
            + usage.cache_write_tokens * self.input * self.cache_write_multiplier
        ) / 1_000_000

    @property
    def age_days(self) -> int:
        return (datetime.now(UTC).date() - date.fromisoformat(self.checked)).days


CHECKED = "2026-08-17"

RATES: dict[str, Rate] = {
    # -- Anthropic -----------------------------------------------------------
    # The only two entries the previous card got right, and not by luck: they
    # were the two its comment said came from the published rate card.
    "claude-opus-5": Rate(5.00, 25.00, ANTHROPIC, CHECKED),
    # Was 3.00/15.00 here. The $2/$10 launched as introductory pricing through
    # 2026-08-31 and has since been made permanent - the scheduled rise to
    # $3/$15 was cancelled. A rate can move *down* and still make the card wrong.
    "claude-sonnet-5": Rate(2.00, 10.00, ANTHROPIC, CHECKED),
    "claude-haiku-4-5": Rate(1.00, 5.00, ANTHROPIC, CHECKED),
    # -- OpenAI --------------------------------------------------------------
    # Was 1.25/10.00, so flagship projections built on it were four times low.
    "gpt-5.6": Rate(
        5.00,
        30.00,
        OPENAI + "gpt-5.6",
        CHECKED,
        tier_note="2x input and 1.5x output above 272k input tokens",
    ),
    # `gpt-5.6-mini` does not exist; the newest mini is 5.4.
    # Was 0.25/2.00 - three times low on input, more than twice on output.
    "gpt-5.4-mini": Rate(0.75, 4.50, OPENAI + "gpt-5.4", CHECKED),
    # -- Google --------------------------------------------------------------
    # `gemini-3.6-pro` was in this table and does not exist - the API 404s it.
    # There is no 3.6 or 3.7 in the pro tier; 3.1 is the newest and a preview.
    "gemini-3.1-pro-preview": Rate(
        2.00, 12.00, GOOGLE, CHECKED, tier_note="$4/$18 above 200k prompt tokens"
    ),
    # Both flash tiers carried flash-lite's price. Note these are themselves
    # promotional: $0.75/$3.75 holds through 2026-12-31 and then doubles.
    "gemini-3.7-flash": Rate(
        0.75, 3.75, GOOGLE, CHECKED, tier_note="rises to $1.50/$7.50 on 2027-01-01"
    ),
    "gemini-3.6-flash": Rate(
        0.75, 3.75, GOOGLE, CHECKED, tier_note="rises to $1.50/$7.50 on 2027-01-01"
    ),
    # Was 0.10/0.40. Worth noting a second model got this one wrong in the other
    # direction, quoting $0.54/$4.50 while arguing our figure was impossible.
    "gemini-3.5-flash-lite": Rate(0.30, 2.50, GOOGLE, CHECKED),
    # -- xAI -----------------------------------------------------------------
    # `grok-4` does not exist; the live list is 4.3 / 4.5 / 4.6.
    "grok-4.6": Rate(2.00, 6.00, XAI, CHECKED, tier_note="doubles above 200k prompt tokens"),
    "grok-4.5": Rate(2.00, 6.00, XAI, CHECKED, tier_note="doubles above 200k prompt tokens"),
    # Was 0.20/0.50, six times low on input and five on output - the worst entry
    # in the file, and the seat that looked cheapest partly because of it.
    "grok-4.3": Rate(1.25, 2.50, XAI, CHECKED, tier_note="doubles above 200k prompt tokens"),
}


class UnknownModel(KeyError):
    """Raised rather than pricing an unrecognised model at zero."""

    def __init__(self, model: str):
        super().__init__(model)
        self.model = model

    def __str__(self) -> str:
        return (
            f"no rate card for {self.model!r}. Add it to arena_orchestrator.pricing.RATES "
            f"before running a match with it - an unpriced model would spend against a "
            f"budget meter that never moves."
        )


def rate_for(model: str) -> Rate:
    try:
        return RATES[model]
    except KeyError:
        raise UnknownModel(model) from None


def cost_of(model: str, usage: Usage) -> float:
    """Dollars for one response. Raises `UnknownModel` if the model is unpriced."""
    return rate_for(model).cost(usage)
