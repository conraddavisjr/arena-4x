"""What a match costs, in dollars.

This is the accountant behind the safety halt, so its one hard rule is that it
never guesses. A model with no rate card raises rather than costing zero: a
silent zero would let an unpriced model run to the turn cap while the budget
meter sat at $0.00, which is the exact failure the cap exists to prevent.

Rates are per million tokens. The Anthropic figures are from the vendor rate
card; the others are recorded here as the single place to correct them, and are
marked with the date they were last checked, because these move.
"""

from __future__ import annotations

from dataclasses import dataclass

from .providers.base import Usage


@dataclass(frozen=True, slots=True)
class Rate:
    """Dollars per million tokens, plus the two cache multipliers.

    Cache reads are roughly a tenth of the input rate and cache writes roughly
    1.25x, which is why the system prefix is worth keeping byte-stable: at
    250 turns the difference between a hit and a miss on a 6k block is most of
    the input bill.
    """

    input: float
    output: float
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.input * self.cache_read_multiplier
            + usage.cache_write_tokens * self.input * self.cache_write_multiplier
        ) / 1_000_000


# Checked 2026-08-14. Anthropic rates are from the published card; the other
# three are the figures the roster was costed against and should be re-checked
# before a flagship run, since vendor pricing changes more often than this file.
RATES: dict[str, Rate] = {
    # Anthropic
    "claude-opus-5": Rate(input=5.00, output=25.00),
    "claude-sonnet-5": Rate(input=3.00, output=15.00),
    "claude-haiku-4-5": Rate(input=1.00, output=5.00),
    # OpenAI
    "gpt-5.6": Rate(input=1.25, output=10.00),
    "gpt-5.6-mini": Rate(input=0.25, output=2.00),
    # Google. `gemini-3.6-pro` was in this table and does not exist - the API
    # 404s it and suggests gemini-2.5-pro. There is no 3.6 or 3.7 in the pro
    # tier at all; 3.1 is the newest, and it is still a preview.
    "gemini-3.1-pro-preview": Rate(input=1.25, output=10.00),
    "gemini-3.7-flash": Rate(input=0.30, output=2.50),
    "gemini-3.6-flash": Rate(input=0.30, output=2.50),
    "gemini-3.5-flash-lite": Rate(input=0.10, output=0.40),
    # xAI
    "grok-4": Rate(input=3.00, output=15.00),
    "grok-4-fast": Rate(input=0.20, output=0.50),
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
