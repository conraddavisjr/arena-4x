"""Two budgets that must never be confused with each other.

**The dollar cap is out-of-fiction.** It is a safety halt on an unattended
multi-day run: when the match has spent its cap it stops, scores the position,
and emits `match_ended` with `reason: "budget_cap"`. The agents never see it and
it is not part of the game.

**The token allowance is in-fiction.** It is an optional experiment
(`agent_budget_awareness`): every agent gets the same output-token allowance,
sees what it has left in its observation, and falls back to passing when it runs
out. That turns token efficiency into a strategy rather than an operational
concern, and it is the thing being measured.

They are kept apart deliberately. The allowance is counted in **tokens, not
dollars**, because per-token prices differ by more than an order of magnitude
across these four vendors - a shared dollar allowance would buy wildly different
amounts of thinking and we would partly be measuring who got the cheaper
contract rather than who reasons better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import cost_of
from .providers.base import Usage


@dataclass
class Ledger:
    """Running spend, per agent and in total.

    The totals are accumulated from what each response actually reported, never
    estimated from the prompt, so the halt fires on real spend. `cost_of` raises
    on an unpriced model rather than adding zero, which is what stops a
    misconfigured roster from running to the turn cap with the meter stuck.
    """

    cap_usd: float
    spent_usd: float = 0.0
    by_agent: dict[str, float] = field(default_factory=dict)
    usage_by_agent: dict[str, Usage] = field(default_factory=dict)
    requests: int = 0

    def charge(self, player_id: str, model: str, usage: Usage) -> float:
        cost = cost_of(model, usage)
        self.spent_usd += cost
        self.requests += 1
        self.by_agent[player_id] = self.by_agent.get(player_id, 0.0) + cost
        self.usage_by_agent[player_id] = self.usage_by_agent.get(player_id, Usage()) + usage
        return cost

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def score_per_100k(self, player_id: str, score: int) -> float:
        """Score gained per 100k tokens - the efficiency figure the lab reports.

        Worth having as a first-class number rather than something computed in
        the dashboard: it is the one metric that compares a cheap model that
        plays adequately against an expensive one that plays slightly better,
        and it is the closest thing this experiment has to a headline result.
        """
        spent = self.usage_by_agent.get(player_id, Usage()).total
        return score / (spent / 100_000) if spent else 0.0


@dataclass
class Allowance:
    """The in-fiction token allowance, when the experiment is switched on.

    Only *output* tokens count. Input is dominated by the observation, which the
    engine writes and the agent cannot control, so charging for it would measure
    the size of the board rather than the agent's restraint.
    """

    per_agent_tokens: int
    spent: dict[str, int] = field(default_factory=dict)

    def charge(self, player_id: str, usage: Usage) -> None:
        self.spent[player_id] = self.spent.get(player_id, 0) + usage.output_tokens

    def remaining(self, player_id: str) -> int:
        return max(0, self.per_agent_tokens - self.spent.get(player_id, 0))

    def exhausted(self, player_id: str) -> bool:
        return self.remaining(player_id) <= 0

    def observation_block(self, player_id: str, turn: int, turn_limit: int) -> dict[str, object]:
        """The `budget` block of the observation, present only when enabled.

        Surfacing a countdown to a model is known to sometimes trigger premature
        wrap-up, where it starts conserving for the wrong reasons. That is a real
        risk of this design and the reason the experiment defaults off - and it
        is also, arguably, itself a finding worth having.
        """
        return {
            "tokens_allowance": self.per_agent_tokens,
            "tokens_spent": self.spent.get(player_id, 0),
            "tokens_remaining": self.remaining(player_id),
            "match_pct_elapsed": round(100 * turn / turn_limit, 1) if turn_limit else 0.0,
        }
