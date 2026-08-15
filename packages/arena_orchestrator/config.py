"""What a match is, before it starts.

Kept as data rather than arguments scattered through the loop, because the whole
roster has to be written into the journal's first record: a match is only
replayable if the thing that produced it is recoverable, and "which model sat in
seat p3" is not derivable from the moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from arena_engine.types import MatchConfig


@dataclass(frozen=True, slots=True)
class Seat:
    """One civ, and the model playing it."""

    player_id: str
    civ_name: str
    provider: str
    model: str | None = None
    # Passed through to the adapter constructor: effort, max_tokens, base_url.
    options: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "player_id": self.player_id,
            "civ_name": self.civ_name,
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything the loop needs that is not the game itself."""

    seed: int
    seats: tuple[Seat, ...]
    match: MatchConfig = field(default_factory=MatchConfig)

    # Part of the state, and therefore part of the state hash. Set once and
    # carried in the journal rather than derived from wherever the run happens
    # to be stored, so a run directory can be moved or copied without the
    # replay hashing to something different.
    match_id: str = "m1"

    # The out-of-fiction safety halt. Not visible to agents; see budget.py.
    budget_usd: float = 75.0

    # The in-fiction experiment. `off` is the baseline; `tokens` gives every
    # agent the same output-token allowance and shows it the countdown.
    agent_budget_awareness: str = "off"
    allowance_tokens: int = 400_000

    # A turn that has not come back by this point is abandoned and the agent
    # passes. Generous, because a frontier model reasoning hard about a late
    # board legitimately takes a while, and killing it early would silently
    # bias the experiment toward whichever vendor happens to be fastest.
    turn_timeout_s: float = 180.0

    # Throttling, per provider. Deliberately conservative: being rejected costs
    # a round trip plus backoff, waiting costs milliseconds.
    requests_per_minute: float = 50.0
    tokens_per_minute: float = 400_000.0

    # How many of the agent's own previous turns of reasoning to carry forward.
    # The dossier is what carries long-horizon memory; this is just enough
    # continuity that a plan spanning two turns is not forgotten between them.
    reasoning_history: int = 3

    def seat(self, player_id: str) -> Seat:
        for seat in self.seats:
            if seat.player_id == player_id:
                return seat
        raise KeyError(player_id)

    @property
    def roster(self) -> list[tuple[str, str]]:
        """The `(player_id, civ_name)` pairs the engine wants."""
        return [(seat.player_id, seat.civ_name) for seat in self.seats]
