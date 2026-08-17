"""A seat that plays competently and costs nothing.

The scripted provider returns canned bodies, which is right for unit tests and
useless for exercising a match: a hundred turns of `pass_turn` never founds a
city, never starts a war, and never produces the board states where the
interesting failures live.

This wraps the engine's heuristic bots in the provider interface instead. The
match that comes out is a real match - cities, wars, a victor - and every layer
above the HTTP call is exercised for real: the observation is built, the JSON is
serialised, the schema validates it, the ledger is charged, the journal is
written, the bundle grows.

It is deliberately a `LLMClient` rather than a shortcut around one. A dry run
that bypassed the seam would prove the loop works with something that is not
how it will actually be used, which is the same as not testing it.

Unlike everything in `providers/`, this imports the engine - that is why it
lives out here rather than in with the adapters, which must stay free of both
the game rules and any vendor SDK.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from arena_engine import bots
from arena_engine.types import State

from .providers.base import Turn, Usage

# Priced like a small model so a dry run produces a believable bill and the
# budget halt is genuinely exercised rather than bypassed by a zero.
DRY_RUN_MODEL = "claude-haiku-4-5"


@dataclass
class BotClient:
    """Plays one seat with the engine's heuristics, through the provider seam.

    Reads the live state by reference rather than parsing it back out of the
    observation. That is a shortcut a real model does not get, and it is fine
    here: the point is to exercise the *loop*, not to prove a heuristic can play
    from a fog-limited view.
    """

    player_id: str
    board: BoardHandle
    model: str = DRY_RUN_MODEL
    name: str = "bot"
    # A bot does not deliberate, so it reports no trace. Settable because the
    # loop's persistence of traces has to be testable without a live vendor -
    # that gap is exactly why traces went unstored for as long as they did.
    thinking: str | None = None
    calls: int = field(default=0, init=False)

    async def complete(self, system: str, user: str, schema: dict[str, Any]) -> Turn:
        self.calls += 1
        action = bots.all_bot_actions(self.board.state).get(self.player_id)
        text = action.model_dump_json() if action is not None else "{}"
        return Turn(
            text=text,
            # Roughly the shape of a real turn: a cached system prefix, a fresh
            # observation, a small structured body.
            usage=Usage(
                input_tokens=len(user) // 4,
                output_tokens=len(text) // 4,
                cache_read_tokens=len(system) // 4,
            ),
            model=self.model,
            latency_ms=0,
            stop_reason="end_turn",
            thinking=self.thinking,
        )

    async def aclose(self) -> None:
        return None


@dataclass
class BoardHandle:
    """A mutable window onto the current state, shared by every bot seat.

    The loop rebinds its `state` local on every turn, so the seats need
    something that follows it rather than the object they were built with.
    """

    state: State | None = None

    def observe(self, state: State) -> None:
        self.state = state


def bot_seats(
    player_ids: Iterable[str], *, thinking: str | None = None
) -> tuple[dict[str, BotClient], BoardHandle]:
    """Clients for every seat, all sharing one view of the board.

    Hand the clients to `Orchestrator(clients=...)` and `handle.observe` to its
    `after_turn` hook. Without the hook the bots would plan every turn against
    the opening position, and the match would go nowhere at great length.
    """
    handle = BoardHandle()
    clients = {pid: BotClient(player_id=pid, board=handle, thinking=thinking) for pid in player_ids}
    return clients, handle
