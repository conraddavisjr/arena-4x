"""One seat at the table: a model, its guard rails, and its memory.

The rule this whole module is built around: **an agent that cannot answer passes
its turn.** Every failure path - timeout, refusal, rate limit, exhausted
retries, unparseable body, blown budget - lands on `pass_turn()` and returns a
reason. Nothing here raises into the match loop. An agent that passes plays
badly, which is a result worth recording; an agent that raises ends a multi-day
match, which is not.

**Illegal orders are not this module's problem.** The engine validates every
order against the live rules and drops the illegal ones individually with an
`order_rejected` event, so a turn where three of five orders are nonsense still
applies the other two. It costs no extra request, it degrades gracefully instead
of all-or-nothing, and the rejections come back to the agent in next turn's
`recent_events`.

**Unusable orders are.** An order that never reaches the engine - `found_city`
with no unit and no name - gets no `order_rejected`, because the reducer never
sees it. `actions.parse` discards it, and for a long time that was the end of
it: no event, no failure, nothing in any log. From outside, a model issuing
nothing but unusable orders is indistinguishable from a model choosing to do
nothing. `claude-haiku-4-5` spent 91% of its orders that way across a 128-turn
match, founded no cities, and was eliminated on turn 35 with every turn recorded
as a success.

So a discard now buys one correction naming the missing fields. This is the only
repair loop here that exists for *semantics* rather than for a malformed body,
and it exists because the silent version cost a civilisation.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from arena_engine import rules
from arena_engine.actions import Action, parse_reporting_drops, pass_turn, why_unusable

from .providers.base import LLMClient, OutOfCredits, ProviderError, Turn
from .resilience import CircuitBreaker, RetryPolicy, Sleeper, TokenBucket, with_retry

# Two attempts total, and the second one is worth having: a schema violation is
# usually a near miss the model corrects immediately when shown the error. A
# third rarely converges and costs a full request.
MAX_PARSE_ATTEMPTS = 2


@dataclass
class Outcome:
    """What one seat did this turn, and what it cost."""

    action: Action
    turns: list[Turn] = field(default_factory=list)
    failure: str | None = None
    repaired: bool = False

    @property
    def passed(self) -> bool:
        return self.failure is not None


@dataclass
class Agent:
    """One civ's seat, holding everything that persists between its turns."""

    player_id: str
    civ_name: str
    client: LLMClient
    bucket: TokenBucket
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: float = 180.0
    history_size: int = 3
    # Injectable for the same reason the resilience module takes one: without
    # it a test that proves a dead provider does not sink the match has to sit
    # through the real backoff ladder, which is forty seconds a case.
    sleep: Sleeper = asyncio.sleep
    # Called for every retry that happens *inside* a turn. Without it a 429 that
    # the ladder absorbed leaves no trace at all - the turn succeeds, nothing is
    # journalled, and a provider that is rate-limiting us on every single call
    # looks identical to one that is not.
    on_retry: Callable[[ProviderError, int, float], None] | None = None
    # The action schema in the dialect this seat's vendor accepts. Held per
    # agent rather than passed per turn because it never changes and, on
    # Anthropic, it is not the same bytes the other seats are sent.
    schema: dict[str, Any] = field(default_factory=dict)
    _history: deque[str] = field(default_factory=lambda: deque(maxlen=3))

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.history_size)
        # Built once and never rebuilt. This string is the cache prefix: it is
        # byte-identical on every turn of every match this agent plays, which is
        # the entire reason the input bill is not ten times what it is. Anything
        # that varies - the turn, the board, the dossier - goes in the user
        # message instead.
        self.system = rules.system_prompt(self.civ_name, self.player_id)

    async def take_turn(self, observation_json: str) -> Outcome:
        user = self._user_message(observation_json)
        try:
            return await asyncio.wait_for(self._attempt(user), timeout=self.timeout_s)
        except TimeoutError:
            return Outcome(action=pass_turn(), failure=f"timeout after {self.timeout_s:.0f}s")
        except OutOfCredits:
            # The one failure this method does not absorb. "An agent that cannot
            # answer passes its turn" is the right policy for a vendor having a
            # bad ten minutes, and the wrong one for an account that cannot pay:
            # that will not recover inside the run, so passing turns quietly
            # converts a billing problem into a corrupted result. Measured - a
            # 300-turn baseline played 29 further turns with one civ holding
            # cities and issuing no orders. Raised so the loop can halt.
            raise
        except ProviderError as error:
            return Outcome(action=pass_turn(), failure=f"{type(error).__name__}: {error}")

    async def _attempt(self, user: str) -> Outcome:
        turns: list[Turn] = []
        prompt = user
        last_error = ""

        for attempt in range(MAX_PARSE_ATTEMPTS):
            # Throttle on the way in rather than being rejected on the way out.
            await self.bucket.acquire(_estimate_tokens(self.system, prompt))
            turn = await self.breaker.call(
                lambda p=prompt: with_retry(
                    lambda: self.client.complete(self.system, p, self.schema),
                    policy=self.retry,
                    sleep=self.sleep,
                    on_retry=self.on_retry,
                ),
                provider=self.client.name,
            )
            turns.append(turn)
            try:
                # `actions.parse`, not `model_validate_json`: the schema some
                # vendors get is a flattened union, so a body can arrive
                # carrying fields that belong to a different order type. Those
                # are trimmed; anything genuinely wrong still raises here.
                action, dropped = parse_reporting_drops(turn.text)
            except (ValidationError, json.JSONDecodeError, ValueError) as error:
                last_error = str(error)[:800]
                # The correction goes in the *user* turn. An assistant-turn
                # prefill is the obvious way to steer a retry and it returns a
                # 400 on every current frontier model.
                prompt = (
                    f"{user}\n\nYour previous response could not be used. "
                    f"It failed validation with:\n{last_error}\n"
                    f"Return a corrected object matching the schema exactly."
                )
                continue

            # An entry that could not be used is worth one correction, not a
            # silent discard. A `found_city` with no unit is unrepairable here -
            # nothing can invent the unit - but the model can repair it, and
            # until this existed nobody ever told it. One seat spent 91% of its
            # orders this way across a whole match and was eliminated on turn 35
            # while every log said the turn had succeeded.
            #
            # Only worth a round trip while an attempt remains and something
            # survived being asked; a model that returns nothing usable twice is
            # answered by the pass below.
            if dropped and attempt < MAX_PARSE_ATTEMPTS - 1:
                faults = "; ".join(why_unusable(d) for d in dropped[:4])
                last_error = f"{len(dropped)} order(s) discarded - {faults}"
                prompt = (
                    f"{user}\n\nYour previous response was accepted but "
                    f"{len(dropped)} of your entries could not be used and were "
                    f"discarded: {faults}. Those actions did not happen. Reissue "
                    f"them with every field they require, and keep the rest of "
                    f"your turn as it was."
                )
                continue

            self._remember(action)
            return Outcome(action=action, turns=turns, repaired=attempt > 0)

        return Outcome(
            action=pass_turn(),
            turns=turns,
            failure=f"schema violation after {MAX_PARSE_ATTEMPTS} attempts: {last_error}",
        )

    def _user_message(self, observation_json: str) -> str:
        """The turn-varying half of the request.

        The dossier is not appended here - it is already inside the observation
        as `your_dossier`, returned verbatim from what the agent wrote last
        turn. Duplicating it would pay for the same tokens twice and risk the
        two copies disagreeing.
        """
        parts = [observation_json]
        if self._history:
            parts.append("Your own recent thinking, most recent last:\n" + "\n".join(self._history))
        return "\n\n".join(parts)

    def _remember(self, action: Action) -> None:
        """Carry a compacted line of reasoning forward.

        Deliberately just the plan, not the full assessment. By turn 200 the
        verbatim deliberation of turn 12 is noise, and the thing that actually
        needs continuity between consecutive turns is "what was I in the middle
        of doing".
        """
        plan = action.reasoning.plan_this_turn.strip()
        if plan:
            self._history.append(f"- {plan}")

    async def aclose(self) -> None:
        await self.client.aclose()


def _estimate_tokens(system: str, user: str) -> int:
    """Rough token count for rate limiting, not for billing.

    Four characters per token is wrong in the third significant figure and
    exactly right for deciding whether to wait 200ms. Asking a vendor to count
    would cost a round trip per turn to protect against a round trip per turn.
    """
    return (len(system) + len(user)) // 4
