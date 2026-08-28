"""The match loop.

    build four observations -> call four models at once -> resolve one turn

Repeat until somebody wins, the turn cap lands, or the budget runs out.

Three decisions are worth stating, because they are what the loop *is*:

**Turns resolve simultaneously.** Every agent sees the same board and none sees
another's move before committing to its own, so there is no turn-order
advantage to control for. It also means the wall clock per turn is the slowest
model rather than the sum of four, which over 300 turns is the difference
between a day and most of a week.

**A failed agent passes; it never propagates.** `Agent.take_turn` cannot raise,
so a provider outage costs one civ its turn and nothing else. The match is the
thing being protected.

**The budget is checked after resolution, not before.** Checking first would
abandon a turn that four agents had already been billed for. Checking after
means the halt lands on a coherent board that can be scored.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arena_replay import BundleWriter

from arena_engine import observation as obs
from arena_engine.actions import Action, pass_turn
from arena_engine.events import Event
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig, State

from . import journal as jl
from .agent import Agent, Outcome
from .budget import Allowance, Ledger
from .config import RunConfig
from .dialects import for_provider
from .journal import Journal
from .providers import build as build_client
from .providers.base import LLMClient, OutOfCredits, Usage
from .resilience import CircuitBreaker, Sleeper, TokenBucket

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "action.schema.json"


def _now() -> str:
    """UTC, to the second, ISO 8601.

    The only clock reading anywhere near the journal. Everything else here is
    deliberately time-free, because determinism comes from the seed and the
    recorded decisions and a replay that depended on when it ran would not be
    one. This is metadata *about* a run - it sits outside the state hash, resume
    never reads it back, and rebuilding a bundle does not change it.

    It exists because nothing knew when a match happened, which is fine for a
    replay and useless for a library of them.
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat()


# Which adapters read a token stream, and therefore can tell a model thinking
# from a connection that has stopped talking. Recorded as a set rather than an
# `if` so the next vendor to gain a stream is an entry here, not a branch - and
# so that the seats *without* one are visible rather than assumed.
#
# Google is the omission. Its interactions surface exposes no token stream, so
# that seat still depends on the transport timeout and the turn backstop.
STREAMING_PROVIDERS = frozenset({"anthropic", "openai"})

# What each vendor calls the reasoning dial. The names differ, the intent does
# not, and keeping the mapping here means the match sets it once.
EFFORT_PARAM = {
    "anthropic": "effort",
    "openai": "reasoning_effort",
    "xai": "reasoning_effort",
    "google": "effort",
}


@dataclass
class MatchResult:
    state: State
    reason: str
    winner: str | None
    ledger: Ledger
    failures: int = 0


@dataclass
class Orchestrator:
    config: RunConfig
    root: Path
    clients: dict[str, LLMClient] | None = None
    schema: dict[str, Any] = field(default_factory=dict)
    bundle: bool = True
    # Fired with the resolved state after every turn, and once before the first.
    # Progress reporting hangs off this, and so does the dry-run harness, which
    # needs its bot seats to see the current board rather than turn zero's.
    after_turn: Callable[[State], None] | None = None
    # Every wait in the loop - backoff and throttling alike - goes through this.
    # Overriding it is what lets a test prove that a provider outage costs one
    # civ its turns without the suite sitting through the real backoff ladder.
    sleep: Sleeper = asyncio.sleep
    # What each civ saw happen to it last turn. The engine writes one event
    # stream for the whole board, so this is the per-agent slice - a civ should
    # be told its own warrior died, not everybody's.
    _recent: dict[str, list[str]] = field(default_factory=dict)
    # Retries absorbed since the last turn boundary, drained into the journal.
    _retries: list[dict[str, Any]] = field(default_factory=list)
    # This turn's cost per civ, drained into the frame when the turn resolves.
    _spend: dict[str, Any] = field(default_factory=dict)

    @property
    def bundle_root(self) -> Path:
        """Where the publishable replay is written. Self-contained by
        construction, so publishing is a directory copy with nothing to omit."""
        return self.root / "bundle"

    def __post_init__(self) -> None:
        if not self.schema:
            self.schema = json.loads(SCHEMA_PATH.read_text())
        self.ledger = Ledger(cap_usd=self.config.budget_usd)
        self.allowance = (
            Allowance(per_agent_tokens=self.config.allowance_tokens)
            if self.config.agent_budget_awareness == "tokens"
            else None
        )
        # One bucket per *provider*, not per seat: the limit belongs to the
        # vendor account, so two Anthropic seats in the same match share it. A
        # bucket each would let the pair burst to twice the configured rate.
        self._buckets: dict[str, TokenBucket] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self.agents: dict[str, Agent] = {}
        effort = self.config.reasoning_effort
        for seat in self.config.seats:
            # The stall gap is match policy rather than seat configuration, so it
            # comes from the run config - but a seat may still override it, and a
            # vendor whose adapter cannot stream simply does not accept it.
            options = dict(seat.options)
            if seat.provider in STREAMING_PROVIDERS:
                options.setdefault("stall_gap_s", self.config.stall_gap_s)
            # Every vendor takes it under a different name, and every vendor now
            # takes it. A seat may still override, which is how an experiment
            # that *wants* asymmetry asks for it explicitly rather than
            # inheriting it from whichever defaults happened to disagree.
            options.setdefault(EFFORT_PARAM.get(seat.provider, "effort"), effort)
            client = (self.clients or {}).get(seat.player_id) or build_client(
                seat.provider, seat.model, **options
            )
            self.agents[seat.player_id] = Agent(
                player_id=seat.player_id,
                civ_name=seat.civ_name,
                client=client,
                # Anthropic compiles the schema into a grammar and rejects this
                # one as too large; it gets a flattened variant. Everyone else
                # takes the base schema unchanged. See dialects.py.
                schema=for_provider(self.schema, seat.provider, seat.model),
                bucket=self._bucket(seat.provider),
                breaker=self._breaker(seat.provider),
                timeout_s=self.config.turn_timeout_s,
                history_size=self.config.reasoning_history,
                sleep=self.sleep,
                on_retry=self._note_retry(seat.player_id, seat.provider),
            )

    def _note_retry(self, player_id: str, provider: str):
        """Record a retry the ladder absorbed, so it is not invisible."""

        def note(error: Exception, attempt: int, delay: float) -> None:
            self._retries.append(
                {
                    "player_id": player_id,
                    "provider": provider,
                    "error": type(error).__name__,
                    "attempt": attempt,
                    "delay_s": round(delay, 2),
                }
            )

        return note

    def _bucket(self, provider: str) -> TokenBucket:
        if provider not in self._buckets:
            self._buckets[provider] = TokenBucket(
                self.config.requests_per_minute,
                self.config.tokens_per_minute,
                sleep=self.sleep,
            )
        return self._buckets[provider]

    def _breaker(self, provider: str) -> CircuitBreaker:
        # Shared per provider for the same reason: "Anthropic is down" is a fact
        # about Anthropic, and each seat learning it separately would spend four
        # full retry ladders to reach the same conclusion.
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker()
        return self._breakers[provider]

    # -----------------------------------------------------------------------

    async def run(self, *, resume: bool = False) -> MatchResult:
        recovered = jl.recover(self.root) if resume else None
        if recovered and recovered.ended:
            raise RuntimeError(
                f"match at {self.root} already ended; nothing to resume. A match "
                f"that reached a game outcome is finished - resuming one would "
                f"play turns past its own ending."
            )

        journal = Journal.open(self.root, resume=bool(recovered))
        match_id = (recovered.match_id if recovered else None) or self.config.match_id
        # A recovered config wins over the configured one for the same reason
        # the recovered match id does: it is what the recorded decisions were
        # made under, and anything else replays to a different board.
        match_config = self.config.match
        if recovered and recovered.match_config:
            match_config = MatchConfig.model_validate(recovered.match_config)
        state, events = new_match(match_id, self.config.seed, self.config.roster, match_config)
        # The bundle goes in its own subdirectory, apart from the journal and
        # the transcripts. Those live in the run root and the bundle is what
        # gets published - and they were all in one directory, which meant
        # serving a match served the system prompts with it. The one property
        # a published match has to have is that it carries nothing but the
        # match, and "remember to exclude two files" is not that property.
        models = {seat.player_id: seat.model or seat.provider for seat in self.config.seats}
        writer = BundleWriter.start(self.bundle_root, state, models) if self.bundle else None

        if recovered:
            # Before anything is spent: the cap belongs to the match, not to
            # this process. Starting at zero would let a run that crashed near
            # its limit spend the whole cap a second time.
            self.ledger.spent_usd = recovered.spent_usd
            # Dollars per seat as well as the total. Seeding these at zero left
            # the report's spend column summing to less than its own total, and
            # `score_per_100k` dividing a whole match's score by the spend of
            # however much of it happened after the last interruption.
            self.ledger.by_agent.update(recovered.spent_by_agent)
            for player_id, counts in recovered.usage_by_agent.items():
                self.ledger.by_agent.setdefault(player_id, 0.0)
                self.ledger.usage_by_agent[player_id] = Usage(**counts)
            state, writer = self._replay(state, recovered, writer)
        else:
            journal.append(
                jl.MATCH_CREATED,
                0,
                match_id=match_id,
                seed=self.config.seed,
                # The whole config, not just the seed. `MatchConfig` is part of
                # `State` and therefore part of the state hash, so a resume that
                # rebuilt it from defaults - or from whatever flags the operator
                # happened to retype - reproduces a different match. Resume was
                # only working because the same `--turns` was passed by hand.
                match_config=self.config.match.model_dump(mode="json"),
                seats=[seat.to_json() for seat in self.config.seats],
                state_hash=state.state_hash(),
                # Wall-clock, purely for the humans. The journal is otherwise
                # deliberately time-free - determinism comes from the seed and
                # the recorded decisions, and a replay must not depend on when
                # it happened - so this is metadata *about* the run rather than
                # an input to it. Nothing reads it back during resume, and it
                # sits outside the state hash.
                started_at=_now(),
            )

        failures = 0
        reason = "turn_limit"
        try:
            if self.after_turn:
                self.after_turn(state)
            while state.victory is None and state.turn < self.config.match.turn_limit:
                try:
                    actions, outcomes = await self._collect(state, journal)
                except OutOfCredits as broke:
                    # Halted rather than absorbed, and this is the one provider
                    # failure treated that way. Everything else here is designed
                    # so a bad vendor costs one civ one turn; an account that
                    # cannot pay costs that civ *every remaining turn*, and the
                    # match quietly becomes a three-way comparison nobody asked
                    # for. Better to stop on a coherent board that can be
                    # scored, exactly as the budget cap does.
                    journal.append(
                        jl.AGENT_FAILURE,
                        state.turn + 1,
                        player_id=getattr(broke, "player_id", None),
                        reason=f"OutOfCredits: {broke}",
                    )
                    reason = "provider_credits"
                    break
                failures += sum(1 for o in outcomes.values() if o.passed)
                state, events = step(state, actions)
                self._remember_events(events)
                if self.after_turn:
                    self.after_turn(state)

                for note in self._retries:
                    journal.append(jl.PROVIDER_RETRY, state.turn, **note)
                self._retries.clear()
                waited = {p: round(b.waited_s, 1) for p, b in self._buckets.items() if b.waited_s}
                if waited:
                    journal.append(jl.THROTTLED, state.turn, seconds_by_provider=waited)
                journal.append(
                    jl.TURN_RESOLVED,
                    state.turn,
                    actions={p: a.model_dump(mode="json") for p, a in actions.items()},
                    state_hash=state.state_hash(),
                )
                if writer:
                    writer.add(state, events, self._spend)
                self._spend = {}

                if self.ledger.exhausted:
                    reason = "budget_cap"
                    break

            if state.victory is not None:
                reason = state.victory.condition
            winner = state.victory.winner if state.victory else None
            journal.append(
                jl.MATCH_ENDED,
                state.turn,
                reason=reason,
                winner=winner,
                spent_usd=round(self.ledger.spent_usd, 4),
                finished_at=_now(),
            )
            if writer:
                writer.finish(
                    state,
                    {"winner": winner, "reason": reason},
                    finished_at=_now(),
                    spent_usd=round(self.ledger.spent_usd, 4),
                )
            return MatchResult(state, reason, winner, self.ledger, failures)
        finally:
            journal.close()
            await self.aclose()

    async def _collect(
        self, state: State, journal: Journal
    ) -> tuple[dict[str, Action], dict[str, Any]]:
        """Four concurrent calls, resolved together."""
        living = [pid for pid in state.civ_ids() if state.players[pid].alive]
        prompts = {pid: self._observation(state, pid) for pid in living}

        # A civ that has spent its token allowance stops being asked. This is
        # what makes the allowance an in-game constraint rather than a number
        # printed in the observation: without it the countdown reached zero and
        # nothing whatsoever happened, which is how it stood - `exhausted()` was
        # written, tested, and never called.
        #
        # It passes rather than being eliminated. A civ that vanishes hands its
        # cities to nobody and rewrites the board for the other three, which
        # would make the *other* seats' results depend on when this one ran dry.
        # A civ that can no longer act is still a real player, still holds
        # territory, and gets outcompeted in public - which is both a fairer
        # consequence and far better evidence.
        broke = [pid for pid in living if self.allowance and self.allowance.exhausted(pid)]
        asked = [pid for pid in living if pid not in broke]

        results = await asyncio.gather(*(self.agents[pid].take_turn(prompts[pid]) for pid in asked))
        outcomes = dict(zip(asked, results, strict=True))
        for pid in broke:
            outcomes[pid] = Outcome(action=pass_turn(), failure="token allowance exhausted")

        actions: dict[str, Action] = {}
        for player_id, outcome in outcomes.items():
            actions[player_id] = outcome.action
            self._account(state, journal, player_id, outcome, prompts[player_id])
        return actions, outcomes

    def _account(self, state: State, journal: Journal, player_id: str, outcome, user: str) -> None:
        agent = self.agents[player_id]
        for turn in outcome.turns:
            cost = self.ledger.charge(player_id, turn.model, turn.usage)
            row = self._spend.setdefault(
                player_id,
                {"usd": 0.0, "input": 0, "output": 0, "cached": 0, "ms": 0, "effort": None},
            )
            row["usd"] = round(row["usd"] + cost, 6)
            row["input"] += turn.usage.input_tokens
            row["output"] += turn.usage.output_tokens
            row["cached"] += turn.usage.cache_read_tokens
            row["ms"] += turn.latency_ms
            row["effort"] = turn.effort
            row["effort_sent"] = turn.effort_sent
            if self.allowance:
                self.allowance.charge(player_id, turn.usage)
            journal.append(
                jl.AGENT_CALL,
                state.turn + 1,
                player_id=player_id,
                model=turn.model,
                usage=turn.usage.__dict__ if hasattr(turn.usage, "__dict__") else None,
                input_tokens=turn.usage.input_tokens,
                output_tokens=turn.usage.output_tokens,
                cache_read_tokens=turn.usage.cache_read_tokens,
                # Recorded because its absence is the symptom of the cache
                # breakpoint being in the wrong place: writes on every turn and
                # reads on none. Cost was always right; the telemetry was not.
                cache_write_tokens=turn.usage.cache_write_tokens,
                cost_usd=round(cost, 6),
                latency_ms=turn.latency_ms,
                stop_reason=turn.stop_reason,
                # The trace itself goes to the transcripts; its size goes here,
                # so "is this seat's reasoning being captured at all" is a
                # question the journal can answer. A seat billing thousands of
                # reasoning tokens while storing none of them is a real state
                # and one worth being able to see without opening 80 payloads.
                reasoning_tokens=turn.usage.reasoning_tokens,
                thinking_chars=len(turn.thinking or ""),
                # What this seat was asked for. Journalled beside the tokens
                # because the two only mean something together: 600k output
                # tokens at `low` and at `high` are different findings.
                effort=turn.effort,
                effort_sent=turn.effort_sent,
            )
            journal.transcript(
                state.turn + 1, player_id, agent.system, user, turn.text, turn.thinking
            )

        # A cache that never reads costs 12.5x the input price of the prefix and
        # fails nothing. Tests can miss it - one did, for a whole live match -
        # so the running match checks for itself.
        if state.turn >= 2 and agent.client.name == "anthropic":
            for turn in outcome.turns:
                if turn.usage.cache_read_tokens == 0 and turn.usage.cache_write_tokens > 0:
                    journal.append(
                        jl.CACHE_MISS,
                        state.turn + 1,
                        player_id=player_id,
                        wrote=turn.usage.cache_write_tokens,
                        note="wrote a fresh cache entry instead of reading one; "
                        "the prefix is varying or the breakpoint is misplaced",
                    )

        if outcome.repaired:
            journal.append(jl.PARSE_REPAIRED, state.turn + 1, player_id=player_id)
        if outcome.failure:
            # Surfaced on the dashboard rather than buried: an agent that passes
            # is playing badly for a reason, and a run where one seat quietly
            # passed forty turns is not a fair comparison.
            journal.append(
                jl.AGENT_FAILURE, state.turn + 1, player_id=player_id, reason=outcome.failure
            )

    def _observation(self, state: State, player_id: str) -> str:
        budget = None
        if self.allowance:
            block = self.allowance.observation_block(
                player_id, state.turn, self.config.match.turn_limit
            )
            budget = obs.BudgetView(**block)
        return obs.to_json(
            obs.build(
                state,
                player_id,
                recent_events=self._recent.get(player_id, []),
                budget=budget,
            )
        )

    def _remember_events(self, events: list[Event]) -> None:
        """Slice the turn's events into what each civ is entitled to know.

        Only events attributed to a civ go to that civ. Engine-level events have
        no actor and are addressed to nobody, and handing an agent the whole
        board's event stream would leak the fog of war straight through the one
        channel that is supposed to respect it.
        """
        self._recent = {}
        for event in events:
            if event.actor:
                self._recent.setdefault(event.actor, []).append(event.text)

    # -----------------------------------------------------------------------

    def _replay(self, state: State, recovered, writer):
        """Rebuild an interrupted match by re-applying its recorded decisions.

        The hash check is the point. A replay that silently diverged - because
        the engine changed between crash and restart, or a line was truncated -
        would resume a *different* match while every log said otherwise, and
        nothing downstream would ever notice.
        """
        if recovered.seed is not None and recovered.seed != self.config.seed:
            raise RuntimeError(
                f"journal was written with seed {recovered.seed}, "
                f"but this run is configured for {self.config.seed}"
            )
        for record in recovered.turns:
            actions = {
                player_id: Action.model_validate(payload)
                for player_id, payload in record["actions"].items()
            }
            state, events = step(state, actions)
            if state.state_hash() != record["state_hash"]:
                raise RuntimeError(
                    f"replay diverged at turn {record['turn']}: the recorded decisions no "
                    f"longer reproduce the recorded state. Refusing to resume into a match "
                    f"the log does not describe."
                )
            if writer:
                # With the spend the journal recorded, not without it. The
                # frames a bundle is made of carry what each seat spent that
                # turn, so replaying them empty left a resumed match reporting
                # $0.00 for every turn before the interruption - the tokens were
                # on disk the whole time and nothing was reading them.
                writer.add(state, events, recovered.spend_by_turn.get(record["turn"], {}))
        return state, writer

    async def aclose(self) -> None:
        await asyncio.gather(
            *(agent.aclose() for agent in self.agents.values()), return_exceptions=True
        )
