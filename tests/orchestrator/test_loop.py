"""The match loop, driven end to end without touching a network.

Every test here plays a real match: real observations, real schema validation,
real ledger, real journal, real bundle. Only the HTTP call is replaced. That is
the whole point of building the dry-run seat as an `LLMClient` rather than as a
shortcut around one - a loop tested with something that is not how it will be
used has not been tested.

The properties being defended are the ones that only show up on day two of an
unattended run: an agent that fails passes rather than propagating, the budget
halt lands on a scoreable board, and a crashed match resumes into *the same
match* rather than a plausible-looking different one.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from arena_engine.types import MatchConfig
from arena_orchestrator import journal as jl
from arena_orchestrator.config import RunConfig, Seat
from arena_orchestrator.dryrun import bot_seats
from arena_orchestrator.loop import Orchestrator
from arena_orchestrator.providers.base import (
    Overloaded,
    RateLimited,
    Refused,
    Turn,
    Usage,
)

CIVS = ["Aurelian Compact", "Iron Concord", "Verdant Pact", "Solari Dominion"]


def make_config(turns: int = 8, **overrides) -> RunConfig:
    # One provider per seat, as a real roster has. It matters: rate limits and
    # circuit breakers are keyed by provider because they are facts about a
    # vendor account, so seats sharing a provider share both. Labelling all four
    # the same here would have three healthy seats continually resetting the
    # failure count of a fourth that is genuinely down.
    seats = tuple(
        Seat(player_id=f"p{i + 1}", civ_name=CIVS[i], provider=f"bot{i + 1}") for i in range(4)
    )
    return RunConfig(seed=4, seats=seats, match=MatchConfig(turn_limit=turns), **overrides)


class Waits:
    """Records what the loop would have slept for, and sleeps for none of it.

    Every backoff and every throttle goes through here, so a test proving that
    a provider outage costs one civ its turns runs in milliseconds instead of
    sitting through five real retry ladders - and it can *assert* the backoff
    happened, which waiting for it never did.
    """

    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)

    @property
    def total(self) -> float:
        return sum(self.seconds)


def make_orchestrator(root: Path, config: RunConfig, **overrides) -> Orchestrator:
    clients, handle = bot_seats(
        (seat.player_id for seat in config.seats), thinking=overrides.pop("thinking", None)
    )
    clients.update(overrides.pop("clients", {}))
    overrides.setdefault("sleep", Waits())
    return Orchestrator(
        config=config, root=root, clients=clients, after_turn=handle.observe, **overrides
    )


def records(root: Path, kind: str) -> list[dict]:
    return [r for r in jl.Journal(root=root).records() if r["type"] == kind]


# ---------------------------------------------------------------------------
# A match happens
# ---------------------------------------------------------------------------


async def test_a_match_plays_and_writes_everything_downstream_needs(tmp_path: Path) -> None:
    config = make_config(turns=8)
    result = await make_orchestrator(tmp_path, config).run()

    assert result.state.turn == 8
    assert result.failures == 0
    # The bundle the viewer reads, written as the match went rather than after,
    # and in its own directory so publishing it cannot sweep up the transcripts.
    assert (tmp_path / "bundle" / "match.json").exists()
    assert len(list((tmp_path / "bundle" / "turns").glob("*.json"))) == 8
    assert not list((tmp_path / "bundle").glob("transcripts*"))
    assert not list((tmp_path / "bundle").glob("journal*"))
    # The journal resume reads, and the transcripts it deliberately does not.
    assert len(records(tmp_path, jl.TURN_RESOLVED)) == 8
    assert len(records(tmp_path, jl.AGENT_CALL)) == 32
    assert (tmp_path / "transcripts.jsonl").stat().st_size > 0


async def test_four_agents_are_called_concurrently_per_turn(tmp_path: Path) -> None:
    """Simultaneous resolution: nobody sees another's move before committing to
    their own, so there is no turn-order advantage to control for."""
    result = await make_orchestrator(tmp_path, make_config(turns=5)).run()
    calls = records(tmp_path, jl.AGENT_CALL)
    by_turn: dict[int, set[str]] = {}
    for call in calls:
        by_turn.setdefault(call["turn"], set()).add(call["player_id"])
    assert all(seats == {"p1", "p2", "p3", "p4"} for seats in by_turn.values())
    assert result.state.turn == 5


async def test_the_agent_sees_only_its_own_events(tmp_path: Path) -> None:
    """Handing an agent the whole board's event stream would leak the fog of war
    through the one channel that is supposed to respect it."""
    config = make_config(turns=6)
    orchestrator = make_orchestrator(tmp_path, config)
    await orchestrator.run()
    for player_id, texts in orchestrator._recent.items():
        assert texts, f"{player_id} was told nothing"
        for other in ("p1", "p2", "p3", "p4"):
            if other != player_id:
                assert not any(t.startswith(f"{other} ") for t in texts)


# ---------------------------------------------------------------------------
# Failure never propagates
# ---------------------------------------------------------------------------


class Broken:
    """A seat whose provider is having a bad day."""

    name = "broken"
    model = "claude-haiku-4-5"

    def __init__(self, error: Exception):
        self._error = error
        self.calls = 0

    async def complete(self, system, user, schema):
        self.calls += 1
        raise self._error

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize(
    "error",
    [Overloaded("529"), RateLimited("429"), Refused("policy")],
    ids=["overloaded", "rate_limited", "refused"],
)
async def test_a_dead_provider_costs_one_civ_its_turns_not_the_match(
    tmp_path: Path, error: Exception
) -> None:
    """The property the whole design rests on. An agent that passes plays badly,
    which is a result. An agent that raises ends a multi-day match."""
    config = make_config(turns=6)
    result = await make_orchestrator(tmp_path, config, clients={"p2": Broken(error)}).run()

    assert result.state.turn == 6
    assert result.failures == 6  # p2 passed every turn
    failures = records(tmp_path, jl.AGENT_FAILURE)
    assert {f["player_id"] for f in failures} == {"p2"}
    # And the other three played normally throughout.
    assert len({c["player_id"] for c in records(tmp_path, jl.AGENT_CALL)}) == 3


async def test_a_retryable_outage_backs_off_and_then_stops_trying(tmp_path: Path) -> None:
    """Retry handles a provider that is busy; the breaker handles one that is
    down. Without the second, six turns would each spend a full ladder to reach
    the same conclusion - most of an hour of wall clock learning nothing."""
    waits = Waits()
    client = Broken(Overloaded("529"))
    result = await make_orchestrator(
        tmp_path, make_config(turns=6), clients={"p2": client}, sleep=waits
    ).run()

    assert result.failures == 6
    # Five attempts a turn until the breaker opens, then none at all.
    assert client.calls < 6 * 5
    assert waits.total > 0, "a retryable error should have backed off"


async def test_a_refusal_is_not_retried(tmp_path: Path) -> None:
    """The model declined. That is an answer, not a transport failure, and
    asking again four times spends money to be declined four more times."""
    waits = Waits()
    client = Broken(Refused("policy"))
    await make_orchestrator(
        tmp_path, make_config(turns=3), clients={"p2": client}, sleep=waits
    ).run()

    assert client.calls == 3  # once a turn, no ladder
    assert waits.seconds == []


class Garbage:
    """Returns a body that is not the schema, then a good one."""

    name = "garbage"
    model = "claude-haiku-4-5"

    def __init__(self, *, always_bad: bool = False):
        self.always_bad = always_bad
        self.calls = 0

    async def complete(self, system, user, schema):
        self.calls += 1
        bad = self.always_bad or self.calls % 2 == 1
        text = "not json at all" if bad else json.dumps({"orders": []})
        return Turn(text=text, usage=Usage(output_tokens=10), model=self.model, latency_ms=1)


async def test_an_unparseable_body_is_repaired_once(tmp_path: Path) -> None:
    """A schema violation is usually a near miss the model corrects when shown
    the error. The correction goes in the user turn - an assistant prefill is
    the obvious way to steer a retry and 400s on every current model."""
    client = Garbage()
    result = await make_orchestrator(tmp_path, make_config(turns=3), clients={"p2": client}).run()

    assert result.failures == 0
    assert len(records(tmp_path, jl.PARSE_REPAIRED)) == 3
    assert client.calls == 6  # two attempts a turn


async def test_a_body_that_never_parses_gives_up_and_passes(tmp_path: Path) -> None:
    client = Garbage(always_bad=True)
    result = await make_orchestrator(tmp_path, make_config(turns=3), clients={"p2": client}).run()

    assert result.failures == 3
    assert client.calls == 6  # not more: a third attempt rarely converges
    assert all("schema violation" in f["reason"] for f in records(tmp_path, jl.AGENT_FAILURE))


async def test_illegal_orders_are_dropped_individually_by_the_engine(tmp_path: Path) -> None:
    """Not repaired here, by design: the engine drops the illegal ones and keeps
    the rest, which costs no extra request and degrades gracefully instead of
    all-or-nothing."""

    class Nonsense:
        name = "nonsense"
        model = "claude-haiku-4-5"

        async def complete(self, system, user, schema):
            payload = {
                "orders": [
                    {"action": "move_unit", "unit_id": "u9999", "to": "40,40"},
                    {"action": "set_research", "tech": "time_travel"},
                ]
            }
            return Turn(
                text=json.dumps(payload),
                usage=Usage(output_tokens=10),
                model=self.model,
                latency_ms=1,
            )

    result = await make_orchestrator(
        tmp_path, make_config(turns=4), clients={"p2": Nonsense()}
    ).run()
    # The turn was accepted - it parsed - and the match carried on.
    assert result.failures == 0
    assert result.state.turn == 4


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


async def test_the_budget_cap_halts_the_match_on_a_scoreable_board(tmp_path: Path) -> None:
    """Checked after resolution, not before: halting mid-turn would abandon a
    turn four agents had already been billed for."""
    config = make_config(turns=50, budget_usd=0.02)
    result = await make_orchestrator(tmp_path, config).run()

    assert result.reason == "budget_cap"
    assert result.state.turn < 50
    assert result.ledger.spent_usd >= 0.02
    ended = records(tmp_path, jl.MATCH_ENDED)[0]
    assert ended["reason"] == "budget_cap"
    # The board is complete and the bundle is finished, so it can still be
    # scored, replayed and published.
    assert (tmp_path / "bundle" / "match.json").exists()
    assert len(list((tmp_path / "bundle" / "turns").glob("*.json"))) == result.state.turn


async def test_spend_is_attributed_per_agent(tmp_path: Path) -> None:
    result = await make_orchestrator(tmp_path, make_config(turns=6)).run()
    assert set(result.ledger.by_agent) == {"p1", "p2", "p3", "p4"}
    assert all(v > 0 for v in result.ledger.by_agent.values())
    assert result.ledger.spent_usd == pytest.approx(sum(result.ledger.by_agent.values()))


async def test_the_token_allowance_reaches_the_observation_when_enabled(
    tmp_path: Path,
) -> None:
    config = make_config(turns=4, agent_budget_awareness="tokens", allowance_tokens=50_000)
    await make_orchestrator(tmp_path, config).run()
    payloads = [
        json.loads(line) for line in (tmp_path / "transcripts.jsonl").read_text().splitlines()
    ]
    assert all('"budget"' in p["user"] for p in payloads)


async def test_the_allowance_is_absent_by_default(tmp_path: Path) -> None:
    """It defaults off because surfacing a countdown is known to sometimes
    trigger premature wrap-up, where a model conserves for the wrong reasons."""
    await make_orchestrator(tmp_path, make_config(turns=3)).run()
    payloads = [
        json.loads(line) for line in (tmp_path / "transcripts.jsonl").read_text().splitlines()
    ]
    assert all('"budget"' not in p["user"] for p in payloads)


# ---------------------------------------------------------------------------
# Crash and resume
# ---------------------------------------------------------------------------


def crash_after(root: Path, turn: int) -> None:
    """Truncate a journal the way a kill -9 between turns would."""
    path = root / "journal.jsonl"
    keep = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["type"] == jl.MATCH_ENDED or record["turn"] > turn:
            continue
        keep.append(line)
    path.write_text("\n".join(keep) + "\n")


async def test_a_crashed_match_resumes_into_the_same_match(tmp_path: Path) -> None:
    """The property that makes resume trustworthy: the same board, not a
    plausible-looking different one. Verified by state hash, because a replay
    that silently diverged would look identical in every log."""
    clean, crashed = tmp_path / "clean", tmp_path / "crashed"
    config = make_config(turns=10)
    await make_orchestrator(clean, config).run()

    crashed.mkdir()
    for name in ("journal.jsonl", "transcripts.jsonl"):
        (crashed / name).write_text((clean / name).read_text())
    crash_after(crashed, turn=6)

    result = await make_orchestrator(crashed, make_config(turns=10)).run(resume=True)

    assert result.state.turn == 10
    assert records(clean, jl.TURN_RESOLVED)[-1]["state_hash"] == result.state.state_hash()


async def test_resume_replays_rather_than_re_paying(tmp_path: Path) -> None:
    """Recovery re-applies recorded decisions; it does not re-ask the models.
    Otherwise a crash on turn 250 would cost the price of the match again."""
    clean, crashed = tmp_path / "clean", tmp_path / "crashed"
    await make_orchestrator(clean, make_config(turns=10)).run()

    crashed.mkdir()
    for name in ("journal.jsonl", "transcripts.jsonl"):
        (crashed / name).write_text((clean / name).read_text())
    crash_after(crashed, turn=6)

    orchestrator = make_orchestrator(crashed, make_config(turns=10))
    result = await orchestrator.run(resume=True)
    # Four turns short of ten, times four seats.
    assert result.ledger.requests == 16


async def test_a_divergent_journal_refuses_to_resume(tmp_path: Path) -> None:
    """An engine change between crash and restart, or a truncated line, would
    otherwise resume a different match while every log said otherwise."""
    root = tmp_path / "run"
    await make_orchestrator(root, make_config(turns=6)).run()
    crash_after(root, turn=4)

    path = root / "journal.jsonl"
    lines = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["type"] == jl.TURN_RESOLVED and record["turn"] == 3:
            record["state_hash"] = "0" * 64
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(RuntimeError, match="replay diverged at turn 3"):
        await make_orchestrator(root, make_config(turns=6)).run(resume=True)


async def test_resuming_a_finished_match_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "run"
    await make_orchestrator(root, make_config(turns=4)).run()
    with pytest.raises(RuntimeError, match="already ended"):
        await make_orchestrator(root, make_config(turns=8)).run(resume=True)


async def test_a_moved_run_directory_still_resumes(tmp_path: Path) -> None:
    """The match id is part of the state hash, so deriving it from the directory
    name meant that copying a run - exactly what an operator does while
    recovering one - replayed to a different hash and looked like corruption."""
    original, moved = tmp_path / "run-a", tmp_path / "run-b"
    await make_orchestrator(original, make_config(turns=8)).run()

    moved.mkdir()
    for name in ("journal.jsonl", "transcripts.jsonl"):
        (moved / name).write_text((original / name).read_text())
    crash_after(moved, turn=5)

    result = await make_orchestrator(moved, make_config(turns=8)).run(resume=True)
    assert result.state.state_hash() == records(original, jl.TURN_RESOLVED)[-1]["state_hash"]


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


class Flaky:
    """Fails a fixed number of times per turn, then succeeds."""

    name = "flaky"
    model = "claude-haiku-4-5"

    def __init__(self, failures_per_turn: int = 2):
        self._budget = failures_per_turn
        self._left = failures_per_turn
        self.calls = 0

    async def complete(self, system, user, schema):
        self.calls += 1
        if self._left:
            self._left -= 1
            raise Overloaded("529")
        self._left = self._budget
        return Turn(
            text=json.dumps({"orders": []}),
            usage=Usage(output_tokens=10),
            model=self.model,
            latency_ms=1,
        )

    async def aclose(self) -> None:
        return None


async def test_a_retry_the_ladder_absorbed_is_still_recorded(tmp_path: Path) -> None:
    """The blind spot this closes. A 429 that the retry ladder swallowed used to
    leave no trace at all: the turn succeeded, nothing was journalled, and a
    provider rate-limiting every single call looked identical to a healthy one.
    The only symptom was a run that took longer than it should, and "the run is
    slow" is not a diagnosis."""
    client = Flaky(failures_per_turn=2)
    await make_orchestrator(tmp_path, make_config(turns=3), clients={"p2": client}).run()

    retries = records(tmp_path, jl.PROVIDER_RETRY)
    assert len(retries) == 6  # two absorbed per turn, three turns
    assert {r["player_id"] for r in retries} == {"p2"}
    assert {r["error"] for r in retries} == {"Overloaded"}
    assert all(r["delay_s"] >= 0 for r in retries)
    # And the turn still succeeded, which is the point of absorbing it.
    assert not records(tmp_path, jl.AGENT_FAILURE)


async def test_resume_carries_prior_spend_so_the_cap_survives_a_crash(tmp_path: Path) -> None:
    """The budget cap belongs to the match, not to the process running it.

    A ledger that restarted at zero on resume let a run that crashed near its
    limit spend the whole cap a second time - and a run that crashed repeatedly
    had no limit at all. Found on a real interrupted shakeout, where the resumed
    process reported $0.22 for a match that had already spent $1.54.
    """
    clean, crashed = tmp_path / "clean", tmp_path / "crashed"
    first = await make_orchestrator(clean, make_config(turns=10)).run()
    assert first.ledger.spent_usd > 0

    crashed.mkdir()
    for name in ("journal.jsonl", "transcripts.jsonl"):
        (crashed / name).write_text((clean / name).read_text())
    crash_after(crashed, turn=6)

    resumed = await make_orchestrator(crashed, make_config(turns=10)).run(resume=True)
    # Everything the interrupted run was billed for, plus the four turns it
    # took to finish - not just the latter.
    assert resumed.ledger.spent_usd == pytest.approx(first.ledger.spent_usd, rel=0.01)
    assert set(resumed.ledger.usage_by_agent) == {"p1", "p2", "p3", "p4"}


async def test_a_resumed_match_still_halts_on_the_budget_cap(tmp_path: Path) -> None:
    """The consequence that matters: carried-forward spend has to actually arm
    the halt, not merely appear in the report."""
    clean, crashed = tmp_path / "clean", tmp_path / "crashed"
    first = await make_orchestrator(clean, make_config(turns=8)).run()

    crashed.mkdir()
    for name in ("journal.jsonl", "transcripts.jsonl"):
        (crashed / name).write_text((clean / name).read_text())
    crash_after(crashed, turn=4)

    # A cap the interrupted portion has already exhausted.
    tight = make_config(turns=8, budget_usd=first.ledger.spent_usd * 0.6)
    resumed = await make_orchestrator(crashed, tight).run(resume=True)
    assert resumed.reason == "budget_cap"


def empties_after(orchestrator, player_id: str, calls: int):
    """Wrap a seat so its account runs dry partway through the match.

    A truncated journal is the wrong shape for this. `crash_after` models a
    process killed between turns, which leaves no ending record at all; running
    out of credits is the opposite case - the loop notices, stops deliberately,
    and writes down why. The bug was in what that written-down reason meant.
    """
    from arena_orchestrator.providers.base import OutOfCredits

    real = orchestrator.clients[player_id]

    class Empty:
        name, model = real.name, real.model

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, system, user, schema):
            self.calls += 1
            if self.calls > calls:
                raise OutOfCredits("You have no credits remaining.", provider=real.name)
            return await real.complete(system, user, schema)

        async def aclose(self):
            return await real.aclose()

    seat = Empty()
    orchestrator.clients[player_id] = seat
    orchestrator.agents[player_id].client = seat
    return orchestrator


async def test_an_account_running_dry_stops_the_match_without_sealing_it(
    tmp_path: Path,
) -> None:
    """Topping up has to be enough to carry on. It was not.

    The halt and the resume were each right on their own and wrong together. A
    seat running out of credits stops the match - correctly, so it does not limp
    on as a three-way comparison - and the stop was recorded as `match_ended`,
    which is the same record a match writes when someone *wins*. Resume read
    that record, concluded there was nothing to continue, and refused. So the
    safeguard protecting a 300-turn run from a billing problem was also the
    thing that made a billing problem cost the whole run.

    Everything needed was already on disk: the journal is flushed per record, so
    every resolved turn survived. Only the permission to continue was missing.
    """
    root = tmp_path / "run"
    config = make_config(turns=12)
    halted = await empties_after(make_orchestrator(root, config), "p2", calls=5).run()

    assert halted.reason == "provider_credits"
    assert halted.state.turn == 5

    recovered = jl.recover(root)
    assert recovered.halted == "provider_credits"
    assert not recovered.ended, "a stopped match is not a finished one"
    assert recovered.resumable

    # The top-up: every seat can pay again.
    resumed = await make_orchestrator(root, config).run(resume=True)
    assert resumed.state.turn == 12
    assert resumed.reason == "turn_limit"


async def test_a_match_that_actually_finished_is_still_refused(tmp_path: Path) -> None:
    """The distinction has to hold in both directions.

    If every ending were resumable, resume would happily play turns past a
    victory - which is a different match, quietly.
    """
    root = tmp_path / "run"
    await make_orchestrator(root, make_config(turns=4)).run()

    recovered = jl.recover(root)
    assert recovered.ended and recovered.halted is None

    with pytest.raises(RuntimeError, match="already ended"):
        await make_orchestrator(root, make_config(turns=8)).run(resume=True)


async def test_a_resumed_match_reports_how_it_ended_not_how_it_stopped(
    tmp_path: Path,
) -> None:
    """Two `match_ended` records, and the later one is the answer.

    A match that halted, was topped up and then finished carries both. Reading
    the first would publish a completed run as having died on a billing problem
    it recovered from - and the bundle rebuild did exactly that, because `next()`
    over a journal finds the earliest match, not the last.
    """
    root = tmp_path / "run"
    config = make_config(turns=12)
    await empties_after(make_orchestrator(root, config), "p2", calls=5).run()
    await make_orchestrator(root, config).run(resume=True)

    assert len(records(root, jl.MATCH_ENDED)) == 2
    recovered = jl.recover(root)
    assert recovered.ended, "it finished this time"
    assert recovered.halted is None, "the halt it recovered from is not its outcome"


async def test_resume_carries_spend_per_seat_and_not_just_the_total(
    tmp_path: Path,
) -> None:
    """The spend column has to add up to the total printed above it.

    Resume carried `spent_usd` forward - the number the cap is enforced on, so
    the safety property was right - and seeded every seat's dollars at zero. The
    report then showed a per-seat column summing to half its own total, and
    `score_per_100k`, the efficiency figure this experiment exists to produce,
    divided a whole match's score by the spend of only the part played after the
    last interruption.

    Caught by running the CLI rather than the loop: the totals line and the
    table disagreed on screen, which no assertion about the ledger was looking
    at.
    """
    clean, halted = tmp_path / "clean", tmp_path / "halted"
    config = make_config(turns=10)
    reference = await make_orchestrator(clean, config).run()

    await empties_after(make_orchestrator(halted, config), "p2", calls=4).run()
    resumed = await make_orchestrator(halted, config).run(resume=True)

    assert sum(resumed.ledger.by_agent.values()) == pytest.approx(
        resumed.ledger.spent_usd, rel=0.01
    ), "the per-seat column does not add up to the total"
    for player_id, dollars in reference.ledger.by_agent.items():
        assert resumed.ledger.by_agent[player_id] == pytest.approx(dollars, rel=0.01), (
            f"{player_id} is missing what it spent before the interruption"
        )


async def test_a_replayed_turn_keeps_the_spend_it_was_billed(tmp_path: Path) -> None:
    """A resumed match must not report $0.00 for the half it already paid for.

    Bundle frames carry what each seat spent that turn, and replay rebuilt them
    from recorded decisions without the recorded costs - so every turn before
    the interruption rendered as free. The tokens were in the journal the whole
    time; nothing was reading them on this path.
    """
    root = tmp_path / "run"
    config = make_config(turns=10)
    await empties_after(make_orchestrator(root, config), "p2", calls=4).run()
    await make_orchestrator(root, config).run(resume=True)

    frames = sorted((root / "bundle" / "turns").glob("*.json"))
    assert len(frames) == 10
    # Turn 1 is replayed rather than played, which is the whole point.
    early = json.loads(frames[0].read_text())
    assert early["spend"], "a replayed turn came back with no spend at all"
    assert sum(seat["usd"] for seat in early["spend"].values()) > 0


async def test_raising_the_cap_lets_a_capped_match_carry_on(tmp_path: Path) -> None:
    """The dollar halt is the same shape of stop, and resumable on the same terms.

    Carried-forward spend still arms the cap, so continuing needs a cap above
    what the match has already spent - raising it is a deliberate act, not a
    reset.
    """
    root = tmp_path / "run"
    generous = make_config(turns=10, budget_usd=100.0)
    reference = await make_orchestrator(tmp_path / "reference", generous).run()
    tight = make_config(turns=10, budget_usd=reference.ledger.spent_usd * 0.5)

    capped = await make_orchestrator(root, tight).run()
    assert capped.reason == "budget_cap"
    assert jl.recover(root).resumable

    resumed = await make_orchestrator(root, generous).run(resume=True)
    assert resumed.state.turn == 10
    assert resumed.reason == "turn_limit"
    assert resumed.ledger.spent_usd == pytest.approx(reference.ledger.spent_usd, rel=0.01)


async def test_thinking_traces_are_persisted_and_kept_out_of_the_bundle(tmp_path: Path) -> None:
    """The trace is bought on every turn; it should not be thrown away.

    Reasoning tokens are billed as output and counted in `reasoning_tokens`, and
    three of the four adapters could already parse the trace out of the
    response. Nothing stored it, so every match paid for deliberation and kept
    none of it - and the one place it would have been noticed, the transcripts,
    was the file nobody thought to check.

    It goes to the transcripts and not the bundle, for the same reason the
    prompts do: a published match carries the match and nothing else.
    """
    orchestrator = make_orchestrator(
        tmp_path, make_config(turns=3), thinking="scripted deliberation"
    )
    await orchestrator.run()

    payloads = [
        json.loads(line) for line in (tmp_path / "transcripts.jsonl").read_text().splitlines()
    ]
    assert payloads, "no transcripts written at all"
    assert all(p.get("thinking") for p in payloads), "a turn was recorded with no trace"

    # Auditable from the journal without opening a single transcript, which is
    # the point: a seat billing reasoning tokens and storing none of them is a
    # real state and has to be visible.
    calls = [
        json.loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
        if '"agent_call"' in line
    ]
    assert calls and all(c["thinking_chars"] > 0 for c in calls)

    blob = "\n".join(
        path.read_text() for path in (tmp_path / "bundle").rglob("*") if path.is_file()
    )
    assert "scripted deliberation" not in blob, "the trace leaked into the published bundle"


async def test_a_vendor_that_offers_no_trace_still_records_the_turn(tmp_path: Path) -> None:
    """Absent means "not offered", not "did not think".

    Google's interactions surface bills thought tokens and exposes no thought
    text, so the field has to be genuinely optional rather than something the
    writer depends on.
    """
    await make_orchestrator(tmp_path, make_config(turns=2)).run()
    payloads = [
        json.loads(line) for line in (tmp_path / "transcripts.jsonl").read_text().splitlines()
    ]
    assert payloads and all("thinking" not in p for p in payloads)
    assert all(p["raw"] for p in payloads), "the rest of the record must survive"


async def test_an_agent_that_spends_its_allowance_stops_being_asked(tmp_path: Path) -> None:
    """The allowance has to bite, or it is decoration.

    It did not. `Allowance.exhausted()` was written and tested and never called
    from the loop, so the countdown reached zero, the observation kept saying
    `tokens_remaining: 0`, and the agent kept right on being asked - which makes
    the whole experiment a number printed on a page rather than a constraint a
    model has to plan against.

    Passing rather than eliminating is deliberate. A civ that vanishes hands its
    cities to nobody and rewrites the board for the other three, which would make
    *their* results depend on when this one ran dry. A civ that can no longer act
    still holds its territory and gets outcompeted in public.
    """
    config = make_config(turns=6, agent_budget_awareness="tokens", allowance_tokens=1)
    orchestrator = make_orchestrator(tmp_path, config)
    result = await orchestrator.run()

    # One token each: everyone is broke after their first turn.
    starved = records(tmp_path, jl.AGENT_FAILURE)
    assert starved, "running out of allowance left no trace at all"
    assert any("allowance exhausted" in r["reason"] for r in starved)

    # And it stopped costing money, which is the observable half.
    calls = collections.Counter(r["player_id"] for r in records(tmp_path, jl.AGENT_CALL))
    assert all(n <= 2 for n in calls.values()), f"a broke agent was still called: {calls}"
    assert result.state.turn == 6, "the match still finishes; nobody is removed"


async def test_the_allowance_does_nothing_when_the_experiment_is_off(tmp_path: Path) -> None:
    """Default off, because surfacing a countdown is known to sometimes trigger
    premature wrap-up. The baseline has to be a match nobody was rationing."""
    await make_orchestrator(tmp_path, make_config(turns=4)).run()
    assert not [
        r for r in records(tmp_path, jl.AGENT_FAILURE) if "allowance" in r.get("reason", "")
    ]
    calls = collections.Counter(r["player_id"] for r in records(tmp_path, jl.AGENT_CALL))
    assert all(n == 4 for n in calls.values()), "every seat plays every turn"


async def test_running_out_of_credits_halts_the_match(tmp_path: Path) -> None:
    """The one provider failure that ends the match instead of costing a turn.

    Every other failure here is survivable by design - an agent that cannot
    answer passes, plays badly, and the match continues, which is right for a
    vendor having a bad ten minutes. An exhausted account is not that: it will
    not recover inside the run, so "keep going" silently converts a billing
    problem into a corrupted result.

    Measured before this existed. A 300-turn baseline reached turn 40, one
    seat's credits ran out, and it played 29 further turns with that civ holding
    its cities and issuing no orders - producing a four-way comparison missing a
    fourth, at a cost of six hours and eighteen dollars if it had finished.
    """
    from arena_orchestrator.providers.base import OutOfCredits

    class Broke:
        name, model = "openai", "gpt-5.4-mini"

        async def complete(self, system, user, schema):
            raise OutOfCredits("You have no credits remaining.", provider="openai")

        async def aclose(self):
            return None

    config = make_config(turns=40)
    orchestrator = make_orchestrator(tmp_path, config, clients={"p2": Broke()})
    result = await orchestrator.run()

    assert result.reason == "provider_credits", "the match must stop, not limp"
    assert result.state.turn < 40, "it stopped early rather than playing on"
    # And it stopped *promptly* - the failure this guards is playing dozens of
    # turns with a dead seat, so a couple of turns is fine and thirty is not.
    assert result.state.turn <= 2, f"played {result.state.turn} turns with a broke seat"

    ended = records(tmp_path, jl.MATCH_ENDED)
    assert ended and ended[0]["reason"] == "provider_credits"
    assert any("OutOfCredits" in r["reason"] for r in records(tmp_path, jl.AGENT_FAILURE))


async def test_a_normal_provider_failure_still_only_costs_a_turn(tmp_path: Path) -> None:
    """The distinction has to hold in both directions, or the halt is just a
    crash with a nicer name. A transient outage must not end a multi-day match."""
    from arena_orchestrator.providers.base import Overloaded

    class Flaky:
        name, model = "openai", "gpt-5.4-mini"

        async def complete(self, system, user, schema):
            raise Overloaded("upstream is having a moment", provider="openai")

        async def aclose(self):
            return None

    config = make_config(turns=5)
    result = await make_orchestrator(tmp_path, config, clients={"p2": Flaky()}).run()
    assert result.reason == "turn_limit", "an outage must not end the match"
    assert result.state.turn == 5


async def test_an_unusable_order_earns_one_correction(tmp_path: Path) -> None:
    """The silent discard that eliminated a civilisation, now audible.

    A `found_city` with no unit and no name never reaches the reducer, so it
    gets no `order_rejected` - `actions.parse` discards it first. Nothing was
    logged, nothing failed, and from outside it looked exactly like a model
    choosing to do nothing. One seat spent 91% of its orders this way across
    128 turns and was eliminated on turn 35.

    It now costs one round trip and a message naming the missing fields.
    """
    from arena_orchestrator.providers.scripted import ScriptedClient

    good = {
        "reasoning": {
            "situation_assessment": "a",
            "threats_and_opportunities": [],
            "plan_this_turn": "b",
        },
        "dossier": {
            "doctrine": "",
            "opponent_models": [],
            "standing_commitments": [],
            "lessons": [],
        },
        "diplomacy": [],
        "orders": [{"action": "set_research", "tech": "pottery"}],
    }
    broken = {**good, "orders": [{"action": "found_city"}]}
    seen: list[str] = []

    def respond(user: str):
        seen.append(user)
        return broken if len(seen) == 1 else good

    config = make_config(turns=1)
    orchestrator = make_orchestrator(tmp_path, config, clients={"p1": ScriptedClient(respond)})
    await orchestrator.run()

    assert len(seen) == 2, "an unusable order should buy exactly one retry"
    # And the correction has to say what was wrong, or it is just a re-ask.
    assert "discarded" in seen[1]
    assert "found_city needs name, unit_id" in seen[1]
    assert records(tmp_path, jl.PARSE_REPAIRED), "the repair went unrecorded"


async def test_a_clean_turn_costs_exactly_one_call(tmp_path: Path) -> None:
    """The correction must not fire on turns that were fine - it would double
    the bill and the wall clock of every match for nothing."""
    from arena_orchestrator.providers.scripted import ScriptedClient

    good = {
        "reasoning": {
            "situation_assessment": "a",
            "threats_and_opportunities": [],
            "plan_this_turn": "b",
        },
        "dossier": {
            "doctrine": "",
            "opponent_models": [],
            "standing_commitments": [],
            "lessons": [],
        },
        "diplomacy": [],
        "orders": [{"action": "set_research", "tech": "pottery"}],
    }
    seen: list[str] = []

    def respond(user: str):
        seen.append(user)
        return good

    await make_orchestrator(
        tmp_path, make_config(turns=3), clients={"p1": ScriptedClient(respond)}
    ).run()
    assert len(seen) == 3, f"clean turns should be one call each, got {len(seen)}"
    assert not records(tmp_path, jl.PARSE_REPAIRED)
