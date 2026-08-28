"""The append-only log a match can be rebuilt from.

**Resume works by replaying decisions, not by restoring state.** The engine is a
pure deterministic reducer, so `new_match(seed)` plus the recorded actions for
turns 1..N reproduces turn N exactly - byte for byte, verified by state hash.
That is a much better deal than snapshotting: the log is small, the recovery
path is the same code as the normal path, and it exercises the determinism
property continuously instead of only in a test.

The hash is recorded with every turn precisely so resume can *check* rather than
assume. If a replay diverges - an engine change between crash and restart, a
corrupted line - it fails loudly at the turn it diverged rather than quietly
continuing a different match.

Two files, because they have different readers:

  `journal.jsonl`      decisions, telemetry, outcomes. Read on resume.
  `transcripts.jsonl`  the exact prompt sent and the raw body returned.

The split is not tidiness. Transcripts are roughly 24KB per agent-turn - about
20MB for a full match, the great majority of everything written - and resume has
to scan the journal on every restart. Keeping them apart means recovery reads
kilobytes rather than tens of megabytes, and it means the forensic record can be
deleted or withheld without touching replayability. It is also why the published
bundle can exclude prompts without losing anything: they were never load-bearing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Record types the orchestrator writes. The engine's own event vocabulary lives
# in arena_engine.events; these are things that happen to the *system* rather
# than in the game, and keeping the two lists apart is what stops the game log
# filling up with HTTP.
MATCH_CREATED = "match_created"
TURN_RESOLVED = "turn_resolved"
AGENT_CALL = "agent_call"
AGENT_FAILURE = "agent_failure"
PARSE_REPAIRED = "parse_repaired"
PROVIDER_RETRY = "provider_retry"
THROTTLED = "throttled"
CACHE_MISS = "cache_miss"
BUDGET_UPDATED = "budget_updated"
MATCH_ENDED = "match_ended"

# `match_ended` covers two things that are not the same, and treating them alike
# turned a recoverable billing problem into a lost run.
#
# A match that *finished* has a game outcome: someone won, or the turn limit
# arrived. There is nothing to resume, and asking to resume one is an operator
# error worth refusing.
#
# A match that *halted* stopped for a reason outside the game - an account that
# could not pay, or the dollar cap. The board is coherent and scoreable, which
# is why the halt exists, but the match is not over: the condition that stopped
# it is one a human fixes with a billing page or a flag. Sealing those as ended
# meant a seat running dry on turn 240 cost the whole run, which is the exact
# outcome the halt was added to prevent - it stopped the match limping, and then
# stopped it continuing.
#
# Resume is still explicit and still human-initiated. What changes is that the
# journal no longer says a stopped match is a finished one.
HALTS = frozenset({"provider_credits", "budget_cap"})


@dataclass
class Journal:
    """Append-only, flushed per record.

    Flushing every write costs a syscall per record and buys the property the
    whole resume path depends on: if the process is killed between turns, what
    is on disk is what happened. A buffered writer would lose the last turns -
    exactly the ones needed to work out what went wrong.
    """

    root: Path
    _seq: int = 0
    _log: Any = field(default=None, repr=False)
    _transcripts: Any = field(default=None, repr=False)

    @classmethod
    def open(cls, root: Path, *, resume: bool = False) -> Journal:
        root.mkdir(parents=True, exist_ok=True)
        journal = cls(root=root)
        if resume:
            journal._seq = sum(1 for _ in journal.records())
        mode = "a" if resume else "w"
        journal._log = (root / "journal.jsonl").open(mode, encoding="utf-8")
        journal._transcripts = (root / "transcripts.jsonl").open(mode, encoding="utf-8")
        return journal

    def append(self, kind: str, turn: int, **payload: Any) -> int:
        self._seq += 1
        record = {"seq": self._seq, "type": kind, "turn": turn, **payload}
        self._log.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log.flush()
        return self._seq

    def transcript(
        self,
        turn: int,
        player_id: str,
        system: str,
        user: str,
        raw: str,
        thinking: str | None = None,
    ) -> None:
        """The forensic record: exactly what was sent, exactly what came back.

        Never read by resume, never included in a published bundle. It is here
        so that "why did p3 do that on turn 47" is answerable months later.

        `thinking` is the model's reasoning trace, which is a different artifact
        from the `reasoning` block in `raw`. That block is the account the model
        writes *for the record*, knowing it will be read and handed back; this is
        the deliberation behind it. Every seat was already paying for these
        tokens - they are billed as output and counted in `reasoning_tokens` -
        and the adapters were already parsing them out of the response. Nothing
        stored them, so on every turn of every match they were bought and thrown
        away.

        Kept here rather than in the bundle for the same reason the prompts are:
        a published match should carry the match and nothing else. Vendors also
        differ on what they will even show - Anthropic returns summarised
        thinking blocks, OpenAI returns reasoning summaries only if asked,
        xAI puts a trace on the message, and Google's interactions surface
        exposes no thought text at all while still billing for it - so an
        absent trace here means "not offered", not "did not think".
        """
        self._transcripts.write(
            json.dumps(
                {
                    "turn": turn,
                    "player_id": player_id,
                    "system": system,
                    "user": user,
                    "raw": raw,
                    **({"thinking": thinking} if thinking else {}),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._transcripts.flush()

    def records(self) -> Iterator[dict[str, Any]]:
        path = self.root / "journal.jsonl"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def close(self) -> None:
        for handle in (self._log, self._transcripts):
            if handle is not None:
                handle.close()


@dataclass(frozen=True, slots=True)
class Recovered:
    """What a journal says about a match that was interrupted."""

    seed: int | None
    match_id: str | None
    # The MatchConfig the run was played under. Part of the state hash, so a
    # rebuild or resume that guesses it produces a different match.
    match_config: dict[str, Any] | None
    turns: list[dict[str, Any]]
    # A game outcome was recorded: won, or the turn limit. Not resumable.
    ended: bool
    # The match stopped for an operational reason, named here. Resumable once
    # the reason is dealt with - credits topped up, or the cap raised.
    halted: str | None
    # What the interrupted run had already spent. Carried forward because the
    # budget cap is a cap on the *match*, not on the current process: a ledger
    # that restarts at zero lets a match that crashes near its limit resume and
    # spend the whole cap again, and a match that crashes repeatedly has no
    # limit at all.
    spent_usd: float = 0.0
    # Per seat, and not merely the total split for display. Resume seeded the
    # per-agent dollars at zero while carrying the total, so a resumed match
    # reported a spend column that did not add up to its own total - and
    # `score_per_100k`, the efficiency figure the whole experiment reports,
    # divided a full match's score by half a match's spend.
    spent_by_agent: dict[str, float] = field(default_factory=dict)
    usage_by_agent: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-turn, per-seat spend, in the shape the bundle writer wants. Replaying
    # a match rebuilds its frames, and frames carry what each seat spent that
    # turn - so without this a resumed run's cost panel reads $0.00 for every
    # turn before the interruption. The tokens were never lost, only unread.
    spend_by_turn: dict[int, dict[str, dict[str, Any]]] = field(default_factory=dict)

    @property
    def last_turn(self) -> int:
        return self.turns[-1]["turn"] if self.turns else 0

    @property
    def resumable(self) -> bool:
        """Is there a match here to carry on with?

        A halted match is; a finished one is not; and a journal that stops
        mid-turn - a killed process - has neither record and is the ordinary
        crash-resume case that always worked.
        """
        return bool(self.turns) and not self.ended


def recover(root: Path) -> Recovered:
    """Read a journal back into the decisions needed to rebuild the match.

    Only complete `turn_resolved` records count. A turn whose record was never
    written did not happen as far as recovery is concerned, which is what makes
    the restart idempotent: the worst case is re-playing one turn that four
    agents had already been billed for, not resuming into a state no log
    describes.
    """
    journal = Journal(root=root)
    seed: int | None = None
    match_id: str | None = None
    match_config: dict[str, Any] | None = None
    turns: list[dict[str, Any]] = []
    outcome: str | None = None
    spent = 0.0
    by_agent: dict[str, float] = {}
    usage: dict[str, dict[str, int]] = {}
    spend_by_turn: dict[int, dict[str, dict[str, Any]]] = {}
    for record in journal.records():
        if record["type"] == AGENT_CALL:
            spent += record.get("cost_usd", 0.0)
            by_agent[record["player_id"]] = round(
                by_agent.get(record["player_id"], 0.0) + record.get("cost_usd", 0.0), 6
            )
            seat = usage.setdefault(record["player_id"], {})
            for field_name in ("input_tokens", "output_tokens", "cache_read_tokens"):
                seat[field_name] = seat.get(field_name, 0) + record.get(field_name, 0)
            row = spend_by_turn.setdefault(record["turn"], {}).setdefault(
                record["player_id"],
                {"usd": 0.0, "input": 0, "output": 0, "cached": 0, "ms": 0, "effort": None},
            )
            # As billed at the time. A rebuild reprices from tokens against the
            # current rate card, which is the right thing for a published
            # artifact; resume is continuing a run rather than restating it, and
            # the ledger it is carrying forward is the one that was charged.
            row["usd"] = round(row["usd"] + record.get("cost_usd", 0.0), 6)
            row["input"] += record.get("input_tokens", 0)
            row["output"] += record.get("output_tokens", 0)
            row["cached"] += record.get("cache_read_tokens", 0)
            row["ms"] += record.get("latency_ms", 0)
            row["effort"] = record.get("effort")
            row["effort_sent"] = record.get("effort_sent")
        if record["type"] == MATCH_CREATED:
            seed = record.get("seed")
            # Recovered rather than re-derived from the directory name. The
            # match id is inside the state hash, so a run directory that was
            # moved or copied - which is exactly what an operator does while
            # recovering one - would otherwise replay to a different hash and
            # look like corruption.
            match_id = record.get("match_id")
            match_config = record.get("match_config")
        elif record["type"] == TURN_RESOLVED:
            turns.append(record)
        elif record["type"] == MATCH_ENDED:
            # Last one wins. A match that halted, was resumed and then finished
            # carries two of these, and the one that describes how it ended is
            # the later one - reading the first would report a resumed match as
            # having stopped for a billing problem it recovered from.
            outcome = record.get("reason")
    turns.sort(key=lambda r: r["turn"])
    return Recovered(
        seed=seed,
        match_id=match_id,
        match_config=match_config,
        turns=turns,
        ended=outcome is not None and outcome not in HALTS,
        halted=outcome if outcome in HALTS else None,
        spent_usd=round(spent, 6),
        spent_by_agent=by_agent,
        usage_by_agent=usage,
        spend_by_turn=spend_by_turn,
    )
