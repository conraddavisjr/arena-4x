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
BUDGET_UPDATED = "budget_updated"
MATCH_ENDED = "match_ended"


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

    def transcript(self, turn: int, player_id: str, system: str, user: str, raw: str) -> None:
        """The forensic record: exactly what was sent, exactly what came back.

        Never read by resume, never included in a published bundle. It is here
        so that "why did p3 do that on turn 47" is answerable months later.
        """
        self._transcripts.write(
            json.dumps(
                {
                    "turn": turn,
                    "player_id": player_id,
                    "system": system,
                    "user": user,
                    "raw": raw,
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
    turns: list[dict[str, Any]]
    ended: bool
    # What the interrupted run had already spent. Carried forward because the
    # budget cap is a cap on the *match*, not on the current process: a ledger
    # that restarts at zero lets a match that crashes near its limit resume and
    # spend the whole cap again, and a match that crashes repeatedly has no
    # limit at all.
    spent_usd: float = 0.0
    usage_by_agent: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def last_turn(self) -> int:
        return self.turns[-1]["turn"] if self.turns else 0


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
    turns: list[dict[str, Any]] = []
    ended = False
    spent = 0.0
    usage: dict[str, dict[str, int]] = {}
    for record in journal.records():
        if record["type"] == AGENT_CALL:
            spent += record.get("cost_usd", 0.0)
            seat = usage.setdefault(record["player_id"], {})
            for field_name in ("input_tokens", "output_tokens", "cache_read_tokens"):
                seat[field_name] = seat.get(field_name, 0) + record.get(field_name, 0)
        if record["type"] == MATCH_CREATED:
            seed = record.get("seed")
            # Recovered rather than re-derived from the directory name. The
            # match id is inside the state hash, so a run directory that was
            # moved or copied - which is exactly what an operator does while
            # recovering one - would otherwise replay to a different hash and
            # look like corruption.
            match_id = record.get("match_id")
        elif record["type"] == TURN_RESOLVED:
            turns.append(record)
        elif record["type"] == MATCH_ENDED:
            ended = True
    turns.sort(key=lambda r: r["turn"])
    return Recovered(
        seed=seed,
        match_id=match_id,
        turns=turns,
        ended=ended,
        spent_usd=round(spent, 6),
        usage_by_agent=usage,
    )
