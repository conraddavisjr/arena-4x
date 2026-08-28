#!/usr/bin/env python
"""Rebuild a match's replay bundle from its journal.

    python scripts/rebuild_bundle.py output/llm-shakeout

The journal records the decisions, not the board, and the engine is a pure
deterministic reducer - so `new_match(seed)` plus those decisions reproduces
every turn exactly. That is what makes resume work, and it means a *bundle* can
be regenerated the same way.

The point is that the bundle format can change without re-running anything. A
30-turn shakeout across four vendors costs real money and an hour of wall clock;
adding a field to a frame should not cost either. Before this existed, the only
way to get a new field into an old match was to play it again.

Verified against the recorded state hashes as it goes, so a rebuild that
diverges from the match that actually happened fails rather than quietly
producing a plausible different one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from arena_replay import BundleWriter  # noqa: E402

from arena_engine.actions import Action  # noqa: E402
from arena_engine.reducer import new_match, step  # noqa: E402
from arena_engine.types import MatchConfig  # noqa: E402
from arena_orchestrator import journal as jl  # noqa: E402
from arena_orchestrator.pricing import UnknownModel, cost_of  # noqa: E402
from arena_orchestrator.providers.base import Usage  # noqa: E402

CIVS = ["Aurelian Compact", "Iron Concord", "Verdant Pact", "Solari Dominion"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a run directory containing journal.jsonl")
    parser.add_argument(
        "--turn-limit",
        type=int,
        default=300,
        help="only for journals written before the match config was recorded",
    )
    args = parser.parse_args()

    recovered = jl.recover(args.run)
    if recovered.seed is None:
        raise SystemExit(f"no match_created record in {args.run}/journal.jsonl")

    created = next(r for r in jl.Journal(root=args.run).records() if r["type"] == jl.MATCH_CREATED)
    roster = [(s["player_id"], s["civ_name"]) for s in created["seats"]]

    # From the journal where it is recorded. `--turn-limit` is only for
    # journals written before the config was journalled at all, and getting it
    # wrong is caught below rather than producing a plausible wrong bundle.
    config = (
        MatchConfig.model_validate(recovered.match_config)
        if recovered.match_config
        else MatchConfig(turn_limit=args.turn_limit)
    )
    state, _ = new_match(recovered.match_id or "m1", recovered.seed, roster, config)
    models = {s["player_id"]: s.get("model") or s["provider"] for s in created["seats"]}
    writer = BundleWriter.start(args.run / "bundle", state, models)

    # Per-turn spend, recovered from the telemetry the journal already keeps.
    #
    # Recomputed from the token counts rather than summing the `cost_usd` each
    # record carries. The tokens are a measurement and cannot go stale; the
    # dollars were derived from a rate card that turned out to be wrong for eight
    # of its ten entries, understating every match by about half. Summing the
    # stored figure would carry that error forward into every rebuild forever,
    # and a bundle is exactly the artifact somebody trusts later.
    #
    # So the journal keeps what was measured, and money is derived on the way
    # out. Correcting a rate now reprices history simply by rebuilding.
    spend_by_turn: dict[int, dict] = {}
    repriced = 0.0
    as_billed = 0.0
    for r in jl.Journal(root=args.run).records():
        if r["type"] != jl.AGENT_CALL:
            continue
        row = spend_by_turn.setdefault(r["turn"], {}).setdefault(
            r["player_id"],
            {"usd": 0.0, "input": 0, "output": 0, "cached": 0, "ms": 0, "effort": None},
        )
        usage = Usage(
            input_tokens=r.get("input_tokens", 0),
            output_tokens=r.get("output_tokens", 0),
            cache_read_tokens=r.get("cache_read_tokens", 0),
            cache_write_tokens=r.get("cache_write_tokens", 0),
        )
        # A model with no rate card raises rather than costing zero, which is the
        # whole point of `cost_of` - but a *rebuild* should not be the thing that
        # fails on it, since the match already happened and its tokens are real.
        try:
            usd = cost_of(r["model"], usage)
        except UnknownModel:
            usd = r.get("cost_usd", 0.0)
        repriced += usd
        as_billed += r.get("cost_usd", 0.0)
        row["usd"] = round(row["usd"] + usd, 6)
        row["input"] += usage.input_tokens
        row["output"] += usage.output_tokens
        row["cached"] += usage.cache_read_tokens
        row["ms"] += r.get("latency_ms", 0)
        row["effort"] = r.get("effort")
        row["effort_sent"] = r.get("effort_sent")

    for record in recovered.turns:
        actions = {
            player_id: Action.model_validate(payload)
            for player_id, payload in record["actions"].items()
        }
        state, events = step(state, actions)
        if state.state_hash() != record["state_hash"]:
            raise SystemExit(
                f"replay diverged at turn {record['turn']}: the recorded decisions no longer "
                f"reproduce the recorded state. Refusing to write a bundle for a match that "
                f"did not happen."
            )
        writer.add(state, events, spend_by_turn.get(record["turn"], {}))

    # First `match_created`, *last* `match_ended`. A match that halted on a
    # billing problem, was topped up and resumed carries two end records, and
    # the one that says how it ended is the later one - taking the first would
    # publish a finished match as having stopped for a problem it recovered
    # from. One pass, because two `next()` calls over the same file read it
    # twice to answer half a question each.
    created = ended = None
    for record in jl.Journal(root=args.run).records():
        if record["type"] == jl.MATCH_CREATED and created is None:
            created = record
        elif record["type"] == jl.MATCH_ENDED:
            ended = record
    root = writer.finish(
        state,
        {
            "winner": ended.get("winner") if ended else None,
            "reason": ended.get("reason") if ended else "incomplete",
        },
        # Carried from the journal rather than stamped now. A rebuild is not a
        # new match, and dating it today would quietly relabel a run from last
        # week - which is precisely the kind of small lie a library of matches
        # cannot afford, because nothing downstream could ever catch it.
        finished_at=(ended or {}).get("finished_at"),
        started_at=(created or {}).get("started_at"),
        # The repriced total, not the one the run recorded at the time.
        spent_usd=round(repriced, 4),
    )
    print(f"{root}")
    print(f"  turns   {state.turn}")
    print(
        f"  outcome {(ended or {}).get('reason', 'incomplete')} -> "
        f"{(ended or {}).get('winner') or '-'}"
    )
    print("  hashes  verified against the journal at every turn")
    if as_billed and abs(repriced - as_billed) > 0.005:
        print(
            f"  cost    ${repriced:.2f} at current rates, recorded as ${as_billed:.2f} "
            f"during the run ({repriced / as_billed:.1f}x)"
        )
    else:
        print(f"  cost    ${repriced:.2f}")


if __name__ == "__main__":
    main()
