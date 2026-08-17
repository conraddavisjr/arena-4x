#!/usr/bin/env python
"""Index every finished match under a directory, for the viewer's library.

    python scripts/build_library.py output

Writes `library.json` beside the runs it found. The viewer reads it to list past
matches; without it the viewer still works and simply has no library.

The index is derived, never authoritative. Everything in it comes from each
bundle's own `match.json`, so deleting `library.json` and rebuilding it is always
safe and a hand-edit is always pointless. That matters because the alternative -
a registry that matches are *registered into* - drifts the first time somebody
moves a directory, and then lists matches that are not there while omitting ones
that are.

**What is deliberately not here: names and stars.** Those are the viewer's, kept
in the browser. A published bundle is a static directory with no write path, so
the honest options were a server (which the viewer exists not to need) or a CLI
that renames on disk (which nobody would run mid-scrub). Browser-local
annotation keeps the viewer inert and still lets a name be typed and kept. The
cost is that it is per-browser and does not travel with a published match, which
is the right trade for a lab notebook and the wrong one for a shared catalogue -
noted here so the day that changes, the reason is on record.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def dated(bundle: Path, meta: dict[str, Any]) -> str:
    """When the match finished, best available.

    Recorded runs carry `finished_at`. The two that predate the timestamp keep
    their real date through the file's mtime, which is close enough to be useful
    and honest enough not to pretend otherwise - it is when the bundle was last
    written, and for those runs that is when they were rebuilt rather than when
    they were played.
    """
    stamped = meta.get("finished_at")
    if stamped:
        return str(stamped)
    return (
        datetime.fromtimestamp(bundle.stat().st_mtime, UTC).replace(microsecond=0).isoformat()
        + "~"  # marks an inferred date, so the viewer can say so
    )


def entry(root: Path, bundle: Path) -> dict[str, Any] | None:
    meta_path = bundle / "match.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text())
    victory = meta.get("victory") or {}
    civs = meta.get("civs") or []
    scores = victory.get("scores") or {}
    winner = victory.get("winner")
    named = {c["player_id"]: c.get("model") or c.get("civ_name") for c in civs}
    return {
        # Relative, because the viewer loads it as `?match=<path>` from the page
        # it was served on. An absolute path would work locally and break the
        # moment the tree is served from anywhere else.
        "path": str(bundle.relative_to(root)),
        "id": bundle.parent.name,
        "turns": meta.get("turns") or meta.get("final_turn") or 0,
        "finished_at": dated(bundle, meta),
        "seed": meta.get("seed"),
        "condition": victory.get("condition"),
        "winner": winner,
        "winner_name": named.get(winner),
        "models": [named[p] for p in sorted(named)],
        "scores": {named[p]: s for p, s in sorted(scores.items()) if p in named},
        "spent_usd": meta.get("spent_usd"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("output"))
    args = parser.parse_args()
    root: Path = args.root

    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")

    # Any depth, so a bundle nested in a dated folder is found without anyone
    # having to remember a layout convention.
    found = [e for bundle in sorted(root.rglob("bundle")) if (e := entry(root, bundle))]
    # Newest first. The viewer sorts starred entries above these, but an index
    # with no useful default order would make that the viewer's problem twice.
    found.sort(key=lambda e: e["finished_at"], reverse=True)

    out = root / "library.json"
    out.write_text(json.dumps({"matches": found}, indent=1, sort_keys=True) + "\n")
    print(f"{out}  {len(found)} match{'' if len(found) == 1 else 'es'}")
    for e in found:
        end = e["condition"] or "incomplete"
        who = e["winner_name"] or "-"
        print(f"  {e['finished_at'][:10]}  {e['id']:<16} {e['turns']:>4} turns  {end} -> {who}")


if __name__ == "__main__":
    main()
