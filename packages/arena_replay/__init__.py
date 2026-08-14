"""Turn a finished match into the static data a viewer can replay.

This is the foundation of both the live dashboard and the publishable export,
because of a decision made deliberately up front: **the viewer is a pure
function of this bundle, and "live" is just the case where the bundle is still
growing.** Building it the other way round - a socket-driven UI with replay
bolted on afterwards - would produce two code paths and an exported viewer that
drifts from the one actually used to watch matches.

Everything a viewer needs is already captured, because the engine is
deterministic and the event log is the source of truth. Nothing extra has to be
recorded to make a match publishable.

## Shape

    match.json      metadata, the roster, and the terrain map exactly once
    turns/NNN.json  one file per turn: what changed, who saw what, who said what
    stats.json      the match dossier

Per-turn files rather than one blob. A 200-turn match carries roughly 800 agent
reasoning payloads, and a single file would be tens of megabytes - the page
would stall before showing anything. Split this way a viewer loads metadata plus
the current turn and prefetches neighbours.

## What is deliberately excluded

Full prompts and raw model responses. They are the bulk of the data (~24KB per
agent-turn, so ~20MB for a match), they are the least interesting part to a
visitor, and keeping them out means a published match cannot leak system-prompt
internals. They stay in Postgres for analysis.

## Why terrain lives in match.json

Terrain almost never changes, so repeating 1027 tiles every turn would dominate
the bundle. It is written once as an ordered array, and everything positional
afterwards - visibility above all - refers to tiles by index into that array
rather than by `"q,r"` string. Visibility is the largest per-turn structure by
far (four civs times a few hundred tiles), and integers are roughly a quarter
the size of quoted coordinate strings.
"""

from arena_replay.bundle import (
    BundleWriter,
    build_bundle,
    match_metadata,
    turn_frame,
)

__all__ = ["BundleWriter", "build_bundle", "match_metadata", "turn_frame"]
