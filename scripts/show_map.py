"""Render a generated map to the terminal.

The tests assert that starts are fair and the land is connected, but neither
tells you whether the map is any *good* to play on. This is for looking at it.

    python scripts/show_map.py --seed 42 --radius 12
"""

from __future__ import annotations

import argparse
from collections import Counter

from arena_engine import hex as hx
from arena_engine import mapgen
from arena_engine.content import TERRAIN, Terrain
from arena_engine.types import MatchConfig

GLYPH: dict[Terrain, str] = {
    Terrain.OCEAN: "\x1b[34m~\x1b[0m",
    Terrain.COAST: "\x1b[36m-\x1b[0m",
    Terrain.GRASSLAND: '\x1b[92m"\x1b[0m',
    Terrain.PLAINS: "\x1b[33m.\x1b[0m",
    Terrain.FOREST: "\x1b[32m*\x1b[0m",
    Terrain.HILLS: "\x1b[37mn\x1b[0m",
    Terrain.DESERT: "\x1b[93m:\x1b[0m",
    Terrain.MOUNTAINS: "\x1b[90m^\x1b[0m",
}
START_COLORS = ["\x1b[1;92m", "\x1b[1;33m", "\x1b[1;96m", "\x1b[1;95m"]
RESOURCE_MARK = "\x1b[1;91mo\x1b[0m"


def render(seed: int, radius: int, show_resources: bool) -> str:
    m = mapgen.generate(seed, radius)
    starts = {h: i for i, h in enumerate(m.starts)}
    lines: list[str] = []

    for r in range(-radius, radius + 1):
        # Pointy-top rows shear by r/2, so indent to keep the hexagon readable.
        indent = " " * abs(r)
        row: list[str] = []
        q_lo = max(-radius, -r - radius)
        q_hi = min(radius, -r + radius)
        for q in range(q_lo, q_hi + 1):
            h = hx.Hex(q, r)
            tile = m.tiles[h.to_key()]
            if h in starts:
                row.append(f"{START_COLORS[starts[h]]}{starts[h] + 1}\x1b[0m")
            elif show_resources and tile.resource is not None:
                row.append(RESOURCE_MARK)
            else:
                row.append(GLYPH[tile.terrain])
        lines.append(indent + " ".join(row))

    counts = Counter(t.terrain for t in m.tiles.values())
    land = sum(n for t, n in counts.items() if TERRAIN[t].passable)
    resources = sum(1 for t in m.tiles.values() if t.resource)

    lines.append("")
    lines.append(f"seed {seed}  radius {radius}  tiles {len(m.tiles)}")
    lines.append(f"land {land / len(m.tiles):.0%}  resources {resources}")
    lines.append("  ".join(f"{t.value}={counts[t]}" for t in Terrain if counts[t]))
    lines.append(
        "starts  "
        + "  ".join(f"{START_COLORS[i]}{i + 1}\x1b[0m {s.to_key()}" for i, s in enumerate(m.starts))
    )
    profile = sorted(hx.distance(m.starts[0], b) for b in m.starts[1:])
    lines.append(f"distance profile from each start: {profile}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    # Follow MatchConfig rather than repeating the number, so the viewer can
    # never drift from the size matches actually run at.
    ap.add_argument("--radius", type=int, default=MatchConfig().radius)
    ap.add_argument("--resources", action="store_true", help="mark resource tiles")
    args = ap.parse_args()
    print(render(args.seed, args.radius, args.resources))


if __name__ == "__main__":
    main()
