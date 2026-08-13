"""Axial hex coordinates.

Pointy-top hexes in axial coordinates `(q, r)`, with the third cube coordinate
`s = -q - r` derived when needed. Axial is the storage format because it is two
integers rather than three; cube is the arithmetic format because distance and
rotation are trivial in it.

`Hex` is a NamedTuple, so it is a plain tuple at runtime: hashable, usable as a
dict key, cheap to copy, and it compares and sorts structurally. The engine
holds these; only the observation layer converts to the `"q,r"` wire format.
"""

from __future__ import annotations

from typing import NamedTuple


class Hex(NamedTuple):
    q: int
    r: int

    @property
    def s(self) -> int:
        """The derived third cube coordinate. Always `q + r + s == 0`."""
        return -self.q - self.r

    def __add__(self, other: Hex) -> Hex:  # type: ignore[override]
        return Hex(self.q + other.q, self.r + other.r)

    def __sub__(self, other: Hex) -> Hex:
        return Hex(self.q - other.q, self.r - other.r)

    def scale(self, k: int) -> Hex:
        return Hex(self.q * k, self.r * k)

    def to_key(self) -> str:
        """The wire format used in observations: `"3,-1"`."""
        return f"{self.q},{self.r}"


def from_key(key: str) -> Hex:
    """Parse the `"q,r"` wire format back into a Hex.

    Raises ValueError on anything malformed, which is what the order validator
    wants: a model that invents a coordinate string gets a rejected order
    rather than a crash.
    """
    q_str, _, r_str = key.partition(",")
    if not _:
        raise ValueError(f"malformed hex key: {key!r}")
    return Hex(int(q_str), int(r_str))


# Clockwise from east. Index order is stable and part of the engine's
# determinism contract: anything that iterates neighbours iterates them in this
# order, so a seeded run is reproducible.
DIRECTIONS: tuple[Hex, ...] = (
    Hex(1, 0),
    Hex(1, -1),
    Hex(0, -1),
    Hex(-1, 0),
    Hex(-1, 1),
    Hex(0, 1),
)

ORIGIN = Hex(0, 0)


def neighbors(h: Hex) -> tuple[Hex, ...]:
    """The six adjacent hexes, in DIRECTIONS order."""
    return tuple(h + d for d in DIRECTIONS)


def neighbor(h: Hex, direction: int) -> Hex:
    return h + DIRECTIONS[direction % 6]


def distance(a: Hex, b: Hex) -> int:
    """Hex grid distance, which is the cube Chebyshev-style metric halved."""
    dq = a.q - b.q
    dr = a.r - b.r
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2


def ring(center: Hex, radius: int) -> list[Hex]:
    """The hexes exactly `radius` away, walked counter-clockwise from one corner.

    A radius of 0 is the centre itself, which is the degenerate case callers
    forget; returning `[center]` rather than `[]` keeps `spiral` simple.
    """
    if radius < 0:
        raise ValueError(f"ring radius must be non-negative, got {radius}")
    if radius == 0:
        return [center]

    results: list[Hex] = []
    # Start on the south-west corner and walk each of the six edges.
    current = center + DIRECTIONS[4].scale(radius)
    for direction in range(6):
        for _ in range(radius):
            results.append(current)
            current = neighbor(current, direction)
    return results


def spiral(center: Hex, radius: int) -> list[Hex]:
    """Every hex within `radius`, ordered centre-outward.

    Deterministic ordering matters here: map generation and vision both walk
    this, and a set-based implementation would reorder between runs.
    """
    results: list[Hex] = []
    for k in range(radius + 1):
        results.extend(ring(center, k))
    return results


def within(center: Hex, radius: int) -> set[Hex]:
    """Every hex within `radius`, as a set. Use when order does not matter."""
    return {
        Hex(center.q + dq, center.r + dr)
        for dq in range(-radius, radius + 1)
        for dr in range(max(-radius, -dq - radius), min(radius, -dq + radius) + 1)
    }


def line(a: Hex, b: Hex) -> list[Hex]:
    """Hexes along the straight line from `a` to `b`, inclusive of both.

    Used for reachability checks and for drawing contact arcs on the dashboard.
    The epsilon nudge avoids ties landing exactly on an edge, which would make
    rounding depend on floating-point noise rather than on geometry.
    """
    n = distance(a, b)
    if n == 0:
        return [a]
    results: list[Hex] = []
    for i in range(n + 1):
        t = i / n
        q = a.q + (b.q - a.q) * t + 1e-6
        r = a.r + (b.r - a.r) * t + 1e-6
        results.append(_round_to_hex(q, r))
    return results


def _round_to_hex(q: float, r: float) -> Hex:
    """Round fractional axial coordinates to the nearest hex.

    Rounds in cube space and repairs whichever component moved furthest, which
    is the standard way to keep `q + r + s == 0` after rounding.
    """
    s = -q - r
    rq, rr, rs = round(q), round(r), round(s)
    dq, dr, ds = abs(rq - q), abs(rr - r), abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    return Hex(int(rq), int(rr))


def rotate_60(h: Hex, times: int = 1) -> Hex:
    """Rotate a hex around the origin in 60-degree steps.

    This is what gives the four starting positions their exact symmetry: each
    player's start is a rotation of the first, so no player can be handed a
    better neighbourhood than another.
    """
    q, r, s = h.q, h.r, h.s
    for _ in range(times % 6):
        q, r, s = -r, -s, -q
    return Hex(q, r)
