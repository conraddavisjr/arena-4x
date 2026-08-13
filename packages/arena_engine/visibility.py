"""Vision, fog of war, and contact.

Two things are computed here and they are deliberately *not* the same thing.

**Visibility** is geometry: which tiles a civ can currently see. Two civs can
have heavily overlapping vision of the same empty valley and remain completely
unaware of one another.

**Contact** is awareness: whether a civ can currently see any unit or city
belonging to another. It is *directed* and frequently asymmetric - a scout that
spots an army while itself sitting unobserved creates contact in one direction
only - and that asymmetry is usually the most decision-relevant fact on the
board. It is also invisible in any single-civ view, which is exactly why the
dashboard's mixed view needs both layers rather than just tinting fog.

Mountains block line of sight, and standing on high ground extends it. That is
what turns the mountain spines the generator produces into real strategic
features: armies can genuinely hide behind them, and taking a ridge buys
information as well as defence.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena_engine import hex as hx
from arena_engine.content import CITY_VISION, UNITS, Terrain
from arena_engine.hex import Hex
from arena_engine.types import RememberedTile, State

# Terrain that stops sight passing *through* it. Hills obstruct nothing on
# their own; only mountains do, which keeps the rule easy for an agent to reason
# about while still making ridges matter.
BLOCKING: frozenset[Terrain] = frozenset({Terrain.MOUNTAINS})

# Standing on high ground sees further.
ELEVATED: frozenset[Terrain] = frozenset({Terrain.HILLS, Terrain.MOUNTAINS})
ELEVATION_BONUS = 1


@dataclass(frozen=True, slots=True)
class Sighting:
    """One directed observation, for the dashboard's contact arcs.

    Carries both endpoints so the arc can be drawn from the watcher to the
    watched rather than merely recording that the pair are in contact.
    """

    observer: str
    observed: str
    observer_pos: Hex
    observed_pos: Hex
    asset: str  # "unit" or "city"


@dataclass(frozen=True, slots=True)
class VisionReport:
    visibility: dict[str, set[Hex]]
    contact: dict[str, set[str]]
    sightings: list[Sighting]

    def sees(self, observer: str, observed: str) -> bool:
        return observed in self.contact.get(observer, set())

    def mutual(self, a: str, b: str) -> bool:
        return self.sees(a, b) and self.sees(b, a)

    def one_way(self, a: str, b: str) -> bool:
        """True when `a` sees `b` but `b` does not see `a`."""
        return self.sees(a, b) and not self.sees(b, a)


def visible_from(state: State, origin: Hex, radius: int) -> set[Hex]:
    """Tiles visible from `origin`, accounting for elevation and blockers.

    The observer's own tile is always visible even if it is a mountain, and a
    blocking tile is itself visible - you can see the ridge, just not past it.
    """
    tile = state.at(origin)
    if tile is not None and tile.terrain in ELEVATED:
        radius += ELEVATION_BONUS

    seen = {origin}
    for target in hx.within(origin, radius):
        if target in seen or state.at(target) is None:
            continue
        if _unobstructed(state, origin, target):
            seen.add(target)
    return seen


def _unobstructed(state: State, origin: Hex, target: Hex) -> bool:
    """Walk the line and stop at the first blocker strictly between the ends."""
    path = hx.line(origin, target)
    for step in path[1:-1]:
        tile = state.at(step)
        if tile is not None and tile.terrain in BLOCKING:
            return False
    return True


def compute(state: State) -> VisionReport:
    """Compute visibility and contact for every living civ.

    Sorted iteration throughout: the sightings list ends up in the event log,
    and an unordered one would make replays differ from live runs.
    """
    visibility: dict[str, set[Hex]] = {p: set() for p in state.living_player_ids()}

    for _, unit in sorted(state.units.items()):
        if unit.owner not in visibility:
            continue
        visibility[unit.owner] |= visible_from(state, unit.hex, UNITS[unit.type].vision)

    for _, city in sorted(state.cities.items()):
        if city.owner not in visibility:
            continue
        visibility[city.owner] |= visible_from(state, city.hex, CITY_VISION)

    # Index assets by tile once rather than scanning every unit per observer.
    assets: dict[Hex, list[tuple[str, str]]] = {}
    for _, unit in sorted(state.units.items()):
        assets.setdefault(unit.hex, []).append((unit.owner, "unit"))
    for _, city in sorted(state.cities.items()):
        assets.setdefault(city.hex, []).append((city.owner, "city"))

    contact: dict[str, set[str]] = {p: set() for p in visibility}
    sightings: list[Sighting] = []
    for observer in sorted(visibility):
        for target in sorted(visibility[observer]):
            for owner, kind in assets.get(target, ()):
                if owner == observer or owner not in visibility:
                    continue
                contact[observer].add(owner)
                sightings.append(
                    Sighting(
                        observer=observer,
                        observed=owner,
                        # The arc is drawn to the sighted asset; the observing
                        # end is resolved by the dashboard from the nearest
                        # owned asset, which keeps this payload small.
                        observer_pos=target,
                        observed_pos=target,
                        asset=kind,
                    )
                )
    return VisionReport(visibility=visibility, contact=contact, sightings=sightings)


def apply(state: State, report: VisionReport) -> None:
    """Write the report onto the state and refresh each civ's remembered map.

    Mutates in place; the reducer owns when this is called. Storing the derived
    view on the snapshot is what lets the dashboard replay contact history
    across the whole match instead of only showing the live turn.
    """
    state.visibility = {
        player: sorted(h.to_key() for h in tiles)
        for player, tiles in sorted(report.visibility.items())
    }
    state.contact = {observer: sorted(seen) for observer, seen in sorted(report.contact.items())}
    _remember(state, report)


def _remember(state: State, report: VisionReport) -> None:
    """Fold currently-visible tiles into each civ's persistent memory.

    Memory is deliberately lossy: terrain and resources persist, live unit
    positions do not. An agent should have to reason about how stale its
    picture is, which is what `last_seen_turn` is for. Enemy cities are the one
    exception, recorded as a note, because a city does not move and forgetting
    one entirely would make the map unusable for planning.
    """
    city_by_pos = {c.pos: c for _, c in sorted(state.cities.items())}

    for player_id, tiles in sorted(report.visibility.items()):
        player = state.players[player_id]
        for h in sorted(tiles):
            key = h.to_key()
            tile = state.tiles.get(key)
            if tile is None:
                continue
            note = None
            city = city_by_pos.get(key)
            if city is not None and city.owner != player_id:
                note = (
                    f"{state.players[city.owner].civ_name} city "
                    f"'{city.name}', population {city.population}"
                )
            player.memory[key] = RememberedTile(
                terrain=tile.terrain,
                resource=tile.resource,
                last_seen_turn=state.turn,
                note=note,
            )


def contact_changes(
    previous: dict[str, list[str]], current: dict[str, set[str]]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Directed contacts gained and lost, for `first_contact` / `contact_lost`.

    Returns sorted lists so the emitted events are byte-stable across runs.
    """
    gained: list[tuple[str, str]] = []
    lost: list[tuple[str, str]] = []
    for observer in sorted(set(previous) | set(current)):
        before = set(previous.get(observer, []))
        after = current.get(observer, set())
        gained.extend((observer, o) for o in sorted(after - before))
        lost.extend((observer, o) for o in sorted(before - after))
    return gained, lost
