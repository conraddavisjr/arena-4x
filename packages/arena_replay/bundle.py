"""Building the replay bundle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arena_engine import economy, victory
from arena_engine.content import UNITS, Domain
from arena_engine.events import Event
from arena_engine.types import State

BUNDLE_VERSION = 1


def match_metadata(state: State, tile_order: list[str]) -> dict[str, Any]:
    """Everything that is fixed for the whole match, written once.

    `tiles` is the ordered terrain array every later frame indexes into. Terrain
    is included here rather than per turn because it is effectively static;
    repeating a thousand tiles every turn would dominate the bundle.
    """
    return {
        "version": BUNDLE_VERSION,
        "match_id": state.match_id,
        "seed": state.seed,
        "radius": state.config.radius,
        "turn_limit": state.config.turn_limit,
        "civs": [
            {
                "player_id": pid,
                "civ_name": state.players[pid].civ_name,
                # Stable per-civ colour index, so the board, the contact matrix
                # and the reasoning columns all agree without the viewer having
                # to invent a mapping.
                "colour": index,
            }
            for index, pid in enumerate(state.civ_ids())
        ],
        "tiles": tile_order,
        "terrain": [state.tiles[key].terrain.value for key in tile_order],
        "resources": {
            str(i): state.tiles[key].resource.value
            for i, key in enumerate(tile_order)
            if state.tiles[key].resource is not None
        },
    }


def _economy(state: State, player_id: str) -> dict[str, Any]:
    """The per-civ economy panel.

    Shown for every civ at once in the god view. Watching four treasuries move
    side by side is how a spectator spots that a civ is quietly going bankrupt
    or hoarding, several turns before it shows up in the board position - and
    it is how an implausible number gets noticed at all.
    """
    gold_per_turn, science_per_turn, culture_per_turn = economy.player_output(state, player_id)
    cities = state.cities_of(player_id)
    units = state.units_of(player_id)
    player = state.players[player_id]

    return {
        "gold": player.gold,
        "gold_per_turn": gold_per_turn,
        "science_per_turn": science_per_turn,
        "culture": player.culture,
        "culture_per_turn": culture_per_turn,
        "food_stored": sum(c.food_stored for c in cities),
        "food_surplus": sum(economy.food_surplus(state, c) for c in cities),
        "production_per_turn": sum(economy.city_yields(state, c).production for c in cities),
        "upkeep": economy.upkeep(state, player_id),
        "tax_pct": player.tax_pct,
        "science_pct": player.science_pct,
        "cities": len(cities),
        "population": sum(c.population for c in cities),
        "units": len(units),
        "military": sum(1 for u in units if not UNITS[u.type].civilian),
        "at_sea": sum(1 for u in units if u.embarked),
        "navy": sum(1 for u in units if UNITS[u.type].domain is Domain.SEA),
        "techs": len(player.known_techs),
        "researching": player.researching,
        "score": victory.score(state, player_id),
        "alive": player.alive,
    }


def _dossiers(state: State) -> dict[str, Any]:
    return {pid: state.players[pid].dossier.model_dump(mode="json") for pid in state.civ_ids()}


def turn_frame(
    state: State,
    events: list[Event],
    index: dict[str, int],
    previous_owners: dict[str, str | None] | None = None,
    previous_improvements: dict[str, str] | None = None,
    previous_dossiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One turn of the replay.

    Positional data refers to tiles by index rather than by `"q,r"`, which is
    roughly a quarter the size once visibility is included - and visibility is
    four civs times a few hundred tiles, the largest structure in the frame.
    """
    owners = {key: tile.owner for key, tile in state.tiles.items()}
    changed_owners = (
        {index[k]: v for k, v in owners.items() if previous_owners.get(k) != v}
        if previous_owners is not None
        else {index[k]: v for k, v in owners.items() if v is not None}
    )
    # A delta, like owners. Improvements accumulate and rarely change, so
    # repeating the full set every frame cost ~2KB a turn for data the viewer
    # already had.
    current_improvements = {
        key: tile.improvement.value
        for key, tile in state.tiles.items()
        if tile.improvement is not None
    }
    improvements = {
        index[k]: v
        for k, v in current_improvements.items()
        if previous_improvements is None or previous_improvements.get(k) != v
    }

    # A delta, like owners and improvements, and for the same reason twice over.
    # A dossier is capped at roughly 2000 tokens and most turns an agent changes
    # one line of it, so repeating four of them every frame would dominate the
    # bundle. It also happens to be exactly the shape the viewer wants: the
    # thing worth watching is not the dossier, it is the *edit* - a model
    # rewriting `trustworthiness: high` to `low` two turns after a betrayal is
    # this whole experiment in one field.
    dossiers = _dossiers(state)
    changed = {
        pid: d
        for pid, d in dossiers.items()
        if previous_dossiers is None or previous_dossiers.get(pid) != d
    }

    return {
        "turn": state.turn,
        "dossiers": changed,
        "units": [
            {
                "id": u.id,
                "owner": u.owner,
                "type": u.type.value,
                "at": index[u.pos],
                "hp": u.hp,
                "embarked": u.embarked or None,
                "fortified": u.fortified or None,
            }
            for _, u in sorted(state.units.items())
        ],
        "cities": [
            {
                "id": c.id,
                "owner": c.owner,
                "name": c.name,
                "at": index[c.pos],
                "population": c.population,
                "buildings": c.buildings,
                "defense": economy.city_defense(state, c),
            }
            for _, c in sorted(state.cities.items())
        ],
        "owners": changed_owners,
        "improvements": improvements,
        # Per civ, so the viewer can switch between god view, one civ's fog, and
        # the mixed contact view without refetching anything.
        "visibility": {
            pid: sorted(index[k] for k in keys if k in index)
            for pid, keys in state.visibility.items()
        },
        "contact": {pid: sorted(seen) for pid, seen in state.contact.items()},
        "economy": {pid: _economy(state, pid) for pid in state.civ_ids()},
        "relations": {
            key: {"state": rel.state.value, "since": rel.since_turn}
            for key, rel in sorted(state.relations.items())
        },
        "reasoning": _reasoning(events),
        "messages": _messages(state, events),
        "events": [
            {
                "type": e.type,
                "actor": e.actor,
                "text": e.text,
                **({"pos": index[e.payload["pos"]]} if e.payload.get("pos") in index else {}),
            }
            for e in events
            if e.type not in _NOISE
        ],
    }


# Event types that would swamp the ticker without telling a spectator anything.
# Movement is already visible on the board; rejected orders are debugging detail.
_NOISE = frozenset({"turn_started", "turn_ended", "unit_moved", "order_rejected", "agent_action"})


def _reasoning(events: list[Event]) -> dict[str, Any]:
    """Each civ's own account of what it was doing this turn."""
    out: dict[str, Any] = {}
    for e in events:
        if e.type != "agent_action" or e.actor is None:
            continue
        out[e.actor] = {
            "assessment": e.payload.get("situation_assessment") or "",
            "plan": e.payload.get("plan_this_turn") or "",
            "threats": e.payload.get("threats_and_opportunities") or [],
            "confidence": e.payload.get("confidence"),
            "orders": e.payload.get("order_count", 0),
        }
    return out


def _messages(state: State, events: list[Event]) -> list[dict[str, Any]]:
    """Everything said this turn, public and private.

    Private messages are included in full. The bundle is ground truth for a
    finished match, not a fog-limited view: a spectator reading a published
    match is meant to see the deal being struck *and* what the civ striking it
    was privately thinking, which is the whole point of the deception panel.
    """
    return [
        {
            "from": e.actor,
            "to": e.payload.get("to"),
            "channel": e.payload.get("channel", "public"),
            # The message body is in the payload; `e.text` is the ticker line
            # ("Aurelian Compact sent a private message"), not the content.
            "text": e.payload.get("text", ""),
        }
        for e in events
        if e.type == "message_sent"
    ]


@dataclass
class BundleWriter:
    """Accumulates frames during a match and writes them out at the end.

    Kept incremental so the same object can back a live match: frames are
    appended turn by turn, and a viewer polling the directory sees the match
    grow. That is the entire difference between live and replay.
    """

    root: Path
    metadata: dict[str, Any]
    index: dict[str, int]
    frames: list[dict[str, Any]] = field(default_factory=list)
    _previous_owners: dict[str, str | None] | None = None
    _previous_improvements: dict[str, str] | None = None
    _previous_dossiers: dict[str, Any] | None = None

    @classmethod
    def start(cls, root: Path, state: State) -> BundleWriter:
        tile_order = sorted(state.tiles)
        return cls(
            root=root,
            metadata=match_metadata(state, tile_order),
            index={key: i for i, key in enumerate(tile_order)},
        )

    def add(self, state: State, events: list[Event]) -> None:
        self.frames.append(
            turn_frame(
                state,
                events,
                self.index,
                self._previous_owners,
                self._previous_improvements,
                self._previous_dossiers,
            )
        )
        self._previous_owners = {k: t.owner for k, t in state.tiles.items()}
        self._previous_improvements = {
            k: t.improvement.value for k, t in state.tiles.items() if t.improvement is not None
        }
        self._previous_dossiers = _dossiers(state)

    def finish(self, state: State, stats: dict[str, Any] | None = None) -> Path:
        turns = self.root / "turns"
        turns.mkdir(parents=True, exist_ok=True)

        for frame in self.frames:
            (turns / f"{frame['turn']:04d}.json").write_text(_dump(frame))

        self.metadata["turns"] = len(self.frames)
        self.metadata["final_turn"] = state.turn
        self.metadata["victory"] = state.victory.model_dump(mode="json") if state.victory else None
        (self.root / "match.json").write_text(_dump(self.metadata))
        (self.root / "stats.json").write_text(_dump(stats or {}))
        return self.root


def _dump(payload: dict[str, Any]) -> str:
    """Compact and key-sorted.

    Sorted so a bundle is byte-reproducible from the same match, which is what
    lets a published replay be diffed or cached by content hash.
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"


def build_bundle(
    root: Path,
    state: State,
    frames: list[tuple[State, list[Event]]],
    stats: dict[str, Any] | None = None,
) -> Path:
    """Convenience wrapper for a match already played to completion."""
    writer = BundleWriter.start(root, state)
    for snapshot, events in frames:
        writer.add(snapshot, events)
    return writer.finish(frames[-1][0] if frames else state, stats)
