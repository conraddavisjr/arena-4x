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


def match_metadata(
    state: State, tile_order: list[str], models: dict[str, str] | None = None
) -> dict[str, Any]:
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
                # Which model played this seat. Carried so a published replay
                # says who was actually competing - a civ name is fiction, and
                # the whole point of the match is the comparison.
                #
                # Deliberately *not* the civ name the agents are given. That
                # string goes into the system prompt, so naming a civ after its
                # model would tell every agent which model it is and show the
                # others too. Whether a model plays differently when it knows it
                # is Opus facing Grok is a genuinely interesting question, and it
                # is not the one the baseline run is asking.
                "model": (models or {}).get(pid),
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
    spend: dict[str, Any] | None = None,
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
        # What this turn cost, per civ. Belongs in the bundle rather than only
        # in the journal because "score gained per 100k tokens" is the closest
        # thing this experiment has to a headline result, and a published replay
        # that cannot show it is missing the comparison it exists to make.
        #
        # Safe to publish: token counts and dollars describe the match, not the
        # prompts. Nothing here reveals what was sent.
        "spend": spend or {},
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
            # `pact_until` was dropped here, and dropping it made the entire
            # treaty system invisible. A non-aggression pact does not change the
            # relation *state* - two civs at peace with a pact are still
            # `neutral` - so a bundle carrying only the state showed nothing
            # while three pacts were signed, one expired, and five proposals
            # were exchanged. It looked like the agents were ignoring the
            # diplomacy mechanics; they were using them fluently.
            key: {
                "state": rel.state.value,
                "since": rel.since_turn,
                "pact_until": rel.pact_until,
            }
            for key, rel in sorted(state.relations.items())
        },
        # Combat, structured. The event ticker can say a fight happened; only
        # this can say who fought, what it cost and who is bleeding. It is what
        # turns "haiku suddenly lost half its army on turn 15" from something
        # you notice by accident into something the panel tells you.
        "combat": _combat(events),
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
_NOISE = frozenset(
    {
        "turn_started",
        "turn_ended",
        "unit_moved",
        "order_rejected",
        "agent_action",
        # The reply's *words* belong in the diplomacy thread, and go there. As a
        # ticker line it only ever restates the `treaty_signed` or
        # `proposal_rejected` sitting directly above it: "grok accepted gemini's
        # non_aggression proposal" followed by "grok answered gemini" is one
        # event reported twice.
        "proposal_answered",
    }
)


def _combat(events: list[Event]) -> list[dict[str, Any]]:
    """Every blow struck this turn, with both sides named."""
    out: list[dict[str, Any]] = []
    for e in events:
        if e.type != "combat_resolved" or "attacker_type" not in e.payload:
            continue
        out.append(
            {
                "attacker": e.actor,
                "defender": e.payload.get("defender_owner"),
                "attacker_type": e.payload.get("attacker_type"),
                "defender_type": e.payload.get("defender_type"),
                "attacker_damage": e.payload.get("attacker_damage", 0),
                "defender_damage": e.payload.get("defender_damage", 0),
                "attacker_died": bool(e.payload.get("attacker_died")),
                "defender_died": bool(e.payload.get("defender_died")),
                "text": e.text,
            }
        )
    return out


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


# What counts as something a civ *said*. A proposal's covering note and the
# reply to it are negotiation in exactly the way a DM is, and gathering only
# `message_sent` was why a match full of pacts read as "Nobody has spoken":
# models put their diplomacy in the proposal, not beside it.
SPOKEN = {
    "message_sent": "message",
    "proposal_made": "proposal",
    "proposal_answered": "reply",
}


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
            # A proposal and its reply are addressed to one civ by construction,
            # so they are private whatever the payload says. Only `send_message`
            # can be a broadcast.
            "channel": e.payload.get("channel", "private" if kind != "message" else "public"),
            "kind": kind,
            # The message body is in the payload; `e.text` is the ticker line
            # ("Aurelian Compact sent a private message"), not the content.
            "text": e.payload.get("text", ""),
            # Only on a reply, and it is the half that carries the meaning: the
            # same words follow an acceptance and a refusal.
            **({"response": e.payload["response"]} if "response" in e.payload else {}),
            **({"type": e.payload["type"]} if kind == "proposal" else {}),
        }
        for e in events
        # A message needs words - an empty chat bubble says nothing. A proposal
        # or a reply does not: the act is the content. A civ that accepted a
        # ten-turn pact in silence still bound itself, and requiring prose here
        # meant the treaty showed up in the relations bar with nothing in the
        # thread to account for it.
        if (kind := SPOKEN.get(e.type)) and (kind != "message" or e.payload.get("text"))
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
    def start(cls, root: Path, state: State, models: dict[str, str] | None = None) -> BundleWriter:
        tile_order = sorted(state.tiles)
        return cls(
            root=root,
            metadata=match_metadata(state, tile_order, models),
            index={key: i for i, key in enumerate(tile_order)},
        )

    def add(self, state: State, events: list[Event], spend: dict[str, Any] | None = None) -> None:
        self.frames.append(
            turn_frame(
                state,
                events,
                self.index,
                self._previous_owners,
                self._previous_improvements,
                self._previous_dossiers,
                spend,
            )
        )
        self._previous_owners = {k: t.owner for k, t in state.tiles.items()}
        self._previous_improvements = {
            k: t.improvement.value for k, t in state.tiles.items() if t.improvement is not None
        }
        self._previous_dossiers = _dossiers(state)

    def finish(self, state: State, stats: dict[str, Any] | None = None, **about: Any) -> Path:
        turns = self.root / "turns"
        turns.mkdir(parents=True, exist_ok=True)

        for frame in self.frames:
            (turns / f"{frame['turn']:04d}.json").write_text(_dump(frame))

        self.metadata["turns"] = len(self.frames)
        self.metadata["final_turn"] = state.turn
        self.metadata["victory"] = state.victory.model_dump(mode="json") if state.victory else None
        # Facts about the run rather than about the match: when it finished, what
        # it cost. They belong in the bundle because the bundle is the thing that
        # gets served and listed, and a library that had to open a journal to
        # date an entry would need the journal published alongside it - which is
        # exactly what the bundle exists to avoid.
        self.metadata.update({k: v for k, v in about.items() if v is not None})
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
