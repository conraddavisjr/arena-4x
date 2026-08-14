"""What an agent sees: `(State, player_id) -> Observation`.

This is the layer that turns full game state into the fog-limited, token-bounded
payload a model actually receives. Two pressures shape every decision here, and
they pull against each other:

**Completeness.** An agent that cannot see something cannot reason about it, and
a missing field shows up as an inexplicably bad decision rather than an error.

**Cost.** Every field is paid for by four agents on every one of ~130 turns. A
field that adds 200 tokens costs roughly 100k tokens per match. So each one has
to earn its place, and anything unbounded has to be capped explicitly rather
than left to grow.

The sharpest instance of that tension is the remembered map. A civ explores
hundreds of tiles over a match, and dumping every one back each turn would be
the single largest block in the payload while consisting almost entirely of
"there is grass at 7,-4". Only remembered tiles that carry information - enemy
cities, resources - are returned, plus a count so the agent knows the scale of
what it has seen. See `_remembered`.
"""

from __future__ import annotations

from typing import Any

from arena_engine import diplomacy, economy, victory
from arena_engine import hex as hx
from arena_engine.content import (
    BARBARIAN_ID,
    CITY_WORK_RADIUS,
    UNITS,
    Improvement,
    Resource,
    Terrain,
)
from arena_engine.hex import Hex
from arena_engine.reducer import legal_actions
from arena_engine.types import Dossier, Model, State

# Caps. Each one is a token-budget decision, not a technical limit.
MAX_REMEMBERED = 40
MAX_INTEL_NOTES = 4
MAX_FRONTIER = 12


class ResearchView(Model):
    current: str | None = None
    turns_remaining: int | None = None
    science_per_turn: int = 0
    known_techs: list[str] = []
    available_next: list[str] = []


class YouView(Model):
    player_id: str
    civ_name: str
    score: int
    gold: int
    gold_per_turn: int
    culture: int
    rates: dict[str, int]
    research: ResearchView


class CityView(Model):
    id: str
    name: str
    pos: str
    population: int
    food_stored: int
    food_to_grow: int
    food_surplus: int
    production_per_turn: int
    building: str | None = None
    turns_remaining: int | None = None
    buildings: list[str] = []
    defense: int
    coastal: bool


class UnitView(Model):
    id: str
    type: str
    pos: str
    hp: int
    moves_left: int
    status: str
    # Attack and defence are deliberately absent. They are static per unit type
    # and already in the rules reference, which is in the cached prefix and so
    # costs ~0.1x - repeating them per unit per turn pays full price for the
    # same constant. Only `hp` varies, and that is here.


class TileView(Model):
    pos: str
    terrain: Terrain
    resource: Resource | None = None
    improvement: Improvement | None = None
    owner: str | None = None
    # Who is standing here, if anyone. Wildlife shows up like any other
    # occupant; an agent has to be able to see the wolf pack next to its settler.
    occupant: dict[str, Any] | None = None


class RememberedView(Model):
    pos: str
    terrain: Terrain
    resource: Resource | None = None
    last_seen_turn: int
    note: str | None = None


class MapView(Model):
    visible: list[TileView]
    remembered: list[RememberedView]
    explored_tiles: int
    frontier: list[str]


class RelationView(Model):
    player_id: str
    civ_name: str
    state: str
    since_turn: int
    pact_until: int | None = None


class ProposalView(Model):
    proposal_id: str
    from_player: str
    type: str
    terms: dict[str, Any]
    message: str | None = None
    expires_turn: int


class MessageView(Model):
    from_player: str
    channel: str
    turn: int
    text: str


class DiplomacyView(Model):
    relations: list[RelationView]
    open_proposals: list[ProposalView]
    inbox: list[MessageView]
    public_log: list[MessageView]


class IntelView(Model):
    player_id: str
    civ_name: str
    known_cities: int
    known_score: int | None = None
    military_estimate: str
    in_contact: bool


class VictoryProgressView(Model):
    your_cities: int
    total_cities: int
    domination_threshold_pct: int
    domination_streak: int
    turn: int
    turn_limit: int
    apex_tech_known: bool


class BudgetView(Model):
    tokens_allowance: int
    tokens_spent: int
    tokens_remaining: int
    match_pct_elapsed: float


class Observation(Model):
    match_id: str
    turn: int
    you: YouView
    cities: list[CityView]
    units: list[UnitView]
    map: MapView
    diplomacy: DiplomacyView
    intel: list[IntelView]
    your_dossier: Dossier
    recent_events: list[str]
    victory_progress: VictoryProgressView
    legal_actions: dict[str, Any]
    budget: BudgetView | None = None


def build(
    state: State,
    player_id: str,
    *,
    recent_events: list[str] | None = None,
    budget: BudgetView | None = None,
) -> Observation:
    """Assemble everything `player_id` may legitimately know this turn."""
    player = state.players[player_id]
    visible = {hx.from_key(k) for k in state.visibility.get(player_id, [])}
    gold_per_turn, science_per_turn, _ = economy.player_output(state, player_id)

    return Observation(
        match_id=state.match_id,
        turn=state.turn,
        you=_you(state, player_id, gold_per_turn, science_per_turn),
        cities=_cities(state, player_id),
        units=_units(state, player_id),
        map=_map(state, player_id, visible),
        diplomacy=_diplomacy(state, player_id),
        intel=_intel(state, player_id, visible),
        your_dossier=player.dossier,
        recent_events=(recent_events or [])[-state.config.recent_events_size :],
        victory_progress=_victory_progress(state, player_id),
        legal_actions=_compact_legal(legal_actions(state, player_id)),
        budget=budget,
    )


def _compact_legal(legal: dict[str, Any]) -> dict[str, Any]:
    """Drop empty options from the legal-action list.

    A late-game civ can field a hundred units, and `legal_actions` grows
    linearly with that. Most entries are mostly nothing: a fortified soldier
    inland has no attacks, no embarks, no disembarks, cannot found a city and
    cannot build. Spelling all of that out as empty lists and `false` was over
    5,000 tokens per turn at turn 120, more than the entire rest of the payload.

    An absent key reads the same as an empty one and costs nothing. The rules
    reference already tells the agent that anything not listed is unavailable.
    """
    units = {}
    for unit_id, options in legal["units"].items():
        kept = {k: v for k, v in options.items() if v not in (None, [], False)}
        units[unit_id] = kept
    return {**legal, "units": units}


def _you(state: State, player_id: str, gold_per_turn: int, science_per_turn: int) -> YouView:
    player = state.players[player_id]
    from arena_engine.content import available_techs, tech_cost

    remaining = None
    if player.researching is not None and science_per_turn > 0:
        outstanding = tech_cost(player.researching) - player.science_stored
        remaining = max(1, -(-outstanding // science_per_turn))

    return YouView(
        player_id=player_id,
        civ_name=player.civ_name,
        score=victory.score(state, player_id),
        gold=player.gold,
        gold_per_turn=gold_per_turn,
        culture=player.culture,
        rates={"tax_pct": player.tax_pct, "science_pct": player.science_pct},
        research=ResearchView(
            current=player.researching,
            turns_remaining=remaining,
            science_per_turn=science_per_turn,
            known_techs=player.known_techs,
            available_next=available_techs(frozenset(player.known_techs)),
        ),
    )


def _cities(state: State, player_id: str) -> list[CityView]:
    from arena_engine.content import food_to_grow

    out: list[CityView] = []
    for city in state.cities_of(player_id):
        production = economy.city_yields(state, city).production
        remaining = None
        if city.building is not None and production > 0:
            outstanding = economy.build_cost(city.building) - city.production_stored
            remaining = max(1, -(-outstanding // production))
        out.append(
            CityView(
                id=city.id,
                name=city.name,
                pos=city.pos,
                population=city.population,
                food_stored=city.food_stored,
                food_to_grow=food_to_grow(city.population),
                food_surplus=economy.food_surplus(state, city),
                production_per_turn=production,
                building=city.building,
                turns_remaining=remaining,
                buildings=city.buildings,
                defense=economy.city_defense(state, city),
                coastal=economy.is_coastal(state, city),
            )
        )
    return out


def _units(state: State, player_id: str) -> list[UnitView]:
    out: list[UnitView] = []
    for unit in state.units_of(player_id):
        if unit.embarked:
            status = "at sea"
        elif unit.working_on is not None:
            status = f"building {unit.working_on.value} ({unit.work_turns_left} turns left)"
        elif unit.fortified:
            status = "fortified"
        else:
            status = "ready"
        out.append(
            UnitView(
                id=unit.id,
                type=unit.type.value,
                pos=unit.pos,
                hp=unit.hp,
                moves_left=unit.moves_left,
                status=status,
            )
        )
    return out


def _occupant(state: State, h: Hex, player_id: str) -> dict[str, Any] | None:
    """Who is standing here, from this civ's point of view."""
    units = state.units_at(h)
    city = state.city_at(h)
    if not units and city is None:
        return None

    entry: dict[str, Any] = {}
    if city is not None:
        entry["city"] = city.name
        entry["city_owner"] = city.owner
        entry["population"] = city.population
    if units:
        owner = units[0].owner
        entry["player_id"] = owner
        entry["mine"] = owner == player_id
        if owner == BARBARIAN_ID:
            entry["wild"] = True
        # Composition rather than a unit list: an agent needs to know it faces
        # two archers, not their ids and exact hit points.
        counts: dict[str, int] = {}
        for unit in units:
            counts[unit.type.value] = counts.get(unit.type.value, 0) + 1
        entry["units"] = dict(sorted(counts.items()))
    return entry


def _map(state: State, player_id: str, visible: set[Hex]) -> MapView:
    tiles: list[TileView] = []
    for h in sorted(visible, key=lambda x: x.to_key()):
        tile = state.at(h)
        if tile is None:
            continue
        tiles.append(
            TileView(
                pos=h.to_key(),
                terrain=tile.terrain,
                resource=tile.resource,
                improvement=tile.improvement,
                owner=tile.owner,
                occupant=_occupant(state, h, player_id),
            )
        )
    return MapView(
        visible=tiles,
        remembered=_remembered(state, player_id, visible),
        explored_tiles=len(state.players[player_id].memory),
        frontier=_frontier(state, visible),
    )


def _remembered(state: State, player_id: str, visible: set[Hex]) -> list[RememberedView]:
    """Stale knowledge worth spending tokens on.

    A civ remembers hundreds of tiles by mid-match. Returning all of them would
    make this the largest block in the payload while being almost entirely
    "there is grass at 7,-4", which changes no decision. Only remembered tiles
    that carry information are returned - a rival city, or a resource worth
    settling for - most recently seen first. `explored_tiles` on the parent
    carries the scale that this list deliberately drops.
    """
    memory = state.players[player_id].memory
    candidates = [
        RememberedView(
            pos=key,
            terrain=entry.terrain,
            resource=entry.resource,
            last_seen_turn=entry.last_seen_turn,
            note=entry.note,
        )
        for key, entry in memory.items()
        if hx.from_key(key) not in visible
        and (entry.note is not None or entry.resource is not None)
    ]
    # Rival cities first - they are the single most decision-relevant thing a
    # civ can remember - then whatever was seen most recently.
    candidates.sort(key=lambda r: (r.note is None, -r.last_seen_turn, r.pos))
    return candidates[:MAX_REMEMBERED]


def _frontier(state: State, visible: set[Hex]) -> list[str]:
    """Unexplored tiles adjacent to what we can see: where to scout next."""
    edge: set[str] = set()
    for h in visible:
        for n in hx.neighbors(h):
            if n not in visible and state.at(n) is not None:
                edge.add(n.to_key())
    return sorted(edge)[:MAX_FRONTIER]


def _diplomacy(state: State, player_id: str) -> DiplomacyView:
    config = state.config
    relations = [
        RelationView(
            player_id=other,
            civ_name=state.players[other].civ_name,
            state=state.relation(player_id, other).state.value,
            since_turn=state.relation(player_id, other).since_turn,
            pact_until=state.relation(player_id, other).pact_until,
        )
        for other in state.civ_ids()
        if other != player_id and state.players[other].alive
    ]
    proposals = [
        ProposalView(
            proposal_id=p.id,
            from_player=p.from_player,
            type=p.type.value,
            terms={k: v for k, v in p.terms.model_dump().items() if v},
            message=p.message,
            expires_turn=p.expires_turn,
        )
        for p in diplomacy.open_proposals_for(state, player_id)
    ]
    return DiplomacyView(
        relations=relations,
        open_proposals=proposals,
        inbox=[
            MessageView(from_player=m.from_player, channel=m.channel, turn=m.turn, text=m.text)
            for m in diplomacy.inbox_for(state, player_id, config.inbox_size)
        ],
        public_log=[
            MessageView(from_player=m.from_player, channel=m.channel, turn=m.turn, text=m.text)
            for m in diplomacy.public_log(state, config.public_log_size)
        ],
    )


def _intel(state: State, player_id: str, visible: set[Hex]) -> list[IntelView]:
    """What we believe about rivals, from what we have actually observed.

    Deliberately partial. City counts come from this civ's own memory rather
    than from the true state, so an agent that has not scouted is working from a
    stale picture - which is the entire point of fog.
    """
    memory = state.players[player_id].memory
    my_power = _military_power(state, player_id)
    contacts = set(state.contact.get(player_id, []))

    out: list[IntelView] = []
    for other in state.civ_ids():
        if other == player_id or not state.players[other].alive:
            continue
        name = state.players[other].civ_name
        known_cities = sum(1 for entry in memory.values() if entry.note and name in entry.note)
        theirs = _military_power(state, other)
        if other in contacts:
            estimate = (
                "stronger"
                if theirs > my_power * 1.2
                else "weaker"
                if theirs < my_power * 0.8
                else "comparable"
            )
        else:
            estimate = "unknown"
        out.append(
            IntelView(
                player_id=other,
                civ_name=name,
                known_cities=known_cities,
                known_score=victory.score(state, other) if other in contacts else None,
                military_estimate=estimate,
                in_contact=other in contacts,
            )
        )
    return out


def _military_power(state: State, player_id: str) -> int:
    return sum(
        UNITS[u.type].attack + UNITS[u.type].defense
        for u in state.units_of(player_id)
        if not UNITS[u.type].civilian
    )


def _victory_progress(state: State, player_id: str) -> VictoryProgressView:
    from arena_engine.content import APEX_TECH

    return VictoryProgressView(
        your_cities=len(state.cities_of(player_id)),
        total_cities=len(state.cities),
        domination_threshold_pct=state.config.domination_threshold_pct,
        domination_streak=state.players[player_id].domination_streak,
        turn=state.turn,
        turn_limit=state.config.turn_limit,
        apex_tech_known=APEX_TECH in state.players[player_id].known_techs,
    )


def to_json(observation: Observation) -> str:
    """The exact string handed to the model.

    Nulls and empty collections are dropped. They are pure cost: an absent
    `resource` key says the same thing as `"resource": null` and saves the
    tokens, and across ~130 turns and four agents that is not a rounding error.
    """
    return observation.model_dump_json(exclude_none=True, exclude_defaults=False, indent=None)


def estimate_tokens(observation: Observation) -> int:
    """Rough token count for budgeting. Characters over 3.5 is close enough
    for JSON to make a cap meaningful without an API round trip."""
    return len(to_json(observation)) // 4 + 1


# The city work radius is quoted in the rules reference, so re-export it here to
# keep the two from drifting.
__all__ = ["Observation", "build", "to_json", "estimate_tokens", "CITY_WORK_RADIUS"]
