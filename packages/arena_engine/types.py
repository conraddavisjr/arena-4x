"""Game state models.

Two conventions here are load-bearing and worth stating up front.

**Hexes are stored as `"q,r"` strings, not tuples.** The state has to round-trip
through JSONB for snapshots and through a canonical dump for the determinism
hash, and JSON has no tuple keys. Converting on access costs an f-string per
lookup, which across a full match is noise. Use `State.at()` rather than
indexing `tiles` directly.

**Anything set-like is stored sorted.** Python set iteration order is stable
within a process but the *serialized* order is not, and the determinism test
compares hashes of serialized state. A `set[str]` field would make that test
flap. Sorted lists everywhere, enforced by validators.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arena_engine.content import Improvement, Resource, Terrain, UnitType
from arena_engine.hex import Hex, from_key


class Model(BaseModel):
    """Base for every state model.

    `extra="forbid"` catches a typo'd field name at construction rather than
    silently dropping it, which is exactly the class of bug that would make a
    determinism failure impossible to trace.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=False)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


class Tile(Model):
    terrain: Terrain
    resource: Resource | None = None
    improvement: Improvement | None = None
    # The city whose borders claim this tile. Ownership follows culture, not
    # occupation: standing an army on a tile does not transfer it.
    owner: str | None = None


class RememberedTile(Model):
    """What a player recalls of a tile it can no longer see.

    Deliberately lossy. It keeps terrain and a free-text note but not live unit
    positions, because the whole point of fog is that an agent has to reason
    about staleness. `last_seen_turn` is what makes that reasoning possible.
    """

    terrain: Terrain
    resource: Resource | None = None
    last_seen_turn: int
    note: str | None = None


# ---------------------------------------------------------------------------
# Units and cities
# ---------------------------------------------------------------------------


class Unit(Model):
    id: str
    owner: str
    type: UnitType
    pos: str
    hp: int = 100
    moves_left: int = 0
    fortified: bool = False
    # At sea. A land unit that embarked is a sea unit for movement and combat
    # until it lands again; it is never cargo inside another unit.
    embarked: bool = False
    # Worker state. `work_turns_left` counts down only while the worker stays
    # put; moving cancels the job, which is checked in the reducer.
    working_on: Improvement | None = None
    work_turns_left: int = 0

    @property
    def hex(self) -> Hex:
        return from_key(self.pos)


class City(Model):
    id: str
    owner: str
    name: str
    pos: str
    population: int = 1
    food_stored: int = 0
    production_stored: int = 0
    # A unit type name or a building name. The engine resolves which by looking
    # it up in UNITS then BUILDINGS, so the two namespaces must not collide.
    building: str | None = None
    buildings: list[str] = Field(default_factory=list)
    hp: int = 100
    # Recomputed each turn from population and available tiles rather than
    # stored authoritatively, so it can never drift from the map.
    worked_tiles: list[str] = Field(default_factory=list)

    @property
    def hex(self) -> Hex:
        return from_key(self.pos)

    @field_validator("buildings", "worked_tiles")
    @classmethod
    def _sorted(cls, v: list[str]) -> list[str]:
        return sorted(v)


# ---------------------------------------------------------------------------
# Diplomacy
# ---------------------------------------------------------------------------


class RelationState(StrEnum):
    NEUTRAL = "neutral"
    PEACE = "peace"
    WAR = "war"
    ALLIANCE = "alliance"


class Relation(Model):
    state: RelationState = RelationState.NEUTRAL
    since_turn: int = 0
    # A non-aggression pact is a promise with an expiry. Breaking it early is
    # legal and emits treaty_broken; that is the interesting case, not a bug.
    pact_until: int | None = None


class ProposalType(StrEnum):
    PEACE = "peace"
    ALLIANCE = "alliance"
    CEASEFIRE = "ceasefire"
    NON_AGGRESSION = "non_aggression"
    TRADE = "trade"


class Terms(Model):
    gold_to_them: int = 0
    gold_to_you: int = 0
    tech_to_them: str | None = None
    tech_to_you: str | None = None
    duration_turns: int = 0


class Proposal(Model):
    id: str
    from_player: str
    to_player: str
    type: ProposalType
    terms: Terms = Field(default_factory=Terms)
    message: str | None = None
    created_turn: int
    expires_turn: int


class Message(Model):
    id: str
    from_player: str
    # None means a public broadcast every civ receives.
    to_player: str | None = None
    channel: Literal["public", "private"]
    turn: int
    text: str


def pair_key(a: str, b: str) -> str:
    """Canonical key for an unordered pair of players.

    Relations are symmetric, so `("p3","p1")` and `("p1","p3")` must land on the
    same record. Sorting rather than nesting also keeps the state a flat dict,
    which serializes cleanly to JSONB.
    """
    if a == b:
        raise ValueError(f"a player has no relation with itself: {a}")
    lo, hi = sorted((a, b))
    return f"{lo}|{hi}"


# ---------------------------------------------------------------------------
# Agent-authored memory
# ---------------------------------------------------------------------------


class Trustworthiness(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class OpponentModel(Model):
    player_id: str
    assessed_intent: str
    trustworthiness: Trustworthiness = Trustworthiness.UNKNOWN
    notes: str | None = None


class Dossier(Model):
    """The agent's self-authored memory.

    The engine stores this and hands it back verbatim next turn. It never edits
    the content; the only thing it may do is truncate an oversized one, and it
    emits `dossier_truncated` when it does so the loss is visible in the log
    rather than silent.
    """

    doctrine: str = ""
    opponent_models: list[OpponentModel] = Field(default_factory=list)
    standing_commitments: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------


class Player(Model):
    id: str
    civ_name: str
    gold: int = 0
    culture: int = 0
    known_techs: list[str] = Field(default_factory=list)
    researching: str | None = None
    science_stored: int = 0
    tax_pct: int = 50
    science_pct: int = 50
    alive: bool = True
    # The barbarian faction. Modelled as a player so units, combat and movement
    # work unchanged, but it is not a civilisation: it never wins, never loses,
    # never negotiates, and must be excluded from every victory calculation.
    neutral: bool = False
    # Turn the civ was eliminated, for the post-game timeline.
    eliminated_turn: int | None = None
    memory: dict[str, RememberedTile] = Field(default_factory=dict)
    dossier: Dossier = Field(default_factory=Dossier)
    # Consecutive turns holding the domination threshold. Domination requires
    # holding it, not merely touching it, so a one-turn spike does not win.
    domination_streak: int = 0

    @field_validator("known_techs")
    @classmethod
    def _sorted(cls, v: list[str]) -> list[str]:
        return sorted(set(v))

    @model_validator(mode="after")
    def _rates_sum_to_100(self) -> Self:
        if self.tax_pct + self.science_pct != 100:
            raise ValueError(
                f"tax_pct + science_pct must equal 100, got {self.tax_pct}+{self.science_pct}"
            )
        return self


# ---------------------------------------------------------------------------
# Match configuration
# ---------------------------------------------------------------------------


class MatchConfig(Model):
    # Radius 18 (1027 tiles) rather than 12 (469). A measured sweep showed map
    # size costs nothing that matters: seconds-per-turn and the observation
    # payload are both flat from radius 12 to 24, because visibility scales with
    # unit count and vision, not with map area. The larger world buys room for
    # real oceans, mountain ranges and varied landscape, and roughly 50% more
    # cities per match. See `mapgen.START_RADIUS_FRACTION` for why the opening
    # distance is held fixed rather than scaling with this.
    radius: int = 18
    turn_limit: int = 300
    domination_threshold_pct: int = 60
    domination_hold_turns: int = 3
    # Which victory modules are live. Names index the registry in victory.py,
    # which is what makes adding a cultural victory a config change.
    victory_conditions: list[str] = Field(
        default_factory=lambda: ["conquest", "domination", "science", "turn_limit"]
    )
    starting_units: list[UnitType] = Field(
        default_factory=lambda: [
            UnitType.SETTLER,
            UnitType.SETTLER,
            UnitType.WARRIOR,
            UnitType.SCOUT,
            UnitType.WORKER,
        ]
    )
    starting_gold: int = 40
    # Where the four starts sit, as a fraction of `radius`. Overriding this is
    # how you trade opening distance against expansion room; see mapgen.
    start_radius_fraction: float | None = None
    # How many turns a proposal stays open before it lapses.
    proposal_ttl: int = 3
    # How many recent public messages an agent sees. Bounded because this is
    # paid for in tokens on every turn by every agent.
    public_log_size: int = 8
    inbox_size: int = 8
    recent_events_size: int = 10


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------


class VictoryResult(Model):
    condition: str
    winner: str | None
    turn: int
    detail: str
    # Final scores for every player, for the post-game table.
    scores: dict[str, int] = Field(default_factory=dict)


class State(Model):
    match_id: str
    seed: int
    turn: int = 0
    config: MatchConfig = Field(default_factory=MatchConfig)

    tiles: dict[str, Tile] = Field(default_factory=dict)
    units: dict[str, Unit] = Field(default_factory=dict)
    cities: dict[str, City] = Field(default_factory=dict)
    players: dict[str, Player] = Field(default_factory=dict)

    relations: dict[str, Relation] = Field(default_factory=dict)
    proposals: dict[str, Proposal] = Field(default_factory=dict)
    messages: list[Message] = Field(default_factory=list)

    # Derived each turn by visibility.py and stored on the snapshot so the
    # dashboard can replay contact history rather than only show it live.
    visibility: dict[str, list[str]] = Field(default_factory=dict)
    contact: dict[str, list[str]] = Field(default_factory=dict)

    # Monotonic counter behind every generated id. Part of the state precisely
    # so that replaying a log reproduces identical ids.
    next_id: int = 1
    victory: VictoryResult | None = None

    # -- accessors ---------------------------------------------------------

    def at(self, h: Hex) -> Tile | None:
        return self.tiles.get(h.to_key())

    def player_ids(self) -> list[str]:
        """Always sorted. Anything that iterates players for resolution order
        must go through here, or turn order becomes dict-insertion dependent."""
        return sorted(self.players)

    def living_player_ids(self) -> list[str]:
        return [p for p in self.player_ids() if self.players[p].alive]

    def civ_ids(self) -> list[str]:
        """Real civilisations, excluding the neutral faction.

        Anything that means "a rival" must use this rather than `player_ids`.
        A victory condition that counted barbarians as a civ would never fire
        conquest, and one that counted a razed barbarian holding as a city would
        skew domination - both silently, with a plausible-looking result.
        """
        return [p for p in self.player_ids() if not self.players[p].neutral]

    def living_civ_ids(self) -> list[str]:
        return [p for p in self.civ_ids() if self.players[p].alive]

    def is_neutral(self, player_id: str) -> bool:
        player = self.players.get(player_id)
        return player is not None and player.neutral

    def units_of(self, player_id: str) -> list[Unit]:
        return [u for _, u in sorted(self.units.items()) if u.owner == player_id]

    def cities_of(self, player_id: str) -> list[City]:
        return [c for _, c in sorted(self.cities.items()) if c.owner == player_id]

    def units_at(self, h: Hex) -> list[Unit]:
        key = h.to_key()
        return [u for _, u in sorted(self.units.items()) if u.pos == key]

    def city_at(self, h: Hex) -> City | None:
        key = h.to_key()
        for _, city in sorted(self.cities.items()):
            if city.pos == key:
                return city
        return None

    def relation(self, a: str, b: str) -> Relation:
        """Relations default to neutral rather than being pre-seeded, so a new
        player added mid-design does not need a migration."""
        return self.relations.get(pair_key(a, b), Relation())

    def at_war(self, a: str, b: str) -> bool:
        """Whether these two may attack each other.

        The neutral faction is permanently hostile to everyone and cannot be
        negotiated with, so it short-circuits the relation lookup entirely.
        Routing it through `relation` would also raise on the pair key, since
        there is no diplomatic record with a wolf.
        """
        if a == b:
            return False
        if self.is_neutral(a) or self.is_neutral(b):
            return True
        return self.relation(a, b).state is RelationState.WAR

    # -- identity ----------------------------------------------------------

    def new_id(self, prefix: str) -> tuple[str, int]:
        """Return `(id, next_counter)`. Callers must write the counter back.

        Deliberately not a mutating method: the reducer owns all state changes,
        and an id generator that quietly mutated would be the one exception.
        """
        return f"{prefix}{self.next_id}", self.next_id + 1

    def canonical_json(self) -> str:
        """Byte-stable serialization. The basis of the determinism test.

        `visibility` and `contact` are excluded because they are derived: two
        engines that agree on every unit and city necessarily agree on vision,
        so including them would only add a second way for the hash to break.
        """
        payload = self.model_dump(mode="json", exclude={"visibility", "contact"})
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def state_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()
