"""Static game content: terrain, resources, improvements, units, buildings, techs.

Everything here is frozen data with no behaviour. Rules that *read* this data
live in the subsystem modules; keeping the tables separate is what lets the
system prompt's rules reference be generated from the same source the engine
plays by, so the two can never drift.

Balance targets, so later tuning has something to aim at:
  - A decisive victory should land between turn 150 and 300.
  - Expansion should stay attractive through roughly turn 80, then contest.
  - No unit should be strictly dominant; every attacker has a counter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Terrain(StrEnum):
    OCEAN = "ocean"
    COAST = "coast"
    GRASSLAND = "grassland"
    PLAINS = "plains"
    FOREST = "forest"
    HILLS = "hills"
    DESERT = "desert"
    MOUNTAINS = "mountains"


class Resource(StrEnum):
    WHEAT = "wheat"
    IRON = "iron"
    HORSES = "horses"
    GOLD_ORE = "gold_ore"


class Improvement(StrEnum):
    FARM = "farm"
    MINE = "mine"
    ROAD = "road"


@dataclass(frozen=True, slots=True)
class Yields:
    """The four things a tile or building can produce."""

    food: int = 0
    production: int = 0
    gold: int = 0
    science: int = 0
    culture: int = 0

    def __add__(self, other: Yields) -> Yields:
        return Yields(
            self.food + other.food,
            self.production + other.production,
            self.gold + other.gold,
            self.science + other.science,
            self.culture + other.culture,
        )


@dataclass(frozen=True, slots=True)
class TerrainSpec:
    yields: Yields
    passable: bool  # to land units
    settleable: bool
    move_cost: int
    defense_pct: int  # percentage bonus to a defender standing here


TERRAIN: dict[Terrain, TerrainSpec] = {
    # Water is workable by an adjacent city but land units cannot enter it and
    # cities cannot be founded on it. There are no boats in v1.
    Terrain.OCEAN: TerrainSpec(Yields(food=1, gold=1), False, False, 1, 0),
    Terrain.COAST: TerrainSpec(Yields(food=2, gold=2), False, False, 1, 0),
    # Grassland feeds, plains balance, forest and hills produce and defend.
    Terrain.GRASSLAND: TerrainSpec(Yields(food=3), True, True, 1, 0),
    Terrain.PLAINS: TerrainSpec(Yields(food=1, production=2, gold=1), True, True, 1, 0),
    Terrain.FOREST: TerrainSpec(Yields(food=1, production=3), True, True, 2, 25),
    Terrain.HILLS: TerrainSpec(Yields(production=2), True, True, 2, 50),
    Terrain.DESERT: TerrainSpec(Yields(production=1, gold=1), True, True, 1, 0),
    # Mountains are pure obstacle: they shape the map into chokepoints, which is
    # what makes the contact view interesting to watch.
    Terrain.MOUNTAINS: TerrainSpec(Yields(production=1), False, False, 1, 100),
}

# What a resource adds on top of its tile, and which terrains it can appear on.
RESOURCE_YIELDS: dict[Resource, Yields] = {
    Resource.WHEAT: Yields(food=2),
    Resource.IRON: Yields(production=2),
    Resource.HORSES: Yields(production=1, food=1),
    Resource.GOLD_ORE: Yields(gold=3),
}

RESOURCE_TERRAIN: dict[Resource, frozenset[Terrain]] = {
    Resource.WHEAT: frozenset({Terrain.GRASSLAND, Terrain.PLAINS}),
    Resource.IRON: frozenset({Terrain.HILLS, Terrain.MOUNTAINS, Terrain.DESERT}),
    Resource.HORSES: frozenset({Terrain.PLAINS, Terrain.GRASSLAND}),
    Resource.GOLD_ORE: frozenset({Terrain.HILLS, Terrain.DESERT, Terrain.MOUNTAINS}),
}

# Strategic resources gate units. A civ with no iron in its borders cannot build
# swordsmen no matter what it has researched, which is what makes land worth
# fighting over rather than merely worth having.
STRATEGIC: frozenset[Resource] = frozenset({Resource.IRON, Resource.HORSES})


@dataclass(frozen=True, slots=True)
class ImprovementSpec:
    yields: Yields
    terrains: frozenset[Terrain]
    turns: int


IMPROVEMENTS: dict[Improvement, ImprovementSpec] = {
    Improvement.FARM: ImprovementSpec(
        Yields(food=1), frozenset({Terrain.GRASSLAND, Terrain.PLAINS, Terrain.DESERT}), 4
    ),
    Improvement.MINE: ImprovementSpec(
        Yields(production=1), frozenset({Terrain.HILLS, Terrain.FOREST, Terrain.DESERT}), 4
    ),
    Improvement.ROAD: ImprovementSpec(
        Yields(gold=1),
        frozenset(t for t in Terrain if TERRAIN[t].passable),
        2,
    ),
}


# ---------------------------------------------------------------------------
# Technology
#
# A DAG across four eras terminating in apex_theory. Costs are per-era rather
# than per-tech so the pacing is legible: a civ that beelines the bottom of the
# tree pays for it in the breadth it skipped.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TechSpec:
    era: int
    cost: int
    prereqs: tuple[str, ...] = ()


ERA_COST = {1: 30, 2: 70, 3: 130, 4: 210}

TECHS: dict[str, TechSpec] = {
    # Era 1: no prerequisites. Everyone starts here with nothing researched.
    "agriculture": TechSpec(1, ERA_COST[1]),
    "pottery": TechSpec(1, ERA_COST[1]),
    "mining": TechSpec(1, ERA_COST[1]),
    "archery": TechSpec(1, ERA_COST[1]),
    # Era 2
    "animal_husbandry": TechSpec(2, ERA_COST[2], ("agriculture",)),
    "bronze_working": TechSpec(2, ERA_COST[2], ("mining",)),
    "writing": TechSpec(2, ERA_COST[2], ("pottery",)),
    "masonry": TechSpec(2, ERA_COST[2], ("mining",)),
    # Era 3
    "horseback_riding": TechSpec(3, ERA_COST[3], ("animal_husbandry",)),
    "iron_working": TechSpec(3, ERA_COST[3], ("bronze_working",)),
    "mathematics": TechSpec(3, ERA_COST[3], ("writing",)),
    "currency": TechSpec(3, ERA_COST[3], ("pottery", "writing")),
    "literature": TechSpec(3, ERA_COST[3], ("writing",)),
    # Era 4
    "drama": TechSpec(4, ERA_COST[4], ("literature",)),
    "engineering": TechSpec(4, ERA_COST[4], ("mathematics", "masonry")),
    "philosophy": TechSpec(4, ERA_COST[4], ("literature", "currency")),
    "construction": TechSpec(4, ERA_COST[4], ("masonry", "currency")),
    # The science victory gate. Deliberately expensive and deep: reaching it
    # means having actually built an economy rather than rushed one branch.
    #
    # Cost was raised from 420 after a bot sweep ended 7 of 8 matches on a
    # science victory around turn 110, which left the conquest and domination
    # paths effectively untested and gave the military game no time to develop.
    "apex_theory": TechSpec(4, 1100, ("engineering", "philosophy")),
}

APEX_TECH = "apex_theory"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class UnitType(StrEnum):
    SETTLER = "settler"
    WORKER = "worker"
    SCOUT = "scout"
    WARRIOR = "warrior"
    ARCHER = "archer"
    SPEARMAN = "spearman"
    HORSEMAN = "horseman"
    SWORDSMAN = "swordsman"
    CATAPULT = "catapult"


@dataclass(frozen=True, slots=True)
class UnitSpec:
    cost: int
    attack: int
    defense: int
    moves: int
    vision: int
    upkeep: int = 0
    req_tech: str | None = None
    req_resource: Resource | None = None
    # Civilian units cannot attack and are captured rather than killed.
    civilian: bool = False
    # Percentage bonus when attacking a city. Catapults exist to break stacks
    # that fortified archers would otherwise make unassailable.
    siege_pct: int = 0


UNITS: dict[UnitType, UnitSpec] = {
    UnitType.SETTLER: UnitSpec(cost=30, attack=0, defense=1, moves=2, vision=2, civilian=True),
    UnitType.WORKER: UnitSpec(cost=20, attack=0, defense=1, moves=2, vision=1, civilian=True),
    UnitType.SCOUT: UnitSpec(cost=15, attack=1, defense=2, moves=3, vision=3),
    UnitType.WARRIOR: UnitSpec(cost=15, attack=6, defense=6, moves=1, vision=2),
    UnitType.ARCHER: UnitSpec(
        cost=25, attack=8, defense=10, moves=1, vision=2, upkeep=1, req_tech="archery"
    ),
    UnitType.SPEARMAN: UnitSpec(
        cost=25, attack=6, defense=13, moves=1, vision=2, upkeep=1, req_tech="bronze_working"
    ),
    UnitType.HORSEMAN: UnitSpec(
        cost=35,
        attack=12,
        defense=6,
        moves=3,
        vision=2,
        upkeep=1,
        req_tech="horseback_riding",
        req_resource=Resource.HORSES,
    ),
    UnitType.SWORDSMAN: UnitSpec(
        cost=40,
        attack=14,
        defense=10,
        moves=1,
        vision=2,
        upkeep=1,
        req_tech="iron_working",
        req_resource=Resource.IRON,
    ),
    UnitType.CATAPULT: UnitSpec(
        cost=50,
        attack=18,
        defense=4,
        moves=1,
        vision=1,
        upkeep=2,
        req_tech="mathematics",
        siege_pct=100,
    ),
}

# Spearmen counter horsemen, the one hard rock-paper-scissors edge in the game.
# Without it a horseback-riding rush is close to unanswerable.
COUNTER_BONUS_PCT = 100
COUNTERS: dict[UnitType, frozenset[UnitType]] = {
    UnitType.SPEARMAN: frozenset({UnitType.HORSEMAN}),
}


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildingSpec:
    cost: int
    yields: Yields = field(default_factory=Yields)
    defense: int = 0
    upkeep: int = 0
    req_tech: str | None = None
    # A wonder can exist once in the whole world, not once per civ.
    wonder: bool = False
    # Percentage of food kept when a city grows. Only granary sets this.
    food_kept_pct: int = 0


BUILDINGS: dict[str, BuildingSpec] = {
    "granary": BuildingSpec(cost=40, req_tech="pottery", upkeep=1, food_kept_pct=50),
    "barracks": BuildingSpec(cost=35, defense=5, upkeep=1),
    "walls": BuildingSpec(cost=45, defense=12, upkeep=1, req_tech="masonry"),
    "library": BuildingSpec(cost=45, yields=Yields(science=3), upkeep=1, req_tech="writing"),
    "market": BuildingSpec(cost=50, yields=Yields(gold=3), upkeep=1, req_tech="currency"),
    "temple": BuildingSpec(cost=35, yields=Yields(culture=2), upkeep=1),
    "amphitheater": BuildingSpec(cost=60, yields=Yields(culture=4), upkeep=2, req_tech="drama"),
    "aqueduct": BuildingSpec(cost=55, yields=Yields(food=2), upkeep=1, req_tech="construction"),
    # Wonders. Culture-heavy on purpose: they are the v1 culture score path, and
    # the hook a full tourism victory would later hang off.
    "great_library": BuildingSpec(
        cost=140, yields=Yields(science=5, culture=3), req_tech="literature", wonder=True
    ),
    "colossus": BuildingSpec(
        cost=120, yields=Yields(gold=5, culture=3), req_tech="bronze_working", wonder=True
    ),
    "pyramids": BuildingSpec(
        cost=160, yields=Yields(production=4, culture=3), req_tech="masonry", wonder=True
    ),
    # The science victory build. Expensive enough that reaching apex_theory is
    # not by itself a win; the civ still has to survive building it, which is
    # the window in which its rivals get to do something about it.
    "apex_project": BuildingSpec(
        cost=700, yields=Yields(culture=5), req_tech=APEX_TECH, wonder=True
    ),
}

APEX_PROJECT = "apex_project"
WONDERS: frozenset[str] = frozenset(name for name, spec in BUILDINGS.items() if spec.wonder)


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------

CITY_BASE_DEFENSE = 8
CITY_WORK_RADIUS = 2
CITY_VISION = 3
# Food needed to reach population n from n-1.
FOOD_TO_GROW_BASE = 14
FOOD_TO_GROW_STEP = 6
# A city cannot be founded within this many hexes of another, which keeps the
# map from degenerating into a settler-spam contest.
MIN_CITY_SPACING = 3


def food_to_grow(population: int) -> int:
    return FOOD_TO_GROW_BASE + FOOD_TO_GROW_STEP * (population - 1)


def tech_cost(name: str) -> int:
    return TECHS[name].cost


def available_techs(known: frozenset[str]) -> list[str]:
    """Techs whose prerequisites are all met and which are not already known.

    Sorted so the observation payload is byte-stable across runs; an unsorted
    set here would show up as spurious diffs in golden-file tests.
    """
    return sorted(
        name
        for name, spec in TECHS.items()
        if name not in known and all(p in known for p in spec.prereqs)
    )
