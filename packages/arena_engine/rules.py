"""The rules reference handed to every agent, generated from the content tables.

Two properties matter here and both come from generating rather than writing it.

**It cannot drift from the engine.** A hand-written rules document describing a
spearman that no longer counters cavalry would make every agent reason from a
false premise, and nothing would fail - the matches would simply be measuring
something other than what we thought.

**It is byte-identical across turns and matches.** This text is the whole of the
cached system prefix. Anthropic prompt caching is a strict prefix match, so a
single varying byte anywhere in here - a timestamp, a turn number, a player id -
invalidates the cache on every request for every agent, turning a ~0.1x read
back into full price. Nothing dynamic may ever be interpolated into this string.
`test_rules.py` asserts that directly by generating it twice and comparing.
"""

from __future__ import annotations

from functools import cache

from arena_engine.content import (
    APEX_PROJECT,
    APEX_TECH,
    BUILDINGS,
    CITY_VISION,
    CITY_WORK_RADIUS,
    COUNTER_BONUS_PCT,
    COUNTERS,
    EMBARK_TECH,
    EMBARKED_DEFENSE_PCT,
    IMPROVEMENTS,
    MIN_CITY_SPACING,
    NEUTRAL_UNITS,
    RESOURCE_TERRAIN,
    RESOURCE_YIELDS,
    STRATEGIC,
    TECHS,
    TERRAIN,
    UNITS,
    WORKER_IMPROVEMENTS,
    Domain,
    Terrain,
    UnitType,
)


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    out.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))).rstrip())
    return "\n".join(out)


def _terrain_section() -> str:
    rows = []
    for terrain in Terrain:
        spec = TERRAIN[terrain]
        y = spec.yields
        who = "land" if spec.passable else ("sea" if spec.navigable else "nobody")
        rows.append(
            [
                terrain.value,
                f"{y.food}/{y.production}/{y.gold}",
                who,
                "yes" if spec.settleable else "no",
                str(spec.move_cost),
                f"+{spec.defense_pct}%" if spec.defense_pct else "-",
            ]
        )
    return _table(rows, ["terrain", "F/P/G", "open to", "city", "move", "defence"])


def _resource_section() -> str:
    rows = []
    for resource in sorted(RESOURCE_YIELDS, key=lambda r: r.value):
        y = RESOURCE_YIELDS[resource]
        parts = [
            f"{n} {label}"
            for label, n in (("food", y.food), ("prod", y.production), ("gold", y.gold))
            if n
        ]
        rows.append(
            [
                resource.value,
                ", ".join(parts),
                ", ".join(sorted(t.value for t in RESOURCE_TERRAIN[resource])),
                "yes" if resource in STRATEGIC else "no",
            ]
        )
    return _table(rows, ["resource", "yield", "found on", "strategic"])


def _unit_section() -> str:
    rows = []
    for unit in UnitType:
        if unit in NEUTRAL_UNITS:
            continue
        spec = UNITS[unit]
        gate = spec.req_tech or "-"
        if spec.req_resource is not None:
            gate = f"{gate} + {spec.req_resource.value}"
        rows.append(
            [
                unit.value,
                str(spec.cost),
                str(spec.attack),
                str(spec.defense),
                str(spec.moves),
                str(spec.vision),
                str(spec.upkeep),
                "sea" if spec.domain is Domain.SEA else "land",
                gate,
            ]
        )
    return _table(
        rows, ["unit", "cost", "atk", "def", "move", "sight", "upkeep", "domain", "requires"]
    )


def _building_section() -> str:
    rows = []
    for name in sorted(BUILDINGS):
        spec = BUILDINGS[name]
        y = spec.yields
        effects = [
            f"{n} {label}"
            for label, n in (
                ("food", y.food),
                ("prod", y.production),
                ("gold", y.gold),
                ("science", y.science),
                ("culture", y.culture),
            )
            if n
        ]
        if spec.defense:
            effects.append(f"+{spec.defense} defence")
        if spec.food_kept_pct:
            effects.append(f"keeps {spec.food_kept_pct}% food on growth")
        rows.append(
            [
                name,
                str(spec.cost),
                str(spec.upkeep),
                ", ".join(effects) or "-",
                spec.req_tech or "-",
                "yes" if spec.wonder else "no",
            ]
        )
    return _table(rows, ["building", "cost", "upkeep", "effect", "requires", "wonder"])


def _tech_section() -> str:
    rows = []
    for name in sorted(TECHS, key=lambda t: (TECHS[t].era, TECHS[t].cost, t)):
        spec = TECHS[name]
        rows.append([name, str(spec.era), str(spec.cost), ", ".join(spec.prereqs) or "-"])
    return _table(rows, ["tech", "era", "cost", "requires"])


def _improvement_section() -> str:
    rows = []
    for improvement in sorted(IMPROVEMENTS, key=lambda i: i.value):
        spec = IMPROVEMENTS[improvement]
        y = spec.yields
        parts = [
            f"+{n} {label}"
            for label, n in (("food", y.food), ("prod", y.production), ("gold", y.gold))
            if n
        ]
        rows.append(
            [
                improvement.value,
                ", ".join(parts),
                ", ".join(sorted(t.value for t in spec.terrains)),
                str(spec.turns),
                "worker" if improvement in WORKER_IMPROVEMENTS else "automatic",
            ]
        )
    return _table(rows, ["improvement", "yield", "terrain", "turns", "built by"])


@cache
def reference() -> str:
    """The full rules reference. Cached because it is identical every call."""
    counters = "; ".join(
        f"{unit.value} vs {', '.join(sorted(t.value for t in targets))} (+{COUNTER_BONUS_PCT}%)"
        for unit, targets in sorted(COUNTERS.items(), key=lambda kv: kv[0].value)
    )
    return f"""\
# ARENA-4X RULES

Four civilisations share one map. You control one of them.

## The turn

All civilisations move simultaneously. You submit reasoning, diplomacy and
orders together; every civilisation's orders for the turn resolve at once, so
you are always acting on information that is one turn old for everything you
cannot currently see.

Orders are validated individually. An illegal order is rejected on its own and
the rest of your turn still applies, so a single mistake never costs you a turn.
`legal_actions` in your observation lists exactly what is legal right now - if
it is not on that list, it will be rejected.

Messages and proposals you send arrive in the recipient's inbox on the *next*
turn. There is no instant negotiation.

## Winning

- **Conquest** - be the only civilisation left holding cities.
- **Domination** - hold at least 60% of all cities in the world for 3
  consecutive turns.
- **Science** - research `{APEX_TECH}` and complete the `{APEX_PROJECT}` wonder.
- **Score** - if the turn limit is reached, the highest score wins. Score counts
  cities, population, techs, military and culture, with cities weighted most.

## Cities

Found with a settler on settleable terrain, at least {MIN_CITY_SPACING} hexes
from any other city. A new city claims the tiles within 2 hexes.

A city works its centre tile free, plus one tile per population, chosen
automatically from the best available within {CITY_WORK_RADIUS} hexes. Cities
grow on surplus food and shrink if food goes negative. City sight range is
{CITY_VISION}.

Commerce from tiles is split between gold and science by your tax rate.
Building science and gold are flat and bypass the split.

## Combat

A single attack damages both sides in proportion to the strength ratio, so
fights wear units down over several turns rather than resolving in one blow.
Terrain, fortification and city defences all add to the defender.

The strongest defender in a stack protects the whole stack, so escorting a
settler with a real soldier works. Civilians alone are captured, not killed.

Counters: {counters}.

Capturing a city requires killing everything military inside it, then moving in
on a later turn. You must be at war with a civilisation to attack it.

## The sea

Water is impassable on foot. With `{EMBARK_TECH}`, a land unit may step onto
water and becomes *embarked*: it defends at {EMBARKED_DEFENSE_PCT}% of its
normal strength, cannot attack, and cannot found cities or build improvements
until it lands. Embarking and landing each end that unit's turn.

Warships beat anything they catch at sea. Escort your crossings.

Ships can only be built in a city adjacent to water, and a coastal city acts as
a port that ships may enter.

## The wilderness

Unclaimed land holds wolves and barbarian raiders. They are permanently hostile,
cannot be negotiated with, and you never need to declare war to fight them.
Wolves hunt lone or civilian units. Raiders march on cities and will sack an
undefended one, reducing its population, or burn a size-1 city outright. They
never occupy a city. Wildlife thins out as land is claimed.

## Your dossier

You write it, the engine stores it untouched, and it comes back to you verbatim
next turn. It is the only thing you carry forward apart from the board itself,
so anything you will want in twenty turns has to be in here.

- `doctrine` - the strategy you are executing across turns, not this turn's
  orders.
- `opponent_models` - one per rival you have formed a view about. Record what
  you think they are doing and how much their word has been worth, and revise
  it when evidence arrives. A promise kept or broken is exactly what should move
  `trustworthiness`. This is the most valuable thing in the dossier: an explicit
  read on a rival is worth far more than a transcript of what they said.
- `standing_commitments` - promises you have made, with the turn they expire.
  You may break them. Everyone will be told if you do.
- `lessons` - what you have learned about how this world works.

Keep it under about two thousand tokens. If it runs over, the engine trims
lessons first and then commitments; your doctrine and your opponent models are
never trimmed, because they are the expensive ones to rebuild.

## Terrain

{_terrain_section()}

## Resources

A strategic resource must lie inside your borders somewhere to build the units
that need it.

{_resource_section()}

## Improvements

{_improvement_section()}

## Units

{_unit_section()}

## Buildings

A wonder can exist only once in the entire world.

{_building_section()}

## Technology

{_tech_section()}
"""


def system_prompt(civ_name: str, player_id: str) -> str:
    """The full system prompt for one agent.

    The civ name and player id are the only per-agent variation, and they are
    placed *before* the rules reference on purpose so that the large invariant
    block stays a contiguous suffix. Callers should still put the whole thing in
    one cached block; see the adapter notes on prompt caching.
    """
    return f"""\
You are the sovereign of {civ_name} ({player_id}) in ARENA-4X, a turn-based 4X
strategy game against three rival civilisations, each controlled by a different
AI.

Achieve success and conquer the game as you see fit.

Each turn you receive the complete state visible to you and the set of actions
currently legal. Return your strategic reasoning first, then your diplomacy and
your orders.

{reference()}"""
