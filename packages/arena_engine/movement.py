"""Domain-aware movement: who may enter which tile, and embarkation.

Every rule about where a unit can go lives here rather than being spread across
the reducer, `legal_actions`, and the bots. Those three must agree exactly - the
whole point of handing an agent `legal_actions` is that anything it offers will
be accepted - so they call one function instead of each reimplementing the
check.

**Embarkation is the Civ-5 model, not a transport model.** A land unit that
steps onto water becomes a sea unit itself: `embarked` flips to true, it defends
at a fraction of its strength, and it cannot attack. No unit is ever loaded
*inside* another. That was a deliberate scope choice: cargo would mean one unit
containing others, and with it position coupling every turn, referential
integrity on every death, and a cascade when a transport sinks - the highest
bug-density structure that could be added to a codebase whose value rests on
byte-identical replay.
"""

from __future__ import annotations

from arena_engine.content import (
    EMBARK_TECH,
    EMBARKED_MOVES,
    TERRAIN,
    UNITS,
    Domain,
)
from arena_engine.hex import Hex
from arena_engine.types import State, Unit


def domain_of(unit: Unit) -> Domain:
    """The domain a unit is currently operating in.

    An embarked land unit is a sea unit for every purpose except what it becomes
    again when it lands.
    """
    if UNITS[unit.type].domain is Domain.SEA:
        return Domain.SEA
    return Domain.SEA if unit.embarked else Domain.LAND


def can_embark(state: State, unit: Unit) -> bool:
    spec = UNITS[unit.type]
    if spec.domain is Domain.SEA or not spec.can_embark:
        return False
    return EMBARK_TECH in state.players[unit.owner].known_techs


def is_port(state: State, h: Hex) -> bool:
    """Whether a sea unit may sit on this tile because a coastal city is there.

    Ships built in a coastal city have to exist somewhere on the turn they are
    completed, and the city tile is land. Treating a coastal city as a port is
    both the conventional rule and the one that avoids spawning a trireme onto a
    water tile that might be occupied or contested.
    """
    city = state.city_at(h)
    if city is None:
        return False
    return any(
        (tile := state.at(n)) is not None and TERRAIN[tile.terrain].navigable for n in _neighbors(h)
    )


def _neighbors(h: Hex) -> tuple[Hex, ...]:
    from arena_engine import hex as hx

    return hx.neighbors(h)


def entry_check(state: State, unit: Unit, target: Hex) -> tuple[bool, str]:
    """Whether `unit` may move onto `target`, and why not if it may not.

    Terrain only. Occupancy and adjacency are the reducer's business, because
    they depend on whose turn it is and what has already resolved.
    """
    tile = state.at(target)
    if tile is None:
        return False, f"{target.to_key()} is off the map"

    spec = TERRAIN[tile.terrain]
    domain = domain_of(unit)

    if domain is Domain.SEA:
        if spec.navigable or is_port(state, target):
            return True, ""
        if UNITS[unit.type].domain is Domain.SEA:
            return False, f"{target.to_key()} is not navigable"
        # An embarked land unit stepping ashore is the normal disembark case.
        if spec.passable:
            return True, ""
        return False, f"{target.to_key()} is impassable"

    if spec.passable:
        return True, ""
    if spec.navigable:
        if not can_embark(state, unit):
            if not UNITS[unit.type].can_embark:
                return False, f"a {unit.type.value} cannot cross water"
            return False, f"crossing water requires {EMBARK_TECH}"
        return True, ""
    return False, f"{target.to_key()} is impassable"


def transition(state: State, unit: Unit, target: Hex) -> str | None:
    """What boarding state change this move causes: "embark", "disembark", None."""
    if UNITS[unit.type].domain is Domain.SEA:
        return None
    tile = state.at(target)
    if tile is None:
        return None
    navigable = TERRAIN[tile.terrain].navigable and not TERRAIN[tile.terrain].passable
    if not unit.embarked and navigable:
        return "embark"
    if unit.embarked and not navigable:
        return "disembark"
    return None


def apply_move(state: State, unit: Unit, target: Hex) -> str | None:
    """Move the unit and update its boarding state. Returns the transition.

    Landing ends the unit's turn. Without that, a stack could cross and assault
    in the same turn, which removes the entire defensive value of a coastline.
    """
    change = transition(state, unit, target)
    tile = state.at(target)
    cost = TERRAIN[tile.terrain].move_cost if tile is not None else 1

    unit.pos = target.to_key()
    unit.fortified = False
    unit.working_on = None

    if change == "embark":
        unit.embarked = True
        unit.moves_left = 0
    elif change == "disembark":
        unit.embarked = False
        unit.moves_left = 0
    else:
        unit.moves_left = max(0, unit.moves_left - cost)
    return change


def moves_for(state: State, unit: Unit) -> int:
    """Movement allowance at the start of a turn.

    Embarked units move at a fixed sea rate rather than their land rate, so a
    scout does not sail three times faster than an army.
    """
    if unit.embarked:
        return EMBARKED_MOVES
    return UNITS[unit.type].moves


def can_attack(state: State, unit: Unit) -> bool:
    """Embarked units are in transit, not in a fight."""
    return not UNITS[unit.type].civilian and not unit.embarked


def can_act_on_land(unit: Unit) -> bool:
    """Founding a city or building an improvement requires being ashore."""
    return not unit.embarked
