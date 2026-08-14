"""Combat resolution.

Deterministic given the match seed and the identities of the two units, via the
derived-draw scheme in `rng`. Two consequences worth knowing:

  - The same attack on the same turn always resolves the same way, so a replay
    reproduces the match exactly.
  - Adding or removing an unrelated combat elsewhere in the turn does not shift
    this one, which a sequential generator would.

The model is a single exchange rather than rounds-to-the-death. One attack does
proportional damage to both sides based on the strength ratio, so a losing
attacker still bloodies the defender and stacks wear down over several turns
instead of evaporating. That gives an agent something to reason about between
turns, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass

from arena_engine import rng
from arena_engine.content import (
    COUNTER_BONUS_PCT,
    COUNTERS,
    EMBARKED_DEFENSE_PCT,
    TERRAIN,
    UNITS,
)
from arena_engine.economy import city_defense
from arena_engine.types import City, State, Unit

FORTIFY_BONUS_PCT = 25
# How far the die can swing a fight, as a fraction either way. Enough that a
# marginal attack is a real gamble, not so much that strength stops mattering.
LUCK_SPREAD = 0.30
# Damage the winner takes as a share of what it dealt. Keeps attrition real.
MAX_DAMAGE = 100


@dataclass(frozen=True, slots=True)
class CombatResult:
    attacker_id: str
    defender_id: str
    attacker_damage: int  # damage dealt TO the attacker
    defender_damage: int  # damage dealt TO the defender
    attacker_died: bool
    defender_died: bool
    attacker_strength: int
    defender_strength: int
    captured: bool = False


def attack_strength(state: State, attacker: Unit, defender: Unit) -> int:
    spec = UNITS[attacker.type]
    strength = spec.attack * 100

    # Counter units get their bonus on attack as well as defence, so a spearman
    # is a genuine answer to cavalry rather than a purely passive wall.
    if defender.type in COUNTERS.get(attacker.type, frozenset()):
        strength += spec.attack * COUNTER_BONUS_PCT

    # Siege applies against cities, which is what catapults exist for.
    if spec.siege_pct and state.city_at(defender.hex) is not None:
        strength += spec.attack * spec.siege_pct

    return max(1, strength * attacker.hp // 100)


def defense_strength(state: State, defender: Unit, attacker: Unit) -> int:
    spec = UNITS[defender.type]
    strength = spec.defense * 100

    # Caught at sea. Terrain, fortification and city bonuses are all irrelevant
    # to a unit in transit, so return immediately rather than layering land
    # bonuses onto something floating on the water.
    if defender.embarked:
        return max(1, spec.defense * EMBARKED_DEFENSE_PCT * defender.hp // 100)

    tile = state.at(defender.hex)
    if tile is not None:
        strength += spec.defense * TERRAIN[tile.terrain].defense_pct

    if defender.fortified:
        strength += spec.defense * FORTIFY_BONUS_PCT

    if attacker.type in COUNTERS.get(defender.type, frozenset()):
        strength += spec.defense * COUNTER_BONUS_PCT

    city = state.city_at(defender.hex)
    if city is not None:
        strength += city_defense(state, city) * 100 // 10

    return max(1, strength * defender.hp // 100)


def resolve(state: State, attacker: Unit, defender: Unit) -> CombatResult:
    """Resolve one attack. Pure: computes the result, changes nothing.

    Civilian defenders are captured rather than killed, which is what makes a
    lightly-escorted settler a real liability and a worthwhile target.
    """
    atk = attack_strength(state, attacker, defender)
    dfn = defense_strength(state, defender, attacker)

    if UNITS[defender.type].civilian:
        return CombatResult(
            attacker_id=attacker.id,
            defender_id=defender.id,
            attacker_damage=0,
            defender_damage=0,
            attacker_died=False,
            defender_died=False,
            attacker_strength=atk,
            defender_strength=dfn,
            captured=True,
        )

    # One derived draw per side, keyed by the pair and the turn. Two draws
    # rather than one so a fight is not a pure coin flip on a single number.
    key = (state.turn, attacker.id, defender.id)
    atk_luck = 1.0 + (rng.roll(state.seed, "combat_atk", *key) - 0.5) * 2 * LUCK_SPREAD
    dfn_luck = 1.0 + (rng.roll(state.seed, "combat_def", *key) - 0.5) * 2 * LUCK_SPREAD

    effective_atk = max(1, int(atk * atk_luck))
    effective_dfn = max(1, int(dfn * dfn_luck))
    total = effective_atk + effective_dfn

    # Each side takes damage proportional to how outmatched it was. An even
    # fight costs both sides half their health; a rout is nearly one-sided.
    defender_damage = min(MAX_DAMAGE, max(1, effective_atk * MAX_DAMAGE // total))
    attacker_damage = min(MAX_DAMAGE, max(1, effective_dfn * MAX_DAMAGE // total))

    return CombatResult(
        attacker_id=attacker.id,
        defender_id=defender.id,
        attacker_damage=attacker_damage,
        defender_damage=defender_damage,
        attacker_died=attacker.hp - attacker_damage <= 0,
        defender_died=defender.hp - defender_damage <= 0,
        attacker_strength=atk,
        defender_strength=dfn,
    )


def best_defender(state: State, tile_units: list[Unit]) -> Unit | None:
    """Which unit in a stack takes the hit.

    The strongest defender protects the whole stack, so escorting a settler
    with a spearman actually works. Ties break on unit id for determinism.
    """
    if not tile_units:
        return None
    military = [u for u in tile_units if not UNITS[u.type].civilian]
    pool = military or tile_units
    return max(pool, key=lambda u: (UNITS[u.type].defense * u.hp, u.id))


def city_falls(state: State, city: City) -> bool:
    """A city is taken when nothing military remains to hold it."""
    return not any(
        not UNITS[u.type].civilian for u in state.units_at(city.hex) if u.owner == city.owner
    )
