"""Victory conditions, as a pluggable registry.

Each condition is an independent function registered by name, and `MatchConfig`
decides which are live. That is what makes adding a cultural or tourism victory
later a new module plus a config entry rather than a change to the reducer.

Scoring is shared by the turn-limit condition and by the post-game table, so
"who was winning" means the same thing in both places.
"""

from __future__ import annotations

from collections.abc import Callable

from arena_engine.content import APEX_PROJECT, APEX_TECH, UNITS
from arena_engine.types import State, VictoryResult

VictoryCheck = Callable[[State], VictoryResult | None]
REGISTRY: dict[str, VictoryCheck] = {}

# Score weights. Cities dominate, because a civ that holds more of the map
# should be ahead even if it is behind on any single sub-measure.
W_CITY = 12
W_POPULATION = 3
W_TECH = 5
W_MILITARY = 1
W_CULTURE = 1
W_WONDER = 15


def register(name: str) -> Callable[[VictoryCheck], VictoryCheck]:
    def decorator(fn: VictoryCheck) -> VictoryCheck:
        REGISTRY[name] = fn
        return fn

    return decorator


def score(state: State, player_id: str) -> int:
    """A civ's standing. Used for the turn-limit win and the final table."""
    cities = state.cities_of(player_id)
    player = state.players[player_id]

    total = len(cities) * W_CITY
    total += sum(c.population for c in cities) * W_POPULATION
    total += len(player.known_techs) * W_TECH
    total += player.culture * W_CULTURE
    total += (
        sum(
            UNITS[u.type].attack + UNITS[u.type].defense
            for u in state.units_of(player_id)
            if not UNITS[u.type].civilian
        )
        * W_MILITARY
        // 4
    )
    total += sum(
        W_WONDER
        for c in cities
        for b in c.buildings
        if b in {APEX_PROJECT, "great_library", "colossus", "pyramids"}
    )
    return total


def scores(state: State) -> dict[str, int]:
    return {p: score(state, p) for p in state.player_ids()}


@register("conquest")
def _conquest(state: State) -> VictoryResult | None:
    """Last civ standing.

    Requires at least two civs to have started, so a single-player debug match
    does not instantly declare a winner on turn 1.
    """
    if len(state.players) < 2:
        return None
    holders = sorted({c.owner for _, c in sorted(state.cities.items())})
    living = state.living_player_ids()
    if len(living) == 1 and len(holders) <= 1:
        winner = living[0]
        return VictoryResult(
            condition="conquest",
            winner=winner,
            turn=state.turn,
            detail=f"{state.players[winner].civ_name} is the last civilisation standing",
            scores=scores(state),
        )
    return None


@register("domination")
def _domination(state: State) -> VictoryResult | None:
    """Hold a supermajority of the world's cities for several turns running.

    The streak requirement is the whole point: capturing one city to cross the
    line for a single turn should not win the game, holding it should. The
    streak counter lives on the player and is maintained here.
    """
    total = len(state.cities)
    if total == 0:
        return None

    threshold = state.config.domination_threshold_pct
    for player_id in state.player_ids():
        player = state.players[player_id]
        held = len(state.cities_of(player_id))
        if player.alive and held * 100 >= total * threshold:
            player.domination_streak += 1
        else:
            player.domination_streak = 0

        if player.domination_streak >= state.config.domination_hold_turns:
            return VictoryResult(
                condition="domination",
                winner=player_id,
                turn=state.turn,
                detail=(
                    f"{player.civ_name} held {held} of {total} cities "
                    f"for {player.domination_streak} consecutive turns"
                ),
                scores=scores(state),
            )
    return None


@register("science")
def _science(state: State) -> VictoryResult | None:
    """Research the apex tech and actually finish the project.

    Two gates rather than one, so a runaway science civ still has to survive
    long enough to build it.
    """
    for player_id in state.living_player_ids():
        player = state.players[player_id]
        if APEX_TECH not in player.known_techs:
            continue
        for city in state.cities_of(player_id):
            if APEX_PROJECT in city.buildings:
                return VictoryResult(
                    condition="science",
                    winner=player_id,
                    turn=state.turn,
                    detail=(f"{player.civ_name} completed the Apex Project in {city.name}"),
                    scores=scores(state),
                )
    return None


@register("turn_limit")
def _turn_limit(state: State) -> VictoryResult | None:
    """Highest score when the clock runs out.

    Ties break on player id. A tie is vanishingly unlikely with these weights,
    but "vanishingly unlikely" is not "cannot happen" over many matches, and a
    non-deterministic winner would be worse than an arbitrary one.
    """
    if state.turn < state.config.turn_limit:
        return None
    table = scores(state)
    living = state.living_player_ids()
    if not living:
        return VictoryResult(
            condition="turn_limit",
            winner=None,
            turn=state.turn,
            detail="no civilisations survived",
            scores=table,
        )
    winner = max(sorted(living), key=lambda p: (table[p], p))
    return VictoryResult(
        condition="turn_limit",
        winner=winner,
        turn=state.turn,
        detail=(
            f"{state.players[winner].civ_name} led on score at the turn limit with {table[winner]}"
        ),
        scores=table,
    )


def check(state: State) -> VictoryResult | None:
    """Run the enabled conditions in config order; first hit wins.

    Order matters and is the caller's to choose: conquest before turn_limit
    means a civ that wins outright on the final turn is recorded as a conquest
    rather than as a score win.
    """
    for name in state.config.victory_conditions:
        checker = REGISTRY.get(name)
        if checker is None:
            raise KeyError(f"unknown victory condition {name!r}; registered: {sorted(REGISTRY)}")
        result = checker(state)
        if result is not None:
            return result
    return None
