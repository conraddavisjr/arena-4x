"""Scripted heuristic bots.

These are not meant to play well. They exist so the engine can be exercised end
to end without spending a cent on API calls: a full 300-turn match between four
of these is the exit criterion for the engine, and the fastest way to find rules
that deadlock, economies that never grow, and victory conditions that never fire.

They are also the control group. When a frontier model plays, "did it beat the
heuristic bot" is a far more informative question than "did it win", and this is
the baseline that makes that comparison possible.

Deliberately deterministic: given the same state they produce the same orders,
so a bot match is as replayable as an agent match.
"""

from __future__ import annotations

from arena_engine import economy, rng
from arena_engine import hex as hx
from arena_engine.actions import (
    Action,
    Attack,
    BuildImprovement,
    DeclareWar,
    Fortify,
    FoundCity,
    MoveUnit,
    Order,
    Propose,
    Reasoning,
    RespondToProposal,
    SendMessage,
    SetProduction,
    SetRates,
    SetResearch,
)
from arena_engine.content import MIN_CITY_SPACING, TERRAIN, UNITS, UnitType
from arena_engine.hex import Hex
from arena_engine.reducer import legal_actions
from arena_engine.types import ProposalType, State, Terms

# Buildings worth having, in the order a city should pursue them. Units are
# chosen by need rather than from a list (see `_wanted`), because a static
# priority list does not work here at all: `warrior` is buildable on every turn
# of the game, so a front-to-back scan picks it every single time. The first
# version of this produced 161 warriors and 4 settlers across a whole match, and
# the map never got settled.
BUILDING_ORDER = [
    "granary",
    "library",
    "barracks",
    "market",
    "temple",
    "walls",
    "aqueduct",
    "amphitheater",
    "pyramids",
    "great_library",
]

# Best military unit available, strongest first.
MILITARY_ORDER = ["swordsman", "horseman", "archer", "spearman", "warrior"]

# Kept out of MILITARY_ORDER so triremes are built as a deliberate minority
# rather than crowding out the army the moment a city touches the coast.
NAVY = "trireme"
NAVY_PER_CITIES = 4

# Target city count before a civ stops making settlers, and the garrison it
# wants per city before it builds anything else.
TARGET_CITIES = 7
GARRISON_PER_CITY = 2

_MILITARY_NAMES = frozenset(MILITARY_ORDER) | {"catapult"}


# Aggression rises over the match so early expansion is peaceful and the
# endgame is not. Without this the bots coexist to the turn limit and the
# conquest and domination conditions never get exercised.
def _aggression(turn: int) -> float:
    return min(0.9, turn / 160)


def act(state: State, player_id: str) -> Action:
    """Produce one turn of orders for a bot civ."""
    legal = legal_actions(state, player_id)
    orders: list[Order] = []
    diplomacy: list = []

    _handle_diplomacy(state, player_id, legal, diplomacy)
    _order_units(state, player_id, legal, orders)
    _order_cities(state, player_id, legal, orders)
    _order_research(state, player_id, legal, orders)
    _order_rates(state, player_id, orders)

    return Action(
        reasoning=Reasoning(
            situation_assessment=f"Heuristic bot, turn {state.turn}.",
            plan_this_turn="Expand, improve, and press any advantage.",
        ),
        diplomacy=diplomacy,
        orders=orders,
    )


def _handle_diplomacy(state: State, player_id: str, legal: dict, out: list) -> None:
    diplo = legal["diplomacy"]

    # Accept anything offered. A bot that never agrees to anything would leave
    # the treaty and trade paths completely untested.
    for proposal_id in diplo["respond_to_proposal"]:
        out.append(
            RespondToProposal(
                action="respond_to_proposal", proposal_id=proposal_id, response="accept"
            )
        )

    # `legal_actions` names civs rather than keying them, because it is written
    # for the models - who are told to address rivals as "Gemini", not "p3".
    # Bots read the same list and have to translate back, and getting this wrong
    # is silent: intersecting names against ids yields the empty set, so every
    # bot simply stopped declaring war and the matches went quiet.
    targets = [pid for pid in (state.resolve(t) for t in diplo["declare_war"]) if pid]
    if not targets:
        return
    # Only declare on a civ we can actually see, and only once aggression has
    # ramped. Picking a target we have never met would be noise, not strategy.
    seen = set(state.contact.get(player_id, []))
    visible_targets = sorted(set(targets) & seen)
    if not visible_targets:
        return
    target = visible_targets[0]
    name = state.players[target].civ_name

    if rng.chance(state.seed, _aggression(state.turn) * 0.06, "bot_war", state.turn, player_id):
        out.append(DeclareWar(action="declare_war", on=target, casus_belli="Territorial ambition."))
        out.append(
            SendMessage(
                action="send_message",
                channel="public",
                text=f"{state.players[player_id].civ_name} declares war upon {name}. "
                f"Their borders press too close to ours.",
            )
        )
        return

    # Sue for peace when losing, and court the uncommitted otherwise. Bots that
    # never spoke left the diplomacy console, the treaty timeline and the
    # deception panel with nothing to render and nothing to test against.
    if state.at_war(player_id, target):
        if _weaker_than(state, player_id, target) and rng.chance(
            state.seed, 0.25, "bot_peace", state.turn, player_id
        ):
            out.append(
                Propose(
                    action="propose",
                    to=target,
                    type=ProposalType.PEACE,
                    message="This war profits neither of us. Let it end.",
                )
            )
            out.append(
                SendMessage(
                    action="send_message",
                    channel="private",
                    to=target,
                    text="We propose peace. Our quarrel has cost us both enough.",
                )
            )
    elif rng.chance(state.seed, 0.04, "bot_pact", state.turn, player_id):
        out.append(
            Propose(
                action="propose",
                to=target,
                type=ProposalType.NON_AGGRESSION,
                terms=Terms(duration_turns=25),
                message="Neither of us gains from a war on this frontier.",
            )
        )
        out.append(
            SendMessage(
                action="send_message",
                channel="private",
                to=target,
                text=f"We have no quarrel with {name}. Let us agree to keep it so.",
            )
        )


def _weaker_than(state: State, player_id: str, other: str) -> bool:
    def power(pid: str) -> int:
        return sum(
            UNITS[u.type].attack + UNITS[u.type].defense
            for u in state.units_of(pid)
            if not UNITS[u.type].civilian
        )

    return power(player_id) < power(other) * 0.8


def _order_units(state: State, player_id: str, legal: dict, out: list[Order]) -> None:
    # `legal_actions` is a snapshot of turn start, and orders resolve after it.
    # Three units ordered onto the same defender means the first one kills it
    # and the other two are rejected with "nothing hostile there" - 1504 wasted
    # orders in a single measured match. Claiming targets within the turn is the
    # bot's job; the engine is right to reject the stale ones.
    claimed_targets: set[str] = set()
    claimed_tiles: set[str] = set()

    for unit in state.units_of(player_id):
        options = legal["units"].get(unit.id)
        if options is None:
            continue

        # At sea is the worst place to be caught, so always look for a landing.
        if unit.embarked and unit.type is not UnitType.SETTLER:
            ashore = _free_moves_from(options, "move_unit_landing", claimed_tiles)
            if ashore:
                step = sorted(ashore)[0]
                claimed_tiles.add(step)
                out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=step))
                continue

        if unit.type is UnitType.SETTLER:
            _order_settler(state, unit, options, out, claimed_tiles)
        elif unit.type is UnitType.WORKER:
            _order_worker(state, unit, options, out, claimed_tiles)
        elif targets := [t for t in options["attack"] if t not in claimed_targets]:
            # Take any fight that is still unclaimed. Combat is attritional, so
            # even a losing attack is pressure the engine needs to exercise.
            claimed_targets.add(targets[0])
            out.append(Attack(action="attack", unit_id=unit.id, target=targets[0]))
        elif unit.type is UnitType.SCOUT:
            _wander(state, unit, options, out, claimed_tiles)
        else:
            _order_soldier(state, player_id, unit, options, out, claimed_tiles)


def _free_moves(options: dict, claimed: set[str]) -> list[str]:
    """Moves not already taken by another unit of ours this turn."""
    return _free_moves_from(options, "move_unit", claimed)


def _free_moves_from(options: dict, key: str, claimed: set[str]) -> list[str]:
    return [m for m in options.get(key, []) if m not in claimed]


def _order_settler(  # noqa: ANN001
    state: State, unit, options: dict, out: list[Order], claimed: set[str]
) -> None:
    if options["found_city"] and _site_quality(state, unit.hex) >= 6:
        out.append(FoundCity(action="found_city", unit_id=unit.id, name=""))
        return
    # Walk towards the best nearby site rather than wandering, or the map never
    # gets settled and every match ends on the turn limit.
    best = _best_site_step(state, unit, _free_moves(options, claimed))
    if best is not None:
        claimed.add(best)
        out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=best))
        return
    # Nowhere worth going on foot. Put to sea and look for open land elsewhere,
    # which is the whole reason embarkation exists.
    overseas = _free_moves_from(options, "move_unit_embarking", claimed) or _free_moves_from(
        options, "move_unit_landing", claimed
    )
    if overseas:
        step = sorted(overseas)[0]
        claimed.add(step)
        out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=step))
    elif options["found_city"]:
        out.append(FoundCity(action="found_city", unit_id=unit.id, name=""))


def _site_quality(state: State, h: Hex) -> int:
    return sum(economy._tile_score(economy.tile_yields(state, n)) for n in hx.within(h, 1)) // 4


def _best_site_step(state: State, unit, moves: list[str]) -> str | None:  # noqa: ANN001
    """Step towards the most promising nearby city site.

    Always returns a move when one exists. An earlier version only returned a
    step when its score cleared zero, and since standing next to your own city
    penalises every neighbouring tile, a settler that spawned beside its capital
    scored negative in all six directions, received no order at all, and stood
    still for the rest of the match. Preferring the least-bad direction always
    beats issuing nothing.
    """
    if not moves:
        return None
    scored = []
    for key in moves:
        h = hx.from_key(key)
        too_close = any(
            hx.distance(c.hex, h) < MIN_CITY_SPACING for _, c in sorted(state.cities.items())
        )
        # Break out of a crowded area rather than orbiting it: when everything
        # nearby is too close, the tiebreak is distance from the nearest city.
        crowding = -100 if too_close else 0
        spread = min(
            (hx.distance(c.hex, h) for _, c in sorted(state.cities.items())),
            default=0,
        )
        scored.append((_site_quality(state, h) + crowding + spread * 4, key))
    return max(scored, key=lambda t: (t[0], t[1]))[1]


def _order_worker(  # noqa: ANN001
    state: State, unit, options: dict, out: list[Order], claimed: set[str]
) -> None:
    if unit.working_on is not None:
        return
    tile = state.at(unit.hex)
    improvements = options["build_improvement"]
    if tile is not None and tile.owner == unit.owner and improvements and tile.improvement is None:
        # Mines on production tiles, farms elsewhere.
        want = "mine" if "mine" in improvements else improvements[0]
        out.append(BuildImprovement(action="build_improvement", unit_id=unit.id, improvement=want))
        return
    _step_towards_own_territory(state, unit, _free_moves(options, claimed), out, claimed)


def _step_towards_own_territory(  # noqa: ANN001
    state: State, unit, moves: list[str], out: list[Order], claimed: set[str]
) -> None:
    if not moves:
        return
    owned = [
        key
        for key in moves
        if (t := state.tiles.get(key)) is not None
        and t.owner == unit.owner
        and t.improvement is None
    ]
    step = sorted(owned or moves)[0]
    claimed.add(step)
    out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=step))


def _order_soldier(  # noqa: ANN001
    state: State, player_id: str, unit, options: dict, out: list[Order], claimed: set[str]
) -> None:
    """Garrison the nearest city, or march on a rival once at war."""
    enemies = [p for p in state.civ_ids() if p != player_id and state.at_war(player_id, p)]
    if enemies:
        target = _nearest_enemy_city(state, unit.hex, enemies)
        moves = _free_moves(options, claimed)
        if target is not None and moves:
            step = min(moves, key=lambda k: (hx.distance(hx.from_key(k), target), k))
            claimed.add(step)
            out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=step))
            return
    if not unit.fortified:
        out.append(Fortify(action="fortify", unit_id=unit.id))


def _nearest_enemy_city(state: State, origin: Hex, enemies: list[str]) -> Hex | None:
    candidates = [c.hex for _, c in sorted(state.cities.items()) if c.owner in enemies]
    if not candidates:
        return None
    return min(candidates, key=lambda h: (hx.distance(origin, h), h.to_key()))


def _wander(  # noqa: ANN001
    state: State, unit, options: dict, out: list[Order], claimed: set[str]
) -> None:
    """Explore away from home, deterministically."""
    moves = _free_moves(options, claimed)
    if not moves:
        return
    step = max(moves, key=lambda k: (hx.distance(hx.ORIGIN, hx.from_key(k)), k))
    if rng.chance(state.seed, 0.35, "bot_wander", state.turn, unit.id):
        step = rng.choice(state.seed, sorted(moves), "bot_wander_pick", state.turn, unit.id)
    claimed.add(step)
    out.append(MoveUnit(action="move_unit", unit_id=unit.id, to=step))


def _order_cities(state: State, player_id: str, legal: dict, out: list[Order]) -> None:
    units = state.units_of(player_id)
    counts = {
        "settler": sum(1 for u in units if u.type is UnitType.SETTLER),
        "worker": sum(1 for u in units if u.type is UnitType.WORKER),
        "military": sum(1 for u in units if not UNITS[u.type].civilian),
        "navy": sum(1 for u in units if u.type is UnitType.TRIREME),
    }
    cities = state.cities_of(player_id)
    # Upkeep is real: a civ that overspends has units disbanded out from under
    # it. Building a soldier you cannot pay for and losing it next turn is pure
    # churn, so check the books before adding to the payroll.
    income, _, _ = economy.player_output(state, player_id)
    broke = income <= 0 and state.players[player_id].gold < 25

    for city in cities:
        if city.building is not None:
            continue
        options = legal["cities"].get(city.id, {}).get("set_production", [])
        if not options:
            continue
        want = _wanted(state, city, len(cities), counts, options, broke)
        if want is None:
            # Nothing worth building. Let production bank rather than making
            # units for their own sake: falling through to "best available"
            # gave a four-city civ 120 units, which is not a strategy and blew
            # the observation payload past its budget.
            continue
        out.append(SetProduction(action="set_production", city_id=city.id, item=want))
        # Count the decision immediately, or every city in the empire decides
        # to build the same settler on the same turn.
        if want == NAVY:
            counts["navy"] += 1
            counts["military"] += 1
        elif want in counts:
            counts[want] += 1
        elif want in _MILITARY_NAMES:
            counts["military"] += 1


def _wanted(
    state: State,
    city,  # noqa: ANN001
    city_count: int,
    counts: dict[str, int],
    options: list[str],
    broke: bool = False,
) -> str | None:
    """Pick a build from what the civ actually lacks, not by list position.

    The list-position approach does not work here: `warrior` is buildable on
    every turn of the game, so scanning a fixed priority list front-to-back
    selects it every time and the civ never builds anything else.
    """
    military_target = max(2, city_count * GARRISON_PER_CITY)
    if broke:
        # Build the way out of it: gold buildings first, then anything free to
        # maintain, and never another unit on the payroll.
        return _first_available(["market", "granary", "library", "temple"], options)

    # A minimum garrison first; an undefended civ gets wiped out early.
    if counts["military"] < min(2, military_target):
        best = _first_available(MILITARY_ORDER, options)
        if best:
            return best

    # Expand while there is room. Population 2 so producing a settler does not
    # reduce the city to nothing.
    if (
        city_count + counts["settler"] < TARGET_CITIES
        and city.population >= 2
        and "settler" in options
    ):
        return "settler"

    if counts["worker"] < city_count and "worker" in options:
        return "worker"

    if counts["military"] < military_target:
        best = _first_available(MILITARY_ORDER, options)
        if best:
            return best

    # A token navy on the coast, to exercise naval combat and screen crossings.
    if NAVY in options and counts.get("navy", 0) < max(1, city_count // NAVY_PER_CITIES):
        return NAVY

    # The science victory, the moment it becomes reachable.
    if "apex_project" in options:
        return "apex_project"

    building = _first_available(BUILDING_ORDER, options)
    if building:
        return building

    # Everything worth having is built. Keep a standing army proportional to the
    # empire rather than an unbounded one: falling through to "best military
    # unit" here produced 120 units on four cities, which is not a strategy and
    # made the observation payload balloon past its budget.
    if counts["military"] < military_target * 2:
        return _first_available(MILITARY_ORDER, options)
    return None


def _affordable(options: list[str], preferred: list[str]) -> str | None:
    """The best preferred option that costs nothing to maintain."""
    free = {u.value for u in UnitType if UNITS[u].upkeep == 0} | {
        b for b in options if not _is_unit_name(b)
    }
    return _first_available([p for p in preferred if p in free], options)


def _is_unit_name(name: str) -> bool:
    return name in {u.value for u in UnitType}


def _first_available(preferred: list[str], options: list[str]) -> str | None:
    available = set(options)
    return next((item for item in preferred if item in available), None)


def _order_rates(state: State, player_id: str, out: list[Order]) -> None:
    """Balance the books with the tax slider before cutting the army.

    Bots never touched this, so a civ running a deficit had units disbanded from
    under it every turn while sitting on a lever that would have fixed it. Left
    alone that produced 794 disbands in a single match, and then - once building
    was made income-aware - civs so cautious they fielded two units between them.
    """
    player = state.players[player_id]
    income, _, _ = economy.player_output(state, player_id)

    if income < 0 or (player.gold < 25 and income <= 0):
        tax = min(100, player.tax_pct + 20)
    elif player.gold > 150 and income > 4:
        # Comfortable: buy research instead of hoarding.
        tax = max(20, player.tax_pct - 20)
    else:
        return

    if tax != player.tax_pct:
        out.append(SetRates(action="set_rates", tax_pct=tax, science_pct=100 - tax))


def _order_research(state: State, player_id: str, legal: dict, out: list[Order]) -> None:
    player = state.players[player_id]
    if player.researching is not None:
        return
    options = legal["set_research"]
    if options:
        out.append(SetResearch(action="set_research", tech=sorted(options)[0]))


def all_bot_actions(state: State) -> dict[str, Action]:
    return {p: act(state, p) for p in state.living_civ_ids()}


def passable_neighbors(state: State, h: Hex) -> list[Hex]:
    return [
        n for n in hx.neighbors(h) if (t := state.at(n)) is not None and TERRAIN[t.terrain].passable
    ]


__all__ = ["act", "all_bot_actions", "BUILDING_ORDER", "MILITARY_ORDER"]
