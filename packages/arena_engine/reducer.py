"""The step function: `(state, actions) -> (state, events)`.

Pure in the sense that matters: it never touches the network, the clock, or a
global RNG, and it deep-copies the incoming state rather than mutating it. Feed
it the same state and the same actions and it produces the same next state and
the same events, forever. That is what the replay-determinism test rests on.

**Simultaneous resolution.** Every civ's orders are validated against the state
as it stood at the start of the turn, then applied in a rotation that advances
each turn, so no civ permanently owns first strike. An order that was legal at
turn start but has become impossible - the target moved, the tile was taken -
is rejected individually with an `order_rejected` event. The rest of that civ's
orders still apply. A single bad order must never cost an agent its whole turn.
"""

from __future__ import annotations

from arena_engine import combat, diplomacy, economy, movement, victory, visibility
from arena_engine import events as ev
from arena_engine import hex as hx
from arena_engine.actions import Action
from arena_engine.content import (
    IMPROVEMENTS,
    MIN_CITY_SPACING,
    TECHS,
    TERRAIN,
    UNITS,
    WONDERS,
    WORKER_IMPROVEMENTS,
    Improvement,
    UnitType,
    available_techs,
)
from arena_engine.events import Event
from arena_engine.types import City, MatchConfig, Player, State, Unit

CITY_NAMES = [
    "Ravenholt",
    "Kaldis",
    "Thornwatch",
    "Emberfall",
    "Highmoor",
    "Stonevale",
    "Duskmere",
    "Ironhold",
    "Westreach",
    "Amberfen",
    "Coldharbour",
    "Sunspire",
]


def new_match(
    match_id: str,
    seed: int,
    roster: list[tuple[str, str]],
    config: MatchConfig | None = None,
) -> tuple[State, list[Event]]:
    """Create turn 0. `roster` is a list of `(player_id, civ_name)`."""
    from arena_engine import mapgen

    config = config or MatchConfig()
    generated = mapgen.generate(
        seed,
        config.radius,
        player_count=len(roster),
        start_radius_fraction=config.start_radius_fraction,
    )
    state = State(match_id=match_id, seed=seed, turn=0, config=config, tiles=dict(generated.tiles))

    for (player_id, civ_name), start in zip(roster, generated.starts, strict=True):
        state.players[player_id] = Player(
            id=player_id, civ_name=civ_name, gold=config.starting_gold
        )
        for index, unit_type in enumerate(config.starting_units):
            # Fan the starting stack out around the site so a single early
            # attack cannot take the whole civ's opening in one blow.
            offset = hx.ORIGIN if index == 0 else hx.neighbors(start)[index % 6] - start
            state.units[f"u{state.next_id}"] = Unit(
                id=f"u{state.next_id}",
                owner=player_id,
                type=unit_type,
                pos=(start + offset).to_key(),
                moves_left=UNITS[unit_type].moves,
            )
            state.next_id += 1

    report = visibility.compute(state)
    visibility.apply(state, report)
    return state, [ev.event(0, ev.TURN_STARTED, f"Match {match_id} begins", seed=seed)]


def step(state: State, actions: dict[str, Action]) -> tuple[State, list[Event]]:
    """Advance one full turn."""
    s = state.model_copy(deep=True)
    s.turn += 1
    out: list[Event] = [ev.event(s.turn, ev.TURN_STARTED, f"Turn {s.turn}")]

    _begin_turn(s, out)
    order = _rotation(s)

    for player_id in order:
        action = actions.get(player_id)
        if action is None or not s.players[player_id].alive:
            continue
        _log_reasoning(s, player_id, action, out)
        _store_dossier(s, player_id, action, out)
        _apply_diplomacy(s, player_id, action, out)

    for player_id in order:
        action = actions.get(player_id)
        if action is None or not s.players[player_id].alive:
            continue
        for single in action.orders:
            _apply_order(s, player_id, single, out)

    _end_turn(s, out)
    _refresh_vision(s, out)
    _check_elimination(s, out)

    result = victory.check(s)
    if result is not None:
        s.victory = result
        out.append(
            ev.event(
                s.turn,
                ev.MATCH_ENDED,
                result.detail,
                actor=result.winner,
                condition=result.condition,
                scores=result.scores,
            )
        )
    out.append(ev.event(s.turn, ev.TURN_ENDED, f"Turn {s.turn} resolved"))
    return s, out


def _rotation(state: State) -> list[str]:
    """Player order for this turn, rotating so first strike moves around."""
    living = state.living_player_ids()
    if not living:
        return []
    shift = state.turn % len(living)
    return living[shift:] + living[:shift]


# ---------------------------------------------------------------------------
# Turn phases
# ---------------------------------------------------------------------------


def _begin_turn(s: State, out: list[Event]) -> None:
    for e in diplomacy.expire_proposals(s) + diplomacy.expire_pacts(s):
        out.append(ev.event(s.turn, e.kind, e.detail, actor=e.actor, other=e.other))

    for _, unit in sorted(s.units.items()):
        unit.moves_left = movement.moves_for(s, unit)
        if unit.working_on is not None and unit.work_turns_left > 0:
            unit.work_turns_left -= 1
            if unit.work_turns_left == 0:
                tile = s.tiles.get(unit.pos)
                if tile is not None:
                    s.tiles[unit.pos] = tile.model_copy(update={"improvement": unit.working_on})
                    out.append(
                        ev.event(
                            s.turn,
                            ev.IMPROVEMENT_BUILT,
                            f"{unit.working_on.value} completed at {unit.pos}",
                            actor=unit.owner,
                            pos=unit.pos,
                            improvement=unit.working_on.value,
                        )
                    )
                unit.working_on = None


def _end_turn(s: State, out: list[Event]) -> None:
    for player_id in s.player_ids():
        player = s.players[player_id]
        if not player.alive:
            continue

        for city in s.cities_of(player_id):
            economy.assign_tiles(s, city)

        gold, science, culture = economy.player_output(s, player_id)
        player.gold = max(0, player.gold + gold)
        player.culture += culture

        completed, tech = economy.research_progress(s, player, science)
        if completed and tech is not None:
            player.known_techs = sorted({*player.known_techs, tech})
            player.researching = None
            out.append(
                ev.event(
                    s.turn,
                    ev.TECH_COMPLETED,
                    f"{player.civ_name} discovered {tech}",
                    actor=player_id,
                    tech=tech,
                )
            )
        # Auto-pick the next tech so a civ never idles its science because the
        # agent forgot to choose. Cheapest available, ties on name.
        if player.researching is None:
            options = available_techs(frozenset(player.known_techs))
            if options:
                player.researching = min(options, key=lambda t: (TECHS[t].cost, t))

        for city in s.cities_of(player_id):
            _advance_city(s, city, out)
            # Re-assign after growth. Tiles are assigned before the food step,
            # so a city that starves down a size in the same turn would keep
            # working more tiles than it has citizens to work them.
            economy.assign_tiles(s, city)


def _develop_worked_water(s: State, city: City) -> None:
    """Put fishing boats on water tiles the city works.

    Automatic rather than worker-built, because a worker cannot walk onto water
    to build anything there. Without this a coastal site is strictly worse than
    an inland one - half its workable tiles would be permanently unimprovable -
    and nobody would ever settle the shore that naval play depends on.
    """
    for key in city.worked_tiles:
        tile = s.tiles.get(key)
        if tile is None or tile.improvement is not None:
            continue
        if TERRAIN[tile.terrain].navigable:
            s.tiles[key] = tile.model_copy(update={"improvement": Improvement.FISHING_BOATS})


def _advance_city(s: State, city: City, out: list[Event]) -> None:
    _develop_worked_water(s, city)
    growth = economy.grow_city(city, economy.food_surplus(s, city))
    if growth == "grew":
        out.append(
            ev.event(
                s.turn,
                ev.CITY_GREW,
                f"{city.name} grew to {city.population}",
                actor=city.owner,
                city_id=city.id,
                population=city.population,
            )
        )
    elif growth == "shrank":
        out.append(
            ev.event(
                s.turn,
                ev.CITY_SHRANK,
                f"{city.name} starved down to {city.population}",
                actor=city.owner,
                city_id=city.id,
                population=city.population,
            )
        )

    if city.building is None:
        return
    city.production_stored += economy.city_yields(s, city).production
    cost = economy.build_cost(city.building)
    if city.production_stored < cost:
        return

    item = city.building

    # Re-check the wonder gate at completion, not just when production was set.
    # `can_build_building` runs when an agent chooses what to build, but several
    # cities can legally choose the same wonder on the same turn, or before any
    # of them finishes. Without this check they all complete it: a measured
    # match produced eight Great Libraries. Refund rather than discard, so the
    # loser of the race keeps its production.
    if not economy.is_unit(item) and item in WONDERS:
        already = any(item in c.buildings for _, c in sorted(s.cities.items()))
        if already:
            city.building = None
            out.append(
                ev.event(
                    s.turn,
                    ev.ORDER_REJECTED,
                    f"{city.name} lost the race to build {item}; production retained",
                    actor=city.owner,
                    city_id=city.id,
                    item=item,
                )
            )
            return

    city.production_stored -= cost
    city.building = None

    if economy.is_unit(item):
        unit_type = UnitType(item)
        uid, s.next_id = s.new_id("u")
        s.units[uid] = Unit(id=uid, owner=city.owner, type=unit_type, pos=city.pos, moves_left=0)
    else:
        city.buildings = sorted({*city.buildings, item})
    out.append(
        ev.event(
            s.turn,
            ev.BUILD_COMPLETED,
            f"{city.name} completed {item}",
            actor=city.owner,
            city_id=city.id,
            item=item,
        )
    )


def _refresh_vision(s: State, out: list[Event]) -> None:
    previous = s.contact
    report = visibility.compute(s)
    gained, lost = visibility.contact_changes(previous, report.contact)
    visibility.apply(s, report)

    for observer, observed in gained:
        out.append(
            ev.event(
                s.turn,
                ev.FIRST_CONTACT,
                f"{s.players[observer].civ_name} sighted {s.players[observed].civ_name}",
                actor=observer,
                observed=observed,
            )
        )
    for observer, observed in lost:
        out.append(
            ev.event(
                s.turn,
                ev.CONTACT_LOST,
                f"{s.players[observer].civ_name} lost sight of {s.players[observed].civ_name}",
                actor=observer,
                observed=observed,
            )
        )


def _check_elimination(s: State, out: list[Event]) -> None:
    for player_id in s.player_ids():
        player = s.players[player_id]
        if not player.alive:
            continue
        if not s.cities_of(player_id) and not s.units_of(player_id):
            player.alive = False
            player.eliminated_turn = s.turn
            out.append(
                ev.event(
                    s.turn,
                    ev.PLAYER_ELIMINATED,
                    f"{player.civ_name} has been eliminated",
                    actor=player_id,
                )
            )


# ---------------------------------------------------------------------------
# Diplomacy
# ---------------------------------------------------------------------------


def _log_reasoning(s: State, player_id: str, action: Action, out: list[Event]) -> None:
    """Record what the agent said it was thinking, before anything is applied.

    Emitted by the engine rather than only by the orchestrator so that a bot
    match produces a fully watchable, fully exportable replay with no API calls
    at all. That is what lets the viewer and the published-match bundle be built
    and tested end to end before the first dollar is spent on a model.

    The orchestrator adds the expensive half around this - the exact prompt, the
    raw response, token usage - which stays in Postgres and is deliberately kept
    out of the published bundle.
    """
    r = action.reasoning
    out.append(
        ev.event(
            s.turn,
            ev.AGENT_ACTION,
            r.plan_this_turn or f"{s.players[player_id].civ_name} acted",
            actor=player_id,
            situation_assessment=r.situation_assessment,
            threats_and_opportunities=r.threats_and_opportunities,
            plan_this_turn=r.plan_this_turn,
            confidence=r.confidence,
            order_count=len(action.orders),
            diplomacy_count=len(action.diplomacy),
        )
    )


def _store_dossier(s: State, player_id: str, action: Action, out: list[Event]) -> None:
    """Persist the agent's self-authored memory verbatim.

    The engine never edits the content. It only truncates an over-long one, and
    it says so in the log rather than silently dropping the tail.
    """
    dossier = action.dossier
    limit = 12
    if len(dossier.lessons) > limit or len(dossier.standing_commitments) > limit:
        dossier = dossier.model_copy(
            update={
                "lessons": dossier.lessons[:limit],
                "standing_commitments": dossier.standing_commitments[:limit],
            }
        )
        out.append(
            ev.event(
                s.turn,
                ev.DOSSIER_TRUNCATED,
                f"{s.players[player_id].civ_name}'s dossier exceeded its size cap",
                actor=player_id,
            )
        )
    s.players[player_id].dossier = dossier


def _apply_diplomacy(s: State, player_id: str, action: Action, out: list[Event]) -> None:
    for item in action.diplomacy:
        match item.action:
            case "send_message":
                target = item.to if item.channel == "private" else None
                if item.channel == "private" and (target not in s.players or target == player_id):
                    _reject(s, out, player_id, "send_message", f"unknown recipient {item.to!r}")
                    continue
                diplomacy.send_message(s, player_id, item.channel, item.text, target)
                out.append(
                    ev.event(
                        s.turn,
                        ev.MESSAGE_SENT,
                        f"{s.players[player_id].civ_name} sent a {item.channel} message",
                        actor=player_id,
                        channel=item.channel,
                        to=target,
                        text=item.text,
                    )
                )
            case "propose":
                if item.to not in s.players or item.to == player_id:
                    _reject(s, out, player_id, "propose", f"unknown recipient {item.to!r}")
                    continue
                p = diplomacy.open_proposal(
                    s, player_id, item.to, item.type, item.terms, item.message
                )
                out.append(
                    ev.event(
                        s.turn,
                        ev.PROPOSAL_MADE,
                        f"{s.players[player_id].civ_name} proposed "
                        f"{item.type.value} to {s.players[item.to].civ_name}",
                        actor=player_id,
                        proposal_id=p.id,
                        to=item.to,
                        type=item.type.value,
                    )
                )
            case "respond_to_proposal":
                proposal = s.proposals.get(item.proposal_id)
                if proposal is None or proposal.to_player != player_id:
                    _reject(
                        s,
                        out,
                        player_id,
                        "respond_to_proposal",
                        f"no open proposal {item.proposal_id!r} addressed to you",
                    )
                    continue
                results = (
                    diplomacy.accept(s, proposal)
                    if item.response == "accept"
                    else [diplomacy.reject(s, proposal)]
                )
                for e in results:
                    out.append(ev.event(s.turn, e.kind, e.detail, actor=e.actor, other=e.other))
            case "declare_war":
                if item.on not in s.players or item.on == player_id:
                    _reject(s, out, player_id, "declare_war", f"unknown target {item.on!r}")
                    continue
                for e in diplomacy.declare_war(s, player_id, item.on):
                    out.append(
                        ev.event(
                            s.turn,
                            e.kind,
                            e.detail,
                            actor=e.actor,
                            other=e.other,
                            casus_belli=item.casus_belli,
                        )
                    )


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------


def _reject(s: State, out: list[Event], player_id: str, what: str, why: str) -> None:
    out.append(
        ev.event(
            s.turn,
            ev.ORDER_REJECTED,
            f"{what} rejected: {why}",
            actor=player_id,
            order=what,
            reason=why,
        )
    )


def _own_unit(s: State, player_id: str, unit_id: str) -> Unit | None:
    unit = s.units.get(unit_id)
    return unit if unit is not None and unit.owner == player_id else None


def _apply_order(s: State, player_id: str, order, out: list[Event]) -> None:  # noqa: ANN001
    match order.action:
        case "move_unit":
            _move(s, player_id, order, out)
        case "attack":
            _attack(s, player_id, order, out)
        case "fortify":
            unit = _own_unit(s, player_id, order.unit_id)
            if unit is None:
                _reject(s, out, player_id, "fortify", f"no such unit {order.unit_id!r}")
                return
            unit.fortified = True
            unit.working_on = None
            out.append(
                ev.event(
                    s.turn,
                    ev.UNIT_FORTIFIED,
                    f"{unit.type.value} fortified",
                    actor=player_id,
                    unit_id=unit.id,
                    pos=unit.pos,
                )
            )
        case "found_city":
            _found_city(s, player_id, order, out)
        case "build_improvement":
            _build_improvement(s, player_id, order, out)
        case "set_production":
            city = s.cities.get(order.city_id)
            if city is None or city.owner != player_id:
                _reject(s, out, player_id, "set_production", f"no such city {order.city_id!r}")
                return
            if order.item not in economy.buildable(s, city):
                _reject(
                    s,
                    out,
                    player_id,
                    "set_production",
                    f"{city.name} cannot build {order.item!r} right now",
                )
                return
            city.building = order.item
        case "set_research":
            player = s.players[player_id]
            if order.tech not in available_techs(frozenset(player.known_techs)):
                _reject(
                    s,
                    out,
                    player_id,
                    "set_research",
                    f"{order.tech!r} is not available to research",
                )
                return
            player.researching = order.tech
            out.append(
                ev.event(
                    s.turn,
                    ev.RESEARCH_SET,
                    f"researching {order.tech}",
                    actor=player_id,
                    tech=order.tech,
                )
            )
        case "set_rates":
            if (
                order.tax_pct + order.science_pct != 100
                or min(order.tax_pct, order.science_pct) < 0
            ):
                _reject(s, out, player_id, "set_rates", "rates must be non-negative and sum to 100")
                return
            s.players[player_id].tax_pct = order.tax_pct
            s.players[player_id].science_pct = order.science_pct


def _move(s: State, player_id: str, order, out: list[Event]) -> None:  # noqa: ANN001
    unit = _own_unit(s, player_id, order.unit_id)
    if unit is None:
        _reject(s, out, player_id, "move_unit", f"no such unit {order.unit_id!r}")
        return
    try:
        target = hx.from_key(order.to)
    except ValueError:
        _reject(s, out, player_id, "move_unit", f"malformed destination {order.to!r}")
        return

    if hx.distance(unit.hex, target) != 1:
        _reject(s, out, player_id, "move_unit", "destination is not adjacent")
        return
    allowed, why = movement.entry_check(s, unit, target)
    if not allowed:
        _reject(s, out, player_id, "move_unit", why)
        return
    if unit.moves_left <= 0:
        _reject(s, out, player_id, "move_unit", "no movement remaining")
        return

    occupants = s.units_at(target)
    if any(u.owner != player_id for u in occupants):
        _reject(s, out, player_id, "move_unit", "tile is occupied by another civ; attack instead")
        return
    city = s.city_at(target)
    if city is not None and city.owner != player_id:
        _reject(s, out, player_id, "move_unit", "tile holds a rival city; attack instead")
        return

    change = movement.apply_move(s, unit, target)
    verb = {"embark": "put to sea at", "disembark": "landed at"}.get(change, "moved to")
    out.append(
        ev.event(
            s.turn,
            ev.UNIT_MOVED,
            f"{unit.type.value} {verb} {order.to}",
            actor=player_id,
            unit_id=unit.id,
            pos=order.to,
            transition=change,
            embarked=unit.embarked,
        )
    )


def _attack(s: State, player_id: str, order, out: list[Event]) -> None:  # noqa: ANN001
    attacker = _own_unit(s, player_id, order.unit_id)
    if attacker is None:
        _reject(s, out, player_id, "attack", f"no such unit {order.unit_id!r}")
        return
    if UNITS[attacker.type].civilian:
        _reject(s, out, player_id, "attack", f"{attacker.type.value} cannot attack")
        return
    if attacker.embarked:
        _reject(s, out, player_id, "attack", "a unit at sea cannot attack; land first")
        return
    try:
        target = hx.from_key(order.target)
    except ValueError:
        _reject(s, out, player_id, "attack", f"malformed target {order.target!r}")
        return
    if hx.distance(attacker.hex, target) != 1:
        _reject(s, out, player_id, "attack", "target is not adjacent")
        return
    if attacker.moves_left <= 0:
        _reject(s, out, player_id, "attack", "no movement remaining")
        return

    defenders = [u for u in s.units_at(target) if u.owner != player_id]
    city = s.city_at(target)
    if not defenders and (city is None or city.owner == player_id):
        _reject(s, out, player_id, "attack", "nothing hostile there")
        return

    owner = defenders[0].owner if defenders else (city.owner if city else None)
    if owner is not None and not s.at_war(player_id, owner):
        _reject(
            s,
            out,
            player_id,
            "attack",
            f"not at war with {s.players[owner].civ_name}; declare war first",
        )
        return

    attacker.moves_left = 0
    attacker.fortified = False

    defender = combat.best_defender(s, defenders)
    if defender is None:
        _capture_city(s, player_id, city, attacker, out)
        return

    result = combat.resolve(s, attacker, defender)
    if result.captured:
        defender.owner = player_id
        out.append(
            ev.event(
                s.turn,
                ev.UNIT_CAPTURED,
                f"captured a {defender.type.value}",
                actor=player_id,
                unit_id=defender.id,
                from_player=owner,
            )
        )
        return

    attacker.hp -= result.attacker_damage
    defender.hp -= result.defender_damage
    out.append(
        ev.event(
            s.turn,
            ev.COMBAT_RESOLVED,
            f"{attacker.type.value} attacked {defender.type.value} at {order.target}",
            actor=player_id,
            attacker=attacker.id,
            defender=defender.id,
            attacker_damage=result.attacker_damage,
            defender_damage=result.defender_damage,
        )
    )

    if result.defender_died:
        del s.units[defender.id]
        out.append(
            ev.event(
                s.turn,
                ev.UNIT_KILLED,
                f"{defender.type.value} destroyed",
                actor=player_id,
                unit_id=defender.id,
                owner=defender.owner,
            )
        )
    if result.attacker_died:
        del s.units[attacker.id]
        out.append(
            ev.event(
                s.turn,
                ev.UNIT_KILLED,
                f"{attacker.type.value} destroyed",
                actor=owner,
                unit_id=attacker.id,
                owner=player_id,
            )
        )

    # Taking the last defender leaves the city open; the attacker walks in next
    # turn rather than capturing in the same blow, which gives the loser one
    # turn to counter and makes a defended city genuinely worth holding.
    if city is not None and result.defender_died and combat.city_falls(s, city):
        out.append(
            ev.event(
                s.turn,
                ev.COMBAT_RESOLVED,
                f"{city.name} stands undefended",
                actor=player_id,
                city_id=city.id,
            )
        )


def _capture_city(
    s: State, player_id: str, city: City | None, attacker: Unit, out: list[Event]
) -> None:
    if city is None:
        return
    former = city.owner
    city.owner = player_id
    city.population = max(1, city.population - 1)
    city.building = None
    city.production_stored = 0
    city.worked_tiles = []
    attacker.pos = city.pos
    for key in list(s.tiles):
        tile = s.tiles[key]
        if tile.owner == former and hx.distance(hx.from_key(key), city.hex) <= 2:
            s.tiles[key] = tile.model_copy(update={"owner": player_id})
    out.append(
        ev.event(
            s.turn,
            ev.CITY_CAPTURED,
            f"{s.players[player_id].civ_name} captured {city.name} from "
            f"{s.players[former].civ_name}",
            actor=player_id,
            city_id=city.id,
            from_player=former,
        )
    )


def _found_city(s: State, player_id: str, order, out: list[Event]) -> None:  # noqa: ANN001
    unit = _own_unit(s, player_id, order.unit_id)
    if unit is None or unit.type is not UnitType.SETTLER:
        _reject(s, out, player_id, "found_city", "unit is not a settler you control")
        return
    if unit.embarked:
        _reject(s, out, player_id, "found_city", "a settler at sea must land first")
        return
    tile = s.at(unit.hex)
    if tile is None or not TERRAIN[tile.terrain].settleable:
        _reject(s, out, player_id, "found_city", "terrain cannot support a city")
        return
    if tile.owner is not None and tile.owner != player_id:
        _reject(s, out, player_id, "found_city", "tile belongs to another civ")
        return
    for _, other in sorted(s.cities.items()):
        if hx.distance(other.hex, unit.hex) < MIN_CITY_SPACING:
            _reject(
                s,
                out,
                player_id,
                "found_city",
                f"too close to {other.name}; minimum spacing is {MIN_CITY_SPACING}",
            )
            return

    cid, s.next_id = s.new_id("c")
    name = order.name or CITY_NAMES[len(s.cities) % len(CITY_NAMES)]
    s.cities[cid] = City(id=cid, owner=player_id, name=name, pos=unit.pos)
    for h in hx.within(unit.hex, 2):
        key = h.to_key()
        existing = s.tiles.get(key)
        if existing is not None and existing.owner is None:
            s.tiles[key] = existing.model_copy(update={"owner": player_id})
    del s.units[unit.id]
    out.append(
        ev.event(
            s.turn,
            ev.CITY_FOUNDED,
            f"{name} founded at {unit.pos}",
            actor=player_id,
            city_id=cid,
            pos=unit.pos,
            name=name,
        )
    )


def _build_improvement(s: State, player_id: str, order, out: list[Event]) -> None:  # noqa: ANN001
    unit = _own_unit(s, player_id, order.unit_id)
    if unit is None or unit.type is not UnitType.WORKER:
        _reject(s, out, player_id, "build_improvement", "unit is not a worker you control")
        return
    if unit.embarked:
        _reject(s, out, player_id, "build_improvement", "a worker at sea must land first")
        return
    improvement = Improvement(order.improvement)
    if improvement not in WORKER_IMPROVEMENTS:
        _reject(
            s, out, player_id, "build_improvement", f"{improvement.value} is not built by workers"
        )
        return
    tile = s.at(unit.hex)
    if tile is None or tile.terrain not in IMPROVEMENTS[improvement].terrains:
        _reject(
            s,
            out,
            player_id,
            "build_improvement",
            f"{improvement.value} cannot be built on this terrain",
        )
        return
    if tile.improvement == improvement:
        _reject(s, out, player_id, "build_improvement", "already built here")
        return
    unit.working_on = improvement
    unit.work_turns_left = IMPROVEMENTS[improvement].turns
    unit.fortified = False


def legal_actions(state: State, player_id: str) -> dict:
    """What this civ may legally do right now.

    Handed to the agent verbatim in its observation. This is the single biggest
    lever on output validity: a model that is told exactly which hexes a unit
    can reach rarely invents one, so the repair loop almost never fires.
    """
    player = state.players[player_id]
    units: dict[str, dict] = {}
    for unit in state.units_of(player_id):
        spec = UNITS[unit.type]
        moves: list[str] = []
        embarks: list[str] = []
        disembarks: list[str] = []
        attacks: list[str] = []
        for n in hx.neighbors(unit.hex):
            tile = state.at(n)
            if tile is None:
                continue
            hostile_units = [u for u in state.units_at(n) if u.owner != player_id]
            city = state.city_at(n)
            hostile_city = city is not None and city.owner != player_id
            if hostile_units or hostile_city:
                owner = hostile_units[0].owner if hostile_units else city.owner  # type: ignore[union-attr]
                if (
                    movement.can_attack(state, unit)
                    and state.at_war(player_id, owner)
                    and unit.moves_left > 0
                ):
                    attacks.append(n.to_key())
                continue
            if unit.moves_left <= 0:
                continue
            # One source of truth with the reducer: anything offered here must
            # be accepted there, which is the whole reason the agent is given a
            # legal-action list at all.
            allowed, _ = movement.entry_check(state, unit, n)
            if not allowed:
                continue
            change = movement.transition(state, unit, n)
            if change == "embark":
                embarks.append(n.to_key())
            elif change == "disembark":
                disembarks.append(n.to_key())
            else:
                moves.append(n.to_key())

        tile = state.at(unit.hex)
        ashore = movement.can_act_on_land(unit)
        can_found = (
            unit.type is UnitType.SETTLER
            and ashore
            and tile is not None
            and TERRAIN[tile.terrain].settleable
            and all(
                hx.distance(c.hex, unit.hex) >= MIN_CITY_SPACING
                for _, c in sorted(state.cities.items())
            )
        )
        improvements = (
            sorted(
                i.value
                for i in WORKER_IMPROVEMENTS
                if tile is not None and tile.terrain in IMPROVEMENTS[i].terrains
            )
            if unit.type is UnitType.WORKER and ashore
            else []
        )
        units[unit.id] = {
            "move": sorted(moves),
            # Kept separate from `move` so an agent can see that stepping here
            # is a commitment - it ends the turn and leaves the unit exposed -
            # rather than an ordinary step it might take by accident.
            "embark": sorted(embarks),
            "disembark": sorted(disembarks),
            "attack": sorted(attacks),
            "fortify": not spec.civilian and not unit.embarked,
            "found_city": can_found,
            "build_improvement": improvements,
            "embarked": unit.embarked,
        }

    others = [p for p in state.player_ids() if p != player_id and state.players[p].alive]
    return {
        "units": units,
        "cities": {
            c.id: {"build": economy.buildable(state, c)} for c in state.cities_of(player_id)
        },
        "research": available_techs(frozenset(player.known_techs)),
        "diplomacy": {
            "can_message": others,
            "can_propose_to": others,
            "can_declare_war_on": [p for p in others if not state.at_war(player_id, p)],
            "respondable_proposals": [p.id for p in diplomacy.open_proposals_for(state, player_id)],
        },
    }
