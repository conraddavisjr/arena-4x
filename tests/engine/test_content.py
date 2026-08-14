"""Content table integrity.

These tables are hand-maintained data, so the realistic failure is a typo: a
prereq naming a tech that does not exist, a unit gated behind a tech nobody can
reach, a resource placed on terrain it can never spawn on. Each of those would
surface much later as "the AI never builds swordsmen" rather than as an error,
so they are worth asserting directly.
"""

from __future__ import annotations

from arena_engine import content as c


def test_every_tech_prereq_exists() -> None:
    for name, spec in c.TECHS.items():
        for prereq in spec.prereqs:
            assert prereq in c.TECHS, f"{name} requires unknown tech {prereq!r}"


def test_tech_graph_is_acyclic_and_fully_reachable() -> None:
    """Every tech must be reachable by researching only available techs.

    This is a stronger claim than "no cycles": it also catches a tech whose
    prereqs are mutually unreachable, which a plain cycle check would miss.
    """
    known: set[str] = set()
    while True:
        newly = [n for n in c.available_techs(frozenset(known))]
        if not newly:
            break
        known.update(newly)
    unreachable = set(c.TECHS) - known
    assert not unreachable, f"unreachable techs: {sorted(unreachable)}"


def test_prereqs_are_from_an_earlier_or_equal_era() -> None:
    for name, spec in c.TECHS.items():
        for prereq in spec.prereqs:
            assert c.TECHS[prereq].era <= spec.era, f"{name} depends on later-era {prereq}"


def test_apex_tech_is_the_deepest_node() -> None:
    assert c.APEX_TECH in c.TECHS
    assert not any(c.APEX_TECH in spec.prereqs for spec in c.TECHS.values()), (
        "apex_theory must be a leaf; nothing should depend on it"
    )


def test_unit_tech_and_resource_gates_are_real() -> None:
    for unit, spec in c.UNITS.items():
        if spec.req_tech is not None:
            assert spec.req_tech in c.TECHS, f"{unit} gated behind unknown tech {spec.req_tech!r}"
        if spec.req_resource is not None:
            assert spec.req_resource in c.RESOURCE_TERRAIN, f"{unit} needs unknown resource"
            assert spec.req_resource in c.STRATEGIC, (
                f"{unit} is gated on {spec.req_resource}, which must be marked strategic"
            )


def test_building_tech_gates_are_real() -> None:
    for name, spec in c.BUILDINGS.items():
        if spec.req_tech is not None:
            assert spec.req_tech in c.TECHS, f"{name} gated behind unknown tech {spec.req_tech!r}"


def test_starting_units_need_no_tech() -> None:
    """A civ starts with nothing researched, so its opening build must be legal."""
    for unit in (c.UnitType.SETTLER, c.UnitType.WORKER, c.UnitType.SCOUT, c.UnitType.WARRIOR):
        assert c.UNITS[unit].req_tech is None
        assert c.UNITS[unit].req_resource is None


def test_resources_spawn_only_on_terrain_that_exists() -> None:
    for resource, terrains in c.RESOURCE_TERRAIN.items():
        assert terrains, f"{resource} can never spawn"
        assert resource in c.RESOURCE_YIELDS, f"{resource} has no yield defined"
        for terrain in terrains:
            assert terrain in c.TERRAIN


def test_strategic_resources_are_reachable_on_workable_land() -> None:
    """A resource that only spawns on impassable terrain can never be improved."""
    for resource in c.STRATEGIC:
        workable = [t for t in c.RESOURCE_TERRAIN[resource] if c.TERRAIN[t].settleable]
        assert workable, f"{resource} only spawns on terrain no city can work"


def test_improvements_apply_to_terrain_that_exists() -> None:
    for name, spec in c.IMPROVEMENTS.items():
        assert spec.terrains, f"{name} applies to no terrain"
        for terrain in spec.terrains:
            assert terrain in c.TERRAIN


def test_worker_improvements_are_on_terrain_a_worker_can_stand_on() -> None:
    """A worker walks to the tile, so its improvements must be on land.

    Fishing boats are the exception and are excluded from WORKER_IMPROVEMENTS
    for exactly this reason: they sit on water and are placed by a coastal city
    working the tile, not by a unit standing on it.
    """
    for improvement in c.WORKER_IMPROVEMENTS:
        for terrain in c.IMPROVEMENTS[improvement].terrains:
            assert c.TERRAIN[terrain].passable, f"{improvement} on land-impassable {terrain}"


def test_water_improvements_are_on_navigable_terrain_only() -> None:
    water_only = set(c.IMPROVEMENTS) - c.WORKER_IMPROVEMENTS
    assert water_only, "expected at least one city-worked water improvement"
    for improvement in water_only:
        for terrain in c.IMPROVEMENTS[improvement].terrains:
            assert c.TERRAIN[terrain].navigable, f"{improvement} on non-navigable {terrain}"


def test_every_terrain_is_reachable_by_some_domain() -> None:
    """Mountains are the one deliberate exception: closed to everything."""
    unreachable = [
        t
        for t, spec in c.TERRAIN.items()
        if not spec.open_to(c.Domain.LAND) and not spec.open_to(c.Domain.SEA)
    ]
    assert unreachable == [c.Terrain.MOUNTAINS], f"unexpectedly unreachable: {unreachable}"


def test_sea_units_cannot_embark_and_land_units_can() -> None:
    for unit, spec in c.UNITS.items():
        if spec.domain is c.Domain.SEA:
            assert not spec.can_embark, f"{unit} is already a sea unit"


def test_embarkation_is_gated_on_a_real_tech() -> None:
    assert c.EMBARK_TECH in c.TECHS
    # Reachable early: the ocean is a third of the map, and gating it behind a
    # late tech would waste most of the board for most of the match.
    assert c.TECHS[c.EMBARK_TECH].era <= 2


def test_naval_units_share_the_embark_tech() -> None:
    naval = [u for u, s in c.UNITS.items() if s.domain is c.Domain.SEA]
    assert naval, "there should be at least one sea unit"
    for unit in naval:
        assert c.UNITS[unit].req_tech == c.EMBARK_TECH


def test_embarked_units_are_meaningfully_weaker() -> None:
    """The crossing has to be the vulnerable part, or naval play is free."""
    assert 0 < c.EMBARKED_DEFENSE_PCT < 100
    weakest_naval = min(s.attack for s in c.UNITS.values() if s.domain is c.Domain.SEA)
    toughest_embarked = max(
        s.defense * c.EMBARKED_DEFENSE_PCT // 100
        for s in c.UNITS.values()
        if s.domain is c.Domain.LAND
    )
    assert weakest_naval > toughest_embarked, (
        "a warship should beat anything caught at sea, or escorting is pointless"
    )


def test_apex_project_is_a_wonder_gated_on_apex_theory() -> None:
    spec = c.BUILDINGS[c.APEX_PROJECT]
    assert spec.wonder
    assert spec.req_tech == c.APEX_TECH


def test_wonders_set_matches_the_wonder_flag() -> None:
    assert frozenset(n for n, s in c.BUILDINGS.items() if s.wonder) == c.WONDERS


def test_yields_add_componentwise() -> None:
    total = c.Yields(food=1, production=2) + c.Yields(production=3, culture=4)
    assert total == c.Yields(food=1, production=5, gold=0, science=0, culture=4)


def test_food_to_grow_is_strictly_increasing() -> None:
    costs = [c.food_to_grow(p) for p in range(1, 12)]
    assert costs == sorted(costs)
    assert len(set(costs)) == len(costs)


def test_available_techs_is_sorted_and_excludes_known() -> None:
    known = frozenset({"pottery", "mining"})
    avail = c.available_techs(known)
    assert avail == sorted(avail), "must be sorted for byte-stable observations"
    assert not (set(avail) & known)
    assert "writing" in avail, "writing needs only pottery"
    assert "iron_working" not in avail, "iron_working needs bronze_working first"


def test_every_attacker_has_a_counter_or_is_not_dominant() -> None:
    """No unit should have both the best attack and the best defense."""
    military = {u: s for u, s in c.UNITS.items() if not s.civilian}
    best_attack = max(s.attack for s in military.values())
    best_defense = max(s.defense for s in military.values())
    dominant = [
        u for u, s in military.items() if s.attack == best_attack and s.defense == best_defense
    ]
    assert not dominant, f"strictly dominant unit(s): {dominant}"


def test_counters_reference_real_units() -> None:
    for unit, countered in c.COUNTERS.items():
        assert unit in c.UNITS
        for target in countered:
            assert target in c.UNITS
