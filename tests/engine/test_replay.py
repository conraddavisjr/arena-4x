"""The replay bundle.

The bundle is the published artifact. Two properties matter beyond it merely
being written: it has to be **self-contained** (a viewer with no network beyond
the bundle can reconstruct every turn) and it has to be **inert** (nothing in it
lets a reader influence, replay differently, or reach back into the system).

The delta encoding is where self-containment could quietly break: borders and
improvements are only written when they change, so a viewer reconstructs them by
folding forward. If a delta were ever dropped the board would render subtly
wrong many turns later, with nothing failing at the time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from arena_replay import BundleWriter

from arena_engine import bots
from arena_engine.reducer import new_match, step
from arena_engine.types import MatchConfig, State

ROSTER = [
    ("p1", "Aurelian Compact"),
    ("p2", "Iron Concord"),
    ("p3", "Verdant Pact"),
    ("p4", "Solari Dominion"),
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, State]:
    root = tmp_path_factory.mktemp("bundle")
    state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=45))
    writer = BundleWriter.start(root, state)
    while state.victory is None and state.turn < 45:
        state, events = step(state, bots.all_bot_actions(state))
        writer.add(state, events)
    writer.finish(state, {"winner": state.victory.winner if state.victory else None})
    return root, state


def read(root: Path, name: str) -> dict:
    return json.loads((root / name).read_text())


def frame(root: Path, turn: int) -> dict:
    return read(root, f"turns/{turn:04d}.json")


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_bundle_has_the_expected_files(bundle) -> None:
    root, state = bundle
    assert (root / "match.json").exists()
    assert (root / "stats.json").exists()
    meta = read(root, "match.json")
    assert len(list((root / "turns").glob("*.json"))) == meta["turns"]


def test_terrain_is_written_once_not_per_turn(bundle) -> None:
    """Repeating a thousand tiles every turn would dominate the bundle."""
    root, _ = bundle
    meta = read(root, "match.json")
    assert len(meta["terrain"]) == len(meta["tiles"])
    assert "terrain" not in frame(root, 5)


def test_positions_are_tile_indices_not_coordinate_strings(bundle) -> None:
    root, _ = bundle
    meta, f = read(root, "match.json"), frame(root, 10)
    limit = len(meta["tiles"])
    for unit in f["units"]:
        assert isinstance(unit["at"], int) and 0 <= unit["at"] < limit
    for tiles in f["visibility"].values():
        assert all(isinstance(i, int) and 0 <= i < limit for i in tiles)


def test_a_first_paint_stays_small(bundle) -> None:
    """Metadata plus one turn is what a visitor waits for before seeing anything."""
    root, _ = bundle
    biggest = max((root / "turns").glob("*.json"), key=lambda f: f.stat().st_size)
    first_paint = (root / "match.json").stat().st_size + biggest.stat().st_size
    assert first_paint < 250_000, f"first paint is {first_paint / 1024:.0f} KB"


# ---------------------------------------------------------------------------
# Self-containment: the delta encoding must fold back to the truth
# ---------------------------------------------------------------------------


def test_folding_border_deltas_reproduces_the_real_map(bundle) -> None:
    """The property the viewer depends on and nothing else would catch.

    Borders are written only when they change. A dropped delta would render the
    board subtly wrong many turns later, with nothing failing at the time.
    """
    root, state = bundle
    meta = read(root, "match.json")
    owners: list[str | None] = [None] * len(meta["tiles"])
    turns = sorted(int(f.stem) for f in (root / "turns").glob("*.json"))

    for turn in turns:
        for index, who in frame(root, turn)["owners"].items():
            owners[int(index)] = who

    truth = [state.tiles[key].owner for key in meta["tiles"]]
    assert owners == truth


def test_folding_improvement_deltas_reproduces_the_real_map(bundle) -> None:
    root, state = bundle
    meta = read(root, "match.json")
    built: dict[int, str] = {}
    for turn in sorted(int(f.stem) for f in (root / "turns").glob("*.json")):
        built.update({int(k): v for k, v in frame(root, turn)["improvements"].items()})

    truth = {
        i: state.tiles[key].improvement.value
        for i, key in enumerate(meta["tiles"])
        if state.tiles[key].improvement is not None
    }
    assert built == truth


def test_every_frame_carries_what_the_panels_need(bundle) -> None:
    root, _ = bundle
    for turn in (1, 12, 30):
        f = frame(root, turn)
        for key in ("units", "cities", "economy", "visibility", "contact", "events"):
            assert key in f, f"turn {turn} is missing {key}"
        for civ in read(root, "match.json")["civs"]:
            econ = f["economy"][civ["player_id"]]
            # The panel the spectator watches to catch a civ going bankrupt.
            for field in ("gold", "gold_per_turn", "food_stored", "food_surplus", "score"):
                assert field in econ


def test_economy_matches_the_engine(bundle) -> None:
    """The numbers on screen have to be the numbers the engine played by."""
    from arena_engine import economy, victory

    root, state = bundle
    last = frame(root, state.turn)
    for player_id in state.civ_ids():
        econ = last["economy"][player_id]
        gold, science, _ = economy.player_output(state, player_id)
        assert econ["gold"] == state.players[player_id].gold
        assert econ["gold_per_turn"] == gold
        assert econ["science_per_turn"] == science
        assert econ["score"] == victory.score(state, player_id)


# ---------------------------------------------------------------------------
# Inertness: a published match must not be a control surface
# ---------------------------------------------------------------------------


def test_the_bundle_carries_no_prompts_or_raw_model_output(bundle) -> None:
    """~20MB of the least interesting data, and the one place a system prompt
    could leak into a published page."""
    root, _ = bundle
    # Every file, not just *.json. This globbed `*.json` and the orchestrator
    # writes its transcripts to `transcripts.jsonl` - so the one test standing
    # between a system prompt and a published web page did not look at the file
    # the system prompt was in.
    for path in (p for p in root.rglob("*") if p.is_file()):
        text = path.read_text().lower()
        for leak in ("system prompt", "you are the sovereign", "api_key", "anthropic"):
            assert leak not in text, f"{path.name} contains {leak!r}"


def test_the_bundle_has_no_endpoints_or_credentials(bundle) -> None:
    root, _ = bundle
    for path in (p for p in root.rglob("*") if p.is_file()):
        text = path.read_text()
        assert "http://" not in text and "https://" not in text
        assert "sk-" not in text


def test_frames_are_byte_reproducible(tmp_path: Path) -> None:
    """Same match, same bytes - so a published replay can be content-addressed."""

    def write(root: Path) -> str:
        state, _ = new_match("t", 9, ROSTER, MatchConfig(turn_limit=20))
        writer = BundleWriter.start(root, state)
        for _ in range(20):
            state, events = step(state, bots.all_bot_actions(state))
            writer.add(state, events)
            if state.victory:
                break
        writer.finish(state)
        return "".join(sorted(p.read_text() for p in root.rglob("*.json")))

    assert write(tmp_path / "a") == write(tmp_path / "b")


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


def test_reasoning_is_captured_for_every_living_civ(bundle) -> None:
    root, _ = bundle
    f = frame(root, 10)
    assert set(f["reasoning"]) == {c["player_id"] for c in read(root, "match.json")["civs"]}
    assert all(r["plan"] for r in f["reasoning"].values())


def test_wildlife_appears_on_the_board_but_not_as_a_civ(bundle) -> None:
    from arena_engine.content import BARBARIAN_ID

    root, _ = bundle
    meta = read(root, "match.json")
    assert BARBARIAN_ID not in {c["player_id"] for c in meta["civs"]}

    seen = any(
        u["owner"] == BARBARIAN_ID
        for turn in range(1, meta["turns"] + 1)
        for u in frame(root, turn)["units"]
    )
    assert seen, "the wilderness should be visible on the board"


def test_noise_is_filtered_from_the_event_ticker(bundle) -> None:
    """Movement is already visible on the board; rejections are debug detail."""
    root, _ = bundle
    for turn in (5, 20, 35):
        kinds = {e["type"] for e in frame(root, turn)["events"]}
        assert not (kinds & {"unit_moved", "order_rejected", "turn_started", "turn_ended"})


def test_civ_colours_are_stable_and_distinct(bundle) -> None:
    root, _ = bundle
    colours = [c["colour"] for c in read(root, "match.json")["civs"]]
    assert sorted(colours) == list(range(len(colours)))


def test_folding_dossier_deltas_reproduces_what_each_agent_wrote(bundle) -> None:
    """The dossier is written as a delta for the same reason borders are, and it
    has the same failure mode: a dropped one renders wrong many turns later with
    nothing failing at the time.

    It is also the artifact the lab exists to read. An agent's dossier is the
    only thing it carries between turns that it authored itself, so a bundle
    that reconstructs it incorrectly misrepresents the one record of what the
    model believed.
    """
    root, state = bundle
    folded: dict[str, dict] = {}
    for turn in sorted(int(f.stem) for f in (root / "turns").glob("*.json")):
        folded.update(frame(root, turn).get("dossiers") or {})

    truth = {pid: state.players[pid].dossier.model_dump(mode="json") for pid in state.civ_ids()}
    assert folded == truth


def test_a_whole_negotiation_reaches_the_frame_not_just_chat() -> None:
    """A proposal's covering note and its reply are things a civ *said*.

    They were not in the bundle, because `_messages` gathered `message_sent`
    alone - and models put almost all of their diplomacy in a proposal's
    `message` rather than beside it in a separate send. The result was a match
    whose civs signed non-aggression pacts every few turns rendering in the
    viewer as "Nobody has spoken", which is not a missing feature but a
    positive falsehood about what happened.
    """
    from arena_replay.bundle import _messages

    from arena_engine.actions import Action, Propose, RespondToProposal, pass_turn
    from arena_engine.types import ProposalType, Terms

    state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=45))
    actions = {p: pass_turn() for p, _ in ROSTER}
    actions["p1"] = Action(
        diplomacy=[
            Propose(
                action="propose",
                to="p2",
                type=ProposalType.NON_AGGRESSION,
                terms=Terms(duration_turns=10),
                message="Greetings gemini-3.6-flash. Let us keep the frontier quiet.",
            )
        ]
    )
    state, events = step(state, actions)
    said = _messages(state, events)
    assert [m["kind"] for m in said] == ["proposal"]
    assert said[0]["text"].startswith("Greetings")
    assert said[0]["to"] == "p2", "a proposal is addressed, so it threads like a DM"
    assert said[0]["channel"] == "private"

    pid = next(iter(state.proposals))
    actions = {p: pass_turn() for p, _ in ROSTER}
    actions["p2"] = Action(
        diplomacy=[
            RespondToProposal(
                action="respond_to_proposal",
                proposal_id=pid,
                response="accept",
                message="Accepted. Let us both expand westward in peace.",
            )
        ]
    )
    state, events = step(state, actions)
    reply = [m for m in _messages(state, events) if m["kind"] == "reply"]
    assert len(reply) == 1, "the answer is the other half of the exchange"
    assert reply[0]["response"] == "accept", "the same words follow a yes and a no"
    assert reply[0]["to"] == "p1"
    assert any(e.type == "treaty_signed" for e in events), "the outcome still fires separately"


def test_a_wordless_proposal_and_reply_still_appear() -> None:
    """The act is the content. Prose is optional and silence is a finding.

    `respond_to_proposal.message` is optional and models use that: one signed a
    ten-turn non-aggression pact with `message: null`. An earlier version of
    this suppressed wordless entries on the reasoning that an empty chat bubble
    is worse than none - true for a message, wrong for a treaty. The pact
    appeared in the relations bar with nothing in the thread to account for it,
    and reading the console you would conclude the engine had signed something
    nobody agreed to.

    A `send_message` with no words is still dropped, because there the words
    *are* the act.
    """
    from arena_replay.bundle import _messages

    from arena_engine.actions import Action, Propose, RespondToProposal, pass_turn
    from arena_engine.types import ProposalType, Terms

    state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=45))
    actions = {p: pass_turn() for p, _ in ROSTER}
    actions["p1"] = Action(
        diplomacy=[Propose(action="propose", to="p2", type=ProposalType.PEACE, terms=Terms())]
    )
    state, events = step(state, actions)
    said = _messages(state, events)
    assert [m["kind"] for m in said] == ["proposal"]
    assert said[0]["text"] == "", "no note written, and none invented"

    pid = next(iter(state.proposals))
    actions = {p: pass_turn() for p, _ in ROSTER}
    actions["p2"] = Action(
        diplomacy=[
            RespondToProposal(action="respond_to_proposal", proposal_id=pid, response="accept")
        ]
    )
    state, events = step(state, actions)
    reply = [m for m in _messages(state, events) if m["kind"] == "reply"]
    assert len(reply) == 1, "a treaty signed in silence is still a treaty signed"
    assert reply[0]["response"] == "accept"
    assert reply[0]["text"] == ""
    assert any(e.type == "treaty_signed" for e in events)


def test_an_unchanged_dossier_is_not_repeated(bundle) -> None:
    """Four dossiers at ~2000 tokens each, repeated on 300 frames, would dominate
    the bundle - and most turns an agent changes one line or none."""
    root, _ = bundle
    turns = sorted(int(f.stem) for f in (root / "turns").glob("*.json"))
    # Heuristic bots never write one at all, so after the first frame there is
    # nothing to repeat. A frame that re-sent an identical dossier would fail.
    seen: dict[str, dict] = {}
    for turn in turns:
        for pid, doc in (frame(root, turn).get("dossiers") or {}).items():
            assert seen.get(pid) != doc, f"turn {turn} repeated {pid}'s unchanged dossier"
            seen[pid] = doc


# ---------------------------------------------------------------------------
# The match library
# ---------------------------------------------------------------------------


def _library():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import build_library

    return build_library


def test_the_library_indexes_finished_matches(tmp_path: Path) -> None:
    """One entry per bundle, derived from the bundle and nothing else.

    Derived rather than registered, because a registry drifts: move a directory
    and it lists matches that are not there while omitting ones that are. This
    can be deleted and rebuilt at any time, which is the property that makes it
    safe to regenerate on every `make library`.
    """
    lib = _library()
    for name, turns in (("alpha", 3), ("beta", 5)):
        state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=turns))
        writer = BundleWriter.start(
            tmp_path / name / "bundle", state, {p: f"model-{p}" for p, _ in ROSTER}
        )
        for _ in range(turns):
            state, events = step(state, bots.all_bot_actions(state))
            writer.add(state, events)
        writer.finish(state, {}, finished_at="2026-08-17T09:00:00+00:00", spent_usd=1.5)

    entries = [lib.entry(tmp_path, tmp_path / n / "bundle") for n in ("alpha", "beta")]
    assert [e["id"] for e in entries] == ["alpha", "beta"]
    for e in entries:
        # Relative, because the viewer loads it as `?match=<path>` from the page
        # it was served on; an absolute path works locally and nowhere else.
        assert not Path(e["path"]).is_absolute()
        assert e["path"].endswith("bundle")
        assert e["finished_at"] == "2026-08-17T09:00:00+00:00"
        assert e["spent_usd"] == 1.5
        assert e["models"] == [f"model-{p}" for p, _ in ROSTER]


def test_an_undated_match_is_marked_as_inferred(tmp_path: Path) -> None:
    """A rebuilt bundle must not claim to have been played today.

    The two matches that predate timestamps have no `finished_at`, so the date
    falls back to the file's mtime - which is when the bundle was last *written*,
    not when it was played. The trailing marker is what lets the viewer say "from
    file" instead of presenting a guess as a fact.
    """
    lib = _library()
    state, _ = new_match("t", 4, ROSTER, MatchConfig(turn_limit=2))
    writer = BundleWriter.start(tmp_path / "old" / "bundle", state)
    for _ in range(2):
        state, events = step(state, bots.all_bot_actions(state))
        writer.add(state, events)
    writer.finish(state, {})

    entry = lib.entry(tmp_path, tmp_path / "old" / "bundle")
    assert entry["finished_at"].endswith("~"), "an inferred date must be marked as one"

    # And a recorded one is never marked, or the marker would mean nothing.
    writer.finish(state, {}, finished_at="2026-08-17T09:00:00+00:00")
    assert not lib.entry(tmp_path, tmp_path / "old" / "bundle")["finished_at"].endswith("~")


def test_a_directory_with_no_bundle_is_skipped_not_an_error(tmp_path: Path) -> None:
    """Runs get killed. A half-written directory should not break the shelf."""
    lib = _library()
    (tmp_path / "crashed" / "bundle").mkdir(parents=True)
    assert lib.entry(tmp_path, tmp_path / "crashed" / "bundle") is None
