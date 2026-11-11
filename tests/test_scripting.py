"""descape/scripting.py: the script transaction, the `editor` facade and the
run-end checks.

The claims under test, in the order the risk runs:

1. **One undo entry or none.** A finished run is one composite record that
   undoes and redoes tiles and units together; a run that changed nothing
   pushes nothing.
2. **Full rollback.** A raise, a SyntaxError and a cancel (caught or not)
   leave every tile and all twelve saved unit fields as the run found them.
3. **Only `editor` changes a unit.** Each way around the facade fails the
   run, and the rollback still completes with the model aligned.
4. **The facade stays inside the write rules**: rotation, variant, placement
   rotation, GAIA-only owners, garrison type and capacity, and terrain a save
   could not write or the game could not load.
"""

from __future__ import annotations

import contextlib
import copy
import math
import textwrap
from pathlib import Path

import pytest

from descape import batch_api, garrison, place_rotation, render, scripting, settings, unit_rotation
from descape.edit_history import EditHistory, tile_state
from descape.scripting import Cancelled, ScriptEntry, ScriptError, ScriptRun
from descape.unit_model import UnitEditModel

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"

_TREE_OAK = 100
_WALL = 102
_HOUSE = 200
_ARCHER_P1 = 201
_VILLAGER_IN_HOUSE = 203
_ARCHER_P2 = 300
_WATCH_TOWER_CONST = 79
_ARCHER_CONST = 4
_OAK_CONST = 349
_RAM_CONST = 35  # holds land units, itself included: the one way to reach the cycle check

WITH_MODEL = pytest.mark.parametrize("with_model", [False, True], ids=["no_model", "model"])


class Doc:
    def __init__(self, with_model: bool = False):
        self.loaded = batch_api.load(FIXTURE_PATH)
        self.history = EditHistory()
        self.model = UnitEditModel(self.loaded) if with_model else None
        self.run: ScriptRun | None = None

    @property
    def tiles(self):
        return self.loaded.map_manager.terrain

    def unit(self, reference_id: int):
        return next(u for u in self.loaded.unit_manager.get_all_units() if u.reference_id == reference_id)

    def state(self):
        """Every tile, each player's units by identity, and all twelve saved fields."""
        lists = [list(units) for units in self.loaded.unit_manager.units]
        values = [[scripting.saved_values(u) for u in units] for units in lists]
        return [tile_state(t) for t in self.tiles], lists, values

    def assert_state(self, before) -> None:
        tiles, lists, values = before
        assert [tile_state(t) for t in self.tiles] == tiles
        now_lists = [list(units) for units in self.loaded.unit_manager.units]
        assert [[id(u) for u in units] for units in now_lists] == [[id(u) for u in units] for units in lists]
        assert [[scripting.saved_values(u) for u in units] for units in lists] == values

    def new_run(self, **kwargs) -> ScriptRun:
        self.run = ScriptRun(self.loaded, self.history, model=self.model, **kwargs)
        return self.run

    def call(self, function, **kwargs):
        return scripting.call(function, self.new_run(**kwargs), title="t", label='Script "t"')

    def source(self, text: str, **kwargs):
        entry = ScriptEntry("t", "t", Path("t.py"))
        return scripting.run_script(entry, textwrap.dedent(text).encode(), self.new_run(**kwargs))

    def assert_rolled_back(self, outcome, before, records=()) -> None:
        assert not outcome.ok or outcome.cancelled
        assert not outcome.incomplete, outcome.detail
        assert outcome.record is None and outcome.changed == ()
        self.assert_state(before)
        assert self.history.records == list(records) and not self.history.in_stroke
        model = self.run.model
        if model is not None:
            model._check_alignment()
            assert not model.has_edits
            # The edit is closed: the next one can open.
            model.begin_unit_edit([0])
            model.abort_unit_edit()

    def move_history(self, direction: str) -> None:
        getattr(self.history, direction)(self.tiles, None, units=self.run.model)


def _ref(editor, reference_id: int):
    return next(u for u in editor.units() if u.reference_id == reference_id)


# -- one undo entry, or none ---------------------------------------------------


@WITH_MODEL
def test_a_run_is_one_composite_record_that_undoes_and_redoes_tiles_and_units(with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()

    def script(editor) -> None:
        for x in range(3):
            editor.set_terrain(x, 0, 6)
        editor.move_unit(_ref(editor, _ARCHER_P1), 30.5, 31.5)

    outcome = doc.call(script)
    assert outcome.ok and outcome.changed == ("terrain", "units")
    assert scripting.summarise(outcome)[0] == 'Script "t" changed terrain and units'
    assert doc.history.records == [outcome.record]
    assert outcome.record.kind == "composite" and outcome.record.label == 'Script "t"'
    assert not doc.history.in_stroke and doc.history.is_dirty
    after = doc.state()
    archer = doc.unit(_ARCHER_P1)
    assert (archer.x, archer.y) == (30.5, 31.5) and doc.tiles[1].terrain_id == 6

    doc.move_history("undo")
    doc.assert_state(before)
    doc.move_history("redo")
    doc.assert_state(after)
    doc.run.model._check_alignment()


@WITH_MODEL
def test_a_run_that_changes_nothing_pushes_nothing(with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()

    def script(editor) -> None:
        archer = _ref(editor, _ARCHER_P1)
        editor.move_unit(archer, archer.x, archer.y)
        editor.set_status(archer, archer.status)
        editor.rotate(archer, archer.rotation)
        editor.reassign(archer, 1)
        tile = editor.tile(0, 0)
        editor.set_terrain(0, 0, tile.terrain_id)
        # Added then removed: the lists end as they began.
        editor.remove(editor.add_unit(_ARCHER_CONST, 40.5, 40.5, 1))

    outcome = doc.call(script)
    assert outcome.ok and outcome.changed == () and outcome.record is None
    assert scripting.summarise(outcome)[0] == 'Script "t" ran, but changed nothing'
    assert doc.history.records == [] and not doc.history.is_dirty
    doc.assert_state(before)
    assert not doc.run.model.has_edits
    doc.run.model._check_alignment()


def test_a_terrain_only_run_never_builds_the_unit_model() -> None:
    doc = Doc()
    built = []
    outcome = doc.call(lambda editor: editor.set_terrain(0, 0, 6), model_getter=lambda: built.append(1))
    assert outcome.changed == ("terrain",) and not built and doc.run.model is None


# -- full rollback -------------------------------------------------------------


def _busy(editor) -> None:
    """Tiles, a move, an add, a remove and a reassign: something of every kind to put back."""
    editor.set_terrain(2, 2, 6)
    editor.set_elevation(60, 60, 3)
    editor.move_unit(_ref(editor, _HOUSE), 50.5, 50.5)
    editor.add_unit(_ARCHER_CONST, 40.5, 40.5, 3, facing=1.0)
    editor.remove(_ref(editor, _ARCHER_P2))
    editor.reassign(_ref(editor, _ARCHER_P1), 2)
    editor.set_status(_ref(editor, _TREE_OAK), 0)


def _raises(editor) -> None:
    _busy(editor)
    raise ValueError("boom")


def _exits(editor) -> None:
    _busy(editor)
    raise SystemExit(3)


def _cancel(done, total, note) -> None:
    raise Cancelled


def _cancelled(editor) -> None:
    _busy(editor)
    editor.progress(1, 2)


def _cancel_caught_as_exception(editor) -> None:
    _busy(editor)
    with contextlib.suppress(Exception):  # the point: this must not catch a cancel
        editor.progress(1, 2)
    editor.set_terrain(3, 3, 6)


def _cancel_swallowed(editor) -> None:
    _busy(editor)
    with contextlib.suppress(BaseException):  # a script that swallows everything is still cancelled
        editor.progress(1, 2)


@WITH_MODEL
@pytest.mark.parametrize(
    ("script", "cancelled"),
    [
        (_raises, False),
        (_exits, False),
        (_cancelled, True),
        (_cancel_caught_as_exception, True),
        (_cancel_swallowed, True),
    ],
)
def test_a_failed_or_cancelled_run_puts_everything_back(script, cancelled, with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()
    outcome = doc.call(script, progress=_cancel)
    assert outcome.cancelled is cancelled
    assert bool(outcome.error) is not cancelled
    assert set(outcome.rolled_back) == {"terrain", "units"} and outcome.touched
    doc.assert_rolled_back(outcome, before)
    assert "nothing changed" in scripting.summarise(outcome)[0]


@WITH_MODEL
def test_a_syntax_error_is_an_ordinary_failure_and_output_is_captured(with_model, capsys) -> None:
    doc = Doc(with_model)
    before = doc.state()
    outcome = doc.source("editor.set_terrain(0, 0, 6\n")
    assert outcome.error.startswith("SyntaxError") and "t.py" in outcome.detail
    doc.assert_rolled_back(outcome, before)

    outcome = doc.source(
        """
        import sys
        print("to stdout")
        print("to stderr", file=sys.stderr)
        if __name__ == "__main__":
            editor.set_terrain(0, 0, 6)
        """
    )
    assert outcome.ok and outcome.changed == ()
    assert outcome.output == "to stdout\nto stderr\n" and capsys.readouterr().out == ""


def test_keyboard_interrupt_rolls_back_then_propagates() -> None:
    doc = Doc(with_model=True)
    before = doc.state()

    def script(editor) -> None:
        _busy(editor)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        doc.call(script)
    doc.assert_state(before)
    assert doc.history.records == [] and not doc.history.in_stroke
    doc.model._check_alignment()


def test_a_run_refuses_to_begin_inside_an_open_edit() -> None:
    doc = Doc(with_model=True)
    doc.history.begin_stroke(doc.tiles)
    outcome = doc.call(lambda editor: editor.set_terrain(0, 0, 6))
    assert outcome.error == "an edit is still in progress" and doc.history.in_stroke
    doc.history.abort_stroke()

    doc.model.begin_unit_edit([1])
    outcome = doc.call(lambda editor: editor.set_terrain(0, 0, 6))
    assert outcome.error == "a unit edit is still in progress" and not doc.history.in_stroke
    assert doc.tiles[0].terrain_id != 6


def test_any_other_base_exception_closes_the_transaction_then_propagates() -> None:
    class Stop(BaseException):
        pass

    doc = Doc(with_model=True)
    before = doc.state()

    def script(editor) -> None:
        _busy(editor)
        raise Stop

    with pytest.raises(Stop):
        doc.call(script)
    doc.assert_state(before)
    assert doc.history.records == [] and not doc.history.in_stroke
    doc.model._check_alignment()
    doc.model.begin_unit_edit([0])  # the unit edit is closed too


@WITH_MODEL
def test_a_unit_moved_away_and_back_is_no_edit(with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()

    def script(editor) -> None:
        archer = _ref(editor, _ARCHER_P1)
        x, y = archer.x, archer.y
        editor.move_unit(archer, 30.5, 30.5)
        editor.move_unit(archer, x, y)

    outcome = doc.call(script)
    assert outcome.ok and outcome.changed == () and doc.history.records == []
    doc.assert_state(before)
    assert not doc.run.model.has_edits


def test_a_listener_raising_after_the_push_does_not_roll_the_run_back() -> None:
    doc = Doc()

    def listener() -> None:
        raise RuntimeError("repaint failed")

    doc.history.on_change = listener
    outcome = doc.call(lambda editor: editor.set_terrain(0, 0, 6))
    assert outcome.ok and outcome.changed == ("terrain",) and "repaint failed" in outcome.detail
    assert doc.history.records == [outcome.record] and doc.tiles[0].terrain_id == 6


@WITH_MODEL
def test_redo_reapplies_adds_removes_reassigns_and_garrison_links(with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()

    def script(editor) -> None:
        _busy(editor)
        ram, other = (editor.add_unit(_RAM_CONST, 70.5 + i, 70.5, 1) for i in range(2))
        editor.garrison(ram, other)
        with pytest.raises(editor.ScriptError, match="cycle"):
            editor.garrison(other, ram)

    outcome = doc.call(script)
    assert outcome.ok and outcome.changed == ("terrain", "units"), outcome.detail
    after = doc.state()
    for _ in range(2):
        doc.move_history("undo")
        doc.assert_state(before)
        doc.move_history("redo")
        doc.assert_state(after)
    doc.run.model._check_alignment()


def test_a_rollback_that_cannot_finish_is_reported_incomplete() -> None:
    doc = Doc()

    def script(editor) -> None:
        archer = _ref(editor, _ARCHER_P1)
        editor.set_terrain(0, 0, 6)
        del archer.x  # unreadable from here on: no check or restore can pass it

    outcome = doc.call(script)
    assert not outcome.ok and outcome.incomplete and outcome.record is None
    short, lines = scripting.summarise(outcome)
    assert "nothing changed" not in short and "close it without saving" in short
    assert any("could not be fully rolled back" in line for line in lines)
    assert doc.tiles[0].terrain_id != 6  # the tiles still went back


def _float_elevation(editor) -> None:
    tile = editor.tile(5, 5)
    tile.elevation = float(tile.elevation)


def _float_unit_const_then_facade_move(editor) -> None:
    archer = _ref(editor, _ARCHER_P2)
    archer.unit_const = float(archer.unit_const)
    editor.move_unit(archer, 30.5, 30.5)


def _float_status_untouched(editor) -> None:
    archer = _ref(editor, _ARCHER_P2)
    archer.status = float(archer.status)


@WITH_MODEL
@pytest.mark.parametrize("script", [_float_elevation, _float_unit_const_then_facade_move, _float_status_untouched])
def test_an_equal_value_of_another_type_fails_the_run_and_goes_back(script, with_model) -> None:
    """A float equal to the int it replaced is in no diff, yet the encoders cannot pack it."""
    doc = Doc(with_model)
    before = doc.state()
    types = [[type(v) for v in scripting.saved_values(u)] for u in doc.loaded.unit_manager.get_all_units()]
    outcome = doc.call(script)
    assert outcome.error and outcome.findings
    doc.assert_rolled_back(outcome, before)
    assert types == [[type(v) for v in scripting.saved_values(u)] for u in doc.loaded.unit_manager.get_all_units()]
    assert scripting.mistyped_tiles(doc.tiles) == []


# -- only `editor` changes a unit ----------------------------------------------


def _direct_rotation(editor) -> None:
    _ref(editor, _ARCHER_P2).rotation = 9


def _direct_rotation_then_facade_move(editor) -> None:
    # A facade write elsewhere first, so the pre-write check is what catches it.
    editor.set_status(_ref(editor, _TREE_OAK), 0)
    archer = _ref(editor, _ARCHER_P2)
    archer.rotation = 9
    editor.move_unit(archer, 30.5, 30.5)


def _swallowed_refusal(editor) -> None:
    editor.set_status(_ref(editor, _TREE_OAK), 0)
    archer = _ref(editor, _ARCHER_P2)
    archer.rotation = 9
    with contextlib.suppress(editor.ScriptError):
        editor.move_unit(archer, 30.5, 30.5)
    archer.rotation = 0.0  # even put back, the run has already failed


def _direct_id_and_caption_after_facade_move(editor) -> None:
    archer = _ref(editor, _ARCHER_P2)
    editor.move_unit(archer, 30.5, 30.5)
    archer.reference_id = 999
    archer.caption_string = "changed"


def _direct_append(editor) -> None:
    editor.loaded.unit_manager.units[1].append(copy.copy(_ref(editor, _ARCHER_P1)))
    # The first facade write must name the append, not trip the model's construction gate on it.
    editor.set_status(_ref(editor, _TREE_OAK), 0)


def _direct_del(editor) -> None:
    editor.set_status(_ref(editor, _TREE_OAK), 0)
    del editor.loaded.unit_manager.units[2][0]


def _direct_write_before_first_facade_call(editor) -> None:
    _ref(editor, _ARCHER_P2).x = 50.5
    editor.set_status(_ref(editor, _TREE_OAK), 0)


def _banned_player_setter(editor) -> None:
    editor.set_status(_ref(editor, _TREE_OAK), 0)
    _ref(editor, _ARCHER_P1).player = 2


@WITH_MODEL
@pytest.mark.parametrize(
    "script",
    [
        _direct_rotation,
        _direct_rotation_then_facade_move,
        _swallowed_refusal,
        _direct_id_and_caption_after_facade_move,
        _direct_append,
        _direct_del,
        _direct_write_before_first_facade_call,
        _banned_player_setter,
    ],
)
def test_a_unit_changed_around_editor_fails_the_run_and_rolls_back(script, with_model) -> None:
    doc = Doc(with_model)
    before = doc.state()
    outcome = doc.call(script)
    assert outcome.findings and "without editor" in outcome.error
    assert "units" in outcome.rolled_back
    doc.assert_rolled_back(outcome, before)


def test_unit_findings_names_the_unit_and_the_fields_and_caps_the_list() -> None:
    doc = Doc()
    image = scripting.capture_image(doc.loaded)
    membership = scripting.Membership(image.lists)
    manager = doc.loaded.unit_manager
    assert scripting.unit_findings(image, {}, membership, manager) == []

    archer = doc.unit(_ARCHER_P1)
    archer.rotation, archer.x = 9, 99.5
    (finding,) = scripting.unit_findings(image, {}, membership, manager)
    assert finding == "Archer (id 201) changed without editor: x 11.5 -> 99.5, rotation 0.0 -> 9"

    # What a facade write recorded is the expected state from then on.
    expected = {id(archer): (archer, scripting.saved_values(archer))}
    assert scripting.unit_findings(image, expected, membership, manager) == []

    for unit in manager.get_all_units():
        unit.status = 77
    findings = scripting.unit_findings(image, {}, membership, manager, limit=2)
    assert len(findings) == 3 and findings[-1] == "and 6 more units"


# -- the facade's unit writes --------------------------------------------------


def test_a_moved_or_reassigned_host_carries_its_garrison() -> None:
    doc = Doc()
    before = doc.state()

    def script(editor) -> None:
        house = _ref(editor, _HOUSE)
        editor.move_unit(house, 50.5, 51.5)
        editor.reassign(house, 2)

    outcome = doc.call(script)
    assert outcome.ok and outcome.changed == ("units",)
    house, villager = doc.unit(_HOUSE), doc.unit(_VILLAGER_IN_HOUSE)
    assert (villager.x, villager.y) == (house.x, house.y) == (50.5, 51.5)
    units = doc.loaded.unit_manager.units
    assert any(u is house for u in units[2]) and any(u is villager for u in units[2])
    doc.move_history("undo")
    doc.assert_state(before)


def test_garrison_co_locates_refuses_type_capacity_and_self_and_unloads_beside_the_host() -> None:
    doc = Doc()
    refused: list[str] = []
    made = {}

    def script(editor) -> None:
        tower = editor.add_unit(_WATCH_TOWER_CONST, 30.5, 30.5, 1)
        archers = [editor.add_unit(_ARCHER_CONST, 40.5 + i, 40.5, 1) for i in range(garrison.capacity(_WATCH_TOWER_CONST) + 1)]
        for archer in archers[:-1]:
            editor.garrison(archer, tower)
        for attempt in (
            lambda: editor.garrison(archers[-1], tower),  # full
            lambda: editor.garrison(tower, tower),  # itself
            lambda: editor.garrison(tower, archers[0]),  # an archer holds nothing
            lambda: editor.garrison(_ref(editor, _TREE_OAK), tower),  # a tree goes inside nothing
        ):
            with pytest.raises(editor.ScriptError) as caught:
                attempt()
            refused.append(str(caught.value))
        editor.garrison(archers[0], None)
        editor.garrison(archers[0], None)  # already out: nothing to do
        made.update(tower=tower, archers=archers)

    outcome = doc.call(script)
    assert outcome.ok, outcome.detail
    tower, archers = made["tower"], made["archers"]
    assert "is full (5 places)" in refused[0]
    for inside in archers[1:-1]:
        assert inside.garrisoned_in_id == tower.reference_id
        assert (inside.x, inside.y, inside.z) == (tower.x, tower.y, tower.z)
    assert archers[-1].garrisoned_in_id == -1 and (archers[-1].x, archers[-1].y) == (45.5, 40.5)
    mm = doc.loaded.map_manager
    bounds = render.unit_tile_bounds(tower, mm.map_width, mm.map_height)
    tile = garrison.unload_tile(bounds, mm.map_width, mm.map_height)
    assert archers[0].garrisoned_in_id == -1
    assert (archers[0].x, archers[0].y) == (tile[0] + 0.5, tile[1] + 0.5)
    doc.run.model._check_alignment()


def test_rotate_set_variant_and_add_unit_stay_inside_their_write_rules() -> None:
    doc = Doc()
    made = {}

    def script(editor) -> None:
        archer, wall, oak = (_ref(editor, r) for r in (_ARCHER_P1, _WALL, _TREE_OAK))
        wall_rotation = wall.rotation
        refusals = (
            lambda: editor.rotate(wall, 1.0),  # a wall's rotation is a shape
            lambda: editor.rotate(oak, 1.0),
            lambda: editor.rotate(archer, math.nan),
            lambda: editor.rotate(archer, math.inf),
            lambda: editor.set_variant(oak, 2.5),
            lambda: editor.set_variant(oak, True),
            lambda: editor.set_variant(oak, 42),  # an oak has 42: 0..41
            lambda: editor.set_variant(oak, -1),
            lambda: editor.set_variant(wall, 1),
            lambda: editor.set_variant(archer, 1),
            lambda: editor.add_unit(999999, 5.5, 5.5, 1),
            lambda: editor.add_unit(_ARCHER_CONST, 500.5, 5.5, 1),  # off the map
            lambda: editor.add_unit(_ARCHER_CONST, 5.5, 5.5, 9),
            lambda: editor.add_unit(_OAK_CONST, 5.5, 5.5, 0, facing=1.0),
            lambda: editor.add_unit(_ARCHER_CONST, 5.5, 5.5, 1, variant=1),
            lambda: editor.add_unit(_OAK_CONST, 5.5, 5.5, 0, variant=42),
            lambda: editor.move_unit(archer, math.nan, 5.5),
            lambda: editor.move_unit(archer, -1, 5.5),
            lambda: editor.set_status(archer, 256),
            lambda: editor.set_capture_flag(archer, 1),  # a 1.58 file has no capture flag
            lambda: editor.reassign(oak, 1),  # GAIA-only
            lambda: editor.remove(_ref(editor, _HOUSE)),  # the villager inside it would dangle
            lambda: editor.move_unit(copy.copy(archer), 5.5, 5.5),  # not a unit of this scenario
        )
        for attempt in refusals:
            with pytest.raises(editor.ScriptError):
                attempt()
        assert wall.rotation == wall_rotation
        editor.rotate(archer, 0.3)
        editor.set_variant(oak, 3)
        made["tree"] = editor.add_unit(_OAK_CONST, 60.5, 60.5, 1, variant=5)
        made["archer"] = editor.add_unit(_ARCHER_CONST, 61.5, 60.5, 1, facing=1.0)
        made["plain"] = editor.add_unit(_WATCH_TOWER_CONST, 70.5, 70.5, 2)

    outcome = doc.call(script)
    assert outcome.ok, outcome.detail
    count = unit_rotation.angle_count_for(_ARCHER_CONST)
    snapped = unit_rotation.facing_to_rotation(unit_rotation.rotation_to_facing(0.3, count), count)
    assert doc.unit(_ARCHER_P1).rotation == snapped != 0.3
    assert doc.unit(_TREE_OAK).rotation == 3.0
    units = doc.loaded.unit_manager.units
    # A GAIA-only object goes to GAIA whatever the player asked for.
    assert any(u is made["tree"] for u in units[0])
    state = place_rotation.PlaceRotation
    assert made["tree"].rotation == place_rotation.rotation_for(state(variant=5, variant_const=_OAK_CONST), _OAK_CONST, 0)
    assert made["archer"].rotation == place_rotation.rotation_for(state(facing=1.0), _ARCHER_CONST, 1)
    assert made["plain"].rotation == place_rotation.rotation_for(state(), _WATCH_TOWER_CONST, 2)
    assert made["tree"].rotation == 5.0 and made["archer"].rotation not in (0.0, 1.0)


def test_a_kept_editor_refuses_to_write_after_its_run() -> None:
    doc = Doc()
    kept = []
    assert doc.call(kept.append).ok
    editor = kept[0]
    assert editor.width == editor.height == 120 and editor.name == FIXTURE_PATH.name
    assert len(editor.units()) == 8 and len(editor.units(1)) == 3 and editor.tile(0, 0) is doc.tiles[0]
    assert editor.selected_units == () and editor.selected_tiles == ()
    for write in (
        lambda: editor.set_terrain(0, 0, 6),
        lambda: editor.move_unit(doc.unit(_ARCHER_P1), 5.5, 5.5),
        lambda: editor.progress(1, 2),
    ):
        with pytest.raises(ScriptError, match="only change the scenario while"):
            write()
    assert doc.history.records == []


def test_the_selection_comes_from_the_window() -> None:
    doc = Doc()
    archer = doc.unit(_ARCHER_P1)
    seen = {}

    def script(editor) -> None:
        seen.update(units=editor.selected_units, tiles=editor.selected_tiles)

    doc.call(script, selected_units=lambda: [archer], selected_tiles=lambda: [(1, 2)])
    assert seen == {"units": (archer,), "tiles": ((1, 2),)}


# -- terrain -------------------------------------------------------------------


def _fails(doc: Doc, script, expect: str) -> None:
    before, records = doc.state(), list(doc.history.records)
    outcome = doc.call(script)
    assert expect in outcome.error, outcome.error
    doc.assert_rolled_back(outcome, before, records)


def test_a_direct_tile_write_is_recorded_when_legal_and_fails_the_run_when_not() -> None:
    doc = Doc()
    before = doc.state()

    def legal(editor) -> None:
        editor.tile(5, 5).elevation = 1
        editor.tile(6, 5).terrain_id = 6

    outcome = doc.call(legal)
    assert outcome.ok and outcome.changed == ("terrain",) and sorted(outcome.record.touched_indices()) == [605, 606]
    doc.move_history("undo")
    doc.assert_state(before)

    def step_of_two(editor) -> None:
        editor.tile(5, 5).elevation = 2

    def too_high(editor) -> None:
        editor.tile(5, 5).elevation = 99

    def bad_terrain(editor) -> None:
        editor.tile(5, 5).terrain_id = 300

    def layer_equal_to_base(editor) -> None:
        tile = editor.tile(5, 5)
        tile.layer = tile.terrain_id

    _fails(doc, step_of_two, "differ by 2 elevation levels")
    # Both tiles of a pair changed: the pair is named once, not once from each side.
    changes = [(605, (0, 0, -1), (0, 3, -1)), (606, (0, 0, -1), (0, 5, -1))]
    doc.tiles[605].elevation, doc.tiles[606].elevation = 3, 5
    found = scripting.terrain_findings(changes, doc.tiles, 120, 120, write_supported=True, has_layer=True, limit=99)
    assert len([f for f in found if "(5, 5) and (6, 5)" in f or "(6, 5) and (5, 5)" in f]) == 1
    doc.tiles[605].elevation, doc.tiles[606].elevation = 0, 0
    _fails(doc, too_high, "elevation 99 is not an int in 0..15")
    _fails(doc, bad_terrain, "terrain_id 300 is not an int in 0..255")
    _fails(doc, layer_equal_to_base, "equals its own terrain_id")


def test_an_illegal_step_the_run_did_not_touch_does_not_fail_it() -> None:
    doc = Doc()
    doc.tiles[10 * 120 + 10].elevation = 4  # already in the file, as far as the run can tell
    outcome = doc.call(lambda editor: editor.set_terrain(50, 50, 6))
    assert outcome.ok and outcome.changed == ("terrain",)


def test_terrain_the_file_cannot_save_is_refused_by_the_facade_and_by_the_run_end_check() -> None:
    doc = Doc()
    doc.loaded.terrain_has_layer = False

    def facade(editor) -> None:
        with pytest.raises(editor.ScriptError, match="no layer field"):
            editor.set_layer(5, 5, 6)

    assert doc.call(facade).ok

    def direct_layer(editor) -> None:
        editor.tile(5, 5).layer = 6

    _fails(doc, direct_layer, "no layer field")

    doc.loaded.terrain_has_layer = True
    doc.loaded.map_is_square = False

    def unsquare(editor) -> None:
        for write in (lambda: editor.set_elevation(5, 5, 1), lambda: editor.raise_elevation([(5, 5)])):
            with pytest.raises(editor.ScriptError, match="square map"):
                write()
        editor.set_terrain(5, 5, 6)  # terrain has no such limit

    assert doc.call(unsquare).changed == ("terrain",)

    doc.loaded.map_is_square = True
    doc.loaded.terrain_write_supported = False

    def read_only(editor) -> None:
        with pytest.raises(editor.ScriptError, match="cannot be written"):
            editor.set_terrain(6, 6, 7)

    assert doc.call(read_only).ok

    def direct(editor) -> None:
        editor.tile(6, 6).terrain_id = 7

    _fails(doc, direct, "cannot be written")


def test_the_terrain_facade_writes_what_the_tools_write() -> None:
    doc = Doc()

    def script(editor) -> None:
        editor.set_terrain(5, 5, 6)
        editor.set_layer(5, 5, 9)
        editor.set_layer(6, 5, editor.tile(6, 5).terrain_id)  # its own terrain: removed
        editor.set_elevation(40, 40, 99)  # clamped
        assert editor.raise_elevation([_ref(editor, _ARCHER_P1), (11, 10), editor.tile(80, 80)], 1) == 2
        for bad in (lambda: editor.set_terrain(5, 5, 256), lambda: editor.tile(120, 0), lambda: editor.set_layer(5, 5, 300)):
            with pytest.raises(editor.ScriptError):
                bad()

    outcome = doc.call(script)
    assert outcome.ok, outcome.error
    tiles = doc.tiles
    assert (tiles[5 * 120 + 5].terrain_id, tiles[5 * 120 + 5].layer, tiles[5 * 120 + 6].layer) == (6, 9, -1)
    assert tiles[40 * 120 + 40].elevation == 15
    assert tiles[10 * 120 + 11].elevation == 1 and tiles[80 * 120 + 80].elevation == 1
    # What the facade wrote is a legal map: the run-end check passed above, and so does this.
    changes = outcome.record.children[0].changes
    assert scripting.terrain_findings(changes, tiles, 120, 120, write_supported=True, has_layer=True) == []


# -- the model's side ----------------------------------------------------------


def test_rollback_to_needs_an_open_whole_list_edit_over_all_nine_players() -> None:
    doc = Doc(with_model=True)
    lists = [list(units) for units in doc.loaded.unit_manager.units]
    with pytest.raises(RuntimeError, match="rollback_to"):
        doc.model.rollback_to(lists)
    doc.model.begin_unit_edit([1])
    with pytest.raises(RuntimeError, match="rollback_to"):
        doc.model.rollback_to(lists)
    doc.model.abort_unit_edit()
    doc.model.begin_unit_edit(range(9), fields_only=True)
    with pytest.raises(RuntimeError, match="rollback_to"):
        doc.model.rollback_to(lists)


# -- folder, consent, switches -------------------------------------------------


def test_discover_scripts_skip_rules_root_and_digest(tmp_path) -> None:
    for name in ("b.py", "a.py", "_private.py", ".hidden.py", "notes.txt"):
        (tmp_path / name).write_text("pass\n")
    (tmp_path / "package.py").mkdir()
    assert [entry.name for entry in scripting.discover_scripts(tmp_path)] == ["a", "b"]
    assert scripting.discover_scripts(tmp_path / "absent") == []

    assert scripting.script_root({scripting.ENV_SCRIPT_PATH: str(tmp_path)}) == tmp_path
    assert scripting.script_root({}).name == "scripts"

    path = tmp_path / "a.py"
    first = scripting.content_digest(path)
    assert first == scripting.content_digest(path.read_bytes())
    path.write_text("pass  # edited\n")
    assert scripting.content_digest(path) != first

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "one.py").write_text("x = 1\n")
    whole = scripting.content_digest(package)
    (package / "one.py").rename(package / "two.py")
    assert scripting.content_digest(package) != whole


def test_consent_is_kept_per_kind_and_per_path_and_survives_a_reload(monkeypatch) -> None:
    assert settings.get_confirmed_scripts() == set() == settings.get_confirmed_addons()
    settings.add_confirmed_script("/s/a.py", "d1")
    settings.add_confirmed_addon("/s/a.py", "d9")
    settings.add_confirmed_script("/s/b.py", "d2")
    settings.add_confirmed_script("/s/a.py", "d3")  # new bytes replace the old consent
    monkeypatch.setattr(settings, "_confirmed_scripts", None)
    monkeypatch.setattr(settings, "_confirmed_addons", None)
    assert settings.get_confirmed_scripts() == {("/s/a.py", "d3"), ("/s/b.py", "d2")}
    # Agreeing to run a file as a script does not load it as an addon.
    assert settings.get_confirmed_addons() == {("/s/a.py", "d9")}

    config = settings._load_config()
    config["confirmed_scripts"] = [{"path": "/ok.py", "digest": "d"}, ["/list.py", "d"], {"path": 3, "digest": "d"}, "x"]
    settings._save_config(config)
    monkeypatch.setattr(settings, "_confirmed_scripts", None)
    assert settings.get_confirmed_scripts() == {("/ok.py", "d")}


def test_addons_disabled_by_names_the_switch() -> None:
    assert settings.addons_disabled_by(["descape", "map.aoe2scenario"], {}) == ""
    assert settings.addons_disabled_by(["descape", "--no-addons"], {}) == "--no-addons"
    assert settings.addons_disabled_by([], {"DESCAPE_NO_ADDONS": "1"}) == "DESCAPE_NO_ADDONS"
    assert settings.addons_disabled_by(None, {"DESCAPE_NO_ADDONS": "0"}) == ""
