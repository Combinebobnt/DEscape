"""GH #186's model half: UnitEditModel.set_status()/set_capture_flag(), the
per-file capture-flag gate (scenario_io.unit_field_supported) and the
UnitState widening that lets undo restore both fields on the live Unit.

Unit State is the `status` byte, in every version. Capture Flag is
`capture_flag`, stored from 1.59 only: before that the library's link is
poisoned (reading raises, and so does assigning anything but None), and after
a depoison() the unit has no such attribute at all. So the 1.58 fixture
carries the refusals and the 1.59 one the edits.

See tests/test_unit_choice_panel.py for the inspector rows and
tests/test_unit_choice_viewer.py for the path from a combo to the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import library_compat, scenario_io
from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel, UnitFieldSnapshot, capture_flag_of, unit_state

V158 = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
V159 = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"

# 1.58: a Player 1 archer and villager. 1.59: an archer on Convert Multiple
# Times (2), a villager on Default (-1), both Alive (2).
_V158_ARCHER = 201
_V158_VILLAGER = 203
_V159_ARCHER = 200
_V159_VILLAGER = 201


def _open(path: Path) -> tuple:
    loaded = load_map_and_units(path)
    return loaded, UnitEditModel(loaded)


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _reload(loaded, model, tmp_path: Path):
    out = tmp_path / "saved.aoe2scenario"
    write_scenario(loaded, out, units=model)
    return load_map_and_units(out)


def _fields(loaded) -> dict:
    """reference_id -> (status, capture flag) for every unit, to prove an
    edit reached one unit and no other."""
    return {u.reference_id: (u.status, capture_flag_of(u)) for u in loaded.unit_manager.get_all_units()}


# -- the per-file gate ---------------------------------------------------------


def test_the_predicate_follows_the_links_own_support_range() -> None:
    v158 = load_map_and_units(V158)
    v159 = load_map_and_units(V159)
    assert scenario_io.unit_field_supported(v159, "capture_flag") is True
    assert scenario_io.unit_field_supported(v158, "capture_flag") is False
    for loaded in (v158, v159):
        assert scenario_io.unit_field_supported(loaded, "status") is True
        assert scenario_io.unit_field_supported(loaded, "no_such_field") is False


def test_the_predicate_answers_for_a_file_with_no_codec(monkeypatch) -> None:
    """unit_codec() may be None (a structure the encoder refuses), so the
    gate cannot hang off codec.unsupported: it must still say True and False."""
    monkeypatch.setattr(scenario_io, "unit_codec", lambda loaded: None)
    v158 = load_map_and_units(V158, fast_units=False)
    v159 = load_map_and_units(V159, fast_units=False)
    assert scenario_io.unit_codec(v159) is None
    assert scenario_io.unit_field_supported(v159, "capture_flag") is True
    assert scenario_io.unit_field_supported(v158, "capture_flag") is False
    assert UnitEditModel(v159).supports_capture_flag is True
    assert UnitEditModel(v158).supports_capture_flag is False


def test_the_model_decides_capture_flag_support_at_construction() -> None:
    assert _open(V159)[1].supports_capture_flag is True
    assert _open(V158)[1].supports_capture_flag is False


# -- the two setters -----------------------------------------------------------


@pytest.mark.parametrize("path", [V158, V159], ids=["1.58", "1.59"])
def test_set_status_dirties_only_that_unit_and_survives_a_reload(path: Path, tmp_path: Path) -> None:
    loaded, model = _open(path)
    before = _fields(loaded)
    reference_id = _V158_ARCHER if path == V158 else _V159_ARCHER
    archer = _unit(loaded, reference_id)
    gen0 = loaded.unit_gen

    model.set_status(archer, 0)

    assert archer.status == 0
    assert model.has_edits
    assert loaded.unit_gen == gen0 + 1
    assert sum(blob is None for blobs in model._blobs for blob in blobs) == 1
    expected = dict(before)
    expected[reference_id] = (0, before[reference_id][1])
    assert _fields(_reload(loaded, model, tmp_path)) == expected


def test_set_capture_flag_dirties_only_that_unit_and_survives_a_reload(tmp_path: Path) -> None:
    loaded, model = _open(V159)
    before = _fields(loaded)
    villager = _unit(loaded, _V159_VILLAGER)
    assert before[_V159_VILLAGER] == (2, -1)
    gen0 = loaded.unit_gen

    model.set_capture_flag(villager, 1)

    assert villager.capture_flag == 1
    assert model.has_edits
    assert loaded.unit_gen == gen0 + 1
    assert sum(blob is None for blobs in model._blobs for blob in blobs) == 1
    expected = dict(before)
    expected[_V159_VILLAGER] = (2, 1)
    assert _fields(_reload(loaded, model, tmp_path)) == expected


def test_set_capture_flag_saves_through_the_commit_when_there_is_no_codec(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(scenario_io, "unit_codec", lambda loaded: None)
    loaded = load_map_and_units(V159, fast_units=False)
    model = UnitEditModel(loaded)
    assert model._codec is None
    model.set_capture_flag(_unit(loaded, _V159_VILLAGER), 3)
    model.set_status(_unit(loaded, _V159_ARCHER), 7)
    saved = _fields(_reload(loaded, model, tmp_path))
    assert saved[_V159_VILLAGER] == (2, 3)
    assert saved[_V159_ARCHER] == (7, 2)


def test_a_raw_value_with_no_label_is_written_as_given(tmp_path: Path) -> None:
    """Neither setter checks the label tables: a script may write any raw
    value the field can hold, the same convention as add()."""
    loaded, model = _open(V159)
    model.set_status(_unit(loaded, _V159_ARCHER), 255)
    model.set_capture_flag(_unit(loaded, _V159_VILLAGER), -128)
    saved = _fields(_reload(loaded, model, tmp_path))
    assert saved[_V159_ARCHER] == (255, 2)
    assert saved[_V159_VILLAGER] == (2, -128)


# -- scope guards --------------------------------------------------------------


@pytest.mark.parametrize("depoisoned", [False, True], ids=["poisoned", "depoisoned"])
@pytest.mark.parametrize("value", [-1, 0, 1])
def test_set_capture_flag_refuses_a_file_saved_before_1_59(depoisoned: bool, value: int) -> None:
    """Loud on both sides of a depoison(). After one the unit has no such
    attribute, so without the gate a bare assignment would create an attribute
    the encoder ignores: an edit the save silently drops."""
    loaded, model = _open(V158)
    archer = _unit(loaded, _V158_ARCHER)
    if depoisoned:
        library_compat.depoison()
    with pytest.raises(ValueError, match=r"stored from 1\.59"):
        model.set_capture_flag(archer, value)
    assert not model.has_edits
    assert "capture_flag" not in vars(archer)
    assert all(blob is not None for blobs in model._blobs for blob in blobs)


@pytest.mark.parametrize("value", [128, -129, 1.0, True, None, "1"])
def test_set_capture_flag_refuses_a_value_outside_s8(value) -> None:
    loaded, model = _open(V159)
    villager = _unit(loaded, _V159_VILLAGER)
    with pytest.raises(ValueError, match="capture_flag"):
        model.set_capture_flag(villager, value)
    assert villager.capture_flag == -1
    assert not model.has_edits


@pytest.mark.parametrize("value", [256, -1, 2.0, True, None, "2"])
def test_set_status_refuses_a_value_outside_u8(value) -> None:
    loaded, model = _open(V158)
    archer = _unit(loaded, _V158_ARCHER)
    with pytest.raises(ValueError, match="status"):
        model.set_status(archer, value)
    assert archer.status == 2
    assert not model.has_edits


@pytest.mark.parametrize("value", [-128, 127])
def test_set_capture_flag_takes_the_s8_bounds(value: int) -> None:
    loaded, model = _open(V159)
    villager = _unit(loaded, _V159_VILLAGER)
    model.set_capture_flag(villager, value)
    assert villager.capture_flag == value


@pytest.mark.parametrize("value", [0, 255])
def test_set_status_takes_the_u8_bounds(value: int) -> None:
    loaded, model = _open(V158)
    archer = _unit(loaded, _V158_ARCHER)
    model.set_status(archer, value)
    assert archer.status == value


def test_both_setters_take_the_parser_enums_and_store_plain_ints() -> None:
    from AoE2ScenarioParser.datasets.trigger_lists import CaptureFlag, ObjectState

    loaded, model = _open(V159)
    villager = _unit(loaded, _V159_VILLAGER)
    model.set_capture_flag(villager, CaptureFlag.ONCE)
    model.set_status(villager, ObjectState.FOUNDATION)
    assert type(villager.capture_flag) is int and villager.capture_flag == 1
    assert type(villager.status) is int and villager.status == 0


def test_both_setters_run_inside_a_fields_only_edit() -> None:
    loaded, model = _open(V159)
    villager = _unit(loaded, _V159_VILLAGER)
    model.begin_unit_edit([1], fields_only=True)
    model.set_status(villager, 3)
    model.set_capture_flag(villager, 2)
    pending = model._pending
    assert isinstance(pending, UnitFieldSnapshot)
    assert len(pending.entries) == 1, "two writes to one unit are one delta entry"
    model.commit_unit_edit("Set both", EditHistory())
    assert (villager.status, villager.capture_flag) == (3, 2)


# -- undo: the live object and the blob ----------------------------------------


def _edit(model, players, fields_only: bool, history, mutate) -> None:
    model.begin_unit_edit(players, fields_only=fields_only)
    mutate()
    model.commit_unit_edit("Set field", history)


@pytest.mark.parametrize("fields_only", [False, True], ids=["whole-list", "field-delta"])
def test_undo_restores_status_on_the_live_unit_and_the_bytes(fields_only: bool, tmp_path: Path) -> None:
    """Without `status` in UnitState the blob comes back (the FILE is right)
    while the live Unit keeps the edit, so the inspector and the next save
    disagree. Both restore paths: restore() and _restore_field_delta()."""
    loaded, model = _open(V159)
    history = EditHistory()
    archer = _unit(loaded, _V159_ARCHER)

    _edit(model, [1], fields_only, history, lambda: model.set_status(archer, 3))
    assert archer.status == 3

    history.undo([], None, None, model)
    assert archer.status == 2
    assert not model.has_edits
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == V159.read_bytes()

    history.redo([], None, None, model)
    assert archer.status == 3
    assert _fields(_reload(loaded, model, tmp_path))[_V159_ARCHER] == (3, 2)


@pytest.mark.parametrize("fields_only", [False, True], ids=["whole-list", "field-delta"])
def test_undo_restores_the_capture_flag_on_the_live_unit_and_the_bytes(fields_only: bool, tmp_path: Path) -> None:
    loaded, model = _open(V159)
    history = EditHistory()
    archer = _unit(loaded, _V159_ARCHER)
    assert archer.capture_flag == 2

    _edit(model, [1], fields_only, history, lambda: model.set_capture_flag(archer, 0))
    assert archer.capture_flag == 0

    history.undo([], None, None, model)
    assert archer.capture_flag == 2
    assert not model.has_edits
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == V159.read_bytes()

    history.redo([], None, None, model)
    assert archer.capture_flag == 0
    assert _fields(_reload(loaded, model, tmp_path))[_V159_ARCHER] == (2, 0)


@pytest.mark.parametrize("depoison_at", ["never", "before the edit", "before the undo"])
@pytest.mark.parametrize("fields_only", [False, True], ids=["whole-list", "field-delta"])
def test_a_pre_1_59_status_undo_never_touches_the_capture_flag(
    fields_only: bool, depoison_at: str, tmp_path: Path
) -> None:
    """Before 1.59 the snapshot holds None for the capture flag and the
    restore skips it: assigning even -1 to the poisoned link raises, and after
    a depoison() it would create an attribute the file has no field for."""
    loaded, model = _open(V158)
    history = EditHistory()
    archer = _unit(loaded, _V158_ARCHER)
    if depoison_at == "before the edit":
        library_compat.depoison()

    _edit(model, [1], fields_only, history, lambda: model.set_status(archer, 7))
    assert archer.status == 7
    if depoison_at == "before the undo":
        library_compat.depoison()

    history.undo([], None, None, model)
    assert archer.status == 2
    assert "capture_flag" not in vars(archer)
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, units=model)
    assert out.read_bytes() == V158.read_bytes()

    history.redo([], None, None, model)
    assert archer.status == 7
    assert "capture_flag" not in vars(archer)


def test_a_snapshot_reads_no_capture_flag_from_a_pre_1_59_file() -> None:
    """The model skips the read by its own gate, not by what the unit happens
    to hold: add() gives a new unit a plain `capture_flag = -1` attribute even
    here, and a snapshot that stored it would assign it back on undo."""
    _loaded, model = _open(V158)
    added = model.add(player=1, unit_const=83, x=30.5, y=30.5, z=0.0, rotation=0.0)
    assert vars(added)["capture_flag"] == -1
    snapshot = model._capture([0, 1, 2])
    assert {state[-1] for pls in snapshot.players.values() for state in pls.states} == {None}
    assert {state[-2] for pls in snapshot.players.values() for state in pls.states} == {2}

    model.begin_unit_edit([1], fields_only=True)
    model.set_status(added, 0)
    assert {entry.state[-1] for entry in model._pending.entries.values()} == {None}
    record = model.commit_unit_edit("Set Unit State", EditHistory())
    assert {entry.state[-1] for entry in record.after.entries.values()} == {None}


def test_unit_state_carries_both_fields_on_a_1_59_unit() -> None:
    loaded, _model = _open(V159)
    archer = _unit(loaded, _V159_ARCHER)
    assert unit_state(archer)[-2:] == (2, 2)
    assert unit_state(archer, capture_flag=False)[-2:] == (2, None)
