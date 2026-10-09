"""UnitEditModel.replace_type() / replace_refusal(): Find and Replace's
in-place type change (GH #144), AGENTS.md's second const exception and its
*Replace* rotation/frame rule.

Default tier, no QApplication, on tests/fixtures/units_120x120.aoe2scenario.
A few cases set a raw field before the edit (e.g. a rotation or frame value
the fixture lacks); those never save, since the blob comes from the bytes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units
from descape.scenario_write import write_scenario
from descape.unit_model import UnitEditModel, replaced_rotation

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
MAP = 120

_REF_OAK = 100
_REF_PINE = 101
_REF_WALL = 102
_REF_HOUSE = 200
_REF_ARCHER_P1 = 201
_REF_VILLAGER_P1 = 203
_REF_ARCHER_P2 = 300

_ARCHER = 4
_KNIGHT = 38
_HOUSE = 70
_WATCH_TOWER = 79
_CASTLE = 82
_BARRACKS = 12
_DONJON = 1665
_GUARD_TOWER = 234
_STONE_WALL = 117
_PALISADE_WALL = 72
_GATE_UP = 64
_GATE_SIBLING = 659
_OAK = 349
_PINE = 350
_GOLD_MINE = 66
_CLIFF = 264
_WAR_GALLEY = 21
_BATTERING_RAM = 35


def _open() -> tuple:
    loaded = load_map_and_units(FIXTURE_PATH)
    return loaded, UnitEditModel(loaded), EditHistory()


def _unit(loaded, reference_id: int):
    return next(u for u in loaded.unit_manager.get_all_units() if u.reference_id == reference_id)


def _slot(loaded, unit) -> tuple[int, int]:
    for player, units in enumerate(loaded.unit_manager.units):
        for index, candidate in enumerate(units):
            if candidate is unit:
                return player, index
    raise AssertionError("unit not found")


def _replace(loaded, model, history, unit, new_const, *, fields_only=False):
    player = _slot(loaded, unit)[0]
    model.begin_unit_edit([player], fields_only=fields_only)
    model.replace_type(unit, new_const, map_w=MAP, map_h=MAP)
    model.commit_unit_edit("Replace", history)


# -- position ------------------------------------------------------------------


@pytest.mark.parametrize("fields_only", [False, True])
def test_same_span_replace_keeps_x_y_byte_exact(fields_only) -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P1)
    model.begin_unit_edit([1], fields_only=True)
    model.set_position(archer, 11.37, 10.81, 0.0)
    model.commit_unit_edit("Move", history)
    _replace(loaded, model, history, archer, _KNIGHT, fields_only=fields_only)
    assert (archer.unit_const, archer.x, archer.y) == (_KNIGHT, 11.37, 10.81)
    oak = _unit(loaded, _REF_OAK)
    _replace(loaded, model, history, oak, _PINE, fields_only=fields_only)
    assert (oak.x, oak.y) == (5.5, 5.5)


@pytest.mark.parametrize("fields_only", [False, True])
def test_different_span_replace_reanchors_and_undo_restores(fields_only) -> None:
    loaded, model, history = _open()
    oak = _unit(loaded, _REF_OAK)
    _replace(loaded, model, history, oak, _HOUSE, fields_only=fields_only)
    # Low corner (5, 5) kept; a 2x2 House anchors on the tile corner.
    assert (oak.unit_const, oak.x, oak.y) == (_HOUSE, 6.0, 6.0)
    history.undo([], None, None, model)
    assert (oak.unit_const, oak.x, oak.y) == (_OAK, 5.5, 5.5)
    history.redo([], None, None, model)
    assert (oak.unit_const, oak.x, oak.y) == (_HOUSE, 6.0, 6.0)


def test_only_the_changed_axis_is_reanchored() -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    model.begin_unit_edit([2], fields_only=True)
    model.set_position(archer, 20.3, 20.7, 0.0)
    model.commit_unit_edit("Move", history)
    # A (4, 1) gate: x's span changes, y's does not.
    _replace(loaded, model, history, archer, _GATE_UP)
    assert archer.x == 22.0
    assert archer.y == 20.7


# -- rotation / initial_animation_frame ----------------------------------------


def test_rotation_table() -> None:
    # ANGLE -> ANGLE: verbatim.
    assert replaced_rotation(_ARCHER, _KNIGHT, 1.25, 3) == (1.25, 3)
    # Cyclable in range: verbatim; out of range: 0/0.
    assert replaced_rotation(_OAK, _PINE, 7.0, 7) == (7.0, 7)
    assert replaced_rotation(_OAK, _GOLD_MINE, 41.0, 41) == (0.0, 0)
    # Wall -> wall keeps the index, radian-encoded or not.
    assert replaced_rotation(_STONE_WALL, _PALISADE_WALL, 2.5132741928100586, 0) == (2.5132741928100586, 0)
    # INERT -> INERT keeps the 7.0 sentinel.
    assert replaced_rotation(_WATCH_TOWER, _GUARD_TOWER, 7.0, 0) == (7.0, 0)
    # Tree -> building, building -> unit: 0/0.
    assert replaced_rotation(_OAK, _HOUSE, 7.0, 7) == (0.0, 0)
    assert replaced_rotation(_HOUSE, _ARCHER, 7.0, 0) == (0.0, 0)
    # Wall -> tree is "any other pairing".
    assert replaced_rotation(_STONE_WALL, _OAK, 3.0, 0) == (0.0, 0)
    # Anything -> gate: 0/0, even from INERT.
    assert replaced_rotation(_ARCHER, _GATE_UP, 1.25, 2) == (0.0, 0)
    assert replaced_rotation(_WATCH_TOWER, _GATE_UP, 7.0, 0) == (0.0, 0)


def test_replace_writes_rotation_and_frame_together() -> None:
    loaded, model, history = _open()
    oak = _unit(loaded, _REF_OAK)
    oak.rotation, oak.initial_animation_frame = 41.0, 41
    _replace(loaded, model, history, oak, _GOLD_MINE)
    assert (oak.rotation, oak.initial_animation_frame) == (0.0, 0)
    pine = _unit(loaded, _REF_PINE)
    _replace(loaded, model, history, pine, _OAK)
    assert pine.rotation == 41.0


@pytest.mark.parametrize("fields_only", [False, True])
def test_widened_unit_state_restores_the_frame_on_undo(fields_only) -> None:
    loaded, model, history = _open()
    oak = _unit(loaded, _REF_OAK)
    oak.rotation, oak.initial_animation_frame = 7.0, 7
    _replace(loaded, model, history, oak, _HOUSE, fields_only=fields_only)
    assert oak.initial_animation_frame == 0
    history.undo([], None, None, model)
    assert (oak.unit_const, oak.rotation, oak.initial_animation_frame) == (_OAK, 7.0, 7)


# -- what is kept ---------------------------------------------------------------


def test_identity_fields_are_preserved() -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    slot = _slot(loaded, archer)
    before = (archer.reference_id, archer.z, archer.status, archer.caption_string, archer.garrisoned_in_id)
    _replace(loaded, model, history, archer, _KNIGHT)
    assert _slot(loaded, archer) == slot
    assert (archer.reference_id, archer.z, archer.status, archer.caption_string, archer.garrisoned_in_id) == before
    villager = _unit(loaded, _REF_VILLAGER_P1)
    house = _unit(loaded, _REF_HOUSE)
    _replace(loaded, model, history, house, _DONJON)
    assert villager.garrisoned_in_id == _REF_HOUSE
    assert house.reference_id == _REF_HOUSE


def test_save_reload_round_trip(tmp_path: Path) -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    _replace(loaded, model, history, archer, _KNIGHT)
    oak = _unit(loaded, _REF_OAK)
    _replace(loaded, model, history, oak, _HOUSE)
    out = tmp_path / "replaced.aoe2scenario"
    write_scenario(loaded, out, units=model)
    again = load_map_and_units(out)
    knight = _unit(again, _REF_ARCHER_P2)
    assert (knight.unit_const, knight.x, knight.y, knight.caption_string) == (_KNIGHT, 20.5, 20.5, "Fixture caption")
    assert _slot(again, knight) == (2, 0)
    house = _unit(again, _REF_OAK)
    assert (house.unit_const, house.x, house.y, house.rotation, house.initial_animation_frame) == (
        _HOUSE,
        6.0,
        6.0,
        0.0,
        0,
    )


# -- raise scope -----------------------------------------------------------------


def _raises(model, unit, new_const, match):
    assert model.replace_refusal(unit, new_const, MAP, MAP) is not None
    model.begin_unit_edit([0, 1, 2])
    with pytest.raises(ValueError, match=match):
        model.replace_type(unit, new_const, map_w=MAP, map_h=MAP)
    model.abort_unit_edit()


def test_raise_cases() -> None:
    loaded, model, _history = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    _raises(model, archer, _ARCHER, "already that type")
    _raises(model, archer, 999_999, "unknown object type")
    _raises(model, archer, _CLIFF, "cliff")
    villager = _unit(loaded, _REF_VILLAGER_P1)
    # Inside a House, which holds no ship.
    _raises(model, villager, _WAR_GALLEY, "cannot go inside")
    house = _unit(loaded, _REF_HOUSE)
    # A Barracks has capacity but admits nothing, so the villager would have no home.
    _raises(model, house, _BARRACKS, "Garrison")
    assert not model.has_edits


def test_gate_siblings_are_refused() -> None:
    loaded, model, _history = _open()
    gate = _unit(loaded, _REF_WALL)
    gate.unit_const, gate.x, gate.y = _GATE_UP, 10.0, 5.5
    _raises(model, gate, _GATE_SIBLING, "orientation")


def test_off_map_and_map_edge_are_refused() -> None:
    loaded, model, history = _open()
    archer = _unit(loaded, _REF_ARCHER_P2)
    model.begin_unit_edit([2], fields_only=True)
    model.set_position(archer, 119.5, 60.5, 0.0)
    model.commit_unit_edit("Move", history)
    _raises(model, archer, _CASTLE, "leaves the map")
    assert model.replace_refusal(archer, _KNIGHT, MAP, MAP) is None
    model.begin_unit_edit([2], fields_only=True)
    model.set_position(archer, 130.5, 60.5, 0.0)
    model.commit_unit_edit("Move", history)
    _raises(model, archer, _KNIGHT, "off the map")


def test_batch_preflight_reads_planned_consts() -> None:
    """Host -> Castle with occupant -> Knight passes only as a batch; host ->
    Watch Tower with occupant -> Knight is tower-holds-knight, refused."""
    loaded, model, _history = _open()
    house = _unit(loaded, _REF_HOUSE)
    villager = _unit(loaded, _REF_VILLAGER_P1)
    assert model.replace_refusal(villager, _KNIGHT, MAP, MAP) is not None
    assert model.replace_refusal(villager, _KNIGHT, MAP, MAP, planned={_REF_HOUSE: _CASTLE}) is None
    assert model.replace_refusal(house, _CASTLE, MAP, MAP, planned={_REF_VILLAGER_P1: _KNIGHT}) is None
    assert model.replace_refusal(house, _WATCH_TOWER, MAP, MAP) is None
    assert model.replace_refusal(house, _WATCH_TOWER, MAP, MAP, planned={_REF_VILLAGER_P1: _KNIGHT}) is not None
    assert model.replace_refusal(villager, _KNIGHT, MAP, MAP, planned={_REF_HOUSE: _WATCH_TOWER}) is not None
    assert not model.has_edits
    assert (house.unit_const, villager.unit_const) == (_HOUSE, 83)


def test_batch_refusals_let_a_host_back_in_once_its_occupant_drops() -> None:
    """House and its villager -> Watch Tower: round one refuses both (a tower
    inside a tower). Once the villager drops, the house -> tower holding a
    villager passes, and the villager's reason is read against the final set."""
    loaded, model, _history = _open()
    house = _unit(loaded, _REF_HOUSE)
    villager = _unit(loaded, _REF_VILLAGER_P1)
    reasons = model.replace_batch_refusals([house, villager], _WATCH_TOWER, MAP, MAP)
    assert reasons == [None, "Garrison: Watch Tower cannot go inside Watch Tower"]
    assert model.replace_refusal(house, _WATCH_TOWER, MAP, MAP, planned={_REF_HOUSE: _WATCH_TOWER}) is None
    assert not model.has_edits


@pytest.mark.parametrize("order", ["occupant_first", "host_first"])
def test_a_preflighted_batch_applies_in_any_order(order) -> None:
    """House and its villager -> Battering Ram (a ram may hold a ram): the
    batch passes, and replace_type() given the batch's planned map does not
    raise on the occupant just because its host is still a House."""
    loaded, model, _history = _open()
    house = _unit(loaded, _REF_HOUSE)
    villager = _unit(loaded, _REF_VILLAGER_P1)
    batch = [villager, house] if order == "occupant_first" else [house, villager]
    assert model.replace_batch_refusals(batch, _BATTERING_RAM, MAP, MAP) == [None, None]
    planned = {u.reference_id: _BATTERING_RAM for u in batch}
    model.begin_unit_edit(sorted({_slot(loaded, u)[0] for u in batch}), fields_only=False)
    for unit in batch:
        model.replace_type(unit, _BATTERING_RAM, map_w=MAP, map_h=MAP, planned=planned)
    model.abort_unit_edit()
    assert (house.unit_const, villager.unit_const) == (_BATTERING_RAM, _BATTERING_RAM)


def test_batch_refusals_never_keep_a_stale_reason() -> None:
    """Every reason is the one replace_refusal() gives against the final
    passing set, and every passing unit passes with the others planned."""
    loaded, model, _history = _open()
    units = [_unit(loaded, r) for r in (_REF_HOUSE, _REF_VILLAGER_P1, _REF_ARCHER_P1, _REF_OAK)]
    for const in (_WATCH_TOWER, _CASTLE, _KNIGHT, _HOUSE, _BARRACKS):
        reasons = model.replace_batch_refusals(units, const, MAP, MAP)
        planned = {u.reference_id: const for u, r in zip(units, reasons, strict=True) if r is None}
        for unit, reason in zip(units, reasons, strict=True):
            assert model.replace_refusal(unit, const, MAP, MAP, planned=planned) == reason, (const, unit.reference_id)


def test_batch_refusals_terminate_on_a_mutual_conflict(monkeypatch) -> None:
    """Two units that each pass only while the other is dropped would
    oscillate; the shrink-only fallback drops both with BATCH_CONFLICT."""
    from descape.unit_model import BATCH_CONFLICT

    loaded, model, _history = _open()
    a, b = _unit(loaded, _REF_ARCHER_P1), _unit(loaded, _REF_ARCHER_P2)
    rival = {a.reference_id: b.reference_id, b.reference_id: a.reference_id}
    calls = []

    def refusal(unit, new_const, map_w, map_h, *, planned=None):
        calls.append(unit.reference_id)
        assert len(calls) < 50, "the rounds do not terminate"
        return "blocked by the other" if rival[unit.reference_id] in (planned or {}) else None

    monkeypatch.setattr(model, "replace_refusal", refusal)
    assert model.replace_batch_refusals([a, b], _KNIGHT, MAP, MAP) == [BATCH_CONFLICT, BATCH_CONFLICT]
