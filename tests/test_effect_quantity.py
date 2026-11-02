"""An effect's stored quantity pair through a trigger re-serialize (TASK-554):
descape/effect_quantity.py.

No default-tier fixture holds a disagreeing pair, and the pair has to be on disk
before pull() sees it, so write_quantity_shapes() makes one. The corpus side is
tests/test_trigger_write_path.py's test_resave_keeps_every_effect_quantity_pair.
"""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from descape import effect_quantity, trigger_clipboard, trigger_fields, trigger_status
from descape.edit_history import EditHistory
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel, remove_entries

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"

_MODIFY_ATTRIBUTE = 51
_MODIFY_ATTRIBUTE_FOR_CLASS = 104
_HIT_POINTS, _MOVEMENT_SPEED, _ATTACK, _MAXIMUM_RANGE, _WORK_RATE = 0, 5, 9, 12, 13
# What the library writes into the dead float half.
_NAN_BYTES = b"\xff\xff\xff\xff"
_NAN = struct.unpack("<f", _NAN_BYTES)[0]
_ATTACK_CLASS, _ATTACK_AMOUNT = 3, 50


class Shape(NamedTuple):
    attribute: int
    whole: int
    part: float


SHAPES: dict[str, Shape] = {
    "sentinel int, stored 0.0": Shape(_MOVEMENT_SPEED, -1, 0.0),
    "real int, stored 0.0": Shape(_WORK_RATE, 100, 0.0),
    "stale int": Shape(_MAXIMUM_RANGE, 33, 12.0),
    "inexact float": Shape(_WORK_RATE, 2, 1.9),
    "sentinel float": Shape(_WORK_RATE, 7, _NAN),
    "stale float": Shape(_HIT_POINTS, 50, 12.0),
    "int-live, float -1.0": Shape(_HIT_POINTS, 5, -1.0),
    "int-live, stored 0.0": Shape(_HIT_POINTS, 50, 0.0),
    "armour pair, stale float": Shape(_ATTACK, _ATTACK_CLASS * 65536 + _ATTACK_AMOUNT, 12.0),
}
# Appended after the fixture's four, with nothing in it but SHAPES.
SHAPE_TRIGGER = 4


def shape_index(label: str) -> int:
    return list(SHAPES).index(label)


def _packed(whole: int, part: float) -> tuple[bytes, bytes]:
    return struct.pack("<i", whole), struct.pack("<f", part)


def _file_pairs() -> list[tuple[bytes, bytes]]:
    return [_packed(shape.whole, shape.part) for shape in SHAPES.values()]


def _slots(loaded) -> list:
    entries = loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data
    return entries[SHAPE_TRIGGER].retriever_map["effect_data"].data


def stored_pairs(path: Path) -> list[tuple[bytes, bytes]]:
    """The shape trigger's raw pairs, as the file holds them."""
    loaded = load_map_and_units(path)
    assert parse_triggers(loaded) is not None
    fields = [slot.retriever_map for slot in _slots(loaded)]
    return [(f["quantity"].get_data_as_bytes(), f["quantity_float"].get_data_as_bytes()) for f in fields]


def write_quantity_shapes(out: Path) -> Path:
    """triggers_120x120 plus a trigger of Modify Attribute effects holding
    SHAPES' raw pairs. The pairs go in behind the library, on a commit wrapped
    on this instance: new effects carry nothing, so effect_quantity.push()
    leaves them, and this stands in for no part of it."""
    loaded = load_map_and_units(FIXTURE_PATH)
    model = TriggerEditModel(loaded)
    manager = model.manager()
    assert len(manager.triggers) == SHAPE_TRIGGER

    def add_shapes(m) -> None:
        trigger = m.add_trigger("Quantity shapes")
        for shape in SHAPES.values():
            effect = trigger._add_effect(_MODIFY_ATTRIBUTE)
            effect.object_attributes = shape.attribute
            if shape.attribute == _ATTACK:
                effect.armour_attack_class, effect.armour_attack_quantity = _ATTACK_CLASS, _ATTACK_AMOUNT

    model.structural_edit(add_shapes)
    commit = manager.commit

    def commit_with_raw_pairs() -> None:
        commit()
        for slot, shape in zip(_slots(loaded), SHAPES.values(), strict=True):
            slot.retriever_map["quantity"].set_data(shape.whole, affect_dirty=False)
            slot.retriever_map["quantity_float"].set_data(shape.part, affect_dirty=False)

    manager.commit = commit_with_raw_pairs
    write_scenario(loaded, out, triggers=model)
    assert stored_pairs(out) == _file_pairs(), "the helper's own file does not hold SHAPES"
    return out


@pytest.fixture(scope="module")
def shapes_path(tmp_path_factory) -> Path:
    return write_quantity_shapes(tmp_path_factory.mktemp("quantity_shapes") / "shapes.aoe2scenario")


def _resaved(shapes_path: Path, tmp_path: Path, edit: Callable) -> list[tuple[bytes, bytes]]:
    """Rename the shape trigger, apply `edit(manager, trigger)`, save, reread."""
    loaded = load_map_and_units(shapes_path)
    model = TriggerEditModel(loaded)
    manager = model.manager()
    trigger = manager.triggers[SHAPE_TRIGGER]
    trigger.name = "Renamed"
    edit(manager, trigger)
    model.mark_dirty(SHAPE_TRIGGER)
    out = tmp_path / "resaved.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    return stored_pairs(out)


# -- a dirty trigger keeps what it stored ----------------------------------------


def test_a_dirty_rename_keeps_every_stored_pair(shapes_path: Path, tmp_path: Path) -> None:
    assert _resaved(shapes_path, tmp_path, lambda manager, trigger: None) == _file_pairs()


def test_two_serializes_of_a_dirty_trigger_agree(shapes_path: Path, tmp_path: Path) -> None:
    loaded = load_map_and_units(shapes_path)
    model = TriggerEditModel(loaded)
    model.manager().triggers[SHAPE_TRIGGER].name = "Renamed"
    model.mark_dirty(SHAPE_TRIGGER)
    assert model.serialize() == model.serialize()
    out = tmp_path / "resaved.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert stored_pairs(out) == _file_pairs()


# -- the heal --------------------------------------------------------------------


def test_a_stored_zero_reads_as_zero_on_a_float_live_effect_only(shapes_path: Path) -> None:
    loaded = load_map_and_units(shapes_path)
    effects = parse_triggers(loaded).triggers[SHAPE_TRIGGER].effects
    read = dict(zip(SHAPES, effects, strict=True))
    spec = trigger_fields.FieldSpec("quantity_float", trigger_fields.FLOAT, attribute="quantity")

    for label in ("sentinel int, stored 0.0", "real int, stored 0.0"):
        value = trigger_status.read_value(read[label], spec.attribute)
        assert value == 0.0 and isinstance(value, float), f"{label}: {value!r}"
        assert trigger_status.is_set(spec, value), f"{label}: a stored 0 reads as unset"
    assert read["stale int"].quantity == 12.0, "a real float keeps the library's reading"
    assert read["int-live, stored 0.0"].quantity == 50, "an int-live effect's float half is dead"


# -- an edit keeps the library's normalization -----------------------------------


def _set_quantity(value: float) -> Callable:
    return lambda effect: setattr(effect, "quantity", value)


def _switch_attribute(attribute: int) -> Callable:
    """object_attributes through cluster_fixup(), as the viewer's funnel writes it."""

    def switch(effect) -> None:
        before = trigger_fields.live_quantity_slot(effect)
        effect.object_attributes = attribute
        after = trigger_fields.live_quantity_slot(effect)
        for name, value in trigger_fields.cluster_fixup(effect, before, after, ("quantity", "quantity_float")):
            setattr(effect, name, value)

    return switch


@pytest.mark.parametrize(
    "label,edit,expected",
    [
        ("stale float", _set_quantity(77), (struct.pack("<i", 77), _NAN_BYTES)),
        ("stale int", _set_quantity(3.5), _packed(-1, 3.5)),
        ("real int, stored 0.0", _set_quantity(3.5), _packed(-1, 3.5)),
        ("stale int", _switch_attribute(_HIT_POINTS), (struct.pack("<i", 12), _NAN_BYTES)),
        ("stale float", _switch_attribute(_WORK_RATE), _packed(-1, 50.0)),
        ("stale float", lambda effect: setattr(effect, "effect_type", _MODIFY_ATTRIBUTE_FOR_CLASS), (struct.pack("<i", 50), _NAN_BYTES)),
        ("stale int", _set_quantity(12.0), _packed(33, 12.0)),
        ("inexact float", _set_quantity(1.9), _packed(2, 1.9)),
    ],
    ids=[
        "int-live edit",
        "float-live edit",
        "float-live edit over a stored 0",
        "attribute switch, float to int",
        "attribute switch, int to float",
        "type switch",
        "back to the stored value",
        "retyped inexact value",
    ],
)
def test_an_edit_writes_the_library_result_and_a_non_edit_keeps_the_pair(
    shapes_path: Path, tmp_path: Path, label: str, edit: Callable, expected: tuple[bytes, bytes]
) -> None:
    index = shape_index(label)
    want = _file_pairs()
    want[index] = expected
    assert _resaved(shapes_path, tmp_path, lambda manager, trigger: edit(trigger.effects[index])) == want


# -- the pair follows the effect, not the slot -----------------------------------


def test_the_pair_follows_its_effect_past_a_deleted_neighbour(shapes_path: Path, tmp_path: Path) -> None:
    gone = shape_index("real int, stored 0.0")
    pairs = _resaved(shapes_path, tmp_path, lambda manager, trigger: remove_entries(trigger, [("effect", gone)]))
    want = _file_pairs()
    del want[gone]
    assert pairs == want


def test_the_pair_follows_its_effect_past_an_anchored_paste(shapes_path: Path, tmp_path: Path) -> None:
    copied = shape_index("armour pair, stale float")

    def paste(manager, trigger) -> None:
        block = trigger_clipboard.copy_entries(manager, SHAPE_TRIGGER, [("effect", copied)])
        trigger_clipboard.paste_entries(manager, SHAPE_TRIGGER, block, ("effect", 0))

    want = _file_pairs()
    want.insert(1, want[copied])
    assert _resaved(shapes_path, tmp_path, paste) == want


# -- undo ------------------------------------------------------------------------


def test_undoing_an_edit_on_a_still_dirty_trigger_keeps_the_pair(shapes_path: Path, tmp_path: Path) -> None:
    """A lone edit plus undo restores the original blob, which proves nothing,
    so the rename in its own record keeps the trigger dirty through the undo."""
    loaded = load_map_and_units(shapes_path)
    model = TriggerEditModel(loaded)
    history = EditHistory()
    trigger = model.manager().triggers[SHAPE_TRIGGER]

    model.begin_trigger_edit([SHAPE_TRIGGER])
    trigger.name = "Renamed"
    model.commit_trigger_edit("Rename", history)

    model.begin_trigger_edit([SHAPE_TRIGGER])
    trigger.effects[shape_index("stale float")].quantity = 77
    record = model.commit_trigger_edit("Set quantity", history)

    model.restore(record.before)
    assert SHAPE_TRIGGER in model.dirty_indices()
    out = tmp_path / "undone.aoe2scenario"
    write_scenario(loaded, out, triggers=model)
    assert stored_pairs(out) == _file_pairs()


# -- a file with no float field ---------------------------------------------------


def test_a_carried_pair_leaves_a_slot_with_no_float_field_alone() -> None:
    """Only a batch script gets one there: the UI refuses a paste across versions."""
    effect = SimpleNamespace(effect_type=_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS)
    setattr(effect, effect_quantity._CARRIED, (_MODIFY_ATTRIBUTE, _HIT_POINTS, 5, 12.0))
    quantity = SimpleNamespace(data=8)
    effect_quantity.push([effect], [SimpleNamespace(retriever_map={"quantity": quantity})])
    assert quantity.data == 8
