"""descape.trigger_range: which trigger effects change a placed building's
maximum range, and to what (GH #49 slice 2). Default tier, no QApplication;
one corpus-marked test pins a real file's collection."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from descape import library_compat, trigger_range
from descape.trigger_fields import UNSET
from descape.trigger_range import UNREAD, UNREADABLE

import conftest

VOCABULARY = library_compat.load_vocabulary("1.59")

CASTLE = 82
WATCH_TOWER = 79
TREBUCHET = 42
HOUSE = 70

_EFFECT_DEFAULTS = {
    "quantity": UNSET,
    "operation": UNSET,
    "source_player": 1,
    "object_list_unit_id": UNSET,
    "object_group": UNSET,
    "object_type": UNSET,
    "object_group2": UNSET,
    "object_type2": UNSET,
    "selected_object_ids": [],
    "area_x1": UNSET,
    "area_y1": UNSET,
    "area_x2": UNSET,
    "area_y2": UNSET,
    "object_attributes": UNSET,
}


def _effect(effect_type: int, **fields) -> SimpleNamespace:
    values = dict(_EFFECT_DEFAULTS)
    if effect_type in trigger_range.ATTRIBUTE_EFFECTS:
        values["object_attributes"] = trigger_range.MAXIMUM_RANGE_ATTRIBUTE
    values.update(fields)
    return SimpleNamespace(effect_type=effect_type, **values)


def _trigger(*effects, name="T", order=None, conditions=(), looping=0, enabled=1) -> SimpleNamespace:
    trigger = SimpleNamespace(
        name=name, effects=list(effects), conditions=list(conditions), looping=looping, enabled=enabled
    )
    if order is not None:
        trigger.effect_order = list(order)
    return trigger


def _collect(*triggers):
    return trigger_range.collect_range_effects(SimpleNamespace(triggers=list(triggers)), VOCABULARY)


def _unit(player_id=1, unit_const=CASTLE, reference_id=500, own_tile=(10, 10)) -> dict:
    return {"player_id": player_id, "unit_const": unit_const, "reference_id": reference_id, "own_tile": own_tile}


def _ranges(effects, base=8.0, **unit):
    return trigger_range.trigger_ranges(trigger_range.index_effects(effects), base, **_unit(**unit))


def _applies(effect, **unit) -> bool:
    return trigger_range.effect_applies(effect, **_unit(**unit))


# -- collection ---------------------------------------------------------------


@pytest.mark.parametrize("effect_type", [32, 51, 79, 104, 105, 106])
def test_every_range_effect_type_is_collected(effect_type) -> None:
    effects = _collect(_trigger(_effect(effect_type, quantity=3, operation=2)))
    assert [e.effect_type for e in effects] == [effect_type]


@pytest.mark.parametrize("effect_type", [51, 79, 104, 105, 106])
def test_an_attribute_effect_on_another_attribute_is_not_collected(effect_type) -> None:
    assert _collect(_trigger(_effect(effect_type, quantity=3, operation=2, object_attributes=1))) == ()


def test_other_effect_types_are_not_collected() -> None:
    assert _collect(_trigger(SimpleNamespace(effect_type=24, quantity=3))) == ()


def test_collection_reads_the_trigger_flags_and_the_class_wide_fields() -> None:
    (effect,) = _collect(
        _trigger(
            _effect(104, quantity=6, operation=1, object_group2=52, object_type2=2, source_player=2),
            name="Init", conditions=[object()], looping=1, enabled=0,
        )
    )
    assert (effect.trigger_name, effect.has_conditions, effect.looping, effect.enabled) == ("Init", True, True, False)
    assert (effect.group, effect.object_type, effect.source_player, effect.quantity) == (52, 2, 2, 6.0)


def test_only_the_attributes_the_vocabulary_lists_are_read() -> None:
    """105 lists no object_group/object_type, so a stray value there is never read."""
    (effect,) = _collect(_trigger(_effect(105, quantity=3, operation=2, object_group=52, object_type=2)))
    assert (effect.group, effect.object_type) == (UNSET, UNSET)


def test_a_by_variable_effect_carries_no_quantity() -> None:
    (effect,) = _collect(_trigger(_effect(79, quantity=3, operation=2, object_list_unit_id=CASTLE)))
    assert effect.quantity is None and effect.by_variable


class _Raising(SimpleNamespace):
    def __getattribute__(self, name):
        if name == "source_player":
            raise RuntimeError("version-gated")
        return super().__getattribute__(name)


def test_a_raising_attribute_reads_as_unset() -> None:
    effect = _Raising(**{**_EFFECT_DEFAULTS, "effect_type": 32, "quantity": 2})
    (collected,) = _collect(_trigger(effect))
    assert collected.source_player == UNSET


@pytest.mark.parametrize("quantity", [None, -1, "x", True])
def test_an_unset_quantity_is_skipped(quantity) -> None:
    effects = _collect(_trigger(_effect(32, quantity=quantity, operation=1)))
    assert effects[0].quantity is None
    assert _ranges(effects) == ()


def test_a_half_set_area_is_no_area() -> None:
    (effect,) = _collect(_trigger(_effect(32, quantity=2, area_x1=-1, area_y1=0, area_x2=5, area_y2=5)))
    assert effect.area is None


# -- operations ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("operation", "quantity", "expected"),
    [(1, 12, 12.0), (2, 4, 12.0), (3, 3, 5.0), (4, 2, 16.0), (5, 2, 4.0), (3, 20, 0.0)],
)
def test_the_operation_table(operation, quantity, expected) -> None:
    effects = _collect(_trigger(_effect(32, quantity=quantity, operation=operation)))
    (result,) = _ranges(effects, base=8.0)
    assert result.value == expected


def test_divide_by_zero_skips_the_effect() -> None:
    assert _ranges(_collect(_trigger(_effect(32, quantity=0, operation=5)))) == ()


def test_legacy_change_object_range_with_operation_unset_adds() -> None:
    (result,) = _ranges(_collect(_trigger(_effect(32, quantity=2))), base=8.0)
    assert result.value == 10.0


@pytest.mark.parametrize("effect_type", [51, 105])
def test_operation_unset_on_the_attribute_effects_skips(effect_type) -> None:
    effects = _collect(_trigger(_effect(effect_type, quantity=2, object_list_unit_id=CASTLE)))
    assert _ranges(effects) == ()


# -- the matching table, row by row --------------------------------------------


def _one(effect_type, **fields):
    (effect,) = _collect(_trigger(_effect(effect_type, quantity=1, operation=2, **fields)))
    return effect


@pytest.mark.parametrize("effect_type", [51, 79])
def test_type_wide_matches_owner_and_const(effect_type) -> None:
    effect = _one(effect_type, object_list_unit_id=CASTLE)
    assert _applies(effect)
    assert not _applies(effect, player_id=2)
    assert not _applies(effect, unit_const=WATCH_TOWER)


def test_type_wide_with_no_const_matches_nothing() -> None:
    assert not _applies(_one(51))


def test_class_wide_matches_owner_class_and_optional_type() -> None:
    assert _applies(_one(104, object_group2=52), unit_const=WATCH_TOWER)
    assert _applies(_one(104, object_group2=52, object_type2=2), unit_const=WATCH_TOWER)
    assert not _applies(_one(104, object_group2=52, object_type2=4), unit_const=WATCH_TOWER)
    assert not _applies(_one(104, object_group2=52), unit_const=CASTLE)
    assert not _applies(_one(104, object_group2=52, source_player=2), unit_const=WATCH_TOWER)
    assert not _applies(_one(104), unit_const=WATCH_TOWER)


@pytest.mark.parametrize("effect_type", [32, 105, 106])
def test_selected_ids_win_over_every_other_filter(effect_type) -> None:
    effect = _one(
        effect_type, selected_object_ids=[500], source_player=3, object_list_unit_id=WATCH_TOWER,
        area_x1=0, area_y1=0, area_x2=1, area_y2=1,
    )
    assert _applies(effect, reference_id=500)
    assert not _applies(effect, reference_id=501)


@pytest.mark.parametrize("effect_type", [32, 105, 106])
def test_no_filter_set_means_every_object_of_that_player(effect_type) -> None:
    effect = _one(effect_type)
    assert _applies(effect) and _applies(effect, unit_const=HOUSE)
    assert not _applies(effect, player_id=2)


@pytest.mark.parametrize(("tile", "inside"), [((5, 5), True), ((9, 9), True), ((5, 9), True), ((4, 5), False),
                                              ((10, 9), False), ((9, 10), False)])
def test_the_area_is_inclusive_on_both_corners(tile, inside) -> None:
    effect = _one(32, area_x1=5, area_y1=5, area_x2=9, area_y2=9)
    assert _applies(effect, own_tile=tile) is inside


def test_each_object_filter_must_pass() -> None:
    assert _applies(_one(32, object_list_unit_id=CASTLE))
    assert not _applies(_one(32, object_list_unit_id=WATCH_TOWER))
    assert _applies(_one(32, object_group=3))
    assert not _applies(_one(32, object_group=52))
    assert _applies(_one(32, object_type=2))
    assert not _applies(_one(32, object_type=4))
    assert _applies(_one(32, object_type=4), unit_const=TREBUCHET)


def test_object_type_comes_from_the_baked_combat_level() -> None:
    assert trigger_range.object_type_of(CASTLE) == 2
    assert trigger_range.object_type_of(TREBUCHET) == 4
    assert trigger_range.object_type_of(999_999) is None


# -- per-trigger combination ----------------------------------------------------


def test_each_trigger_gives_its_own_value() -> None:
    effects = _collect(
        _trigger(_effect(32, quantity=4, operation=2), name="A"),
        _trigger(_effect(32, quantity=3, operation=2), name="B"),
    )
    assert [(r.trigger_name, r.value) for r in _ranges(effects, base=8.0)] == [("A", 12.0), ("B", 11.0)]


def test_a_triggers_effects_apply_in_display_order() -> None:
    """List order SET 12 then ADD 2 gives 14; display order [1, 0] runs ADD first, so SET wins: 12."""
    set_12 = _effect(32, quantity=12, operation=1)
    add_2 = _effect(32, quantity=2, operation=2)
    (identity,) = _ranges(_collect(_trigger(set_12, add_2)), base=8.0)
    (reordered,) = _ranges(_collect(_trigger(set_12, add_2, order=[1, 0])), base=8.0)
    assert (identity.value, reordered.value) == (14.0, 12.0)


def test_unresolved_counts_matching_by_variable_effects() -> None:
    effects = _collect(_trigger(_effect(79, object_list_unit_id=CASTLE), _effect(106), _effect(106, source_player=2)))
    index = trigger_range.index_effects(effects)
    assert trigger_range.unresolved_count(index, **_unit()) == 2
    assert trigger_range.trigger_ranges(index, 8.0, **_unit()) == ()


def test_the_index_matches_a_linear_scan() -> None:
    rng = random.Random(49)
    triggers = []
    for t in range(40):
        effects = []
        for _ in range(rng.randint(1, 4)):
            effect_type = rng.choice([32, 51, 79, 104, 105, 106])
            fields = {
                "quantity": rng.choice([1, 2, 5]),
                "operation": rng.choice([UNSET, 1, 2, 4]),
                "source_player": rng.randint(1, 3),
                "object_list_unit_id": rng.choice([UNSET, CASTLE, WATCH_TOWER]),
                "object_group": rng.choice([UNSET, 3, 52]),
                "object_type2": rng.choice([UNSET, 2]),
                "object_group2": rng.choice([UNSET, 3, 52]),
            }
            if rng.random() < 0.3:
                fields["selected_object_ids"] = rng.sample(range(500, 520), 3)
            if rng.random() < 0.3:
                x, y = rng.randint(0, 15), rng.randint(0, 15)
                fields.update(area_x1=x, area_y1=y, area_x2=x + 4, area_y2=y + 4)
            effects.append(_effect(effect_type, **fields))
        triggers.append(_trigger(*effects, name=f"T{t}"))
    effects = _collect(*triggers)
    index = trigger_range.index_effects(effects)
    for _ in range(200):
        unit = _unit(
            player_id=rng.randint(1, 3), unit_const=rng.choice([CASTLE, WATCH_TOWER]),
            reference_id=rng.randint(500, 520), own_tile=(rng.randint(0, 20), rng.randint(0, 20)),
        )
        linear = [e for e in effects if trigger_range.effect_applies(e, **unit)]
        indexed = [e for e in index.candidates(player_id=unit["player_id"], reference_id=unit["reference_id"])
                   if trigger_range.effect_applies(e, **unit)]
        assert indexed == linear


# -- describe() -----------------------------------------------------------------


def test_describe_one_value() -> None:
    effects = _collect(_trigger(_effect(32, quantity=4, operation=2), name="Init", conditions=[1], looping=1))
    line, tooltip = trigger_range.describe(_ranges(effects, base=10.0), 0, ringable=True)
    assert line == "Range with triggers: 14"
    assert tooltip.splitlines()[0] == '14: "Init", change object range, add 4, has conditions, loops'
    assert tooltip.splitlines()[-1] == trigger_range.NOT_SIMULATED


def test_describe_two_values_and_a_disabled_trigger() -> None:
    effects = _collect(
        _trigger(_effect(32, quantity=12, operation=1), name="A"),
        _trigger(_effect(105, quantity=3, operation=2), name="B", enabled=0),
    )
    line, tooltip = trigger_range.describe(_ranges(effects, base=8.0), 0, ringable=True)
    assert line == "Range with triggers: 12, 11"
    assert tooltip.splitlines()[1] == '11: "B", modify object attribute, add 3, disabled at start'


def test_describe_the_variable_only_line() -> None:
    assert trigger_range.describe((), 2, ringable=True)[0] == (
        "Range set from a variable by 2 trigger effects (not drawn)"
    )
    assert trigger_range.describe((), 1, ringable=True)[0] == (
        "Range set from a variable by 1 trigger effect (not drawn)"
    )


def test_describe_the_no_data_states() -> None:
    assert trigger_range.describe((), 0, state=UNREAD, ringable=True)[0] == (
        "Trigger range changes not read yet: open Triggers mode once"
    )
    assert trigger_range.describe((), 0, state=UNREADABLE, ringable=True)[0] == (
        "This file's triggers can't be read, so trigger range changes are not shown"
    )
    assert trigger_range.describe((), 0, state=UNREAD, ringable=False) == ("", "")
    assert trigger_range.describe((), 0, ringable=True) == ("", "")


# -- vocabulary sweep -------------------------------------------------------------

# What each effect must list wherever it exists, so a library bump that renames a
# targeting field fails here instead of silently blinding the collector.
_REQUIRED_ATTRIBUTES = {
    32: {"quantity", "operation", "source_player", "object_list_unit_id", "object_group", "object_type",
         "selected_object_ids", "area_x1", "area_y1", "area_x2", "area_y2"},
    51: {"quantity", "operation", "source_player", "object_list_unit_id", "object_attributes"},
    79: {"operation", "source_player", "object_list_unit_id", "object_attributes", "variable"},
    104: {"quantity", "operation", "source_player", "object_group2", "object_type2", "object_attributes"},
    105: {"quantity", "operation", "source_player", "object_list_unit_id", "object_attributes",
          "selected_object_ids", "area_x1", "area_y1", "area_x2", "area_y2"},
    106: {"operation", "source_player", "object_list_unit_id", "object_attributes", "variable",
          "selected_object_ids", "area_x1", "area_y1", "area_x2", "area_y2"},
}


def test_every_shipped_vocabulary_lists_the_pinned_attributes() -> None:
    missing = []
    seen = set()
    for version in library_compat.vocabulary_versions():
        effects = library_compat.load_vocabulary(version).effects
        for effect_id, required in _REQUIRED_ATTRIBUTES.items():
            definition = effects.get(effect_id)
            if definition is None:
                continue
            seen.add(effect_id)
            gap = required - set(definition.attributes)
            if gap:
                missing.append(f"v{version} {effect_id}: {sorted(gap)}")
    assert missing == []
    assert seen == set(_REQUIRED_ATTRIBUTES)


# -- corpus -------------------------------------------------------------------------


@pytest.mark.corpus
def test_old_allies_collects_its_nine_range_effects(request) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers

    path = conftest._scenario_dir(request.config) / "old-allies-final-v2.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"{path.name} not in examples/")
    loaded = load_map_and_units(str(path))
    manager = parse_triggers(loaded)
    effects = trigger_range.collect_range_effects(manager, library_compat.load_vocabulary(loaded.scenario_version))
    assert sorted(e.effect_type for e in effects) == [32] + [105] * 8
