"""descape.trigger_status: the Qt-free completeness status behind the trigger
panel's colours (GH #166). Default tier, synthetic entries over the real 1.59
vocabulary; every table test passes its own requirements, so the committed
table's contents never change what these assert."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import library_compat, trigger_fields, trigger_status
from descape.trigger_fields import FieldSpec
from descape.trigger_status import CONDITION, EFFECT, Requirement, Status, StatusContext

VOCABULARY = library_compat.load_vocabulary("1.59")

_SEND_CHAT, _ACTIVATE_TRIGGER, _CREATE_OBJECT, _TASK_OBJECT, _KILL_OBJECT, _MODIFY_ATTRIBUTE = 3, 8, 11, 12, 14, 51
_DISPLAY_INSTRUCTIONS = 20
_REPLACE_OBJECT = 43
_TRIGGER_ACTIVE, _OWN_OBJECTS = 79, 1
_ARMOR, _HIT_POINTS = 8, 0


def _effect(effect_type: int, **fields):
    defaults = dict(VOCABULARY.effects[effect_type].default_attributes)
    defaults.update(effect_type=effect_type, **fields)
    return SimpleNamespace(**defaults)


def _condition(condition_type: int, **fields):
    defaults = dict(VOCABULARY.conditions[condition_type].default_attributes)
    defaults.update(condition_type=condition_type, **fields)
    return SimpleNamespace(**defaults)


def _context(requirements=None, *, units=(), triggers=(0, 1, 2), vocabulary=VOCABULARY) -> StatusContext:
    placed = frozenset(units)
    return StatusContext(
        vocabulary=vocabulary,
        trigger_ids=frozenset(triggers),
        unit_exists=placed.__contains__,
        requirements={"conditions": {}, "effects": {}} if requirements is None else requirements,
    )


def _effects_table(**rows: Requirement) -> dict:
    return {"conditions": {}, "effects": rows}


def _status(kind, entry, context=None):
    return trigger_status.entry_status(kind, entry, context or _context())


# -- is_set ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        (trigger_fields.INT, -1, False),
        (trigger_fields.INT, 0, True),
        (trigger_fields.INT, 7, True),
        (trigger_fields.ENUM, -1, False),
        (trigger_fields.ENUM, 0, True),
        (trigger_fields.REFERENCE, -1, False),
        (trigger_fields.REFERENCE, 4, True),
        (trigger_fields.FLOAT, -1.0, False),
        (trigger_fields.FLOAT, -1, False),
        (trigger_fields.FLOAT, 0.5, True),
        (trigger_fields.INT_LIST, [], False),
        (trigger_fields.INT_LIST, [-1], False),
        (trigger_fields.INT_LIST, [3], True),
        (trigger_fields.INT_LIST, [-1, 3], True),
        (trigger_fields.STR, "", False),
        (trigger_fields.STR, "   \n", False),
        (trigger_fields.STR, " hi ", True),
        (trigger_fields.BOOL, 0, True),
        (trigger_fields.BOOL, -1, True),
    ],
)
def test_is_set_per_kind(kind, value, expected) -> None:
    assert trigger_status.is_set(FieldSpec("f", kind), value) is expected


@pytest.mark.parametrize("kind", [trigger_fields.INT, trigger_fields.FLOAT, trigger_fields.STR,
                                  trigger_fields.INT_LIST, trigger_fields.ENUM, trigger_fields.BOOL])
def test_none_is_unset_for_every_kind(kind) -> None:
    """A field this version lacks reads None after depoison()."""
    assert trigger_status.is_set(FieldSpec("f", kind), None) is False


# -- universal checks ------------------------------------------------------------


def test_no_vocabulary_is_unchecked() -> None:
    context = _context(vocabulary=None)
    assert _status(EFFECT, _effect(_CREATE_OBJECT), context).status is Status.UNCHECKED


def test_a_clean_entry_with_no_table_row_is_ok() -> None:
    assert _status(EFFECT, _effect(_CREATE_OBJECT)) == trigger_status.EntryStatus(Status.OK)


@pytest.mark.parametrize(("kind", "entry", "reason"), [
    (EFFECT, SimpleNamespace(effect_type=0), "no effect type chosen"),
    (CONDITION, SimpleNamespace(condition_type=0), "no condition type chosen"),
])
def test_type_zero_none_is_a_problem(kind, entry, reason) -> None:
    assert _status(kind, entry) == trigger_status.EntryStatus(Status.PROBLEM, (reason,))


@pytest.mark.parametrize(("kind", "entry"), [
    (EFFECT, SimpleNamespace(effect_type=999)),
    (CONDITION, SimpleNamespace(condition_type=999)),
])
def test_an_unknown_type_id_is_unchecked_not_a_problem(kind, entry) -> None:
    """Nothing to check it against (user decision, 2026-10-07)."""
    assert _status(kind, entry) == trigger_status.UNCHECKED


def test_a_dangling_unit_reference_is_a_problem_and_a_placed_one_is_not() -> None:
    entry = _effect(_KILL_OBJECT, selected_object_ids=[5, 6])
    dangling = _status(EFFECT, entry, _context(units={5}))
    assert dangling.status is Status.PROBLEM
    assert dangling.reasons == ("unit 6 (selected object ids) is not placed on the map",)
    assert _status(EFFECT, entry, _context(units={5, 6})).status is Status.OK


@pytest.mark.parametrize(("kind", "entry"), [
    (EFFECT, _effect(_ACTIVATE_TRIGGER, trigger_id=9)),
    # Condition 79's trigger_id has presentation "": keyed on the name, not presentation.
    (CONDITION, _condition(_TRIGGER_ACTIVE, trigger_id=9)),
])
def test_a_dangling_trigger_id_is_a_problem(kind, entry) -> None:
    assert _status(kind, entry).reasons == ("trigger 9 does not exist",)
    entry.trigger_id = 2
    assert _status(kind, entry).status is Status.OK
    entry.trigger_id = -1
    assert _status(kind, entry).status is Status.OK


def test_the_trigger_id_check_is_keyed_on_the_referencing_types_only() -> None:
    """send_chat has no trigger_id; a stray attribute on another type is not read."""
    assert _status(EFFECT, _effect(_SEND_CHAT, trigger_id=9)).status is Status.OK


def test_an_incoherent_quantity_cluster_is_a_problem_with_its_own_text() -> None:
    broken = _effect(_MODIFY_ATTRIBUTE, object_attributes=_ARMOR, armour_attack_quantity=None,
                     armour_attack_class=None, quantity=45)
    broken._armour_attack_quantity = None
    broken._armour_attack_class = None
    text = trigger_fields.cluster_incoherence(broken)
    assert text
    assert _status(EFFECT, broken).reasons == (text,)
    fine = _effect(_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS, quantity=45)
    fine._quantity = 45
    assert _status(EFFECT, fine).status is Status.OK


@pytest.mark.parametrize(("fields", "reason"), [
    ({"area_x1": -1, "area_y1": 5, "area_x2": 8, "area_y2": 9}, "area is only partly set"),
    ({"location_x": 3, "location_y": -1}, "location is only partly set"),
    # None reads as -1, trigger_geometry's rule.
    ({"location_x": 3, "location_y": None}, "location is only partly set"),
])
def test_a_half_set_coordinate_group_is_a_problem(fields, reason) -> None:
    assert _status(EFFECT, _effect(_TASK_OBJECT, **fields)).reasons == (reason,)


@pytest.mark.parametrize("fields", [
    {},
    {"area_x1": 0, "area_y1": 5, "area_x2": 8, "area_y2": 9},
    {"location_x": 0, "location_y": 0},
])
def test_whole_or_empty_coordinate_groups_are_fine(fields) -> None:
    assert _status(EFFECT, _effect(_TASK_OBJECT, **fields)).status is Status.OK


# -- table semantics -------------------------------------------------------------


def test_all_tokens_must_each_be_set() -> None:
    context = _context(_effects_table(send_chat=Requirement(all=("message", "string_id"))))
    assert _status(EFFECT, _effect(_SEND_CHAT, message="  "), context).reasons == (
        "missing message", "missing string id")
    assert _status(EFFECT, _effect(_SEND_CHAT, message="hi", string_id=4), context).status is Status.OK


def test_one_any_group_must_be_fully_set() -> None:
    context = _context(_effects_table(send_chat=Requirement(any=(("message",), ("string_id", "sound_name")))))
    assert _status(EFFECT, _effect(_SEND_CHAT), context).reasons == ("needs message or string id + sound name",)
    # Half of the second group is not enough.
    assert _status(EFFECT, _effect(_SEND_CHAT, string_id=4), context).status is Status.PROBLEM
    assert _status(EFFECT, _effect(_SEND_CHAT, string_id=4, sound_name="x"), context).status is Status.OK
    assert _status(EFFECT, _effect(_SEND_CHAT, message="hi"), context).status is Status.OK


def test_a_token_this_version_does_not_list_is_skipped_not_failed() -> None:
    context = _context(_effects_table(send_chat=Requirement(all=("wall_x1", "location"))))
    assert _status(EFFECT, _effect(_SEND_CHAT), context).status is Status.OK


def test_an_any_group_of_skipped_tokens_is_dropped_never_passing_vacuously() -> None:
    context = _context(_effects_table(send_chat=Requirement(any=(("wall_x1",), ("message",)))))
    assert _status(EFFECT, _effect(_SEND_CHAT), context).reasons == ("needs message",)


def test_a_row_whose_every_any_group_is_dropped_has_no_any_constraint() -> None:
    context = _context(_effects_table(send_chat=Requirement(any=(("wall_x1",), ("area",)))))
    assert _status(EFFECT, _effect(_SEND_CHAT), context).status is Status.OK


def test_area_pseudo_field_needs_all_four_corners() -> None:
    context = _context(_effects_table(kill_object=Requirement(all=("area",))))
    assert _status(EFFECT, _effect(_KILL_OBJECT), context).reasons == ("missing area",)
    full = _effect(_KILL_OBJECT, area_x1=0, area_y1=0, area_x2=3, area_y2=3)
    assert _status(EFFECT, full, context).status is Status.OK


def test_location_pseudo_field_coordinates() -> None:
    context = _context(_effects_table(create_object=Requirement(all=("location",))))
    assert _status(EFFECT, _effect(_CREATE_OBJECT), context).reasons == ("missing location",)
    assert _status(EFFECT, _effect(_CREATE_OBJECT, location_x=0, location_y=4), context).status is Status.OK


def test_location_pseudo_field_counts_a_resolving_object_reference_only() -> None:
    table = _effects_table(task_object=Requirement(all=("location",)))
    by_reference = _effect(_TASK_OBJECT, location_object_reference=42)
    assert _status(EFFECT, by_reference, _context(table, units={42})).status is Status.OK
    # A stale reference must not read as a set location.
    stale = _status(EFFECT, by_reference, _context(table, units=()))
    assert stale.status is Status.PROBLEM
    assert "missing location" in stale.reasons


def test_location_reference_does_not_count_on_a_type_that_does_not_list_it() -> None:
    """create_object has no location_object_reference: a stray attribute is not read."""
    context = _context(_effects_table(create_object=Requirement(all=("location",))), units={42})
    entry = _effect(_CREATE_OBJECT, location_object_reference=42)
    assert _status(EFFECT, entry, context).reasons == ("missing location",)


def test_values_are_read_through_the_spec_attribute_and_a_raising_getter_is_unset() -> None:
    """quantity_float aliases to quantity; the broken-pair quantity getter raises."""
    context = _context(_effects_table(modify_attribute=Requirement(all=("quantity_float",))))
    ok = _effect(_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS, quantity=2)
    ok._quantity = 2
    assert _status(EFFECT, ok, context).status is Status.OK

    class Raising(SimpleNamespace):
        @property
        def quantity(self):
            raise ValueError("broken armour pair")

    raising = Raising(**vars(_effect(_MODIFY_ATTRIBUTE, object_attributes=_HIT_POINTS)))
    raising._quantity = 2
    assert _status(EFFECT, raising, context).reasons == ("missing quantity float",)


def test_universal_and_table_reasons_both_show() -> None:
    context = _context(_effects_table(task_object=Requirement(all=("object_list_unit_id",))))
    entry = _effect(_TASK_OBJECT, selected_object_ids=[7])
    assert _status(EFFECT, entry, context).reasons == (
        "unit 7 (selected object ids) is not placed on the map", "missing source player unit")


def test_a_missing_field_is_named_by_its_types_own_caption() -> None:
    """GH #212: the reason reads as the form row does, so Replace Object's
    second unit is the target player's."""
    context = _context(_effects_table(replace_object=Requirement(all=("object_list_unit_id", "object_list_unit_id_2"))))
    unset = _effect(_REPLACE_OBJECT, object_list_unit_id=4, object_list_unit_id_2=-1)
    assert _status(EFFECT, unset, context).reasons == ("missing target player unit",)
    neither = _effect(_REPLACE_OBJECT, object_list_unit_id=-1, object_list_unit_id_2=-1)
    assert _status(EFFECT, neither, context).reasons == ("missing source player unit", "missing target player unit")
    assert _status(EFFECT, _effect(_REPLACE_OBJECT, object_list_unit_id=4, object_list_unit_id_2=5), context).status is Status.OK


def test_an_any_group_names_fields_by_caption_and_pseudo_fields_by_token() -> None:
    table = _effects_table(kill_object=Requirement(any=(("object_list_unit_id", "area"), ("selected_object_ids",))))
    assert _status(EFFECT, _effect(_KILL_OBJECT), _context(table)).reasons == (
        "needs source player unit + area or selected object ids",)


def test_token_state_matches_the_table_semantics_for_the_census() -> None:
    context = _context(units={42})
    assert trigger_status.token_state(EFFECT, "location", _effect(_TASK_OBJECT, location_object_reference=42), context)
    assert trigger_status.token_state(EFFECT, "area", _effect(_TASK_OBJECT), context) is False
    assert trigger_status.token_state(EFFECT, "area", _effect(_SEND_CHAT), context) is None
    assert trigger_status.token_state(EFFECT, "message", _effect(_SEND_CHAT, message="x"), context) is True
    assert trigger_status.token_state(EFFECT, "area", _effect(_TASK_OBJECT), _context(vocabulary=None)) is None


def test_the_committed_table_loads_and_parses() -> None:
    table = trigger_status.load_table()
    assert set(table) == {"conditions", "effects"}
    for rows in table.values():
        assert all(isinstance(row, Requirement) for row in rows.values())


def test_parse_table_reads_all_any_and_why() -> None:
    raw = {"effects": {"send_chat": {"any": [["message"], ["string_id"]], "why": "text"}}}
    assert trigger_status.parse_table(raw) == {
        "conditions": {},
        "effects": {"send_chat": Requirement(any=(("message",), ("string_id",)), why="text")},
    }


def test_context_without_requirements_uses_the_committed_table(monkeypatch) -> None:
    monkeypatch.setattr(trigger_status, "load_table",
                        lambda: _effects_table(send_chat=Requirement(all=("message",))))
    context = StatusContext(vocabulary=VOCABULARY)
    assert trigger_status.entry_status(EFFECT, _effect(_SEND_CHAT), context).reasons == ("missing message",)


# -- rollups ---------------------------------------------------------------------


def _trigger(
    name="Spawn", conditions=(), effects=(), enabled=True, display_as_objective=0, header=0, display_on_screen=0
):
    return SimpleNamespace(name=name, conditions=list(conditions), effects=list(effects), enabled=enabled,
                           display_as_objective=display_as_objective, header=header,
                           display_on_screen=display_on_screen)


def test_a_divider_is_none_whatever_its_entries_say() -> None:
    trigger = _trigger("--- Setup ---", effects=[SimpleNamespace(effect_type=0)])
    assert trigger_status.evaluate_trigger(trigger, _context()).trigger.status is Status.NONE
    assert trigger_status.evaluate_trigger(_trigger("-----"), _context()).trigger.status is Status.NONE


def test_a_divider_entry_rows_are_still_evaluated() -> None:
    trigger = _trigger("--- Setup ---", effects=[SimpleNamespace(effect_type=0)])
    assert trigger_status.evaluate_trigger(trigger, _context()).effects[0].status is Status.PROBLEM


def test_zero_effects_is_a_problem() -> None:
    result = trigger_status.evaluate_trigger(_trigger(conditions=[_condition(_OWN_OBJECTS)]), _context())
    assert result.trigger == trigger_status.EntryStatus(Status.PROBLEM, ("has no effects",))
    assert trigger_status.needs_effects(_trigger()) is True


@pytest.mark.parametrize("flag", ["display_as_objective", "header", "display_on_screen"])
def test_an_objective_header_or_on_screen_display_may_have_no_effects(flag) -> None:
    """User decisions, 2026-10-07: such a trigger may exist only to show text."""
    trigger = _trigger(conditions=[_condition(_OWN_OBJECTS)], **{flag: 1})
    assert trigger_status.needs_effects(trigger) is False
    cleared = _trigger(conditions=[_condition(_OWN_OBJECTS)], **{flag: 0})
    assert trigger_status.evaluate_trigger(cleared, _context()).trigger.reasons == ("has no effects",)
    assert trigger_status.evaluate_trigger(trigger, _context()).trigger == trigger_status.EntryStatus(Status.OK)
    # The exemption is for the missing effects only; a broken child still shows.
    broken = _trigger(conditions=[SimpleNamespace(condition_type=0)], **{flag: 1})
    assert trigger_status.evaluate_trigger(broken, _context()).trigger == trigger_status.EntryStatus(
        Status.PROBLEM, ("condition 1 (none): no condition type chosen",))


def test_a_trigger_without_the_objective_flags_attributes_still_needs_effects() -> None:
    bare = SimpleNamespace(name="Spawn", conditions=[], effects=[])
    assert trigger_status.needs_effects(bare) is True
    assert trigger_status.needs_effects(_trigger("--- Setup ---")) is False


def test_an_unknown_type_rolls_up_unchecked_unless_another_entry_is_a_problem() -> None:
    trigger = _trigger(effects=[_effect(_SEND_CHAT), SimpleNamespace(effect_type=999)])
    result = trigger_status.evaluate_trigger(trigger, _context())
    assert [s.status for s in result.effects] == [Status.OK, Status.UNCHECKED]
    assert result.trigger == trigger_status.UNCHECKED
    trigger.effects.append(SimpleNamespace(effect_type=0))
    assert trigger_status.evaluate_trigger(trigger, _context()).trigger == trigger_status.EntryStatus(
        Status.PROBLEM, ("effect 3 (none): no effect type chosen",))


def test_a_child_problem_propagates_with_its_prefix() -> None:
    context = _context(_effects_table(create_object=Requirement(all=("location",))))
    trigger = _trigger(conditions=[_condition(_OWN_OBJECTS)],
                       effects=[_effect(_SEND_CHAT), _effect(_CREATE_OBJECT)])
    result = trigger_status.evaluate_trigger(trigger, context)
    assert [s.status for s in result.effects] == [Status.OK, Status.PROBLEM]
    assert result.conditions == (trigger_status.EntryStatus(Status.OK),)
    assert result.trigger == trigger_status.EntryStatus(
        Status.PROBLEM, ("effect 2 (create object): missing location",))


def test_rollup_reasons_are_numbered_and_listed_by_display_slot() -> None:
    """GH #213: the entry tree lists conditions and effects in display order,
    so "effect 1" is its first row, whatever list index that entry has."""
    context = _context(_effects_table(create_object=Requirement(all=("location",))))
    trigger = _trigger(
        conditions=[SimpleNamespace(condition_type=0), _condition(_OWN_OBJECTS)],
        effects=[SimpleNamespace(effect_type=0), _effect(_SEND_CHAT), _effect(_CREATE_OBJECT)],
    )
    trigger.condition_order = [1, 0]
    trigger.effect_order = [2, 1, 0]
    result = trigger_status.evaluate_trigger(trigger, context)
    # The per-entry statuses stay keyed by list index.
    assert [s.status for s in result.conditions] == [Status.PROBLEM, Status.OK]
    assert [s.status for s in result.effects] == [Status.PROBLEM, Status.OK, Status.PROBLEM]
    assert result.trigger.reasons == (
        "condition 2 (none): no condition type chosen",
        "effect 1 (create object): missing location",
        "effect 3 (none): no effect type chosen",
    )

    # An array that is not a permutation of the entries reads as list order.
    trigger.effect_order = [0, 0, 1]
    assert trigger_status.evaluate_trigger(trigger, context).trigger.reasons[1:] == (
        "effect 1 (none): no effect type chosen",
        "effect 3 (create object): missing location",
    )


def test_a_clean_trigger_is_ok_and_enabled_is_ignored() -> None:
    trigger = _trigger(effects=[_effect(_SEND_CHAT)], enabled=False)
    assert trigger_status.evaluate_trigger(trigger, _context()).trigger.status is Status.OK


def test_unchecked_children_make_an_unchecked_rollup_without_problems() -> None:
    unchecked = trigger_status.UNCHECKED
    ok = trigger_status.EntryStatus(Status.OK)
    trigger = _trigger(effects=[object()])
    assert trigger_status.trigger_status(trigger, [("effect 1 (x)", unchecked), ("effect 2 (y)", ok)]) == unchecked
    problem = trigger_status.EntryStatus(Status.PROBLEM, ("bad",))
    assert trigger_status.trigger_status(trigger, [("effect 1 (x)", unchecked), ("effect 2 (y)", problem)]) == (
        trigger_status.EntryStatus(Status.PROBLEM, ("effect 2 (y): bad",)))


def test_no_vocabulary_leaves_everything_unchecked_but_dividers() -> None:
    context = _context(vocabulary=None)
    plain = trigger_status.evaluate_trigger(_trigger(conditions=[object()]), context)
    assert plain == trigger_status.TriggerStatuses(trigger_status.UNCHECKED, (trigger_status.UNCHECKED,), ())
    divider = trigger_status.evaluate_trigger(_trigger("== Intro =="), context)
    assert divider.trigger.status is Status.NONE


def test_group_status_of_nothing_is_ok() -> None:
    assert trigger_status.group_status([]) == trigger_status.EntryStatus(Status.OK)


def test_entry_label_is_one_based_with_the_type_name() -> None:
    assert trigger_status.entry_label(EFFECT, 0, _effect(_CREATE_OBJECT), VOCABULARY) == "effect 1 (create object)"
    assert trigger_status.entry_label(CONDITION, 4, SimpleNamespace(condition_type=999), VOCABULARY) == (
        "condition 5 (type 999)")


def test_type_zero_effect_is_labelled_none() -> None:
    assert trigger_status.entry_label(EFFECT, 0, SimpleNamespace(effect_type=0), VOCABULARY) == "effect 1 (none)"


def test_display_instructions_with_defaults_is_ok_under_the_universal_rules() -> None:
    """A fixed-slot effect never trips the cluster check, whatever its quantity holds."""
    entry = _effect(_DISPLAY_INSTRUCTIONS, quantity=None)
    assert _status(EFFECT, entry).status is Status.OK


# -- per-type plans (TASK-215) ---------------------------------------------------


def _counting(monkeypatch, module, name: str) -> list:
    calls: list = []
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def test_one_context_derives_each_types_plan_once(monkeypatch) -> None:
    """Anything derived from the type alone is built once per type and context:
    a pass visits 160k entries on the largest corpus file."""
    from descape import trigger_geometry, unit_references

    coordinate = _counting(monkeypatch, trigger_geometry, "coordinate_fields")
    unit_fields = _counting(monkeypatch, unit_references, "unit_reference_fields")
    specs = _counting(monkeypatch, trigger_fields, "field_specs")
    context = _context(_effects_table(kill_object=Requirement(all=("selected_object_ids",))), units={5})
    trigger = _trigger(effects=[_effect(_KILL_OBJECT, selected_object_ids=[5]) for _ in range(50)])
    for _ in range(2):
        result = trigger_status.evaluate_trigger(trigger, context)
        assert [s.status for s in result.effects] == [Status.OK] * 50
    assert (len(coordinate), len(unit_fields), len(specs)) == (1, 1, 1)


def test_an_all_ok_trigger_never_builds_entry_labels(monkeypatch) -> None:
    labels = _counting(monkeypatch, trigger_status, "entry_label")
    trigger = _trigger(conditions=[_condition(_OWN_OBJECTS)], effects=[_effect(_SEND_CHAT), _effect(_CREATE_OBJECT)])
    assert trigger_status.evaluate_trigger(trigger, _context()).trigger is trigger_status.OK
    assert labels == []


def test_one_problem_entry_still_gets_its_label_in_display_slot_order(monkeypatch) -> None:
    context = _context(_effects_table(create_object=Requirement(all=("location",))))
    trigger = _trigger(effects=[_effect(_SEND_CHAT), _effect(_SEND_CHAT), _effect(_CREATE_OBJECT)])
    trigger.effect_order = [1, 2, 0]
    result = trigger_status.evaluate_trigger(trigger, context)
    assert result.trigger == trigger_status.EntryStatus(Status.PROBLEM, ("effect 2 (create object): missing location",))


def test_two_contexts_with_different_requirements_do_not_share_a_verdict() -> None:
    import dataclasses

    entry = _effect(_CREATE_OBJECT)
    strict = _context(_effects_table(create_object=Requirement(all=("location",))))
    lenient = _context()
    assert _status(EFFECT, entry, strict).status is Status.PROBLEM
    assert _status(EFFECT, entry, lenient).status is Status.OK
    assert _status(EFFECT, entry, strict).status is Status.PROBLEM
    # A copy gets its own plans, never the original's.
    copied = dataclasses.replace(strict, requirements=_effects_table())
    assert _status(EFFECT, entry, copied).status is Status.OK
    assert _status(EFFECT, entry, strict).status is Status.PROBLEM


def _with_references(trigger, context=None):
    statuses, unit_refs, targets = trigger_status.evaluate_with_references(trigger, context or _context())
    assert statuses == trigger_status.evaluate_trigger(trigger, context or _context())
    return statuses, unit_refs, targets


def test_evaluate_with_references_without_a_vocabulary_is_false_and_empty() -> None:
    trigger = _trigger(effects=[_effect(_KILL_OBJECT, selected_object_ids=[5]), _effect(_ACTIVATE_TRIGGER, trigger_id=2)])
    statuses, unit_refs, targets = _with_references(trigger, _context(vocabulary=None))
    assert statuses.trigger is trigger_status.UNCHECKED
    assert (unit_refs, targets) == (False, ())


def test_evaluate_with_references_a_set_unit_id_counts_placed_or_not() -> None:
    trigger = _trigger(effects=[_effect(_SEND_CHAT), _effect(_KILL_OBJECT, selected_object_ids=[-1, 5])])
    assert _with_references(trigger, _context(units={5}))[1:] == (True, ())
    assert _with_references(trigger, _context(units=()))[1:] == (True, ())


def test_evaluate_with_references_an_unset_only_unit_field_does_not_count() -> None:
    trigger = _trigger(effects=[_effect(_KILL_OBJECT, selected_object_ids=[-1])])
    assert _with_references(trigger)[1:] == (False, ())


def test_evaluate_with_references_collects_every_trigger_id_in_stored_order() -> None:
    """Valid, dangling and UNSET alike, conditions then effects: a count change can flip any of them."""
    trigger = _trigger(
        conditions=[_condition(_TRIGGER_ACTIVE, trigger_id=9)],
        effects=[_effect(_ACTIVATE_TRIGGER, trigger_id=2), _effect(_SEND_CHAT), _effect(_ACTIVATE_TRIGGER, trigger_id=-1)],
    )
    statuses, unit_refs, targets = _with_references(trigger)
    assert (unit_refs, targets) == (False, (9, 2, -1))
    assert statuses.conditions[0].reasons == ("trigger 9 does not exist",)
    assert [s.status for s in statuses.effects] == [Status.OK] * 3


def test_evaluate_with_references_keys_targets_on_the_type_even_when_the_vocabulary_lacks_it() -> None:
    """An unknown type id is UNCHECKED, but a trigger-id type still names its target."""
    import dataclasses

    stripped = dataclasses.replace(
        VOCABULARY, effects={k: v for k, v in VOCABULARY.effects.items() if k != _ACTIVATE_TRIGGER}
    )
    trigger = _trigger(effects=[_effect(_ACTIVATE_TRIGGER, trigger_id=9), SimpleNamespace(effect_type=999, trigger_id=4)])
    statuses, unit_refs, targets = _with_references(trigger, _context(vocabulary=stripped))
    assert [s.status for s in statuses.effects] == [Status.UNCHECKED, Status.UNCHECKED]
    assert (unit_refs, targets) == (False, (9,))
