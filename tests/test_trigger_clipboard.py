"""trigger_clipboard (GH #27): copy_block() / paste_into() over a real
manager. Qt-free.

The fixture's trigger 2 ("Fixture: references") activates trigger 0 and
deactivates trigger 1, which is what every reference case here rides on.
"""

from __future__ import annotations

from pathlib import Path

from descape import trigger_clipboard
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.trigger_organize import is_divider

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _manager():
    manager = parse_triggers(load_map_and_units(FIXTURE))
    assert manager is not None
    return manager


def _refs(trigger) -> list[int]:
    return [ce.trigger_id for ce in trigger_clipboard.get_trigger_referencing_ce(trigger)]


def test_a_reference_inside_the_block_points_at_the_pasted_copy() -> None:
    manager = _manager()
    block = trigger_clipboard.copy_block(manager, [0, 2])
    pasted = trigger_clipboard.paste_into(manager, block).triggers
    assert [manager.triggers.index(t) for t in pasted] == [4, 5]
    # The copy of 2 activates the copy of 0 (4); 1 was outside and keeps 1.
    assert _refs(pasted[1]) == [4, 1]
    assert _refs(manager.triggers[2]) == [0, 1], "the source is untouched"


def test_an_outside_reference_follows_its_target_across_a_delete() -> None:
    """Ids renumber; object identity does not. Deleting trigger 0 after the
    copy moves trigger 1 to index 0, and the pasted reference follows it."""
    manager = _manager()
    block = trigger_clipboard.copy_block(manager, [2])
    target = manager.triggers[1]
    manager.remove_triggers([0])
    pasted = trigger_clipboard.paste_into(manager, block).triggers
    assert manager.triggers.index(target) == 0
    # Trigger 0 was outside too, and is gone: that link lands on -1.
    assert _refs(pasted[0]) == [-1, 0]


def test_a_paste_into_another_document_leaves_outside_links_unset() -> None:
    source = _manager()
    block = trigger_clipboard.copy_block(source, [2])
    destination = _manager()
    pasted = trigger_clipboard.paste_into(destination, block).triggers
    assert _refs(pasted[0]) == [-1, -1]


def test_pasted_triggers_carry_the_copy_suffix_and_a_pasted_divider_stops_being_one() -> None:
    """Accepted consequence (GH #27 decision): the suffix folds a pasted
    section header into the section before it. Pinned so it stays deliberate."""
    manager = _manager()
    manager.triggers[0].name = "--- Setup ---"
    assert is_divider(manager.triggers[0].name)
    pasted = trigger_clipboard.paste_into(manager, trigger_clipboard.copy_block(manager, [0])).triggers
    assert pasted[0].name == "--- Setup --- (copy)"
    assert not is_divider(pasted[0].name)


def test_the_block_follows_display_order_not_argument_order() -> None:
    manager = _manager()
    manager.trigger_display_order = [3, 2, 1, 0]
    block = trigger_clipboard.copy_block(manager, [0, 3, 1])
    assert [t.name for t in block.triggers] == ["Fixture: variable", "Fixture: armour split", "Fixture: setup"]


def test_the_clipboard_does_not_see_later_edits_to_its_sources() -> None:
    manager = _manager()
    block = trigger_clipboard.copy_block(manager, [0])
    manager.triggers[0].name = "renamed after the copy"
    assert block.triggers[0].name == "Fixture: setup"


def test_a_paste_after_deleting_every_trigger_returns_the_stored_objects() -> None:
    """Into an empty list import_triggers() returns copies that are not what
    manager.triggers holds; paste_into() must hand back the stored ones, with
    their inner lists stamped to this manager."""
    manager = _manager()
    block = trigger_clipboard.copy_block(manager, [0, 1])
    manager.remove_triggers(list(range(len(manager.triggers))))
    pasted = trigger_clipboard.paste_into(manager, block).triggers
    assert all(a is b for a, b in zip(pasted, manager.triggers, strict=True))
    for trigger in pasted:
        assert trigger._effects._uuid == manager._uuid
        assert trigger._conditions._uuid == manager._uuid


def test_a_cross_document_paste_restamps_the_inner_lists() -> None:
    source = _manager()
    destination = _manager()
    pasted = trigger_clipboard.paste_into(destination, trigger_clipboard.copy_block(source, [0])).triggers
    assert pasted[0]._effects._uuid == destination._uuid
    assert pasted[0]._conditions._uuid == destination._uuid


# -- GH #3: pasting into another document -------------------------------------

VERSION = "1.58"  # the fixture's and the blank template's


# The library's scenario store is weak, and new_effect/new_condition look the
# scenario up by uuid, so the sources' LoadedScenarios are kept alive here.
_KEEP_ALIVE: list = []


def _cross_source():
    """The fixture's manager, with placed-unit references on its "variable"
    trigger (3): a Task Object effect's selection and location object, and a
    Bring Object condition's unit. Trigger 3 already references the named
    variable 0."""
    loaded = load_map_and_units(FIXTURE)
    _KEEP_ALIVE.append(loaded)
    manager = parse_triggers(loaded)
    trigger = manager.triggers[3]
    trigger.new_effect.task_object(selected_object_ids=[7, 8], location_object_reference=9, source_player=1)
    trigger.new_condition.bring_object_to_area(unit_object=7, area_x1=1, area_y1=1, area_x2=2, area_y2=2)
    return manager


def _cross_block(manager, indices=(0, 2, 3)):
    return trigger_clipboard.copy_block(manager, list(indices), doc_id="A", scenario_version=VERSION)


def test_a_block_records_the_named_variables_it_references() -> None:
    block = _cross_block(_cross_source())
    assert block.variable_names == ((0, "fixture_var"),)
    assert _cross_block(_manager(), (0, 1)).variable_names == (), "no variable field, no name"


def test_a_cross_document_paste_into_a_blank_map_round_trips(tmp_path: Path) -> None:
    """GH #3's headline case, through the real write path: copy three triggers
    from A, paste them into a blank map B, add an effect to one afterwards,
    write B and reload it."""
    from descape.scenario_new import load_blank_scenario
    from descape.scenario_write import write_scenario
    from descape.trigger_model import TriggerEditModel

    source = _cross_source()
    block = _cross_block(source)
    source_effects = len(source.triggers[3].effects)

    blank = load_blank_scenario(120)
    assert blank.scenario_version == VERSION
    model = TriggerEditModel(blank)
    results = []
    model.structural_edit(lambda m: results.append(trigger_clipboard.paste_into(m, block, doc_id="B")))
    (result,) = results
    assert result.cross_document
    # 3 reference fields added above, plus the fixture's own 5 on trigger 3.
    assert (result.cleared_trigger_links, result.cleared_unit_refs) == (1, 8)
    assert (result.named_variables, result.variable_name_conflicts) == (1, 0)
    assert all(a is b for a, b in zip(result.triggers, model.manager().triggers, strict=True))

    # Added after the paste: it must land in B's sections, not A's.
    model.structural_edit(lambda m: m.triggers[2].new_effect.send_chat(source_player=1, message="after paste"), [2])
    out = tmp_path / "b.aoe2scenario"
    write_scenario(blank, out, triggers=model)

    reloaded_scenario = load_map_and_units(out)  # kept alive: the library's store is weak
    reloaded = parse_triggers(reloaded_scenario)
    assert reloaded is not None
    assert [t.name for t in reloaded.triggers] == [
        "Fixture: setup (copy)",
        "Fixture: references (copy)",
        "Fixture: variable (copy)",
    ]
    assert _refs(reloaded.triggers[1]) == [0, -1], "the internal link follows the copy, the outside one is unset"
    variable_trigger = reloaded.triggers[2]
    for task in (e for e in variable_trigger.effects if e.effect_type == 12):
        assert (list(task.selected_object_ids), task.location_object_reference) == ([], -1)
    bring = next(c for c in variable_trigger.conditions if c.condition_type == 1)
    assert bring.unit_object == -1
    assert (bring.area_x1, bring.area_x2) == (1, 2), "areas are kept verbatim"
    assert [(v.variable_id, v.name) for v in reloaded.variables] == [(0, "fixture_var")]
    assert variable_trigger.effects[-1].message == "after paste"
    assert len(source.triggers[3].effects) == source_effects, "the source document is untouched"


def test_a_destination_slot_already_named_differently_keeps_its_name() -> None:
    destination = _manager()
    destination.variables[0].name = "their_var"
    result = trigger_clipboard.paste_into(destination, _cross_block(_cross_source()), doc_id="B")
    assert (result.named_variables, result.variable_name_conflicts) == (0, 1)
    assert [(v.variable_id, v.name) for v in destination.variables] == [(0, "their_var")]
    assert "kept this scenario's name for 1 variable" in trigger_clipboard.paste_report(result)


def test_a_same_document_paste_keeps_unit_references() -> None:
    manager = _cross_source()
    result = trigger_clipboard.paste_into(manager, _cross_block(manager, (3,)), doc_id="A")
    assert not result.cross_document and result.cleared_unit_refs == 0
    # The last Task Object is the one added above; the fixture carries its own first.
    task = [e for e in result.triggers[0].effects if e.effect_type == 12][-1]
    assert (list(task.selected_object_ids), task.location_object_reference) == ([7, 8], 9)


def test_the_paste_report_names_what_was_cleared() -> None:
    report = trigger_clipboard.paste_report(
        trigger_clipboard.PasteResult([object()] * 12, True, 3, 5, 2, 0)
    )
    assert report == (
        "Pasted 12 triggers from another scenario: cleared 3 trigger links and 5 unit references, named 2 variables."
    )
    quiet = trigger_clipboard.PasteResult([object()], True)
    assert trigger_clipboard.paste_report(quiet) == "Pasted 1 trigger from another scenario; nothing needed clearing."
