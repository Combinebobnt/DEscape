"""Trigger panel: retyping a condition or effect through the type picker."""

from __future__ import annotations

from pathlib import Path

import pytest
from trigger_panel_helpers import (
    _ARCHER,
    _CASTLE,
    _REPLACE_OBJECT,
    TRIGGER_FIXTURE,
    _effect_type_id,
    _effects0,
    _facet_window,
    _group,
    _no_install_env,  # noqa: F401 -- autouse fixture
    _pick,
    _triggers_window,
)

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


# -- retyping an existing condition or effect (GH #37) -----------------------


def _select_first_effect(panel) -> int:
    """Select the first effect the fixture carries, and return its trigger
    index. Asserts rather than skips, for _select_first_condition's reason."""
    for i in range(panel.tree.topLevelItemCount()):
        panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
        effects = _group(panel, "effect")
        if effects.childCount():
            panel.entry_tree.setCurrentItem(effects.child(0))
            return i
    raise AssertionError("the fixture carries no effects, so retype tests cannot run")


def _effects_of(window, trigger_index: int):
    manager = window.trigger_panel._manager()
    return manager.triggers[trigger_index].effects


def test_the_retype_picker_offers_one_kind_and_opens_on_the_current_type() -> None:
    """Cross-kind retyping is not offered -- conditions and effects are
    separate lists -- and the picker opens where the user already is."""
    from PyQt5.QtCore import Qt

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        current_type = _effects_of(window, trigger_index)[0].effect_type

        panel._request_entry_op("retype")
        assert panel.detail_stack.currentIndex() == 1
        assert panel.picker_tree.topLevelItemCount() == 1
        assert "Effects" in panel.picker_tree.topLevelItem(0).text(0)
        assert panel.picker_add_button.text() == "Change"

        picked = panel.picker_tree.currentItem()
        assert picked is not None, "the retype picker opened with nothing selected"
        assert picked.data(0, Qt.UserRole) == ("effect", current_type)
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_opening_and_cancelling_the_retype_picker_records_nothing(tmp_path: Path) -> None:
    from descape.scenario_write import write_scenario

    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_effect(panel)
        panel._request_entry_op("retype")
        _pick(panel, "effect", "send chat")
        panel.picker_filter.setText("chat")
        panel._close_picker()

        assert panel.detail_stack.currentIndex() == 0
        assert panel._picker_entry_ref is None, "the entry latch outlived the picker"
        assert window.trigger_edits is None, "opening the picker must not build an edit model"
        assert not window.edit_history.is_dirty

        out = tmp_path / "cancelled.aoe2scenario"
        write_scenario(window.scenario, out, triggers=window.trigger_edits)
        assert out.read_bytes() == TRIGGER_FIXTURE.read_bytes()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_picking_the_type_the_entry_already_has_records_nothing() -> None:
    """commit_trigger_edit() pushes unconditionally, so _accept_pick() is the
    only place a no-op retype can be stopped."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        _select_first_effect(panel)
        panel._request_entry_op("retype")
        # The picker already opened on the current type, so accept as-is.
        panel._accept_pick()

        assert panel.detail_stack.currentIndex() == 0
        assert window.trigger_edits is None
        assert not window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_the_retype_targets_the_entry_latched_at_open() -> None:
    """The detail tree stays live behind the picker in Add mode; in Change mode
    a selection change in between must not move the target."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        effects = _effects_of(window, trigger_index)
        if len(effects) < 2:
            for i in range(panel.tree.topLevelItemCount()):
                panel.tree.setCurrentItem(panel.tree.topLevelItem(i))
                if _group(panel, "effect").childCount() >= 2:
                    trigger_index = i
                    break
            effects = _effects_of(window, trigger_index)
        assert len(effects) >= 2, "the fixture has no trigger with two effects"
        panel.entry_tree.setCurrentItem(_group(panel, "effect").child(0))

        before = [e.effect_type for e in effects]
        panel._request_entry_op("retype")
        # Move the detail selection to the *second* effect while the picker is
        # up. The tree is hidden there, but a programmatic selection still
        # fires _on_entry_selected, and the latch is what has to survive it --
        # re-reading the selection at accept time would retype the wrong entry.
        panel.entry_tree.setCurrentItem(_group(panel, "effect").child(1))
        _pick(panel, "effect", "send chat")
        panel._accept_pick()

        after = [e.effect_type for e in window.trigger_edits.manager().triggers[trigger_index].effects]
        assert after[0] != before[0], "the latched entry was not retyped"
        assert after[1:] == before[1:], "a retype moved off the entry latched at open"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_retype_keeps_the_entry_in_place_and_re_anchors_its_uuid() -> None:
    """Build-then-replace, not append: the index, the list length and the
    display-order array all have to come out unchanged, and UuidList.__setitem__
    is what re-anchors the fresh object to the document."""
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        trigger_index = _select_first_effect(panel)
        effects = _effects_of(window, trigger_index)
        before_len = len(effects)
        before_order = list(
            panel._manager().triggers[trigger_index].effect_order
        )

        window.entry_structural_edit("retype", trigger_index, "effect", 0, 3)

        trigger = window.trigger_edits.manager().triggers[trigger_index]
        assert len(trigger.effects) == before_len, "a retype changed the list length"
        assert trigger.effects[0].effect_type == 3
        assert list(trigger.effect_order) == before_order, "a retype rewrote the display order"
        assert trigger.effects[0]._uuid == trigger._uuid, "the fresh entry was never re-anchored"
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_retype_carries_the_shared_fields_and_reports_the_rest() -> None:
    window = _triggers_window()
    try:
        panel = window.trigger_panel
        # Trigger 0's effect is display_instructions, whose message/source_player
        # send_chat also lists, and whose display_time it does not.
        trigger_index = _select_first_effect(panel)
        effect = _effects_of(window, trigger_index)[0]
        effect.message = "carried"
        effect.source_player = 2
        effect.display_time = 15

        window.entry_structural_edit("retype", trigger_index, "effect", 0, 3)

        fresh = window.trigger_edits.manager().triggers[trigger_index].effects[0]
        assert fresh.effect_type == 3
        assert fresh.message == "carried"
        assert fresh.source_player == 2
        assert "display time" in window.status_log.toPlainText()
    finally:
        window.edit_history.mark_saved()
        window.close()


def test_a_retype_names_a_dropped_field_by_the_old_types_caption() -> None:
    """GH #212: Replace Object's second unit is "target player unit" on its
    form, so the line saying it did not carry over calls it that too."""
    window, indices = _facet_window(_REPLACE_OBJECT, {"object_list_unit_id": _ARCHER, "object_list_unit_id_2": _CASTLE})
    try:
        kill_object = _effect_type_id(window.trigger_panel, "kill_object")
        window.entry_structural_edit("retype", 0, "effect", indices[0], kill_object)

        fresh = _effects0(window)[indices[0]]
        assert fresh.effect_type == kill_object
        assert fresh.object_list_unit_id == _ARCHER, "the first unit is a field both types list"
        reports = [line for line in window.status_log.toPlainText().splitlines() if line.startswith("Changed effect to")]
        assert len(reports) == 1
        # facet2 drops too; only the unit field's wording is this test's.
        assert "did not carry over (target player unit" in reports[0]
        assert "object list unit id" not in reports[0]
    finally:
        window.edit_history.mark_saved()
        window.close()
