"""Coverage for descape/units_panel.py (S4+S5 of the units-sidebar-catalog
work): the object catalog + owner combo, and the selected-unit(s) inspector,
standalone -- no ViewerWindow and no loaded scenario, matching
tests/test_players_panel.py's own convention.

Every assertion here is about the panel's own contract: what show_unit()/
show_selection_count()/clear() render, the sticky pending object const, and
that a programmatic populate never reports through on_unit_field while a
genuine edit does. The write path and the selection-reconciliation logic
stay on ViewerWindow and are covered by tests/test_unit_selection_viewer.py
and tests/test_unit_edit_viewer.py instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QSpinBox

from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_pick import UnitEntry
from descape.units_panel import UnitsPanel

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_TREE_CONST = 349  # a real GAIA doodad -- rotation is a variant index, not an angle
_ARCHER_CONST = 4  # a real creatable unit -- rotation is a genuine angle
_TOWER_CONST = 79  # Watch Tower -- a real garrison host, 5 places
_GATE_CONST = 64  # stone gate, ne orientation, angle_count 1
_WALL_CONST = 117  # stone wall -- VARIANT, not rotatable


@dataclass
class SyntheticUnit:
    x: float
    y: float
    unit_const: int
    reference_id: int
    rotation: float = 0.0
    z: float = 0.0
    garrisoned_in_id: int = -1


def _entry(player_id=1, **kwargs) -> UnitEntry:
    defaults = {"x": 4.5, "y": 8.5, "unit_const": _ARCHER_CONST, "reference_id": 101}
    defaults.update(kwargs)
    unit = SyntheticUnit(**defaults)
    return UnitEntry(player_id=player_id, unit=unit, own_x=int(unit.x), own_y=int(unit.y), order=0)


def _panel(**kwargs):
    conftest.ensure_qapp()
    return UnitsPanel(**kwargs)


def test_starts_with_no_unit_selected() -> None:
    panel = _panel()
    assert panel.unit_inspector_empty.isVisibleTo(panel)
    assert not panel.unit_inspector_grid.isVisibleTo(panel)


def test_show_unit_populates_the_fields_and_hides_the_empty_label() -> None:
    panel = _panel()
    entry = _entry()
    panel.show_unit(entry)
    assert not panel.unit_inspector_empty.isVisibleTo(panel)
    assert panel.unit_inspector_grid.isVisibleTo(panel)
    assert panel.unit_field_labels["reference_id"].text() == str(entry.unit.reference_id)
    assert panel.unit_field_labels["unit_const"].text() == str(_ARCHER_CONST)
    assert panel.unit_field_editors["player"].currentData() == 1
    assert panel.unit_field_editors["x"].value() == pytest.approx(entry.unit.x)


def test_show_unit_none_blanks_and_hides_the_grid() -> None:
    panel = _panel()
    panel.show_unit(_entry())
    panel.show_unit(None)
    assert panel.unit_inspector_empty.isVisibleTo(panel)
    assert not panel.unit_inspector_grid.isVisibleTo(panel)
    assert panel.unit_field_labels["reference_id"].text() == ""


def test_clear_is_equivalent_to_show_unit_none() -> None:
    panel = _panel()
    panel.show_unit(_entry())
    panel.clear()
    assert panel.unit_inspector_empty.isVisibleTo(panel)
    assert not panel.unit_inspector_grid.isVisibleTo(panel)


def test_gaia_owner_is_labelled_gaia_not_player_0() -> None:
    panel = _panel()
    panel.show_unit(_entry(player_id=GAIA_PLAYER_ID))
    assert panel.unit_field_editors["player"].currentData() == GAIA_PLAYER_ID


def test_show_selection_count_renders_the_count_and_hides_the_grid() -> None:
    panel = _panel()
    panel.show_unit(_entry())
    panel.show_selection_count(3)
    assert panel.unit_inspector_empty.text() == "3 units selected"
    assert panel.unit_inspector_empty.isVisibleTo(panel)
    assert not panel.unit_inspector_grid.isVisibleTo(panel)


def test_show_selection_count_zero_renders_no_unit_selected() -> None:
    panel = _panel()
    panel.show_selection_count(0)
    assert panel.unit_inspector_empty.text() == "No unit selected"


def test_rotation_editor_visible_for_an_angle_const_hidden_for_a_variant_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    assert panel.unit_field_editors["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert not panel.unit_field_labels["rotation"].isVisibleTo(panel.unit_inspector_grid)

    # A wall, not a tree: a tree's variant is editable since GH #123.
    panel.show_unit(_entry(unit_const=_WALL_CONST, rotation=37))
    assert not panel.unit_field_editors["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_field_labels["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_field_labels["rotation"].text() == "37"


def test_a_gate_shows_its_orientation_index_never_its_stored_rotation() -> None:
    """GH #61: a gate's orientation lives in its const, so the field shows the
    const's index in Rotate's order, never the junk sentinel 7.0 as a facing."""
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=88, rotation=7.0))  # stone gate, se
    spin = panel.unit_field_editors["rotation"]
    assert spin.isVisibleTo(panel.unit_inspector_grid)
    assert not panel.unit_field_labels["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert (spin.minimum(), spin.maximum(), spin.value()) == (0, 3, 2)
    assert spin.toolTip().startswith("Orientation 2 of 4: NW to SE (const 88).")
    assert not panel.unit_rotation_note.isVisibleTo(panel)
    assert received == []


def test_a_gate_orientation_wheel_step_wraps_like_rotate() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=667))  # stone gate, n: the last in Rotate's order
    panel.unit_field_editors["rotation"].stepBy(1)
    assert received == [("rotation", 0)]


def test_a_group_of_gates_edits_together_and_a_gate_with_a_wall_does_not() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_GATE_CONST), _entry(unit_const=88)])
    spin = panel.unit_field_editors["rotation"]
    assert spin.isVisibleTo(panel.unit_inspector_grid)
    assert spin.text() == "(mixed)"
    assert spin.toolTip().startswith("Orientation for every selected gate.")
    assert not panel.unit_rotation_note.isVisibleTo(panel)
    panel.show_group([_entry(unit_const=_GATE_CONST), _entry(unit_const=_WALL_CONST)])
    assert not spin.isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_field_labels["rotation"].text() == "(n/a)"


def test_gate_orientation_names_match_every_gate_graphics_direction_token() -> None:
    """The tooltip's names are pinned to the data, not hand-trusted."""
    import json
    import re
    from pathlib import Path

    from descape import gate_orientation, units_panel

    graphics = json.loads((Path(units_panel.__file__).parent / "unit_graphic_map.json").read_text())
    checked = 0
    for group in gate_orientation.groups().values():
        for index, const in enumerate(group):
            match = re.search(r"_(ne|se|e|n)_", graphics["graphics"][str(const)]["file_name"])
            if match:
                # The game's own names: "Gate Southwest To Northeast" is b_..._ne_..., etc.
                expected = {"ne": "SW to NE", "e": "W to E", "se": "NW to SE", "n": "N to S"}[match.group(1)]
                assert units_panel._GATE_ORIENTATION_NAMES[index] == expected, const
                checked += 1
    assert checked == 48


def test_stats_block_shows_all_five_rows_for_a_full_combat_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    assert panel.unit_stats_header.isVisibleTo(panel)
    for field_id, expected in (
        ("hp", "30"), ("attack", "4"), ("melee_armour", "0"), ("pierce_armour", "0"), ("range", "4"),
    ):
        caption, value = panel.unit_stat_rows[field_id]
        assert caption.isVisibleTo(panel)
        assert value.isVisibleTo(panel)
        assert value.text() == expected


def test_stats_block_hides_combat_rows_for_an_hp_only_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TREE_CONST))
    assert panel.unit_stats_header.isVisibleTo(panel)
    _hp_caption, hp_value = panel.unit_stat_rows["hp"]
    assert hp_value.isVisibleTo(panel)
    assert hp_value.text() == "20"
    for field_id in ("attack", "melee_armour", "pierce_armour", "range"):
        caption, value = panel.unit_stat_rows[field_id]
        assert not caption.isVisibleTo(panel)
        assert not value.isVisibleTo(panel)


def test_stats_block_hides_entirely_for_an_unknown_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=999999))
    assert not panel.unit_stats_header.isVisibleTo(panel)
    assert not panel.unit_stats_grid.isVisibleTo(panel)
    assert not panel.unit_stats_note.isVisibleTo(panel)


def test_stats_block_hides_and_blanks_on_show_unit_none() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    panel.show_unit(None)
    assert not panel.unit_stats_header.isVisibleTo(panel)
    _hp_caption, hp_value = panel.unit_stat_rows["hp"]
    assert hp_value.text() == ""


def test_on_unit_field_is_not_called_during_show_unit() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry())
    assert received == []


def test_on_unit_field_fires_once_per_genuine_edit() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry())
    panel.unit_field_editors["x"].setValue(99.0)
    assert len(received) == 1
    assert received[0][0] == "x"
    assert received[0][1] == 99.0


def test_on_unit_field_default_is_a_no_op() -> None:
    """The panel must stay constructible with no callback, the same
    contract PlayersPanel/TriggerPanel's callbacks have."""
    panel = _panel()
    panel.show_unit(_entry())
    panel.unit_field_editors["x"].setValue(123.0)  # must not raise


def test_float_editors_have_keyboard_tracking_off() -> None:
    panel = _panel()
    for field_id in ("x", "y", "z"):
        assert panel.unit_field_editors[field_id].keyboardTracking() is False


def test_rotation_caption_carries_the_full_caveat_as_a_tooltip() -> None:
    """The "shown as a facing, stored in radians" half of the old note stays
    reachable regardless of which const is selected -- via the Rotation
    caption's own tooltip, not the conditional unit_rotation_note."""
    panel = _panel()
    assert "radians" in panel.unit_rotation_label.toolTip()


@pytest.mark.font_sensitive
def test_min_width_and_height_stay_within_a_reasonable_floor() -> None:
    """A size floor, matching the trigger/players panel suites' own checks:
    this page must not raise the whole window's minimum."""
    panel = _panel()
    panel.show_unit(_entry())
    hint = panel.minimumSizeHint()
    assert hint.width() <= UnitsPanel.MIN_USEFUL_WIDTH


@pytest.mark.font_sensitive
def test_the_garrison_buttons_fit_the_floor_with_a_host_shown() -> None:
    """GH #142: the Garrison block is hidden in the width test above, and its
    widest button row is what grows past a fixed floor at a large font."""
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST))
    panel.show_garrison([("Archer", "Player 1", 11), ("Villager", "GAIA", 12)], 5)
    assert panel.garrison_buttons.isVisibleTo(panel)
    width = panel.garrison_buttons.minimumSizeHint().width()
    assert width <= UnitsPanel.MIN_USEFUL_WIDTH, f"garrison buttons demand {width} px"


def test_splitter_defaults_to_the_catalog_getting_at_least_two_thirds() -> None:
    """The catalog is the thing being browsed; the inspector is a small
    detail box below it -- default proportions should favor it, not split
    evenly or worse. A user drag overrides this; this only covers the
    as-constructed state."""
    panel = _panel()
    catalog_size, inspector_size = panel.splitter.sizes()
    assert catalog_size >= 2 * inspector_size


# -- catalog pane (S5) -------------------------------------------------------


def test_catalog_starts_with_no_pending_object() -> None:
    panel = _panel()
    assert panel.selected_object_const() is None
    assert panel.placing_label.text() == "Placing: (none)"


def test_owner_combo_defaults_to_player_1() -> None:
    panel = _panel()
    assert panel.owner_id() == 1


def test_select_owner_round_trips() -> None:
    panel = _panel()
    assert panel.select_owner(3) is True
    assert panel.owner_id() == 3


def test_select_owner_gaia() -> None:
    panel = _panel()
    assert panel.select_owner(GAIA_PLAYER_ID) is True
    assert panel.owner_id() == GAIA_PLAYER_ID


def test_select_owner_out_of_range_fails_and_leaves_the_combo_alone() -> None:
    panel = _panel()
    assert panel.select_owner(99) is False
    assert panel.owner_id() == 1


def test_select_object_sets_the_pending_const_and_placing_label() -> None:
    panel = _panel()
    panel.select_object(4)  # Archer
    assert panel.selected_object_const() == 4
    assert "Archer".lower() in panel.placing_label.text().lower()


def test_placing_label_carries_the_full_name_as_a_tooltip() -> None:
    """A one-line label doesn't wrap or elide -- a 36-character object name
    silently clips at MIN_USEFUL_WIDTH, so the full name stays reachable via
    hover. Caught by tools/gen_units_panel_eyeball.py's screenshot pass."""
    panel = _panel()
    panel.select_object(2448)  # FLAGSHIP OF NEARCHOS DOCKED MOVEABLE
    assert "FLAGSHIP OF NEARCHOS DOCKED MOVEABLE" in panel.placing_label.toolTip()


def test_clicking_a_row_sets_the_pending_const_but_does_not_place() -> None:
    received = []
    panel = _panel(on_place_requested=received.append)
    panel.catalog_view.select(4)
    assert panel.selected_object_const() == 4
    assert received == []


def test_a_filter_that_hides_the_pending_row_leaves_it_pending() -> None:
    """The sticky-pending contract: a filter change or stray click that
    lands the current item on nothing real must not blank the pending value
    back to "choose an object first"."""
    panel = _panel()
    panel.select_object(4)
    panel.catalog_view.filter_edit.setText("this matches nothing at all")
    assert panel.selected_object_const() == 4


def test_activating_a_row_sets_the_pending_const_and_requests_placement() -> None:
    received = []
    panel = _panel(on_place_requested=received.append)
    panel.catalog_view.select(4)
    panel.catalog_view.tree.itemActivated.emit(panel.catalog_view.tree.currentItem(), 0)
    assert panel.selected_object_const() == 4
    assert received == [4]


def test_on_place_requested_default_is_a_no_op() -> None:
    panel = _panel()
    panel.catalog_view.select(4)
    panel.catalog_view.tree.itemActivated.emit(panel.catalog_view.tree.currentItem(), 0)  # must not raise


def test_catalog_tree_uses_click_focus() -> None:
    """MapView's arrow-key unit nudge only fires while MapView itself has
    focus -- an always-visible 1000+ row tree defaulting to Qt's normal
    focus policy would silently steal it. See units_panel.py's own
    _build_catalog_pane docstring."""
    panel = _panel()
    assert panel.catalog_view.tree.focusPolicy() == Qt.ClickFocus


# --- GH #71: group mode ------------------------------------------------------


def test_show_group_shows_the_grid_and_keeps_the_count_line() -> None:
    panel = _panel()
    panel.show_group([_entry(reference_id=1), _entry(reference_id=2), _entry(reference_id=3)])
    assert panel.unit_inspector_grid.isVisibleTo(panel)
    assert panel.unit_inspector_empty.isVisibleTo(panel)
    assert panel.unit_inspector_empty.text() == "3 units selected"
    for field_id in ("reference_id", "garrisoned_in_id"):
        assert not panel.unit_field_labels[field_id].isVisibleTo(panel)
        assert not panel.unit_field_captions[field_id].isVisibleTo(panel)


def test_show_group_shows_common_values_when_the_members_agree() -> None:
    panel = _panel()
    panel.show_group([_entry(x=12.5, rotation=1.0), _entry(x=12.5, y=3.5, rotation=1.0)])
    assert panel.unit_field_labels["name"].text() != "(mixed)"
    assert panel.unit_field_labels["unit_const"].text() == str(_ARCHER_CONST)
    assert panel.unit_field_editors["x"].text() == "12.50"
    assert panel.unit_field_editors["y"].text() == "(mixed)"
    assert panel.unit_field_editors["rotation"].text() == "3"  # 1.0 rad = 2.55 of 16
    assert panel.unit_field_editors["player"].currentData() == 1
    assert panel.unit_stats_header.isVisibleTo(panel)


def test_show_group_marks_disagreeing_fields_mixed() -> None:
    panel = _panel()
    panel.show_group(
        [
            _entry(player_id=1, x=1.5, rotation=0.5),
            _entry(player_id=2, x=2.5, rotation=1.5, unit_const=_TREE_CONST),
            _entry(player_id=1, x=1.5, rotation=0.5),
        ]
    )
    assert panel.unit_field_labels["name"].text() == "(mixed)"
    assert panel.unit_field_labels["unit_const"].text() == "(mixed)"
    assert panel.unit_field_editors["x"].text() == "(mixed)"
    combo = panel.unit_field_editors["player"]
    assert combo.currentText() == "(mixed)"
    assert combo.currentData() is None
    # Only the two archers count for rotation, and they agree.
    assert panel.unit_field_editors["rotation"].text() == "1"  # 0.5 rad = 1.27 of 16
    assert panel.unit_rotation_note.isVisibleTo(panel)
    assert panel.unit_rotation_note.text().startswith("Typing a Rotation turns the 2 selected units")
    assert "The other 1 of 3" in panel.unit_rotation_note.text()
    assert not panel.unit_stats_header.isVisibleTo(panel)


def test_group_rotation_compares_facings_not_raw_radians() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=1.0), _entry(rotation=1.0 + 1e-9)])
    assert panel.unit_field_editors["rotation"].text() == "3"


def test_group_rotation_is_mixed_across_angle_members() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    assert panel.unit_field_editors["rotation"].text() == "(mixed)"
    assert not panel.unit_rotation_note.isVisibleTo(panel)


def test_group_with_no_editable_member_shows_na_and_hides_the_editor() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_GATE_CONST, rotation=7), _entry(unit_const=_WALL_CONST, rotation=2)])
    grid = panel.unit_inspector_grid
    assert not panel.unit_field_editors["rotation"].isVisibleTo(grid)
    assert panel.unit_field_labels["rotation"].isVisibleTo(grid)
    assert panel.unit_field_labels["rotation"].text() == "(n/a)"
    assert panel.unit_rotation_note.text().startswith("Typing a Rotation changes none of the 2 selected")


def test_show_unit_after_group_restores_single_unit_state() -> None:
    panel = _panel()
    panel.show_group([_entry(player_id=1, x=1.5, rotation=0.5), _entry(player_id=2, x=2.5, rotation=1.5)])
    panel.show_unit(_entry(x=4.5, rotation=0.0))
    combo = panel.unit_field_editors["player"]
    assert combo.findText("(mixed)") < 0
    assert combo.count() == 9
    spin = panel.unit_field_editors["x"]
    assert spin.minimum() == -(2**15)
    assert spin.text() == "4.50"
    # A real facing 0 sits at the minimum, where special text would render.
    rotation = panel.unit_field_editors["rotation"]
    assert rotation.text() == "0"
    assert rotation.minimum() == 0
    assert panel.unit_field_labels["reference_id"].isVisibleTo(panel)
    assert panel.unit_field_captions["garrisoned_in_id"].isVisibleTo(panel)
    assert not panel.unit_inspector_empty.isVisibleTo(panel)


def test_group_note_text_is_restored_for_a_single_variant_unit() -> None:
    panel = _panel()
    panel.show_group([_entry(), _entry(unit_const=_WALL_CONST, rotation=2)])
    panel.show_unit(_entry(unit_const=_WALL_CONST, rotation=2))
    assert panel.unit_rotation_note.text().startswith("Rotation is shown as a facing")
    assert panel.unit_field_labels["rotation"].text() == "2"


def test_populating_across_single_group_transitions_reports_nothing() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    mixed = [_entry(player_id=1, x=1.5, rotation=0.5), _entry(player_id=2, x=2.5, rotation=1.5)]
    agreeing = [_entry(player_id=3, x=9.5), _entry(player_id=3, x=9.5)]
    panel.show_unit(_entry(rotation=0.0))
    panel.show_group(mixed)
    panel.show_unit(_entry(player_id=4, x=5.5, rotation=2.0))
    panel.show_group(mixed)
    panel.show_group(agreeing)
    panel.show_group(mixed)
    panel.show_unit(None)
    panel.show_group(mixed)
    panel.show_selection_count(0)
    assert received == []


def test_a_genuine_group_edit_reports_but_the_mixed_placeholder_does_not() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_group([_entry(player_id=1, x=1.5), _entry(player_id=2, x=2.5)])
    spin = panel.unit_field_editors["x"]
    spin.setValue(50.0)
    assert received == [("x", 50.0)]
    # Back onto the sentinel (e.g. a wheel step down) is not an edit.
    spin.setValue(spin.minimum())
    combo = panel.unit_field_editors["player"]
    combo.setCurrentIndex(combo.findData(5))
    combo.setCurrentIndex(0)
    assert received == [("x", 50.0), ("player", 5)]


# --- GH #118: X/Y/Z clamp to the map ------------------------------------------


def _map_panel(size=(120, 120), **kwargs):
    return _panel(map_size=lambda: size, **kwargs)


def _type(spin, text: str) -> None:
    """Real keystrokes over the whole text, then Return. The validator
    rejects a keystroke that would leave the range, so "500" lands as 50."""
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, text)
    QTest.keyClick(spin, Qt.Key_Return)


def test_x_cannot_be_typed_or_stepped_past_the_map_edge() -> None:
    received = []
    panel = _map_panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry())
    x_spin = panel.unit_field_editors["x"]
    _type(x_spin, "500")
    assert x_spin.value() == 50.0  # the third digit was refused
    _type(x_spin, "-5")
    assert x_spin.value() == 50.0  # the minus sign was refused
    x_spin.setValue(500.0)  # a wheel/step or programmatic set clamps instead
    assert x_spin.value() == pytest.approx(119.99)
    x_spin.setValue(-5.0)
    assert x_spin.value() == 0.0
    assert received == [("x", 50.0), ("x", pytest.approx(119.99)), ("x", 0.0)]


def test_x_and_y_use_the_width_and_height_separately() -> None:
    panel = _map_panel(size=(120, 168))
    panel.show_unit(_entry())
    assert panel.unit_field_editors["x"].maximum() == pytest.approx(119.99)
    assert panel.unit_field_editors["y"].maximum() == pytest.approx(167.99)


def test_z_caps_at_the_elevation_limit() -> None:
    panel = _map_panel()
    panel.show_unit(_entry())
    z_spin = panel.unit_field_editors["z"]
    _type(z_spin, "40")
    assert z_spin.value() == 4.0
    z_spin.setValue(40.0)
    assert z_spin.value() == 15.0
    assert z_spin.minimum() == 0.0


def test_an_off_map_stored_x_shows_as_stored_without_an_edit() -> None:
    """Widen, don't clamp. The in-map unit first narrows the range, so this
    guards show_unit's range-before-value ordering: set the other way round,
    999 would silently clamp to the previous unit's 119.99."""
    received = []
    panel = _map_panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(x=4.5))
    assert panel.unit_field_editors["x"].maximum() == pytest.approx(119.99)
    panel.show_unit(_entry(x=999.0))
    x_spin = panel.unit_field_editors["x"]
    assert x_spin.text() == "999.00"
    assert x_spin.maximum() == 999.0
    # The next in-map unit narrows it again.
    panel.show_unit(_entry(x=4.5))
    assert x_spin.maximum() == pytest.approx(119.99)
    assert received == []


def test_a_group_with_an_off_map_member_widens_and_still_shows_mixed() -> None:
    panel = _map_panel()
    panel.show_group([_entry(x=4.5), _entry(x=999.0)])
    x_spin = panel.unit_field_editors["x"]
    assert x_spin.text() == "(mixed)"
    assert x_spin.maximum() == 999.0


def test_the_mixed_sentinel_sits_one_display_step_below_the_live_minimum() -> None:
    received = []
    panel = _map_panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_group([_entry(x=4.5), _entry(x=9.5)])
    x_spin = panel.unit_field_editors["x"]
    assert x_spin.minimum() == pytest.approx(-0.01)
    assert x_spin.value() == x_spin.minimum()
    # Nothing off-map is reachable in mixed mode: -0.5 clamps onto the sentinel.
    x_spin.setValue(1.5)
    x_spin.setValue(-0.5)
    assert x_spin.text() == "(mixed)"
    assert received == [("x", 1.5)]
    received.clear()
    # An off-map member moves the sentinel below its own value.
    panel.show_group([_entry(x=-5.0), _entry(x=9.5)])
    assert x_spin.minimum() == pytest.approx(-5.01)
    x_spin.setValue(x_spin.minimum())
    assert received == []


def test_no_map_keeps_the_wide_fallback_range() -> None:
    panel = _panel(map_size=lambda: None)
    panel.show_unit(_entry())
    for field_id in ("x", "y", "z"):
        spin = panel.unit_field_editors[field_id]
        assert (spin.minimum(), spin.maximum()) == (-(2**15), 2**15)
    panel.show_group([_entry(x=1.5), _entry(x=2.5)])
    assert panel.unit_field_editors["x"].minimum() == -(2**15) - 0.01


# --- GH #61: whole-number facings --------------------------------------------

_TREBUCHET_CONST = 42  # angle_count 32
_CONST_6 = 2607  # angle_count 6, holds 16-grid values in the corpus


def _rotation_spin(panel):
    return panel.unit_field_editors["rotation"]


def test_rotation_editor_is_a_wrapping_whole_facing() -> None:
    panel = _panel()
    panel.show_unit(_entry(rotation=math.pi / 2))
    spin = _rotation_spin(panel)
    assert isinstance(spin, QSpinBox)
    assert spin.wrapping()
    assert spin.keyboardTracking() is False
    assert spin.suffix() == ""
    assert (spin.minimum(), spin.maximum()) == (0, 15)
    assert spin.value() == 4


def test_a_trebuchet_ranges_over_its_own_32_facings() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST, rotation=math.pi))
    spin = _rotation_spin(panel)
    assert (spin.minimum(), spin.maximum()) == (0, 31)
    assert spin.value() == 16


def test_a_six_direction_const_ranges_to_5_and_shows_the_nearest_facing() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_CONST_6, rotation=3 * math.pi / 4))
    spin = _rotation_spin(panel)
    assert spin.maximum() == 5
    assert spin.value() == 2  # 2.25 frames


def test_switching_from_a_32_to_a_16_count_unit_sets_range_before_value() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST, rotation=20 * 2 * math.pi / 32))
    assert _rotation_spin(panel).value() == 20
    panel.show_unit(_entry(rotation=3 * 2 * math.pi / 16))
    assert _rotation_spin(panel).maximum() == 15
    assert _rotation_spin(panel).value() == 3
    # And back up: facing 20 must not clamp to the archer's 15.
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST, rotation=20 * 2 * math.pi / 32))
    assert _rotation_spin(panel).value() == 20
    assert received == []


def test_the_junk_sentinel_populates_as_its_wrapped_facing_without_an_edit() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(rotation=7.0))
    assert _rotation_spin(panel).value() == 2
    assert received == []


def test_the_editor_tooltip_carries_the_exact_stored_radians() -> None:
    panel = _panel()
    panel.show_unit(_entry(rotation=3 * 2 * math.pi / 16))
    assert _rotation_spin(panel).toolTip() == "Facing 3 of 16 (stored: 1.1781 rad)"


def test_a_wheel_step_reports_the_next_facing_and_wraps() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(rotation=15 * 2 * math.pi / 16))
    spin = _rotation_spin(panel)
    spin.stepBy(1)
    assert received == [("rotation", 0)]


def test_a_wheel_step_off_a_mixed_group_facing_reports_facing_0() -> None:
    """The "(mixed)" placeholder parks one below 0, so the first step up is a
    real edit to facing 0, not a no-op on the sentinel."""
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    spin = _rotation_spin(panel)
    assert spin.text() == "(mixed)"
    spin.stepBy(1)
    assert received == [("rotation", 0)]


def test_a_mixed_count_group_edits_on_the_finest_grid() -> None:
    """Archer (16) + trebuchet (32): the scale is 32, and a shared direction
    reads as one facing on it."""
    panel = _panel()
    panel.show_group([_entry(rotation=math.pi / 2), _entry(unit_const=_TREBUCHET_CONST, rotation=math.pi / 2)])
    spin = _rotation_spin(panel)
    assert spin.maximum() == 31
    assert spin.text() == "8"
    assert "32-direction scale" in spin.toolTip()
    assert "(16)" in spin.toolTip()


def test_a_mixed_count_group_facing_differently_shows_mixed() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.0), _entry(unit_const=_TREBUCHET_CONST, rotation=math.pi)])
    spin = _rotation_spin(panel)
    assert spin.text() == "(mixed)"
    assert spin.maximum() == 31


def test_a_mixed_facing_group_then_a_single_unit_restores_the_minimum() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    assert _rotation_spin(panel).minimum() == -1
    panel.show_group([_entry(rotation=0.5), _entry(rotation=0.5)])
    assert _rotation_spin(panel).minimum() == 0
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST))
    assert (_rotation_spin(panel).minimum(), _rotation_spin(panel).maximum()) == (0, 31)


# --- GH #123: the Rotation field shows a tree's variant --------------------

_PINE_CONST = 350  # 27 variants; _TREE_CONST (oak) has 42


def test_a_single_tree_shows_an_editable_variant_spinbox() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=_TREE_CONST, rotation=37.0))
    spin = _rotation_spin(panel)
    grid = panel.unit_inspector_grid
    assert spin.isVisibleTo(grid)
    assert not panel.unit_field_labels["rotation"].isVisibleTo(grid)
    assert (spin.minimum(), spin.maximum()) == (0, 41)
    assert spin.value() == 37
    assert spin.toolTip() == "Variant 37 of 42 (stored: 37)"
    assert not panel.unit_rotation_note.isVisibleTo(panel)
    assert received == []


def test_switching_variant_and_facing_units_sets_range_before_value() -> None:
    """An oak's variant 37 must not clamp to the pine's 0..26 shown before it."""
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=_PINE_CONST, rotation=20.0))
    panel.show_unit(_entry(unit_const=_TREE_CONST, rotation=37.0))
    assert _rotation_spin(panel).value() == 37
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST, rotation=math.pi))
    assert _rotation_spin(panel).value() == 16
    assert received == []


def test_a_variant_wheel_step_reports_the_next_variant_and_wraps() -> None:
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    panel.show_unit(_entry(unit_const=_TREE_CONST, rotation=41.0))
    _rotation_spin(panel).stepBy(1)
    assert received == [("rotation", 0)]


def test_an_oak_and_pine_group_edits_the_fewest_variants() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=5.0), _entry(unit_const=_PINE_CONST, rotation=5.0)])
    spin = _rotation_spin(panel)
    assert spin.maximum() == 26
    assert spin.text() == "5"
    assert "the fewest any selected object has" in spin.toolTip()
    assert not panel.unit_rotation_note.isVisibleTo(panel)


def test_a_tree_group_beyond_the_shared_range_can_only_show_mixed() -> None:
    """An oak at 35 exceeds a pine's 0..26, but then the two differ, so the
    box reads "(mixed)" rather than a clamped 26."""
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=35.0), _entry(unit_const=_PINE_CONST, rotation=5.0)])
    assert _rotation_spin(panel).text() == "(mixed)"


def test_an_archer_and_tree_group_is_facing_mode_and_the_note_counts_the_tree() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(unit_const=_TREE_CONST, rotation=7.0)])
    spin = _rotation_spin(panel)
    assert spin.maximum() == 15
    assert spin.text() == "1"
    assert panel.unit_rotation_note.isVisibleTo(panel)
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation turns the 1 selected unit with a facing.")
    assert "Rotate (. and ,) still steps trees" in note


def test_the_group_note_leads_with_what_a_typed_rotation_turns() -> None:
    """GH #71: "won't rotate" read as "trees can't rotate", though Rotate steps them."""
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=0.5), _entry(unit_const=_TREE_CONST, rotation=7.0)])
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation turns the 2 selected units with a facing.")
    assert "The other 1 of 3 (trees, walls, gates, buildings, scenery) keep theirs." in note
    assert "won't rotate" not in note
    assert "Rotate (. and ,) still steps trees and scenery through their variants." in note


def test_the_group_note_names_the_rebound_rotate_keys(monkeypatch) -> None:
    from descape import settings

    keys = {"unit_rotate_cw": "R", "unit_rotate_ccw": ""}
    monkeypatch.setattr(settings, "get_keybind", lambda action_id: keys.get(action_id, ""))
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(unit_const=_GATE_CONST)])
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation turns the 1 selected unit with a facing.")
    assert "Rotate (R) still turns gates through their four orientations." in note


def test_the_note_for_a_group_with_nothing_typeable_says_so() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_GATE_CONST, rotation=7), _entry(unit_const=_WALL_CONST, rotation=2)])
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation changes none of the 2 selected:")
    assert "Rotate (. and ,) still turns gates" in note


def test_a_tree_and_wall_group_is_variant_mode_and_the_note_counts_the_wall() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=7.0), _entry(unit_const=_WALL_CONST, rotation=2.0)])
    spin = _rotation_spin(panel)
    assert spin.isVisibleTo(panel.unit_inspector_grid)
    assert (spin.minimum(), spin.maximum()) == (0, 41)
    assert spin.text() == "7"
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation picks that variant for the 1 selected object with graphic variants.")
    assert "The other 1 of 2 (walls, cliffs, gates, single-frame objects) keep theirs." in note
    assert "won't rotate" not in note
    assert "Rotate" not in note.removeprefix("Typing a Rotation")


def test_a_tree_and_gate_variant_group_note_says_rotate_turns_the_gate() -> None:
    """The tail names only what the typed value skips: the gate, never the tree it already steps."""
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=7.0), _entry(unit_const=_GATE_CONST)])
    note = panel.unit_rotation_note.text()
    assert note.startswith("Typing a Rotation picks that variant for the 1 selected object with graphic variants.")
    assert note.endswith("Rotate (. and ,) still turns gates through their four orientations.")
    assert "steps trees" not in note


# --- GH #42: the Garrison block -----------------------------------------


def test_the_garrison_block_is_hidden_until_it_is_given_rows() -> None:
    panel = _panel()
    panel.show_unit(_entry())
    assert not panel.garrison_tree.isVisibleTo(panel)
    assert not panel.garrison_header.isVisibleTo(panel)


def test_show_garrison_lists_the_rows_and_counts_against_capacity() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11), ("Villager", "GAIA", 12)], 5)

    assert panel.garrison_tree.isVisibleTo(panel)
    assert "Garrison (2 / 5)" in panel.garrison_header.text()
    assert panel.garrison_tree.topLevelItemCount() == 2
    assert panel.garrison_tree.topLevelItem(0).text(0) == "Archer"
    assert panel.garrison_tree.topLevelItem(1).text(1) == "GAIA"
    assert not panel.garrison_note.isVisibleTo(panel)
    assert panel.garrison_add_button.isEnabled()
    assert not panel.garrison_delete_button.isEnabled()


def test_add_is_disabled_at_capacity_and_the_note_explains_an_overfull_host() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    rows = [(f"Archer {i}", "Player 1", 20 + i) for i in range(6)]
    panel.show_garrison(rows, 5)

    assert not panel.garrison_add_button.isEnabled()
    assert panel.garrison_add_button.toolTip() == "Full: the game gives this one 5 places"
    assert panel.garrison_note.isVisibleTo(panel)
    assert "5" in panel.garrison_note.text()


def test_a_wrong_type_occupant_is_reported_not_corrected() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Mangonel", "Player 1", 11)], 5, wrong_type=1)

    assert panel.garrison_note.isVisibleTo(panel)
    assert "cannot garrison here" in panel.garrison_note.text()
    assert panel.garrison_tree.topLevelItemCount() == 1


def test_selecting_a_row_enables_delete_and_reports_the_selected_ids() -> None:
    reported = []
    panel = _panel(on_garrison_delete=reported.append)
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11), ("Villager", "Player 1", 12)], 5)

    panel.garrison_tree.topLevelItem(1).setSelected(True)
    assert panel.garrison_delete_button.isEnabled()
    assert panel.garrison_selection() == [12]
    panel.garrison_delete_button.click()
    assert reported == [[12]]


def test_add_and_double_click_report_through_their_callbacks() -> None:
    added, navigated = [], []
    panel = _panel(on_garrison_add=lambda: added.append(True), on_garrison_navigate=navigated.append)
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11)], 5)

    panel.garrison_add_button.click()
    panel._garrison_double_clicked(panel.garrison_tree.topLevelItem(0), 0)
    assert added == [True]
    assert navigated == [11]


def test_a_group_selection_or_a_new_single_unit_hides_the_garrison_block() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11)], 5)

    panel.show_group([_entry(reference_id=1), _entry(reference_id=2)])
    assert not panel.garrison_tree.isVisibleTo(panel)

    panel.show_garrison([("Archer", "Player 1", 11)], 5)
    panel.show_unit(_entry(reference_id=3))
    assert not panel.garrison_tree.isVisibleTo(panel)


# --- GH #115: the Garrison block's manager buttons ---------------------------


def _row_buttons(panel):
    return (
        panel.garrison_copy_button,
        panel.garrison_unload_button,
        panel.garrison_delete_button,
        panel.garrison_owner_combo,
    )


def test_the_buttons_keep_the_planned_order() -> None:
    panel = _panel()
    order = [
        panel.garrison_add_button,
        panel.garrison_pick_button,
        panel.garrison_copy_button,
        panel.garrison_paste_button,
        panel.garrison_unload_button,
        panel.garrison_delete_button,
        panel.garrison_owner_combo,
    ]
    widgets = []
    for row in panel.garrison_button_rows:
        layout = row.layout()
        widgets += [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget()]
    assert widgets == order


def test_the_row_buttons_need_a_selected_row() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11), ("Villager", "Player 1", 12)], 5)
    assert not any(w.isEnabled() for w in _row_buttons(panel))

    panel.garrison_tree.topLevelItem(0).setSelected(True)
    assert all(w.isEnabled() for w in _row_buttons(panel))

    panel.garrison_tree.clearSelection()
    assert not any(w.isEnabled() for w in _row_buttons(panel))


def test_paste_follows_the_viewers_can_paste_flag() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([], 5)
    assert not panel.garrison_paste_button.isEnabled()
    panel.show_garrison([], 5, can_paste=True)
    assert panel.garrison_paste_button.isEnabled()


def test_pick_from_map_is_checkable_and_disabled_on_a_full_host() -> None:
    toggled = []
    panel = _panel(on_garrison_pick=toggled.append)
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([], 5)
    assert panel.garrison_pick_button.isCheckable()
    assert panel.garrison_pick_button.isEnabled()

    panel.garrison_pick_button.click()
    assert toggled == [True]

    full = [(f"Archer {i}", "Player 1", 20 + i) for i in range(5)]
    panel.show_garrison(full, 5)
    assert panel.garrison_pick_button.isEnabled(), "an armed pick can always be switched off"

    panel.set_garrison_pick_armed(False)
    panel.show_garrison(full, 5)
    assert not panel.garrison_pick_button.isEnabled()


def test_set_garrison_pick_armed_reports_nothing_back() -> None:
    toggled = []
    panel = _panel(on_garrison_pick=toggled.append)
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([], 5)

    panel.set_garrison_pick_armed(True)
    assert panel.garrison_pick_button.isChecked()
    panel.set_garrison_pick_armed(False)
    assert not panel.garrison_pick_button.isChecked()
    assert toggled == []


def test_copy_paste_unload_and_owner_report_through_their_callbacks() -> None:
    copied, pasted, unloaded, owners = [], [], [], []
    panel = _panel(
        on_garrison_copy=copied.append,
        on_garrison_paste=lambda: pasted.append(True),
        on_garrison_unload=unloaded.append,
        on_garrison_owner=lambda ids, pid: owners.append((ids, pid)),
    )
    panel.show_unit(_entry(unit_const=_TOWER_CONST, reference_id=10))
    panel.show_garrison([("Archer", "Player 1", 11), ("Villager", "Player 1", 12)], 5, can_paste=True)
    panel.garrison_tree.topLevelItem(1).setSelected(True)

    panel.garrison_copy_button.click()
    panel.garrison_paste_button.click()
    panel.garrison_unload_button.click()
    combo = panel.garrison_owner_combo
    combo.setCurrentIndex(combo.findData(3))
    combo.activated.emit(combo.currentIndex())

    assert copied == [[12]]
    assert pasted == [True]
    assert unloaded == [[12]]
    assert owners == [([12], 3)]
    assert combo.currentData() is None, "the combo goes back to its prompt row"


def test_the_owner_combo_follows_player_labels() -> None:
    from descape import player_labels

    panel = _panel()
    combo = panel.garrison_owner_combo
    labels = player_labels.DEFAULT_LABELS
    renamed = type(labels)(
        text=tuple("Renamed" if i == 2 else t for i, t in enumerate(labels.text)),
        full=tuple("Renamed" if i == 2 else t for i, t in enumerate(labels.full)),
    )
    panel.refresh_player_labels(renamed, None)
    assert combo.itemText(combo.findData(2)) == "Renamed"


def test_select_object_alone_is_a_no_op_on_a_filtered_out_row() -> None:
    """The premise behind pick_object() (GH #125)."""
    panel = _panel()
    panel.select_object(4)
    panel.catalog_view.filter_edit.setText("this matches nothing at all")
    panel.select_object(117)
    assert panel.selected_object_const() == 4


def test_pick_object_clears_a_filter_that_hides_the_row() -> None:
    panel = _panel()
    panel.select_object(4)
    panel.catalog_view.filter_edit.setText("this matches nothing at all")
    assert panel.pick_object(117)
    assert panel.selected_object_const() == 117
    assert panel.catalog_view.filter_edit.text() == ""


def test_pick_object_keeps_a_filter_that_shows_the_row() -> None:
    panel = _panel()
    panel.catalog_view.filter_edit.setText("wall")
    assert panel.pick_object(72)
    assert panel.selected_object_const() == 72
    assert panel.catalog_view.filter_edit.text() == "wall"


def test_pick_object_refuses_an_unknown_const() -> None:
    panel = _panel()
    panel.select_object(4)
    assert not panel.pick_object(-12345)
    assert panel.selected_object_const() == 4
