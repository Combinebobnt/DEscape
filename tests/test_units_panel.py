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

from dataclasses import dataclass

import pytest
from PyQt5.QtCore import Qt

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

    panel.show_unit(_entry(unit_const=_TREE_CONST, rotation=37))
    assert not panel.unit_field_editors["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_field_labels["rotation"].isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_field_labels["rotation"].text() == "37"


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
    """The "shown raw, in radians" half of the old always-visible note stays
    reachable regardless of which const is selected -- via the Rotation
    caption's own tooltip, not the conditional unit_rotation_note."""
    panel = _panel()
    assert "radians" in panel.unit_rotation_label.toolTip()


def test_min_width_and_height_stay_within_a_reasonable_floor() -> None:
    """A size floor, matching the trigger/players panel suites' own checks:
    this page must not raise the whole window's minimum."""
    panel = _panel()
    panel.show_unit(_entry())
    hint = panel.minimumSizeHint()
    assert hint.width() <= UnitsPanel.MIN_USEFUL_WIDTH


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
