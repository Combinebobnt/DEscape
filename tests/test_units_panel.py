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
from PyQt5.QtWidgets import QLabel, QSpinBox

from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_pick import UnitEntry
from descape.units_panel import _ROTATION_TOOLTIP, UnitsPanel

import conftest
from conftest import _BUILDING_GREEN, _UNIT_RED, _fake_install

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


def _group_rotation_note(panel) -> str:
    """The group note leading the Rotation caption's tooltip (GH #198), "" for the caveat alone."""
    tip = panel.unit_rotation_label.toolTip()
    assert tip.endswith(_ROTATION_TOOLTIP)
    return tip.removesuffix(_ROTATION_TOOLTIP).removesuffix("\n\n")


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
    assert _group_rotation_note(panel) == ""
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
    assert _group_rotation_note(panel) == ""
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


_STAT_CAPTIONS = {
    "hp": "Hit points", "attack": "Attack", "melee_armour": "Melee armour",
    "pierce_armour": "Pierce armour", "range": "Range",
}


def _stat_cell(panel, field_id: str) -> tuple[int, int] | None:
    """(row, column) of a stat pair's icon in the stats grid, or None if it isn't placed."""
    grid = panel.unit_stats_grid.layout()
    index = grid.indexOf(panel.unit_stat_rows[field_id][0])
    return None if index < 0 else grid.getItemPosition(index)[:2]


def test_stats_block_shows_all_five_rows_for_a_full_combat_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    assert panel.unit_stats_grid.isVisibleTo(panel)
    for field_id, expected in (
        ("hp", "30"), ("attack", "4"), ("melee_armour", "0"), ("pierce_armour", "0"), ("range", "4"),
    ):
        caption, value = panel.unit_stat_rows[field_id]
        assert caption.isVisibleTo(panel)
        assert value.isVisibleTo(panel)
        assert value.text() == expected
        assert caption.toolTip() == caption.accessibleName() == _STAT_CAPTIONS[field_id]


def test_the_stats_pack_two_pairs_to_a_row() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    cells = {field_id: _stat_cell(panel, field_id) for field_id in _STAT_CAPTIONS}
    assert cells == {"hp": (0, 0), "attack": (0, 2), "melee_armour": (1, 0), "pierce_armour": (1, 2), "range": (2, 0)}


def test_a_tree_packs_to_one_pair_after_a_full_unit() -> None:
    """GH #148: the present pairs re-pack, so a tree shows HP alone in the first cell, not a gap."""
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    panel.show_unit(_entry(unit_const=_TREE_CONST, rotation=7.0))
    assert _stat_cell(panel, "hp") == (0, 0)
    for field_id in ("attack", "melee_armour", "pierce_armour", "range"):
        assert _stat_cell(panel, field_id) is None
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    assert _stat_cell(panel, "range") == (2, 0)


def test_with_no_install_the_stat_captions_are_text(monkeypatch) -> None:
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    for field_id, caption_text in _STAT_CAPTIONS.items():
        caption = panel.unit_stat_rows[field_id][0]
        assert caption.text() == f"{caption_text}:"
        assert caption.pixmap() is None or caption.pixmap().isNull()
    assert panel.portrait_source in ("sprite", None)
    if panel.portrait_source is None:
        assert not panel.unit_portrait.isVisibleTo(panel)


def test_stats_block_hides_combat_rows_for_an_hp_only_const() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TREE_CONST))
    assert panel.unit_stats_grid.isVisibleTo(panel)
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
    assert not panel.unit_stats_grid.isVisibleTo(panel)
    assert panel.unit_header.isVisibleTo(panel), "the name and Unit ID still show"
    assert not panel.unit_portrait.isVisibleTo(panel)


def test_stats_block_hides_and_blanks_on_show_unit_none() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    panel.show_unit(None)
    assert not panel.unit_stats_grid.isVisibleTo(panel)
    assert not panel.unit_header.isVisibleTo(panel)
    _hp_caption, hp_value = panel.unit_stat_rows["hp"]
    assert hp_value.text() == ""


def test_name_and_unit_id_sit_in_the_header_and_the_stats_note_is_a_tooltip() -> None:
    """GH #148: no Name/Unit ID grid rows, no "Base stats" title, no visible note line."""
    from PyQt5.QtWidgets import QLabel

    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    for field_id in ("name", "unit_const"):
        label = panel.unit_field_labels[field_id]
        assert panel.unit_header.isAncestorOf(label)
        assert not panel.unit_inspector_grid.isAncestorOf(label)
    assert panel.unit_field_labels["name"].font().bold()
    assert panel.unit_field_captions["unit_const"].text() == "Unit ID:"
    assert panel.unit_stats_grid.isAncestorOf(panel.unit_stat_rows["hp"][1])
    assert panel.unit_header.isAncestorOf(panel.unit_stats_grid)
    assert "civ bonuses" in panel.unit_stats_grid.toolTip()
    visible = [label.text() for label in panel.findChildren(QLabel) if label.isVisibleTo(panel)]
    assert not [text for text in visible if "Base stats" in text or text.startswith("Base values")]


# -- GH #148: portrait and stat icons from a fake install ----------------------
# _fake_install and the fake_install fixture live in conftest.py, shared with test_unit_card.py.


def _portrait_centre(panel) -> tuple[int, int, int, int]:
    image = panel.unit_portrait.pixmap().toImage()
    colour = image.pixelColor(image.width() // 2, image.height() // 2)
    return colour.red(), colour.green(), colour.blue(), colour.alpha()


def _settle(panel) -> None:
    """GH #68: portraits arrive from the shared loader's worker, not inside show_unit()."""
    conftest.settle_thumbs(panel.thumb_loader)


def test_an_install_gives_the_archer_its_portrait_and_icon_captions(fake_install) -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    _settle(panel)
    assert panel.portrait_source == "icon"
    assert panel.unit_portrait.isVisibleTo(panel)
    assert _portrait_centre(panel) == _UNIT_RED
    for field_id, caption_text in _STAT_CAPTIONS.items():
        caption = panel.unit_stat_rows[field_id][0]
        assert caption.text() == ""
        assert caption.pixmap().height() == caption.fontMetrics().height()
        assert caption.toolTip() == caption.accessibleName() == caption_text


def test_a_castle_takes_the_building_portrait_not_unit_007(fake_install) -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=82))
    _settle(panel)
    assert panel.portrait_source == "icon"
    assert _portrait_centre(panel) == _BUILDING_GREEN


def test_a_group_shows_the_portrait_only_for_one_shared_const(fake_install) -> None:
    panel = _panel()
    panel.show_group([_entry(reference_id=1), _entry(reference_id=2)])
    _settle(panel)
    assert panel.portrait_source == "icon"
    assert panel.unit_portrait.isVisibleTo(panel)
    panel.show_group([_entry(reference_id=1), _entry(unit_const=_TREE_CONST, reference_id=2)])
    _settle(panel)
    assert panel.unit_field_labels["name"].text() == "(mixed)"
    assert panel.portrait_source is None
    assert not panel.unit_portrait.isVisibleTo(panel)


def test_the_icons_follow_an_install_change(tmp_path) -> None:
    """Loaded once per install: dropping the install brings the text captions back."""
    from descape import asset_source

    panel = _panel()
    asset_source.set_install_path_override(_fake_install(tmp_path))
    try:
        panel.show_unit(_entry(unit_const=_ARCHER_CONST))
        _settle(panel)
        assert panel.unit_stat_rows["hp"][0].text() == ""
        assert panel.portrait_source == "icon"
    finally:
        asset_source.set_install_path_override(None)
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    _settle(panel)
    assert panel.unit_stat_rows["hp"][0].text() == "Hit points:"
    assert panel.portrait_source != "icon"


# Measured before GH #148 the same way, with an Archer at MIN_USEFUL_WIDTH and the test font: 475 px.
_PRE_GH148_ARCHER_MIN_HEIGHT = 475


def test_the_header_makes_the_inspector_shorter_than_before() -> None:
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 900)
    panel.show()
    try:
        QApplication.processEvents()
        panel.show_unit(_entry(unit_const=_ARCHER_CONST))
        QApplication.processEvents()
        assert panel.inspector_host.minimumHeight() < _PRE_GH148_ARCHER_MIN_HEIGHT
    finally:
        panel.close()


@pytest.mark.font_sensitive
@pytest.mark.parametrize("with_install", [False, True])
def test_the_header_fits_the_floor(with_install, tmp_path, monkeypatch) -> None:
    """Portrait plus two pair columns, icon or text captions, beside a scroll bar at MIN_USEFUL_WIDTH."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.set_install_path_override(_fake_install(tmp_path) if with_install else None)
    try:
        panel = _panel()
        panel.show_unit(_entry(unit_const=_ARCHER_CONST))
        area = panel.inspector_area
        margins = panel.inspector_layout.contentsMargins()
        room = (
            UnitsPanel.MIN_USEFUL_WIDTH - 2 * area.frameWidth() - area.verticalScrollBar().sizeHint().width()
            - margins.left() - margins.right()
        )
        width = panel.unit_header.minimumSizeHint().width()
        assert width <= room, f"header demands {width} px of {room}"
    finally:
        asset_source.set_install_path_override(None)


@pytest.mark.corpus
def test_the_real_install_gives_an_archer_its_portrait() -> None:
    from descape import asset_source

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    _settle(panel)
    assert panel.portrait_source == "icon"
    assert not panel.unit_portrait.pixmap().isNull()
    assert all(not panel.unit_stat_rows[field_id][0].pixmap().isNull() for field_id in _STAT_CAPTIONS)


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
    caption's own tooltip, which a group's note only ever leads (GH #198)."""
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


def _units_window(width: int, height: int):
    """A shown window at `width` x `height`, blank template loaded, in Units mode."""
    from PyQt5.QtWidgets import QApplication

    from descape.scenario_io import BLANK_TEMPLATE_PATH

    window = conftest.shown_window(width, height)
    window.load_scenario(BLANK_TEMPLATE_PATH)
    window.mode_combo.setCurrentText("Units")
    # Twice: the default split runs on a zero timer the first show arms.
    QApplication.processEvents()
    QApplication.processEvents()
    assert window.units_panel.isVisible()
    return window


@pytest.mark.parametrize(("width", "height"), [(800, 600), (1280, 720), (1920, 1080)])
def test_splitter_defaults_to_the_catalog_getting_at_least_two_thirds(width: int, height: int) -> None:
    """The catalog is the thing being browsed; the inspector is a small detail box below it.
    Pass conditions, on a shown window's first Units entry: catalog >= 2 x inspector, and the
    catalog pane at least its own minimum height. A user drag overrides this."""
    window = _units_window(width, height)
    try:
        panel = window.units_panel
        catalog_size, inspector_size = panel.splitter.sizes()
        assert catalog_size >= 2 * inspector_size, (catalog_size, inspector_size)
        assert catalog_size >= panel.splitter.widget(0).minimumSizeHint().height()
    finally:
        conftest.close_window(window)


def _assert_the_inspector_fits_its_content(panel) -> None:
    from PyQt5.QtCore import QPoint

    viewport = panel.inspector_area.viewport()
    assert viewport.height() >= panel.inspector_host.minimumHeight(), (viewport.height(), panel.inspector_host.minimumHeight())
    card = panel.placing_card
    top = card.mapTo(viewport, QPoint(0, 0)).y()
    assert top >= 0 and top + card.height() <= viewport.height(), (top, card.height(), viewport.height())


def test_the_default_window_shows_the_whole_placing_card() -> None:
    """TASK-420, 1280x720, Units mode, nothing selected, an Archer picked. Pass conditions: the
    inspector viewport is at least its content's height, and the placing card lies inside it."""
    from PyQt5.QtWidgets import QApplication

    window = _units_window(1280, 720)
    try:
        panel = window.units_panel
        panel.select_object(_ARCHER_CONST)
        QApplication.processEvents()
        assert panel.placing_card.isVisible()
        _assert_the_inspector_fits_its_content(panel)
    finally:
        conftest.close_window(window)


def test_a_dragged_split_survives_a_taller_window() -> None:
    """TASK-420: the default split is one-shot. Pass condition: after setting the split to
    [x, 120] and growing the window 200 px, the inspector is still 120 px."""
    from PyQt5.QtWidgets import QApplication

    window = _units_window(1280, 720)
    try:
        splitter = window.units_panel.splitter
        total = sum(splitter.sizes())
        splitter.setSizes([total - 120, 120])
        QApplication.processEvents()
        assert splitter.sizes()[1] == 120
        window.resize(1280, 920)
        QApplication.processEvents()
        QApplication.processEvents()
        assert sum(splitter.sizes()) > total
        assert splitter.sizes()[1] == 120
    finally:
        conftest.close_window(window)


def test_a_standalone_panel_shown_once_fits_its_inspector() -> None:
    """TASK-420: the path a resize-only hook misses (a standalone show() resizes while hidden).
    Pass condition: resize, show, two event passes, and the inspector fits its content."""
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    try:
        panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 600)
        panel.show()
        QApplication.processEvents()
        QApplication.processEvents()
        _assert_the_inspector_fits_its_content(panel)
    finally:
        panel.close()


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
    panel.catalog_view.list.itemActivated.emit(panel.catalog_view.list.currentItem())
    assert panel.selected_object_const() == 4
    assert received == [4]


def test_on_place_requested_default_is_a_no_op() -> None:
    panel = _panel()
    panel.catalog_view.select(4)
    panel.catalog_view.list.itemActivated.emit(panel.catalog_view.list.currentItem())  # must not raise


def test_catalog_tree_uses_click_focus() -> None:
    """MapView's arrow-key unit nudge only fires while MapView itself has
    focus -- an always-visible list of hundreds of rows defaulting to Qt's
    normal focus policy would silently steal it. See units_panel.py's own
    _build_catalog_pane comment."""
    panel = _panel()
    assert panel.catalog_view.list.focusPolicy() == Qt.ClickFocus


def test_a_category_switch_leaves_the_pending_object_alone() -> None:
    """GH #68: categories, classes and the filter only browse; the pick is sticky."""
    changed = []
    panel = _panel(on_pending_changed=lambda: changed.append(True))
    panel.select_object(_ARCHER_CONST)
    changed.clear()
    panel.catalog_view.category_buttons["Buildings"].click()
    panel.catalog_view.class_combo.setCurrentIndex(1)
    panel.catalog_view.display_buttons["names"].click()
    assert panel.selected_object_const() == _ARCHER_CONST
    assert panel.placing_card.const_label.text() == str(_ARCHER_CONST)
    assert changed == []


# --- GH #164: Villager (Random) ---------------------------------------------


def _visible_catalog_values(panel) -> list[int]:
    view = panel.catalog_view.list
    return [view.item(i).data(Qt.UserRole) for i in range(view.count()) if not view.item(i).isHidden()]


def _random_panel(seed=164, **kwargs):
    import random

    from descape.place_random import RANDOM_VILLAGER

    panel = _panel(rng=random.Random(seed), **kwargs)
    panel.catalog_view.filter_edit.setText("villager")
    panel.catalog_view.select(RANDOM_VILLAGER)
    return panel


def test_filtering_villager_shows_the_random_row_beside_both_villagers() -> None:
    from descape.place_random import RANDOM_VILLAGER

    panel = _panel()
    panel.catalog_view.filter_edit.setText("villager")
    assert sorted(_visible_catalog_values(panel)) == sorted([83, 293, RANDOM_VILLAGER])


def test_picking_villager_random_places_a_real_villager() -> None:
    from descape.place_random import RANDOM_VILLAGER

    panel = _random_panel()
    assert panel.catalog_view.current_value() == RANDOM_VILLAGER
    assert panel.selected_object_const() in (83, 293)
    assert panel.placing_label.text() == "Placing: Villager (Random)"
    assert panel.selected_object_label() == "Villager (Random)"
    assert panel.selected_object_choices() == (83, 293)


def test_reroll_pending_rolls_both_villagers_and_refreshes_the_ghost() -> None:
    changed = []
    panel = _random_panel(on_pending_changed=lambda: changed.append(True))
    changed.clear()
    rolled = set()
    for _ in range(20):
        panel.reroll_pending()
        rolled.add(panel.selected_object_const())
    assert rolled == {83, 293}
    assert len(changed) == 20
    assert panel.placing_label.text() == "Placing: Villager (Random)"


def test_reroll_pending_is_a_no_op_for_a_real_pick() -> None:
    changed = []
    panel = _panel(on_pending_changed=lambda: changed.append(True))
    panel.select_object(4)
    changed.clear()
    panel.reroll_pending()
    assert panel.selected_object_const() == 4
    assert changed == []
    assert panel.selected_object_choices() == (4,)


def test_a_real_pick_leaves_random_mode() -> None:
    panel = _random_panel()
    assert panel.pick_object(83)
    assert panel.selected_object_const() == 83
    assert panel.selected_object_choices() == (83,)
    for _ in range(10):
        panel.reroll_pending()
        assert panel.selected_object_const() == 83
    assert panel.placing_label.text() != "Placing: Villager (Random)"


def test_nothing_picked_has_no_choices_or_label() -> None:
    panel = _panel()
    assert panel.selected_object_choices() is None
    assert panel.selected_object_label() is None


def test_the_placing_card_shows_villager_random_as_the_male_villager() -> None:
    """GH #164 + TASK-337: the card names the pick and shows its preview
    const's stats, not the roll; Unit ID lists both consts the next placement
    may be (orchestrator ruling, 2026-10-14)."""
    panel = _random_panel()
    card = panel.placing_card
    assert card.name_label.text() == "Placing: Villager (Random)"
    assert card.const_label.text() == "83 / 293 (random)"
    assert _shown_stats(card)["hp"] == "25"
    assert card.minimumSizeHint().width() <= UnitsPanel.MIN_USEFUL_WIDTH
    for _ in range(6):
        panel.reroll_pending()
        assert card.const_label.text() == "83 / 293 (random)"
        assert _shown_stats(card)["hp"] == "25"


# --- TASK-337: the placing card ------------------------------------------------


def _shown_stats(card) -> dict[str, str]:
    return {
        field_id: value.text()
        for field_id, (caption, value) in card.stat_rows.items()
        if caption.isVisibleTo(card) and value.isVisibleTo(card)
    }


def _longest_entry():
    from descape import object_catalog

    return max((entry for entry in object_catalog.objects() if not entry.hidden), key=lambda entry: len(entry.name))


def test_the_catalog_has_no_side_preview() -> None:
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    assert getattr(panel.catalog_view, "preview", None) is None
    assert panel.placing_card.isAncestorOf(panel.placing_label)
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 800)
    panel.show()
    try:
        QApplication.processEvents()
        view = panel.catalog_view.list
        right = view.mapTo(panel, view.rect().topRight()).x()
        assert right == panel.width() - 1, "the list's right edge is the pane's"
    finally:
        panel.close()


def test_picking_a_row_fills_the_placing_card() -> None:
    panel = _panel()
    panel.catalog_view.select(_ARCHER_CONST)
    card = panel.placing_card
    assert card.const_label.text() == str(_ARCHER_CONST)
    assert _shown_stats(card) == {"hp": "30", "attack": "4", "melee_armour": "0", "pierce_armour": "0", "range": "4"}


def test_a_filter_that_hides_the_pending_row_leaves_the_card_on_it() -> None:
    panel = _panel()
    panel.select_object(_ARCHER_CONST)
    text = panel.placing_label.text()
    panel.catalog_view.filter_edit.setText("this matches nothing at all")
    assert panel.placing_label.text() == text
    assert panel.placing_card.const_label.text() == str(_ARCHER_CONST)


def test_the_pick_and_the_map_selection_each_show_on_their_own_card() -> None:
    panel = _panel()
    panel.select_object(_ARCHER_CONST)
    panel.show_unit(_entry(unit_const=82))  # Castle
    assert panel.placing_card.const_label.text() == str(_ARCHER_CONST)
    assert panel.selection_card.const_label.text() == "82"
    assert _shown_stats(panel.selection_card)["hp"] == "4800"
    panel.show_unit(None)
    assert panel.placing_card.const_label.text() == str(_ARCHER_CONST)
    assert _shown_stats(panel.placing_card)["hp"] == "30"


def test_the_two_cards_share_one_slot_in_the_inspector() -> None:
    """A selection shows over the pick; the pick shows again once nothing is selected."""
    panel = _panel()
    panel.select_object(_ARCHER_CONST)
    assert panel.inspector_host.isAncestorOf(panel.placing_card)
    assert panel.placing_card.isVisibleTo(panel)
    assert not panel.selection_card.isVisibleTo(panel)
    panel.show_unit(_entry(unit_const=82))
    assert panel.selection_card.isVisibleTo(panel)
    assert not panel.placing_card.isVisibleTo(panel)
    panel.show_group([_entry(unit_const=82), _entry(unit_const=_ARCHER_CONST, reference_id=102)])
    assert not panel.placing_card.isVisibleTo(panel)
    panel.show_selection_count(0)
    assert panel.placing_card.isVisibleTo(panel)
    assert not panel.selection_card.isVisibleTo(panel)


def _holed_install(tmp_path):
    """The fake install, with the Archer's portrait opaque black but for a
    transparent centre: DE leaves the player-colour region that way."""
    from PIL import Image

    root = _fake_install(tmp_path)
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    image.paste((0, 0, 0, 0), (8, 8, 24, 24))
    image.save(root / "widgetui" / "textures" / "ingame" / "units" / "017_50730.DDS", format="DDS")
    return root


def _centre(pixmap) -> tuple[int, int, int]:
    image = pixmap.toImage()
    colour = image.pixelColor(image.width() // 2, image.height() // 2)
    return colour.red(), colour.green(), colour.blue()


def test_portraits_fill_the_player_colour_region_with_the_owners_colour(tmp_path, monkeypatch) -> None:
    from descape import asset_source
    from descape.terrain_palette import PLAYER_COLORS

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.set_install_path_override(_holed_install(tmp_path))
    try:
        panel = _panel()
        panel.select_object(_ARCHER_CONST)
        _settle(panel)
        assert panel.placing_card.portrait_source == "icon"
        assert _centre(panel.placing_card.portrait.pixmap()) == PLAYER_COLORS[1]  # the Owner combo's default
        panel.select_owner(2)
        assert _centre(panel.placing_card.portrait.pixmap()) == PLAYER_COLORS[2]
        colors = list(PLAYER_COLORS)
        colors[2] = (10, 20, 30)
        from descape import player_labels

        panel.refresh_player_labels(player_labels.DEFAULT_LABELS, colors)
        assert _centre(panel.placing_card.portrait.pixmap()) == (10, 20, 30), "the scenario's own colour"
        panel.show_unit(_entry(player_id=3, unit_const=_ARCHER_CONST))
        _settle(panel)
        assert _centre(panel.unit_portrait.pixmap()) == colors[3]
        panel.show_group([_entry(player_id=3), _entry(player_id=4, reference_id=102)])
        assert _centre(panel.unit_portrait.pixmap()) == colors[GAIA_PLAYER_ID], "mixed owners: neutral"
    finally:
        asset_source.set_install_path_override(None)


def test_catalog_rows_fill_the_player_colour_region_with_the_owners_colour(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QApplication

    from descape import asset_source, settings
    from descape.terrain_palette import PLAYER_COLORS

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    monkeypatch.setattr(settings, "get_units_catalog_mode", lambda: "icons")
    asset_source.set_install_path_override(_holed_install(tmp_path))
    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 800)
    panel.show()
    try:
        panel.select_object(_ARCHER_CONST)
        QApplication.processEvents()
        _settle(panel)
        view = panel.catalog_view
        px = view.list.iconSize().width()

        def row_centre():
            return _centre(view._items[_ARCHER_CONST].icon().pixmap(px, px))

        assert row_centre() == PLAYER_COLORS[1]
        panel.select_owner(5)
        assert row_centre() == PLAYER_COLORS[5]
    finally:
        panel.close()
        asset_source.set_install_path_override(None)


@pytest.mark.font_sensitive
@pytest.mark.parametrize("with_install", [False, True])
def test_the_placing_card_fits_the_floor(with_install, tmp_path, monkeypatch) -> None:
    """With the longest name picked: the name elides, so it cannot widen the card."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.set_install_path_override(_fake_install(tmp_path) if with_install else None)
    try:
        panel = _panel()
        longest = _longest_entry()
        panel.select_object(longest.id)
        assert panel.placing_label.text() == f"Placing: {longest.name}"
        width = panel.placing_card.minimumSizeHint().width()
        assert width <= UnitsPanel.MIN_USEFUL_WIDTH, f"placing card demands {width} px"
    finally:
        asset_source.set_install_path_override(None)


def test_refresh_thumbnails_reaches_both_cards(tmp_path) -> None:
    from descape import asset_source

    panel = _panel()
    panel.select_object(_ARCHER_CONST)
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    _settle(panel)
    asset_source.set_install_path_override(_fake_install(tmp_path))
    try:
        panel.refresh_thumbnails()
        _settle(panel)
        for card in (panel.placing_card, panel.selection_card):
            assert card.stat_rows["hp"][0].text() == ""
            assert card.portrait_source == "icon"
    finally:
        asset_source.set_install_path_override(None)


def test_the_catalog_list_keeps_its_geometry_across_picks() -> None:
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 800)
    panel.show()
    try:
        geometries = {}
        for const in (_ARCHER_CONST, _TREE_CONST, _longest_entry().id):
            panel.select_object(const)
            QApplication.processEvents()
            QApplication.processEvents()
            tree = panel.catalog_view.list
            origin = tree.mapTo(panel, tree.rect().topLeft())
            geometries[const] = (
                tree.geometry().getRect(), (origin.x(), origin.y()), panel.placing_card.geometry().getRect()
            )
        assert len(set(geometries.values())) == 1, geometries
    finally:
        panel.close()


@pytest.mark.parametrize("case", ["select_while_hidden", "scroll_after_the_pick"])
def test_the_placing_cards_portrait_is_decoded_before_the_lists_viewport(case, fake_install, monkeypatch) -> None:
    """The list asks for its viewport after the card asked: on a hidden pane
    select()'s scrollToItem does (a shown list scrolls before the card asks),
    and so does any scroll after a pick. The card is still served first."""
    from PyQt5.QtWidgets import QApplication

    from descape import catalog_thumbs, object_catalog, unit_card

    executor = conftest.ManualExecutor()
    monkeypatch.setattr(catalog_thumbs, "_thumb_pool", lambda: executor)
    town_center = 109  # far down Buildings, so select() scrolls the list
    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 700)
    panel.show()
    try:
        QApplication.processEvents()
        if case == "select_while_hidden":
            panel.hide()
        panel.select_object(town_center)
        bar = panel.catalog_view.list.verticalScrollBar()
        if case == "scroll_after_the_pick":
            bar.setValue(bar.value() - bar.pageStep())
        QApplication.processEvents()
        assert bar.value() > 0, "the list scrolled, or this proves nothing"
        for index, (_fn, _args, future) in enumerate(executor.submitted):
            if not future.done():
                executor.run(index)  # the one decode in flight before the card asked
        served = len(executor.submitted)
        deadline = 100
        while len(executor.submitted) == served and deadline:
            QApplication.processEvents()
            deadline -= 1
        _fn, (_install, icon, _building, px), _future = executor.submitted[served]
        assert (icon, px) == (object_catalog.icon_for(town_center), unit_card.PORTRAIT_PX)
    finally:
        panel.close()


def test_the_panel_stays_under_the_trigger_panels_minimum_height() -> None:
    """The left stack's minimum is the tallest page's: the placing card must not make this page that one."""
    from descape.trigger_panel import TriggerPanel

    panel = _panel()
    height = panel.minimumSizeHint().height()
    assert height < TriggerPanel().minimumSizeHint().height()


def test_the_placing_card_costs_the_catalog_pane_no_height() -> None:
    """It shares the inspector's scrolled slot with the selection card (measured 348 px -> 231 px)."""
    panel = _panel()
    catalog_pane = panel.splitter.widget(0)
    assert not catalog_pane.isAncestorOf(panel.placing_card)
    assert panel.inspector_area.isAncestorOf(panel.placing_card)


@pytest.mark.corpus
def test_the_real_install_fills_the_placing_card_for_brasidas() -> None:
    from descape import asset_source

    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    panel = _panel()
    panel.select_object(2317)
    card = panel.placing_card
    assert _shown_stats(card)["hp"] == "500"
    _settle(panel)
    assert card.portrait_source == "icon"
    assert not card.portrait.pixmap().isNull()


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
    assert panel.unit_stats_grid.isVisibleTo(panel)


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
    # GH #68: the Unit ID row counts the types (two here: archer and tree).
    assert panel.unit_field_labels["unit_const"].text() == "(mixed: 2 types)"
    assert panel.unit_field_editors["x"].text() == "(mixed)"
    combo = panel.unit_field_editors["player"]
    assert combo.currentText() == "(mixed)"
    assert combo.currentData() is None
    # Only the two archers count for rotation, and they agree.
    assert panel.unit_field_editors["rotation"].text() == "1"  # 0.5 rad = 1.27 of 16
    assert _group_rotation_note(panel)
    assert _group_rotation_note(panel).startswith("Typing a Rotation turns the 2 selected units")
    assert "The other 1 of 3" in _group_rotation_note(panel)
    assert not panel.unit_stats_grid.isVisibleTo(panel)
    assert not panel.unit_portrait.isVisibleTo(panel)


def test_group_rotation_compares_facings_not_raw_radians() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=1.0), _entry(rotation=1.0 + 1e-9)])
    assert panel.unit_field_editors["rotation"].text() == "3"


def test_group_rotation_is_mixed_across_angle_members() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    assert panel.unit_field_editors["rotation"].text() == "(mixed)"
    assert _group_rotation_note(panel) == ""


def test_group_with_no_editable_member_shows_na_and_hides_the_editor() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_GATE_CONST, rotation=7), _entry(unit_const=_WALL_CONST, rotation=2)])
    grid = panel.unit_inspector_grid
    assert not panel.unit_field_editors["rotation"].isVisibleTo(grid)
    assert panel.unit_field_labels["rotation"].isVisibleTo(grid)
    assert panel.unit_field_labels["rotation"].text() == "(n/a)"
    assert _group_rotation_note(panel).startswith("Typing a Rotation changes none of the 2 selected")


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


def test_group_note_gives_way_to_a_tooltip_for_a_single_wall() -> None:
    """GH #147: after a group, a lone wall drops the note and explains itself in the value's tooltip."""
    panel = _panel()
    panel.show_group([_entry(), _entry(unit_const=_WALL_CONST, rotation=2)])
    assert _group_rotation_note(panel)
    panel.show_unit(_entry(unit_const=_WALL_CONST, rotation=2))
    assert _group_rotation_note(panel) == ""
    assert panel.unit_field_labels["rotation"].text() == "2"
    assert "variant index" in panel.unit_field_labels["rotation"].toolTip()


_BARRACKS_CONST = 12  # INERT: a single-frame building
_STATUE_B_CONST = 1280  # VARIANT with one real variant


def test_a_single_frame_object_gets_a_tooltip_not_the_note() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_BARRACKS_CONST, rotation=7.0))
    label = panel.unit_field_labels["rotation"]
    assert label.isVisibleTo(panel.unit_inspector_grid)
    assert label.text() == "7"
    assert label.toolTip().startswith("Single-frame object")
    assert _group_rotation_note(panel) == ""


def test_a_one_variant_object_gets_its_own_tooltip() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_STATUE_B_CONST, rotation=0.0))
    assert panel.unit_field_labels["rotation"].toolTip().startswith("Only one graphic variant")
    assert _group_rotation_note(panel) == ""


def test_the_readonly_rotation_tooltip_never_outlives_its_unit() -> None:
    panel = _panel()
    label = panel.unit_field_labels["rotation"]
    panel.show_unit(_entry(unit_const=_WALL_CONST, rotation=2))
    assert label.toolTip()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    assert label.toolTip() == ""
    panel.show_unit(_entry(unit_const=_BARRACKS_CONST, rotation=7.0))
    assert label.toolTip()
    panel.show_group([_entry(unit_const=_GATE_CONST, rotation=7), _entry(unit_const=_WALL_CONST, rotation=2)])
    assert label.text() == "(n/a)"
    assert label.toolTip() == ""
    panel.show_unit(_entry(unit_const=_WALL_CONST, rotation=2))
    assert label.toolTip()
    panel.show_unit(None)
    assert label.toolTip() == ""


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
    assert _group_rotation_note(panel) == ""
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
    assert _group_rotation_note(panel) == ""


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
    assert _group_rotation_note(panel)
    note = _group_rotation_note(panel)
    assert note.startswith("Typing a Rotation turns the 1 selected unit with a facing.")
    assert "Rotate (. and ,) still steps trees" in note


def test_the_group_note_leads_with_what_a_typed_rotation_turns() -> None:
    """GH #71: "won't rotate" read as "trees can't rotate", though Rotate steps them."""
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=0.5), _entry(unit_const=_TREE_CONST, rotation=7.0)])
    note = _group_rotation_note(panel)
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
    note = _group_rotation_note(panel)
    assert note.startswith("Typing a Rotation turns the 1 selected unit with a facing.")
    assert "Rotate (R) still turns gates through their four orientations." in note


def test_the_note_for_a_group_with_nothing_typeable_says_so() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_GATE_CONST, rotation=7), _entry(unit_const=_WALL_CONST, rotation=2)])
    note = _group_rotation_note(panel)
    assert note.startswith("Typing a Rotation changes none of the 2 selected:")
    assert "Rotate (. and ,) still turns gates" in note


def test_a_tree_and_wall_group_is_variant_mode_and_the_note_counts_the_wall() -> None:
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=7.0), _entry(unit_const=_WALL_CONST, rotation=2.0)])
    spin = _rotation_spin(panel)
    assert spin.isVisibleTo(panel.unit_inspector_grid)
    assert (spin.minimum(), spin.maximum()) == (0, 41)
    assert spin.text() == "7"
    note = _group_rotation_note(panel)
    assert note.startswith("Typing a Rotation picks that variant for the 1 selected object with graphic variants.")
    assert "The other 1 of 2 (walls, cliffs, gates, single-frame objects) keep theirs." in note
    assert "won't rotate" not in note
    assert "Rotate" not in note.removeprefix("Typing a Rotation")


def test_a_tree_and_gate_variant_group_note_says_rotate_turns_the_gate() -> None:
    """The tail names only what the typed value skips: the gate, never the tree it already steps."""
    panel = _panel()
    panel.show_group([_entry(unit_const=_TREE_CONST, rotation=7.0), _entry(unit_const=_GATE_CONST)])
    note = _group_rotation_note(panel)
    assert note.startswith("Typing a Rotation picks that variant for the 1 selected object with graphic variants.")
    assert note.endswith("Rotate (. and ,) still turns gates through their four orientations.")
    assert "steps trees" not in note


# --- GH #198: the group note leads the Rotation caption's tooltip ----------

_FACING_GROUP_NOTE = (
    "Typing a Rotation turns the 1 selected unit with a facing. "
    "The other 1 of 2 (trees, walls, gates, buildings, scenery) keep theirs."
)
_VARIANT_GROUP_NOTE = (
    "Typing a Rotation picks that variant for the 1 selected object with graphic variants. "
    "The other 1 of 2 (walls, cliffs, gates, single-frame objects) keep theirs."
)
_NO_ROTATION_GROUP_NOTE = (
    "Typing a Rotation changes none of the 2 selected: walls, cliffs, "
    "gates and single-frame objects keep their stored value."
)
# Per rotation_field_mode: the group's members, and the note its tooltip leads with.
_GROUP_NOTE_CASES = {
    "facing": ([{}, {"unit_const": _BARRACKS_CONST}], _FACING_GROUP_NOTE),
    "variant": (
        [{"unit_const": _TREE_CONST, "rotation": 7.0}, {"unit_const": _WALL_CONST, "rotation": 2.0}],
        _VARIANT_GROUP_NOTE,
    ),
    "none": ([{"unit_const": _BARRACKS_CONST}, {"unit_const": _WALL_CONST, "rotation": 2.0}], _NO_ROTATION_GROUP_NOTE),
}


@pytest.mark.parametrize("case", sorted(_GROUP_NOTE_CASES))
def test_a_group_note_leads_the_rotation_captions_tooltip(case) -> None:
    members, note = _GROUP_NOTE_CASES[case]
    panel = _panel()
    panel.show_group([_entry(**member) for member in members])
    assert panel.unit_rotation_label.isVisibleTo(panel.unit_inspector_grid)
    assert panel.unit_rotation_label.toolTip() == f"{note}\n\n{_ROTATION_TOOLTIP}"


@pytest.mark.parametrize("case", sorted(_GROUP_NOTE_CASES))
def test_no_label_in_the_panel_shows_a_group_note(case) -> None:
    members, _note = _GROUP_NOTE_CASES[case]
    panel = _panel()
    panel.show_group([_entry(**member) for member in members])
    assert [label.text() for label in panel.findChildren(QLabel) if "Typing a Rotation" in label.text()] == []


@pytest.mark.parametrize("unit_const", [_ARCHER_CONST, _TREE_CONST, _WALL_CONST, _GATE_CONST, _BARRACKS_CONST])
def test_one_unit_keeps_the_rotation_captions_tooltip_as_it_was(unit_const) -> None:
    """Before a group and straight after one, whose note must not outlive it."""
    panel = _panel()
    panel.show_unit(_entry(unit_const=unit_const))
    assert panel.unit_rotation_label.toolTip() == _ROTATION_TOOLTIP
    panel.show_group([_entry(), _entry(unit_const=_BARRACKS_CONST)])
    assert panel.unit_rotation_label.toolTip() != _ROTATION_TOOLTIP
    panel.show_unit(_entry(unit_const=unit_const))
    assert panel.unit_rotation_label.toolTip() == _ROTATION_TOOLTIP


def test_a_cleared_selection_drops_the_group_note_from_the_tooltip() -> None:
    panel = _panel()
    panel.show_group([_entry(), _entry(unit_const=_BARRACKS_CONST)])
    panel.show_selection_count(0)
    assert panel.unit_rotation_label.toolTip() == _ROTATION_TOOLTIP


def _group_inspector_height(second_const: int) -> int:
    """The scroll content's height for an Archer plus `second_const`, on a fresh shown panel."""
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 900)
    panel.show()
    try:
        QApplication.processEvents()
        panel.show_group([_entry(), _entry(unit_const=second_const)])
        QApplication.processEvents()
        return panel.inspector_host.minimumHeight()
    finally:
        panel.close()


def test_a_group_note_costs_the_inspector_no_line_of_height() -> None:
    """Both groups mix two consts; only the Barracks one has a member a typed Rotation skips."""
    assert _group_inspector_height(_BARRACKS_CONST) == _group_inspector_height(_TREBUCHET_CONST)


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
        panel.garrison_cut_button,
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


def test_a_refused_pick_object_keeps_the_filter_and_the_hidden_box() -> None:
    """TASK-356: the refusal cleared the filter while the status line said "unchanged"."""
    panel = _panel()
    panel.catalog_view.filter_edit.setText("wall")
    panel.select_object(72)
    assert not panel.pick_object(-12345)
    assert panel.catalog_view.filter_edit.text() == "wall"
    assert not panel.catalog_view.show_hidden_checkbox.isChecked()
    assert panel.selected_object_const() == 72


_EDITOR_HIDDEN_CONST = 118  # Builder: in the catalog, hidden from the in-game editor


def test_pick_object_ticks_show_hidden_for_an_editor_hidden_const() -> None:
    from descape import object_catalog

    entry = object_catalog.entry(object_catalog.objects(), _EDITOR_HIDDEN_CONST)
    assert entry is not None and entry.hidden, "the premise: a catalog row the box hides"
    panel = _panel()
    panel.select_object(4)
    box = panel.catalog_view.show_hidden_checkbox
    assert not box.isChecked()

    assert panel.pick_object(_EDITOR_HIDDEN_CONST)

    assert box.isChecked()
    assert panel.selected_object_const() == _EDITOR_HIDDEN_CONST
    assert panel.catalog_view.current_value() == _EDITOR_HIDDEN_CONST


def test_select_object_now_ticks_show_hidden_itself() -> None:
    """GH #68's select() contract: the browser shows hidden rows for a hidden
    const, where ValuePickerView's select() could not land on one."""
    panel = _panel()
    panel.select_object(4)
    panel.select_object(_EDITOR_HIDDEN_CONST)
    assert panel.catalog_view.show_hidden_checkbox.isChecked()
    assert panel.selected_object_const() == _EDITOR_HIDDEN_CONST


def test_pick_object_leaves_show_hidden_alone_for_an_ordinary_const() -> None:
    panel = _panel()
    assert panel.pick_object(117)
    assert not panel.catalog_view.show_hidden_checkbox.isChecked()


# --- TASK-023: the facing dial beside the Rotation spinbox ------------------


def _dial(panel):
    return panel.unit_rotation_dial


def _dial_shown(panel) -> bool:
    return _dial(panel).isVisibleTo(panel.unit_inspector_grid)


def _send_mouse(widget, kind, point, buttons) -> None:
    """A real QMouseEvent: QTest.mouseMove carries no held button on a QWidget."""
    from PyQt5.QtCore import QEvent, QPointF
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    event_type = {"press": QEvent.MouseButtonPress, "move": QEvent.MouseMove, "release": QEvent.MouseButtonRelease}[kind]
    button = Qt.NoButton if kind == "move" else Qt.LeftButton
    local = QPointF(point)
    event = QMouseEvent(event_type, local, local, widget.mapToGlobal(point.toPoint()), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def _click(dial, point) -> None:
    _send_mouse(dial, "press", point, Qt.LeftButton)
    _send_mouse(dial, "release", point, Qt.NoButton)


def _wheel(dial, units: int) -> None:
    from PyQt5.QtCore import QPoint, QPointF
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    centre = QPointF(dial.rect().center())
    event = QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, units), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(dial, event)


def _recording_panel():
    received = []
    panel = _panel(on_unit_field=lambda spec, value: received.append((spec.field_id, value)))
    return panel, received


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param([{"unit_const": _ARCHER_CONST}], id="archer"),
        pytest.param([{"unit_const": _GATE_CONST}], id="gate"),
        pytest.param([{"unit_const": _GATE_CONST}, {"unit_const": 88}], id="gate-group"),
    ],
)
def test_the_dial_shows_for_facings_and_gates(entries) -> None:
    panel = _panel()
    if len(entries) == 1:
        panel.show_unit(_entry(**entries[0]))
    else:
        panel.show_group([_entry(**kwargs) for kwargs in entries])
    assert _dial_shown(panel)
    assert _rotation_spin(panel).isVisibleTo(panel.unit_inspector_grid)
    assert _dial(panel).accessibleName() == "Rotation dial"
    assert _dial(panel).toolTip() == _rotation_spin(panel).toolTip()
    assert _dial(panel).focusPolicy() == Qt.NoFocus


@pytest.mark.parametrize(
    ("unit_const", "editable"),
    [(_TREE_CONST, True), (_WALL_CONST, False), (_BARRACKS_CONST, False)],
    ids=["tree", "wall", "inert"],
)
def test_the_dial_hides_for_variants_and_read_only_consts(unit_const: int, editable: bool) -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_ARCHER_CONST))
    panel.show_unit(_entry(unit_const=unit_const, rotation=3.0))
    assert not _dial_shown(panel)
    assert _rotation_spin(panel).isVisibleTo(panel.unit_inspector_grid) is editable
    assert panel.unit_field_labels["rotation"].isVisibleTo(panel.unit_inspector_grid) is not editable


def test_the_dial_takes_the_selections_mode_and_count() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=_TREBUCHET_CONST, rotation=math.pi))
    assert (_dial(panel).mode, _dial(panel).count, _dial(panel).value) == ("facing", 32, 16)
    panel.show_unit(_entry(unit_const=88))  # stone gate, orientation 2
    assert (_dial(panel).mode, _dial(panel).count, _dial(panel).value) == ("gate", 4, 2)
    panel.show_unit(_entry(rotation=3 * 2 * math.pi / 16))
    assert (_dial(panel).mode, _dial(panel).count, _dial(panel).value) == ("facing", 16, 3)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("facing", [0, 3, 8, 13])
def test_a_click_on_a_spoke_sets_that_facing_once(flat: bool, facing: int) -> None:
    panel, received = _recording_panel()
    panel.set_rotation_projection(flat)
    panel.show_unit(_entry(rotation=5 * 2 * math.pi / 16))
    _click(_dial(panel), _dial(panel).spoke_point(facing))
    expected = [] if facing == 5 else [("rotation", facing)]
    assert received == expected
    assert _rotation_spin(panel).value() == facing
    assert _dial(panel).value == facing


def test_a_click_on_the_current_facing_records_nothing() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=5 * 2 * math.pi / 16))
    _click(_dial(panel), _dial(panel).spoke_point(5))
    assert received == []


def test_a_drag_across_three_spokes_commits_once_with_the_release_spoke() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=0.0))
    dial = _dial(panel)
    _send_mouse(dial, "press", dial.spoke_point(2), Qt.LeftButton)
    _send_mouse(dial, "move", dial.spoke_point(3), Qt.LeftButton)
    assert dial.shown_value == 3, "the drag previews the spoke under the cursor"
    assert received == [] and _rotation_spin(panel).value() == 0
    _send_mouse(dial, "move", dial.spoke_point(4), Qt.LeftButton)
    _send_mouse(dial, "release", dial.spoke_point(4), Qt.NoButton)
    assert received == [("rotation", 4)]
    assert dial.shown_value == 4


def test_a_click_on_a_gate_axis_picks_that_orientation_from_either_end() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(unit_const=_GATE_CONST))  # orientation 0
    dial = _dial(panel)
    _click(dial, dial.spoke_point(3))
    _click(dial, _opposite(dial, dial.spoke_point(1)))
    assert received == [("rotation", 3), ("rotation", 1)]


def test_a_press_at_the_centre_commits_nothing() -> None:
    from PyQt5.QtCore import QPointF

    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=0.0))
    dial = _dial(panel)
    _click(dial, QPointF(dial.width() / 2, dial.height() / 2))
    assert received == []


def test_wheel_up_from_the_last_facing_wraps_one_call_per_notch() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=15 * 2 * math.pi / 16))
    _wheel(_dial(panel), 120)
    _wheel(_dial(panel), 120)
    _wheel(_dial(panel), -120)
    assert received == [("rotation", 0), ("rotation", 1), ("rotation", 0)]


def test_a_mixed_group_dial_has_no_pointer_and_a_click_sets_one_value() -> None:
    panel, received = _recording_panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    assert _rotation_spin(panel).text() == "(mixed)"
    assert _dial(panel).value is None and _dial(panel).shown_value is None
    _click(_dial(panel), _dial(panel).spoke_point(6))
    assert received == [("rotation", 6)]


def test_populating_moves_the_dial_and_records_nothing() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=3 * 2 * math.pi / 16))
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    panel.show_unit(_entry(unit_const=_GATE_CONST))
    panel.show_group([_entry(rotation=math.pi), _entry(rotation=math.pi)])
    assert _dial(panel).value == 8
    panel.show_unit(None)
    assert received == []


def test_set_rotation_projection_moves_the_spokes_and_records_nothing() -> None:
    panel, received = _recording_panel()
    panel.show_unit(_entry(rotation=0.0))
    iso = _dial(panel).spoke_point(2)
    panel.set_rotation_projection(True)
    flat = _dial(panel).spoke_point(2)
    assert (iso.x(), iso.y()) != pytest.approx((flat.x(), flat.y()), abs=2)
    assert _dial(panel).flat is True
    assert received == []


def _dial_ink(panel):
    """The dial painted on its own, as a numpy array, with the panel's highlight colour."""
    from PyQt5.QtGui import QImage, QPalette

    from testkit.qt_capture import qimage_rgb888_to_array

    dial = _dial(panel)
    image = dial.grab().toImage().convertToFormat(QImage.Format_RGB888)
    highlight = dial.palette().color(QPalette.Highlight)
    return qimage_rgb888_to_array(image), (highlight.red(), highlight.green(), highlight.blue())


def _highlight_near(panel, point, radius=2) -> bool:
    array, rgb = _dial_ink(panel)
    x, y = round(point.x()), round(point.y())
    patch = array[max(0, y - radius): y + radius + 1, max(0, x - radius): x + radius + 1, :3]
    return bool((abs(patch.astype(int) - rgb).sum(axis=2) < 30).any())


@pytest.mark.parametrize("flat", [False, True])
def test_the_pointer_is_painted_along_the_current_spoke(flat: bool) -> None:
    panel = _panel()
    panel.set_rotation_projection(flat)
    panel.show_unit(_entry(rotation=3 * 2 * math.pi / 16))
    dial = _dial(panel)
    assert _highlight_near(panel, dial.spoke_point(3, 0.5))
    assert not _highlight_near(panel, dial.spoke_point(11, 0.5))


def test_a_mixed_group_paints_no_pointer() -> None:
    panel = _panel()
    panel.show_group([_entry(rotation=0.5), _entry(rotation=1.5)])
    array, rgb = _dial_ink(panel)
    assert not (abs(array[:, :, :3].astype(int) - rgb).sum(axis=2) < 30).any()


def _opposite(dial, point):
    from PyQt5.QtCore import QPointF

    return QPointF(dial.width() - point.x(), dial.height() - point.y())


def test_the_gate_dial_highlights_the_current_axis_end_to_end() -> None:
    panel = _panel()
    panel.show_unit(_entry(unit_const=88))  # orientation 2
    dial = _dial(panel)
    end = dial.spoke_point(2, 0.6)
    assert _highlight_near(panel, end)
    assert _highlight_near(panel, _opposite(dial, end))
    assert not _highlight_near(panel, dial.spoke_point(0, 0.6))


def _shown_panel(entry):
    from PyQt5.QtWidgets import QApplication

    panel = _panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 900)
    panel.show()
    QApplication.processEvents()
    panel.show_unit(entry)
    QApplication.processEvents()
    return panel


def _row_height(panel, field_id: str) -> int:
    """A grid row's laid-out height: its caption fills the cell vertically."""
    return panel.unit_field_captions[field_id].height()


@pytest.mark.font_sensitive
@pytest.mark.parametrize("unit_const", [_ARCHER_CONST, _GATE_CONST])
def test_the_dial_row_fits_the_floor_with_no_horizontal_scrollbar(unit_const: int) -> None:
    panel = _shown_panel(_entry(unit_const=unit_const))
    try:
        assert _dial(panel).isVisible()
        assert panel.inspector_area.horizontalScrollBar().maximum() == 0
        assert panel.minimumSizeHint().width() <= UnitsPanel.MIN_USEFUL_WIDTH
        dial = _dial(panel).geometry()
        assert panel.unit_rotation_cell.rect().contains(dial)
    finally:
        panel.close()


@pytest.mark.parametrize("unit_const", [_WALL_CONST, _BARRACKS_CONST])
def test_nothing_covers_a_read_only_rotation_label(unit_const: int) -> None:
    """The label's GH #147 tooltip and text selection need the mouse: no empty dial cell on top of it."""
    panel = _shown_panel(_entry(unit_const=unit_const, rotation=3.0))
    try:
        label = panel.unit_field_labels["rotation"]
        assert label.isVisible()
        centre = label.mapTo(panel.unit_inspector_grid, label.rect().center())
        assert panel.unit_inspector_grid.childAt(centre) is label
    finally:
        panel.close()


def test_a_variant_or_read_only_rotation_row_is_as_tall_as_before_the_dial() -> None:
    """The hidden dial takes no height: a tree's row is as tall as the X spinbox row, a wall's as Reference ID's label row."""
    tree = _shown_panel(_entry(unit_const=_TREE_CONST, rotation=3.0))
    wall = _shown_panel(_entry(unit_const=_WALL_CONST, rotation=3.0))
    archer = _shown_panel(_entry())
    try:
        assert _row_height(tree, "rotation") == _row_height(tree, "x") > 0
        assert _row_height(wall, "rotation") == _row_height(wall, "reference_id") > 0
        assert _row_height(archer, "rotation") >= _dial(archer).height()
    finally:
        for panel in (tree, wall, archer):
            panel.close()


# --- GH #154: a group's X/Y/Z steps shift every member ----------------------


def _step_panel(**kwargs):
    """A map-sized panel recording both callbacks, in call order, into one list."""
    calls = []
    panel = _map_panel(
        on_unit_field=lambda spec, value: calls.append(("set", spec.field_id, value)),
        on_unit_field_step=lambda spec, delta: calls.append(("step", spec.field_id, delta)),
        **kwargs,
    )
    return panel, calls


def _mixed_x_group(panel) -> None:
    panel.show_group([_entry(x=4.5), _entry(x=9.5)])
    assert panel.unit_field_editors["x"].text() == "(mixed)"


def test_a_group_step_reports_a_delta_and_leaves_the_mixed_field_alone() -> None:
    panel, calls = _step_panel()
    _mixed_x_group(panel)
    spin = panel.unit_field_editors["x"]
    spin.stepBy(1)
    spin.stepBy(-1)
    assert calls == [("step", "x", 1.0), ("step", "x", -1.0)]
    assert spin.text() == "(mixed)"


def test_both_arrows_are_live_on_the_group_sentinel_but_not_on_a_single_units_minimum() -> None:
    from PyQt5.QtWidgets import QAbstractSpinBox

    panel, _calls = _step_panel()
    _mixed_x_group(panel)
    spin = panel.unit_field_editors["x"]
    assert spin.stepEnabled() == QAbstractSpinBox.StepUpEnabled | QAbstractSpinBox.StepDownEnabled
    panel.show_unit(_entry(x=0.0))
    assert spin.value() == spin.minimum() == 0.0
    assert spin.stepEnabled() == QAbstractSpinBox.StepUpEnabled


def _wheel_spin(spin, units: int) -> None:
    from PyQt5.QtCore import QPoint, QPointF
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    centre = QPointF(spin.rect().center())
    event = QWheelEvent(centre, centre, QPoint(0, 0), QPoint(0, units), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(spin, event)


def test_keys_and_the_wheel_on_a_shown_group_field_report_relative_steps() -> None:
    from PyQt5.QtWidgets import QApplication

    panel, calls = _step_panel()
    panel.resize(UnitsPanel.MIN_USEFUL_WIDTH, 900)
    panel.show()
    try:
        QApplication.processEvents()
        _mixed_x_group(panel)
        spin = panel.unit_field_editors["x"]
        for key in (Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown):
            QTest.keyClick(spin, key)
        _wheel_spin(spin, 120)
        _wheel_spin(spin, -120)
        assert calls == [("step", "x", -1.0), ("step", "x", 1.0), ("step", "x", -10.0), ("step", "x", 1.0),
                         ("step", "x", -1.0)]
        assert spin.text() == "(mixed)"
    finally:
        panel.close()


def test_a_group_that_agrees_still_steps_and_the_panel_keeps_the_value() -> None:
    panel, calls = _step_panel()
    panel.show_group([_entry(x=4.5, y=8.5), _entry(x=4.5, y=12.5)])
    spin = panel.unit_field_editors["x"]
    assert spin.value() == 4.5
    spin.stepBy(1)
    assert calls == [("step", "x", 1.0)]
    assert spin.value() == 4.5


def test_z_steps_relatively_and_rotation_still_sets_a_value() -> None:
    panel, calls = _step_panel()
    panel.show_group([_entry(z=1.0), _entry(z=3.0)])
    panel.unit_field_editors["z"].stepBy(1)
    panel.unit_field_editors["rotation"].stepBy(1)
    assert calls == [("step", "z", 1.0), ("set", "rotation", 1)]


def test_a_single_unit_steps_absolutely_as_before() -> None:
    panel, calls = _step_panel()
    panel.show_unit(_entry(x=4.5))
    panel.unit_field_editors["x"].stepBy(1)
    assert calls == [("set", "x", 5.5)]


def test_leaving_group_mode_drops_relative_steps_and_the_tooltip() -> None:
    panel, _calls = _step_panel()
    coords = [panel.unit_field_editors[field_id] for field_id in ("x", "y", "z")]
    for leave in (lambda: panel.show_unit(_entry()), panel.clear):
        _mixed_x_group(panel)
        assert all(spin.relative_steps and spin.toolTip().startswith("Arrows move every") for spin in coords)
        leave()
        assert not any(spin.relative_steps or spin.toolTip() for spin in coords)


def test_text_typed_but_not_committed_lands_before_the_step() -> None:
    panel, calls = _step_panel()
    _mixed_x_group(panel)
    spin = panel.unit_field_editors["x"]
    spin.lineEdit().selectAll()
    QTest.keyClicks(spin, "50")
    assert calls == [], "keyboard tracking is off: nothing lands until a commit"
    QTest.keyClick(spin, Qt.Key_Up)
    assert calls == [("set", "x", 50.0), ("step", "x", 1.0)]


def test_a_populate_reports_no_step() -> None:
    panel, calls = _step_panel()
    _mixed_x_group(panel)
    panel._populating = True
    try:
        panel.unit_field_editors["x"].stepBy(1)
    finally:
        panel._populating = False
    assert calls == []


# --- GH #32: GAIA-only objects and the Owner combo ------------------------------

_MACAW = 816
_GOLD_MINE = 66


def _listed(panel) -> set[int]:
    view = panel.catalog_view
    return {view.list.item(i).data(Qt.UserRole) for i in range(view.list.count()) if not view.list.item(i).isHidden()}


def test_hide_mode_lists_gaia_only_objects_only_for_gaia() -> None:
    panel = _panel()
    assert panel.owner_id() == 1
    assert _MACAW not in _listed(panel) and _ARCHER_CONST in _listed(panel)
    panel.select_owner(GAIA_PLAYER_ID)
    assert {_MACAW, _ARCHER_CONST} <= _listed(panel)
    panel.catalog_view.category_buttons["Others"].click()
    assert _GOLD_MINE in _listed(panel)
    panel.select_owner(2)
    assert _GOLD_MINE not in _listed(panel)


def test_keep_mode_lists_them_for_every_owner() -> None:
    from descape import settings

    settings.set_gaia_only_catalog_mode("keep")
    panel = _panel()
    assert _MACAW in _listed(panel)
    panel.select_object(_MACAW)
    panel.select_owner(3)
    assert panel.selected_object_const() == _MACAW, "keep mode never drops the pick"


def test_the_setting_takes_effect_on_refresh_owner_filter() -> None:
    from descape import settings

    panel = _panel()
    assert _MACAW not in _listed(panel)
    settings.set_gaia_only_catalog_mode("keep")
    panel.refresh_owner_filter()
    assert _MACAW in _listed(panel)
    settings.set_gaia_only_catalog_mode("hide")
    panel.refresh_owner_filter()
    assert _MACAW not in _listed(panel)


def test_a_gaia_only_pick_is_dropped_when_owner_turns_to_a_player() -> None:
    pending = []
    panel = _panel(on_pending_changed=lambda: pending.append(panel.selected_object_const()))
    panel.select_owner(GAIA_PLAYER_ID)
    panel.select_object(_MACAW)
    assert panel.selected_object_const() == _MACAW
    panel.select_owner(1)
    assert panel.selected_object_const() is None, "no ghost for an object Player 1 cannot own"
    assert pending[-1] is None
    assert "(none)" in panel.placing_label.text()


def test_a_dropped_gaia_only_row_picks_again_on_one_click_after_owner_returns_to_gaia() -> None:
    panel = _panel()
    panel.select_owner(GAIA_PLAYER_ID)
    panel.select_object(_MACAW)
    panel.select_owner(1)
    panel.select_owner(GAIA_PLAYER_ID)
    assert panel.selected_object_const() is None
    assert panel.catalog_view.current_value() is None, "the dropped row does not come back looking picked"
    panel.catalog_view.select(_MACAW)
    assert panel.selected_object_const() == _MACAW


def test_a_player_pick_survives_an_owner_change() -> None:
    panel = _panel()
    panel.select_object(_ARCHER_CONST)
    panel.select_owner(GAIA_PLAYER_ID)
    panel.select_owner(4)
    assert panel.selected_object_const() == _ARCHER_CONST


def test_pick_object_of_a_gaia_only_const_sets_owner_to_gaia_in_hide_mode() -> None:
    panel = _panel()
    assert panel.pick_object(_MACAW)
    assert panel.owner_id() == GAIA_PLAYER_ID
    assert panel.selected_object_const() == _MACAW
    # A player unit keeps the owner.
    panel.select_owner(5)
    assert panel.pick_object(_ARCHER_CONST)
    assert panel.owner_id() == 5


def test_pick_object_leaves_the_owner_alone_in_keep_mode() -> None:
    from descape import settings

    settings.set_gaia_only_catalog_mode("keep")
    panel = _panel()
    assert panel.pick_object(_MACAW)
    assert panel.owner_id() == 1


def test_a_switch_to_return_of_rome_hides_its_own_gaia_only_set() -> None:
    from descape import game_dataset

    lion, gazelle = 1029, 22
    panel = _panel()
    game_dataset.set_active(game_dataset.ROR)
    panel.reload_catalog()
    assert not {lion, gazelle} & _listed(panel)
    panel.select_owner(GAIA_PLAYER_ID)
    assert {lion, gazelle} <= _listed(panel)


def _player_rows_offered(combo) -> dict[int, bool]:
    """player_id -> shown in the popup and selectable."""
    model, view = combo.model(), combo.view()
    return {
        combo.itemData(row): model.item(row).isEnabled() and not view.isRowHidden(row)
        for row in range(combo.count())
        if combo.itemData(row) is not None
    }


def test_the_owner_field_offers_only_gaia_for_a_gaia_only_selection() -> None:
    panel = _panel()
    combo = panel.unit_field_editors["player"]
    panel.show_unit(_entry(player_id=GAIA_PLAYER_ID, unit_const=_MACAW))
    rows = _player_rows_offered(combo)
    assert rows[GAIA_PLAYER_ID] and not any(rows[p] for p in range(1, 9))
    panel.show_unit(_entry(player_id=1, unit_const=_ARCHER_CONST))
    assert all(_player_rows_offered(combo).values())
    # A group hides them only when every member is GAIA-only.
    panel.show_group([_entry(player_id=0, unit_const=_MACAW), _entry(player_id=1, unit_const=_ARCHER_CONST)])
    assert all(_player_rows_offered(combo).values())
    panel.show_group([_entry(player_id=0, unit_const=_MACAW), _entry(player_id=2, unit_const=_GOLD_MINE)])
    assert not _player_rows_offered(combo)[3]


def test_a_bad_files_player_owned_deer_still_shows_its_owner() -> None:
    panel = _panel()
    combo = panel.unit_field_editors["player"]
    panel.show_unit(_entry(player_id=2, unit_const=65))
    assert combo.currentData() == 2
    assert _player_rows_offered(combo)[GAIA_PLAYER_ID] and not _player_rows_offered(combo)[2]


def test_the_garrison_owner_combo_offers_only_gaia_for_gaia_only_rows() -> None:
    panel = _panel()
    rows = [("Deer", "Player 1", 501), ("Villager", "Player 1", 502)]
    panel.show_garrison(rows, 10, gaia_only=[501])
    tree = panel.garrison_tree
    tree.topLevelItem(0).setSelected(True)
    assert not _player_rows_offered(panel.garrison_owner_combo)[2]
    tree.topLevelItem(1).setSelected(True)
    assert _player_rows_offered(panel.garrison_owner_combo)[2], "a mixed selection keeps them"
