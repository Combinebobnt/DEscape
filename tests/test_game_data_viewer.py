"""GH #67 in the window: a Return of Rome file opens in RoR game data and
editable, Settings > Advanced > Game data overrides the display and locks a
mismatched document, which is never written, and a clipboard block never
crosses games. The wall menu, cliff families and Villager (Random) follow
every open, not only the Settings switch.

The RoR fixture is testkit.ror_fixture's: the blank template with a RoR
header (variant 2, DLC list [11]); the user's RoR samples are local-only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import game_dataset, player_fields, scenario_io, settings, unit_sprites
from descape.options_model import players_write_supported
from descape.region_clipboard import RegionBlock
from testkit.ror_fixture import ROR_DLCS, ror_copy

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


@pytest.fixture()
def ror_path(tmp_path):
    return ror_copy(tmp_path)


@pytest.fixture()
def window():
    window = conftest.blank_window(load=False)
    yield window
    conftest.close_window(window)


def _logged(window, monkeypatch) -> list[str]:
    lines: list[str] = []
    real = window._log_status
    monkeypatch.setattr(window, "_log_status", lambda message: (lines.append(message), real(message)))
    return lines


def test_the_fixture_reads_as_return_of_rome(ror_path):
    loaded = scenario_io.load_map_and_units(ror_path)
    assert loaded.game_variant == game_dataset.ROR_VARIANT
    assert scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH).game_variant == 1


def test_the_fixture_has_a_ror_dlc_list(ror_path):
    loaded = scenario_io.load_map_and_units(ror_path)
    assert loaded._scenario.sections["FileHeader"].retriever_map["unknown_numbers"].data == list(ROR_DLCS)


def test_a_ror_file_opens_in_ror_game_data_and_editable(window, ror_path):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    window.mode_combo.setCurrentText("Terrain")
    assert window.draw_action.isEnabled()
    window.load_scenario(ror_path)
    assert game_dataset.is_ror()
    assert settings.get_game_data() == settings.GAME_DATA_DEFAULT, "Auto, not an override"
    assert window.game_data_lock_reason() is None
    assert window.game_data_banner.isHidden()
    assert "Return of Rome" in window.windowTitle()
    assert window.save_action.isEnabled() and window.save_as_action.isEnabled()
    window.mode_combo.setCurrentText("Terrain")
    assert window.draw_action.isEnabled()
    assert window._ensure_unit_edits() is not None
    assert {"civilization", "starting_age", "food"} <= window._editable_player_fields()


def _wall_menu(window) -> list[int]:
    return [action.data() for action in window.wall_pick_button.menu().actions()]


def _cliff_consts(window) -> set[int]:
    return {piece.unit_const for pieces in window._cliff_families.values() for piece in pieces}


def test_the_unit_tool_menus_follow_each_open(window, ror_path):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert 2678 in _wall_menu(window)
    assert 2190 in _cliff_consts(window)
    window._set_last_wall(2678)
    family = window.cliff_family_combo.findData(unit_sprites.graphic_map()[264]["file_name"])
    window.cliff_family_combo.setCurrentIndex(family)
    window.cliff_piece_combo.setCurrentIndex(2)
    piece = window.cliff_piece_combo.currentData()

    window.load_scenario(ror_path)
    assert window.cliff_piece_combo.currentData() == piece, "a family and piece both datasets have stay picked"
    assert _wall_menu(window) == [72, 117, 155, 231], "the RoR editor's walls, by an open, not set_game_data()"
    assert window._last_wall_const == 117 and "Medium Wall" in window.wall_pick_button.text()
    assert _cliff_consts(window) == set(unit_sprites.cliff_consts())
    assert window.cliff_family_combo.count() == len(window._cliff_families)
    assert all(not window.cliff_family_combo.itemData(i).startswith("const_") for i in range(window.cliff_family_combo.count()))
    assert -1 in window.units_panel.catalog_view._by_id, "Villager (Random)"

    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert 2678 in _wall_menu(window) and 2190 in _cliff_consts(window)
    assert -1 in window.units_panel.catalog_view._by_id


def test_a_ror_file_saves_its_edits_and_keeps_its_header(window, ror_path, tmp_path):
    window.load_scenario(ror_path)
    model = window._ensure_unit_edits()
    with window._unit_edit(model, "Add Hoplite", [1]):
        hoplite = model.add(1, 93, 20.5, 20.5)
        model.set_rotation(hoplite, 1.0)
    specs = {spec.field_id: spec for spec in player_fields.specs_for(window.scenario)}
    window.set_player_field(specs["civilization"], 1, "EGYPTIAN-CIV")
    window.set_player_field(specs["starting_age"], 1, 4)
    from descape.messages_fields import MESSAGE_FIELDS

    window.set_message_field(MESSAGE_FIELDS[0].field_id, "ror instructions")
    window.mode_combo.setCurrentText("Players")
    spin = window.players_panel.player_count_spin
    assert spin.isEnabled()
    spin.setValue(3)
    window.trigger_structural_edit("new", [])
    out = tmp_path / "out.aoe2scenario"
    window.scenario.path = out
    window._untitled = False
    window.save()
    assert out.is_file() and not window.edit_history.is_dirty

    reloaded = scenario_io.load_map_and_units(out)
    header = reloaded._scenario.sections["FileHeader"].retriever_map
    assert reloaded.game_variant == game_dataset.ROR_VARIANT
    assert header["unknown_numbers"].data == list(ROR_DLCS)
    units = [u for u in reloaded.unit_manager.get_all_units() if u.unit_const == 93]
    assert [u.rotation for u in units] == [1.0]
    civ = player_fields.specs_for(reloaded)
    by_id = {spec.field_id: spec for spec in civ}
    assert player_fields.current_value(reloaded, by_id["civilization"], 1) == "EGYPTIAN-CIV"
    assert player_fields.current_value(reloaded, by_id["starting_age"], 1) == 4
    assert players_write_supported(reloaded)
    assert "ror instructions" in str(reloaded._scenario.sections["Messages"].retriever_map[MESSAGE_FIELDS[0].retriever].data)
    assert player_fields.defined_player_count(reloaded) == 3
    from descape.trigger_model import TriggerEditModel

    assert TriggerEditModel(reloaded).trigger_count == 1
    assert reloaded.player_art[1] == (1, 4), "Egyptian art at Bronze Age"


def test_ror_starting_ages_show_ror_names_in_the_players_panel(window, ror_path):
    from PyQt5.QtWidgets import QComboBox

    window.load_scenario(ror_path)
    window.mode_combo.setCurrentText("Players")
    labels = [
        [combo.itemText(i) for i in range(combo.count())]
        for combo in window.players_panel.findChildren(QComboBox)
    ]
    assert ["Stone Age", "Tool Age", "Bronze Age", "Iron Age", "Post-Iron Age"] in labels


def test_an_aoe2_file_is_unlocked_and_reopening_one_drops_ror(window, ror_path):
    window.load_scenario(ror_path)
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert game_dataset.active() == game_dataset.AOE2
    assert window.game_data_lock_reason() is None
    assert window.game_data_banner.isHidden()
    assert "Return of Rome" not in window.windowTitle()
    assert window.save_action.isEnabled()


def test_the_setting_switches_the_data_locks_and_applies_to_the_next_open(window):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    catalog = window.units_panel.catalog_view
    assert catalog._by_id[93].name != "Hoplite"
    window.set_game_data(game_dataset.ROR)
    assert game_dataset.is_ror()
    assert settings.get_game_data() == game_dataset.ROR
    assert catalog._by_id[93].name == "Hoplite", "the Units catalog was rebuilt from the RoR table"
    reason = window.game_data_lock_reason()
    assert "Age of Empires II file" in reason and "Settings > Advanced" in reason
    assert window.scenario.game_variant == 1, "display only: the file's own flag is untouched"
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert game_dataset.active() == game_dataset.AOE2 and window.game_data_lock_reason() is None

    window.set_game_data(game_dataset.ROR)
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert game_dataset.is_ror(), "the setting outlives one document"
    assert window.game_data_lock_reason() is not None
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert catalog._by_id[93].name != "Hoplite"


def test_the_advanced_tab_combo_saves_and_applies_the_setting():
    dialog, window = conftest.dialog_and_window()
    try:
        window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
        combo = dialog.game_data_combo
        assert combo.currentData() == settings.GAME_DATA_DEFAULT
        assert [combo.itemData(i) for i in range(combo.count())] == [cid for cid, _label in settings.GAME_DATA_CHOICES]
        combo.setCurrentIndex(combo.findData(game_dataset.ROR))
        assert game_dataset.is_ror() and window.game_data_lock_reason() is not None
        settings._game_data = None
        assert settings.get_game_data() == game_dataset.ROR, "persisted to config.yaml"
        combo.setCurrentIndex(combo.findData(settings.GAME_DATA_DEFAULT))
        assert game_dataset.active() == game_dataset.AOE2
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_switch_keeps_a_pick_the_new_catalog_has_and_drops_one_it_lacks(window):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    panel = window.units_panel
    assert panel.pick_object(82)
    window.set_game_data(game_dataset.ROR)
    assert panel.selected_object_const() == 82, "the Castle is a hidden RoR object"
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert panel.pick_object(1808)
    window.set_game_data(game_dataset.ROR)
    assert panel.selected_object_const() is None, "Mule Cart's RoR slot is empty"
    assert "(none)" in panel.placing_label.text()
    window.set_game_data(settings.GAME_DATA_DEFAULT)


def test_the_setting_switches_with_no_document_open(window):
    window.set_game_data(game_dataset.ROR)
    assert game_dataset.is_ror()
    assert _wall_menu(window) == [72, 117, 155, 231]
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert 2678 in _wall_menu(window)


def test_view_has_no_game_data_menu(window):
    view = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert all("Game Data" not in action.text().replace("&", "") for action in view.actions())


@pytest.mark.parametrize("shown", ["aoe2_on_ror", "ror_on_aoe2"])
def test_save_save_as_and_autosave_refuse_a_mismatched_document(window, ror_path, monkeypatch, shown):
    from descape import viewer

    if shown == "aoe2_on_ror":
        window.set_game_data(game_dataset.AOE2)
        window.load_scenario(ror_path)
    else:
        window.set_game_data(game_dataset.ROR)
        window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert window.game_data_lock_reason() is not None
    assert not window.save_action.isEnabled()
    assert window._ensure_unit_edits() is None
    assert window._editable_player_fields() == frozenset()
    writes = []
    monkeypatch.setattr(viewer, "write_scenario", lambda *args, **kwargs: writes.append(args))
    monkeypatch.setattr(viewer.QMessageBox, "information", lambda *args, **kwargs: None)
    asked = []
    monkeypatch.setattr(viewer.QFileDialog, "getSaveFileName", lambda *args, **kwargs: asked.append(args) or ("", ""))
    lines = _logged(window, monkeypatch)
    window.save()
    window.save_as()
    assert not asked, "Save As refuses before asking for a path"
    # A dirty document with autosave on still writes no slot.
    monkeypatch.setattr(settings, "get_autosave_enabled", lambda: True)
    tiles = window.scenario.map_manager.terrain
    window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", 15))
    window._autosave_tick()
    assert writes == []
    assert sum(line.startswith("Save refused.") for line in lines) == 2


def test_a_region_copied_under_the_other_game_is_not_pasted(window, monkeypatch):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    block = RegionBlock(1, 1, (15,), (0,), (-1,), (), dataset=game_dataset.ROR)
    window._clipboard_history.push(block, _thumbnail(), "1x1")
    window._hover_tile = (5, 5)
    lines = _logged(window, monkeypatch)
    before = window.scenario.map_manager.terrain[5 * window.scenario.map_manager.map_width + 5].terrain_id
    window.paste_region()
    assert any(line.startswith("Paste refused: Copied with Return of Rome") for line in lines)
    assert window.scenario.map_manager.terrain[5 * window.scenario.map_manager.map_width + 5].terrain_id == before
    assert not window.edit_history.is_dirty


def test_a_copy_from_a_locked_view_keeps_its_files_game(window, ror_path, monkeypatch):
    """Shown with AoE2 data, a RoR file's ids are still RoR's: the copy may not
    land in an AoE2 file. And an AoE2 file's copy shown as RoR pastes into AoE2."""
    window.set_game_data(game_dataset.AOE2)
    window.load_scenario(ror_path)
    assert window.game_data_lock_reason() is not None
    window._region = (0, 0, 2, 2)
    window.copy_region()
    assert window._clipboard_history.active_block.dataset == game_dataset.ROR
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    assert window.game_data_lock_reason() is None
    window._hover_tile = (5, 5)
    lines = _logged(window, monkeypatch)
    window.paste_region()
    assert any(line.startswith("Paste refused: Copied with Return of Rome") for line in lines)
    assert not window.edit_history.is_dirty

    window.set_game_data(game_dataset.ROR)
    assert window.game_data_lock_reason() is not None
    window._region = (0, 0, 2, 2)
    window.copy_region()
    assert window._clipboard_history.active_block.dataset == game_dataset.AOE2
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert window._cross_game_refusal(window._clipboard_history.active_block) == ""


def test_a_block_records_the_game_data_it_was_copied_under():
    assert RegionBlock(1, 1, (0,), (0,), (-1,), ()).dataset == game_dataset.AOE2
    game_dataset.set_active(game_dataset.ROR)
    assert RegionBlock(1, 1, (0,), (0,), (-1,), ()).dataset == game_dataset.ROR


def test_a_trigger_block_from_the_other_game_is_refused(window):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    block = SimpleNamespace(dataset=game_dataset.ROR, scenario_version=window.scenario.scenario_version, source_doc_id="x")
    assert window._paste_refusal(block).startswith("Copied with Return of Rome game data")


def _thumbnail():
    import numpy as np

    return np.zeros((1, 1, 3), dtype=np.uint8)


def test_a_kept_hidden_pick_ticks_the_show_hidden_box_and_a_reused_id_is_dropped(window):
    window.load_scenario(scenario_io.BLANK_TEMPLATE_PATH)
    panel = window.units_panel
    box = panel.catalog_view.show_hidden_checkbox
    box.setChecked(False)
    assert panel.pick_object(82) and not box.isChecked(), "the Castle is visible in AoE2"
    window.set_game_data(game_dataset.ROR)
    assert panel.selected_object_const() == 82 and box.isChecked(), "hidden in RoR: the box shows it"
    window.set_game_data(settings.GAME_DATA_DEFAULT)
    assert panel.pick_object(93)
    window.set_game_data(game_dataset.ROR)
    assert panel.selected_object_const() is None, "93 is Spearman in AoE2 and Hoplite in RoR"
    window.set_game_data(settings.GAME_DATA_DEFAULT)
