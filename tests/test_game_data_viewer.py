"""GH #67 in the window: a Return of Rome file opens in RoR game data and
read-only, Settings > Advanced > Game data overrides the display, a locked
document is never written, and a clipboard block never crosses games.

The RoR fixture is the blank template with its header's variant field
(FileHeader.unknown_value_2) set to 2, built here: the user's RoR samples are
local-only. Only the variant moves, so the file is otherwise byte-identical."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from descape import game_dataset, scenario_io, settings
from descape.region_clipboard import RegionBlock

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _ror_copy(tmp_path):
    raw = bytearray(scenario_io.BLANK_TEMPLATE_PATH.read_bytes())
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    spans = scenario_io._header_field_spans(loaded.header_bytes, loaded._scenario.sections["FileHeader"].retriever_map)
    start, end = spans["unknown_value_2"]
    assert end - start == 4 and struct.unpack_from("<I", raw, start)[0] == 1
    struct.pack_into("<I", raw, start, game_dataset.ROR_VARIANT)
    path = tmp_path / "ror_blank.aoe2scenario"
    path.write_bytes(bytes(raw))
    return path


@pytest.fixture()
def ror_path(tmp_path):
    return _ror_copy(tmp_path)


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


def test_a_ror_file_opens_in_ror_game_data_and_locked(window, ror_path):
    window.load_scenario(ror_path)
    assert game_dataset.is_ror()
    assert settings.get_game_data() == settings.GAME_DATA_DEFAULT, "Auto, not an override"
    reason = window.game_data_lock_reason()
    assert reason is not None and "Return of Rome" in reason
    assert window.game_data_banner.text() == reason
    assert not window.game_data_banner.isHidden()
    assert "Return of Rome" in window.windowTitle()
    assert not window.save_action.isEnabled() and not window.save_as_action.isEnabled()
    assert not window.draw_action.isEnabled()
    assert window._ensure_unit_edits() is None
    assert window._editable_player_fields() == frozenset()


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


def test_view_has_no_game_data_menu(window):
    view = next(a.menu() for a in window.menuBar().actions() if a.text() == "&View")
    assert all("Game Data" not in action.text().replace("&", "") for action in view.actions())


def test_save_save_as_and_autosave_refuse_a_locked_document(window, ror_path, monkeypatch):
    from descape import viewer

    window.load_scenario(ror_path)
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
