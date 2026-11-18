"""descape.settings_snapshot: the keys each Settings tab owns, and the snapshot
behind the dialog's Cancel and Reset (GH #191). Qt-free."""

from __future__ import annotations

import yaml

from descape import settings, settings_file, settings_snapshot, tool_options
from descape.settings_snapshot import ABSENT, DIALOG_TAB_KEYS

_TOOLS_LEGACY = tuple(key for key, _name in settings._LEGACY_TOOL_OPTION_KEYS)


def _config() -> dict:
    path = settings.CONFIG_PATH
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def _write(config: dict) -> None:
    settings.CONFIG_PATH.write_text(yaml.safe_dump(config))
    settings_snapshot._clear_memos(settings_snapshot.ALL_DIALOG_KEYS)


def test_each_tab_owns_its_keys() -> None:
    assert DIALOG_TAB_KEYS["General"] == ("zoom_centered_on_cursor", "xwayland_on_wayland")
    assert set(DIALOG_TAB_KEYS["Mods"]) == {
        "mod_art_excluded",
        "mod_art_included",
        "data_mod_hide_unchanged",
        "data_mod_label_shown",
        "data_mod_terrain_group",
    }
    assert {"theme", "theme_colors", "dark_mode", "potato_mode", "grid_lightness", "elev_step_divisor"} <= set(
        DIALOG_TAB_KEYS["Appearance"]
    )
    assert set(DIALOG_TAB_KEYS["Tools"]) == {"tool_options", *_TOOLS_LEGACY}
    assert set(DIALOG_TAB_KEYS["Saving"]) == {
        "autosave_enabled",
        "autosave_interval_min",
        "autosave_retention",
        "autosave_location",
        "autosave_dir",
        "backups_enabled",
    }
    assert set(DIALOG_TAB_KEYS["Keybinds"]) == {"keybinds", "wheel_bindings"}
    assert set(DIALOG_TAB_KEYS["Advanced"]) == {"game_data", "gaia_only_catalog_mode", "units_catalog_right_click"}


def test_every_settings_tab_key_in_portable_keys_is_covered() -> None:
    """A portable key filed under a tab label reaches that tab's list without a second edit."""
    tabs = set(settings_snapshot.TABS)
    for key, (group, _label) in settings_file.PORTABLE_KEYS.items():
        if group in tabs:
            assert key in DIALOG_TAB_KEYS[group], key
    assert len(set(settings_snapshot.ALL_DIALOG_KEYS)) == len(settings_snapshot.ALL_DIALOG_KEYS)
    # Paths stay out: the install and profile folder apply on their own buttons.
    assert not {"aoe2de_install", "aoe2de_profile"} & set(settings_snapshot.ALL_DIALOG_KEYS)


def test_snapshot_marks_absent_keys_and_copies_deeply() -> None:
    _write({"theme": "dark", "overlay_colors": {"ruler_line": "#123456"}})
    snapshot = settings_snapshot.snapshot_settings(("theme", "overlay_colors", "grid_blend"))
    assert snapshot == {"theme": "dark", "overlay_colors": {"ruler_line": "#123456"}, "grid_blend": ABSENT}
    settings.set_overlay_color("ruler_line", "#abcdef")
    assert snapshot["overlay_colors"] == {"ruler_line": "#123456"}
    assert settings_snapshot.differs(snapshot)


def test_restore_round_trips_absent_and_legacy_keys() -> None:
    legacy = {
        "potato_mode": True,
        "dark_mode": True,
        "grid_lightness": 40,
        "elev_step_divisor": 4,
        **dict.fromkeys(_TOOLS_LEGACY, True),
    }
    _write({**legacy, "recent_files": ["/x.aoe2scenario"]})
    snapshot = settings_snapshot.snapshot_settings(settings_snapshot.ALL_DIALOG_KEYS)
    before = (
        settings.get_graphics_quality(),
        settings.get_theme(),
        settings.get_grid_blend(),
        settings.get_elev_step_pct(),
        settings.get_tool_option_defaults(),
    )
    # Each setter drops its legacy key and writes its successor.
    settings.set_graphics_quality(4)
    settings.set_theme("light")
    settings.set_grid_blend(-10)
    settings.set_elev_step_pct(100)
    settings.set_tool_option_default("paint_trees", False)
    settings.set_keybind("file_open", "Ctrl+F12")
    assert not set(legacy) & set(_config())

    settings_snapshot.restore_settings(snapshot)

    config = _config()
    assert {key: config[key] for key in legacy} == legacy
    assert "keybinds" not in config and "graphics_quality" not in config and "tool_options" not in config
    assert config["recent_files"] == ["/x.aoe2scenario"], "keys outside the dialog are left alone"
    after = (
        settings.get_graphics_quality(),
        settings.get_theme(),
        settings.get_grid_blend(),
        settings.get_elev_step_pct(),
        settings.get_tool_option_defaults(),
    )
    assert after == before
    assert settings.get_keybind("file_open") == settings.get_default_keybind("file_open")
    assert not settings_snapshot.differs(snapshot)


def test_reset_gives_every_getter_its_default() -> None:
    _write(
        {
            "zoom_centered_on_cursor": False,
            "autosave_interval_min": 30,
            "paint_trees": False,
            "tool_options": {"brush_size": 5},
            "game_data": "ror",
            "recent_files": ["/x.aoe2scenario"],
        }
    )
    assert settings.get_tool_option_default("paint_trees") is False
    assert settings.get_zoom_centered_on_cursor() is False

    settings_snapshot.reset_settings(DIALOG_TAB_KEYS["Tools"])
    assert settings.get_tool_option_defaults() == tool_options.defaults()
    assert settings.get_zoom_centered_on_cursor() is False, "another tab's keys stay"

    for tab in ("General", "Saving", "Advanced"):
        settings_snapshot.reset_settings(DIALOG_TAB_KEYS[tab])
    assert settings.get_zoom_centered_on_cursor() is True
    assert settings.get_autosave_interval_min() == settings.AUTOSAVE_INTERVAL_DEFAULT
    assert settings.get_game_data() == settings.GAME_DATA_DEFAULT
    assert _config() == {"recent_files": ["/x.aoe2scenario"]}


def test_every_dialog_key_has_a_memo_that_clearing_reaches() -> None:
    """For each key: its memo globals exist, and after _clear_memos the getters
    re-read the file, so a restored or reset value is never shadowed by a stale memo."""
    for key in settings_snapshot.ALL_DIALOG_KEYS:
        for name in settings_snapshot.memo_names(key):
            assert hasattr(settings, name), (key, name)
            setattr(settings, name, object())
        settings_snapshot._clear_memos((key,))
        for name in settings_snapshot.memo_names(key):
            assert getattr(settings, name) is None, (key, name)


def test_a_direct_config_edit_shows_through_the_getters_after_clearing() -> None:
    settings.get_pan_speed()
    settings.get_wheel_bindings()
    settings.get_autosave_dir()
    config = _config()
    config.update(
        {"pan_speed": 900, "wheel_bindings": {"plain": "none"}, "dark_mode": False, "autosave_dir": "relative"}
    )
    settings.CONFIG_PATH.write_text(yaml.safe_dump(config))
    settings_snapshot._clear_memos(settings_snapshot.ALL_DIALOG_KEYS)
    assert settings.get_pan_speed() == 900
    assert settings.get_wheel_bindings()["plain"] == "none"
    assert settings.get_theme() == "light"
    assert settings.get_autosave_dir() == ""
    assert settings.get_autosave_dir_refusal() is not None
