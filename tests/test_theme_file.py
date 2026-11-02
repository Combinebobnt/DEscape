"""Theme export/import: descape/theme_file.py, settings.apply_appearance_values()
and Settings > Appearance's Export theme... / Import theme... buttons."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from descape import settings, theme_file, themes, yaml_file_io

import conftest


def _non_default_look() -> None:
    settings.set_theme("solarized_dark")
    settings.set_theme_color("highlight", "#336699")
    settings.set_ui_font_family("DejaVu Sans")
    settings.set_ui_font_size(14)
    settings.set_overlay_color("ruler_line", "#ff00ff")
    settings.set_ruler_label_font_px(24)
    settings.set_distance_tick_font_px(10)
    settings.set_stack_badge_position("top_left")
    settings.set_grid_blend(30)
    settings.set_grid_thickness(3)


def _reset_memos(monkeypatch) -> None:
    from testkit.settings_isolation import MEMOIZED_GLOBALS

    for name in MEMOIZED_GLOBALS:
        monkeypatch.setattr(settings, name, None)


def _write(path: Path, doc) -> Path:
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _doc(**keys) -> dict:
    return {"format": theme_file.FORMAT_ID, "version": theme_file.FORMAT_VERSION, **keys}


# --- round trip and apply ----------------------------------------------------


def test_export_then_import_round_trips_to_an_identical_document(tmp_path, monkeypatch) -> None:
    _non_default_look()
    exported = theme_file.build_document()
    out = tmp_path / f"mine{theme_file.FILE_SUFFIX}"
    theme_file.write_theme_file(out, exported)

    # A fresh machine: an empty config, nothing memoized.
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "other" / "config.yaml")
    _reset_memos(monkeypatch)
    assert theme_file.build_document() != exported
    plan = theme_file.parse_theme_file(out)
    assert plan.problems == []
    settings.apply_appearance_values(plan.values)
    assert theme_file.build_document() == exported
    _reset_memos(monkeypatch)  # and from disk, not just the memo globals
    assert theme_file.build_document() == exported


def test_an_editor_preset_with_a_surface_override_round_trips(tmp_path, monkeypatch) -> None:
    settings.set_theme("editor_dark_wood_green")
    settings.set_theme_color("surface", "#3a2010")
    settings.set_theme_color("button", "#0a4a04")
    exported = theme_file.build_document()
    assert exported["theme"] == {"preset": "editor_dark_wood_green", "colors": {"surface": "#3a2010", "button": "#0a4a04"}}
    out = tmp_path / f"editor{theme_file.FILE_SUFFIX}"
    theme_file.write_theme_file(out, exported)

    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "other" / "config.yaml")
    _reset_memos(monkeypatch)
    assert settings.get_theme() == "editor_parchment_red"
    plan = theme_file.parse_theme_file(out)
    assert plan.problems == []
    settings.apply_appearance_values(plan.values)
    assert theme_file.build_document() == exported
    _reset_memos(monkeypatch)
    assert theme_file.build_document() == exported


def test_the_document_is_look_only() -> None:
    doc = theme_file.build_document()
    assert set(doc) == {
        "format",
        "version",
        "theme",
        "ui_font",
        "overlay_colors",
        "ruler_label_font_px",
        "distance_tick_font_px",
        "stack_badge_position",
        "grid_blend",
        "grid_thickness",
    }


def test_apply_appearance_values_saves_exactly_once(tmp_path, monkeypatch) -> None:
    _non_default_look()
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", theme_file.build_document()))
    saves = []
    real_save = settings._save_config
    monkeypatch.setattr(settings, "_save_config", lambda config: (saves.append(1), real_save(config)))
    settings.apply_appearance_values(plan.values)
    assert len(saves) == 1


def test_a_partial_file_touches_only_its_keys(tmp_path, monkeypatch) -> None:
    _non_default_look()
    before = theme_file.build_document()
    config_before = yaml.safe_load(settings.CONFIG_PATH.read_text())
    plan = theme_file.parse_theme_file(
        _write(tmp_path / "t.yaml", _doc(grid_blend=-10, overlay_colors={"region_fill": "#010203"}))
    )
    assert plan.problems == []
    settings.apply_appearance_values(plan.values)
    after = theme_file.build_document()
    assert after["grid_blend"] == -10
    assert after["overlay_colors"] == {**before["overlay_colors"], "region_fill": "#010203"}
    for key in set(before) - {"grid_blend", "overlay_colors"}:
        assert after[key] == before[key], key
    config_after = yaml.safe_load(settings.CONFIG_PATH.read_text())
    changed = {k for k in config_after if config_after[k] != config_before.get(k)}
    assert changed == {"grid_blend", "overlay_colors"}


def test_importing_a_theme_pops_legacy_dark_mode(tmp_path) -> None:
    settings.CONFIG_PATH.write_text("dark_mode: true\n")
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", _doc(theme={"preset": "dim", "colors": {}})))
    settings.apply_appearance_values(plan.values)
    on_disk = yaml.safe_load(settings.CONFIG_PATH.read_text())
    assert "dark_mode" not in on_disk
    assert on_disk["theme"] == "dim"


# --- per-key problems --------------------------------------------------------


@pytest.mark.parametrize(
    ("keys", "fragment", "not_applied"),
    [
        ({"colour_scheme": "dark"}, "unknown key 'colour_scheme'", "colour_scheme"),
        ({"theme": {"preset": "no_such", "colors": {}}}, "unknown preset 'no_such'", "theme"),
        ({"theme": "dark"}, "theme: expected a mapping", "theme"),
        ({"overlay_colors": {"no_such_overlay": "#000000"}}, "unknown id 'no_such_overlay'", "overlay_colors"),
        ({"overlay_colors": {"ruler_line": "orange"}}, "'orange' is not a '#rrggbb'", "overlay_colors"),
        ({"overlay_colors": ["#000000"]}, "overlay_colors: expected a mapping", "overlay_colors"),
        ({"ruler_label_font_px": 99}, "ruler_label_font_px: 99 is outside", "ruler_label_font_px"),
        ({"distance_tick_font_px": 2}, "distance_tick_font_px: 2 is outside", "distance_tick_font_px"),
        ({"grid_blend": 500}, "grid_blend: 500 is outside", "grid_blend"),
        ({"grid_thickness": 7}, "grid_thickness: 7 is not one of", "grid_thickness"),
        ({"grid_thickness": "2"}, "grid_thickness: expected a whole number", "grid_thickness"),
        ({"grid_blend": True}, "grid_blend: expected a whole number", "grid_blend"),
        ({"stack_badge_position": "middle"}, "stack_badge_position: 'middle' is not one of", "stack_badge_position"),
        ({"ui_font": {"size": 99}}, "ui_font.size: 99 is outside", "ui_font_size"),
        ({"ui_font": {"size": "big"}}, "ui_font.size: expected a whole number", "ui_font_size"),
        ({"ui_font": {"family": 12}}, "ui_font.family: expected text", "ui_font_family"),
    ],
)
def test_each_bad_key_is_reported_and_not_applied(tmp_path, keys, fragment, not_applied) -> None:
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", _doc(**keys)))
    assert any(fragment in problem for problem in plan.problems), plan.problems
    assert not_applied not in plan.values


def test_a_bad_role_colour_skips_only_that_role(tmp_path) -> None:
    doc = _doc(theme={"preset": "dim", "colors": {"window": "#ABCDEF", "text": "nope", "no_such_role": "#000000"}})
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc))
    assert plan.values["theme"] == {"preset": "dim", "colors": {"window": "#abcdef"}}
    assert len(plan.problems) == 2


def test_out_of_range_values_are_never_clamped_into_settings(tmp_path) -> None:
    before = theme_file.build_document()
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", _doc(ruler_label_font_px=999, grid_blend=-999)))
    settings.apply_appearance_values(plan.values)
    assert theme_file.build_document() == before


def test_an_uninstalled_font_is_kept_and_noted(tmp_path) -> None:
    plan = theme_file.parse_theme_file(
        _write(tmp_path / "t.yaml", _doc(ui_font={"family": "Nowhere Sans", "size": None})),
        installed_families=["DejaVu Sans"],
    )
    assert plan.values["ui_font_family"] == "Nowhere Sans"
    assert plan.values["ui_font_size"] is None
    assert plan.problems == []
    assert any("Nowhere Sans" in note for note in plan.notes)


# --- whole-file refusals -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (yaml.safe_dump(_doc(version=theme_file.FORMAT_VERSION + 1)), "newer than"),
        (yaml.safe_dump({"format": "something-else", "version": 1}), "not a DEscape theme file"),
        (yaml.safe_dump({"version": 1}), "not a DEscape theme file"),
        (yaml.safe_dump(["format", "descape-theme"]), "not a mapping"),
        ("just a string\n", "not a mapping"),
        ("format: descape-theme\nversion: [unterminated\n", "not valid YAML"),
        (yaml.safe_dump(_doc(version="1")), "version"),
    ],
)
def test_a_non_theme_file_is_refused_whole(tmp_path, text, fragment) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(text)
    with pytest.raises(theme_file.ThemeFileRefused, match=fragment):
        theme_file.parse_theme_file(path)


def test_an_oversized_file_is_refused_before_it_is_parsed(tmp_path, monkeypatch) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(_doc()) + "#" * theme_file.MAX_FILE_BYTES)
    monkeypatch.setattr(yaml, "load", lambda *_a, **_k: pytest.fail("parsed an oversized file"))
    with pytest.raises(theme_file.ThemeFileRefused, match="limit"):
        theme_file.parse_theme_file(path)


def test_deep_nesting_is_refused_not_a_recursion_error(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(_doc()) + "theme: " + "[" * 5000 + "\n")
    with pytest.raises(theme_file.ThemeFileRefused, match="nested too deeply"):
        theme_file.parse_theme_file(path)


def test_an_integer_past_the_conversion_limit_is_refused(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(_doc()) + "grid_blend: " + "9" * 5000 + "\n")
    with pytest.raises(theme_file.ThemeFileRefused, match="not valid YAML"):
        theme_file.parse_theme_file(path)


def _alias_bomb(levels: int, fanout: int) -> str:
    lines = ['l0: &l0 "xxxxxxxxxx"']
    for level in range(1, levels + 1):
        refs = ", ".join([f"*l{level - 1}"] * fanout)
        lines.append(f"l{level}: &l{level} [{refs}]")
    return "\n".join(lines) + f"\nstack_badge_position: *l{levels}\n"


def test_an_alias_bomb_is_refused_before_it_expands(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(_doc()) + _alias_bomb(5, 10))
    assert path.stat().st_size < 1024
    with pytest.raises(theme_file.ThemeFileRefused, match="aliases"):
        theme_file.parse_theme_file(path)


def test_problem_messages_cap_every_echoed_value(tmp_path) -> None:
    big = ["x" * 50] * 2000
    doc = _doc(
        stack_badge_position=big,
        theme={"preset": "x" * 5000, "colors": {}},
        overlay_colors={"x" * 5000: "#000000", "ruler_line": "y" * 5000},
        ui_font={"size": 10**4000, "family": "f" * 5000},
        grid_blend=10**4000,
        grid_thickness=10**4000,
        **{"z" * 5000: 1},
    )
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc), installed_families=[])
    messages = plan.problems + plan.notes
    assert len(messages) == 9, messages
    assert all(len(m) < 300 for m in messages), [len(m) for m in messages]


@pytest.mark.parametrize(
    "doc",
    [
        {"format": "f" * 5000, "version": 1},
        {"format": theme_file.FORMAT_ID, "version": ["v" * 50] * 2000},
        {"format": theme_file.FORMAT_ID, "version": 10**4000},
    ],
)
def test_refusal_messages_cap_the_echoed_value(tmp_path, doc) -> None:
    with pytest.raises(theme_file.ThemeFileRefused) as refused:
        theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc))
    assert len(str(refused.value)) < 300


def test_export_never_writes_anchors_so_import_reads_it_back(tmp_path) -> None:
    shared = {}
    out = tmp_path / f"mine{theme_file.FILE_SUFFIX}"
    theme_file.write_theme_file(out, _doc(theme={"preset": "dim", "colors": shared}, overlay_colors=shared))
    assert "&" not in out.read_text()
    assert theme_file.parse_theme_file(out).problems == []


# --- writing -----------------------------------------------------------------


def test_a_subscribed_path_is_refused_with_nothing_written(tmp_path) -> None:
    subscribed = tmp_path / "mods" / "subscribed" / "12345"
    subscribed.mkdir(parents=True)
    with pytest.raises(theme_file.ThemeWriteBlockedError):
        theme_file.write_theme_file(subscribed / "t.yaml", theme_file.build_document())
    assert list(subscribed.iterdir()) == []


@pytest.mark.parametrize("py313_resolve", [False, True])
@conftest.requires_symlinks
def test_a_looped_path_is_refused_with_nothing_written(tmp_path, monkeypatch, py313_resolve) -> None:
    out = tmp_path / "out"
    out.mkdir()
    loop = conftest.symlink_loop(out)
    doc = theme_file.build_document()
    if py313_resolve:
        conftest.resolve_like_py313(monkeypatch)
    with pytest.raises(theme_file.ThemeWriteBlockedError, match="cannot be resolved"):
        theme_file.write_theme_file(loop / "t.yaml", doc)
    assert sorted(p.name for p in out.iterdir()) == ["loop-a", "loop-b"]


@conftest.requires_symlinks
def test_a_symlink_into_mods_subscribed_is_refused_with_nothing_written(tmp_path) -> None:
    subscribed = tmp_path / "mods" / "subscribed" / "12345"
    subscribed.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(subscribed, target_is_directory=True)
    with pytest.raises(theme_file.ThemeWriteBlockedError):
        theme_file.write_theme_file(link / "t.yaml", theme_file.build_document())
    assert list(subscribed.iterdir()) == []


def test_a_compatdata_path_is_written(tmp_path) -> None:
    proton = tmp_path / "compatdata" / "813780"
    proton.mkdir(parents=True)
    theme_file.write_theme_file(proton / "t.yaml", theme_file.build_document())
    assert [p.name for p in proton.iterdir()] == ["t.yaml"]


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits (st_mode is only 0o666 or 0o444)")
def test_the_written_file_is_not_mode_600(tmp_path) -> None:
    out = tmp_path / "t.yaml"
    theme_file.write_theme_file(out, theme_file.build_document())
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(out.stat().st_mode) == 0o644 & ~umask
    assert stat.S_IMODE(out.stat().st_mode) != 0o600


def test_writing_leaves_no_temp(tmp_path) -> None:
    out = tmp_path / "t.yaml"
    theme_file.write_theme_file(out, theme_file.build_document())
    assert [p.name for p in tmp_path.iterdir()] == ["t.yaml"]


def test_overwriting_replaces_the_file(tmp_path) -> None:
    out = tmp_path / "t.yaml"
    out.write_text("old contents\n")
    theme_file.write_theme_file(out, theme_file.build_document())
    assert yaml.safe_load(out.read_text()) == theme_file.build_document()


def test_the_whole_number_check_is_the_shared_one() -> None:
    """Theme and settings files read versions and sizes through one int check, not a copy each."""
    assert theme_file._is_int is yaml_file_io.is_int


def test_summary_lists_changes_problems_and_notes(tmp_path) -> None:
    doc = _doc(theme={"preset": "dark", "colors": {}}, grid_thickness=2, bogus=1, ui_font={"family": "Nowhere Sans"})
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc), installed_families=[])
    summary = theme_file.format_import_summary(plan)
    assert "Theme: Wood (red) -> Dark" in summary
    assert "Grid thickness: 1 px -> 2 px" in summary
    assert "unknown key 'bogus'" in summary
    assert "Nowhere Sans" in summary.split("Notes:")[1]


def test_summary_of_a_no_op_file_says_so(tmp_path) -> None:
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", theme_file.build_document()))
    assert theme_file.format_import_summary(plan).startswith("Nothing in this file differs")


# --- Settings > Appearance ---------------------------------------------------


@pytest.fixture
def qt_dialog():
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    from descape import gc_hold, viewer_dialogs

    # The theme's own state (both palettes, style, _LIGHT_PALETTE) goes back through the shared helper.
    with conftest.restoring_app_theme() as app:
        saved_font, saved_default_font = app.font(), viewer_dialogs._DEFAULT_FONT
        viewer_dialogs._DEFAULT_FONT = None
        dialog, window = conftest.dialog_and_window()
        yield dialog
        dialog.close()
        conftest.close_window(window)
        viewer_dialogs._DEFAULT_FONT = saved_default_font
        with gc_hold.paused():
            app.setFont(saved_font)


def _answer_import(monkeypatch, path: Path, apply: bool) -> list[str]:
    from PyQt5.QtWidgets import QFileDialog

    from descape.viewer import SettingsDialog

    shown: list[str] = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(SettingsDialog, "_confirm_theme_import", lambda _self, summary: shown.append(summary) or apply)
    return shown


@pytest.mark.gui
def test_import_applies_live_and_refreshes_every_widget(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape import grid_overlay

    doc = _doc(
        theme={"preset": "high_contrast_light", "colors": {"highlight": "#123456"}},
        overlay_colors={"ruler_line": "#00ff00"},
        ruler_label_font_px=20,
        distance_tick_font_px=16,
        stack_badge_position="above",
        grid_blend=40,
        grid_thickness=4,
    )
    shown = _answer_import(monkeypatch, _write(tmp_path / "t.yaml", doc), apply=True)
    saves = []
    real_save = settings._save_config
    monkeypatch.setattr(settings, "_save_config", lambda config: (saves.append(1), real_save(config)))
    qt_dialog.theme_import_button.click()

    assert len(shown) == 1 and "Theme: Wood (red) -> High contrast light" in shown[0]
    assert len(saves) == 1  # the widget refresh re-persisted nothing
    palette = QApplication.instance().palette()
    assert palette.color(QPalette.Active, QPalette.Highlight).name() == "#123456"
    assert palette.color(QPalette.Active, QPalette.Window).name() == themes.PRESETS["high_contrast_light"][1]["window"]
    assert qt_dialog.theme_combo.currentData() == "high_contrast_light"
    assert qt_dialog._theme_swatches["highlight"].toolTip() == "#123456"
    assert qt_dialog._overlay_swatches["ruler_line"].toolTip() == "#00ff00"
    assert qt_dialog.ruler_label_font_spin.value() == 20
    assert qt_dialog.distance_tick_font_spin.value() == 16
    assert qt_dialog.stack_badge_position_combo.currentData() == "above"
    assert qt_dialog.grid_blend_slider.value() == 40
    assert qt_dialog.grid_thickness_slider.value() == grid_overlay.thickness_index(4)
    assert qt_dialog.grid_thickness_value_label.text() == "4 px"
    view = qt_dialog._window.map_view
    assert view._stack_badge_position == "above"
    assert (view._grid_blend, view._grid_thickness) == (40, 4)


@pytest.mark.gui
def test_cancelling_an_import_changes_nothing(tmp_path, monkeypatch, qt_dialog) -> None:
    before = theme_file.build_document()
    _answer_import(monkeypatch, _write(tmp_path / "t.yaml", _doc(grid_blend=40)), apply=False)
    qt_dialog.theme_import_button.click()
    assert theme_file.build_document() == before
    assert qt_dialog.grid_blend_slider.value() == before["grid_blend"]


@pytest.mark.gui
def test_a_refused_import_warns_and_changes_nothing(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtWidgets import QMessageBox

    before = theme_file.build_document()
    path = tmp_path / "t.yaml"
    path.write_text("format: something-else\nversion: 1\n")
    shown = _answer_import(monkeypatch, path, apply=True)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    qt_dialog.theme_import_button.click()
    assert len(warned) == 1 and "not a DEscape theme file" in warned[0][2]
    assert shown == []
    assert theme_file.build_document() == before


@pytest.mark.gui
def test_export_writes_the_current_look(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtWidgets import QFileDialog

    settings.set_grid_blend(12)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "mine"), "")))
    qt_dialog.theme_export_button.click()
    out = tmp_path / f"mine{theme_file.FILE_SUFFIX}"
    assert yaml.safe_load(out.read_text()) == theme_file.build_document()


@pytest.mark.gui
@pytest.mark.parametrize("replace", [False, True])
def test_export_asks_before_replacing_the_suffixed_file(tmp_path, monkeypatch, qt_dialog, replace) -> None:
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    out = tmp_path / f"mine{theme_file.FILE_SUFFIX}"
    out.write_text("keep me\n")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "mine"), "")))
    asked = []
    answer = QMessageBox.Yes if replace else QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or answer))
    qt_dialog.theme_export_button.click()
    assert len(asked) == 1 and str(out) in asked[0][2]
    if replace:
        assert yaml.safe_load(out.read_text()) == theme_file.build_document()
    else:
        assert out.read_text() == "keep me\n"


@pytest.mark.gui
def test_export_into_mods_subscribed_is_blocked_and_reported(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    subscribed = tmp_path / "mods" / "subscribed"
    subscribed.mkdir(parents=True)
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(subscribed / f"t{theme_file.FILE_SUFFIX}"), "")),
    )
    reported = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: reported.append(a)))
    qt_dialog.theme_export_button.click()
    assert len(reported) == 1 and reported[0][1] == "Export blocked"
    assert list(subscribed.iterdir()) == []


# --- the owed in-app passes, offscreen (TASK-100) ----------------------------


@pytest.mark.gui
def test_an_export_then_import_through_the_buttons_restores_theme_overlay_colour_and_grid_thickness(
    tmp_path, monkeypatch, qt_dialog
) -> None:
    """Pass conditions:
    - Export theme... writes the current look to the picked path;
    - after the theme, the ruler_line overlay colour and Grid thickness change through their
      widgets, Import theme... of that file shows a summary with the overlay-colour
      "Overlay colour Line and endpoints: <new> -> <old>" line, plus theme and thickness lines;
    - Apply restores all three in settings and in the theme combo, the overlay swatch and the
      thickness slider.
    """
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import QColorDialog, QFileDialog

    from descape import grid_overlay

    before = (settings.get_theme(), settings.get_overlay_color("ruler_line"), settings.get_grid_thickness())
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(tmp_path / "mine"), "")))
    qt_dialog.theme_export_button.click()
    out = tmp_path / f"mine{theme_file.FILE_SUFFIX}"
    assert out.is_file()

    qt_dialog.theme_combo.setCurrentIndex(qt_dialog.theme_combo.findData("solarized_light"))
    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#00ff00")))
    qt_dialog._overlay_swatches["ruler_line"].click()
    qt_dialog.grid_thickness_slider.setValue(grid_overlay.thickness_index(4))
    changed = (settings.get_theme(), settings.get_overlay_color("ruler_line"), settings.get_grid_thickness())
    assert changed == ("solarized_light", "#00ff00", 4)
    assert all(a != b for a, b in zip(before, changed, strict=True))

    shown = _answer_import(monkeypatch, out, apply=True)
    qt_dialog.theme_import_button.click()
    assert len(shown) == 1
    lines = shown[0].splitlines()
    assert f"  Overlay colour Line and endpoints: #00ff00 -> {before[1]}" in lines
    assert f"  Theme: Solarized light -> {themes.preset_label(before[0])}" in lines
    assert f"  Grid thickness: 4 px -> {before[2]} px" in lines

    assert (settings.get_theme(), settings.get_overlay_color("ruler_line"), settings.get_grid_thickness()) == before
    assert qt_dialog.theme_combo.currentData() == before[0]
    assert qt_dialog._overlay_swatches["ruler_line"].toolTip() == before[1]
    assert qt_dialog.grid_thickness_slider.value() == grid_overlay.thickness_index(before[2])


# The listed contents of the W11 hand-made broken theme, rebuilt here so no test reads build/.
_BROKEN_THEME = """\
format: descape-theme
version: 1
theme: {preset: solarized_dark, colors: {highlight: '#336699', text: nope}}
overlay_colors: {ruler_line: '#00ff00', no_such_overlay: '#000000'}
ruler_label_font_px: 99
grid_thickness: 3
ui_font: {family: Nowhere Sans}
bogus_key: 1
"""


@pytest.mark.gui
def test_importing_the_broken_theme_through_the_dialog_reports_each_part_and_applies_only_the_good_ones(
    tmp_path, monkeypatch, qt_dialog
) -> None:
    """Pass conditions, the summary the Import button shows:
    - Will change: Theme -> Solarized dark, the Selection #336699 chrome override, the
      ruler_line overlay colour -> #00ff00, Grid thickness -> 3 px (and the kept UI font);
    - Skipped: theme.colors.text 'nope', no_such_overlay, ruler_label_font_px 99, bogus_key;
    - Notes: the 'Nowhere Sans' not-installed note;
    and after Apply: those four took effect, the ruler label spin box and its setting are
    unchanged, and no 'text' role override was stored.
    """
    path = tmp_path / f"broken{theme_file.FILE_SUFFIX}"
    path.write_text(_BROKEN_THEME)
    ruler_px_before = settings.get_ruler_label_font_px()
    assert qt_dialog.ruler_label_font_spin.value() == ruler_px_before
    overlay_before = settings.get_overlay_color("ruler_line")
    thickness_before = settings.get_grid_thickness()
    theme_before = settings.get_theme()

    shown = _answer_import(monkeypatch, path, apply=True)
    qt_dialog.theme_import_button.click()
    assert len(shown) == 1
    will, skipped, notes = shown[0].split("\n\n")
    assert will.splitlines() == [
        "Will change:",
        f"  Theme: {themes.preset_label(theme_before)} -> Solarized dark",
        "  Chrome colour overrides: Selection #336699",
        "  UI font: Nowhere Sans",
        f"  Overlay colour Line and endpoints: {overlay_before} -> #00ff00",
        f"  Grid thickness: {thickness_before} px -> 3 px",
    ]
    skipped_lines = skipped.splitlines()
    assert skipped_lines[0] == "Skipped (not applied):"
    assert len(skipped_lines) == 5
    for fragment in ("theme.colors.text: 'nope'", "unknown id 'no_such_overlay'", "ruler_label_font_px: 99", "'bogus_key'"):
        assert sum(fragment in line for line in skipped_lines) == 1, fragment
    assert notes.startswith("Notes:\n") and "'Nowhere Sans' is not installed" in notes

    assert settings.get_theme() == "solarized_dark"
    assert settings.get_theme_colors() == {"highlight": "#336699"}
    assert settings.get_overlay_color("ruler_line") == "#00ff00"
    assert settings.get_grid_thickness() == 3
    assert settings.get_ruler_label_font_px() == ruler_px_before
    assert qt_dialog.ruler_label_font_spin.value() == ruler_px_before


_STARTUP_PROBE = """
import sys
from pathlib import Path
config = Path(sys.argv[1])
import descape.asset_source as asset_source
import descape.settings as settings
asset_source.CONFIG_PATH = config
settings.CONFIG_PATH = config
asset_source.LEGACY_CONFIG_PATH = config.parent / "no-legacy-config.yaml"
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication
import descape.viewer as viewer

def _probe_exec(app):
    palette = app.palette()
    print("HIGHLIGHT", palette.color(QPalette.Active, QPalette.Highlight).name())
    print("WINDOW", palette.color(QPalette.Active, QPalette.Window).name())
    for widget in app.topLevelWidgets():
        if isinstance(widget, viewer.ViewerWindow):
            widget.edit_history.mark_saved()
            widget.close()
    return 0

QApplication.exec_ = _probe_exec
sys.argv = ["descape"]
try:
    viewer.main()
except SystemExit as e:
    print("EXIT", e.code)
"""


@pytest.mark.gui
def test_main_applies_a_persisted_imported_theme_at_startup(tmp_path) -> None:
    """A theme imported through apply_appearance_values() (the Import button's persist step)
    and then a fresh `viewer.main()` in its own process (one QApplication per process).

    Pass conditions: before its event loop starts, main()'s QApplication palette has the
    imported preset's Window colour and the imported Highlight override, and main() exits 0.
    """
    import os
    import subprocess
    import sys

    doc = _doc(theme={"preset": "high_contrast_light", "colors": {"highlight": "#123456"}})
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc))
    settings.apply_appearance_values(plan.values)
    config = tmp_path / "config.yaml"
    assert yaml.safe_load(config.read_text())["theme"] == "high_contrast_light"

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QPA_PLATFORMTHEME="", XDG_CONFIG_HOME=str(tmp_path / "xdg"))
    proc = subprocess.run(
        [sys.executable, "-c", _STARTUP_PROBE, str(config)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = dict(line.split(" ", 1) for line in proc.stdout.splitlines() if line.startswith(("HIGHLIGHT ", "WINDOW ", "EXIT ")))
    assert out.get("EXIT") == "0", proc.stderr[-2000:]
    assert out["HIGHLIGHT"] == "#123456"
    assert out["WINDOW"] == themes.PRESETS["high_contrast_light"][1]["window"]
