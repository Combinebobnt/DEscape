"""Theme export/import: descape/theme_file.py, settings.apply_appearance_values()
and Settings > Appearance's Export theme... / Import theme... buttons."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from descape import settings, theme_file, themes
from descape.scenario_io import FORBIDDEN_WRITE_MARKER

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


def test_a_compatdata_path_is_refused_with_nothing_written(tmp_path) -> None:
    proton = tmp_path / FORBIDDEN_WRITE_MARKER / "813780"
    proton.mkdir(parents=True)
    with pytest.raises(theme_file.ThemeWriteBlockedError):
        theme_file.write_theme_file(proton / "t.yaml", theme_file.build_document())
    assert list(proton.iterdir()) == []


def test_a_symlink_into_compatdata_is_refused_with_nothing_written(tmp_path) -> None:
    proton = tmp_path / FORBIDDEN_WRITE_MARKER / "813780"
    proton.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(proton, target_is_directory=True)
    with pytest.raises(theme_file.ThemeWriteBlockedError):
        theme_file.write_theme_file(link / "t.yaml", theme_file.build_document())
    assert list(proton.iterdir()) == []


def test_the_written_file_is_not_mode_600_and_leaves_no_temp(tmp_path) -> None:
    out = tmp_path / "t.yaml"
    theme_file.write_theme_file(out, theme_file.build_document())
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(out.stat().st_mode) == 0o644 & ~umask
    assert stat.S_IMODE(out.stat().st_mode) != 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["t.yaml"]


def test_overwriting_replaces_the_file(tmp_path) -> None:
    out = tmp_path / "t.yaml"
    out.write_text("old contents\n")
    theme_file.write_theme_file(out, theme_file.build_document())
    assert yaml.safe_load(out.read_text()) == theme_file.build_document()


def test_summary_lists_changes_problems_and_notes(tmp_path) -> None:
    doc = _doc(theme={"preset": "dark", "colors": {}}, grid_thickness=2, bogus=1, ui_font={"family": "Nowhere Sans"})
    plan = theme_file.parse_theme_file(_write(tmp_path / "t.yaml", doc), installed_families=[])
    summary = theme_file.format_import_summary(plan)
    assert "Theme: Light -> Dark" in summary
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
    conftest.ensure_qapp()
    from PyQt5.QtGui import QPalette
    from PyQt5.QtWidgets import QApplication

    from descape import viewer_dialogs

    app = QApplication.instance()
    saved = (QPalette(app.palette()), app.style().objectName(), app.font(), viewer_dialogs._LIGHT_PALETTE, viewer_dialogs._DEFAULT_FONT)
    viewer_dialogs._LIGHT_PALETTE = None
    viewer_dialogs._DEFAULT_FONT = None
    dialog, window = conftest.dialog_and_window()
    yield dialog
    dialog.close()
    conftest.close_window(window)
    viewer_dialogs._LIGHT_PALETTE, viewer_dialogs._DEFAULT_FONT = saved[3], saved[4]
    app.setStyle(saved[1])
    app.setPalette(saved[0])
    app.setFont(saved[2])


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

    assert len(shown) == 1 and "Theme: Light -> High contrast light" in shown[0]
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
def test_export_into_compatdata_is_blocked_and_reported(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtWidgets import QFileDialog, QMessageBox

    proton = tmp_path / FORBIDDEN_WRITE_MARKER
    proton.mkdir()
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(proton / f"t{theme_file.FILE_SUFFIX}"), ""))
    )
    reported = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: reported.append(a)))
    qt_dialog.theme_export_button.click()
    assert len(reported) == 1 and reported[0][1] == "Export blocked"
    assert list(proton.iterdir()) == []
