"""GH #182: Elevation View and the View > Layers rows persist across launches,
driven through real offscreen ViewerWindows.

Each test nulls the settings memo between the write and building the second
window, so that window reads config.yaml rather than the in-process cache: a
test that skipped this would pass on a wrong YAML shape.

Every ViewerWindow() constructed here goes through conftest.close_window().
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from descape import settings, view_layers, viewer

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]


def _config(tmp_path: Path) -> dict:
    path = tmp_path / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def _flipped(spec: view_layers.LayerSpec) -> bool:
    return not spec.default


@pytest.mark.parametrize("load", [False, True], ids=["no_map", "map_open"])
def test_a_layer_toggle_persists_into_a_fresh_window(tmp_path: Path, monkeypatch, load: bool) -> None:
    first = conftest.blank_window(load=load)
    try:
        for spec in view_layers.LAYERS:
            first.layer_actions[spec.layer_id].setChecked(_flipped(spec))
    finally:
        conftest.close_window(first)
    assert _config(tmp_path)["view_layers"] == {spec.layer_id: _flipped(spec) for spec in view_layers.LAYERS}

    monkeypatch.setattr(settings, "_view_layers", None)
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(viewer.ViewerWindow, "_on_layer_toggled", lambda self, lid, on: calls.append((lid, on)))
    second = conftest.blank_window(load=False)
    try:
        for spec in view_layers.LAYERS:
            assert second.layer_actions[spec.layer_id].isChecked() is _flipped(spec), spec.layer_id
            assert getattr(second._layers, spec.layer_id) is _flipped(spec), spec.layer_id
        assert calls == [], "construction must not fire _on_layer_toggled"
    finally:
        conftest.close_window(second)


def test_a_persisted_layer_reaches_the_cache_on_open(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("view_layers:\n  terrain_textures: false\n")
    window = conftest.blank_window()
    try:
        assert window.layer_actions["terrain_textures"].isChecked() is False
        assert window._cache.layers.terrain_textures is False
    finally:
        conftest.close_window(window)


def test_a_persisted_sloped_style_is_the_launch_style(tmp_path: Path) -> None:
    from descape.scenario_io import BLANK_TEMPLATE_PATH

    (tmp_path / "config.yaml").write_text("terrain_style: sloped\n")
    window = conftest.blank_window(load=False)
    try:
        assert window.terrain_style_combo.currentText() == "Sloped"
        assert window._terrain_style == "sloped"
        assert window.iso_action.isEnabled() is False
        window.load_scenario(BLANK_TEMPLATE_PATH)
        assert window.scenario is not None
        assert window.map_view._terrain_style == "sloped"
    finally:
        conftest.close_window(window)


def test_a_style_switch_persists_into_a_fresh_window(tmp_path: Path, monkeypatch) -> None:
    first = conftest.blank_window()
    try:
        first.terrain_style_combo.setCurrentText("Flat")
        assert first._terrain_style == "flat"
    finally:
        conftest.close_window(first)
    assert _config(tmp_path)["terrain_style"] == "flat"

    monkeypatch.setattr(settings, "_terrain_style", None)
    second = conftest.blank_window(load=False)
    try:
        assert second.terrain_style_combo.currentText() == "Flat"
        assert second._terrain_style == "flat"
        assert second.iso_action.isEnabled() is True
    finally:
        conftest.close_window(second)


def test_a_style_switch_refused_while_busy_does_not_persist(tmp_path: Path) -> None:
    window = conftest.blank_window()
    try:
        window._busy = True
        try:
            window.terrain_style_combo.setCurrentText("Sloped")
        finally:
            window._busy = False
        assert window._terrain_style == "stepped"
        assert window.terrain_style_combo.currentText() == "Stepped"
    finally:
        conftest.close_window(window)
    assert "terrain_style" not in _config(tmp_path)
