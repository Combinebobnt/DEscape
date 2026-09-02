"""File > Save (in-place, as opposed to Save As -- see tests/test_new_map.py
for that one). ViewerWindow.save() writes back to the currently-open file's
own path with no dialog, except on an untitled document, where there is no
path yet to write back to and it must fall through to save_as() instead --
mirroring the "no in-place Save" comment write_scenario() itself never
enforced, only the viewer's old menu wiring did.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import conftest
from descape.scenario_io import load_map_and_units

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _window():
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    return window


def _close(window) -> None:
    window.edit_history.mark_saved()
    window.close()


def test_save_writes_back_to_the_open_path_with_no_dialog(tmp_path, monkeypatch) -> None:
    target = tmp_path / "mymap.aoe2scenario"
    shutil.copyfile(BLANK_FIXTURE, target)

    import descape.viewer as viewer_module

    def _unexpected_dialog(*args, **kwargs):
        raise AssertionError("save() on a titled document must not open Save As's dialog")

    monkeypatch.setattr(
        viewer_module.QFileDialog, "getSaveFileName", staticmethod(_unexpected_dialog)
    )

    window = _window()
    try:
        window.load_scenario(target)
        assert window.save_action.isEnabled()

        before_mtime = target.stat().st_mtime_ns
        window.save()

        assert window.scenario.path == target
        assert window._untitled is False
        assert not window.edit_history.is_dirty
        assert target.stat().st_mtime_ns != before_mtime  # actually rewrote the file

        reloaded = load_map_and_units(target)
        assert reloaded.map_manager.map_width == reloaded.map_manager.map_height == 120
    finally:
        _close(window)


def test_save_as_on_an_already_open_file_retargets_it(tmp_path, monkeypatch) -> None:
    """Save As on a normally-opened (non-untitled) document must adopt the
    new path too, not just the untitled case test_new_map.py covers -- the
    usual editor convention: after Save As, the window is editing the file
    it just wrote, not the one it was opened from."""
    original = tmp_path / "original.aoe2scenario"
    shutil.copyfile(BLANK_FIXTURE, original)
    dest = tmp_path / "renamed.aoe2scenario"

    import descape.viewer as viewer_module

    monkeypatch.setattr(
        viewer_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(dest), "")),
    )

    window = _window()
    try:
        window.load_scenario(original)
        original_mtime = original.stat().st_mtime_ns

        window.save_as()

        assert dest.is_file()
        assert window.scenario.path == dest
        assert window._untitled is False

        # A follow-up in-place Save must now write back to the new path, not
        # silently re-target the file the window was originally opened from.
        window.save()
        assert dest.stat().st_mtime_ns != original_mtime  # dest actually rewritten
        assert original.stat().st_mtime_ns == original_mtime  # original untouched
    finally:
        _close(window)


def test_save_on_untitled_document_falls_through_to_save_as(tmp_path, monkeypatch) -> None:
    dest = tmp_path / "newmap.aoe2scenario"

    import descape.viewer as viewer_module

    monkeypatch.setattr(
        viewer_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(dest), "")),
    )

    window = _window()
    try:
        window.new_map()
        assert window._untitled is True

        window.save()

        assert dest.is_file()
        assert window._untitled is False
        assert window.scenario.path == dest
    finally:
        _close(window)
