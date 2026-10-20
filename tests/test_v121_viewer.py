"""TASK-054's in-app half on the real Workshop v1.21 files: the same 25 files
tests/test_v121_alignment.py reads, opened in a real offscreen ViewerWindow.
That file proves the load and write paths headless; this one drives them
through the window (every mode entered, the real Save As, the Players and
Map Options panels a user would see). A small fixed subset, not all 25:
each file costs a full window load and mode cycle.

Corpus-tier for the alignment file's reason: these files exist only on this
machine and can never be committed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest

WORKSHOP_DIR = Path("~/games/steam/steamapps/workshop/content/221380").expanduser()

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
    pytest.mark.skipif(
        not WORKSHOP_DIR.is_dir(),
        reason="the real v1.21 Workshop corpus only exists on this machine",
    ),
]

# The smallest file, the one TASK-054's edited pair came from, and a 2v3.
_SUBSET = (
    "Coop-Scenario_Rebellion against the Mad King.aoe2scenario",
    "Coop-Scenario_Attila the Hun.aoe2scenario",
    "Matt's Coop 2v3.aoe2scenario",
)


def _resolve(name: str) -> Path:
    from test_v121_alignment import _v121_files

    matches = [path for path in _v121_files() if path.name == name]
    assert len(matches) == 1, f"{name} is not in the v1.21 Workshop corpus"
    return matches[0]


def _pump() -> None:
    from PyQt5.QtWidgets import QApplication

    for _ in range(3):
        QApplication.processEvents()


def _v121_window(path: Path):
    window = conftest.shown_window(1500, 900)
    window.load_scenario(path)
    assert window.scenario is not None, f"{path.name} failed to load"
    assert window.scenario.structure_source == "repo"
    _pump()
    return window


@pytest.mark.parametrize("name", _SUBSET)
def test_save_as_after_every_mode_is_byte_identical(name: str, tmp_path: Path, monkeypatch) -> None:
    """Passes when the file loads through the repo v1.21 structure, every
    mode in the combo can be entered (the combo shows it and the window's
    mode follows), the document stays clean throughout, and the real
    ViewerWindow.save_as(), with only its file dialog stubbed, writes a copy
    byte-identical to the original and retargets the window to it."""
    import descape.viewer as viewer_module

    path = _resolve(name)
    original = path.read_bytes()
    dest = tmp_path / "copy.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog, "getSaveFileName", staticmethod(lambda *args, **kwargs: (str(dest), ""))
    )
    window = _v121_window(path)
    try:
        modes = [window.mode_combo.itemText(i) for i in range(window.mode_combo.count())]
        assert len(modes) == 8
        for mode in [*modes, "View"]:
            window.mode_combo.setCurrentText(mode)
            _pump()
            assert window.mode_combo.currentText() == mode
            assert not window.edit_history.is_dirty, f"entering {mode} dirtied the document"

        window.save_as()

        assert dest.read_bytes() == original
        assert window.scenario.path == dest
        assert path.read_bytes() == original
    finally:
        conftest.close_window(window)


_EXEC_ORDER_UNLISTED = (
    "Map Options: this file's Triggers section can't be read, so the "
    "trigger execution-order setting is not listed."
)


@pytest.mark.parametrize("name", _SUBSET)
def test_players_are_read_only_and_map_options_lists_no_exec_order(name: str) -> None:
    """TASK-054 step 3's Players / Map Options half.

    Passes when, on a real v1.21 file: the window's editable Players field
    set is empty, every Players row for P1 is a disabled input or a plain
    label, and the panel's summary says "read-only" exactly once ("all
    read-only"); and Map Options has no `legacy_exec_order` spec or widget,
    its summary claims nothing read-only (its scalars do verify on v1.21),
    and entering it logs the unreadable-Triggers explanation exactly once."""
    from PyQt5.QtWidgets import QLabel

    window = _v121_window(_resolve(name))
    try:
        assert window._editable_player_fields() == frozenset()
        window.mode_combo.setCurrentText("Players")
        _pump()
        panel = window.players_panel
        assert panel._specs
        live = [
            spec.field_id
            for spec in panel._specs
            if not isinstance(panel.widget_for(spec.field_id), QLabel) and panel.widget_for(spec.field_id).isEnabled()
        ]
        assert live == []
        summary = panel.status.text()
        assert summary.endswith(", all read-only.")
        assert summary.count("read-only") == 1

        before = window.status_log.toPlainText()
        window.mode_combo.setCurrentText("Map Options")
        _pump()
        options = window.map_options_panel
        assert "legacy_exec_order" not in {spec.field_id for spec in options._specs}
        assert options.widget_for("legacy_exec_order") is None
        assert "read-only" not in options.status.text()
        logged = window.status_log.toPlainText()[len(before):]
        assert logged.count(_EXEC_ORDER_UNLISTED) == 1
    finally:
        conftest.close_window(window)
