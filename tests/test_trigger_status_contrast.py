"""TASK-188 (GH #166): the Triggers panel's status colours under each theme.

Every status label in the app is shaded with `themes.legible()` until it
reaches 4.5:1 on what it sits on (7:1 on a high-contrast preset), through
`viewer_dialogs.legible_status_color()`. These tests hold each marked
trigger, condition and effect row to that floor against the tree's Base
colour, under every theme preset, applied the way the app does, and after a
live theme switch. Before TASK-424 the rows took the raw setting hex: on Wood,
the default theme, an OK row measured 1.44:1 and a problem row 1.99:1.
Ratios are read off the live palette and row brushes: PyQt 5.15, Fusion,
offscreen QPA. A selected row paints in the highlight colours, not measured here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import themes

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

TRIGGER_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
# The fixture's one trigger with problems (test_trigger_panel.py's GH #166 section).
_BROKEN_TRIGGER = 3


def _params():
    for preset_id in themes.PRESETS:
        for kind in ("ok", "problem"):
            yield pytest.param(preset_id, kind, id=f"{preset_id}-{kind}")


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install (see test_trigger_panel.py)."""
    from descape import asset_source

    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _marked_rows(panel, status) -> list[tuple[str, object, int]]:
    """(label, item, column) for a trigger row, a condition row and an effect
    row whose status is `status`, from the fixture's own statuses."""
    from PyQt5.QtCore import Qt

    index = 0 if status.name == "OK" else _BROKEN_TRIGGER
    rows = [("trigger", panel._item_for_index[index], panel._COL_NAME)]
    panel.select_trigger(_BROKEN_TRIGGER)
    for kind, top in (("condition", 1), ("effect", 2)):
        group = panel.entry_tree.topLevelItem(top)
        match = next(
            (
                group.child(r)
                for r in range(group.childCount())
                if getattr(group.child(r).data(0, panel._STATUS_ROLE), "status", None) is status
            ),
            None,
        )
        assert match is not None, f"fixture assumption: trigger {_BROKEN_TRIGGER} has a {status.name} {kind}"
        rows.append((kind, match, 0))
    for label, item, col in rows:
        assert item.data(col, Qt.ForegroundRole) is not None, f"fixture assumption: the {label} row is coloured"
    return rows


def _apply_preset(app, preset_id: str) -> None:
    from descape import settings
    from descape.viewer_dialogs import apply_theme

    settings.set_theme(preset_id)
    apply_theme(app, settings.get_theme(), settings.get_theme_colors())


def _below_floor(rows, preset_id: str) -> list[str]:
    """One line per row whose foreground misses the floor on its tree's Base."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPalette

    floor = 7.0 if preset_id in themes.HIGH_CONTRAST_PRESETS else 4.5
    failures = []
    for label, item, col in rows:
        base = item.treeWidget().palette().color(QPalette.Active, QPalette.Base).name()
        foreground = item.data(col, Qt.ForegroundRole).color().name()
        ratio = themes.contrast_ratio(foreground, base)
        if ratio < floor:
            failures.append(f"{label} row {foreground} on {base}: {ratio:.2f}:1 (floor {floor})")
    return failures


def _status(kind: str):
    from descape.trigger_status import Status

    return Status.OK if kind == "ok" else Status.PROBLEM


@pytest.mark.parametrize(("preset_id", "kind"), list(_params()))
def test_a_status_row_is_legible_on_the_trees_base(preset_id: str, kind: str) -> None:
    """Passes when: under `preset_id`, applied and persisted as the app does,
    a trigger row, a condition row and an effect row of status `kind` each
    draw a foreground reaching legible()'s floor (4.5:1, 7:1 on a
    high-contrast preset) against their tree's Base colour."""
    with conftest.restoring_app_theme() as app:
        _apply_preset(app, preset_id)
        window = conftest.shown_window(1500, 900)
        try:
            window.load_scenario(TRIGGER_FIXTURE)
            window.mode_combo.setCurrentText("Triggers")
            failures = _below_floor(_marked_rows(window.trigger_panel, _status(kind)), preset_id)
            assert not failures, f"under {preset_id}: " + "; ".join(failures)
        finally:
            window.edit_history.mark_saved()
            window.close()


@pytest.mark.parametrize("kind", ["ok", "problem"])
@pytest.mark.parametrize(("before", "after"), [("dark", "light"), ("light", "dark")])
def test_a_theme_switch_reshades_the_rows_already_shown(before: str, after: str, kind: str) -> None:
    """Passes when: rows styled under `before` reach the floor on `after`'s
    Base once `after` is applied live, with no repopulate or reselect (each
    pair's `before` shade misses `after`'s floor, so a stale brush fails)."""
    from PyQt5.QtWidgets import QApplication

    with conftest.restoring_app_theme() as app:
        _apply_preset(app, before)
        window = conftest.shown_window(1500, 900)
        try:
            window.load_scenario(TRIGGER_FIXTURE)
            window.mode_combo.setCurrentText("Triggers")
            rows = _marked_rows(window.trigger_panel, _status(kind))
            assert not _below_floor(rows, before), "precondition: the rows reach the floor before the switch"
            _apply_preset(app, after)
            QApplication.processEvents()
            failures = _below_floor(rows, after)
            assert not failures, f"after {before} -> {after}: " + "; ".join(failures)
        finally:
            window.edit_history.mark_saved()
            window.close()
