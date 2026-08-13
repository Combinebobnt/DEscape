"""File > New Map: generates a blank tiles x tiles map in memory from the
shipped donor template (descape.scenario_new.blank_scenario_bytes) and loads
the result through the normal load path, marking the document untitled --
rather than building a scenario from scratch through AoE2ScenarioParser's own
(unverified, non-byte-stable) serializer. See scenario_new.py's docstring and
descape/viewer.py's new_map()/new_map_custom().

Non-GUI tests pin the donor template asset's own shape (dimensions, version,
emptiness) and the write-guard that keeps Save As from ever overwriting a
shipped template -- both load-bearing for tests/test_write_path.py and
tests/test_lazy_viewport.py, which assume this same file's geometry. The byte
production itself -- reproducing real game exports at other sizes, bounds
checking, precondition guards -- is tests/test_scenario_new.py's job, not
this module's; GUI tests here trigger only a few of the seven standard sizes
to keep this file's own runtime down, on the understanding that the
generator underneath every size is already covered there.

GUI tests exercise the actual QAction/dialog wiring and the untitled-state
bookkeeping in descape/viewer.py (ViewerWindow._untitled, UNTITLED_PATH,
save_as() adoption). Every ViewerWindow() constructed here must end its test
with edit_history.mark_saved() before close() -- close() runs
_confirm_discard_changes(), which pops a modal QMessageBox on a dirty
document and would hang an offscreen run (see tests/test_lazy_viewport.py's
_check_exposed_rect_smaller() for the same pattern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import conftest
from descape.scenario_io import (
    BLANK_TEMPLATE_PATH,
    BLANK_TEMPLATE_SIZES,
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    blank_template_path,
    load_map_and_units,
)
from descape.scenario_new import (
    LARGE_MAP_CONFIRM_TILES,
    STANDARD_MAP_SIZE_NAMES,
    STANDARD_MAP_SIZES,
)
from descape.scenario_write import WriteBlockedError, write_scenario


@pytest.mark.parametrize("tiles", BLANK_TEMPLATE_SIZES)
def test_blank_template_is_shipped_and_loads(tiles: int) -> None:
    path = blank_template_path(tiles)
    assert path.is_file()
    s = load_map_and_units(path)
    mm = s.map_manager
    assert mm.map_width == mm.map_height == tiles
    assert s.map_is_square
    assert s.terrain_write_supported
    assert s.scenario_version == "1.58"
    assert all(t.terrain_id == 0 and t.elevation == 0 for t in mm.terrain)
    assert sum(len(units) for units in s.unit_manager.units) == 0


def test_write_scenario_refuses_to_write_into_template_dir() -> None:
    s = load_map_and_units(BLANK_TEMPLATE_PATH)
    original_bytes = BLANK_TEMPLATE_PATH.read_bytes()
    with pytest.raises(WriteBlockedError):
        write_scenario(s, BLANK_TEMPLATE_PATH)
    # A future sibling template must be covered by the same guard, without
    # needing to exist on disk first -- the check is on the destination's
    # parent directory, not on colliding with an existing file. Deliberately
    # not one of BLANK_TEMPLATE_SIZES (which now really are shipped files) --
    # this is testing the guard's blanket directory coverage, not colliding
    # with a real one.
    with pytest.raises(WriteBlockedError):
        write_scenario(s, TEMPLATE_DIR / "blank_960x960.aoe2scenario")
    assert BLANK_TEMPLATE_PATH.read_bytes() == original_bytes


def test_write_scenario_still_writes_elsewhere(tmp_path) -> None:
    s = load_map_and_units(BLANK_TEMPLATE_PATH)
    dest = tmp_path / "copy.aoe2scenario"
    write_scenario(s, dest)
    assert dest.is_file()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_action_creates_untitled_document() -> None:
    conftest.ensure_qapp()
    from PyQt5.QtGui import QKeySequence

    from descape.viewer import UNTITLED_NAME, ViewerWindow

    window = ViewerWindow()
    try:
        # Trigger the QAction itself, not window.new_map() directly, so the
        # menu wiring (_build_menu_bar's file_menu.addAction) is covered too.
        window.new_action.trigger()

        assert window.scenario is not None
        assert window._untitled is True
        assert window.scenario.path.name == UNTITLED_NAME
        assert window.scenario.path != BLANK_TEMPLATE_PATH
        assert window.windowTitle() == f"{UNTITLED_NAME} — DEscape"
        assert window.edit_history.is_dirty is False
        assert window.save_as_action.isEnabled()
        assert window.close_action.isEnabled()

        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == BLANK_TEMPLATE_TILES

        assert window.new_action.isEnabled()
        assert window.new_action.shortcut().isEmpty()
        assert window.new_custom_action.shortcut() == QKeySequence("Ctrl+N")
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_size_submenu_actions_create_matching_size_document() -> None:
    """window.new_size_actions holds one QAction per STANDARD_MAP_SIZES
    (descape/viewer.py's File > New Map submenu), derived from the constant so
    adding a size later needs no edit here, plus a separate Custom size…
    action that is deliberately NOT part of new_size_actions (see
    viewer.py's new_map_custom() docstring for why).

    Triggers only three entries -- the 120 default, one mid-size (168), and
    480 -- not all seven: each trigger is a full parse plus offscreen render,
    and byte production at every size is already covered by
    tests/test_scenario_new.py. This test's job is the signal wiring, not
    the byte production."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        assert [a.text() for a in window.new_size_actions] == [
            f"{tiles}×{tiles} ({STANDARD_MAP_SIZE_NAMES[tiles]})" for tiles in STANDARD_MAP_SIZES
        ]
        assert window.new_custom_action.text() == "&Custom size…"
        assert window.new_custom_action not in window.new_size_actions

        for tiles in (168, 480):
            action = next(
                a
                for a in window.new_size_actions
                if a.text() == f"{tiles}×{tiles} ({STANDARD_MAP_SIZE_NAMES[tiles]})"
            )
            action.trigger()
            mm = window.scenario.map_manager
            assert mm.map_width == mm.map_height == tiles
            assert window._untitled is True
            window.edit_history.mark_saved()  # keep _confirm_discard_changes() quiet for the next iteration
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_preset_480_does_not_confirm(monkeypatch) -> None:
    """Pins the deliberate asymmetry in new_map_custom()'s docstring: the
    labelled 480 preset never pops the large-map confirmation, only a typed
    Custom size above LARGE_MAP_CONFIRM_TILES does. Fails loudly (instead of
    hanging on an unclickable modal) if a future 'consistency fix' routes
    presets through the same confirm."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    def fail_if_called(*args, **kwargs):
        raise AssertionError("QMessageBox.question must not be called for the 480 preset")

    monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fail_if_called))

    window = ViewerWindow()
    try:
        action = next(a for a in window.new_size_actions if a.text() == "480×480 (Ludicrous)")
        action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 480
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_action_creates_that_size(monkeypatch) -> None:
    """168 <= LARGE_MAP_CONFIRM_TILES, so this needs no QMessageBox patch at
    all -- if new_map_custom() ever popped a confirm here, the unpatched
    QMessageBox.question would hang the offscreen run rather than silently
    pass, which is the point."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    monkeypatch.setattr(
        viewer_module.QInputDialog, "getInt", staticmethod(lambda *a, **k: (168, True))
    )

    window = ViewerWindow()
    try:
        window.new_custom_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 168
        assert window._untitled is True
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_on_dirty_document_confirms_discard_exactly_once(monkeypatch) -> None:
    """new_map_custom() checks _confirm_discard_changes() itself and then
    calls ViewerWindow._create_new_map() directly, not new_map() -- if it
    called new_map() instead, a dirty document would trigger the discard
    prompt twice (once in new_map_custom(), once more inside new_map()),
    after the user already answered the size dialog and possibly the
    large-map confirm too."""
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    monkeypatch.setattr(
        viewer_module.QInputDialog, "getInt", staticmethod(lambda *a, **k: (168, True))
    )

    window = ViewerWindow()
    try:
        window.new_map()
        window.edit_history.cursor = 1  # force is_dirty True with no real edit needed
        assert window.edit_history.is_dirty

        discard_calls = []

        def fake_discard_prompt(*args, **kwargs):
            discard_calls.append(args)
            return QMessageBox.Discard

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(fake_discard_prompt))
        window.new_map_custom()

        assert len(discard_calls) == 1
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == 168
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_cancelled_leaves_document_untouched(monkeypatch) -> None:
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    monkeypatch.setattr(
        viewer_module.QInputDialog, "getInt", staticmethod(lambda *a, **k: (300, False))
    )

    window = ViewerWindow()
    try:
        assert window.scenario is None
        window.new_custom_action.trigger()
        assert window.scenario is None
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_custom_size_above_threshold_confirms_before_creating(monkeypatch) -> None:
    conftest.ensure_qapp()
    from PyQt5.QtWidgets import QMessageBox

    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    tiles = LARGE_MAP_CONFIRM_TILES + 60
    monkeypatch.setattr(
        viewer_module.QInputDialog, "getInt", staticmethod(lambda *a, **k: (tiles, True))
    )

    window = ViewerWindow()
    try:
        calls = []

        def cancel(*args, **kwargs):
            calls.append(args)
            return QMessageBox.Cancel

        monkeypatch.setattr(viewer_module.QMessageBox, "question", staticmethod(cancel))
        window.new_custom_action.trigger()
        assert len(calls) == 1
        assert window.scenario is None  # cancelled -- nothing created

        monkeypatch.setattr(
            viewer_module.QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.Yes)
        )
        window.new_custom_action.trigger()
        mm = window.scenario.map_manager
        assert mm.map_width == mm.map_height == tiles
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_new_map_does_not_touch_disk() -> None:
    """Insurance for the in-memory generation decision (descape.scenario_io.
    load_map_and_units_from_bytes) -- File > New Map must never write to
    descape/templates/, since nothing in the design needs it to."""
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    before = {p: p.read_bytes() for p in TEMPLATE_DIR.glob("*.aoe2scenario")}
    window = ViewerWindow()
    try:
        window.new_map(240)
        after = {p: p.read_bytes() for p in TEMPLATE_DIR.glob("*.aoe2scenario")}
        assert after == before
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_save_as_default_never_points_at_the_template() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_map()
        start_path = window._save_as_start_path()
        assert Path(start_path).is_absolute()
        assert Path(start_path).parent.resolve() != TEMPLATE_DIR.resolve()
        assert "blank_" not in Path(start_path).name
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("tiles", (BLANK_TEMPLATE_TILES, 168))
def test_save_as_on_new_map_adopts_path(tmp_path, monkeypatch, tiles: int) -> None:
    """168, alongside the donor's own 120, proves Save As works end-to-end on
    a *generated* (not shipped-file) size too."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    dest = tmp_path / "mymap.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(dest), "")),
    )

    window = ViewerWindow()
    try:
        window.new_map(tiles)
        window.save_as()

        assert dest.is_file()
        reloaded = load_map_and_units(dest)
        assert reloaded.map_manager.map_width == reloaded.map_manager.map_height == tiles

        assert window._untitled is False
        assert window.scenario.path == dest
        assert window.windowTitle() == f"{dest.name} — DEscape"
        assert BLANK_TEMPLATE_PATH.read_bytes()  # donor template itself untouched (still readable/non-empty)
    finally:
        window.edit_history.mark_saved()
        window.close()


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_open_after_new_clears_untitled() -> None:
    conftest.ensure_qapp()
    from descape.viewer import ViewerWindow

    window = ViewerWindow()
    try:
        window.new_map()
        assert window._untitled is True

        window.load_scenario(BLANK_TEMPLATE_PATH)  # a normal open, no untitled=

        assert window._untitled is False
        assert window.scenario.path == BLANK_TEMPLATE_PATH
        assert window.windowTitle() == f"{BLANK_TEMPLATE_PATH.name} — DEscape"
    finally:
        window.edit_history.mark_saved()
        window.close()
