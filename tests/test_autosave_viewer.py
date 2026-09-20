"""ViewerWindow's autosave timer, tick and Saving tab -- the half of the
feature that needs a real window. descape/autosave.py's own slot/index
mechanics are tests/test_autosave.py (Qt-free, default tier).

SettingsDialog is constructed directly, never through
ViewerWindow._show_settings(), which exec_()s and would hang offscreen --
the same rule tests/test_elev_step_slider.py documents. Teardown goes
through conftest.close_window(), which mark_saved()s first so closeEvent's
discard prompt can't block forever.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from descape import autosave, settings

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def _dirty(window) -> None:
    """A real terrain edit, so edit_history.is_dirty and the cursor both
    move -- the tick's first two gates read exactly those."""
    tiles = window.scenario.map_manager.terrain
    new_terrain = 15 if tiles[0].terrain_id == 2 else 2
    window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", new_terrain))
    # What a real paint's own tail does; the "*" assertion below is only
    # meaningful against a title that was actually put in step first.
    window._update_title()


def _slots() -> list[Path]:
    return sorted(autosave.autosave_dir().glob(f"*{autosave.AUTOSAVE_SUFFIX}"))


def _window_on(tmp_path):
    # Its own subdirectory: tmp_path also holds the isolated config.yaml, and
    # the backup assertions below count what lands beside the scenario.
    map_dir = tmp_path / "maps"
    map_dir.mkdir()
    target = map_dir / "mymap.aoe2scenario"
    shutil.copyfile(BLANK_FIXTURE, target)
    window = conftest.blank_window(load=False)
    window.load_scenario(target)
    assert window.scenario is not None
    return window, target


def test_a_tick_on_a_clean_document_writes_nothing(tmp_path) -> None:
    window, _target = _window_on(tmp_path)
    try:
        window._autosave_tick()
        assert _slots() == []
    finally:
        conftest.close_window(window)


def test_a_tick_on_a_dirty_document_writes_a_slot_and_leaves_it_dirty(tmp_path) -> None:
    """The single most important assertion in this feature. An autosave is a
    recovery snapshot, not a save: clearing the dirty state would drop the
    title's "*" AND make _confirm_discard_changes() return True, so the user
    would close without a prompt and lose the work."""
    window, target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._autosave_tick()

        assert len(_slots()) == 1
        assert window.edit_history.is_dirty
        assert window.windowTitle().startswith("*")
        assert target.stat().st_size == BLANK_FIXTURE.stat().st_size  # real file untouched
        assert "Autosaved to" in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


def test_a_second_tick_with_no_further_edit_writes_nothing(tmp_path) -> None:
    """The _autosaved_at_cursor gate: a document edited once and then left
    alone is dirty forever, and would otherwise re-serialize identical bytes
    every interval for the rest of the session."""
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._autosave_tick()
        window.status_log.clear()
        window._autosave_tick()

        assert len(_slots()) == 1
        assert "Autosaved to" not in window.status_log.toPlainText()
    finally:
        conftest.close_window(window)


def test_a_busy_tick_writes_nothing_and_arms_the_retry(tmp_path) -> None:
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._busy = True
        window._autosave_tick()

        assert _slots() == []
        assert window._autosave_retry_timer.isActive()
    finally:
        window._busy = False
        conftest.close_window(window)


def test_a_retry_that_saw_a_stroke_re_arms_instead_of_writing(tmp_path) -> None:
    """The settle rule. Without this the retry fires on the first clear
    tick, which can land exactly as the user begins the next stroke -- the
    mid-drag freeze the gates exist to avoid. Both retry behaviours pass
    every other assertion in this file, so this is the only one that pins
    it."""
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._busy = True
        window._autosave_tick()  # arms the retry, flags the stroke
        window._busy = False

        window._autosave_tick()  # the retry firing: clear, but a stroke was seen
        assert _slots() == []
        assert window._autosave_retry_timer.isActive()

        window._autosave_tick()  # the next one, after a clear window
        assert len(_slots()) == 1
    finally:
        conftest.close_window(window)


def test_a_tick_mid_stroke_writes_nothing(tmp_path) -> None:
    """Serializing between begin_stroke() and commit_stroke() would write a
    half-applied stroke."""
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        tiles = window.scenario.map_manager.terrain
        window.edit_history.begin_stroke(tiles[:1])
        window._autosave_tick()

        assert _slots() == []
        assert window._autosave_retry_timer.isActive()
    finally:
        window.edit_history.commit_stroke("paint", tiles[:1])
        conftest.close_window(window)


def test_a_successful_save_discards_that_documents_autosaves(tmp_path) -> None:
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._autosave_tick()
        assert len(_slots()) == 1

        window.save()
        assert _slots() == []
        assert autosave.entries() == []
    finally:
        conftest.close_window(window)


def test_save_as_discards_the_untitled_slots_it_was_autosaved_under(tmp_path, monkeypatch) -> None:
    """Pins the captured-old-key rule: save_as() retargets scenario.path and
    clears _untitled BEFORE returning, so a freshly-computed key would name
    the new path, discard nothing, and leak the untitled slots into Recover
    under a stale "Untitled" label -- while still looking green."""
    import descape.viewer as viewer_module

    dest = tmp_path / "named.aoe2scenario"
    monkeypatch.setattr(
        viewer_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(dest), "")),
    )

    window = conftest.blank_window(load=False)
    try:
        window.load_scenario(BLANK_FIXTURE, untitled=True)
        assert window._untitled
        _dirty(window)
        window._autosave_tick()
        assert len(_slots()) == 1

        window.save_as()

        assert not window._untitled
        assert _slots() == []
        assert autosave.entries() == []
    finally:
        conftest.close_window(window)


@pytest.mark.parametrize("backups_enabled", [True, False])
def test_the_backups_switch_gates_the_bak_and_orig_pair(tmp_path, backups_enabled: bool) -> None:
    """Guards the shipped, already-verified backup path in both directions:
    a one-way assertion would pass on a save() that simply stopped writing
    backups at all."""
    settings.set_backups_enabled(backups_enabled)
    window, target = _window_on(tmp_path)
    try:
        _dirty(window)
        window.save()

        produced = sorted(p.suffix for p in target.parent.iterdir() if p != target)
        assert (produced != []) is backups_enabled, produced
    finally:
        conftest.close_window(window)


def test_the_saving_tab_controls_persist_and_re_arm_the_timer(tmp_path) -> None:
    from descape.viewer import SettingsDialog

    window = conftest.blank_window(load=False)
    dialog = SettingsDialog(window)
    try:
        dialog.autosave_interval_combo.setCurrentIndex(
            settings.AUTOSAVE_INTERVAL_CHOICES.index(15)
        )
        assert settings.get_autosave_interval_min() == 15
        assert window._autosave_timer.interval() == 15 * 60 * 1000

        dialog.autosave_enabled_check.setChecked(False)
        assert settings.get_autosave_enabled() is False
        assert not window._autosave_timer.isActive()

        dialog.autosave_retention_combo.setCurrentIndex(
            settings.AUTOSAVE_RETENTION_CHOICES.index(10)
        )
        assert settings.get_autosave_retention() == 10
        dialog.autosave_location_combo.setCurrentIndex(1)
        assert settings.get_autosave_location() == "sidecar"
        dialog.backups_enabled_check.setChecked(False)
        assert settings.get_backups_enabled() is False
    finally:
        dialog.close()
        conftest.close_window(window)


def test_a_disabled_autosave_tick_writes_nothing(tmp_path) -> None:
    window, _target = _window_on(tmp_path)
    try:
        settings.set_autosave_enabled(False)
        _dirty(window)
        window._autosave_tick()
        assert _slots() == []
    finally:
        conftest.close_window(window)


def test_the_recover_dialog_lists_a_slot_and_opens_it_untitled(tmp_path) -> None:
    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._autosave_tick()

        window.show_recover_autosave()
        dialog = window._recover_dialog
        assert dialog.tree.topLevelItemCount() == 1

        entry = dialog.selected_entry()
        window.edit_history.mark_saved()  # so _confirm_discard_changes() doesn't prompt
        window._open_autosave_entry(entry)

        assert window._untitled
        assert window.scenario is not None
    finally:
        conftest.close_window(window)


def test_the_tick_writes_what_a_real_save_would(tmp_path) -> None:
    """Byte-identity against a write built from the same edit models, not
    from a hand-listed set: the tick is a third write call site that has to
    stay in lockstep with save()/save_as() forever, and the original plan's
    tick call already omitted messages= once, which would have silently
    dropped every Messages-mode edit."""
    from descape.scenario_write import write_scenario

    window, _target = _window_on(tmp_path)
    try:
        _dirty(window)
        window._autosave_tick()
        slot, = _slots()

        reference = tmp_path / "reference.aoe2scenario"
        write_scenario(window.scenario, reference, backup=False, **window._edit_model_kwargs())

        assert slot.read_bytes() == reference.read_bytes()
    finally:
        conftest.close_window(window)


def test_every_write_scenario_edit_model_reaches_the_tick(tmp_path) -> None:
    """The lockstep guard, reflectively: a new edit-model parameter added to
    write_scenario() and wired into save() but not here fails this, rather
    than silently shipping a tick that drops that whole domain's edits."""
    import inspect

    from descape.scenario_write import write_scenario

    window = conftest.blank_window(load=False)
    try:
        params = set(inspect.signature(write_scenario).parameters) - {"scenario", "out_path", "backup"}
        assert params == set(window._edit_model_kwargs())
    finally:
        conftest.close_window(window)


def test_an_autosave_round_trips_a_messages_mode_edit(tmp_path) -> None:
    """The concrete failure the lockstep guard above prevents. A generic
    byte-identity assertion alone would not have caught the original
    omission: both sides would have omitted messages=."""
    from descape.messages_fields import MESSAGE_FIELDS
    from descape.scenario_io import load_map_and_units

    window, _target = _window_on(tmp_path)
    try:
        spec = MESSAGE_FIELDS[0]
        window.set_message_field(spec.field_id, "autosaved instructions")
        window._autosave_tick()
        slot, = _slots()

        recovered = load_map_and_units(slot)
        assert "autosaved instructions" in str(
            recovered._scenario.sections["Messages"].retriever_map[spec.retriever].data
        )
    finally:
        conftest.close_window(window)
