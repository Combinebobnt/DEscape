"""`.bak`/`.orig` backup files written alongside a save. Two independent
pieces are exercised: descape/backup.py's make_backups() directly, and
scenario_write.write_scenario()'s wiring of it (the no-op short circuit,
the guard-before-backup ordering, backup=False, and the atomic final
write). The last case also drives ViewerWindow.save() to confirm a
blocked save leaves the document dirty.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

import conftest
import descape.backup as backup_module
from descape.backup import BackupFailedError, bak_path, make_backups, orig_path
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import FORBIDDEN_WRITE_MARKER, TEMPLATE_DIR, load_map_and_units
from descape.scenario_write import WriteBlockedError, write_scenario


def _copy_fixture(tmp_path: Path, name: str = "map.aoe2scenario") -> Path:
    target = tmp_path / name
    shutil.copyfile(FIXTURE_PATH, target)
    return target


def _flip_one_tile(scenario) -> None:
    """A single real edit, enough to make the next write_scenario() call
    produce bytes that differ from whatever is already on disk."""
    tile = scenario.map_manager.terrain[0]
    tile.terrain_id = 15 if tile.terrain_id == 2 else 2


def test_first_save_creates_bak_and_orig_both_matching_pre_save_bytes(tmp_path) -> None:
    target = _copy_fixture(tmp_path)
    before = target.read_bytes()

    s = load_map_and_units(target)
    _flip_one_tile(s)
    result = write_scenario(s, target)

    assert result.wrote
    assert set(result.backups) == {bak_path(target), orig_path(target)}
    assert bak_path(target).read_bytes() == before
    assert orig_path(target).read_bytes() == before


def test_second_save_refreshes_bak_but_not_orig(tmp_path) -> None:
    target = _copy_fixture(tmp_path)
    original_bytes = target.read_bytes()

    s = load_map_and_units(target)
    _flip_one_tile(s)
    write_scenario(s, target)
    after_first_save = target.read_bytes()
    orig_mtime_after_first = orig_path(target).stat().st_mtime_ns

    s2 = load_map_and_units(target)
    _flip_one_tile(s2)
    result = write_scenario(s2, target)

    assert result.wrote
    assert bak_path(target).read_bytes() == after_first_save
    assert orig_path(target).read_bytes() == original_bytes
    assert orig_path(target).stat().st_mtime_ns == orig_mtime_after_first


def test_a_noop_save_does_not_refresh_bak(tmp_path) -> None:
    target = _copy_fixture(tmp_path)

    s = load_map_and_units(target)
    _flip_one_tile(s)
    write_scenario(s, target)  # save #1: seeds .bak with the pre-edit state

    s2 = load_map_and_units(target)
    _flip_one_tile(s2)
    write_scenario(s2, target)  # save #2: .bak now holds save #1's state
    bak_bytes = bak_path(target).read_bytes()
    bak_mtime = bak_path(target).stat().st_mtime_ns
    target_mtime = target.stat().st_mtime_ns

    s3 = load_map_and_units(target)  # no edits made
    result = write_scenario(s3, target)  # save #3: byte-identical, a no-op

    assert not result.wrote
    assert result.backups == []
    assert bak_path(target).read_bytes() == bak_bytes
    assert bak_path(target).stat().st_mtime_ns == bak_mtime
    assert target.stat().st_mtime_ns == target_mtime


def test_a_noop_save_on_a_never_saved_file_creates_no_orig(tmp_path) -> None:
    target = _copy_fixture(tmp_path)
    original_bytes = target.read_bytes()

    s = load_map_and_units(target)
    result = write_scenario(s, target)  # no edits at all -- byte-identical

    assert not result.wrote
    assert not bak_path(target).exists()
    assert not orig_path(target).exists()

    s2 = load_map_and_units(target)
    _flip_one_tile(s2)
    result2 = write_scenario(s2, target)

    assert result2.wrote
    assert orig_path(target).read_bytes() == original_bytes


def test_save_as_to_a_fresh_name_creates_no_backups(tmp_path) -> None:
    s = load_map_and_units(FIXTURE_PATH)
    dest = tmp_path / "fresh.aoe2scenario"
    result = write_scenario(s, dest)

    assert result.wrote
    assert result.backups == []
    assert not bak_path(dest).exists()
    assert not orig_path(dest).exists()


def test_save_as_onto_an_existing_file_backs_up_that_file(tmp_path) -> None:
    existing = _copy_fixture(tmp_path, "existing.aoe2scenario")
    before = existing.read_bytes()

    s = load_map_and_units(FIXTURE_PATH)
    _flip_one_tile(s)
    result = write_scenario(s, existing)

    assert result.wrote
    assert set(result.backups) == {bak_path(existing), orig_path(existing)}
    assert bak_path(existing).read_bytes() == before
    assert orig_path(existing).read_bytes() == before


def test_blocked_write_leaves_zero_backup_files(tmp_path) -> None:
    s = load_map_and_units(FIXTURE_PATH)

    compatdata_target = tmp_path / "compatdata" / "map.aoe2scenario"
    assert FORBIDDEN_WRITE_MARKER in str(compatdata_target)
    with pytest.raises(WriteBlockedError):
        write_scenario(s, compatdata_target)
    assert not bak_path(compatdata_target).exists()
    assert not orig_path(compatdata_target).exists()

    template_target = TEMPLATE_DIR / "blank_960x960.aoe2scenario"
    with pytest.raises(WriteBlockedError):
        write_scenario(s, template_target)
    assert not bak_path(template_target).exists()
    assert not orig_path(template_target).exists()


def test_backup_failure_aborts_the_save(tmp_path, monkeypatch) -> None:
    target = _copy_fixture(tmp_path)
    before = target.read_bytes()

    s = load_map_and_units(target)
    _flip_one_tile(s)
    write_scenario(s, target)  # seed a real .bak/.orig
    bak_bytes = bak_path(target).read_bytes()
    orig_bytes = orig_path(target).read_bytes()
    target_bytes_before_failure = target.read_bytes()

    def _raise(_target: Path) -> list[Path]:
        raise BackupFailedError("simulated backup failure")

    monkeypatch.setattr(backup_module, "make_backups", _raise)

    s2 = load_map_and_units(target)
    _flip_one_tile(s2)
    with pytest.raises(BackupFailedError):
        write_scenario(s2, target)

    assert target.read_bytes() == target_bytes_before_failure
    assert bak_path(target).read_bytes() == bak_bytes
    assert orig_path(target).read_bytes() == orig_bytes
    assert before  # sanity: fixture actually had bytes to compare against


def test_backup_false_writes_the_file_and_creates_nothing_else(tmp_path) -> None:
    target = _copy_fixture(tmp_path)

    s = load_map_and_units(target)
    _flip_one_tile(s)
    result = write_scenario(s, target, backup=False)

    assert result.wrote
    assert result.backups == []
    assert not bak_path(target).exists()
    assert not orig_path(target).exists()


def test_zero_edit_round_trip_is_still_byte_identical_through_the_atomic_write(tmp_path) -> None:
    s = load_map_and_units(FIXTURE_PATH)
    out = tmp_path / "zero_edit.aoe2scenario"
    write_scenario(s, out)

    written = out.read_bytes()
    assert written[: len(s.header_bytes)] == s.header_bytes
    body = _decompress_bytes(written[len(s.header_bytes) :])
    assert body == s.decompressed_body


def test_make_backups_returns_empty_for_a_nonexistent_target(tmp_path) -> None:
    assert make_backups(tmp_path / "does_not_exist.aoe2scenario") == []


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_failed_save_leaves_is_dirty_true(tmp_path, monkeypatch) -> None:
    """Piggybacks on the backup-failure case (8), but through the viewer:
    BackupFailedError is a WriteBlockedError, so ViewerWindow.save()'s
    existing except WriteBlockedError branch already returns before
    edit_history.mark_saved() runs -- this pins that the document is left
    dirty, not just that the on-disk bytes are untouched."""
    conftest.ensure_qapp()
    import descape.viewer as viewer_module
    from descape.viewer import ViewerWindow

    target = _copy_fixture(tmp_path)

    def _raise(_target: Path) -> list[Path]:
        raise BackupFailedError("simulated backup failure")

    monkeypatch.setattr(backup_module, "make_backups", _raise)
    # QMessageBox.critical() is a modal exec_() call, same offscreen-hang
    # hazard as .question()/.warning() elsewhere in this suite.
    monkeypatch.setattr(viewer_module.QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    window = ViewerWindow()
    try:
        window.load_scenario(target)
        tiles = window.scenario.map_manager.terrain
        new_terrain = 15 if tiles[0].terrain_id == 2 else 2
        window.edit_history.apply("paint", tiles, lambda: setattr(tiles[0], "terrain_id", new_terrain))

        assert window.edit_history.is_dirty
        window.save()
        assert window.edit_history.is_dirty
    finally:
        window.edit_history.mark_saved()
        window.close()
