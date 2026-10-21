"""`.bak`/`.orig` backup files written alongside a save. Two independent
pieces are exercised: descape/backup.py's make_backups() directly, and
scenario_write.write_scenario()'s wiring of it (the no-op short circuit,
the guard-before-backup ordering, backup=False, and the atomic final
write). The last case also drives ViewerWindow.save() to confirm a
blocked save leaves the document dirty.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

import pytest
from AoE2ScenarioParser.scenarios.aoe2_scenario import _decompress_bytes

import descape.backup as backup_module
from descape.backup import BackupFailedError, bak_path, make_backups, orig_path
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from descape.scenario_io import TEMPLATE_DIR, load_map_and_units, names_workshop_subscribed, resolve_detecting_loops
from descape.scenario_write import WriteBlockedError, write_scenario

import conftest


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

    subscribed_target = tmp_path / "mods" / "subscribed" / "12345" / "map.aoe2scenario"
    assert names_workshop_subscribed(subscribed_target)
    with pytest.raises(WriteBlockedError):
        write_scenario(s, subscribed_target)
    assert not bak_path(subscribed_target).exists()
    assert not orig_path(subscribed_target).exists()

    template_target = TEMPLATE_DIR / "blank_960x960.aoe2scenario"
    with pytest.raises(WriteBlockedError):
        write_scenario(s, template_target)
    assert not bak_path(template_target).exists()
    assert not orig_path(template_target).exists()


def test_a_compatdata_target_writes_with_its_backups(tmp_path) -> None:
    """The game profile under a Proton prefix is writable; an overwrite there
    backs up like anywhere else."""
    real_dir = tmp_path / "compatdata" / "813780"
    real_dir.mkdir(parents=True)
    target = _copy_fixture(real_dir)
    before = target.read_bytes()

    s = load_map_and_units(target)
    _flip_one_tile(s)
    result = write_scenario(s, target)

    assert result.wrote
    assert target.read_bytes() != before
    assert bak_path(target).read_bytes() == before
    assert orig_path(target).read_bytes() == before


def _linked_subscribed_folder(tmp_path: Path) -> tuple[Path, Path]:
    """(real, link): a scenario inside a Workshop mods/subscribed/ folder, and
    the same file opened through a directory symlink whose own path lacks it."""
    real_dir = tmp_path / "mods" / "subscribed" / "12345"
    real_dir.mkdir(parents=True)
    real = _copy_fixture(real_dir)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir, target_is_directory=True)
    return real, link_dir / real.name


@conftest.requires_symlinks
def test_a_directory_symlink_into_a_subscribed_folder_is_refused(tmp_path) -> None:
    """The hard rule through a linked folder: the path as written carries no
    mods/subscribed/, so only the resolved check can catch it."""
    from descape import autosave

    real, link_path = _linked_subscribed_folder(tmp_path)
    assert not names_workshop_subscribed(link_path)  # else this passes without the resolve
    before = real.read_bytes()

    s = load_map_and_units(link_path)
    key = autosave.doc_key(link_path, "0123456789abcdef")
    assert autosave.slot_path(key, link_path.name, link_path, "sidecar").parent == autosave.autosave_dir()

    _flip_one_tile(s)  # unmodified bytes would return before any backup, proving nothing
    with pytest.raises(WriteBlockedError):
        write_scenario(s, link_path)
    assert real.read_bytes() == before
    assert sorted(p.name for p in real.parent.iterdir()) == [real.name]


@conftest.requires_symlinks
def test_a_file_symlink_into_a_subscribed_folder_is_refused_not_replaced(tmp_path) -> None:
    """Before the resolved check, Save replaced such a link with a real file."""
    real, _ = _linked_subscribed_folder(tmp_path)
    link_path = tmp_path / "linked.aoe2scenario"
    link_path.symlink_to(real)
    assert not names_workshop_subscribed(link_path)

    s = load_map_and_units(link_path)
    _flip_one_tile(s)
    with pytest.raises(WriteBlockedError):
        write_scenario(s, link_path)
    assert link_path.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", link_path.name, "mods"]
    assert sorted(p.name for p in real.parent.iterdir()) == [real.name]


def test_a_marked_path_is_blocked_before_resolve_can_fail(tmp_path, monkeypatch) -> None:
    """A symlink loop or unreadable parent makes resolve() raise; a path that
    already names mods/subscribed/ must still read "blocked", not "Save failed"."""
    s = load_map_and_units(FIXTURE_PATH)

    def _raise(self, strict=False):
        raise RuntimeError("Symlink loop")

    monkeypatch.setattr(Path, "resolve", _raise)
    with pytest.raises(WriteBlockedError):
        write_scenario(s, tmp_path / "Mods" / "Subscribed" / "map.aoe2scenario")  # case-folded, for Windows


@conftest.requires_symlinks
def test_a_looped_save_path_is_refused_with_zero_backups(tmp_path) -> None:
    """Fail closed: a path that can't be resolved can't be shown to be outside
    a mods/subscribed/ folder. Unrefused, the atomic write would replace the link."""
    a, b = tmp_path / "a.aoe2scenario", tmp_path / "b.aoe2scenario"
    a.symlink_to(b)
    b.symlink_to(a)

    s = load_map_and_units(FIXTURE_PATH)
    _flip_one_tile(s)  # unmodified bytes could return before the write, proving nothing
    with pytest.raises(WriteBlockedError, match="cannot be resolved"):
        write_scenario(s, a)
    assert a.is_symlink() and b.is_symlink()
    for p in (a, b):
        assert not bak_path(p).exists() and not bak_path(p).is_symlink()
        assert not orig_path(p).exists() and not orig_path(p).is_symlink()


@conftest.requires_symlinks
def test_a_looped_save_path_is_refused_when_resolve_does_not_raise(tmp_path, monkeypatch) -> None:
    """Python 3.13+ semantics: non-strict resolve() returns a loop unresolved
    instead of raising. Pins the stat() check on 3.11, where a real loop raises anyway."""
    a, b = tmp_path / "a.aoe2scenario", tmp_path / "b.aoe2scenario"
    a.symlink_to(b)
    b.symlink_to(a)
    s = load_map_and_units(FIXTURE_PATH)
    _flip_one_tile(s)

    conftest.resolve_like_py313(monkeypatch)
    with pytest.raises(WriteBlockedError, match="cannot be resolved"):
        write_scenario(s, a)
    assert a.is_symlink() and b.is_symlink()
    for p in (a, b):
        assert not bak_path(p).exists() and not bak_path(p).is_symlink()
        assert not orig_path(p).exists() and not orig_path(p).is_symlink()


@pytest.mark.parametrize("py313_resolve", [False, True])
@conftest.requires_symlinks
def test_resolve_detecting_loops_raises_on_every_loop_shape(tmp_path, monkeypatch, py313_resolve) -> None:
    """Bare, with a missing tail (the Save As case), and with a ".." after the
    loop, which 3.13+ resolve() applies as text to land on a plain missing path."""
    a = conftest.symlink_loop(tmp_path)
    if py313_resolve:
        conftest.resolve_like_py313(monkeypatch)
    for path in (a, a / "new.aoe2scenario", a / ".." / "new.aoe2scenario"):
        with pytest.raises((OSError, RuntimeError)) as info:  # 3.11's real resolve() raises RuntimeError first
            resolve_detecting_loops(path)
        if py313_resolve:
            assert info.value.errno == errno.ELOOP


@conftest.requires_symlinks
def test_resolve_detecting_loops_returns_a_missing_or_plain_path(tmp_path) -> None:
    plain = tmp_path / "plain.aoe2scenario"
    plain.write_bytes(b"")
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    assert resolve_detecting_loops(plain) == plain.resolve()
    assert resolve_detecting_loops(tmp_path / "missing" / "new.aoe2scenario") == (
        tmp_path / "missing" / "new.aoe2scenario"
    ).resolve()
    assert resolve_detecting_loops(link / "plain.aoe2scenario") == plain.resolve()
    assert resolve_detecting_loops(plain / "child.aoe2scenario") == plain.resolve() / "child.aoe2scenario"  # ENOTDIR


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
