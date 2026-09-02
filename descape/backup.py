"""On-save write insurance, written beside the target file:

- `.bak` -- the conventional previous-version backup. Refreshed on every
  save, so it always holds the state immediately before the most recent
  write.
- `.orig` -- a one-time pristine snapshot of a pre-existing file, captured
  lazily the first time a save is about to overwrite it. Never touched by
  any later save. Deleting `.orig` and saving again recreates it from the
  then-current file -- deliberate, not a bug: deleting it is read as "reset
  the baseline", and remembering that one once existed would be more state
  than that earns.

Qt-free so it tests in the default tier, matching unit_pick.py/settings.py.

Each copy goes through a temp sibling + os.replace, not a bare
shutil.copyfile onto the destination -- a bare copy truncates the
destination first, so a disk-full failure partway through would destroy the
previous good `.bak` and leave garbage in its place. The temp sits next to
the backup itself, so os.replace never crosses a filesystem boundary.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from descape.scenario_write import WriteBlockedError

BAK_SUFFIX = ".bak"
ORIG_SUFFIX = ".orig"


class BackupFailedError(WriteBlockedError):
    """Raised when a `.bak`/`.orig` copy fails partway through, leaving every
    pre-existing backup intact. Subclasses WriteBlockedError so viewer.py's
    existing WriteBlockedError handling in save()/save_as() already aborts
    correctly with no separate except clause -- shows "Save blocked", logs,
    and returns without edit_history.mark_saved(), leaving the document
    dirty and the on-disk file untouched."""


def bak_path(target: Path) -> Path:
    return target.with_name(target.name + BAK_SUFFIX)


def orig_path(target: Path) -> Path:
    return target.with_name(target.name + ORIG_SUFFIX)


def _atomic_copy(src: Path, dest: Path) -> None:
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def make_backups(target: Path) -> list[Path]:
    """No-op returning [] if `target` does not exist (a Save As to a fresh
    name backs up nothing). Otherwise:
      1. `.orig` -- created only if absent.
      2. `.bak` -- always refreshed, from `target`'s state before this call.
    Returns the paths actually written, for the caller to log.

    `.orig` is written first, `.bak` second: if the disk is full, aborting
    before touching `.bak` loses less. Raises BackupFailedError on any
    OSError, having left every pre-existing backup intact -- `_atomic_copy`
    never truncates a destination in place."""
    if not target.exists():
        return []
    written: list[Path] = []
    try:
        orig = orig_path(target)
        if not orig.exists():
            _atomic_copy(target, orig)
            written.append(orig)
        bak = bak_path(target)
        _atomic_copy(target, bak)
        written.append(bak)
    except OSError as e:
        raise BackupFailedError(f"Failed to back up {target}: {e}") from e
    return written
