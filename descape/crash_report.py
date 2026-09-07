"""Crash dump writer/pruner/sweeper for unhandled exceptions and hard aborts.

Qt-free -- no PyQt import here -- so this tests in the default tier, same
reasoning as backup.py. `dump_dir` is always a parameter, never a module
constant: asset_source.py's write_config_file() documents why (settings.py
needs its own monkeypatched CONFIG_PATH to actually take effect), and a
module-level DUMP_DIR here would fight the same tests/conftest.py
_isolated_settings autouse fixture.

Dump lifecycle: a fresh dump is `crash-<ts>-<hex>.txt`. Once the app has
shown it to the user (in-app dialog or the next-launch sweep),
mark_reported() renames it to the same name plus `.reported` -- so it never
nags twice, but the file is still there to attach to a bug report.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

DUMP_PREFIX = "crash-"
DUMP_SUFFIX = ".txt"
REPORTED_SUFFIX = ".reported"
FAULTHANDLER_LOG_NAME = "faulthandler.log"


def build_report(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb,
    *,
    version: str,
    log_text: str,
    qt_version: str = "unknown",
    frozen: bool = False,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = (
        f"DEscape crash report\n"
        f"Time: {timestamp}\n"
        f"Version: {version}\n"
        f"Frozen build: {frozen}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"Platform: {platform.platform()}\n"
        f"Qt: {qt_version}\n"
        f"\nTraceback:\n"
    )
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    return f"{header}{tb_text}\nDebug log:\n{log_text}\n"


def _dump_filename() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{DUMP_PREFIX}{timestamp}-{uuid.uuid4().hex[:8]}{DUMP_SUFFIX}"


def write_report(text: str, dump_dir: Path) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dest = dump_dir / _dump_filename()
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, dest)
    return dest


def prune(dump_dir: Path, keep: int = 5) -> None:
    """Deletes the oldest reports beyond `keep`, by mtime. Only touches
    normal dumps, not `.reported` ones or the live faulthandler log."""
    if not dump_dir.is_dir():
        return
    dumps = sorted(
        dump_dir.glob(f"{DUMP_PREFIX}*{DUMP_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
    )
    for path in dumps[:-keep] if keep > 0 else dumps:
        path.unlink(missing_ok=True)


def pending_reports(dump_dir: Path) -> list[Path]:
    if not dump_dir.is_dir():
        return []
    return sorted(dump_dir.glob(f"{DUMP_PREFIX}*{DUMP_SUFFIX}"), key=lambda p: p.stat().st_mtime)


def mark_reported(path: Path) -> Path:
    dest = path.with_name(path.name + REPORTED_SUFFIX)
    os.replace(path, dest)
    return dest


def extract_summary(report_text: str) -> str:
    """The one-line 'ExceptionType: message' summary out of a report built by
    build_report() -- the last non-blank line of its Traceback section. Used
    by the next-launch sweep, which only has the dump file on disk, not the
    original exception object."""
    marker = "\nTraceback:\n"
    idx = report_text.find(marker)
    if idx == -1:
        lines = report_text.splitlines()
        return lines[0] if lines else ""
    tb_section = report_text[idx + len(marker) :]
    end = tb_section.find("\nDebug log:\n")
    if end != -1:
        tb_section = tb_section[:end]
    lines = [line for line in tb_section.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def fingerprint(exc_type: type[BaseException], exc_tb) -> tuple:
    """(exc_type, last traceback frame's filename+lineno) -- stable enough to
    recognize "the same bug re-raising" (e.g. a raising paintEvent hit on
    every repaint) without needing exact traceback equality."""
    frames = traceback.extract_tb(exc_tb)
    location = (frames[-1].filename, frames[-1].lineno) if frames else (None, None)
    return (exc_type, location)


class RateLimiter:
    """Writes a dump for the first occurrence of each fingerprint, up to a
    hard cap on total dumps per process. Without this, a raising paintEvent
    would write a dump and open a dialog on every repaint."""

    def __init__(self, max_dumps: int = 3) -> None:
        self.max_dumps = max_dumps
        self.written = 0
        self._seen: set[tuple] = set()

    def should_write(self, fp: tuple) -> bool:
        if fp in self._seen or self.written >= self.max_dumps:
            return False
        self._seen.add(fp)
        self.written += 1
        return True


def rotate_faulthandler_log(dump_dir: Path) -> Path | None:
    """If a previous session's faulthandler.log has content, rename it into
    the normal pending/.reported pipeline before this session's
    faulthandler.enable() truncates it. Returns the new path, or None if
    there was nothing (or nothing non-empty) to rotate."""
    log_path = dump_dir / FAULTHANDLER_LOG_NAME
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return None
    dump_dir.mkdir(parents=True, exist_ok=True)
    mtime = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    dest = dump_dir / f"{DUMP_PREFIX}{mtime}-faulthandler{DUMP_SUFFIX}"
    os.replace(log_path, dest)
    return dest
