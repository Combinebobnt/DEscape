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

A tester attaches dumps to public issues, so every dump (a built report and
a rotated faulthandler log alike) goes through scrub_user_paths() first: the
home directory becomes `~`, and any other path segment equal to the username
(a `/media/<user>/` mount) becomes `<user>`.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import platform
import re
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path

from descape import debug_log

DUMP_PREFIX = "crash-"
DUMP_SUFFIX = ".txt"
REPORTED_SUFFIX = ".reported"
FAULTHANDLER_LOG_NAME = "faulthandler.log"
SCRUB_TMP_NAME = ".faulthandler-scrub.tmp"
USER_PLACEHOLDER = "<user>"
_SEPARATORS = "/\\"


def _usernames(home: str) -> set[str]:
    names = set()
    # getuser() raises OSError, KeyError or ImportError depending on the Python version.
    with contextlib.suppress(Exception):
        names.add(getpass.getuser())
    if home:
        names.add(re.split(r"[/\\]", home)[-1])
    return {name for name in names if name}


def scrub_user_paths(text: str) -> str:
    """The home directory replaced with `~`, then every whole path segment
    equal to the username with `<user>`.

    Home and username are read per call, so this answers for the environment
    the program runs in. A home of `/` or an empty one is left alone, since
    replacing it would rewrite unrelated text. The home must end at a
    separator or a non-name character, so home `/home/al` leaves
    `/home/alpha` alone. A segment needs a separator before it, and after it
    a separator, any character but a word character, `.` or `-`, or the end
    of the text, so user `al` never touches `/alpha/` or `/al.bak`; that rule
    is the only guard, for any username.
    """
    home = os.path.expanduser("~").rstrip(_SEPARATORS)
    if home:
        text = re.sub(rf"{re.escape(home)}(?=[/\\]|[^\w.\-]|$)", "~", text)
    for name in sorted(_usernames(home), key=len, reverse=True):
        pattern = rf"(?<=[/\\]){re.escape(name)}(?=[/\\]|[^\w.\-]|$)"
        text = re.sub(pattern, USER_PLACEHOLDER, text)
    return text


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
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
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
    # Once, on the whole text, so no section can carry a path past it.
    return scrub_user_paths(f"{header}{tb_text}\nDebug log:\n{log_text}\n")


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
    there was nothing (or nothing non-empty) to rotate. The rotated copy is
    scrubbed (scrub_user_paths) in place on a best-effort basis: a failed
    scrub is logged and the unscrubbed copy kept, since the sweep scrubs the
    text again before showing it."""
    log_path = dump_dir / FAULTHANDLER_LOG_NAME
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return None
    dump_dir.mkdir(parents=True, exist_ok=True)
    mtime = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y%m%d-%H%M%S")
    dest = dump_dir / f"{DUMP_PREFIX}{mtime}-faulthandler{DUMP_SUFFIX}"
    os.replace(log_path, dest)
    try:
        _scrub_file(dest)
    except OSError as exc:
        # Must not block launch; the sweep scrubs the text again before showing it.
        debug_log.log(f"crash report: could not scrub {dest.name} ({exc!r})")
    return dest


def _scrub_file(path: Path) -> None:
    """Rewrites `path` scrubbed, via a temp in the same dir and os.replace, so
    a failure leaves the unscrubbed original rather than a truncated one."""
    text = path.read_text(encoding="utf-8", errors="replace")
    scrubbed = scrub_user_paths(text)
    if scrubbed == text:
        return
    # One fixed name, overwritten each time, so failures never pile up temps.
    # Not crash-*.txt, so a leftover never reaches the sweep.
    tmp = path.parent / SCRUB_TMP_NAME
    tmp.write_text(scrubbed, encoding="utf-8")
    # A leftover temp may carry a stale mode; keep the original's.
    os.chmod(tmp, path.stat().st_mode & 0o7777)
    os.replace(tmp, path)
