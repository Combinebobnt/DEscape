"""Autosave slot naming, the on-disk index, rotation, listing and pruning.

Qt-free, like backup.py and settings.py, so it runs in the default test tier.
Owns everything about *where* a recovery snapshot lives and which ones
survive; it never serializes a scenario itself -- descape/viewer.py's tick
calls write_scenario() and hands the resulting path back here.

An autosave slot is a legitimate, fully-formed .aoe2scenario written through
the same path a real save takes, not a second-class artifact. What makes it a
snapshot rather than a save is only that it goes to its own name and never
clears the document's unsaved-changes state.

The suffix is APPENDED, not substituted -- June_Event.aoe2scenario.20260918-
051200.autosave -- following descape/backup.py's own rationale: that keeps
slots out of the Open dialog's *.aoe2scenario filter and out of
tests/conftest.py's corpus glob. Naming them foo.autosave.aoe2scenario would
pollute both.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from descape import asset_source
from descape.scenario_io import FORBIDDEN_WRITE_MARKER, TEMPLATE_DIR

AUTOSAVE_SUFFIX = ".autosave"
AUTOSAVE_DIRNAME = "autosave"
INDEX_NAME = "index.yaml"
STALE_AFTER_DAYS = 30


@dataclass(frozen=True)
class AutosaveEntry:
    """One row of the Recover dialog. `display_path` is the document the slot
    came from, as text -- an untitled document has no real one, so `untitled`
    is what the dialog labels on, never a path that exists."""

    key: str
    path: Path
    display_path: str
    untitled: bool
    saved_at: float
    size: int


def autosave_dir() -> Path:
    """The central slot directory, beside config.yaml. Resolved at call time,
    not import time: tests (and the eyeball tools) monkeypatch
    asset_source.CONFIG_PATH, and a module-level constant would freeze the
    real user's directory in before they ever got the chance."""
    return asset_source.CONFIG_PATH.parent / AUTOSAVE_DIRNAME


def index_path() -> Path:
    return autosave_dir() / INDEX_NAME


def doc_key(path: Path | None, doc_id: str) -> str:
    """The rotation key for one document. A named document keys on its
    resolved path, NOT on doc_id, so reopening the same file in a later
    session rotates the same slots instead of growing a fresh set forever.

    An untitled document has no path to key on and gets its own per-document
    doc_id instead, which is the one unbounded case here: twenty File > New
    maps leave twenty keys, each holding up to `retention` slots, until
    prune_stale()'s cutoff. Bounded by time rather than by count, which is
    accepted deliberately -- the slots are small and File > New is not a
    high-frequency action. If it ever bites, the fix is a MAX_KEYS cap
    evicting least-recently-touched, not a shorter cutoff.
    """
    if path is None:
        return hashlib.sha1(f"untitled:{doc_id}".encode(), usedforsecurity=False).hexdigest()[:12]
    return hashlib.sha1(str(path.resolve()).encode(), usedforsecurity=False).hexdigest()[:12]


def _falls_back_to_central(source: Path | None) -> bool:
    """Three source paths cannot be served beside the file, so sidecar mode
    writes centrally for them instead of silently doing nothing:

    1. An untitled document, which has no path at all.
    2. A compatdata/ path -- write_scenario() refuses any such out_path
       (the AGENTS.md hard rule). These Workshop/Proton files are precisely
       the ones the user has no second copy of, so falling back is the whole
       point rather than a corner case.
    3. A shipped template, which write_scenario() also refuses.

    Because the slot path is then never under compatdata/, that hard rule
    holds structurally: write_scenario()'s own guard is what enforces it, and
    nothing here weakens it.
    """
    if source is None:
        return True
    if FORBIDDEN_WRITE_MARKER in str(source):
        return True
    try:
        return source.resolve().parent == TEMPLATE_DIR.resolve()
    except OSError:
        return True


def timestamp_text(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if timestamp is None else timestamp).strftime("%Y%m%d-%H%M%S")


def slot_path(
    key: str,
    display_name: str,
    source: Path | None,
    location: str,
    timestamp: float | None = None,
) -> Path:
    """Where this document's next slot goes. Central filenames carry the key
    so two documents that share a file name (a copy in another folder, say)
    cannot collide in one flat directory; sidecar ones don't need it, being
    already namespaced by their own folder."""
    stamp = timestamp_text(timestamp)
    if location == "sidecar" and not _falls_back_to_central(source):
        return source.parent / f"{source.name}.{stamp}{AUTOSAVE_SUFFIX}"
    return autosave_dir() / f"{display_name}.{key}.{stamp}{AUTOSAVE_SUFFIX}"


def _load_index() -> dict:
    path = index_path()
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _save_index(index: dict) -> None:
    asset_source.write_config_file(index_path(), index)


def record(key: str, slot: Path, display_path: str, untitled: bool, timestamp: float | None = None) -> None:
    """Add one written slot to the index. The index is a convenience for the
    dialog, not the source of truth -- entries() drops rows whose file has
    vanished, and the .autosave files themselves are what actually matter."""
    index = _load_index()
    doc = index.setdefault(key, {})
    doc["display_path"] = display_path
    doc["untitled"] = untitled
    saved_at = time.time() if timestamp is None else timestamp
    entries = [e for e in doc.get("entries", []) if e.get("file") != str(slot)]
    entries.append({
        "file": str(slot),
        "saved_at": saved_at,
        "bytes": slot.stat().st_size if slot.is_file() else 0,
    })
    doc["entries"] = entries
    _save_index(index)


def rotate(key: str, retention: int) -> list[Path]:
    """Delete all but the newest `retention` slots for `key`, returning what
    was deleted. Scoped to one key on purpose: two documents autosaving in
    the same central directory must never prune each other."""
    index = _load_index()
    doc = index.get(key)
    if not doc:
        return []
    entries = sorted(doc.get("entries", []), key=lambda e: e.get("saved_at", 0.0), reverse=True)
    keep, drop = entries[:retention], entries[retention:]
    deleted = []
    for entry in drop:
        path = Path(entry["file"])
        try:
            path.unlink()
            deleted.append(path)
        except OSError:
            # A slot deleted behind our back (or on a read-only medium) is
            # not worth failing an autosave over; the index row goes either
            # way, so the next rotate() won't retry it forever.
            pass
    doc["entries"] = keep
    index[key] = doc
    _save_index(index)
    return deleted


def entries() -> list[AutosaveEntry]:
    """Every live slot, newest first, for the Recover dialog. Index rows
    whose file has vanished are dropped rather than listed as openable."""
    out = []
    for key, doc in _load_index().items():
        for entry in doc.get("entries", []):
            path = Path(entry.get("file", ""))
            if not path.is_file():
                continue
            out.append(AutosaveEntry(
                key=key,
                path=path,
                display_path=doc.get("display_path", ""),
                untitled=bool(doc.get("untitled", False)),
                saved_at=float(entry.get("saved_at", 0.0)),
                size=int(entry.get("bytes", 0)),
            ))
    out.sort(key=lambda e: e.saved_at, reverse=True)
    return out


def discard(key: str) -> None:
    """Drop one document's slots and index rows -- what a successful real
    save calls. Once the user's own file holds the state, its autosaves are
    obsolete, and leaving them would let Recover offer something OLDER than
    what is on disk."""
    index = _load_index()
    doc = index.pop(key, None)
    if doc is None:
        return
    for entry in doc.get("entries", []):
        with contextlib.suppress(OSError):
            Path(entry["file"]).unlink()
    _save_index(index)


def prune_stale(max_age_days: int = STALE_AFTER_DAYS) -> list[str]:
    """Drop every key whose newest slot is older than the cutoff, returning
    the keys dropped. Runs once at startup; this is what bounds the untitled
    keys doc_key() deliberately leaves unbounded by count."""
    # Epoch arithmetic directly: `saved_at` is time.time(), and going via a
    # naive local datetime would wobble by an hour across a DST boundary.
    cutoff = time.time() - max_age_days * 86400
    dropped = []
    for key, doc in list(_load_index().items()):
        newest = max((float(e.get("saved_at", 0.0)) for e in doc.get("entries", [])), default=0.0)
        if newest < cutoff:
            discard(key)
            dropped.append(key)
    return dropped
