"""Qt-free reading and writing of DEscape's own YAML files: theme files
(theme_file) and settings files (settings_file).

Reading refuses a whole file that can't be one of ours (oversize, not YAML,
aliases, deep nesting, no top-level mapping, the wrong format id, a bad or
newer version), so a caller only ever sees a mapping. Writing never truncates
an existing file in place, and an export refuses a mods/subscribed/ path.
"""

from __future__ import annotations

import os
import reprlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from descape.scenario_io import is_workshop_subscribed


class _AliasRefused(Exception):
    pass


class NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases, so a tiny alias bomb can't expand into
    a huge document (or a huge problem message)."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise _AliasRefused()
        return super().compose_node(parent, index)


class NoAliasDumper(yaml.SafeDumper):
    """Export never emits anchors, so import's alias refusal can't reject our own file."""

    def ignore_aliases(self, data) -> bool:
        return True


# Caps every echoed value in a problem or note, whatever the file holds.
_SHORT = reprlib.Repr()
_SHORT.maxstring = 60
_SHORT.maxother = 60
_SHORT.maxlevel = 2
_SHORT_MAX = 80


def short(value) -> str:
    # reprlib never builds the full repr; the slice bounds its per-container fan-out.
    text = _SHORT.repr(value)
    return text if len(text) <= _SHORT_MAX else text[: _SHORT_MAX - 3] + "..."


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class YamlFormat:
    """One file format's whole-file checks. `noun` fills "not a DEscape <noun>
    file"; `refused` is raised for every refusal except a newer version, which
    raises `newer_refused` when set. `other_formats` maps another DEscape format
    id to a sentence naming what reads it, added to a wrong-format refusal."""

    format_id: str
    max_version: int
    max_bytes: int
    noun: str
    refused: type[Exception]
    newer_refused: type[Exception] | None = None
    other_formats: Mapping[str, str] = field(default_factory=dict)


def read_root(path: Path, fmt: YamlFormat) -> dict:
    """The parsed mapping, or fmt.refused with the one reason why not. A file
    that can't be read is refused from its OSError, so `__cause__` tells a
    missing file apart."""
    noun = fmt.noun
    try:
        size = path.stat().st_size
    except OSError as e:
        raise fmt.refused(f"cannot read the file: {e}") from e
    if size > fmt.max_bytes:
        raise fmt.refused(f"the file is {size} bytes, over the {fmt.max_bytes}-byte limit for a {noun} file")
    try:
        root = yaml.load(path.read_text(encoding="utf-8"), Loader=NoAliasLoader)  # noqa: S506 (a SafeLoader)
    except (OSError, UnicodeDecodeError) as e:
        raise fmt.refused(f"cannot read the file: {e}") from e
    except yaml.YAMLError as e:
        raise fmt.refused(f"not valid YAML: {e}") from e
    except _AliasRefused as e:
        raise fmt.refused(f"not a DEscape {noun} file (it uses YAML anchors/aliases, which a {noun} file never has)") from e
    except RecursionError as e:
        raise fmt.refused(f"not a DEscape {noun} file (nested too deeply)") from e
    except ValueError as e:  # e.g. an integer over Python's 4300-digit conversion limit
        raise fmt.refused(f"not valid YAML: {short(str(e))}") from e
    if not isinstance(root, dict):
        raise fmt.refused(f"not a DEscape {noun} file (the top level is not a mapping)")
    found = root.get("format")
    if found != fmt.format_id:
        hint = fmt.other_formats.get(found) if isinstance(found, str) else None
        raise fmt.refused(
            f"not a DEscape {noun} file (format is {short(found)}, expected {fmt.format_id!r})"
            + (f". {hint}" if hint else "")
        )
    version = root.get("version")
    if not _is_int(version) or version < 1:
        raise fmt.refused(f"unreadable {noun} file version {short(version)}")
    if version > fmt.max_version:
        raise (fmt.newer_refused or fmt.refused)(
            f"{noun} file version {short(version)} is newer than this DEscape reads ({fmt.max_version})"
        )
    return root


def dump(doc: dict) -> str:
    return yaml.dump(doc, Dumper=NoAliasDumper, default_flow_style=False, sort_keys=False)


def atomic_write(path: Path, data: str | bytes, *, mode: int | None = None) -> None:
    """Writes through a temp file in path's directory and os.replace(), so an
    existing file is never truncated in place. Text is written as UTF-8. The
    file gets `mode`, or 0o644 less the umask when None."""
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        if isinstance(data, bytes):
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data)
        if mode is None:
            # mkstemp makes the file 0o600; these files are meant to be shared.
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o644 & ~umask
        tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_export(path: Path | str, doc: dict, *, what: str, blocked: type[Exception]) -> None:
    """Refuses a mods/subscribed/ path, as written or resolved, or an
    unresolvable one, raising `blocked` and writing nothing. Otherwise dumps
    `doc` without anchors through atomic_write(). `what` names the file in the
    refusal ("a theme")."""
    path = Path(path)
    try:
        is_blocked = is_workshop_subscribed(path)
    except (OSError, RuntimeError) as e:  # a symlink loop raises on every Python version
        raise blocked(f"refusing to write {what} to a path that cannot be resolved ({e}): {path}") from e
    if is_blocked:
        raise blocked(f"refusing to write {what} into a subscribed-mods (mods/subscribed/) folder: {path}")
    atomic_write(path, dump(doc))
