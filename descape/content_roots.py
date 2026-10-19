"""Read-only discovery of the folders AoE2:DE loads content from: the install,
each per-user profile, and each profile's mods. Qt-free and content-agnostic;
a consumer (AI scripts, scenarios, xs, mod graphics) asks find_resources() for
one resources/_common subfolder across every root.

Profile and mods are optional by design. A missing, unreadable or
undetectable profile or mods folder yields no roots of that kind, logged to
debug_log only, never a dialog, so every consumer still works install-only.

Strictly read-only: this is a discovery module, and the mods/subscribed/
folders it lists are re-synced by the game's mod manager, which the hard
rule forbids writing to.
tests/test_content_roots.py scans this module's source for write calls.

Profile location, in priority order (the same shape as
asset_source.get_install_path()):
  1. the AOE2DE_PROFILE_PATH environment variable
  2. an "aoe2de_profile" key in config.yaml (the profile folder row in
     Settings > Mods writes it, through asset_source.save_profile_path_to_config())
  3. Windows: %USERPROFILE%/Games/Age of Empires 2 DE
  4. elsewhere: derived from the install path, via its nearest `steamapps`
     ancestor and the Proton prefix of app 813780. Never guesses ~/.steam,
     since a Steam library can live anywhere.
An explicit path (1 or 2) that is not a directory means no profile, not a
fall-through to autodetection.

The profile folder holds one numeric folder per account ("0" for offline,
a SteamID otherwise), each with resources/_common/ and mods/{subscribed,
local}/. A mods/mod-status.json, when present, carries each mod's Enabled
flag and Priority keyed by a "subscribed//<folder>" style Path; a mod it lists
as disabled is skipped, and the Priority rides on the root for consumers that
resolve overrides (mod_graphics). A mod folder it does not list at all is kept (the game adds new
subscriptions on its next launch), and a missing or malformed status file
means every mod folder is kept.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

from descape import asset_source, debug_log

PROFILE_ENV_VAR = "AOE2DE_PROFILE_PATH"
PROFILE_CONFIG_KEY = "aoe2de_profile"
AOE2DE_STEAM_APP_ID = "813780"
PROFILE_FOLDER_NAME = "Age of Empires 2 DE"
_PROTON_PROFILE_SUBPATH = (
    Path("compatdata") / AOE2DE_STEAM_APP_ID / "pfx" / "drive_c" / "users" / "steamuser" / "Games"
    / PROFILE_FOLDER_NAME
)
COMMON_SUBPATH = Path("resources") / "_common"
MOD_STATUS_FILE = "mod-status.json"

KIND_INSTALL = "install"
KIND_PROFILE = "profile"
KIND_MOD_SUBSCRIBED = "mod_subscribed"
KIND_MOD_LOCAL = "mod_local"
_MOD_KINDS = ((KIND_MOD_SUBSCRIBED, "subscribed"), (KIND_MOD_LOCAL, "local"))


@dataclass(frozen=True)
class ContentRoot:
    kind: str  # KIND_INSTALL / KIND_PROFILE / KIND_MOD_SUBSCRIBED / KIND_MOD_LOCAL
    label: str  # human-readable, for a source column when names collide
    path: Path  # the folder that contains resources/_common
    profile_id: str | None = None  # the numeric profile folder name, None for install
    mod_id: str | None = None  # the mod's folder name, None unless a mod
    priority: int | None = None  # mod-status.json's Priority (1 is the in-game list's top), None if unlisted
    status_path: str | None = None  # mod-status.json's Path form, "subscribed//<folder>", for any mod

    @property
    def common(self) -> Path:
        """resources/_common under this root, matched case-insensitively (a
        mod may ship Resources/); the lowercase path when neither exists."""
        return child_dir(self.path, *COMMON_SUBPATH.parts) or self.path / COMMON_SUBPATH


def child_dir(folder: Path, *parts: str) -> Path | None:
    """folder/<parts...> where each part matches an existing directory name
    case-insensitively (an exact match first), or None if one is missing."""
    current = Path(folder)
    for part in parts:
        exact = current / part
        if exact.is_dir():
            current = exact
            continue
        try:
            entries = list(current.iterdir())
        except OSError:
            return None
        wanted = part.casefold()
        match = next((p for p in sorted(entries) if p.name.casefold() == wanted and p.is_dir()), None)
        if match is None:
            return None
        current = match
    return current


def _config_value(key: str) -> str | None:
    # Read at call time, not bound at import: tests redirect asset_source.CONFIG_PATH.
    config_path = asset_source.CONFIG_PATH
    if not config_path.is_file():
        return None
    try:
        config = yaml.safe_load(config_path.read_text()) or {}
    except (yaml.YAMLError, OSError) as exc:
        debug_log.log(f"content roots: config unreadable ({exc!r})")
        return None
    raw = config.get(key) if isinstance(config, dict) else None
    return str(raw) if raw else None


def _explicit_dir(raw: str, source: str) -> Path | None:
    path = Path(raw)
    if path.is_dir():
        return path
    debug_log.log(f"content roots: {source} profile path {raw!r} is not a folder; no profile")
    return None


def derive_profile_from_install(install: Path) -> Path | None:
    """The Proton-prefix profile folder for the Steam library holding
    `install`, or None if `install` has no `steamapps` ancestor or the
    folder does not exist."""
    for ancestor in (install, *install.parents):
        if ancestor.name == "steamapps":
            candidate = ancestor / _PROTON_PROFILE_SUBPATH
            return candidate if candidate.is_dir() else None
    return None


def profile_path(install: Path | None = None, *, platform: str | None = None) -> Path | None:
    """The AoE2:DE profile folder (the parent of the numeric profile id
    folders), or None. See the module docstring for the precedence.
    `install` defaults to asset_source.get_install_path(); `platform` to
    sys.platform (both parameters exist for tests)."""
    env_raw = os.environ.get(PROFILE_ENV_VAR)
    if env_raw:
        return _explicit_dir(env_raw, PROFILE_ENV_VAR)
    config_raw = _config_value(PROFILE_CONFIG_KEY)
    if config_raw:
        return _explicit_dir(config_raw, PROFILE_CONFIG_KEY)

    platform = sys.platform if platform is None else platform
    if platform == "win32":
        home = os.environ.get("USERPROFILE")
        if not home:
            return None
        candidate = Path(home) / "Games" / PROFILE_FOLDER_NAME
        return candidate if candidate.is_dir() else None

    if install is None:
        install = asset_source.get_install_path()
    if install is None:
        return None
    return derive_profile_from_install(install)


SOURCE_ENV = "env"
SOURCE_CONFIG = "config"
SOURCE_AUTO = "auto"
SOURCE_NONE = "none"
SOURCE_ENV_INVALID = "env-invalid"
SOURCE_CONFIG_INVALID = "config-invalid"


def profile_source(install: Path | None = None, *, platform: str | None = None) -> tuple[str, str | None]:
    """(source, raw) for what profile_path() answers: SOURCE_ENV or SOURCE_CONFIG
    with the raw setting, their *_INVALID twins when it is not a folder (so
    profile_path() is None), or SOURCE_AUTO / SOURCE_NONE with raw None."""
    env_raw = os.environ.get(PROFILE_ENV_VAR)
    if env_raw:
        return (SOURCE_ENV if Path(env_raw).is_dir() else SOURCE_ENV_INVALID), env_raw
    config_raw = _config_value(PROFILE_CONFIG_KEY)
    if config_raw:
        return (SOURCE_CONFIG if Path(config_raw).is_dir() else SOURCE_CONFIG_INVALID), config_raw
    found = profile_path(install, platform=platform)
    return (SOURCE_AUTO if found is not None else SOURCE_NONE), None


def _sorted_dirs(folder: Path) -> list[Path]:
    """Sub-folders of `folder` by name; [] (logged) if it cannot be listed."""
    try:
        entries = list(folder.iterdir())
    except OSError as exc:
        debug_log.log(f"content roots: cannot list {folder} ({exc!r})")
        return []
    return sorted((p for p in entries if p.is_dir()), key=lambda p: p.name)


def _status_parts(raw) -> tuple[str, str] | None:
    parts = [part for part in re.split(r"[\\/]+", str(raw)) if part]
    return (parts[0], parts[-1]) if len(parts) >= 2 else None


def _mod_status(mods_dir: Path) -> tuple[frozenset[tuple[str, str]], dict[tuple[str, str], int]]:
    """One parse of mod-status.json: the (kind folder, mod folder) pairs it
    marks disabled, and each listed mod's integer Priority. A non-integer
    Priority leaves that mod unlisted for ordering; an unreadable file gives
    nothing of either."""
    status_path = mods_dir / MOD_STATUS_FILE
    if not status_path.is_file():
        return frozenset(), {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        mods = data["Mods"]
        disabled = set()
        priorities: dict[tuple[str, str], int] = {}
        for mod in mods:
            pair = None if mod.get("Path") is None else _status_parts(mod["Path"])
            if pair is None:
                continue
            if not mod.get("Enabled", True):
                disabled.add(pair)
            priority = mod.get("Priority")
            if isinstance(priority, int) and not isinstance(priority, bool):
                priorities[pair] = priority
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        debug_log.log(f"content roots: {status_path} unreadable, keeping every mod ({exc!r})")
        return frozenset(), {}
    return frozenset(disabled), priorities


def _disabled_mods(mods_dir: Path) -> frozenset[tuple[str, str]]:
    """(kind folder, mod folder) pairs mod-status.json marks disabled."""
    return _mod_status(mods_dir)[0]


def _profile_roots(profile: Path) -> list[ContentRoot]:
    roots: list[ContentRoot] = []
    for id_dir in _sorted_dirs(profile):
        if not id_dir.name.isdigit():
            continue
        profile_id = id_dir.name
        roots.append(ContentRoot(KIND_PROFILE, f"Profile {profile_id}", id_dir, profile_id=profile_id))
        mods_dir = id_dir / "mods"
        if not mods_dir.is_dir():
            continue
        disabled, priorities = _mod_status(mods_dir)
        for kind, folder in _MOD_KINDS:
            kind_dir = mods_dir / folder
            if not kind_dir.is_dir():
                continue
            for mod_dir in _sorted_dirs(kind_dir):
                if (folder, mod_dir.name) in disabled:
                    continue
                roots.append(ContentRoot(
                    kind, f"Mod ({folder}): {mod_dir.name}", mod_dir,
                    profile_id=profile_id, mod_id=mod_dir.name,
                    priority=priorities.get((folder, mod_dir.name)),
                    status_path=f"{folder}//{mod_dir.name}",
                ))
    return roots


def skipped_mods(profile: Path) -> list[tuple[str, str]]:
    """(kind folder, mod folder) for each mod on disk that _profile_roots()
    leaves out because mod-status.json disables it, in profile id order."""
    skipped: list[tuple[str, str]] = []
    for id_dir in _sorted_dirs(Path(profile)):
        mods_dir = id_dir / "mods"
        if not id_dir.name.isdigit() or not mods_dir.is_dir():
            continue
        disabled = _disabled_mods(mods_dir)
        skipped.extend(sorted(pair for pair in disabled if (mods_dir / pair[0] / pair[1]).is_dir()))
    return skipped


def active_profile_id(profile: Path) -> str | None:
    """The numeric profile id whose mods/mod-status.json was written last (the
    account the game last ran), or None when no id folder has one."""
    newest: tuple[float, str] | None = None
    for id_dir in _sorted_dirs(Path(profile)):
        if not id_dir.name.isdigit():
            continue
        try:
            mtime = (id_dir / "mods" / MOD_STATUS_FILE).stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, id_dir.name)
    return None if newest is None else newest[1]


def has_profile_ids(folder: Path) -> bool:
    """Whether `folder` holds a numeric per-account folder, the shape a profile folder has."""
    return any(p.name.isdigit() for p in _sorted_dirs(Path(folder)))


_UNSET = object()


def content_roots(install=_UNSET, profile=_UNSET) -> list[ContentRoot]:
    """Every content root, in order: the install, then per numeric profile id
    (by name) the profile itself, its enabled subscribed mods, then its
    enabled local mods. `install`/`profile` default to the live lookups;
    pass None to leave that kind out."""
    if install is _UNSET:
        install = asset_source.get_install_path()
    if profile is _UNSET:
        profile = profile_path(install)
    roots: list[ContentRoot] = []
    if install is not None:
        roots.append(ContentRoot(KIND_INSTALL, "Install", Path(install)))
    if profile is not None:
        roots.extend(_profile_roots(Path(profile)))
    return roots


def find_resources(
    rel_dir: str | Path, suffixes: Iterable[str], roots: Iterable[ContentRoot] | None = None
) -> list[tuple[ContentRoot, Path]]:
    """(root, file) for every file directly inside resources/_common/<rel_dir>
    of each root whose suffix case-folds to one of `suffixes` (".ai"
    matches "x.Ai"), the folders matched case-insensitively too. Root order,
    then file name order. Not recursive."""
    wanted = frozenset(s.casefold() for s in suffixes)
    if roots is None:
        roots = content_roots()
    found: list[tuple[ContentRoot, Path]] = []
    for root in roots:
        folder = child_dir(root.common, *Path(rel_dir).parts)
        if folder is None:
            continue
        try:
            entries = list(folder.iterdir())
        except OSError as exc:
            debug_log.log(f"content roots: cannot list {folder} ({exc!r})")
            continue
        found.extend(
            (root, path)
            for path in sorted(entries, key=lambda p: p.name)
            if path.suffix.casefold() in wanted and path.is_file()
        )
    return found
