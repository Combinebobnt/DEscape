"""The game's scenario folders: where Open and Save As start, and the list
File > Browse Game Scenarios… shows. Qt-free.

The game lists scenarios from <profile>/<id>/resources/_common/scenario/ and
each mod's resources/_common/scenario/, which content_roots finds. This
module adds only what is scenario-specific, so content_roots stays
content-agnostic.

Strictly read-only, like content_roots: a mod's folder under mods/subscribed/
is re-synced by the game's mod manager, which the hard rule forbids writing
to. tests/test_scenario_library.py scans this module's source for write calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from descape import content_roots, debug_log
from descape.content_roots import KIND_INSTALL, KIND_MOD_SUBSCRIBED, KIND_PROFILE, ContentRoot

SCENARIO_DIR = "scenario"
SUFFIXES = (".aoe2scenario", ".scn", ".scx")  # pre-DE files open read-only (TASK-429)

_UNSET = object()


def _profile_scenario_dir(id_dir: Path) -> Path | None:
    try:
        return content_roots.child_dir(id_dir, *content_roots.COMMON_SUBPATH.parts, SCENARIO_DIR)
    except OSError:  # Python 3.11's is_dir() raises on an unsearchable folder
        return None


def default_folder(profile=_UNSET) -> Path | None:
    """<profile>/<id>/resources/_common/scenario for the active profile id
    (content_roots.active_profile_id()), else the id whose scenario folder
    was modified last, or None. Never created. `profile` defaults to
    content_roots.profile_path(); None means no profile."""
    if profile is _UNSET:
        profile = content_roots.profile_path()
    if profile is None:
        return None
    profile = Path(profile)
    active = content_roots.active_profile_id(profile)
    if active is not None:
        found = _profile_scenario_dir(profile / active)
        if found is not None:
            return found
    newest: tuple[float, Path] | None = None
    try:
        id_dirs = sorted(p for p in profile.iterdir() if p.name.isdigit() and p.is_dir())
    except OSError:
        return None
    for id_dir in id_dirs:
        folder = _profile_scenario_dir(id_dir)
        if folder is None:
            continue
        try:
            mtime = folder.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, folder)
    return None if newest is None else newest[1]


@dataclass(frozen=True)
class ScenarioEntry:
    path: Path
    root: ContentRoot
    size: int
    mtime: float

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def in_subscribed(self) -> bool:
        return self.root.kind == KIND_MOD_SUBSCRIBED

    @property
    def source(self) -> str:
        """"Profile <id>", or "<title> (subscribed mod)" / "(local mod)", with
        ", disabled in game" inside the brackets for a mod the game disables."""
        if self.root.kind == KIND_PROFILE:
            return f"Profile {self.root.profile_id}"
        if self.root.kind == KIND_INSTALL:
            return "Install"
        kind = "subscribed mod" if self.in_subscribed else "local mod"
        disabled = "" if self.root.enabled else ", disabled in game"
        return f"{self.root.title or self.root.mod_id} ({kind}{disabled})"


def discover(roots=None) -> list[ScenarioEntry]:
    """Every scenario file in each root's scenario folder, disabled mods
    included, in root order then name order. A file whose stat() fails is
    skipped and logged."""
    if roots is None:
        roots = content_roots.content_roots(include_disabled=True)
    entries: list[ScenarioEntry] = []
    for root, path in content_roots.find_resources(SCENARIO_DIR, SUFFIXES, roots):
        try:
            found = path.stat()
        except OSError as exc:
            debug_log.log(f"scenario library: cannot stat {path} ({exc!r})", level=debug_log.WARNING)
            continue
        entries.append(ScenarioEntry(path, root, found.st_size, found.st_mtime))
    return entries
