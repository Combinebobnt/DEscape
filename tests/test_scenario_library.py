"""descape/scenario_library.py: the default scenario folder and discovery,
against fake profile trees under tmp_path."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path

from descape import content_roots, scenario_library
from descape.scenario_library import default_folder, discover


def _touch(path: Path, data: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _scenario_dir(profile: Path, profile_id: str, *, caps: bool = False) -> Path:
    parts = ("Resources", "_common", "Scenario") if caps else ("resources", "_common", "scenario")
    folder = profile / profile_id / Path(*parts)
    folder.mkdir(parents=True)
    return folder


def _set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def test_default_folder_prefers_the_active_id(tmp_path):
    profile = tmp_path / "profile"
    _scenario_dir(profile, "0")
    active = _scenario_dir(profile, "7656")
    _touch(profile / "7656" / "mods" / "mod-status.json", b"{}")
    _set_mtime(active, 1_000)  # older than 0's, but 7656 is the id the game last ran
    assert default_folder(profile) == active


def test_default_folder_skips_an_active_id_with_no_scenario_folder(tmp_path):
    profile = tmp_path / "profile"
    offline = _scenario_dir(profile, "0")
    _touch(profile / "7656" / "mods" / "mod-status.json", b"{}")
    assert default_folder(profile) == offline


def test_default_folder_falls_back_to_the_newest_scenario_folder(tmp_path):
    profile = tmp_path / "profile"
    offline = _scenario_dir(profile, "0")
    steam = _scenario_dir(profile, "7656")
    _set_mtime(offline, 1_000)
    _set_mtime(steam, 2_000)
    assert default_folder(profile) == steam
    _set_mtime(offline, 3_000)
    assert default_folder(profile) == offline


def test_default_folder_matches_case_insensitively(tmp_path):
    profile = tmp_path / "profile"
    folder = _scenario_dir(profile, "7656", caps=True)
    assert default_folder(profile).samefile(folder)


def test_default_folder_none_without_profile_or_folder(tmp_path):
    assert default_folder(None) is None
    profile = tmp_path / "profile"
    (profile / "0" / "resources" / "_common").mkdir(parents=True)
    assert default_folder(profile) is None
    assert not (profile / "0" / "resources" / "_common" / "scenario").exists()  # never created
    assert default_folder(tmp_path / "missing") is None


def _library_tree(tmp_path: Path) -> tuple[Path, Path]:
    install = tmp_path / "install"
    (install / "resources" / "_common").mkdir(parents=True)  # no scenario folder
    profile = tmp_path / "profile"
    own = _scenario_dir(profile, "0")
    _touch(own / "Offline.aoe2scenario", b"x" * 10)
    _touch(own / "notes.txt")
    _touch(own / "Old.scx", b"x" * 3)
    _touch(own / "UserPatch.scx2", b"x" * 4)
    caps = _scenario_dir(profile, "7656", caps=True)
    _touch(caps / "Steam.aoe2scenario", b"x" * 20)
    mods = profile / "7656" / "mods"
    _touch(mods / "subscribed" / "12_Sub" / "resources" / "_common" / "scenario" / "Sub.aoe2scenario")
    _touch(mods / "local" / "Mine" / "resources" / "_common" / "scenario" / "Local.aoe2scenario")
    _touch(mods / "local" / "Off" / "resources" / "_common" / "scenario" / "Off.aoe2scenario")
    status = {"Mods": [
        {"Enabled": True, "Path": "subscribed//12_Sub", "Title": "Sub Mod"},
        {"Enabled": False, "Path": "local//Off"},
    ]}
    _touch(mods / "mod-status.json", json.dumps(status).encode())
    return install, profile


def test_discover_lists_every_root_with_its_source(tmp_path):
    install, profile = _library_tree(tmp_path)
    entries = discover(content_roots.content_roots(install, profile, include_disabled=True))
    got = {e.path.name: (e.source, e.in_subscribed, e.size) for e in entries}
    assert got == {
        "Offline.aoe2scenario": ("Profile 0", False, 10),
        "Old.scx": ("Profile 0", False, 3),
        "UserPatch.scx2": ("Profile 0", False, 4),
        "Steam.aoe2scenario": ("Profile 7656", False, 20),
        "Sub.aoe2scenario": ("Sub Mod (subscribed mod)", True, 0),
        "Local.aoe2scenario": ("Mine (local mod)", False, 0),
        "Off.aoe2scenario": ("Off (local mod, disabled in game)", False, 0),
    }
    offline = next(e for e in entries if e.path.name == "Offline.aoe2scenario")
    assert offline.mtime == offline.path.stat().st_mtime
    assert offline.name == "Offline"


def test_discover_skips_a_file_whose_stat_fails(tmp_path, monkeypatch):
    install, profile = _library_tree(tmp_path)
    roots = content_roots.content_roots(install, profile, include_disabled=True)
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "Steam.aoe2scenario" and not kwargs and not args:
            raise PermissionError("denied")
        return real_stat(self, *args, **kwargs)

    found = [(r, p) for r, p in content_roots.find_resources("scenario", scenario_library.SUFFIXES, roots)]
    monkeypatch.setattr(content_roots, "find_resources", lambda *a, **k: found)
    monkeypatch.setattr(Path, "stat", flaky_stat)
    names = [e.path.name for e in discover(roots)]
    assert "Steam.aoe2scenario" not in names
    assert "Offline.aoe2scenario" in names


_WRITE_CALLS = re.compile(
    r"write_text|write_bytes|\.mkdir\(|\.touch\(|os\.remove|\.rename\(|\.replace\(|\.unlink\(|shutil|"
    r"""open\([^)]*['"][wax+]"""
)


def test_module_is_read_only():
    assert _WRITE_CALLS.findall(inspect.getsource(scenario_library)) == []
