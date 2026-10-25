"""GH #58: finding the player's data mods across profile ids, enabled or not."""

from __future__ import annotations

import json
from pathlib import Path

from descape import content_roots, data_mods


def _mod(profile: Path, pid: str, kind: str, folder: str, *, dat: bool = True, resources: str = "resources") -> Path:
    root = profile / pid / "mods" / kind / folder
    target = root / resources / "_common" / ("dat" if dat else "drs") / ("empires2_x2_p1.dat" if dat else "x.sld")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")
    return root


def _status(profile: Path, pid: str, mods: list[dict]) -> None:
    (profile / pid / "mods" / "mod-status.json").write_text(json.dumps({"Mods": mods, "Unsub": []}))


def test_discovery_lists_disabled_data_mods_once_and_ignores_the_rest(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    _mod(profile, "111", "subscribed", "93961_Exploding Villagers")
    _mod(profile, "111", "subscribed", "48_Chunky Trees", dat=False)
    _mod(profile, "111", "local", "Gensokyo", resources="Resources")
    _mod(profile, "222", "subscribed", "93961_Exploding Villagers")
    _status(profile, "111", [
        {"Enabled": False, "Path": "subscribed//93961_Exploding Villagers", "Priority": 3,
         "Title": "Exploding Villagers", "WorkshopID": 93961},
        {"Enabled": True, "Path": "subscribed//48_Chunky Trees", "Priority": 1, "Title": "Chunky", "WorkshopID": 48},
    ])
    install = tmp_path / "install"
    install.mkdir()
    roots = content_roots.content_roots(install=install, profile=profile, include_disabled=True)
    assert all(root.enabled for root in content_roots.content_roots(install=install, profile=profile))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    found = data_mods.discover(roots)
    assert [(m.status_path, m.title, m.workshop_id) for m in found] == [
        ("subscribed//93961_Exploding Villagers", "Exploding Villagers", "93961"),
        ("local//Gensokyo", "Gensokyo", None),
    ]
    assert found[0].dat.is_file() and found[0].root.parent.parent.parent.name == "111"


def test_the_active_profiles_copy_wins(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    _mod(profile, "111", "subscribed", "7_M")
    _mod(profile, "222", "subscribed", "7_M")
    _status(profile, "111", [])
    _status(profile, "222", [{"Enabled": False, "Path": "subscribed//7_M", "Title": "Newer"}])
    newer = profile / "222" / "mods" / "mod-status.json"
    older = profile / "111" / "mods" / "mod-status.json"
    import os

    os.utime(older, (1, 1))
    os.utime(newer, (2_000_000_000, 2_000_000_000))
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    found = data_mods.discover(content_roots.content_roots(install=None, profile=profile, include_disabled=True))
    assert [(m.title, m.root.parent.parent.parent.name) for m in found] == [("Newer", "222")]


def test_a_missing_status_file_names_mods_by_folder(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    _mod(profile, "0", "local", "My Mod")
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    found = data_mods.discover(content_roots.content_roots(install=None, profile=profile, include_disabled=True))
    assert [(m.status_path, m.title) for m in found] == [("local//My Mod", "My Mod")]


def test_install_and_profile_roots_are_never_data_mods(tmp_path, monkeypatch):
    install = tmp_path / "install"
    profile = tmp_path / "profile"
    for root in (install, profile / "111"):
        (root / "resources" / "_common" / "dat").mkdir(parents=True)
        (root / "resources" / "_common" / "dat" / "empires2_x2_p1.dat").write_bytes(b"x")
    monkeypatch.setenv("AOE2DE_PROFILE_PATH", str(profile))
    roots = content_roots.content_roots(install=install, profile=profile, include_disabled=True)
    assert {root.kind for root in roots} == {content_roots.KIND_INSTALL, content_roots.KIND_PROFILE}
    assert data_mods.discover(roots) == []
