"""descape/content_roots.py: install/profile/mod discovery against fake trees
under tmp_path. _isolated_settings (conftest.py, autouse) redirects
CONFIG_PATH, so a test sees no config unless it writes one."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest
import yaml

from descape import asset_source, content_roots
from descape.content_roots import (
    KIND_INSTALL,
    KIND_MOD_LOCAL,
    KIND_MOD_SUBSCRIBED,
    KIND_PROFILE,
    find_resources,
    profile_path,
)

roots_of = content_roots.content_roots


@pytest.fixture(autouse=True)
def _no_profile_env(monkeypatch):
    monkeypatch.delenv(content_roots.PROFILE_ENV_VAR, raising=False)


def _touch(path: Path, data: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _install(tmp_path: Path) -> Path:
    install = tmp_path / "steamapps" / "common" / "AoE2DE"
    ai = install / "resources" / "_common" / "ai"
    _touch(ai / "E3-p2.ai")
    _touch(ai / "E3-p2.per", b"(defrule)")
    _touch(ai / "Promisory" / "x.ai")  # include folders are not scanned
    return install


def _profile(tmp_path: Path) -> Path:
    return tmp_path / "steamapps" / content_roots._PROTON_PROFILE_SUBPATH


def _mod(profile: Path, pid: str, folder: str, name: str, ai_name: str | None = None) -> Path:
    mod = profile / pid / "mods" / folder / name
    (mod / "resources" / "_common").mkdir(parents=True, exist_ok=True)
    if ai_name:
        _touch(mod / "resources" / "_common" / "ai" / ai_name)
    return mod


def _full_tree(tmp_path: Path) -> tuple[Path, Path]:
    install = _install(tmp_path)
    profile = _profile(tmp_path)
    for pid in ("0", "7656"):
        (profile / pid / "resources" / "_common" / "ai").mkdir(parents=True)
    for junk in ("logs", "metadata"):
        (profile / junk).mkdir()
    _touch(profile / "7656" / "resources" / "_common" / "ai" / "Mine.AI")
    _mod(profile, "7656", "subscribed", "12_Alpha AI", "Alpha.ai")
    _mod(profile, "7656", "subscribed", "34_Off AI", "Off.ai")
    _mod(profile, "7656", "subscribed", "56_Unlisted", "Unlisted.ai")
    _mod(profile, "7656", "local", "Local Thing", "Local.ai")
    _touch(profile / "7656" / "mods" / "subscribed" / "info.json", b"{}")
    _touch(profile / "7656" / "mods" / "local" / "Packed.zip")
    status = {"Mods": [
        {"Enabled": True, "Path": "subscribed//12_Alpha AI"},
        {"Enabled": False, "Path": "subscribed//34_Off AI"},
        {"Enabled": True, "Path": "local//Local Thing"},
    ], "Unsub": []}
    _touch(profile / "7656" / "mods" / "mod-status.json", json.dumps(status).encode())
    return install, profile


def test_full_tree_order_and_kinds(tmp_path):
    install, profile = _full_tree(tmp_path)
    roots = roots_of(install, profile)
    assert [(r.kind, r.profile_id, r.mod_id) for r in roots] == [
        (KIND_INSTALL, None, None),
        (KIND_PROFILE, "0", None),
        (KIND_PROFILE, "7656", None),
        (KIND_MOD_SUBSCRIBED, "7656", "12_Alpha AI"),
        (KIND_MOD_SUBSCRIBED, "7656", "56_Unlisted"),
        (KIND_MOD_LOCAL, "7656", "Local Thing"),
    ]
    assert roots[0].common == install / "resources" / "_common"


def test_find_resources_case_insensitive_and_not_recursive(tmp_path):
    install, profile = _full_tree(tmp_path)
    found = find_resources("ai", [".ai"], roots_of(install, profile))
    names = [(root.kind, path.name) for root, path in found]
    assert names == [
        (KIND_INSTALL, "E3-p2.ai"),
        (KIND_PROFILE, "Mine.AI"),
        (KIND_MOD_SUBSCRIBED, "Alpha.ai"),
        (KIND_MOD_SUBSCRIBED, "Unlisted.ai"),
        (KIND_MOD_LOCAL, "Local.ai"),
    ]


def test_install_only(tmp_path):
    install = _install(tmp_path)
    roots = roots_of(install, None)
    assert [r.kind for r in roots] == [KIND_INSTALL]
    assert [p.name for _r, p in find_resources("ai", [".AI"], roots)] == ["E3-p2.ai"]


def test_nothing_configured(tmp_path):
    assert roots_of(None, None) == []
    assert find_resources("ai", [".ai"], []) == []


def test_profile_without_mods(tmp_path):
    install = _install(tmp_path)
    profile = _profile(tmp_path)
    (profile / "0" / "resources" / "_common").mkdir(parents=True)
    roots = roots_of(install, profile)
    assert [(r.kind, r.profile_id) for r in roots] == [(KIND_INSTALL, None), (KIND_PROFILE, "0")]


def test_malformed_mod_status_keeps_every_mod(tmp_path):
    install, profile = _full_tree(tmp_path)
    _touch(profile / "7656" / "mods" / "mod-status.json", b"{not json")
    mods = [r.mod_id for r in roots_of(install, profile) if r.mod_id]
    assert mods == ["12_Alpha AI", "34_Off AI", "56_Unlisted", "Local Thing"]


def test_unreadable_profile_folder_yields_install_only(tmp_path, monkeypatch):
    # chmod is ignored under root CI, so the listing failure is injected instead.
    install, profile = _full_tree(tmp_path)
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self == profile:
            raise PermissionError(13, "Permission denied", str(self))
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert [r.kind for r in roots_of(install, profile)] == [KIND_INSTALL]


def test_profile_path_derived_from_install(tmp_path):
    install = _install(tmp_path)
    assert profile_path(install, platform="linux") is None
    _profile(tmp_path).mkdir(parents=True)
    assert profile_path(install, platform="linux") == _profile(tmp_path)


def test_profile_path_install_outside_steamapps(tmp_path):
    install = tmp_path / "AoE2DE"
    install.mkdir()
    assert profile_path(install, platform="linux") is None


def test_profile_path_env_beats_config(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    asset_source.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"aoe2de_profile": str(cfg_dir)}))
    assert profile_path(None, platform="linux") == cfg_dir
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, str(env_dir))
    assert profile_path(None, platform="linux") == env_dir


def test_explicit_missing_path_means_no_profile(tmp_path, monkeypatch):
    install = _install(tmp_path)
    _profile(tmp_path).mkdir(parents=True)
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, str(tmp_path / "missing"))
    assert profile_path(install, platform="linux") is None


def test_profile_path_windows(tmp_path, monkeypatch):
    home = tmp_path / "home"
    target = home / "Games" / "Age of Empires 2 DE"
    monkeypatch.setenv("USERPROFILE", str(home))
    assert profile_path(None, platform="win32") is None
    target.mkdir(parents=True)
    assert profile_path(None, platform="win32") == target


_WRITE_CALLS = re.compile(
    r"write_text|write_bytes|\.mkdir\(|\.touch\(|os\.remove|\.rename\(|\.replace\(|shutil|"
    r"""open\([^)]*['"][wax+]"""
)


def test_module_is_read_only():
    source = inspect.getsource(content_roots)
    assert _WRITE_CALLS.findall(source) == []
