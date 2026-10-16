"""descape/content_roots.py: install/profile/mod discovery against fake trees
under tmp_path. _isolated_settings (conftest.py, autouse) redirects
CONFIG_PATH and drops AOE2DE_PROFILE_PATH, so a test sees no config and no
profile override unless it sets one."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

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


def test_conftest_drops_the_profile_env_var() -> None:
    import os

    assert content_roots.PROFILE_ENV_VAR not in os.environ


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


def _write_config(**keys) -> None:
    asset_source.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump(keys))


def test_profile_source_auto_and_none(tmp_path):
    install = _install(tmp_path)
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_NONE, None)
    _profile(tmp_path).mkdir(parents=True)
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_AUTO, None)


def test_profile_source_config_then_env_beats_it(tmp_path, monkeypatch):
    install = _install(tmp_path)
    _profile(tmp_path).mkdir(parents=True)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    _write_config(aoe2de_profile=str(cfg_dir))
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_CONFIG, str(cfg_dir))
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, str(env_dir))
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_ENV, str(env_dir))


def test_profile_source_invalid_explicit_paths(tmp_path, monkeypatch):
    """An explicit non-folder never falls through to auto-detect, so it is reported as such."""
    install = _install(tmp_path)
    _profile(tmp_path).mkdir(parents=True)
    missing = str(tmp_path / "missing")
    _write_config(aoe2de_profile=missing)
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_CONFIG_INVALID, missing)
    assert profile_path(install, platform="linux") is None
    monkeypatch.setenv(content_roots.PROFILE_ENV_VAR, missing)
    assert content_roots.profile_source(install, platform="linux") == (content_roots.SOURCE_ENV_INVALID, missing)


def test_skipped_mods_lists_disabled_mods_that_exist(tmp_path):
    _install_dir, profile = _full_tree(tmp_path)
    status_path = profile / "7656" / "mods" / "mod-status.json"
    status = json.loads(status_path.read_text())
    status["Mods"].append({"Enabled": False, "Path": "subscribed//78_Gone"})  # not on disk
    status["Mods"].append({"Enabled": False, "Path": "local//Local Thing"})
    _touch(status_path, json.dumps(status).encode())
    assert content_roots.skipped_mods(profile) == [("local", "Local Thing"), ("subscribed", "34_Off AI")]
    kept = {r.mod_id for r in roots_of(_install_dir, profile) if r.mod_id}
    assert kept.isdisjoint({"34_Off AI", "Local Thing"})


def test_skipped_mods_none_without_a_status_file(tmp_path):
    _install_dir, profile = _full_tree(tmp_path)
    (profile / "7656" / "mods" / "mod-status.json").unlink()
    assert content_roots.skipped_mods(profile) == []
    assert content_roots.skipped_mods(tmp_path / "missing") == []


def test_has_profile_ids(tmp_path):
    _install_dir, profile = _full_tree(tmp_path)
    assert content_roots.has_profile_ids(profile)
    assert not content_roots.has_profile_ids(profile / "logs")
    assert not content_roots.has_profile_ids(tmp_path / "missing")


def _status(profile: Path, pid: str, mods: list[dict]) -> Path:
    return _touch(profile / pid / "mods" / "mod-status.json", json.dumps({"Mods": mods, "Unsub": []}).encode())


def test_priority_and_status_path(tmp_path):
    install, profile = _full_tree(tmp_path)
    _status(profile, "7656", [
        {"Enabled": True, "Path": "subscribed//12_Alpha AI", "Priority": 3},
        {"Enabled": False, "Path": "subscribed//34_Off AI", "Priority": 1},
        {"Enabled": True, "Path": "local//Local Thing", "Priority": "2"},
    ])
    mods = {r.mod_id: (r.priority, r.status_path) for r in roots_of(install, profile) if r.mod_id}
    assert mods == {
        "12_Alpha AI": (3, "subscribed//12_Alpha AI"),
        "56_Unlisted": (None, "subscribed//56_Unlisted"),  # not in the status file at all
        "Local Thing": (None, "local//Local Thing"),  # a non-int Priority counts as unlisted
    }


def test_active_profile_id_is_the_newest_status_file(tmp_path):
    import os

    _install_dir, profile = _full_tree(tmp_path)
    assert content_roots.active_profile_id(profile) == "7656"
    newer = _status(profile, "0", [])
    old = profile / "7656" / "mods" / "mod-status.json"
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))
    assert content_roots.active_profile_id(profile) == "0"
    assert content_roots.active_profile_id(tmp_path / "missing") is None


def test_capitalised_resources_folder_is_found(tmp_path):
    install = _install(tmp_path)
    profile = _profile(tmp_path)
    mod = profile / "7656" / "mods" / "subscribed" / "90_Caps"
    _touch(mod / "Resources" / "_Common" / "AI" / "Caps.ai")
    roots = roots_of(install, profile)
    caps = next(r for r in roots if r.mod_id == "90_Caps")
    assert caps.common == mod / "Resources" / "_Common"
    assert [(r.mod_id, p.name) for r, p in find_resources("ai", [".ai"], roots)][-1] == ("90_Caps", "Caps.ai")


_WRITE_CALLS = re.compile(
    r"write_text|write_bytes|\.mkdir\(|\.touch\(|os\.remove|\.rename\(|\.replace\(|shutil|"
    r"""open\([^)]*['"][wax+]"""
)


def test_module_is_read_only():
    source = inspect.getsource(content_roots)
    assert _WRITE_CALLS.findall(source) == []


# The mod art readers: `.replace(` is left out, since str.replace is legitimate there.
_MOD_ART_WRITE_CALLS = re.compile(
    r"write_text|write_bytes|\.mkdir\(|\.touch\(|os\.remove|\.rename\(|\.unlink\(|shutil|os\.replace|"
    r"""open\([^)]*['"][wax+]"""
)


def test_mod_art_modules_are_read_only():
    from descape import mod_graphics, smx_decoder

    for module in (mod_graphics, smx_decoder):
        assert _MOD_ART_WRITE_CALLS.findall(inspect.getsource(module)) == [], module.__name__
