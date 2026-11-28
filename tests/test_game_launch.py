"""Tools > Launch in AoE2:DE's Qt-free half (GH #4, descape/game_launch.py):
the launch copy's target, the ownership ledger, the launch command table and
the running-game check. No test spawns anything."""

from __future__ import annotations

from pathlib import Path

import pytest

from descape import asset_source, content_roots, game_launch, qt_platform

_PROTON_PROFILE = Path("steamapps/compatdata/813780/pfx/drive_c/users/steamuser/Games/Age of Empires 2 DE")


@pytest.fixture
def config(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "config" / "config.yaml"
    path.parent.mkdir()
    monkeypatch.setattr(asset_source, "CONFIG_PATH", path)
    monkeypatch.delenv(content_roots.PROFILE_ENV_VAR, raising=False)
    return path


def _profile(root: Path, ids=("76561190000000001",)) -> Path:
    profile = root / _PROTON_PROFILE
    for pid in ids:
        (profile / pid / "resources" / "_common" / "scenario").mkdir(parents=True)
    return profile


def test_target_is_the_fixed_name_in_the_profile_scenario_folder(tmp_path, config) -> None:
    profile = _profile(tmp_path)
    path, refusal = game_launch.launch_target(profile)
    assert refusal is None
    assert path == next(profile.iterdir()) / "resources" / "_common" / "scenario" / game_launch.LAUNCH_NAME


def test_target_refusals_say_what_is_missing(tmp_path, config) -> None:
    assert game_launch.launch_target(None) == (None, game_launch.NO_PROFILE_REFUSAL)
    bare = tmp_path / "profile"
    (bare / "123").mkdir(parents=True)
    assert game_launch.launch_target(bare) == (None, game_launch.NO_SCENARIO_FOLDER_REFUSAL)
    assert game_launch.NO_PROFILE_REFUSAL != game_launch.NO_SCENARIO_FOLDER_REFUSAL


def _writer(content: bytes, calls: list):
    def write(path, backup):
        calls.append((path, backup))
        Path(path).write_bytes(content)
        return "result"
    return write


def test_ledger_tracks_what_descape_wrote(tmp_path, config) -> None:
    a, b = tmp_path / "a" / game_launch.LAUNCH_NAME, tmp_path / "b" / game_launch.LAUNCH_NAME
    a.parent.mkdir(), b.parent.mkdir()
    assert not game_launch.ledger_matches(a), "a missing file is never owned"
    calls: list = []
    assert game_launch.write_launch_copy(a, _writer(b"one", calls)) == "result"
    assert calls == [(str(a), True)], "the launch copy keeps the profile's backups"
    assert game_launch.ledger_matches(a)
    game_launch.write_launch_copy(a, _writer(b"two", calls))
    assert game_launch.ledger_matches(a), "a repeat write moves the entry along"
    assert not game_launch.ledger_matches(b), "entries are per path"
    b.write_bytes(b"two")
    assert not game_launch.ledger_matches(b), "same bytes at a foreign path are still foreign"
    game_launch.write_launch_copy(b, _writer(b"three", calls))
    assert game_launch.ledger_matches(a) and game_launch.ledger_matches(b), "two targets kept apart"
    a.write_bytes(b"saved in-game")
    assert not game_launch.ledger_matches(a), "an edit outside DEscape breaks ownership"


def test_a_failed_ledger_write_does_not_fail_the_launch(tmp_path, config, monkeypatch) -> None:
    def boom(_path):
        raise OSError("read-only")
    monkeypatch.setattr(game_launch, "record_owned", boom)
    target = tmp_path / game_launch.LAUNCH_NAME
    assert game_launch.write_launch_copy(target, _writer(b"x", [])) == "result"
    assert target.read_bytes() == b"x"


def test_an_unreadable_ledger_reads_as_empty(tmp_path, config) -> None:
    target = tmp_path / game_launch.LAUNCH_NAME
    target.write_bytes(b"x")
    game_launch.ledger_path().parent.mkdir(parents=True)
    game_launch.ledger_path().write_text("{not json", encoding="utf-8")
    assert not game_launch.ledger_matches(target)
    game_launch.record_owned(target)
    assert game_launch.ledger_matches(target), "a bad ledger is replaced, not fatal"


def test_a_foreign_copy_is_kept_under_a_name_the_load_list_skips(tmp_path) -> None:
    target = tmp_path / game_launch.LAUNCH_NAME
    target.write_bytes(b"saved in-game")
    kept = game_launch.keep_foreign(target)
    assert kept.parent == tmp_path and kept.name.endswith(".bak") and not kept.name.endswith(".aoe2scenario")
    assert kept.read_bytes() == b"saved in-game" and target.read_bytes() == b"saved in-game"
    assert game_launch.keep_foreign_name(target, 0) != game_launch.keep_foreign_name(target, 86400)


def test_launch_args_default_and_override(config) -> None:
    assert game_launch.launch_args() == list(game_launch.DEFAULT_LAUNCH_ARGS)
    config.write_text("aoe2de_launch_args: 'EDITOR \"DEscape launch\"'\n", encoding="utf-8")
    assert game_launch.launch_args() == ["EDITOR", "DEscape launch"]
    config.write_text("aoe2de_launch_args: 'EDITOR \"unclosed'\n", encoding="utf-8")
    assert game_launch.launch_args() == list(game_launch.DEFAULT_LAUNCH_ARGS)


def _which(found: dict[str, str]):
    return found.get


def test_launch_command_table_in_order(tmp_path) -> None:
    app = game_launch.AOE2DE_APP_ID
    flatpak_profile = Path.home() / ".var" / "app" / "com.valvesoftware.Steam" / "data" / _PROTON_PROFILE
    both = _which({"flatpak": "/usr/bin/flatpak", "steam": "/usr/bin/steam", "xdg-open": "/usr/bin/xdg-open"})
    assert game_launch.launch_command(["EDITOR"], platform="linux", which=both, profile=flatpak_profile) == [
        "/usr/bin/flatpak", "run", "com.valvesoftware.Steam", "-applaunch", app, "EDITOR",
    ], "the Flatpak client owns a Flatpak profile even with steam on PATH"
    native = tmp_path / _PROTON_PROFILE
    assert game_launch.launch_command(["EDITOR"], platform="linux", which=both, profile=native) == [
        "/usr/bin/steam", "-applaunch", app, "EDITOR",
    ]
    no_steam = _which({"xdg-open": "/usr/bin/xdg-open"})
    assert game_launch.launch_command(["EDITOR", "My Map"], platform="linux", which=no_steam, profile=native) == [
        "/usr/bin/xdg-open", f"steam://run/{app}//EDITOR%20%27My%20Map%27/",
    ]
    exe = r"C:\Program Files (x86)\Steam\steam.exe"
    assert game_launch.launch_command(["EDITOR"], platform="win32", winreg_lookup=lambda: exe) == [
        exe, "-applaunch", app, "EDITOR",
    ]
    assert game_launch.launch_command(["EDITOR"], platform="win32", winreg_lookup=lambda: None) == (
        f"steam://run/{app}//EDITOR/"
    )
    assert game_launch.launch_command(["/EDITOR"], platform="win32", winreg_lookup=lambda: None) == (
        f"steam://run/{app}//%2FEDITOR/"
    ), "a slash is encoded, not read as a URL separator"
    with pytest.raises(ValueError, match="macOS"):
        game_launch.launch_command(["EDITOR"], platform="darwin")


def test_platform_refusal() -> None:
    assert game_launch.refusal_for_platform("linux") is None
    assert game_launch.refusal_for_platform("win32") is None
    assert game_launch.refusal_for_platform("darwin") == game_launch.MACOS_REFUSAL
    assert game_launch.refusal_for_platform("freebsd14")


def test_game_running_scans_proc(tmp_path) -> None:
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "12").mkdir()
    (proc / "12" / "cmdline").write_bytes(b"/usr/bin/bash\0")
    assert game_launch.game_running(proc, platform="linux") is False
    (proc / "34").mkdir()
    (proc / "34" / "cmdline").write_bytes(b"Z:\\steam\\common\\AoE2DE\\AoE2DE_s.exe\0EDITOR\0")
    assert game_launch.game_running(proc, platform="linux") is True
    assert game_launch.game_running(proc, platform="win32") is None


def test_spawn_detaches_with_the_restored_library_path(monkeypatch) -> None:
    seen = {}
    def popen(argv, **kw):
        seen.update(argv=argv, **kw)

    monkeypatch.setattr(game_launch.subprocess, "Popen", popen)
    env = {"LD_LIBRARY_PATH": "/bundle", "LD_LIBRARY_PATH_ORIG": "/usr/lib", "HOME": "/h"}
    game_launch.spawn(["steam", "-applaunch"], env, frozen=True)
    assert seen["argv"] == ["steam", "-applaunch"] and seen["start_new_session"]
    assert seen["env"] == {"LD_LIBRARY_PATH": "/usr/lib", "HOME": "/h"}


def test_system_tool_env() -> None:
    env = {"LD_LIBRARY_PATH": "/bundle", "X": "1"}
    assert qt_platform.system_tool_env(env, frozen=False) == env
    assert qt_platform.system_tool_env(env, frozen=True) == {"X": "1"}
    assert qt_platform.system_tool_env({**env, "LD_LIBRARY_PATH_ORIG": "/o"}, frozen=True) == {
        "LD_LIBRARY_PATH": "/o", "X": "1",
    }
