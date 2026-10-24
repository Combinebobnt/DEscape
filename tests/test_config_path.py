"""Verifies the config.yaml relocation to the OS-standard per-user config
location (descape/asset_source.py's CONFIG_PATH/LEGACY_CONFIG_PATH,
write_config_file, migrate_legacy_config).

Checks:
  1. Missing-parent-directory coverage -- the one thing the suite's own
     autouse `_isolated_settings` fixture cannot supply, since its fake
     path is `tmp_path / "config.yaml"`, whose parent always exists. A
     full green default tier proves nothing about a fresh machine with no
     `~/.config/DEscape/` yet.
  2. The resolver -- real XDG_CONFIG_HOME branches on Linux, and the
     pinned platformdirs argument set (appauthor=False, roaming=True) via
     a spy, which is what a macOS/Windows assertion has to fall back to
     from this machine.
  3. Migration -- copied when legacy is present and the target is not;
     target left untouched when it already exists; a true no-op (no
     directory created) when neither exists; OSError from mkdir/copy2
     degrades to returning None rather than raising.

Every migration test patches LEGACY_CONFIG_PATH itself, on top of the
autouse fixture's CONFIG_PATH redirect: `_isolated_settings` does not
touch LEGACY_CONFIG_PATH, so an unpatched one still points at this
developer's real repo-root config.yaml.
"""

from __future__ import annotations

import sys

import pytest
import yaml

from descape import asset_source, settings


def test_missing_parent_directory_is_created_on_write(tmp_path, monkeypatch):
    deep_path = tmp_path / "no" / "such" / "dir" / "config.yaml"
    monkeypatch.setattr(asset_source, "CONFIG_PATH", deep_path)
    monkeypatch.setattr(settings, "CONFIG_PATH", deep_path)

    settings.set_theme("dark")
    assert deep_path.is_file()
    assert yaml.safe_load(deep_path.read_text())["theme"] == "dark"

    other_deep_path = tmp_path / "another" / "missing" / "tree" / "config.yaml"
    monkeypatch.setattr(asset_source, "CONFIG_PATH", other_deep_path)
    install = tmp_path  # any real directory works; save doesn't validate it
    asset_source.save_install_path_to_config(install)
    assert other_deep_path.is_file()
    assert yaml.safe_load(other_deep_path.read_text())["aoe2de_install"] == str(install)


@pytest.mark.skipif(
    sys.platform == "win32", reason="platformdirs ignores XDG_CONFIG_HOME on Windows (%APPDATA% via SHGetFolderPath)"
)
def test_resolver_uses_xdg_config_home_when_set(monkeypatch, tmp_path):
    fake_xdg = tmp_path / "fake-xdg-home"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(fake_xdg))
    path = asset_source._default_config_path()
    assert path == asset_source.Path(fake_xdg) / "DEscape" / "config.yaml"


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="platformdirs falls back to ~/.config only on Linux (macOS: ~/Library/Application Support, Windows: %APPDATA%)",
)
def test_resolver_falls_back_to_dot_config_when_xdg_unset(monkeypatch, tmp_path):
    fake_home = tmp_path / "fake-home"
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(fake_home))
    path = asset_source._default_config_path()
    assert path == asset_source.Path(fake_home) / ".config" / "DEscape" / "config.yaml"


def test_resolver_pins_the_platformdirs_argument_set(monkeypatch):
    calls = []

    def fake_user_config_dir(appname, appauthor=None, roaming=None):
        calls.append((appname, appauthor, roaming))
        return "/spied/path"

    monkeypatch.setattr(asset_source.platformdirs, "user_config_dir", fake_user_config_dir)
    path = asset_source._default_config_path()
    assert path == asset_source.Path("/spied/path") / "config.yaml"
    assert calls == [("DEscape", False, True)], (
        "appauthor=False avoids an interposed author directory on Windows, "
        "roaming=True is required to hit %APPDATA%\\DEscape\\ rather than "
        "AppData\\Local -- both must be passed, not just appname"
    )


def test_migration_copies_when_legacy_present_and_target_absent(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.yaml"
    legacy.parent.mkdir()
    legacy.write_text("aoe2de_install: /some/path\n")
    target = tmp_path / "new" / "config.yaml"
    monkeypatch.setattr(asset_source, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(asset_source, "CONFIG_PATH", target)

    result = asset_source.migrate_legacy_config()

    assert result == target
    assert target.is_file()
    assert target.read_text() == legacy.read_text()
    assert legacy.is_file(), "the legacy file must be copied, not moved"


def test_migration_leaves_an_existing_target_untouched(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.yaml"
    legacy.parent.mkdir()
    legacy.write_text("aoe2de_install: /legacy/path\n")
    target = tmp_path / "new" / "config.yaml"
    target.parent.mkdir()
    target.write_text("aoe2de_install: /already/here\n")
    monkeypatch.setattr(asset_source, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(asset_source, "CONFIG_PATH", target)

    result = asset_source.migrate_legacy_config()

    assert result is None
    assert target.read_text() == "aoe2de_install: /already/here\n"


def test_migration_is_a_true_noop_when_neither_file_exists(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.yaml"
    target = tmp_path / "new" / "config.yaml"
    monkeypatch.setattr(asset_source, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(asset_source, "CONFIG_PATH", target)

    result = asset_source.migrate_legacy_config()

    assert result is None
    assert not target.parent.exists(), "a no-op migration must not create the target directory"


def test_migration_degrades_to_none_on_oserror(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy" / "config.yaml"
    legacy.parent.mkdir()
    legacy.write_text("aoe2de_install: /some/path\n")
    target = tmp_path / "new" / "config.yaml"
    monkeypatch.setattr(asset_source, "LEGACY_CONFIG_PATH", legacy)
    monkeypatch.setattr(asset_source, "CONFIG_PATH", target)

    def raise_oserror(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(asset_source.shutil, "copy2", raise_oserror)

    result = asset_source.migrate_legacy_config()

    assert result is None


# Every config.yaml reader in a child whose locale encoding is not UTF-8: LC_ALL=C
# (ASCII) on Linux and macOS, the ANSI code page (cp1252 on CI) on Windows.
_LOCALE_PROBE = """
import json, locale, sys
from pathlib import Path
config = Path(sys.argv[1])
import descape.asset_source as asset_source
asset_source.CONFIG_PATH = config
import descape.settings as settings
settings.CONFIG_PATH = config
from descape import autosave, content_roots
result = {
    "encoding": locale.getpreferredencoding(False),
    "settings": settings._load_config(),
    "install": str(asset_source.get_install_path()),
    "language": asset_source.get_language(),
    "autosave_index": autosave._load_index(),
}
asset_source.save_install_path_to_config(Path(sys.argv[2]))
result["profile_after_save"] = content_roots._config_value(content_roots.PROFILE_CONFIG_KEY)
print(json.dumps(result))
"""


def test_a_utf8_config_with_a_bom_reads_under_a_non_utf8_locale(tmp_path) -> None:
    """README says to copy the UTF-8 example; Windows reads text as cp1252 by
    default, which mangled a non-ASCII path and kept a BOM in the first key."""
    import codecs
    import json
    import locale
    import subprocess

    from test_collection_env import ROOT, sanitized_env

    from descape import autosave

    if sys.platform == "win32" and codecs.lookup(locale.getpreferredencoding(False)).name == "utf-8":
        pytest.skip("this Windows uses UTF-8 as its ANSI code page (LC_ALL has no effect there)")
    install = tmp_path / "install"
    install.mkdir()
    other_install = tmp_path / "other_install"
    other_install.mkdir()
    profile = "C:\\Users\\José\\Games\\Age of Empires 2 DE"
    config = tmp_path / "config.yaml"
    text = (
        f"aoe2de_install: {install}\n"
        "# Copied from config.example.yaml…\n"
        "language: fr\n"
        f"aoe2de_profile: {profile}\n"
        "theme: dark\n"
    )
    config.write_bytes(("\ufeff" + text).encode("utf-8"))
    # The autosave index sits beside config.yaml and goes through the same reader.
    (tmp_path / autosave.AUTOSAVE_DIRNAME).mkdir()
    (tmp_path / autosave.AUTOSAVE_DIRNAME / autosave.INDEX_NAME).write_bytes("\ufeffk: {a: é}\n".encode("utf-8"))
    env = sanitized_env(tmp_path)
    env.update(LC_ALL="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    result = subprocess.run(
        [sys.executable, "-c", _LOCALE_PROBE, str(config), str(other_install)],
        cwd=ROOT,
        env=env,
        check=False,  # the asserts below are the check, and want the output
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    seen = json.loads(result.stdout.splitlines()[-1])
    assert seen["encoding"].replace("-", "").lower() != "utf8", f"fixture: the child's locale is {seen['encoding']}"
    assert seen["settings"] == {"aoe2de_install": str(install), "language": "fr", "aoe2de_profile": profile, "theme": "dark"}
    assert seen["install"] == str(install)
    assert seen["language"] == "fr"
    assert seen["autosave_index"] == {"k": {"a": "é"}}
    assert seen["profile_after_save"] == profile, "a read-modify-write keeps the other keys"
    assert yaml.safe_load(config.read_bytes().decode("utf-8"))["aoe2de_install"] == str(other_install)


@pytest.mark.parametrize("locale_encoding", ["cp1252", "UTF-8"])
def test_a_config_saved_in_the_ansi_encoding_still_reads(tmp_path, monkeypatch, locale_encoding: str) -> None:
    """An older Windows Notepad saved ANSI, not UTF-8: such a file keeps its
    characters, also under a UTF-8 locale, where they would otherwise turn
    into replacement characters and the next save would keep those."""
    config = tmp_path / "config.yaml"
    config.write_bytes("aoe2de_profile: C:\\Users\\José\\Games\n".encode("cp1252"))
    monkeypatch.setattr(asset_source.locale, "getpreferredencoding", lambda _do_setlocale=True: locale_encoding)
    assert yaml.safe_load(asset_source.read_config_text(config)) == {"aoe2de_profile": "C:\\Users\\José\\Games"}
