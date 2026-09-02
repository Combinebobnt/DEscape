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

import yaml

from descape import asset_source, settings


def test_missing_parent_directory_is_created_on_write(tmp_path, monkeypatch):
    deep_path = tmp_path / "no" / "such" / "dir" / "config.yaml"
    monkeypatch.setattr(asset_source, "CONFIG_PATH", deep_path)
    monkeypatch.setattr(settings, "CONFIG_PATH", deep_path)

    settings.set_dark_mode(True)
    assert deep_path.is_file()
    assert yaml.safe_load(deep_path.read_text())["dark_mode"] is True

    other_deep_path = tmp_path / "another" / "missing" / "tree" / "config.yaml"
    monkeypatch.setattr(asset_source, "CONFIG_PATH", other_deep_path)
    install = tmp_path  # any real directory works; save doesn't validate it
    asset_source.save_install_path_to_config(install)
    assert other_deep_path.is_file()
    assert yaml.safe_load(other_deep_path.read_text())["aoe2de_install"] == str(install)


def test_resolver_uses_xdg_config_home_when_set(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/fake-xdg-home")
    path = asset_source._default_config_path()
    assert path == asset_source.Path("/tmp/fake-xdg-home") / "DEscape" / "config.yaml"


def test_resolver_falls_back_to_dot_config_when_xdg_unset(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", "/tmp/fake-home")
    path = asset_source._default_config_path()
    assert path == asset_source.Path("/tmp/fake-home") / ".config" / "DEscape" / "config.yaml"


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
