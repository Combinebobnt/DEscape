"""Settings export/import (TASK-497, GH #268): descape/settings_file.py, its
startup apply in viewer.main() and Settings > General's Export settings... /
Import settings... buttons."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from descape import asset_source, settings, settings_file, theme_file, yaml_file_io

import conftest


def _reset_memos(monkeypatch) -> None:
    from testkit.settings_isolation import MEMOIZED_GLOBALS

    for name in MEMOIZED_GLOBALS:
        monkeypatch.setattr(settings, name, None)


def _use_config(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(settings, "CONFIG_PATH", path)
    monkeypatch.setattr(asset_source, "CONFIG_PATH", path)
    _reset_memos(monkeypatch)


def _write_yaml(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _doc(settings_entries: dict | None = None, **overrides) -> dict:
    return {
        "format": settings_file.FORMAT_ID,
        "version": settings_file.FORMAT_VERSION,
        "exported_by": "DEscape test",
        "known_keys": list(settings_file.PORTABLE_KEYS),
        "settings": settings_entries or {},
        **overrides,
    }


def _stage(tmp_path: Path, doc: dict) -> settings_file.ImportPlan:
    source = _write_yaml(tmp_path / "in" / f"s{settings_file.FILE_SUFFIX}", doc)
    plan = settings_file.parse_settings_file(source)
    settings_file.stage_import(plan, source)
    return plan


def _config() -> Path:
    return settings.CONFIG_PATH


def _portable(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in settings_file.PORTABLE_KEYS}


def _local(config: dict) -> dict:
    return {k: v for k, v in config.items() if k in settings_file.LOCAL_KEYS and k != settings_file.IMPORT_APPLIED_KEY}


def _applied() -> dict:
    """config.yaml less the import id an apply records beside the settings."""
    config = yaml.safe_load(_config().read_text())
    config.pop(settings_file.IMPORT_APPLIED_KEY, None)
    return config


# --- round trip, replace and legacy keys --------------------------------------


def test_a_round_trip_replaces_the_portable_keys_and_keeps_the_local_ones(tmp_path, monkeypatch) -> None:
    """Config A: customized keybinds and tool options, a legacy dark_mode, local keys.
    Config B: other local keys, a portable customization A lacks, a theme A's dark_mode must beat."""
    a = tmp_path / "a" / "config.yaml"
    _use_config(monkeypatch, a)
    settings.set_keybind("file_open", "Ctrl+Shift+O")
    settings.set_tool_option_default("brush_size", 3)
    settings.set_window_size(1000, 800)
    asset_source.save_install_path_to_config(tmp_path / "install-a")
    config_a = {**yaml.safe_load(a.read_text()), "dark_mode": True}
    _write_yaml(a, config_a)
    exported = tmp_path / f"a{settings_file.FILE_SUFFIX}"
    settings_file.write_settings_file(exported, settings_file.build_document())

    b = tmp_path / "config.yaml"
    _use_config(monkeypatch, b)
    settings.set_theme("solarized_light")
    settings.set_grid_overlay(True)
    settings.set_window_size(900, 700)
    settings.add_recent_file(tmp_path / "m.aoe2scenario")
    asset_source.save_install_path_to_config(tmp_path / "install-b")
    old_b = b.read_bytes()

    plan = settings_file.parse_settings_file(exported)
    assert (plan.problems, plan.notes) == ([], [])
    settings_file.stage_import(plan, exported)
    line = settings_file.apply_pending_import()

    new_b = yaml.safe_load(b.read_text())
    assert _portable(new_b) == _portable(config_a)
    assert _local(new_b) == _local(yaml.safe_load(old_b))
    assert "grid_overlay" not in new_b
    _reset_memos(monkeypatch)
    assert settings.get_theme() == "dark"
    assert settings.get_keybind("file_open") == "Ctrl+Shift+O"
    assert settings.get_tool_option_default("brush_size") == 3
    assert settings_file.backup_path().read_bytes() == old_b
    assert not settings_file.pending_path().exists()
    assert line == f"Settings imported from {exported}; the previous settings are in {settings_file.backup_path()}."


def test_a_key_the_exporter_did_not_know_keeps_its_local_value(tmp_path) -> None:
    settings.set_grid_overlay(True)
    settings.set_pan_speed(900)
    known = [key for key in settings_file.PORTABLE_KEYS if key != "grid_overlay"]
    _stage(tmp_path, _doc({}, known_keys=known))
    settings_file.apply_pending_import()
    config = yaml.safe_load(_config().read_text())
    assert config["grid_overlay"] is True
    assert "pan_speed" not in config


def test_an_older_exporters_legacy_key_beats_a_local_successor(tmp_path, monkeypatch) -> None:
    settings.set_theme("solarized_light")
    known = [key for key in settings_file.PORTABLE_KEYS if key not in ("theme", "theme_colors")]
    _stage(tmp_path, _doc({"dark_mode": True}, known_keys=known))
    settings_file.apply_pending_import()
    assert "theme" not in yaml.safe_load(_config().read_text())
    _reset_memos(monkeypatch)
    assert settings.get_theme() == "dark"


def test_a_newer_exporters_known_successor_drops_a_stale_local_legacy_key(tmp_path, monkeypatch) -> None:
    _write_yaml(_config(), {"dark_mode": True, "pan_speed": 900})
    _stage(tmp_path, _doc({}, known_keys=["theme"]))
    settings_file.apply_pending_import()
    assert _applied() == {"pan_speed": 900}
    _reset_memos(monkeypatch)
    assert settings.get_theme() != "dark"


@pytest.mark.parametrize(
    ("local", "expected"),
    [
        ({"paint_trees": False, "brush_size": 3}, {"brush_size": 3}),
        ({"paint_trees": False}, None),
    ],
)
def test_a_legacy_tool_option_in_the_file_drops_only_that_local_option(tmp_path, monkeypatch, local, expected) -> None:
    _write_yaml(_config(), {"tool_options": local})
    _stage(tmp_path, _doc({"paint_trees": True}, known_keys=["paint_trees"]))
    settings_file.apply_pending_import()
    config = yaml.safe_load(_config().read_text())
    assert config.get("tool_options") == expected
    assert config["paint_trees"] is True
    _reset_memos(monkeypatch)
    assert settings.get_tool_option_default("paint_trees") is True


# --- export ---------------------------------------------------------------------


def test_export_carries_the_portable_entries_as_stored_and_no_local_key(tmp_path) -> None:
    portable = {
        "tool_options": {"brush_size": 3},
        "potato_mode": True,
        "keybinds": {"file_open": "Ctrl+Shift+O"},
        "language": "de",
    }
    _write_yaml(_config(), {**dict.fromkeys(settings_file.LOCAL_KEYS, "/here"), **portable})
    doc = settings_file.build_document()
    assert doc["settings"] == portable
    assert doc["known_keys"] == list(settings_file.PORTABLE_KEYS)
    out = tmp_path / f"mine{settings_file.FILE_SUFFIX}"
    settings_file.write_settings_file(out, doc)
    assert yaml.safe_load(out.read_text()) == doc
    assert not set(settings_file.parse_settings_file(out).values) & set(settings_file.LOCAL_KEYS)


def test_a_subscribed_path_is_refused_with_nothing_written(tmp_path) -> None:
    subscribed = tmp_path / "mods" / "subscribed" / "12345"
    subscribed.mkdir(parents=True)
    with pytest.raises(settings_file.SettingsWriteBlockedError):
        settings_file.write_settings_file(subscribed / f"s{settings_file.FILE_SUFFIX}", settings_file.build_document())
    assert list(subscribed.iterdir()) == []


@pytest.mark.parametrize("exporter", ["settings", "theme"])
@pytest.mark.parametrize(
    "target_name",
    [settings.CONFIG_PATH.name, settings_file.PENDING_NAME, settings_file.BAD_PENDING_NAME, settings_file.BACKUP_NAME],
)
def test_an_export_refuses_config_yaml_and_the_files_beside_it(tmp_path, exporter, target_name) -> None:
    """A `.yaml` name gets no suffix, so Export could otherwise replace config.yaml or an import file."""
    settings.set_pan_speed(800)
    target = settings.CONFIG_PATH.parent / target_name
    if not target.exists():
        target.write_text("keep me\n")
    before = target.read_bytes()
    write, doc, blocked = (
        (settings_file.write_settings_file, settings_file.build_document(), settings_file.SettingsWriteBlockedError)
        if exporter == "settings"
        else (theme_file.write_theme_file, theme_file.build_document(), theme_file.ThemeWriteBlockedError)
    )
    with pytest.raises(blocked, match=target_name):
        write(target, doc)
    assert target.read_bytes() == before


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
@conftest.requires_symlinks
def test_an_export_refuses_a_symlinked_configs_target_and_a_path_through_a_link(tmp_path) -> None:
    real = _write_yaml(tmp_path / "dotfiles" / "config.yaml", {"pan_speed": 800})
    _config().symlink_to(real)
    link_dir = tmp_path / "link"
    link_dir.symlink_to(_config().parent, target_is_directory=True)
    before = real.read_bytes()
    for target in (real, link_dir / "config.yaml"):
        with pytest.raises(settings_file.SettingsWriteBlockedError):
            settings_file.write_settings_file(target, settings_file.build_document())
    assert real.read_bytes() == before
    assert _config().is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard link semantics")
def test_an_export_refuses_a_hard_link_of_config_yaml(tmp_path) -> None:
    """Neither the name nor resolve() matches a hard link: only yaml_file_io._same_file()'s samefile() does."""
    settings.set_pan_speed(800)
    link = tmp_path / "elsewhere" / "linked.yaml"
    link.parent.mkdir()
    os.link(_config(), link)
    before = _config().read_bytes()
    with pytest.raises(settings_file.SettingsWriteBlockedError, match=_config().name):
        settings_file.write_settings_file(link, settings_file.build_document())
    assert _config().read_bytes() == before and link.read_bytes() == before
    assert os.path.samefile(_config(), link), "the link must survive, or the write went through"


@conftest.requires_symlinks
def test_a_symlink_into_mods_subscribed_is_refused_with_nothing_written(tmp_path) -> None:
    subscribed = tmp_path / "mods" / "subscribed" / "12345"
    subscribed.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(subscribed, target_is_directory=True)
    with pytest.raises(settings_file.SettingsWriteBlockedError):
        settings_file.write_settings_file(link / f"s{settings_file.FILE_SUFFIX}", settings_file.build_document())
    assert list(subscribed.iterdir()) == []


# --- parsing and refusals --------------------------------------------------------


def test_an_unknown_setting_is_a_problem_and_a_local_one_a_note(tmp_path) -> None:
    doc = _doc({"pan_speed": 900, "bogus": 1, "aoe2de_install": "/elsewhere"}, extra=1)
    plan = settings_file.parse_settings_file(_write_yaml(tmp_path / "s.yaml", doc), installed_families=[])
    assert plan.values == {"pan_speed": 900}
    assert plan.problems == ["unknown key 'extra' outside settings", "unknown setting 'bogus'"]
    assert len(plan.notes) == 1 and plan.notes[0].startswith("aoe2de_install stays on this machine")


def test_known_keys_this_build_does_not_know_are_dropped_silently(tmp_path) -> None:
    doc = _doc({}, known_keys=["pan_speed", "from_a_newer_build", "aoe2de_install"])
    plan = settings_file.parse_settings_file(_write_yaml(tmp_path / "s.yaml", doc))
    assert plan.known_keys == ["pan_speed"]
    assert plan.problems == []


def test_an_uninstalled_font_is_noted(tmp_path) -> None:
    doc = _doc({"ui_font_family": "Nowhere Sans"})
    plan = settings_file.parse_settings_file(_write_yaml(tmp_path / "s.yaml", doc), installed_families=["Other"])
    assert plan.values == {"ui_font_family": "Nowhere Sans"}
    assert any("'Nowhere Sans' is not installed" in note for note in plan.notes)


def _alias_bomb(levels: int, fanout: int) -> str:
    lines = ['l0: &l0 "xxxxxxxxxx"']
    for level in range(1, levels + 1):
        refs = ", ".join([f"*l{level - 1}"] * fanout)
        lines.append(f"l{level}: &l{level} [{refs}]")
    return "\n".join(lines) + "\n"


def _nested(depth: int):
    value: object = 1
    for _ in range(depth):
        value = [value]
    return value


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (yaml.safe_dump(_doc()) + "#" * settings_file.MAX_FILE_BYTES, "limit"),
        ("format: descape-settings\nversion: [unterminated\n", "not valid YAML"),
        (yaml.safe_dump(_doc()) + _alias_bomb(5, 10), "aliases"),
        (yaml.safe_dump(["format", "descape-settings"]), "not a mapping"),
        (yaml.safe_dump({"format": "something-else", "version": 1}), "not a DEscape settings file"),
        (yaml.safe_dump(theme_file.build_document()), "Import theme"),
        (yaml.safe_dump(_doc(version=settings_file.FORMAT_VERSION + 1)), "newer than"),
        (yaml.safe_dump(_doc(version="1")), "version"),
        (yaml.safe_dump(_doc(settings=[1])), "settings is not a mapping"),
        (yaml.safe_dump({k: v for k, v in _doc().items() if k != "settings"}), "settings is not a mapping"),
        (yaml.safe_dump(_doc(known_keys="pan_speed")), "known_keys"),
        (yaml.safe_dump(_doc(known_keys=[1])), "known_keys"),
        (yaml.safe_dump({k: v for k, v in _doc().items() if k != "known_keys"}), "known_keys"),
        (yaml.safe_dump(_doc({"tool_options": _nested(40)})), "nested too deeply"),
    ],
    # Short ids: the oversized text as an id overflows Windows' 32767-char PYTEST_CURRENT_TEST.
    ids=[
        "oversized",
        "bad-yaml",
        "alias-bomb",
        "not-a-mapping",
        "other-format",
        "theme-file",
        "newer-version",
        "string-version",
        "settings-list",
        "settings-missing",
        "known-keys-string",
        "known-keys-ints",
        "known-keys-missing",
        "too-deep",
    ],
)
def test_a_file_that_is_not_a_settings_file_is_refused_whole_and_nothing_is_staged(tmp_path, text, fragment) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(text)
    with pytest.raises(settings_file.SettingsFileRefused, match=fragment):
        settings_file.parse_settings_file(path)
    assert settings_file.pending_import() is None


def test_a_settings_file_offered_to_the_theme_importer_names_import_settings(tmp_path) -> None:
    path = _write_yaml(tmp_path / "s.yaml", _doc({"pan_speed": 900}))
    with pytest.raises(theme_file.ThemeFileRefused) as refused:
        theme_file.parse_theme_file(path)
    assert str(refused.value).startswith("not a DEscape theme file")
    assert "Import settings" in str(refused.value)


# --- describe ---------------------------------------------------------------------


def _describe(tmp_path, entries: dict, **doc_overrides) -> list[str]:
    path = _write_yaml(tmp_path / "s.yaml", _doc(entries, **doc_overrides))
    return settings_file.describe_changes(settings_file.parse_settings_file(path))


def test_one_differing_keybind_counts_one_binding_not_every_action(tmp_path) -> None:
    settings.set_keybind("file_open", "Ctrl+Shift+O")  # writes every action
    stored = yaml.safe_load(_config().read_text())["keybinds"]
    lines = _describe(tmp_path, {"keybinds": {**stored, "file_open": "Ctrl+Alt+O"}})
    assert lines == ["Will change:", "  Keybinds: 1 binding differs"]


def test_a_retired_or_unknown_keybind_counts_nothing(tmp_path) -> None:
    lines = _describe(tmp_path, {"keybinds": {"tool_wall_run": "W", "no_such_action": "Q"}})
    assert lines == ["Nothing in this file differs from the current settings."]


def test_a_files_legacy_dark_mode_reads_as_a_theme_change(tmp_path) -> None:
    settings.set_theme("solarized_light")
    lines = _describe(tmp_path, {"dark_mode": True})
    assert lines == ["Will change:", "  Appearance: Theme: 'solarized_light' -> True (as dark_mode)"]


def test_resets_are_listed_under_their_own_heading(tmp_path) -> None:
    settings.set_grid_overlay(True)
    settings.set_view_layer("terrain_blend", False)
    lines = _describe(tmp_path, {"pan_speed": 900})
    assert lines == [
        "Will change:",
        "  Appearance: Pan speed: default -> 900",
        "",
        "Will reset to default:",
        "  View menu: Layers: 1 layer differs",
        "  View menu: Grid > Show: True -> default",
    ]


def test_every_summary_line_names_its_control_not_its_config_key(tmp_path) -> None:
    """One file changing every shown portable key at once, a dict-valued one in a single entry:
    one line each, none naming a raw key, and a label that repeats its group (Keybinds, Tools)
    shown once."""
    shown = [key for key in settings_file.PORTABLE_KEYS if key not in settings_file.LEGACY_SUCCESSORS]
    entries: dict = dict.fromkeys(shown, "zz-changed")
    for key, (defaults, _one, _many) in settings_file._entry_ids().items():
        entries[key] = {next(iter(defaults)) if defaults else "messages.instructions": "zz-changed"}
    lines = [line.strip() for line in _describe(tmp_path, entries) if line.startswith("  ")]
    assert len(lines) == len(shown), lines
    for line in lines:
        assert not set(line.split(": ")[:2]) & set(settings_file.PORTABLE_KEYS), line
    assert "Keybinds: 1 binding differs" in lines
    assert "Tools: 1 tool default differs" in lines


def test_the_summary_adds_problems_notes_and_the_next_launch_rule(tmp_path) -> None:
    doc = _doc({"pan_speed": 900, "bogus": 1, "window_size": [1, 2]})
    plan = settings_file.parse_settings_file(_write_yaml(tmp_path / "s.yaml", doc))
    summary = settings_file.format_import_summary(plan)
    assert "Skipped (not applied):\n  unknown setting 'bogus'" in summary
    assert "Notes:\n  window_size stays on this machine" in summary
    assert summary.endswith("These apply the next time DEscape starts.")


# --- apply ------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["- a\n- list\n", "theme: [unterminated\n", "just text\n"])
def test_an_unreadable_config_refuses_export_and_apply_leaves_it_untouched(tmp_path, text) -> None:
    _config().write_text(text)
    before = _config().read_bytes()
    with pytest.raises(settings_file.ConfigUnreadable):
        settings_file.build_document()
    _stage(tmp_path, _doc({"pan_speed": 900}))
    line = settings_file.apply_pending_import()
    assert line is not None and line.startswith("Settings import failed and nothing changed")
    assert _config().read_bytes() == before
    assert not settings_file.pending_path().exists()
    assert settings_file.bad_pending_path().is_file()
    assert not settings_file.backup_path().exists()


def test_an_empty_config_reads_as_empty_for_export_and_apply(tmp_path) -> None:
    _config().write_text("")
    assert settings_file.build_document()["settings"] == {}
    _stage(tmp_path, _doc({"pan_speed": 900}))
    line = settings_file.apply_pending_import()
    assert _applied() == {"pan_speed": 900}
    assert settings_file.backup_path().read_bytes() == b""
    assert settings_file.BACKUP_NAME in line


@pytest.mark.parametrize("older_bad", [False, True])
def test_a_refused_config_write_leaves_it_untouched_and_moves_the_pending_file_aside(
    tmp_path, monkeypatch, older_bad
) -> None:
    settings.set_pan_speed(800)
    before = _config().read_bytes()
    _stage(tmp_path, _doc({"pan_speed": 900}))
    staged = settings_file.pending_path().read_bytes()
    if older_bad:
        settings_file.bad_pending_path().write_text("an older failure\n")
    target = _config().resolve()
    real_replace = os.replace

    def refusing(src, dst):
        if Path(dst) == target:
            raise PermissionError("config.yaml is held open by another process")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", refusing)
    line = settings_file.apply_pending_import()
    assert line is not None and "held open by another process" in line
    assert _config().read_bytes() == before
    assert not settings_file.pending_path().exists()
    assert settings_file.bad_pending_path().read_bytes() == staged
    assert not [p.name for p in _config().parent.iterdir() if p.name.endswith(".tmp")]


def test_a_failure_after_the_config_write_is_reported_as_applied(tmp_path, monkeypatch) -> None:
    _stage(tmp_path, _doc({"pan_speed": 900}))
    pending = settings_file.pending_path()
    real_unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self == pending:
            raise PermissionError("locked")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    line = settings_file.apply_pending_import()
    assert line is not None and line.startswith("Settings imported from") and "could not be removed" in line
    assert _applied() == {"pan_speed": 900}
    assert not settings_file.bad_pending_path().exists()


def test_a_pending_file_a_failed_unlink_left_is_not_reapplied_over_later_edits(tmp_path, monkeypatch) -> None:
    """The unlink after a good config write fails; the user then edits a portable key and
    relaunches. That apply must only remove the pending file: the edit survives and the backup
    still holds the pre-import config."""
    settings.set_pan_speed(800)
    old = _config().read_bytes()
    _stage(tmp_path, _doc({"pan_speed": 900}))
    pending = settings_file.pending_path()
    real_unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self == pending:
            raise PermissionError("locked")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)
    assert "could not be removed" in settings_file.apply_pending_import()
    monkeypatch.setattr(Path, "unlink", real_unlink)
    _reset_memos(monkeypatch)
    assert settings.get_pan_speed() == 900
    settings.set_pan_speed(1200)

    line = settings_file.apply_pending_import()
    assert line is not None and "already applied" in line
    assert settings_file.backup_path().read_bytes() == old
    _reset_memos(monkeypatch)
    assert settings.get_pan_speed() == 1200
    assert not pending.exists()


def _lock_pending(monkeypatch) -> None:
    """Path.unlink refuses the staged file, as one held open on Windows does."""
    pending = settings_file.pending_path()
    real_unlink = Path.unlink

    def locked(self, missing_ok=False):
        if self == pending:
            raise PermissionError("locked")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked)


def test_a_pending_file_a_failed_unlink_left_is_not_shown_as_pending(tmp_path, monkeypatch) -> None:
    """Settings > General said such an import still applies at the next start."""
    _stage(tmp_path, _doc({"pan_speed": 900}))
    assert settings_file.pending_import() is not None
    _lock_pending(monkeypatch)
    assert "could not be removed" in settings_file.apply_pending_import()
    assert settings_file.pending_path().exists()
    assert settings_file.pending_import() is None


def test_a_second_failed_unlink_says_already_applied_not_failed(tmp_path, monkeypatch) -> None:
    _stage(tmp_path, _doc({"pan_speed": 900}))
    pending = settings_file.pending_path()
    _lock_pending(monkeypatch)
    assert "could not be removed" in settings_file.apply_pending_import()
    applied = _config().read_bytes()

    line = settings_file.apply_pending_import()
    assert line is not None and "already applied" in line and f"{pending} could not be removed" in line
    assert pending.exists() and not settings_file.bad_pending_path().exists()
    assert _config().read_bytes() == applied


def test_a_pending_file_another_instance_took_is_nothing_pending(tmp_path, monkeypatch) -> None:
    settings.set_pan_speed(800)
    before = _config().read_bytes()
    _stage(tmp_path, _doc({"pan_speed": 900}))
    real_read_root = yaml_file_io.read_root

    def taken(path, fmt):
        settings_file.pending_path().unlink()
        return real_read_root(path, fmt)

    monkeypatch.setattr(yaml_file_io, "read_root", taken)
    assert settings_file.apply_pending_import() is None
    assert _config().read_bytes() == before
    assert not settings_file.bad_pending_path().exists()


def test_a_rerun_against_the_applied_config_backs_up_and_writes_nothing(tmp_path, monkeypatch) -> None:
    settings.set_pan_speed(800)
    settings.set_theme("dark")
    old = _config().read_bytes()
    _stage(tmp_path, _doc({"pan_speed": 900, "dark_mode": False}))
    staged = settings_file.pending_path().read_bytes()
    settings_file.apply_pending_import()
    applied = _config().read_bytes()
    assert settings_file.backup_path().read_bytes() == old

    settings_file.pending_path().write_bytes(staged)  # a crash before the unlink, or a second instance
    writes = []
    real_write = yaml_file_io.atomic_write
    monkeypatch.setattr(yaml_file_io, "atomic_write", lambda path, data, **kw: (writes.append(path), real_write(path, data, **kw)))
    line = settings_file.apply_pending_import()
    assert writes == []
    assert line is not None and "already applied" in line
    assert _config().read_bytes() == applied
    assert settings_file.backup_path().read_bytes() == old
    assert not settings_file.pending_path().exists()


_ABSENT = object()


def _pending_doc(**header) -> dict:
    """A staged file as stage_import() writes it, less its id; an _ABSENT value drops that key."""
    doc = {
        "format": settings_file.PENDING_FORMAT_ID,
        "version": settings_file.PENDING_VERSION,
        "mode": "replace",
        "source": "x",
        "known_keys": ["pan_speed"],
        "settings": {"pan_speed": 900},
        **header,
    }
    return {k: v for k, v in doc.items() if v is not _ABSENT}


@pytest.mark.parametrize(
    "header",
    [{"version": settings_file.PENDING_VERSION + 1}, {"mode": "merge"}],
)
def test_a_pending_file_from_a_newer_build_stays_staged_and_changes_nothing(tmp_path, header) -> None:
    settings.set_pan_speed(800)
    before = _config().read_bytes()
    _write_yaml(settings_file.pending_path(), _pending_doc(**header))
    line = settings_file.apply_pending_import()
    assert line is not None and line.startswith("Settings import not applied, left staged")
    assert settings_file.pending_path().is_file()
    assert not settings_file.bad_pending_path().exists()
    assert _config().read_bytes() == before


@pytest.mark.parametrize("mode", [_ABSENT, None, 7, ["replace"]])
def test_a_pending_file_whose_mode_is_not_text_is_moved_aside_and_changes_nothing(tmp_path, mode) -> None:
    """Only a text mode other than replace can be a newer build's; anything else no build applies."""
    settings.set_pan_speed(800)
    before = _config().read_bytes()
    _write_yaml(settings_file.pending_path(), _pending_doc(mode=mode))
    line = settings_file.apply_pending_import()
    assert line is not None and line.startswith("Settings import failed and nothing changed")
    assert not settings_file.pending_path().exists()
    assert settings_file.bad_pending_path().is_file()
    assert _config().read_bytes() == before


@pytest.mark.parametrize(
    ("header", "fragment"),
    [
        ({"version": settings_file.PENDING_VERSION + 1}, "a newer DEscape staged it"),
        ({"mode": "merge"}, "a newer DEscape staged it"),
        ({"mode": None}, "it cannot be read"),
        ({"settings": [1]}, "it cannot be read"),
    ],
)
def test_pending_import_says_why_this_build_will_not_apply_a_staged_file(tmp_path, header, fragment) -> None:
    _write_yaml(settings_file.pending_path(), _pending_doc(**header))
    pending = settings_file.pending_import()
    assert pending is not None and pending.refusal is not None and pending.refusal.startswith(fragment)
    assert pending.settings == {}


def test_a_corrupt_pending_file_is_moved_aside_and_changes_nothing(tmp_path) -> None:
    settings.set_pan_speed(800)
    before = _config().read_bytes()
    settings_file.pending_path().write_text("format: [broken\n")
    line = settings_file.apply_pending_import()
    assert line is not None and line.startswith("Settings import failed")
    assert _config().read_bytes() == before
    assert settings_file.bad_pending_path().read_text() == "format: [broken\n"
    assert not settings_file.pending_path().exists()


def test_with_no_config_the_apply_writes_one_and_names_no_backup(tmp_path) -> None:
    settings_file.backup_path().write_text("an older backup\n")
    _stage(tmp_path, _doc({"pan_speed": 900}))
    assert not _config().exists()
    line = settings_file.apply_pending_import()
    assert _applied() == {"pan_speed": 900}
    assert settings_file.backup_path().read_text() == "an older backup\n"
    assert line is not None and settings_file.BACKUP_NAME not in line


def test_nothing_pending_returns_none(tmp_path) -> None:
    assert settings_file.apply_pending_import() is None
    assert settings_file.pending_import() is None


def test_discard_removes_the_pending_file(tmp_path) -> None:
    _stage(tmp_path, _doc({"pan_speed": 900}))
    pending = settings_file.pending_import()
    assert (pending.settings, pending.refusal) == ({"pan_speed": 900}, None)
    settings_file.discard_pending_import()
    assert settings_file.pending_import() is None


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX mode bits (st_mode is only 0o666 or 0o444)")
def test_an_applied_config_and_its_backup_keep_the_configs_mode(tmp_path) -> None:
    settings.set_pan_speed(800)
    _config().chmod(0o600)
    _stage(tmp_path, _doc({"pan_speed": 900}))
    settings_file.apply_pending_import()
    assert stat.S_IMODE(_config().stat().st_mode) == 0o600
    assert stat.S_IMODE(settings_file.backup_path().stat().st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics")
@conftest.requires_symlinks
def test_a_symlinked_config_stays_a_symlink_to_the_updated_target(tmp_path) -> None:
    real = _write_yaml(tmp_path / "dotfiles" / "config.yaml", {"pan_speed": 800})
    _config().symlink_to(real)
    _stage(tmp_path, _doc({"pan_speed": 900}))
    settings_file.apply_pending_import()
    assert _config().is_symlink()
    assert _config().resolve() == real.resolve()
    assert _applied() == {"pan_speed": 900}


# --- startup (viewer.main()) ------------------------------------------------------

# test_theme_file.py's startup probe, plus the status log's last line.
_STARTUP_PROBE = """
import sys
from pathlib import Path
config = Path(sys.argv[1])
import descape.asset_source as asset_source
import descape.settings as settings
asset_source.CONFIG_PATH = config
settings.CONFIG_PATH = config
asset_source.LEGACY_CONFIG_PATH = config.parent / "no-legacy-config.yaml"
from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QApplication
import descape.viewer as viewer

def _probe_exec(app):
    palette = app.palette()
    print("HIGHLIGHT", palette.color(QPalette.Active, QPalette.Highlight).name())
    print("WINDOW", palette.color(QPalette.Active, QPalette.Window).name())
    for widget in app.topLevelWidgets():
        if isinstance(widget, viewer.ViewerWindow):
            print("STATUS", (widget.status_log.toPlainText().splitlines() or ["-"])[-1])
            widget.edit_history.mark_saved()
            widget.close()
    return 0

QApplication.exec_ = _probe_exec
sys.argv = ["descape"]
try:
    viewer.main()
except SystemExit as e:
    print("EXIT", e.code)
"""


@pytest.mark.gui
def test_main_applies_a_staged_import_before_the_first_settings_read(tmp_path) -> None:
    """A staged import whose theme differs, then a fresh `viewer.main()` in its own process.

    Pass conditions: before its event loop starts, main()'s palette has the imported preset's
    Window colour and Highlight override; the status log names the import; afterwards
    config.yaml holds the import, the backup holds the old config and nothing is pending.
    """
    import subprocess
    import sys

    from descape import themes

    settings.set_theme("dark")
    settings.set_window_size(1000, 800)
    old = _config().read_bytes()
    _stage(tmp_path, _doc({"theme": "high_contrast_light", "theme_colors": {"highlight": "#123456"}}))

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QPA_PLATFORMTHEME="", XDG_CONFIG_HOME=str(tmp_path / "xdg"))
    proc = subprocess.run(
        [sys.executable, "-c", _STARTUP_PROBE, str(_config())],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = dict(line.split(" ", 1) for line in proc.stdout.splitlines() if line.startswith(("HIGHLIGHT ", "WINDOW ", "STATUS ", "EXIT ")))
    assert out.get("EXIT") == "0", proc.stderr[-2000:]
    assert out["HIGHLIGHT"] == "#123456"
    assert out["WINDOW"] == themes.PRESETS["high_contrast_light"][1]["window"]
    assert out["STATUS"].startswith("Settings imported from")
    config = yaml.safe_load(_config().read_text())
    assert (config["theme"], config["theme_colors"]) == ("high_contrast_light", {"highlight": "#123456"})
    assert config["window_size"] == [1000, 800]
    assert settings_file.backup_path().read_bytes() == old
    assert not settings_file.pending_path().exists()


# --- Settings > General -----------------------------------------------------------


@pytest.fixture
def qt_dialog():
    if not conftest.PYQT5_AVAILABLE:
        pytest.skip("PyQt5 not importable")
    dialog, window = conftest.dialog_and_window()
    yield dialog
    dialog.close()
    conftest.close_window(window)


def _answer_import(monkeypatch, path: Path, apply: bool, quit_now: bool = False) -> tuple[list[str], list[int]]:
    from PyQt5.QtWidgets import QFileDialog

    from descape.viewer import SettingsDialog

    shown: list[str] = []
    asked: list[int] = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(SettingsDialog, "_confirm_settings_import", lambda _self, summary: shown.append(summary) or apply)
    monkeypatch.setattr(SettingsDialog, "_ask_quit_after_import", lambda _self: asked.append(1) or quit_now)
    return shown, asked


def _save_to(monkeypatch, path: Path) -> None:
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(path), "")))


# LOCAL_KEYS -> the words the tooltip and the guide name it by; None for one no user sets.
_LOCAL_KEY_WORDS: dict[str, str | None] = {
    "aoe2de_install": "folder paths",
    "aoe2de_profile": "folder paths",
    "autosave_dir": "folder paths",
    "recent_files": "recent files",
    "new_map": "New Map dialog's last choices",
    "data_mod_by_scenario": "Data Mod",
    "window_size": "window",
    "split_sizes": "window",
    "log_height": "window",
    "xwayland_on_wayland": "X11 compatibility on Wayland",
    settings_file.IMPORT_APPLIED_KEY: None,
}


def test_the_guide_names_everything_that_stays_on_this_computer() -> None:
    assert set(_LOCAL_KEY_WORDS) == set(settings_file.LOCAL_KEYS)
    guide = (Path(__file__).resolve().parent.parent / "descape" / "help" / "settings.md").read_text(encoding="utf-8")
    general = guide.split("## General", 1)[1].split("\n## ", 1)[0]
    missing = {key: words for key, words in _LOCAL_KEY_WORDS.items() if words is not None and words not in general}
    assert not missing


@pytest.mark.gui
def test_the_settings_file_tooltip_names_everything_that_stays_on_this_computer(qt_dialog) -> None:
    tooltip = qt_dialog.settings_export_button.toolTip()
    assert qt_dialog.settings_import_button.toolTip() == tooltip
    missing = {key: words for key, words in _LOCAL_KEY_WORDS.items() if words is not None and words not in tooltip}
    assert not missing


@pytest.mark.gui
def test_the_general_tab_has_both_buttons_and_no_pending_line(qt_dialog) -> None:
    from PyQt5.QtWidgets import QTabWidget

    tabs = qt_dialog.findChild(QTabWidget)
    general = tabs.widget(next(i for i in range(tabs.count()) if tabs.tabText(i) == "General"))
    assert qt_dialog.settings_export_button.text() == "Export settings…"
    assert qt_dialog.settings_import_button.text() == "Import settings…"
    assert general.isAncestorOf(qt_dialog.settings_export_button)
    assert general.isAncestorOf(qt_dialog.settings_import_button)
    assert qt_dialog.settings_pending_row.isHidden()


@pytest.mark.gui
def test_an_import_stages_and_shows_the_pending_line_and_discard_removes_both(tmp_path, monkeypatch, qt_dialog) -> None:
    path = _write_yaml(tmp_path / f"mine{settings_file.FILE_SUFFIX}", _doc({"pan_speed": 900}))
    shown, asked = _answer_import(monkeypatch, path, apply=True)
    qt_dialog.settings_import_button.click()
    assert len(shown) == 1 and "  Appearance: Pan speed: default -> 900" in shown[0].splitlines()
    assert asked == [1]
    assert settings_file.pending_import().settings == {"pan_speed": 900}
    assert not qt_dialog.settings_pending_row.isHidden()
    assert qt_dialog.settings_pending_label.text() == f"Settings from {path.name} apply the next time DEscape starts."
    assert settings.get_pan_speed() == settings.PAN_SPEED_DEFAULT  # nothing applies live

    qt_dialog.settings_pending_discard_button.click()
    assert settings_file.pending_import() is None
    assert qt_dialog.settings_pending_row.isHidden()


@pytest.mark.gui
def test_a_dialog_opened_with_an_import_staged_shows_the_pending_line(tmp_path, qt_dialog) -> None:
    from descape.viewer import SettingsDialog

    _stage(tmp_path, _doc({"pan_speed": 900}))
    second = SettingsDialog(qt_dialog._window)
    try:
        assert not second.settings_pending_row.isHidden()
        assert second.settings_pending_label.text().startswith(f"Settings from s{settings_file.FILE_SUFFIX} apply")
    finally:
        second.close()


@pytest.mark.gui
def test_the_pending_line_says_when_this_build_will_not_apply_the_staged_file(qt_dialog) -> None:
    from descape.viewer import SettingsDialog

    _write_yaml(settings_file.pending_path(), _pending_doc(mode="merge"))
    second = SettingsDialog(qt_dialog._window)
    try:
        assert not second.settings_pending_row.isHidden()
        text = second.settings_pending_label.text()
        assert "'merge'" in text and "next time DEscape starts" not in text
        assert not second.settings_pending_discard_button.isHidden()
    finally:
        second.close()


@pytest.mark.gui
def test_a_cancelled_import_stages_nothing(tmp_path, monkeypatch, qt_dialog) -> None:
    path = _write_yaml(tmp_path / "s.yaml", _doc({"pan_speed": 900}))
    shown, asked = _answer_import(monkeypatch, path, apply=False)
    qt_dialog.settings_import_button.click()
    assert len(shown) == 1 and asked == []
    assert settings_file.pending_import() is None
    assert qt_dialog.settings_pending_row.isHidden()


@pytest.mark.gui
@pytest.mark.parametrize(
    ("file_text", "config_text"), [("format: something-else\nversion: 1\n", None), (None, "theme: [unterminated\n")]
)
def test_a_refused_import_warns_and_stages_nothing(tmp_path, monkeypatch, qt_dialog, file_text, config_text) -> None:
    from PyQt5.QtWidgets import QMessageBox

    path = tmp_path / "s.yaml"
    if file_text is None:
        _write_yaml(path, _doc({"pan_speed": 900}))
    else:
        path.write_text(file_text)
    if config_text is not None:
        _config().write_text(config_text)
    shown, _asked = _answer_import(monkeypatch, path, apply=True)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a)))
    qt_dialog.settings_import_button.click()
    assert len(warned) == 1 and warned[0][1] == "Import refused"
    assert shown == []
    assert settings_file.pending_import() is None


@pytest.mark.gui
def test_quit_now_closes_the_dialog_and_then_the_window(tmp_path, monkeypatch, qt_dialog) -> None:
    from PyQt5.QtWidgets import QApplication

    path = _write_yaml(tmp_path / "s.yaml", _doc({"pan_speed": 900}))
    _answer_import(monkeypatch, path, apply=True, quit_now=True)
    window = qt_dialog._window
    real_dialog_close, real_window_close = qt_dialog.close, window.close
    order: list[str] = []
    qt_dialog.close = lambda: order.append("dialog")
    window.close = lambda: order.append("window")
    try:
        qt_dialog.settings_import_button.click()
        assert order == ["dialog"]  # the window's close waits for the event loop
        QApplication.processEvents()
        assert order == ["dialog", "window"]
    finally:
        qt_dialog.close, window.close = real_dialog_close, real_window_close
    assert settings_file.pending_path().is_file()


@pytest.mark.gui
def test_an_export_holds_a_slider_change_still_on_its_apply_timer(tmp_path, monkeypatch, qt_dialog) -> None:
    _save_to(monkeypatch, tmp_path / "mine")
    qt_dialog.pan_speed_slider.setValue(1500)
    assert qt_dialog._pan_speed_apply_timer.isActive()
    qt_dialog.settings_export_button.click()
    doc = yaml.safe_load((tmp_path / f"mine{settings_file.FILE_SUFFIX}").read_text())
    assert doc["settings"]["pan_speed"] == 1500
    assert not set(doc["settings"]) & set(settings_file.LOCAL_KEYS)


@pytest.mark.gui
@pytest.mark.parametrize("replace", [False, True])
def test_export_asks_before_replacing_the_suffixed_file(tmp_path, monkeypatch, qt_dialog, replace) -> None:
    from PyQt5.QtWidgets import QMessageBox

    out = tmp_path / f"mine{settings_file.FILE_SUFFIX}"
    out.write_text("keep me\n")
    _save_to(monkeypatch, tmp_path / "mine")
    asked = []
    answer = QMessageBox.Yes if replace else QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(a) or answer))
    qt_dialog.settings_export_button.click()
    assert len(asked) == 1 and str(out) in asked[0][2]
    if replace:
        assert yaml.safe_load(out.read_text())["format"] == settings_file.FORMAT_ID
    else:
        assert out.read_text() == "keep me\n"


@pytest.mark.gui
@pytest.mark.parametrize(
    ("subscribed", "config_text", "title"),
    [(True, None, "Export blocked"), (False, "theme: [unterminated\n", "Settings unreadable")],
)
def test_a_blocked_or_unreadable_export_is_reported_and_writes_nothing(
    tmp_path, monkeypatch, qt_dialog, subscribed, config_text, title
) -> None:
    from PyQt5.QtWidgets import QMessageBox

    folder = tmp_path / "mods" / "subscribed" if subscribed else tmp_path / "out"
    folder.mkdir(parents=True)
    if config_text is not None:
        _config().write_text(config_text)
    _save_to(monkeypatch, folder / f"s{settings_file.FILE_SUFFIX}")
    reported = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: reported.append(a)))
    qt_dialog.settings_export_button.click()
    assert len(reported) == 1 and reported[0][1] == title
    assert list(folder.iterdir()) == []
