"""Qt-free settings files, Settings > General > Export settings... / Import settings... (GH #268).

A settings file carries config.yaml's portable keys (PORTABLE_KEYS) exactly as
config.yaml stores them; LOCAL_KEYS (paths, last-used state, this screen's
geometry) never travel. Every config.yaml key is in exactly one of the two
maps, which tests/test_settings_file_keys.py enforces against the code.

Import never applies live. It checks the file's structure, summarizes what
would change and stages the file beside config.yaml; viewer.main() applies it
at the next launch, before anything reads a setting (apply_pending_import()).
Applying replaces: every portable key the exporting build knew is dropped
first, so this machine matches the source, while a key the exporter did not
know keeps its local value. Values are not checked here: each getter falls
back per key or per id, as for a hand-edited config.yaml.
"""

from __future__ import annotations

import contextlib
import math
import os
import secrets
import stat
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from descape import __version__, asset_source, settings, theme_file, themes, tool_options, view_layers, yaml_file_io
from descape.yaml_file_io import short as _short

FORMAT_ID = "descape-settings"
FORMAT_VERSION = 1
FILE_SUFFIX = ".descape-settings.yaml"
MAX_FILE_BYTES = 512 * 1024
PENDING_FORMAT_ID = "descape-settings-pending"
PENDING_VERSION = 1
PENDING_NAME = "config.import-pending.yaml"
BAD_PENDING_NAME = "config.import-pending.bad.yaml"
BACKUP_NAME = "config.before-import.yaml"
# Far deeper than any setting's own shape (a tool option's list sits 3 deep).
MAX_DEPTH = 16

# Group labels: the Settings tab, menu or panel each key is set from.
_GENERAL = "General"
_MODS = "Mods"
_APPEARANCE = "Appearance"
_TOOLS = "Tools"
_SAVING = "Saving"
_KEYBINDS = "Keybinds"
_ADVANCED = "Advanced"
_VIEW = "View menu"
_TOOLBAR = "Toolbar"
_PANELS = "Panels"
_OTHER = "Other"

# key -> (group label, the label of the control that sets it). The change summary shows both, or
# the group alone where the label repeats it. Legacy keys travel too, so the importing build
# migrates them on read; each takes its successor's label.
PORTABLE_KEYS: dict[str, tuple[str, str]] = {
    "zoom_centered_on_cursor": (_GENERAL, "Mouse wheel zoom"),
    "mod_art_excluded": (_MODS, "Mod art left unticked"),
    "mod_art_included": (_MODS, "Mod art ticked while disabled in game"),
    "theme": (_APPEARANCE, "Theme"),
    "theme_colors": (_APPEARANCE, "Chrome colours"),
    "dark_mode": (_APPEARANCE, "Theme"),
    "ui_font_family": (_APPEARANCE, "UI font"),
    "ui_font_size": (_APPEARANCE, "UI font size"),
    "preload_zoom_levels": (_APPEARANCE, "Preload neighbouring zoom levels"),
    "graphics_quality": (_APPEARANCE, "Graphics quality"),
    "potato_mode": (_APPEARANCE, "Graphics quality"),
    "elev_step_pct": (_APPEARANCE, "Isometric elevation height"),
    "elev_step_divisor": (_APPEARANCE, "Isometric elevation height"),
    "pan_speed": (_APPEARANCE, "Pan speed"),
    "wall_opacity_pct": (_APPEARANCE, "Wall opacity"),
    "overlay_colors": (_APPEARANCE, "Tool overlay colors"),
    "overlay_opacity": (_APPEARANCE, "Tool overlay opacity"),
    "ruler_label_font_px": (_APPEARANCE, "Ruler label size"),
    "distance_tick_font_px": (_APPEARANCE, "Distance ticks label size"),
    "stack_badge_position": (_APPEARANCE, "Stacked-unit badge position"),
    "terrain_grid_swatch_px": (_APPEARANCE, "Terrain grid swatch size"),
    "grid_blend": (_APPEARANCE, "Grid line blend"),
    "grid_lightness": (_APPEARANCE, "Grid line blend"),
    "grid_thickness": (_APPEARANCE, "Grid thickness"),
    "toolbar_style": (_APPEARANCE, "Toolbar buttons"),
    "trigger_status_colors": (_APPEARANCE, "Trigger status colours"),
    "trigger_status_marker": (_APPEARANCE, "Trigger status"),
    "trigger_status_color_ok_rows": (_APPEARANCE, "Mark triggers that pass"),
    "trigger_status_color_sections": (_APPEARANCE, "Colour section rows"),
    "tool_options": (_TOOLS, _TOOLS),
    **{legacy_key: (_TOOLS, _TOOLS) for legacy_key, _name in settings._LEGACY_TOOL_OPTION_KEYS},
    "autosave_enabled": (_SAVING, "Autosave on a timer"),
    "autosave_interval_min": (_SAVING, "Autosave every"),
    "autosave_retention": (_SAVING, "Keep per document"),
    "autosave_location": (_SAVING, "Autosaves go"),
    "backups_enabled": (_SAVING, "Write .bak and .orig backups on save"),
    "keybinds": (_KEYBINDS, _KEYBINDS),
    "wheel_bindings": (_KEYBINDS, "Mouse Wheel"),
    "game_data": (_ADVANCED, "Game data"),
    "gaia_only_catalog_mode": (_ADVANCED, "GAIA-only objects"),
    "view_layers": (_VIEW, "Layers"),
    "distance_ticks": (_VIEW, "Distance Ticks > Show"),
    "distance_tick_interval": (_VIEW, "Distance Ticks > Every N tiles"),
    "stack_badges": (_VIEW, "Show Stacked-Unit Badges"),
    "selection_by_owner": (_VIEW, "Colour Selection by Owner"),
    "grid_overlay": (_VIEW, "Grid > Show"),
    "grid_follow_elevation": (_VIEW, "Grid > Follow Terrain Elevation"),
    "grid_major_every": (_VIEW, "Grid > Stronger line every N tiles"),
    "footprint_outlines": (_VIEW, "Footprint Outlines > Show"),
    "footprint_scope": (_VIEW, "Footprint Outlines > which units"),
    "footprint_merged": (_VIEW, "Footprint Outlines > Merge tiles"),
    "footprint_by_owner": (_VIEW, "Footprint Outlines > Colour by Owner"),
    "range_rings": (_VIEW, "Range Rings > Selected Buildings"),
    "range_rings_by_owner": (_VIEW, "Range Rings > Colour by Owner"),
    "range_rings_revealers": (_VIEW, "Range Rings > All Map Revealers"),
    "range_rings_buildings": (_VIEW, "Range Rings > All Buildings"),
    "range_rings_selected_revealers": (_VIEW, "Range Rings > Selected Map Revealers"),
    "range_rings_triggers": (_VIEW, "Range Rings > Trigger Changes"),
    "camera_markers": (_VIEW, "Show Player Cameras"),
    "trigger_overlay": (_VIEW, "Show Trigger Overlay"),
    "trigger_ghosts": (_VIEW, "Show Trigger-Created Units"),
    "terrain_style": (_TOOLBAR, "Elevation View"),
    "units_catalog_mode": (_PANELS, "Units catalog view"),
    "terrain_picker_mode": (_PANELS, "Terrain picker view"),
    "text_box_lines": (_PANELS, "Text box heights"),
    "button_grid_all_triggers": (_PANELS, "Show all triggers' changes"),
    "language": (_OTHER, "Language"),
}

# The staged import's id, written into config.yaml in the same write as its settings.
IMPORT_APPLIED_KEY = "import_applied"

# key -> why it stays on this machine.
LOCAL_KEYS: dict[str, str] = {
    "aoe2de_install": "the game install path",
    "aoe2de_profile": "the game profile path",
    "autosave_dir": "a folder path",
    "recent_files": "this machine's recently opened files",
    "new_map": "the New Map dialog's last choices",
    "data_mod_by_scenario": "keyed by this machine's scenario paths",
    "window_size": "window layout for this screen",
    "split_sizes": "window layout for this screen",
    "log_height": "window layout for this screen",
    "xwayland_on_wayland": "a Linux session workaround",
    IMPORT_APPLIED_KEY: "this machine's record of the last applied import",
}

# Legacy key -> (its successor key, the tool_options entry it fed or None). The getters read
# a legacy key only while its successor is absent, so an import drops the two together (D9).
LEGACY_SUCCESSORS: dict[str, tuple[str, str | None]] = {
    "potato_mode": ("graphics_quality", None),
    "dark_mode": ("theme", None),
    "grid_lightness": ("grid_blend", None),
    "elev_step_divisor": ("elev_step_pct", None),
    **{legacy_key: ("tool_options", name) for legacy_key, name in settings._LEGACY_TOOL_OPTION_KEYS},
}


class SettingsFileRefused(ValueError):
    """The file is not a settings file this version can read; nothing is staged."""


class SettingsWriteBlockedError(ValueError):
    """The export target is in a subscribed-mods (mods/subscribed/) folder, is
    one of config_files(), or cannot be resolved; nothing was written."""


class ConfigUnreadable(ValueError):
    """config.yaml exists but is not a YAML mapping. Export and apply refuse
    rather than write an empty file from it or replace it."""


class _PendingRefused(ValueError):
    pass


class _PendingNewer(_PendingRefused):
    pass


_FORMAT = yaml_file_io.YamlFormat(
    FORMAT_ID,
    FORMAT_VERSION,
    MAX_FILE_BYTES,
    "settings",
    SettingsFileRefused,
    other_formats={theme_file.FORMAT_ID: "It is a theme file: use Import theme… in Settings > Appearance."},
)
_PENDING_FORMAT = yaml_file_io.YamlFormat(
    PENDING_FORMAT_ID, PENDING_VERSION, MAX_FILE_BYTES, "staged settings", _PendingRefused, _PendingNewer
)
_HEADER_KEYS = ("format", "version", "exported_by", "known_keys", "settings")


@dataclass
class ImportPlan:
    """`values` is what an apply sets: the file's portable entries, raw.
    `known_keys` is the exporting build's portable keys that this build knows
    too. `problems` were skipped; `notes` deserve a mention."""

    values: dict = field(default_factory=dict)
    known_keys: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def pending_path() -> Path:
    return settings.CONFIG_PATH.parent / PENDING_NAME


def bad_pending_path() -> Path:
    return settings.CONFIG_PATH.parent / BAD_PENDING_NAME


def backup_path() -> Path:
    return settings.CONFIG_PATH.parent / BACKUP_NAME


def config_files() -> tuple[Path, ...]:
    """config.yaml and the files an import keeps beside it: no export may replace one."""
    return (settings.CONFIG_PATH, pending_path(), bad_pending_path(), backup_path())


def read_current_config() -> tuple[bytes | None, dict]:
    """config.yaml's bytes and its parsed mapping; (None, {}) when there is
    none and {} for an empty one. Unlike settings._load_config(), a file that
    does not parse, or parses to anything but a mapping, raises
    ConfigUnreadable. An OSError reading it propagates."""
    path = settings.CONFIG_PATH
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None, {}
    try:
        parsed = yaml.safe_load(asset_source.decode_config_bytes(data))
    except (yaml.YAMLError, ValueError, RecursionError) as e:
        raise ConfigUnreadable(f"{path} is not valid YAML: {_short(str(e))}") from e
    if parsed is None:
        return data, {}
    if not isinstance(parsed, dict):
        raise ConfigUnreadable(f"{path} does not hold a mapping of settings")
    return data, parsed


def build_document() -> dict:
    """The portable entries of config.yaml as stored, under the file header."""
    _data, config = read_current_config()
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "exported_by": f"DEscape {__version__}",
        "known_keys": list(PORTABLE_KEYS),
        "settings": {key: config[key] for key in PORTABLE_KEYS if key in config},
    }


def write_settings_file(path: Path | str, doc: dict) -> None:
    """Refuses a mods/subscribed/ path, as written or resolved, an
    unresolvable one, or one of config_files(), raising
    SettingsWriteBlockedError and writing nothing. Otherwise writes through a
    temp file and os.replace()."""
    yaml_file_io.write_export(path, doc, what="settings", blocked=SettingsWriteBlockedError, protected=config_files())


def _too_deep(value) -> bool:
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > MAX_DEPTH:
                return True
            stack.extend((child, depth + 1) for child in (item.values() if isinstance(item, dict) else item))
    return False


def _structure(root: dict, refused: type[Exception], noun: str) -> tuple[list[str], dict]:
    """The root's known_keys and settings, or `refused` when either has the wrong shape."""
    known, entries = root.get("known_keys"), root.get("settings")
    if not isinstance(known, list) or not all(isinstance(key, str) for key in known):
        raise refused(f"not a DEscape {noun} file (known_keys is not a list of setting names)")
    if not isinstance(entries, dict):
        raise refused(f"not a DEscape {noun} file (settings is not a mapping)")
    if _too_deep(entries):
        raise refused(f"not a DEscape {noun} file (nested too deeply)")
    return known, entries


def parse_settings_file(path: Path | str, installed_families: Collection[str] | None = None) -> ImportPlan:
    """Raises SettingsFileRefused for a file that can't be a settings file at
    all. An unknown setting is a problem, a local one a note; neither is kept.
    `installed_families`, when given, turns an uninstalled UI font into a note."""
    root = yaml_file_io.read_root(Path(path), _FORMAT)
    known, entries = _structure(root, SettingsFileRefused, "settings")
    plan = ImportPlan(known_keys=[key for key in known if key in PORTABLE_KEYS])
    for key in root:
        if key not in _HEADER_KEYS:
            plan.problems.append(f"unknown key {_short(key)} outside settings")
    for key, value in entries.items():
        if key in PORTABLE_KEYS:
            plan.values[key] = value
        elif key in LOCAL_KEYS:
            plan.notes.append(f"{key} stays on this machine ({LOCAL_KEYS[key]}), so it is not imported")
        else:
            plan.problems.append(f"unknown setting {_short(key)}")
    family = plan.values.get("ui_font_family")
    if isinstance(family, str) and family and installed_families is not None and family not in installed_families:
        plan.notes.append(f"UI font {_short(family)} is not installed here; the platform default is used until it is")
    return plan


def _same(a, b) -> bool:
    """Equal and of the same type throughout, so True never matches 1, and NaN matches NaN."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and math.isnan(a):
        return math.isnan(b)
    return a == b


def _target_config(current: dict, known_keys: Collection[str], values: dict) -> dict:
    """What config.yaml becomes: every key both builds know dropped, the legacy
    keys that follow a dropped or imported successor dropped too (D9), then the
    file's entries set. LOCAL_KEYS are never touched."""
    known = set(known_keys) & set(PORTABLE_KEYS)
    target = {key: value for key, value in current.items() if key not in known}
    for legacy_key, (successor, option) in LEGACY_SUCCESSORS.items():
        if successor in known:  # a newer exporter: the legacy key would shadow the successor's default
            target.pop(legacy_key, None)
        if legacy_key not in values:
            continue
        # An older exporter's legacy key: a local successor would shadow it.
        if option is None:
            target.pop(successor, None)
        elif isinstance(target.get(successor), dict) and option in target[successor]:
            entries = {name: v for name, v in target[successor].items() if name != option}
            if entries:
                target[successor] = entries
            else:
                del target[successor]
    target.update({key: value for key, value in values.items() if key in PORTABLE_KEYS})
    return target


_MISSING = object()


def _entry_ids() -> dict[str, tuple[dict, str, str]]:
    """Dict-valued keys -> (each known id's default, singular noun, plural noun).
    An empty default map means the key holds overrides for open-ended ids."""
    return {
        "keybinds": ({aid: d for aid, _label, d in settings.REBINDABLE_ACTIONS}, "binding", "bindings"),
        "wheel_bindings": (dict(settings.DEFAULT_WHEEL_BINDINGS), "wheel binding", "wheel bindings"),
        "tool_options": (tool_options.defaults(), "tool default", "tool defaults"),
        "overlay_colors": ({cid: d for cid, _label, d in settings.OVERLAY_COLORS}, "colour", "colours"),
        "view_layers": ({spec.layer_id: spec.default for spec in view_layers.LAYERS}, "layer", "layers"),
        "overlay_opacity": ({gid: d for gid, _label, d in settings.OVERLAY_OPACITIES}, "opacity", "opacities"),
        "trigger_status_colors": ({sid: d for sid, _label, d in settings.TRIGGER_STATUS_COLORS}, "colour", "colours"),
        "theme_colors": (dict.fromkeys(themes.ROLE_IDS, _MISSING), "colour override", "colour overrides"),
        "text_box_lines": ({}, "text box height", "text box heights"),
    }


def _entries(config: dict, key: str) -> dict:
    """A dict-valued key's stored entries as its getter reads them: renamed
    keybinds carried over, legacy tool options folded in."""
    stored = config.get(key)
    entries = dict(stored) if isinstance(stored, dict) else {}
    if key == "keybinds":
        for old, new in settings._RENAMED_KEYBINDS.items():
            if old in entries and new not in entries:
                entries[new] = entries[old]
    if key == "tool_options":
        for legacy_key, name in settings._LEGACY_TOOL_OPTION_KEYS:
            if name not in entries and legacy_key in config:
                entries[name] = bool(config[legacy_key])
    return entries


def _present(config: dict, key: str) -> bool:
    return key in config or any(successor == key and legacy in config for legacy, (successor, _o) in LEGACY_SUCCESSORS.items())


def _scalar(config: dict, key: str):
    """A scalar key's stored value, or its legacy key's as "<value> (as <legacy>)", or _MISSING."""
    if key in config:
        return config[key]
    for legacy_key, (successor, option) in LEGACY_SUCCESSORS.items():
        if successor == key and option is None and legacy_key in config:
            return (legacy_key, config[legacy_key])
    return _MISSING


def _show(value) -> str:
    if value is _MISSING:
        return "default"
    if isinstance(value, tuple):
        legacy_key, raw = value
        return f"{_short(raw)} (as {legacy_key})"
    return _short(value)


def describe_changes(plan: ImportPlan) -> list[str]:
    """What applying `plan` would change against config.yaml as it is now:
    the changed settings, then the ones reset to default, one line each under
    their group and control label. A dict-valued setting is compared entry by
    entry with both sides filled in from its defaults, and shows as a count. A
    legacy key reads as its successor's change. Raises ConfigUnreadable like
    apply would."""
    _data, current = read_current_config()
    target = _target_config(current, plan.known_keys, plan.values)
    entry_ids = _entry_ids()
    changed: list[str] = []
    reset: list[str] = []
    for key, (group, label) in PORTABLE_KEYS.items():
        if key in LEGACY_SUCCESSORS:
            continue  # shown as its successor's change
        where = group if label == group else f"{group}: {label}"
        if key in entry_ids:
            defaults, one, many = entry_ids[key]
            old, new = {**defaults, **_entries(current, key)}, {**defaults, **_entries(target, key)}
            ids = defaults if defaults else {k for k in (*old, *new) if isinstance(k, str)}
            count = sum(not _same(old.get(i, _MISSING), new.get(i, _MISSING)) for i in ids)
            if not count:
                continue
            line = f"{where}: {count} {one} differs" if count == 1 else f"{where}: {count} {many} differ"
        else:
            old, new = _scalar(current, key), _scalar(target, key)
            if old is new or (old is not _MISSING and new is not _MISSING and _same(old, new)):
                continue
            line = f"{where}: {_show(old)} -> {_show(new)}"
        (reset if _present(current, key) and not _present(target, key) else changed).append(line)
    if not changed and not reset:
        return ["Nothing in this file differs from the current settings."]
    lines = ["Will change:", *(f"  {line}" for line in changed)] if changed else []
    if reset:
        lines += [*([""] if lines else []), "Will reset to default:", *(f"  {line}" for line in reset)]
    return lines


def format_import_summary(plan: ImportPlan) -> str:
    """The import dialog's body: what changes, then what was skipped and why."""
    parts = describe_changes(plan)
    if plan.problems:
        parts += ["", "Skipped (not applied):", *(f"  {p}" for p in plan.problems)]
    if plan.notes:
        parts += ["", "Notes:", *(f"  {n}" for n in plan.notes)]
    parts += ["", "These apply the next time DEscape starts."]
    return "\n".join(parts)


def stage_import(plan: ImportPlan, source_path: Path | str) -> Path:
    """Writes the pending import beside config.yaml, for the next launch to
    apply. Its fresh `id` is what tells apply that config.yaml already holds it."""
    path = pending_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "format": PENDING_FORMAT_ID,
        "version": PENDING_VERSION,
        "mode": "replace",
        "id": secrets.token_hex(8),
        "source": str(source_path),
        "known_keys": list(plan.known_keys),
        "settings": dict(plan.values),
    }
    yaml_file_io.atomic_write(path, yaml_file_io.dump(doc))
    return path


@dataclass(frozen=True)
class PendingImport:
    """A staged import, its keys re-filtered to this build's portable keys.
    `source` and `import_id` are None when the file lacks them. `refusal`, when
    set, is why this build will not apply it; the other fields are then empty."""

    source: str | None = None
    import_id: str | None = None
    known_keys: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    refusal: str | None = None


def pending_import() -> PendingImport | None:
    """The staged import, None when none is, or when config.yaml already
    records its id: a failed unlink left it, and the next launch only removes
    it. A newer build's file stays staged; an unreadable one is moved aside
    at the next launch."""
    try:
        staged = _read_pending(pending_path())
    except _PendingNewer as e:
        return PendingImport(refusal=f"a newer DEscape staged it ({e})")
    except _PendingRefused as e:
        return PendingImport(refusal=f"it cannot be read ({e})")
    if staged is None or staged.import_id is None:
        return staged
    try:
        _data, current = read_current_config()
    except (ConfigUnreadable, OSError):
        return staged
    return None if _already_applied(staged, current) else staged


def _already_applied(staged: PendingImport, current: dict) -> bool:
    return staged.import_id is not None and current.get(IMPORT_APPLIED_KEY) == staged.import_id


def discard_pending_import() -> None:
    pending_path().unlink(missing_ok=True)


def _read_pending(path: Path) -> PendingImport | None:
    """The staged import, or None when the file is gone (another instance
    took it). Raises _PendingNewer for a file a newer build staged, which stays
    staged, and _PendingRefused for one no build can apply."""
    try:
        root = yaml_file_io.read_root(path, _PENDING_FORMAT)
    except _PendingRefused as e:
        if not isinstance(e, _PendingNewer) and isinstance(e.__cause__, FileNotFoundError):
            return None
        raise
    mode = root.get("mode")
    if not isinstance(mode, str):
        raise _PendingRefused(f"not a DEscape staged settings file (its mode {_short(mode)} is not text)")
    if mode != "replace":
        raise _PendingNewer(f"its mode {_short(mode)} is not one this DEscape applies")
    known, entries = _structure(root, _PendingRefused, "staged settings")
    source, import_id = root.get("source"), root.get("id")
    return PendingImport(
        source if isinstance(source, str) else None,
        import_id if isinstance(import_id, str) and import_id else None,
        [key for key in known if key in PORTABLE_KEYS],
        {key: value for key, value in entries.items() if key in PORTABLE_KEYS},
    )


def apply_pending_import() -> str | None:
    """Applies a staged import to config.yaml, before anything reads a
    setting: viewer.main() calls it at startup. None when nothing is staged,
    else a one-line status for the log. Never raises (D8): a failure before
    config.yaml is written leaves it untouched and moves the staged file to
    BAD_PENDING_NAME. The old config.yaml is copied to BACKUP_NAME first, and
    only when the import changes it, so a re-run never backs up an imported
    config (D7). The config write also records the import's id
    (IMPORT_APPLIED_KEY), so a pending file left behind by a failed unlink is
    only removed later, never re-applied over edits made since."""
    pending = pending_path()
    if not pending.exists():
        return None
    written = False
    source = "a settings file"
    try:
        try:
            staged = _read_pending(pending)
        except _PendingNewer as e:
            return f"Settings import not applied, left staged in {pending}: {e}"
        if staged is None:
            return None
        source = staged.source or source
        data, current = read_current_config()
        if _already_applied(staged, current):
            try:
                pending.unlink(missing_ok=True)
            except OSError as e:
                # Not the generic handler's "nothing changed": the import did apply, at an earlier launch.
                return f"Settings from {source} were already applied; {pending} could not be removed: {e}"
            return f"Settings from {source} were already applied."
        target = _target_config(current, staged.known_keys, staged.settings)
        changes = not _same(target, current)
        if staged.import_id is not None:
            target[IMPORT_APPLIED_KEY] = staged.import_id
        if not _same(target, current):
            config = settings.CONFIG_PATH.resolve()  # a symlinked config.yaml stays a symlink
            mode = stat.S_IMODE(config.stat().st_mode) if data is not None else None
            if changes and data is not None:
                yaml_file_io.atomic_write(backup_path(), data, mode=mode)
            config.parent.mkdir(parents=True, exist_ok=True)
            yaml_file_io.atomic_write(config, yaml.safe_dump(target, default_flow_style=False, sort_keys=False), mode=mode)
            written = True
        pending.unlink(missing_ok=True)
        if not changes:
            return f"Settings from {source} were already applied."
        backup = f"; the previous settings are in {backup_path()}" if data is not None else ""
        return f"Settings imported from {source}{backup}."
    except Exception as e:  # noqa: BLE001 -- a failed import must never stop DEscape starting (D8)
        if written:
            return f"Settings imported from {source}, but {pending} could not be removed: {e}"
        with contextlib.suppress(OSError):
            os.replace(pending, bad_pending_path())
        return f"Settings import failed and nothing changed ({e}); the staged file is now {bad_pending_path()}"
