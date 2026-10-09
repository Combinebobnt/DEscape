"""Qt-free theme files, Settings > Appearance > Export theme... / Import theme...

A theme file is YAML carrying the look-only Appearance settings: the chrome
theme (preset + per-role overrides), the UI font, the tool overlay colours,
the ruler and distance-tick label sizes, the stacked-unit badge position and
the grid's blend and thickness. Machine, performance and feel settings
(graphics quality, preloading, elevation height, pan speed) never travel.

Import validates rather than trusts. A file that isn't a theme file at all is
refused whole; anything else is checked key by key, and a bad key is reported
and skipped, never clamped or guessed at. A partial file changes only the keys
it carries.
"""

from __future__ import annotations

import os
import reprlib
import tempfile
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from descape import grid_overlay, settings, themes
from descape.scenario_io import FORBIDDEN_WRITE_MARKER, is_under_compatdata

FORMAT_ID = "descape-theme"
FORMAT_VERSION = 1
FILE_SUFFIX = ".descape-theme.yaml"
MAX_FILE_BYTES = 256 * 1024

# Range-checked, never clamped: an out-of-range value is a problem, not a fix-up.
_INT_RANGES: dict[str, tuple[int, int]] = {
    "ruler_label_font_px": (settings.RULER_LABEL_FONT_PX_MIN, settings.RULER_LABEL_FONT_PX_MAX),
    "distance_tick_font_px": (settings.DISTANCE_TICK_FONT_PX_MIN, settings.DISTANCE_TICK_FONT_PX_MAX),
    "grid_blend": (grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX),
}
_STACK_BADGE_IDS = tuple(pid for pid, _label in settings.STACK_BADGE_POSITIONS)
_OVERLAY_IDS = tuple(cid for cid, _label, _default in settings.OVERLAY_COLORS)


class ThemeWriteBlockedError(ValueError):
    """The export target is under a Proton compatdata/ prefix; nothing was written."""


class ThemeFileRefused(ValueError):
    """The file is not a theme file this version can read; nothing applies."""


class _AliasRefused(Exception):
    pass


class _NoAliasLoader(yaml.SafeLoader):
    """SafeLoader that refuses aliases, so a tiny alias bomb can't expand into
    a huge document (or a huge problem message)."""

    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise _AliasRefused()
        return super().compose_node(parent, index)


class _NoAliasDumper(yaml.SafeDumper):
    """Export never emits anchors, so import's alias refusal can't reject our own file."""

    def ignore_aliases(self, data) -> bool:
        return True


# Caps every echoed value in a problem or note, whatever the file holds.
_SHORT = reprlib.Repr()
_SHORT.maxstring = 60
_SHORT.maxother = 60
_SHORT.maxlevel = 2
_SHORT_MAX = 80


def _short(value) -> str:
    # reprlib never builds the full repr; the slice bounds its per-container fan-out.
    text = _SHORT.repr(value)
    return text if len(text) <= _SHORT_MAX else text[: _SHORT_MAX - 3] + "..."


@dataclass
class ImportPlan:
    """`values` is what apply_appearance_values() takes: only keys that passed
    validation, already normalized. `problems` were skipped; `notes` apply
    anyway but deserve a mention."""

    values: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def build_document() -> dict:
    """The current look settings, read through the settings getters."""
    return {
        "format": FORMAT_ID,
        "version": FORMAT_VERSION,
        "theme": {"preset": settings.get_theme(), "colors": settings.get_theme_colors()},
        "ui_font": {"family": settings.get_ui_font_family(), "size": settings.get_ui_font_size()},
        "overlay_colors": {cid: settings.get_overlay_color(cid) for cid in _OVERLAY_IDS},
        "ruler_label_font_px": settings.get_ruler_label_font_px(),
        "distance_tick_font_px": settings.get_distance_tick_font_px(),
        "stack_badge_position": settings.get_stack_badge_position(),
        "grid_blend": settings.get_grid_blend(),
        "grid_thickness": settings.get_grid_thickness(),
    }


def write_theme_file(path: Path | str, doc: dict) -> None:
    """Refuses a compatdata path, as written or resolved, and writes nothing.
    Otherwise writes through a temp file in the target directory and
    os.replace(), so an existing file is never truncated in place."""
    path = Path(path)
    if is_under_compatdata(path):
        raise ThemeWriteBlockedError(f"refusing to write a theme under a Proton '{FORBIDDEN_WRITE_MARKER}' folder: {path}")
    text = yaml.dump(doc, Dumper=_NoAliasDumper, default_flow_style=False, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        # mkstemp makes the file 0o600; a theme is meant to be shared.
        umask = os.umask(0)
        os.umask(umask)
        tmp.chmod(0o644 & ~umask)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_root(path: Path) -> dict:
    """The parsed mapping, or ThemeFileRefused with the one reason why not."""
    try:
        size = path.stat().st_size
    except OSError as e:
        raise ThemeFileRefused(f"cannot read the file: {e}") from e
    if size > MAX_FILE_BYTES:
        raise ThemeFileRefused(f"the file is {size} bytes, over the {MAX_FILE_BYTES}-byte limit for a theme file")
    try:
        root = yaml.load(path.read_text(encoding="utf-8"), Loader=_NoAliasLoader)  # noqa: S506 (a SafeLoader)
    except (OSError, UnicodeDecodeError) as e:
        raise ThemeFileRefused(f"cannot read the file: {e}") from e
    except yaml.YAMLError as e:
        raise ThemeFileRefused(f"not valid YAML: {e}") from e
    except _AliasRefused as e:
        raise ThemeFileRefused("not a DEscape theme file (it uses YAML anchors/aliases, which a theme file never has)") from e
    except RecursionError as e:
        raise ThemeFileRefused("not a DEscape theme file (nested too deeply)") from e
    except ValueError as e:  # e.g. an integer over Python's 4300-digit conversion limit
        raise ThemeFileRefused(f"not valid YAML: {_short(str(e))}") from e
    if not isinstance(root, dict):
        raise ThemeFileRefused("not a DEscape theme file (the top level is not a mapping)")
    if root.get("format") != FORMAT_ID:
        raise ThemeFileRefused(f"not a DEscape theme file (format is {_short(root.get('format'))}, expected {FORMAT_ID!r})")
    version = root.get("version")
    if not _is_int(version) or version < 1:
        raise ThemeFileRefused(f"unreadable theme file version {_short(version)}")
    if version > FORMAT_VERSION:
        raise ThemeFileRefused(f"theme file version {_short(version)} is newer than this DEscape reads ({FORMAT_VERSION})")
    return root


def _parse_colors(raw, known: Collection[str], what: str, plan: ImportPlan) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        plan.problems.append(f"{what}: expected a mapping of id to '#rrggbb', got {type(raw).__name__}")
        return None
    colors: dict[str, str] = {}
    for key, value in raw.items():
        if key not in known:
            plan.problems.append(f"{what}: unknown id {_short(key)}")
            continue
        try:
            colors[key] = themes.normalize_hex(value)
        except ValueError:
            plan.problems.append(f"{what}.{key}: {_short(value)} is not a '#rrggbb' colour")
    return colors


def _parse_theme(raw, plan: ImportPlan) -> None:
    if not isinstance(raw, dict):
        plan.problems.append(f"theme: expected a mapping, got {type(raw).__name__}; theme skipped")
        return
    for key in raw:
        if key not in ("preset", "colors"):
            plan.problems.append(f"theme: unknown key {_short(key)}")
    preset = raw.get("preset")
    if not isinstance(preset, str) or preset not in themes.PRESETS:
        plan.problems.append(f"theme.preset: unknown preset {_short(preset)}; theme skipped")
        return
    colors = _parse_colors(raw.get("colors", {}), themes.ROLE_IDS, "theme.colors", plan)
    plan.values["theme"] = {"preset": preset, "colors": colors or {}}


def _parse_ui_font(raw, plan: ImportPlan, installed_families: Collection[str] | None) -> None:
    if not isinstance(raw, dict):
        plan.problems.append(f"ui_font: expected a mapping, got {type(raw).__name__}")
        return
    for key in raw:
        if key not in ("family", "size"):
            plan.problems.append(f"ui_font: unknown key {_short(key)}")
    if "family" in raw:
        family = raw["family"]
        if not isinstance(family, str):
            plan.problems.append(f"ui_font.family: expected text, got {type(family).__name__}")
        else:
            plan.values["ui_font_family"] = family
            if family and installed_families is not None and family not in installed_families:
                plan.notes.append(f"UI font {_short(family)} is not installed here; the platform default is used until it is")
    if "size" in raw:
        size = raw["size"]
        lo, hi = settings.UI_FONT_SIZE_MIN, settings.UI_FONT_SIZE_MAX
        if size is None or (_is_int(size) and lo <= size <= hi):
            plan.values["ui_font_size"] = size
        elif _is_int(size):
            plan.problems.append(f"ui_font.size: {_short(size)} is outside {lo}-{hi}")
        else:
            plan.problems.append(f"ui_font.size: expected a whole number or null, got {type(size).__name__}")


def _parse_int(key: str, value, plan: ImportPlan) -> None:
    if not _is_int(value):
        plan.problems.append(f"{key}: expected a whole number, got {type(value).__name__}")
        return
    if key == "grid_thickness":
        if value not in grid_overlay.THICKNESS_STOPS:
            stops = ", ".join(str(s) for s in grid_overlay.THICKNESS_STOPS)
            plan.problems.append(f"grid_thickness: {_short(value)} is not one of {stops}")
            return
    else:
        lo, hi = _INT_RANGES[key]
        if not lo <= value <= hi:
            plan.problems.append(f"{key}: {_short(value)} is outside {lo}-{hi}")
            return
    plan.values[key] = value


def parse_theme_file(path: Path | str, installed_families: Collection[str] | None = None) -> ImportPlan:
    """Raises ThemeFileRefused for a file that can't be a theme file at all.
    `installed_families`, when given, turns an uninstalled font family into a
    note; the family is still applied."""
    root = _read_root(Path(path))
    plan = ImportPlan()
    for key, raw in root.items():
        if key in ("format", "version"):
            continue
        if key == "theme":
            _parse_theme(raw, plan)
        elif key == "ui_font":
            _parse_ui_font(raw, plan, installed_families)
        elif key == "overlay_colors":
            colors = _parse_colors(raw, _OVERLAY_IDS, "overlay_colors", plan)
            if colors:
                plan.values["overlay_colors"] = colors
        elif key == "stack_badge_position":
            if isinstance(raw, str) and raw in _STACK_BADGE_IDS:
                plan.values[key] = raw
            else:
                plan.problems.append(f"stack_badge_position: {_short(raw)} is not one of {', '.join(_STACK_BADGE_IDS)}")
        elif key in ("ruler_label_font_px", "distance_tick_font_px", "grid_blend", "grid_thickness"):
            _parse_int(key, raw, plan)
        else:
            plan.problems.append(f"unknown key {_short(key)}")
    return plan


def describe_changes(values: dict) -> list[str]:
    """One line per setting the plan would actually change, against the
    current settings."""
    lines: list[str] = []
    theme = values.get("theme")
    if theme is not None:
        current_preset, current_colors = settings.get_theme(), settings.get_theme_colors()
        if theme["preset"] != current_preset:
            lines.append(f"Theme: {themes.preset_label(current_preset)} -> {themes.preset_label(theme['preset'])}")
        if theme["colors"] != current_colors:
            shown = ", ".join(f"{themes.role_label(r)} {h}" for r, h in theme["colors"].items()) or "none"
            lines.append(f"Chrome colour overrides: {shown}")
    if "ui_font_family" in values and values["ui_font_family"] != settings.get_ui_font_family():
        lines.append(f"UI font: {values['ui_font_family'] or 'platform default'}")
    if "ui_font_size" in values and values["ui_font_size"] != settings.get_ui_font_size():
        size = values["ui_font_size"]
        lines.append(f"UI font size: {f'{size} pt' if size is not None else 'platform default'}")
    for color_id, hex_str in values.get("overlay_colors", {}).items():
        current = settings.get_overlay_color(color_id)
        if hex_str != current:
            lines.append(f"Overlay colour {settings.get_overlay_color_label(color_id)}: {current} -> {hex_str}")
    getters = {
        "ruler_label_font_px": ("Ruler label size", settings.get_ruler_label_font_px, " px"),
        "distance_tick_font_px": ("Distance ticks label size", settings.get_distance_tick_font_px, " px"),
        "stack_badge_position": ("Stacked-unit badge position", settings.get_stack_badge_position, ""),
        "grid_blend": ("Grid line blend", settings.get_grid_blend, ""),
        "grid_thickness": ("Grid thickness", settings.get_grid_thickness, " px"),
    }
    for key, (label, getter, unit) in getters.items():
        if key in values and values[key] != getter():
            lines.append(f"{label}: {getter()}{unit} -> {values[key]}{unit}")
    return lines


def format_import_summary(plan: ImportPlan) -> str:
    """The import dialog's body: what changes, then what was skipped and why."""
    changes = describe_changes(plan.values)
    parts = ["Will change:", *(f"  {line}" for line in changes)] if changes else ["Nothing in this file differs from the current settings."]
    if plan.problems:
        parts += ["", "Skipped (not applied):", *(f"  {p}" for p in plan.problems)]
    if plan.notes:
        parts += ["", "Notes:", *(f"  {n}" for n in plan.notes)]
    return "\n".join(parts)
