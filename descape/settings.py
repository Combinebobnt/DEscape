"""General app settings -- not game-resource loading (see asset_source.py for
that), but still persisted to the same config.yaml, under their own keys.

Simpler than asset_source.py's install-path handling on purpose: no env-var
tier, no cross-process override concept -- just an in-memory value loaded
once and written straight through to disk on change.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from descape.asset_source import CONFIG_PATH
from descape import iso_geometry

_zoom_centered_on_cursor: bool | None = None


def _load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return {}


# Settings > Appearance's graphics-quality slider -- render.
# tile_pixels_for_map() scales its base tile_px by 2**(quality -
# GRAPHICS_QUALITY_DEFAULT), so each stage away from Default is a power-of-
# two step (halving/doubling), matching what the old boolean "potato mode"
# already did for its one stage (quality 2 here reproduces it exactly).
GRAPHICS_QUALITY_LABELS: dict[int, str] = {1: "Potatest", 2: "Potato", 3: "Default", 4: "Enhanced"}
GRAPHICS_QUALITY_DEFAULT = 3
GRAPHICS_QUALITY_MIN = min(GRAPHICS_QUALITY_LABELS)
GRAPHICS_QUALITY_MAX = max(GRAPHICS_QUALITY_LABELS)

_graphics_quality: int | None = None


def get_graphics_quality() -> int:
    """1-4 graphics quality stage (see GRAPHICS_QUALITY_LABELS), falling
    back to GRAPHICS_QUALITY_DEFAULT if never set or malformed. Migrates
    the legacy boolean "potato_mode" key on first read if "graphics_quality"
    itself isn't present in config.yaml (True -> 2/"Potato", the equivalent
    halved resolution; False/missing -> the default)."""
    global _graphics_quality
    if _graphics_quality is None:
        config = _load_config()
        raw = config.get("graphics_quality")
        if raw is None and "potato_mode" in config:
            raw = 2 if config.get("potato_mode") else GRAPHICS_QUALITY_DEFAULT
        _graphics_quality = raw if raw in GRAPHICS_QUALITY_LABELS else GRAPHICS_QUALITY_DEFAULT
    return _graphics_quality


def set_graphics_quality(value: int) -> None:
    if value not in GRAPHICS_QUALITY_LABELS:
        raise ValueError(f"graphics_quality must be one of {sorted(GRAPHICS_QUALITY_LABELS)}, got {value!r}")
    global _graphics_quality
    _graphics_quality = value
    config = _load_config()
    config["graphics_quality"] = value
    config.pop("potato_mode", None)  # fully migrated once set through the new control
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


_dark_mode: bool | None = None


def get_dark_mode() -> bool:
    """Whether the dark app-chrome theme is on -- see viewer.apply_theme()
    for what that does and doesn't affect. Off (light) by default."""
    global _dark_mode
    if _dark_mode is None:
        _dark_mode = bool(_load_config().get("dark_mode", False))
    return _dark_mode


def set_dark_mode(enabled: bool) -> None:
    global _dark_mode
    _dark_mode = enabled
    config = _load_config()
    config["dark_mode"] = enabled
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


def get_zoom_centered_on_cursor() -> bool:
    """Whether mouse-wheel zoom is anchored under the cursor (True, the
    default) or at the view's center (False). See
    MapView.set_zoom_anchor_mode for how this is applied."""
    global _zoom_centered_on_cursor
    if _zoom_centered_on_cursor is None:
        _zoom_centered_on_cursor = bool(_load_config().get("zoom_centered_on_cursor", True))
    return _zoom_centered_on_cursor


def set_zoom_centered_on_cursor(enabled: bool) -> None:
    global _zoom_centered_on_cursor
    _zoom_centered_on_cursor = enabled
    config = _load_config()
    config["zoom_centered_on_cursor"] = enabled
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


# Stepped rendering mode's elev_step, as a percent of half_h -- see
# iso_geometry.canvas_size_and_origin's elev_step_pct param and
# ELEV_STEP_DEFAULT_PCT's own comment for why headroom above the default is
# offered here (unlike the old divisor-based control this replaces).
ELEV_STEP_PCT_MIN = 25
ELEV_STEP_PCT_MAX = 200

# The control is a stop space, not a continuous range: every off-stop value
# enumerates a shallower mip ladder than the nearest stop would (see
# maintainer/docs/PLAN_MIPS.md), so the slider's value space IS the stop
# index and every entry point snaps.
ELEV_STEP_PCT_STEP = 25
ELEV_STEP_PCT_STOPS = tuple(range(ELEV_STEP_PCT_MIN, ELEV_STEP_PCT_MAX + 1, ELEV_STEP_PCT_STEP))

_elev_step_pct: int | None = None


def snap_elev_step_pct(value: int) -> int:
    """Nearest legal stop to `value`, clamped into range first. No tie-break
    rule is needed: a stop midpoint is never an integer at STEP=25."""
    value = max(ELEV_STEP_PCT_MIN, min(ELEV_STEP_PCT_MAX, value))
    return min(ELEV_STEP_PCT_STOPS, key=lambda stop: abs(stop - value))


def elev_step_index(pct: int) -> int:
    """1-based stop index for a pct, for driving the slider's value space.
    Snaps first, so an off-stop caller gets the nearest stop rather than a
    ValueError out of the lookup."""
    return ELEV_STEP_PCT_STOPS.index(snap_elev_step_pct(pct)) + 1


def elev_step_pct_for_index(index: int) -> int:
    """Inverse of elev_step_index() -- the pct a slider position means."""
    return ELEV_STEP_PCT_STOPS[index - 1]


def get_elev_step_pct() -> int:
    """Current Stepped-mode elev_step, as a percent of half_h, always one of
    ELEV_STEP_PCT_STOPS. Falls back to iso_geometry.ELEV_STEP_DEFAULT_PCT if
    never set, above ELEV_STEP_PCT_MAX, or malformed; an in-range off-stop
    value (including one written below today's MIN by an older build) snaps
    to its nearest stop instead. Migrates the legacy "elev_step_divisor" key
    (this control's previous divisor-based form) on first read if
    "elev_step_pct" itself isn't present."""
    global _elev_step_pct
    if _elev_step_pct is None:
        config = _load_config()
        raw = config.get("elev_step_pct")
        if raw is None:
            legacy_divisor = config.get("elev_step_divisor")
            if isinstance(legacy_divisor, int) and legacy_divisor > 0:
                raw = 100 // legacy_divisor
        # Gates on 1 <= raw <= MAX rather than snapping bare, so a nonsense
        # value still falls back to the default instead of snapping to MAX.
        if isinstance(raw, int) and 1 <= raw <= ELEV_STEP_PCT_MAX:
            _elev_step_pct = snap_elev_step_pct(raw)
        else:
            _elev_step_pct = iso_geometry.ELEV_STEP_DEFAULT_PCT
    return _elev_step_pct


def set_elev_step_pct(value: int) -> None:
    # Snaps rather than only clamping -- a programmatic caller would
    # otherwise persist an off-stop value and silently shrink the mip set.
    global _elev_step_pct
    value = snap_elev_step_pct(value)
    _elev_step_pct = value
    config = _load_config()
    config["elev_step_pct"] = value
    config.pop("elev_step_divisor", None)  # fully migrated once set through the new control
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720
# Also passed directly to ViewerWindow.setMinimumSize() -- one shared
# constant rather than two hardcoded 800/600 pairs that could drift apart.
MIN_WINDOW_WIDTH = 800
MIN_WINDOW_HEIGHT = 600

_window_size: tuple[int, int] | None = None


def get_window_size() -> tuple[int, int]:
    """Last persisted main-window (width, height), or the DEFAULT_WINDOW_*
    constants if never saved or the stored value is missing/malformed/below
    ViewerWindow's minimum size -- never hands back something that would
    open off-usable-size."""
    global _window_size
    if _window_size is None:
        raw = _load_config().get("window_size")
        width, height = DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                raw_width, raw_height = int(raw[0]), int(raw[1])
                if raw_width >= MIN_WINDOW_WIDTH and raw_height >= MIN_WINDOW_HEIGHT:
                    width, height = raw_width, raw_height
            except (TypeError, ValueError):
                pass
        _window_size = (width, height)
    return _window_size


def set_window_size(width: int, height: int) -> None:
    global _window_size
    _window_size = (width, height)
    config = _load_config()
    config["window_size"] = [width, height]
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))


@dataclass(frozen=True)
class ToolDef:
    """One entry per toolbar Tool (Pan, Terrain, Elevate, Set Elevation, and
    whatever comes next) -- the single place a new tool gets registered.
    REBINDABLE_ACTIONS below (and so the Keybinds settings tab) is generated
    from this list, not hand-duplicated, specifically so a new tool can't be
    added here without also getting a keybind row -- and viewer.py's
    per-tool dicts (EDIT_TOOLS/_TOOL_LABELS/_STROKE_LABELS) are generated
    from it too, so those can't silently drift out of sync with this list
    either. Adding a tool still needs its own QAction/toolbar wiring in
    viewer.py (that part is inherently bespoke -- each tool's enable
    condition differs), but ViewerWindow._build_keybind_actions() looks up
    that QAction by `f"{tool_id}_action"` via getattr with no default, so a
    tool listed here without that wiring fails loudly at construction
    (AttributeError) instead of silently missing its shortcut."""

    tool_id: str  # matches self.<tool_id>_action in viewer.py, e.g. "terrain"
    label: str  # display label, e.g. "Terrain", "Elevate"
    is_edit_tool: bool = True  # False for "pan" -- doesn't mutate scenario data, no undo record
    stroke_label: str = ""  # EditHistory record label; only meaningful when is_edit_tool
    default_key: str = ""  # keybind default -- "" (unbound until set) is a fine default
    # True for one-shot click tools (Paint Can) that must not run the
    # drag-stroke path -- one edit per press, never re-fired per
    # drag-entered tile. Routed by viewer.py's CLICK_TOOLS and
    # MapView.on_click_edit instead of the on_stroke_start/tile/end trio.
    click_only: bool = False
    # Toolbar param widget this tool reads, if any: "" | "terrain" | "level".
    # Drives which of the two tool-param widgets viewer.py shows/hides for
    # the active tool -- see _TOOL_PARAM there.
    param_widget: str = ""
    # Whether this tool's stroke applies across a brush footprint (size +
    # shape) instead of always exactly one tile. A separate bool rather than
    # folding into param_widget: brush is orthogonal to a tool's "primary
    # value" param -- Terrain wants terrain type AND brush, Set Elevation
    # wants level AND brush, Elevate wants brush with no param_widget at all
    # -- so param_widget's existing single-valued "" | "terrain" | "level"
    # semantics stay exactly as they are. Paint Can is click_only and never
    # sets this: one flood fill per click has no brush to speak of.
    supports_brush: bool = False


TOOLS: list[ToolDef] = [
    ToolDef("pan", "Pan", is_edit_tool=False, default_key="M"),
    ToolDef("terrain", "Terrain", stroke_label="Paint terrain", default_key="T", param_widget="terrain", supports_brush=True),
    ToolDef("fill", "Paint Can", stroke_label="Fill terrain", default_key="P", click_only=True, param_widget="terrain"),
    ToolDef("elevation", "Elevate", stroke_label="Elevate", default_key="R", supports_brush=True),
    ToolDef("set_level", "Set Elevation", stroke_label="Set elevation", default_key="L", param_widget="level", supports_brush=True),
]

# (action_id, display label, default key sequence string) -- the one source
# of truth for what's rebindable, consumed by both viewer.py (to build the
# actual QAction shortcuts) and the Keybinds settings tab (to build its
# rows). The tool_* entries are generated from TOOLS above, not hand-listed,
# so every tool automatically gets a row here. No Qt dependency here on
# purpose, matching this whole module's split from viewer.py -- QKeySequence
# strings are plain text (e.g. "Ctrl+P"), no need to import Qt just to store
# them.
REBINDABLE_ACTIONS: list[tuple[str, str, str]] = [
    ("file_new", "New Map", "Ctrl+N"),
    ("file_open", "Open Map", "Ctrl+O"),
    ("file_close", "Close Map", "Ctrl+W"),
    ("file_save_as", "Save As", "Ctrl+Shift+S"),
    ("file_exit", "Exit", "Ctrl+Q"),
    ("edit_undo", "Undo", "Ctrl+Z"),
    ("edit_redo", "Redo", "Ctrl+Y"),
    # v2.7 copy/paste: "Ctrl+C"/"Ctrl+V" rather than a bare
    # letter like every tool above. Every Edit-menu action in this list uses
    # a literal string default -- not a QKeySequence.<Standard> enum -- for
    # the reason given on ToolDef's default_key and the module docstring:
    # this module is deliberately Qt-free, and a first-run macOS user seeing
    # "Ctrl+N" instead of "Cmd+N" as the displayed (not matched) default is
    # an accepted tradeoff. No collision with mode_view's bare "V" above:
    # Qt treats "V" and "Ctrl+V" as distinct key sequences.
    ("edit_copy", "Copy Tile", "Ctrl+C"),
    ("edit_paste", "Paste Tile", "Ctrl+V"),
    ("edit_settings", "Settings…", ""),
    ("view_isometric", "Isometric View", "Ctrl+I"),
    ("help_about", "About", ""),
    ("help_debug_log", "Debug Log", ""),
    ("mode_view", "View Mode", "V"),
    ("mode_edit", "Edit Mode", "E"),
] + [(f"tool_{t.tool_id}", f"{t.label} Tool", t.default_key) for t in TOOLS] + [
    # The active tool's own "primary value" -- today that's only Set
    # Elevation's Level spinbox (elevation_level_spin), gated the same way
    # its toolbar spinbox is (viewer.py's _update_tool_enabled). "]"/"["
    # rather than a bare letter for the same reason edit_copy/paste picked
    # a non-letter binding: an established convention (brush-size style
    # inc/dec) to follow instead of inventing one.
    ("adjust_increment", "Increase Tool Value", "]"),
    ("adjust_decrement", "Decrease Tool Value", "["),
]
_DEFAULT_KEYBINDS: dict[str, str] = {action_id: default for action_id, _, default in REBINDABLE_ACTIONS}

_keybinds: dict[str, str] | None = None  # None = not loaded from disk yet


def _load_keybinds() -> dict[str, str]:
    global _keybinds
    if _keybinds is None:
        _keybinds = dict(_DEFAULT_KEYBINDS)
        _keybinds.update(_load_config().get("keybinds", {}))
    return _keybinds


def get_keybind(action_id: str) -> str:
    """Current key sequence string for action_id (e.g. "Ctrl+P"), or "" if
    explicitly cleared. Not the same as "unset" -- every action in
    REBINDABLE_ACTIONS always has an entry, defaulted from
    _DEFAULT_KEYBINDS the first time this loads."""
    return _load_keybinds().get(action_id, "")


def get_default_keybind(action_id: str) -> str:
    return _DEFAULT_KEYBINDS.get(action_id, "")


def set_keybind(action_id: str, key_sequence: str) -> None:
    keybinds = _load_keybinds()
    keybinds[action_id] = key_sequence
    config = _load_config()
    config["keybinds"] = keybinds
    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False))
