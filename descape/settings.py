"""General app settings -- not game-resource loading (see asset_source.py for
that), but still persisted to the same config.yaml (OS-standard per-user
location, not repo-relative), under their own keys.

Simpler than asset_source.py's install-path handling on purpose: no env-var
tier, no cross-process override concept -- just an in-memory value loaded
once and written straight through to disk on change.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from descape import asset_source, edge_ticks, iso_geometry
from descape.asset_source import CONFIG_PATH

_zoom_centered_on_cursor: bool | None = None


def _load_config() -> dict:
    if not CONFIG_PATH.is_file():
        return {}
    try:
        return yaml.safe_load(CONFIG_PATH.read_text()) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _save_config(config: dict) -> None:
    # Looks up this module's own CONFIG_PATH at call time, not import time,
    # so a test's monkeypatch of settings.CONFIG_PATH (constraint: it's a
    # bind-by-value import, so there are two module attributes to patch)
    # still takes effect.
    asset_source.write_config_file(CONFIG_PATH, config)


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
    _save_config(config)


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
    _save_config(config)


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
    _save_config(config)


# View > Distance Ticks: the ruler strip of tick marks drawn in the void
# just outside the map's own border, with two persisted halves (whether it
# is drawn at all, and how many tiles apart the minor ticks sit). Unlike
# the unit filter and Show sprites (deliberately per-session), this one is
# persisted because it is passive chrome that never changes what the map
# itself shows.
_distance_ticks: bool | None = None


def get_distance_ticks() -> bool:
    """Whether the map-edge distance ruler is drawn. Off by default, so a
    first run looks exactly as it did before the feature existed."""
    global _distance_ticks
    if _distance_ticks is None:
        _distance_ticks = bool(_load_config().get("distance_ticks", False))
    return _distance_ticks


def set_distance_ticks(enabled: bool) -> None:
    global _distance_ticks
    _distance_ticks = enabled
    config = _load_config()
    config["distance_ticks"] = enabled
    _save_config(config)


_distance_tick_interval: int | None = None


def get_distance_tick_interval() -> int:
    """Tiles between adjacent minor ticks, always one of
    edge_ticks.TICK_INTERVALS. Gates on membership rather than clamping into
    a range, the same shape get_graphics_quality uses: an off-list value is
    nonsense rather than a near miss, so it falls back to the default."""
    global _distance_tick_interval
    if _distance_tick_interval is None:
        raw = _load_config().get("distance_tick_interval")
        _distance_tick_interval = (
            raw if raw in edge_ticks.TICK_INTERVALS else edge_ticks.TICK_INTERVAL_DEFAULT
        )
    return _distance_tick_interval


def set_distance_tick_interval(value: int) -> None:
    if value not in edge_ticks.TICK_INTERVALS:
        raise ValueError(
            f"distance_tick_interval must be one of {list(edge_ticks.TICK_INTERVALS)}, got {value!r}"
        )
    global _distance_tick_interval
    _distance_tick_interval = value
    config = _load_config()
    config["distance_tick_interval"] = value
    _save_config(config)


# Stepped rendering mode's elev_step, as a percent of half_h -- see
# iso_geometry.canvas_size_and_origin's elev_step_pct param and
# ELEV_STEP_DEFAULT_PCT's own comment for why headroom above the default is
# offered here (unlike the old divisor-based control this replaces).
ELEV_STEP_PCT_MIN = 25
ELEV_STEP_PCT_MAX = 200

# The control is a stop space, not a continuous range: every off-stop value
# enumerates a shallower mip ladder than the nearest stop would, so the
# slider's value space IS the stop index and every entry point snaps.
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
    _save_config(config)


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
    _save_config(config)


# Left info/trigger panel vs. map view. The first thing besides the main window
# itself to persist geometry -- the trigger browser needs far more width than
# the info panel ever did, and a draggable split beats guessing a number that
# every later panel would have to be re-guessed for.
MIN_SPLIT_PANE = 120

_split_sizes: tuple[int, int] | None = None


def get_split_sizes() -> tuple[int, int] | None:
    """Last persisted (left, right) content-splitter widths, or None if never
    saved or the stored value is missing/malformed. None means "let Qt size it
    from the layout", which is what a first run wants."""
    global _split_sizes
    if _split_sizes is None:
        raw = _load_config().get("split_sizes")
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            try:
                left, right = int(raw[0]), int(raw[1])
                if left >= MIN_SPLIT_PANE and right >= MIN_SPLIT_PANE:
                    _split_sizes = (left, right)
            except (TypeError, ValueError):
                pass
    return _split_sizes


def set_split_sizes(left: int, right: int) -> None:
    global _split_sizes
    if left < MIN_SPLIT_PANE or right < MIN_SPLIT_PANE:
        return
    _split_sizes = (left, right)
    config = _load_config()
    config["split_sizes"] = [left, right]
    _save_config(config)


# Map view vs. system log, inside the right pane. Their own constants rather
# than reusing MIN_SPLIT_PANE, which is a width -- one name for both axes
# would lie.
MIN_LOG_PANE = 48
MIN_MAP_PANE = 120

_log_height: int | None = None


def get_log_height() -> int | None:
    """Last persisted system-log height in pixels, or None if never saved or
    the stored value is missing/malformed. None means "use the built-in
    four-line default", which is what a first run wants."""
    global _log_height
    if _log_height is None:
        raw = _load_config().get("log_height")
        try:
            value = int(raw)
        except (TypeError, ValueError):
            pass
        else:
            if value >= MIN_LOG_PANE:
                _log_height = value
    return _log_height


def set_log_height(height: int) -> None:
    global _log_height
    if height < MIN_LOG_PANE:
        return
    _log_height = height
    config = _load_config()
    config["log_height"] = height
    _save_config(config)


@dataclass(frozen=True)
class ToolDef:
    """One entry per toolbar Tool (Pan, Draw, Elevate, Set Elevation, and
    whatever comes next) -- the single place a new tool gets registered.
    REBINDABLE_ACTIONS below (and so the Keybinds settings tab) is generated
    from this list, not hand-duplicated, specifically so a new tool can't be
    added here without also getting a keybind row -- and viewer_common.py's
    per-tool dicts (EDIT_TOOLS/_TOOL_LABELS/_STROKE_LABELS) are generated
    from it too, so those can't silently drift out of sync with this list
    either. Adding a tool still needs its own QAction/toolbar wiring in
    viewer.py (that part is inherently bespoke -- each tool's enable
    condition differs), but ViewerWindow._build_keybind_actions() looks up
    that QAction by `f"{tool_id}_action"` via getattr with no default, so a
    tool listed here without that wiring fails loudly at construction
    (AttributeError) instead of silently missing its shortcut."""

    tool_id: str  # matches self.<tool_id>_action in viewer.py, e.g. "draw"
    label: str  # display label, e.g. "Draw", "Elevate"
    is_edit_tool: bool = True  # False for "pan" -- doesn't mutate scenario data, no undo record
    stroke_label: str = ""  # EditHistory record label; only meaningful when is_edit_tool
    default_key: str = ""  # keybind default -- "" (unbound until set) is a fine default
    # True for one-shot click tools (Paint Can) that must not run the
    # drag-stroke path -- one edit per press, never re-fired per
    # drag-entered tile. Routed by viewer_common.py's CLICK_TOOLS and
    # MapView.on_click_edit instead of the on_stroke_start/tile/end trio.
    click_only: bool = False
    # Toolbar param widget this tool reads, if any: "" | "terrain" | "level".
    # Drives which of the two tool-param widgets viewer.py shows/hides for
    # the active tool -- see _TOOL_PARAM in viewer_common.py.
    param_widget: str = ""
    # Whether this tool's stroke applies across a brush footprint (size +
    # shape) instead of always exactly one tile. A separate bool rather than
    # folding into param_widget: brush is orthogonal to a tool's "primary
    # value" param -- Draw wants terrain type AND brush, Set Elevation
    # wants level AND brush, Elevate wants brush with no param_widget at all
    # -- so param_widget's existing single-valued "" | "terrain" | "level"
    # semantics stay exactly as they are. Paint Can is click_only and never
    # sets this: one flood fill per click has no brush to speak of.
    supports_brush: bool = False
    # Which mode(s) this tool's toolbar button shows in; empty means every
    # mode. Drives viewer_common.tool_applicable() and
    # ViewerWindow._update_tool_enabled()'s per-mode visibility loop --
    # mode-inapplicable tools hide outright rather than just greying out.
    # Deliberately separate from enablement (has_map / write_ok /
    # elevation_ok, still driving setEnabled): a Terrain tool on a
    # non-square map should stay greyed and visible, not vanish.
    modes: tuple[str, ...] = ()


TOOLS: list[ToolDef] = [
    ToolDef("pan", "Pan", is_edit_tool=False, default_key="M"),
    ToolDef(
        "draw", "Draw", stroke_label="Paint terrain", default_key="D",
        param_widget="terrain", supports_brush=True, modes=("terrain",),
    ),
    ToolDef(
        "fill", "Paint Can", stroke_label="Fill terrain", default_key="P",
        click_only=True, param_widget="terrain", modes=("terrain",),
    ),
    ToolDef("elevation", "Elevate", stroke_label="Elevate", default_key="E", supports_brush=True, modes=("terrain",)),
    ToolDef(
        "set_level", "Set Elevation", stroke_label="Set elevation", default_key="L",
        param_widget="level", supports_brush=True, modes=("terrain",),
    ),
    # Measures, never mutates, so is_edit_tool=False puts it alongside Pan
    # rather than the edit tools. It took "R" from Elevate, which moved to the
    # "E" freed by the old mode_edit -> mode_terrain rename; see
    # _migrate_elevate_off_r for what that costs an existing config.
    ToolDef("ruler", "Ruler", is_edit_tool=False, default_key="R"),
    # Phase 3.5b's b1.4, Units-mode only. Originally gated only via setEnabled
    # (ViewerWindow._update_tool_enabled's own "unit_editable"), which greyed
    # it out but left it VISIBLE in every other mode -- switched to the same
    # modes=("units",) hide-not-grey treatment Convert already uses below,
    # per a later request that it hide outside Units mode like the four
    # Terrain tools already hide outside Terrain. Ships unbound: every letter
    # M/D/P/E/L/R plus every mode letter is already taken, and
    # view_distance_ticks already sets the precedent for a control shipping
    # with no default key rather than running a collision audit
    # (settings._migrate_elevate_off_r's own docstring spells out what a
    # shared QKeySequence does -- fires NEITHER action, silently).
    ToolDef(
        "place_unit", "Place Unit", stroke_label="Place unit", default_key="",
        click_only=True, param_widget="object", modes=("units",),
    ),
    # Phase 3.5b's b2.5 (D3): a brush, not a click-once tool, so it reuses
    # the generic stroke mechanism (begin/tile/end) every brush tool already
    # has -- one undo record per drag, mirroring Draw/Elevate. Units-mode
    # only, unbound by default for the same reason Place Unit is: every
    # short letter key is already spoken for.
    ToolDef(
        "convert", "Convert", stroke_label="Convert units", default_key="",
        param_widget="convert", supports_brush=True, modes=("units",),
    ),
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
    # The quick "default size" New action (File > New Map > <blank size>),
    # distinct from file_new above which is Custom size's Ctrl+N. Ships
    # unbound like view_distance_ticks below -- no default suggested by the
    # feature request that added this entry, just making it user-bindable.
    ("file_new_default", "New Map (Default Size)", ""),
    ("file_open", "Open Map", "Ctrl+O"),
    ("file_close", "Close Map", "Ctrl+W"),
    ("file_save", "Save", "Ctrl+S"),
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
    # an accepted tradeoff.
    ("edit_copy", "Copy Tile", "Ctrl+C"),
    ("edit_paste", "Paste Tile", "Ctrl+V"),
    ("edit_settings", "Settings…", ""),
    # Moved off Ctrl+I when mode_view claimed it below -- see that entry.
    ("view_isometric", "Isometric View", "Ctrl+Shift+I"),
    # Ships unbound, which needs no collision audit (a duplicate binding
    # silently kills both actions, an open item in TODO.md) and leaves the
    # obvious "R" mnemonic free for the separately backlogged Ruler tool.
    # Kept adjacent to view_isometric so _build_keybinds_tab does not emit a
    # second "View" header.
    ("view_distance_ticks", "Distance Ticks", ""),
    # Ships unbound, same reasoning as view_distance_ticks above.
    ("view_show_sprites", "Show Sprites", ""),
    ("help_about", "About", ""),
    ("help_debug_log", "Debug Log", ""),
    # Ships unbound, same reasoning as view_distance_ticks above.
    ("help_perf_trace", "Perf Trace", ""),
    # All eight mode_* entries use Ctrl+<letter> for a consistent mode-switch
    # group, distinct from every tool's bare-letter shortcut. mode_view takes
    # Ctrl+I (not the more obvious Ctrl+V, which edit_paste already owns) --
    # that in turn pushed view_isometric off Ctrl+I onto Ctrl+Shift+I above.
    # mode_terrain takes Ctrl+E rather than Ctrl+T since mode_triggers
    # already established Ctrl+T first. Kept contiguous because
    # _build_keybinds_tab only compares against the previous row, so a
    # non-contiguous prefix would emit a second "Modes" header.
    ("mode_view", "View Mode", "Ctrl+I"),
    ("mode_terrain", "Terrain Mode", "Ctrl+E"),
    ("mode_units", "Units Mode", "Ctrl+U"),
    ("mode_triggers", "Triggers Mode", "Ctrl+T"),
    ("mode_map_options", "Map Options Mode", "Ctrl+M"),
    ("mode_players", "Players Mode", "Ctrl+P"),
    ("mode_diplomacy", "Diplomacy Mode", "Ctrl+D"),
    # Ctrl+M is mode_map_options' -- same "M already taken" collision
    # view_isometric hit against mode_view above, resolved the same way
    # (Shift added, mnemonic letter kept) rather than picking an unrelated
    # free letter.
    ("mode_messages", "Messages Mode", "Ctrl+Shift+M"),
    # The Filters popup's toggles (_build_filters_button). All ship unbound,
    # same reasoning as view_distance_ticks above -- these are per-session,
    # not-persisted toggles, and no default was requested for them.
    ("filter_show_gaia", "Show GAIA", ""),
    ("filter_show_trees", "Show Trees", ""),
    ("filter_all_players", "All Players", ""),
    ("filter_no_players", "No Players", ""),
    ("filter_show_all", "Show All (Filters)", ""),
    ("filter_hide_all", "Hide All (Filters)", ""),
] + [
    # Per-mode player selection: sets the active mode's own player selector
    # (Units' place/convert owner, Players panel, Diplomacy panel) -- see
    # ViewerWindow._select_player(). range(9) rather than viewer.py's
    # MAX_PLAYER_ID on purpose: importing it would drag this Qt-free,
    # widely-imported config module onto unit_filter -> terrain_palette,
    # which parses two JSON files at import time. A test cross-checks the
    # count instead.
    (f"player_select_{pid}", f"Select {'GAIA' if pid == 0 else f'Player {pid}'}", str(pid))
    for pid in range(9)
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
_ACTION_LABELS: dict[str, str] = {action_id: label for action_id, label, _ in REBINDABLE_ACTIONS}

_keybinds: dict[str, str] | None = None  # None = not loaded from disk yet


def _migrate_elevate_off_r(keybinds: dict[str, str], persisted: dict) -> None:
    """Elevate's default moved from "R" to "E" when the Ruler tool claimed R.
    A config written before that change still pins tool_elevation: R, and
    config beats defaults, so without this both actions would end up sharing R.
    Two QActions on one window sharing a sequence make Qt treat every press as
    an ambiguous shortcut overload and fire NEITHER, with the only diagnostic
    on a stderr a GUI user never sees.

    Translates on read and never writes from a getter, the same shape
    get_graphics_quality's potato_mode migration uses: the corrected value
    persists on its own the next time set_keybind rewrites the whole dict.

    The "tool_ruler not in persisted" guard is what makes this one-shot, and
    it needs no schema-version key: a config written before the Ruler existed
    cannot name it, and set_keybind always writes every action, so any config
    written since does. Without that guard, a user who later assigned R to
    Elevate on purpose would have it silently undone on every single launch.

    Accepted limitation: someone who deliberately chose R for Elevate before
    the change is indistinguishable from someone who never touched it, and
    gets moved to E too."""
    if "tool_ruler" not in persisted and persisted.get("tool_elevation") == "R":
        keybinds["tool_elevation"] = "E"


def _reconcile_load_time_collisions(keybinds: dict[str, str], persisted: dict) -> None:
    """General counterpart to _migrate_elevate_off_r: a code update can add or
    move a DEFAULT keybind that collides with an unrelated action's persisted
    binding, and unlike an in-app assignment (set_keybind's own auto-clear +
    warn) there is no user action to catch this at -- it would otherwise be
    silent from the very next launch. Same Qt failure mode as everywhere else
    keybind collisions are discussed: two QActions sharing a sequence fire
    NEITHER, not "last one wins".

    Priority: a persisted value that differs from _DEFAULT_KEYBINDS wins over
    one that doesn't -- "config beats defaults", the same principle
    _migrate_elevate_off_r's own docstring names, generalized. Checked against
    _DEFAULT_KEYBINDS rather than mere presence in `persisted`: set_keybind()
    persists the WHOLE dict, so anyone who has ever touched a single keybind
    has every action pinned in their config, most of them equal to the
    default -- presence alone can't distinguish a deliberate choice from an
    incidental one.

    Iterates REBINDABLE_ACTIONS's declared order, both for which action
    claims a sequence first and as the tie-break when priority is equal (two
    customized values colliding, or two defaults colliding -- the latter
    should never happen for real, see test_default_keybinds_have_no_
    duplicate_sequences, but this stays deterministic either way).

    Silent, no warning surface -- matches _migrate_elevate_off_r's own
    precedent, and unlike set_keybind()'s assign-time case there is no
    Settings dialog open yet to show one in."""

    def is_customized(action_id: str) -> bool:
        return action_id in persisted and persisted[action_id] != _DEFAULT_KEYBINDS.get(action_id, "")

    claimed: dict[str, str] = {}  # sequence -> action_id currently holding it
    for action_id, _label, _default in REBINDABLE_ACTIONS:
        seq = keybinds.get(action_id, "")
        if not seq:
            continue
        holder = claimed.get(seq)
        if holder is None:
            claimed[seq] = action_id
            continue
        if is_customized(action_id) and not is_customized(holder):
            keybinds[holder] = ""
            claimed[seq] = action_id
        else:
            keybinds[action_id] = ""


def _load_keybinds() -> dict[str, str]:
    global _keybinds
    if _keybinds is None:
        persisted = _load_config().get("keybinds", {})
        _keybinds = dict(_DEFAULT_KEYBINDS)
        _keybinds.update(persisted)
        _migrate_elevate_off_r(_keybinds, persisted)
        _reconcile_load_time_collisions(_keybinds, persisted)
    return _keybinds


def get_keybind(action_id: str) -> str:
    """Current key sequence string for action_id (e.g. "Ctrl+P"), or "" if
    explicitly cleared. Not the same as "unset" -- every action in
    REBINDABLE_ACTIONS always has an entry, defaulted from
    _DEFAULT_KEYBINDS the first time this loads."""
    return _load_keybinds().get(action_id, "")


def get_default_keybind(action_id: str) -> str:
    return _DEFAULT_KEYBINDS.get(action_id, "")


def get_action_label(action_id: str) -> str:
    return _ACTION_LABELS.get(action_id, action_id)


def set_keybind(action_id: str, key_sequence: str) -> str | None:
    """Persists key_sequence for action_id. Two QActions sharing a sequence
    make Qt fire neither on press (see TODO.md's keybind-collision item), so
    if key_sequence is already bound to a different action, that other
    action is auto-cleared here rather than left to silently break both.
    Returns the auto-cleared action_id, or None if there was no collision.

    An empty key_sequence is exempt from the check in both directions --
    several actions ship unbound on purpose, so two actions both being ""
    is not a collision.
    """
    keybinds = _load_keybinds()
    cleared_action_id = None
    if key_sequence:
        for other_id, other_sequence in keybinds.items():
            if other_id != action_id and other_sequence == key_sequence:
                keybinds[other_id] = ""
                cleared_action_id = other_id
                break
    keybinds[action_id] = key_sequence
    config = _load_config()
    config["keybinds"] = keybinds
    _save_config(config)
    return cleared_action_id
