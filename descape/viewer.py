"""PyQt5 viewer: open an .aoe2scenario file, pan/zoom the rendered map, and read
per-tile / per-unit details. Terrain mode's Draw/Elevate/Set Elevation tools
(v2) paint the map and save out via File > Save or Save As -- see
descape.scenario_write for the write path itself.

This module is now the main window and its settings dialog: ViewerWindow owns
the menus, toolbar, mode switching, the render pipeline (_render_current /
_apply_dirty / _cache) and every edit callback. The widgets it drives live
beside it -- map_view.py, trigger_panel.py, map_options_panel.py,
viewer_canvas.py, viewer_dialogs.py, and the viewer_common.py helpers they
share. Those import one way, never back into this module."""

from __future__ import annotations

import copy
import dataclasses
import faulthandler
import functools
import html
import math
import os
import random
import sys
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QFont,
    QFontInfo,
    QIcon,
    QKeySequence,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFontComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import (
    __version__,
    asset_source,
    autosave,
    brush,
    cliff_catalog,
    cliff_chain,
    clipboard_history,
    composite_backend,
    crash_report,
    debug_log,
    diplomacy_fields,
    disables_fields,
    edge_ticks,
    garrison,
    gate_orientation,
    grid_overlay,
    iso_geometry,
    level_warm,
    library_compat,
    map_analysis,
    margin_warm,
    object_catalog,
    option_fields,
    perf_trace,
    player_fields,
    player_stats,
    region_clipboard,
    ruler,
    scatter,
    settings,
    terrain_classes,
    terrain_units,
    trigger_clipboard,
    trigger_fields,
    trigger_geometry,
    trigger_organize,
    unit_fields,
    unit_pick,
    unit_references,
    unit_rotation,
    unit_sprites,
    unit_variant,
    view_layers,
    wall_run,
)
from descape.analysis_dialog import AnalysisDialog
from descape.autosave_dialog import RecoverAutosaveDialog
from descape.batch_api import set_terrain
from descape.beach_edges import apply_beach_ring
from descape.clipboard_dialog import ClipboardHistoryDialog
from descape.constant_picker import CatalogBrowseDialog, preview_pixmap
from descape.diplomacy_panel import DiplomacyPanel
from descape.disables_dialog import DisablesDialog
from descape.edit_history import (
    CompositeDiffRecord,
    DiffRecord,
    EditHistory,
    MessagesDiffRecord,
    OptionsDiffRecord,
    TileDiffRecord,
    UnitDiffRecord,
    tile_state,
)
from descape.elevation_tools import set_tiles_elevation
from descape.fill_tools import contiguous_region, flood_fill_terrain
from descape.history_dialog import EditHistoryDialog
from descape.map_options_panel import MapOptionsPanel
from descape.map_view import MapView
from descape.messages_fields import MESSAGE_FIELDS
from descape.messages_model import MessageEditsUnavailableError, MessagesEditModel
from descape.messages_panel import MessagesPanel
from descape.mirror_tools import (
    ANGULAR_MODES,
    MODE_BY_ID,
    MODES,
    angular_mirror_axes,
    plan_mirror,
    plan_mirror_units,
)
from descape.options_model import (
    OptionEditsUnavailableError,
    OptionsEditModel,
    diplomacy_write_supported,
    disables_write_supported,
    options_write_supported,
    player_count_write_supported,
    players_write_supported,
)
from descape.players_panel import PlayersPanel
from descape.render import (
    NON_BUILDING_SPAN,
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
    unit_at,
    unit_mark_color,
    unit_occupied_tiles,
    unit_sprite_draws_at,
    unit_tile_bounds,
    wall_variant_rotation_overrides,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
    UnitSplice,
)
from descape.scatter_dialog import ScatterDialog
from descape.scenario_io import (
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    LoadedScenario,
    UnsupportedStructureVersion,
    load_map_and_units,
    load_map_and_units_from_bytes,
    parse_triggers,
    refresh_player_colors,
    xs_attachment,
)
from descape.scenario_new import (
    LARGE_MAP_CONFIRM_TILES,
    MAX_MAP_TILES,
    MIN_MAP_TILES,
    STANDARD_MAP_SIZE_NAMES,
    STANDARD_MAP_SIZES,
    BlankGenerationError,
    MapSizeError,
    blank_scenario_bytes,
)
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import name_for_terrain_id, tile_span
from descape.terrain_panel import TerrainPanel
from descape.terrain_style import STYLE_LABELS, label_for, style_for_label
from descape.toolbar_overflow import partition
from descape.trigger_model import (
    TriggerEditModel,
    TriggerEditsUnavailableError,
    display_order_moved_to_slot,
    display_order_with_block_inserted,
    display_order_with_copy_inserted,
    exec_order_write_supported,
    moved_display_order_block,
)
from descape.trigger_panel import TriggerPanel
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID, UnitFilter
from descape.unit_model import UnitEditModel, UnitEditsUnavailableError, span_low_corner
from descape.units_panel import UnitsPanel
from descape.viewer_common import (
    _STROKE_LABELS,
    _TOOL_LABELS,
    _TOOL_PARAM,
    _TOOL_SHAPE,
    BRUSH_TOOLS,
    FREE_PLACE_TOOLS,
    _swatch_icon,
    brush_applicable,
    tool_applicable,
)
from descape.viewer_dialogs import CrashReportDialog, DebugLogDialog, apply_theme, apply_ui_font

# D2's arrow-key nudge amounts, in tiles: a fine nudge, and Shift's whole-tile
# step (matching the tile-centre grid a click/place snaps to).
_UNIT_NUDGE_STEP = 0.1
_UNIT_NUDGE_STEP_SHIFT = 1.0

# GH #75: a group drag ghosts every member up to this many, then the grabbed one only.
GROUP_GHOST_CAP = 200

# Free-mode headroom below a span-1 unit's far map edge; a float32 coordinate
# at W - 1e-6 rounds back up to W, which is off-map.
_FREE_EDGE_MARGIN = 1e-3


def _axis_move_range(coord: float, low: int, span: int, size: int, whole_tiles: bool) -> tuple[float, float]:
    """(lowest, highest) delta along one axis that keeps a footprint on the map."""
    if whole_tiles:
        lo, hi = -low, size - (low + span)
    elif span > 1:
        # _span_start's half-tile branch: tile + span/2 is the anchor of a footprint starting at tile.
        lo, hi = span / 2 - coord, size - span / 2 - coord
    else:
        lo, hi = -coord, size - _FREE_EDGE_MARGIN - coord
    # A footprint already overhanging an edge doesn't force a move back inward.
    return min(0, lo), max(0, hi)


def clamp_group_delta(units, dx, dy, map_w: int, map_h: int, whole_tiles: bool):
    """(dx, dy, clamped): the group delta pulled into the range every member
    allows, so the whole group stops where its outermost unit reaches the map
    edge (GH #75). Members already off-map do not limit it."""
    lo_x = lo_y = -math.inf
    hi_x = hi_y = math.inf
    for unit in units:
        if unit_tile_bounds(unit, map_w, map_h) is None:
            continue
        span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
        low_x, low_y = span_low_corner(unit)
        ax_lo, ax_hi = _axis_move_range(unit.x, low_x, span_x, map_w, whole_tiles)
        ay_lo, ay_hi = _axis_move_range(unit.y, low_y, span_y, map_h, whole_tiles)
        lo_x, hi_x = max(lo_x, ax_lo), min(hi_x, ax_hi)
        lo_y, hi_y = max(lo_y, ay_lo), min(hi_y, ay_hi)
    cx, cy = min(max(dx, lo_x), hi_x), min(max(dy, lo_y), hi_y)
    return cx, cy, (cx, cy) != (dx, dy)

# Shown when free placement was asked for and the screen -> map-point inverse
# had no answer for that pixel. A module constant so a test can assert on the
# real string rather than on a paraphrase of it.
FREE_PLACE_FALLBACK_MESSAGE = (
    "Free placement: no exact point under the cursor here, snapped to the tile centre instead"
)

# Which left-panel page each mode shows. The panel swaps rather than growing
# a third column -- the mode already says which one is relevant, so a dock
# would only need its own show/hide logic duplicating what a stack gets free.
_LEFT_PAGE_INFO = 0
_LEFT_PAGE_TRIGGERS = 1
_LEFT_PAGE_UNITS = 2
# Appended, never renumbered: tests/test_trigger_panel.py and
# tests/test_unit_selection_viewer.py assert left_stack.currentIndex() by
# literal integer for pages 0-2.
_LEFT_PAGE_MAP_OPTIONS = 3
_LEFT_PAGE_PLAYERS = 4
# Appended at 5, per the rule above, since Players mode already claimed 4
# by the time Diplomacy mode landed.
_LEFT_PAGE_DIPLOMACY = 5
# Appended at 6, same rule, since Diplomacy mode already claimed 5 by the
# time Messages mode landed.
_LEFT_PAGE_MESSAGES = 6
# Appended at 7, same rule: the Terrain-mode picker page (GH #56).
_LEFT_PAGE_TERRAIN = 7
_LEFT_PAGE_FOR_MODE = {
    "triggers": _LEFT_PAGE_TRIGGERS,
    "units": _LEFT_PAGE_UNITS,
    "map_options": _LEFT_PAGE_MAP_OPTIONS,
    "players": _LEFT_PAGE_PLAYERS,
    "diplomacy": _LEFT_PAGE_DIPLOMACY,
    "messages": _LEFT_PAGE_MESSAGES,
    "terrain": _LEFT_PAGE_TERRAIN,
}

# Mode combo label -> the id self.mode holds. Only multi-word labels need an
# entry; everything else is the lowercased label, which is what self.mode was
# built from before this existed. Without it "Map Options" would become the id
# "map options", with a space, while every gate in this file compares against a
# single lowercase word.
_MODE_ID_FOR_LABEL = {"Map Options": "map_options"}

# The one Map Options row whose write path is TriggerEditModel rather than
# OptionsEditModel. Named here rather than matched inline, because the split is
# checked in four places (the edit fan-out, the editable-field gate, the
# pending-value merge, and the read-only note) and a typo in any one of them
# fails silently in the direction of dropping the edit.
_EXEC_ORDER_FIELD = "legacy_exec_order"
_EXEC_ORDER_SECTION = "Triggers"


def _mode_id(mode_text: str) -> str:
    return _MODE_ID_FOR_LABEL.get(mode_text, mode_text.lower())


def _xs_info_lines(loaded) -> list[str]:
    """The info panel's two read-only XS attachment lines. Honest about what is
    not known yet, like the trigger-tail line above them."""
    script_name, embedded = xs_attachment(loaded)
    if script_name is None:
        name_text = "(not stored before scenario 1.40)"
    else:
        name_text = script_name or "(none)"
    if embedded is not None:
        embedded_text = f"{embedded:,} chars" if embedded else "(none)"
    elif loaded.trigger_read_supported is False:
        embedded_text = "(unknown: Triggers section can't be read)"
    else:
        embedded_text = "(unknown until triggers are parsed)"
    return [f"XS script file: {name_text}", f"Embedded XS: {embedded_text}"]


def _all_diplomacy_cell_ids(loaded) -> frozenset[str]:
    """Every stance and allied-victory cell id the grid stores (the full 8x8
    including the diagonal, and all 8 allied-victory flags) -- what the
    editable-field and read-only-reason gates are computed against, not just
    the 7 opponent rows a panel actually shows for one selected player."""
    return frozenset(
        {
            **diplomacy_fields.stance_offsets(loaded),
            **diplomacy_fields.allied_victory_offsets(loaded),
        }
    )


def _diplomacy_field_label(cell_id: str) -> str:
    """A human label for an undo/status entry, from a raw cell id -- the
    Diplomacy-grid counterpart to an OptionFieldSpec's own `.label`, which
    these synthetic ids have no spec to carry."""
    parsed = diplomacy_fields.parse_cell_id(cell_id)
    if parsed[0] == "stance":
        _, row, col = parsed
        return f"P{row} stance toward P{col}"
    _, player = parsed
    return f"P{player} allied victory"


_MESSAGE_FIELD_LABELS = {spec.field_id: spec.label for spec in MESSAGE_FIELDS}


def _message_field_label(field_id: str) -> str:
    """A human label for an undo/status entry, from a raw Messages field id --
    the id-field counterpart ("hints_id") has no MessageFieldSpec of its own,
    so it borrows its text field's label with a suffix."""
    if field_id.endswith("_id"):
        base = field_id[: -len("_id")]
        return f"{_MESSAGE_FIELD_LABELS.get(base, base)} string ID"
    return _MESSAGE_FIELD_LABELS.get(field_id, field_id)


# Lives at object_catalog.display_name: units_panel.py needs the same
# display convention, and importing viewer.py from there would invert the
# dependency. Aliased here so this file's own 10 call sites stay untouched.
_unit_name = object_catalog.display_name


def _entries_of(trigger, kind: str):
    """A trigger's live condition or effect list, by kind (not a copy)."""
    return trigger.conditions if kind == "condition" else trigger.effects


def _garrison_referrers(model, unit) -> list:
    """UnitEditModel.referencing(unit), minus the one case it deliberately
    keeps: a unit whose own reference_id is -1.

    referencing() preserves the -1 bucket on purpose (its own docstring, so
    a byte-identical answer on a pathological file), and -1 is exactly what
    every unit carries when it is inside nothing. Read literally, a host with
    reference_id -1 therefore "holds" every ungarrisoned unit in the file --
    which would cascade a move or a delete across the whole map, and fill the
    Inspector's Garrison list with it. -1 means "inside nothing" in
    UnitFilter.matches() and in the map-analysis check; it means that here."""
    if unit.reference_id == -1:
        return []
    return model.referencing(unit)


@functools.cache
def _wall_consts() -> frozenset[int]:
    """Place Unit's wall-run consts (GH #98). Lazy: the graphic map it
    reads is not needed to import this module."""
    return frozenset(unit_sprites.wall_family_consts())


# Sourced from iso_geometry, not redefined here, so Set Elevation's spinbox
# range and Stepped mode's canvas sizing (descape.render.elevations_and_proj)
# can never drift apart -- see that constant's own comment for why Phase 4
# needs them to agree exactly.
ELEVATION_LEVEL_MAX = iso_geometry.MAX_ELEVATION

# Above this many dirty tiles, Stepped mode's own _apply_dirty() path (per-
# tile dilation into a Python set, then one patch() over what ends up being
# effectively the whole canvas -- see dirty_screen_bbox_iso's own docstring)
# is slower than just re-rendering: measured with tools/bench_fill_latency.py
# against a full-map Paint Can fill (this project's own worst case for dirty
# tile count) -- 14,400 tiles (blank_120x120) stayed under the dilation path
# at ~105ms, but 230,400 tiles (blank_480x480) cost ~2.3s there versus ~35ms
# for a full _render_current() (Phase B-C's lazy chunk cache means "full
# re-render" no longer means "recomposite every pixel up front" -- see that
# method's own docstring). Set well above the fine case and well below the
# slow one; the exact crossover between them hasn't been measured.
STEPPED_FULL_RERENDER_THRESHOLD = 20_000

# Above this many tiles, a Paint Can fill with Trees or Eye candy checked
# confirms before writing anything -- a 480x480 map's full-map fill can plan
# >200k units (descape/terrain_units.py), which is not finishable at
# UnitEditModel's current per-op cost. Draw has no equivalent guard: its
# counts are bounded by brush size x drag length, and edit_history.
# abort_stroke() cannot un-paint tiles already written mid-drag, so there is
# no safe way to cancel out of a Draw stroke partway through anyway.
TERRAIN_UNIT_CONFIRM_THRESHOLD = 2000

# Display-only sentinel assigned to LoadedScenario.path for a File > New map.
# Deliberately relative and non-existent: LoadedScenario.path is only ever read
# for display and for Save As's default name -- write_scenario() takes an
# explicit destination and never reads it (see that field's own comment in
# scenario_io.py).
UNTITLED_NAME = "Untitled.aoe2scenario"
UNTITLED_PATH = Path(UNTITLED_NAME)

# How long an autosave tick waits before looking again, when it arrived
# while the window was busy or mid-stroke. Short enough that a long paint
# session doesn't starve autosave for a whole interval.
AUTOSAVE_RETRY_MS = 5000

STATUS_OK_COLOR = "#4caf50"
STATUS_ERROR_COLOR = "#e05252"

# The hover label's idle text, shown whenever the cursor isn't over a tile.
HOVER_IDLE_TEXT = "Hover the map for tile info"


class SettingsDialog(QDialog):
    """Tabbed settings dialog: General (app-wide behavior, including the
    AoE2DE install path -- not its own tab since it's a one-time setup
    step, not a recurring one), Appearance (theme, render quality), and
    Keybinds. Split into tabs so more settings in any category have an
    obvious place to land without reshaping this dialog."""

    def __init__(self, parent: ViewerWindow):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 360)
        self._window = parent

        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_appearance_tab(), "Appearance")
        tabs.addTab(self._build_saving_tab(), "Saving")
        tabs.addTab(self._build_keybinds_tab(), "Keybinds")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _build_general_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        zoom_label = QLabel("Mouse wheel zoom:")
        layout.addWidget(zoom_label)
        self.zoom_cursor_radio = QRadioButton("Zoom in centered on mouse cursor location")
        self.zoom_view_center_radio = QRadioButton("Zoom in overall (view center)")
        zoom_group = QButtonGroup(tab)
        zoom_group.addButton(self.zoom_cursor_radio)
        zoom_group.addButton(self.zoom_view_center_radio)
        if settings.get_zoom_centered_on_cursor():
            self.zoom_cursor_radio.setChecked(True)
        else:
            self.zoom_view_center_radio.setChecked(True)
        # Connecting only one radio's toggled is enough: within an exclusive
        # QButtonGroup, this one flips to False exactly when the other flips
        # to True, so both directions are covered by this single signal.
        self.zoom_cursor_radio.toggled.connect(self._on_zoom_mode_toggled)
        layout.addWidget(self.zoom_cursor_radio)
        layout.addWidget(self.zoom_view_center_radio)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        self._add_install_section(layout)

        layout.addStretch(1)
        return tab

    def _on_zoom_mode_toggled(self, centered_on_cursor: bool) -> None:
        settings.set_zoom_centered_on_cursor(centered_on_cursor)
        self._window.map_view.set_zoom_anchor_mode(centered_on_cursor)
        mode = "centered on mouse cursor" if centered_on_cursor else "overall (view center)"
        self._window._log_status(f"Zoom mode: {mode}")

    def _build_appearance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        dark_checkbox = QCheckBox("Dark mode")
        dark_checkbox.setChecked(settings.get_dark_mode())
        dark_checkbox.setToolTip(
            "App chrome only (menus, dialogs, toolbars) -- the map view's own "
            "colors (real terrain textures, unit dots, hover highlights) are "
            "unaffected, since they represent game data or are already dark."
        )
        dark_checkbox.toggled.connect(self._on_dark_mode_toggled)
        layout.addWidget(dark_checkbox)

        ui_font_row = QHBoxLayout()
        ui_font_row.addWidget(QLabel("UI font:"))
        app_font = QApplication.instance().font()
        self.ui_font_combo = QFontComboBox()
        self.ui_font_size_spin = QSpinBox()
        self.ui_font_size_spin.setRange(settings.UI_FONT_SIZE_MIN, settings.UI_FONT_SIZE_MAX)
        self.ui_font_size_spin.setSuffix(" pt")
        ui_font_tooltip = (
            "App chrome only (menus, dialogs, panels, toolbars, the status "
            "log) -- the map view's own overlay text (ruler readout, "
            "distance-tick numbers, stacked-unit badges) keeps its own fixed "
            "sizes, since it is measured in device pixels against fixed label "
            "boxes. Reset returns both to the platform default."
        )
        self.ui_font_combo.setToolTip(ui_font_tooltip)
        self.ui_font_size_spin.setToolTip(ui_font_tooltip)
        ui_font_reset = QPushButton("Reset to default")
        ui_font_reset.setToolTip(ui_font_tooltip)
        # Values first, signals after -- same order dark_checkbox and
        # graphics_quality_slider use, and load-bearing here: populating a
        # QFontComboBox emits currentFontChanged, which connected first would
        # persist a family the user never chose and pin "" (follow the
        # platform) to whatever the platform default happened to be.
        self._set_ui_font_widgets(settings.get_ui_font_family(), settings.get_ui_font_size(), app_font)
        self._ui_font_apply_timer = QTimer(self)
        self._ui_font_apply_timer.setSingleShot(True)
        self._ui_font_apply_timer.timeout.connect(self._apply_ui_font)
        self.ui_font_combo.currentFontChanged.connect(self._on_ui_font_family_changed)
        self.ui_font_size_spin.valueChanged.connect(self._on_ui_font_size_changed)
        ui_font_reset.clicked.connect(self._on_ui_font_reset)
        ui_font_row.addWidget(self.ui_font_combo, stretch=1)
        ui_font_row.addWidget(self.ui_font_size_spin, stretch=0)
        ui_font_row.addWidget(ui_font_reset, stretch=0)
        layout.addLayout(ui_font_row)

        preload_checkbox = QCheckBox("Preload neighbouring zoom levels")
        preload_checkbox.setChecked(settings.get_preload_zoom_levels())
        preload_checkbox.setToolTip(
            "After a map opens, resolve the next zoom levels' unit sprites in "
            "idle time so the first zoom doesn't stutter, and warm a margin of "
            "chunks just outside the viewport as you pan so scrolling into new "
            "territory doesn't stutter either. The window stays interactive "
            "throughout; the cost is a few seconds of background work per "
            "open, plus ongoing idle-time work while panning, and holding "
            "extra chunks/levels in memory. Applies immediately."
        )
        preload_checkbox.toggled.connect(self._on_preload_zoom_levels_toggled)
        layout.addWidget(preload_checkbox)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Graphics quality:"))
        self.graphics_quality_slider = QSlider(Qt.Horizontal)
        self.graphics_quality_slider.setRange(settings.GRAPHICS_QUALITY_MIN, settings.GRAPHICS_QUALITY_MAX)
        self.graphics_quality_slider.setTickInterval(1)
        self.graphics_quality_slider.setTickPosition(QSlider.TicksBelow)
        self.graphics_quality_slider.setSingleStep(1)
        self.graphics_quality_slider.setPageStep(1)
        self.graphics_quality_slider.setToolTip(
            "Scales map render resolution (see render.tile_pixels_for_map) -- "
            "applies to the next render. Enhanced only doubles resolution on "
            "large maps (already-downscaled); on small maps it's a no-op, "
            "since doubling their base resolution is well past what's been "
            "measured safe for memory."
        )
        self.graphics_quality_slider.setValue(settings.get_graphics_quality())
        self.graphics_quality_label = QLabel()
        self._update_graphics_quality_label(settings.get_graphics_quality())
        self._graphics_quality_apply_timer = QTimer(self)
        self._graphics_quality_apply_timer.setSingleShot(True)
        self._graphics_quality_apply_timer.timeout.connect(self._apply_graphics_quality)
        self.graphics_quality_slider.valueChanged.connect(self._on_graphics_quality_slider_changed)
        quality_row.addWidget(self.graphics_quality_slider, stretch=1)
        quality_row.addWidget(self.graphics_quality_label)
        layout.addLayout(quality_row)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Isometric elevation height:"))
        self.elev_step_slider = QSlider(Qt.Horizontal)
        # Value space is the 1-based stop index, not the pct -- setSingleStep
        # alone only governs arrow keys and the wheel, so a drag would still
        # produce arbitrary off-stop pcts via QStyle::sliderValueFromPosition.
        # Same shape as graphics_quality_slider above.
        self.elev_step_slider.setRange(1, len(settings.ELEV_STEP_PCT_STOPS))
        self.elev_step_slider.setTickInterval(1)
        self.elev_step_slider.setTickPosition(QSlider.TicksBelow)
        self.elev_step_slider.setSingleStep(1)
        self.elev_step_slider.setPageStep(1)
        self.elev_step_slider.setToolTip(
            "Stepped and Flat + Isometric View rendering only -- how tall one "
            "elevation level's displacement reads on screen. Flat + Isometric "
            "View always renders at elevation 0, so this only governs its "
            "diamond-tile geometry there, not an actual step. Default is Tall; "
            "the range above "
            "it is real headroom, not just cosmetic overshoot -- AoE2's real "
            "elevation transitions are smooth multi-tile ramps rather than a "
            "single hard edge, which is what keeps a taller step from making "
            "tiles hide behind their taller neighbors as easily as a single "
            "sharp step would."
        )
        self.elev_step_slider.setValue(settings.elev_step_index(settings.get_elev_step_pct()))
        self.elev_step_value_label = QLabel()
        self._update_elev_step_label(settings.get_elev_step_pct())
        self._elev_step_apply_timer = QTimer(self)
        self._elev_step_apply_timer.setSingleShot(True)
        self._elev_step_apply_timer.timeout.connect(self._apply_elev_step_pct)
        self.elev_step_slider.valueChanged.connect(self._on_elev_step_slider_changed)
        self.elev_step_slider.sliderReleased.connect(self._apply_elev_step_pct)
        height_row.addWidget(self.elev_step_slider, stretch=1)
        height_row.addWidget(self.elev_step_value_label)
        layout.addLayout(height_row)

        pan_row = QHBoxLayout()
        pan_row.addWidget(QLabel("Pan speed:"))
        self.pan_speed_slider = QSlider(Qt.Horizontal)
        self.pan_speed_slider.setRange(settings.PAN_SPEED_MIN, settings.PAN_SPEED_MAX)
        self.pan_speed_slider.setTickInterval(200)
        self.pan_speed_slider.setTickPosition(QSlider.TicksBelow)
        self.pan_speed_slider.setSingleStep(50)
        self.pan_speed_slider.setPageStep(100)
        self.pan_speed_slider.setToolTip(
            "How fast holding a Pan key (Settings > Keybinds) scrolls the map, "
            "in viewport pixels per second. Measured in screen pixels like the "
            "middle-drag pan, so a fixed speed covers fewer tiles the further "
            "you zoom in. Takes effect on the next hold."
        )
        self.pan_speed_slider.setValue(settings.get_pan_speed())
        self.pan_speed_value_label = QLabel()
        self._update_pan_speed_label(settings.get_pan_speed())
        # Debounced like both sliders above, for a different reason: every
        # settings setter does a full _load_config/_save_config YAML round
        # trip, so a live valueChanged would be hundreds of disk writes
        # across one drag of this 200-2000 range.
        self._pan_speed_apply_timer = QTimer(self)
        self._pan_speed_apply_timer.setSingleShot(True)
        self._pan_speed_apply_timer.timeout.connect(self._apply_pan_speed)
        self.pan_speed_slider.valueChanged.connect(self._on_pan_speed_slider_changed)
        self.pan_speed_slider.sliderReleased.connect(self._apply_pan_speed)
        pan_row.addWidget(self.pan_speed_slider, stretch=1)
        pan_row.addWidget(self.pan_speed_value_label)
        layout.addLayout(pan_row)

        layout.addWidget(self._build_overlay_colors_group())

        font_row = QHBoxLayout()
        font_row.addWidget(QLabel("Ruler label size:"))
        self.ruler_label_font_spin = QSpinBox()
        self.ruler_label_font_spin.setRange(settings.RULER_LABEL_FONT_PX_MIN, settings.RULER_LABEL_FONT_PX_MAX)
        self.ruler_label_font_spin.setSuffix(" px")
        self.ruler_label_font_spin.setValue(settings.get_ruler_label_font_px())
        self.ruler_label_font_spin.valueChanged.connect(self._on_ruler_label_font_px_changed)
        font_row.addWidget(self.ruler_label_font_spin, stretch=0)
        font_row.addStretch(1)
        layout.addLayout(font_row)

        distance_tick_font_row = QHBoxLayout()
        distance_tick_font_row.addWidget(QLabel("Distance ticks label size:"))
        self.distance_tick_font_spin = QSpinBox()
        self.distance_tick_font_spin.setRange(
            settings.DISTANCE_TICK_FONT_PX_MIN, settings.DISTANCE_TICK_FONT_PX_MAX
        )
        self.distance_tick_font_spin.setSuffix(" px")
        self.distance_tick_font_spin.setValue(settings.get_distance_tick_font_px())
        self.distance_tick_font_spin.valueChanged.connect(self._on_distance_tick_font_px_changed)
        distance_tick_font_row.addWidget(self.distance_tick_font_spin, stretch=0)
        distance_tick_font_row.addStretch(1)
        layout.addLayout(distance_tick_font_row)

        # View > Grid's appearance. A drag is bracketed rather than debounced:
        # its first tick swaps the baked grid for GridItem's preview (one
        # eviction), each tick after that is two QPens and an update(), and
        # sliderReleased or the timer (keyboard and wheel) re-bakes. Blend is
        # a plain value-space slider, so it moves smoothly across its range;
        # only thickness has stops left to snap to.
        self._grid_apply_timer = QTimer(self)
        self._grid_apply_timer.setSingleShot(True)
        self._grid_apply_timer.timeout.connect(self._end_grid_preview)
        grid_blend_row = QHBoxLayout()
        grid_blend_row.addWidget(QLabel("Grid line blend:"))
        self.grid_blend_slider = self._blend_slider(settings.get_grid_blend())
        self.grid_blend_value_label = QLabel(self._blend_label(settings.get_grid_blend()))
        self.grid_blend_slider.valueChanged.connect(self._on_grid_blend_changed)
        self.grid_blend_slider.sliderReleased.connect(self._end_grid_preview)
        grid_blend_row.addWidget(self.grid_blend_slider, stretch=1)
        grid_blend_row.addWidget(self.grid_blend_value_label)
        layout.addLayout(grid_blend_row)

        grid_thickness_row = QHBoxLayout()
        grid_thickness_row.addWidget(QLabel("Grid thickness:"))
        self.grid_thickness_slider = self._stop_slider(
            len(grid_overlay.THICKNESS_STOPS), grid_overlay.thickness_index(settings.get_grid_thickness())
        )
        self.grid_thickness_value_label = QLabel(f"{settings.get_grid_thickness()} px")
        self.grid_thickness_slider.valueChanged.connect(self._on_grid_thickness_changed)
        self.grid_thickness_slider.sliderReleased.connect(self._end_grid_preview)
        grid_thickness_row.addWidget(self.grid_thickness_slider, stretch=1)
        grid_thickness_row.addWidget(self.grid_thickness_value_label)
        layout.addLayout(grid_thickness_row)

        layout.addStretch(1)
        return tab

    @staticmethod
    def _stop_slider(stop_count: int, index: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, stop_count)
        slider.setTickInterval(1)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setValue(index)
        return slider

    @staticmethod
    def _blend_slider(value: int) -> QSlider:
        """Every value in the range, not a stop ladder. The ticks are only a
        landmark for the invisible centre; dragging still lands anywhere."""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(grid_overlay.BLEND_MIN, grid_overlay.BLEND_MAX)
        slider.setTickInterval(grid_overlay.BLEND_MAX // 2)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        slider.setValue(value)
        return slider

    @staticmethod
    def _blend_label(value: int) -> str:
        if not value:
            return "off"
        return f"{'light' if value > 0 else 'dark'} {abs(value)}"

    def _on_grid_blend_changed(self, value: int) -> None:
        settings.set_grid_blend(value)
        self.grid_blend_value_label.setText(self._blend_label(value))
        self._preview_grid_appearance()

    def _on_grid_thickness_changed(self, index: int) -> None:
        value = grid_overlay.thickness_for_index(index)
        settings.set_grid_thickness(value)
        self.grid_thickness_value_label.setText(f"{value} px")
        self._preview_grid_appearance()

    def _preview_grid_appearance(self) -> None:
        window = self._window
        view = window.map_view
        if view.grid_bake_live() and not view.grid_previewing():
            window._apply_grid_change(view.begin_grid_preview)
        view.set_grid_appearance(settings.get_grid_blend(), settings.get_grid_thickness())
        if view.grid_previewing() and not (
            self.grid_blend_slider.isSliderDown() or self.grid_thickness_slider.isSliderDown()
        ):
            self._grid_apply_timer.start(200)

    def _end_grid_preview(self) -> None:
        self._grid_apply_timer.stop()
        if self._window.map_view.grid_previewing():
            self._window._apply_grid_change(self._window.map_view.end_grid_preview)

    def done(self, result: int) -> None:
        # A keyboard change's timer dies with the dialog, which would leave
        # the grid stuck in its unbaked preview.
        self._end_grid_preview()
        super().done(result)

    # OVERLAY_COLORS id prefix (before the first "_") -> section header text,
    # mirroring _KEYBIND_SECTION_TITLES below.
    _OVERLAY_SECTION_TITLES: ClassVar[dict[str, str]] = {
        "highlight": "Terrain brush",
        "pan": "Pan",
        "unit": "Units",
        "ruler": "Ruler",
        "region": "Select tool",
        "mirror": "Map mirroring",
        "footprint": "Footprint outlines",
        "range": "Range rings",
        "analysis": "Map Analysis",
        "trigger": "Trigger overlay",
    }

    def _build_overlay_colors_group(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(QLabel("Tool overlay colors:"))

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        grid.setColumnStretch(1, 1)

        self._overlay_swatches: dict[str, QPushButton] = {}
        row = 0
        current_section = None
        for color_id, label, _default in settings.OVERLAY_COLORS:
            section = color_id.split("_", 1)[0]
            if section != current_section:
                if current_section is not None:
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Sunken)
                    grid.addWidget(divider, row, 0, 1, 3)
                    row += 1
                section_label = QLabel(f"<b>{self._OVERLAY_SECTION_TITLES.get(section, section.title())}</b>")
                grid.addWidget(section_label, row, 0, 1, 3)
                row += 1
                current_section = section

            grid.addWidget(QLabel(label), row, 0)

            swatch = QPushButton()
            swatch.setFixedWidth(60)
            swatch.setToolTip(settings.get_overlay_color(color_id))
            swatch.clicked.connect(lambda _checked, cid=color_id: self._pick_overlay_color(cid))
            self._overlay_swatches[color_id] = swatch
            self._set_swatch_color(swatch, settings.get_overlay_color(color_id))
            grid.addWidget(swatch, row, 1)

            default_btn = QPushButton("Default")
            default_btn.clicked.connect(lambda _checked, cid=color_id: self._reset_overlay_color(cid))
            grid.addWidget(default_btn, row, 2)

            row += 1

        grid.setRowStretch(row, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        scroll.setMinimumHeight(220)
        outer.addWidget(scroll)
        return container

    @staticmethod
    def _set_swatch_color(swatch: QPushButton, hex_str: str) -> None:
        swatch.setStyleSheet(f"background-color: {hex_str};")
        swatch.setToolTip(hex_str)

    def _pick_overlay_color(self, color_id: str) -> None:
        current = QColor(settings.get_overlay_color(color_id))
        chosen = QColorDialog.getColor(current, self)
        if not chosen.isValid():
            return
        self._apply_overlay_color(color_id, chosen.name())

    def _reset_overlay_color(self, color_id: str) -> None:
        self._apply_overlay_color(color_id, settings.get_default_overlay_color(color_id))

    def _apply_overlay_color(self, color_id: str, hex_str: str) -> None:
        settings.set_overlay_color(color_id, hex_str)
        self._set_swatch_color(self._overlay_swatches[color_id], hex_str)
        self._window.map_view.apply_overlay_colors()
        label = settings.get_overlay_color_label(color_id)
        self._window._log_status(f"Overlay colour: {label} -> {hex_str}")

    def _on_ruler_label_font_px_changed(self, value: int) -> None:
        settings.set_ruler_label_font_px(value)
        self._window.map_view.apply_ruler_label_font()
        self._window._log_status(f"Ruler label size: {value}px")

    def _on_distance_tick_font_px_changed(self, value: int) -> None:
        settings.set_distance_tick_font_px(value)
        self._window.map_view.apply_distance_tick_font()
        self._window._log_status(f"Distance ticks label size: {value}px")

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        settings.set_dark_mode(enabled)
        apply_theme(QApplication.instance(), enabled)
        self._window._log_status(f"Dark mode: {'on' if enabled else 'off'}")

    def _set_ui_font_widgets(self, family: str, size: int | None, app_font) -> None:
        """Opens the pair on the persisted values, falling back to the live
        app font for either one that is unset. Signals are blocked: this
        also runs from Reset, where re-firing them would persist exactly
        what Reset just cleared. QFontInfo, not app_font.pointSize(), since
        the latter is -1 for a pixel-sized font and QSpinBox would silently
        clamp that to the range minimum."""
        self.ui_font_combo.blockSignals(True)
        self.ui_font_size_spin.blockSignals(True)
        self.ui_font_combo.setCurrentFont(QFont(family) if family else app_font)
        current_pt = QFontInfo(app_font).pointSize()
        self.ui_font_size_spin.setValue(
            size
            if size is not None
            else max(settings.UI_FONT_SIZE_MIN, min(settings.UI_FONT_SIZE_MAX, current_pt))
        )
        self.ui_font_combo.blockSignals(False)
        self.ui_font_size_spin.blockSignals(False)

    def _on_ui_font_family_changed(self, font) -> None:
        # Persisted here, per widget, rather than in the debounced applier:
        # an applier reading both widgets would write the combo's displayed
        # family on a size-only change, turning "follow the platform" into a
        # permanent pin on whatever was showing.
        settings.set_ui_font_family(font.family())
        self._ui_font_apply_timer.start(200)

    def _on_ui_font_size_changed(self, value: int) -> None:
        settings.set_ui_font_size(value)
        self._ui_font_apply_timer.start(200)

    def _on_ui_font_reset(self) -> None:
        settings.set_ui_font_family("")
        settings.set_ui_font_size(None)
        apply_ui_font(QApplication.instance(), "", None)
        self._set_ui_font_widgets("", None, QApplication.instance().font())
        self._window._update_log_min_height()
        self._window._log_status("UI font: platform default")

    def _apply_ui_font(self) -> None:
        """Debounced, the same 200ms shape as _apply_graphics_quality and for
        the same reason: a spinbox held on its arrow fires once per step, and
        every app.setFont() is a full-app relayout."""
        family = settings.get_ui_font_family()
        size = settings.get_ui_font_size()
        apply_ui_font(QApplication.instance(), family, size)
        self._window._update_log_min_height()
        shown_family = family or "platform default"
        shown_size = f"{size} pt" if size is not None else "default size"
        self._window._log_status(f"UI font: {shown_family}, {shown_size}")

    def _on_preload_zoom_levels_toggled(self, enabled: bool) -> None:
        """Persists only -- turning it OFF also cancels both warms already in
        flight (the neighbour-level layer warm AND the margin-ring chunk
        warm, maintainer plan 2026-09-07's A6), so the choice takes effect
        immediately rather than once queued work happens to finish. Turning
        it ON does not start either retroactively: the level warm's
        neighbours are derived from the fit baseline at open time (and the
        current view may be nowhere near it), and the margin warm only ever
        starts from the next viewport-changed poll fire."""
        settings.set_preload_zoom_levels(enabled)
        if not enabled:
            self._window._level_warmer.cancel()
            self._window._margin_warmer.cancel()
        self._window._log_status(f"Preload zoom levels: {'on' if enabled else 'off'}")

    def _update_graphics_quality_label(self, quality: int) -> None:
        self.graphics_quality_label.setText(settings.GRAPHICS_QUALITY_LABELS[quality])

    def _on_graphics_quality_slider_changed(self, value: int) -> None:
        self._update_graphics_quality_label(value)
        # Debounced (not applied on every intermediate value) since a full
        # re-render is expensive and dragging the slider across all 4 stops
        # would otherwise fire one per stop -- see the elev_step slider's
        # matching comment below, same reasoning.
        self._graphics_quality_apply_timer.start(200)

    def _apply_graphics_quality(self) -> None:
        value = self.graphics_quality_slider.value()
        settings.set_graphics_quality(value)
        self._window._log_status(f"Graphics quality: {settings.GRAPHICS_QUALITY_LABELS[value]}")
        self._window.refresh_map()

    def _update_elev_step_label(self, pct: int) -> None:
        suffix = " (Tall, default)" if pct == iso_geometry.ELEV_STEP_DEFAULT_PCT else ""
        self.elev_step_value_label.setText(f"{pct}%{suffix}")

    def _on_elev_step_slider_changed(self, value: int) -> None:
        # value is a stop index -- _update_elev_step_label still takes a pct,
        # so its "(Tall, default)" test against ELEV_STEP_DEFAULT_PCT holds.
        self._update_elev_step_label(settings.elev_step_pct_for_index(value))
        # Each application is a full Stepped-mode re-render
        # (elevations_and_proj + IsoChunkCache rebuild) -- expensive enough
        # on a real map that applying it live per-pixel would make dragging
        # itself laggy. While the mouse is down, just update the label and
        # wait for sliderReleased to apply. Keyboard arrow presses have no
        # press/release pair, so fall back to a short debounce timer for
        # those instead.
        if self.elev_step_slider.isSliderDown():
            return
        self._elev_step_apply_timer.start(200)

    def _apply_elev_step_pct(self) -> None:
        value = settings.elev_step_pct_for_index(self.elev_step_slider.value())
        settings.set_elev_step_pct(value)
        self._window._log_status(f"Isometric elevation height: {value}%")
        self._window.refresh_map()

    def _update_pan_speed_label(self, px_per_s: int) -> None:
        suffix = " (default)" if px_per_s == settings.PAN_SPEED_DEFAULT else ""
        self.pan_speed_value_label.setText(f"{px_per_s} px/s{suffix}")

    def _on_pan_speed_slider_changed(self, value: int) -> None:
        self._update_pan_speed_label(value)
        if self.pan_speed_slider.isSliderDown():
            return
        self._pan_speed_apply_timer.start(200)

    def _apply_pan_speed(self) -> None:
        # No refresh_map(): MapView._pan_step reads get_pan_speed() on every
        # tick, so the new value is live on the next hold with no wiring.
        settings.set_pan_speed(self.pan_speed_slider.value())

    # action_id prefix (before the first "_") -> section header text. Covers
    # today's sections (REBINDABLE_ACTIONS's "file_*"/"edit_*"/"view_*"/
    # "help_*"/"mode_*"/"tool_*"/"adjust_*" entries); an unlisted future
    # prefix still gets a section of its own, just titled from the raw
    # prefix instead of a curated name.
    _KEYBIND_SECTION_TITLES: ClassVar[dict[str, str]] = {
        "file": "File",
        "edit": "Edit",  # renamed from "Copy/Paste" -- now covers the whole Edit menu
        "map": "Map",
        "analysis": "Analysis",
        "view": "View",
        "help": "Help",
        "mode": "Modes",
        "filter": "Filters",
        "player": "Player Selection",
        "unit": "Units",
        "tool": "Tools",
        "adjust": "Tool Value",
    }

    def _build_saving_tab(self) -> QWidget:
        """Autosave's four controls plus the .bak/.orig switch.

        Every widget is assigned to self, not left a bare local like the
        General tab's dark_checkbox: a bare local is unreachable from any
        GUI test, and these have to be drivable from one. Each value is set
        BEFORE its signal is connected, or merely constructing this dialog
        would write config.yaml -- the same trap the View menu's tick
        actions already document.

        No OK/Apply/Cancel to design around: this dialog is Close-only, so
        each handler persists and applies in one step, like
        _on_dark_mode_toggled. Combos rather than spin boxes for the two
        membership-gated values, which makes their off-list ValueError
        unreachable from the UI.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.autosave_enabled_check = QCheckBox("Autosave on a timer")
        self.autosave_enabled_check.setToolTip(
            "Writes a recovery snapshot to its own rotating slot, never to "
            "the file you have open, and never clears the unsaved-changes "
            "marker. A tick is skipped entirely while a document is clean, "
            "unchanged since its last autosave, or mid-stroke."
        )
        self.autosave_enabled_check.setChecked(settings.get_autosave_enabled())
        self.autosave_enabled_check.toggled.connect(self._on_autosave_enabled_toggled)
        layout.addWidget(self.autosave_enabled_check)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Autosave every:"))
        self.autosave_interval_combo = QComboBox()
        for minutes in settings.AUTOSAVE_INTERVAL_CHOICES:
            self.autosave_interval_combo.addItem(f"{minutes} min", minutes)
        self.autosave_interval_combo.setCurrentIndex(
            settings.AUTOSAVE_INTERVAL_CHOICES.index(settings.get_autosave_interval_min())
        )
        self.autosave_interval_combo.currentIndexChanged.connect(self._on_autosave_interval_changed)
        interval_row.addWidget(self.autosave_interval_combo)
        interval_row.addStretch(1)
        layout.addLayout(interval_row)

        retention_row = QHBoxLayout()
        retention_row.addWidget(QLabel("Keep per document:"))
        self.autosave_retention_combo = QComboBox()
        for count in settings.AUTOSAVE_RETENTION_CHOICES:
            self.autosave_retention_combo.addItem(f"{count} autosave{'s' if count != 1 else ''}", count)
        self.autosave_retention_combo.setCurrentIndex(
            settings.AUTOSAVE_RETENTION_CHOICES.index(settings.get_autosave_retention())
        )
        self.autosave_retention_combo.currentIndexChanged.connect(self._on_autosave_retention_changed)
        retention_row.addWidget(self.autosave_retention_combo)
        retention_row.addStretch(1)
        layout.addLayout(retention_row)

        location_row = QHBoxLayout()
        location_row.addWidget(QLabel("Autosaves go:"))
        self.autosave_location_combo = QComboBox()
        self.autosave_location_combo.addItem("In the DEscape config folder", "central")
        self.autosave_location_combo.addItem("Beside the scenario file", "sidecar")
        self.autosave_location_combo.setToolTip(
            "Beside the file falls back to the config folder for any document "
            "it can't serve: an untitled one, a Steam Workshop/Proton path, or "
            "a shipped template."
        )
        self.autosave_location_combo.setCurrentIndex(
            settings.AUTOSAVE_LOCATION_CHOICES.index(settings.get_autosave_location())
        )
        self.autosave_location_combo.currentIndexChanged.connect(self._on_autosave_location_changed)
        location_row.addWidget(self.autosave_location_combo)
        location_row.addStretch(1)
        layout.addLayout(location_row)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        self.backups_enabled_check = QCheckBox("Write .bak and .orig backups on save")
        self.backups_enabled_check.setToolTip(
            ".bak refreshes on every save that changes bytes; .orig is a "
            "one-time pristine snapshot of the file as it was first opened."
        )
        self.backups_enabled_check.setChecked(settings.get_backups_enabled())
        self.backups_enabled_check.toggled.connect(self._on_backups_enabled_toggled)
        layout.addWidget(self.backups_enabled_check)

        layout.addStretch(1)
        return tab

    def _on_autosave_enabled_toggled(self, enabled: bool) -> None:
        settings.set_autosave_enabled(enabled)
        self._window._reset_autosave_timer()
        self._window._log_status(f"Autosave: {'on' if enabled else 'off'}")

    def _on_autosave_interval_changed(self, index: int) -> None:
        minutes = self.autosave_interval_combo.itemData(index)
        settings.set_autosave_interval_min(minutes)
        self._window._reset_autosave_timer()
        self._window._log_status(f"Autosave interval: {minutes} min")

    def _on_autosave_retention_changed(self, index: int) -> None:
        # Retention and location both take effect on the next tick, so
        # neither needs _reset_autosave_timer().
        count = self.autosave_retention_combo.itemData(index)
        settings.set_autosave_retention(count)
        self._window._log_status(f"Autosaves kept per document: {count}")

    def _on_autosave_location_changed(self, index: int) -> None:
        location = self.autosave_location_combo.itemData(index)
        settings.set_autosave_location(location)
        self._window._log_status(f"Autosave location: {location}")

    def _on_backups_enabled_toggled(self, enabled: bool) -> None:
        settings.set_backups_enabled(enabled)
        self._window._log_status(f"Save backups: {'on' if enabled else 'off'}")

    def _build_keybinds_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        grid.setColumnStretch(1, 1)

        self._keybind_edits: dict[str, QKeySequenceEdit] = {}
        row = 0
        current_section = None
        for action_id, label, _default in settings.REBINDABLE_ACTIONS:
            section = action_id.split("_", 1)[0]
            if section != current_section:
                if current_section is not None:
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Sunken)
                    grid.addWidget(divider, row, 0, 1, 4)
                    row += 1
                section_label = QLabel(f"<b>{self._KEYBIND_SECTION_TITLES.get(section, section.title())}</b>")
                grid.addWidget(section_label, row, 0, 1, 4)
                row += 1
                current_section = section

            grid.addWidget(QLabel(label), row, 0)

            edit = QKeySequenceEdit(QKeySequence(settings.get_keybind(action_id)))
            edit.keySequenceChanged.connect(
                lambda seq, aid=action_id: self._on_keybind_changed(aid, seq)
            )
            # QKeySequenceEdit has no setAlignment of its own -- it wraps an
            # internal QLineEdit, which does.
            line_edit = edit.findChild(QLineEdit)
            if line_edit is not None:
                line_edit.setAlignment(Qt.AlignCenter)
            grid.addWidget(edit, row, 1)
            self._keybind_edits[action_id] = edit

            default_btn = QPushButton("Default")
            default_btn.clicked.connect(lambda _checked, aid=action_id: self._reset_keybind(aid))
            grid.addWidget(default_btn, row, 2)

            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda _checked, aid=action_id: self._keybind_edits[aid].clear())
            grid.addWidget(clear_btn, row, 3)

            row += 1

        grid.setRowStretch(row, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        outer_layout.addWidget(scroll)

        # Feedback surface for a keybind-collision auto-clear (see
        # _on_keybind_changed) -- deliberately unbolded, since
        # test_keybinds_tab_renders_menu_section_headers collects section
        # headers by filtering QLabel.text().startswith("<b>").
        self._keybind_warning_label = QLabel("")
        self._keybind_warning_label.setWordWrap(True)
        outer_layout.addWidget(self._keybind_warning_label)

        return tab

    def _on_keybind_changed(self, action_id: str, key_sequence: QKeySequence) -> None:
        text = key_sequence.toString()
        cleared_action_id = settings.set_keybind(action_id, text)
        self._window.apply_keybind(action_id)
        self._window._log_status(f"Keybind changed: {action_id} -> {text or '(cleared)'}")
        if cleared_action_id is None:
            self._keybind_warning_label.clear()
            return
        self._window.apply_keybind(cleared_action_id)
        cleared_edit = self._keybind_edits.get(cleared_action_id)
        if cleared_edit is not None:
            cleared_edit.blockSignals(True)
            cleared_edit.clear()
            cleared_edit.blockSignals(False)
        cleared_label = settings.get_action_label(cleared_action_id)
        self._keybind_warning_label.setText(
            f"'{text}' was already assigned to {cleared_label} -- that binding has been cleared."
        )

    def _reset_keybind(self, action_id: str) -> None:
        default = settings.get_default_keybind(action_id)
        holder = settings.keybind_holder(default, exclude=action_id)
        if holder is not None:
            # Any other action holding this default is necessarily off its own
            # default (test_default_keybinds_have_no_duplicate_sequences pins that
            # the shipped defaults are unique), so a deliberate binding is at
            # stake -- restoring a default must not outrank it. Typing the
            # sequence in is still an explicit override.
            holder_label = settings.get_action_label(holder)
            self._keybind_warning_label.setText(
                f"'{default}' is assigned to {holder_label} -- "
                f"{settings.get_action_label(action_id)}'s default was not restored."
            )
            self._window._log_status(
                f"Keybind unchanged: {action_id} ('{default}' held by {holder_label})"
            )
            return
        # setKeySequence() emits keySequenceChanged, so _on_keybind_changed()
        # handles the actual persist + apply + log.
        self._keybind_edits[action_id].setKeySequence(QKeySequence(default))

    def _add_install_section(self, layout: QVBoxLayout) -> None:
        """Appends the AoE2DE install-path controls into the given layout --
        folded into the General tab (moved out of its own "Game Resources"
        tab) since this is a one-time setup step, not a recurring setting
        someone flips back and forth."""
        install_label = QLabel("AoE2DE install path (optional, for real terrain colors):")
        install_label.setWordWrap(True)
        layout.addWidget(install_label)

        install_row = QHBoxLayout()
        self.install_path_edit = QLineEdit()
        existing_install = asset_source.get_install_path()
        if existing_install is not None:
            self.install_path_edit.setText(str(existing_install))
        install_row.addWidget(self.install_path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_install_path)
        install_row.addWidget(browse_btn)
        layout.addLayout(install_row)

        load_install_btn = QPushButton("Load")
        load_install_btn.clicked.connect(self.load_install_path)
        layout.addWidget(load_install_btn)

        self.install_status_label = QLabel("")
        self.install_status_label.setWordWrap(True)
        layout.addWidget(self.install_status_label)

        # Reflects whatever's actually configured right now (dialog-open
        # time), not just the outcome of the last "Load" click in this
        # session -- e.g. a path set via AOE2DE_INSTALL_PATH or a prior
        # session's config.yaml would otherwise leave this blank forever.
        if existing_install is not None:
            ok, message = asset_source.validate_install_path(existing_install)
            self._set_install_status(message, ok=ok)
        else:
            self._set_install_status("No install path configured -- using flat fallback colors.", ok=False)

    def browse_install_path(self) -> None:
        start_dir = self.install_path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select AoE2DE install folder", start_dir)
        if path:
            self.install_path_edit.setText(path)

    def load_install_path(self) -> None:
        text = self.install_path_edit.text().strip()
        if not text:
            self._set_install_status("Enter or browse to an AoE2DE install path first.", ok=False)
            return

        ok, message = asset_source.validate_install_path(Path(text))
        if not ok:
            self._set_install_status(message, ok=False)
            self._window._log_status(f"Install path rejected ({text}): {message}")
            return

        path = Path(text)
        asset_source.set_install_path_override(path)
        asset_source.save_install_path_to_config(path)
        self._set_install_status(message, ok=True)
        self._window._log_status(f"Install path set to {path}: {message}")
        self._window.refresh_map()
        # The long-lived picker keeps its old swatches otherwise (see refresh_swatches()).
        self._window.terrain_panel.view.refresh_swatches()

    def _set_install_status(self, message: str, ok: bool) -> None:
        color = STATUS_OK_COLOR if ok else STATUS_ERROR_COLOR
        self.install_status_label.setStyleSheet(f"color: {color};")
        self.install_status_label.setText(message)


class MirrorDialog(QDialog):
    """Map mirroring (Stage 1: terrain + elevation). The repo's first
    accept/reject dialog:
    SettingsDialog and DebugLogDialog above are both Close-only and apply
    every change immediately on its own widget signal, so there is no OK/
    Apply/Cancel convention here to copy -- this establishes one, with a
    third "Preview" state in between.

    Preview applies the real edit as a normal undo record (via
    ViewerWindow.on_mirror(), same shape as on_fill()) and leaves the dialog
    open, so the user judges actual rendered pixels. Changing any option, or
    Cancel, undoes that preview first -- guarded by EditHistory.peek_undo()
    so it only auto-undoes if the preview's own record is still the top of
    the stack, never eating an edit made some other way in the meantime.
    Apply keeps whatever the last Preview already applied (or, if nothing
    was previewed under the current options, applies now) and closes.
    """

    def __init__(self, parent: ViewerWindow):
        super().__init__(parent)
        self.setWindowTitle("Mirror Map")
        self.resize(460, 460)
        self._window = parent
        # The DiffRecord Preview last pushed, or None if no preview is
        # currently applied under the CURRENTLY selected options (an option
        # change or Cancel undoes it and resets this to None -- see
        # _undo_own_preview()).
        self._preview_record = None

        layout = QVBoxLayout(self)

        note = QLabel(
            "The map renders as a diamond on screen (isometric projection) -- "
            "mode names below describe the screen tips they pair up, not raw "
            "array directions. The shaded area on the map is the source slice. "
            "The 3-way and 6-way modes split the map into equal wedges by tile distance, "
            "so the wedges look unequal on screen."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QGridLayout()
        row = 0
        form.addWidget(QLabel("Symmetry:"), row, 0)
        self.mode_combo = QComboBox()
        two_way = [m for m in MODES if len(m.group) == 2]
        four_way = [m for m in MODES if len(m.group) == 4]
        eight_way = [m for m in MODES if len(m.group) == 8]
        for mode in two_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        self.mode_combo.insertSeparator(self.mode_combo.count())
        for mode in four_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        self.mode_combo.insertSeparator(self.mode_combo.count())
        for mode in eight_way:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        # Angular modes have an empty `group`, so the length buckets above
        # never see them: they get their own labelled block.
        self.mode_combo.insertSeparator(self.mode_combo.count())
        self.mode_combo.addItem("Approximate (tile-space wedges):")
        header = self.mode_combo.model().item(self.mode_combo.count() - 1)
        header.setFlags(header.flags() & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
        for mode in ANGULAR_MODES:
            self.mode_combo.addItem(mode.label, mode.mode_id)
        form.addWidget(self.mode_combo, row, 1)
        row += 1

        form.addWidget(QLabel("Source slice:"), row, 0)
        self.slice_combo = QComboBox()
        form.addWidget(self.slice_combo, row, 1)
        row += 1
        layout.addLayout(form)

        self.terrain_checkbox = QCheckBox("Terrain (type and blend layer)")
        self.terrain_checkbox.setChecked(True)
        layout.addWidget(self.terrain_checkbox)
        self.elevation_checkbox = QCheckBox("Elevation")
        self.elevation_checkbox.setChecked(True)
        self.elevation_checkbox.setToolTip(
            "Copy elevation too. Off by default for the 3-way and 6-way modes: resampling a "
            "slope skips rows, so almost every hilly map ends up with steps the game rejects"
        )
        layout.addWidget(self.elevation_checkbox)
        # Which kind of mode the Elevation default was last set for, so a
        # switch between lattice and angular modes re-applies it once.
        self._elevation_default_kind = "lattice"
        self.units_checkbox = QCheckBox("Units")
        self.units_checkbox.setChecked(False)
        self.units_checkbox.setToolTip(
            "Rewrite every other slice's units from the source slice. Gates are swapped to "
            "the orientation sibling the mirror turns them into"
        )
        layout.addWidget(self.units_checkbox)

        # Ownership rotation is opt-in and off by default, matching the
        # in-game Map Copy tool's own Change Player option: there is no
        # per-mode rule that is right for every scenario, since the slice
        # count often does not divide the player count.
        owner_row = QHBoxLayout()
        self.ownership_checkbox = QCheckBox("Rotate ownership by")
        self.ownership_checkbox.setChecked(False)
        self.ownership_spin = QSpinBox()
        self.ownership_spin.setRange(1, 8)
        self.ownership_spin.setValue(1)
        self.ownership_spin.setEnabled(False)
        self.ownership_spin.setToolTip(
            "Players per slice, along the scenario's own defined-player list. GAIA is "
            "never rotated. With fewer defined players than slices, ownership wraps around"
        )
        owner_row.addWidget(self.ownership_checkbox)
        owner_row.addWidget(self.ownership_spin)
        owner_row.addStretch(1)
        layout.addLayout(owner_row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addStretch(1)

        btn_row = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.apply_button = QPushButton("Apply")
        self.cancel_button = QPushButton("Cancel")
        btn_row.addStretch(1)
        btn_row.addWidget(self.preview_button)
        btn_row.addWidget(self.apply_button)
        btn_row.addWidget(self.cancel_button)
        layout.addLayout(btn_row)

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.slice_combo.currentIndexChanged.connect(self._on_option_changed)
        self.terrain_checkbox.toggled.connect(self._on_option_changed)
        self.elevation_checkbox.toggled.connect(self._on_option_changed)
        self.units_checkbox.toggled.connect(self._on_option_changed)
        self.ownership_checkbox.toggled.connect(self.ownership_spin.setEnabled)
        self.ownership_checkbox.toggled.connect(self._on_option_changed)
        self.ownership_spin.valueChanged.connect(self._on_option_changed)
        self.preview_button.clicked.connect(self._on_preview)
        self.apply_button.clicked.connect(self._on_apply)
        self.cancel_button.clicked.connect(self.reject)
        self.finished.connect(self._on_finished)

        self._on_mode_changed()  # populates slice_combo and the initial overlay/summary

    def _current_mode(self):
        return MODE_BY_ID[self.mode_combo.currentData()]

    def _on_mode_changed(self) -> None:
        self._undo_own_preview()
        mode = self._current_mode()
        if mode.kind != self._elevation_default_kind:
            # Measured over examples/: 6 of 285 hilly file x angular mode x
            # wedge cases had no seam violation, so angular defaults it off.
            self._elevation_default_kind = mode.kind
            self.elevation_checkbox.blockSignals(True)
            self.elevation_checkbox.setChecked(mode.kind == "lattice")
            self.elevation_checkbox.blockSignals(False)
        self.slice_combo.blockSignals(True)
        self.slice_combo.clear()
        for index, label in enumerate(mode.slice_labels):
            self.slice_combo.addItem(label, index)
        self.slice_combo.blockSignals(False)
        self._on_option_changed()

    def _on_option_changed(self) -> None:
        self._undo_own_preview()
        self._refresh_overlay_and_summary()

    def _compute_plan(self):
        mode = self._current_mode()
        mm = self._window.scenario.map_manager
        slice_index = self.slice_combo.currentData()
        if slice_index is None:
            slice_index = 0
        return plan_mirror(
            mm,
            mode.mode_id,
            slice_index,
            self.terrain_checkbox.isChecked(),
            self.elevation_checkbox.isChecked(),
        )

    def _compute_unit_plan(self, plan):
        """The units half, or None when the Units box is off or this file's
        units are read-only. Shares plan_mirror()'s own source_indices rather
        than re-deriving which tiles are the source."""
        if not self.units_checkbox.isChecked():
            return None
        model = self._window._ensure_unit_edits()
        if model is None:
            # Read-only units: untick rather than just returning None, or
            # _ensure_unit_edits()' warning box reappears on every option
            # change for the rest of the dialog's life.
            self.units_checkbox.setChecked(False)
            return None
        mode = self._current_mode()
        slice_index = self.slice_combo.currentData() or 0
        loaded = self._window.scenario
        return plan_mirror_units(
            loaded.map_manager,
            mode.mode_id,
            slice_index,
            loaded.unit_manager.units,
            plan.source_indices,
            player_ids=player_fields.defined_player_ids(loaded, self._window._pending_options()),
            ownership_steps=self.ownership_spin.value() if self.ownership_checkbox.isChecked() else 0,
            referencing=model.referencing,
            unreachable=plan.unreachable,
        )

    def _refresh_overlay_and_summary(self) -> None:
        if self._window.scenario is None:
            return
        plan = self._compute_plan()
        unit_plan = self._compute_unit_plan(plan)
        mm = self._window.scenario.map_manager
        n = mm.map_width
        tiles = [divmod(idx, n)[::-1] for idx in plan.source_indices]
        axes = self._axis_lines(self._current_mode(), n)
        self._window.map_view.show_mirror_overlay(tiles, axes)

        blockers = []
        if plan.elevation_violations:
            blockers.append(
                f"{len(plan.elevation_violations)} elevation seam violation(s) would exceed "
                f"the +/-1 limit"
            )
        if unit_plan is not None:
            if unit_plan.straddling:
                blockers.append(
                    f"{len(unit_plan.straddling)} unit(s) straddle the symmetry axis "
                    f"(their image would overlap the original)"
                )
            if unit_plan.garrisoned_blockers:
                blockers.append(
                    f"{len(unit_plan.garrisoned_blockers)} unit(s) outside the source slice "
                    f"hold a garrison and cannot be replaced"
                )
        # Angular modes only: what the approximation leaves alone, reported
        # whether or not the plan is blocked.
        leftovers = []
        if plan.unreachable:
            leftovers.append(
                f"{len(plan.unreachable)} corner tile(s) have no source inside the map and stay as they are"
            )
        if unit_plan is not None and unit_plan.off_map:
            leftovers.append(
                f"{len(unit_plan.off_map)} unit image(s) would land off the map or in an untouched "
                f"corner and are skipped"
            )
        if blockers:
            self.summary_label.setText(
                f"{len(plan.changes)} tile(s) would change -- BLOCKED: "
                + "; ".join(blockers)
                + ". Pick a different mode, or move the offending objects."
                + "".join(f" {part[0].upper()}{part[1:]}." for part in leftovers)
            )
            self.summary_label.setStyleSheet(f"color: {STATUS_ERROR_COLOR};")
        else:
            parts = [f"{len(plan.changes)} tile(s) will change"]
            if unit_plan is not None:
                parts.append(
                    f"{len(unit_plan.removals)} unit(s) replaced by {len(unit_plan.images)} image(s)"
                )
                if unit_plan.gates:
                    parts.append(f"{len(unit_plan.gates)} gate(s) reoriented")
                if unit_plan.unsquare_spans:
                    parts.append(
                        f"{len(unit_plan.unsquare_spans)} non-square footprint(s) cannot be "
                        f"reflected onto the other axis"
                    )
            parts.extend(leftovers)
            self.summary_label.setText("; ".join(parts))
            self.summary_label.setStyleSheet("")

    def _axis_lines(self, mode, n: int):
        """Straight symmetry-axis lines, Flat style only -- Stepped/Sloped's
        projected geometry makes a literal straight line non-trivial to
        place correctly, and the shaded source slice (tile-exact in every
        style via _tile_polygon) already conveys the boundary on its own.
        One line per reflection generator actually in the mode's group;
        pure-rotation modes (5, 6, 10, 11) have no reflection axis and draw
        none; mode 12 draws its three mirror axes."""
        if self._window.map_view._terrain_style != "flat":
            return []
        tp = self._window.map_view._tile_pixels
        size = n * tp
        lines = []
        if mode.angular is not None:
            # Scene space is tile space scaled, so a tile-space axis is a
            # straight scene line through the centre, clipped to the square.
            half = size / 2
            for dx, dy in angular_mirror_axes(mode.angular):
                t = half / max(abs(dx), abs(dy))
                lines.append((QPointF(half + t * dx, half + t * dy), QPointF(half - t * dx, half - t * dy)))
            return lines
        if "mx" in mode.group:  # u=0 -- the vertical centre line
            lines.append((QPointF(size / 2, 0), QPointF(size / 2, size)))
        if "my" in mode.group:  # v=0 -- the horizontal centre line
            lines.append((QPointF(0, size / 2), QPointF(size, size / 2)))
        if "d" in mode.group:  # v-u=0 -- the main diagonal, (0,0)-(n-1,n-1)
            lines.append((QPointF(0, 0), QPointF(size, size)))
        if "a" in mode.group:  # u+v=0 -- the anti-diagonal, (n-1,0)-(0,n-1)
            lines.append((QPointF(size, 0), QPointF(0, size)))
        return lines

    def _undo_own_preview(self) -> None:
        if self._preview_record is not None and self._window.edit_history.peek_undo() is self._preview_record:
            self._window.undo()
        self._preview_record = None

    def _on_preview(self) -> None:
        self._undo_own_preview()
        plan = self._compute_plan()
        dirty = self._window.on_mirror(plan, self._compute_unit_plan(plan))
        if dirty:
            self._preview_record = self._window.edit_history.peek_undo()
        self._refresh_overlay_and_summary()

    def _on_apply(self) -> None:
        if self._preview_record is not None and self._window.edit_history.peek_undo() is self._preview_record:
            self.accept()
            return
        plan = self._compute_plan()
        dirty = self._window.on_mirror(plan, self._compute_unit_plan(plan))
        if dirty is None:
            return  # refused (elevation violations) -- status already logged, stay open
        self.accept()

    def _on_finished(self, result: int) -> None:
        if result == QDialog.Rejected:
            self._undo_own_preview()
        self._window.map_view.clear_mirror_overlay()


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Tool-overflow state (Stage 3 of the tool-overflow plan). Set
        # before the resize() call below: that call fires resizeEvent(),
        # which calls _apply_toolbar_overflow(), before _build_toolbar()
        # has run -- _tool_overflow_measured=False makes that call a no-op
        # until showEvent() has cached real (post-realization) widths.
        self._tool_overflow_measured = False
        self._tool_button_widths: dict[str, int] = {}
        self._pinned_toolbar_width = 0
        self._toolbar_overflow_updating = False

        width, height = settings.get_window_size()
        self.resize(width, height)
        self.setMinimumSize(settings.MIN_WINDOW_WIDTH, settings.MIN_WINDOW_HEIGHT)

        self.scenario: LoadedScenario | None = None
        # True while the current scenario came from File > New and has never been
        # saved: its LoadedScenario.path is UNTITLED_PATH (a display-only sentinel,
        # not a real file), so Save As must default to UNTITLED_NAME instead.
        # Cleared by a successful Save As, by File > Open, and by Close.
        self._untitled = False
        # Holds whichever style's chunk cache is current -- an IsoChunkCache
        # (Stepped), a FlatChunkCache (Flat, since Phase B-E; previously a
        # separate self._img numpy array, retired in that phase), or a
        # SlopedChunkCache (Sloped, since Track C3). Set by
        # _render_current(), matching self._terrain_style/self.map_view's
        # own paired state.
        self._cache: IsoChunkCache | FlatChunkCache | SlopedChunkCache | None = None
        # Stepped and Sloped, set together with self._cache by
        # _render_current() -- the exact elevation snapshot and
        # IsoProjection self._cache was built from, and the SAME
        # array/object self.map_view holds (passed by reference into
        # MapView.set_source(), never copied) so dirty_screen_bbox_iso()/
        # dirty_screen_bbox_sloped()'s in-place elevations mutation is
        # visible to the viewer's own hit-testing, and to self._cache
        # itself, without a separate update step (Risk #6 in the parent
        # plan). None in Flat mode.
        self._iso_elevations: np.ndarray | None = None
        self._iso_proj: iso_geometry.IsoProjection | None = None
        self.mode = "view"
        self._current_tool = "pan"
        # The Cliff tool's in-flight chain (Track B Stage 2), None between
        # strokes. Initialised here rather than only in _begin_cliff_stroke()
        # so a release that somehow arrives without a matching press is a
        # no-op instead of an AttributeError.
        self._cliff_stroke: cliff_chain.ChainStroke | None = None
        # One of terrain_style.TERRAIN_STYLES -- see the toolbar combo built
        # in _build_toolbar(). Defaults to "stepped" (Flat alone no longer
        # shows elevation at all -- see render_tile()'s docstring). Persists
        # across a File > Open (a freshly loaded scenario renders in
        # whichever style was already selected), unlike self._current_tool
        # which File > Close resets to "pan".
        self._terrain_style = "stepped"
        # True for the duration of load_scenario()/_render_current()'s
        # blocking work -- an explicit guard against re-entering any of
        # load_scenario/refresh_map/on_terrain_style_changed while one is
        # already running, on top of (not instead of) setEnabled(False):
        # that call only blocks Qt from *delivering new input events* to
        # disabled widgets, it doesn't stop a direct/programmatic call into
        # one of these methods (e.g. from a QTimer callback) from reaching
        # them regardless of enabled state. This flag is what actually
        # makes such a call a no-op rather than a reentrant interleave.
        self._busy = False
        # Tools > Map Analysis results, modeless and reused; see _show_analysis().
        self._analysis_dialog: AnalysisDialog | None = None
        # The in-flight first-paint report a load is accumulating canvas
        # paint time into, or None when nothing is pending. See
        # _on_canvas_paint_timed().
        self._pending_paint_report: dict | None = None
        # Restarted by every timed paint, so it fires once the event loop
        # first goes idle and one line covers however many paints Qt chose
        # to split the composite across.
        self._paint_report_timer = QTimer(self)
        self._paint_report_timer.setSingleShot(True)
        self._paint_report_timer.setInterval(0)
        self._paint_report_timer.timeout.connect(self._emit_paint_report)
        # The incremental level warm's driver (2026-09-04 plan). One per
        # window, not per document: start() replaces whatever it was doing,
        # and _cancel_warms() below is called by every path that mutates
        # the scenario, the elevations array, or the cache itself. Idle until
        # something queues levels on it, so a window that never opens a file
        # never schedules a tick.
        self._level_warmer = level_warm.LevelWarmer()
        # The margin-ring chunk warm's driver (2026-09-07 plan's Phase A).
        # Same one-per-window/idle-until-queued shape as _level_warmer above,
        # started only from _on_viewport_changed() and stopped by
        # _cancel_warms() alongside it.
        self._margin_warmer = margin_warm.MarginWarmer()
        # The load-time margin warm's driver (2026-09-07 plan's load-time
        # margin warm). A SECOND MarginWarmer instance, not a third class --
        # same tick contract as _margin_warmer above, but a different region
        # (a neighbour mip's own viewport-sized patch, queued once that
        # mip's sprite layer finishes) that must never cancel or be
        # cancelled by the navigation-driven ring above. _load_warm_queue
        # holds (mip, chunks) pairs not yet started because a previous one
        # was still draining -- see _pump_load_warm().
        self._load_warmer = margin_warm.MarginWarmer()
        self._load_warm_queue: list[tuple[int, list[tuple[int, int]]]] = []
        # The last viewport_chunk_target() _on_viewport_changed() saw, kept
        # here (not on MapView, which tracks its own copy for a different
        # purpose -- deciding when to stop polling) so a pan's DIRECTION can
        # be diffed across fires and handed to margin_warm.ring_chunks() as
        # `lead`. None means "no baseline yet" (first fire, or right after a
        # cancel), which ring_chunks() reads the same way as lead=(0, 0).
        self._last_viewport_chunk_target: tuple[int, int, int, int, int] | None = None
        # One history per loaded scenario -- load_scenario()/close_scenario()
        # call .reset() so it never leaks state across files. See
        # descape.edit_history for why this is the single choke point every
        # terrain/elevation edit goes through.
        self.edit_history = EditHistory()
        # -- Autosave (Settings > Saving) -------------------------------
        # Repeating, interval from settings; armed by _reset_autosave_timer().
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        # Single-shot, for a tick that arrived while the window was busy or
        # mid-stroke. Connects back to the same tick, so the settle rule in
        # _autosave_tick() is what stops it firing as the next stroke begins.
        self._autosave_retry_timer = QTimer(self)
        self._autosave_retry_timer.setSingleShot(True)
        self._autosave_retry_timer.timeout.connect(self._autosave_tick)
        # The history cursor the last successful autosave captured, so a
        # document edited once and then left alone doesn't re-serialize
        # identical bytes every interval for the rest of the session.
        self._autosaved_at_cursor: int | None = None
        # Set whenever a tick or retry found a stroke in progress, cleared by
        # the retry that then re-armed -- see _autosave_tick()'s settle rule.
        self._stroke_seen_since_retry = False
        # Per-document, reassigned by load_scenario(): an untitled document
        # has no path to key its autosaves on, so it keys on this instead.
        self._doc_id = uuid4().hex
        # File > Recover from Autosave…, modeless and reused.
        self._recover_dialog: RecoverAutosaveDialog | None = None
        # Phase 4b's trigger edit state, on the same history as terrain edits.
        # Created lazily by the editing UI, never at file-open time: parsing
        # triggers costs 3.5s on the worst corpus file and 4a deliberately kept
        # that off the open path. None until a trigger edit is actually made,
        # which is also what makes save_as()'s triggers= argument inert until
        # then. Owned here rather than on TriggerPanel because that panel
        # clears and re-parses on every entry into Triggers mode, so anything
        # it owned would die on a mode switch.
        self.trigger_edits: TriggerEditModel | None = None
        # The Map Options panel's write model, on the same lazy terms and for
        # the same reason: built on the first real option edit, so a document
        # whose options were only browsed still saves byte-identically.
        self.option_edits: OptionsEditModel | None = None
        # Phase 3.5b's unit edit state, on the same lazy terms as the two
        # above -- built on the first real unit edit (place/move/delete/
        # reassign), so a document whose units were only browsed/selected
        # still saves byte-identically.
        self.unit_edits: UnitEditModel | None = None
        # Messages mode's write model, on the same lazy terms as the three
        # above -- built on the first real edit to Instructions/Hints/Victory/
        # Loss/History/Scouts or a string-table id, so a document whose
        # Messages were only browsed still saves byte-identically.
        self.message_edits: MessagesEditModel | None = None
        # Phase 2.8's region clipboard: a single slot, not a manager --
        # overwritten on every Copy Region, None means empty. Carries
        # terrain+elevation+units together (RegionBlock), unlike v2.7's
        # retired per-tool clipboard -- see copy_region()/paste_region().
        # Deliberately survives close_scenario() (a clipboard history
        # outliving the file it was copied from is normal clipboard
        # semantics, and Paste is already disabled with no map loaded via
        # _update_tool_enabled()) -- only self._region below is map-relative
        # and needs clearing then.
        self._clipboard_history = clipboard_history.ClipboardHistory()
        # GH #27's trigger clipboard: one slot, session-only, and like the
        # region clipboard it survives close_scenario(), so it pastes into
        # another file (GH #3). Paste is same scenario_version only.
        self._trigger_clipboard: trigger_clipboard.TriggerBlock | None = None
        self._clipboard_dialog: ClipboardHistoryDialog | None = None
        # GH #30's History window. Assigned before on_change is hooked up
        # below, since reset()/mark_saved() fire that hook and the refresh
        # reads this attribute.
        self._history_dialog: EditHistoryDialog | None = None
        # Every EditHistory mutation repaints the History window, rather than
        # the ~30 push sites each growing a refresh call of their own.
        self.edit_history.on_change = self._refresh_history_dialog
        # Armed by a paste that actually wrote something; while it is set, the
        # committed region can be dragged to re-place that same snapshot
        # somewhere else. Always assigned through _set_paste_move(), never
        # directly, so MapView's own flag cannot drift from it.
        self._paste_move: region_clipboard.PasteMove | None = None
        # The Select tool's committed region, half-open tile-space (tx0, ty0,
        # tx1, ty1) -- None means no selection. Mirrored on MapView (its own
        # copy backs the overlay); this is the copy Copy Region/undo-kind
        # gating reads. Set by on_region_selected(), the one handler
        # MapView's drag-commit/Escape-clear and this window's Select
        # All/Deselect all funnel through.
        self._region: tuple[int, int, int, int] | None = None
        # Last-accepted ScatterDialog state, sticky for this session only.
        # Nothing here is persisted, so there is no new settings key.
        self._scatter_defaults: dict = {}
        # Updated on every mouse move by on_hover() regardless of whether a
        # scenario is loaded -- copy/paste fire from a keyboard shortcut, not
        # a mouse click, so they need "what tile is under the mouse right
        # now" captured at hover time rather than at the moment the key is
        # pressed (MapView has no notion of "current tile" itself).
        self._hover_tile: tuple[int, int] | None = None
        # A collection from day one, though phase 3 only ever holds 0 or 1 --
        # see on_click_select() for why. Holds (player_id, reference_id)
        # keys, never entries or list positions.
        self._selection: list[tuple[int, int]] = []
        # Cycle Variant's randomize source; tests reseed it.
        self._variant_rng = random.Random()
        # Repeat-click cycling through a unit stack: (anchor tile, index into
        # that tile's stack members, key selected). -1 means the first click
        # picked a unit outside the group. Only trusted while _selection still
        # equals [key], so a selection changed any other way restarts it.
        self._stack_cycle: tuple[tuple[int, int], int, tuple[int, int]] | None = None
        # GH #75: (key, status suffix) of a plain click on a group member,
        # finished on release if no drag started; and the press's cover tile.
        self._pending_collapse: tuple[tuple[int, int], str] | None = None
        self._drag_grab_tile: tuple[int, int] | None = None
        # The drag key whose over-cap preview already logged its status line.
        self._ghost_cap_logged: tuple[int, int] | None = None
        # Convert brush: the model whose edit is open for the current drag
        # (None outside a stroke), and the timer coalescing its live repaint.
        self._convert_model: UnitEditModel | None = None
        # Trigger Pick from map's target, (trigger, entry kind, entry index,
        # field, is_list), or None when disarmed. Read by _needs_unit_index().
        self._unit_picker: tuple | None = None
        # Built on first ask, dropped on any unit mutation or document change.
        self._unit_ref_index: unit_references.ReferenceIndex | None = None
        self._convert_done: set[tuple[int, int]] = set()
        self._convert_touched_tiles: set[tuple[int, int]] = set()
        # One UnitSplice per reassign not yet handed to the cache, in mutation order.
        self._convert_pending_splices: list[UnitSplice] = []
        self._convert_refresh_timer = QTimer(self)
        self._convert_refresh_timer.setSingleShot(True)
        self._convert_refresh_timer.timeout.connect(self._flush_convert_refresh)
        self._update_title()

        left = QVBoxLayout()

        self.hover_label = QLabel(HOVER_IDLE_TEXT)
        self.hover_label.setWordWrap(True)
        left.addWidget(self.hover_label)

        # GH #5: one player's breakdown at a time; item data is the player id (0 = GAIA).
        self.stats_player_combo = QComboBox()
        self.stats_player_combo.setEnabled(False)
        self.stats_player_combo.currentIndexChanged.connect(lambda _index: self._update_player_stats())
        left.addWidget(self.stats_player_combo)
        self.player_stats_rows: list[tuple[str, str, str]] = []  # what the label shows, for tests
        self.player_stats_label = QLabel("")
        self.player_stats_label.setTextFormat(Qt.RichText)
        self.player_stats_label.setWordWrap(True)
        self.player_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.addWidget(self.player_stats_label)

        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        left.addWidget(self.info, stretch=1)

        info_widget = QWidget()
        info_widget.setLayout(left)

        # The left column swaps contents per mode rather than growing a third
        # column: page 0 is the map-statistics panel every mode has always had,
        # page 1 the trigger browser, page 2 the unit inspector, page 3 the
        # scenario-wide settings form. MapView stays live beside any of them.
        # Indices are named in _LEFT_PAGE_* above -- keep addWidget order and
        # those constants in step.
        self.trigger_panel = TriggerPanel(
            on_trigger_field=self.set_trigger_field,
            on_entry_field=self.set_entry_fields,
            on_trigger_structural=self.trigger_structural_edit,
            on_entry_structural=self.entry_structural_edit,
            on_variable_structural=self.variable_structural_edit,
            on_selection_changed=self._on_trigger_selection_changed,
            on_tag_rename=self.rename_trigger_tag,
            on_tag_remove=self.remove_trigger_tag,
        )
        self.trigger_panel.on_focus_changed = self._on_trigger_focus_changed
        self.trigger_panel.describe_unit_reference = self._describe_unit_reference
        self.trigger_panel.on_pick_unit = self._on_pick_unit_requested
        # GH #41: (trigger index, entry ref or None) the map overlay draws, or
        # None. Held here, not read off the panel, so an undo in View mode can
        # re-derive it while the panel sits unrefreshed.
        self._trigger_overlay_ref: tuple[int, tuple[str, int] | None] | None = None
        self.map_options_panel = MapOptionsPanel(on_option_field=self.set_option_field)
        self.players_panel = PlayersPanel(
            on_player_field=self.set_player_field,
            on_player_count=self.set_player_count,
            on_disables_requested=self._show_disables_dialog,
            on_set_view=self.set_player_view,
            on_go_to_view=self.go_to_player_view,
            on_reset_view=self.reset_player_view,
            on_player_selected=self._on_player_selected,
        )
        self.diplomacy_panel = DiplomacyPanel(
            on_diplomacy_field=self.set_diplomacy_field, on_option_field=self.set_option_field
        )
        self.messages_panel = MessagesPanel(on_message_field=self.set_message_field)
        self.units_panel = UnitsPanel(
            on_unit_field=self._on_unit_field_changed,
            on_place_requested=self._on_units_panel_place_requested,
            on_garrison_add=self._on_garrison_add,
            on_garrison_delete=self._on_garrison_delete,
            on_garrison_navigate=self._on_garrison_navigate,
        )
        self.terrain_panel = TerrainPanel()
        self.terrain_panel.set_hover_text(HOVER_IDLE_TEXT)
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(info_widget)
        self.left_stack.addWidget(self.trigger_panel)
        self.left_stack.addWidget(self.units_panel)
        self.left_stack.addWidget(self.map_options_panel)
        self.left_stack.addWidget(self.players_panel)
        self.left_stack.addWidget(self.diplomacy_panel)
        self.left_stack.addWidget(self.messages_panel)
        self.left_stack.addWidget(self.terrain_panel)

        self.map_view = MapView(
            self.on_hover,
            self.on_edit_stroke_start,
            self.on_edit_stroke_tiles,
            self.on_edit_stroke_end,
            self.on_click_edit,
            self.on_shape_commit,
            self.on_click_select,
            self.on_unit_place,
            self.on_unit_move,
            self.on_unit_nudge,
            self.on_unit_delete,
            self.on_marquee_select,
            self.on_region_selected,
            self.on_region_move,
            self.on_unit_drag_preview,
            self.on_ruler_measured,
            self.on_ruler_changed,
            # Placeholder: the real target, self._update_zoom_status, reads
            # self.zoom_status_label, which _build_status_bar() below hasn't
            # created yet. A synchronous resizeEvent between here and there
            # (splitter/geometry setup) would otherwise call it too early and
            # raise AttributeError -- see the reassignment after
            # _build_status_bar().
            lambda: None,
        )
        # GH #98: a wall const picked in the catalog makes Place Unit a drag.
        self.map_view.set_place_shape_query(self._place_shape)
        # GH #75: a click on a group member collapses on release, and a drag moves the group.
        self.map_view.set_unit_drag_hooks(self.on_unit_click_release, self._is_group_key)

        # Always-visible short status history, distinct from both the
        # transient single-line QMainWindow.statusBar() message and the
        # Help > Debug Log dialog (a separate popup, not always on screen).
        # _log_status() is the one place that feeds all three destinations
        # that matter for a given message.
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        line_height = self.status_log.fontMetrics().lineSpacing()
        self.status_log.setMaximumBlockCount(200)
        self._update_log_min_height()

        # The log sits beside the left column rather than under it, so the
        # column runs the full height of the central widget. A vertical split
        # rather than a fixed row makes the height draggable; it persists via
        # settings.get_log_height, the same way the horizontal split does.
        log_split = QSplitter(Qt.Vertical)
        log_split.addWidget(self.map_view)
        log_split.addWidget(self.status_log)
        log_split.setStretchFactor(0, 1)
        log_split.setStretchFactor(1, 0)
        log_split.setChildrenCollapsible(False)
        self.map_view.setMinimumWidth(settings.MIN_SPLIT_PANE)
        # setChildrenCollapsible(False) alone bottoms out at QGraphicsView's own
        # 70 px minimumSizeHint, which on a letterboxed render shows only
        # background -- collapsed in every sense but Qt's. This floor is free:
        # measured, the window's minimumSizeHint height is 493 with or without
        # it, because left_stack's own 410 dominates the content splitter.
        self.map_view.setMinimumHeight(settings.MIN_MAP_PANE)
        # Stretch factor 0 pins the log to exactly the second number while the
        # map absorbs every resize, so the first one is a don't-care (measured:
        # 100, 712 and 10000 all give the same result once shown).
        log_split.setSizes([10_000, settings.get_log_height() or line_height * 4 + 12])
        self.log_splitter = log_split

        # A splitter rather than the old setMaximumWidth(340): a trigger list
        # plus its condition/effect detail does not fit in 340 px, and picking a
        # second hardcoded width would only have to be re-picked when phase 4b
        # adds a field editor. Position persists (settings.get_split_sizes).
        content = QSplitter(Qt.Horizontal)
        content.addWidget(self.left_stack)
        content.addWidget(log_split)
        content.setStretchFactor(0, 0)
        content.setStretchFactor(1, 1)
        content.setChildrenCollapsible(False)
        self.left_stack.setMinimumWidth(settings.MIN_SPLIT_PANE)
        saved_split = settings.get_split_sizes()
        content.setSizes(list(saved_split) if saved_split else [340, 900])
        self.content_splitter = content

        self.setCentralWidget(content)

        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()
        # Wired here rather than passed into MapView's constructor -- see the
        # placeholder lambda left there.
        self.map_view._on_zoom_changed = self._update_zoom_status
        self._update_zoom_status()
        # A2.5: the margin warm's entry point (2026-09-07 plan). No ordering
        # trap here the way _on_zoom_changed has -- _on_viewport_changed()
        # touches only self._cache/settings/self.map_view, none of which
        # need _build_status_bar() to have run first.
        self.map_view.on_viewport_changed = self._on_viewport_changed
        self.map_view.on_unit_pick = self._on_unit_picked
        self.map_view.on_unit_picker_cancel = self.disarm_unit_picker
        self._build_keybind_actions()
        self._update_tool_enabled()
        # Once per launch: this is what bounds the untitled autosave keys
        # doc_key() deliberately leaves unbounded by count. Never fatal --
        # a failure here must not stop the window from opening.
        try:
            autosave.prune_stale()
        except OSError as e:
            debug_log.log(f"Autosave prune skipped: {type(e).__name__}: {e}")
        self._reset_autosave_timer()

        # Connected only now that self.map_view exists, then set to match
        # MapView's own default (isometric on) so the action reflects reality
        # without needing map_view to already exist at widget-creation time.
        self.iso_action.toggled.connect(self._on_iso_toggled)
        self.iso_action.setChecked(self.map_view._isometric)
        # Has nothing to do while Elevation View = Stepped (today's default
        # -- see self._terrain_style above), same gating on_terrain_style_
        # changed() applies on every later switch.
        self.iso_action.setEnabled(self._terrain_style == "flat")

        # Registered last, not at the top of __init__: the crash hook's
        # no-host fallback exists precisely for an exception during this
        # constructor, and registering early would make a mid-__init__
        # crash find a "live" host and queue a dialog into an event loop
        # that never starts, surfacing nowhere -- sweep included.
        register_crash_host(self)

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        # One action per real AoE2:DE map size (descape.scenario_new.STANDARD_MAP_SIZES
        # -- confirmed empirically against every real scenario in examples/, see that
        # module's docstring), each generated on demand from the shipped 120x120 donor
        # rather than loaded from its own file. new_action stays the 120x120 entry, but
        # Ctrl+N lives on new_custom_action below instead -- picking a size is more
        # useful as the keyboard-driven default than always defaulting to Tiny.
        new_menu = file_menu.addMenu("&New Map")
        self.new_action = QAction(
            f"{BLANK_TEMPLATE_TILES}×{BLANK_TEMPLATE_TILES} ({STANDARD_MAP_SIZE_NAMES[BLANK_TEMPLATE_TILES]})",
            self,
        )
        self.new_action.triggered.connect(lambda: self.new_map(BLANK_TEMPLATE_TILES))
        new_menu.addAction(self.new_action)
        self.new_size_actions = [self.new_action]
        for tiles in STANDARD_MAP_SIZES:
            if tiles == BLANK_TEMPLATE_TILES:
                continue
            action = QAction(f"{tiles}×{tiles} ({STANDARD_MAP_SIZE_NAMES[tiles]})", self)
            # checked=False must come first: QAction.triggered passes a
            # checked: bool positional arg, and PyQt5 binds it into whichever
            # parameter is first available -- a lambda with only `tiles=tiles`
            # gets `checked` bound into `tiles` instead of the captured
            # default, silently passing tiles=0/False (confirmed: hangs the
            # test suite on a QMessageBox.critical() for the resulting
            # nonexistent blank_0x0.aoe2scenario, with nothing able to click
            # "OK" in an offscreen run).
            action.triggered.connect(lambda checked=False, tiles=tiles: self.new_map(tiles))
            new_menu.addAction(action)
            self.new_size_actions.append(action)
        new_menu.addSeparator()
        # Deliberately NOT appended to new_size_actions -- that list means "one
        # action per preset size", and new_map_custom() is the only path that can
        # ever pop the >LARGE_MAP_CONFIRM_TILES confirmation dialog (see its own
        # docstring for why the presets, including 480, never confirm).
        self.new_custom_action = QAction("&Custom size…", self)
        # Same checked=False trap as above -- applies even with no captured value,
        # since QAction.triggered still passes a positional bool.
        self.new_custom_action.triggered.connect(lambda checked=False: self.new_map_custom())
        new_menu.addAction(self.new_custom_action)
        self.open_action = QAction("&Open .aoe2scenario…", self)
        self.open_action.triggered.connect(self.open_file)
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_files_menu()
        self.close_action = QAction("&Close Map", self)
        self.close_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.close_action.triggered.connect(self.close_scenario)
        file_menu.addAction(self.close_action)
        file_menu.addSeparator()
        # In-place Save was withheld until in-game round-trip verification
        # passed, on the theory that a write-path bug should only ever be
        # able to damage a file the user explicitly named as a new
        # destination, not their only copy. That verification has since
        # passed repeatedly (terrain edits confirmed 2026-08-05; trigger
        # edits, including walls and variables, confirmed through
        # 2026-08-27).
        self.save_action = QAction("&Save", self)
        self.save_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.save_action.triggered.connect(self.save)
        file_menu.addAction(self.save_action)
        self.save_as_action = QAction("Save &As…", self)
        self.save_as_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(self.save_as_action)
        # Enabled unconditionally, unlike the save actions: recovery is most
        # needed when nothing is open.
        self.recover_autosave_action = QAction("&Recover from Autosave…", self)
        self.recover_autosave_action.triggered.connect(self.show_recover_autosave)
        file_menu.addAction(self.recover_autosave_action)
        file_menu.addSeparator()
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setEnabled(False)  # re-gated by _update_edit_actions()
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setEnabled(False)  # re-gated by _update_edit_actions()
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        # v2.7 copy/paste. No setShortcut() here, or anywhere else
        # in this whole menu -- every Edit-menu action's shortcut comes from
        # the settings-backed keybind system instead (see
        # _build_keybind_actions()/apply_keybind()), same as every tool
        # action, so its shortcut comes from settings.REBINDABLE_ACTIONS's
        # matching "edit_*" entry.
        self.copy_action = QAction("&Copy Region", self)
        self.copy_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        # Mode-dispatched (GH #27): triggers in Triggers mode, a region otherwise.
        self.copy_action.triggered.connect(self._dispatch_copy)
        edit_menu.addAction(self.copy_action)
        self.paste_action = QAction("&Paste Region", self)
        self.paste_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_action.triggered.connect(self._dispatch_paste)
        edit_menu.addAction(self.paste_action)
        # No setShortcut(): the whole menu takes its shortcuts from the
        # keybind system. Ships unbound, like Settings… directly below.
        self.clipboard_history_action = QAction("Clipboard &History…", self)
        self.clipboard_history_action.triggered.connect(self._show_clipboard_history)
        edit_menu.addAction(self.clipboard_history_action)
        # GH #30's edit-history window. Ships unbound like the two dialog
        # openers either side of it; "Histor&y" because Clipboard History
        # above already owns H in this menu.
        self.history_action = QAction("Histor&y…", self)
        self.history_action.triggered.connect(self._show_history_dialog)
        edit_menu.addAction(self.history_action)
        # Phase 2.8: the Select tool's whole-map-select / clear-selection
        # pair -- both share on_region_selected() with MapView's own
        # drag-commit/Escape-clear paths (see that method's docstring).
        self.select_all_action = QAction("Select &All", self)
        self.select_all_action.triggered.connect(self._dispatch_select_all)
        edit_menu.addAction(self.select_all_action)
        self.deselect_action = QAction("&Deselect", self)
        self.deselect_action.triggered.connect(self._dispatch_deselect)
        edit_menu.addAction(self.deselect_action)
        # Gated on the committed region alone, like Copy/Paste above, so it
        # needs no round trip back to Terrain mode to stay reachable.
        self.scatter_action = QAction("Sca&tter Units in Region…", self)
        self.scatter_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.scatter_action.triggered.connect(self.scatter_units_in_region)
        edit_menu.addAction(self.scatter_action)

        # Phase 3.5b's b3. NOT a ToolDef: Rotate acts on the existing
        # selection the way nudge and delete do, so there is no click mode to
        # enter. Built here, in the menu pass, because _build_toolbar() runs
        # after this one and reuses the two fine actions as toolbar buttons.
        #
        # The coarse pair exists as its own two actions rather than as a Shift
        # modifier on the fine ones: a QAction's shortcut IS the key
        # combination, so unlike the arrow-key nudge there is no live
        # `modifiers` for this path to read.
        edit_menu.addSeparator()
        rotate_menu = edit_menu.addMenu("&Rotate Selection")
        self._rotate_actions: list[QAction] = []
        for attr, label, handler in (
            ("rotate_ccw_action", "Rotate ↺", lambda: self.on_unit_rotate(-1)),
            ("rotate_cw_action", "Rotate ↻", lambda: self.on_unit_rotate(1)),
            ("rotate_ccw_coarse_action", "Rotate ↺ (Quarter Turn)",
             lambda: self.on_unit_rotate_coarse(-1)),
            ("rotate_cw_coarse_action", "Rotate ↻ (Quarter Turn)",
             lambda: self.on_unit_rotate_coarse(1)),
        ):
            action = QAction(label, self)
            action.setEnabled(False)  # re-gated by _update_tool_enabled()
            action.setToolTip(
                "Turn the selected units (Units mode). Gates cycle through their four "
                "orientations instead, since a gate stores its facing in its object type. "
                "Walls and most GAIA objects store a graphic variant in the rotation field "
                "and are skipped; use Edit > Cycle Variant for trees and scenery"
            )
            action.triggered.connect(handler)
            rotate_menu.addAction(action)
            setattr(self, attr, action)
            self._rotate_actions.append(action)

        # Cycle Variant: trees, doodads and scenery store a graphic-variant
        # index in `rotation`. Same selection-acting shape as Rotate above.
        variant_menu = edit_menu.addMenu("Cycle &Variant")
        self._variant_actions: list[QAction] = []
        for attr, label, handler in (
            ("variant_prev_action", "Previous Variant", lambda: self.on_unit_variant(-1)),
            ("variant_next_action", "Next Variant", lambda: self.on_unit_variant(1)),
            ("variant_random_action", "Random Variant", lambda: self.on_unit_variant(randomize=True)),
        ):
            action = QAction(label, self)
            action.setEnabled(False)  # re-gated by _update_tool_enabled()
            action.setToolTip(
                "Change the graphic variant of the selected trees, plants, rocks and scenery "
                "(Units mode). Walls, cliffs and gates are skipped: the game derives their "
                "shape from their neighbours"
            )
            action.triggered.connect(handler)
            variant_menu.addAction(action)
            setattr(self, attr, action)
            self._variant_actions.append(action)

        # GH #75. Not a double-click: MapView forwards the second press of a
        # pair as a normal press on purpose, so a fast double-click already cycles twice.
        self.select_stack_action = QAction("Select Whole &Stack", self)
        self.select_stack_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.select_stack_action.setToolTip(
            "Widen the selection to every unit stacked with a selected one (Units mode), "
            "so a drag moves the whole stack. A marquee over a stack selects it all too"
        )
        self.select_stack_action.triggered.connect(self.on_select_whole_stack)
        edit_menu.addAction(self.select_stack_action)

        edit_menu.addSeparator()
        # GH #57's Disabled Objects dialog, duplicated from the Players
        # panel's own button so it can be keybound. No setShortcut() here,
        # same reason as every other menu action in this method -- the
        # shortcut comes from settings.REBINDABLE_ACTIONS' "edit_disables".
        self.disables_action = QAction("&Disabled Objects…", self)
        self.disables_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.disables_action.triggered.connect(self._show_disables_dialog)
        edit_menu.addAction(self.disables_action)

        self.settings_action = QAction("&Settings…", self)
        self.settings_action.triggered.connect(self._show_settings)
        edit_menu.addAction(self.settings_action)

        # Map mirroring (Stage 1: terrain + elevation). New top-level menu
        # between Edit and View, one action. No setShortcut() here, same
        # reason as every other menu action in this method -- the shortcut
        # comes from the settings-backed keybind system
        # (settings.REBINDABLE_ACTIONS's "map_mirror" row).
        map_menu = menu_bar.addMenu("&Map")
        self.mirror_action = QAction("&Mirror Map…", self)
        self.mirror_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.mirror_action.triggered.connect(self._show_mirror_dialog)
        map_menu.addAction(self.mirror_action)

        view_menu = menu_bar.addMenu("&View")
        self.iso_action = QAction("&Isometric View (game-style)", self)
        self.iso_action.setCheckable(True)
        self.iso_action.setToolTip(
            "Flat Elevation View only. Switches Flat to a real isometric "
            "render -- terrain as iso diamonds, units as upright ground-"
            "anchored sprites -- instead of the plain top-down grid with a "
            "view rotation applied. Stepped already renders its own real "
            "per-tile projection into the image, so this has nothing left to "
            "do there and is disabled while Elevation View = Stepped."
        )
        view_menu.addAction(self.iso_action)

        # P3-g. Lives in View, not Filters, because it is not a filter: it
        # changes HOW units are drawn (real .sld sprites instead of coloured
        # marks), not WHICH ones. That also puts it beside iso_action, the
        # only other style-gated view boolean -- the two are near-mirrors,
        # iso_action being Flat-only and this one Stepped-only. Both are
        # re-gated by _update_tool_enabled(), which is also where the "not yet
        # in this view" tooltips are set.
        #
        # Per-session state only, deliberately NOT persisted to settings.py,
        # same rule the unit filter is held to: a view state that survived a
        # restart would present an unexplained map with the explanation two
        # clicks deep in a menu the user has no reason to open.
        #
        # _sprites_enabled is the WINDOW's copy, distinct from the live cache's
        # own flag: a cache is rebuilt from scratch on every style switch and
        # every file open, so something outside it has to remember the choice
        # across those rebuilds. _render_current() re-applies it, exactly the
        # way it re-applies _unit_filter.
        self._sprites_enabled = True
        # (overrides_dict, {unit_key: rotation_override}) from the mid-drag
        # preview -- see _ghost_rotation_override() for why it is keyed on the
        # dict object itself rather than on a generation number.
        self._ghost_rotation: tuple[dict, dict[tuple[int, int], float | None]] | None = None
        self.show_sprites_action = QAction("Show sprites", self, checkable=True, checked=True)
        self.show_sprites_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.show_sprites_action.toggled.connect(self._on_sprites_toggled)
        view_menu.addAction(self.show_sprites_action)

        # Persisted, unlike Show sprites: passive chrome like Distance Ticks.
        # Only ever drawn in Units mode, so it needs no gating here.
        self.stack_badges_action = QAction(
            "Show Stacked-Unit &Badges", self, checkable=True, checked=settings.get_stack_badges()
        )
        self.stack_badges_action.setToolTip(
            "In Units mode, a count over every spot where a unit is hidden under another. "
            "Click the spot repeatedly to cycle through the stack."
        )
        view_menu.addAction(self.stack_badges_action)

        # Persisted passive chrome like the badges above; Units mode only.
        self.selection_owner_colour_action = QAction(
            "Colour Selection by &Owner", self, checkable=True, checked=settings.get_selection_by_owner()
        )
        self.selection_owner_colour_action.setToolTip(
            "In Units mode, outline selected units in their owner's player colour. "
            "GAIA and the marquee keep the Selection colour from Settings > Appearance."
        )
        view_menu.addAction(self.selection_owner_colour_action)

        # GH #49. Persisted passive chrome like the two above, but off by
        # default: it draws over the sprites, so it is opt-in.
        self.range_rings_action = QAction(
            "Show &Range Rings", self, checkable=True, checked=settings.get_range_rings()
        )
        self.range_rings_action.setToolTip(
            "In Units mode, a circle around each selected building showing its attack range. "
            "Buildings only, and only ones that have a range. No technology or trigger "
            "effects are applied."
        )
        view_menu.addAction(self.range_rings_action)

        # GH #22. Persisted passive chrome, off by default for the same
        # reason as the rings: it draws on the map rather than beside it.
        # Not mode-gated -- a starting camera belongs to the scenario, not to
        # Players mode; only the emphasis on the selected player is.
        self.player_cameras_action = QAction(
            "Show &Player Cameras", self, checkable=True, checked=settings.get_camera_markers()
        )
        self.player_cameras_action.setToolTip(
            "A camera glyph on each player's starting view tile, in that player's colour. "
            "Players whose view is unset get none. In Players mode the selected player's "
            "marker is emphasised."
        )
        view_menu.addAction(self.player_cameras_action)

        # GH #41. Persisted passive chrome, ON by default: it draws nothing
        # until a trigger is selected, and persists across modes so the
        # selected trigger can be read on the map from View.
        self.trigger_overlay_action = QAction(
            "Show &Trigger Overlay", self, checkable=True, checked=settings.get_trigger_overlay()
        )
        self.trigger_overlay_action.setToolTip(
            "The trigger selected in Triggers mode, drawn on the map: its areas, its locations, "
            "and a line with the distance from an area to its destination. The selected "
            "condition or effect is drawn strongest."
        )
        view_menu.addAction(self.trigger_overlay_action)

        # The ruler strip of tick marks outside the map border. Deliberately
        # NOT gated by mode or terrain style, unlike its two neighbours
        # above: it needs only the map dimensions plus, for the iso styles,
        # the projection, so there is nothing to grey and nothing to explain
        # in a greyed tooltip.
        ticks_menu = view_menu.addMenu("&Distance Ticks")
        self.distance_ticks_action = QAction(
            "&Show", self, checkable=True, checked=settings.get_distance_ticks()
        )
        self.distance_ticks_action.setToolTip(
            "Ruler-style tick marks just outside the map border, with a tile "
            "number on every fourth one. Constant size at any zoom."
        )
        ticks_menu.addAction(self.distance_ticks_action)
        ticks_menu.addSeparator()

        interval_group = QActionGroup(self)
        interval_group.setExclusive(True)
        current_interval = settings.get_distance_tick_interval()
        self.distance_tick_interval_actions: dict[int, QAction] = {}
        for tiles in edge_ticks.TICK_INTERVALS:
            action = QAction(f"Every &{tiles} tiles", self, checkable=True, checked=tiles == current_interval)
            interval_group.addAction(action)
            ticks_menu.addAction(action)
            self.distance_tick_interval_actions[tiles] = action

        # Connected only after every checked= above has settled. Both
        # handlers write config.yaml, so connecting first would make each
        # ViewerWindow() perform a disk write during _build_menu_bar.
        self.distance_ticks_action.toggled.connect(self._on_distance_ticks_toggled)
        self.stack_badges_action.toggled.connect(self._on_stack_badges_toggled)
        self.selection_owner_colour_action.toggled.connect(self._on_selection_owner_colour_toggled)
        self.range_rings_action.toggled.connect(self._on_range_rings_toggled)
        self.player_cameras_action.toggled.connect(self._on_player_cameras_toggled)
        self.trigger_overlay_action.toggled.connect(self._on_trigger_overlay_toggled)
        for tiles, action in self.distance_tick_interval_actions.items():
            # `on` must lead: toggled passes checked as the first positional
            # arg, so a lambda with only `tiles=tiles` gets it bound into
            # `tiles` instead (the trap New Map's own size actions document).
            # `on and ...` is required rather than stylistic: QActionGroup
            # unchecks the outgoing action before checking the incoming one,
            # so without the guard each switch fires twice, the first time
            # carrying the stale interval.
            action.toggled.connect(lambda on, tiles=tiles: on and self._on_distance_tick_interval(tiles))

        # Not gated by mode or terrain style, like Distance Ticks. Connected
        # after checked= settles, for the same config-write reason.
        grid_menu = view_menu.addMenu("&Grid")
        self.grid_action = QAction("&Show", self, checkable=True, checked=settings.get_grid_overlay())
        self.grid_action.setToolTip(
            "A line on every tile boundary, every fourth one stronger. Lightness and "
            "thickness are in Settings > Appearance."
        )
        grid_menu.addAction(self.grid_action)
        grid_menu.addSeparator()
        self.grid_follow_action = QAction(
            "&Follow Terrain Elevation", self, checkable=True,
            checked=settings.get_grid_follow_elevation(),
        )
        self.grid_follow_action.setToolTip(
            "Drape the grid over raised terrain instead of drawing it on the flat "
            "ground plane. No effect in Flat."
        )
        grid_menu.addAction(self.grid_follow_action)

        # Same two persisted halves as Distance Ticks, so the same submenu
        # shape. Not gated by mode or style either: MapView already holds
        # everything it needs.
        footprint_menu = view_menu.addMenu("&Footprint Outlines")
        self.footprint_action = QAction(
            "&Show", self, checkable=True, checked=settings.get_footprint_outlines()
        )
        self.footprint_action.setToolTip(
            "Outline the tiles each unit occupies, readable over sprite art and "
            "between two buildings that share an edge."
        )
        footprint_menu.addAction(self.footprint_action)
        footprint_menu.addSeparator()
        footprint_group = QActionGroup(self)
        footprint_group.setExclusive(True)
        current_scope = settings.get_footprint_scope()
        self.footprint_scope_actions: dict[str, QAction] = {}
        for scope, label in (
            (unit_pick.FOOTPRINT_SCOPE_MULTITILE, "&Multi-tile buildings"),
            (unit_pick.FOOTPRINT_SCOPE_BUILDINGS, "All &buildings"),
            (unit_pick.FOOTPRINT_SCOPE_ALL, "All &units"),
        ):
            action = QAction(label, self, checkable=True, checked=scope == current_scope)
            footprint_group.addAction(action)
            footprint_menu.addAction(action)
            self.footprint_scope_actions[scope] = action

        self.grid_action.toggled.connect(self._on_grid_overlay_toggled)
        self.grid_follow_action.toggled.connect(self._on_grid_follow_toggled)
        self.footprint_action.toggled.connect(self._on_footprint_outlines_toggled)
        for scope, action in self.footprint_scope_actions.items():
            # `on` leads, and the guard is required, for the same two reasons
            # the tick intervals above document.
            action.toggled.connect(lambda on, scope=scope: on and self._on_footprint_scope(scope))

        # View > Layers: render-category questions, a third axis beside
        # Filters (WHICH units) and Show sprites (WHETHER unit art draws).
        # Session-only, like Show sprites and the unit filter, since every
        # row changes what the map itself shows -- see view_layers.py's own
        # module docstring.
        #
        # A registry loop rather than a block per layer: a new row plugs into
        # view_layers.LAYERS and the menu, the greying and the keybind table
        # all follow.
        #
        # self._layers is the WINDOW's copy, for exactly the reason
        # _sprites_enabled above is: a cache is rebuilt from scratch on every
        # style switch and file open, so something outside it has to remember
        # the choice. _render_current() re-applies it.
        self._layers = view_layers.LayerState()
        layers_menu = view_menu.addMenu("&Layers")
        self.layer_actions: dict[str, QAction] = {}
        for spec in view_layers.LAYERS:
            action = QAction(spec.label, self, checkable=True, checked=spec.default)
            action.setEnabled(False)  # re-gated by _update_tool_enabled()
            action.setToolTip(spec.tooltip)
            layers_menu.addAction(action)
            self.layer_actions[spec.layer_id] = action
        # Second loop, after every checked= above has settled, same discipline
        # the Distance Ticks block documents. `on` must lead for the same
        # reason too: toggled passes checked as the first positional arg.
        for layer_id, action in self.layer_actions.items():
            action.toggled.connect(lambda on, lid=layer_id: self._on_layer_toggled(lid, on))

        tools_menu = menu_bar.addMenu("&Tools")
        self.analysis_action = QAction("Map &Analysis…", self)
        self.analysis_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.analysis_action.triggered.connect(self._show_analysis)
        tools_menu.addAction(self.analysis_action)

        help_menu = menu_bar.addMenu("&Help")
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)
        help_menu.addSeparator()
        self.debug_log_action = QAction("&Debug Log", self)
        self.debug_log_action.triggered.connect(self._show_debug_log)
        help_menu.addAction(self.debug_log_action)
        # No settings.get_/set_ backing on purpose (draw-perf plan Step 1): a
        # debug toggle that defaults off each session is the right behavior
        # anyway, and it sidesteps QSettings' string-to-bool read trap.
        self.perf_trace_action = QAction(
            "Perf &Trace", self, checkable=True, checked=perf_trace.is_enabled()
        )
        self.perf_trace_action.setToolTip(
            "Traces per-stroke drag-paint latency by phase, and pan/zoom repaints, to Help > Debug Log. "
            "Also settable via the DESCAPE_PERF_TRACE=1 environment variable."
        )
        self.perf_trace_action.toggled.connect(perf_trace.enable)
        help_menu.addAction(self.perf_trace_action)

    def _on_unit_field_changed(self, spec: unit_fields.UnitFieldSpec, value) -> None:
        """The single funnel every unit inspector editor reports through --
        mirrors set_trigger_field()'s role for the trigger form.

        Applies to every selected unit (GH #71): X/Y/Z and Owner set all of
        them to the typed value, and Rotation sets every ANGLE member to one
        facing, skipping the rest. A one-unit selection is a group of one, so
        both cases share this path, and each edit is one undo record.

        UnitsPanel itself suppresses this during a programmatic populate (its
        own `_populating` guard, mirroring PlayersPanel._changed()'s), so
        this only ever fires for a genuine user edit -- but a genuinely
        no-op edit (typing the same X back) must still not record one,
        hence the per-unit equality checks below.
        """
        if not self._selection:
            return
        model = self._ensure_unit_edits()
        if model is None:
            self._update_unit_inspector_from_selection()
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return
        single = len(entries) == 1

        if spec.field_id == "player":
            new_player = int(value)
            changing = [e for e in entries if e.player_id != new_player]
            if not changing:
                return
            label = "Reassign unit" if single else f"Reassign {len(changing)} units"
            players = sorted({e.player_id for e in changing} | {new_player})
            with self._unit_edit(model, label, players):
                for entry in changing:
                    model.reassign(entry.unit, new_player)
                # Inside the block: _unit_edit's exit reconciles the selection
                # and would drop the stale (old_player, ref) keys.
                self._selection = [unit_pick.unit_key(new_player, e.unit) for e in entries]
            owner = "GAIA" if new_player == GAIA_PLAYER_ID else f"Player {new_player}"
            if single:
                self._log_status(f"Reassigned unit to {owner}")
            else:
                self._log_status(f"Reassigned {len(changing)} units to {owner}")
            return

        if spec.field_id == "rotation":
            # VARIANT/INERT members are skipped: their rotation is passed
            # through verbatim (AGENTS.md hard rule), and set_rotation raises.
            angle_entries = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
            skipped = len(entries) - len(angle_entries)
            if not angle_entries:
                return
            # `value` is a whole facing (GH #61) on the panel's scale: the
            # shared direction count, or the largest when members differ.
            scale = unit_rotation.facing_scale(e.unit.unit_const for e in angle_entries)
            changes = []
            for entry in angle_entries:
                count = unit_rotation.angle_count_for(entry.unit.unit_const)
                facing = unit_rotation.snap_facing(int(value), scale, count)
                # Already showing this facing: no write, so an off-grid stored
                # value is only snapped by a real edit.
                if unit_rotation.rotation_to_facing(entry.unit.rotation, count) == facing:
                    continue
                rotation = unit_rotation.facing_to_rotation(facing, count)
                if rotation != entry.unit.rotation:
                    changes.append((entry, rotation))
            if not changes:
                return
            label = "Set unit Rotation" if single else f"Set rotation on {len(changes)} units"
            # Splice-eligible (Batch D's D5): rotation never moves the
            # footprint, so old/new own_tile and old/new tiles are identical
            # -- the splice still exists so the cache re-resolves this
            # unit's now-different sprite frame, and the pick index is left
            # untouched (nothing pickable moved).
            players = sorted({e.player_id for e, _ in changes})
            splices: list[UnitSplice] = []
            with self._unit_edit(model, label, players, splices, fields_only=True):
                for entry, rotation in changes:
                    unit = entry.unit
                    old_own, old_tiles = self._unit_footprint(unit)
                    idx = self._unit_list_index(entry.player_id, unit)
                    model.set_rotation(unit, rotation)
                    new_own, new_tiles = self._unit_footprint(unit)
                    splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            if skipped:
                self._log_status(f"{label} ({skipped} skipped: not rotatable)")
            return

        axis = spec.field_id

        def axis_value(unit):
            return getattr(unit, "z", 0.0) if axis == "z" else getattr(unit, axis)

        changing = [e for e in entries if axis_value(e.unit) != value]
        if not changing:
            return
        label = f"Set unit {spec.label}" if single else f"Set {spec.label} on {len(changing)} units"
        # A typed coordinate carries the host's garrison with it (GH #42),
        # counted out of the label: the user edited the units they selected.
        # An occupant already on the typed value has nothing to write.
        changing += [e for e in self._garrison_occupant_entries(model, changing) if axis_value(e.unit) != value]
        mm = self.scenario.map_manager
        players = sorted({e.player_id for e in changing})
        splices = []
        with self._unit_edit(model, label, players, splices, fields_only=True):
            for entry in changing:
                unit = entry.unit
                x, y, z = unit.x, unit.y, getattr(unit, "z", 0.0)
                if axis == "x":
                    x = value
                elif axis == "y":
                    y = value
                elif axis == "z":
                    z = value
                old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                model.set_position(unit, x, y, z)
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
                self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)

    def _update_unit_inspector_from_selection(self) -> None:
        """Puts the inspector back in step with the model after an edit that
        was refused (_ensure_unit_edits() returned None) -- the unit-editor
        counterpart to _repopulate_map_options(). The panel's live values are
        what _on_unit_field_changed's equality guard compares against, so
        leaving a refused edit on screen would make the next genuine change
        to that field look like a no-op."""
        if self.mode != "units" or not self._selection:
            return
        self._refresh_selection_view()

    def _refresh_selection_view(self) -> None:
        """The single reconciliation point for self._selection (b2.2):
        drops any key that no longer resolves against the live unit index,
        pushes the surviving entries to MapView's N-way highlight, and
        updates the inspector -- one entry shows its fields (D5), 2+ show the
        group fields (GH #71), and zero shows unit_inspector_empty's text.

        Every path that used to read/write self._selection[0] directly and
        silently discard the rest (on_click_select, on_unit_nudge,
        on_unit_delete, _after_unit_mutation, _refresh_selection_after_filter)
        goes through this now, so a dead key dropped by one path can't
        linger and desync from another -- this had to land before the
        marquee, not after.
        """
        index = self.map_view._unit_index
        entries = []
        if index is not None:
            for key in self._selection:
                entry = index.entry_for_key(key)
                if entry is not None:
                    entries.append(entry)
        self._selection = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
        self.map_view.set_unit_selection(entries)
        # Rotate is the first action gated on there BEING a selection, so this
        # reconciliation point is now also where that gate is re-evaluated.
        # Not a recursion risk despite _update_tool_enabled()'s own
        # pan_action.setChecked(True) branch: that re-enters _on_tool_selected,
        # which reaches _update_tool_enabled() again (terminating, per its own
        # comment) but never comes back through here or _after_unit_mutation.
        self._update_tool_enabled()
        if len(entries) == 1:
            self.units_panel.show_unit(entries[0])
            self._apply_garrison_block(entries[0])
            return
        if entries:
            self.units_panel.show_group(entries)
            return
        self.units_panel.show_selection_count(0)

    # -- Inspector garrison block (GH #42) ------------------------------

    def _garrison_occupants(self, unit) -> list:
        """(player_id, occupant) for every unit garrisoned inside `unit`.

        Deliberately does NOT build the edit model: this runs on every
        selection change, and _ensure_unit_edits()' construction gate scans
        the whole file and can refuse with a dialog. An already-built model
        answers the membership question off its O(1) reverse-map; without
        one this is a walk of the nine lists, the same cost as the pick
        index's own build.
        """
        if unit.reference_id == -1:
            return []  # see _garrison_referrers: -1 is "inside nothing"
        model = self.unit_edits
        inside = {id(u) for u in _garrison_referrers(model, unit)} if model is not None else None
        rows = []
        for player_id, units in enumerate(self.scenario.unit_manager.units):
            for candidate in units:
                if candidate is unit:
                    continue
                if inside is not None:
                    if id(candidate) in inside:
                        rows.append((player_id, candidate))
                elif getattr(candidate, "garrisoned_in_id", -1) == unit.reference_id:
                    rows.append((player_id, candidate))
        return rows

    def _apply_garrison_block(self, entry) -> None:
        """Shows the Inspector's Garrison list for a host, or hides it.

        Shown for a const the game lets hold something, and also for one
        that already holds something it shouldn't -- a file is displayed as
        it is, never corrected, and an occupant hidden by the default filter
        would otherwise be unreachable.
        """
        unit = entry.unit
        occupants = self._garrison_occupants(unit)
        if not occupants and not garrison.can_hold(unit.unit_const):
            self.units_panel.hide_garrison()
            return
        rows = [
            (
                _unit_name(occupant.unit_const),
                "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}",
                occupant.reference_id,
            )
            for player_id, occupant in occupants
        ]
        wrong_type = sum(1 for _pid, o in occupants if not garrison.accepts(unit.unit_const, o.unit_const))
        self.units_panel.show_garrison(rows, garrison.capacity(unit.unit_const), wrong_type)

    def _selected_host_entry(self):
        """The one selected unit's entry, or None -- the garrison block only
        ever exists for a single selection."""
        index = self.map_view._unit_index
        if self.scenario is None or index is None or len(self._selection) != 1:
            return None
        return index.entry_for_key(self._selection[0])

    def _on_garrison_add(self) -> None:
        """The Garrison block's Add... button: pick an object the host
        admits, then create it inside. The picker is restricted to
        garrison.eligible_consts(), so a refusal is normally unreachable
        from the UI -- _garrison_add_const() still checks, since it is also
        the path a test (and any later caller) drives."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        eligible = garrison.eligible_consts(entry.unit.unit_const)
        catalog = [row for row in object_catalog.objects() if row.id in eligible]
        if not catalog:
            self._log_status(f"Garrison: nothing can go inside {_unit_name(entry.unit.unit_const)}")
            return
        dialog = CatalogBrowseDialog(catalog, parent=self)
        dialog.setWindowTitle(f"Garrison {_unit_name(entry.unit.unit_const)}")
        if dialog.exec_() != QDialog.Accepted:
            return
        unit_const = dialog.selected_id()
        if unit_const is not None:
            self._garrison_add_const(entry, unit_const)

    def _garrison_add_const(self, entry, unit_const: int) -> None:
        """Add...'s write half, without the modal picker.

        The new unit takes the host's own player, point and z, rotation 0
        and the host's reference_id -- garrisoned_in_id is write-once
        (unit_model.py), so creating the unit inside is the only way in, and
        this is not a fields_only edit.
        """
        host = entry.unit
        if not garrison.accepts(host.unit_const, unit_const):
            self._log_status(
                f"Garrison: {_unit_name(unit_const)} cannot go inside {_unit_name(host.unit_const)}"
            )
            return
        capacity = garrison.capacity(host.unit_const)
        if len(self._garrison_occupants(host)) >= capacity:
            self._log_status(f"Garrison: {_unit_name(host.unit_const)} is full ({capacity} places)")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        player = entry.player_id
        splices: list[UnitSplice] = []
        with self._unit_edit(model, "Garrison unit", [player], splices):
            unit = model.add(
                player,
                unit_const,
                host.x,
                host.y,
                getattr(host, "z", 0.0),
                garrisoned_in_id=host.reference_id,
            )
            new_own, new_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(player, unit)
            splices.append(UnitSplice(player, idx, unit, None, new_own, (), new_tiles))
            index = self.map_view._unit_index
            if index is not None:
                # A no-op while Show Garrisoned is off, which is the point:
                # matches() is the single gate, so drawn and pickable stay
                # in step without this path knowing about the filter.
                unit_pick.patch_index_for_add(self.scenario, index, player, unit, self._unit_filter)
        self._log_status(f"Garrisoned {_unit_name(unit_const)} inside {_unit_name(host.unit_const)}")

    def _on_garrison_delete(self, reference_ids) -> None:
        """The Garrison block's Delete button: the selected occupants go in
        one undo record, along with anything garrisoned inside them."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        wanted = set(reference_ids)
        targets = [(pid, u) for pid, u in self._garrison_occupants(entry.unit) if u.reference_id in wanted]
        if not targets:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        rows = [unit_pick.UnitEntry(pid, u, int(u.x), int(u.y), -1) for pid, u in targets]
        rows += self._garrison_occupant_entries(model, rows)
        players = sorted({row.player_id for row in rows})
        label = "Remove from garrison" if len(rows) == 1 else f"Remove {len(rows)} from garrison"
        with self._unit_edit(model, label, players):
            model.remove_many([row.unit for row in rows])
        self._log_status(f"Removed {len(rows)} unit(s) from {_unit_name(entry.unit.unit_const)}'s garrison")

    def _on_garrison_navigate(self, reference_id: int) -> None:
        """Double-click on an occupant row. Selecting one needs it in the
        pick index, and the default filter keeps it out -- so this says what
        to turn on rather than silently doing nothing."""
        entry = self._selected_host_entry()
        if entry is None:
            return
        if not self._unit_filter.show_garrisoned:
            self._log_status("Turn on Filters > Show Garrisoned Units to select a unit inside another one")
            return
        target = next(
            ((pid, u) for pid, u in self._garrison_occupants(entry.unit) if u.reference_id == reference_id),
            None,
        )
        if target is None:
            return
        player_id, occupant = target
        key = unit_pick.unit_key(player_id, occupant)
        index = self.map_view._unit_index
        if index is None or index.entry_for_key(key) is None:
            return
        self._selection = [key]
        self._refresh_selection_view()
        self._log_status(f"Selected {_unit_name(occupant.unit_const)} (garrisoned)")

    def _build_filters_button(self, toolbar) -> None:
        """The Filters popup -- phase 3's P3-b.

        A QMenu of checkable QActions rather than a custom widget: keyboard
        navigation and testability both come free, and a menu is the
        conventional home for "a pile of independent toggles".

        Visible in EVERY mode, unlike selection/hover, which are Units-mode
        only. Filtering changes what is rendered, so it affects View and
        Terrain too -- gating it on a mode would mean hiding GAIA required
        switching modes first.

        Per-session state only, deliberately NOT persisted to settings.py.
        A filter that survived a restart could present an apparently
        unit-less map with no visible cause -- the toggle that explains it
        is two clicks deep in a menu the user has no reason to open.
        """
        self._unit_filter = UnitFilter()

        self.filters_button = QToolButton()
        self.filters_button.setText("Filters")
        self.filters_button.setPopupMode(QToolButton.InstantPopup)
        self.filters_button.setToolTip(
            "Show/hide units by owner or kind -- affects every mode, and hidden units can't be selected"
        )
        self.filters_button.setEnabled(False)  # re-gated by _update_tool_enabled()

        menu = QMenu(self.filters_button)
        self.show_gaia_action = QAction("Show GAIA", self, checkable=True, checked=True)
        self.show_gaia_action.setToolTip("GAIA owns cliffs, gold and stone as well as trees")
        self.show_gaia_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_gaia_action)

        self.show_trees_action = QAction("Show Trees", self, checkable=True, checked=True)
        self.show_trees_action.setToolTip(
            "Trees are the bulk of a typical file's ~9,988 GAIA objects -- hiding them alone "
            "usually leaves the map readable"
        )
        self.show_trees_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_trees_action)

        self.show_walls_action = QAction("Show Walls", self, checkable=True, checked=True)
        self.show_walls_action.setToolTip(
            "Walls and gates -- up to 2,004 in one example file, and they hide what is "
            "behind and inside them"
        )
        self.show_walls_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_walls_action)

        self.show_eye_candy_action = QAction("Show Eye Candy", self, checkable=True, checked=True)
        self.show_eye_candy_action.setToolTip(
            "Grass, rocks, flowers, stumps and barrels -- up to 3,093 in one example file. "
            "Gold, stone and berry bushes are resources, not eye candy, and stay visible"
        )
        self.show_eye_candy_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_eye_candy_action)

        self.show_invisible_action = QAction("Show Invisible Objects", self, checkable=True, checked=True)
        self.show_invisible_action.setToolTip(
            "Invisible Objects, Map Revealers and Blockers: no art in-game, drawn here as a coloured box"
        )
        self.show_invisible_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_invisible_action)

        # The one entry that ships UNCHECKED (GH #42): a garrisoned unit is
        # inside its host in game, and the file stores it at the host's own
        # point, so leaving it on draws a tower's occupants stacked on the
        # tower. Turning it on is how an occupant becomes selectable again.
        self.show_garrisoned_action = QAction("Show Garrisoned Units", self, checkable=True, checked=False)
        self.show_garrisoned_action.setToolTip(
            "Units inside a building or a transport -- hidden by default, as in game. "
            "The Inspector's Garrison block lists and edits them without this"
        )
        self.show_garrisoned_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_garrisoned_action)

        menu.addSeparator()
        self.player_actions: dict[int, QAction] = {}
        for player_id in range(1, MAX_PLAYER_ID + 1):
            action = QAction(f"Player {player_id}", self, checkable=True, checked=True)
            action.toggled.connect(self._on_filter_changed)
            menu.addAction(action)
            self.player_actions[player_id] = action

        menu.addSeparator()
        self.filter_all_players_action = QAction("All Players", self)
        self.filter_all_players_action.triggered.connect(lambda: self._set_all_players(True))
        menu.addAction(self.filter_all_players_action)
        self.filter_no_players_action = QAction("No Players", self)
        self.filter_no_players_action.triggered.connect(lambda: self._set_all_players(False))
        menu.addAction(self.filter_no_players_action)

        menu.addSeparator()
        self.filter_show_all_action = QAction("Show All", self)
        self.filter_show_all_action.setToolTip(
            "Check every entry above -- GAIA, Trees, Walls, Eye Candy, Invisible Objects, "
            "Garrisoned Units, and all players"
        )
        self.filter_show_all_action.triggered.connect(lambda: self._set_all_filters(True))
        menu.addAction(self.filter_show_all_action)
        self.filter_hide_all_action = QAction("Hide All", self)
        self.filter_hide_all_action.setToolTip(
            "Uncheck every entry above -- GAIA, Trees, Walls, Eye Candy, Invisible Objects, "
            "Garrisoned Units, and all players"
        )
        self.filter_hide_all_action.triggered.connect(lambda: self._set_all_filters(False))
        menu.addAction(self.filter_hide_all_action)

        self.filters_button.setMenu(menu)
        toolbar.addWidget(self.filters_button)
        # Show Garrisoned ships unchecked, so the app's default filter is no
        # longer UnitFilter(): re-read the menu now, or _unit_filter claims
        # occupants are shown until the first toggle and a freshly loaded
        # file draws them stacked on their hosts.
        self._unit_filter = self._current_unit_filter()

    def _needs_unit_index(self) -> bool:
        """Units mode needs the pick index, and so does View > Footprint
        Outlines, which is reachable from every mode. Without the second
        term the overlay silently draws nothing outside Units mode, since
        the index is built on demand rather than at load."""
        return self.mode == "units" or settings.get_footprint_outlines() or self._unit_picker is not None

    def _rebuild_unit_index(self) -> None:
        """Rebuilds the pick index for the current scenario + filter and hands
        it to MapView.

        Built on demand (entering Units mode, or a filter change while
        already in it) rather than at load: an ~11k-unit file costs a real
        walk, and opening a map for terrain work must not pay for it.
        """
        # A new index can reorder or resize any stack the cycle points into.
        self._stack_cycle = None
        if self.scenario is None:
            self.map_view.set_selection_player_colors(None)
            self.map_view.set_unit_index(None)
            return
        # Before set_unit_index(), whose own highlight refresh then draws with these.
        self.map_view.set_selection_player_colors(self.scenario.player_colors)
        self.map_view.set_unit_index(unit_pick.build_index(self.scenario, self._unit_filter))

    def on_click_select(self, pos, modifiers) -> tuple[int, int] | None:
        """MapView's sixth injected callable -- a left click in Units mode.

        Selection is a COLLECTION from day one (self._selection), even though
        phase 3 only ever held 0 or 1: future marquee/bulk items would
        otherwise force an immediate rewrite of a scalar _selected_unit. And
        identity is the (player_id, reference_id) key resolved late through
        the index, never a list position or an object reference -- phase
        3.5's add/remove invalidates both of those.

        b2.2 wires the modifiers this always received but ignored (settled
        fact 3): Ctrl toggles the clicked unit's own membership; Shift adds
        it without ever removing. Neither is "range" selection -- units have
        no ordering for that to mean anything against -- so both degrade to
        a plain set operation over the existing selection. A modified click
        on empty ground leaves the selection alone (Ctrl/Shift mean "adjust
        the set", not "replace it"); a plain click on empty ground clears it,
        same as before this slice.

        A plain click on a stack (unit_pick.stack_groups()) selects exactly
        what pick_unit returns; each repeat click on the same stack selects
        the next unit down, wrapping. Ctrl/Shift are set operations, not
        stack navigation, and reset the cycle.

        Returns the key the click acted on (None on empty ground), which
        MapView uses as the move-drag key.

        GH #75: a plain click on a member of a 2+ selection leaves the group
        selected, so a drag can move all of it; on_unit_click_release()
        collapses it to that unit if the press never became a drag.
        """
        self._pending_collapse = None
        self._drag_grab_tile = None
        self._ghost_cap_logged = None
        found = self.map_view.pick_unit_cover_at(pos)
        if found is None:
            self._stack_cycle = None
            if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
                self._selection = []
                self._refresh_selection_view()
                self._log_status("Selection cleared")
            return None
        entry, cover_tile = found
        self._drag_grab_tile = cover_tile
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        stack_note = ""
        if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)) and key in self._selection and len(
            self._selection
        ) >= 2:
            # Still cycles now, so the click after the collapse carries on down the stack.
            entry, stack_note = self._cycle_stack(entry, cover_tile)
            key = unit_pick.unit_key(entry.player_id, entry.unit)
            self._pending_collapse = (key, stack_note)
            return key
        if modifiers & Qt.ControlModifier:
            self._stack_cycle = None
            if key in self._selection:
                self._selection = [k for k in self._selection if k != key]
            else:
                self._selection = [*self._selection, key]
        elif modifiers & Qt.ShiftModifier:
            self._stack_cycle = None
            if key not in self._selection:
                self._selection = [*self._selection, key]
        else:
            entry, stack_note = self._cycle_stack(entry, cover_tile)
            key = unit_pick.unit_key(entry.player_id, entry.unit)
            self._selection = [key]
        self._refresh_selection_view()
        if len(self._selection) == 1:
            self._log_status(
                f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g}){stack_note}"
            )
        else:
            self._log_status(f"{len(self._selection)} units selected")
        return key

    def on_unit_click_release(self, key: tuple[int, int]) -> None:
        """MapView's release of a press on a unit that never became a drag:
        finishes the plain click on_click_select put off (GH #75)."""
        pending, self._pending_collapse = self._pending_collapse, None
        if pending is None or pending[0] != key:
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            return
        self._selection = [key]
        self._refresh_selection_view()
        self._log_status(
            f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g}){pending[1]}"
        )

    def _is_group_key(self, key) -> bool:
        """Whether a drag on `key` moves the whole selection (GH #75)."""
        return key is not None and len(self._selection) >= 2 and key in self._selection

    def _cycle_stack(self, picked, cover_tile: tuple[int, int]):
        """(entry to select, status suffix) for a plain click that picked
        `picked` while covering `cover_tile`, advancing _stack_cycle."""
        members = self.map_view.stack_group_at(cover_tile)
        if not members:
            self._stack_cycle = None
            return picked, ""
        cycle = self._stack_cycle
        continuing = cycle is not None and cycle[0] == cover_tile and self._selection == [cycle[2]]
        chosen, idx = unit_pick.stack_cycle_step(members, picked, cycle[1] if continuing else None)
        self._stack_cycle = (cover_tile, idx, unit_pick.unit_key(chosen.player_id, chosen.unit))
        if idx == -1:
            return chosen, ""
        return chosen, f" ({idx + 1} of {len(members)} stacked here)"

    def on_marquee_select(self, keys: list[tuple[int, int]], modifiers) -> None:
        """MapView's eleventh injected callable -- a completed marquee drag
        in Units mode (b2.3). `keys` are the (player_id, reference_id) keys
        of every unit MapView's units_in_rect() query found under the
        dragged rectangle, already filter-respecting.

        Same modifier convention as on_click_select's own toggle/add (b2.2),
        applied to the whole covered set at once: Ctrl toggles each key's
        membership (a second marquee over an already-selected group
        deselects exactly that overlap); Shift adds without removing; plain
        replaces the selection outright. An empty marquee with a modifier
        held leaves the existing selection alone -- Ctrl/Shift both mean
        "adjust the set", never "replace it" -- matching on_click_select's
        own empty-ground behavior.
        """
        adjusting = bool(modifiers & (Qt.ControlModifier | Qt.ShiftModifier))
        if not keys:
            if not adjusting:
                self._selection = []
            else:
                return
        elif modifiers & Qt.ControlModifier:
            existing = set(self._selection)
            toggled = existing.symmetric_difference(keys)
            self._selection = list(toggled)
        elif modifiers & Qt.ShiftModifier:
            merged = list(self._selection)
            for key in keys:
                if key not in merged:
                    merged.append(key)
            self._selection = merged
        else:
            self._selection = list(keys)
        self._refresh_selection_view()
        if not self._selection:
            self._log_status("Selection cleared")
        elif len(self._selection) == 1:
            index = self.map_view._unit_index
            entry = index.entry_for_key(self._selection[0]) if index is not None else None
            if entry is not None:
                self._log_status(
                    f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})"
                )
        else:
            self._log_status(f"{len(self._selection)} units selected")

    def on_select_whole_stack(self) -> None:
        """Edit > Select Whole Stack (GH #75): adds every member of every stack
        holding a selected unit, keeping order and dropping duplicates.

        Found by membership, not by tile: a stack is keyed by its hidden
        unit's own tile (unit_pick.stack_scan), which need not be the selected
        unit's tile."""
        if not self._selection:
            self._log_status("Select Whole Stack: nothing is selected")
            return
        selected = set(self._selection)
        widened = list(self._selection)
        seen = set(widened)
        stacked = False
        for members in self.map_view._stack_groups.values():
            keys = [unit_pick.unit_key(m.player_id, m.unit) for m in members]
            if selected.isdisjoint(keys):
                continue
            stacked = True
            for key in keys:
                if key not in seen:
                    seen.add(key)
                    widened.append(key)
        if not stacked:
            self._log_status("Select Whole Stack: no selected unit is stacked")
            return
        self._stack_cycle = None
        self._selection = widened
        self._refresh_selection_view()
        self._log_status(f"{len(self._selection)} units selected (stack)")

    def _placement_point(self, pos, modifiers) -> tuple[float, float] | None:
        """Where a click puts a unit -- the one place the snapped/free choice
        is made, so Place Unit and Move Unit cannot disagree about it.

        Snapped (the default) is `tile + 0.5`: `_pick_tile()` resolves the
        clicked SCREEN pixel to an integer tile in all three render styles
        (Flat's division, Stepped's screen_to_tile, Sloped's pick-plane
        lookup). Free is `_pick_map_point()`, the continuous inverse added by
        free placement's Stage 2 -- which supersedes the "Sloped has no
        analytic inverse" reason this feature used to be deferred for. That
        claim is true of the TERRAIN surface and was never true of the
        point-placement one, which inverts closed-form.

        Free is reached by the toolbar checkbox or by holding Alt for a
        one-off (D2's modifier), either one. Falls back to the snapped point
        whenever the inverse returns None -- a degenerate denominator or a
        failed round-trip, both reachable on a steep Sloped ramp -- rather
        than refusing the click: a placement that silently does nothing is
        worse than one that snaps.

        **The fallback says so.** Without a status line a tester who asked
        for free placement and got a tile centre cannot tell "the inverse
        refused this pixel" from "the checkbox didn't apply" from "the window
        manager ate the Alt", and those want three different responses."""
        point, fell_back = self._resolve_placement_point(pos, modifiers)
        if fell_back:
            self._log_status(FREE_PLACE_FALLBACK_MESSAGE)
        return point

    def _free_placement_requested(self, modifiers) -> bool:
        """The toolbar checkbox, or a held Alt for a one-off."""
        # `modifiers` is None from a caller that has no real event behind it
        # (a test, or a programmatic place), so it must not be bit-tested.
        alt = modifiers is not None and bool(modifiers & Qt.AltModifier)
        return self.free_place_check.isChecked() or alt

    def _resolve_placement_point(self, pos, modifiers) -> tuple[tuple[float, float] | None, bool]:
        """`_placement_point` without the status line -- (point, fell_back).

        Split out for the mid-drag move preview, which resolves this same
        point on EVERY mouse-move: on a steep Sloped ramp the inverse can
        refuse pixel after pixel, and logging each one would fill the status
        log with one line per frame. The preview reads `fell_back` and ignores
        it; the two committing callers log it, exactly as before."""
        tile = self.map_view._pick_tile(pos)
        if tile is None:
            return None, False
        if self._free_placement_requested(modifiers):
            point = self.map_view._pick_map_point(pos)
            if point is not None:
                return point, False
            return (tile[0] + 0.5, tile[1] + 0.5), True
        return (tile[0] + 0.5, tile[1] + 0.5), False

    def on_unit_place(self, pos, modifiers) -> None:
        """MapView's seventh injected callable -- a left click with the
        Place Unit tool active (b1.4).

        Snapped by default, free with the param toolbar's checkbox or a held
        Alt -- see `_placement_point`, which is where that choice lives."""
        if self.scenario is None:
            return
        point = self._placement_point(pos, modifiers)
        if point is None:
            return
        object_id = self.units_panel.selected_object_const()
        if object_id is None:
            self._log_status("Place Unit: choose an object first")
            return
        player = self.units_panel.owner_id()
        model = self._ensure_unit_edits()
        if model is None:
            return
        x, y = point
        splices: list[UnitSplice] = []
        with self._unit_edit(model, "Place unit", [player], splices):
            unit = model.add(player, object_id, x, y)
            # Placed units become the selection (same reasoning as
            # reassign's own key update): the id just chosen is the one worth
            # showing in the inspector next, not whatever was selected before.
            self._selection = [unit_pick.unit_key(player, unit)]
            new_own, new_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(player, unit)
            splices.append(UnitSplice(player, idx, unit, None, new_own, (), new_tiles))
            index = self.map_view._unit_index
            if index is not None:
                unit_pick.patch_index_for_add(self.scenario, index, player, unit, self._unit_filter)
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        status = f"Placed {_unit_name(object_id)} for {owner_text} at ({x:g}, {y:g})"
        # The same gap the cliff tool surfaces for Show GAIA, generalized:
        # placing something the live filter hides looks exactly like the tool
        # doing nothing. Asked of the filter rather than of a const set, so it
        # covers walls, gates, eye candy, trees and owner alike, and picks up
        # whatever gate the next one adds for free.
        if not self._unit_filter.matches(player, unit):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)

    def scatter_units_in_region(self) -> None:
        """Edit > Scatter Units in Region…: N randomized copies of the Units
        panel's chosen object across the committed Select region, as one undo
        step.

        The object and the default owner come from the Units panel, which is
        a persistent widget whose readers work in every mode, so this action
        needs no mode gate of its own (see _update_tool_enabled).
        """
        if self.scenario is None or self._region is None:
            return
        object_id = self.units_panel.selected_object_const()
        if object_id is None:
            # Place Unit's own rule, so a scatter with nothing chosen says so
            # rather than opening a dialog that cannot be completed.
            self._log_status("Scatter: choose an object in the Units panel first")
            return
        dialog = ScatterDialog(
            self.scenario,
            self._region,
            object_id,
            self.units_panel.owner_id(),
            parent=self,
            last=self._scatter_defaults,
        )
        # Before _ensure_unit_edits(), so cancelling never pays for the lazy
        # unit-model build.
        if dialog.exec_() != QDialog.Accepted:
            return
        self._scatter_defaults = dialog.state()
        params = dialog.params()
        model = self._ensure_unit_edits()
        if model is None:
            return
        player = params.player
        requested = params.requested(len(params.tiles))
        # splices=None, so one wholesale _after_unit_mutation() for the whole
        # scatter, the same trade the wall run makes. The dialog already
        # subtracted occupied tiles, hence avoid_occupied=False.
        with self._unit_edit(model, "Scatter units", [player]):
            placed = scatter.scatter_units(
                self.scenario,
                model,
                params.tiles,
                params.unit_const,
                player=player,
                count=params.count,
                density=params.density,
                seed=params.seed,
                jitter=params.jitter,
                min_spacing=params.min_spacing,
                avoid_occupied=False,
            )
            # Inside the block, so the exit reconciles the selection once.
            # Only in Units mode: that reconciliation is mode-gated, so
            # elsewhere the placed units are left unselected.
            if self.mode == "units":
                self._selection = [unit_pick.unit_key(player, unit) for unit in placed]
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        status = f"Scattered {len(placed)} x {_unit_name(params.unit_const)} for {owner_text}"
        if requested is not None and len(placed) < requested:
            status += f" (asked for {requested}; clamped by eligible tiles/spacing)"
        # on_unit_place()'s clause: a scatter the live filter hides looks
        # exactly like the action doing nothing.
        if placed and not self._unit_filter.matches(player, placed[0]):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)

    def _on_units_panel_place_requested(self, object_id: int) -> None:
        """UnitsPanel's double-click/Enter path -- this replaces "accept the
        dialog" from the old modal CatalogBrowseDialog. Guarding on
        isEnabled() rather than unconditionally checking the action matters:
        on a DISABLED action, setChecked(True) would still flip its checked
        state (QActionGroup does not itself block that), and the next
        _update_tool_enabled() pass would then force it back to Pan with a
        spurious status line -- see that method's own forced-back-to-Pan
        comment.
        """
        if not self.place_unit_action.isEnabled():
            self._log_status("Place Unit is unavailable here")
            return
        self.place_unit_action.setChecked(True)
        # The common flow -- pick an object, place it, then nudge it -- needs
        # focus back on the map, or arrow keys move the tree's current row
        # instead of the just-placed unit (unit_pick's own on_unit_nudge is
        # a MapView.keyPressEvent handler, so it only fires with map focus).
        self.map_view.setFocus()

    def _populate_cliff_families(self) -> None:
        """One-time (construction-time) fill of the Family combo -- cliff
        families are static within a session, unlike Piece (repopulated per
        family) and Frame (re-ranged per piece) below."""
        self._cliff_families = cliff_catalog.families()
        for file_name in sorted(self._cliff_families, key=cliff_catalog.family_label):
            self.cliff_family_combo.addItem(cliff_catalog.family_label(file_name), file_name)
        self._on_cliff_family_changed(self.cliff_family_combo.currentIndex())

    def _on_cliff_family_changed(self, _index: int) -> None:
        """Repopulates Piece for the newly-selected family. Signals blocked
        while rebuilding: an intermediate empty-then-refilled combo would
        otherwise fire spurious currentIndexChanged calls into
        _on_cliff_piece_changed with a stale or None unit_const."""
        file_name = self.cliff_family_combo.currentData()
        self.cliff_piece_combo.blockSignals(True)
        self.cliff_piece_combo.clear()
        for piece in self._cliff_families.get(file_name, ()):
            self.cliff_piece_combo.addItem(piece.label, piece.unit_const)
        self.cliff_piece_combo.blockSignals(False)
        self._on_cliff_piece_changed(self.cliff_piece_combo.currentIndex())

    def _on_cliff_piece_changed(self, _index: int) -> None:
        """Re-ranges Frame to the newly-selected piece's own real frame
        count (cliff_catalog.frame_count -- Marble/Short Marble are 23
        frames, every other family 24; A3's range-check bug is exactly what
        made hardcoding this dangerous)."""
        unit_const = self.cliff_piece_combo.currentData()
        self.cliff_frame_spin.blockSignals(True)
        self.cliff_frame_spin.setRange(0, max(0, cliff_catalog.frame_count(unit_const) - 1) if unit_const is not None else 0)
        self.cliff_frame_spin.blockSignals(False)
        self._update_cliff_preview()

    def _update_cliff_preview(self) -> None:
        unit_const = self.cliff_piece_combo.currentData()
        pixmap = None
        if unit_const is not None:
            pixmap = preview_pixmap(unit_const, float(self.cliff_frame_spin.value()))
        self.cliff_preview_label.setPixmap(pixmap if pixmap is not None else QPixmap())

    # -- Cliff tool (Track B Stage 1 + 2) ---------------------------------
    #
    # Accumulate-then-commit, exactly like the Convert brush above and for
    # the same two reasons: UnitEditModel.begin_unit_edit(players) wants the
    # complete touched-player set up front, and here a node's own piece
    # isn't even decidable mid-drag (its neighbour set isn't complete until
    # the node after it exists). One undo record per stroke either way.
    #
    # A press-release without moving is a one-tile stroke, so Stage 1's
    # click-to-place gesture survives the widening unchanged: a lone node
    # has no neighbours, and cliff_chain falls that case back to the
    # Piece/Frame picker's own values rather than the connectivity table.
    #
    # Consequence worth naming: nothing repaints until release, since the
    # stroke mutates nothing before then. That's Convert's own established
    # trade and it answers the concern Stage 1's plan section raised about a
    # drag multiplying _after_unit_mutation()'s whole-canvas invalidation --
    # there is still exactly one per drag.

    def _begin_cliff_stroke(self) -> None:
        """Fixes the stroke's inputs at press time -- the picker can't change
        while the mouse is captured, and holding them keeps one drag
        self-consistent, the same reasoning _begin_convert_stroke() gives for
        its own destination combo."""
        self._cliff_stroke = None
        if self.scenario is None:
            return
        unit_const = self.cliff_piece_combo.currentData()
        if unit_const is None:
            self._log_status("Cliff: choose a family and piece first")
            return
        mm = self.scenario.map_manager
        self._cliff_stroke = cliff_chain.ChainStroke(
            map_width=mm.map_width,
            map_height=mm.map_height,
            existing=cliff_chain.existing_cliffs(self.scenario, mm.map_width, mm.map_height),
            fallback_const=unit_const,
            fallback_frame=self.cliff_frame_spin.value(),
        )

    def _cliff_stroke_tile(self, x: int, y: int) -> None:
        if self._cliff_stroke is not None:
            self._cliff_stroke.add_cursor_tile(x, y)

    def _end_cliff_stroke(self) -> None:
        """Commits the whole run through the same UnitEditModel.add() write
        path Place Unit uses.

        Always GAIA (status quo for every real cliff, and the AGENTS.md hard
        rule on GAIA rotation being a variant index, not an angle, is exactly
        why `rotation` is passed through verbatim below rather than via
        set_rotation(), which raises for a variant const by design).
        """
        stroke = self._cliff_stroke
        self._cliff_stroke = None
        if stroke is None or self.scenario is None:
            return
        nodes = stroke.nodes()
        if not nodes:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        label = "Place cliff" if len(nodes) == 1 else f"Place {len(nodes)} cliffs"
        # Selection is deliberately left untouched, unlike on_unit_place():
        # Terrain mode has no selection-visual concept, and _after_unit_
        # mutation() only rebuilds the pick index _selection resolves
        # against when self.mode == "units" (see its own docstring) -- so
        # pointing _selection at a placed cliff here would set it against
        # an index that this mutation doesn't rebuild.
        with self._unit_edit(model, label, [GAIA_PLAYER_ID]):
            for node in nodes:
                model.add(
                    GAIA_PLAYER_ID, node.unit_const, node.x, node.y,
                    rotation=float(node.rotation), initial_animation_frame=node.rotation,
                )
        # object_catalog.object_name(), not the usual _unit_name(): cliffs
        # are largely absent from the four library datasets _unit_name()'s
        # combined_object_name() merges (confirmed for 1339/2178/2199), so
        # that path falls through to "UNKNOWN_<id>" for most of them. The
        # piece combo's own label is already resolved correctly (Piece
        # combo, added via cliff_catalog.families()); reuse it instead of
        # re-resolving the name a second, worse way. A chain can end up on a
        # different piece than the combo shows (it inherits an adjacent
        # cliff's size), so only the single-node case names the piece.
        if len(nodes) == 1:
            status = f"Placed {self.cliff_piece_combo.currentText()} at ({nodes[0].x:g}, {nodes[0].y:g})"
        else:
            status = f"Placed a chain of {len(nodes)} cliffs"
            unresolved = sum(1 for node in nodes if not node.resolved)
            if unresolved:
                status += f" ({unresolved} kept the picked frame -- no measured shape for that junction)"
        # A5's own known gap, surfaced here rather than only in the tooltip --
        # placing a cliff with GAIA hidden looks exactly like the tool doing
        # nothing.
        if not self.show_gaia_action.isChecked():
            status += " (Show GAIA is off, so it won't be visible)"
        self._log_status(status)

    # -- Place Unit's wall branch (GH #98; was the Wall Run tool) and the
    # -- Wall Rectangle tool (2026-09-21 wall enclosure plan) ---------------
    #
    # A drag_shape path, not a stroke: MapView owns the rubber band and calls
    # on_shape_commit() once at release with the exact tile path, so there is
    # no per-tile half here at all. That matters beyond tidiness -- a wall's
    # variant index is a function of its neighbours, and a node's neighbour
    # set is not complete until the path is.

    def _place_shape(self) -> str:
        """MapView's place-shape query: "wall" when the Units catalog's
        picked const is one of the 8 wall consts, else ""."""
        return "wall" if self.units_panel.selected_object_const() in _wall_consts() else ""

    def _commit_wall_run(self, tiles, unit_const) -> None:
        """Places a whole wall run (any tile list: a Place Unit drag or a
        Wall Rectangle ring) as one undo record, with `unit_const` from the
        Units catalog.

        Follows _end_cliff_stroke()'s shape, with one structural difference
        forced by the junction rewrites: the touched-player set has to be
        complete BEFORE begin_unit_edit(), and it includes the owner of every
        pre-existing wall the run reshapes. add()/add_many() are refused
        inside a fields_only edit, so both halves run as one
        fields_only=False edit. wall_run.touched_players() is that set.

        GAIA is allowed: render.stored_rotation() keeps a GAIA wall's index
        (all 8 wall consts are rotation_is_variant), so it draws shaped.
        """
        tool_name = "Wall Rectangle" if self._current_tool == "wall_rect" else "Place Unit"
        # Wall Rectangle takes any catalog pick, and for Place Unit the pick
        # can change mid-drag after the press latched "wall".
        if unit_const not in _wall_consts():
            self._log_status(f"{tool_name}: pick a wall in the catalog")
            return
        player = self.units_panel.owner_id()
        model = self._ensure_unit_edits()
        if model is None:
            return
        mm = self.scenario.map_manager
        existing_tiles, existing_walls = wall_run.wall_scene(
            self.scenario, mm.map_width, mm.map_height
        )
        plan = wall_run.plan_wall_run(
            tiles,
            unit_const=unit_const,
            existing_tiles=existing_tiles,
            existing_walls=existing_walls,
        )
        if not plan.nodes and not plan.rewrites:
            self._log_status(f"{tool_name}: those {plan.skipped} tiles already hold a wall")
            return
        name = _unit_name(unit_const)
        label = "Place wall" if len(plan.nodes) == 1 else f"Place {len(plan.nodes)} walls"
        # splices=None, so one wholesale _after_unit_mutation() per drag --
        # the cliff tool's own trade. Selection is left untouched for the
        # reason _end_cliff_stroke() gives.
        with self._unit_edit(model, label, wall_run.touched_players(player, plan)):
            units = wall_run.apply_wall_plan(model, player, plan)
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        status = f"Placed {len(plan.nodes)} x {name} for {owner_text}"
        if plan.rewrites:
            status += f", reshaping {len(plan.rewrites)} adjacent"
        if plan.skipped:
            status += f" ({plan.skipped} tiles already held a wall)"
        # on_unit_place()'s clause: a run the live filter hides (Show Walls
        # off) looks exactly like the tool doing nothing.
        if units and not self._unit_filter.matches(player, units[0]):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)

    def on_unit_move(self, key: tuple[int, int], pos, modifiers) -> None:
        """MapView's eighth injected callable -- a press-drag-release past
        UNIT_DRAG_THRESHOLD_PX on a picked unit (b1.5). `key` is the
        (player_id, reference_id) MapView picked at press time; resolved
        late through the live index here, same as every other selection-key
        use, since add/remove/reassign elsewhere in the same session can
        invalidate it before the release arrives.
        """
        if self.scenario is None:
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            return
        if self._is_group_key(key):
            self._move_group(entry, pos, modifiers)
            return
        point = self._placement_point(pos, modifiers)
        if point is None:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        x, y = point
        unit = entry.unit
        # Effectively never fires in free mode (two float coordinates rarely
        # land on the same bits twice), and still correct for the snapped
        # path, which is the only reason it is still here.
        if (x, y) == (unit.x, unit.y):
            return
        mm = self.scenario.map_manager
        # The garrison rides along, onto the host's new point (GH #42).
        occupants = self._garrison_occupant_entries(model, [entry])
        old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
        old_own, old_tiles = self._unit_footprint(unit)
        idx = self._unit_list_index(entry.player_id, unit)
        players = sorted({entry.player_id} | {e.player_id for e in occupants})
        splices: list[UnitSplice] = []
        with self._unit_edit(model, "Move unit", players, splices, fields_only=True):
            model.set_position(unit, x, y, unit.z)
            new_own, new_tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)
            for occupant in occupants:
                self._move_occupant_to(model, occupant, x, y, splices)
        self._log_status(f"Moved {_unit_name(unit.unit_const)} to ({x:g}, {y:g})")

    def _group_delta(self, anchor, pos, modifiers, commit: bool):
        """(entries, dx, dy, clamped) for a group drag grabbing `anchor`, or
        None. Snapped moves by whole tiles from the press's cover tile; free
        moves the anchor onto the point, as a single free move does."""
        index = self.map_view._unit_index
        if index is None:
            return None
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return None
        pos = self.map_view._clamped_to_map_rect(pos)
        point, fell_back = self._resolve_placement_point(pos, modifiers)
        if commit and fell_back:
            self._log_status(FREE_PLACE_FALLBACK_MESSAGE)
        free = self._free_placement_requested(modifiers)
        if point is None:
            # Off the diamond in Stepped/Sloped: the ground-plane point, which the clamp then pulls in.
            ground = self.map_view.ground_map_point(pos)
            if ground is None:
                return None
            point = ground if free else (math.floor(ground[0]) + 0.5, math.floor(ground[1]) + 0.5)
        unit = anchor.unit
        if free:
            dx, dy = point[0] - unit.x, point[1] - unit.y
        else:
            grab = self._drag_grab_tile or (int(unit.x), int(unit.y))
            dx, dy = math.floor(point[0]) - grab[0], math.floor(point[1]) - grab[1]
        mm = self.scenario.map_manager
        dx, dy, clamped = clamp_group_delta(
            [e.unit for e in entries], dx, dy, mm.map_width, mm.map_height, whole_tiles=not free
        )
        return entries, dx, dy, clamped

    def _move_group(self, anchor, pos, modifiers) -> None:
        """on_unit_move for a key in a 2+ selection: every member moves by the
        same clamped delta, in one undo record (GH #75)."""
        self._pending_collapse = None
        resolved = self._group_delta(anchor, pos, modifiers, commit=True)
        if resolved is None:
            return
        entries, dx, dy, clamped = resolved
        if (dx, dy) == (0, 0):
            if clamped:
                self._log_status(f"{len(entries)} units already at the map edge; nothing moved")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        self._move_units(model, entries, dx, dy, f"Move {len(entries)} units")
        status = f"Moved {len(entries)} units by ({dx:g}, {dy:g})"
        if clamped:
            status += ", stopped at the map edge"
        self._log_status(status)

    def _move_occupant_to(self, model, entry, x: float, y: float, splices: list) -> None:
        """One garrisoned unit onto (x, y), inside its host's own open
        _unit_edit. `z` stays the occupant's own: the host's move doesn't
        change the ground under it, and the two are already co-located."""
        unit = entry.unit
        mm = self.scenario.map_manager
        old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
        old_own, old_tiles = self._unit_footprint(unit)
        idx = self._unit_list_index(entry.player_id, unit)
        model.set_position(unit, x, y, unit.z)
        new_own, new_tiles = self._unit_footprint(unit)
        splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
        self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)

    def _garrison_occupant_entries(self, model, entries) -> list:
        """Every unit garrisoned inside one of `entries`, as UnitEntry rows,
        deduped against `entries` itself (GH #42).

        Occupants are hidden by default now, so they cannot be selected
        alongside their host: a move or a delete that left them behind would
        strand them at the host's old point, or dangle their
        garrisoned_in_id. Every such op expands its selection through here
        first.

        A hidden occupant has no index entry to return, so one is
        synthesized with `order=-1`: these rows are only ever read for
        player_id/unit (the move loops and the delete pre-flight), never
        appended to a UnitIndex, and the real entry is reused when Show
        Garrisoned is on. The walk is transitive -- an occupant may itself
        hold a garrison -- and identity-keyed, since Unit isn't hashable.
        """
        model.warm_garrison_map()
        seen = {id(e.unit) for e in entries}
        pending = [e.unit for e in entries]
        occupants = []
        while pending:
            for occupant in _garrison_referrers(model, pending.pop()):
                if id(occupant) in seen:
                    continue
                seen.add(id(occupant))
                occupants.append(occupant)
                pending.append(occupant)
        if not occupants:
            return []
        owners = {id(u): pid for pid, units in enumerate(self.scenario.unit_manager.units) for u in units}
        index = self.map_view._unit_index
        rows = []
        for unit in occupants:
            player_id = owners[id(unit)]
            entry = index.entry_for_key(unit_pick.unit_key(player_id, unit)) if index is not None else None
            rows.append(entry or unit_pick.UnitEntry(player_id, unit, int(unit.x), int(unit.y), -1))
        return rows

    def _move_units(self, model, entries, dx: float, dy: float, label: str) -> None:
        """Moves every entry by (dx, dy) in ONE undo record, splicing each
        unit's footprint. `rotation` and `z` pass through verbatim.

        A host's garrison rides along (GH #42): its occupants sit at its own
        point and cannot be selected while hidden, so the same delta applies
        to them in the same undo record."""
        entries = list(entries) + self._garrison_occupant_entries(model, entries)
        players = sorted({e.player_id for e in entries})
        mm = self.scenario.map_manager
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, players, splices, fields_only=True):
            for entry in entries:
                unit = entry.unit
                old_bounds = unit_tile_bounds(unit, mm.map_width, mm.map_height)
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                model.set_position(unit, unit.x + dx, unit.y + dy, unit.z)
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
                self._patch_unit_index_for_move(entry.player_id, unit, old_bounds)

    def on_unit_drag_preview(self, key: tuple[int, int], pos, modifiers) -> None:
        """MapView's fourteenth injected callable -- one mouse-move of an
        in-progress unit drag, past UNIT_DRAG_THRESHOLD_PX (the mid-drag move
        preview). `key` is resolved late through the live index, same as
        on_unit_move's.

        Resolving lives here rather than in MapView because MapView holds no
        scenario: it has no player_colors, no team_indices and no sprites
        toggle. The toggle in particular must be read from
        self._sprites_enabled -- the WINDOW's copy -- and never off the live
        cache, which is a derived mirror rebuilt from scratch on every style
        switch and file open.

        Every failure to resolve clears the ghost rather than leaving a stale
        one at the last destination that DID resolve: a preview frozen a few
        tiles behind the cursor is a worse lie than no preview."""
        if self.scenario is None:
            self.map_view.set_unit_ghost()
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(key) if index is not None else None
        if entry is None:
            self.map_view.set_unit_ghost()
            return
        if self._is_group_key(key):
            self._preview_group(key, entry, pos, modifiers)
            return
        # The same resolver on_unit_move commits through, so the preview
        # cannot lie about where the unit lands -- including under free
        # placement's checkbox and its held-Alt one-off. Its fallback status
        # line is deliberately dropped here; see _resolve_placement_point.
        point, _fell_back = self._resolve_placement_point(pos, modifiers)
        if point is None:
            self.map_view.set_unit_ghost()
            return
        ghost = unit_at(entry.unit, *point)
        draws = self._ghost_sprite_draws(entry, ghost)
        if draws:
            self.map_view.set_unit_ghost(draws=draws)
            return
        polygons, color = self._ghost_mark(entry, ghost)
        self.map_view.set_unit_ghost(polygons=polygons, color=color)

    def _ghost_mark(self, entry, ghost):
        """(polygons, color) of the coloured-mark ghost for `entry` at `ghost`."""
        ghost_entry = unit_pick.UnitEntry(
            entry.player_id, ghost, int(ghost.x), int(ghost.y), entry.order
        )
        return (
            self.map_view._unit_polygons_for(ghost_entry),
            unit_mark_color(ghost, self.scenario.player_colors[entry.player_id]),
        )

    def _preview_group(self, key, anchor, pos, modifiers) -> None:
        """One ghost per group member at the same clamped delta the commit
        uses. Past GROUP_GHOST_CAP members only the anchor is ghosted."""
        resolved = self._group_delta(anchor, pos, modifiers, commit=False)
        if resolved is None:
            self.map_view.set_unit_ghost()
            return
        entries, dx, dy, _clamped = resolved
        members = entries
        if len(entries) > GROUP_GHOST_CAP:
            members = [anchor]
            # Once per drag: this runs on every mouse-move.
            if self._ghost_cap_logged != key:
                self._ghost_cap_logged = key
                self._log_status(f"Moving {len(entries)} units (preview shows the grabbed one only)")
        mm = self.scenario.map_manager
        draws: list = []
        marks: list = []
        for entry in members:
            ghost = unit_at(entry.unit, entry.unit.x + dx, entry.unit.y + dy)
            # A member nudged off-map stays off-map, and has nothing to draw.
            if unit_tile_bounds(ghost, mm.map_width, mm.map_height) is None:
                continue
            sprite = self._ghost_sprite_draws(entry, ghost)
            if sprite:
                draws.extend(sprite)
            else:
                marks.append(self._ghost_mark(entry, ghost))
        self.map_view.set_unit_ghosts(draws=draws, marks=marks)

    def _ghost_sprite_draws(self, entry, ghost) -> list:
        """The dragged unit's real sprite pieces at the ghost's destination,
        or [] to fall back to the coloured mark.

        **Flat is always the mark**, never a sprite. Flat does draw sprites
        now, as footprint-fitted icons, but a Flat ghost with an icon is out
        of scope here (GH #53 Part B left it as the mark)."""
        view = self.map_view
        if not self._sprites_enabled or view._terrain_style == "flat":
            return []
        if view._iso_proj is None or view._iso_elevations is None:
            return []
        cache = view._sloped_cache()
        return unit_sprite_draws_at(
            self.scenario,
            view._iso_proj,
            view._iso_elevations,
            None if cache is None else cache.corner_rise,
            entry.player_id,
            ghost,
            rotation_override=self._ghost_rotation_override(entry),
            # The window's live value, not the default: a ghost left on 1.0
            # would drag a tree at full size and snap it small on drop.
            tree_scale=self._layers.tree_scale,
            hero_glow=self._layers.hero_glow,
        )

    def _ghost_rotation_override(self, entry) -> float | None:
        """This unit's wall-connectivity-derived rotation, resolved once per
        drag rather than per mouse-move.

        Two walks sit behind it that must not run per frame: the memo behind
        wall_variant_rotation_overrides() rebuilds over every unit in the file
        whenever the unit generation bumps, and _unit_list_index() is a linear
        scan by identity. The cache is keyed on the overrides dict OBJECT, not
        on a generation number -- the memo hands back a new dict whenever it
        rebuilds, so an `is` test against the one this value was derived from
        is exactly "still the same answer", with nothing to keep in sync.

        Boundary worth knowing during an in-app pass: this is the override the
        unit's SOURCE neighbours give it. A wall dropped somewhere with
        different neighbours has its index re-derived by the render on
        release, so the ghost can show the wall shape it has now rather than
        the one it lands as. Recomputing connectivity per move is the
        whole-file walk this preview exists to avoid."""
        overrides = wall_variant_rotation_overrides(self.scenario)
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        cached = self._ghost_rotation
        if cached is None or cached[0] is not overrides:
            cached = (overrides, {})
            self._ghost_rotation = cached
        memo = cached[1]
        # Keyed per unit so a group drag (GH #75) scans each member once, not once per frame.
        if key not in memo:
            index = self._unit_list_index(entry.player_id, entry.unit)
            memo[key] = overrides.get((entry.player_id, index))
        return memo[key]

    def on_unit_nudge(self, dx: int, dy: int, modifiers) -> bool:
        """MapView's ninth injected callable -- an arrow key in Units mode
        (b1.5's second half, generalized to the whole selection by b2.4).

        **The return value is load-bearing, not incidental.** True means the
        nudge happened and owns the key; False means it refused, and MapView
        falls through to a view_pan_* pan on the same press (see its
        keyPressEvent). Every early return below is a case where panning is
        the right answer, "nothing is selected" most of all.
        One record per press, matching every other discrete-keypress edit in
        this app (e.g. the ]/[ tool-value step) -- not merged across
        repeats, unlike a typed-digit spinbox edit. That holds for a group
        nudge too: every selected unit moves in the SAME undo step, not one
        step per unit, since begin_unit_edit(players) already captures every
        touched player's whole list up front (unit_model.py's own
        PlayerListSnapshot)."""
        if self.scenario is None or not self._selection:
            return False
        index = self.map_view._unit_index
        if index is None:
            return False
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return False
        model = self._ensure_unit_edits()
        if model is None:
            return False
        step = _UNIT_NUDGE_STEP_SHIFT if modifiers & Qt.ShiftModifier else _UNIT_NUDGE_STEP
        label = "Nudge unit" if len(entries) == 1 else f"Nudge {len(entries)} units"
        self._move_units(model, entries, dx * step, dy * step, label)
        if len(entries) == 1:
            unit = entries[0].unit
            self._log_status(f"Moved {_unit_name(unit.unit_const)} to ({unit.x:g}, {unit.y:g})")
        else:
            self._log_status(f"Moved {len(entries)} units")
        return True

    def on_unit_rotate(self, steps: int, gate_steps: int | None = None) -> None:
        """Turns every selected unit whose `rotation` is genuinely an angle by
        `steps` whole stored frames, and cycles every selected gate through
        `gate_steps` of its own four orientations. Phase 3.5b's b3.

        Modelled on on_unit_nudge: the whole selection moves in ONE undo
        record, since begin_unit_edit(players) already captures every touched
        player's list up front. Positive steps rotate clockwise on screen (see
        unit_rotation.rotate_step and gate_orientation.cycle_const).

        **Each unit rotates about its own centre.** Orbiting a selection about
        a shared pivot is deliberately out of scope -- that is a group
        transform, not a per-unit field edit, and nothing else in the unit
        edit path moves a unit the caller did not name.

        `gate_steps` defaults to `steps` and exists because a gate step is 45
        degrees while a rotation step is one stored frame: the coarse path
        passes a quarter turn in each unit, so a mixed selection turns the same
        visible amount instead of the gates silently doing a full 4-step
        identity cycle.

        A gate whose new footprint would hang off the map is skipped and
        counted: (4, 1) to (1, 4) grows three tiles on the other axis, and
        this is the layer that knows the map's dimensions. The model stays
        dimension-free.

        Selected units whose semantics are VARIANT (walls, trees, most GAIA
        doodads) or INERT (every other single-frame graphic) are skipped
        rather than refused: a marquee over a village will always include
        some, and failing the whole action for them would make Rotate unusable
        exactly where it is most wanted.

        Mode-gated explicitly, unlike nudge/delete: those arrive through
        MapView's injected callables and so are Units-mode-only for free,
        while this is a QAction whose shortcut is live in every mode.
        """
        if self.scenario is None or self.mode != "units" or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        gate_steps = steps if gate_steps is None else gate_steps
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        rotatable = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
        gates = [e for e in entries if gate_orientation.is_gate(e.unit.unit_const)]
        cyclable = [e for e in gates if self._gate_cycle_fits(e.unit, gate_steps)]
        blocked = len(gates) - len(cyclable)
        skipped = len(entries) - len(rotatable) - len(gates)
        if not rotatable and not cyclable:
            reasons = []
            if blocked:
                reasons.append(f"{blocked} gate(s) would hang off the map edge")
            if skipped:
                reasons.append(f"{skipped} selected unit(s) store a graphic variant, not an angle")
            if reasons:
                self._log_status(f"Rotate: {'; '.join(reasons)}")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        touched = rotatable + cyclable
        label = "Rotate unit" if len(touched) == 1 else f"Rotate {len(touched)} units"
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, sorted({e.player_id for e in touched}), splices, fields_only=True):
            for entry in rotatable:
                unit = entry.unit
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                angle_count = unit_rotation.angle_count_for(unit.unit_const)
                model.set_rotation(unit, unit_rotation.rotate_step(unit.rotation, angle_count, steps))
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            # cyclable's set_unit_const() changes span, which
            # unit_pick.patch_index_for_move() is explicitly not scoped to
            # (see its own docstring) -- one full rebuild covers the whole
            # batch, including the rotatable half above, which never needed
            # one on its own.
            for entry in cyclable:
                unit = entry.unit
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                model.set_unit_const(unit, gate_orientation.cycle_const(unit.unit_const, gate_steps))
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
            if cyclable:
                self._rebuild_unit_index()
        self._log_status(self._rotate_status(rotatable, cyclable, blocked, skipped))

    def _gate_cycle_fits(self, unit, gate_steps: int) -> bool:
        """Whether this gate's next footprint still fits on the map.

        Measured from span_low_corner(), the same low corner
        UnitEditModel.set_unit_const() re-anchors from, so the two sides
        cannot disagree about which tiles the swap would claim. Both ends are
        checked: a span-4 axis puts the low corner two tiles below the unit's
        own, which can already be negative near the map's origin.
        """
        new_const = gate_orientation.cycle_const(unit.unit_const, gate_steps)
        low_x, low_y = span_low_corner(unit)
        span_x, span_y = tile_span(new_const, NON_BUILDING_SPAN)
        mm = self.scenario.map_manager
        return (
            low_x >= 0
            and low_y >= 0
            and low_x + span_x <= mm.map_width
            and low_y + span_y <= mm.map_height
        )

    def _rotate_status(self, rotatable, cyclable, blocked: int, skipped: int) -> str:
        """on_unit_rotate's report line: what turned, what cycled, what was
        left alone. Names the single unit when exactly one thing was touched,
        the way every other unit action's status does."""
        parts = []
        if rotatable:
            if len(rotatable) == 1 and not cyclable:
                unit = rotatable[0].unit
                count = unit_rotation.angle_count_for(unit.unit_const)
                facing = unit_rotation.rotation_to_facing(unit.rotation, count)
                parts.append(f"Rotated {_unit_name(unit.unit_const)} to facing {facing}/{count}")
            else:
                parts.append(f"Rotated {len(rotatable)} unit{'' if len(rotatable) == 1 else 's'}")
        if cyclable:
            if len(cyclable) == 1 and not rotatable:
                unit = cyclable[0].unit
                parts.append(f"Cycled {_unit_name(unit.unit_const)} to ({unit.x:g}, {unit.y:g})")
            else:
                parts.append(f"cycled {len(cyclable)} gate{'' if len(cyclable) == 1 else 's'}")
        done = ", ".join(parts)
        if blocked:
            done += f"; {blocked} gate(s) skipped (would hang off the map edge)"
        if skipped:
            done += f"; {skipped} skipped (rotation is a graphic variant, not an angle)"
        return done

    def on_unit_variant(self, steps: int = 0, randomize: bool = False) -> None:
        """Steps (or randomizes) the graphic variant of every selected cyclable
        unit, in one undo record. Modelled on on_unit_rotate(): non-cyclable
        units are skipped and counted rather than refused, and the QAction's
        shortcut is live in every mode, so the mode gate is explicit.
        """
        if self.scenario is None or self.mode != "units" or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        cyclable = [e for e in entries if unit_variant.is_cyclable(e.unit.unit_const)]
        skipped = len(entries) - len(cyclable)
        if not cyclable:
            if skipped:
                self._log_status(f"Cycle Variant: {skipped} selected unit(s) have no graphic variants")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        verb = "Randomize" if randomize else "Cycle"
        label = f"{verb} unit variant" if len(cyclable) == 1 else f"{verb} {len(cyclable)} units' variant"
        splices: list[UnitSplice] = []
        with self._unit_edit(model, label, sorted({e.player_id for e in cyclable}), splices, fields_only=True):
            for entry in cyclable:
                unit = entry.unit
                old_own, old_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(entry.player_id, unit)
                angle_count = unit_rotation.angle_count_for(unit.unit_const)
                variant_count = unit_variant.variant_count_for(unit.unit_const)
                if randomize:
                    value = unit_variant.random_variant(unit.rotation, angle_count, variant_count, self._variant_rng)
                else:
                    value = unit_variant.cycle_step(unit.rotation, angle_count, variant_count, steps)
                model.set_variant(unit, value)
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, new_own, old_tiles, new_tiles))
        if len(cyclable) == 1:
            unit = cyclable[0].unit
            count = unit_variant.variant_count_for(unit.unit_const)
            done = f"{'Randomized' if randomize else 'Cycled'} {_unit_name(unit.unit_const)} to variant {int(unit.rotation)}/{count}"
        else:
            done = f"{'Randomized' if randomize else 'Cycled'} {len(cyclable)} units' variant"
        if skipped:
            done += f"; {skipped} skipped (this object has no graphic variants)"
        self._log_status(done)

    def on_unit_rotate_coarse(self, direction: int) -> None:
        """Rotate by the closest whole number of frames to a quarter turn.

        The coarse step is its own action rather than a Shift modifier on the
        fine one: a QAction's shortcut IS the key combination, so unlike
        on_unit_nudge (which gets a live `modifiers` from MapView's key event)
        there is no modifier for this path to read. Four keybind rows, two
        entry points.

        Gates count their own quarter turn: two of their four 45-degree
        orientation steps, regardless of what frame count the selection's
        first rotatable unit has.
        """
        gate_steps = direction * gate_orientation.QUARTER_TURN_STEPS
        entry = self._first_rotatable_entry()
        if entry is None:
            if self._first_cyclable_entry() is None:
                self.on_unit_rotate(direction)  # nothing to turn; reuse its status/no-op path
            else:
                self.on_unit_rotate(direction, gate_steps=gate_steps)
            return
        angle_count = unit_rotation.angle_count_for(entry.unit.unit_const)
        steps = direction * unit_rotation.quarter_turn_steps(angle_count)
        self.on_unit_rotate(steps, gate_steps=gate_steps)

    def _first_rotatable_entry(self):
        """The first selected entry whose rotation is an angle, or None.

        A mixed selection can span several angle_counts, and a quarter turn is
        a different number of frames in each. This picks the step size from
        the first rotatable unit and applies it to all of them, so one press
        stays one undo record; a selection of mixed graphics turns by the same
        frame count rather than the same visual angle. Rotating each by its
        own quarter turn is the alternative, and it is worse: two adjacent
        units would silently drift apart in facing on repeated presses.
        """
        index = self.map_view._unit_index
        if index is None or self.mode != "units":
            return None
        for key in self._selection:
            entry = index.entry_for_key(key)
            if entry is not None and unit_rotation.rotation_is_angle(entry.unit.unit_const):
                return entry
        return None

    def _first_cyclable_entry(self):
        """The first selected gate entry, or None.

        Sibling of _first_rotatable_entry() rather than a widening of it: a
        gate-only selection has no angle_count to take a step size from, and
        without this the coarse path would fall through to a single 45-degree
        step instead of the quarter turn the action promises.
        """
        index = self.map_view._unit_index
        if index is None or self.mode != "units":
            return None
        for key in self._selection:
            entry = index.entry_for_key(key)
            if entry is not None and gate_orientation.is_gate(entry.unit.unit_const):
                return entry
        return None

    def on_unit_delete(self, modifiers) -> None:
        """MapView's tenth injected callable -- the Delete key in Units mode
        (b1.6).

        No confirm dialog (revised from D4's original design): every other
        edit tool in this app (Draw, Elevate, Delete Unit's own reassign/
        move/place siblings) commits straight to the undo-backed
        EditHistory with no "are you sure", and Ctrl+Z is the same one
        keystroke away regardless. remove()'s own garrison-reference guard
        stays a hard refusal, unrelated to confirmation -- a dangling
        reference is legal on disk and the in-game editor permits it, so
        this must not additionally hard-block on one.

        Deferred from D4's full design: naming which triggers reference this
        unit's reference_id needs a per-trigger vocabulary scan
        (trigger_fields.field_specs() over every condition/effect, matched
        against the Unit/Unit[] presentations) that this slice doesn't
        build.

        b2.4 generalizes this to the whole selection, as one undo record --
        but unlike nudge/move, a group delete can be PARTIALLY refused (some
        selected units garrison-referenced, others not), and `_unit_edit`
        must never be entered for zero real changes (its own docstring: a
        spurious enter records a phantom undo step). So the referencing
        check runs as a pre-flight over the whole group, via
        UnitEditModel.referencing() -- the same guard remove() itself
        applies, factored out so both call sites can't drift apart -- and
        only the units it clears go into one commit.
        """
        if self.scenario is None or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        if not entries:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        # One garrison reverse-map walk for the whole pre-flight, instead of
        # leaving the first referencing() call to find it cold.
        model.warm_garrison_map()
        # Deleting a host deletes what it holds, in the same undo step (GH
        # #42): occupants are hidden by default, so they can't be selected
        # alongside it, and leaving them would dangle their link. The
        # refusal below therefore only survives for a unit whose holder is
        # NOT in the batch -- a dangling reference or an unselected host.
        occupants = self._garrison_occupant_entries(model, entries)
        cascaded_ids = {id(e.unit) for e in occupants}
        entries = list(entries) + occupants
        batch = {id(e.unit) for e in entries}
        # model.referencing(), not _garrison_referrers(): this decides what
        # remove_many() will accept, and the model keeps the -1 bucket. So a
        # unit whose own reference_id is -1 is still refused here, exactly as
        # it was before the cascade, rather than raising out of the model.
        removable = [e for e in entries if all(id(u) in batch for u in model.referencing(e.unit))]
        blocked = len(entries) - len(removable)
        if not removable:
            self._log_status("Delete failed: referenced by a garrison")
            QMessageBox.warning(
                self, "Cannot delete unit" if len(entries) == 1 else "Cannot delete units",
                "Referenced by another unit's garrison." if len(entries) == 1
                else "All selected units are referenced by another unit's garrison.",
            )
            return
        players = sorted({e.player_id for e in removable})
        label = "Delete unit" if len(removable) == 1 else f"Delete {len(removable)} units"
        if len(removable) == 1:
            # Splice-eligible (Batch D's D5's "single Delete"): a batch of
            # 2+ removals keeps the wholesale path below instead, since each
            # unit's own list index (render_cache.UnitSplice.index, needed
            # for wall_variant_rotation_overrides' key) shifts under a
            # LATER removal from the same player's list -- see UnitSplice's
            # own docstring. That 2+ path passes no splices, so no per-unit
            # index has to survive a later removal and it takes remove_many().
            entry = removable[0]
            unit = entry.unit
            old_own, old_tiles = self._unit_footprint(unit)
            idx = self._unit_list_index(entry.player_id, unit)
            splices: list[UnitSplice] = []
            with self._unit_edit(model, label, players, splices):
                model.remove(unit)
                splices.append(UnitSplice(entry.player_id, idx, unit, old_own, None, old_tiles, ()))
                # No patch_index_for_remove() exists (see unit_pick.py's own
                # D4 note) -- a full rebuild is the only path for a removal.
                self._rebuild_unit_index()
        else:
            with self._unit_edit(model, label, players):
                model.remove_many([e.unit for e in removable])
        # Only the cascade that actually ran, not every occupant found: a
        # host whose own removal was refused takes its occupants with it.
        cascaded = sum(1 for e in removable if id(e.unit) in cascaded_ids)
        carried = f" ({cascaded} garrisoned)" if cascaded else ""
        if blocked:
            self._log_status(f"Deleted {len(removable)}{carried}; {blocked} refused (garrison reference)")
            QMessageBox.warning(
                self, "Some units not deleted",
                f"{blocked} unit(s) are referenced by another unit's garrison and were not deleted.",
            )
        elif len(removable) - cascaded == 1:
            name = next(_unit_name(e.unit.unit_const) for e in removable if id(e.unit) not in cascaded_ids)
            self._log_status(f"Deleted {name}{carried}")
        else:
            self._log_status(f"Deleted {len(removable) - cascaded} units{carried}")

    def _refresh_selection_after_filter(self) -> None:
        """A filter toggle can hide currently selected units. Rebuild the
        index, then drop every selection key that no longer resolves --
        leaving a highlight around a unit that no longer paints would be a
        cue pointing at nothing.

        The index rebuild also covers the footprint overlay, which is index-
        driven and live outside Units mode; the selection half is Units-mode
        only, as before."""
        if not self._needs_unit_index():
            return
        self._rebuild_unit_index()
        if self.mode != "units":
            return
        self._refresh_selection_view()

    def _set_all_players(self, checked: bool) -> None:
        """Bulk-sets the per-player checkboxes with signals blocked, then
        applies once. Without the block this would fire _on_filter_changed
        up to eight times, each one evicting and recompositing the whole
        canvas."""
        for action in self.player_actions.values():
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)
        self._on_filter_changed()

    def _set_all_filters(self, checked: bool) -> None:
        """Same as _set_all_players, but also the six kind toggles -- the
        menu's "Show All"/"Hide All" shortcuts at the bottom."""
        for action in (
            self.show_gaia_action,
            self.show_trees_action,
            self.show_walls_action,
            self.show_eye_candy_action,
            self.show_invisible_action,
            self.show_garrisoned_action,
        ):
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)
        for action in self.player_actions.values():
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)
        self._on_filter_changed()

    def _current_unit_filter(self) -> UnitFilter:
        """Reads the menu's checkboxes into a UnitFilter.

        players stays None when every player is checked rather than becoming
        a full frozenset, so an all-checked menu produces a filter equal to
        UnitFilter() -- which is what lets set_unit_filter()'s
        unchanged-filter guard skip a pointless full-canvas eviction.
        """
        checked = frozenset(pid for pid, action in self.player_actions.items() if action.isChecked())
        all_checked = len(checked) == len(self.player_actions)
        return UnitFilter(
            show_gaia=self.show_gaia_action.isChecked(),
            show_trees=self.show_trees_action.isChecked(),
            show_walls=self.show_walls_action.isChecked(),
            show_eye_candy=self.show_eye_candy_action.isChecked(),
            show_invisible=self.show_invisible_action.isChecked(),
            show_garrisoned=self.show_garrisoned_action.isChecked(),
            players=None if all_checked else checked,
        )

    def _on_filter_changed(self) -> None:
        """Applies the menu's current state to the live chunk cache and
        repaints.

        Two steps, both required: the cache's own set_unit_filter() rebuilds
        its unit structures and evicts every composited chunk (units are
        baked into chunk PIXELS), and then MapView.invalidate_region()
        schedules the actual Qt repaint. Doing only the first leaves the old
        pixels on screen until something else happens to repaint them.
        """
        self._unit_filter = self._current_unit_filter()
        if self.scenario is None or self._cache is None:
            return
        self._cache.set_unit_filter(self._unit_filter)
        canvas_w, canvas_h = self._cache.canvas_dims(0)
        self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        self._refresh_selection_after_filter()
        self._log_status(f"Unit filter: {self._filter_summary()}")

    def _on_sprites_toggled(self, checked: bool) -> None:
        """Turns real .sld unit sprites on or off in Stepped -- P3-g's toggle.

        Stores the window's own copy first, unconditionally, so the choice
        survives being made with no map open and is picked up by the next
        _render_current().

        Wrapped in the same wait-cursor/setEnabled(False) busy pattern
        on_fill() uses, and for a stronger reason: turning sprites ON pays a
        0.5-4.1s cold .sld decode inside set_sprites_enabled() (P3-g4's
        measurement), which is long enough that the window would otherwise
        sit frozen with nothing to explain it.

        It deliberately does NOT copy on_fill's `if self._busy: return` early
        exit, and that asymmetry is the trap here rather than an oversight.
        on_fill is a plain method, so bailing out of it costs nothing. This is
        a `toggled` slot, which Qt fires AFTER the QAction's checked state has
        already flipped -- so an early return would leave the menu showing
        checked while the cache stayed off, a desync the user could only
        escape by toggling twice. self._busy is still SET for the duration
        (restored in the finally), so this call still blocks a reentrant
        on_fill/refresh_map; it just never refuses to run on account of it.
        setEnabled(False) disables the whole window for the warm anyway, so
        the re-entrancy on_fill guards against cannot reach the toolbar.

        Any future programmatic show_sprites_action.setChecked() needs the
        same care: unblocked, it lands here and fires a multi-second warm.
        Block signals around it (the pattern _set_all_players already uses)
        unless a real cache rebuild is actually wanted."""
        self._sprites_enabled = checked
        # View > Layers' farm row is requires_sprites, so it greys/ungreys
        # the moment this flips, with no map change to trigger the usual
        # re-gate.
        self._update_tool_enabled()
        # Before the processEvents() below, which is a real event-loop turn a
        # warm tick could run in -- and before set_sprites_enabled() bumps
        # the source gen out from under one.
        self._cancel_warms()
        if self.scenario is None or self._cache is None:
            return
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        was_busy = self._busy
        self._busy = True
        try:
            t0 = time.perf_counter()
            self._cache.set_sprites_enabled(checked)
            # refresh_canvas_dims() FIRST (Track P3-g6): SlopedChunkCache's
            # canvas_dims() can grow once sprites are on (a low elev_step_pct
            # stop's canvas fix), and invalidate_region()'s own rect below
            # must reach that newly-widened strip, not the pre-toggle size --
            # a no-op call on every OTHER style/config, whose canvas_dims()
            # never moves.
            self.map_view.refresh_canvas_dims()
            self.map_view.invalidate_region((0, 0, *self._cache.canvas_dims(0)))
            # Sloped farms drape only with sprites on, so their outlines change shape here.
            self.map_view.refresh_footprint_overlay()
            elapsed = time.perf_counter() - t0
        finally:
            self._busy = was_busy
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
        self._log_status(f"Show sprites: {'on' if checked else 'off'} (applied in {elapsed:.2f}s)")

    def _on_layer_toggled(self, layer_id: str, on: bool) -> None:
        """Turns one View > Layers row on or off.

        Stores the window's own copy first, unconditionally, so the choice
        survives being made with no map open and is picked up by the next
        _render_current() -- same contract _on_sprites_toggled() documents,
        including why this is not allowed to early-return on self._busy (a
        `toggled` slot fires after the QAction has already flipped, so
        bailing out would leave the menu and the cache desynced).

        _cancel_warms() before set_layers(), not after: a LevelWarmer
        generator already in flight was built with the OLD build-time layer
        values and would install a stale layer over the fresh one.

        Both of the last two steps are required, not one or the other. The
        cache eviction happens inside set_layers(); MapView still holds its
        own painted scene items, so it needs its own invalidate_region() to
        repaint from the now-empty cache."""
        self._layers = dataclasses.replace(self._layers, **{layer_id: on})
        self._cancel_warms()
        if self.scenario is None or self._cache is None:
            self._update_tool_enabled()
            return
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        was_busy = self._busy
        self._busy = True
        try:
            t0 = time.perf_counter()
            self._cache.set_layers(self._layers)
            self.map_view.invalidate_region((0, 0, *self._cache.canvas_dims(0)))
            elapsed = time.perf_counter() - t0
        finally:
            self._busy = was_busy
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
        label = next(s.label for s in view_layers.LAYERS if s.layer_id == layer_id).replace("&", "")
        self._log_status(f"{label}: {'on' if on else 'off'} (applied in {elapsed:.2f}s)")

    def _filter_summary(self) -> str:
        if self._unit_filter.is_default:
            return "showing all units"
        parts = []
        if not self._unit_filter.show_gaia:
            parts.append("GAIA hidden")
        if not self._unit_filter.show_trees:
            parts.append("trees hidden")
        if not self._unit_filter.show_walls:
            parts.append("walls hidden")
        if not self._unit_filter.show_eye_candy:
            parts.append("eye candy hidden")
        if not self._unit_filter.show_invisible:
            parts.append("invisible objects hidden")
        if not self._unit_filter.show_garrisoned:
            parts.append("garrisoned units hidden")
        if self._unit_filter.players is not None:
            shown = sorted(self._unit_filter.players)
            parts.append(f"players {shown}" if shown else "no players")
        return ", ".join(parts)

    def _build_tool_overflow_button(self, toolbar) -> None:
        """A labelled "More Tools" dropdown for whichever tool buttons don't
        fit row 1 at the current window width, replacing Qt's own anonymous
        `>>` chevron.

        Same QToolButton + QMenu shape as _build_filters_button, for the
        same reason (keyboard navigation and testability for free). Starts
        hidden: partition()'s own two-pass contract is "if everything fits,
        no button and no overflow", and _apply_toolbar_overflow() is what
        shows it once something actually needs it.
        """
        self.more_tools_button = QToolButton()
        self.more_tools_button.setText("More Tools ▾")
        self.more_tools_button.setPopupMode(QToolButton.InstantPopup)
        self.more_tools_button.setToolTip(
            "Tools that don't fit in the toolbar at the current window width"
        )
        self.more_tools_menu = QMenu(self.more_tools_button)
        self.more_tools_button.setMenu(self.more_tools_menu)
        self.more_tools_action = toolbar.addWidget(self.more_tools_button)
        self.more_tools_action.setVisible(False)

    def _cache_tool_button_widths(self) -> None:
        """One-time measurement, called from showEvent() -- pre-show()
        sizeHint()s are unreliable, so this must run after realization, not
        inside _build_toolbar(). Reads widgetForAction() while every tool
        action is still a toolbar member (Stage 3 hasn't moved anything into
        the overflow menu yet), so partition() never has to re-measure a
        tool currently sitting in the dropdown -- widgetForAction() returns
        None once an action has been removed from a toolbar.

        A hidden (mode-inapplicable, see ToolDef.modes) action's widget
        still reports its real sizeHint() -- confirmed empirically -- so
        this doesn't need to visit every mode first.
        """
        for tool in settings.TOOLS:
            action = getattr(self, f"{tool.tool_id}_action")
            widget = self.main_toolbar.widgetForAction(action)
            self._tool_button_widths[tool.tool_id] = widget.sizeHint().width() if widget else 0

        # The pinned prefix's width (Mode/Elevation View/Filters and their
        # labels/separators) isn't measured directly -- it's whatever the
        # toolbar's own sizeHint doesn't account for in currently-visible
        # tool buttons. More Tools starts hidden (see
        # _build_tool_overflow_button), so it contributes nothing here.
        visible_tool_width = sum(
            self._tool_button_widths[t.tool_id]
            for t in settings.TOOLS
            if getattr(self, f"{t.tool_id}_action").isVisible()
        )
        self._pinned_toolbar_width = self.main_toolbar.sizeHint().width() - visible_tool_width

    def _apply_toolbar_overflow(self) -> None:
        """Stage 3's one apply function -- computes bar-membership and
        menu-membership together from (active mode, measured widths), and
        is the only place either is ever touched. Called from
        _update_tool_enabled() (mode/tool changes) and resizeEvent()
        (width changes); both routes converge here so there is exactly one
        place that decides where a tool action currently lives.

        No-ops until _cache_tool_button_widths() has run at least once
        (see _tool_overflow_measured), and guards its own re-entrancy since
        removeAction()/insertAction() trigger a relayout that re-fires
        resizeEvent().
        """
        if not self._tool_overflow_measured or self._toolbar_overflow_updating:
            return
        self._toolbar_overflow_updating = True
        try:
            applicable_ids = [
                t.tool_id for t in settings.TOOLS if tool_applicable(t.tool_id, self.mode)
            ]
            applicable = set(applicable_ids)

            # Mode-inapplicable tools are never overflow candidates -- they
            # stay (invisible, via _update_tool_enabled's own setVisible
            # loop) on the main toolbar, exactly as before Stage 3 existed.
            # This also reclaims a tool that was sitting in the menu from a
            # previous mode and has since become inapplicable.
            for tool in settings.TOOLS:
                if tool.tool_id in applicable:
                    continue
                action = getattr(self, f"{tool.tool_id}_action")
                if action not in self.main_toolbar.actions():
                    self.main_toolbar.insertAction(self.more_tools_action, action)

            item_widths = [(t, self._tool_button_widths.get(t, 0)) for t in applicable_ids]
            available_px = self.width() - self._pinned_toolbar_width
            more_button_px = self.more_tools_button.sizeHint().width()
            on_bar_ids, overflow_ids = partition(available_px, item_widths, more_button_px)
            overflow_set = set(overflow_ids)

            for tool_id in overflow_ids:
                action = getattr(self, f"{tool_id}_action")
                if action in self.main_toolbar.actions():
                    self.main_toolbar.removeAction(action)
            for tool_id in on_bar_ids:
                action = getattr(self, f"{tool_id}_action")
                if action not in self.main_toolbar.actions():
                    self.main_toolbar.insertAction(self.more_tools_action, action)

            self.more_tools_menu.clear()
            for tool_id in overflow_ids:
                self.more_tools_menu.addAction(getattr(self, f"{tool_id}_action"))

            self.more_tools_action.setVisible(bool(overflow_ids))
            if self._current_tool in overflow_set:
                label = _TOOL_LABELS.get(self._current_tool, self._current_tool)
                self.more_tools_button.setText(f"{label} ▾")
            else:
                self.more_tools_button.setText("More Tools ▾")
        finally:
            self._toolbar_overflow_updating = False

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        # Kept for the Stage 3 overflow logic (_apply_toolbar_overflow),
        # which needs to add/remove tool actions on this specific toolbar.
        self.main_toolbar = toolbar

        toolbar.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        # "Units" (plural) rather than the plan's "Unit", matching Triggers.
        # on_mode_changed() lowercases, so the label decides the gate string
        # everywhere -- self.mode == "units".
        self.mode_combo.addItems(
            ["View", "Terrain", "Units", "Triggers", "Map Options", "Players", "Diplomacy", "Messages"]
        )
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        toolbar.addWidget(self.mode_combo)
        toolbar.addSeparator()

        # Separate from Mode/the tool group -- Elevation View picks the
        # rendering pipeline (Flat's plain, un-displaced terrain vs.
        # Stepped's real per-tile Z-height compositor -- the only one of the
        # two that actually shows elevation at all now that Phase 3's
        # follow-up removed the old brightness-based elevation hint), an
        # orthogonal axis to View/Terrain mode or which brush tool is active.
        # Internal identifiers/log lines still say "terrain style" (the
        # underlying concept -- which renderer is active); only this
        # user-facing label and combo text changed to "Elevation View".
        # Defaults to Stepped, not Flat -- Flat alone no longer conveys
        # elevation at all, so Stepped is the more useful default view.
        toolbar.addWidget(QLabel(" Elevation View: "))
        self.terrain_style_combo = QComboBox()
        self.terrain_style_combo.addItems(list(STYLE_LABELS))
        self.terrain_style_combo.setCurrentText("Stepped")  # before connect(): no spurious signal
        self.terrain_style_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.terrain_style_combo.setToolTip(
            "How the map shows height. Flat: no elevation. Stepped: each tile raised to its "
            "height, as blocks. Sloped: the ground slopes smoothly between heights."
        )
        self.terrain_style_combo.currentTextChanged.connect(self.on_terrain_style_changed)
        toolbar.addWidget(self.terrain_style_combo)
        toolbar.addSeparator()

        self._build_filters_button(toolbar)
        toolbar.addSeparator()

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.pan_action.setEnabled(False)  # re-enabled by _update_tool_enabled() once a map loads
        self.pan_action.setToolTip("Drag to pan the map")
        self.pan_action.toggled.connect(lambda on: on and self._on_tool_selected("pan"))
        tool_group.addAction(self.pan_action)
        toolbar.addAction(self.pan_action)

        self.draw_action = QAction("Draw", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.draw_action.setToolTip(
            "Paint the selected terrain type over the brush footprint (Terrain mode) -- drag to paint a trail"
        )
        self.draw_action.toggled.connect(lambda on: on and self._on_tool_selected("draw"))
        tool_group.addAction(self.draw_action)
        toolbar.addAction(self.draw_action)

        self.fill_action = QAction("Paint Can", self)
        self.fill_action.setCheckable(True)
        self.fill_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.fill_action.setToolTip(
            "Flood-fill the connected region of matching terrain with the selected "
            "terrain type (Terrain mode) -- one click, no drag"
        )
        self.fill_action.toggled.connect(lambda on: on and self._on_tool_selected("fill"))
        tool_group.addAction(self.fill_action)
        toolbar.addAction(self.fill_action)

        self.elevation_action = QAction("Elevate", self)
        self.elevation_action.setCheckable(True)
        self.elevation_action.setEnabled(False)
        self.elevation_action.setToolTip(
            "Left click/drag to raise the brush footprint's elevation by 1; right click/drag "
            "(or Shift+left) to lower (Terrain mode, square maps only)"
        )
        self.elevation_action.toggled.connect(lambda on: on and self._on_tool_selected("elevation"))
        tool_group.addAction(self.elevation_action)
        toolbar.addAction(self.elevation_action)

        self.set_level_action = QAction("Set Elevation", self)
        self.set_level_action.setCheckable(True)
        self.set_level_action.setEnabled(False)
        self.set_level_action.setToolTip(
            "Click/drag to set the brush footprint's elevation to the level below "
            "(Terrain mode, square maps only)"
        )
        self.set_level_action.toggled.connect(lambda on: on and self._on_tool_selected("set_level"))
        tool_group.addAction(self.set_level_action)
        toolbar.addAction(self.set_level_action)

        # Draw Line / Draw Rectangle (ToolDef.drag_shape). Press-drag-
        # release, previewing as a highlight and committing one undo record
        # at release -- nothing is mutated until the button comes up, so a
        # rubber band that shrinks back over its own path leaves no residue.
        self.draw_line_action = QAction("Draw Line", self)
        self.draw_line_action.setCheckable(True)
        self.draw_line_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.draw_line_action.setToolTip(
            "Drag to paint a straight line of the chosen terrain (Terrain mode). "
            "Hold Shift to snap the line to one of 16 directions"
        )
        self.draw_line_action.toggled.connect(lambda on: on and self._on_tool_selected("draw_line"))
        tool_group.addAction(self.draw_line_action)
        toolbar.addAction(self.draw_line_action)

        self.draw_rect_action = QAction("Draw Rectangle", self)
        self.draw_rect_action.setCheckable(True)
        self.draw_rect_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.draw_rect_action.setToolTip(
            "Drag to paint a rectangle of the chosen terrain (Terrain mode), filled "
            "or outlined. Hold Shift to square it"
        )
        self.draw_rect_action.toggled.connect(lambda on: on and self._on_tool_selected("draw_rect"))
        tool_group.addAction(self.draw_rect_action)
        toolbar.addAction(self.draw_rect_action)

        # Track B Stage 1 + 2 of the 2026-09-05 cliffs plan. Terrain-mode
        # like the four tools above; a click places the picked piece, a drag
        # lays an auto-connecting chain.
        self.cliff_action = QAction("Cliff", self)
        self.cliff_action.setCheckable(True)
        self.cliff_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.cliff_action.setToolTip(
            "Click to place the chosen cliff family/piece/frame (Terrain mode), "
            "anchored on the clicked tile by its real footprint span. "
            "Drag to lay a connected run: each piece's shape is chosen from "
            "its neighbours, keeping the size you picked"
        )
        self.cliff_action.toggled.connect(lambda on: on and self._on_tool_selected("cliff"))
        tool_group.addAction(self.cliff_action)
        toolbar.addAction(self.cliff_action)

        # Reads a tile's terrain + elevation into the toolbar params instead
        # of mutating -- same has_map/write_ok gate as Draw/Paint Can/Cliff
        # below (no squareness requirement: it never calls set_elevation()).
        self.eyedropper_action = QAction("Eyedropper", self)
        self.eyedropper_action.setCheckable(True)
        self.eyedropper_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.eyedropper_action.setToolTip(
            "Click a tile to load its terrain and elevation into the toolbar "
            "params (Terrain mode)"
        )
        self.eyedropper_action.toggled.connect(lambda on: on and self._on_tool_selected("eyedropper"))
        tool_group.addAction(self.eyedropper_action)
        toolbar.addAction(self.eyedropper_action)

        # Phase 2.8: drags a tile rectangle for Copy/Paste Region. Gated on
        # has_map/write_ok alone below, same as Eyedropper -- it reads the
        # map and never writes on its own (Paste is the thing that writes).
        self.select_action = QAction("Select", self)
        self.select_action.setCheckable(True)
        self.select_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.select_action.setToolTip(
            "Drag a rectangle to select a region for Copy/Paste Region (Terrain mode). "
            "Ctrl+A selects the whole map, Ctrl+Shift+A/Escape clears it"
        )
        self.select_action.toggled.connect(lambda on: on and self._on_tool_selected("select"))
        tool_group.addAction(self.select_action)
        toolbar.addAction(self.select_action)

        # Gated on has_map alone below, exactly like Pan and unlike every edit
        # tool: it reads the map and never writes, so no mode, write or
        # squareness gate applies to it.
        self.ruler_action = QAction("Ruler", self)
        self.ruler_action.setCheckable(True)
        self.ruler_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.ruler_action.setToolTip(
            "Measure the distance between two tiles (every mode) - drag, or click twice. "
            "Escape or right-click clears. Changes nothing."
        )
        self.ruler_action.toggled.connect(lambda on: on and self._on_tool_selected("ruler"))
        tool_group.addAction(self.ruler_action)
        toolbar.addAction(self.ruler_action)

        # Phase 3.5b's b1.4, gated on Units mode the same way Draw/Fill/
        # Elevate/Set Elevation gate on Terrain mode -- see
        # _update_tool_enabled's unit_editable.
        self.place_unit_action = QAction("Place Unit", self)
        self.place_unit_action.setCheckable(True)
        self.place_unit_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.place_unit_action.setToolTip(
            "Click to place the chosen object for the chosen owner (Units mode) -- "
            "snapped to the clicked tile's centre. With a wall picked, drag to place a run "
            "(one undo step per drag): it bends once, and Shift makes it one straight segment. "
            "Walls are always tile-snapped, so free placement doesn't apply to them."
        )
        self.place_unit_action.toggled.connect(lambda on: on and self._on_tool_selected("place_unit"))
        tool_group.addAction(self.place_unit_action)
        toolbar.addAction(self.place_unit_action)

        # The wall enclosure plan (2026-09-21), gated on Units mode like
        # Place Unit. Takes its wall const from the Units catalog, as Place
        # Unit's wall branch does; see _commit_wall_run().
        self.wall_rect_action = QAction("Wall Rectangle", self)
        self.wall_rect_action.setCheckable(True)
        self.wall_rect_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.wall_rect_action.setToolTip(
            "Drag to place an outline rectangle of walls for the chosen owner, using the wall "
            "picked in the Units catalog (Units mode) -- one undo step per drag. "
            "Shift makes it a square."
        )
        self.wall_rect_action.toggled.connect(lambda on: on and self._on_tool_selected("wall_rect"))
        tool_group.addAction(self.wall_rect_action)
        toolbar.addAction(self.wall_rect_action)

        # Phase 3.5b's b2.5, gated on Units mode the same way Place Unit is.
        self.convert_action = QAction("Convert", self)
        self.convert_action.setCheckable(True)
        self.convert_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.convert_action.setToolTip(
            "Drag over units owned by a checked source player to reassign them to the chosen "
            "owner (Units mode) -- one undo step per drag"
        )
        self.convert_action.toggled.connect(lambda on: on and self._on_tool_selected("convert"))
        tool_group.addAction(self.convert_action)
        toolbar.addAction(self.convert_action)

        # GH #59, Triggers mode only. The template is the entry tree's current
        # effect, checked per click in stamp_create_objects().
        self.create_objects_action = QAction("Create Objects", self)
        self.create_objects_action.setCheckable(True)
        self.create_objects_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.create_objects_action.setToolTip(
            "Select a Create Object effect, then click the map: adds one copy of it per tile "
            "under the brush (Triggers mode), skipping tiles that already have one -- "
            "one undo step per click"
        )
        self.create_objects_action.toggled.connect(
            lambda on: on and self._on_tool_selected("create_objects")
        )
        tool_group.addAction(self.create_objects_action)
        toolbar.addAction(self.create_objects_action)

        # Every tool action's only host has been this toolbar up to now.
        # Stage 3's overflow button moves some of these into a QMenu instead
        # (toolbar.removeAction()), and a QAction with no host at all can't
        # fire its shortcut -- so give every tool action a second, permanent
        # home on the window itself, independent of toolbar membership. Same
        # precedent as the mode_* actions in _build_keybind_actions().
        for _tool in settings.TOOLS:
            self.addAction(getattr(self, f"{_tool.tool_id}_action"))

        self._build_tool_overflow_button(toolbar)

        # The two fine Rotate actions again, as their own toolbar-only
        # QActions (not self.rotate_ccw_action/rotate_cw_action -- those stay
        # menu-only). A QAction's visibility isn't per-widget, so sharing one
        # action between the toolbar and Edit > Rotate Selection would force
        # the same show/hide behaviour on both; the toolbar buttons hide
        # outside Units mode (see _rotate_toolbar_actions in
        # _update_tool_enabled()) while the menu keeps its grey-not-hidden
        # treatment. Not added to tool_group: Rotate acts on the selection
        # rather than entering a click mode, so there is nothing for the
        # exclusive group to deselect. The coarse pair stays menu/keybind-only,
        # to keep this already-crowded row from growing two more buttons.
        self.rotate_ccw_toolbar_action = QAction("Rotate ↺", self)
        self.rotate_ccw_toolbar_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.rotate_ccw_toolbar_action.setToolTip(self.rotate_ccw_action.toolTip())
        self.rotate_ccw_toolbar_action.triggered.connect(lambda: self.on_unit_rotate(-1))
        self.rotate_cw_toolbar_action = QAction("Rotate ↻", self)
        self.rotate_cw_toolbar_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.rotate_cw_toolbar_action.setToolTip(self.rotate_cw_action.toolTip())
        self.rotate_cw_toolbar_action.triggered.connect(lambda: self.on_unit_rotate(1))
        self._rotate_toolbar_actions = [self.rotate_ccw_toolbar_action, self.rotate_cw_toolbar_action]
        toolbar.addAction(self.rotate_ccw_toolbar_action)
        toolbar.addAction(self.rotate_cw_toolbar_action)

        # Second toolbar row for tool params: row 1 fills with tool buttons
        # long before these did, and Qt's own overflow chevron ate the
        # params first since it overflows right-to-left. A break-separated
        # row removes that competition.
        self.addToolBarBreak()
        param_toolbar = self.addToolBar("Tool Options")
        param_toolbar.setMovable(False)

        # A permanent spacer, added once and never hidden: an empty
        # QToolBar collapses to zero height, and this row must hold its
        # height even when no param applies (Pan/Ruler) so switching tools
        # never shifts the layout. Sized off a real QComboBox's own hint
        # rather than a guessed pixel constant.
        param_row_spacer = QLabel("")
        param_row_spacer.setFixedHeight(QComboBox().sizeHint().height())
        param_toolbar.addWidget(param_row_spacer)

        # addWidget()/addSeparator() both hand back the QAction Qt actually
        # lays the item out with -- captured here (not discarded) because
        # hiding a toolbar-embedded widget has to go through that action's
        # setVisible(), not the widget's own hide()/setVisible(), or the
        # action's layout slot is left behind. Visibility (which of these
        # three shows at all) and enabled state are both re-gated by
        # _update_tool_enabled() per the active tool's param_widget.
        self.tool_param_separator_action = param_toolbar.addSeparator()
        # The terrain itself is chosen in the Terrain-mode sidebar page (GH #56), not on this row.
        # Wired here, once this row exists: auto beach below re-gates on a water-family pick.
        self.terrain_panel.terrain_changed.connect(self._on_terrain_changed)

        # Auto beach (2026-08-31 plan). A checkbox on Draw rather than a
        # separate Water tool -- the user's call, and it costs no ToolDef,
        # no QAction and no keybind audit. Session-only, never persisted,
        # matching the brush pair below.
        self.auto_beach_check = QCheckBox("Auto beach")
        self.auto_beach_check.setChecked(False)
        self.auto_beach_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.auto_beach_check.toggled.connect(self._on_auto_beach_toggled)
        self.auto_beach_param_action = param_toolbar.addWidget(self.auto_beach_check)

        self.beach_combo = QComboBox()
        # data None is "Auto": derive the beach per ring tile from its own
        # land terrain's climate -- see terrain_classes.auto_beach_for.
        self.beach_combo.addItem("Auto", None)
        for beach_id in terrain_classes.beach_terrains():
            self.beach_combo.addItem(name_for_terrain_id(beach_id), beach_id)
        self.beach_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.beach_combo.setToolTip(
            "Which beach terrain the shoreline uses. Auto picks it from the "
            "land tile's climate (ice land gets an ice beach)"
        )
        self.beach_param_action = param_toolbar.addWidget(self.beach_combo)

        self.beach_width_spin = QSpinBox()
        self.beach_width_spin.setRange(1, 3)
        self.beach_width_spin.setValue(1)
        self.beach_width_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        # No stutter warning here: measured with tools/bench_draw_stroke.py
        # --beach-width, brush 9 in Stepped costs 1.16x a plain Draw stroke
        # at width 1 and 1.46x at width 3 (1.32x/1.61x with sprites on) --
        # well inside the plan's 1.3x ship gate on the default, and nothing
        # like the 2.35x it projected against the older, slower baseline.
        self.beach_width_spin.setToolTip(
            "How many tiles wide the shoreline is"
        )
        self.beach_width_param_action = param_toolbar.addWidget(self.beach_width_spin)

        # descape/terrain_units.py's auto-placed trees/eye-candy, gated by
        # terrain_param_ok in _update_tool_enabled() since both Draw and Paint Can read them.
        # Persisted (settings.get/set_paint_trees/eye_candy), unlike brush
        # size/shape below: these change what gets written to the file.
        self.paint_trees_check = QCheckBox("Trees")
        self.paint_trees_check.setChecked(settings.get_paint_trees())
        self.paint_trees_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paint_trees_check.setToolTip(
            "Auto-place the matching GAIA tree on each tile a forest terrain is painted onto "
            "(varied graphic per tile), like the in-game editor's Eye Candy option. Painting a "
            "different terrain over a tile removes its tree."
        )
        self.paint_trees_check.toggled.connect(self._on_paint_trees_toggled)
        self.paint_trees_param_action = param_toolbar.addWidget(self.paint_trees_check)

        self.paint_eye_candy_check = QCheckBox("Eye candy")
        self.paint_eye_candy_check.setChecked(settings.get_paint_eye_candy())
        self.paint_eye_candy_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paint_eye_candy_check.setToolTip(
            "Also auto-place non-tree doodads a terrain carries (grass tufts, jungle underbrush). "
            "Painting a different terrain over a tile removes them too."
        )
        self.paint_eye_candy_check.toggled.connect(self._on_paint_eye_candy_toggled)
        self.paint_eye_candy_param_action = param_toolbar.addWidget(self.paint_eye_candy_check)

        self.level_param_label_action = param_toolbar.addWidget(QLabel(" Level: "))
        self.elevation_level_spin = QSpinBox()
        self.elevation_level_spin.setRange(0, ELEVATION_LEVEL_MAX)
        self.elevation_level_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.level_param_spin_action = param_toolbar.addWidget(self.elevation_level_spin)

        # Cliff's own param trio (Track B Stage 1): Family picks the shared
        # graphic, Piece picks the unit_const (its own real footprint span --
        # A4's fix, never a hand-typed suffix table), Frame picks the stored
        # `rotation`/`initial_animation_frame` that selects which of the
        # graphic's shapes gets drawn -- see cliff_catalog.py's own docstring
        # for why family and piece are independent axes. The preview label
        # is sized to the row's own fixed height (param_row_spacer above),
        # not constant_picker's larger dialog-preview box, so a wide Cliff
        # param group doesn't grow the one row every other tool also uses.
        self.cliff_family_label_action = param_toolbar.addWidget(QLabel(" Cliff family: "))
        self.cliff_family_combo = QComboBox()
        self.cliff_family_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.cliff_family_combo.currentIndexChanged.connect(self._on_cliff_family_changed)
        self.cliff_family_param_action = param_toolbar.addWidget(self.cliff_family_combo)

        self.cliff_piece_label_action = param_toolbar.addWidget(QLabel(" Piece: "))
        self.cliff_piece_combo = QComboBox()
        self.cliff_piece_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.cliff_piece_combo.currentIndexChanged.connect(self._on_cliff_piece_changed)
        self.cliff_piece_param_action = param_toolbar.addWidget(self.cliff_piece_combo)

        self.cliff_frame_label_action = param_toolbar.addWidget(QLabel(" Frame: "))
        self.cliff_frame_spin = QSpinBox()
        self.cliff_frame_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.cliff_frame_spin.valueChanged.connect(self._update_cliff_preview)
        self.cliff_frame_param_action = param_toolbar.addWidget(self.cliff_frame_spin)

        self.cliff_preview_label = QLabel()
        preview_side = param_row_spacer.height()
        self.cliff_preview_label.setFixedSize(preview_side, preview_side)
        self.cliff_preview_label.setAlignment(Qt.AlignCenter)
        self.cliff_preview_param_action = param_toolbar.addWidget(self.cliff_preview_label)

        self._populate_cliff_families()

        # Place Unit's own param pair (the object catalog and the owner
        # combo) moved into the Units-mode sidebar (UnitsPanel) -- both are
        # now always visible/live there rather than gated behind this
        # toolbar and the active tool. See units_panel.py.

        # Convert's own extra param (D3/b2.5) -- which owners' units a drag
        # reassigns. Destination reads units_panel.owner_id() (the same combo
        # Place Unit uses); brush size/shape below is generic to
        # every ToolDef.supports_brush tool and needs no Convert-specific
        # wiring at all.
        self.convert_sources_button = QToolButton()
        self.convert_sources_button.setText("Sources")
        self.convert_sources_button.setPopupMode(QToolButton.InstantPopup)
        self.convert_sources_button.setToolTip("Which owners' units this brush reassigns")
        self.convert_sources_button.setEnabled(False)  # re-gated by _update_tool_enabled()
        sources_menu = QMenu(self.convert_sources_button)
        self.convert_source_actions: dict[int, QAction] = {}
        gaia_source_action = QAction("GAIA", self, checkable=True, checked=True)
        sources_menu.addAction(gaia_source_action)
        self.convert_source_actions[GAIA_PLAYER_ID] = gaia_source_action
        for player_id in range(1, MAX_PLAYER_ID + 1):
            action = QAction(f"Player {player_id}", self, checkable=True, checked=True)
            sources_menu.addAction(action)
            self.convert_source_actions[player_id] = action
        self.convert_sources_button.setMenu(sources_menu)
        self.convert_sources_param_action = param_toolbar.addWidget(self.convert_sources_button)

        # Draw Rectangle's Fill/Outline toggle. Session-only like the brush
        # pair below and unlike the Trees/Eye candy checks above: it changes
        # which tiles one drag paints, not what gets written to a tile, so
        # there is nothing here worth carrying between launches.
        self.rect_fill_combo = QComboBox()
        self.rect_fill_combo.addItem("Filled", True)
        self.rect_fill_combo.addItem("Outline", False)
        self.rect_fill_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.rect_fill_combo.setToolTip(
            "Filled paints the whole rectangle; Outline paints its border only, "
            "thickened by the brush"
        )
        self.rect_fill_combo.currentIndexChanged.connect(self._on_rect_fill_changed)
        self.rect_fill_param_action = param_toolbar.addWidget(self.rect_fill_combo)

        # Brush size/shape -- shown for any ToolDef.supports_brush tool
        # (Terrain, Elevate, Set Elevation), same visibility/enabled
        # convention as the two params above. Deliberately session-only, not
        # read from or written to settings.py's config: every launch starts
        # at BRUSH_SIZE_MIN/square, so nothing here persists.
        self.brush_param_label_action = param_toolbar.addWidget(QLabel(" Brush: "))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(brush.BRUSH_SIZE_MIN, brush.BRUSH_SIZE_MAX)
        self.brush_size_spin.setValue(brush.BRUSH_SIZE_MIN)
        self.brush_size_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_size_spin.valueChanged.connect(self._on_brush_changed)
        self.brush_size_spin_action = param_toolbar.addWidget(self.brush_size_spin)

        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItem("Square", brush.BRUSH_SHAPE_SQUARE)
        self.brush_shape_combo.addItem("Circle", brush.BRUSH_SHAPE_CIRCLE)
        self.brush_shape_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_shape_combo.currentIndexChanged.connect(self._on_brush_changed)
        self.brush_shape_combo_action = param_toolbar.addWidget(self.brush_shape_combo)

        # Phase 2.8's paste filters -- which categories a Paste Region
        # actually writes. Visible only while Select is active (see
        # _update_tool_enabled()); all checked by default and session-only,
        # never persisted, same convention as the brush pair above.
        self.paste_terrain_check = QCheckBox("Terrain")
        self.paste_terrain_check.setChecked(True)
        self.paste_terrain_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_terrain_param_action = param_toolbar.addWidget(self.paste_terrain_check)

        self.paste_elevation_check = QCheckBox("Elevation")
        self.paste_elevation_check.setChecked(True)
        self.paste_elevation_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_elevation_param_action = param_toolbar.addWidget(self.paste_elevation_check)

        self.paste_units_check = QCheckBox("Units")
        self.paste_units_check.setChecked(True)
        self.paste_units_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_units_param_action = param_toolbar.addWidget(self.paste_units_check)

        # D2's free-placement toggle -- shown for any ToolDef.supports_free_place
        # tool (Place Unit). Session-only like the brush and paste-filter
        # controls above: tool options are deliberately not persisted, so
        # there is no settings getter/setter pair behind this one either.
        self.free_place_check = QCheckBox("Free placement")
        self.free_place_check.setToolTip(
            "Place a unit exactly under the cursor instead of at the tile's centre. "
            "Hold Alt for a one-off free placement without changing this."
        )
        self.free_place_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.free_place_param_action = param_toolbar.addWidget(self.free_place_check)

    def _build_status_bar(self) -> None:
        self.mode_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.mode_status_label)
        self._update_mode_status()
        # A dedicated slot rather than folding this into mode_status_label:
        # it has its own lifecycle (empty whenever there is no live ruler
        # measurement, however that measurement ends), not tied to mode/tool/
        # style changes the way that label is.
        self.ruler_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.ruler_status_label)
        # A separate label from mode_status_label, not text appended to it:
        # zoom changes once per wheel tick, mode/tool/style changes rarely,
        # and folding them together would rebuild the mode string on every
        # tick and couple two unrelated cadences.
        self.zoom_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.zoom_status_label)

    def _widen_left_column(self, min_width: int) -> None:
        """Grow the left pane to `min_width` on entering a mode whose panel
        needs it, if it is narrower.

        Only ever widens, and only up to that threshold, so a width the user
        dragged to themselves (and which closeEvent persisted) is left alone.
        The info panel's old 340 px is unusable for a trigger list, but picking
        a fixed Triggers width instead would throw away the user's own choice
        every time they switched modes.

        The threshold is a parameter in both places it is read -- the guard and
        the min() -- because each panel names its own MIN_USEFUL_WIDTH, and
        parameterising only one would silently widen to the other panel's.
        """
        sizes = self.content_splitter.sizes()
        if len(sizes) != 2 or sizes[0] >= min_width:
            return
        total = sizes[0] + sizes[1]
        left = min(min_width, total - settings.MIN_SPLIT_PANE)
        if left > sizes[0]:
            self.content_splitter.setSizes([left, total - left])

    def _update_mode_status(self) -> None:
        tool = _TOOL_LABELS.get(self._current_tool, self._current_tool)
        style = label_for(self._terrain_style)
        # The combo's own label, not self.mode.capitalize(): byte-identical
        # for the four single-word modes, and correct for "Map Options", whose
        # id is map_options and whose capitalize() would read "Map_options".
        self.mode_status_label.setText(
            f"  Mode: {self.mode_combo.currentText()}  |  Tool: {tool}  |  Style: {style}  "
        )

    def _update_zoom_status(self) -> None:
        pct = self.map_view.zoom_percent_of_fit()
        self.zoom_status_label.setText("  Zoom: --  " if pct is None else f"  Zoom: {pct:.0f}%  ")

    def _adjust_tool_value(self, delta: int) -> None:
        """Backs the "]"/"[" tool_value_inc/dec_action shortcuts -- see their
        own comment in _build_keybind_actions for the brush-vs-level
        priority rule. Reads each spinbox's live isEnabled() (kept correct
        by _update_tool_enabled, which gates the actions themselves the same
        way) rather than re-deriving BRUSH_TOOLS/_TOOL_PARAM membership
        here, so there is exactly one place that decides which widget is
        "the active tool's value" at any given moment."""
        if self.brush_size_spin.isEnabled():
            spin = self.brush_size_spin
        elif self.elevation_level_spin.isEnabled():
            spin = self.elevation_level_spin
        else:
            return
        spin.stepUp() if delta > 0 else spin.stepDown()

    def _build_keybind_actions(self) -> None:
        # Mode switching has no QAction of its own (it's a toolbar QComboBox)
        # -- these two exist purely to carry a shortcut. Not added to any
        # menu, just registered on the window (addAction) so Qt still
        # processes the shortcut globally while the window has focus.
        self.mode_view_action = QAction("Switch to View Mode", self)
        self.mode_view_action.triggered.connect(lambda: self.mode_combo.setCurrentText("View"))
        self.addAction(self.mode_view_action)

        self.mode_terrain_action = QAction("Switch to Terrain Mode", self)
        self.mode_terrain_action.triggered.connect(lambda: self.mode_combo.setCurrentText("Terrain"))
        self.addAction(self.mode_terrain_action)

        self.mode_units_action = QAction("Switch to Units Mode", self)
        self.mode_units_action.triggered.connect(lambda: self.mode_combo.setCurrentText("Units"))
        self.addAction(self.mode_units_action)

        self.mode_triggers_action = QAction("Switch to Triggers Mode", self)
        self.mode_triggers_action.triggered.connect(
            lambda: self.mode_combo.setCurrentText("Triggers")
        )
        self.addAction(self.mode_triggers_action)

        self.mode_map_options_action = QAction("Switch to Map Options Mode", self)
        self.mode_map_options_action.triggered.connect(
            lambda: self.mode_combo.setCurrentText("Map Options")
        )
        self.addAction(self.mode_map_options_action)

        self.mode_players_action = QAction("Switch to Players Mode", self)
        self.mode_players_action.triggered.connect(lambda: self.mode_combo.setCurrentText("Players"))
        self.addAction(self.mode_players_action)

        self.mode_diplomacy_action = QAction("Switch to Diplomacy Mode", self)
        self.mode_diplomacy_action.triggered.connect(
            lambda: self.mode_combo.setCurrentText("Diplomacy")
        )
        self.addAction(self.mode_diplomacy_action)

        self.mode_messages_action = QAction("Switch to Messages Mode", self)
        self.mode_messages_action.triggered.connect(lambda: self.mode_combo.setCurrentText("Messages"))
        self.addAction(self.mode_messages_action)

        # Per-mode player selection (0-8) -- see _select_player() for the
        # dispatch. No menu entry, same reasoning as the mode_* actions above.
        self.player_select_actions: dict[int, QAction] = {}
        for pid in range(MAX_PLAYER_ID + 1):
            label = "GAIA" if pid == GAIA_PLAYER_ID else f"Player {pid}"
            action = QAction(f"Select {label}", self)
            action.triggered.connect(lambda _=False, p=pid: self._select_player(p))
            self.addAction(action)
            self.player_select_actions[pid] = action

        # The active tool's own "primary value" -- brush size for any
        # supports_brush tool (Draw, Elevate, Set Elevation), falling back
        # to Set Elevation's Level spinbox when the active tool has no brush.
        # Brush wins the overlap on Set Elevation, which has both: this is
        # what "]"/"[" -- picked as brush-size-style inc/dec keys, see
        # REBINDABLE_ACTIONS's adjust_increment/adjust_decrement comment in
        # settings.py -- were always meant to drive. Gated the same way the
        # spinboxes themselves are (see _update_tool_enabled) so the
        # shortcut is dead whenever neither would apply. stepUp/stepDown
        # clamp to the target spinbox's own range, same as clicking its
        # arrows would.
        self.tool_value_inc_action = QAction("Increase Tool Value", self)
        self.tool_value_inc_action.triggered.connect(lambda: self._adjust_tool_value(+1))
        self.addAction(self.tool_value_inc_action)

        self.tool_value_dec_action = QAction("Decrease Tool Value", self)
        self.tool_value_dec_action.triggered.connect(lambda: self._adjust_tool_value(-1))
        self.addAction(self.tool_value_dec_action)

        # The four pan actions are deliberately INERT: never addAction'd,
        # never given a shortcut, nothing connected to triggered. They exist
        # only so each view_pan_* id has a Keybinds-tab row and so
        # test_keybinds.py's REBINDABLE_ACTIONS <-> _keybind_actions set
        # equality holds. The key itself is routed to MapView by
        # apply_keybind, which a QAction shortcut could not do -- it consumes
        # the press and never reports the release a hold timer needs.
        for action_id in MapView.PAN_DIRECTIONS:
            setattr(self, f"{action_id}_action", QAction(settings.get_action_label(action_id), self))

        # Every tool's toolbar action reused directly -- no separate action
        # needed, just give it a shortcut too. Looked up via getattr rather
        # than hand-listed so this stays in sync with settings.TOOLS
        # automatically: a tool registered there without a matching
        # self.<tool_id>_action (built in _build_toolbar, before this runs)
        # is a wiring bug, and this is deliberately not defensive about it
        # -- getattr with no default raises immediately, which is the point
        # (see settings.ToolDef's own docstring).
        self._keybind_actions = {
            "file_new": self.new_custom_action,
            "file_new_default": self.new_action,
            "file_open": self.open_action,
            "file_close": self.close_action,
            "file_save": self.save_action,
            "file_save_as": self.save_as_action,
            "file_recover_autosave": self.recover_autosave_action,
            "file_exit": self.exit_action,
            "edit_undo": self.undo_action,
            "edit_redo": self.redo_action,
            "edit_copy": self.copy_action,
            "edit_paste": self.paste_action,
            "edit_select_all": self.select_all_action,
            "edit_deselect": self.deselect_action,
            "edit_clipboard_history": self.clipboard_history_action,
            "edit_history": self.history_action,
            "edit_scatter_units": self.scatter_action,
            "edit_disables": self.disables_action,
            "edit_settings": self.settings_action,
            "map_mirror": self.mirror_action,
            "analysis_run": self.analysis_action,
            "view_isometric": self.iso_action,
            "view_distance_ticks": self.distance_ticks_action,
            "view_show_sprites": self.show_sprites_action,
            "view_stack_badges": self.stack_badges_action,
            "view_grid_overlay": self.grid_action,
            "view_grid_follow": self.grid_follow_action,
            "view_footprint_outlines": self.footprint_action,
            "view_selection_owner_colour": self.selection_owner_colour_action,
            "view_range_rings": self.range_rings_action,
            "view_player_cameras": self.player_cameras_action,
            "view_trigger_overlay": self.trigger_overlay_action,
            **{f"view_layer_{lid}": action for lid, action in self.layer_actions.items()},
            "view_pan_up": self.view_pan_up_action,
            "view_pan_down": self.view_pan_down_action,
            "view_pan_left": self.view_pan_left_action,
            "view_pan_right": self.view_pan_right_action,
            "help_about": self.about_action,
            "help_debug_log": self.debug_log_action,
            "help_perf_trace": self.perf_trace_action,
            "mode_view": self.mode_view_action,
            "mode_terrain": self.mode_terrain_action,
            "mode_units": self.mode_units_action,
            "mode_triggers": self.mode_triggers_action,
            "mode_map_options": self.mode_map_options_action,
            "mode_players": self.mode_players_action,
            "mode_diplomacy": self.mode_diplomacy_action,
            "mode_messages": self.mode_messages_action,
            "filter_show_gaia": self.show_gaia_action,
            "filter_show_trees": self.show_trees_action,
            "filter_show_walls": self.show_walls_action,
            "filter_show_eye_candy": self.show_eye_candy_action,
            "filter_show_invisible": self.show_invisible_action,
            "filter_show_garrisoned": self.show_garrisoned_action,
            "filter_all_players": self.filter_all_players_action,
            "filter_no_players": self.filter_no_players_action,
            "filter_show_all": self.filter_show_all_action,
            "filter_hide_all": self.filter_hide_all_action,
            "adjust_increment": self.tool_value_inc_action,
            "adjust_decrement": self.tool_value_dec_action,
            "unit_rotate_ccw": self.rotate_ccw_action,
            "unit_rotate_cw": self.rotate_cw_action,
            "unit_rotate_ccw_coarse": self.rotate_ccw_coarse_action,
            "unit_rotate_cw_coarse": self.rotate_cw_coarse_action,
            "unit_variant_prev": self.variant_prev_action,
            "unit_variant_next": self.variant_next_action,
            "unit_variant_random": self.variant_random_action,
            "unit_select_stack": self.select_stack_action,
        }
        for tool in settings.TOOLS:
            self._keybind_actions[f"tool_{tool.tool_id}"] = getattr(self, f"{tool.tool_id}_action")
        for pid, action in self.player_select_actions.items():
            self._keybind_actions[f"player_select_{pid}"] = action
        for action_id in self._keybind_actions:
            self.apply_keybind(action_id)

    def apply_keybind(self, action_id: str) -> None:
        # Pan needs press AND release to drive its hold timer; a QAction
        # shortcut swallows the press and never reports the release, so these
        # four route to MapView's own key handlers instead of setShortcut().
        # A knowing deviation from "exactly one generic keybind path" --
        # confined to this one branch.
        if action_id in MapView.PAN_DIRECTIONS:
            self.map_view.set_pan_binding(action_id, settings.get_keybind(action_id))
            return
        action = self._keybind_actions.get(action_id)
        if action is None:
            return
        key_text = settings.get_keybind(action_id)
        action.setShortcut(QKeySequence(key_text) if key_text else QKeySequence())

    def _select_player(self, player_id: int) -> None:
        """Backs the player_select_0..8 shortcuts. There is no global
        "current player" (Filters' per-player checkboxes are visibility
        toggles, not one -- see units_panel.py's own Owner combo), so this
        dispatches to whichever mode-local selector is active and never
        syncs the other two."""
        if self.mode == "units":
            self.units_panel.select_owner(player_id)
            owner_text = "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}"
            self._log_status(f"Place/Convert owner: {owner_text}")
        elif self.mode == "players":
            if self.players_panel.select_player(player_id):
                self._log_status(f"Players: switched to Player {player_id}")
            elif player_id == GAIA_PLAYER_ID:
                self._log_status("GAIA has no editable fields in Players mode")
            else:
                self._log_status("No scenario loaded")
        elif self.mode == "diplomacy":
            if self.diplomacy_panel.select_player(player_id):
                self._log_status(f"Diplomacy: switched to Player {player_id}")
            else:
                self._log_status(f"Player {player_id} is not defined in this scenario")
        elif _LEFT_PAGE_FOR_MODE.get(self.mode, _LEFT_PAGE_INFO) == _LEFT_PAGE_INFO:
            # View shows page 0, where the player stats combo lives (Terrain has its own page, GH #56).
            index = self.stats_player_combo.findData(player_id)
            if index >= 0:
                self.stats_player_combo.setCurrentIndex(index)
                self._log_status(f"Player stats: {self.stats_player_combo.itemText(index)}")
            elif self.scenario is None:
                self._log_status("No scenario loaded")
            else:
                self._log_status(f"Player {player_id} is not defined in this scenario")
        else:
            self._log_status(f"Player selection has no target in {self.mode_combo.currentText()} mode")

    def _on_distance_ticks_toggled(self, checked: bool) -> None:
        settings.set_distance_ticks(checked)
        self.map_view.set_edge_ticks(checked)

    def _on_grid_overlay_toggled(self, checked: bool) -> None:
        settings.set_grid_overlay(checked)
        self._apply_grid_change(lambda: self.map_view.set_grid_overlay(checked))

    def _on_grid_follow_toggled(self, checked: bool) -> None:
        settings.set_grid_follow_elevation(checked)
        self._apply_grid_change(lambda: self.map_view.set_grid_follow_elevation(checked))

    def _apply_grid_change(self, change: Callable[[], bool]) -> None:
        """Every View > Grid change that can re-bake the grid into the chunk
        cache: the toggle, Follow Terrain Elevation, and both ends of a
        Settings > Appearance slider drag. _on_layer_toggled's shape:
        _cancel_warms() first, since a warm in flight would install work done
        under the old spec, and _start_level_warm() after. No wait cursor:
        set_grid() only evicts, and the recomposite happens in paint()."""
        self._cancel_warms()
        change()
        self._start_level_warm()

    def _on_footprint_outlines_toggled(self, checked: bool) -> None:
        settings.set_footprint_outlines(checked)
        if checked and self._needs_unit_index():
            self._rebuild_unit_index()
        self.map_view.set_footprint_outlines(checked)

    def _on_footprint_scope(self, scope: str) -> None:
        settings.set_footprint_scope(scope)
        self.map_view.set_footprint_scope(scope)

    def _on_stack_badges_toggled(self, checked: bool) -> None:
        settings.set_stack_badges(checked)
        self.map_view.set_stack_badges(checked)

    def _on_selection_owner_colour_toggled(self, checked: bool) -> None:
        settings.set_selection_by_owner(checked)
        self.map_view.set_selection_by_owner(checked)

    def _on_range_rings_toggled(self, checked: bool) -> None:
        settings.set_range_rings(checked)
        self.map_view.set_range_rings(checked)

    def _on_player_cameras_toggled(self, checked: bool) -> None:
        settings.set_camera_markers(checked)
        self.map_view.set_camera_markers_enabled(checked)
        # Pushes the list too, not just the flag: a toggle can be the first
        # thing that happens after a load, and MapView is never told about a
        # scenario it was not handed markers for.
        self._refresh_camera_markers()

    def _on_trigger_overlay_toggled(self, checked: bool) -> None:
        settings.set_trigger_overlay(checked)
        self.map_view.set_trigger_overlay_enabled(checked)

    def _on_trigger_selection_changed(self) -> None:
        """The panel's selection-set callback: Copy gating, and the overlay,
        since a multi-selection or a deselect never reaches the entry form."""
        self._update_tool_enabled()
        self._on_trigger_focus_changed()

    def _on_trigger_focus_changed(self) -> None:
        """The panel's current trigger or entry moved: re-read it and redraw.
        Several triggers selected draws nothing, as the panel shows no form."""
        panel = self.trigger_panel
        index = panel.current_trigger_index()
        if index is None or len(panel.selected_trigger_indices()) > 1:
            self._trigger_overlay_ref = None
        else:
            self._trigger_overlay_ref = (index, panel.current_entry_ref())
        # A compare, not a plain disarm: a list pick's own write re-enters here.
        target = self._unit_picker
        if target is not None and self._trigger_overlay_ref != (target[0], (target[1], target[2])):
            self.disarm_unit_picker()
        self._refresh_trigger_overlay()

    # -- trigger unit references ---------------------------------------------

    def _unit_reference_index(self) -> unit_references.ReferenceIndex:
        if self.scenario is None:
            return unit_references.EMPTY_INDEX
        if self._unit_ref_index is None:
            self._unit_ref_index = unit_references.build_reference_index(self.scenario)
        return self._unit_ref_index

    def _describe_unit_reference(self, ref_id) -> str:
        return unit_references.describe(self._unit_reference_index(), ref_id)

    def _on_unit_references_moved(self) -> None:
        """A unit mutation can move, remove or reassign what a trigger names."""
        self._unit_ref_index = None
        self.trigger_panel.refresh_unit_reference_labels()
        # The held ref, not a panel read: undo is global, and the panel may sit unrefreshed.
        self._refresh_trigger_overlay()

    def _on_pick_unit_requested(self, target) -> None:
        """The panel's Pick button: a target arms, None disarms."""
        if target is None:
            self.disarm_unit_picker()
        else:
            self.arm_unit_picker(target)

    def arm_unit_picker(self, target: tuple) -> None:
        """Pick from map for one Unit/Unit[] field. Leaves self._selection
        alone: the field's current units are drawn through MapView's selection
        layer only, and set back to nothing on disarm."""
        if self.scenario is None:
            return
        self.pan_action.setChecked(True)
        self._unit_picker = target
        self.map_view.set_unit_picker(True)
        if self.map_view._unit_index is None:
            self._rebuild_unit_index()
        self.trigger_panel.set_pick_armed(target)
        self._refresh_picker_highlight()
        self.map_view.setFocus()
        verb = "add or remove units in" if target[4] else "set"
        hint = f"Click a unit to {verb} {target[3].replace('_', ' ')}; Esc to finish"
        if self._unit_filter != UnitFilter():
            hint += " (units hidden by Filters can't be picked)"
        self._log_status(hint)

    def disarm_unit_picker(self) -> None:
        if self._unit_picker is None:
            return
        self._unit_picker = None
        self.map_view.set_unit_picker(False)
        if self.mode != "units":
            self.map_view.set_unit_selection([])
        self.trigger_panel.set_pick_armed(None)
        if not self._needs_unit_index() and self.map_view._unit_index is not None:
            self._stack_cycle = None
            self.map_view.set_unit_index(None)

    def _refresh_picker_highlight(self) -> None:
        """The armed field's units, through MapView's selection layer."""
        target, index = self._unit_picker, self.map_view._unit_index
        if target is None or index is None:
            return
        refs = self._unit_reference_index()
        entries = []
        for ref_id in self.trigger_panel.unit_reference_ids(target):
            ref = refs.get(ref_id)
            entry = None if ref is None else index.entry_for_key((ref.player_id, ref.reference_id))
            if entry is not None:
                entries.append(entry)
        self.map_view.set_unit_selection(entries)

    def _on_unit_picked(self, key: tuple[int, int]) -> None:
        """MapView's pick: written through the panel's funnel as one undo record."""
        target = self._unit_picker
        if target is None:
            return
        ref_id = key[1]
        stays_armed = self.trigger_panel.apply_picked_reference(target, ref_id)
        if self._unit_picker is None:
            return
        if stays_armed:
            self._refresh_picker_highlight()
            self._log_status(f"Picked {self._describe_unit_reference(ref_id)}")
        else:
            self.disarm_unit_picker()

    def _rederive_trigger_overlay(self) -> None:
        """After a trigger edit or undo: the snapshot is plain ints, so it is
        stale the moment a coordinate or an entry index changes. Outside
        Triggers mode the panel is not refreshed, so the held ref is used."""
        if self.mode == "triggers":
            self._on_trigger_focus_changed()
        else:
            self._refresh_trigger_overlay()

    def _refresh_trigger_overlay(self) -> None:
        """Re-derives the overlay from a fresh parse_triggers(), never a held
        manager (the library's field gating is process-global). Read-only, so
        it never goes through TriggerEditModel."""
        ref = self._trigger_overlay_ref
        loaded = self.scenario
        # trigger_read_supported first: a ref only exists once the panel parsed.
        if ref is None or loaded is None or loaded.trigger_read_supported is not True:
            self.map_view.clear_trigger_overlay()
            return
        manager = parse_triggers(loaded)
        index, entry_ref = ref
        if (
            manager is None
            or not library_compat.vocabulary_is_available(loaded.scenario_version)
            or not 0 <= index < len(manager.triggers)
        ):
            self._trigger_overlay_ref = None
            self.map_view.clear_trigger_overlay()
            return
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        shapes = trigger_geometry.shapes_for_trigger(
            manager.triggers[index], vocabulary, trigger_index=index, references=self._unit_reference_index()
        )
        self.map_view.show_trigger_overlay(shapes, entry_ref)

    def _on_distance_tick_interval(self, tiles: int) -> None:
        settings.set_distance_tick_interval(tiles)
        self.map_view.set_edge_tick_interval(tiles)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"DEscape {__version__} by Combinebobnt\n\n"
            "External map editor for Age of Empires 2: Definitive Edition scenarios.\n\n"
            "Copyright (C) 2026 Combinebobnt\n"
            "Licensed under the GNU General Public License v3.0 or later. ",
        )

    def on_mode_changed(self, mode_text: str) -> None:
        self.disarm_unit_picker()
        self.mode = _mode_id(mode_text)
        # Entering a trigger-parsing mode is what makes Files, and so the
        # embedded-XS info line, readable: refresh it once when that changes.
        triggers_known = None if self.scenario is None else self.scenario.trigger_read_supported
        if self.mode != "terrain":
            # Triggers and Units both gate every edit tool off exactly as View
            # does, so they need the same fall back to Pan rather than leaving
            # a now-disabled tool checked.
            self.pan_action.setChecked(True)
        self.map_view.set_mode(self.mode)
        # set_mode() already dropped MapView's visual selection when leaving
        # Units. Drop ViewerWindow's key too, or the two diverge: re-entering
        # Units would show an empty inspector while _selection still held a
        # stale key, and _refresh_selection_after_filter could then resolve an
        # entry for a unit with no highlight and no inspector row. Phase 3.5
        # inherits _selection as the selection API, so it must not lie.
        self._selection = []
        self.left_stack.setCurrentIndex(_LEFT_PAGE_FOR_MODE.get(self.mode, _LEFT_PAGE_INFO))
        # The markers themselves are mode-free; their emphasis is not, so
        # entering or leaving Players mode has to re-push them.
        self._refresh_camera_markers()
        if self.mode == "units":
            self._widen_left_column(UnitsPanel.MIN_USEFUL_WIDTH)
            self._rebuild_unit_index()
            self.units_panel.clear()
        if self.mode == "triggers":
            self._widen_left_column(TriggerPanel.MIN_USEFUL_WIDTH)
            # Parsed on demand, not at load: the largest corpus file's Triggers
            # section is 1.17 MB and opening a map for terrain work must not
            # pay for it.
            self._show_triggers()
        if self.mode == "map_options":
            self._widen_left_column(MapOptionsPanel.MIN_USEFUL_WIDTH)
            self._show_map_options()
        if self.mode == "players":
            self._widen_left_column(PlayersPanel.MIN_USEFUL_WIDTH)
            self._show_players()
        if self.mode == "diplomacy":
            self._widen_left_column(DiplomacyPanel.MIN_USEFUL_WIDTH)
            self._show_diplomacy()
        if self.mode == "messages":
            self._widen_left_column(MessagesPanel.MIN_USEFUL_WIDTH)
            self._repopulate_messages()
        if self.mode == "terrain":
            self._widen_left_column(TerrainPanel.MIN_USEFUL_WIDTH)
        if self.scenario is not None and self.scenario.trigger_read_supported != triggers_known:
            self._update_info()
        self._update_tool_enabled()
        self._update_mode_status()
        # The combo's own text, not mode_text: _update_tool_enabled() above
        # can have forced the mode back (selecting Units while already in
        # Sloped), and reporting the mode the user asked for rather than the
        # one they got would contradict the combo they're looking at. Reading
        # the label rather than capitalizing self.mode also keeps "Map Options"
        # spelled the way the combo spells it.
        self._log_status(f"Mode changed to {self.mode_combo.currentText()}")

    def _show_triggers(self) -> None:
        """Populate the Triggers panel. The one repopulate path, directly
        parallel to _show_map_options() and its stated "the one repopulate
        path" rationale: every caller that changes what the panel should show
        resolves the pending exec-order value here once, rather than
        repeating the resolution at each call site.

        set_exec_order() normalises self.trigger_edits.exec_order back to
        None when set to the file's own byte, so a flip-and-flip-back
        correctly stops showing "unsaved change" here too.
        """
        pending = (
            self.trigger_edits.exec_order
            if self.trigger_edits is not None and self.trigger_edits.exec_order_supported
            else None
        )
        # The repopulate rebuilds the form, and the Pick button with it.
        self.disarm_unit_picker()
        self.trigger_panel.show_scenario(self.scenario, pending)
        self._sync_trigger_clipboard_state()
        # A repopulate can leave nothing current, which emits no entry change.
        self._on_trigger_focus_changed()

    def _show_map_options(self) -> None:
        """Populate the Map Options panel, parsing the Triggers section first.

        The parse is what makes the trigger execution-order row appear at all:
        option_fields.specs_for() reads a field's presence off the loaded
        sections, and "Triggers" is not among them until parse_triggers() has
        run. Same on-demand cost Triggers mode already pays, and not paid at
        load for the same reason.

        A file whose Triggers section cannot be parsed (the 1.54/trigger-3.9
        set) simply has no exec-order row, which is indistinguishable in the
        form from a scenario version that never stored the flag. Logged rather
        than left silent, since those are different situations to a user.

        Also the one repopulate path, so every caller that changes what the
        panel should show (mode entry, an undo, a redo) goes through the same
        pending-value and gate resolution rather than repeating it.
        """
        if self.scenario is None:
            self.map_options_panel.clear_document()
            return
        triggers_ok = parse_triggers(self.scenario) is not None
        self.map_options_panel.show_scenario(
            self.scenario,
            values=self._pending_option_values(),
            editable_fields=self._editable_option_fields(),
            read_only_reasons=self._map_options_read_only_reasons(),
            notes=self._map_options_notes(),
        )
        if not triggers_ok:
            self._log_status(
                "Map Options: this file's Triggers section can't be read, so the "
                "trigger execution-order setting is not listed."
            )

    def _pending_option_values(self) -> dict[str, int]:
        """Raw values that differ from what the file's retrievers hold.

        Neither write path mutates a retriever -- the scalars are byte patches
        and exec-order is one byte of a spliced tail -- so a repopulate that
        did not pass these would reset every row to the file's value while the
        models still held the edits. That is not only a wrong display: the
        panel's own equality guard compares against what it last showed, so
        re-setting a field to its pending value would then report an edit the
        model already has, and record a second undo step for it.
        """
        values: dict[str, int] = {}
        if self.option_edits is not None:
            values.update(self.option_edits.pending_values())
        if self.trigger_edits is not None and self.trigger_edits.exec_order_supported:
            values[_EXEC_ORDER_FIELD] = self.trigger_edits.exec_order
        return values

    def _editable_option_fields(self) -> frozenset[str]:
        """Which Map Options rows this file will accept edits for.

        Two gates, resolved separately, because neither says anything about the
        other's rows: options_model walks GlobalVictory/Diplomacy/Map/Options
        and explicitly skips the Triggers section, while trigger_write_supported
        covers only the Triggers section. A file with an untrustworthy Map
        anchor can still edit exec-order, and a file that failed the trigger
        alignment gate can still edit every scalar.
        """
        specs = option_fields.specs_for(self.scenario)
        editable = set()
        if options_write_supported(self.scenario, specs):
            editable.update(
                spec.field_id for spec in specs if spec.field_id != _EXEC_ORDER_FIELD
            )
        if exec_order_write_supported(self.scenario):
            editable.add(_EXEC_ORDER_FIELD)
        return frozenset(editable)

    _EXEC_ORDER_READ_ONLY = (
        "Read-only for this file: its Triggers section failed the alignment gate a "
        "save would splice through, so this flag cannot be written back."
    )
    _SCALARS_READ_ONLY = (
        "Read-only for this file: its map-option block failed verification, so "
        "patching this setting would land at an offset that cannot be trusted."
    )

    def _map_options_read_only_reasons(self) -> dict[str, str]:
        """Per-row tooltip text for every listed row that is not editable.

        A greyed row with no tooltip is the failure this exists to prevent, and
        the exec-order row is the live case: it carries no spec tooltip of its
        own, so before this a user hovering it learned nothing. Two texts, one
        per gate, for the same reason the gates are separate -- a row greyed
        because the Map anchor failed and a row greyed because the Triggers
        section failed are different situations to whoever has to act on it.

        Rows shown as-stored are deliberately absent here: their own label
        already carries a tooltip accounting for the raw number, which is the
        more specific explanation and must not be displaced.
        """
        editable = self._editable_option_fields()
        reasons = {}
        for spec in option_fields.specs_for(self.scenario):
            if spec.field_id in editable:
                continue
            reasons[spec.field_id] = (
                self._EXEC_ORDER_READ_ONLY
                if spec.section == _EXEC_ORDER_SECTION
                else self._SCALARS_READ_ONLY
            )
        return reasons

    def _map_options_notes(self) -> list[str]:
        """The status-line summary alongside those tooltips. Only the
        exec-order gate needs one: a greyed scalar is already covered by the
        panel's own read-only count, while a single greyed row among editable
        ones would otherwise go unremarked in the summary."""
        specs = option_fields.specs_for(self.scenario)
        listed = any(spec.field_id == _EXEC_ORDER_FIELD for spec in specs)
        if listed and not exec_order_write_supported(self.scenario):
            return [
                (
                    "Trigger execution order is read-only for this file: its Triggers "
                    "section failed the alignment gate that a save would splice through."
                )
            ]
        return []

    def set_option_field(self, spec, value: int) -> None:
        """MapOptionsPanel's one callback: the user set `spec` to raw `value`.

        Fans out to the two write paths the panel deliberately does not know
        apart: TriggerEditModel for exec-order, which lives inside the Triggers
        region and is emitted by that model's serialize() tail, and
        OptionsEditModel for every byte-patched scalar. Wiring exec-order to
        the options model instead would leave self.trigger_edits None on a save
        whose only change was that flag, and write_scenario() would emit the
        original bytes with no error at all.
        """
        if spec.section == _EXEC_ORDER_SECTION:
            self._set_exec_order(spec, value)
        else:
            self._set_option_scalar(spec, value)

    def _set_exec_order(self, spec, value: int) -> None:
        """The exec-order row, through the trigger model's own edit pair so it
        gets a TriggerDiffRecord -- which is also what makes the flag and
        trigger_display_order restore together on an undo."""
        model = self._ensure_trigger_edits()
        if model is None:
            self._repopulate_map_options()
            return
        if not model.exec_order_supported:
            self._log_status(
                "Trigger execution order cannot be written back for this file."
            )
            self._repopulate_map_options()
            return
        with self._trigger_edit(model, f"Set {spec.label}") as m:
            m.set_exec_order(value)

    @contextmanager
    def _option_edit(self, model: OptionsEditModel, field_id: str, label: str):
        """Shared undo/redo wiring for one OptionsEditModel field edit:
        reads `before` from the model, lets the caller mutate it, and on a
        successful exit pushes one OptionsDiffRecord and updates title/edit
        actions/status. Extracted out of _set_option_scalar() once a second
        call site needed it (Players mode, step 3c) -- that method said
        this was coming the moment it did.

        No begin/commit pair is needed, unlike the trigger side: the before
        value is readable from the model at any time, so there is no timing
        window to protect. If the caller's mutation raises, the code after
        `yield` never runs and no record is pushed -- the same "nothing
        recorded on a refused edit" behaviour _set_option_scalar() always
        had.
        """
        before = model.current_value(field_id)
        yield model
        after = model.current_value(field_id)
        self.edit_history.push_options_record(OptionsDiffRecord(label, field_id, before, after))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)

    def _set_option_scalar(self, spec, value: int) -> None:
        """Every other Map Options row: one in-place byte patch, one
        OptionsDiffRecord, via the shared _option_edit() contextmanager."""
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_map_options()
            return
        try:
            with self._option_edit(model, spec.field_id, f"Set {spec.label}"):
                model.set_value(spec.field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Map option {spec.field_id} not set: {e}")
            self._repopulate_map_options()

    def _ensure_option_edits(self) -> OptionsEditModel | None:
        """The document's OptionsEditModel, built on first use.

        Deferred to the first real edit exactly as _ensure_trigger_edits() is,
        and for the same reason: a browse-only session must save
        byte-identically. Returns None for a file the model refuses, which the
        panel's gate should already have kept unreachable -- reported rather
        than swallowed, since silently ignoring an edit the user made is the
        worse failure.
        """
        if self.option_edits is not None:
            return self.option_edits
        if self.scenario is None:
            return None
        try:
            self.option_edits = OptionsEditModel(self.scenario)
        except OptionEditsUnavailableError as e:
            self._log_status(f"Map options are read-only for this file: {e}")
            QMessageBox.warning(self, "Map options are read-only", str(e))
            return None
        return self.option_edits

    def _ensure_unit_edits(self) -> UnitEditModel | None:
        """The document's UnitEditModel, built on first use.

        Deferred to the first real unit edit exactly as _ensure_option_edits()
        is, and for the same reason: a browse-only session must save
        byte-identically, and the construction gate below scans every unit in
        the file (up to 10,871 on the largest corpus file).

        That one-time scan is a multi-hundred-ms freeze on a large file, so
        it gets a wait cursor. Only the first call reaches it; the cached
        returns above are free and push no cursor. No processEvents() to go
        with it, unlike every other wait-cursor site here: this runs from
        mid-stroke (_apply_terrain_unit_plan) and from inside other busy
        wraps (paste, fill), where a real event-loop turn would be a
        reentrancy hole, and the setEnabled(False)/_busy pair those sites
        use to close it cannot nest.
        """
        if self.unit_edits is not None:
            return self.unit_edits
        if self.scenario is None:
            return None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Restored before the warning box below, not after it: a modal
            # dialog under a wait cursor reads as a still-frozen window.
            try:
                self.unit_edits = UnitEditModel(self.scenario)
            finally:
                QApplication.restoreOverrideCursor()
        except UnitEditsUnavailableError as e:
            self._log_status(f"Units are read-only for this file: {e}")
            QMessageBox.warning(self, "Units are read-only", str(e))
            return None
        return self.unit_edits

    def _repopulate_map_options(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself (an
        undo). The panel's live values are what its equality guard compares
        against, so leaving a refused edit on screen would make the next
        genuine change to that field look like a no-op."""
        if self.mode == "map_options":
            self._show_map_options()

    def _show_players(self) -> None:
        """Populate the Players panel -- the single repopulate path for
        Players mode (mode entry, a new document, an edit, an undo/redo, or
        a refused edit all call only this, via _repopulate_players() for
        the latter three -- see PlayersPanel.show_scenario()'s own
        docstring for why one method covers all of them without resetting
        the player selection on every edit).

        Unlike _show_map_options(), no section needs parsing on demand
        first: every player_fields.py spec lives in DataHeader/Options/
        PlayerDataTwo/Map/Units, all of which load_map_and_units() already
        parses, so there is nothing here analogous to the Triggers
        on-demand parse.
        """
        if self.scenario is None:
            self.players_panel.clear_document()
            return
        self.players_panel.show_scenario(
            self.scenario,
            editable_fields=self._editable_player_fields(),
            read_only_reasons=self._players_read_only_reasons(),
            pending_values=self._pending_player_values(),
            player_count=self._current_player_count(),
            disables_editable=self._disables_editable(),
            disables_carrier_ok=self._options_carrier_ok(),
        )

    def _editable_player_fields(self) -> frozenset[str]:
        """Which Players mode field ids (bare, not "player:<field>:<player>")
        this file will accept edits for -- the same set for every player,
        since players_write_supported() is a document-wide gate, not a
        per-player one.

        Two gates, resolved separately per the maintainer plan's decision
        1: options_write_supported() is the *carrier* OptionsEditModel's own
        gate -- construction never reaches the player block at all if this
        fails, so player edits inherit that failure regardless of whether
        the per-player block itself would have verified. players_write_
        supported() is player_fields.verify_player_block(), independent in
        the other direction: a PlayerDataTwo/Units-only failure must not
        greyed out Map Options or the Diplomacy grid.

        Number of Players is unioned in from its own third gate rather than
        sharing this one -- see _editable_player_count_field(). It survives
        a players_write_supported() failure, and can fail on its own while
        every per-player row here stays editable.
        """
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        per_player: frozenset[str] = frozenset()
        if players_write_supported(self.scenario):
            per_player = frozenset(
                player_fields.parse_player_field_id(key)[0]
                for key in player_fields.write_targets(self.scenario)
            )
        return per_player | self._editable_player_count_field()

    def _editable_player_count_field(self) -> frozenset[str]:
        """Number of Players' own gate, kept out of
        _editable_player_fields()' shared body because it is genuinely a
        different gate: it still inherits the carrier model's
        options_write_supported(), but instead of players_write_supported()
        it needs player_count_write_supported() -- the eight `active` flags
        plus FileHeader.player_count's own offset. The two disagree on a
        real corpus file (a scenario version 1.37 one, whose header walk
        does not reconcile), which is exactly why folding them together
        would grey out eleven editable rows to protect one.
        """
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        if not player_count_write_supported(self.scenario):
            return frozenset()
        return frozenset({player_fields.PLAYER_COUNT_FIELD_ID})

    _PLAYER_CARRIER_GATE_REASON = (
        "Read-only for this file: its map-option block failed verification, "
        "which the per-player write path shares a model with."
    )
    _PLAYER_GATE_REASON = (
        "Read-only for this file: its per-player block failed verification, so "
        "patching this setting would land at an offset that cannot be trusted."
    )
    _PLAYER_COUNT_GATE_REASON = (
        "Read-only for this file: this setting is stored twice (each player's "
        "active flag, and a count in the file header), and this file's header "
        "layout could not be resolved, so the two copies could not be kept in "
        "step. Every other player setting here is unaffected."
    )

    def _players_read_only_reasons(self) -> dict[str, str]:
        """Per-row tooltip text for ordinary Tier-1 rows a gate refused.

        Tier 2 (personality) and player_type are deliberately absent here:
        PlayersPanel resolves their own reasons unconditionally, since
        those are facts about the field kind, not about this file -- see
        that module's docstring. tribe_name is an ordinary Tier-1 row as of
        step 3d, and civilization/architecture as of the civ/architecture
        maintainer plan's Step B, and all three get the same gate reason as
        any other Tier-1 field -- Step A's brief window where
        civilization/architecture needed a field-kind-specific reason
        (unwritable on a 1.56+ file for a reason unrelated to this file's
        own per-player block verifying) closed once Step B's resizing
        splice made every version writable through the same
        players_write_supported() gate.
        """
        if self.scenario is None:
            return {}
        editable = self._editable_player_fields()
        carrier_ok = options_write_supported(self.scenario, option_fields.specs_for(self.scenario))
        reasons = {}
        for spec in player_fields.specs_for(self.scenario):
            if spec.field_id in editable or spec.field_id in player_fields._NEVER_WRITABLE:
                continue
            reasons[spec.field_id] = (
                self._PLAYER_CARRIER_GATE_REASON if not carrier_ok else self._PLAYER_GATE_REASON
            )
        if player_fields.PLAYER_COUNT_FIELD_ID not in editable:
            reasons[player_fields.PLAYER_COUNT_FIELD_ID] = (
                self._PLAYER_CARRIER_GATE_REASON if not carrier_ok
                else self._PLAYER_COUNT_GATE_REASON
            )
        return reasons

    def _pending_player_values(self) -> dict[str, dict[int, int | str]]:
        """{field_id: {player_id: value}} for every pending Players mode
        edit, across all 8 players at once -- PlayersPanel needs every
        player's pending value up front so switching the combo can show a
        different player's edit without a round trip back here. Same
        reasoning as _pending_option_values(): a byte-patch write never
        mutates the retriever it patches, so a repopulate that skipped this
        would reset the form to the file's values while the model still
        held the edits.
        """
        values: dict[str, dict[int, int]] = {}
        if self.option_edits is not None:
            for key, value in self.option_edits.pending_values().items():
                if not key.startswith("player:"):
                    continue
                field_id, player_id = player_fields.parse_player_field_id(key)
                values.setdefault(field_id, {})[player_id] = value
        return values

    def _pending_options(self) -> dict[str, int | str]:
        """The carrier model's pending edits, or {} before it exists --
        what player_fields.defined_player_ids()/_count() take as their
        `pending` argument."""
        return {} if self.option_edits is None else self.option_edits.pending_values()

    def _current_player_count(self) -> int:
        """Number of Players as edited, falling back to what the file
        stores -- see player_fields.defined_player_count()."""
        if self.scenario is None:
            return 0
        return player_fields.defined_player_count(self.scenario, self._pending_options())

    def _current_active_players(self) -> list[int]:
        """The player numbers Diplomacy mode's grid should show, as edited
        -- see player_fields.defined_player_ids(), whose pending branch
        makes this a contiguous prefix."""
        if self.scenario is None:
            return []
        return player_fields.defined_player_ids(self.scenario, self._pending_options())

    def set_player_count(self, value: int) -> None:
        """PlayersPanel's second callback: the user set Number of Players.

        Rides the same OptionsEditModel and the same OptionsDiffRecord
        every other Players mode row does (decision 1), so undo/redo needs
        nothing of its own.

        The trailing _repopulate_diplomacy() is a no-op today -- this
        spinbox only exists in Players mode, and entering Diplomacy runs
        _show_diplomacy() from scratch, which is what actually resizes the
        grid off the pending count. It is here for the same reason
        _move_history() calls both: a second entry point for this edit
        would otherwise leave a live grid stale.
        """
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_players()
            return
        try:
            with self._option_edit(
                model, player_fields.PLAYER_COUNT_FIELD_ID, f"Set number of players to {value}"
            ):
                model.set_value(player_fields.PLAYER_COUNT_FIELD_ID, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Number of players not set: {e}")
            self._repopulate_players()
            return
        self._repopulate_diplomacy()
        self._repopulate_stats_players()
        # Lowering the count drops the markers of the players it deactivated.
        self._refresh_camera_markers()

    def set_player_field(self, spec, player_id: int, value: int | str) -> None:
        """PlayersPanel's one callback: the user set `spec` to raw `value`
        for `player_id`. Every Players mode row rides the same
        OptionsEditModel Map Options and Diplomacy do (decision 1), so this
        needs no equivalent of set_option_field()'s exec-order/scalar fan-out
        -- there is only ever one destination.
        """
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_players()
            return
        field_id = player_fields.player_field_id(spec.field_id, player_id)
        try:
            with self._option_edit(model, field_id, f"Set P{player_id} {spec.label}"):
                model.set_value(field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Player field {field_id} not set: {e}")
            self._repopulate_players()
            return
        # After the _option_edit block, never inside it: set_value() runs in
        # the yield, so a recompute from in there would read pre-edit pending
        # values and then wrongly conclude nothing changed.
        self._after_player_color_change()

    # -- Point of View and the camera markers (GH #22) ---------------------

    def _pov_specs(self):
        """The (x, y) Point of View spec pair for the open file, or None
        when it has neither -- a pre-1.40 file, or a 1.41 one, whose
        mis-framed copy player_fields.specs_for() drops deliberately. One
        lookup shared by every reader below, since specs_for() walks every
        spec's retriever."""
        if self.scenario is None:
            return None
        specs = {s.field_id: s for s in player_fields.specs_for(self.scenario)}
        x_spec = specs.get(player_fields.POV_X_FIELD)
        y_spec = specs.get(player_fields.POV_Y_FIELD)
        if x_spec is None or y_spec is None:
            return None
        return x_spec, y_spec

    def _player_view(self, player_id: int, specs=None) -> tuple[int, int] | None:
        """P{n}'s Point of View as currently edited, or None when the file
        has no such field or the view is unset (-1 on either axis).

        Pending edits lead, the file's own value follows -- the same
        precedence PlayersPanel shows on screen, so an unsaved Set View
        moves the marker immediately rather than at the next save."""
        specs = specs or self._pov_specs()
        if specs is None:
            return None
        pending = self._pending_player_values()
        values = []
        for spec in specs:
            value = pending.get(spec.field_id, {}).get(player_id)
            if value is None:
                value = player_fields.current_value(self.scenario, spec, player_id)
            values.append(value)
        x, y = values
        if not isinstance(x, int) or not isinstance(y, int) or x < 0 or y < 0:
            return None
        return x, y

    def set_player_view(self, player_id: int) -> None:
        """PlayersPanel's Set View: write the tile at the centre of the map
        view as P{n}'s starting camera."""
        tile = self.map_view.viewport_centre_tile()
        if tile is None:
            self._log_status("View centre is off the map -- Set View wrote nothing")
            return
        self._write_player_view(player_id, tile[0], tile[1], f"Set P{player_id} Point of View")

    def reset_player_view(self, player_id: int) -> None:
        """PlayersPanel's Reset View: back to (-1, -1), which is what every
        unset slot across the corpus stores. Whether the game's own Reset
        writes the same pair is the in-game pass's question."""
        unset = player_fields.POV_UNSET
        self._write_player_view(player_id, unset, unset, f"Reset P{player_id} Point of View")

    def go_to_player_view(self, player_id: int) -> None:
        """PlayersPanel's Go to View: scroll to P{n}'s starting camera,
        clamped to the map the same way _navigate_to_finding clamps."""
        view = self._player_view(player_id)
        if view is None or self.scenario is None:
            return
        mm = self.scenario.map_manager
        x = max(0, min(mm.map_width - 1, view[0]))
        y = max(0, min(mm.map_height - 1, view[1]))
        self.map_view.center_on_tile(x, y)
        self._log_status(f"Centred on P{player_id}'s point of view ({x}, {y})")

    def _write_player_view(self, player_id: int, x: int, y: int, label: str) -> None:
        """Both coordinates as ONE undo step -- the composite shape
        set_disabled_ids() uses, and for the same reason: this is a single
        user gesture, and two set_player_field() calls would cost two
        Ctrl+Z and leave a half-moved camera in between."""
        specs = self._pov_specs()
        model = self._ensure_option_edits()
        if specs is None or model is None:
            self._repopulate_players()
            return
        records = []
        for spec, value in zip(specs, (x, y), strict=True):
            field_id = player_fields.player_field_id(spec.field_id, player_id)
            try:
                before = model.current_value(field_id)
                model.set_value(field_id, value)
            except (KeyError, ValueError) as e:
                self._log_status(f"Player field {field_id} not set: {e}")
                continue
            if model.current_value(field_id) == before:
                continue
            records.append(OptionsDiffRecord(label, field_id, before, value))
        if not records:
            self._repopulate_players()
            return
        if len(records) == 1:
            self.edit_history.push_options_record(records[0])
        else:
            self.edit_history.push_composite_record(CompositeDiffRecord(label, records))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)
        # The edit came from a button, not from a widget, so the panel's own
        # _values/_pending_values never saw it: without this the spinboxes
        # would still show the pre-edit numbers and the buttons' enabled
        # states would be stale.
        self._repopulate_players()
        self._refresh_camera_markers()

    def _on_player_selected(self, player_id: int) -> None:
        """PlayersPanel's selection notice. Only the markers' emphasis
        follows it; which player the panel is showing is read back off the
        panel rather than copied here (see its current_player_id())."""
        self._refresh_camera_markers()

    def _refresh_camera_markers(self) -> None:
        """View > Player Cameras: one marker per active player whose Point
        of View is set and lands on the map, in that player's colour.

        Emphasis applies only in Players mode -- elsewhere there is no
        selected player for it to mean anything about."""
        specs = self._pov_specs()
        if self.scenario is None or specs is None:
            self.map_view.set_camera_markers([], None)
            return
        mm = self.scenario.map_manager
        markers = []
        for player_id in range(1, self._current_player_count() + 1):
            view = self._player_view(player_id, specs)
            if view is None:
                continue
            x, y = view
            if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
                continue
            markers.append((player_id, x, y, QColor(*self.scenario.player_colors[player_id])))
        emphasised = self.players_panel.current_player_id() if self.mode == "players" else None
        self.map_view.set_camera_markers(markers, emphasised)

    # -- Disabled Objects (GH #57) ----------------------------------------

    def _options_carrier_ok(self) -> bool:
        """Whether the carrier OptionsEditModel's own map-option gate passes
        -- what tells the Disabled Objects button's tooltip which of its two
        gates refused."""
        if self.scenario is None:
            return False
        return options_write_supported(self.scenario, option_fields.specs_for(self.scenario))

    def _disables_editable(self) -> bool:
        """Whether the Disabled Objects dialog can be opened and applied.

        Two gates, the same shape _editable_diplomacy_fields() uses and for
        the same reason: options_write_supported() is the *carrier*
        OptionsEditModel's own gate, so a disables edit inherits its failure
        regardless; disables_write_supported() is independent in the other
        direction, so an Options-region failure must not grey out a single
        Map Options row.
        """
        if self.scenario is None:
            return False
        if not self._options_carrier_ok():
            return False
        return disables_write_supported(self.scenario)

    def _current_disabled_ids(self) -> dict[tuple[str, int], tuple[int, ...]]:
        """All 24 lists as the dialog should open on them: the model's
        pending value where there is one, the file's stored value otherwise.

        Read through the model rather than the file whenever one exists, for
        the same reason _pending_option_values() exists at all -- a splice
        write never mutates the retrievers it rebuilds from, so re-reading
        the file would show the original and a second OK would re-record an
        edit the model already has.
        """
        current: dict[tuple[str, int], tuple[int, ...]] = {}
        model = self.option_edits
        for field_id in disables_fields.all_field_ids():
            key = disables_fields.parse_disables_field_id(field_id)
            if model is not None and model.disables_supported:
                current[key] = tuple(model.current_value(field_id))
            else:
                current[key] = disables_fields.current_ids(self.scenario, *key)
        return current

    def _build_disables_dialog(self) -> DisablesDialog | None:
        """Constructs and wires the dialog without showing it.

        Split from _show_disables_dialog() so the GUI tests have something to
        drive: exec_() is modal and would hang an offscreen run outright
        (tests/test_keybinds.py's own docstring records the same trap for
        SettingsDialog).
        """
        if self.scenario is None or not self._disables_editable():
            return None
        return DisablesDialog(
            self._current_disabled_ids(),
            player_colors=self.scenario.player_colors,
            on_accept=self.set_disabled_ids,
            parent=self,
        )

    def _show_disables_dialog(self) -> None:
        dialog = self._build_disables_dialog()
        if dialog is None:
            return
        dialog.exec_()

    def set_disabled_ids(self, changes: Mapping[tuple[str, int], Sequence[int]]) -> None:
        """DisablesDialog's one callback: every list the user changed, in one
        call, applied as ONE undo step.

        One composite record rather than one per list, because the dialog is
        modal: the whole session was a single user gesture (open, edit,
        OK), and unwinding it should cost one Ctrl+Z, the same reasoning
        phase 2.8's region paste established.
        """
        if not changes:
            return
        model = self._ensure_option_edits()
        if model is None or not model.disables_supported:
            self._repopulate_players()
            return
        records = []
        for (category, player_id), ids in changes.items():
            field_id = disables_fields.disables_field_id(category, player_id)
            try:
                before = model.current_value(field_id)
                model.set_value(field_id, ids)
            except (KeyError, ValueError) as e:
                self._log_status(f"Disabled {category} for P{player_id} not set: {e}")
                continue
            if model.current_value(field_id) == before:
                continue
            records.append(
                OptionsDiffRecord(
                    f"Set P{player_id} disabled {category}", field_id, before, tuple(ids)
                )
            )
        if not records:
            self._repopulate_players()
            return
        if len(records) == 1:
            self.edit_history.push_options_record(records[0])
        else:
            self.edit_history.push_composite_record(
                CompositeDiffRecord("Set disabled objects", records)
            )
        self._update_title()
        self._update_edit_actions()
        self._log_status(
            f"Set disabled objects ({len(records)} list{'' if len(records) == 1 else 's'})"
        )

    def _after_player_color_change(self) -> None:
        """Shared tail for every edit that COULD have changed a player's
        colour -- Players mode's own form and the undo/redo of one. The
        Players-mode counterpart of _after_unit_mutation(), which this
        deliberately does not reuse: that method also rebuilds the unit pick
        index and resets the stack cycle, neither of which a recolour
        touches.

        Recompute strictly BEFORE invalidating. invalidate_units() bumps
        _source_gen and FlatChunkCache overrides it with an eager rebuild, so
        re-deriving afterwards would bake the stale tuples straight back in.

        Gating the invalidate on an actual difference is what makes this safe
        to call unconditionally: this runs for every player field (gold,
        wood, lock_personality...) and the undo/redo branch fires for Map
        Options and Diplomacy edits too, so an ungated call would evict the
        whole canvas on a gold-amount edit. The recompute itself is free (8
        ints to two 9-tuples).

        All four live player_colors/team_indices readers are unit-drawing
        paths, so terrain needs nothing here.

        The recolour work itself is _apply_player_color_change(); this
        wrapper exists only to hang the camera-marker refresh off a path
        that returns early inside that method on every edit that did NOT
        change a colour -- which includes a typed Point of View value and
        the undo of one, both of which move a marker.
        """
        self._apply_player_color_change()
        # After the recolour, never before it: the markers draw in
        # scenario.player_colors, which _apply_player_color_change() is what
        # re-derives.
        self._refresh_camera_markers()

    def _apply_player_color_change(self) -> None:
        if self.scenario is None or self.option_edits is None:
            return
        pending_colors = self._pending_player_values().get("color", {})
        if not refresh_player_colors(self.scenario, pending_colors):
            return
        # A live selection coloured by owner follows the recolour.
        self.map_view.set_selection_player_colors(self.scenario.player_colors)
        self.map_view.refresh_unit_highlight()
        # An in-flight warm is walking unit data this recolour just changed;
        # re-armed below, same as _after_unit_mutation()'s own third
        # invalidation.
        self._cancel_warms()
        if self._cache is not None:
            self._cache.invalidate_units()
            canvas_w, canvas_h = self._cache.canvas_dims(0)
            self._cache.invalidate_region((0, 0, canvas_w, canvas_h))
            self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        self._start_level_warm()
        # Unconditional, not mode-gated: a colour edit made in Players mode
        # must not leave Diplomacy's combo stale for whenever the user next
        # switches to it.
        self.players_panel.refresh_player_swatches(self.scenario)
        self.diplomacy_panel.refresh_player_swatches(self.scenario)
        self._refresh_stats_swatches()

    def _repopulate_players(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself
        (an undo) -- same reasoning as _repopulate_map_options()."""
        if self.mode == "players":
            self._show_players()

    def _show_diplomacy(self) -> None:
        """Populate the Diplomacy panel -- the single repopulate path for
        Diplomacy mode (mode entry, a new document, an edit, an undo/redo,
        or a refused edit all call only this, via _repopulate_diplomacy()
        for the latter three -- see PlayersPanel.show_scenario()'s own
        docstring for why one method covers all of them without resetting
        the player selection on every edit; DiplomacyPanel.show_scenario()
        follows the same rule).

        Like Players mode and unlike Map Options, no section needs parsing
        on demand first: the Diplomacy section is already fully parsed by
        load_map_and_units().
        """
        if self.scenario is None:
            self.diplomacy_panel.clear_document()
            return
        specs = self._diplomacy_option_specs()
        spec_ids = {spec.field_id for spec in specs}
        editable = self._editable_diplomacy_fields() | (
            self._editable_option_fields() & spec_ids
        )
        reasons = dict(self._diplomacy_read_only_reasons())
        reasons.update(
            {k: v for k, v in self._map_options_read_only_reasons().items() if k in spec_ids}
        )
        values = dict(self._pending_diplomacy_values())
        values.update(
            {k: v for k, v in self._pending_option_values().items() if k in spec_ids}
        )
        self.diplomacy_panel.show_scenario(
            self.scenario,
            option_specs=specs,
            editable_fields=editable,
            read_only_reasons=reasons,
            pending_values=values,
            active_players=self._current_active_players(),
        )

    def _diplomacy_option_specs(self) -> tuple[option_fields.OptionFieldSpec, ...]:
        """The Teams group's specs (step 5): option_fields.specs_for()
        covers every panel's rows together, so this filters to the ones
        DiplomacyPanel renders -- the same split MapOptionsPanel makes for
        its own panel."""
        if self.scenario is None:
            return ()
        return tuple(s for s in option_fields.specs_for(self.scenario) if s.panel == "diplomacy")

    def _editable_diplomacy_fields(self) -> frozenset[str]:
        """Which Diplomacy cell ids (stance:row:col, allied_victory:player)
        this file will accept edits for.

        Same two-gate shape as _editable_player_fields(), and for the same
        reason: options_write_supported() is the *carrier* OptionsEditModel's
        own gate -- construction never adds the grid's cells at all if this
        fails, so grid edits inherit that failure regardless of whether the
        grid itself would have verified. diplomacy_write_supported() is
        independent in the other direction: a Diplomacy-section-only failure
        must not grey out Map Options or Players mode.
        """
        if self.scenario is None:
            return frozenset()
        if not options_write_supported(self.scenario, option_fields.specs_for(self.scenario)):
            return frozenset()
        if not diplomacy_write_supported(self.scenario):
            return frozenset()
        return frozenset(_all_diplomacy_cell_ids(self.scenario))

    _DIPLOMACY_CARRIER_GATE_REASON = (
        "Read-only for this file: its map-option block failed verification, "
        "which the Diplomacy grid's write path shares a model with."
    )
    _DIPLOMACY_GATE_REASON = (
        "Read-only for this file: its Diplomacy section failed verification, so "
        "patching this cell would land at an offset that cannot be trusted."
    )

    def _diplomacy_read_only_reasons(self) -> dict[str, str]:
        """Per-cell tooltip text for every cell that is not editable -- same
        reasoning as _players_read_only_reasons()."""
        if self.scenario is None:
            return {}
        editable = self._editable_diplomacy_fields()
        carrier_ok = options_write_supported(self.scenario, option_fields.specs_for(self.scenario))
        reasons = {}
        for cell_id in _all_diplomacy_cell_ids(self.scenario):
            if cell_id in editable:
                continue
            reasons[cell_id] = (
                self._DIPLOMACY_CARRIER_GATE_REASON if not carrier_ok else self._DIPLOMACY_GATE_REASON
            )
        return reasons

    def _pending_diplomacy_values(self) -> dict[str, int]:
        """Raw values that differ from what the file's retrievers hold --
        same reasoning as _pending_option_values(): a byte-patch write never
        mutates the retriever it patches, so a repopulate that skipped this
        would reset every cell to the file's value while the model still
        held the edits."""
        values: dict[str, int] = {}
        if self.option_edits is not None:
            values = {
                key: value
                for key, value in self.option_edits.pending_values().items()
                if key.startswith(("stance:", "allied_victory:"))
            }
        return values

    def set_diplomacy_field(self, cell_id: str, value: int) -> None:
        """DiplomacyPanel's one callback: the user set the stance or
        allied-victory cell `cell_id` to raw `value`. Every Diplomacy row
        rides the same OptionsEditModel Map Options and Players do (decision
        1), so this needs no exec-order-style fan-out -- there is only ever
        one destination.
        """
        model = self._ensure_option_edits()
        if model is None:
            self._repopulate_diplomacy()
            return
        try:
            with self._option_edit(model, cell_id, f"Set {_diplomacy_field_label(cell_id)}"):
                model.set_value(cell_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Diplomacy field {cell_id} not set: {e}")
            self._repopulate_diplomacy()

    def _repopulate_diplomacy(self) -> None:
        """Put the form back in step with the models, after an edit that was
        refused or one applied from somewhere other than the panel itself
        (an undo) -- same reasoning as _repopulate_map_options()."""
        if self.mode == "diplomacy":
            self._show_diplomacy()

    _MESSAGES_READ_ONLY_REASON = (
        "Read-only for this file: its Messages section failed load-time "
        "verification, so a splice would land at an offset that cannot be "
        "trusted."
    )

    def _repopulate_messages(self) -> None:
        """Populate the Messages panel -- the single repopulate path for
        Messages mode (mode entry, a new document, an edit, an undo/redo, or
        a refused edit all call only this). A no-op outside the mode, same
        guard as _repopulate_diplomacy().

        Passes pending_values() the same way _pending_option_values() does:
        the splice write path never mutates the parsed retrievers, so a
        repopulate that skipped this would reset the form to the file's text
        while the model still held the edits.
        """
        if self.mode != "messages":
            return
        if self.scenario is None:
            self.messages_panel.clear_document()
            return
        editable = self.scenario.messages_write_supported
        self.messages_panel.show_scenario(
            self.scenario,
            values=self._pending_message_values(),
            editable=editable,
            read_only_reasons={} if editable else {"*": self._MESSAGES_READ_ONLY_REASON},
        )

    def _pending_message_values(self) -> dict[str, str | int]:
        """Raw values that differ from what the file's retrievers hold --
        same reasoning as _pending_option_values(): the splice write path
        never mutates a retriever, so a repopulate that skipped this would
        reset every field to the file's value while the model still held the
        edits."""
        if self.message_edits is None:
            return {}
        return self.message_edits.pending_values()

    def set_message_field(self, field_id: str, value: str | int) -> None:
        """MessagesPanel's one callback: the user set `field_id` (a text
        field id, or an "<id>_id" string-table id) to raw `value`."""
        model = self._ensure_message_edits()
        if model is None:
            self._repopulate_messages()
            return
        try:
            with self._messages_edit(model, field_id, f"Set {_message_field_label(field_id)}"):
                model.set_value(field_id, value)
        except (KeyError, ValueError) as e:
            self._log_status(f"Message field {field_id} not set: {e}")
            self._repopulate_messages()

    @contextmanager
    def _messages_edit(self, model: MessagesEditModel, field_id: str, label: str):
        """Shared undo/redo wiring for one MessagesEditModel field edit, the
        Messages-mode counterpart to _option_edit()."""
        before = model.current_value(field_id)
        yield model
        after = model.current_value(field_id)
        self.edit_history.push_messages_record(MessagesDiffRecord(label, field_id, before, after))
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)

    def _ensure_message_edits(self) -> MessagesEditModel | None:
        """The document's MessagesEditModel, built on first use.

        Deferred to the first real edit exactly as _ensure_option_edits() is,
        and for the same reason: a browse-only session must save
        byte-identically. Returns None for a file the model refuses, which
        the panel's gate should already have kept unreachable -- reported
        rather than swallowed, since silently ignoring an edit the user made
        is the worse failure.
        """
        if self.message_edits is not None:
            return self.message_edits
        if self.scenario is None:
            return None
        try:
            self.message_edits = MessagesEditModel(self.scenario)
        except MessageEditsUnavailableError as e:
            self._log_status(f"Messages are read-only for this file: {e}")
            QMessageBox.warning(self, "Messages are read-only", str(e))
            return None
        return self.message_edits

    def _force_mode(self, mode_text: str) -> None:
        """Switches the mode combo WITHOUT re-entering on_mode_changed.

        The re-entrancy is real, not hypothetical: this is called from
        _update_tool_enabled(), which on_mode_changed() itself calls, and
        mode_combo.currentTextChanged is wired to on_mode_changed. A plain
        setCurrentText() would recurse. Signals are blocked and the state
        on_mode_changed would have set is applied explicitly instead --
        deliberately not including its trigger-panel branch, since this only
        ever forces toward View.
        """
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText(mode_text)
        self.mode_combo.blockSignals(False)
        # Via _mode_id even though this only ever forces toward "View": two
        # label -> id conventions in one file is how the next multi-word mode
        # breaks silently.
        self.mode = _mode_id(mode_text)
        self.map_view.set_mode(self.mode)
        self.left_stack.setCurrentIndex(_LEFT_PAGE_FOR_MODE.get(self.mode, _LEFT_PAGE_INFO))
        self._selection = []
        self.map_view.set_unit_selection(None)
        self.units_panel.clear()
        self.pan_action.setChecked(True)
        # Same reason as on_mode_changed's own call: this leaves Players mode
        # too, and the emphasis has to go with it.
        self._refresh_camera_markers()
        self._log_status(f"Mode forced to {mode_text} -- Sloped has no unit selection yet")

    def _update_tool_enabled(self) -> None:
        # Tools (and Close/Save As) have nothing to act on before a map is
        # loaded -- grayed out rather than left clickable-but-pointless.
        # Tool params (Trees / Eye candy / Level) go further and hide outright
        # when the active tool doesn't read them at all (see the
        # _TOOL_PARAM block below). Mode applicability is a separate hide
        # (not grey) axis of its own now -- see the ToolDef.modes visibility
        # loop below -- so greying here covers only "wrong map state to use
        # right now" (no map / non-square map), not mode. Elevation and Set
        # Level further require terrain_write_supported (the load-time
        # terrain-block verification passed -- see scenario_io.py) and
        # map_is_square (MapManager.set_elevation, and even single-tile
        # elevation edits via elevation_tools.set_tile_elevation, raise
        # ValueError on any non-square map -- confirmed against
        # AoE2ScenarioParser's source, not just an assumption). Terrain
        # painting has no such constraint, only terrain_write_supported --
        # Paint Can shares that same gate (write_ok, not elevation_ok): it
        # never calls get_tile()/set_elevation() either, so squareness is
        # irrelevant to it too.
        has_map = self.scenario is not None
        write_ok = has_map and self.scenario.terrain_write_supported
        elevation_ok = write_ok and self.scenario.map_is_square
        # Phase 3's decision #6 scope cut (Stepped had no bounded
        # incremental redraw, so editing there meant a full ~1-2s
        # re-render per touched tile) is lifted as of Phase 4 --
        # refresh_region_iso() gives Stepped its own bounded redraw now,
        # same as Flat's own. Editing is gated the same way in both styles
        # from here on.
        # Place Unit's (and Convert's) own enablement gate -- has_map only,
        # with no write_ok-style check here: whether the file's units are
        # actually writable is discovered lazily by _ensure_unit_edits() on
        # the first real attempt, same as every trigger/option edit action.
        # Mode-inapplicability is handled separately, by ToolDef.modes'
        # visibility loop below, same as the four Terrain tools.
        unit_editable = has_map and self.mode == "units"
        # The four tools below carried a `sloped_editable` gate until Track
        # C4's Step 4: Sloped had no hit-testing, so none of them could
        # resolve a click to a tile there. C4 landed (pick plane, warped
        # outline, incremental patch path), so Terrain editing is
        # style-independent again and they gate on write_ok/elevation_ok
        # alone, exactly as in Flat and Stepped.
        #
        # C4 SPLIT that gate rather than deleting it, keeping a
        # `sloped_units_ok` half that forced Units mode back to View in
        # Sloped -- deliberately, so releasing the Terrain tools would not
        # silently release unit selection along with them. Track C5 landed
        # the missing half (unit_rise_px placement, and unit_pick's own
        # sloped pick/highlight branches), so there is nothing left for that
        # flag to gate and it is gone: mode is style-independent now, the
        # same way the tools became after C4.
        self.pan_action.setEnabled(has_map)
        # Same single gate as Pan: measuring reads the map and mutates nothing,
        # so it stays reachable in every mode and every Elevation View.
        self.ruler_action.setEnabled(has_map)
        # Mode no longer gates these four -- ToolDef.modes drives visibility
        # instead (below). A Terrain tool on write-unsupported/non-square
        # data now stays greyed AND visible rather than vanishing; a
        # mode-inapplicable tool hides outright instead of greying.
        self.draw_action.setEnabled(has_map and write_ok)
        self.fill_action.setEnabled(has_map and write_ok)
        # Same gate as Draw: terrain only, no elevation call anywhere in
        # either tool's commit path, so squareness is irrelevant to them.
        self.draw_line_action.setEnabled(has_map and write_ok)
        self.draw_rect_action.setEnabled(has_map and write_ok)
        self.elevation_action.setEnabled(has_map and elevation_ok)
        self.set_level_action.setEnabled(has_map and elevation_ok)
        # Map mirroring (Stage 1). Same gate as Set Level/Elevate --
        # plan_mirror's elevation half raw-assigns tile.elevation, the same
        # thing MapManager.set_elevation does, so it needs the same
        # map_is_square guarantee. Terrain-only mirroring never calls it
        # either, but there is no "terrain only" toggle at the gating layer
        # (the dialog's own Elevation checkbox handles that), so the whole
        # action shares elevation_ok rather than write_ok.
        self.mirror_action.setEnabled(has_map and elevation_ok)
        # Its own gate, not write_ok: the disables region is spliced, and
        # verifies (or not) independently of the terrain block every other
        # entry in this loop keys off. See _disables_editable().
        self.disables_action.setEnabled(self._disables_editable())
        # Read-only, so any loaded map; each check reports what it can't run.
        self.analysis_action.setEnabled(has_map)
        # Same gate as Draw/Paint Can (has_map and write_ok, no squareness
        # requirement): a cliff placement never touches the elevation
        # recursion elevation_ok exists for.
        self.cliff_action.setEnabled(has_map and write_ok)
        # Same gate as Draw/Paint Can/Cliff: a pick never calls
        # set_elevation() either, so squareness is irrelevant to it too.
        self.eyedropper_action.setEnabled(has_map and write_ok)
        # Same gate as Eyedropper: Select only reads (Copy) and its own
        # write (Paste) is gated separately below, so squareness is
        # irrelevant to the tool itself -- only to whether the Elevation
        # paste checkbox may be checked, gated further down.
        self.select_action.setEnabled(has_map and write_ok)
        self.place_unit_action.setEnabled(unit_editable)
        self.wall_rect_action.setEnabled(unit_editable)
        self.convert_action.setEnabled(unit_editable)
        # Deliberately coarse: this runs on mode changes, not entry-tree
        # clicks, so the template itself is validated per click instead.
        trigger_editable = has_map and self.mode == "triggers" and self.trigger_panel._editable
        self.create_objects_action.setEnabled(trigger_editable)
        # Rotate needs a selection as well as Units mode, unlike the two tools
        # above -- it acts on what is already selected rather than on a click,
        # so with nothing selected there is no target and the button should
        # say so by being grey rather than by logging a refusal. Not narrowed
        # to rotatable/gate consts: a selection's per-unit skips are reported
        # in the status line, and greying on content would flicker per click.
        for action in self._rotate_actions + self._variant_actions:
            action.setEnabled(unit_editable and bool(self._selection))
        self.select_stack_action.setEnabled(unit_editable and bool(self._selection))
        # Greyed rather than hidden outside Units mode, unlike Place Unit and
        # Convert: those hide via ToolDef.modes, but these four are the same
        # QAction objects the Edit menu holds, and a QAction's visibility is
        # not per-widget -- hiding them for the toolbar would empty the Edit >
        # Rotate Selection submenu too.
        #
        # The toolbar's own Rotate buttons are separate QAction objects (see
        # _build_toolbar()), so they can hide outright outside Units mode
        # like Place Unit/Convert do, without touching the Edit menu.
        for action in self._rotate_toolbar_actions:
            action.setVisible(unit_editable)
            action.setEnabled(unit_editable and bool(self._selection))
        self.close_action.setEnabled(has_map)
        self.save_action.setEnabled(write_ok)
        self.save_as_action.setEnabled(write_ok)
        self.terrain_style_combo.setEnabled(has_map)

        # Per-mode visibility (hide, not grey) for every registered tool --
        # ToolDef.modes empty means every mode, so Pan and Ruler stay visible
        # throughout; Place Unit and Convert hide outside Units mode here,
        # on top of (not instead of) their own has_map-based setEnabled above.
        for tool in settings.TOOLS:
            getattr(self, f"{tool.tool_id}_action").setVisible(tool_applicable(tool.tool_id, self.mode))

        # Filtering is read-only and style-independent, so it needs neither
        # write_ok nor a mode gate -- only something to filter.
        self.filters_button.setEnabled(has_map)

        # P3-g's View > Show sprites. Gated tighter than anything in Filters,
        # which is why it does not live there: it needs a configured AoE2DE
        # install to have any .sld to decode. That check earns its place --
        # without it every sprite_for()/icon_for() returns None and the toggle
        # is a silent no-op with no visible cause.
        #
        # EVERY style has a sprite compositor now: Stepped (P3-g3), Sloped
        # (P3-g6) and Flat (P3-g7, footprint-fitted icons rather than iso
        # sprites -- see unit_sprites.icon_for()). So there is no style gate
        # left here, only has_map and install_ok.
        #
        # Greying deliberately does NOT uncheck the action (contrast the
        # forced-back-to-Pan block below): the checked state IS the memory
        # that restores sprites once an install is configured, and
        # _render_current() reads self._sprites_enabled rather than the live
        # cache, so a disabled-but-checked action is a consistent state here
        # rather than the desync it would be for a tool.
        install_ok = asset_source.is_available()
        self.show_sprites_action.setEnabled(has_map and install_ok)
        if not has_map:
            sprite_tip = "Open a map first"
        elif not install_ok:
            sprite_tip = "Needs a configured AoE2DE install path (Settings) to read unit sprites from"
        else:
            sprite_tip = (
                "Draw units as their real game sprites instead of coloured marks -- "
                "turning this on decodes them once and can take a few seconds"
            )
        self.show_sprites_action.setToolTip(sprite_tip)

        # View > Layers, one loop over the registry. Greying deliberately
        # does NOT uncheck an action, for the same reason Show sprites above
        # doesn't: the checked state IS the memory that restores the layer
        # once it becomes available again, and _render_current() reads
        # self._layers rather than the live cache, so a disabled-but-checked
        # action is a consistent state here. Resetting it on grey would
        # silently flip the toggle on a style round trip.
        #
        # _on_sprites_toggled() calls this method for exactly this loop's
        # sake, so the farm row greys the moment sprites go off.
        for spec in view_layers.LAYERS:
            enabled, tip = view_layers.availability(
                spec, style=self._render_style, sprites_enabled=self._sprites_enabled, has_map=has_map
            )
            action = self.layer_actions[spec.layer_id]
            action.setEnabled(enabled)
            action.setToolTip(tip)

        # Disabling a checked QAction doesn't uncheck it or notify MapView --
        # e.g. Elevation selected, then a non-square file is opened over it.
        # Left alone, the toolbar button would sit checked-but-greyed-out
        # while MapView still had "elevation" as its live _tool, so a click
        # on the map would still start a stroke and hit set_tile_elevation's
        # ValueError on the new file. Force back to Pan whenever the
        # currently-selected tool is no longer enabled, or (Stage 2) no
        # longer applies to the active mode -- either way MapView must not
        # keep a live tool with no visible button.
        #
        # TOOLS-derived rather than hand-listed, but Pan and Ruler stay
        # explicitly excluded: Pan has no "back to Pan" to force, and
        # _on_tool_selected's own recursion-termination argument depends on
        # Pan having no entry here (it re-enters this method via
        # pan_action.setChecked(True), and that re-entry must be a no-op).
        # Ruler has no enablement/applicability reason to ever force back to
        # Pan -- gated on has_map alone, reachable in every mode.
        current_action = {
            t.tool_id: getattr(self, f"{t.tool_id}_action")
            for t in settings.TOOLS
            if t.tool_id not in ("pan", "ruler")
        }.get(self._current_tool)
        if current_action is not None and (
            not current_action.isEnabled() or not tool_applicable(self._current_tool, self.mode)
        ):
            self.pan_action.setChecked(True)

        # Tool params (terrain group / Level): which one shows, if either,
        # depends on the *final* self._current_tool for this call -- same
        # reason the copy/paste block below reads it only after the
        # forced-back-to-Pan block above has had its say. One boolean per
        # param feeds both the widget's visibility and its enabled state
        # (and, for Level, the ]/[ step-value keybinds too) so a hidden
        # param can never be left live behind the scenes.
        param = _TOOL_PARAM.get(self._current_tool, "")
        # Eyedropper writes both the sidebar terrain and Level from one pick (see
        # pick_tile_value()), so Level must stay visible while it's
        # active -- driven off its own action rather than _TOOL_PARAM, which
        # only ever names one param per tool. The terrain group keeps its pre-GH #56 gate.
        if self._current_tool == "eyedropper":
            terrain_param_ok = self.eyedropper_action.isEnabled()
            level_param_ok = self.eyedropper_action.isEnabled()
        else:
            # current_action rather than the old hardcoded draw/fill pair:
            # those two literals only ever meant "the active terrain tool's
            # own button", which is what current_action is -- and Draw Line/
            # Draw Rectangle are terrain tools the pair would have missed.
            terrain_param_ok = (
                param == "terrain" and current_action is not None and current_action.isEnabled()
            )
            level_param_ok = param == "level" and self.set_level_action.isEnabled()
        convert_param_ok = param == "convert" and self.convert_action.isEnabled()
        cliff_param_ok = param == "cliff" and self.cliff_action.isEnabled()
        # Brush size/shape -- reuses current_action (computed above for the
        # forced-back-to-Pan check) rather than re-deriving write_ok/
        # elevation_ok: brush_ok should track exactly whether the active
        # tool's own toolbar action is enabled, same as terrain_param_ok/
        # level_param_ok already do via draw_action/fill_action/
        # set_level_action above. Elevate has no param_widget at all today
        # (param == ""), so this is the first param group it ever shows --
        # see the separator line below, which previously assumed Elevate
        # showed nothing.
        # brush_applicable(), not BRUSH_TOOLS membership: Draw Rectangle
        # carries a brush in Outline mode only. See that predicate.
        brush_ok = (
            brush_applicable(self._current_tool, rect_filled=self._rect_filled())
            and current_action is not None
            and current_action.isEnabled()
        )
        rect_param_ok = _TOOL_SHAPE.get(self._current_tool) == "rect" and (
            current_action is not None and current_action.isEnabled()
        )
        # terrain_param_ok gates no picker of its own (the sidebar page stays live throughout
        # Terrain mode); it still gates Trees, Eye candy and auto beach below.
        # Auto beach. The `== "draw"` term is load-bearing, not tidiness:
        # Paint Can shares param_widget="terrain", so without it the
        # checkbox would appear for Fill and silently do nothing. The
        # shape tools are excluded for the same reason -- their commit
        # path is on_shape_commit, not the Draw stroke branch the ring
        # hangs off.
        auto_beach_ok = (
            terrain_param_ok
            and self._current_tool == "draw"
            and terrain_classes.is_water_family(self.terrain_panel.terrain_id())
        )
        self.auto_beach_param_action.setVisible(terrain_param_ok and self._current_tool == "draw")
        self.auto_beach_check.setEnabled(auto_beach_ok)
        self.auto_beach_check.setToolTip(
            "Paint a beach shoreline around the water this stroke lays down"
            if auto_beach_ok
            else "Pick a water terrain to enable the automatic shoreline"
        )
        # The combo and width follow the checkbox, so a hidden setting can
        # never sit live behind the scenes.
        beach_param_ok = auto_beach_ok and self.auto_beach_check.isChecked()
        self.beach_param_action.setVisible(beach_param_ok)
        self.beach_combo.setEnabled(beach_param_ok)
        self.beach_width_param_action.setVisible(beach_param_ok)
        self.beach_width_spin.setEnabled(beach_param_ok)
        self.paint_trees_param_action.setVisible(terrain_param_ok)
        self.paint_trees_check.setEnabled(terrain_param_ok)
        self.paint_eye_candy_param_action.setVisible(terrain_param_ok)
        self.paint_eye_candy_check.setEnabled(terrain_param_ok)
        self.level_param_label_action.setVisible(level_param_ok)
        self.level_param_spin_action.setVisible(level_param_ok)
        self.elevation_level_spin.setEnabled(level_param_ok)
        self.convert_sources_param_action.setVisible(convert_param_ok)
        self.convert_sources_button.setEnabled(convert_param_ok)
        self.rect_fill_param_action.setVisible(rect_param_ok)
        self.rect_fill_combo.setEnabled(rect_param_ok)
        self.brush_param_label_action.setVisible(brush_ok)
        self.brush_size_spin_action.setVisible(brush_ok)
        self.brush_size_spin.setEnabled(brush_ok)
        self.brush_shape_combo_action.setVisible(brush_ok)
        self.brush_shape_combo.setEnabled(brush_ok)
        self.tool_value_inc_action.setEnabled(level_param_ok or brush_ok)
        self.tool_value_dec_action.setEnabled(level_param_ok or brush_ok)
        self.cliff_family_label_action.setVisible(cliff_param_ok)
        self.cliff_family_param_action.setVisible(cliff_param_ok)
        self.cliff_family_combo.setEnabled(cliff_param_ok)
        self.cliff_piece_label_action.setVisible(cliff_param_ok)
        self.cliff_piece_param_action.setVisible(cliff_param_ok)
        self.cliff_piece_combo.setEnabled(cliff_param_ok)
        self.cliff_frame_label_action.setVisible(cliff_param_ok)
        self.cliff_frame_param_action.setVisible(cliff_param_ok)
        self.cliff_frame_spin.setEnabled(cliff_param_ok)
        self.cliff_preview_param_action.setVisible(cliff_param_ok)

        # Phase 2.8's Copy/Paste Region -- gated on the committed region and
        # clipboard alone, NOT on the active tool: a region survives a tool
        # switch (see MapView.set_tool()'s own comment), so Copy/Paste must
        # stay available under Draw or Elevate just as much as under Select
        # itself, mirroring how Rotate acts on self._selection regardless of
        # which tool is active.
        if self.mode == "triggers":
            editable = self.trigger_panel._editable
            self.copy_action.setText("&Copy Triggers")
            self.paste_action.setText("&Paste Triggers")
            self.copy_action.setEnabled(editable and bool(self.trigger_panel.selected_trigger_indices()))
            self.paste_action.setEnabled(editable and self._trigger_paste_allowed())
            reason = self._trigger_paste_refusal() if self._trigger_clipboard is not None else ""
            # Empty restores Qt's default, the action's own text.
            self.paste_action.setToolTip(reason)
            self.select_all_action.setEnabled(self.trigger_panel.tree.topLevelItemCount() > 0)
            self.deselect_action.setEnabled(bool(self.trigger_panel.selected_trigger_indices()))
        else:
            self.copy_action.setText("&Copy Region")
            self.paste_action.setText("&Paste Region")
            self.paste_action.setToolTip("")
            self.copy_action.setEnabled(has_map and self._region is not None)
            self.paste_action.setEnabled(has_map and self._region_clipboard is not None)
            self.select_all_action.setEnabled(has_map)
            self.deselect_action.setEnabled(has_map and self._region is not None)
        # No mode gate, for Copy/Paste's reason above: Select only exists in
        # Terrain mode, so gating scatter on Units mode would force a round
        # trip through Terrain, Select, drag, Units on every use.
        self.scatter_action.setEnabled(has_map and self._region is not None)

        # The paste filter checkboxes: visible only while Select is active,
        # same convention as every other tool param above. Elevation is
        # additionally gated on elevation_ok (map_is_square) -- paste_region()
        # would otherwise hand set_tiles_elevation a non-square map and hit
        # MapManager.get_tile's ValueError mid-stroke. _close_stroke_on_error()
        # keeps that from wedging EditHistory, but the paste would still fail.
        select_param_ok = self._current_tool == "select" and self.select_action.isEnabled()
        self.paste_terrain_param_action.setVisible(select_param_ok)
        self.paste_terrain_check.setEnabled(select_param_ok)
        self.paste_elevation_param_action.setVisible(select_param_ok)
        self.paste_elevation_check.setEnabled(select_param_ok and elevation_ok)
        self.paste_units_param_action.setVisible(select_param_ok)
        self.paste_units_check.setEnabled(select_param_ok)

        # D2's free-placement toggle -- same tool-param convention as every
        # group above, derived from FREE_PLACE_TOOLS rather than naming
        # Place Unit here, so a second free-placing tool needs only a ToolDef
        # field. It goes in the separator roll-up below too: a new param group
        # that forgets that line leaves the separator hidden while its own
        # widget shows.
        free_place_ok = (
            self._current_tool in FREE_PLACE_TOOLS
            and current_action is not None
            and current_action.isEnabled()
        )
        self.free_place_param_action.setVisible(free_place_ok)
        self.free_place_check.setEnabled(free_place_ok)

        self.tool_param_separator_action.setVisible(
            terrain_param_ok or level_param_ok or convert_param_ok
            or cliff_param_ok or brush_ok or select_param_ok or rect_param_ok
            or beach_param_ok or free_place_ok
        )

        # Stage 3: mode may have just changed which tools are applicable,
        # and the active tool may have just changed which one the More
        # Tools button's own label should track -- both are this function's
        # business already, so recompute overflow membership here too.
        self._apply_toolbar_overflow()

    def _sync_map_view_brush(self) -> None:
        """Pushes the toolbar's current brush size/shape into MapView's
        hover-preview state -- but only for a tool that actually has one
        (BRUSH_TOOLS); Pan and Paint Can always preview a single tile
        regardless of what size/shape the spinbox/combo were last left at
        for Terrain/Elevate/Set Elevation. Called both when the brush
        widgets change and when the active tool changes, so the preview is
        never stale in either direction -- and when Draw Rectangle's
        Fill/Outline toggle moves, which is the third thing that can change
        the answer (see brush_applicable): without that call MapView would
        keep the last brush size and dilate the hover preview while the
        brush widgets are hidden."""
        # Pushed from here rather than from the combo's own handler so the
        # two can't drift: every path that syncs the brush syncs this too.
        self.map_view.set_rect_filled(self._rect_filled())
        if brush_applicable(self._current_tool, rect_filled=self._rect_filled()):
            self.map_view.set_brush(self.brush_size_spin.value(), self.brush_shape_combo.currentData())
        else:
            self.map_view.set_brush(brush.BRUSH_SIZE_MIN, brush.BRUSH_SHAPE_SQUARE)

    def _rect_filled(self) -> bool:
        """Draw Rectangle's Fill/Outline state. A method rather than a plain
        attribute so the combo stays the single source of truth -- nothing
        mirrors it into a field that could drift."""
        return bool(self.rect_fill_combo.currentData())

    def _on_rect_fill_changed(self) -> None:
        # Toggling Filled/Outline changes whether the brush applies at all,
        # so this has to re-run the whole param gate, not just push the
        # brush down into MapView.
        self._update_tool_enabled()
        self._sync_map_view_brush()
        self.map_view.refresh_highlight(self._hover_tile)

    def _on_brush_changed(self) -> None:
        self._sync_map_view_brush()
        self.map_view.refresh_highlight(self._hover_tile)

    def _on_terrain_changed(self, terrain_id: int) -> None:
        """The sidebar picker's hookup, for a user pick and the Eyedropper's
        set_terrain() alike: re-gates the param row (auto beach needs a
        water-family terrain) and logs the new terrain."""
        self._update_tool_enabled()
        self._log_status(f"Terrain type: {name_for_terrain_id(terrain_id)}")

    def _on_auto_beach_toggled(self, checked: bool) -> None:
        # Re-gates the combo and width, which are visible only while the box
        # is ticked.
        self._update_tool_enabled()

    def _on_paint_trees_toggled(self, checked: bool) -> None:
        settings.set_paint_trees(checked)

    def _on_paint_eye_candy_toggled(self, checked: bool) -> None:
        settings.set_paint_eye_candy(checked)

    def _on_tool_selected(self, tool: str) -> None:
        if tool != "pan":
            # Pick from map lives on Pan; any other tool takes the clicks back.
            self.disarm_unit_picker()
        # set_tool() first: it can synchronously close a dangling stroke via
        # _end_stroke() -> on_edit_stroke_end(), which labels the undo record
        # from self._current_tool -- that must still read the OUTGOING tool.
        self.map_view.set_tool(tool)
        self._current_tool = tool
        self._sync_map_view_brush()
        self._update_mode_status()
        self._log_status(f"Tool changed to {tool}")
        # v2.7 copy/paste: Copy/Paste's enabled state depends on which tool
        # is active (and, for Paste, whether the clipboard's kind matches
        # it -- see _update_tool_enabled()'s own comment) but this method is
        # the one path that changes self._current_tool without going
        # through _update_tool_enabled() itself (unlike on_mode_changed/
        # on_terrain_style_changed/load_scenario/close_scenario, which all
        # call it already). Without this, switching tools (e.g. Terrain ->
        # Elevate with a terrain-kind clipboard) would leave Paste enabled
        # from the previous tool's evaluation until something else happened
        # to trigger a refresh. Safe against the recursion this method can
        # itself trigger indirectly (_update_tool_enabled() forcing Pan back
        # on via pan_action.setChecked(True) when the current tool becomes
        # disabled, which re-enters this method) -- that re-entry finds Pan
        # already the current tool, whose lookup in _update_tool_enabled()'s
        # forced-Pan check is a no-op, so it terminates rather than looping.
        self._update_tool_enabled()

    def on_terrain_style_changed(self, text: str) -> None:
        # Validated rather than a bare .lower(): the combo only ever holds
        # STYLE_LABELS, so an unknown label here means a caller invented one.
        style = style_for_label(text)
        if style == self._terrain_style:
            return
        if self._busy:
            # A render is already in progress (reentrant signal -- see
            # load_scenario's comment on self._busy). Refuse the change and
            # put the combo back to what's actually loaded rather than
            # leaving it displaying a style nothing was ever rendered at;
            # blockSignals so this doesn't recurse back into this handler.
            self.terrain_style_combo.blockSignals(True)
            self.terrain_style_combo.setCurrentText(label_for(self._terrain_style))
            self.terrain_style_combo.blockSignals(False)
            return
        self._terrain_style = style
        # Flat's own view-rotation checkbox has nothing to do while a
        # Stepped image is loaded (its projection is already baked into the
        # pixels) -- see the checkbox's own tooltip.
        self.iso_action.setEnabled(style == "flat")
        self._update_tool_enabled()
        self._update_mode_status()
        if self.scenario is not None:
            self._busy = True
            try:
                # reset_view=False: same document, just a different Elevation
                # View. The projection genuinely changes shape (Flat's plain
                # grid vs Stepped/Sloped's isometric projection), so the
                # restored zoom/pan is an approximation of the same map area
                # rather than pixel-exact -- still far closer to what the
                # user was looking at than snapping back to full-map fit.
                elapsed, tile_px = self._render_current(reset_view=False)
                self._log_status(f"Elevation view: {text} (tile_px={tile_px}) prepared in {elapsed:.2f}s")
            finally:
                self._busy = False

    def _on_iso_toggled(self, checked: bool) -> None:
        """Isometric View's own handler (Flat+Isometric plan, Step 3),
        replacing the old plain `lambda checked: self.map_view.set_isometric
        (checked)` connection now that toggling it in Flat has a RENDER path
        to switch, not just a view transform to flip.

        set_isometric() first, always (F4): it must run BEFORE the guarded
        re-render below, since self._render_style reads this action's own
        checked state and _render_current()'s set_source() call ends by
        calling set_isometric() again -- which, now that the style reaching
        MapView is "stepped", takes MapView's own early return and skips the
        QTransform. That is the only thing that ever turns the transform
        off; nothing else does.

        Guarded per F3: setChecked() below (construction-time, and every
        on_terrain_style_changed() switch) fires this handler before any
        scenario exists or while a render is already in progress -- both
        must be a no-op past the view-transform call. Stepped/Sloped never
        reach the render below either: the checkbox is disabled in both, and
        only Flat's render path actually changes with this toggle."""
        self.map_view.set_isometric(checked)
        # Re-gate before the guarded render, not after: this action is one of
        # the two inputs to self._render_style, so flipping it moves Flat
        # between "really Flat" and "renders through IsoChunkCache" -- and
        # View > Layers' farm row is gated on exactly that. Placed above the
        # early returns so the busy/no-map paths re-gate too; with no map
        # every row is greyed anyway.
        self._update_tool_enabled()
        if self.scenario is None or self._busy or self._terrain_style != "flat":
            return
        self._busy = True
        try:
            elapsed, tile_px = self._render_current(reset_view=False)
            state = "on" if checked else "off"
            self._log_status(f"Isometric View {state} (tile_px={tile_px}) prepared in {elapsed:.2f}s")
        finally:
            self._busy = False

    # -- Edit-tool strokes (terrain paint / elevation raise-lower / set level) --
    # One stroke (mouse-down through release, or a defensive close on leave --
    # see MapView) is exactly one descape.edit_history.EditHistory record and
    # one undo step, but repaints live as it's dragged. See edit_history.py's
    # module docstring for why nothing here writes to a tile directly.

    def on_edit_stroke_start(self) -> None:
        if self.scenario is None:
            return
        if self._current_tool == "convert":
            self._begin_convert_stroke()
            return
        # Above begin_stroke(), like convert: a cliff stroke writes units,
        # not terrain, so falling through would open an EditHistory terrain
        # stroke that commits empty at release -- the plan's own third
        # Stage 1 invariant, now live.
        if self._current_tool == "cliff":
            self._begin_cliff_stroke()
            return
        self.edit_history.begin_stroke(self.scenario.map_manager.terrain)
        # index -> the tile state as of the last time we handed that index to
        # _apply_dirty. A dict, not a set of indices: elevation propagation
        # can change one tile SEVERAL times over a single drag, and a
        # membership-only set silently drops every change after the first --
        # see on_edit_stroke_tile's own comment.
        self._stroke_seen_state: dict[int, tuple[int, int, int]] = {}
        # Painted-tile dedupe for this stroke, keyed on the actual tile a
        # brush touched -- see on_edit_stroke_tile's own comment for why
        # this must be separate from MapView._stroke_touched (which is
        # keyed on the CURSOR tile, and is only a cheap early-out, not a
        # correctness guarantee once a brush is bigger than one tile).
        self._stroke_painted: set[tuple[int, int]] = set()
        # Auto beach, snapshotted HERE rather than read per touch. The
        # checkbox, beach combo and width spin are all live widgets, so
        # reading them inside on_edit_stroke_tile would half-apply a
        # mid-stroke change across one undo record. A separate Water tool
        # would not have had this problem; the checkbox does.
        self._stroke_auto_beach = (
            (
                self.auto_beach_check.isEnabled() and self.auto_beach_check.isChecked(),
                self.beach_combo.currentData(),
                self.beach_width_spin.value(),
            )
            if self._current_tool == "draw"
            else (False, None, 0)
        )

    def on_edit_stroke_tile(self, x: int, y: int, modifiers) -> None:
        self.on_edit_stroke_tiles([(x, y)], modifiers)

    def on_edit_stroke_tiles(self, tiles, modifiers) -> None:
        """MapView's per-event stroke entry: `tiles` is the cursor path in
        order. Mutates tile by tile exactly as that many single-tile calls
        would (Elevate's propagation order depends on it), then runs the
        O(map) stroke scan and _apply_dirty once for the whole event."""
        if self.scenario is None:
            return
        if self._current_tool == "convert":
            for x, y in tiles:
                self._convert_stroke_tile(x, y)
            return
        if self._current_tool == "cliff":
            for x, y in tiles:
                self._cliff_stroke_tile(x, y)
            return
        # Closed early by a raise below: the rest of this drag writes nothing.
        if not self.edit_history.in_stroke:
            return
        label = _STROKE_LABELS.get(self._current_tool, "Edit")
        with self._close_stroke_on_error(label):
            changed = False
            for x, y in tiles:
                changed |= self._mutate_stroke_tile(x, y, modifiers)
            if changed:
                self._apply_stroke_dirty()

    @contextmanager
    def _close_stroke_on_error(self, label: str):
        """Wraps work done inside an open EditHistory stroke. On a raise (e.g.
        MapManager.get_tile() on a non-square map) the stroke is committed
        before the exception propagates, so _stroke_before can't stay set and
        wedge every later begin_stroke(). Committed, not aborted: tiles already
        written can't be un-painted (abort_stroke()'s docstring), and this
        keeps them on the undo stack and repainted instead of silently live."""
        try:
            yield
        except BaseException:
            if self.edit_history.in_stroke:
                touched = self.edit_history.commit_stroke(label, self.scenario.map_manager.terrain)
                self._stroke_seen_state = {}
                self._stroke_painted = set()
                self._stroke_auto_beach = (False, None, 0)
                self._update_edit_actions()
                self._update_title()
                # The repaint can hit the same fault; the original exception is the one to report.
                with suppress(Exception):
                    self._apply_dirty(touched)
            raise

    def _mutate_stroke_tile(self, x: int, y: int, modifiers) -> bool:
        """One cursor tile's footprint edit. True if it wrote anything."""
        mm = self.scenario.map_manager
        # Every drag tool reaching this method today (terrain/elevation/
        # set_level) has supports_brush=True, so this is BRUSH_TOOLS in
        # practice -- checked explicitly rather than assumed, so a future
        # is_edit_tool tool with no brush still degrades to its old
        # single-tile behavior instead of silently expanding.
        if self._current_tool in BRUSH_TOOLS:
            footprint = brush.brush_tiles(
                x, y, self.brush_size_spin.value(), self.brush_shape_combo.currentData(), mm.map_width, mm.map_height
            )
        elif 0 <= x < mm.map_width and 0 <= y < mm.map_height:
            footprint = [(x, y)]
        else:
            footprint = []
        # Dedupe against tiles this stroke already painted -- NOT against
        # MapView._stroke_touched's cursor-tile set. With a brush bigger
        # than one tile, a single painted tile falls under many distinct
        # cursor tiles during a drag; without this, Elevate's accumulating
        # +/-1 (see below) would raise/lower the same tile once per cursor
        # tile that overlapped it, not once per stroke.
        footprint = [t for t in footprint if t not in self._stroke_painted]
        if not footprint:
            return False

        if self._current_tool == "draw":
            terrain_id = self.terrain_panel.terrain_id()
            for tx, ty in footprint:
                tile = mm.get_tile(tx, ty)
                tile.terrain_id = terrain_id
                # Clear a stale double-terrain blend -- render.py doesn't draw
                # `layer`, but the game does, and leaving it set after changing
                # terrain_id would make this tool's own render lie about what
                # the game will actually show.
                tile.layer = -1
            # The shoreline, laid AFTER the core so a tile this touch just
            # made water can never be beached, and strictly BEFORE the
            # stroke_dirty_indices block below -- after it, the ring tiles
            # would stay unpainted on screen until the next cursor touch,
            # which reads as a rendering bug. `footprint` is already the
            # dedupe-filtered core, which is correct: a tile skipped because
            # an earlier touch painted it is already water, and its ring was
            # laid then.
            beach_on, beach_id, beach_width = self._stroke_auto_beach
            if beach_on:
                apply_beach_ring(mm, footprint, terrain_id, beach_id, beach_width)
        elif self._current_tool == "elevation":
            delta = -1 if modifiers & Qt.ShiftModifier else 1
            # Clamp to the legal range Set Elevation's spinbox already
            # enforces (ELEVATION_LEVEL_MAX) -- Elevate had no clamp at all
            # before Phase 4: going below 0 wrote a negative int that
            # scenario_write.py's raw byte patch (`body[o+1] = tile.elevation`)
            # would crash on at save time, and going above the legal range
            # is exactly the case the fixed-range canvas sizing every Stepped
            # projection uses (iso_geometry.canvas_size_and_origin, see
            # iso_geometry.MAX_ELEVATION) now depends on never happening for
            # Stepped mode's incremental redraw to stay safe. Both real, not
            # hypothetical -- confirmed by reading scenario_write.py and by
            # descape.render.dirty_screen_bbox_iso's own guard below.
            #
            # Every target's current elevation is read here, before ANY tile
            # in this footprint is written -- set_tiles_elevation's
            # propagation pass can still change a not-yet-processed
            # footprint tile's elevation (that's the whole point of a
            # footprint-wide xys), and reading "current" late would apply
            # delta on top of that propagated value instead of the value
            # this cursor tile's drag actually found.
            targets = [
                (tx, ty, max(0, min(ELEVATION_LEVEL_MAX, mm.get_tile(tx, ty).elevation + delta)))
                for tx, ty in footprint
            ]
            set_tiles_elevation(mm, targets)
        elif self._current_tool == "set_level":
            level = self.elevation_level_spin.value()
            set_tiles_elevation(mm, [(tx, ty, level) for tx, ty in footprint])
        else:
            return False

        self._stroke_painted.update(footprint)
        return True

    def _apply_stroke_dirty(self) -> None:
        mm = self.scenario.map_manager
        # Cumulative dirty set since stroke start, minus what's already been
        # redrawn AT ITS CURRENT STATE this stroke -- avoids repainting the
        # same tile repeatedly as the drag continues over tiles elevation
        # propagation already touched. See EditHistory.stroke_dirty_indices's
        # docstring for the cost of this (a linear scan) at this project's map
        # sizes -- this is exactly why it runs once per mouse event (after
        # every footprint on the event's gap-filled cursor path is written),
        # never once per brush or path tile: doing that would multiply an
        # already-O(map) scan on every mouse-move. Same warning on_fill's
        # own docstring carries for its single full-map fill.
        #
        # Compared on STATE, not on index membership. stroke_dirty_indices is
        # cumulative (everything differing from the stroke-start snapshot), so
        # once a tile appears it stays for the rest of the drag -- and
        # set_tiles_elevation's propagation routinely changes one tile several
        # times as the brush moves over it. Subtracting a plain set of indices
        # therefore synced each tile exactly once, at its FIRST value, and
        # froze it there: _apply_dirty -> dirty_screen_bbox_iso is the only
        # thing that writes MapView._iso_elevations, so that snapshot drifted
        # permanently out of sync with tile.elevation. Measured on one
        # 8-step Set Elevation drag: 131 tiles wrong, 35 of them sitting at
        # elevation 6 while the snapshot still read lower, some off by 2.
        # Top faces still looked right (they render from tile.elevation), but
        # the hover highlight and screen_to_tile hit-testing read the array,
        # and _render_tile_iso mixes the two when it computes skirt/contact-
        # shadow deltas as tile.elevation - elevations[neighbour].
        with perf_trace.phase("stroke_scan"):
            all_dirty = self.edit_history.stroke_dirty_indices(mm.terrain)
        # One tile_state() per cumulative dirty index, not two passes over it:
        # _stroke_seen_state only moves where the state actually changed.
        new_dirty = set()
        for i in all_dirty:
            state = tile_state(mm.terrain[i])
            if state != self._stroke_seen_state.get(i):
                self._stroke_seen_state[i] = state
                new_dirty.add(i)
        self._apply_dirty(new_dirty)

    def on_edit_stroke_end(self) -> None:
        if self.scenario is None:
            return
        if self._current_tool == "convert":
            self._end_convert_stroke()
            return
        if self._current_tool == "cliff":
            self._end_cliff_stroke()
            return
        # _close_stroke_on_error() already committed it; MapView still sends the release.
        if not self.edit_history.in_stroke:
            return
        label = _STROKE_LABELS.get(self._current_tool, "Edit")
        mm = self.scenario.map_manager
        if self._current_tool == "draw":
            # Not commit_stroke(): terrain_units needs the built-but-unpushed
            # record first, to decide whether it rides alone or inside a
            # CompositeDiffRecord with the trees/eye-candy it triggers.
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain)
            self._push_terrain_unit_record(self._apply_terrain_unit_plan(tile_record, mm))
        else:
            self.edit_history.commit_stroke(label, mm.terrain)
        self._stroke_seen_state = {}
        self._stroke_painted = set()
        self._stroke_auto_beach = (False, None, 0)
        self._update_edit_actions()
        self._update_title()
        perf_trace.flush(label.lower().replace(" ", "-"))

    def on_shape_commit(self, tiles) -> None:
        """Draw Line / Draw Rectangle: one undo record per drag, applied at
        release with the exact tile set MapView's preview named. MapView
        computes that set (see its own _shape_tiles) so the committed shape
        can never disagree with the previewed one.

        Shaped like on_fill() rather than the stroke handlers, for the same
        reason: there is no live drag to give feedback during, so the O(map)
        stroke_dirty_indices() scan per touched tile buys nothing here.
        Routed through _apply_terrain_unit_plan() too, so Trees/Eye candy
        follow a painted line exactly as they follow a Draw stroke. Wrapped
        in on_fill()'s busy/wait-cursor guard: a full-map filled rectangle
        is the same order of work as a full-map flood fill (~540ms in Flat),
        long enough that the window would otherwise look hung."""
        if self.scenario is None or self._busy or not tiles:
            return
        # Place Unit's wall branch (GH #98) and Wall Rectangle are the
        # non-terrain shape commits, so this method's whole body below is the
        # terrain branch. Dispatched before the tree/eye-candy size prompt,
        # which has nothing to say about units.
        if self._current_tool in ("place_unit", "wall_rect"):
            self._commit_wall_run(tiles, self.units_panel.selected_object_const())
            return
        mm = self.scenario.map_manager
        terrain_id = self.terrain_panel.terrain_id()
        label = _STROKE_LABELS.get(self._current_tool, "Edit")

        # Same pre-write size guard as on_fill's -- sized before anything is
        # written, so Cancel leaves the map untouched. Draw is exempt there
        # because a stroke's count is bounded by brush size x drag length; a
        # filled rectangle has no such bound, so it needs the guard.
        paints_units = self.paint_trees_check.isChecked() or self.paint_eye_candy_check.isChecked()
        if paints_units and len(tiles) > TERRAIN_UNIT_CONFIRM_THRESHOLD:
            reply = QMessageBox.question(
                self,
                "Large edit",
                f"This covers {len(tiles)} tiles and can place a large number of "
                f"trees/eye candy units, which may take a while. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.edit_history.begin_stroke(mm.terrain)
            with self._close_stroke_on_error(label):
                for tx, ty in tiles:
                    set_terrain(mm.get_tile(tx, ty), terrain_id)
            tile_record = self.edit_history.build_stroke_record(label, mm.terrain)
            record = self._apply_terrain_unit_plan(tile_record, mm)
            self._push_terrain_unit_record(record)
            if isinstance(record, TileDiffRecord):
                # No CompositeDiffRecord, so _push_terrain_unit_record skipped
                # the repaint -- and unlike a stroke, nothing repainted
                # incrementally on the way here. Same branch on_fill takes.
                self._apply_dirty(record.touched_indices())
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        # Logged even when nothing changed, same reasoning as on_fill's: a
        # silent no-op reads as a broken tool.
        self._log_status(
            f"{label}: {len(tiles)} tiles set to {name_for_terrain_id(terrain_id)}"
        )

    def _apply_terrain_unit_plan(self, tile_record: TileDiffRecord | None, mm) -> DiffRecord | None:
        """Given an unpushed TileDiffRecord from one terrain-paint gesture
        (Draw's stroke or Paint Can's one-shot fill), rolls Trees/Eye candy
        per descape/terrain_units.py's measured placement model and returns
        the record to push: `tile_record` unchanged if nothing was planned
        (both checkboxes off, no terrain actually changed, or this file's
        units are read-only), or a CompositeDiffRecord bundling it with a
        UnitDiffRecord so one Ctrl+Z undoes both -- exactly paste_region()'s
        own shape. `tile_record` may itself be None (a no-op stroke/fill);
        that passes straight through, matching build_stroke_record's own
        "nothing changed" convention.
        """
        if tile_record is None:
            return None
        trees = self.paint_trees_check.isChecked()
        doodads = self.paint_eye_candy_check.isChecked()
        if not trees and not doodads:
            return tile_record

        width = mm.map_width
        tiles_by_index = {i: (i % width, i // width) for i, _old, _new in tile_record.changes}
        enabled_consts: frozenset[int] = frozenset()
        if trees:
            enabled_consts |= terrain_units.TREE_CONSTS
        if doodads:
            enabled_consts |= terrain_units.DOODAD_CONSTS
        existing: dict[tuple[int, int], list] = {}
        for unit in self.scenario.unit_manager.units[GAIA_PLAYER_ID]:
            if unit.unit_const in enabled_consts:
                existing.setdefault((int(unit.x), int(unit.y)), []).append(unit)

        plan = terrain_units.plan_terrain_units(
            tile_record.changes, tiles_by_index, existing, trees=trees, doodads=doodads, rng=random.Random()
        )
        if not plan:
            return tile_record

        model = self._ensure_unit_edits()
        if model is None:
            self._log_status("Trees/eye candy skipped -- units are read-only for this file")
            return tile_record

        model.begin_unit_edit([GAIA_PLAYER_ID])
        try:
            if plan.removes:
                model.remove_many(plan.removes)
            if plan.adds:
                model.add_many(GAIA_PLAYER_ID, plan.adds)
        except Exception:
            model.abort_unit_edit()
            raise
        unit_record = model.commit_unit_edit(tile_record.label, self.edit_history, push=False)
        return CompositeDiffRecord(tile_record.label, children=[tile_record, unit_record])

    def _push_terrain_unit_record(self, record: DiffRecord | None) -> None:
        """Pushes what _apply_terrain_unit_plan() returned. A plain
        TileDiffRecord needs no extra repaint here -- Draw's own live
        per-tile _apply_dirty() during the drag (on_edit_stroke_tile) and
        Paint Can's own busy-guarded caller already cover the terrain --
        but a CompositeDiffRecord means units changed too, which needs the
        unit-cache invalidation/repaint _after_unit_mutation() does and
        which nothing upstream has done yet.
        """
        if record is None:
            return
        if isinstance(record, CompositeDiffRecord):
            self.edit_history.push_composite_record(record)
            self._apply_dirty(record.touched_indices())
            self._after_unit_mutation()
        else:
            self.edit_history.push_tile_record(record)

    # -- Convert brush (phase 3.5b's b2.5) --------------------------------
    #
    # Deliberately NOT built on edit_history.begin_stroke/commit_stroke like
    # the terrain strokes above: those diff TileState over a fixed-size
    # array, and a unit edit is a UnitEditModel record instead. The edit is
    # opened once at stroke start declaring ALL nine players (which ones a
    # drag will touch isn't known yet), each touched unit is reassigned
    # live, and the edit commits once at stroke end: one record per drag,
    # like every other brush tool.
    #
    # The repaint is coalesced by _convert_refresh_timer. The pick index is
    # deliberately left stale until stroke end: a full rebuild costs ~60ms on
    # the largest corpus file, and _convert_touched_tiles already guarantees
    # no tile is examined twice, so stale entries can't be reassigned twice.
    #
    # Each reassign queues a render_cache.UnitSplice, so a flush splices only
    # the converted units instead of rebuilding every unit-derived structure.

    CONVERT_REFRESH_MS = 100

    def _begin_convert_stroke(self) -> None:
        # A stroke that never reached _end_convert_stroke (none known) must
        # not leave its edit open, or the next unit edit raises.
        if self._convert_model is not None:
            self._convert_model.abort_unit_edit()
            self._convert_model = None
            # Its unflushed reassigns are still live in the model, so repaint them.
            if self._convert_pending_splices:
                self._after_unit_mutation()
        self._convert_pending_splices = []
        self._convert_done = set()
        self._convert_touched_tiles = set()
        # Read once and held for the whole drag rather than re-read per
        # touched tile: the combo can't actually change while the mouse is
        # captured, and holding it fixed keeps every reassignment in one
        # drag consistent even if that ever stopped being true.
        self._convert_destination = self.units_panel.owner_id()
        model = self._ensure_unit_edits()
        if model is None:
            return
        model.begin_unit_edit(list(range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1)))
        self._convert_model = model

    def _convert_stroke_tile(self, x: int, y: int) -> None:
        model = self._convert_model
        if model is None or self.scenario is None:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        mm = self.scenario.map_manager
        footprint = brush.brush_tiles(
            x, y, self.brush_size_spin.value(), self.brush_shape_combo.currentData(), mm.map_width, mm.map_height
        )
        sources = {pid for pid, action in self.convert_source_actions.items() if action.isChecked()}
        destination = self._convert_destination
        destination_units = self.scenario.unit_manager.units[destination]
        converted = False
        for tx, ty in footprint:
            if (tx, ty) in self._convert_touched_tiles:
                continue
            self._convert_touched_tiles.add((tx, ty))
            for order in index.by_tile.get((tx, ty), ()):
                entry = index.entries[order]
                key = unit_pick.unit_key(entry.player_id, entry.unit)
                # A multi-tile unit appears in several touched tiles' buckets.
                if key in self._convert_done:
                    continue
                if entry.player_id in sources and entry.player_id != destination:
                    old_own, old_tiles = self._unit_footprint(entry.unit)
                    model.reassign(entry.unit, destination)
                    # reassign() appends, so the index is the tail, not an O(n) _unit_list_index() scan.
                    self._convert_pending_splices.append(UnitSplice(
                        destination, len(destination_units) - 1, entry.unit,
                        old_own, old_own, old_tiles, old_tiles, old_player_id=entry.player_id,
                    ))
                    self._convert_done.add(key)
                    converted = True
        # Throttle, not debounce: restarting a running timer on every tile
        # would hold the repaint off for as long as the drag keeps moving.
        if converted and not self._convert_refresh_timer.isActive():
            self._convert_refresh_timer.start(self.CONVERT_REFRESH_MS)

    def _flush_convert_refresh(self) -> None:
        """The mid-stroke repaint: cache and canvas only, index left stale
        (see this section's header)."""
        if self._convert_model is not None and self._convert_pending_splices:
            pending, self._convert_pending_splices = self._convert_pending_splices, []
            self._after_unit_mutation(changed=pending, defer_index=True)

    def _end_convert_stroke(self) -> None:
        self._convert_refresh_timer.stop()
        model = self._convert_model
        self._convert_model = None
        count = len(self._convert_done)
        self._convert_done = set()
        self._convert_touched_tiles = set()
        pending, self._convert_pending_splices = self._convert_pending_splices, []
        if model is None:
            return
        if not count:
            model.abort_unit_edit()
            return
        label = "Convert unit" if count == 1 else f"Convert {count} units"
        model.commit_unit_edit(label, self.edit_history)
        # rebuild_index is load-bearing: the pick index was left stale all stroke.
        self._after_unit_mutation(changed=pending, rebuild_index=True)
        self._update_title()
        self._update_edit_actions()
        owner_text = "GAIA" if self._convert_destination == GAIA_PLAYER_ID else f"Player {self._convert_destination}"
        self._log_status(f"Converted {count} unit(s) to {owner_text}")

    # -- Region select/copy/paste (phase 2.8) -----------------------------

    def on_region_selected(self, region: tuple[int, int, int, int] | None) -> None:
        """The one handler for every way the committed region changes: a
        completed Select drag or an Escape-clear (both reported by MapView
        via the callback it was constructed with), and this window's own
        Select All/Deselect actions, which call it directly. Keeps
        self._region and MapView's own rendering copy in sync from either
        direction -- see MapView.set_region()'s docstring for the split."""
        self._region = region
        # Cleared unconditionally: this is the single sync point for every
        # committed-region change, and a selection that is no longer the
        # paste's own is not movable. paste_region()/on_region_move() re-arm
        # AFTER calling this, which is why their arming comes last.
        self._set_paste_move(None)
        self.map_view.set_region(region)
        self._update_tool_enabled()

    def select_all(self) -> None:
        if self.scenario is None:
            return
        mm = self.scenario.map_manager
        region = (0, 0, mm.map_width, mm.map_height)
        self.on_region_selected(region)
        self._log_status(f"Selected {mm.map_width}x{mm.map_height} region (whole map)")

    def deselect(self) -> None:
        if self.scenario is None:
            return
        self.on_region_selected(None)
        self._log_status("Deselected")

    @property
    def _region_clipboard(self):
        """The active clipboard entry's block, or None.

        Permanent, not a migration shim: every reader of this name -- Paste
        Region, _update_tool_enabled()'s gate, tools/verify_copy_paste.py,
        and three test modules -- means exactly "the active clipboard block",
        so the property is semantically honest. Read-only on purpose: with no
        setter, any leftover assignment raises AttributeError loudly at
        construction rather than silently shadowing the history."""
        return self._clipboard_history.active_block

    def copy_region(self) -> None:
        """Copy Region: snapshots the committed selection's terrain,
        elevation and units, and pushes it onto the clipboard history as the
        new active entry. No undo record -- nothing is mutated."""
        if self.scenario is None or self._region is None:
            return
        tx0, ty0, tx1, ty1 = self._region
        block = region_clipboard.copy_region(
            self.scenario.map_manager, self.scenario.unit_manager, tx0, ty0, tx1, ty1
        )
        self._clipboard_history.push(
            block, clipboard_history.thumbnail_rgb(block), clipboard_history.default_label(block)
        )
        # Every history mutation clears the move state, with no exceptions to
        # remember: a move replays its OWN snapshot, so leaving it armed would
        # still be correct, but dragging after the user has shifted clipboard
        # intent would move the PREVIOUSLY pasted content.
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        w, h = tx1 - tx0, ty1 - ty0
        self._log_status(
            f"Copied {w}x{h} region ({len(block.units)} units) "
            f"(entry {len(self._clipboard_history.entries)} of {clipboard_history.MAX_ENTRIES})"
        )

    def _clipboard_rows(self):
        return [
            (e.entry_id, e.label, e.block.width, e.block.height, len(e.block.units), e.thumbnail)
            for e in self._clipboard_history.entries
        ]

    def _show_clipboard_history(self) -> None:
        if self._clipboard_dialog is None:
            self._clipboard_dialog = ClipboardHistoryDialog(
                self,
                on_activate=self._on_clipboard_activate,
                on_delete=self._on_clipboard_delete,
                on_clear=self._on_clipboard_clear,
                on_rename=self._on_clipboard_rename,
            )
        self._refresh_clipboard_dialog()
        self._clipboard_dialog.show()
        self._clipboard_dialog.raise_()
        self._clipboard_dialog.activateWindow()

    def _refresh_clipboard_dialog(self) -> None:
        if self._clipboard_dialog is None:
            return
        self._clipboard_dialog.set_entries(
            self._clipboard_rows(), self._clipboard_history.active_id
        )

    def _history_rows(self):
        """One row per DiffRecord, oldest first. id(record) as the row id:
        the History window restores its selection by id because _push()'s
        redo truncation and overflow trim both shift every later index."""
        return [
            (id(record), record.label, ", ".join(sorted(record.kinds())))
            for record in self.edit_history.records
        ]

    def _show_history_dialog(self) -> None:
        if self._history_dialog is None:
            self._history_dialog = EditHistoryDialog(self, on_jump=self._on_history_jump)
        # show() before the refresh, not after: the refresh below deliberately
        # skips a hidden window, and Close only hides this one.
        self._history_dialog.show()
        self._history_dialog.raise_()
        self._history_dialog.activateWindow()
        self._refresh_history_dialog()

    def _refresh_history_dialog(self) -> None:
        """EditHistory.on_change's target -- fires on every history mutation,
        including during load and close, so it no-ops until the window exists
        and again once the user has closed it (close() only hides a QDialog,
        and rebuilding a few hundred hidden rows per stroke is pure waste)."""
        if self._history_dialog is None or not self._history_dialog.isVisible():
            return
        self._history_dialog.set_rows(
            self._history_rows(), self.edit_history.cursor, self.edit_history.saved_at_cursor
        )

    def _on_history_jump(self, target: int) -> None:
        """Undo or redo the whole span between the cursor and `target` as one
        gesture, then run undo/redo's own refresh tail.

        Refused rather than raised on a bad span: this is reached from a
        double-click inside a non-modal dialog, and a RuntimeError out of
        require_target() (a record whose model is gone) or a ValueError out of
        a stale row would otherwise surface as a QMessageBox behind the
        dialog."""
        if self.scenario is None:
            self._log_status("Jump: no scenario is open")
            return
        if self._busy or self._stroke_in_progress():
            self._log_status("Jump: an edit is still in progress")
            return
        start = self.edit_history.cursor
        try:
            kinds = self.edit_history.span_kinds(target)
            dirty = self.edit_history.jump_to(
                target,
                self.scenario.map_manager.terrain,
                self.trigger_edits,
                self.option_edits,
                self.unit_edits,
                self.message_edits,
            )
        except (RuntimeError, ValueError) as exc:
            self._log_status(f"Jump refused: {exc}")
            return
        if target == start:
            self._log_status("Already at that history entry")
            return
        self._refresh_after_history_move(kinds, dirty)
        steps = abs(target - start)
        direction = "back" if target < start else "forward"
        label = self.edit_history.records[target - 1].label if target > 0 else "Opened file"
        plural = "" if steps == 1 else "s"
        self._log_status(f"Jumped {direction} {steps} step{plural} to: {label}")

    def _on_clipboard_activate(self, entry_id: int) -> None:
        if not self._clipboard_history.set_active(entry_id):
            return
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        entry = self._clipboard_history.active
        self._log_status(f"Clipboard entry active: {entry.label}")

    def _on_clipboard_delete(self, entry_id: int) -> None:
        if not self._clipboard_history.remove(entry_id):
            return
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        self._log_status("Deleted clipboard entry")

    def _on_clipboard_clear(self) -> None:
        if not self._clipboard_history.entries:
            return
        self._clipboard_history.clear()
        self._set_paste_move(None)
        self._update_tool_enabled()
        self._refresh_clipboard_dialog()
        self._log_status("Cleared the clipboard history")

    def _on_clipboard_rename(self, entry_id: int, label: str) -> None:
        if not self._clipboard_history.rename(entry_id, label):
            return
        self._set_paste_move(None)
        self._refresh_clipboard_dialog()
        self._log_status(f"Renamed clipboard entry to {label}")

    def paste_region(self) -> None:
        """Paste Region: writes the clipboard's checked categories anchored
        on the hovered tile (falling back to the current region's own
        top-left with no hover), as ONE undo step regardless of how many
        categories are checked -- see edit_history.CompositeDiffRecord.
        Terrain+elevation ride a single TileDiffRecord (they share
        TileState); units ride a separate UnitDiffRecord built with
        push=False so it doesn't land on the stack until it's known how many
        other children there are."""
        block = self._clipboard_history.active_block
        if self.scenario is None or block is None:
            return
        if self._hover_tile is not None:
            tx0, ty0 = self._hover_tile
        elif self._region is not None:
            tx0, ty0 = self._region[0], self._region[1]
        else:
            self._log_status("Paste Region: no target tile -- hover the map or select a region first")
            return

        do_terrain = self.paste_terrain_check.isChecked()
        # Defensive re-check of the same gate the checkbox's own setEnabled()
        # already encodes (_update_tool_enabled): the checkbox could be
        # stale mid-call the same way the old tool-scoped gating's own
        # comment documented. set_tiles_elevation raises ValueError via
        # MapManager.get_tile on a non-square map. _paste_block_at() closes its
        # stroke on a raise, so it can't wedge later edits, but the paste would
        # still fail halfway.
        do_elevation = self.paste_elevation_check.isChecked() and self.scenario.map_is_square
        do_units = self.paste_units_check.isChecked()
        record, parts, units_skipped = self._paste_block_at(
            block, tx0, ty0, do_terrain, do_elevation, do_units
        )
        self.on_region_selected(self._pasted_region(block, tx0, ty0))
        # Arming comes LAST, after on_region_selected(), which clears the
        # move state unconditionally as the single sync point for every
        # committed-region change. A paste that pushed nothing (no category
        # checked, or a destination already identical) arms nothing: there is
        # no record to replace and so nothing to move.
        if record is not None:
            self._set_paste_move(
                region_clipboard.PasteMove(
                    block, (tx0, ty0), do_terrain, do_elevation, do_units, record
                )
            )

        label = ", ".join(parts) if parts else "nothing (no category checked)"
        if record is None and parts:
            self._log_status(f"Pasted {label} at ({tx0}, {ty0}) (no change)")
        else:
            self._log_status(f"Pasted {label} at ({tx0}, {ty0})")
        if units_skipped:
            self._log_status("Units were skipped -- units are read-only for this file")

    def _set_paste_move(self, state) -> None:
        """The ONE assignment path for _paste_move, so MapView's movable flag
        is pushed from the same place and can never disagree with it."""
        self._paste_move = state
        self.map_view.set_region_movable(state is not None)

    def _refresh_paste_move_validity(self) -> None:
        """One identity check, not a list of event hooks: the state is valid
        only while the paste's own record is still the next thing undo would
        take. That single condition already covers every way it should lapse --
        any other edit pushes a record on top, an undo or redo moves the cursor
        off it, and loading or closing a file resets the history entirely.

        Called from _update_edit_actions(), which is already the
        post-mutation and post-undo/redo sweep point. The two things this
        cannot see (a new selection, and a clipboard change) clear the state
        explicitly at their own call sites."""
        if self._paste_move is None:
            return
        if self.edit_history.peek_undo() is not self._paste_move.record:
            self._set_paste_move(None)

    def on_region_move(self, dx: int, dy: int) -> None:
        """Commit a region drag: undo the paste that armed this, then re-paste
        the same snapshot at the new anchor.

        Because the undo rewinds the cursor and the re-paste truncates at it
        (EditHistory._push opens with del records[cursor:]), the history ends
        with ONE record covering the paste and every move of it -- no pop, no
        in-place record mutation, no new coalescing API."""
        if self.scenario is None:
            return
        self._refresh_paste_move_validity()
        if self._paste_move is None:
            self._log_status("Move Region: the pasted region is no longer movable")
            return
        # Captured here, as this step's last act: _update_edit_actions() below
        # runs the validity sweep, which clears self._paste_move the instant
        # the undo moves peek_undo() off the paste record. Everything after
        # this line must read the local.
        state = self._paste_move

        self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo", quiet=True)

        # The old location is restored BEFORE the new paste is written, so a
        # short drag (where the two footprints overlap heavily) reads restored
        # terrain when set_tiles_elevation propagates its skirt.
        tx0, ty0 = state.anchor[0] + dx, state.anchor[1] + dy
        block = state.block
        record, _parts, _units_skipped = self._paste_block_at(
            block, tx0, ty0, state.do_terrain, state.do_elevation, state.do_units
        )
        if record is None:
            # The destination already matched the block byte for byte. Leave
            # it un-armed rather than re-arming on nothing to move to. This
            # strands a redo entry: records[cursor - 1] still holds the old
            # paste, so Ctrl+Z here undoes whatever preceded that paste and
            # Ctrl+Y brings it back. Coherent, and accepted as-is.
            self._log_status(f"Moved pasted region to ({tx0}, {ty0}) (no change)")
            return
        self.on_region_selected(self._pasted_region(block, tx0, ty0))
        self._set_paste_move(
            region_clipboard.PasteMove(
                block, (tx0, ty0), state.do_terrain, state.do_elevation, state.do_units, record
            )
        )
        self._log_status(f"Moved pasted region to ({tx0}, {ty0})")

    def _pasted_region(self, block, tx0: int, ty0: int) -> tuple[int, int, int, int]:
        """Where the selection goes after a block lands at (tx0, ty0), so the
        user sees the result and a second paste is idempotent. Clamped to the
        map like _clipped_bounds() -- an off-map anchor or an oversized block
        would otherwise hand MapView a region reaching past the elevation
        grid's edge and crash in _tile_polygon."""
        mm = self.scenario.map_manager
        return (
            max(0, tx0),
            max(0, ty0),
            min(mm.map_width, tx0 + block.width),
            min(mm.map_height, ty0 + block.height),
        )

    def _paste_block_at(
        self, block, tx0: int, ty0: int, do_terrain: bool, do_elevation: bool, do_units: bool
    ):
        """The write half of Paste Region, with the anchor and the three
        category flags passed in rather than read from _hover_tile and the
        checkboxes -- so Paste Region and a region MOVE share exactly one
        paste write path.

        Returns (pushed_record, parts, units_skipped); pushed_record is None
        when nothing changed, which is what tells the caller there is no
        record to replace and therefore nothing to move.

        One undo step regardless of how many categories are checked: terrain
        and elevation ride a single TileDiffRecord (they share TileState),
        units ride a separate UnitDiffRecord built with push=False so it
        doesn't land on the stack until it's known how many other children
        there are -- see edit_history.CompositeDiffRecord."""
        mm = self.scenario.map_manager
        children: list = []
        parts: list[str] = []
        units_skipped = False

        if do_terrain or do_elevation:
            self.edit_history.begin_stroke(mm.terrain)
            with self._close_stroke_on_error("Paste Region"):
                if do_terrain:
                    region_clipboard.paste_terrain(mm, block, tx0, ty0)
                    parts.append("terrain")
                if do_elevation:
                    targets = region_clipboard.elevation_targets(block, tx0, ty0, mm.map_width, mm.map_height)
                    set_tiles_elevation(mm, targets)
                    parts.append("elevation")
            tile_record = self.edit_history.build_stroke_record("Paste Region", mm.terrain)
            if tile_record is not None:
                children.append(tile_record)

        if do_units and block.units:
            unit_edits = self._ensure_unit_edits()
            if unit_edits is None:
                units_skipped = True
            else:
                targets = region_clipboard.unit_paste_targets(block, tx0, ty0, mm.map_width, mm.map_height)
                if targets:
                    owners = {t.unit.player for t in targets}
                    unit_edits.begin_unit_edit(owners)
                    new_ids: dict[int, int] = {}
                    for t in targets:
                        u = t.unit
                        added = unit_edits.add(
                            player=u.player,
                            unit_const=u.unit_const,
                            x=t.x,
                            y=t.y,
                            z=u.z,
                            rotation=u.rotation,  # verbatim -- never transformed
                            status=u.status,
                            initial_animation_frame=u.initial_animation_frame,
                            garrisoned_in_id=new_ids.get(t.holder_slot, -1),
                            caption_string_id=u.caption_string_id,
                            caption_string=u.caption_string,
                        )
                        new_ids[t.slot] = added.reference_id
                    unit_record = unit_edits.commit_unit_edit("Paste Region", self.edit_history, push=False)
                    children.append(unit_record)
                parts.append("units")

        pushed = None
        dirty: list[int] = []
        if len(children) == 1:
            pushed = children[0]
            if isinstance(pushed, TileDiffRecord):
                self.edit_history.push_tile_record(pushed)
            else:
                self.edit_history.push_unit_record(pushed)
            dirty = pushed.touched_indices()
        elif len(children) > 1:
            pushed = CompositeDiffRecord("Paste Region", children=children)
            self.edit_history.push_composite_record(pushed)
            dirty = pushed.touched_indices()

        if children:
            self._apply_dirty(dirty)
            if any(isinstance(c, UnitDiffRecord) for c in children):
                self._after_unit_mutation()
        self._update_edit_actions()
        self._update_title()
        return pushed, parts, units_skipped

    def on_fill(self, x: int, y: int, modifiers) -> None:
        """Paint Can: one flood fill per left click -- MapView routes
        CLICK_TOOLS here directly (see mousePressEvent), never through the
        stroke handlers above. Shaped like paste_region() just above, not like
        on_edit_stroke_tile(): begin_stroke/build_stroke_record once at the
        end (via _apply_terrain_unit_plan()) rather than
        begin_stroke/stroke_dirty_indices/commit_stroke, since there's no
        drag to give live feedback during and stroke_dirty_indices() is an
        O(map) scan per call -- fine once per touched brush tile, far too
        slow once per filled tile. `modifiers` is accepted only for
        signature symmetry with on_edit_stroke_tile/on_click_edit and is
        deliberately ignored -- a fill has no Shift/right-button inverse.

        Wraps the whole operation in the same wait-cursor/setEnabled(False)/
        self._busy pattern load_scenario()/refresh_map()/_render_current()
        already use: tools/bench_fill_latency.py measured a full-map fill on
        a 480x480 map at ~540ms (Flat) to ~2.4s (Stepped, before
        STEPPED_FULL_RERENDER_THRESHOLD's fallback) end to end -- long
        enough that the window would otherwise sit frozen with no visible
        indication anything is happening, and that a second click dispatched
        via processEvents() mid-fill could reenter this method while
        self.scenario is still being mutated. Applied unconditionally rather
        than only above some measured size, matching every other guarded
        method in this class -- the guard is cheap for the common small-fill
        case too."""
        if self.scenario is None or self._busy:
            return
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        terrain_id = self.terrain_panel.terrain_id()

        # Large-operation guard (terrain_units plan, §8): sized BEFORE
        # anything is written, so Cancel leaves the map untouched. Paint Can
        # only -- Draw's counts are bounded by brush size x drag length (see
        # TERRAIN_UNIT_CONFIRM_THRESHOLD's own comment), and skipped
        # entirely when neither checkbox is on, since a terrain-only fill
        # has no per-tile unit cost to warn about.
        if self.paint_trees_check.isChecked() or self.paint_eye_candy_check.isChecked():
            region_size = len(contiguous_region(mm, x, y))
            if region_size > TERRAIN_UNIT_CONFIRM_THRESHOLD:
                reply = QMessageBox.question(
                    self,
                    "Large fill",
                    f"This fill covers {region_size} tiles and can place a large number of "
                    f"trees/eye candy units, which may take a while. Continue?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if reply != QMessageBox.Yes:
                    return

        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            self.edit_history.begin_stroke(mm.terrain)
            flood_fill_terrain(mm, x, y, terrain_id)
            tile_record = self.edit_history.build_stroke_record(_STROKE_LABELS["fill"], mm.terrain)
            record = self._apply_terrain_unit_plan(tile_record, mm)
            self._push_terrain_unit_record(record)
            dirty = record.touched_indices() if record is not None else []
            if isinstance(record, TileDiffRecord):
                # No CompositeDiffRecord (no tree/doodad change) -- the
                # terrain repaint _push_terrain_unit_record() skips still
                # has to happen once, here, since Paint Can (unlike Draw)
                # never repaints incrementally during the fill itself.
                self._apply_dirty(dirty)
            elapsed = time.perf_counter() - t0
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        name = name_for_terrain_id(terrain_id)
        # Always log, including the no-op case, same reasoning as
        # paste_region()'s own comment above -- a silent no-op reads as a
        # broken keybind, and a click that can rewrite the whole map
        # deserves a record either way.
        if dirty:
            self._log_status(f"Filled {len(dirty)} tiles with {name} from ({x}, {y}) applied in {elapsed:.2f}s")
        else:
            self._log_status(f"Fill at ({x}, {y}): already {name} (no change)")

    def on_click_edit(self, x: int, y: int, modifiers) -> None:
        """MapView's CLICK_TOOLS destination -- dispatches to whichever
        one-shot tool is actually active, since Paint Can (on_fill) used to
        be the only member and was wired here directly. Eyedropper is the
        second."""
        if self._current_tool == "eyedropper":
            self.pick_tile_value(x, y, modifiers)
        elif self._current_tool == "create_objects":
            self.stamp_create_objects(x, y)
        else:
            self.on_fill(x, y, modifiers)

    def pick_tile_value(self, x: int, y: int, modifiers) -> None:
        """Eyedropper: reads a tile's terrain into the sidebar picker and its
        elevation into the Level param, no undo record since nothing is mutated.
        `modifiers` is accepted only for signature symmetry with on_fill/
        on_edit_stroke_tile and is deliberately ignored.

        Indexes mm.terrain directly rather than calling mm.get_tile(), which
        raises on any non-square map (fill_tools.py documents this) -- a
        pick must work on every write_ok map, squareness included or not."""
        if self.scenario is None:
            return
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        tile = mm.terrain[y * mm.map_width + x]
        terrain_id = tile.terrain_id
        elevation = tile.elevation

        if self.terrain_panel.set_terrain(terrain_id):
            terrain_part = name_for_terrain_id(terrain_id)
        else:
            terrain_part = f"{name_for_terrain_id(terrain_id)} (not in the terrain picker, unchanged)"

        if 0 <= elevation <= ELEVATION_LEVEL_MAX:
            self.elevation_level_spin.setValue(elevation)
            elevation_part = f"elevation {elevation}"
        else:
            elevation_part = f"elevation {elevation} (out of range, unchanged)"

        self._log_status(f"Picked {terrain_part}, {elevation_part} from ({x}, {y})")

    def on_mirror(self, plan, unit_plan=None) -> list[int] | None:
        """Map mirroring's apply path (mirror_tools.plan_mirror -> here),
        called by MirrorDialog for both Preview and Apply. Copies on_fill()'s
        busy-guard/single-history-record/incremental-repaint shape near-
        verbatim -- see that method's own docstring for why (no stroke, and
        stroke_dirty_indices() is an O(map) scan per call).

        Returns None -- refusing to apply, no record pushed -- when
        plan.elevation_violations is non-empty: that check exists precisely
        because an illegal ±1 jump crashes AoE2:DE at load time
        (scenario_write.py), and per the plan there is no safe auto-repair
        (any repair would break the symmetry the mirror was asked to
        produce). Otherwise returns the list of dirty tile indices, same
        shape as on_fill() -- empty if the map was already symmetric under
        the chosen mode/slice, which is a genuine no-op, not a refusal.

        `plan.changes` is already fully computed with no possibility of
        raising, so the mutator handed to the history here is a bare
        assignment loop, per plan_mirror's own "everything computed before
        any mutation" contract.

        `unit_plan` (Stage 2) is mirror_tools.plan_mirror_units()' output or
        None. When present, its removals and images ride a UnitDiffRecord
        built with push=False, and the two records go on the stack as one
        CompositeDiffRecord -- Paste Region's own shape, so one Ctrl+Z undoes
        terrain, elevation and units together. A blocked unit plan refuses
        the whole operation, terrain included, for the same reason an
        elevation seam violation does: a partial mirror is not what was
        asked for."""
        if self.scenario is None or self._busy:
            return None
        if unit_plan is not None and unit_plan.blocked:
            self._log_status(self._mirror_unit_refusal(unit_plan))
            return None
        if plan.elevation_violations:
            width = self.scenario.map_manager.map_width
            sample = ", ".join(
                f"({i % width}, {i // width})-({j % width}, {j // width})"
                for i, j in plan.elevation_violations[:5]
            )
            self._log_status(
                f"Mirror refused: {len(plan.elevation_violations)} elevation seam "
                f"violation(s) would exceed the +/-1 limit (e.g. {sample}) -- pick a "
                f"different mode, or flatten the source region's edges first"
            )
            return None

        mm = self.scenario.map_manager
        changes = plan.changes

        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            children: list = []
            # The unit half goes FIRST, before a single tile is touched: it is
            # the half that can still raise (remove_many refuses a garrisoned
            # unit), and it rolls itself back through abort_unit_edit(). Doing
            # it after the tile loop would leave the map mirrored with an
            # unpushed record and no undo.
            unit_record = self._apply_mirror_units(unit_plan)
            # begin_stroke/build_stroke_record rather than apply(): the tile
            # record has to stay unpushed until it is known whether a unit
            # record rides with it.
            self.edit_history.begin_stroke(mm.terrain)
            for idx, (terrain_id, elevation, layer) in changes:
                tile = mm.terrain[idx]
                tile.terrain_id, tile.elevation, tile.layer = terrain_id, elevation, layer
            tile_record = self.edit_history.build_stroke_record("Mirror Map", mm.terrain)
            if tile_record is not None:
                children.append(tile_record)
            if unit_record is not None:
                children.append(unit_record)

            if len(children) == 1:
                record = children[0]
                if isinstance(record, TileDiffRecord):
                    self.edit_history.push_tile_record(record)
                else:
                    self.edit_history.push_unit_record(record)
                dirty = record.touched_indices()
            elif len(children) > 1:
                composite = CompositeDiffRecord("Mirror Map", children=children)
                self.edit_history.push_composite_record(composite)
                dirty = composite.touched_indices()
            else:
                dirty = []
            self._apply_dirty(dirty)
            if unit_record is not None:
                self._after_unit_mutation()
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        if dirty:
            units = "" if unit_plan is None else (
                f", {len(unit_plan.removals)} unit(s) replaced by {len(unit_plan.images)}"
            )
            self._log_status(f"Mirrored {len(dirty)} tiles{units}")
        else:
            self._log_status("Mirror: already symmetric under this mode (no change)")
        return dirty

    def _mirror_unit_refusal(self, unit_plan) -> str:
        """Why a unit plan blocks. Names the first few offenders by tile, the
        same shape on_mirror's elevation-seam message uses."""
        reasons = []
        if unit_plan.straddling:
            sample = ", ".join(f"({int(u.x)}, {int(u.y)})" for u in unit_plan.straddling[:5])
            reasons.append(
                f"{len(unit_plan.straddling)} unit(s) straddle the symmetry axis, so their "
                f"image would overlap the original (e.g. {sample})"
            )
        if unit_plan.garrisoned_blockers:
            sample = ", ".join(f"({int(u.x)}, {int(u.y)})" for u in unit_plan.garrisoned_blockers[:5])
            reasons.append(
                f"{len(unit_plan.garrisoned_blockers)} unit(s) outside the source slice hold a "
                f"garrison and cannot be replaced (e.g. {sample})"
            )
        return (
            "Mirror refused: " + "; ".join(reasons)
            + " -- pick a different mode, or move the offending objects off the seam"
        )

    def _apply_mirror_units(self, unit_plan):
        """Removes the destination slices' units and adds the source's images,
        as one unpushed UnitDiffRecord. None when there is nothing to do.

        All nine players are declared up front, following the Convert brush:
        ownership rotation can move a unit to any of them, and the snapshot
        has to cover every list the edit touches.
        """
        if unit_plan is None or not (unit_plan.removals or unit_plan.images):
            return None
        model = self._ensure_unit_edits()
        if model is None:
            return None
        model.begin_unit_edit(range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1))
        try:
            if unit_plan.removals:
                model.remove_many(unit_plan.removals)
            for image in unit_plan.images:
                source = image.source
                model.add(
                    player=image.player,
                    unit_const=image.unit_const,
                    x=image.x,
                    y=image.y,
                    z=source.z,
                    rotation=image.rotation,
                    status=source.status,
                    initial_animation_frame=source.initial_animation_frame,
                    garrisoned_in_id=-1,  # a copy never inherits a garrison link
                    caption_string_id=source.caption_string_id,
                    caption_string=source.caption_string,
                )
        except Exception:
            model.abort_unit_edit()
            raise
        return model.commit_unit_edit("Mirror Map", self.edit_history, push=False)

    def _apply_dirty(self, dirty_indices) -> None:
        """Repaints exactly the given tile indices -- the incremental path
        used by strokes (above) and undo/redo, instead of a full
        refresh_map() (which reallocates and recomposites the entire map;
        fine once per file open, far too slow per brush touch).

        Stepped mode (Phase B-C, on top of Phase 4/5's terrain+unit
        interleaving): dirty_screen_bbox_iso() plays refresh_region_iso()'s
        old "half 1" role -- mutates self._iso_elevations in place and
        returns the invalidated bbox, without needing an img array at all
        (see descape.render's own docstring for that split, Phase B-B) --
        then self._cache.patch(bbox) recomposites just the touched chunks
        and self.map_view.invalidate_region(bbox) schedules the actual Qt
        repaint, which pulls the freshly-patched pixels straight from the
        cache. dirty_screen_bbox_iso() can decline (return None) if an edit
        pushed a tile's elevation outside the range self._iso_proj was
        sized for -- see that function's own docstring for why this is a
        defensive fallback, not an expected path given viewer.py's own
        elevation tools already clamp to that same range. Falling back to a
        full _render_current() keeps correctness even if that assumption is
        ever violated (e.g. a hand-edited file loaded with an elevation
        already out of range), at the cost of one full re-render instead of
        an instant patch for that one stroke tile.

        Sloped mode (Track C4's Step 3): structurally the same two-part path
        as Stepped, through dirty_screen_bbox_sloped() instead. The one thing
        that differs is invisible from here and lives on the cache: patch()
        re-derives corner_rise from the elevations the bbox call just mutated,
        so the height field the patch composites against is the post-edit one.

        Every Sloped edit route lands in that branch: the four Terrain tools
        (released by Track C4's Step 4), Paint Can via on_fill(), Paste via
        paste_region(), and Undo/Redo. Undo is worth naming because it was
        reachable in Sloped BEFORE Step 4 -- _update_tool_enabled()'s gate
        never covered it, since _update_edit_actions() enables it from
        edit_history alone -- so "edit in Stepped, switch to Sloped, Ctrl+Z"
        hit this branch while every tool was still greyed out. Before the
        branch existed that undo fell through to the Flat branch below --
        whose assert passed, since Sloped carried no snapshot -- and patched
        axis-aligned tile squares onto an isometric canvas. See
        tests/test_sloped_edit.py's checks 7 and 8.

        Flat mode (Phase B-E): no elevation term and no fallback case --
        the dirty rect is simply the union of the dirty tiles' own pixel
        squares. Unlike Stepped, there's no footprint-expansion step here
        either: composite_rect_flat() redraws terrain AND every overlapping
        unit in full stacking order within whatever rect it's given (see
        that function's own docstring for why that makes the old
        refresh_units_over() fixed-point dirty-tile expansion unnecessary).
        Patches per-tile rects, not one union bbox -- a union over a
        scattered undo set could span the whole map, turning patch() into a
        full recomposite.

        Re-arms the level warm at its tail (Batch B step B1). Every branch
        below cancels the warms first, and _start_level_warm() had exactly one
        caller (load_scenario), so before this the first stroke, paste or undo
        of a session left the neighbour mips cold for the rest of it, and the
        next zoom to one paid its 0.6-4.3s level build inside paint(). The
        body lives in _apply_dirty_render() so the re-arm covers its early
        returns too (both full-rerender thresholds, both out-of-range bbox
        fallbacks, the no-cache guards), not just the patch path. An exception
        propagates past the re-arm, so no warm ever starts against a
        half-patched cache."""
        if not dirty_indices or self.scenario is None:
            return
        self._apply_dirty_render(dirty_indices)
        # Re-arms what the body's _cancel_warms() dropped. Kept below the
        # guard above, whose early return cancelled nothing to re-arm.
        self._start_level_warm()

    def _apply_dirty_render(self, dirty_indices) -> None:
        """_apply_dirty's body, split out so that every early return below
        still reaches that method's tail re-arm of the level warm."""
        # Before any of the elevation-array mutation below (and before
        # patch(), which rebuilds source caches off it) -- see
        # _cancel_warms's docstring. One call here covers every stroke,
        # Paint Can, Paste and Undo/Redo route into the three style branches.
        self._cancel_warms()
        mm = self.scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        if self._render_style == "stepped":
            if self._cache is None:
                return
            # Above STEPPED_FULL_RERENDER_THRESHOLD dirty tiles (a Paint Can
            # fill spanning a large fraction of the map, measured with
            # tools/bench_fill_latency.py -- see that constant's own
            # comment), dirty_screen_bbox_iso's per-tile dilation is slower
            # than just re-rendering. Skips the dilation call entirely
            # rather than running it and discarding the result -- the
            # dilation cost is exactly what this branch exists to avoid.
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Elevation view: {len(dirty_indices)} tiles dirty, "
                    f"re-rendered full map instead of patching (prepared in {elapsed:.2f}s)"
                )
                return
            # with_sprites MUST be the live cache's own flag, never a
            # constant: the widening this asks for and the sprites _level()
            # actually draws have to agree, or an elevation stroke beside a
            # large building leaves a stale sprite fragment behind (P3-g5).
            # tests/test_sprite_toggle_viewer.py pins this exact argument.
            #
            # elevation_changed (draw-perf plan Step 3): populated in place,
            # then threaded into patch() so a terrain-paint-only edit skips
            # rebuilding units_by_tile/building_bboxes/sprites entirely --
            # see IsoChunkCache._refresh_source_caches's own docstring.
            elevation_changed: set = set()
            with perf_trace.phase("bbox"):
                bbox = dirty_screen_bbox_iso(
                    self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                    with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                    flatten_elevations=(self._terrain_style == "flat"),
                )
            if bbox is None:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Elevation view: edit exceeded the cached elevation range, "
                    f"re-rendered full map (prepared in {elapsed:.2f}s)"
                )
            else:
                with perf_trace.phase("patch"):
                    self._cache.patch(bbox, elevation_changed=elevation_changed)
                # The draped grid reads the elevations the bbox call above
                # mutated. Only on an edit that moved one: a terrain-only
                # stroke cannot move a grid line. Qt coalesces the resulting
                # update() calls, so a stroke needs no debounce of its own.
                if elevation_changed:
                    self.map_view.refresh_elevation_overlays()
                with perf_trace.phase("invalidate"):
                    self.map_view.invalidate_region(bbox)
        elif self._render_style == "sloped":
            if self._cache is None:
                return
            # Same two-part shape as Stepped, and the same threshold: the
            # dilation this guards is a per-tile Python set walk, so its cost
            # doesn't depend on the terrain style at all, and Sloped's own
            # _render_current() fallback is only ~1.8-2.0x Stepped's
            # (tools/bench_sloped_patch.py) -- so 20,000 stays conservative
            # here rather than needing its own measured constant.
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Sloped view: {len(dirty_indices)} tiles dirty, "
                    f"re-rendered full map instead of patching (prepared in {elapsed:.2f}s)"
                )
                return
            # with_sprites MUST be the live cache's own flag, same load-bearing
            # warning the Stepped branch above carries -- tests/
            # test_sprite_toggle_viewer.py pins this exact argument too.
            #
            # elevation_changed: same draw-perf plan Step 3 wiring as the
            # Stepped branch above -- see SlopedChunkCache._refresh_source_
            # caches's own docstring for why this class uses one gate for
            # corner_rise/building_bboxes/sprites together, not the narrower
            # per-unit one IsoChunkCache uses.
            elevation_changed: set = set()
            with perf_trace.phase("bbox"):
                bbox = dirty_screen_bbox_sloped(
                    self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                    with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                )
            if bbox is None:
                elapsed, _ = self._render_current(reset_view=False)
                self._log_status(
                    f"Sloped view: edit exceeded the cached elevation range, "
                    f"re-rendered full map (prepared in {elapsed:.2f}s)"
                )
            else:
                # patch() rebuilds corner_rise from the elevations the call
                # above just mutated, before compositing anything -- see
                # SlopedChunkCache._refresh_source_caches(). It REBINDS the
                # array, so the grid overlay has to be handed the new one.
                with perf_trace.phase("patch"):
                    self._cache.patch(bbox, elevation_changed=elevation_changed)
                if elevation_changed:
                    self.map_view.refresh_elevation_overlays()
                with perf_trace.phase("invalidate"):
                    self.map_view.invalidate_region(bbox)
        else:
            assert self._render_style == "flat" and self._iso_elevations is None, (
                "Flat (non-isometric) mode must never carry an elevation snapshot"
            )
            if self._cache is None:
                return
            with perf_trace.phase("bbox"):
                coords = [(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices]
                rects = [(x * tile_px, y * tile_px, (x + 1) * tile_px, (y + 1) * tile_px) for x, y in coords]
            with perf_trace.phase("patch"):
                self._cache.patch_rects(rects)
            with perf_trace.phase("invalidate"):
                for rect in rects:
                    self.map_view.invalidate_region(rect)

    # -- trigger editing ----------------------------------------------------
    #
    # Every trigger mutation in the app goes through _trigger_edit(). The model
    # contract behind it (trigger_model.py) fails silently in every direction:
    # an edit that skips mark_dirty() splices its pre-edit bytes back on save,
    # and a mutation that sets model dirtiness without pushing a record closes
    # the document with no save prompt. Routing every call site through one
    # helper is what keeps those from being per-call-site mistakes.

    def _ensure_trigger_edits(self) -> TriggerEditModel | None:
        """The document's TriggerEditModel, built on first use.

        Construction is deferred to the first real edit, not done at panel
        populate: that is what keeps a browse-only session's save
        byte-identical. A file whose triggers parsed can still refuse to be
        edited (the alignment gate), so this can return None after a
        successful parse.
        """
        if self.trigger_edits is not None:
            return self.trigger_edits
        if self.scenario is None:
            return None
        try:
            self.trigger_edits = TriggerEditModel(self.scenario)
        except TriggerEditsUnavailableError as e:
            self._log_status(f"Triggers are read-only for this file: {e}")
            QMessageBox.warning(self, "Triggers are read-only", str(e))
            return None
        return self.trigger_edits

    @contextmanager
    def _trigger_edit(
        self,
        model: TriggerEditModel,
        label: str,
        content_touched: Sequence[int] = (),
        refresh: str = "",
        refresh_arg=None,
    ):
        """Open one recorded trigger edit, and close it whatever happens.

        `content_touched` names every trigger this edit may rewrite in place,
        in pre-edit numbering. Signature diffing cannot see that class of
        change, so an undeclared one is spliced away on save.

        `refresh` is how much of the panel the edit invalidates: "" for a field
        edit, which patches its own row; "entries" for a condition or effect
        added or removed; "order" for a display-order-only move, which patches
        the moved row's position and nothing else (`refresh_arg` carries the
        (trigger_index, delta) that move needs); "variables" for a variable
        added or removed, which no tree in the panel shows at all; "panel" for
        anything that changes the trigger list itself.

        Enter only once a real value change is confirmed. commit_trigger_edit()
        pushes unconditionally by design, so entering on a focus-out, on a
        programmatic widget populate, or on merely opening the picker would
        record a phantom undo step and dirty the trigger.
        """
        model.begin_trigger_edit(content_touched)
        error: Exception | None = None
        try:
            yield model
        except Exception as e:  # noqa: BLE001 -- reported below, never swallowed
            error = e
        finally:
            # Commits even on failure: a raised structural_edit() is not a
            # no-op, it marks every blob dirty, so that state still needs a
            # record to undo.
            model.commit_trigger_edit(label, self.edit_history)
            self._after_trigger_edit(label, error, refresh, refresh_arg)

    def _after_trigger_edit(
        self, label: str, error: Exception | None, refresh: str = "", refresh_arg=None
    ) -> None:
        if refresh and self.mode == "triggers":
            if refresh == "panel":
                # Rebuilt rather than patched, for the same reason
                # _move_history rebuilds on an undo: a structural edit can
                # change membership, order and ids at once, and there is no
                # partial update that is cheaper to get right than a
                # repopulate.
                self._show_triggers()
            elif refresh == "order" and error is None:
                # The one tier that is neither a full repopulate nor a no-op:
                # a display-order move changes nothing but one row's position,
                # and Move Up/Down is a repeated click -- "panel"'s full
                # rebuild would scroll a 590-trigger list to the top on every
                # press, which is the exact slowness DEscape exists to escape.
                #
                # Only on success. move_row() blindly swaps two rows without
                # reading the model, unlike show_scenario() which re-reads it
                # regardless -- so on a raised structural_edit(), where the
                # display order was already rolled back to its pre-edit value
                # (structural_edit()'s except branch), patching the tree here
                # would show the move as if it succeeded while the model (and
                # the next save) disagree. Falls back to the truthful full
                # rebuild instead, same as every other failed structural edit.
                self.trigger_panel.move_row(*refresh_arg)
            elif refresh == "order":
                self._show_triggers()
            elif refresh == "variables":
                # Neither tree in the panel shows a variable, so a full
                # repopulate would be pure cost. Only the dialog is stale, and
                # this is a no-op when it is not open.
                self.trigger_panel.refresh_variables()
            else:
                self.trigger_panel.refresh_entries()
        # Every tier, field edits included: a typed area_x2 must move the outline.
        self._rederive_trigger_overlay()
        # A player field edit changes page 0's trigger counts; a memoized re-read, 0-3ms.
        self._update_player_stats()
        self._update_title()
        self._update_edit_actions()
        if error is None:
            self._log_status(label)
            return
        # A failed structural edit leaves the document correct and saveable but
        # without its minimal-diff guarantee, so it must not read as a clean
        # no-op.
        self._log_status(f"{label} failed: {type(error).__name__}: {error}")
        QMessageBox.warning(
            self,
            "Trigger edit failed",
            f"{error}\n\nThe document is still correct and can be saved, but this file's "
            f"triggers will now be rewritten in full rather than spliced.",
        )

    def set_trigger_field(self, index: int, spec: trigger_fields.FieldSpec, value) -> None:
        """Write one of a trigger's own fields. The single funnel every trigger
        field widget reports through, so the contract lives in one place."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        with self._trigger_edit(model, f"Set trigger {spec.label}", content_touched=[index]) as m:
            setattr(m.manager().triggers[index], spec.name, value)

    def _tag_carriers(self, model: TriggerEditModel, tag: str) -> list[int]:
        return [
            index
            for index, trigger in enumerate(model.manager().triggers)
            if trigger_organize.parse_tag(trigger.name or "") == tag
        ]

    def _rename_carriers(self, model: TriggerEditModel, label: str, renamed: dict[int, str]) -> None:
        """Assign every precomputed name in one recorded edit. Each carrier is
        declared in content_touched, so exactly those blobs re-serialize."""
        with self._trigger_edit(model, label, content_touched=list(renamed), refresh="panel") as m:
            triggers = m.manager().triggers
            for index, name in renamed.items():
                triggers[index].name = name

    def rename_trigger_tag(self, old: str, new: str) -> None:
        """Rename tag `old` to `new` on every trigger carrying it, as one undo
        step. Onto an existing tag this merges the two facets; the panel asks
        first. A no-op (empty, unchanged, stale tag) records nothing."""
        new = new.strip()
        if not new or new == old:
            return
        model = self._ensure_trigger_edits()
        if model is None:
            return
        carriers = self._tag_carriers(model, old)
        if not carriers:
            return
        triggers = model.manager().triggers
        renamed = {index: trigger_organize.retag_name(triggers[index].name, new) for index in carriers}
        # Validated before the edit opens: commit_trigger_edit() pushes
        # unconditionally, so a refusal inside it would still record a step.
        for index, name in renamed.items():
            if trigger_organize.parse_tag(name) != new:
                original = triggers[index].name
                prefix, _, suffix = trigger_organize.split_tag(original)
                opener, closer = prefix.strip(), suffix.lstrip()[0]
                reason = (
                    f'it contains "{closer}", which ends a {opener}{closer} tag'
                    if closer in new
                    else "it would not read back as that tag"
                )
                QMessageBox.warning(
                    self,
                    "Cannot rename tag",
                    f'Tag "{old}" cannot be renamed to "{new}": {reason} in "{original}".',
                )
                return
        follow = self.mode == "triggers" and self.trigger_panel.current_tag() == old
        count = len(carriers)
        label = f'Rename tag "{old}" to "{new}" ({count} trigger{"s" if count != 1 else ""})'
        self._rename_carriers(model, label, renamed)
        if follow:
            # After the rebuild: the combo's restore-by-data fell back to "All
            # tags" because `old` no longer exists.
            self.trigger_panel.set_tag_filter(new)

    def remove_trigger_tag(self, tag: str) -> None:
        """Strip tag `tag` (and the whitespace after it) from every trigger
        carrying it, as one undo step. A stale tag records nothing."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        carriers = self._tag_carriers(model, tag)
        if not carriers:
            return
        triggers = model.manager().triggers
        renamed = {index: trigger_organize.strip_tag(triggers[index].name) for index in carriers}
        count = len(carriers)
        label = f'Remove tag "{tag}" ({count} trigger{"s" if count != 1 else ""})'
        self._rename_carriers(model, label, renamed)

    def set_entry_field(
        self, index: int, kind: str, entry_index: int, spec: trigger_fields.FieldSpec, value
    ) -> None:
        """Write one field of one condition or effect: set_entry_fields()'s
        N = 1 call."""
        self.set_entry_fields(index, [(kind, entry_index)], spec, value)

    def set_entry_fields(
        self, index: int, refs: Sequence[tuple[str, int]], spec: trigger_fields.FieldSpec, value
    ) -> None:
        """Write one field across conditions and effects of one trigger, as
        one undo record (GH #60). The second and last funnel the panel reports
        edits through.

        Only entries not already holding `value` are written, and that set is
        resolved before the bracket opens: entering it on a no-op would record
        a phantom step, and a redundant setattr still runs Effect.quantity's
        armour/attack split.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        if not 0 <= index < len(manager.triggers):
            return
        trigger = manager.triggers[index]
        targets = []
        for kind, entry_index in refs:
            siblings = _entries_of(trigger, kind)
            if not 0 <= entry_index < len(siblings):
                continue
            try:
                differs = getattr(siblings[entry_index], spec.attribute) != value
            except Exception:  # noqa: BLE001 -- a version-gated read; write as before
                differs = True
            if differs:
                targets.append((kind, entry_index))
        if not targets:
            return
        if len(targets) == 1:
            label = f"Set {targets[0][0]} {spec.label}"
        else:
            label = f"Set {spec.label} on {len(targets)} entries"
        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        reports = []
        with self._trigger_edit(model, label, content_touched=[index]) as m:
            trigger = m.manager().triggers[index]
            for kind, entry_index in targets:
                entry = _entries_of(trigger, kind)[entry_index]
                before = trigger_fields.live_quantity_slot(entry) if kind == "effect" else ""
                setattr(entry, spec.attribute, value)
                if kind != "effect":
                    continue
                # Same undo record: an object_attributes switch can re-point
                # serialization at a cluster slot that was never populated.
                definitions, type_attribute = self._trigger_vocabulary(kind)
                definition = (definitions or {}).get(getattr(entry, type_attribute, None))
                writes = trigger_fields.cluster_fixup(
                    entry,
                    before,
                    trigger_fields.live_quantity_slot(entry),
                    definition.attributes if definition is not None else (),
                )
                for name, new in writes:
                    setattr(entry, name, new)
                if writes:
                    reports.append(self._cluster_fixup_report(label, writes))
        if reports:
            # One status line: N effects switched the same way say the same thing.
            self._log_status(reports[0] if len(set(reports)) == 1 else f"{label} - adjusted {len(reports)} effects' quantity fields")

    @staticmethod
    def _cluster_fixup_report(label: str, writes) -> str:
        """The status line for an edit that rewrote other cluster fields. Said
        out loud: a silent field rewrite is worse than a noisy one."""
        if not writes:
            return ""
        parts = [
            f"cleared {name.replace('_', ' ')}"
            if value is None
            else f"set {name.replace('_', ' ')} to {value:g}"
            for name, value in writes
        ]
        return f"{label} - {', '.join(parts)}"

    def copy_triggers(self) -> None:
        """Put the selected triggers on the trigger clipboard (GH #27). Reads
        only: no model is built and nothing is recorded."""
        if self.scenario is None:
            return
        manager = parse_triggers(self.scenario)
        indices = self.trigger_panel.selected_trigger_indices()
        if manager is None or not indices:
            return
        self._trigger_clipboard = trigger_clipboard.copy_block(
            manager, indices, doc_id=self._doc_id, scenario_version=self.scenario.scenario_version
        )
        self._log_status(f"Copied {self._trigger_clipboard.label} to the clipboard.")
        self._sync_trigger_clipboard_state()

    def _dispatch_copy(self, _checked=False) -> None:
        if self.mode == "triggers":
            self.copy_triggers()
        else:
            self.copy_region()

    def _dispatch_paste(self, _checked=False) -> None:
        if self.mode == "triggers":
            self.paste_triggers()
        else:
            self.paste_region()

    def _dispatch_select_all(self, _checked=False) -> None:
        """Ctrl+A: every trigger in Triggers mode (GH #3), the whole map elsewhere."""
        if self.mode == "triggers":
            self.trigger_panel.select_all()
        else:
            self.select_all()

    def _dispatch_deselect(self, _checked=False) -> None:
        if self.mode == "triggers":
            self.trigger_panel.clear_trigger_selection()
        else:
            self.deselect()

    def paste_triggers(self) -> None:
        """Paste the trigger clipboard below the current trigger."""
        current = self.trigger_panel.current_trigger_index()
        self.trigger_structural_edit("paste", [] if current is None else [current])

    def _trigger_paste_refusal(self) -> str:
        """Why the trigger clipboard cannot be pasted here, or "" when it can.
        A block from another scenario version can carry fields, and effect
        types, this one does not store (cross-version paste is not supported)."""
        block = self._trigger_clipboard
        if block is None or self.scenario is None:
            return "Nothing to paste."
        here = self.scenario.scenario_version
        if block.scenario_version != here:
            return f"Copied from a {block.scenario_version} scenario; this one is {here}."
        if block.source_doc_id != self._doc_id and not library_compat.vocabulary_is_available(here):
            return f"No trigger vocabulary for scenario {here}, so unit references cannot be cleared."
        return ""

    def _trigger_paste_allowed(self) -> bool:
        return not self._trigger_paste_refusal()

    def _sync_trigger_clipboard_state(self) -> None:
        reason = self._trigger_paste_refusal() if self._trigger_clipboard is not None else ""
        self.trigger_panel.set_clipboard_state(self._trigger_paste_allowed(), reason)
        self._update_tool_enabled()

    def _unique_trigger_name(self, manager) -> str:
        """"New trigger", or the first free "New trigger N".

        Duplicate names are legal in-game, but phase 4c's folders and labels
        will have to key on names -- trigger_id does not survive a reorder --
        so a new trigger is not handed a name the document already uses.
        """
        taken = {(t.name or "") for t in manager.triggers}
        if "New trigger" not in taken:
            return "New trigger"
        n = 2
        while f"New trigger {n}" in taken:
            n += 1
        return f"New trigger {n}"

    def trigger_structural_edit(self, op: str, indices: Sequence[int], arg=None) -> None:
        """Add, copy, delete, or reorder triggers. The third funnel, and the
        only one that goes through TriggerEditModel.structural_edit().

        `indices` are list indices, the panel's selection in display order;
        "new" ignores them. However many there are, the op is one
        _trigger_edit bracket and so one undo record: TriggerDiffRecord is a
        whole-model before/after snapshot, so N mutations coalesce.

        Copy makes N independent duplicates in place: a copied trigger's
        (de)activate references still point at the originals, because
        copy_trigger() deepcopies. (Copy + Paste is the other verb, and moves
        a linked block; see trigger_clipboard.py.) Copy costs N renumbers,
        each source re-resolved by identity before its own call.

        Paste appends the clipboard block (import_triggers(index=-1), which
        renumbers nothing) and then inserts it into display order directly
        below `indices[0]`, or at the end with no anchor. Caveat: under
        legacy_exec_order == 1 the id order is the execution order, so a pasted
        block always executes last on a legacy file however it is displayed.
        A block copied from another document (GH #3) also clears unit
        references and names variables, in this same record; see
        trigger_clipboard.py.

        No `content_touched` on any of these. The library renumbers trigger_id
        across the whole list and remaps (de)activate-trigger references to
        match -- remove_triggers() even resets a reference to a deleted trigger
        to -1 -- but every one of those is the id-remap class, which the
        reference-signature diff detects on its own. trigger_id itself is not a
        serialized field (verified against versions/DE/*/structure.json), so a
        trigger that only moved keeps its blob. A pure display-order move is
        even further from that: it renumbers nothing at all.

        Copy is the one op that additionally has to repair display order.
        copy_trigger()'s append_after_source path ends at reorder_triggers(),
        whose triggers-setter resets trigger_display_order to identity -- so
        without display_order_with_copy_inserted() this would silently
        flatten a custom order (6 of 14 parseable corpus files carry one), and
        on a legacy-exec-order file silently rewrite what order the whole
        scenario executes in. Confirmed by direct measurement against
        atilla_1_scn_resaved before this fix existed.

        move_up/move_down never call reorder_triggers()/move_triggers() at
        all -- reordering here means permuting trigger_display_order only,
        ids are never renumbered. The panel only offers these while sorted
        by display order and unfiltered (_update_buttons()), so index
        always names a trigger
        currently at the display slot the button acts on.

        Section management carries its payload in `arg`: "new_section" takes
        the title and lands a divider at the end of `indices[0]`'s section (or
        of display order), "move_to_section" takes the target section's
        header index (None for the leading section). Both are display-order
        edits like move_up/move_down, so on a legacy_exec_order == 0 file a
        move to another section also changes when the trigger runs.
        """
        if op == "new_section":
            self._new_section(indices, arg)
            return
        if op == "move_to_section":
            self._move_to_section(indices, arg)
            return
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        indices = list(dict.fromkeys(indices))
        if not all(0 <= i < len(manager.triggers) for i in indices):
            return
        if op not in ("new", "paste") and not indices:
            return
        block = self._trigger_clipboard
        if op == "paste":
            # Owned here, not only by the greyed action: a keybind can race it.
            # Refused before _trigger_edit opens, so no empty record is pushed.
            reason = self._trigger_paste_refusal()
            if reason:
                if block is not None:
                    QMessageBox.warning(self, "Cannot paste these triggers", reason)
                return
        paste_results: list = []
        # Paste's anchor: the block lands directly below it in display order.
        anchor = indices[0] if op == "paste" and indices else None
        before_len = len(manager.triggers)
        sources = [manager.triggers[i] for i in indices]
        copies: list = []

        def mutate(m):
            if op == "new":
                return m.add_trigger(self._unique_trigger_name(m))
            if op == "copy":
                for source in sources:
                    # Re-resolved by identity: each copy renumbers the list.
                    index = next(i for i, t in enumerate(m.triggers) if t is source)
                    before_triggers = list(m.triggers)
                    before_order = list(m.trigger_display_order)
                    copies.append(m.copy_trigger(index))
                    m.trigger_display_order = display_order_with_copy_inserted(
                        before_triggers, before_order, m.triggers, index
                    )
                return None
            if op in ("move_up", "move_down"):
                delta = -1 if op == "move_up" else 1
                m.trigger_display_order = moved_display_order_block(
                    list(m.trigger_display_order), indices, delta
                )
                return None
            if op == "paste":
                # Captured before the import, which flattens display order;
                # pushed back through the setter (restore()'s Trap 7).
                before_order = list(m.trigger_display_order)
                result = trigger_clipboard.paste_into(m, block, doc_id=self._doc_id)
                paste_results.append(result)
                new_indices = list(range(before_len, before_len + len(result.triggers)))
                m.trigger_display_order = display_order_with_block_inserted(before_order, new_indices, anchor)
                return None
            return m.remove_triggers(sorted(indices))

        count = len(indices)
        noun = f"trigger {indices[0]}" if count == 1 else f"{count} triggers"
        if op == "new":
            label, refresh, refresh_arg = "New trigger", "panel", None
        elif op == "paste":
            label, refresh, refresh_arg = f"Paste {block.label}", "panel", None
        elif op == "copy":
            label, refresh, refresh_arg = f"Copy {noun}", "panel", None
        elif op in ("move_up", "move_down"):
            delta = -1 if op == "move_up" else 1
            label = f"Move {noun} {'up' if delta < 0 else 'down'}"
            # The one-row fast path stays single-row; N rows repopulate.
            refresh, refresh_arg = ("order", (indices[0], delta)) if count == 1 else ("panel", None)
        else:
            label, refresh, refresh_arg = f"Delete {noun}", "panel", None

        with self._trigger_edit(model, label, refresh=refresh, refresh_arg=refresh_arg) as m:
            m.structural_edit(mutate)

        if paste_results and paste_results[0].cross_document:
            # "panel" repopulates the trees only; named variables also stale the dialog.
            self.trigger_panel.refresh_variables()
            self._log_status(trigger_clipboard.paste_report(paste_results[0]))
        live = model.manager().triggers
        if op == "new":
            select = [len(live) - 1]
        elif op == "paste":
            # Valid because the import appends without renumbering.
            select = list(range(before_len, len(live))) or indices
        elif op == "copy":
            select = [i for i, t in enumerate(live) if any(t is c for c in copies)] or [indices[0]]
        elif op == "delete":
            select = [min(*indices, len(live) - 1)]
        else:
            # Ids never change on a move.
            select = indices
        self.trigger_panel.select_triggers(select)

    def _land_on(self, indices: Sequence[int]) -> None:
        """Select `indices` after a section edit, expanding the first one's
        section so a collapsed target does not swallow the result."""
        if indices:
            self.trigger_panel.reveal_trigger(indices[0])
        self.trigger_panel.select_triggers(indices)

    def _refuse_section_title(self, name: str, title: str) -> bool:
        """Warn and return True when `name` would not carry `title` back.
        Called before the edit opens, so a refusal records nothing."""
        reason = trigger_organize.divider_title_error(name, title)
        if reason:
            QMessageBox.warning(self, "Cannot use that section title", f'"{title}": {reason}.')
        return bool(reason)

    def _new_section(self, indices: Sequence[int], title) -> None:
        """Add a divider titled `title`, formatted like the file's own, at the
        end of `indices[0]`'s section: an empty section right after it. With
        no anchor it goes at the end of display order."""
        title = (title or "").strip()
        model = self._ensure_trigger_edits()
        if model is None or not title:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        order = list(manager.trigger_display_order)
        name = trigger_organize.format_divider(title, trigger_organize.divider_format(names))
        if self._refuse_section_title(name, title):
            return
        anchor = next((i for i in indices if 0 <= i < len(names)), None)
        target = len(order)
        if anchor is not None:
            header = trigger_organize.section_header_of(names, order, anchor)
            target = trigger_organize.section_end_slot(names, order, header)

        def mutate(m):
            m.add_trigger(name)
            # The getter appends the new index (Trap 7); set it back through
            # the setter at the target slot.
            new_index = len(m.triggers) - 1
            live = [i for i in m.trigger_display_order if i != new_index]
            live.insert(target, new_index)
            m.trigger_display_order = live

        with self._trigger_edit(model, f'New section "{title}"', refresh="panel") as m:
            m.structural_edit(mutate)
        self._land_on([len(model.manager().triggers) - 1])

    def _move_to_section(self, indices: Sequence[int], header_index) -> None:
        """Move the triggers at `indices`, as one block in display order, to
        the end of the section headed by `header_index` (None: the leading
        section). A pure display-order permutation, like Move Up/Down."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        names = [t.name or "" for t in manager.triggers]
        indices = list(dict.fromkeys(indices))
        if not indices or not all(0 <= i < len(names) for i in indices):
            return
        # Moving a header is "Move a whole section", a separate item.
        if any(trigger_organize.is_divider(names[i]) for i in indices):
            return
        order = list(manager.trigger_display_order)
        try:
            target = trigger_organize.section_end_slot(names, order, header_index)
        except ValueError:
            return
        moved = display_order_moved_to_slot(order, indices, target)
        if moved == order:
            return
        section = (
            "before the first section"
            if header_index is None
            else f'"{trigger_organize.section_label(names[header_index].strip())}"'
        )
        count = len(indices)
        noun = f"trigger {indices[0]}" if count == 1 else f"{count} triggers"

        def mutate(m):
            m.trigger_display_order = moved

        where = section if header_index is None else f"to {section}"
        with self._trigger_edit(model, f"Move {noun} {where}", refresh="panel") as m:
            m.structural_edit(mutate)
        self._land_on(sorted(indices, key=moved.index))

    def entry_structural_edit(
        self,
        op: str,
        index: int,
        kind: str,
        entry_index: int | Sequence[tuple[str, int]],
        type_id: int,
    ) -> None:
        """Add, copy, delete, or retype a condition or effect.

        `new` and `retype` take one `entry_index`. `copy` and `delete` also
        take a sequence of (kind, index) refs there, so a selection spanning
        both lists is one undo record (GH #60): delete removes in descending
        index order, copy appends in the order given.

        Deliberately not a structural_edit(): none of these changes the trigger
        list, so the alignment check has nothing to reconcile. They are content
        edits on one trigger, and `content_touched` is what marks its blob --
        without it the pre-edit bytes splice straight back over the new
        condition on save, with no error anywhere.

        Nothing here touches `condition_order` / `effect_order`, and that is
        not an oversight: both are lazily regenerated by their own getter, which
        checks the list against a stored hash and rebuilds the array when it has
        changed. serialize() commits through the library, which reads the
        property, so the order heals before it is ever written. Verified by
        write-and-reload, not assumed.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        if not 0 <= index < len(manager.triggers):
            return

        definitions, type_attribute = self._trigger_vocabulary(kind)
        if op == "retype" and definitions is None:
            # Unreachable from the UI, which gates the Type… button on the
            # panel's own vocabulary handle. Refuses rather than guessing.
            return

        if isinstance(entry_index, int):
            refs = [(kind, entry_index)]
        elif op in ("copy", "delete"):
            refs = [(str(k), int(i)) for k, i in entry_index]
        else:
            return
        if not refs:
            return
        label = f"{op.title()} {refs[0][0]}" if len(refs) == 1 else f"{op.title()} {len(refs)} entries"

        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        select: list[tuple[str, int]] = []
        report = ""
        with self._trigger_edit(model, label, content_touched=[index], refresh="entries") as m:
            trigger = m.manager().triggers[index]
            entries = trigger.conditions if kind == "condition" else trigger.effects
            if op == "retype":
                dropped = trigger_fields.retype_entry(
                    trigger, kind, entry_index, type_id, definitions, type_attribute
                )
                select = [(kind, entry_index)]
                report = self._retype_report(kind, definitions[type_id], dropped)
            elif op == "new":
                # The generic private entry point rather than the public
                # new_condition.<name>() wrappers: 71 vocabulary names across
                # the shipped versions have no wrapper method at all
                # ("enable/disable_object", "or"). See test_private_api_guard.
                if kind == "condition":
                    trigger._add_condition(type_id)
                else:
                    trigger._add_effect(type_id)
                select = [(kind, len(entries) - 1)]
            elif op == "copy":
                # deepcopy rather than a field-by-field copy, for restore()'s
                # own trap-5 reason: copying fields across would walk into
                # Effect.quantity's armour/attack bit-split.
                for ref_kind, ref_index in refs:
                    siblings = _entries_of(trigger, ref_kind)
                    siblings.append(copy.deepcopy(siblings[ref_index]))
                    select.append((ref_kind, len(siblings) - 1))
            else:
                # Descending, so every index still to be removed stays valid.
                for ref_kind, ref_index in sorted(refs, key=lambda ref: ref[1], reverse=True):
                    if ref_kind == "condition":
                        trigger.remove_condition(condition_index=ref_index)
                    else:
                        trigger.remove_effect(effect_index=ref_index)
                # The first ref's list, at the lowest removed slot, clamped.
                first_kind = refs[0][0]
                lowest = min(i for k, i in refs if k == first_kind)
                landing = min(lowest, len(_entries_of(trigger, first_kind)) - 1)
                select = [(first_kind, landing)] if landing >= 0 else []

        # Outside the pair: the tree is rebuilt by _after_trigger_edit, so the
        # row to land on only exists once that has run.
        if self.mode == "triggers" and select:
            self.trigger_panel.select_entries(select)
        if report and select:
            self._log_status(report)

    # The game's Create Object effect id, the only template GH #59 accepts.
    _CREATE_OBJECT_EFFECT = 11

    def stamp_create_objects(self, x: int, y: int) -> None:
        """GH #59's Create Objects tool: one copy of the current Create Object
        effect per tile of the brush footprint at (x, y), as one undo record.

        A funnel of its own rather than an entry_structural_edit() op, since
        that signature has no room for a tile list. Every refusal runs before
        _trigger_edit() opens, so a refused click records nothing. Tiles that
        already hold the same object for the same player are skipped, so a
        second click on the same spot stacks nothing.
        """
        panel = self.trigger_panel
        if self.scenario is None or self.mode != "triggers" or not panel._editable:
            self._log_status("Create Objects: triggers are not editable in this file")
            return
        index = panel.current_trigger_index()
        if index is None:
            self._log_status("Create Objects: select a trigger first")
            return
        if len(panel.selected_trigger_indices()) > 1:
            self._log_status("Create Objects: select a single trigger, not several")
            return
        if panel.picker_showing():
            self._log_status("Create Objects: close the type picker first")
            return
        ref = panel.current_entry_ref()
        if ref is None or ref[0] != "effect":
            self._log_status("Create Objects: select a Create Object effect to stamp")
            return
        entry_index = ref[1]
        model = self._ensure_trigger_edits()
        if model is None:
            return
        triggers = model.manager().triggers
        if not 0 <= index < len(triggers):
            return
        effects = triggers[index].effects
        if not 0 <= entry_index < len(effects):
            return
        template = effects[entry_index]
        if template.effect_type != self._CREATE_OBJECT_EFFECT:
            self._log_status("Create Objects: select a Create Object effect to stamp")
            return

        mm = self.scenario.map_manager
        footprint = brush.brush_tiles(
            x, y, self.brush_size_spin.value(), self.brush_shape_combo.currentData(),
            mm.map_width, mm.map_height,
        )
        occupied = {
            (effect.location_x, effect.location_y)
            for effect in effects
            if effect.effect_type == self._CREATE_OBJECT_EFFECT
            and effect.object_list_unit_id == template.object_list_unit_id
            and effect.source_player == template.source_player
        }
        tiles = [tile for tile in footprint if tile not in occupied]
        skipped = len(footprint) - len(tiles)
        if not tiles:
            self._log_status(f"Create Objects: all {skipped} tiles already have this object")
            return

        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        select = -1
        with self._trigger_edit(
            model, f"Create {len(tiles)} objects", content_touched=[index], refresh="entries"
        ) as m:
            trigger = m.manager().triggers[index]
            source = trigger.effects[entry_index]
            for tx, ty in tiles:
                # deepcopy for restore()'s trap-5 reason (entry_structural_edit's
                # "copy" op): a field-by-field copy walks into the quantity split.
                effect = copy.deepcopy(source)
                effect.location_x, effect.location_y = tx, ty
                trigger.effects.append(effect)
            select = len(trigger.effects) - 1

        if select < 0:
            return
        if self.mode == "triggers":
            panel.select_entry("effect", select)
        suffix = f" ({skipped} tiles already had one)" if skipped else ""
        self._log_status(f"Created {len(tiles)} Create Object effects{suffix}")

    def _trigger_vocabulary(self, kind: str):
        """(type id -> definition, type attribute) for the loaded scenario's
        version, or (None, attribute) when the library ships none for it.

        Read off the library's own per-version JSON rather than its
        module-level dicts, which are rewritten per load: load_vocabulary() is
        @cache'd, so asking per edit is free.
        """
        type_attribute = "condition_type" if kind == "condition" else "effect_type"
        loaded = self.scenario
        if loaded is None or not library_compat.vocabulary_is_available(
            loaded.scenario_version
        ):
            return (None, type_attribute)
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        entries = vocabulary.conditions if kind == "condition" else vocabulary.effects
        return (entries, type_attribute)

    @staticmethod
    def _retype_report(kind: str, new_definition, dropped: tuple[str, ...]) -> str:
        """The status line for one retype. A line rather than a modal: Delete
        has no confirmation either, and the whole retype is one undo step."""
        name = new_definition.name.replace("_", " ")
        if not dropped:
            return f"Changed {kind} to {name}."
        fields = ", ".join(field.replace("_", " ") for field in dropped)
        plural = "" if len(dropped) == 1 else "s"
        return f"Changed {kind} to {name}; {len(dropped)} field{plural} did not carry over ({fields})."

    # Variables are capped at 256 by the format, and add_variable() searches
    # 0..255 for the lowest free id and raises IndexError when there is none.
    _MAX_VARIABLES = 256

    def variable_structural_edit(self, op: str, variable_id: int, name: str) -> None:
        """Add, remove, or rename a trigger variable. The fifth funnel (4b.6c).

        Goes through structural_edit() like the trigger one does, for a
        different reason: no blob goes stale and no trigger is touched, but
        _variable_signature() is diffed in there and is the only thing that
        sets _variables_dirty. Mutating manager.variables outside it leaves the
        old variable block spliced back verbatim, old count included, and the
        edit vanishes on save with nothing raising.

        Removal is a plain list mutation because TriggerManager has no
        remove_variable(). Nothing renumbers: the surviving variables keep
        their ids, so the block simply writes one entry fewer and the removed
        id becomes a gap. Conditions and effects referencing that id are left
        pointing at a variable that no longer has a name, which is what the
        in-game editor does with them too.

        The one trap that follows from the gap: add_variable() takes the
        *lowest* free id, so removing a variable and adding another hands the
        new one the removed one's id, and with it every reference the document
        still holds to it.

        Rename mutates the existing Variable object's `name` in place --
        there is no library method that replaces it in the list, and
        replacing it here would just be add-then-remove with an extra id
        churn. `_variable_signature()` reads `.name` fresh on every diff, so
        structural_edit() still notices; TriggerSnapshot.variables is a deep
        copy rather than a list of references, so the pre-edit name is not
        overwritten by the very mutation it exists to undo.

        Both cases are pre-validated before the pair is opened rather than
        inside it. commit_trigger_edit() pushes unconditionally, and a raised
        structural_edit() marks every blob dirty and costs the document its
        minimal-diff guarantee, so an edit that cannot succeed must not be
        opened at all.
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()

        if op == "add":
            name = (name or "").strip()
            if not name:
                return
            if len(manager.variables) >= self._MAX_VARIABLES:
                self._log_status(f"Add variable failed: all {self._MAX_VARIABLES} variable ids are in use")
                QMessageBox.warning(
                    self,
                    "No variable id available",
                    f"This scenario already uses all {self._MAX_VARIABLES} variable ids.",
                )
                return
            label = f"Add variable {name}"

            def mutate(m):
                return m.add_variable(name)

        elif op == "rename":
            name = (name or "").strip()
            target = next((v for v in manager.variables if v.variable_id == variable_id), None)
            if not name or target is None or target.name == name:
                return
            label = f"Rename variable {variable_id} to {name}"

            def mutate(m):
                for variable in m.variables:
                    if variable.variable_id == variable_id:
                        variable.name = name
                        return
                return

        else:
            if not any(v.variable_id == variable_id for v in manager.variables):
                return
            label = f"Remove variable {variable_id}"

            def mutate(m):
                for variable in list(m.variables):
                    if variable.variable_id == variable_id:
                        m.variables.remove(variable)
                        return
                return

        with self._trigger_edit(model, label, refresh="variables") as m:
            m.structural_edit(mutate)

    # -- unit editing --------------------------------------------------------
    #
    # Every unit mutation (place/move/delete/reassign) goes through
    # _unit_edit(). Unlike _trigger_edit(), which commits even on a raised
    # structural_edit() because a failed structural edit still dirties every
    # blob, every UnitEditModel operation validates BEFORE it mutates (add()'s
    # player-range check, remove()'s garrison-reference guard, reassign()'s
    # range check, and set_position()/reassign()'s _locate() lookup all raise
    # first, before touching any field) -- so an exception here always means
    # nothing was touched, and aborting rather than committing a no-op record
    # is correct.

    @contextmanager
    def _unit_edit(
        self,
        model: UnitEditModel,
        label: str,
        players: Sequence[int],
        splices: list | None = None,
        fields_only: bool = False,
    ):
        """Open one recorded unit edit, and close it whatever happens.

        `players` is the caller's declaration of which player list(s) the
        upcoming edit will touch -- one for place/move/delete, both source and
        destination for reassign -- passed straight through to
        UnitEditModel.begin_unit_edit(). Enter only once a real change is
        confirmed (e.g. the new value differs from the current one): unlike
        _trigger_edit, there is no failure mode here that still needs a
        record, so a spurious enter would record a genuine phantom undo step.

        splices (Batch D's D5): None (the default) keeps this method's exact
        pre-D5 behaviour -- a wholesale _after_unit_mutation(None). A caller
        doing a single-unit splice-eligible edit (Move/Nudge/Rotate/Set
        field/gate orientation/Place/single Delete) instead passes an
        initially-empty list and appends one render_cache.UnitSplice to it
        per touched unit from INSIDE the `with` block, after that unit's own
        model.set_*()/add()/remove() call -- a splice needs the unit's
        post-edit state, which doesn't exist before then. This method only
        reads `splices` back once the block has exited normally, so every
        append must already have happened by that point (asserted below:
        entering with a non-None `splices` and leaving it empty means the
        caller forgot to record the very edit it declared, which would
        silently skip the cache/index refresh this whole mechanism exists to
        do).

        fields_only (Batch D's D6): passed straight through to
        UnitEditModel.begin_unit_edit(). True for every splice-eligible
        caller above except Place and single Delete -- add()/remove() are
        membership mutators UnitEditModel refuses to run inside a
        fields_only edit, so only the pure set_position/set_rotation/
        set_unit_const callers (Move/Nudge/Rotate/Set field/gate
        orientation) may pass it."""
        model.begin_unit_edit(players, fields_only=fields_only)
        try:
            yield model
        except Exception:
            model.abort_unit_edit()
            raise
        model.commit_unit_edit(label, self.edit_history)
        if splices is not None:
            assert splices, "a splice-tracking _unit_edit exited with no UnitSplice recorded"
        self._after_unit_mutation(splices)
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)

    def _unit_footprint(self, unit) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
        """(own_tile, occupied_tiles) for `unit` at its CURRENT x/y/unit_const
        -- the snapshot a render_cache.UnitSplice needs on each side of a
        single-unit edit (Batch D's D5), taken by calling this once before
        and once after the model mutation."""
        mm = self.scenario.map_manager
        own_tile = (int(unit.x), int(unit.y))
        tiles = tuple(unit_occupied_tiles(unit, mm.map_width, mm.map_height) or ())
        return own_tile, tiles

    def _unit_list_index(self, player_id: int, unit) -> int:
        """`unit`'s position in scenario.unit_manager.units[player_id] --
        render._resolve_unit_sprite()'s own `i`, which a render_cache.
        UnitSplice built outside UnitEditModel (Batch D's D5) has no other
        way to learn. A linear scan by identity: Unit isn't hashable and
        UnitEditModel._locate() is that model's own private lookup, not
        exposed for a caller building a splice from outside it. Safe only
        for a splice-eligible op, which never reorders the list before this
        runs -- see render_cache.UnitSplice's own docstring."""
        for index, candidate in enumerate(self.scenario.unit_manager.units[player_id]):
            if candidate is unit:
                return index
        raise ValueError("unit not found in its own player's list")

    def _patch_unit_index_for_move(self, player_id: int, unit, old_bounds) -> None:
        """Batch D's D5: the unit-pick index counterpart of a splice-eligible
        Move/Nudge/Set-field edit. A no-op if the index doesn't exist yet
        (not in Units mode) -- the same guard _rebuild_unit_index()'s own
        callers rely on, since there is nothing here to keep in step with."""
        index = self.map_view._unit_index
        if index is not None:
            unit_pick.patch_index_for_move(self.scenario, index, player_id, unit, old_bounds)

    def _after_unit_mutation(
        self, changed: list[UnitSplice] | None = None, defer_index: bool = False, rebuild_index: bool = False
    ) -> None:
        """Shared invalidation tail for every unit mutation, whether driven
        by a tool (_unit_edit's own commit, above) or by undo/redo
        (_move_history's "unit" branch).

        Units paint in every mode (_build_filters_button's own docstring:
        filtering "affects View and Terrain too"), and Ctrl+Z is global, so
        the cache-invalidate-and-repaint step below must NOT be mode-gated --
        only the selection/inspector reconciliation that follows is, mirroring
        _refresh_selection_after_filter()'s own mode gate.

        Two DISTINCT invalidate_region()s are both required, the same pair
        _on_filter_changed()'s own set_unit_filter() call bundles internally:
        self._cache.invalidate_units() only rebuilds the cache's SOURCE
        structures (units_by_tile, per-level sprite layers via the source-gen
        bump) -- it never touches self._cache's own dict of already-
        COMPOSITED chunk pixel arrays, so a chunk rendered before this edit
        stays cached with the pre-edit pixels. _ChunkCacheBase.
        invalidate_region() (called on the CACHE, not on self.map_view) is
        the separate primitive that actually evicts those, forcing a real
        recomposite next paint. self.map_view.invalidate_region() only
        schedules that repaint; without the cache-level call first it just
        re-blits the same stale pixels. Missing this step is exactly what
        made place/move/delete/reassign change the model correctly while the
        map kept showing the pre-edit unit -- caught after the fact, not by
        any of this slice's own tests, none of which asserted on pixels.

        changed (Batch D's D5): None (the default, and every pre-D5 caller's
        exact behaviour) means "unknown" and takes the wholesale branch
        below. A single-unit splice-eligible edit instead passes a non-empty
        list of render_cache.UnitSplice -- see that class's own docstring --
        and _patch_unit_edit_cache() below scopes both the source splice
        (D4's invalidate_units(changed)) and the actual repaint to the bbox
        that edit's old/new footprints could have touched, instead of the
        whole canvas.

        The unit-pick index is deliberately NOT touched here when `changed`
        is given: patch_index_for_move()/patch_index_for_add() each need
        that specific op's own pre-edit bounds/filter state, which only the
        call site still has by the time this method runs -- so every
        `changed`-passing caller has already patched (Move/Nudge/Set field/
        Place) or fully rebuilt (gate orientation's span change, single
        Delete) self.map_view._unit_index itself, before ever reaching here.

        defer_index: the Convert brush's mid-stroke repaint only. Repaints
        and re-arms the warm, but skips the index rebuild and selection
        reconciliation; the stroke's final call does both.

        rebuild_index: forces the full _rebuild_unit_index() even though
        `changed` is given. For a caller whose splices were never matched by
        an index patch, i.e. the Convert stroke's final call.
        """
        # A third invalidation, on the same terms as the two below: an
        # in-flight warm is walking the unit list this edit just changed.
        # Re-armed below, once the cache is invalidated (Batch B step B1).
        self._cancel_warms()
        # Scoped edits patch the index without _rebuild_unit_index(), which
        # otherwise resets this; a moved unit can leave or join a stack.
        self._stack_cycle = None
        if self._cache is not None:
            if changed is not None:
                self._patch_unit_edit_cache(changed)
            else:
                self._cache.invalidate_units()
                canvas_w, canvas_h = self._cache.canvas_dims(0)
                self._cache.invalidate_region((0, 0, canvas_w, canvas_h))
                self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        # Above the mode gate, not at the literal tail: undo is global, so a
        # unit edit reverted from Terrain mode must re-arm the warm too.
        self._start_level_warm()
        self._unit_ref_index = None
        if defer_index:
            return
        self._on_unit_references_moved()
        # Unconditional: undo is global. A repopulate, since a reassign can give an inactive slot units.
        self._repopulate_stats_players()
        if self.mode != "units":
            # The footprint overlay is index-driven and live in every mode, so
            # a unit edit made outside Units mode (paste, mirror, undo) still
            # has to reach it. A full rebuild rather than a patch: the in-place
            # patching below is a Units-mode path, so outside it the index may
            # never have been touched at all.
            if settings.get_footprint_outlines():
                self._rebuild_unit_index()
            return
        if changed is None or rebuild_index:
            self._rebuild_unit_index()
        else:
            # The index was patched in place, so nothing derived from it --
            # the stacks, the footprint outlines -- was recomputed.
            self.map_view.refresh_after_index_patch()
        self._refresh_selection_view()

    def _patch_unit_edit_cache(self, changed: list[UnitSplice]) -> None:
        """The scoped branch of _after_unit_mutation() (Batch D's D5):
        splices self._cache's unit-derived sources (D4's own invalidate_
        units(changed)) and repaints only the bbox `changed`'s old/new
        footprints could have touched, rather than invalidating the whole
        canvas the way the wholesale branch does.

        invalidate_units(changed) may still fall back to a full source
        rebuild internally for one or more entries (a wall/connector const,
        or a tile shared with another unit -- see render_cache.
        _splice_eligible()) -- that only changes how the SOURCE data got
        refreshed, never how much of the canvas needs recompositing: the
        bbox below is sized from `changed`'s own footprints regardless,
        which is always a safe (over-)approximation of what a full source
        refresh could have moved on screen, since nothing but this edit's
        own unit(s) actually changed.

        Flat has no elevation term and no dirty_screen_bbox_* counterpart --
        mirrors _apply_dirty_render's own Flat branch, patching one rect per
        touched tile rather than a union bbox."""
        # Convert's stroke end can drain an empty list; Flat would treat [] as wholesale.
        if not changed:
            return
        self._cache.invalidate_units(changed)
        old_tiles = {t for s in changed for t in s.old_tiles}
        new_tiles = {t for s in changed for t in s.new_tiles}
        touched = old_tiles | new_tiles
        if not touched:
            return
        mm = self.scenario.map_manager
        if self._render_style == "flat":
            tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
            rects = [(x * tile_px, y * tile_px, (x + 1) * tile_px, (y + 1) * tile_px) for x, y in touched]
            self._cache.patch_rects(rects)
            for rect in rects:
                self.map_view.invalidate_region(rect)
            return
        dirty_indices = [y * mm.map_width + x for x, y in touched]
        elevation_changed: set = set()
        if self._render_style == "stepped":
            bbox = dirty_screen_bbox_iso(
                self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                flatten_elevations=(self._terrain_style == "flat"), extra_anchor_tiles=old_tiles,
            )
        else:
            bbox = dirty_screen_bbox_sloped(
                self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True,
                with_sprites=self._cache.sprites_enabled, elevation_changed=elevation_changed,
                extra_anchor_tiles=old_tiles,
            )
        if bbox is None:
            # Defensive, not expected (see dirty_screen_bbox_iso's own
            # docstring): a unit edit never moves a tile's elevation, so this
            # only fires if the terrain was already out of range beforehand.
            canvas_w, canvas_h = self._cache.canvas_dims(0)
            self._cache.invalidate_region((0, 0, canvas_w, canvas_h))
            self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
            return
        self._cache.patch(bbox, elevation_changed=elevation_changed)
        self.map_view.invalidate_region(bbox)

    def _update_edit_actions(self) -> None:
        self._refresh_paste_move_validity()
        can_undo = self.scenario is not None and self.edit_history.can_undo
        can_redo = self.scenario is not None and self.edit_history.can_redo
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)

    def undo(self) -> None:
        if self.scenario is None:
            return
        self._move_history(self.edit_history.peek_undo(), self.edit_history.undo, "Undo")

    def redo(self) -> None:
        if self.scenario is None:
            return
        self._move_history(self.edit_history.peek_redo(), self.edit_history.redo, "Redo")

    def _move_history(self, record, move, action: str, quiet: bool = False) -> None:
        """One undo or redo step: read what the record needs *before* the
        cursor moves, move it, then hand the rest to
        _refresh_after_history_move() (shared with the History window's jump,
        so the two cannot drift apart).

        Batch D's D6: a plain UnitDiffRecord for a fields_only edit (Move/
        Nudge/Rotate/Set field/gate orientation) exposes unit_field_entries,
        which lets the unit branch below build one render_cache.UnitSplice
        per touched unit -- the same scoped _after_unit_mutation() path
        _unit_edit() itself takes -- instead of the wholesale invalidation
        every other unit record (place/delete/reassign/paste, or a
        CompositeDiffRecord, neither of which carries that attribute) still
        gets. The pre-mutation footprint has to be read before `move()` runs
        below: by the time the unit branch is reached the model has already
        been put back to the record's other side.
        """
        if record is None:
            # Still logs, exactly as paste_region() and fill do on a genuine
            # no-op: silently doing nothing reads as a broken keybind rather
            # than as an empty history.
            self._update_edit_actions()
            self._log_status(f"Nothing to {action.lower()}")
            return
        field_entries = getattr(record, "unit_field_entries", None)
        old_footprints = (
            [(player_id, index, unit, *self._unit_footprint(unit)) for player_id, index, unit in field_entries]
            if field_entries
            else None
        )
        dirty = move(
            self.scenario.map_manager.terrain,
            self.trigger_edits,
            self.option_edits,
            self.unit_edits,
            self.message_edits,
        )
        splices = None
        if old_footprints is not None:
            splices = []
            for player_id, index, unit, old_own, old_tiles in old_footprints:
                new_own, new_tiles = self._unit_footprint(unit)
                splices.append(UnitSplice(player_id, index, unit, old_own, new_own, old_tiles, new_tiles))
        # `record.kinds()`, not `record.kind`: a CompositeDiffRecord
        # (phase 2.8's region paste) can carry more than one domain in a
        # single record, and each still needs the same refresh a plain
        # record of that domain would get.
        self._refresh_after_history_move(record.kinds(), dirty, splices)
        # quiet: a region move undoes its own previous paste as an internal
        # step, and logging "Undo: Paste Region" there would read as the
        # user's own undo. Suppresses only this line, nothing else.
        if not quiet:
            self._log_status(f"{action}: {record.label}")

    def _refresh_after_history_move(self, kinds, dirty, splices=None) -> None:
        """Everything the window has to put back in step once the history
        cursor has moved and the models have been restored -- shared by
        undo/redo (_move_history above) and by the History window's multi-step
        jump (_on_history_jump), so the two cannot drift apart.

        Branches on `kinds` rather than on a record: a trigger record needs
        the trigger panel rebuilt and yields no tile indices, a tile record
        needs the incremental repaint, either a trigger or an option record
        needs the Map Options form put back in step with the models, and a
        unit record needs the render caches invalidated and the
        selection/inspector reconciled (_after_unit_mutation()).

        _update_title() runs here rather than in _apply_dirty(), which
        early-returns on an empty index list: since a trigger undo produces no
        dirty tiles, the window's "*" marker would otherwise never update on
        one. Every other _apply_dirty() caller already updates the title
        itself.

        `splices` is Batch D's D6 scoped unit path: a list of UnitSplice for a
        fields_only record, or None for the wholesale invalidation. A jump
        always passes None -- its span reads each record's old footprints
        immediately before that record moves, which a one-shot multi-record
        move cannot do.
        """
        # Undo/redo rewrites what the armed field and its form show.
        self.disarm_unit_picker()
        self._apply_dirty(dirty)
        if "unit" in kinds:
            if splices is not None:
                # _after_unit_mutation() deliberately leaves the pick index
                # alone whenever `changed` is given (its own docstring),
                # trusting the call site to have handled it already -- here
                # that's this method, standing in for whichever tool call
                # site originally built the record. A full rebuild rather
                # than a move/add-shaped patch: this path is undo/redo, not
                # the interactive drag D6 exists to make cheap, and it is
                # the exact rebuild the pre-D6 wholesale branch always paid
                # for anyway.
                if self.mode == "units":
                    self._rebuild_unit_index()
                self._after_unit_mutation(splices)
            else:
                self._after_unit_mutation()
        if "trigger" in kinds and self.mode == "triggers":
            # Rebuilt wholesale rather than patched: the panel is a read-only
            # view over the parsed manager, and an undo can change trigger
            # membership, order and ids at once.
            self._show_triggers()
        if "trigger" in kinds:
            # Mode-free: the overlay persists into View, where the panel is not refreshed.
            self._rederive_trigger_overlay()
        if kinds & {"options", "trigger"}:
            # Both kinds, not just "options": the exec-order row is shown in
            # the Map Options panel but recorded as a trigger edit, so undoing
            # one produces a "trigger" record while the user is looking at this
            # form. A no-op outside the mode.
            self._repopulate_map_options()
        if "options" in kinds:
            # A Players mode field, and a Diplomacy grid cell, both ride an
            # OptionsDiffRecord too (decision 1), so their own undo/redo
            # must refresh those panels the same way -- a no-op outside
            # their own mode, same as the call above.
            self._repopulate_players()
            self._repopulate_diplomacy()
            # Runs after the caller's move has applied the undo/redo to
            # option_edits, so the recompute reads the post-move pending values.
            self._after_player_color_change()
            # An undone Number of Players edit changes the stats combo's list.
            self._repopulate_stats_players()
        elif "trigger" in kinds:
            self._update_player_stats()
        if "messages" in kinds:
            self._repopulate_messages()
        self._update_title()
        self._update_edit_actions()

    def _update_title(self) -> None:
        if self.scenario is None:
            self.setWindowTitle("DEscape")
            return
        marker = "*" if self.edit_history.is_dirty else ""
        self.setWindowTitle(f"{marker}{self.scenario.path.name} — DEscape")

    def _confirm_discard_changes(self) -> bool:
        """True if it's OK to proceed: no scenario loaded, no unsaved edits,
        or the user explicitly chose to discard them. Call before anything
        that would drop the current scenario -- open, close, or window close."""
        if self.scenario is None or not self.edit_history.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{self.scenario.path.name} has unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard

    def _save_as_start_path(self) -> str:
        if self._untitled:
            # Never the template's own directory (Path.home() is always
            # writable and never TEMPLATE_DIR).
            return str(Path.home() / UNTITLED_NAME)
        src = self.scenario.path
        # File > Open can reach the shipped template itself; without this,
        # the default would aim inside TEMPLATE_DIR, which write_scenario()
        # unconditionally refuses to write into.
        parent = Path.home() if src.parent.resolve() == TEMPLATE_DIR.resolve() else src.parent
        return str(parent / src.name)

    def show_recover_autosave(self) -> None:
        """File > Recover from Autosave…. Modeless and reused, like the
        clipboard-history and analysis dialogs."""
        if self._recover_dialog is None:
            self._recover_dialog = RecoverAutosaveDialog(self, self._open_autosave_entry)
        else:
            self._recover_dialog.refresh()
        self._recover_dialog.show()
        self._recover_dialog.raise_()
        self._recover_dialog.activateWindow()

    def _open_autosave_entry(self, entry) -> None:
        """Open a slot as an UNTITLED document: save() then falls through to
        Save As, so a recovered document can never silently overwrite the
        original the user is trying to compare it against."""
        if not self._confirm_discard_changes():
            return
        if self._recover_dialog is not None:
            self._recover_dialog.close()
        self.load_scenario(entry.path, untitled=True)
        if self.scenario is not None:
            self._log_status(f"Recovered {entry.path.name} as a new untitled document")

    def _edit_model_kwargs(self) -> dict:
        """The edit models write_scenario() needs, in one place. save(),
        save_as() and the autosave tick are three call sites that have to
        stay in lockstep forever, and the last one already cost a real gap
        once (an autosave that omitted messages= would have silently dropped
        every Messages-mode edit). Adding a model here reaches all three."""
        return {
            "triggers": self.trigger_edits,
            "options": self.option_edits,
            "units": self.unit_edits,
            "messages": self.message_edits,
        }

    def _autosave_doc_key(self) -> str:
        """This document's rotation key -- its resolved path, or its own
        doc_id while untitled."""
        path = None if (self._untitled or self.scenario is None) else self.scenario.path
        return autosave.doc_key(path, self._doc_id)

    def _stroke_in_progress(self) -> bool:
        """True between begin_stroke() and commit_stroke(), or while
        MapView is mid-drag. Serializing here would write a half-applied
        stroke."""
        return self.edit_history.in_stroke or self.map_view._stroke_active

    def _reset_autosave_timer(self) -> None:
        """Re-arm from the current settings -- called at startup and by the
        Saving tab's enabled/interval handlers. Retention and location need
        no restart; they are read fresh on each tick."""
        self._autosave_timer.stop()
        if settings.get_autosave_enabled():
            self._autosave_timer.start(settings.get_autosave_interval_min() * 60 * 1000)

    def _autosave_tick(self) -> None:
        """Write this document's next recovery slot, if it is worth writing
        and this is a safe moment to write it.

        The three gates are all free, and each earns its place. is_dirty:
        write_scenario()'s own no-op short circuit sits AFTER full
        serialization, so letting a clean document reach it would burn the
        whole ~0.9s (measured, largest corpus file) to discover there was
        nothing to do. The cursor comparison is the same idea one level in:
        a document edited once and then left alone stays dirty forever.
        _busy/mid-stroke is a correctness gate rather than an optimization
        -- a QTimer callback can be delivered inside any of viewer.py's
        processEvents() calls, i.e. mid-load or mid-render.

        **The settle rule.** The retry timer connects back here, so during a
        long paint session it re-arms every 5s and would otherwise fire on
        the first clear tick, which can land exactly as the user begins the
        next stroke. A retry that saw a stroke since it was armed re-arms
        once more instead of writing: one extra retry wait after the last
        stroke, and no write ever begins inside a paint burst.

        Never calls mark_saved() and never touches saved_at_cursor. A
        recovery slot does not mean the document matches its real file;
        clearing either would drop the title's "*" AND make
        _confirm_discard_changes() return True, so the user would close
        without a prompt and lose the work. It calls write_scenario()
        directly rather than save() for the same class of reason: save()
        routes an untitled document into a modal Save As dialog.
        """
        if self.scenario is None or not settings.get_autosave_enabled():
            return
        if not self.edit_history.is_dirty:
            return
        if self.edit_history.cursor == self._autosaved_at_cursor:
            return
        if self._busy or self._stroke_in_progress():
            self._stroke_seen_since_retry = True
            self._autosave_retry_timer.start(AUTOSAVE_RETRY_MS)
            return
        if self._stroke_seen_since_retry:
            self._stroke_seen_since_retry = False
            self._autosave_retry_timer.start(AUTOSAVE_RETRY_MS)
            return

        key = self._autosave_doc_key()
        source = None if self._untitled else self.scenario.path
        slot = autosave.slot_path(
            key, self.scenario.path.name, source, settings.get_autosave_location()
        )
        try:
            slot.parent.mkdir(parents=True, exist_ok=True)
            # backup=False is mandatory, not a preference: the default would
            # put a .bak/.orig pair beside every rotating slot.
            write_scenario(self.scenario, slot, backup=False, **self._edit_model_kwargs())
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            # Swallowed to a log line, never a modal. An autosave that pops a
            # QMessageBox mid-edit is worse than one that quietly fails; the
            # user's real file is untouched either way.
            self._log_status(f"Autosave failed: {type(e).__name__}: {e}")
            return
        autosave.record(key, slot, str(self.scenario.path), untitled=self._untitled)
        autosave.rotate(key, settings.get_autosave_retention())
        self._autosaved_at_cursor = self.edit_history.cursor
        self._log_status(f"Autosaved to {slot.name}")

    def save(self) -> None:
        if self.scenario is None:
            return
        if self._untitled:
            # No path to write back to yet -- same first-save-needs-a-name
            # case Save As already handles.
            self.save_as()
            return
        path = str(self.scenario.path)
        autosave_key = self._autosave_doc_key()
        try:
            result = write_scenario(
                self.scenario,
                path,
                backup=settings.get_backups_enabled(),
                **self._edit_model_kwargs(),
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}")
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            self._log_status(f"Save failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.edit_history.mark_saved()
        self._update_title()
        # The real file now holds this state, so its recovery slots are
        # obsolete -- leaving them would let Recover offer something OLDER
        # than what is on disk.
        autosave.discard(autosave_key)
        self._autosaved_at_cursor = self.edit_history.cursor
        if not result.wrote:
            self._log_status("No changes to save")
        else:
            names = ", ".join(p.name for p in result.backups)
            suffix = f" (backed up {names})" if names else ""
            self._log_status(f"Saved to {path}{suffix}")

    def save_as(self) -> None:
        if self.scenario is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scenario as", self._save_as_start_path(), "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        # Captured BEFORE the retarget below, which reassigns scenario.path and
        # clears _untitled: from that point on this document keys by its new
        # path, so discarding a freshly-computed key would drop the wrong
        # (empty) one and leak the untitled slots into Recover under a stale
        # "Untitled" label.
        autosave_key = self._autosave_doc_key()
        try:
            result = write_scenario(
                self.scenario,
                path,
                backup=settings.get_backups_enabled(),
                **self._edit_model_kwargs(),
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}")
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
            self._log_status(f"Save failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        # Save As always retargets the open document to the file just written,
        # matching the usual editor convention (Notepad, VS Code, etc.): the
        # window is now editing that file, not the one it was opened from.
        self.scenario.path = Path(path)
        self._untitled = False
        self._update_info()  # the info panel's "File:" line just changed
        self.edit_history.mark_saved()
        self._update_title()
        # The real file now holds this state, so its recovery slots are
        # obsolete -- leaving them would let Recover offer something OLDER
        # than what is on disk.
        autosave.discard(autosave_key)
        self._autosaved_at_cursor = self.edit_history.cursor
        if not result.wrote:
            self._log_status("No changes to save")
        else:
            names = ", ".join(p.name for p in result.backups)
            suffix = f" (backed up {names})" if names else ""
            self._log_status(f"Saved to {path}{suffix}")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # First-ever show: cache real (post-realization) tool button widths
        # before anything has been moved into the overflow menu, then run
        # the first overflow pass now that _apply_toolbar_overflow() has
        # something real to measure against. See _cache_tool_button_widths's
        # own docstring for why this can't happen earlier, in
        # _build_toolbar().
        if not self._tool_overflow_measured:
            self._cache_tool_button_widths()
            self._tool_overflow_measured = True
        self._apply_toolbar_overflow()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_toolbar_overflow()

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_changes():
            event.ignore()
            return
        # The warm's real lifetime boundary. Without this a window closed
        # mid-warm leaves a live QTimer holding a bound tick, which keeps the
        # orphaned cache (and the whole scenario behind it) alive and keeps
        # ticking on any event loop that outlives the window -- one shared
        # QApplication across a test session being the case that makes it
        # visible rather than merely wasteful.
        self._cancel_warms()
        # Same lifetime reasoning as the warm above, and the teardown
        # trigger_panel.py already does for its own variables dialog:
        # close() doesn't destroy the window, so a stray top-level dialog
        # outlives it on a shared test QApplication.
        if self._clipboard_dialog is not None:
            self._clipboard_dialog.close()
            self._clipboard_dialog.deleteLater()
            self._clipboard_dialog = None
        # Same lifetime reasoning as the clipboard dialog above.
        if self._history_dialog is not None:
            self._history_dialog.close()
            self._history_dialog.deleteLater()
            self._history_dialog = None
        # Same lifetime problem as the warm above: close() doesn't destroy the
        # window, so an armed drain timer would still fire on a shared loop.
        # The record itself is deliberately NOT cleared here: a close
        # dispatched from inside a load's own processEvents() would then null
        # it out from under load_scenario's remaining writes to it.
        self._paint_report_timer.stop()
        settings.set_window_size(self.width(), self.height())
        sizes = self.content_splitter.sizes()
        if len(sizes) == 2:
            settings.set_split_sizes(sizes[0], sizes[1])
        log_sizes = self.log_splitter.sizes()
        if len(log_sizes) == 2:
            settings.set_log_height(log_sizes[1])
        event.accept()

    def _show_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec_()

    def _show_debug_log(self) -> None:
        dialog = DebugLogDialog(self)
        dialog.exec_()

    def _show_mirror_dialog(self) -> None:
        if self.scenario is None:
            return
        dialog = MirrorDialog(self)
        dialog.exec_()

    def _show_analysis(self) -> None:
        """Tools > Map Analysis: a read-only pass, then the modeless results
        dialog. Busy-guarded like load_scenario(): the trigger parse inside
        can take seconds on a large file."""
        if self.scenario is None or self._busy:
            return
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            started = time.perf_counter()
            report = map_analysis.analyze(self.scenario)
            elapsed = time.perf_counter() - started
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._log_status(f"Map Analysis: {report.headline} ({elapsed:.2f}s)")
        # The analysis just parsed the triggers; the next mode switch can't notice the flip.
        self._update_info()
        if self._analysis_dialog is None:
            self._analysis_dialog = AnalysisDialog(
                self,
                on_navigate=self._navigate_to_finding,
                on_select=self._focus_analysis_marker,
                on_closed=self.map_view.clear_analysis_markers,
            )
        self._analysis_dialog.set_report(report, self.scenario.path.name)
        mm = self.scenario.map_manager
        self.map_view.set_analysis_markers(map_analysis.marker_anchors(report, mm.map_width, mm.map_height))
        self._analysis_dialog.show()
        self._analysis_dialog.raise_()
        self._analysis_dialog.activateWindow()

    def _close_analysis_dialog(self) -> None:
        """Its findings describe the document they were run on; a stale row
        must not navigate inside a different map."""
        if self._analysis_dialog is not None:
            self._analysis_dialog.close()
            self._analysis_dialog.deleteLater()
            self._analysis_dialog = None
        self.map_view.clear_analysis_markers()

    def _focus_analysis_marker(self, finding: map_analysis.Finding | None) -> None:
        """The dialog's current row: ring its marker, clamped like the anchors."""
        if finding is None or self.scenario is None:
            self.map_view.set_analysis_marker_focus(None)
            return
        mm = self.scenario.map_manager
        self.map_view.set_analysis_marker_focus(map_analysis.marker_tile(finding, mm.map_width, mm.map_height))

    def _navigate_to_finding(self, finding: map_analysis.Finding) -> None:
        if self.scenario is None:
            return
        tile = finding.tile
        if finding.unit_key is not None:
            # The pick index, and so selection, only exists in Units mode.
            self.mode_combo.setCurrentText("Units")
            index = self.map_view._unit_index
            entry = index.entry_for_key(finding.unit_key) if index is not None else None
            if entry is not None:
                self._selection = [finding.unit_key]
                self._refresh_selection_view()
                tile = (int(entry.unit.x), int(entry.unit.y))
                self._log_status(
                    f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})"
                )
        if tile is None:
            return
        mm = self.scenario.map_manager
        x = max(0, min(mm.map_width - 1, tile[0]))
        y = max(0, min(mm.map_height - 1, tile[1]))
        self.map_view.center_on_tile(x, y)

    def on_ruler_measured(self, measurement) -> None:
        """MapView's completed-measurement callback. Logs the same string the
        on-map label shows, plus both endpoints, so the reading survives being
        scrolled off screen and can be compared against an earlier one; the
        label alone only ever shows the most recent measurement."""
        a_x, a_y = measurement.a
        b_x, b_y = measurement.b
        self._log_status(
            f"Ruler: {ruler.format_measurement(measurement)} "
            f"from ({a_x}, {a_y}) to ({b_x}, {b_y})"
        )

    def on_ruler_changed(self, measurement) -> None:
        """MapView's live-update callback: fires on every drag frame, not
        just a completed one, so this is what keeps the status bar in sync
        while the on-map label is off screen. `measurement` is None on every
        clear (Escape, right-click, tool/mode switch, File > Close/New)."""
        self.ruler_status_label.setText(
            f"  {ruler.format_measurement(measurement)}  " if measurement is not None else ""
        )

    def _update_log_min_height(self) -> None:
        """Two lines of the log's CURRENT font, re-run whenever the UI font
        changes so the floor doesn't stay at construction-time metrics.

        setMinimumHeight, not the setFixedHeight this had while the log was a
        plain layout row: a fixed height sets min == max, which pins a
        QSplitter child and leaves the handle looking draggable but inert.
        The persisted log_height is deliberately not rescaled with the font:
        it is a splitter position the user dragged, and moving it under them
        would be worse than a pane they re-drag once."""
        line_height = self.status_log.fontMetrics().lineSpacing()
        self.status_log.setMinimumHeight(max(settings.MIN_LOG_PANE, line_height * 2 + 12))

    def _log_status(self, message: str) -> None:
        self.status_log.appendPlainText(message)
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
        debug_log.log(message)

    @property
    def _render_style(self) -> str:
        """Which projection the renderer builds for, as opposed to
        self._terrain_style (what the user picked in the Elevation View
        combo). They differ only for Flat + Isometric View on: that
        combination renders through a real IsoChunkCache built on an
        all-zero elevation array (Flat+Isometric plan), so every RENDERING
        read (cache construction, dirty-bbox dispatch, the value passed as
        MapView.set_source(terrain_style=...)) must use this, never
        self._terrain_style directly. Every UI-facing read (the checkbox's
        enabled state, the status bar, the combo-change handler) stays on
        self._terrain_style, since those describe what the user picked, not
        what got built -- see the Flat+Isometric plan's F1 for the full
        split."""
        if self._terrain_style == "flat" and self.iso_action.isChecked():
            return "stepped"
        return self._terrain_style

    @property
    def _style_log_label(self) -> str:
        """self._terrain_style, annotated for a log line when it and
        self._render_style disagree -- bare "stepped" would claim the wrong
        style outright, and bare "flat" would hide that a rebuild is
        happening at all."""
        if self._render_style != self._terrain_style:
            return f"{self._terrain_style} (isometric)"
        return self._terrain_style

    def _cancel_warms(self) -> None:
        """Drops any in-flight level warm AND margin-ring chunk warm --
        called by every path that mutates the scenario, mutates the
        elevations array, or replaces the cache, BEFORE it mutates.

        That ordering is the whole race argument: a warm slice/chunk only
        ever resumes between event-loop iterations, so if every mutating
        path cancels first, a slice/chunk and a mutation can never
        interleave, and none of a threaded design's locks/snapshots/tokens
        are needed. The install predicates in render_cache's
        level_warm_job() are the second line of defence for a level-warm
        path nobody enumerated, not the first; the margin warm has no
        install step to revalidate (get_chunk() populates the cache
        directly), so its own second line of defence is
        is_level_resident() at start() time instead.

        One body with three cancels (rather than nine new call sites
        enumerated separately) is what keeps this correct as the set of
        warms grows: the correctness argument is "every mutating path
        cancels first," and that can't drift out of sync with itself. Also
        resets the margin warm's pan-direction baseline (see
        self._last_viewport_chunk_target's own comment) -- a stale one just
        means the next ring after this mutation starts with no direction
        preference, never a wrong answer. Drops _load_warm_queue too: a
        pending (mip, chunks) pair that never got to _load_warmer.start()
        is exactly as stale as one already ticking.

        Cheap and idempotent, so an over-broad call site costs at most a warm
        that has to be restarted on the next open -- always the right side to
        err on here."""
        self._level_warmer.cancel()
        self._margin_warmer.cancel()
        self._load_warmer.cancel()
        self._load_warm_queue = []
        self._last_viewport_chunk_target = None

    def _start_level_warm(self) -> None:
        """Queues the neighbouring mip levels' sprite layers for an idle-time
        warm -- the 2026-09-04 plan's hook point.

        Called from load_scenario AFTER _render_current(), not from inside
        it: _render_current is also the re-render path for every terrain-
        style/quality/slider change, and those already cancel a warm rather
        than starting one.

        Also called at the tail of _apply_dirty() and _after_unit_mutation()
        (Batch B step B1), the deliberate exception to that: an edit cancels
        the warm and then re-arms it, so the neighbour mips do not stay cold
        for the rest of the session once the first edit lands. A stroke
        therefore pays a start/cancel pair per step, which the plan accepts
        pending a Perf Trace pass. The style/quality re-render paths reached
        through _render_current still only cancel.

        Silent no-op with no map, no viewport (every headless test, and a
        window not yet shown) or a single-level ladder -- _fit_baseline_scale
        returns None for the first two, which is why the level is derived
        from it rather than from mip_for_scale, whose clamp would happily
        return a fabricated level 0 instead."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        fit = self.map_view._fit_baseline_scale()
        if fit is None:
            return
        # devicePixelRatio for the same reason CanvasItem._mip_for_painter
        # reads deviceTransform() rather than worldTransform(): the level a
        # paint actually selects is the DEVICE-space one, so a HiDPI window
        # would otherwise warm the neighbours of a level it never paints.
        mips = level_warm.neighbour_mips(self._cache, fit * self.map_view.devicePixelRatioF())
        self._level_warmer.start(self._cache, mips, on_job_done=self._queue_load_warm)
        # A mip level_warm_job() found nothing to warm for (already current)
        # fires no on_job_done at all -- check its residency here, once,
        # rather than the callback ever having to distinguish "no job" from
        # "job not done yet". The common file-open case is neither (every
        # neighbour mip starts non-resident), so this loop is usually a
        # no-op; see _queue_load_warm's own guard for why it's still safe
        # to call unconditionally.
        for mip in mips:
            self._queue_load_warm(mip)

    def _queue_load_warm(self, mip: int) -> None:
        """Queues the load-time chunk warm for one neighbour mip once its
        sprite layer is actually resident (2026-09-07 plan's load-time
        margin warm, Step 2/3) -- _start_level_warm's LevelWarmer.start()
        on_job_done callback, and also called directly from there for a mip
        already resident when the level warm started.

        Re-checks is_level_resident(mip) itself rather than trusting "the
        job finished" (on_job_done fires on a dropped job too, and a job's
        own install() can return False when a real paint got there first --
        in that second case the level IS resident, just not because of
        this warm, and this must still proceed). Silent no-op with no
        cache, the setting off, a still-not-resident mip, or a degenerate/
        off-grid projection at that mip -- viewport_chunk_target_at()
        returns None for the last case, mirroring _on_viewport_changed's
        own guard shape.

        viewport_chunk_target_at()'s raw range is bounded to a real
        viewport's worth (margin_warm.bounded_chunk_range(), sized via
        MapView.viewport_chunk_span()) before it ever reaches
        load_warm_chunks() -- load_scenario() always calls
        _start_level_warm() right after a fit-to-view render, so the raw
        range is the WHOLE neighbour-mip grid, not the ~20-chunk patch this
        feature is sized for. See bounded_chunk_range's own docstring."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        if not self._cache.is_level_resident(mip):
            return
        target = self.map_view.viewport_chunk_target_at(mip)
        if target is None:
            return
        span_w, span_h = self.map_view.viewport_chunk_span(self._cache, mip)
        cx0, cy0, cx1, cy1 = margin_warm.bounded_chunk_range(*target, span_w, span_h)
        chunks = margin_warm.load_warm_chunks(self._cache, mip, cx0, cy0, cx1, cy1)
        if not chunks:
            return
        self._load_warm_queue.append((mip, chunks))
        self._pump_load_warm()

    def _pump_load_warm(self) -> None:
        """Starts the next queued (mip, chunks) pair on _load_warmer, if it
        isn't already busy with one -- the sequencing _queue_load_warm needs
        because up to two neighbour mips can each become ready at different
        times, and _load_warmer.start() (MarginWarmer's own contract)
        replaces whatever is in flight rather than queuing alongside it.
        Chained off _load_warmer's own on_drained callback, so the second
        mip's chunks start the moment the first mip's queue empties, with
        no polling."""
        if self._load_warmer.is_active or not self._load_warm_queue:
            return
        mip, chunks = self._load_warm_queue.pop(0)
        self._load_warmer.start(self._cache, mip, chunks, on_drained=self._pump_load_warm)

    def _on_viewport_changed(self) -> None:
        """The margin warm's entry point (2026-09-07 plan's A2.5/A3) -- wired
        to self.map_view.on_viewport_changed, called whenever a viewport
        poll fire finds a new viewport_chunk_target().

        Silent no-op with no cache, the setting off, or a degenerate
        viewport (viewport_chunk_target() returns None) -- mirroring
        _start_level_warm's own guard shape so a headless/never-shown
        window (every default-tier test) stays safe.

        `lead` is the sign pair of the viewport origin's movement since the
        LAST fire this method itself recorded -- not MapView's own
        _last_viewport_target, which exists only to decide when the poll
        should stop and would go stale the moment this method's fire
        target catches up to it. Falls back to (0, 0) (see
        margin_warm.ring_chunks' own docstring for why that's a correct
        answer, not a missing one) on the first fire, right after a cancel,
        or whenever the mip itself changed -- a pan direction from a
        DIFFERENT level's chunk grid means nothing here."""
        if self._cache is None or not settings.get_preload_zoom_levels():
            return
        target = self.map_view.viewport_chunk_target()
        if target is None:
            return
        mip, cx0, cy0, cx1, cy1 = target
        prev = self._last_viewport_chunk_target
        self._last_viewport_chunk_target = target
        lead = (0, 0)
        if prev is not None and prev[0] == mip:
            prev_cx0, prev_cy0 = prev[1], prev[2]
            lead = (
                (cx0 > prev_cx0) - (cx0 < prev_cx0),
                (cy0 > prev_cy0) - (cy0 < prev_cy0),
            )
        ring = margin_warm.ring_chunks(self._cache, mip, cx0, cy0, cx1, cy1, lead=lead)
        self._margin_warmer.start(self._cache, mip, ring)

    def _on_canvas_paint_timed(self, elapsed: float, mip: int) -> None:
        """MapCanvasItem's per-paint stopwatch callback (installed by
        _render_current, reported on only by load_scenario).

        Every paint counts, and none is filtered: no property of a single
        paint says "this was the real one". render_rect() returns pixels
        whether it composited them now or hit the LRU, and a paint of an 8px
        sliver is indistinguishable up front from the fit-to-view composite.
        Summing until the event loop goes idle needs no such proxy: the
        sliver contributes its 0.01s, the composite contributes its 5.10s,
        and the total is right in either order.

        The timer is armed only once load_scenario() has marked the report
        ready, so a paint that fires synchronously inside _render_current()'s
        own processEvents() (which does dispatch zero-delay timers) still
        accumulates but cannot print ahead of the load's own line."""
        report = self._pending_paint_report
        if report is None:
            return
        report["paint"] += elapsed
        report["paints"] += 1
        # The mip reported is the costliest paint's, not the last one's: a
        # trailing sliver repaint can select a different level.
        if elapsed >= report["max_paint"]:
            report["max_paint"] = elapsed
            report["mip"] = mip
        if report["ready"]:
            self._paint_report_timer.start()

    def _emit_paint_report(self) -> None:
        """Prints the deferred-composite follow-up to a load's `Loaded ...`/
        `Created ...` line, once painting has stopped, then uninstalls the
        stopwatch so steady-state painting is back to one `is not None`
        check.

        `total` is parse + prepare + summed paint time, i.e. work actually
        done, not wall clock from load start to here, which would fold in
        however long Qt sat idle and make the number a reading of how busy
        the machine was rather than of the file."""
        report = self._pending_paint_report
        self._pending_paint_report = None
        self.map_view.set_paint_timed_callback(None)
        if report is None or not report["paints"]:
            return
        total = report["parse"] + report["prepare"] + report["paint"]
        count = report["paints"]
        counted = "" if count == 1 else f", {count} paints"
        self._log_status(
            f"First paint composited in {report['paint']:.2f}s "
            f"(total {total:.2f}s, mip {report['mip']}{counted})"
        )

    def _render_current(self, *, reset_view: bool = True) -> tuple[float, int]:
        """Renders/prepares self.scenario at the currently selected Terrain
        Style and pushes it to self.map_view -- the render+display step
        shared by load_scenario/refresh_map/on_terrain_style_changed/
        _apply_dirty's full-redraw fallback, each of which wants a
        differently-worded status log line, so that part stays theirs.
        Returns (elapsed_seconds, tile_px).

        reset_view is threaded straight through to MapView.set_source() --
        see that parameter's own docstring. Default True matches only
        load_scenario() (a genuinely new document); _apply_dirty's
        fallback, refresh_map(), and on_terrain_style_changed() all pass
        False to keep the zoom/pan the user already had.

        Neither style blocks on the full canvas anymore as of Phase B-E
        (Flat was the last one still doing so -- render_scenario() always
        compositing the whole map up front, ~1-2s for a real map). Stepped
        (Phase B-C): elevations_and_proj() reads elevations and sizes the
        projection WITHOUT compositing a single pixel. Flat (Phase B-E):
        FlatChunkCache's own construction is similarly cheap -- it only
        precomputes unit_draws (_flat_unit_draws()), not any pixels. Both
        chunk caches then composite lazily, chunk by chunk, only for
        whatever MapCanvasItem's paint() actually asks for.
        This is the whole point of Track B:
        opening/switching to either view no longer pays for the full
        canvas up front.

        setEnabled(False) for the same span, not just the cursor: processEvents()
        re-enters the event loop, which would otherwise let a second click
        (Open, the Elevation View combo, a toolbar tool) dispatch a
        reentrant call into this same render path mid-flight -- e.g. a
        second render starting while self._cache/self.map_view are still
        being written by the first. Now purely a reentrancy guard rather
        than also covering a visible freeze -- the wait cursor still shows
        briefly for the cache construction above, but there's no multi-
        second compositing pass left for the window to actually appear
        frozen during. A disabled window still repaints (the cursor/log
        message stay visible), it just stops accepting input."""
        # Unconditional, and FIRST: this method replaces self._cache outright
        # on every path below, so anything still warming the old one is
        # warming a cache nothing will ever paint from. Covers refresh_map,
        # the terrain-style combo, the graphics-quality and elev-step
        # sliders, the Isometric View toggle and a re-entrant load in one
        # place, rather than one call per caller.
        self._cancel_warms()
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            # Armed before building the cache, not after: the deferred
            # composite cost this exists to capture happens on the FIRST
            # paint of the new MapCanvasItem set_source() below constructs,
            # which can fire before this method returns (a synchronous
            # processEvents() elsewhere in the call stack). reset_view is
            # True only for load_scenario's genuinely-new-document call;
            # every other caller re-renders the same document. The
            # _on_canvas_paint_timed stopwatch every set_source() below
            # installs is unconditional for the same reason: it no-ops
            # unless load_scenario() left a report pending.
            perf_trace.arm("load" if reset_view else "re-render")
            mm = self.scenario.map_manager
            tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
            if self._render_style == "stepped":
                elevations, proj = elevations_and_proj(self.scenario)
                # Flat + Isometric View (Flat+Isometric plan): force every
                # tile's elevation to 0 so terrain paints as a real iso
                # projection at elevation 0 rather than the scenario's real
                # heights -- proj is sized from the fixed legal constants
                # (MIN_ELEVATION/MAX_ELEVATION), never from observed data, so
                # an all-zero array yields a proj field-identical to real
                # Stepped's. self._terrain_style, not self._render_style: this
                # branch is also real Stepped, which must NOT flatten.
                if self._terrain_style == "flat":
                    elevations = np.zeros_like(elevations)
                self._iso_elevations, self._iso_proj = elevations, proj
                # sprites= at CONSTRUCTION rather than a set_sprites_enabled()
                # call afterwards: the setter's job is to rebuild and evict a
                # live cache, all of which this fresh one would be paying for
                # nothing. Same reason unit_filter is passed here too.
                self._cache = IsoChunkCache(
                    self.scenario, elevations, proj, tile_px,
                    unit_filter=self._unit_filter, sprites=self._sprites_enabled,
                    layers=self._layers,
                )
                self.map_view.set_source(
                    tile_px, terrain_style="stepped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view, on_paint_timed=self._on_canvas_paint_timed,
                )
            elif self._render_style == "sloped":
                # Same snapshot contract as Stepped above, and it must be the
                # SAME array object the cache and MapView hold (Risk #6): the
                # sloped patch path mutates it in place, so a copy here would
                # let the pick plane drift from the pixels.
                elevations, corner_rise, proj = sloped_elevations_and_proj(self.scenario)
                self._iso_elevations, self._iso_proj = elevations, proj
                # sprites= at CONSTRUCTION (Track P3-g6), same reason the
                # Stepped branch above passes it there rather than calling
                # set_sprites_enabled() afterwards: that setter's job is to
                # rebuild and evict a LIVE cache, all of which this fresh one
                # would be paying for nothing.
                self._cache = SlopedChunkCache(
                    self.scenario, elevations, corner_rise, proj, tile_px, unit_filter=self._unit_filter,
                    sprites=self._sprites_enabled, layers=self._layers,
                )
                self.map_view.set_source(
                    tile_px, terrain_style="sloped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view, on_paint_timed=self._on_canvas_paint_timed,
                )
            else:
                self._iso_elevations, self._iso_proj = None, None
                # sprites= at CONSTRUCTION (Track P3-g7), same reason the two
                # branches above pass it there rather than calling
                # set_sprites_enabled() afterwards.
                self._cache = FlatChunkCache(
                    self.scenario, tile_px, unit_filter=self._unit_filter,
                    sprites=self._sprites_enabled, layers=self._layers,
                )
                self.map_view.set_source(
                    tile_px, terrain_style="flat", cache=self._cache, reset_view=reset_view,
                    on_paint_timed=self._on_canvas_paint_timed,
                )
            # set_source() drops the pick index along with every other scene
            # item, so a style switch made while in Units mode has to rebuild
            # it -- otherwise selection silently stops working until the mode
            # is toggled off and back on.
            if self._needs_unit_index():
                self._rebuild_unit_index()
            # set_source() rebuilt the markers from MapView's own stored
            # list, which a NEW document's has not replaced yet -- so push
            # this scenario's. One call covers load, New Map, a map resize
            # and a style switch.
            self._refresh_camera_markers()
            return time.perf_counter() - t0, tile_px
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def refresh_map(self) -> None:
        # Re-render the currently open map, if any, so a settings change (e.g.
        # a newly-configured install path) takes effect immediately instead of
        # only on the next file open. self._busy guard: see load_scenario's
        # own comment on it -- same reentrancy concern, one shared flag.
        # reset_view=False: same document, same style, just picking up a
        # setting -- nothing about that should yank the user back to fit.
        if self.scenario is not None and not self._busy:
            self._busy = True
            try:
                elapsed, tile_px = self._render_current(reset_view=False)
                self._log_status(
                    f"Re-rendered map (tile_px={tile_px}, style={self._style_log_label}) "
                    f"prepared in {elapsed:.2f}s"
                )
            finally:
                self._busy = False
            # Re-gate too, not just re-render. "Show sprites" is the one action
            # whose enablement depends on the install path, and its greyed-out
            # tooltip tells the user to go configure one -- so without this,
            # following that instruction re-renders the map and leaves the
            # toggle stubbornly greyed until the next file open.
            self._update_tool_enabled()

    def open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scenario", "", "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        self.load_scenario(Path(path))

    def _rebuild_recent_files_menu(self) -> None:
        """Repopulates File > Open Recent from settings.get_recent_files().
        Called at menu-bar build time and again after every successful open
        -- QMenu keeps no live binding to the underlying list, so this is
        the one place the widget is kept in sync with it. Filters out
        entries whose file no longer exists rather than pruning them from
        the persisted list, so a file on a removable drive reappears once
        it's back."""
        self.recent_menu.clear()
        paths = [p for p in settings.get_recent_files() if Path(p).is_file()]
        if not paths:
            empty_action = QAction("(No recent files)", self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)
            return
        for path_str in paths:
            action = QAction(Path(path_str).name, self)
            action.setToolTip(path_str)
            # Marks this as a real file entry, distinct from the "Clear
            # Recent Files" action below and the disabled placeholder above
            # -- both of which would otherwise be indistinguishable from a
            # file action by toolTip() alone, since Qt falls back to the
            # (mnemonic-stripped) text when no tooltip was set explicitly.
            action.setData(path_str)
            action.triggered.connect(lambda checked=False, p=path_str: self._open_recent(p))
            self.recent_menu.addAction(action)
        self.recent_menu.addSeparator()
        clear_action = QAction("&Clear Recent Files", self)
        clear_action.triggered.connect(self._clear_recent_files)
        self.recent_menu.addAction(clear_action)

    def _open_recent(self, path_str: str) -> None:
        if not self._confirm_discard_changes():
            return
        self.load_scenario(Path(path_str))

    def _clear_recent_files(self) -> None:
        settings.clear_recent_files()
        self._rebuild_recent_files_menu()

    def new_map(self, tiles: int = BLANK_TEMPLATE_TILES) -> None:
        """Starts a blank `tiles`x`tiles` map by generating it in memory from
        the shipped 120x120 donor template (descape.scenario_new.
        blank_scenario_bytes -- one of STANDARD_MAP_SIZES via File > New Map's
        size submenu, or any size in [MIN_MAP_TILES, MAX_MAP_TILES] via
        new_map_custom()) and loading the result through the normal load path,
        then marking the document untitled -- see load_scenario()'s `untitled`
        handling. Not built from AoE2ScenarioParser's own from_default():
        that would need a second write path through the library's own
        (unverified, non-byte-stable) serializer instead of the already
        in-game-verified byte-patch path this reuses unchanged.

        The new map inherits the donor's FileHeader verbatim (creator name,
        timestamps) and DataHeader.filename -- the byte-patch splice can't
        rewrite those by design: filename is DataHeader's terminal,
        length-prefixed field, and rewriting it to a different-length name
        would shift every section offset after it. Inherent to reusing a
        donor rather than a defect in it.
        """
        if not self._confirm_discard_changes():
            return
        self._create_new_map(tiles)

    def new_map_custom(self) -> None:
        """File > New Map > Custom size…. Split out from new_map() so the
        size policy (scenario_new.validate_tiles) is testable with no dialog
        at all. Confirms first for anything larger than
        LARGE_MAP_CONFIRM_TILES -- deliberately only on this path, never for
        the labelled preset entries (including 480): a preset is a deliberate
        choice on a size already in-game proven, so a confirm there would
        just be a nag, while a typed-in custom size has no such history.

        Checks _confirm_discard_changes() up front, before either dialog, and
        calls _create_new_map() directly rather than new_map() -- new_map()
        would check discard again, which would mean answering it twice (once
        here, once more inside new_map()) whenever the document is dirty."""
        if not self._confirm_discard_changes():
            return
        tiles, ok = QInputDialog.getInt(
            self,
            "Custom map size",
            f"Map size (tiles, square, {MIN_MAP_TILES}-{MAX_MAP_TILES}):",
            BLANK_TEMPLATE_TILES,
            MIN_MAP_TILES,
            MAX_MAP_TILES,
            1,
        )
        if not ok:
            return
        if tiles > LARGE_MAP_CONFIRM_TILES:
            reply = QMessageBox.question(
                self,
                "Large map",
                f"{tiles}×{tiles} is larger than the {LARGE_MAP_CONFIRM_TILES}×"
                f"{LARGE_MAP_CONFIRM_TILES} standard size. Large maps take "
                "longer to create and render, and use significantly more "
                "memory. Create it anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        self._create_new_map(tiles)

    def _create_new_map(self, tiles: int) -> None:
        """The generate-and-load half of File > New Map, shared by new_map()
        and new_map_custom() -- both of which have already run
        _confirm_discard_changes() themselves before calling this, so it
        does no confirming of its own."""
        try:
            data = blank_scenario_bytes(tiles)
        except (MapSizeError, BlankGenerationError) as e:
            self._log_status(f"New Map failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "New Map failed", str(e))
            return
        self.load_scenario(UNTITLED_PATH, untitled=True, data=data)

    def load_scenario(self, path: Path, *, untitled: bool = False, data: bytes | None = None) -> None:
        # Loading + rendering a large map is a multi-second blocking call
        # (see _render_current()'s own docstring) -- the log line and status
        # bar message below are queued but not actually painted until
        # control returns to the event loop, so without this
        # setOverrideCursor()+processEvents() pair the window would just sit
        # frozen with no visible indication anything is happening until the
        # whole load completes. self._busy (see its own comment in
        # __init__) plus setEnabled(False) together guard against the same
        # reentrancy risk _render_current() documents for its own
        # processEvents() call -- e.g. a second Open dispatched from inside
        # this one's processEvents() while self.scenario is still being
        # reassigned. A true no-op, not just queued: checked before
        # anything else runs.
        if self._busy:
            return
        self._busy = True
        self._close_analysis_dialog()
        # Dropped before anything else: a previous load whose paint never
        # came (or came late) must not have its line attributed to this file.
        self._paint_report_timer.stop()
        self._pending_paint_report = None
        try:
            self._log_status(f"Loading scenario: {path}")
            self.statusBar().showMessage(
                "Creating new map..." if untitled else f"Loading {path.name}..."
            )
            self.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                t_parse0 = time.perf_counter()
                self.scenario = (
                    load_map_and_units_from_bytes(data, path)
                    if data is not None
                    else load_map_and_units(path)
                )
                parse_elapsed = time.perf_counter() - t_parse0
            except UnsupportedStructureVersion as e:
                # Named cause instead of the library's raw exception text: a
                # v1.32/v1.35 file is unopenable by design, not a broken file.
                self.statusBar().clearMessage()
                self._log_status(
                    f"Failed to load {path}: unsupported scenario version {e.scenario_version}"
                )
                QMessageBox.critical(self, "Failed to load", str(e))
                return
            except Exception as e:  # noqa: BLE001 -- GUI boundary, reports below
                self.statusBar().clearMessage()
                self._log_status(f"Failed to load {path}: {type(e).__name__}: {e}")
                QMessageBox.critical(self, "Failed to load", f"{type(e).__name__}: {e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
                self.setEnabled(True)

            # Only reached on a successful load, so a failed File > New can never
            # mislabel whatever was previously open (self.scenario still holds it,
            # untouched, if the try block above returned early).
            self._untitled = untitled
            if untitled:
                self.scenario.path = UNTITLED_PATH
            else:
                settings.add_recent_file(path)
                self._rebuild_recent_files_menu()
            display_name = self.scenario.path.name

            self.edit_history.reset()
            # A new document is a new autosave key and a fresh "never
            # autosaved" state; the untitled key rides _doc_id, so reusing
            # the old one would rotate a different document's slots.
            self._doc_id = uuid4().hex
            self._autosaved_at_cursor = None
            self._stroke_seen_since_retry = False
            # Dropped alongside the history it pushes records onto: a model
            # held across a document switch would splice the previous file's
            # trigger bytes into this one, patch the previous file's option
            # offsets into it, or splice the previous file's unit bytes in.
            self.trigger_edits = None
            self.option_edits = None
            self.unit_edits = None
            self.message_edits = None
            # Map-relative, like the hover position -- a region rect from
            # whatever was open before must not outlive it (the new map may
            # not even be big enough to contain it). The clipboard is NOT
            # reset here; see close_scenario()'s own comment on why it
            # survives.
            self._region = None
            # The trigger it named belongs to the old document, as do the picker and the references.
            self._trigger_overlay_ref = None
            self.disarm_unit_picker()
            self._unit_ref_index = None

            # Opened before the render: the first paint can fire inside
            # _render_current()'s processEvents(), and counts as this load's.
            self._pending_paint_report = {
                "parse": parse_elapsed, "prepare": 0.0, "paint": 0.0,
                "paints": 0, "max_paint": -1.0, "mip": 0, "ready": False,
            }
            # Renders at whichever Terrain Style was already selected --
            # File > Open doesn't reset it back to Flat. _render_current()
            # pushes its own wait cursor/setEnabled(False) for this step,
            # nested safely inside this method's own (see both docstrings).
            # The "Loading..." status bar message deliberately stays up
            # through this render step too, not just the parse above --
            # cleared only once everything is actually done, right below.
            elapsed, tile_px = self._render_current()
            self._pending_paint_report["prepare"] = elapsed
            # Here rather than inside _render_current(): this is the one
            # render path that opens a document the user is about to zoom
            # around in. _render_current()'s other callers are re-renders
            # that cancel a warm instead (see its own first line).
            self._start_level_warm()
            mm = self.scenario.map_manager
            self._update_info()
            # Only when the panel is actually on screen -- otherwise opening a
            # map would parse a Triggers section nobody asked to see.
            if self.mode == "triggers":
                self._show_triggers()
            else:
                self.trigger_panel.clear_document()
            # Its own if/else rather than an elif chained onto the one above:
            # chaining would leave whichever panel lost the race populated
            # with the previous document.
            if self.mode == "map_options":
                self._show_map_options()
            else:
                self.map_options_panel.clear_document()
            # Its own if/else too, for the same reason as the map_options
            # pair above -- test_opening_a_second_map_while_in_the_mode_
            # repopulates pins this for Players mode as well.
            if self.mode == "players":
                self._show_players()
            else:
                self.players_panel.clear_document()
            # Its own if/else too, for the same reason as the two pairs
            # above.
            if self.mode == "diplomacy":
                self._show_diplomacy()
            else:
                self.diplomacy_panel.clear_document()
            # Its own if/else too, for the same reason as the pairs above.
            if self.mode == "messages":
                self._repopulate_messages()
            else:
                self.messages_panel.clear_document()
            self._update_tool_enabled()
            self._update_edit_actions()
            self._update_title()
            self.statusBar().clearMessage()
            total_elapsed = parse_elapsed + elapsed
            self._log_status(
                f"{'Created' if untitled else 'Loaded'} {display_name} "
                f"({mm.map_width}x{mm.map_height} tiles, "
                f"{sum(len(u) for u in self.scenario.unit_manager.units):,} units, "
                f"tile_px={tile_px}, style={self._style_log_label}) "
                f"in {total_elapsed:.2f}s (parse {parse_elapsed:.2f}s, prepare {elapsed:.2f}s)"
            )
            # Only now, with that line printed, may the follow-up drain. No
            # paint at all (a window never shown) means no second line.
            self._pending_paint_report["ready"] = True
            if self._pending_paint_report["paints"]:
                self._paint_report_timer.start()
            if not self.scenario.terrain_write_supported:
                self._log_status(
                    f"Warning: {display_name}'s terrain block failed load-time verification -- "
                    "Terrain mode's terrain/elevation tools and Save/Save As are disabled for "
                    "this file (still fully viewable)."
                )
            elif not self.scenario.map_is_square:
                # Non-square is NOT actually supported today, despite what this message
                # used to claim: AoE2ScenarioParser's MapManager.map_size raises
                # ValueError("Map is not a square...") from tile.x/tile.y/get_tile(),
                # reached at ~15 call sites across render.py/viewer.py/elevation_tools.py/
                # batch_api.py -- so a non-square map fails during the render this same
                # load_scenario() call triggers below, before this note would ever help.
                # A possible follow-up: deriving tile coordinates from the terrain
                # index instead of these library properties.
                self._log_status(
                    f"Note: {display_name} is a non-square map ({mm.map_width}x{mm.map_height}) -- "
                    "non-square maps are not supported and will likely fail to render."
                )
        finally:
            self._busy = False

    def close_scenario(self) -> None:
        if self.scenario is None:
            return
        if not self._confirm_discard_changes():
            return
        name = self.scenario.path.name
        self.scenario = None
        self._close_analysis_dialog()
        self._untitled = False
        # Dropped alongside the cache it was warming -- otherwise a closed
        # document keeps ticking, holding the whole scenario alive.
        self._cancel_warms()
        self._cache = None
        self._iso_elevations, self._iso_proj = None, None
        self.edit_history.reset()
        self._doc_id = uuid4().hex
        self._autosaved_at_cursor = None
        self._stroke_seen_since_retry = False
        self._autosave_retry_timer.stop()
        self.trigger_edits = None
        self.option_edits = None
        self.unit_edits = None
        self.message_edits = None
        self._trigger_overlay_ref = None
        self.disarm_unit_picker()
        self._unit_ref_index = None
        self.map_view.clear_image()
        # No edit tool has anything to act on with no map open; forcing Pan
        # (rather than just disabling the edit tools) keeps MapView's own
        # tool state in sync too -- same mechanism on_mode_changed already
        # uses when Terrain mode becomes unavailable.
        self.pan_action.setChecked(True)
        self.info.setPlainText("")
        self._repopulate_stats_players()  # scenario is None: clears and disables the combo
        self.trigger_panel.clear_document()
        self.map_options_panel.clear_document()
        self.players_panel.clear_document()
        self.diplomacy_panel.clear_document()
        self.messages_panel.clear_document()
        self._set_hover_text(HOVER_IDLE_TEXT)
        # A stale (x, y) from the just-closed map must not outlive it -- the
        # bounds check in paste_region()/copy_region() would likely catch a
        # mismatch against a differently-sized map opened next anyway, but
        # relying on that coincidence is exactly the kind of leak
        # edit_history.reset() above is already here to prevent for edit
        # history. The clipboard HISTORY deliberately survives a close, for
        # the same reason the single slot it replaced did (a
        # clipboard outliving the file it was copied from is normal
        # clipboard semantics, and Paste is already disabled with no map
        # loaded via _update_tool_enabled() below) -- self._region, unlike
        # the clipboard, IS map-relative (a stale tile rect indexing a map
        # that no longer exists), so it clears here alongside the hover
        # position. map_view.clear_image() above already dropped its own
        # rendering copy of the same value.
        self._hover_tile = None
        self._region = None
        self._update_tool_enabled()
        self._update_edit_actions()
        self._update_title()
        self._log_status(f"Closed {name}")

    def _update_info(self) -> None:
        """Both halves of page 0: the static text below, then the player
        stats block via _repopulate_stats_players()."""
        s = self.scenario
        mm = s.map_manager

        terrain_hist = Counter(t.terrain_id for t in mm.terrain)
        lines = [
            f"File: {s.path.name}",
            f"Scenario version: {s.scenario_version}",
            f"Map size: {mm.map_width} x {mm.map_height}",
            f"Trigger tail (not parsed): {len(s.trigger_tail):,} bytes",
            *_xs_info_lines(s),
            "",
            "Terrain (top 8):",
        ]
        total_tiles = len(mm.terrain)
        for tid, count in terrain_hist.most_common(8):
            lines.append(f"  {name_for_terrain_id(tid):24s} {100 * count / total_tiles:4.1f}%")

        self.info.setPlainText("\n".join(lines))
        # The trigger-tail/XS lines above and the stats block's trigger row flip together.
        self._repopulate_stats_players()

    def _stats_player_label(self, player_id: int, active: set[int]) -> str:
        if player_id == GAIA_PLAYER_ID:
            return "GAIA"
        return f"Player {player_id}" if player_id in active else f"Player {player_id} (inactive)"

    def _repopulate_stats_players(self) -> None:
        """GH #5's combo: GAIA, the defined players as edited, then any
        inactive slot that still owns placements. Keeps the selected player
        when it survives, then re-renders the stats block."""
        combo = self.stats_player_combo
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if self.scenario is not None:
            active = self._current_active_players()
            units = self.scenario.unit_manager.units
            orphaned = [p for p in range(1, len(units)) if units[p] and p not in active]
            active_set = set(active)
            for player_id in [GAIA_PLAYER_ID, *active, *orphaned]:
                combo.addItem(
                    _swatch_icon(self.scenario.player_colors[player_id]),
                    self._stats_player_label(player_id, active_set),
                    player_id,
                )
            index = combo.findData(previous) if previous is not None else -1
            if index < 0:
                # Player 1 by default, else GAIA.
                index = 1 if combo.count() > 1 else 0
            combo.setCurrentIndex(index)
        combo.setEnabled(combo.count() > 0)
        combo.blockSignals(False)
        self._update_player_stats()

    def _refresh_stats_swatches(self) -> None:
        if self.scenario is None:
            return
        combo = self.stats_player_combo
        for i in range(combo.count()):
            combo.setItemIcon(i, _swatch_icon(self.scenario.player_colors[combo.itemData(i)]))

    def _update_player_stats(self) -> None:
        """Renders the selected player's breakdown (GH #5). Cheap enough to
        run on every unit mutation: a walk of one player's list plus, only
        when the Triggers section is already parsed, a memoized re-read."""
        player_id = self.stats_player_combo.currentData()
        if self.scenario is None or player_id is None:
            self.player_stats_rows = []
            self.player_stats_label.setText("")
            return
        c = player_stats.counts_for(self.scenario, player_id)
        # (label, count, note); a row with no count shows its note in the count's place.
        rows = [
            ("Placements", f"{c.placements:,}", ""),
            ("Units", f"{c.units:,}", ""),
            ("Buildings", f"{c.buildings:,}", f"(walls & gates {c.walls:,})"),
            ("Trees", f"{c.trees:,}", ""),
            ("Eye candy", f"{c.eye_candy:,}", ""),
        ]
        supported = self.scenario.trigger_read_supported
        if supported is None:
            rows.append(("Triggers", "", "not counted yet (enter Triggers mode or run Map Analysis)"))
        elif supported is False:
            rows.append(("Triggers", "", map_analysis._TRIGGERS_UNAVAILABLE))
        else:
            summary = player_stats.trigger_summary(self.scenario)
            if summary is None:
                rows.append(("Triggers", "", "no trigger vocabulary for this scenario version"))
            else:
                rows.append((
                    "Triggers",
                    f"{summary.per_player.get(player_id, 0):,}",
                    f"(of {summary.total:,}; {summary.without_player:,} reference no player)",
                ))
        self.player_stats_rows = rows
        # A table, not padded text: the UI font is proportional, so spaces can't align numbers.
        cells = []
        for label, count, note in rows:
            if count:
                cells.append(
                    f"<tr><td>{label}</td><td align='right'>&nbsp;&nbsp;{count}</td>"
                    f"<td>&nbsp;&nbsp;{html.escape(note)}</td></tr>"
                )
            else:
                cells.append(f"<tr><td>{label}</td><td colspan='2'>&nbsp;&nbsp;{html.escape(note)}</td></tr>")
        self.player_stats_label.setText(
            f"<b>{html.escape(self.stats_player_combo.currentText())}</b>"
            f"<table cellspacing='0' cellpadding='1'>{''.join(cells)}</table>"
        )

    def _set_hover_text(self, text: str) -> None:
        # Page 0's label and the Terrain page's one-line copy, so the readout survives the page swap.
        self.hover_label.setText(text)
        self.terrain_panel.set_hover_text(text)

    def on_hover(self, tile: tuple[int, int] | None) -> None:
        self._hover_tile = tile
        if self.scenario is None:
            return
        if tile is None:
            self._set_hover_text(HOVER_IDLE_TEXT)
            return
        x, y = tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            self._set_hover_text(HOVER_IDLE_TEXT)
            return
        # Flat indexing, not get_tile_safe(): that returns None for every tile of a non-square map.
        tile = mm.terrain[y * mm.map_width + x]
        self._set_hover_text(
            f"({x}, {y})  {name_for_terrain_id(tile.terrain_id)}  elevation={tile.elevation}"
        )


# Module-globals for the crash hook: it installs before QApplication exists
# (see install_crash_hooks()), so it can't hold a constructor-injected
# reference to anything -- these are its only state.
_crash_host_ref: weakref.ReferenceType[ViewerWindow] | None = None
_crash_rate_limiter = crash_report.RateLimiter()
_crash_in_handler = False
# Held for process lifetime so the fd outlives this function -- letting it
# get GC'd would close the file, and this handle is the only thing that
# produces anything for a hard abort (a bare Fatal Python error, not a
# catchable exception; see tests/conftest.py).
_faulthandler_log_handle = None


def register_crash_host(window: ViewerWindow) -> None:
    global _crash_host_ref
    _crash_host_ref = weakref.ref(window)


def _crash_host() -> ViewerWindow | None:
    return _crash_host_ref() if _crash_host_ref is not None else None


def _crash_dump_dir() -> Path:
    return asset_source.CONFIG_PATH.parent / "crashes"


def _qt_version_string() -> str:
    from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

    return f"PyQt5 {PYQT_VERSION_STR} / Qt {QT_VERSION_STR}"


def _open_crash_dialog(summary: str, dump_path: Path, dump_text: str) -> None:
    host = _crash_host()
    dialog = CrashReportDialog(host, summary=summary, dump_path=dump_path, dump_text=dump_text)
    dialog.exec_()


def _handle_exception(exc_type, exc_value, exc_tb) -> None:
    """Common body for sys.excepthook and threading.excepthook. Confirmed by
    probe: this PyQt5 does not abort the event loop on an unhandled
    exception, so the in-app dialog is the primary surface, not a fallback."""
    global _crash_in_handler
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    if _crash_in_handler:
        # A bug in this handler must not swallow the original traceback.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    _crash_in_handler = True
    try:
        fingerprint = crash_report.fingerprint(exc_type, exc_tb)
        if not _crash_rate_limiter.should_write(fingerprint):
            # Without this a raising paintEvent would write a dump and open
            # a dialog on every repaint.
            debug_log.log(f"Crash re-raised, already reported: {exc_type.__name__}")
            return
        dump_dir = _crash_dump_dir()
        text = crash_report.build_report(
            exc_type,
            exc_value,
            exc_tb,
            version=__version__,
            log_text=debug_log.get_log_text(),
            qt_version=_qt_version_string(),
            frozen=asset_source.IS_FROZEN,
        )
        dump_path = crash_report.write_report(text, dump_dir)
        crash_report.prune(dump_dir)
        debug_log.log(f"Crash report written to {dump_path}")
        summary = f"{exc_type.__name__}: {exc_value}"
        if _crash_host() is not None:
            # Queued, never inline: this hook can fire from inside Qt's own
            # paint/event dispatch, and a nested modal exec_() there is how
            # you get a second crash inside the first.
            QTimer.singleShot(0, lambda: _open_crash_dialog(summary, dump_path, text))
        # else: no host yet -- e.g. an exception during ViewerWindow's own
        # construction. By design, not a gap: singleShot(0, ...) would queue
        # into an event loop that never starts. The next-launch sweep covers
        # this case instead.
    except Exception:  # noqa: BLE001 -- last-resort handler; anything at all
        # here must fall through to the real excepthook rather than propagate.
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    finally:
        _crash_in_handler = False


def _handle_thread_exception(args) -> None:
    # threading.excepthook takes one ExceptHookArgs namedtuple, not three
    # positionals like sys.excepthook -- this adapter is the only reason the
    # two hooks can share _handle_exception.
    _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)


def install_crash_hooks() -> None:
    global _faulthandler_log_handle
    dump_dir = _crash_dump_dir()
    # Rotate first, then enable: opening the log for writing truncates it,
    # which would destroy a previous hard-abort's trace before the sweep
    # ever gets a chance to surface it.
    crash_report.rotate_faulthandler_log(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Deliberately not a context manager: faulthandler writes to this handle for
    # the process's whole life, so closing it would disarm the crash trace.
    _faulthandler_log_handle = open(dump_dir / crash_report.FAULTHANDLER_LOG_NAME, "w")  # noqa: SIM115
    faulthandler.enable(file=_faulthandler_log_handle)
    sys.excepthook = _handle_exception
    # Cheap insurance for a future worker thread, not a present need --
    # descape/ is currently single-threaded.
    threading.excepthook = _handle_thread_exception


def _sweep_pending_crash_reports() -> None:
    dump_dir = _crash_dump_dir()
    pending = crash_report.pending_reports(dump_dir)
    if not pending:
        return
    host = _crash_host()
    names = ", ".join(p.name for p in pending)
    debug_log.log(f"Crash report(s) found from a previous session: {names}")
    if host is not None:
        host._log_status(f"Crash report(s) found from a previous session: {names}")
    for path in pending:
        text = path.read_text(encoding="utf-8", errors="replace")
        dialog = CrashReportDialog(
            host,
            summary=crash_report.extract_summary(text),
            dump_path=path,
            dump_text=text,
            from_last_session=True,
        )
        dialog.exec_()
        crash_report.mark_reported(path)


def main() -> None:
    # First statement, not merely "before QApplication": migrate_legacy_
    # config() below does file I/O and can raise on its own, and the dump's
    # debug_log snapshot only exists if the hook outlives the first log line.
    install_crash_hooks()
    debug_log.log("Application started")
    debug_log.log(composite_backend.describe())
    migrated = asset_source.migrate_legacy_config()
    if migrated is not None:
        debug_log.log(f"Migrated config from {asset_source.LEGACY_CONFIG_PATH} to {migrated}")
    debug_log.log(f"Config file: {asset_source.CONFIG_PATH}")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "app_icon.png")))
    apply_theme(app, settings.get_dark_mode())
    # Before ViewerWindow(), so every widget is built at the final font and
    # nothing has to be re-measured after the fact.
    apply_ui_font(app, settings.get_ui_font_family(), settings.get_ui_font_size())
    window = ViewerWindow()
    window.show()
    ready_file = os.environ.get("DESCAPE_READY_FILE")
    if ready_file:
        Path(ready_file).touch()
    if len(sys.argv) > 1:
        window.load_scenario(Path(sys.argv[1]))
    # After the ready-file touch, not before: bootstrap.py blocks on that
    # touch to know the app is up, and a modal sweep dialog inserted earlier
    # would hold it up until READY_TIMEOUT fires -- on exactly the launch
    # mode (double-click via bootstrap) that hard-abort dumps come from.
    QTimer.singleShot(0, _sweep_pending_crash_reports)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
