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
import faulthandler
import os
import sys
import threading
import time
import weakref
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
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
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
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

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import (
    __version__,
    asset_source,
    brush,
    cliff_catalog,
    cliff_chain,
    crash_report,
    debug_log,
    diplomacy_fields,
    edge_ticks,
    iso_geometry,
    level_warm,
    margin_warm,
    object_catalog,
    option_fields,
    perf_trace,
    player_fields,
    ruler,
    settings,
    trigger_fields,
    unit_fields,
    unit_pick,
    unit_rotation,
)
from descape.constant_picker import CatalogLineEdit, preview_pixmap
from descape.diplomacy_panel import DiplomacyPanel
from descape.edit_history import EditHistory, MessagesDiffRecord, OptionsDiffRecord, tile_state
from descape.elevation_tools import set_tile_elevation, set_tiles_elevation
from descape.fill_tools import flood_fill_terrain
from descape.map_options_panel import MapOptionsPanel
from descape.map_view import MapView
from descape.messages_fields import MESSAGE_FIELDS
from descape.messages_model import MessageEditsUnavailableError, MessagesEditModel
from descape.messages_panel import MessagesPanel
from descape.players_panel import PlayersPanel
from descape.render import (
    dirty_screen_bbox_iso,
    dirty_screen_bbox_sloped,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)
from descape.scenario_io import (
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    LoadedScenario,
    load_map_and_units,
    load_map_and_units_from_bytes,
    parse_triggers,
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
from descape.options_model import (
    OptionEditsUnavailableError,
    OptionsEditModel,
    diplomacy_write_supported,
    options_write_supported,
    player_count_write_supported,
    players_write_supported,
)
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import name_for_terrain_id
from descape.trigger_model import (
    TriggerEditModel,
    TriggerEditsUnavailableError,
    display_order_with_copy_inserted,
    exec_order_write_supported,
    moved_display_order,
)
from descape.trigger_panel import TriggerPanel
from descape.unit_filter import GAIA_PLAYER_ID, UnitFilter
from descape.unit_model import UnitEditModel, UnitEditsUnavailableError
from descape.viewer_common import (
    BRUSH_TOOLS,
    _STROKE_LABELS,
    _TOOL_LABELS,
    _TOOL_PARAM,
    tool_applicable,
)
from descape.viewer_dialogs import CrashReportDialog, DebugLogDialog, apply_theme

# GAIA plus 8 real players -- scenario_io.LoadedScenario's
# number_of_unit_sections. The Filters menu offers a checkbox per real
# player; GAIA gets its own separate entry (see UnitFilter.matches).
MAX_PLAYER_ID = 8

# D2's arrow-key nudge amounts, in tiles: a fine nudge, and Shift's whole-tile
# step (matching the tile-centre grid a click/place snaps to).
_UNIT_NUDGE_STEP = 0.1
_UNIT_NUDGE_STEP_SHIFT = 1.0

# unit_fields.UnitFieldSpec.conditional -> the rule it names. The specs stay
# Qt-free and data-only by naming a rule as a string; this is the one place
# that resolves it, so the rule itself stays independently testable.
_FIELD_CONDITIONALS = {"rotation_is_angle": unit_rotation.rotation_is_angle}

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
_LEFT_PAGE_FOR_MODE = {
    "triggers": _LEFT_PAGE_TRIGGERS,
    "units": _LEFT_PAGE_UNITS,
    "map_options": _LEFT_PAGE_MAP_OPTIONS,
    "players": _LEFT_PAGE_PLAYERS,
    "diplomacy": _LEFT_PAGE_DIPLOMACY,
    "messages": _LEFT_PAGE_MESSAGES,
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


def _unit_name(unit_const: int) -> str:
    """Mirrors terrain_palette.name_for_terrain_id's shape, UNKNOWN_<id>
    fallback included: a scenario can legitimately reference a unit_const no
    dataset covers, and that must read as a known gap rather than a crash.

    object_catalog.combined_object_name() supplies the four-dataset merge
    (~1,355 members total, first dataset wins on an ID collision); the
    title-casing and UNKNOWN_<id> fallback are this call site's own display
    convention, not shared with trigger_fields.resolve_reference's ALL CAPS
    one.
    """
    name = object_catalog.combined_object_name(unit_const)
    return name.title() if name else f"UNKNOWN_{unit_const}"

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

# Display-only sentinel assigned to LoadedScenario.path for a File > New map.
# Deliberately relative and non-existent: LoadedScenario.path is only ever read
# for display and for Save As's default name -- write_scenario() takes an
# explicit destination and never reads it (see that field's own comment in
# scenario_io.py).
UNTITLED_NAME = "Untitled.aoe2scenario"
UNTITLED_PATH = Path(UNTITLED_NAME)

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

    def __init__(self, parent: "ViewerWindow"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 360)
        self._window = parent

        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_appearance_tab(), "Appearance")
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

        layout.addStretch(1)
        return tab

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        settings.set_dark_mode(enabled)
        apply_theme(QApplication.instance(), enabled)
        self._window._log_status(f"Dark mode: {'on' if enabled else 'off'}")

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

    # action_id prefix (before the first "_") -> section header text. Covers
    # today's sections (REBINDABLE_ACTIONS's "file_*"/"edit_*"/"view_*"/
    # "help_*"/"mode_*"/"tool_*"/"adjust_*" entries); an unlisted future
    # prefix still gets a section of its own, just titled from the raw
    # prefix instead of a curated name.
    _KEYBIND_SECTION_TITLES = {
        "file": "File",
        "edit": "Edit",  # renamed from "Copy/Paste" -- now covers the whole Edit menu
        "view": "View",
        "help": "Help",
        "mode": "Modes",
        "filter": "Filters",
        "player": "Player Selection",
        "unit": "Units",
        "tool": "Tools",
        "adjust": "Tool Value",
    }

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

    def _set_install_status(self, message: str, ok: bool) -> None:
        color = STATUS_OK_COLOR if ok else STATUS_ERROR_COLOR
        self.install_status_label.setStyleSheet(f"color: {color};")
        self.install_status_label.setText(message)


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        # "flat" or "stepped" -- see the Elevation View toolbar combo built
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
        # v2.7 copy/paste: a single clipboard slot, not a
        # manager -- overwritten on every Copy. None means empty. Tagged by
        # "kind" so copy/paste can dispatch by content rather than by
        # whatever tool happened to be active when it was copied:
        # {"kind": "terrain", "terrain_id": ..., "layer": ...} or
        # {"kind": "elevation", "value": ...}. Set by copy_tile(); read (and
        # its kind checked against the active tool) by paste_tile() and
        # _update_tool_enabled()'s paste-gating.
        self._clipboard: dict | None = None
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
        # Guards the inspector's in-place editors (D5) against recording a
        # phantom undo step while _update_unit_inspector() programmatically
        # sets their values -- players_panel._changed()'s own _populating
        # guard, borrowed here since the inspector stays inline rather than
        # becoming a panel class.
        self._unit_inspector_populating = False
        self._update_title()

        left = QVBoxLayout()

        self.hover_label = QLabel(HOVER_IDLE_TEXT)
        self.hover_label.setWordWrap(True)
        left.addWidget(self.hover_label)

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
            on_entry_field=self.set_entry_field,
            on_trigger_structural=self.trigger_structural_edit,
            on_entry_structural=self.entry_structural_edit,
            on_variable_structural=self.variable_structural_edit,
        )
        self.map_options_panel = MapOptionsPanel(on_option_field=self.set_option_field)
        self.players_panel = PlayersPanel(
            on_player_field=self.set_player_field, on_player_count=self.set_player_count
        )
        self.diplomacy_panel = DiplomacyPanel(
            on_diplomacy_field=self.set_diplomacy_field, on_option_field=self.set_option_field
        )
        self.messages_panel = MessagesPanel(on_message_field=self.set_message_field)
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(info_widget)
        self.left_stack.addWidget(self.trigger_panel)
        self.left_stack.addWidget(self._build_unit_inspector())
        self.left_stack.addWidget(self.map_options_panel)
        self.left_stack.addWidget(self.players_panel)
        self.left_stack.addWidget(self.diplomacy_panel)
        self.left_stack.addWidget(self.messages_panel)

        self.map_view = MapView(
            self.on_hover,
            self.on_edit_stroke_start,
            self.on_edit_stroke_tile,
            self.on_edit_stroke_end,
            self.on_fill,
            self.on_click_select,
            self.on_unit_place,
            self.on_unit_move,
            self.on_unit_nudge,
            self.on_unit_delete,
            self.on_marquee_select,
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

        # Always-visible short status history, distinct from both the
        # transient single-line QMainWindow.statusBar() message and the
        # Help > Debug Log dialog (a separate popup, not always on screen).
        # _log_status() is the one place that feeds all three destinations
        # that matter for a given message.
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        line_height = self.status_log.fontMetrics().lineSpacing()
        self.status_log.setMaximumBlockCount(200)
        # setMinimumHeight, not the setFixedHeight this had while the log was a
        # plain layout row: a fixed height sets min == max, which pins a
        # QSplitter child and leaves the handle looking draggable but inert.
        self.status_log.setMinimumHeight(max(settings.MIN_LOG_PANE, line_height * 2 + 12))

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
        self._build_keybind_actions()
        self._update_tool_enabled()

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
        self.copy_action = QAction("&Copy Tile", self)
        self.copy_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.copy_action.triggered.connect(self.copy_tile)
        edit_menu.addAction(self.copy_action)
        self.paste_action = QAction("&Paste Tile", self)
        self.paste_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_action.triggered.connect(self.paste_tile)
        edit_menu.addAction(self.paste_action)

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
                "Turn the selected units (Units mode). Only units whose rotation is a real "
                "facing turn -- walls, gates and most GAIA objects store a graphic variant "
                "in that field instead, and are skipped"
            )
            action.triggered.connect(handler)
            rotate_menu.addAction(action)
            setattr(self, attr, action)
            self._rotate_actions.append(action)

        edit_menu.addSeparator()
        self.settings_action = QAction("&Settings…", self)
        self.settings_action.triggered.connect(self._show_settings)
        edit_menu.addAction(self.settings_action)

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
        self.show_sprites_action = QAction("Show sprites", self, checkable=True, checked=True)
        self.show_sprites_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.show_sprites_action.toggled.connect(self._on_sprites_toggled)
        view_menu.addAction(self.show_sprites_action)

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
        for tiles, action in self.distance_tick_interval_actions.items():
            # `on` must lead: toggled passes checked as the first positional
            # arg, so a lambda with only `tiles=tiles` gets it bound into
            # `tiles` instead (the trap New Map's own size actions document).
            # `on and ...` is required rather than stylistic: QActionGroup
            # unchecks the outgoing action before checking the incoming one,
            # so without the guard each switch fires twice, the first time
            # carrying the stale interval.
            action.toggled.connect(lambda on, tiles=tiles: on and self._on_distance_tick_interval(tiles))

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
            "Traces per-stroke drag-paint latency by phase to Help > Debug Log. "
            "Also settable via the DESCAPE_PERF_TRACE=1 environment variable."
        )
        self.perf_trace_action.toggled.connect(perf_trace.enable)
        help_menu.addAction(self.perf_trace_action)

    def _build_unit_inspector(self) -> QWidget:
        """The Units-mode left page -- phase 3's P3-e laid this out
        read-only "so phase 3.5 swaps each value label for an editor IN
        PLACE"; this is that swap.
        unit_fields.FIELDS is the single source of which rows are editable
        and what kind of editor they get.
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        self.unit_inspector_empty = QLabel("No unit selected")
        self.unit_inspector_empty.setWordWrap(True)
        layout.addWidget(self.unit_inspector_empty)

        self.unit_inspector_grid = QWidget()
        grid = QGridLayout(self.unit_inspector_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        self.unit_field_labels: dict[str, QLabel] = {}
        self.unit_field_editors: dict[str, QWidget] = {}
        for row, spec in enumerate(unit_fields.FIELDS):
            grid.addWidget(QLabel(f"{spec.label}:"), row, 0)
            # A conditional field gets BOTH widgets, in the same cell: the
            # editor for consts its rule admits, and the plain read-only label
            # every other const keeps. _apply_conditional_fields() shows one
            # and hides the other per selection, and a hidden widget takes no
            # layout space, so the row never doubles in height.
            if not spec.editable or spec.conditional:
                value = QLabel("")
                value.setWordWrap(True)
                value.setTextInteractionFlags(Qt.TextSelectableByMouse)
                grid.addWidget(value, row, 1)
                self.unit_field_labels[spec.field_id] = value
            if not spec.editable:
                continue
            if spec.kind == unit_fields.PLAYER:
                combo = QComboBox()
                combo.addItem("GAIA", GAIA_PLAYER_ID)
                for player_id in range(1, MAX_PLAYER_ID + 1):
                    combo.addItem(f"Player {player_id}", player_id)
                combo.currentIndexChanged.connect(
                    lambda _, s=spec, w=combo: self._on_unit_field_changed(s, w.currentData())
                )
                grid.addWidget(combo, row, 1)
                self.unit_field_editors[spec.field_id] = combo
            else:  # FLOAT
                spin = QDoubleSpinBox()
                spin.setDecimals(spec.decimals)
                spin.setRange(spec.minimum, spec.maximum)
                # setKeyboardTracking(False) is load-bearing the same way
                # viewer_common._make_spinbox's own comment explains: without
                # it, typing a multi-digit value over an old one fires
                # valueChanged once per digit, recording one undo step per
                # keystroke instead of one per commit (trap 3).
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(
                    lambda value, s=spec: self._on_unit_field_changed(s, value)
                )
                grid.addWidget(spin, row, 1)
                self.unit_field_editors[spec.field_id] = spin
        layout.addWidget(self.unit_inspector_grid)

        # A top-level AGENTS.md hard rule, and this panel is the first place
        # it becomes user-visible: for ~65% of GAIA objects `rotation` is a
        # tree/doodad graphic-variant index (values like 7..53), not an
        # angle. Shown raw (radians, never degree-formatted) rather than
        # converted, which would be a confident lie most of the time -- and
        # editable only where it genuinely is an angle.
        self.unit_rotation_note = QLabel(
            "Rotation is shown raw, in radians. It is editable only for units "
            "whose rotation is a real facing -- for most GAIA objects, walls "
            "and gates it is a graphic-variant index, not an angle."
        )
        self.unit_rotation_note.setWordWrap(True)
        layout.addWidget(self.unit_rotation_note)

        layout.addStretch(1)
        self._show_inspector_fields(False)
        return page

    def _show_inspector_fields(self, visible: bool) -> None:
        self.unit_inspector_grid.setVisible(visible)
        self.unit_rotation_note.setVisible(visible)
        self.unit_inspector_empty.setVisible(not visible)

    def _update_unit_inspector(self, entry) -> None:
        if entry is None:
            self._show_inspector_fields(False)
            for label in self.unit_field_labels.values():
                label.setText("")
            return
        unit = entry.unit
        texts = {
            "name": _unit_name(unit.unit_const),
            "unit_const": str(unit.unit_const),
            "rotation": f"{unit.rotation:g}",
            "reference_id": str(unit.reference_id),
            "garrisoned_in_id": str(getattr(unit, "garrisoned_in_id", -1)),
        }
        for field, text in texts.items():
            self.unit_field_labels[field].setText(text)
        # Guards the editors below against _on_unit_field_changed recording a
        # phantom undo step from this programmatic populate -- the same
        # _populating guard players_panel._changed() uses.
        self._unit_inspector_populating = True
        try:
            self.unit_field_editors["player"].setCurrentIndex(
                max(self.unit_field_editors["player"].findData(entry.player_id), 0)
            )
            self.unit_field_editors["x"].setValue(unit.x)
            self.unit_field_editors["y"].setValue(unit.y)
            self.unit_field_editors["z"].setValue(getattr(unit, "z", 0.0))
            self._apply_conditional_fields(unit)
        finally:
            self._unit_inspector_populating = False
        self._show_inspector_fields(True)

    def _apply_conditional_fields(self, unit) -> None:
        """Swaps each conditional field between its editor and its read-only
        label for THIS unit's const -- today only Rotation, whose rule is
        unit_rotation.rotation_is_angle.

        Only ever called with the populating guard already held: it sets an
        editor's value, and the resulting valueChanged must not record a
        phantom undo step.

        The value is normalized through rotate_step(..., 0) on the way in
        rather than handed to setValue raw. An angle const can still carry a
        stored value outside [0, 2*pi) (7.0 appears 574 times in the corpus),
        and the spinbox would silently CLAMP that to its own maximum -- i.e.
        display a number the file does not contain.
        """
        for spec in unit_fields.FIELDS:
            if not spec.conditional:
                continue
            allowed = _FIELD_CONDITIONALS[spec.conditional](unit.unit_const)
            editor = self.unit_field_editors[spec.field_id]
            editor.setVisible(allowed)
            self.unit_field_labels[spec.field_id].setVisible(not allowed)
            if allowed and spec.field_id == "rotation":
                editor.setValue(
                    unit_rotation.rotate_step(
                        unit.rotation, unit_rotation.angle_count_for(unit.unit_const), 0
                    )
                )

    def _on_unit_field_changed(self, spec: unit_fields.UnitFieldSpec, value) -> None:
        """The single funnel every unit inspector editor reports through --
        mirrors set_trigger_field()'s role for the trigger form.

        The unchanged-value early return matters here for a second reason
        beyond avoiding a phantom undo step: _update_unit_inspector()
        populates every editor via setValue()/setCurrentIndex(), and Qt
        fires the change signal even for a programmatic set that lands on
        the same value as a prior one when _unit_inspector_populating is
        already guarding it -- but a genuinely no-op edit (typing the same
        X back) must not record either.
        """
        # len != 1, not just falsy: the inspector's fields are hidden
        # whenever 2+ units are selected (_refresh_selection_view), so this
        # is defensive against a signal somehow firing from a hidden editor
        # rather than an expected path.
        if self._unit_inspector_populating or len(self._selection) != 1:
            return
        model = self._ensure_unit_edits()
        if model is None:
            self._update_unit_inspector_from_selection()
            return
        index = self.map_view._unit_index
        entry = index.entry_for_key(self._selection[0]) if index is not None else None
        if entry is None:
            return
        unit = entry.unit

        if spec.field_id == "player":
            new_player = int(value)
            if new_player == entry.player_id:
                return
            with self._unit_edit(model, "Reassign unit", [entry.player_id, new_player]):
                model.reassign(unit, new_player)
                self._selection = [unit_pick.unit_key(new_player, unit)]
            self._log_status(
                f"Reassigned unit to {'GAIA' if new_player == GAIA_PLAYER_ID else f'Player {new_player}'}"
            )
            return

        if spec.field_id == "rotation":
            # Defensive, not expected: the editor is hidden for a const whose
            # rotation is a variant index, so a signal from it would mean the
            # conditional swap failed. Refusing here keeps set_rotation()'s
            # own raise off the UI path.
            if not unit_rotation.rotation_is_angle(unit.unit_const):
                return
            rotation = unit_rotation.rotate_step(
                float(value), unit_rotation.angle_count_for(unit.unit_const), 0
            )
            if rotation == unit.rotation:
                return
            with self._unit_edit(model, "Set unit Rotation", [entry.player_id]):
                model.set_rotation(unit, rotation)
            return

        current = getattr(unit, spec.field_id)
        if value == current:
            return
        x, y, z = unit.x, unit.y, getattr(unit, "z", 0.0)
        if spec.field_id == "x":
            x = value
        elif spec.field_id == "y":
            y = value
        elif spec.field_id == "z":
            z = value
        with self._unit_edit(model, f"Set unit {spec.label}", [entry.player_id]):
            model.set_position(unit, x, y, z)

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
        updates the inspector -- one entry shows its fields (D5), zero or
        2+ show unit_inspector_empty's count text instead.

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
            self._update_unit_inspector(entries[0])
            return
        self._update_unit_inspector(None)
        self.unit_inspector_empty.setText(f"{len(entries)} units selected" if entries else "No unit selected")

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
        self.filter_show_all_action.setToolTip("Check every entry above -- GAIA, Trees, and all players")
        self.filter_show_all_action.triggered.connect(lambda: self._set_all_filters(True))
        menu.addAction(self.filter_show_all_action)
        self.filter_hide_all_action = QAction("Hide All", self)
        self.filter_hide_all_action.setToolTip("Uncheck every entry above -- GAIA, Trees, and all players")
        self.filter_hide_all_action.triggered.connect(lambda: self._set_all_filters(False))
        menu.addAction(self.filter_hide_all_action)

        self.filters_button.setMenu(menu)
        toolbar.addWidget(self.filters_button)

    def _rebuild_unit_index(self) -> None:
        """Rebuilds the pick index for the current scenario + filter and hands
        it to MapView.

        Built on demand (entering Units mode, or a filter change while
        already in it) rather than at load: an ~11k-unit file costs a real
        walk, and opening a map for terrain work must not pay for it.
        """
        if self.scenario is None:
            self.map_view.set_unit_index(None)
            return
        self.map_view.set_unit_index(unit_pick.build_index(self.scenario, self._unit_filter))

    def on_click_select(self, pos, modifiers) -> None:
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
        """
        entry = self.map_view.pick_unit_at(pos)
        if entry is None:
            if not (modifiers & (Qt.ControlModifier | Qt.ShiftModifier)):
                self._selection = []
                self._refresh_selection_view()
                self._log_status("Selection cleared")
            return
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        if modifiers & Qt.ControlModifier:
            if key in self._selection:
                self._selection = [k for k in self._selection if k != key]
            else:
                self._selection = self._selection + [key]
        elif modifiers & Qt.ShiftModifier:
            if key not in self._selection:
                self._selection = self._selection + [key]
        else:
            self._selection = [key]
        self._refresh_selection_view()
        if len(self._selection) == 1:
            self._log_status(f"Selected {_unit_name(entry.unit.unit_const)} at ({entry.unit.x:g}, {entry.unit.y:g})")
        else:
            self._log_status(f"{len(self._selection)} units selected")

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

    def on_unit_place(self, pos, modifiers) -> None:
        """MapView's seventh injected callable -- a left click with the
        Place Unit tool active (b1.4).

        Grid-snapped only (D2's default): `_pick_tile()` already resolves the
        clicked SCREEN pixel to an integer tile in all three render styles
        (Flat's division, Stepped's screen_to_tile, Sloped's pick-plane
        lookup), so `tile + 0.5` is the snap -- no float inversion needed for
        this path. Free placement (D2's toggle) is deferred: Sloped's terrain
        genuinely has no analytic inverse (unit_pick.pick_unit's own
        docstring), so a free-float placement there needs its own pick-plane-
        style solution, not a typing exercise.
        """
        if self.scenario is None:
            return
        tile = self.map_view._pick_tile(pos)
        if tile is None:
            return
        object_id = self.place_object_edit.value()
        if object_id is None:
            self._log_status("Place Unit: choose an object first")
            return
        player = self.place_owner_combo.currentData()
        model = self._ensure_unit_edits()
        if model is None:
            return
        x, y = tile[0] + 0.5, tile[1] + 0.5
        with self._unit_edit(model, "Place unit", [player]):
            unit = model.add(player, object_id, x, y)
            # Placed units become the selection (same reasoning as
            # reassign's own key update): the id just chosen is the one worth
            # showing in the inspector next, not whatever was selected before.
            self._selection = [unit_pick.unit_key(player, unit)]
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        self._log_status(f"Placed {_unit_name(object_id)} for {owner_text} at ({x:g}, {y:g})")

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
        tile = self.map_view._pick_tile(pos)
        if tile is None:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        x, y = tile[0] + 0.5, tile[1] + 0.5
        unit = entry.unit
        if (x, y) == (unit.x, unit.y):
            return
        with self._unit_edit(model, "Move unit", [entry.player_id]):
            model.set_position(unit, x, y, unit.z)
        self._log_status(f"Moved {_unit_name(unit.unit_const)} to ({x:g}, {y:g})")

    def on_unit_nudge(self, dx: int, dy: int, modifiers) -> None:
        """MapView's ninth injected callable -- an arrow key in Units mode
        (b1.5's second half, generalized to the whole selection by b2.4).
        One record per press, matching every other discrete-keypress edit in
        this app (e.g. the ]/[ tool-value step) -- not merged across
        repeats, unlike a typed-digit spinbox edit. That holds for a group
        nudge too: every selected unit moves in the SAME undo step, not one
        step per unit, since begin_unit_edit(players) already captures every
        touched player's whole list up front (unit_model.py's own
        PlayerListSnapshot)."""
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
        step = _UNIT_NUDGE_STEP_SHIFT if modifiers & Qt.ShiftModifier else _UNIT_NUDGE_STEP
        players = sorted({e.player_id for e in entries})
        label = "Nudge unit" if len(entries) == 1 else f"Nudge {len(entries)} units"
        with self._unit_edit(model, label, players):
            for entry in entries:
                unit = entry.unit
                model.set_position(unit, unit.x + dx * step, unit.y + dy * step, unit.z)
        if len(entries) == 1:
            unit = entries[0].unit
            self._log_status(f"Moved {_unit_name(unit.unit_const)} to ({unit.x:g}, {unit.y:g})")
        else:
            self._log_status(f"Moved {len(entries)} units")

    def on_unit_rotate(self, steps: int) -> None:
        """Turns every selected unit whose `rotation` is genuinely an angle by
        `steps` whole stored frames -- phase 3.5b's b3.

        Modelled on on_unit_nudge: the whole selection moves in ONE undo
        record, since begin_unit_edit(players) already captures every touched
        player's list up front. Positive steps rotate clockwise on screen (see
        unit_rotation.rotate_step).

        **Each unit rotates about its own centre.** Orbiting a selection about
        a shared pivot is deliberately out of scope -- that is a group
        transform, not a per-unit field edit, and nothing else in the unit
        edit path moves a unit the caller did not name.

        Selected units whose semantics are VARIANT (walls, trees, most GAIA
        doodads) or INERT (gates and every other single-frame graphic) are
        skipped rather than refused: a marquee over a village will always
        include some, and failing the whole action for them would make Rotate
        unusable exactly where it is most wanted.

        Mode-gated explicitly, unlike nudge/delete: those arrive through
        MapView's injected callables and so are Units-mode-only for free,
        while this is a QAction whose shortcut is live in every mode.
        """
        if self.scenario is None or self.mode != "units" or not self._selection:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        rotatable = [e for e in entries if unit_rotation.rotation_is_angle(e.unit.unit_const)]
        skipped = len(entries) - len(rotatable)
        if not rotatable:
            if entries:
                self._log_status(
                    f"Rotate: {skipped} selected unit(s) store a graphic variant, not an angle"
                )
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        label = "Rotate unit" if len(rotatable) == 1 else f"Rotate {len(rotatable)} units"
        with self._unit_edit(model, label, sorted({e.player_id for e in rotatable})):
            for entry in rotatable:
                unit = entry.unit
                angle_count = unit_rotation.angle_count_for(unit.unit_const)
                model.set_rotation(unit, unit_rotation.rotate_step(unit.rotation, angle_count, steps))
        if len(rotatable) == 1:
            unit = rotatable[0].unit
            done = f"Rotated {_unit_name(unit.unit_const)} to {unit.rotation:g} rad"
        else:
            done = f"Rotated {len(rotatable)} units"
        if skipped:
            done += f"; {skipped} skipped (rotation is a graphic variant, not an angle)"
        self._log_status(done)

    def on_unit_rotate_coarse(self, direction: int) -> None:
        """Rotate by the closest whole number of frames to a quarter turn.

        The coarse step is its own action rather than a Shift modifier on the
        fine one: a QAction's shortcut IS the key combination, so unlike
        on_unit_nudge (which gets a live `modifiers` from MapView's key event)
        there is no modifier for this path to read. Four keybind rows, two
        entry points.
        """
        entry = self._first_rotatable_entry()
        if entry is None:
            self.on_unit_rotate(direction)  # nothing to rotate; reuse its status/no-op path
            return
        angle_count = unit_rotation.angle_count_for(entry.unit.unit_const)
        self.on_unit_rotate(direction * unit_rotation.quarter_turn_steps(angle_count))

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
        removable = [e for e in entries if not model.referencing(e.unit)]
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
        with self._unit_edit(model, label, players):
            for entry in removable:
                model.remove(entry.unit)
        if blocked:
            self._log_status(f"Deleted {len(removable)}; {blocked} refused (garrison reference)")
            QMessageBox.warning(
                self, "Some units not deleted",
                f"{blocked} unit(s) are referenced by another unit's garrison and were not deleted.",
            )
        elif len(removable) == 1:
            self._log_status(f"Deleted {_unit_name(removable[0].unit.unit_const)}")
        else:
            self._log_status(f"Deleted {len(removable)} units")

    def _refresh_selection_after_filter(self) -> None:
        """A filter toggle can hide currently selected units. Rebuild the
        index, then drop every selection key that no longer resolves --
        leaving a highlight around a unit that no longer paints would be a
        cue pointing at nothing."""
        if self.mode != "units":
            return
        self._rebuild_unit_index()
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
        """Same as _set_all_players, but also GAIA and Trees -- the menu's
        "Show All"/"Hide All" shortcuts at the bottom."""
        self.show_gaia_action.blockSignals(True)
        self.show_gaia_action.setChecked(checked)
        self.show_gaia_action.blockSignals(False)
        self.show_trees_action.blockSignals(True)
        self.show_trees_action.setChecked(checked)
        self.show_trees_action.blockSignals(False)
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
            elapsed = time.perf_counter() - t0
        finally:
            self._busy = was_busy
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
        self._log_status(f"Show sprites: {'on' if checked else 'off'} (applied in {elapsed:.2f}s)")

    def _filter_summary(self) -> str:
        if self._unit_filter.is_default:
            return "showing all units"
        parts = []
        if not self._unit_filter.show_gaia:
            parts.append("GAIA hidden")
        if not self._unit_filter.show_trees:
            parts.append("trees hidden")
        if self._unit_filter.players is not None:
            shown = sorted(self._unit_filter.players)
            parts.append(f"players {shown}" if shown else "no players")
        return ", ".join(parts)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

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
        self.terrain_style_combo.addItems(["Flat", "Stepped", "Sloped"])
        self.terrain_style_combo.setCurrentText("Stepped")  # before connect(): no spurious signal
        self.terrain_style_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
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
            "snapped to the clicked tile's centre"
        )
        self.place_unit_action.toggled.connect(lambda on: on and self._on_tool_selected("place_unit"))
        tool_group.addAction(self.place_unit_action)
        toolbar.addAction(self.place_unit_action)

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
        self.terrain_param_label_action = param_toolbar.addWidget(QLabel(" Terrain type: "))
        self.terrain_combo = QComboBox()
        self.terrain_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        for terrain_id in sorted(TerrainId, key=lambda t: t.name):
            self.terrain_combo.addItem(name_for_terrain_id(terrain_id.value), terrain_id.value)
        self.terrain_param_combo_action = param_toolbar.addWidget(self.terrain_combo)

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

        # Place Unit's own param pair -- D1 (reuse the trigger form's own
        # catalog picker rather than building a new one) and the b1.4
        # decision to give placement its own owner combo (later reused as
        # the Convert brush's destination, b2.5) since Filters' per-player
        # checkboxes are visibility toggles, not a "current player".
        self.place_object_label_action = param_toolbar.addWidget(QLabel(" Object: "))
        self.place_object_edit = CatalogLineEdit(object_catalog.objects(), default_category="Units")
        self.place_object_edit.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.place_object_param_action = param_toolbar.addWidget(self.place_object_edit)

        self.place_owner_label_action = param_toolbar.addWidget(QLabel(" Owner: "))
        self.place_owner_combo = QComboBox()
        self.place_owner_combo.addItem("GAIA", GAIA_PLAYER_ID)
        for player_id in range(1, MAX_PLAYER_ID + 1):
            self.place_owner_combo.addItem(f"Player {player_id}", player_id)
        self.place_owner_combo.setCurrentIndex(1)  # Player 1, per b1.4's own default
        self.place_owner_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.place_owner_param_action = param_toolbar.addWidget(self.place_owner_combo)

        # Convert's own extra param (D3/b2.5) -- which owners' units a drag
        # reassigns. Destination reuses place_owner_combo above rather than
        # a second combo (the comment on that combo's own construction
        # already names this reuse); brush size/shape below is generic to
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
        style = self._terrain_style.capitalize()
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
            "file_exit": self.exit_action,
            "edit_undo": self.undo_action,
            "edit_redo": self.redo_action,
            "edit_copy": self.copy_action,
            "edit_paste": self.paste_action,
            "edit_settings": self.settings_action,
            "view_isometric": self.iso_action,
            "view_distance_ticks": self.distance_ticks_action,
            "view_show_sprites": self.show_sprites_action,
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
        }
        for tool in settings.TOOLS:
            self._keybind_actions[f"tool_{tool.tool_id}"] = getattr(self, f"{tool.tool_id}_action")
        for pid, action in self.player_select_actions.items():
            self._keybind_actions[f"player_select_{pid}"] = action
        for action_id in self._keybind_actions:
            self.apply_keybind(action_id)

    def apply_keybind(self, action_id: str) -> None:
        action = self._keybind_actions.get(action_id)
        if action is None:
            return
        key_text = settings.get_keybind(action_id)
        action.setShortcut(QKeySequence(key_text) if key_text else QKeySequence())

    def _select_player(self, player_id: int) -> None:
        """Backs the player_select_0..8 shortcuts. There is no global
        "current player" (Filters' per-player checkboxes are visibility
        toggles, not one -- see place_owner_combo's own construction
        comment), so this dispatches to whichever mode-local selector is
        active and never syncs the other two."""
        if self.mode == "units":
            self.place_owner_combo.setCurrentIndex(self.place_owner_combo.findData(player_id))
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
        else:
            self._log_status(f"Player selection has no target in {self.mode_combo.currentText()} mode")

    def _on_distance_ticks_toggled(self, checked: bool) -> None:
        settings.set_distance_ticks(checked)
        self.map_view.set_edge_ticks(checked)

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
        self.mode = _mode_id(mode_text)
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
        if self.mode == "units":
            self._rebuild_unit_index()
            self._update_unit_inspector(None)
        if self.mode == "triggers":
            self._widen_left_column(TriggerPanel.MIN_USEFUL_WIDTH)
            # Parsed on demand, not at load: the largest corpus file's Triggers
            # section is 1.17 MB and opening a map for terrain work must not
            # pay for it.
            self.trigger_panel.show_scenario(self.scenario)
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
        self._update_tool_enabled()
        self._update_mode_status()
        # The combo's own text, not mode_text: _update_tool_enabled() above
        # can have forced the mode back (selecting Units while already in
        # Sloped), and reporting the mode the user asked for rather than the
        # one they got would contradict the combo they're looking at. Reading
        # the label rather than capitalizing self.mode also keeps "Map Options"
        # spelled the way the combo spells it.
        self._log_status(f"Mode changed to {self.mode_combo.currentText()}")

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
                "Trigger execution order is read-only for this file: its Triggers "
                "section failed the alignment gate that a save would splice through."
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
        """
        if self.unit_edits is not None:
            return self.unit_edits
        if self.scenario is None:
            return None
        try:
            self.unit_edits = UnitEditModel(self.scenario)
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
            for key, value in self.option_edits.pending_values().items():
                if key.startswith("stance:") or key.startswith("allied_victory:"):
                    values[key] = value
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
        self._update_unit_inspector(None)
        self.pan_action.setChecked(True)
        self._log_status(f"Mode forced to {mode_text} -- Sloped has no unit selection yet")

    def _update_tool_enabled(self) -> None:
        # Tools (and Close/Save As) have nothing to act on before a map is
        # loaded -- grayed out rather than left clickable-but-pointless.
        # Tool params (Terrain type / Level) go further and hide outright
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
        self.elevation_action.setEnabled(has_map and elevation_ok)
        self.set_level_action.setEnabled(has_map and elevation_ok)
        # Same gate as Draw/Paint Can (has_map and write_ok, no squareness
        # requirement): a cliff placement never touches the elevation
        # recursion elevation_ok exists for.
        self.cliff_action.setEnabled(has_map and write_ok)
        self.place_unit_action.setEnabled(unit_editable)
        self.convert_action.setEnabled(unit_editable)
        # Rotate needs a selection as well as Units mode, unlike the two tools
        # above -- it acts on what is already selected rather than on a click,
        # so with nothing selected there is no target and the button should
        # say so by being grey rather than by logging a refusal.
        for action in self._rotate_actions:
            action.setEnabled(unit_editable and bool(self._selection))
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

        # Tool params (Terrain type / Level): which one shows, if either,
        # depends on the *final* self._current_tool for this call -- same
        # reason the copy/paste block below reads it only after the
        # forced-back-to-Pan block above has had its say. One boolean per
        # param feeds both the widget's visibility and its enabled state
        # (and, for Level, the ]/[ step-value keybinds too) so a hidden
        # param can never be left live behind the scenes.
        param = _TOOL_PARAM.get(self._current_tool, "")
        terrain_param_ok = param == "terrain" and (self.draw_action.isEnabled() or self.fill_action.isEnabled())
        level_param_ok = param == "level" and self.set_level_action.isEnabled()
        object_param_ok = param == "object" and self.place_unit_action.isEnabled()
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
        brush_ok = self._current_tool in BRUSH_TOOLS and current_action is not None and current_action.isEnabled()
        self.terrain_param_label_action.setVisible(terrain_param_ok)
        self.terrain_param_combo_action.setVisible(terrain_param_ok)
        self.terrain_combo.setEnabled(terrain_param_ok)
        self.level_param_label_action.setVisible(level_param_ok)
        self.level_param_spin_action.setVisible(level_param_ok)
        self.elevation_level_spin.setEnabled(level_param_ok)
        self.place_object_label_action.setVisible(object_param_ok)
        self.place_object_param_action.setVisible(object_param_ok)
        self.place_object_edit.setEnabled(object_param_ok)
        # Owner combo: shown for Place (its own "owner to place as") AND
        # Convert (its destination player, D3/b2.5's own reuse decision --
        # see this widget's construction comment).
        self.place_owner_label_action.setVisible(object_param_ok or convert_param_ok)
        self.place_owner_param_action.setVisible(object_param_ok or convert_param_ok)
        self.place_owner_combo.setEnabled(object_param_ok or convert_param_ok)
        self.convert_sources_param_action.setVisible(convert_param_ok)
        self.convert_sources_button.setEnabled(convert_param_ok)
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
        self.tool_param_separator_action.setVisible(
            terrain_param_ok or level_param_ok or object_param_ok or convert_param_ok
            or cliff_param_ok or brush_ok
        )

        # v2.7 copy/paste -- deliberately placed after the
        # forced-back-to-Pan block above, not before: that block can flip
        # self._current_tool to "pan" synchronously (pan_action.
        # setChecked(True) -> _on_tool_selected("pan"), which itself calls
        # back into this method -- see that method's own comment) mid-call,
        # and copy/paste's gating needs to see the *final* tool for this
        # call, not whatever was selected when it started.
        #
        # Copy is enabled only for the edit-category tools that have
        # per-tile data to copy (Draw, Paint Can, Elevate, Set Elevation
        # -- never Pan), and only while that tool's own action is actually
        # enabled (not just selected) -- reusing draw_action/fill_action/
        # elevation_action/set_level_action's already-computed
        # write_ok/elevation_ok gates above rather than re-deriving them
        # here. Paint Can copies/pastes the same single hovered tile Draw
        # does -- Paste is not redefined as "fill with the clipboard
        # terrain" -- so it shares Draw's "terrain" clipboard kind below.
        # Copy's terrain branch (see copy_tile()) has a second effect beyond
        # the clipboard/Paste pair this comment block describes: it also
        # loads the picked terrain_id into terrain_combo, so it doubles as
        # a lightweight eyedropper for whatever tool Draw/Fill paint with
        # next. That doesn't change any gating here -- it's an extra write
        # inside the already-gated branch, not a new enabled state.
        #
        # Place Unit, Convert and Cliff are deliberately NOT keys here (unlike
        # the force-back-to-Pan dict above, which they ARE in): none has
        # per-tile terrain/elevation data, so omitting them makes
        # current_tool_action None and copy_ok False while any is active --
        # exactly the intended "Copy/Paste don't apply to placing/converting/
        # cliffing" behaviour, not a gap. Adding any would instead enable
        # Copy with kind_for_tool="elevation" (the ternary below), letting
        # Paste splice unrelated elevation data into a unit placement. Pan and
        # Ruler are excluded for the same reasons as the force-back-to-Pan
        # dict above.
        #
        # TOOLS-derived rather than hand-listed, matching current_action
        # above.
        current_tool_action = {
            t.tool_id: getattr(self, f"{t.tool_id}_action")
            for t in settings.TOOLS
            if t.tool_id not in ("pan", "ruler", "place_unit", "convert", "cliff")
        }.get(self._current_tool)
        copy_ok = current_tool_action is not None and current_tool_action.isEnabled()
        self.copy_action.setEnabled(copy_ok)

        # Paste additionally needs a non-empty clipboard whose kind matches
        # what the active tool would produce. Decision (a real open
        # question, not obvious either way): disable Paste outright on a
        # kind mismatch -- e.g. Copy while
        # on Draw, switch to Elevate, hit Paste -- rather than letting
        # the clipboard's own kind silently override the active tool.
        # Chosen for consistency with every other action this method
        # already gates: all of them fail toward "visibly greyed out with
        # an obvious reason" rather than a behavior that depends on state
        # the toolbar doesn't show. Draw and Elevate/Set Elevation both
        # copy/paste through the same "elevation" clipboard kind (see
        # copy_tile()), since Elevate and Set Elevation already share the
        # same underlying tile field.
        clipboard_kind = self._clipboard["kind"] if self._clipboard is not None else None
        kind_for_tool = "terrain" if self._current_tool in ("draw", "fill") else "elevation"
        self.paste_action.setEnabled(copy_ok and clipboard_kind == kind_for_tool)

    def _sync_map_view_brush(self) -> None:
        """Pushes the toolbar's current brush size/shape into MapView's
        hover-preview state -- but only for a tool that actually has one
        (BRUSH_TOOLS); Pan and Paint Can always preview a single tile
        regardless of what size/shape the spinbox/combo were last left at
        for Terrain/Elevate/Set Elevation. Called both when the brush
        widgets change and when the active tool changes, so the preview is
        never stale in either direction."""
        if self._current_tool in BRUSH_TOOLS:
            self.map_view.set_brush(self.brush_size_spin.value(), self.brush_shape_combo.currentData())
        else:
            self.map_view.set_brush(brush.BRUSH_SIZE_MIN, brush.BRUSH_SHAPE_SQUARE)

    def _on_brush_changed(self) -> None:
        self._sync_map_view_brush()
        self.map_view.refresh_highlight(self._hover_tile)

    def _on_tool_selected(self, tool: str) -> None:
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
        style = text.lower()
        if style == self._terrain_style:
            return
        if self._busy:
            # A render is already in progress (reentrant signal -- see
            # load_scenario's comment on self._busy). Refuse the change and
            # put the combo back to what's actually loaded rather than
            # leaving it displaying a style nothing was ever rendered at;
            # blockSignals so this doesn't recurse back into this handler.
            self.terrain_style_combo.blockSignals(True)
            self.terrain_style_combo.setCurrentText(self._terrain_style.capitalize())
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

    def on_edit_stroke_tile(self, x: int, y: int, modifiers) -> None:
        if self.scenario is None:
            return
        if self._current_tool == "convert":
            self._convert_stroke_tile(x, y)
            return
        if self._current_tool == "cliff":
            self._cliff_stroke_tile(x, y)
            return
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
            return

        if self._current_tool == "draw":
            terrain_id = self.terrain_combo.currentData()
            for tx, ty in footprint:
                tile = mm.get_tile(tx, ty)
                tile.terrain_id = terrain_id
                # Clear a stale double-terrain blend -- render.py doesn't draw
                # `layer`, but the game does, and leaving it set after changing
                # terrain_id would make this tool's own render lie about what
                # the game will actually show.
                tile.layer = -1
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
            return

        self._stroke_painted.update(footprint)

        # Cumulative dirty set since stroke start, minus what's already been
        # redrawn AT ITS CURRENT STATE this stroke -- avoids repainting the
        # same tile repeatedly as the drag continues over tiles elevation
        # propagation already touched. See EditHistory.stroke_dirty_indices's
        # docstring for the cost of this (a linear scan) at this project's map
        # sizes -- this is exactly why the brush footprint is expanded HERE,
        # once per cursor tile, rather than by calling this method once per
        # brush tile from MapView: doing that would multiply an already-O(map)
        # scan by the brush's area on every mouse-move. Same warning on_fill's
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
        new_dirty = {i for i in all_dirty if tile_state(mm.terrain[i]) != self._stroke_seen_state.get(i)}
        for i in all_dirty:
            self._stroke_seen_state[i] = tile_state(mm.terrain[i])
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
        label = _STROKE_LABELS.get(self._current_tool, "Edit")
        self.edit_history.commit_stroke(label, self.scenario.map_manager.terrain)
        self._stroke_seen_state = {}
        self._stroke_painted = set()
        self._update_edit_actions()
        self._update_title()
        perf_trace.flush(label.lower().replace(" ", "-"))

    # -- Convert brush (phase 3.5b's b2.5) --------------------------------
    #
    # Deliberately NOT built on edit_history.begin_stroke/commit_stroke like
    # the terrain strokes above: those diff TileState over a fixed-size
    # array, but a unit edit needs UnitEditModel.begin_unit_edit(players)
    # called with the COMPLETE set of touched players up front (it snapshots
    # exactly those players' lists for undo) -- and which players a Convert
    # drag will touch isn't known until the drag ends. So this stroke
    # ACCUMULATES (unit key -> destination) during the drag without mutating
    # anything, and only opens the real UnitEditModel edit once, at
    # _end_convert_stroke(), once the full touched-player set is known. One
    # record per drag either way, matching every other brush tool's own
    # "one record per stroke" contract.

    def _begin_convert_stroke(self) -> None:
        self._convert_pending: dict[tuple[int, int], int] = {}
        self._convert_touched_tiles: set[tuple[int, int]] = set()
        # Read once and held for the whole drag rather than re-read per
        # touched tile: the combo can't actually change while the mouse is
        # captured, and holding it fixed keeps every reassignment in one
        # drag consistent even if that ever stopped being true.
        self._convert_destination = self.place_owner_combo.currentData()

    def _convert_stroke_tile(self, x: int, y: int) -> None:
        if self.scenario is None:
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
        for tx, ty in footprint:
            if (tx, ty) in self._convert_touched_tiles:
                continue
            self._convert_touched_tiles.add((tx, ty))
            for order in index.by_tile.get((tx, ty), ()):
                entry = index.entries[order]
                if entry.player_id in sources and entry.player_id != destination:
                    self._convert_pending[unit_pick.unit_key(entry.player_id, entry.unit)] = destination

    def _end_convert_stroke(self) -> None:
        pending = self._convert_pending
        self._convert_pending = {}
        self._convert_touched_tiles = set()
        if not pending or self.scenario is None:
            return
        index = self.map_view._unit_index
        if index is None:
            return
        targets = []
        for key, destination in pending.items():
            entry = index.entry_for_key(key)
            if entry is not None:
                targets.append((entry, destination))
        if not targets:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        players = sorted({e.player_id for e, _ in targets} | {d for _, d in targets})
        label = "Convert unit" if len(targets) == 1 else f"Convert {len(targets)} units"
        with self._unit_edit(model, label, players):
            for entry, destination in targets:
                model.reassign(entry.unit, destination)
        owner_text = "GAIA" if self._convert_destination == GAIA_PLAYER_ID else f"Player {self._convert_destination}"
        self._log_status(f"Converted {len(targets)} unit(s) to {owner_text}")

    # -- Copy/paste (v2.7) -- one clipboard slot, keyed off
    # self._hover_tile (set by on_hover() on every mouse move) rather than a
    # click, since these fire from a keyboard shortcut. Copy reads whatever
    # field the active edit tool cares about straight off the hovered tile
    # (no undo record -- nothing is mutated). Paste is a one-shot,
    # non-interactive edit -- exactly what edit_history.EditHistory.apply()
    # exists for (see its own docstring), unlike the drag-stroke tools above
    # which use begin_stroke/stroke_dirty_indices/commit_stroke directly for
    # live per-tile feedback mid-drag; a keyboard paste has no drag to give
    # feedback during.

    def copy_tile(self) -> None:
        if self.scenario is None or self._hover_tile is None:
            return
        x, y = self._hover_tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        tile = mm.get_tile(x, y)
        if self._current_tool in ("draw", "fill"):
            # `layer` is captured alongside terrain_id but deliberately never
            # read back on paste (see paste_tile()'s mutate_fn) -- paste
            # always resets layer to -1 instead, matching every other
            # terrain-write path in this tool. Kept in the dict anyway
            # (costs nothing) so it's visible here that this was considered,
            # not overlooked.
            self._clipboard = {"kind": "terrain", "terrain_id": tile.terrain_id, "layer": tile.layer}
            # Also load the picked terrain into terrain_combo -- the actual
            # source Draw/Fill paint with (on_edit_stroke_tile/on_fill both
            # read terrain_combo.currentData(), never the clipboard). Without
            # this, Copy only fed single-tile Paste; a drag-painted stroke
            # right after Copy still used whatever terrain_combo was already
            # showing, which is the "copying for drawing doesn't work"
            # complaint this fixes. findData() returns -1 for a terrain_id
            # not in the picker's list (shouldn't happen -- no corpus file
            # has ever produced one -- but setCurrentIndex(-1) would blank
            # the combo and make currentData() return None, which
            # on_edit_stroke_tile() writes straight into tile.terrain_id) --
            # guarded against below.
            idx = self.terrain_combo.findData(tile.terrain_id)
            terrain_name = name_for_terrain_id(tile.terrain_id)
            if idx >= 0:
                self.terrain_combo.setCurrentIndex(idx)
                self._log_status(f"Copied terrain ({terrain_name}) from ({x}, {y}) -- now selected for drawing")
            else:
                self._log_status(
                    f"Copied terrain ({terrain_name}) from ({x}, {y}) -- not in the terrain picker, "
                    "drawing selection unchanged"
                )
        elif self._current_tool in ("elevation", "set_level"):
            self._clipboard = {"kind": "elevation", "value": tile.elevation}
            self._log_status(f"Copied elevation ({tile.elevation}) from ({x}, {y})")
        else:
            return
        # Paste's enabled state depends on the clipboard's kind (see
        # _update_tool_enabled()'s comment) -- refresh it now rather than
        # waiting for some unrelated event to do so, or a fresh Copy
        # wouldn't visibly enable Paste until then.
        self._update_tool_enabled()

    def paste_tile(self) -> None:
        if self.scenario is None or self._hover_tile is None or self._clipboard is None:
            return
        x, y = self._hover_tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        kind = self._clipboard["kind"]
        # Defensive re-check of the same type-mismatch decision
        # _update_tool_enabled() already encodes in paste_action's enabled
        # state (disable on a kind mismatch -- see that method's comment):
        # this guards paste_tile() itself against being invoked directly
        # (e.g. by a test, or a future caller) bypassing the QAction.
        if kind == "terrain" and self._current_tool not in ("draw", "fill"):
            return
        if kind == "elevation" and self._current_tool not in ("elevation", "set_level"):
            return

        def mutate() -> None:
            if kind == "terrain":
                t = mm.get_tile(x, y)
                t.terrain_id = self._clipboard["terrain_id"]
                # Same layer-reset every other terrain-write path in this
                # tool applies (the Draw tool's own click handler in
                # on_edit_stroke_tile(), batch_api.set_terrain) -- a stale
                # double-terrain blend left over from whatever terrain_id
                # used to be there would make this tool's own render lie
                # about what the game will actually show. Preserving the
                # copied tile's own `layer` verbatim was the alternative,
                # but that would make a pasted tile behave differently from
                # one painted with the same terrain_id by any other path in
                # the tool, for no real benefit.
                t.layer = -1
            else:
                # Not a raw `tile.elevation =` write -- goes through the
                # same neighbor-propagation real Elevate/Set Elevation
                # edits already use. No clamping needed here (unlike
                # on_edit_stroke_tile()'s Elevate branch): the copied value
                # was already a legal elevation on its source tile, not a
                # delta that could go out of range.
                set_tile_elevation(mm, x, y, self._clipboard["value"])

        dirty = self.edit_history.apply("Paste", mm.terrain, mutate)
        # Always log, even on a genuine no-op (dirty == [], e.g. pasting the
        # terrain a tile already has) -- EditHistory.apply()/commit_stroke()
        # deliberately push no record for that case (see commit_stroke()'s
        # docstring), but silently doing nothing here would look like the
        # keybind itself was broken. _apply_dirty() already no-ops on an
        # empty dirty_indices, so it's still safe to call unconditionally.
        self._apply_dirty(dirty)
        self._update_edit_actions()
        self._update_title()
        if dirty:
            self._log_status(f"Pasted {kind} to ({x}, {y})")
        else:
            self._log_status(f"Pasted {kind} to ({x}, {y}) (no change)")

    def on_fill(self, x: int, y: int, modifiers) -> None:
        """Paint Can: one flood fill per left click -- MapView routes
        CLICK_TOOLS here directly (see mousePressEvent), never through the
        stroke handlers above. Shaped like paste_tile() just above, not like
        on_edit_stroke_tile(): a one-shot edit_history.apply() rather than
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
        terrain_id = self.terrain_combo.currentData()
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            dirty = self.edit_history.apply(
                _STROKE_LABELS["fill"], mm.terrain, lambda: flood_fill_terrain(mm, x, y, terrain_id)
            )
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
        # paste_tile()'s own comment above -- a silent no-op reads as a
        # broken keybind, and a click that can rewrite the whole map
        # deserves a record either way.
        if dirty:
            self._log_status(f"Filled {len(dirty)} tiles with {name} from ({x}, {y}) applied in {elapsed:.2f}s")
        else:
            self._log_status(f"Fill at ({x}, {y}): already {name} (no change)")

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
        paste_tile(), and Undo/Redo. Undo is worth naming because it was
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
        full recomposite."""
        if not dirty_indices or self.scenario is None:
            return
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
                # SlopedChunkCache._refresh_source_caches().
                with perf_trace.phase("patch"):
                    self._cache.patch(bbox, elevation_changed=elevation_changed)
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
                self.trigger_panel.show_scenario(self.scenario)
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
                self.trigger_panel.show_scenario(self.scenario)
            elif refresh == "variables":
                # Neither tree in the panel shows a variable, so a full
                # repopulate would be pure cost. Only the dialog is stale, and
                # this is a no-op when it is not open.
                self.trigger_panel.refresh_variables()
            else:
                self.trigger_panel.refresh_entries()
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

    def set_entry_field(
        self, index: int, kind: str, entry_index: int, spec: trigger_fields.FieldSpec, value
    ) -> None:
        """Write one field of a condition or effect. The second and last funnel
        the panel reports edits through."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        label = f"Set {kind} {spec.label}"
        with self._trigger_edit(model, label, content_touched=[index]) as m:
            trigger = m.manager().triggers[index]
            entries = trigger.conditions if kind == "condition" else trigger.effects
            setattr(entries[entry_index], spec.name, value)

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

    def trigger_structural_edit(self, op: str, index: int) -> None:
        """Add, copy, delete, or reorder a trigger. The third funnel, and the
        only one that goes through TriggerEditModel.structural_edit().

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
        """
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        if op != "new" and not 0 <= index < len(manager.triggers):
            return

        def mutate(m):
            if op == "new":
                return m.add_trigger(self._unique_trigger_name(m))
            if op == "copy":
                before_triggers = list(m.triggers)
                before_order = list(m.trigger_display_order)
                new_trigger = m.copy_trigger(index)
                m.trigger_display_order = display_order_with_copy_inserted(
                    before_triggers, before_order, m.triggers, index
                )
                return new_trigger
            if op in ("move_up", "move_down"):
                delta = -1 if op == "move_up" else 1
                m.trigger_display_order = moved_display_order(
                    list(m.trigger_display_order), index, delta
                )
                return None
            return m.remove_trigger(index)

        if op == "new":
            label, select, refresh, refresh_arg = "New trigger", len(manager.triggers), "panel", None
        elif op == "copy":
            # copy_trigger() appends the copy and then moves it directly below
            # its source, so the copy lands at index + 1, not at the end.
            label, select, refresh, refresh_arg = f"Copy trigger {index}", index + 1, "panel", None
        elif op in ("move_up", "move_down"):
            delta = -1 if op == "move_up" else 1
            label = f"Move trigger {index} {'up' if delta < 0 else 'down'}"
            select, refresh, refresh_arg = index, "order", (index, delta)
        else:
            label, select, refresh, refresh_arg = f"Delete trigger {index}", index, "panel", None

        with self._trigger_edit(model, label, refresh=refresh, refresh_arg=refresh_arg) as m:
            m.structural_edit(mutate)
        # Idempotent for move_up/move_down: the "order" refresh tier already
        # leaves the moved trigger selected, and `select` (its list index)
        # never changed, so this just re-confirms the same row through the
        # freshly patched _row_for_index.
        self.trigger_panel.select_trigger(select)

    def entry_structural_edit(
        self, op: str, index: int, kind: str, entry_index: int, type_id: int
    ) -> None:
        """Add, copy, or delete a condition or effect.

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

        # Pre-set, because _trigger_edit() reports a raised body rather than
        # re-raising it: on failure the block below never assigns.
        select = -1
        with self._trigger_edit(
            model, f"{op.title()} {kind}", content_touched=[index], refresh="entries"
        ) as m:
            trigger = m.manager().triggers[index]
            entries = trigger.conditions if kind == "condition" else trigger.effects
            if op == "new":
                # The generic private entry point rather than the public
                # new_condition.<name>() wrappers: 71 vocabulary names across
                # the shipped versions have no wrapper method at all
                # ("enable/disable_object", "or"). See test_private_api_guard.
                if kind == "condition":
                    trigger._add_condition(type_id)
                else:
                    trigger._add_effect(type_id)
                select = len(entries) - 1
            elif op == "copy":
                # deepcopy rather than a field-by-field copy, for restore()'s
                # own trap-5 reason: copying fields across would walk into
                # Effect.quantity's armour/attack bit-split.
                entries.append(copy.deepcopy(entries[entry_index]))
                select = len(entries) - 1
            else:
                if kind == "condition":
                    trigger.remove_condition(condition_index=entry_index)
                else:
                    trigger.remove_effect(effect_index=entry_index)
                select = min(entry_index, len(entries) - 1)

        # Outside the pair: the tree is rebuilt by _after_trigger_edit, so the
        # row to land on only exists once that has run.
        if self.mode == "triggers" and select >= 0:
            self.trigger_panel.select_entry(kind, select)

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
                        return None
                return None

        else:
            if not any(v.variable_id == variable_id for v in manager.variables):
                return
            label = f"Remove variable {variable_id}"

            def mutate(m):
                for variable in list(m.variables):
                    if variable.variable_id == variable_id:
                        m.variables.remove(variable)
                        return None
                return None

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
    def _unit_edit(self, model: UnitEditModel, label: str, players: Sequence[int]):
        """Open one recorded unit edit, and close it whatever happens.

        `players` is the caller's declaration of which player list(s) the
        upcoming edit will touch -- one for place/move/delete, both source and
        destination for reassign -- passed straight through to
        UnitEditModel.begin_unit_edit(). Enter only once a real change is
        confirmed (e.g. the new value differs from the current one): unlike
        _trigger_edit, there is no failure mode here that still needs a
        record, so a spurious enter would record a genuine phantom undo step.
        """
        model.begin_unit_edit(players)
        try:
            yield model
        except Exception:
            model.abort_unit_edit()
            raise
        model.commit_unit_edit(label, self.edit_history)
        self._after_unit_mutation()
        self._update_title()
        self._update_edit_actions()
        self._log_status(label)

    def _after_unit_mutation(self) -> None:
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
        """
        # A third invalidation, on the same terms as the two below: an
        # in-flight warm is walking the unit list this edit just changed.
        self._cancel_warms()
        if self._cache is not None:
            self._cache.invalidate_units()
            canvas_w, canvas_h = self._cache.canvas_dims(0)
            self._cache.invalidate_region((0, 0, canvas_w, canvas_h))
            self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        if self.mode != "units":
            return
        self._rebuild_unit_index()
        self._refresh_selection_view()

    def _update_edit_actions(self) -> None:
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

    def _move_history(self, record, move, action: str) -> None:
        """Shared undo/redo tail, branching on the record *before* the cursor
        moves -- a trigger record needs the trigger panel rebuilt and yields no
        tile indices, a tile record needs the incremental repaint, either a
        trigger or an option record needs the Map Options form put back in
        step with the models, and a unit record needs the render caches
        invalidated and the selection/inspector reconciled
        (_after_unit_mutation()).

        _update_title() is called here rather than from _apply_dirty(), which
        early-returns on an empty index list: since a trigger undo produces no
        dirty tiles, the window's "*" marker would otherwise never update on
        one. Every other _apply_dirty() caller already updates the title
        itself.
        """
        if record is None:
            # Still logs, exactly as paste_tile() and fill do on a genuine
            # no-op: silently doing nothing reads as a broken keybind rather
            # than as an empty history.
            self._update_edit_actions()
            self._log_status(f"Nothing to {action.lower()}")
            return
        dirty = move(
            self.scenario.map_manager.terrain,
            self.trigger_edits,
            self.option_edits,
            self.unit_edits,
            self.message_edits,
        )
        self._apply_dirty(dirty)
        if record.kind == "unit":
            self._after_unit_mutation()
        if record.kind == "trigger" and self.mode == "triggers":
            # Rebuilt wholesale rather than patched: the panel is a read-only
            # view over the parsed manager, and an undo can change trigger
            # membership, order and ids at once.
            self.trigger_panel.show_scenario(self.scenario)
        if record.kind in ("options", "trigger"):
            # Both kinds, not just "options": the exec-order row is shown in
            # the Map Options panel but recorded as a trigger edit, so undoing
            # one produces a "trigger" record while the user is looking at this
            # form. A no-op outside the mode.
            self._repopulate_map_options()
        if record.kind == "options":
            # A Players mode field, and a Diplomacy grid cell, both ride an
            # OptionsDiffRecord too (decision 1), so their own undo/redo
            # must refresh those panels the same way -- a no-op outside
            # their own mode, same as the call above.
            self._repopulate_players()
            self._repopulate_diplomacy()
        if record.kind == "messages":
            self._repopulate_messages()
        self._update_title()
        self._update_edit_actions()
        self._log_status(f"{action}: {record.label}")

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

    def save(self) -> None:
        if self.scenario is None:
            return
        if self._untitled:
            # No path to write back to yet -- same first-save-needs-a-name
            # case Save As already handles.
            self.save_as()
            return
        path = str(self.scenario.path)
        try:
            result = write_scenario(
                self.scenario,
                path,
                triggers=self.trigger_edits,
                options=self.option_edits,
                units=self.unit_edits,
                messages=self.message_edits,
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}")
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:
            self._log_status(f"Save failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        self.edit_history.mark_saved()
        self._update_title()
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
        try:
            # triggers= is what keeps a trigger edit from being silently
            # dropped on save. Inert until the editing UI creates the model:
            # write_scenario() treats None exactly as the two-argument call it
            # replaces, so this changes no byte of any save made today.
            result = write_scenario(
                self.scenario,
                path,
                triggers=self.trigger_edits,
                options=self.option_edits,
                units=self.unit_edits,
                messages=self.message_edits,
            )
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}")
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:
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
        if not result.wrote:
            self._log_status("No changes to save")
        else:
            names = ", ".join(p.name for p in result.backups)
            suffix = f" (backed up {names})" if names else ""
            self._log_status(f"Saved to {path}{suffix}")

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

        One body with two cancels (rather than six new call sites enumerated
        separately) is what keeps this correct as the set of warms grows:
        the correctness argument is "every mutating path cancels first," and
        that can't drift out of sync with itself. Also resets the margin
        warm's pan-direction baseline (see self._last_viewport_chunk_target's
        own comment) -- a stale one just means the next ring after this
        mutation starts with no direction preference, never a wrong answer.

        Cheap and idempotent, so an over-broad call site costs at most a warm
        that has to be restarted on the next open -- always the right side to
        err on here."""
        self._level_warmer.cancel()
        self._margin_warmer.cancel()
        self._last_viewport_chunk_target = None

    def _start_level_warm(self) -> None:
        """Queues the neighbouring mip levels' sprite layers for an idle-time
        warm -- the 2026-09-04 plan's hook point.

        Called from load_scenario AFTER _render_current(), not from inside
        it: _render_current is also the re-render path for every terrain-
        style/quality/slider change, and those already cancel a warm rather
        than starting one.

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
        self._level_warmer.start(self._cache, mips)

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
            # every other caller re-renders the same document.
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
                )
                self.map_view.set_source(
                    tile_px, terrain_style="stepped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view,
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
                    sprites=self._sprites_enabled,
                )
                self.map_view.set_source(
                    tile_px, terrain_style="sloped", cache=self._cache, elevations=elevations, proj=proj,
                    reset_view=reset_view,
                )
            else:
                self._iso_elevations, self._iso_proj = None, None
                # sprites= at CONSTRUCTION (Track P3-g7), same reason the two
                # branches above pass it there rather than calling
                # set_sprites_enabled() afterwards.
                self._cache = FlatChunkCache(
                    self.scenario, tile_px, unit_filter=self._unit_filter,
                    sprites=self._sprites_enabled,
                )
                self.map_view.set_source(
                    tile_px, terrain_style="flat", cache=self._cache, reset_view=reset_view
                )
            # set_source() drops the pick index along with every other scene
            # item, so a style switch made while in Units mode has to rebuild
            # it -- otherwise selection silently stops working until the mode
            # is toggled off and back on.
            if self.mode == "units":
                self._rebuild_unit_index()
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
            except Exception as e:
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
            display_name = self.scenario.path.name

            self.edit_history.reset()
            # Dropped alongside the history it pushes records onto: a model
            # held across a document switch would splice the previous file's
            # trigger bytes into this one, patch the previous file's option
            # offsets into it, or splice the previous file's unit bytes in.
            self.trigger_edits = None
            self.option_edits = None
            self.unit_edits = None
            self.message_edits = None

            # Renders at whichever Terrain Style was already selected --
            # File > Open doesn't reset it back to Flat. _render_current()
            # pushes its own wait cursor/setEnabled(False) for this step,
            # nested safely inside this method's own (see both docstrings).
            # The "Loading..." status bar message deliberately stays up
            # through this render step too, not just the parse above --
            # cleared only once everything is actually done, right below.
            elapsed, tile_px = self._render_current()
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
                self.trigger_panel.show_scenario(self.scenario)
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
        self._untitled = False
        # Dropped alongside the cache it was warming -- otherwise a closed
        # document keeps ticking, holding the whole scenario alive.
        self._cancel_warms()
        self._cache = None
        self._iso_elevations, self._iso_proj = None, None
        self.edit_history.reset()
        self.trigger_edits = None
        self.option_edits = None
        self.unit_edits = None
        self.message_edits = None
        self.map_view.clear_image()
        # No edit tool has anything to act on with no map open; forcing Pan
        # (rather than just disabling the edit tools) keeps MapView's own
        # tool state in sync too -- same mechanism on_mode_changed already
        # uses when Terrain mode becomes unavailable.
        self.pan_action.setChecked(True)
        self.info.setPlainText("")
        self.trigger_panel.clear_document()
        self.map_options_panel.clear_document()
        self.players_panel.clear_document()
        self.diplomacy_panel.clear_document()
        self.messages_panel.clear_document()
        self.hover_label.setText(HOVER_IDLE_TEXT)
        # A stale (x, y) from the just-closed map must not outlive it -- the
        # bounds check in paste_tile()/copy_tile() would likely catch a
        # mismatch against a differently-sized map opened next anyway, but
        # relying on that coincidence is exactly the kind of leak
        # edit_history.reset() above is already here to prevent for edit
        # history. self._clipboard deliberately survives a close (a
        # clipboard outliving the file it was copied from is normal
        # clipboard semantics, and Paste is already disabled with no map
        # loaded via _update_tool_enabled() below) -- only the hover
        # position is map-relative state that needs clearing here.
        self._hover_tile = None
        self._update_tool_enabled()
        self._update_edit_actions()
        self._update_title()
        self._log_status(f"Closed {name}")

    def _update_info(self) -> None:
        s = self.scenario
        mm, um = s.map_manager, s.unit_manager

        terrain_hist = Counter(t.terrain_id for t in mm.terrain)
        lines = [
            f"File: {s.path.name}",
            f"Scenario version: {s.scenario_version}",
            f"Map size: {mm.map_width} x {mm.map_height}",
            f"Trigger tail (not parsed): {len(s.trigger_tail):,} bytes",
            "",
            "Terrain (top 8):",
        ]
        total_tiles = len(mm.terrain)
        for tid, count in terrain_hist.most_common(8):
            lines.append(f"  {name_for_terrain_id(tid):24s} {100 * count / total_tiles:4.1f}%")

        lines += ["", "Units per player:"]
        for player_id, units in enumerate(um.units):
            label = "GAIA" if player_id == 0 else f"Player {player_id}"
            lines.append(f"  {label:10s} {len(units):5d}")
        lines.append(f"  {'Total':10s} {sum(len(u) for u in um.units):5d}")

        self.info.setPlainText("\n".join(lines))

    def on_hover(self, tile: tuple[int, int] | None) -> None:
        self._hover_tile = tile
        if self.scenario is None:
            return
        if tile is None:
            self.hover_label.setText(HOVER_IDLE_TEXT)
            return
        x, y = tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            self.hover_label.setText(HOVER_IDLE_TEXT)
            return
        tile = mm.get_tile_safe(x, y)
        if tile is None:
            return
        self.hover_label.setText(
            f"({x}, {y})  {name_for_terrain_id(tile.terrain_id)}  elevation={tile.elevation}"
        )


# Module-globals for the crash hook: it installs before QApplication exists
# (see install_crash_hooks()), so it can't hold a constructor-injected
# reference to anything -- these are its only state.
_crash_host_ref: "weakref.ReferenceType[ViewerWindow] | None" = None
_crash_rate_limiter = crash_report.RateLimiter()
_crash_in_handler = False
# Held for process lifetime so the fd outlives this function -- letting it
# get GC'd would close the file, and this handle is the only thing that
# produces anything for a hard abort (a bare Fatal Python error, not a
# catchable exception; see tests/conftest.py).
_faulthandler_log_handle = None


def register_crash_host(window: "ViewerWindow") -> None:
    global _crash_host_ref
    _crash_host_ref = weakref.ref(window)


def _crash_host() -> "ViewerWindow | None":
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
    except Exception:
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
    _faulthandler_log_handle = open(dump_dir / crash_report.FAULTHANDLER_LOG_NAME, "w")
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
    migrated = asset_source.migrate_legacy_config()
    if migrated is not None:
        debug_log.log(f"Migrated config from {asset_source.LEGACY_CONFIG_PATH} to {migrated}")
    debug_log.log(f"Config file: {asset_source.CONFIG_PATH}")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "app_icon.png")))
    apply_theme(app, settings.get_dark_mode())
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
