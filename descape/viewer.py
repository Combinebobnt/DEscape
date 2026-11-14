"""PyQt5 viewer: open an .aoe2scenario file, pan/zoom the rendered map, and read
per-tile / per-unit details. Terrain mode's Draw/Elevate/Set Elevation tools
(v2) paint the map and save out via File > Save or Save As -- see
descape.scenario_write for the write path itself.

This module is the main window. ViewerWindow owns the menus, toolbar, mode
switching, the render pipeline and every edit callback, and is this module's
class plus 19 mixins, one module per area:

- viewer.py: the class head, __init__, mode switching, the title, the Qt
  event overrides and icon re-ink, the status log, the info-tree and combo
  widgets __init__ builds, the crash hooks and main().
- viewer_menus.py: the menu bar, keybind actions, About and Help, and the
  Settings, Debug Log and Mirror dialog openers.
- viewer_toolbar.py: tool overflow, the toolbar, the status bar, the
  editor panel and status log docks and their saved layout, and panel
  sizing.
- viewer_unit_inspector.py: unit field edits, selection reconcile and the
  garrison block.
- viewer_unit_move.py: the unit index, click and marquee select, drag, group
  move, ghosts and nudge.
- viewer_unit_place.py: placement, gate plans, the place preview, scatter,
  wall menus, the Cliff tool and the wall run.
- viewer_unit_ops.py: rotate, variant, delete, reassign and replace.
- viewer_unit_edit.py: _unit_edit, _after_unit_mutation and the chunk-cache
  patch path.
- viewer_history.py: undo, redo and the refresh after a history move.
- viewer_view.py: unit filters, sprites and layers, overlay toggles.
- viewer_trigger_overlay.py: area edit, the tile and unit pickers and
  trigger overlay drawing.
- viewer_options.py: Map Options, Players, AI, Point of View, Disabled
  Objects, Swap Players, Diplomacy and Messages.
- viewer_tool_state.py: _update_tool_enabled, brush sync, tool options, tool
  and style selection.
- viewer_strokes.py: terrain strokes, shape commit, Convert, the Unit Brush
  and the Eraser.
- viewer_region.py: region select, the clipboard and its dialogs, paste,
  lift and transform, fill, the eyedropper and mirror.
- viewer_render.py: _apply_dirty, warms, placeholders, reach, the viewport,
  _render_current and refresh_map.
- viewer_triggers.py: trigger editing, from _ensure_trigger_edits to
  variable_structural_edit.
- viewer_file.py: game data, autosave, save, open, recent files, New Map,
  resize, load_scenario and close_scenario.
- viewer_find.py: Analysis, Find and Replace, trigger range rings and rulers.
- viewer_info.py: the Info page, player stats and hover text.

The widgets and dialogs it drives live beside it: map_view*.py,
trigger_panel*.py, map_options_panel.py, viewer_canvas.py, viewer_dialogs.py,
settings_dialog.py, mirror_dialog.py, and the viewer_common.py helpers they
share. Those import one way, never back into this module at runtime.

Rules of the split (tests/test_viewer_split.py checks them):

- A mixin defines no __init__, no Qt event override and no signal: all
  instance state is created in ViewerWindow.__init__.
- No name is defined by two of the 20 classes.
- The mixins precede QMainWindow in ViewerWindow's bases.
- A mixin module never imports this one at runtime. A module-level name two
  modules need lives in viewer_common.py.
- Tests and tools set a viewer module global through testkit.viewer_family,
  never on a module directly.

A branch cut before the split is ported with tools/split_class.py's
`split --like`."""

from __future__ import annotations

import faulthandler
import gc
import os
import random
import sys
import threading
import weakref
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import numpy as np
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer
from PyQt5.QtGui import (
    QIcon,
    QKeySequence,
    QPalette,
)
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAction,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QStackedWidget,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from descape import (
    __version__,
    ai_scripts,
    asset_source,
    autosave,
    catalog_thumbs,
    civ_art,
    cliff_chain,
    clipboard_history,
    composite_backend,
    crash_report,
    debug_log,
    eraser,
    game_dataset,
    garrison,
    gc_hold,
    help_window,
    iso_geometry,
    level_warm,
    library_compat,
    margin_warm,
    mod_graphics,
    move_turn,
    perf_trace,
    player_labels,
    qt_messages,
    qt_platform,
    region_clipboard,
    region_transform,
    settings,
    settings_file,
    sprite_prefetch,
    terrain_target,
    tool_option_bindings,
    trigger_clipboard,
    trigger_geometry,
    ui_icons,
    unit_brush,
    unit_references,
    unit_sprites,
    win_taskbar,
)
from descape.analysis_dialog import AnalysisDialog
from descape.autosave_dialog import RecoverAutosaveDialog
from descape.clipboard_dialog import ClipboardHistoryDialog
from descape.data_mod_menu import DataModMenu
from descape.diplomacy_panel import DiplomacyPanel
from descape.edit_history import (
    EditHistory,
)
from descape.find_dialog import FindDialog
from descape.history_dialog import EditHistoryDialog
from descape.map_options_panel import MapOptionsPanel
from descape.map_view import MapView
from descape.messages_model import MessagesEditModel
from descape.messages_panel import MessagesPanel
from descape.new_map_dialog import NewMapDialog
from descape.options_model import (
    OptionsEditModel,
)
from descape.players_panel import PlayersPanel
from descape.render_cache import FlatChunkCache, IsoChunkCache, SlopedChunkCache
from descape.render_cache_splice import UnitSplice
from descape.scenario_io import (
    LoadedScenario,
    unit_field_supported,
)
from descape.terrain_panel import TerrainPanel
from descape.text_edits import _MultiLineEdit
from descape.trigger_model import (
    TriggerEditModel,
)
from descape.trigger_panel import TriggerPanel
from descape.trigger_panel_status import StatusStyle
from descape.unit_model import (
    UnitEditModel,
)
from descape.units_panel import UnitsPanel
from descape.viewer_common import (
    _DEFAULT_WALL_CONST,
    _LEFT_PAGE_FOR_MODE,
    _LEFT_PAGE_INFO,
    HOVER_IDLE_TEXT,
    FontScaledWidth,
    StatRow,
    _RotationDrag,
    _sync_hi_res,
    _team_colors,
)
from descape.viewer_dialogs import CrashReportDialog, apply_theme, apply_ui_font
from descape.viewer_file import _ViewerFileMixin
from descape.viewer_find import _ViewerFindMixin
from descape.viewer_history import _ViewerHistoryMixin
from descape.viewer_info import _ViewerInfoMixin
from descape.viewer_menus import _ViewerMenusMixin
from descape.viewer_options import _ViewerOptionsMixin
from descape.viewer_region import _ViewerRegionMixin
from descape.viewer_render import _ViewerRenderMixin
from descape.viewer_strokes import _ViewerStrokesMixin
from descape.viewer_tool_state import _ViewerToolStateMixin
from descape.viewer_toolbar import _ViewerToolbarMixin
from descape.viewer_trigger_overlay import _ViewerTriggerOverlayMixin
from descape.viewer_triggers import _ViewerTriggersMixin
from descape.viewer_unit_edit import _ViewerUnitEditMixin
from descape.viewer_unit_inspector import _ViewerUnitInspectorMixin
from descape.viewer_unit_move import _ViewerUnitMoveMixin
from descape.viewer_unit_ops import _ViewerUnitOpsMixin
from descape.viewer_unit_place import _ViewerUnitPlaceMixin
from descape.viewer_view import _ViewerViewMixin

# Mode combo label -> the id self.mode holds. Only multi-word labels need an
# entry; everything else is the lowercased label, which is what self.mode was
# built from before this existed. Without it "Map Options" would become the id
# "map options", with a space, while every gate in this file compares against a
# single lowercase word.
_MODE_ID_FOR_LABEL = {"Map Options": "map_options"}


# Placeholder paints across a level change (TASK-031.25 D8): a new window's default.
# Only main() turns it on, so tests and tools that build a ViewerWindow measure the real level.
LIVE_PLACEHOLDERS = False
# The zoom-out reach warm (TASK-031.11 D9): a new window's default, opted in beside placeholders.
REACH_WARM = False


def _mode_id(mode_text: str) -> str:
    return _MODE_ID_FOR_LABEL.get(mode_text, mode_text.lower())


def _trigger_status_style() -> StatusStyle:
    """The Triggers panel's status marker, from Settings > Appearance (GH #166)."""
    return StatusStyle(
        marker=settings.get_trigger_status_marker(),
        ok=settings.get_trigger_status_color("ok"),
        problem=settings.get_trigger_status_color("problem"),
        color_ok_rows=settings.get_trigger_status_color_ok_rows(),
        color_sections=settings.get_trigger_status_color_sections(),
    )


class _InfoTree(QTreeWidget):
    """View page 0's one table (GH #145). Takes Ctrl+C (the platform Copy
    key) from the window's Copy Region action while focused and copies the
    selected rows as tab-separated text.

    fit_columns() keeps the three columns inside the viewport (TASK-407): Value at
    its content width, Note filling the rest while there is room and hidden when not, Stat
    taking the rest and eliding. Copy still reads a hidden Note."""

    _NOTE_MIN_CHARS = 8
    _STAT_MIN_CHARS = 3
    _INDENT_SCALE = 0.6

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._apply_indentation()
        self.expanded.connect(lambda _index: self.fit_columns())
        self.collapsed.connect(lambda _index: self.fit_columns())

    def _apply_indentation(self) -> None:
        # A tighter tree than the style default leaves the elided Stat labels more room.
        self.resetIndentation()
        self.setIndentation(round(self.indentation() * self._INDENT_SCALE))

    def fit_columns(self) -> None:
        """Size Stat and Value and show or hide Note for the current viewport."""
        # A width set below can show a scroll bar, whose nested refit this pass's stale widths then undo.
        for _ in range(3):
            viewport = self.viewport().width()
            if viewport <= 0 or self.columnCount() < 3:
                return
            self._fit_columns_to(viewport)
            if self.viewport().width() == viewport:
                return

    def _fit_columns_to(self, viewport: int) -> None:
        header = self.header()
        chars = self.fontMetrics().averageCharWidth()
        value = self.sizeHintForColumn(1)
        stat_hint = self.sizeHintForColumn(0)
        room = viewport - value
        show_note = room >= stat_hint + self._NOTE_MIN_CHARS * chars
        header.setSectionHidden(2, not show_note)
        # Every width set here, no stretchLastSection: Qt left a re-shown Note stretched to a stale width.
        if show_note:
            self.setColumnWidth(0, stat_hint)
            self.setColumnWidth(1, value)
            self.setColumnWidth(2, room - stat_hint)
        else:
            # Depth-2 rows start three indents in; below the floor this scrolls instead of collapsing.
            stat = max(3 * self.indentation() + self._STAT_MIN_CHARS * chars, room)
            self.setColumnWidth(0, stat)
            self.setColumnWidth(1, max(value, viewport - stat))

    def viewportEvent(self, event) -> bool:
        handled = super().viewportEvent(event)
        # The viewport, not the tree: a vertical scroll bar appearing narrows it with no tree resize.
        if event.type() == QEvent.Resize:
            self.fit_columns()
        return handled

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.StyleChange:
            self._apply_indentation()
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self.fit_columns()

    def event(self, event) -> bool:
        if event.type() == QEvent.ShortcutOverride and event.matches(QKeySequence.Copy):
            event.accept()
            return True
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            QApplication.clipboard().setText(self.selected_text())
            event.accept()
            return
        super().keyPressEvent(event)

    def selected_text(self) -> str:
        """The selected rows in display order, one per line, cells tab-separated."""
        lines = []
        iterator = QTreeWidgetItemIterator(self)
        while iterator.value() is not None:
            item = iterator.value()
            if item.isSelected():
                cells = [item.text(col) for col in range(self.columnCount())]
                # The Player combo's own row copies as "Player<TAB><selection>".
                widget = self.itemWidget(item, 0)
                combo = widget.findChild(QComboBox) if widget is not None else None
                if combo is not None:
                    cells = [item.parent().text(0), combo.currentText()]
                lines.append("\t".join(cells).rstrip("\t"))
            iterator += 1
        return "\n".join(lines)


class _ShrinkingComboBox(QComboBox):
    """Content width when the cell allows, down to about 12 characters when not, so the
    View table's Player combo elides inside its cell rather than past the viewport (TASK-407)."""

    _MIN_CHARS = 12

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        text = self.fontMetrics().averageCharWidth() * self._MIN_CHARS + self.iconSize().width() + 4
        floor = self.style().sizeFromContents(QStyle.CT_ComboBox, option, QSize(text, hint.height()), self)
        return QSize(min(hint.width(), floor.width()), hint.height())

    def displayed_text(self, option: QStyleOptionComboBox | None = None) -> str:
        """The current text as painted: elided on the right to the edit field (Qt 5 only clips it)."""
        if option is None:
            option = QStyleOptionComboBox()
            self.initStyleOption(option)
        field = self.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, self)
        room = field.width() - 2 - (option.iconSize.width() + 4 if not option.currentIcon.isNull() else 0)
        return self.fontMetrics().elidedText(option.currentText, Qt.ElideRight, max(0, room))

    def paintEvent(self, event) -> None:
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.Text))
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        option.currentText = self.displayed_text(option)
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)


class _ComboCell(QWidget):
    """Holds a combo left-aligned in a tree cell, its maximum width capped to the cell's.
    The cap also bounds the adjustSize() an AdjustToContents combo runs after a relabel."""

    def __init__(self, combo: QComboBox) -> None:
        super().__init__()
        self._combo = combo
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(combo)
        layout.addStretch(1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._combo.setMaximumWidth(max(1, self.width()))


class ViewerWindow(
    _ViewerMenusMixin, _ViewerToolbarMixin, _ViewerUnitInspectorMixin,
    _ViewerUnitMoveMixin, _ViewerUnitPlaceMixin, _ViewerUnitOpsMixin,
    _ViewerUnitEditMixin, _ViewerHistoryMixin, _ViewerViewMixin,
    _ViewerTriggerOverlayMixin, _ViewerOptionsMixin, _ViewerToolStateMixin,
    _ViewerStrokesMixin, _ViewerRegionMixin, _ViewerRenderMixin,
    _ViewerTriggersMixin, _ViewerFileMixin, _ViewerFindMixin, _ViewerInfoMixin,
    QMainWindow,
):
    # GH #142: the View info page's minimumSizeHint width at the baseline font (the stats table).
    # A class attribute: FontScaledWidth only resolves on attribute access through a class.
    INFO_PAGE_MIN_USEFUL_WIDTH = FontScaledWidth(210)
    # File > Open Recent's Clear entry, None while the list is empty.
    clear_recent_action: QAction | None = None
    _icon_reink_pending = False

    def __init__(self):
        super().__init__()

        # Mod Art (TASK-378) is module state the renderers read. Seeded before any panel: the
        # Terrain panel's swatches resolve textures at construction, and that lookup is cached.
        mod_graphics.set_enabled(settings.get_view_layers().mod_art)
        mod_graphics.set_excluded(settings.get_mod_art_excluded())
        mod_graphics.set_included(settings.get_mod_art_included())
        # GH #241: Enhanced draws the game's x2 unit art where x1 would be enlarged.
        _sync_hi_res()
        # GH #272: a mod's reach is walked off the GUI thread; an edit repaints the whole map until it lands.
        unit_sprites.set_background_reach(True)
        asset_source.clear_art_caches()
        # (mod_graphics.generation(), whether any scanned mod ships art), for the row's greying.
        self._mod_art_available: tuple[int, bool] | None = None

        # Tool-overflow state (Stage 3 of the tool-overflow plan). Set
        # before the resize() call below: that call fires resizeEvent(),
        # which calls _apply_toolbar_overflow(), before _build_toolbar()
        # has run -- _tool_overflow_measured=False makes that call a no-op
        # until showEvent() has cached real (post-realization) widths.
        self._tool_overflow_measured = False
        self._tool_button_widths: dict[str, int] = {}
        self._pinned_toolbar_width = 0
        self._rotate_toolbar_width = 0
        self._toolbar_overflow_updating = False

        width, height = settings.get_window_size()
        self.resize(width, height)
        self.setMinimumSize(settings.MIN_WINDOW_WIDTH, settings.MIN_WINDOW_HEIGHT)

        self.scenario: LoadedScenario | None = None
        # GH #130: every player selector's labels, and the last (labels, colours) pushed.
        self._player_labels = player_labels.DEFAULT_LABELS
        self._player_labels_pushed: tuple | None = None
        # True while the current scenario came from File > New and has never been
        # saved: its LoadedScenario.path is UNTITLED_PATH (a display-only sentinel,
        # not a real file), so Save As must default to UNTITLED_NAME instead.
        # Cleared by a successful Save As, by File > Open, and by Close.
        self._untitled = False
        # File > New Map…'s panel, built on first open and reused (GH #243).
        self._new_map_panel: NewMapDialog | None = None
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
        # GH #160's guide window, created on first use by _show_help().
        self._help_window: help_window.HelpWindow | None = None
        self._current_tool = "pan"
        # TASK-287: the tool active when the Eyedropper was entered; None after a mode switch.
        self._eyedropper_return_tool: str | None = None
        # GH #125's wall button: the session value, like every Tool Options control (Settings > Tools
        # holds the default); a committed run makes its wall the last-used one.
        self._last_wall_const = _DEFAULT_WALL_CONST
        # The Cliff tool's in-flight chain (Track B Stage 2), None between
        # strokes. Initialised here rather than only in _begin_cliff_stroke()
        # so a release that somehow arrives without a matching press is a
        # no-op instead of an AttributeError.
        self._cliff_stroke: cliff_chain.ChainStroke | None = None
        # The map's cliffs, the scenario and its unit_gen as that stroke's press saw them.
        self._cliff_stroke_existing: list[cliff_chain.ExistingCliff] = []
        self._cliff_stroke_gen = 0
        self._cliff_stroke_scenario: weakref.ref | None = None
        # One of terrain_style.TERRAIN_STYLES -- see the toolbar combo built
        # in _build_toolbar(). Defaults to "stepped" (Flat alone no longer
        # shows elevation at all -- see render_tile()'s docstring). Persists
        # across a File > Open (a freshly loaded scenario renders in
        # whichever style was already selected), unlike self._current_tool
        # which File > Close resets to "pan", and across launches via
        # settings.get_terrain_style() (GH #182, user decision).
        self._terrain_style = settings.get_terrain_style()
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
        # GH #221: a .aoe2scenario dropped anywhere on the window opens it.
        self.setAcceptDrops(True)
        # Seconds the last load's pre-freeze gc.collect() took (_freeze_loaded_document).
        self._load_gc_collect_s: float | None = None
        # Tools > Map Analysis results, modeless and reused; see _show_analysis().
        self._analysis_dialog: AnalysisDialog | None = None
        # Edit > Find and Replace (GH #144), modeless and reused; see _show_find_dialog().
        self._find_dialog: FindDialog | None = None
        # Every Find and Replace warning and confirm goes through these, so a
        # test can stub them: a real box hangs a headless worker.
        self._warn: Callable[[str, str], None] = self._default_warn
        self._confirm: Callable[..., dict | None] = self._default_confirm
        # Bumped by every trigger edit and trigger undo/redo; keys Find's
        # cached reference counts (find_triggers.trigger_ref_counts()).
        self._trigger_gen = 0
        self._trigger_ref_cache: tuple[int, dict | None] | None = None
        # (_trigger_gen, trigger_range effects) behind _trigger_range_effects(); never a sentinel.
        self._trigger_range_cache: tuple[int, tuple] | None = None
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
        # and _cancel_warms() (viewer_render.py) is called by every path that mutates
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
        # The neighbour levels of the last level warm's anchor, pending level or not (TASK-031.25).
        self._warm_set: list[int] = []
        # The zoom-out reach (TASK-031.11): allowed per process (D9). Its (mip, range) entries
        # run on _load_warmer below every load entry (D3); _reach_keys is the last arm's set (D6).
        self._reach_warm_allowed = REACH_WARM
        self._reach_queue: list[tuple[int, tuple[int, int, int, int]]] = []
        self._reach_running: tuple[int, tuple[int, int, int, int]] | None = None
        self._reach_keys: tuple = ()
        # Placeholder paints (TASK-031.25): allowed per process (D8), and live only with preload on.
        self._placeholders_allowed = LIVE_PLACEHOLDERS
        # The canvas item that last reported a new pending level; both timers no-op for any other.
        self._placeholder_item = None
        # Re-anchors the warm out of paint, on the next event-loop turn.
        self._placeholder_pending_timer = QTimer(self)
        self._placeholder_pending_timer.setSingleShot(True)
        self._placeholder_pending_timer.setInterval(0)
        self._placeholder_pending_timer.timeout.connect(self._on_placeholder_pending)
        # Settles a placeholder some path left with no warm (PLACEHOLDER_STUCK_MS, read at arm time).
        self._placeholder_watchdog = QTimer(self)
        self._placeholder_watchdog.setSingleShot(True)
        self._placeholder_watchdog.timeout.connect(self._on_placeholder_watchdog)
        # The load's idle UnitEditModel build (_schedule_unit_model_prebuild).
        self._unit_model_prebuild = level_warm.IdleCall()
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
        # GH #38: the focused multi-line box whose own undo stack Edit > Undo/Redo
        # follow until focus-out commits it (_on_app_focus_changed()).
        self._text_undo_box: _MultiLineEdit | None = None
        # GH #64: a floating panel's box focus left, until focus lands somewhere (_on_app_focus_changed()).
        self._left_floating_box: _MultiLineEdit | None = None
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
        # The content_key serial (an int, so it pins no DiffRecord) the last autosave
        # captured: a document edited once and then left alone isn't rewritten every interval.
        self._autosaved_content: int | None = None
        # (folder, reason) of the default-folder fallback last logged, so a refused folder logs once per state.
        self._autosave_fallback_logged: tuple[str, str] | None = None
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
        # GH #126 Personality choices: scanned on the first Players-mode show,
        # dropped on an install-path change. Failed resolves are remembered
        # per key so the row stays disabled; the custom gate is per document.
        self._ai_choices: tuple[ai_scripts.AiChoice, ...] | None = None
        self._ai_roots: list | None = None
        self._ai_unresolvable: dict[str, str] = {}
        self._personality_custom_cache: tuple[object, bool] | None = None
        # Phase 3.5b's unit edit state, on the same lazy terms as the two
        # above -- built on the first real unit edit (place/move/delete/
        # reassign), so a document whose units were only browsed/selected
        # still saves byte-identically.
        self.unit_edits: UnitEditModel | None = None
        # TASK-548: None outside a compass drag.
        self._rotation_drag: _RotationDrag | None = None
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
        # GH #137's conditions/effects slot, beside it rather than sharing it, with the same lifetime.
        self._entry_clipboard: trigger_clipboard.EntryBlock | None = None
        # The _doc_id LEGACY_REORDER_MESSAGE was last said for.
        self._legacy_reorder_warned: str | None = None
        # The (_doc_id, exec mode) pairs an ID move's status line was said for.
        self._id_move_warned: set[tuple[str, str]] = set()
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
        # TASK-256: a unit selection or Select region being flipped or turned
        # in place, so the run stays one undo record. Assigned only through _set_lift().
        self._lift: region_clipboard.Lift | None = None
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
        # Last-applied MirrorDialog and ResizeDialog state, sticky the same way (TASK-514).
        self._mirror_defaults: dict = {}
        self._resize_defaults: dict = {}
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
        # The key the last Select press acted on (None on empty ground), which
        # its marquee must not toggle or add a second time (TASK-252).
        self._marquee_press_key: tuple[int, int] | None = None
        # GH #75: the cover tile of the press a group drag grabs by.
        self._drag_grab_tile: tuple[int, int] | None = None
        # The drag key whose over-cap preview already logged its status line.
        self._ghost_cap_logged: tuple[int, int] | None = None
        # GH #115: ((drag key, host key, unit_gen), refusal) for the drag preview's host outline.
        self._drop_preview_cache: tuple | None = None
        # TASK-549: ((doc, host key, const, unit_gen), refusal) for Place Unit into a host.
        self._place_garrison_cache: tuple | None = None
        # TASK-327, reset per press, load and close (_reset_drag_rule_memos()): a single drag's rule
        # answer, ((key, point, const, unit_gen), refused); a group's refused member ids, (selection, dx,
        # dy, unit_gen, gate-walls box, turn, ids); and its BlockerGrid, (selection, unit_gen, grid).
        # `selection` is compared with `is`.
        self._drag_rule_memo: tuple | None = None
        self._group_flags_memo: tuple | None = None
        self._group_grid_memo: tuple | None = None
        # GH #280: the Rotate inputs a live Move drag holds until its release, zeroed at the next press,
        # and each member's stand-in under it, (turn, unit_gen, id(unit), group) -> unit or stand-in.
        self._drag_turn = move_turn.MoveTurn()
        self._drag_stand_in_memo: dict[tuple, object] = {}
        # Convert brush: the model whose edit is open for the current drag
        # (None outside a stroke), and the timer coalescing its live repaint.
        self._convert_model: UnitEditModel | None = None
        # Trigger Pick from map's target, (trigger, entry kind, entry index,
        # field, is_list), or None when disarmed. Read by _needs_unit_index().
        # An eyedropper's adds a sixth slot, PICK_TYPE (GH #276).
        self._unit_picker: tuple | None = None
        # GH #115: the Garrison block's Pick from map host key, or None; never armed with _unit_picker.
        self._garrison_picker: tuple[int, int] | None = None
        # The tool it was armed from, which a user cancel returns to (TASK-252).
        self._garrison_return_tool: str | None = None
        # GH #138's Set Location / Set Area target: _unit_picker's shape plus a sixth
        # slot, "point" or "rect". Never armed with either picker above.
        self._tile_picker: tuple | None = None
        # GH #115: the Garrison block's Copy, as garrison.GarrisonClip rows. Session-only.
        self._garrison_clipboard: tuple = ()
        # GH #204: the Garrison block's Cut, or None. Never held together with a Copy.
        self._garrison_cut: garrison.GarrisonCut | None = None
        # Built on first ask, dropped on any unit mutation or document change.
        self._unit_ref_index: unit_references.ReferenceIndex | None = None
        self._convert_done: set[tuple[int, int]] = set()
        # GH #32: units this stroke left alone because the destination may not own them,
        # and the occupants of such a refused host, which stay with it.
        self._convert_refused: set[tuple[int, int]] = set()
        self._convert_held: set[tuple[int, int]] = set()
        # reference_id -> (player_id, unit) at first need in a stroke, for an occupant's host (TASK-474).
        self._convert_hosts: dict | None = None
        self._convert_touched_tiles: set[tuple[int, int]] = set()
        # One UnitSplice per Convert reassign or Unit Brush add not yet handed to
        # the cache, in mutation order, and its repaint throttle. Shared by the two.
        self._unit_stroke_pending_splices: list[UnitSplice] = []
        self._unit_stroke_refresh_timer = QTimer(self)
        self._unit_stroke_refresh_timer.setSingleShot(True)
        self._unit_stroke_refresh_timer.timeout.connect(self._flush_unit_stroke_refresh)
        # GH #170's Unit Brush: the open drag's plan state (None outside one) and its model.
        self._unit_brush: unit_brush.BrushStroke | None = None
        self._unit_brush_model: UnitEditModel | None = None
        # Units in the stroke's owner lists at its press, so its end sees every add, recorded or not.
        self._unit_brush_base = 0
        # The open Draw stroke's Target, snapshotted at its start (_init_stroke_state).
        self._stroke_target = terrain_target.TERRAIN
        # GH #184's Eraser: the open drag's state (None outside one) and its repaint throttle.
        self._erase: eraser.EraseStroke | None = None
        self._erase_refresh_timer = QTimer(self)
        self._erase_refresh_timer.setSingleShot(True)
        self._erase_refresh_timer.timeout.connect(self._flush_erase_refresh)
        self._update_title()

        left = QVBoxLayout()

        self.hover_label = QLabel(HOVER_IDLE_TEXT)
        self.hover_label.setWordWrap(True)
        left.addWidget(self.hover_label)

        # GH #145: everything below the hover line is one table with its own scrollbar,
        # top-level rows File, Player, Terrain (top 8), Technical. Section items persist;
        # only their children are rebuilt.
        self.info_tree = _InfoTree()
        self.info_tree.setColumnCount(3)
        self.info_tree.setHeaderLabels(["Stat", "Value", "Note"])
        self.info_tree.setSortingEnabled(False)
        self.info_tree.setUniformRowHeights(True)
        self.info_tree.setRootIsDecorated(True)
        self.info_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self.info_tree.header()
        # Fixed, not ResizeToContents, Interactive or stretched: _InfoTree.fit_columns() sets every width.
        header.setStretchLastSection(False)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.Fixed)
        self._info_sections: dict[str, QTreeWidgetItem] = {}
        for title in ("File", "Player", "Terrain (top 8)", "Technical"):
            item = QTreeWidgetItem([title])
            self.info_tree.addTopLevelItem(item)
            item.setExpanded(True)
            self._info_sections[title] = item
        for item in self._info_sections.values():
            item.setFirstColumnSpanned(True)
        left.addWidget(self.info_tree, stretch=1)

        # GH #5: one player's breakdown at a time; item data is the player id (0 = GAIA).
        # On the Player section's own spanned first row: one column gave it ~90 px, clipping tribe names.
        self.stats_player_combo = _ShrinkingComboBox()
        self.stats_player_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.stats_player_combo.setEnabled(False)
        self.stats_player_combo.currentIndexChanged.connect(lambda _index: self._update_player_stats())
        self._stats_player_row = QTreeWidgetItem(self._info_sections["Player"])
        self._stats_player_row.setFirstColumnSpanned(True)
        # Left-aligned at content width; fit_columns() keeps the cell inside the viewport, so a long label elides.
        self.info_tree.setItemWidget(self._stats_player_row, 0, _ComboCell(self.stats_player_combo))
        # What the Player section shows, for tests: (label, count, note, parent label).
        self.player_stats_rows: list[StatRow] = []

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
            on_entry_move=self.entry_move,
            on_variable_structural=self.variable_structural_edit,
            on_selection_changed=self._on_trigger_selection_changed,
            on_tag_rename=self.rename_trigger_tag,
            on_tag_remove=self.remove_trigger_tag,
            on_copy_triggers=self.copy_triggers,
            on_cut_triggers=self.cut_triggers,
            on_copy_entries=self.copy_entries_to_clipboard,
            on_tag_add=self.add_trigger_tag,
            on_tag_remove_selected=self.remove_trigger_tag_from,
            on_entry_field_group=self.set_entry_field_group,
        )
        self.trigger_panel.on_focus_changed = self._on_trigger_focus_changed
        self.trigger_panel.on_instruction_view = self._on_instruction_view
        self.trigger_panel.describe_unit_reference = self._describe_unit_reference
        self.trigger_panel.on_pick_unit = self._on_pick_unit_requested
        self.trigger_panel.unit_reference_tile = self._unit_reference_tile
        self.trigger_panel.on_go_to_group = self.go_to_trigger_group
        self.trigger_panel.text_box_lines = settings.get_text_box_lines
        self.trigger_panel.on_text_box_lines = settings.set_text_box_lines
        # GH #166: called per recompute, never captured, since _patch_unit_ref_index() can drop the index.
        self.trigger_panel.status_context = self._trigger_status_context
        self.trigger_panel.edited_trigger_indices = self._edited_trigger_indices
        self.trigger_panel.status_style = _trigger_status_style
        # GH #41: (trigger indices, emphasised entry refs) the map overlay draws,
        # or None. Held here, not read off the panel, so an undo in View mode can
        # re-derive it while the panel sits unrefreshed.
        self._trigger_overlay_ref: tuple[tuple[int, ...], frozenset[tuple[str, int]]] | None = None
        # TASK-550: the panel's last InstructionView, kept across modes so the box can come back.
        self._instruction_view = None
        # The shapes the overlay last drew, and GH #229's Edit Area target in them:
        # (trigger index, entry kind, entry index, area coords), or None.
        self._trigger_overlay_shapes: list[trigger_geometry.TriggerShape] = []
        # GH #167: the last ghost-cap status key, so a re-derive does not repeat its line.
        self._trigger_ghost_cap_logged: tuple | None = None
        self._area_edit_target: tuple | None = None
        # GH #41's: (trigger index, entry kind, entry index, location tile), or None.
        self._location_edit_target: tuple | None = None
        # Edit Area's own enable gate, Create Objects' trigger_editable.
        self._area_edit_editable = False
        self.map_options_panel = MapOptionsPanel(on_option_field=self.set_option_field)
        self.players_panel = PlayersPanel(
            on_player_field=self.set_player_field,
            on_player_count=self.set_player_count,
            on_disables_requested=self._show_disables_dialog,
            on_set_view=self.set_player_view,
            on_centre_view=self.centre_player_view,
            on_go_to_view=self.go_to_player_view,
            on_reset_view=self.reset_player_view,
            on_player_selected=self._on_player_selected,
            on_swap_requested=self._show_swap_dialog_from_panel,
        )
        self.diplomacy_panel = DiplomacyPanel(
            on_diplomacy_field=self.set_diplomacy_field, on_option_field=self.set_option_field
        )
        self.messages_panel = MessagesPanel(on_message_field=self.set_message_field)
        self.messages_panel.text_box_lines = settings.get_text_box_lines
        self.messages_panel.on_text_box_lines = settings.set_text_box_lines
        self.units_panel = UnitsPanel(
            on_unit_field=self._on_unit_field_changed,
            on_unit_field_step=self._on_unit_field_stepped,
            on_rotation_drag=self._on_rotation_drag,
            on_place_requested=self._on_units_panel_place_requested,
            on_pending_changed=self._refresh_place_ghost,
            on_garrison_add=self._on_garrison_add,
            on_garrison_delete=self._on_garrison_delete,
            on_garrison_navigate=self._on_garrison_navigate,
            map_size=lambda: (
                None
                if self.scenario is None
                else (self.scenario.map_manager.map_width, self.scenario.map_manager.map_height)
            ),
            on_garrison_pick=self._on_garrison_pick,
            on_garrison_copy=self._on_garrison_copy,
            on_garrison_paste=self._on_garrison_paste,
            on_garrison_unload=self._on_garrison_unload,
            on_garrison_owner=self._on_garrison_owner,
            on_garrison_cut=self._on_garrison_cut,
            on_garrison_duplicate=self._on_garrison_duplicate,
            # GH #186: a pull like map_size, never the lazily built unit_edits model.
            codec_field_supported=lambda name: self.scenario is None or unit_field_supported(self.scenario, name),
            # GH #372: pulled too, so an unsaved Players edit shows at once.
            portrait_civ_for=lambda player_id: civ_art.player_portrait_civ(self.scenario, player_id),
        )
        self.terrain_panel = TerrainPanel(
            display_mode=settings.get_terrain_picker_mode(), swatch_px=settings.get_terrain_grid_swatch_px()
        )
        self.terrain_panel.display_mode_changed.connect(settings.set_terrain_picker_mode)
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
            self._on_region_gesture,
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
        # GH #124: the wall drag preview leaves occupied tiles out.
        self.map_view.set_place_blocked_query(self._blocked_for_walls)
        # GH #365: Unit Line and Unit Shape read the pick and options once, at the press.
        self.map_view.set_shape_press_query(self._shape_press)
        # GH #128: Place Unit's hover ghost.
        self.map_view.set_place_preview(self.on_place_preview)
        # GH #167: the trigger-created units' ghosts, resolved here like the hover ghost.
        self.map_view.resolve_trigger_ghost = self._resolve_trigger_ghost
        # A lambda, not the function: read at call time, as new_unit() reads it.
        self.map_view.prepare_trigger_ghost_resolve = lambda: library_compat.depoison()  # noqa: PLW0108
        # GH #172: the wheel's rotate binding (Ctrl+wheel by default) turns that ghost while it shows,
        # a live Move drag (GH #280) and, under the "_selection" choices, the selection (GH #356),
        # through the router the Rotate keys take.
        self.map_view.set_place_ghost_query(self._place_ghost_active)
        self.map_view.set_place_rotate(self.on_unit_rotate)
        # GH #210: the wheel's brush-size binding. Lambdas, since _build_toolbar() makes the spinbox later.
        self.map_view.set_wheel_brush_hooks(
            lambda: self.brush_size_spin.isEnabled(),  # noqa: PLW0108
            lambda steps: self.brush_size_spin.stepBy(steps),  # noqa: PLW0108
        )
        # GH #245: the rotate binding also turns a just-pasted region, through the Rotate keys' own path.
        self.map_view.set_wheel_paste_hooks(
            # Mid-stroke or mid-drag on_region_transform() refuses, so the wheel does nothing there instead.
            lambda: self._region_turn_target() and not (self._stroke_in_progress() or self.map_view.region_drag_live()),
            lambda direction: self.on_region_transform(
                region_transform.ROTATE_CW
                if direction * MapView.PLACE_WHEEL_DIRECTION > 0
                else region_transform.ROTATE_CCW
            ),
        )
        # GH #356: no rotatability check, so a wall-only selection still gets on_unit_rotate's status line.
        # An armed Place target owns the wheel: a notch off the map, where its ghost hides, turns nothing.
        self.map_view.set_wheel_selection_query(
            lambda: bool(
                self.scenario is not None
                and self.mode == "units"
                and self._selection
                and not self._stroke_in_progress()
                and not self._place_rotate_armed()
            )
        )
        self.map_view.set_cliff_preview(self.on_cliff_preview)
        # The Move tool's press picks what to drag; GH #75: a drag on a group member moves the group.
        self.map_view.set_unit_drag_hooks(self.on_move_press, self._is_group_key)
        # TASK-325: under Units Select, Escape clears the unit selection before the region.
        self.map_view.set_unit_escape_hook(self._units_escape)
        # TASK-256: Enter keeps a live flip or turn, Escape reverts it, ahead of the two clears.
        self.map_view.set_lift_hooks(self._apply_lift_key, self.on_lift_revert)

        # Always-visible short status history, distinct from both the
        # transient single-line QMainWindow.statusBar() message and the
        # Help > Debug Log dialog (a separate popup, not always on screen).
        # _log_status() is the one place that feeds all three destinations
        # that matter for a given message.
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumBlockCount(200)
        self._update_log_min_height()

        self.map_view.setMinimumWidth(settings.MIN_SPLIT_PANE)
        # QGraphicsView's own 70 px minimumSizeHint shows only background on a
        # letterboxed render: collapsed in every sense but Qt's.
        self.map_view.setMinimumHeight(settings.MIN_MAP_PANE)
        self.left_stack.setMinimumWidth(settings.MIN_SPLIT_PANE)
        # GH #64: the map is the central widget; the mode panel and the log are docks.
        self.setCentralWidget(self.map_view)
        self._build_docks()

        # GH #58: File > Data Mod and its parse; before the menu and status bar it adds to.
        self.data_mod_menu = DataModMenu(self)
        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()
        # After _build_toolbar(): the saved state places the toolbars too.
        self._restore_window_layout()
        # The type picker opening or closing re-gates Edit > Cut/Copy/Paste, focus move or not.
        self.trigger_panel.detail_stack.currentChanged.connect(lambda _index: self._update_clipboard_actions())
        # Wired here rather than passed into MapView's constructor -- see the
        # placeholder lambda left there.
        self.map_view._on_zoom_changed = self._update_zoom_status
        self._update_zoom_status()
        # A2.5: the margin warm's entry point (2026-09-07 plan). No ordering
        # trap here the way _on_zoom_changed has -- _on_viewport_changed()
        # touches only self._cache/settings/self.map_view, none of which
        # need _build_status_bar() to have run first.
        self.map_view.on_viewport_changed = self._on_viewport_changed
        # After _build_toolbar(): the hook re-gates the Clear rulers button it builds.
        self.map_view.on_pinned_rulers_changed = self._on_pinned_rulers_changed
        self.map_view.on_pinned_rulers_edited = self._on_pinned_rulers_edited
        # GH #108: the "not added to Undo" note is logged once until a ruler is recorded again.
        self._ruler_unrecorded_noted = False
        self._ruler_hint_logged = False
        # TASK-252: Units mode opens on Pan, which no longer selects, so the first Pan click on a unit says why.
        self._pan_hint_logged = False
        self.map_view.on_pan_click = self._on_pan_click
        self.map_view.on_unit_pick = self._on_map_unit_picked
        self.map_view.on_unit_picker_cancel = self._on_unit_picker_cancel
        self.map_view.on_tile_pick = self._on_tile_picked
        self.map_view.on_rect_pick = self._on_rect_picked
        self.map_view.on_area_edit = self._on_area_edited
        self.map_view.on_location_edit = self._on_location_edited
        self.map_view.on_tile_picker_cancel = self.disarm_unit_picker
        self.map_view.on_unit_eyedrop = self.pick_unit_value
        self._tool_base_tips: dict[str, str] = {}
        self._build_keybind_actions()
        self._update_tool_enabled()
        # Settings > Tools' defaults onto the Tool Options row, once every action and handler exists.
        self.tool_option_bindings = tool_option_bindings.build(self)
        self._restore_tool_options()
        # After _build_keybind_actions(), which fills the table apply_icons() reads.
        ui_icons.apply_icons(self)
        self._apply_toolbar_style()
        # Once per launch: this is what bounds the untitled autosave keys
        # doc_key() deliberately leaves unbounded by count. Never fatal --
        # a failure here must not stop the window from opening.
        try:
            autosave.prune_stale()
        except OSError as e:
            debug_log.warning(f"Autosave prune skipped: {type(e).__name__}: {e}")
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
        # The first point where both _render_style inputs exist.
        self._sync_rotation_dial_projection()

        # App-global, so the handler ignores widgets outside this window.
        QApplication.instance().focusChanged.connect(self._on_app_focus_changed)

        # Registered last, not at the top of __init__: the crash hook's
        # no-host fallback exists precisely for an exception during this
        # constructor, and registering early would make a mid-__init__
        # crash find a "live" host and queue a dialog into an event loop
        # that never starts, surfacing nowhere -- sweep included.
        register_crash_host(self)

    def on_mode_changed(self, mode_text: str) -> None:
        with perf_trace.op("mode-switch"):
            self._switch_mode(mode_text)

    def _switch_mode(self, mode_text: str) -> None:
        # TASK-287: a remembered tool may have no button in the new mode.
        self._eyedropper_return_tool = None
        self.disarm_unit_picker()
        self.on_lift_finalize()  # TASK-256: a mode switch keeps a flip or turn as it stands
        self.mode = _mode_id(mode_text)
        self._apply_mode_shortcuts()
        # Entering a trigger-parsing mode is what makes Files, and so the
        # embedded-XS info line, readable: refresh it once when that changes.
        triggers_known = None if self.scenario is None else self.scenario.trigger_read_supported
        if self.mode != "terrain":
            # Every mode but Terrain opens on Pan: Units mode too, even from
            # Select, which it also shows (TASK-252 kept that rule), and the
            # others gate every edit tool off, so a checked one would dangle.
            self.pan_action.setChecked(True)
        with perf_trace.phase("view_mode"):
            self.map_view.set_mode(self.mode)
        # The keys survive every switch (TASK-359). A dormant mode hides the
        # highlight once here; a live one re-pushes it below, after the index work.
        if not self._selection_live():
            self.map_view.set_unit_selection(None)
        with perf_trace.phase("left_page"):
            self.left_stack.setCurrentIndex(_LEFT_PAGE_FOR_MODE.get(self.mode, _LEFT_PAGE_INFO))
            self._apply_left_pane_floor()
            self._show_panel_dock()
        # The markers themselves are mode-free; their emphasis is not, so
        # entering or leaving Players mode has to re-push them.
        with perf_trace.phase("camera_markers"):
            self._refresh_camera_markers()
        if self.mode == "units":
            with perf_trace.phase("widen_left"):
                self._widen_left_column(UnitsPanel.MIN_USEFUL_WIDTH)
            with perf_trace.phase("unit_index"):
                # Current or None in every mode (_after_unit_mutation's invariant), so reuse it.
                if self.map_view._unit_index is None:
                    self._rebuild_unit_index()
                else:
                    self._stack_cycle = None
                    # Players mode can recolour; set_mode("units") already re-showed the badges.
                    self.map_view.set_team_colors(_team_colors(self.scenario))
            with perf_trace.phase("units_panel"):
                # The kept selection's highlight and inspector come back (TASK-359).
                self._refresh_selection_view()
        elif self._selection_live():
            with perf_trace.phase("selection_view"):
                self._reconcile_selection()
        panel_mode = self.mode in ("triggers", "map_options", "players", "diplomacy", "messages")
        with perf_trace.phase("panel") if panel_mode else nullcontext():
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
            with perf_trace.phase("widen_left"):
                self._widen_left_column(TerrainPanel.MIN_USEFUL_WIDTH)
        if self.mode == "view":
            with perf_trace.phase("widen_left"):
                self._widen_left_column(self.INFO_PAGE_MIN_USEFUL_WIDTH)
        if self.scenario is not None and self.scenario.trigger_read_supported != triggers_known:
            with perf_trace.phase("info"):
                self._update_info()
        # Unphased: its forced Pan is a nested tool-switch op, already its own phase.
        self._update_tool_enabled()
        with perf_trace.phase("mode_status"):
            self._update_mode_status()
        # Last: a file loaded in another mode left a stale form until _show_triggers() above.
        self._push_instruction_overlay()
        # The combo's own text, not mode_text: _update_tool_enabled() above
        # can have forced the mode back (selecting Units while already in
        # Sloped), and reporting the mode the user asked for rather than the
        # one they got would contradict the combo they're looking at. Reading
        # the label rather than capitalizing self.mode also keeps "Map Options"
        # spelled the way the combo spells it.
        self._log_status(f"Mode changed to {self.mode_combo.currentText()}")

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
        self._apply_mode_shortcuts()
        self.map_view.set_mode(self.mode)
        self.left_stack.setCurrentIndex(_LEFT_PAGE_FOR_MODE.get(self.mode, _LEFT_PAGE_INFO))
        self._apply_left_pane_floor()
        self._show_panel_dock()
        self._selection = []
        self.map_view.set_unit_selection(None)
        self.units_panel.clear()
        self.pan_action.setChecked(True)
        # Same reason as on_mode_changed's own call: this leaves Players mode
        # too, and the emphasis has to go with it.
        self._refresh_camera_markers()
        self._push_instruction_overlay()
        self._log_status(f"Mode forced to {mode_text} -- Sloped has no unit selection yet")

    def _update_title(self) -> None:
        if self.scenario is None:
            self.setWindowTitle("DEscape")
            return
        marker = "*" if self.edit_history.is_dirty else ""
        tag = game_dataset.game().title_tag
        game = tag if tag is not None else self.data_mod_menu.title_suffix()
        self.setWindowTitle(f"{marker}{self.scenario.path.name}{game} — DEscape")

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

    def changeEvent(self, event) -> None:
        # Deferred and coalesced: apply_theme() sends several, and painting mid apply_theme() can segfault offscreen.
        if event.type() == QEvent.PaletteChange and not self._icon_reink_pending:
            self._icon_reink_pending = True
            QTimer.singleShot(0, self._reink_icons)
        super().changeEvent(event)

    def _reink_icons(self) -> None:
        self._icon_reink_pending = False
        ui_icons.apply_icons(self)

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
        # The poll first: a fire after the cancel re-arms the warms (_on_viewport_changed).
        self.map_view.stop_viewport_poll()
        # And out of zoom frames, or _cancel_warms() restarts their settle wait.
        self.map_view.set_preframes_allowed(False)
        self._stop_placeholder_timers()
        self._cancel_warms()
        self._unit_model_prebuild.cancel()
        # Same lifetime reasoning: the document frozen at load must be collectable again.
        gc.unfreeze()
        # And the warm gc hold's threshold and idle timer must not outlive the window.
        gc_hold.reset()
        # A window closed mid-stroke never sees the release that would re-enable gc.
        self.map_view.resume_stroke_gc()
        # Same lifetime reasoning as the warm above, and the teardown
        # trigger_panel_variables.py already does for its own variables dialog:
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
        self._close_find_dialog()
        # Same again for the trigger panel's Variables dialog.
        self.trigger_panel.close_variables_dialog()
        # Same lifetime problem as the warm above: close() doesn't destroy the
        # window, so an armed drain timer would still fire on a shared loop.
        # The record itself is deliberately NOT cleared here: a close
        # dispatched from inside a load's own processEvents() would then null
        # it out from under load_scenario's remaining writes to it.
        self._paint_report_timer.stop()
        self.data_mod_menu.shutdown()
        settings.set_window_size(self.width(), self.height())
        self._close_docks()
        event.accept()

    def _update_log_min_height(self) -> None:
        """Two lines of the log's CURRENT font, re-run whenever the UI font
        changes so the floor doesn't stay at construction-time metrics.

        setMinimumHeight, not a fixed height: min == max would pin the log
        dock and leave its separator looking draggable but inert. The saved
        dock height is deliberately not rescaled with the font: the user
        dragged it, and moving it under them would be worse than a dock they
        re-drag once."""
        line_height = self.status_log.fontMetrics().lineSpacing()
        self.status_log.setMinimumHeight(max(settings.MIN_LOG_PANE, line_height * 2 + 12))

    def _log_status(self, message: str, level: str = debug_log.INFO) -> None:
        self.status_log.appendPlainText(message)
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
        debug_log.log(message, level=level)

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

    def dragEnterEvent(self, event) -> None:
        if self._drop_paths(event):
            self._accept_as_copy(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        self.dragEnterEvent(event)

    def dropEvent(self, event) -> None:
        paths = self._drop_paths(event)
        if not paths:
            event.ignore()
            return
        self._accept_as_copy(event)
        # Deferred: the discard prompt's modal loop inside dropEvent would stall the drag source.
        QTimer.singleShot(0, lambda: self._open_dropped(paths))


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
            debug_log.error(f"Crash re-raised, already reported: {exc_type.__name__}")
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
        debug_log.error(f"Crash report written to {dump_path}")
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
    debug_log.warning(f"Crash report(s) found from a previous session: {names}")
    if host is not None:
        host._log_status(f"Crash report(s) found from a previous session: {names}", level=debug_log.WARNING)
    for path in pending:
        # Scrubbed here too: a failed rotation scrub, or a dump from before scrubbing existed.
        text = crash_report.scrub_user_paths(path.read_text(encoding="utf-8", errors="replace"))
        dialog = CrashReportDialog(
            host,
            summary=crash_report.extract_summary(text),
            dump_path=path,
            dump_text=text,
            from_last_session=True,
        )
        dialog.exec_()
        crash_report.mark_reported(path)


def _app_window() -> ViewerWindow:
    """The app's window: the one ViewerWindow that allows placeholder paints
    (TASK-031.25 D8), the zoom-out reach warm (TASK-031.11 D9) and
    pre-rendered zoom frames (TASK-031.71)."""
    window = ViewerWindow()
    window.set_placeholders_allowed(True)
    window.set_reach_warm_allowed(True)
    window.map_view.set_preframes_allowed(True)
    return window


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
    # Before the first settings read, so no memo holds a pre-import value (TASK-497).
    import_line = settings_file.apply_pending_import()
    if import_line is not None:
        debug_log.log(import_line)
    debug_log.log(f"Config file: {asset_source.CONFIG_PATH}")
    inherited_line = qt_platform.inherited_line()
    if inherited_line is not None:
        debug_log.log(inherited_line)
    platform_line = qt_platform.apply_default(prefer_xcb=settings.get_xwayland_on_wayland())
    if platform_line is not None:
        debug_log.log(platform_line)
    theme_line = qt_platform.apply_default_theme()
    if theme_line is not None:
        debug_log.log(theme_line)
    taskbar_line = win_taskbar.set_app_user_model_id()
    if taskbar_line is not None:
        debug_log.log(taskbar_line)
    qt_messages.install()
    app = QApplication(sys.argv)
    debug_log.log(f"Qt platform: {QApplication.platformName()}")
    # Wayland's app_id, matching packaging/DEscape.desktop (StartupWMClass is X11-only).
    app.setDesktopFileName("DEscape")
    app.setWindowIcon(QIcon(str(Path(__file__).resolve().parent / "app_icon.png")))
    apply_theme(app, settings.get_theme(), settings.get_theme_colors())
    # Before ViewerWindow(), so every widget is built at the final font and
    # nothing has to be re-measured after the fact.
    apply_ui_font(app, settings.get_ui_font_family(), settings.get_ui_font_size())
    window = _app_window()
    if import_line is not None:
        window.status_log.appendPlainText(import_line)  # debug_log has it already
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
    code = app.exec_()
    asset_source.shutdown_prefetch_pool()
    sprite_prefetch.shutdown_pool()
    catalog_thumbs.shutdown_thumb_pool()
    sys.exit(code)


if __name__ == "__main__":
    main()
