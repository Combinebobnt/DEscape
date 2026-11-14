"""ViewerWindow's toolbar: tool overflow, the toolbar builder and style, the status bar, the editor
panel and status log docks and their saved layout (GH #64), panel sizing and the mode, zoom and
tool-value status."""

from __future__ import annotations

from PyQt5.QtCore import QByteArray, QSize, Qt
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QLabel,
    QMainWindow,
    QMenu,
    QSpinBox,
    QToolButton,
    QWidget,
)

from descape import (
    brush,
    eraser,
    settings,
    terrain_classes,
    terrain_target,
    tool_option_bindings,
    tool_options,
    ui_icons,
    unit_shape,
)
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_style import STYLE_LABELS, label_for
from descape.toolbar_overflow import partition
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.viewer_common import (
    _TOOL_LABELS,
    ELEVATION_LEVEL_MAX,
    KeepOpenMenu,
    tool_applicable,
)

# settings.TOOLBAR_STYLES id -> the Qt style it applies.
_TOOLBAR_BUTTON_STYLES = {
    "icons": Qt.ToolButtonIconOnly,
    "icons_text": Qt.ToolButtonTextBesideIcon,
    "text": Qt.ToolButtonTextOnly,
}

# GH #64: saveState()/restoreState()'s version. Bump it whenever the set of docks or toolbars
# changes, so an older blob is refused whole rather than half-applied.
WINDOW_STATE_VERSION = 1
DEFAULT_PANEL_WIDTH = 340
# The status log's default height: four lines plus its frame.
_DEFAULT_LOG_LINES = 4
_LOG_FRAME_PX = 12


class _ViewerToolbarMixin:
    """ViewerWindow's toolbar, status bar and docks (viewer.py's docstring has the split's rules)."""

    def _build_docks(self) -> None:
        """GH #64: the editor panel (left_stack, whose pages swap with the
        mode) and the status log as docks that float, re-dock and close.
        _restore_window_layout() places them once the toolbars exist."""
        features = QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        self.panel_dock = QDockWidget(settings.MODE_LABELS[self.mode], self)
        self.panel_dock.setObjectName("panel_dock")
        self.panel_dock.setWidget(self.left_stack)
        self.panel_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.panel_dock.setFeatures(features)
        self.log_dock = QDockWidget("Status Log", self)
        self.log_dock.setObjectName("log_dock")
        self.log_dock.setWidget(self.status_log)
        self.log_dock.setFeatures(features)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.panel_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        # The panel column runs the full height, with the log under the map beside it.
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        # No animation: every frame of one resizes the map, and each resize renders. No tabs: a
        # panel tabbed behind the log would stay hidden through the mode change meant to show it.
        self.setDockOptions(self.dockOptions() & ~(QMainWindow.AnimatedDocks | QMainWindow.AllowTabbedDocks))
        # Not the dock's own toggleViewAction(), which takes the dock's title, the mode's name.
        self.panel_dock_action = QAction("&Editor Panel", self)
        self.panel_dock_action.setCheckable(True)
        self.panel_dock_action.setChecked(True)
        self.panel_dock_action.setToolTip(
            "Show or hide the current mode's panel. Drag its title bar to float it or dock it on the right"
        )
        self.panel_dock_action.triggered.connect(self.panel_dock.setVisible)
        self.panel_dock.toggleViewAction().toggled.connect(self.panel_dock_action.setChecked)
        # The log's title never changes, so its own toggle action keeps this text.
        log_action = self.log_dock.toggleViewAction()
        log_action.setText("&Status Log")
        log_action.setToolTip("Show or hide the status log. Drag its title bar to float it or dock it elsewhere")
        self.reset_layout_action = QAction("&Reset Layout", self)
        reset_tip = "Dock the editor panel left and the status log under the map, and show any hidden toolbar"
        self.reset_layout_action.setToolTip(reset_tip)
        self.reset_layout_action.setStatusTip(reset_tip)
        self.reset_layout_action.triggered.connect(lambda _checked=False: self._apply_default_layout())

    def _show_panel_dock(self) -> None:
        """Titles the panel dock with the mode, and re-shows it if it was
        closed: a mode change means its page is about to be needed."""
        self.panel_dock.setWindowTitle(self.mode_combo.currentText())
        self.panel_dock.show()

    def _apply_default_layout(self, panel_width: int | None = None, log_height: int | None = None) -> None:
        """Panel docked left, log docked under the map, both shown, and the
        toolbars shown: a first run, a saved state that will not restore, and
        View > Panels > Reset Layout. `log_height` is the bare log's, as the
        splitter layout before GH #64 saved it; the dock's title bar goes on top."""
        for dock, area in ((self.panel_dock, Qt.LeftDockWidgetArea), (self.log_dock, Qt.BottomDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()
        self.main_toolbar.show()
        self.tool_options_toolbar.show()
        if log_height is None:
            log_height = self.status_log.fontMetrics().lineSpacing() * _DEFAULT_LOG_LINES + _LOG_FRAME_PX
        title_bar = self.log_dock.sizeHint().height() - self.status_log.sizeHint().height()
        self.resizeDocks([self.panel_dock], [panel_width or DEFAULT_PANEL_WIDTH], Qt.Horizontal)
        self.resizeDocks([self.log_dock], [log_height + title_bar], Qt.Vertical)

    def _restore_window_layout(self) -> None:
        """The saved dock and toolbar arrangement, else the default one with
        the pre-GH #64 splitter sizes as its seed. A float saved on a screen
        that is gone comes back on a live one: Qt itself moves it
        (tests/test_window_layout.py pins that)."""
        state = settings.get_window_state()
        if state is not None and self.restoreState(QByteArray(state), WINDOW_STATE_VERSION):
            return
        split = settings.get_split_sizes()
        self._apply_default_layout(split[0] if split else None, settings.get_log_height())

    def _close_docks(self) -> None:
        """Saves the arrangement, then hides a floating dock: a window of its
        own, it would stay on screen after this window closes."""
        settings.set_window_state(bytes(self.saveState(WINDOW_STATE_VERSION)))
        for dock in (self.panel_dock, self.log_dock):
            if dock.isFloating():
                dock.hide()

    def _owns_widget(self, widget: QWidget | None) -> bool:
        """Whether `widget` is in this window, its docks included. isAncestorOf()
        alone stops at a floating dock, which is a window of its own."""
        return widget is not None and any(w.isAncestorOf(widget) for w in (self, self.panel_dock, self.log_dock))

    def _focus_map_view(self) -> None:
        """Keys back to the map after a panel click. From a floating panel,
        setFocus() alone leaves them in the panel's own window."""
        self.activateWindow()
        self.map_view.setFocus()

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
        # TASK-108: the Units-mode Rotate pair, measured hidden or not; _toolbar_pinned_width() adds it while shown.
        rotate_widgets = [self.main_toolbar.widgetForAction(a) for a in self._rotate_toolbar_actions]
        self._rotate_toolbar_width = sum(w.sizeHint().width() for w in rotate_widgets if w is not None)
        visible_rotate_width = self._rotate_toolbar_width if self._rotate_toolbar_shown() else 0
        self._pinned_toolbar_width = self.main_toolbar.sizeHint().width() - visible_tool_width - visible_rotate_width

    def _rotate_toolbar_shown(self) -> bool:
        return any(a.isVisible() for a in self._rotate_toolbar_actions)

    def _toolbar_pinned_width(self) -> int:
        """Everything on the bar that never overflows, the Rotate pair included while Units mode shows it."""
        return self._pinned_toolbar_width + (self._rotate_toolbar_width if self._rotate_toolbar_shown() else 0)

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
            available_px = self.width() - self._toolbar_pinned_width()
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
            self._update_more_tools_button(overflow_set)
        finally:
            self._toolbar_overflow_updating = False

    def _update_more_tools_button(self, overflow_ids: set[str] | None = None) -> None:
        """The button wears the active tool's label and icon while that tool sits in its menu."""
        if overflow_ids is None:
            in_menu = self.more_tools_menu.actions()
            overflow_ids = {t.tool_id for t in settings.TOOLS if getattr(self, f"{t.tool_id}_action") in in_menu}
        if self._current_tool in overflow_ids:
            label = _TOOL_LABELS.get(self._current_tool, self._current_tool)
            self.more_tools_button.setText(f"{label} ▾")
            new_icon = ui_icons.tool_icon(self._current_tool)
        else:
            self.more_tools_button.setText("More Tools ▾")
            new_icon = ui_icons.icon(ui_icons.WINDOW_ICONS["more_tools_button"])
        # setIcon() always relayouts the toolbar, and this runs on every resize.
        if new_icon.cacheKey() != self.more_tools_button.icon().cacheKey():
            self.more_tools_button.setIcon(new_icon)

    def _apply_toolbar_style(self) -> None:
        """Settings > Appearance's Toolbar buttons, then a fresh overflow pass on the new widths."""
        style = _TOOLBAR_BUTTON_STYLES[settings.get_toolbar_style()]
        self.main_toolbar.setToolButtonStyle(style)
        # Widgets added with addWidget() don't follow the toolbar's style.
        self.filters_button.setToolButtonStyle(style)
        self.more_tools_button.setToolButtonStyle(style)
        if not self._tool_overflow_measured:
            return
        # The width cache needs every tool on the bar (widgetForAction() is None in the menu).
        self._toolbar_overflow_updating = True
        try:
            for tool in settings.TOOLS:
                action = getattr(self, f"{tool.tool_id}_action")
                if action not in self.main_toolbar.actions():
                    self.main_toolbar.insertAction(self.more_tools_action, action)
            self.more_tools_menu.clear()
            self.more_tools_action.setVisible(False)
            # Hidden as at the first measure (before any map), so the pinned width doesn't depend on the mode.
            rotate_shown = [a.isVisible() for a in self._rotate_toolbar_actions]
            for action in self._rotate_toolbar_actions:
                action.setVisible(False)
            self.main_toolbar.layout().invalidate()
            self._cache_tool_button_widths()
            for action, shown in zip(self._rotate_toolbar_actions, rotate_shown, strict=True):
                action.setVisible(shown)
        finally:
            self._toolbar_overflow_updating = False
        self._apply_toolbar_overflow()

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")  # saveState() skips an unnamed toolbar
        toolbar.setMovable(False)
        # Fixed, not the style's metric, so the bar height and the overflow widths match on every platform.
        toolbar.setIconSize(QSize(24, 24))
        # Kept for the Stage 3 overflow logic (_apply_toolbar_overflow),
        # which needs to add/remove tool actions on this specific toolbar.
        self.main_toolbar = toolbar

        toolbar.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        # "Units" (plural) rather than the plan's "Unit", matching Triggers.
        # on_mode_changed() lowercases, so the label decides the gate string
        # everywhere -- self.mode == "units".
        self.mode_combo.addItems(list(settings.MODE_LABELS.values()))
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
        self.terrain_style_combo.setCurrentText(label_for(self._terrain_style))  # before connect(): no spurious signal
        self.terrain_style_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        # GH #55: one sentence per entry, shown on that entry; the combo's own tooltip joins them.
        style_tips = {
            "Flat": "Flat: no elevation.",
            "Stepped": "Stepped: each tile raised to its height, as blocks.",
            "Sloped": "Sloped: the ground slopes smoothly between heights.",
        }
        self.terrain_style_combo.setToolTip("How the map shows height. " + " ".join(style_tips.values()))
        for i in range(self.terrain_style_combo.count()):
            self.terrain_style_combo.setItemData(i, style_tips[self.terrain_style_combo.itemText(i)], Qt.ToolTipRole)
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
        self.pan_action.setToolTip("Drag to pan the map (every mode). Changes nothing")
        self.pan_action.toggled.connect(lambda on: on and self._on_tool_selected("pan"))
        tool_group.addAction(self.pan_action)
        toolbar.addAction(self.pan_action)

        self.draw_action = QAction("Draw", self)
        self.draw_action.setCheckable(True)
        self.draw_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.draw_action.setToolTip(
            "Paint the selected terrain type over the brush footprint (Terrain mode). Drag to paint a trail. "
            "Target chooses the base terrain or the layered terrain"
        )
        self.draw_action.toggled.connect(lambda on: on and self._on_tool_selected("draw"))
        tool_group.addAction(self.draw_action)
        toolbar.addAction(self.draw_action)

        self.fill_action = QAction("Paint Can", self)
        self.fill_action.setCheckable(True)
        self.fill_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.fill_action.setToolTip(
            "Flood-fill the connected region of matching terrain with the selected "
            "terrain type (Terrain mode). One click, no drag. Clicking a region that already "
            "has that terrain clears its layered terrain. Target chooses the base terrain or "
            "the layered terrain"
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
            "Hold Shift to snap the line to one of 16 directions. "
            "Target chooses the base terrain or the layered terrain"
        )
        self.draw_line_action.toggled.connect(lambda on: on and self._on_tool_selected("draw_line"))
        tool_group.addAction(self.draw_line_action)
        toolbar.addAction(self.draw_line_action)

        self.draw_rect_action = QAction("Draw Rectangle", self)
        self.draw_rect_action.setCheckable(True)
        self.draw_rect_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.draw_rect_action.setToolTip(
            "Drag to paint a rectangle of the chosen terrain (Terrain mode), filled "
            "or outlined. Hold Shift to square it. "
            "Target chooses the base terrain or the layered terrain"
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
            "centred on the clicked tile by its real footprint span. "
            "Drag to lay a connected run within the picked family: the drag "
            "direction sets which way the face points (front-facing drawn toward "
            "screen-right, back-facing toward screen-left). Start the drag on an "
            "existing cliff to continue it as one wall. "
            "Right-click or right-drag to erase cliffs; a run left with one neighbour gets its end cap back."
        )
        self.cliff_action.toggled.connect(lambda on: on and self._on_tool_selected("cliff"))
        tool_group.addAction(self.cliff_action)
        toolbar.addAction(self.cliff_action)

        # Phase 2.8's region select in Terrain mode, TASK-252's unit select in
        # Units mode: one action, dispatched on mode. It reads and never
        # writes on its own (Paste is the thing that writes).
        self.select_action = QAction("Select", self)
        self.select_action.setCheckable(True)
        self.select_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.select_action.setToolTip(
            "Terrain mode: drag a rectangle to select a region for Copy/Paste Region. "
            "Edit > Select All selects the whole map, Edit > Deselect or Escape clears it. "
            "Units mode: click a unit to select it (again to walk a stack), Ctrl+click toggles, "
            "Shift+click adds, and a drag draws a box, even from on top of a unit"
        )
        self.select_action.toggled.connect(lambda on: on and self._on_tool_selected("select"))
        tool_group.addAction(self.select_action)
        toolbar.addAction(self.select_action)

        # TASK-012 (GH #47), Units mode only: drags units, nothing else. Gated
        # like Place Unit; on_unit_move records the undo step.
        self.move_action = QAction("Move", self)
        self.move_action.setCheckable(True)
        self.move_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.move_action.setToolTip(
            "Drag a unit to move it (Units mode), or drag a selected unit to move the whole "
            "selection. A press on a stack drags its selected unit, so pick a lower one with "
            "Select first. Rotate turns it during the drag. Drop on a building to garrison; "
            "Shift keeps it a move"
        )
        self.move_action.toggled.connect(lambda on: on and self._on_tool_selected("move"))
        tool_group.addAction(self.move_action)
        toolbar.addAction(self.move_action)

        # Reads a tile's terrain + elevation into the toolbar params instead
        # of mutating -- same has_map/write_ok gate as Draw/Paint Can/Cliff
        # below (no squareness requirement: it never calls set_elevation()).
        # Added after Move to match settings.TOOLS, which overflow assumes.
        self.eyedropper_action = QAction("Eyedropper", self)
        self.eyedropper_action.setCheckable(True)
        self.eyedropper_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.eyedropper_action.setToolTip(
            "Click a tile to load its terrain and elevation into the toolbar "
            "params (Terrain mode). With Target on Layer it loads the tile's layered "
            "terrain where there is one. In Units mode, click a unit to make it the object "
            "Place Unit places, with its owner and its facing or variant; Shift+click takes the type only. "
            "With Switch back after a pick on, Ctrl+click keeps the Eyedropper"
        )
        self.eyedropper_action.toggled.connect(lambda on: on and self._on_tool_selected("eyedropper"))
        tool_group.addAction(self.eyedropper_action)
        toolbar.addAction(self.eyedropper_action)

        # Gated on has_map alone below, exactly like Pan and unlike every edit
        # tool: it reads the map and never writes, so no mode, write or
        # squareness gate applies to it.
        self.ruler_action = QAction("Ruler", self)
        self.ruler_action.setCheckable(True)
        self.ruler_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.ruler_action.setToolTip(
            "Measure the distance between two tiles (every mode) - drag, or click twice. "
            "Finished measurements stay on the map. Right-click an endpoint to remove one, "
            "or use Clear rulers. Changes nothing."
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
            "Click to place the chosen object for the chosen owner (Units mode), "
            "snapped to the grid and centred on the cursor, where the game's placement rules allow it "
            "(Free placement lifts them). With a wall picked, drag to place a run "
            "(one undo step per drag): it bends once, and Shift makes it one straight segment. "
            "Walls are always tile-snapped and unrestricted, so Off-grid and Free placement don't apply to them. "
            "A mouse wheel set to rotate (Settings > Keybinds) or the Rotate keys turn the preview "
            "before you click, and later placements keep that facing or variant."
        )
        self.place_unit_action.toggled.connect(lambda on: on and self._on_tool_selected("place_unit"))
        tool_group.addAction(self.place_unit_action)
        toolbar.addAction(self.place_unit_action)

        # GH #170, gated on Units mode like Convert. See the Unit Brush section.
        self.unit_brush_action = QAction("Unit Brush", self)
        self.unit_brush_action.setCheckable(True)
        self.unit_brush_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_brush_action.setToolTip(
            "Drag to paint the chosen object for the chosen owner across the brush (Units mode), "
            "one undo step per drag. Each tile gets one roll per drag at the Density, under the "
            "game's placement rules (Free placement lifts them), and a drag never stacks on itself. "
            "Hover the map and use the Rotate keys or the wheel's rotate binding to turn the preview; "
            "later drags paint that facing. Walls, gates and cliffs have their own tools."
        )
        self.unit_brush_action.toggled.connect(lambda on: on and self._on_tool_selected("unit_brush"))
        tool_group.addAction(self.unit_brush_action)
        toolbar.addAction(self.unit_brush_action)

        # GH #365, gated on Units mode like Place Unit: a line or a shape of the
        # catalog pick, walls as a wall run (_commit_wall_run()), anything else
        # as a lattice of units (_commit_unit_shape()). They replace Wall
        # Rectangle and Wall Circle.
        self.unit_line_action = QAction("Unit Line", self)
        self.unit_line_action.setCheckable(True)
        self.unit_line_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_line_action.setToolTip(
            "Drag to place a line of the chosen object for the chosen owner (Units mode), "
            "one undo step per drag, under the game's placement rules (Free placement lifts them). "
            "Spacing leaves that many tiles between units. A wall is placed as a wall run. "
            "Shift snaps the line to an even angle."
        )
        self.unit_line_action.toggled.connect(lambda on: on and self._on_tool_selected("unit_line"))
        tool_group.addAction(self.unit_line_action)
        toolbar.addAction(self.unit_line_action)

        self.unit_shape_action = QAction("Unit Shape", self)
        self.unit_shape_action.setCheckable(True)
        self.unit_shape_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_shape_action.setToolTip(
            "Drag a box to place a rectangle, ellipse or triangle of the chosen object for the chosen "
            "owner (Units mode), outlined or Filled, one undo step per drag, under the game's placement "
            "rules (Free placement lifts them). A wall is placed as a joined wall outline. "
            "Shift makes a square, a circle or an even triangle."
        )
        self.unit_shape_action.toggled.connect(lambda on: on and self._on_tool_selected("unit_shape"))
        tool_group.addAction(self.unit_shape_action)
        toolbar.addAction(self.unit_shape_action)

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

        # GH #184, Terrain and Units mode both. Gated on has_map like Convert;
        # unit writability is found lazily on the first stroke.
        self.eraser_action = QAction("Eraser", self)
        self.eraser_action.setCheckable(True)
        self.eraser_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.eraser_action.setToolTip(
            "Drag to erase the objects under the brush that match a ticked Target -- one undo "
            "step per drag. Objects hidden by Filters are never erased; erasing a building "
            "also erases its garrison."
        )
        self.eraser_action.toggled.connect(lambda on: on and self._on_tool_selected("eraser"))
        tool_group.addAction(self.eraser_action)
        toolbar.addAction(self.eraser_action)

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
        self.rotate_ccw_toolbar_action.setToolTip(self._rotate_units_tip)
        self.rotate_ccw_toolbar_action.triggered.connect(lambda: self.on_unit_rotate(-1))
        self.rotate_cw_toolbar_action = QAction("Rotate ↻", self)
        self.rotate_cw_toolbar_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.rotate_cw_toolbar_action.setToolTip(self._rotate_units_tip)
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
        param_toolbar.setObjectName("tool_options_toolbar")
        param_toolbar.setMovable(False)
        self.tool_options_toolbar = param_toolbar  # walked by the tool_options drift test

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
        # Wired here, once this row exists: auto beach below re-gates on the pick.
        self.terrain_panel.terrain_changed.connect(self._on_terrain_changed)

        # Which of a tile's two terrains the terrain tools write: its base, or
        # the layered terrain drawn over it. Session-only with no Settings > Tools
        # default (tool_options leaves it out): every launch starts on Terrain.
        self.terrain_target_label_action = param_toolbar.addWidget(QLabel(" Target: "))
        self.terrain_target_combo = QComboBox()
        self.terrain_target_combo.addItem("Terrain", terrain_target.TERRAIN)
        self.terrain_target_combo.addItem("Layer", terrain_target.LAYER)
        self.terrain_target_combo.addItem("Clear layer", terrain_target.CLEAR)
        self.terrain_target_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.terrain_target_combo.currentIndexChanged.connect(self._on_terrain_target_changed)
        self.terrain_target_param_action = param_toolbar.addWidget(self.terrain_target_combo)

        # Auto beach (2026-08-31 plan). A checkbox on Draw rather than a
        # separate Water tool -- the user's call, and it costs no ToolDef,
        # no QAction and no keybind audit. Like every control on this row from
        # here on, it holds the session value and starts at its Settings > Tools
        # default (tool_options.OPTIONS; _restore_tool_options()).
        self.auto_beach_check = QCheckBox("Auto beach")
        self.auto_beach_check.setChecked(tool_options.default("auto_beach"))
        self.auto_beach_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.auto_beach_check.toggled.connect(self._on_auto_beach_toggled)
        self.auto_beach_param_action = param_toolbar.addWidget(self.auto_beach_check)

        self.beach_combo = QComboBox()
        # data None is "Auto": derive the beach per ring tile from the land
        # on its shore side's climate -- see terrain_classes.auto_beach_for.
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
        self.beach_width_spin.setRange(tool_options.option("beach_width").minimum, tool_options.option("beach_width").maximum)
        self.beach_width_spin.setValue(tool_options.default("beach_width"))
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
        # A tick lasts the session, like the rest of the row; Settings > Tools holds the default.
        self.paint_trees_check = QCheckBox("Trees")
        self.paint_trees_check.setChecked(tool_options.default("paint_trees"))
        self.paint_trees_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paint_trees_check.setToolTip(
            "Auto-place the matching GAIA tree on each tile a forest terrain is painted onto "
            "(varied graphic per tile), like the in-game editor's Eye Candy option. Painting a "
            "different terrain over a tile removes its tree."
        )
        self.paint_trees_param_action = param_toolbar.addWidget(self.paint_trees_check)

        self.paint_eye_candy_check = QCheckBox("Eye candy")
        self.paint_eye_candy_check.setChecked(tool_options.default("paint_eye_candy"))
        self.paint_eye_candy_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paint_eye_candy_check.setToolTip(
            "Also auto-place non-tree doodads a terrain carries (grass tufts, jungle underbrush). "
            "Painting a different terrain over a tile removes them too."
        )
        self.paint_eye_candy_param_action = param_toolbar.addWidget(self.paint_eye_candy_check)

        self.level_param_label_action = param_toolbar.addWidget(QLabel(" Level: "))
        self.elevation_level_spin = QSpinBox()
        self.elevation_level_spin.setRange(0, ELEVATION_LEVEL_MAX)
        self.elevation_level_spin.setValue(tool_options.default("elevation_level"))
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
        sources_menu = KeepOpenMenu(self.convert_sources_button)
        self.convert_source_actions: dict[int, QAction] = {}
        default_sources = tool_options.default("convert_sources")
        for player_id in range(GAIA_PLAYER_ID, MAX_PLAYER_ID + 1):
            action = QAction(self._player_labels.text[player_id], self, checkable=True, checked=player_id in default_sources)
            sources_menu.addAction(action)
            self.convert_source_actions[player_id] = action
        self.convert_sources_button.setMenu(sources_menu)
        self.convert_sources_param_action = param_toolbar.addWidget(self.convert_sources_button)

        # GH #170's Unit Brush. Read once at press.
        self.unit_brush_density_label_action = param_toolbar.addWidget(QLabel(" Density: "))
        self.unit_brush_density_spin = QSpinBox()
        density = tool_options.option("unit_brush_density")
        self.unit_brush_density_spin.setRange(density.minimum, density.maximum)
        self.unit_brush_density_spin.setValue(density.default)
        self.unit_brush_density_spin.setSuffix(" %")
        self.unit_brush_density_spin.setToolTip(
            "The chance each tile under the brush gets a unit. Every tile is rolled once per drag, "
            "so going over it again adds nothing"
        )
        self.unit_brush_density_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_brush_density_param_action = param_toolbar.addWidget(self.unit_brush_density_spin)
        self.unit_brush_random_check = QCheckBox("Random facing/variant")
        self.unit_brush_random_check.setChecked(tool_options.default("unit_brush_random"))
        self.unit_brush_random_check.setToolTip(
            "Give each unit a random facing, or a random variant for trees and other scenery. "
            "Unticked, every unit gets the facing or variant last set by turning the preview, "
            "here or in Place Unit, or by the Eyedropper. Turning the preview unticks it"
        )
        self.unit_brush_random_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_brush_random_param_action = param_toolbar.addWidget(self.unit_brush_random_check)

        # GH #365's Unit Line and Unit Shape. Read once at press (_shape_press()).
        self.unit_shape_kind_combo = QComboBox()
        for value, label in unit_shape.SHAPE_KINDS:
            self.unit_shape_kind_combo.addItem(label, value)
        self.unit_shape_kind_combo.setCurrentIndex(
            tool_option_bindings.find_data(self.unit_shape_kind_combo, tool_options.default("unit_shape_kind"))
        )
        self.unit_shape_kind_combo.setToolTip(
            "The shape the drag box draws. Triangle stands its base on the box edge at the drag's start; "
            "Right Triangle puts its right angle there"
        )
        self.unit_shape_kind_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_shape_kind_param_action = param_toolbar.addWidget(self.unit_shape_kind_combo)
        self.unit_shape_filled_check = QCheckBox("Filled")
        self.unit_shape_filled_check.setChecked(tool_options.default("unit_shape_filled"))
        self.unit_shape_filled_check.setToolTip("Fill the shape with units, not just its outline. Walls ignore it")
        self.unit_shape_filled_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_shape_filled_param_action = param_toolbar.addWidget(self.unit_shape_filled_check)
        self.unit_shape_spacing_label_action = param_toolbar.addWidget(QLabel(" Spacing: "))
        self.unit_shape_spacing_spin = QSpinBox()
        spacing = tool_options.option("unit_shape_spacing")
        self.unit_shape_spacing_spin.setRange(spacing.minimum, spacing.maximum)
        self.unit_shape_spacing_spin.setValue(spacing.default)
        self.unit_shape_spacing_spin.setToolTip(
            "Empty tiles left between neighbouring units, measured from the largest picked object. Walls ignore it"
        )
        self.unit_shape_spacing_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_shape_spacing_param_action = param_toolbar.addWidget(self.unit_shape_spacing_spin)
        self.unit_shape_random_check = QCheckBox("Random facing/variant")
        self.unit_shape_random_check.setChecked(tool_options.default("unit_shape_random"))
        self.unit_shape_random_check.setToolTip(
            "Give each unit a random facing, or a random variant for trees and other scenery. "
            "Unticked, every unit gets the facing or variant last set by turning Place Unit's preview "
            "or by the Eyedropper"
        )
        self.unit_shape_random_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.unit_shape_random_param_action = param_toolbar.addWidget(self.unit_shape_random_check)

        # The Eraser's targets (GH #184), modelled on Sources above. A unit is
        # erased if it matches any ticked target; see descape/eraser.py.
        self.eraser_targets_button = QToolButton()
        self.eraser_targets_button.setText("Targets")
        self.eraser_targets_button.setPopupMode(QToolButton.InstantPopup)
        self.eraser_targets_button.setToolTip("Which objects this brush erases")
        self.eraser_targets_button.setEnabled(False)  # re-gated by _update_tool_enabled()
        targets_menu = KeepOpenMenu(self.eraser_targets_button)
        targets_menu.setToolTipsVisible(True)
        everything_action = targets_menu.addAction("Everything")
        everything_action.setToolTip("Tick every target except Cliffs")
        everything_action.triggered.connect(self._tick_every_erase_target)
        targets_menu.keep_open(everything_action)
        targets_menu.addSeparator()
        default_targets = tool_options.default("eraser_targets")
        self.eraser_target_actions: dict[str, QAction] = {}
        for name, label in eraser.TARGET_LABELS:
            if name == "cliffs":
                targets_menu.addSeparator()
            action = QAction(label, self, checkable=True, checked=name in default_targets)
            targets_menu.addAction(action)
            self.eraser_target_actions[name] = action
        self.eraser_target_actions["overlay"].setToolTip("Clear the layered terrain on every brush tile")
        self.eraser_target_actions["cliffs"].setToolTip(
            "Off, no other target erases a cliff: erasing one piece breaks its chain"
        )
        self.eraser_targets_button.setMenu(targets_menu)
        self.eraser_targets_param_action = param_toolbar.addWidget(self.eraser_targets_button)

        # GH #125: on row 2, not row 1, whose pinned width is measured once at show time.
        # GH #190: shown for the wall tools only (_WALL_TOOLS), and in the separator roll-up.
        self.wall_pick_button = QToolButton()
        self.wall_pick_button.setPopupMode(QToolButton.MenuButtonPopup)
        self.wall_pick_button.setToolTip(
            "Pick a wall to place. Click for the last-used wall, ▾ for another."
        )
        self.wall_pick_button.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.wall_pick_button.setMenu(QMenu(self.wall_pick_button))  # filled by _refresh_wall_menu()
        self.wall_pick_button.clicked.connect(lambda: self._arm_wall(self._last_wall_const))
        self.wall_pick_param_action = param_toolbar.addWidget(self.wall_pick_button)
        self._refresh_wall_menu()
        # Carries the unit_wall_pick shortcut only, like the mode_* actions: no menu entry.
        self.wall_pick_action = QAction("Pick Last-Used Wall", self)
        self.wall_pick_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.wall_pick_action.triggered.connect(lambda: self._arm_wall(self._last_wall_const))
        self.addAction(self.wall_pick_action)

        # Draw Rectangle's Fill/Outline toggle, the bool tool_options calls rect_filled.
        self.rect_fill_combo = QComboBox()
        self.rect_fill_combo.addItem("Filled", True)
        self.rect_fill_combo.addItem("Outline", False)
        self.rect_fill_combo.setCurrentIndex(
            tool_option_bindings.find_data(self.rect_fill_combo, tool_options.default("rect_filled"))
        )
        self.rect_fill_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.rect_fill_combo.setToolTip(
            "Filled paints the whole rectangle; Outline paints its border only, "
            "thickened by the brush"
        )
        self.rect_fill_combo.currentIndexChanged.connect(self._on_rect_fill_changed)
        self.rect_fill_param_action = param_toolbar.addWidget(self.rect_fill_combo)

        # Brush size/shape/thickness -- shown for any ToolDef.supports_brush tool
        # (Terrain, Elevate, Set Elevation), same visibility/enabled
        # convention as the two params above. One value each, shared by every brush tool.
        self.brush_param_label_action = param_toolbar.addWidget(QLabel(" Brush: "))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(brush.BRUSH_SIZE_MIN, brush.BRUSH_SIZE_MAX)
        self.brush_size_spin.setValue(tool_options.default("brush_size"))
        self.brush_size_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_size_spin.valueChanged.connect(self._on_brush_changed)
        self.brush_size_spin_action = param_toolbar.addWidget(self.brush_size_spin)

        self.brush_shape_combo = QComboBox()
        for shape in brush.BRUSH_SHAPES:
            # The two line labels follow the view: _refresh_brush_shape_labels().
            self.brush_shape_combo.addItem(brush.shape_label(shape, isometric=True), shape)
        self.brush_shape_combo.setCurrentIndex(brush.BRUSH_SHAPES.index(tool_options.default("brush_shape")))
        self.brush_shape_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_shape_combo.currentIndexChanged.connect(self._on_brush_shape_changed)
        self.brush_shape_combo_action = param_toolbar.addWidget(self.brush_shape_combo)

        # TASK-257: outline shapes only, so it shows and hides with the shape.
        self.brush_thickness_label_action = param_toolbar.addWidget(QLabel(" Thickness: "))
        self.brush_thickness_spin = QSpinBox()
        self.brush_thickness_spin.setRange(brush.BRUSH_THICKNESS_MIN, brush.BRUSH_THICKNESS_MAX)
        self.brush_thickness_spin.setValue(tool_options.default("brush_thickness"))
        self.brush_thickness_spin.setToolTip(
            "How many tiles wide the outline is, measured inward from the brush edge. "
            "A thickness that reaches the centre paints the filled shape"
        )
        self.brush_thickness_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_thickness_spin.valueChanged.connect(self._on_brush_changed)
        self.brush_thickness_spin_action = param_toolbar.addWidget(self.brush_thickness_spin)

        # Phase 2.8's paste filters -- which categories a Paste Region
        # actually writes. Visible only while Select is active (see
        # _update_tool_enabled()).
        self.paste_terrain_check = QCheckBox("Terrain")
        self.paste_terrain_check.setChecked(tool_options.default("paste_terrain"))
        self.paste_terrain_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_terrain_param_action = param_toolbar.addWidget(self.paste_terrain_check)

        self.paste_elevation_check = QCheckBox("Elevation")
        self.paste_elevation_check.setChecked(tool_options.default("paste_elevation"))
        self.paste_elevation_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_elevation_param_action = param_toolbar.addWidget(self.paste_elevation_check)

        self.paste_units_check = QCheckBox("Units")
        self.paste_units_check.setChecked(tool_options.default("paste_units"))
        self.paste_units_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_units_param_action = param_toolbar.addWidget(self.paste_units_check)

        # GH #176: lifts placement_rules' terrain and object rules for Place Unit clicks and
        # Move drags. Read at the click or drop, and by the ghosts (TASK-327).
        self.unrestricted_check = QCheckBox("Free placement")
        self.unrestricted_check.setChecked(tool_options.default("free_placement"))
        self.unrestricted_check.setToolTip(
            "Place or move a unit where the game would refuse it: on terrain it can't stand on, "
            "or on top of another object. It still snaps to the grid; tick Off-grid to lift that too."
        )
        self.unrestricted_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        # The hover ghost turns refused or not with no mouse move; a drag ghost on its next move.
        self.unrestricted_check.toggled.connect(self._refresh_place_ghost)
        self.unrestricted_param_action = param_toolbar.addWidget(self.unrestricted_check)

        # D2's off-grid toggle -- shown for any ToolDef.supports_off_grid
        # tool (Place Unit).
        self.off_grid_check = QCheckBox("Off-grid")
        self.off_grid_check.setChecked(tool_options.default("off_grid"))
        self.off_grid_check.setToolTip(
            "Place a unit exactly under the cursor instead of snapping it to the grid. "
            "Hold Alt for a one-off. Placement rules still apply."
        )
        self.off_grid_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        # The click reads the box directly; only the hover ghost needs telling.
        self.off_grid_check.toggled.connect(self._refresh_place_ghost)
        self.off_grid_param_action = param_toolbar.addWidget(self.off_grid_check)

        # GH #124. Read at press (the preview) and at commit; nothing else needs telling.
        self.skip_occupied_check = QCheckBox("Walls skip occupied tiles")
        self.skip_occupied_check.setChecked(tool_options.default("walls_skip_occupied"))
        self.skip_occupied_check.setToolTip(
            "Walls only. Trees, resources, buildings, units, obstacles and invisible blockers "
            "leave a gap in the run; passable eye candy does not. Existing walls and gates "
            "are never doubled either way."
        )
        self.skip_occupied_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.skip_occupied_param_action = param_toolbar.addWidget(self.skip_occupied_check)

        # GH #194. Read when a move commits, even while hidden (an arrow nudge works under any Units tool).
        self.gate_walls_check = QCheckBox("Moving a gate updates walls")
        self.gate_walls_check.setChecked(tool_options.default("gate_walls"))
        self.gate_walls_check.setToolTip(
            "When a gate moves, remove the walls under its new position and fill the gap it "
            "leaves with the wall it was set in."
        )
        self.gate_walls_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.gate_walls_param_action = param_toolbar.addWidget(self.gate_walls_check)

        # TASK-287, under the Eyedropper in either mode. The pick reads it directly.
        self.eyedropper_return_check = QCheckBox("Switch back after a pick")
        self.eyedropper_return_check.setChecked(tool_options.default("eyedropper_returns"))
        self.eyedropper_return_check.setToolTip(
            "After a pick, go back to the tool used before the Eyedropper. Ctrl+click picks "
            "and stays. A pick that changed nothing stays too."
        )
        self.eyedropper_return_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.eyedropper_return_param_action = param_toolbar.addWidget(self.eyedropper_return_check)

        # GH #229, under Pan in Triggers mode: Edit Area lets Pan drag the shown entry's
        # area and location mark; Move and Resize pick which parts grab, and Tile borders
        # widens the resize border. The last three grey out while it is off.
        self.area_edit_check = QCheckBox("Edit Area")
        self.area_edit_check.setToolTip(
            "Drag the selected trigger area's middle to move it, its edges and corners to "
            "resize it, or its location mark to move the location. One undo step per drag"
        )
        self.area_edit_check.setChecked(tool_options.default("area_edit"))
        self.area_edit_check.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.area_edit_check.toggled.connect(self._on_area_edit_toggled)
        self._area_option_params = [(param_toolbar.addWidget(self.area_edit_check), self.area_edit_check)]
        self._area_option_params.append((param_toolbar.addSeparator(), None))
        self.area_move_check = QCheckBox("Move")
        self.area_move_check.setToolTip("Drag the middle of the area, or the location mark, to move it")
        self.area_move_check.setChecked(tool_options.default("area_move"))
        self.area_resize_check = QCheckBox("Resize")
        self.area_resize_check.setToolTip("Drag an edge or corner of the area to resize it")
        self.area_resize_check.setChecked(tool_options.default("area_resize"))
        self.area_tile_borders_check = QCheckBox("Tile borders")
        self.area_tile_borders_check.setToolTip(
            "Resize from the whole outer ring of tiles instead of a thin border, on sides at least 3 tiles long"
        )
        self.area_tile_borders_check.setChecked(tool_options.default("area_tile_borders"))
        for check in (self.area_move_check, self.area_resize_check, self.area_tile_borders_check):
            check.setEnabled(False)  # re-gated by _update_tool_enabled()
            check.toggled.connect(self._on_area_options_toggled)
            self._area_option_params.append((param_toolbar.addWidget(check), check))
        self.map_view.set_area_edit_options(
            self.area_move_check.isChecked(), self.area_resize_check.isChecked(), self.area_tile_borders_check.isChecked()
        )

        # GH #108/#109: removes every pinned ruler. Shown with the Ruler only,
        # enabled only while one exists (re-gated by _on_pinned_rulers_changed too).
        self.clear_rulers_button = QToolButton()
        self.clear_rulers_button.setText("Clear rulers")
        self.clear_rulers_button.setToolTip("Remove every measurement pinned on the map")
        self.clear_rulers_button.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.clear_rulers_button.clicked.connect(self.map_view.clear_rulers)
        self.clear_rulers_param_action = param_toolbar.addWidget(self.clear_rulers_button)
        # GH #108: the remove gesture used to live only in the toolbar tooltip.
        self.ruler_hint_label = QLabel(" Right-click a ruler's end to remove it ")
        self.ruler_hint_param_action = param_toolbar.addWidget(self.ruler_hint_label)

    def _build_status_bar(self) -> None:
        # GH #67: why the document is read-only, while it is (_show_game_data_status()).
        self.game_data_banner = QLabel()
        self.game_data_banner.setVisible(False)
        self.statusBar().addPermanentWidget(self.game_data_banner)
        # GH #58: "Parsing <mod>…" while a data mod's .dat is read.
        self.statusBar().addPermanentWidget(self.data_mod_menu.parsing_label)
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

    def _apply_left_pane_floor(self) -> None:
        """The left pane's drag floor: in Units mode the Units page's own
        minimumSizeHint width, read on each entry so it follows the content
        (TASK-362, user decision 2026-10-14); in Terrain mode
        TerrainPanel.drag_floor() (TASK-551); in Triggers mode the trigger
        panel's minimumSizeHint width, set by its button rows (TASK-560);
        MIN_SPLIT_PANE in every other mode. Below the page's hint its placing
        card and category buttons clip."""
        floor = settings.MIN_SPLIT_PANE
        if self.mode == "units":
            floor = max(floor, self.units_panel.minimumSizeHint().width())
        elif self.mode == "terrain":
            floor = max(floor, self.terrain_panel.drag_floor())
        elif self.mode == "triggers":
            floor = max(floor, self.trigger_panel.minimumSizeHint().width())
        self.left_stack.setMinimumWidth(floor)

    def _widen_left_column(self, min_width: int) -> None:
        """Grow the editor panel to `min_width` on entering a mode whose page
        needs it, if it is narrower.

        Only ever widens, and only up to that threshold, so a width the user
        dragged to themselves (and which the saved layout kept) is left alone.
        The info panel's old 340 px is unusable for a trigger list, but picking
        a fixed Triggers width instead would throw away the user's own choice
        every time they switched modes.

        The threshold is a parameter in both places it is read -- the guard and
        the min() -- because each panel names its own MIN_USEFUL_WIDTH, and
        parameterising only one would silently widen to the other panel's.

        Docked, the map keeps MIN_SPLIT_PANE; floating, the panel's own window
        grows. A closed panel is left alone.
        """
        dock = self.panel_dock
        width = dock.width()
        if dock.isHidden() or width >= min_width:
            return
        if dock.isFloating():
            dock.resize(min_width, dock.height())
            return
        target = min(min_width, width + self.map_view.width() - settings.MIN_SPLIT_PANE)
        if target > width:
            self.resizeDocks([dock], [target], Qt.Horizontal)
            # Now, as the splitter's setSizes() did: the page filled next fits itself to this width.
            self.layout().activate()

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
