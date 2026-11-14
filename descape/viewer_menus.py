"""ViewerWindow's menus: the menu bar, keybind actions, About and Help, and the Settings, Debug Log and Mirror dialog openers."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QKeySequence,
)
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QDialog,
    QMenu,
    QMessageBox,
)

from descape import (
    __version__,
    edge_ticks,
    facet_preview,
    grid_overlay,
    help_guide,
    help_window,
    perf_trace,
    place_rotation,
    region_transform,
    settings,
    unit_pick,
    view_layers,
    viewer_canvas,
)
from descape.map_view import MapView
from descape.mirror_dialog import MirrorDialog
from descape.scenario_io import (
    BLANK_TEMPLATE_TILES,
)
from descape.settings_dialog import SettingsDialog
from descape.terrain_style import TERRAIN_STYLES
from descape.unit_filter import GAIA_PLAYER_ID, MAX_PLAYER_ID
from descape.viewer_common import (
    _LEFT_PAGE_FOR_MODE,
    _LEFT_PAGE_INFO,
    _SCATTER_TIP,
    _TOOL_LABELS,
    KeepOpenMenu,
)
from descape.viewer_dialogs import DebugLogDialog

# Help > About's library credit (GH #113). Static: the frozen build ships no dist-info, and
# tests/test_about_credit.py checks it against the installed one.
PARSER_CREDIT = (
    "Scenario parsing by AoE2ScenarioParser, by Kerwin Sneijders\n"
    "https://github.com/KSneijders/AoE2ScenarioParser\n"
    "Licensed under the GNU General Public License v3.0."
)


class _ViewerMenusMixin:
    """ViewerWindow's menu bar and keybinds (viewer.py's docstring has the split's rules)."""

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        # GH #243: one dialog for size, game data, terrain and elevation, in place
        # of the old one-action-per-size submenu.
        self.new_map_action = QAction("&New Map…", self)
        # checked=False must come first: QAction.triggered passes a positional
        # checked bool, which PyQt5 would otherwise bind into the lambda's first
        # parameter.
        self.new_map_action.triggered.connect(lambda checked=False: self.new_map_dialog())
        self.new_map_action.setToolTip("Start a blank scenario: pick its size, game data, terrain and elevation")
        file_menu.addAction(self.new_map_action)
        # file_new_default's action: a 120x120 Grass 1 map with no dialog. In no
        # menu, so it is added to the window for its keybind to fire.
        self.new_action = QAction("New Map (Default Size)", self)
        self.new_action.triggered.connect(lambda checked=False: self.new_map(BLANK_TEMPLATE_TILES))
        self.addAction(self.new_action)
        self.open_action = QAction("&Open .aoe2scenario…", self)
        self.open_action.setToolTip("Open a scenario file from disk")
        self.open_action.triggered.connect(self.open_file)
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu("Open &Recent")
        self.recent_menu.menuAction().setToolTip("Reopen a scenario you opened before")
        self._rebuild_recent_files_menu()
        self.close_action = QAction("&Close Map", self)
        self.close_action.setToolTip("Close the open scenario, asking first if it has unsaved changes")
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
        self.save_action.setToolTip("Write your changes back to the open file")
        self.save_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.save_action.triggered.connect(self.save)
        file_menu.addAction(self.save_action)
        self.save_as_action = QAction("Save &As…", self)
        self.save_as_action.setToolTip("Save the scenario to a new file name")
        self.save_as_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(self.save_as_action)
        # Enabled unconditionally, unlike the save actions: recovery is most
        # needed when nothing is open.
        self.recover_autosave_action = QAction("Reco&ver from Autosave…", self)  # Open Recent owns R
        self.recover_autosave_action.setToolTip("List the automatic backup copies of your edits and open one")
        self.recover_autosave_action.triggered.connect(self.show_recover_autosave)
        file_menu.addAction(self.recover_autosave_action)
        file_menu.addSeparator()
        self.data_mod_menu.build(file_menu)
        file_menu.addSeparator()
        self.exit_action = QAction("E&xit", self)
        self.exit_action.setToolTip("Quit DEscape, asking first if there are unsaved changes")
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setToolTip("Take back the last edit")
        self.undo_action.setEnabled(False)  # re-gated by _update_edit_actions()
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setToolTip("Put back the edit you last undid")
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
        # GH #137, GH #192: Cut takes triggers, conditions, effects and Units mode's selected units; there is no Cut Region.
        self.cut_action = QAction("Cu&t Triggers", self)
        self.cut_action.setEnabled(False)  # re-gated by _update_clipboard_actions()
        self.cut_action.triggered.connect(self._dispatch_cut)
        edit_menu.addAction(self.cut_action)
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
        self.clipboard_history_action.setToolTip(
            "The last few regions or units you copied this session; pick which one Paste uses"
        )
        self.clipboard_history_action.triggered.connect(self._show_clipboard_history)
        edit_menu.addAction(self.clipboard_history_action)
        # GH #30's edit-history window. Ships unbound like the two dialog
        # openers either side of it; "Histor&y" because Clipboard History
        # above already owns H in this menu.
        self.history_action = QAction("Histor&y…", self)
        self.history_action.setToolTip("Every edit you can undo, in order; double-click one to jump back or forward to it")
        self.history_action.triggered.connect(self._show_history_dialog)
        edit_menu.addAction(self.history_action)
        # Phase 2.8's whole-map region pair, through on_region_selected(); under
        # Units Select they act on the unit selection instead (TASK-325).
        self.select_all_action = QAction("Select &All", self)
        self.select_all_action.triggered.connect(self._dispatch_select_all)
        edit_menu.addAction(self.select_all_action)
        self.deselect_action = QAction("&Deselect", self)
        self.deselect_action.triggered.connect(self._dispatch_deselect)
        edit_menu.addAction(self.deselect_action)
        # Enabled by a committed region (in any mode) or a Units-mode selection
        # (GH #196); _update_tool_enabled() owns the gate. "U&nits": Cut owns T.
        self.scatter_action = QAction("Scatter U&nits in Region…", self)
        self.scatter_action.setToolTip(_SCATTER_TIP)
        self.scatter_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.scatter_action.triggered.connect(self.scatter_units_in_region)
        edit_menu.addAction(self.scatter_action)
        # GH #144. No default shortcut: rebindable from Settings like the rest.
        self.find_action = QAction("&Find and Replace…", self)
        self.find_action.setToolTip("Find placed objects by name, owner, kind, type or area, then replace, delete or reassign them")
        self.find_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.find_action.triggered.connect(self._show_find_dialog)
        edit_menu.addAction(self.find_action)

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
        rotate_menu = edit_menu.addMenu("R&otate Selection")  # Redo owns R
        rotate_menu.menuAction().setToolTip(
            "Turn the selected units one step or a quarter turn each (Units mode), or a pasted or selected "
            "region as a block; Rotate Group turns the selected units as one group"
        )
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
            # The toolbar pair reuses the first sentences only: it never turns a region.
            self._rotate_units_tip = (
                "Turn the selected units (Units mode). Gates cycle through their four "
                "orientations instead, since a gate stores its facing in its object type, "
                "and trees and scenery step through their graphic variants. Walls, cliffs "
                "and single-frame objects are skipped. While placing, with the cursor over "
                "the map, turns the placement preview instead, and while dragging with Move, "
                "turns the drag, landing with the drop"
            )
            action.setToolTip(
                self._rotate_units_tip + ". With no unit selected and a region just pasted, "
                "turns that whole region a quarter turn, as one undo step with the paste; with a "
                "region selected instead, turns it a quarter turn in place, the units on it included"
            )
            action.triggered.connect(handler)
            rotate_menu.addAction(action)
            setattr(self, attr, action)
            self._rotate_actions.append(action)
        # TASK-256: the selection as one group about its shared centre, in place.
        rotate_menu.addSeparator()
        rotate_menu.setToolTipsVisible(True)
        self._rotate_group_actions: list[QAction] = []
        for attr, label, step in (
            ("rotate_group_ccw_action", "Rotate Group ↺ (Quarter Turn)", region_transform.ROTATE_CCW),
            ("rotate_group_cw_action", "Rotate Group ↻ (Quarter Turn)", region_transform.ROTATE_CW),
        ):
            action = QAction(label, self)
            action.setEnabled(False)  # re-gated by _update_rotate_enabled()
            action.setToolTip(
                "Turn the selected units (Units mode) a quarter turn as one group about their shared "
                "centre. Each unit keeps its identity, so triggers that name it still work. A region "
                "you just pasted turns first; with no units selected, the selected region turns in "
                "place. Turn again to keep going: Enter applies, Esc undoes"
            )
            action.triggered.connect(lambda _=False, s=step: self.on_selection_transform(s))
            rotate_menu.addAction(action)
            setattr(self, attr, action)
            self._rotate_group_actions.append(action)

        # Cycle Variant: trees, doodads and scenery store a graphic-variant
        # index in `rotation`. Same selection-acting shape as Rotate above.
        variant_menu = edit_menu.addMenu("Cycle &Variant")
        variant_menu.menuAction().setToolTip("Change the graphic variant of the selected trees and scenery (Units mode)")
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

        # GH #197 mirrors a just-pasted region as a block; TASK-256 a unit
        # selection or a committed region in place, on_selection_transform()'s order.
        flip_menu = edit_menu.addMenu("F&lip Selection")  # Find and Replace owns F
        flip_menu.menuAction().setToolTip(
            "Mirror a just-pasted region, the selected units or the selected region across its own edges, "
            "swapping its West and North or its West and South corners"
        )
        flip_menu.setToolTipsVisible(True)
        self._flip_actions: list[QAction] = []
        # Action attrs keep their old names: keybind IDs and icons are keyed on them.
        for attr, label, step, corners, flat in (
            ("flip_horizontal_action", "Flip &West ↔ North", region_transform.FLIP_WEST_NORTH, "West and North", "left/right"),
            ("flip_vertical_action", "Flip West ↔ &South", region_transform.FLIP_WEST_SOUTH, "West and South", "top/bottom"),
        ):
            action = QAction(label, self)
            action.setEnabled(False)  # re-gated by _update_rotate_enabled()
            action.setToolTip(
                f"Mirror in place across the selection's own edges, swapping its {corners} corners ({flat} in "
                "Top-down Flat): a region you just pasted, else the selected units (Units "
                "mode) as one group across their box, else the selected region. "
                "Units keep their identity, so triggers that name them still work. Flip again to "
                "keep going: Enter applies, Esc undoes"
            )
            action.triggered.connect(lambda _=False, s=step: self.on_selection_transform(s))
            flip_menu.addAction(action)
            setattr(self, attr, action)
            self._flip_actions.append(action)

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
        self.settings_action = QAction("S&ettings…", self)  # Select Whole Stack owns S
        self.settings_action.setToolTip("General, appearance, tool default, saving and keybind preferences")
        self.settings_action.triggered.connect(self._show_settings)
        edit_menu.addAction(self.settings_action)

        # Map mirroring (Stage 1: terrain + elevation). New top-level menu
        # between Edit and View, one action. No setShortcut() here, same
        # reason as every other menu action in this method -- the shortcut
        # comes from the settings-backed keybind system
        # (settings.REBINDABLE_ACTIONS's "map_mirror" row).
        map_menu = menu_bar.addMenu("&Map")
        # Moved here from the File menu, a user request ("map_resize", was file_resize).
        self.resize_action = QAction("&Resize Map…", self)
        self.resize_action.setToolTip("Grow or shrink the open map, keeping terrain, objects and triggers in place")
        self.resize_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        # Same checked=False trap as the New Map actions.
        self.resize_action.triggered.connect(lambda checked=False: self.resize_map())
        map_menu.addAction(self.resize_action)
        self.mirror_action = QAction("&Mirror Map…", self)
        self.mirror_action.setToolTip("Copy one part of the map onto the rest by reflection or rotation, for a fair layout")
        self.mirror_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.mirror_action.triggered.connect(self._show_mirror_dialog)
        map_menu.addAction(self.mirror_action)
        # GH #57's Disabled Objects dialog, duplicated from the Players panel's button so it can be
        # keybound ("map_disables"). Moved here from the Edit menu, a user request.
        self.disables_action = QAction("&Disabled Objects…", self)
        self.disables_action.setToolTip("Choose which units, buildings and technologies each player cannot use")
        self.disables_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.disables_action.triggered.connect(self._show_disables_dialog)
        map_menu.addAction(self.disables_action)
        # GH #185, the Players panel's button as a keybindable menu item ("map_swap_players").
        self.swap_players_action = QAction("&Swap Players…", self)
        self.swap_players_action.setToolTip("Exchange two players' slots: units, settings, trigger references and diplomacy")
        self.swap_players_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.swap_players_action.triggered.connect(lambda checked=False: self._show_swap_dialog())
        map_menu.addAction(self.swap_players_action)

        view_menu = KeepOpenMenu("&View", menu_bar)
        menu_bar.addMenu(view_menu)
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

        # TASK-599: the zoom stop where one tile spans 96 device px, so x1 sprites draw 1:1.
        self.zoom_native_action = QAction("Zoom to &Native Sprite Size", self)
        self.zoom_native_action.setEnabled(False)  # re-gated by _update_zoom_native_action()
        self.zoom_native_action.triggered.connect(lambda checked=False: self._zoom_to_native())
        view_menu.addAction(self.zoom_native_action)
        view_menu.aboutToShow.connect(self._update_zoom_native_action)

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
        # (key, ghost item) of the last hover ghost; see on_place_preview().
        self._place_ghost_memo: tuple[tuple, object] | None = None
        # GH #172: the facing or variant Place Unit's ghost and next click use.
        self._place_rotation = place_rotation.PlaceRotation()
        # The Cliff tool's ghosts: the hover's (key, item) like the one above,
        # and one stroke's per-node (sprite draws, mark) by (const, rotation, x, y).
        self._cliff_hover_memo: tuple[tuple, object] | None = None
        self._cliff_ghost_parts: dict[tuple, tuple[list, tuple | None]] = {}
        # The hover's own such parts, and the look (style, elevation, colour) they were built for.
        self._cliff_hover_parts: dict[tuple, tuple[list, tuple | None]] = {}
        self._cliff_hover_look: tuple | None = None
        # (scenario weakref, (unit_gen, width, height), cliffs, tile index): see _cliff_existing().
        self._cliff_existing_memo: tuple | None = None
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
            "Grass and other passable eye candy never counts as hidden. "
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

        # GH #22. Persisted passive chrome, off by default for the same
        # reason as the rings below: it draws on the map rather than beside it.
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
        # GH #167. Persisted, ON by default for the overlay's reason, and independent of it.
        self.trigger_ghosts_action = QAction(
            "Show Trigger-&Created Units", self, checkable=True, checked=settings.get_trigger_ghosts()
        )
        self.trigger_ghosts_action.setToolTip(
            "The units the selected trigger's Create Object and Place Foundation effects would create, "
            "drawn see-through on their tiles."
            + ("" if facet_preview.GHOST_FACING_MEASURED else " Facing is not shown yet.")
        )
        view_menu.addAction(self.trigger_ghosts_action)
        # TASK-550. Persisted, ON by default for the overlay's reason, but Triggers mode only:
        # it covers the map's top centre, which is working space in Terrain or Units.
        self.instruction_overlay_action = QAction(
            "Show Instruction on &Map", self, checkable=True, checked=settings.get_instruction_overlay()
        )
        self.instruction_overlay_action.setToolTip(
            "In Triggers mode, the selected Display Instructions effect's message drawn over the map, "
            "sized and placed as the game draws it."
        )
        view_menu.addAction(self.instruction_overlay_action)

        # GH #49. Persisted passive chrome, but off by default: it draws over
        # the sprites, so it is opt-in.
        range_menu = KeepOpenMenu("&Range Rings", view_menu)
        view_menu.addMenu(range_menu)
        range_menu.menuAction().setToolTip(
            "Attack-range circles around buildings, sight circles around map revealers, and their colour"
        )
        self.range_rings_action = QAction(
            "&Selected Buildings", self, checkable=True, checked=settings.get_range_rings()
        )
        self.range_rings_action.setToolTip(
            "In Units mode, a circle around each selected building showing its attack range. "
            "Buildings only, and only ones that have a range. No technology effects are applied."
        )
        range_menu.addAction(self.range_rings_action)
        # GH #49 slice 2: dashed rings beside Selected Buildings' solid ones. On by default.
        self.range_rings_triggers_action = QAction(
            "&Trigger Changes", self, checkable=True, checked=settings.get_range_rings_triggers()
        )
        self.range_rings_triggers_action.setToolTip(
            "A dashed ring for each trigger that changes a selected building's range, one per trigger, "
            "as if that trigger fired once. Conditions, order and loops are not simulated. Needs the "
            "triggers read once: open Triggers mode."
        )
        range_menu.addAction(self.range_rings_triggers_action)
        # GH #155 / #153: always-on layers that follow the unit index, not the selection.
        self.range_rings_buildings_action = QAction(
            "All &Buildings", self, checkable=True, checked=settings.get_range_rings_buildings()
        )
        self.range_rings_buildings_action.setToolTip(
            "A circle around every building that has a range, selected or not, in every mode. "
            "Siege units are not included. No technology or trigger effects are applied."
        )
        range_menu.addAction(self.range_rings_buildings_action)
        self.range_rings_selected_revealers_action = QAction(
            "Selected &Map Revealers", self, checkable=True, checked=settings.get_range_rings_selected_revealers()
        )
        self.range_rings_selected_revealers_action.setToolTip(
            "In Units mode, a circle around each selected map revealer showing the area it reveals "
            "(its line of sight, e.g. 4, 10 or 20 tiles). Invisible Objects have no sight radius in "
            "the game data, so they draw none; trigger changes are not applied."
        )
        range_menu.addAction(self.range_rings_selected_revealers_action)
        self.range_rings_revealers_action = QAction(
            "All Map &Revealers", self, checkable=True, checked=settings.get_range_rings_revealers()
        )
        self.range_rings_revealers_action.setToolTip(
            "A circle around every map revealer showing the area it reveals (its line of sight, "
            "e.g. 4, 10 or 20 tiles), in every mode. Invisible Objects have no sight radius in the "
            "game data, so they draw none; trigger changes are not applied. Hidden while Show "
            "Invisible Objects is off."
        )
        range_menu.addAction(self.range_rings_revealers_action)
        range_menu.addSeparator()
        self.range_rings_by_owner_action = QAction(
            "Colour by O&wner", self, checkable=True, checked=settings.get_range_rings_by_owner()
        )
        self.range_rings_by_owner_action.setToolTip(
            "Each ring in its owner's player colour. GAIA keeps the Range ring colour "
            "from Settings > Appearance."
        )
        range_menu.addAction(self.range_rings_by_owner_action)

        # The ruler strip of tick marks outside the map border. Deliberately
        # NOT gated by mode or terrain style, unlike its two neighbours
        # above: it needs only the map dimensions plus, for the iso styles,
        # the projection, so there is nothing to grey and nothing to explain
        # in a greyed tooltip.
        ticks_menu = KeepOpenMenu("&Distance Ticks", view_menu)
        view_menu.addMenu(ticks_menu)
        ticks_menu.menuAction().setToolTip("Tile-count tick marks along the map border, and how far apart they are")
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
            action.setToolTip(f"A tick every {tiles} tiles along the map border")
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
        self.range_rings_by_owner_action.toggled.connect(self._on_range_rings_by_owner_toggled)
        self.range_rings_triggers_action.toggled.connect(self._on_range_rings_triggers_toggled)
        self.range_rings_buildings_action.toggled.connect(self._on_range_rings_buildings_toggled)
        self.range_rings_revealers_action.toggled.connect(self._on_range_rings_revealers_toggled)
        self.range_rings_selected_revealers_action.toggled.connect(self._on_range_rings_selected_revealers_toggled)
        self.player_cameras_action.toggled.connect(self._on_player_cameras_toggled)
        self.trigger_overlay_action.toggled.connect(self._on_trigger_overlay_toggled)
        self.trigger_ghosts_action.toggled.connect(self._on_trigger_ghosts_toggled)
        self.instruction_overlay_action.toggled.connect(self._on_instruction_overlay_toggled)
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
        grid_menu = KeepOpenMenu("&Grid", view_menu)
        view_menu.addMenu(grid_menu)
        grid_menu.menuAction().setToolTip("Tile grid lines over the map")
        self.grid_action = QAction("&Show", self, checkable=True, checked=settings.get_grid_overlay())
        self.grid_action.setToolTip(
            "A line on every tile boundary, with a stronger line at the spacing below. "
            "Lightness and thickness are in Settings > Appearance."
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
        grid_menu.addSeparator()
        major_every_group = QActionGroup(self)
        major_every_group.setExclusive(True)
        current_major_every = settings.get_grid_major_every()
        self.grid_major_every_actions: dict[int, QAction] = {}
        for tiles in grid_overlay.MAJOR_EVERY_STOPS:
            action = QAction(
                f"Stronger line every &{tiles} tiles", self, checkable=True, checked=tiles == current_major_every,
            )
            action.setToolTip(f"A stronger grid line every {tiles} tiles; the others stay one per tile")
            major_every_group.addAction(action)
            grid_menu.addAction(action)
            self.grid_major_every_actions[tiles] = action

        # Same two persisted halves as Distance Ticks, so the same submenu
        # shape. Not gated by mode or style either: MapView already holds
        # everything it needs.
        footprint_menu = KeepOpenMenu("&Footprint Outlines", view_menu)
        view_menu.addMenu(footprint_menu)
        footprint_menu.menuAction().setToolTip("Outlines of the tiles units occupy, and which units get one")
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
        for scope, label, tip in (
            (
                unit_pick.FOOTPRINT_SCOPE_MULTITILE,
                "&Multi-tile objects",
                (
                    "Outline only objects bigger than one tile: multi-tile buildings, deep-water "
                    "fish, dolphins and whales, large doodads, the 1x3 invisible blockers and "
                    "cliffs. Shore fish, turtles and oysters are one tile: View > Layers > Fish "
                    "Outlines rings them."
                ),
            ),
            (unit_pick.FOOTPRINT_SCOPE_BUILDINGS, "All &buildings", "Outline every building"),
            (
                unit_pick.FOOTPRINT_SCOPE_UNITS,
                "Units &only",
                (
                    "Outline every unit and object except buildings, walls and gates. Trebuchets and "
                    "other siege count as units. With Show Garrisoned Units on, a unit inside a "
                    "building outlines one tile at the building's point."
                ),
            ),
            (unit_pick.FOOTPRINT_SCOPE_ALL, "All &units", "Outline every unit, building and object"),
        ):
            action = QAction(label, self, checkable=True, checked=scope == current_scope)
            action.setToolTip(tip)
            footprint_group.addAction(action)
            footprint_menu.addAction(action)
            self.footprint_scope_actions[scope] = action
        footprint_menu.addSeparator()
        self.footprint_merged_action = QAction(
            "M&erge tiles", self, checkable=True, checked=settings.get_footprint_merged()
        )
        self.footprint_merged_action.setToolTip(
            "One outline around each footprint instead of one per tile. A diagonal gate, and a "
            "farm on steps of more than one level in Sloped, keep one outline per tile."
        )
        footprint_menu.addAction(self.footprint_merged_action)
        self.footprint_by_owner_action = QAction(
            "Colour by O&wner", self, checkable=True, checked=settings.get_footprint_by_owner()
        )
        self.footprint_by_owner_action.setToolTip(
            "Each outline in its owner's player colour. GAIA keeps the Footprint outlines colour "
            "from Settings > Appearance."
        )
        footprint_menu.addAction(self.footprint_by_owner_action)

        self.grid_action.toggled.connect(self._on_grid_overlay_toggled)
        self.grid_follow_action.toggled.connect(self._on_grid_follow_toggled)
        for tiles, action in self.grid_major_every_actions.items():
            # `on` leads and the guard is required, as for the tick intervals.
            action.toggled.connect(lambda on, tiles=tiles: on and self._on_grid_major_every(tiles))
        self.footprint_action.toggled.connect(self._on_footprint_outlines_toggled)
        for scope, action in self.footprint_scope_actions.items():
            # `on` leads, and the guard is required, for the same two reasons
            # the tick intervals above document.
            action.toggled.connect(lambda on, scope=scope: on and self._on_footprint_scope(scope))
        self.footprint_merged_action.toggled.connect(self._on_footprint_merged_toggled)
        self.footprint_by_owner_action.toggled.connect(self._on_footprint_by_owner_toggled)

        # View > Layers: render-category questions, a third axis beside
        # Filters (WHICH units) and Show sprites (WHETHER unit art draws).
        # Persisted across launches by user decision (GH #182), unlike Show
        # sprites and the unit filter: settings.get_view_layers() seeds it and
        # _on_layer_toggled() writes each change back.
        #
        # A registry loop rather than a block per layer: a new row plugs into
        # view_layers.LAYERS and the menu, the greying and the keybind table
        # all follow.
        #
        # self._layers is the WINDOW's copy, for exactly the reason
        # _sprites_enabled above is: a cache is rebuilt from scratch on every
        # style switch and file open, so something outside it has to remember
        # the choice. _render_current() re-applies it.
        self._layers = settings.get_view_layers()
        # Fish Outlines is a MapView item, not cache state: MapView keeps the flag across loads.
        self.map_view.set_fish_outlines(self._layers.fish_outlines)
        layers_menu = KeepOpenMenu("&Layers", view_menu)
        view_menu.addMenu(layers_menu)
        layers_menu.menuAction().setToolTip("Show or hide whole kinds of things the map draws")
        self.layer_actions: dict[str, QAction] = {}
        for spec in view_layers.LAYERS:
            action = QAction(spec.label, self, checkable=True, checked=getattr(self._layers, spec.layer_id))
            action.setEnabled(False)  # re-gated by _update_tool_enabled()
            action.setToolTip(spec.tooltip)
            layers_menu.addAction(action)
            self.layer_actions[spec.layer_id] = action
        # Second loop, after every checked= above has settled, same discipline
        # the Distance Ticks block documents. `on` must lead for the same
        # reason too: toggled passes checked as the first positional arg.
        for layer_id, action in self.layer_actions.items():
            action.toggled.connect(lambda on, lid=layer_id: self._on_layer_toggled(lid, on))

        # GH #64: the docks _build_docks() made. The log's own toggle action keeps its fixed title.
        view_menu.addSeparator()
        panels_menu = KeepOpenMenu("Pan&els", view_menu)
        view_menu.addMenu(panels_menu)
        panels_menu.menuAction().setToolTip("Show or hide the editor panel and the status log")
        panels_menu.addAction(self.panel_dock_action)
        panels_menu.addAction(self.log_dock.toggleViewAction())
        panels_menu.addSeparator()
        panels_menu.addAction(self.reset_layout_action)

        tools_menu = menu_bar.addMenu("&Tools")
        self.analysis_action = QAction("Map &Analysis…", self)
        self.analysis_action.setToolTip("Check the open map for common problems before you ship it. Changes nothing")
        self.analysis_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.analysis_action.triggered.connect(self._show_analysis)
        tools_menu.addAction(self.analysis_action)

        help_menu = menu_bar.addMenu("&Help")
        # GH #160: the guide, web links, the debug tools, then About last (the platform convention).
        self.help_guide_action = QAction("DEscape &Guide", self)
        self.help_guide_action.setToolTip("The built-in guide, open at the page for the current mode")
        # The lambda drops triggered's `checked`, which would otherwise arrive as topic_id.
        self.help_guide_action.triggered.connect(lambda _checked=False: self._show_help())
        help_menu.addAction(self.help_guide_action)
        self.help_keys_action = QAction("&Keyboard and Mouse", self)
        self.help_keys_action.setToolTip("Your current keys and mouse wheel settings, and the fixed gestures")
        self.help_keys_action.triggered.connect(lambda: self._show_help(help_guide.KEYS_TOPIC_ID))
        help_menu.addAction(self.help_keys_action)
        help_menu.addSeparator()
        self.web_guide_action = QAction("A&oE2 Scenario Editor Guide (ugc.aoe2.rocks)", self)
        self.web_guide_action.setToolTip(
            "Opens the AoE2 community guide to the in-game Scenario Editor in your web browser"
        )
        self.web_guide_action.triggered.connect(lambda: self._open_help_url(help_guide.UGC_SCENARIOS_URL))
        help_menu.addAction(self.web_guide_action)
        help_menu.addSeparator()
        self.release_notes_action = QAction(f"&Release Notes for {__version__}", self)
        self.release_notes_action.setToolTip(
            f"Opens the GitHub release page for DEscape {__version__} in your web browser"
        )
        self.release_notes_action.triggered.connect(
            lambda: self._open_help_url(help_guide.release_notes_url(__version__))
        )
        help_menu.addAction(self.release_notes_action)
        self.report_bug_action = QAction("Report a &Bug…", self)
        self.report_bug_action.setToolTip(
            "Opens a new GitHub issue in your web browser with your DEscape version and OS "
            "filled in. Nothing is sent until you submit it there"
        )
        self.report_bug_action.triggered.connect(
            lambda: self._open_help_url(help_guide.report_bug_url(__version__, *help_guide.environment_summary()))
        )
        help_menu.addAction(self.report_bug_action)
        self.github_action = QAction("DEscape on Git&Hub", self)
        self.github_action.setToolTip("Opens DEscape's GitHub page in your web browser")
        self.github_action.triggered.connect(lambda: self._open_help_url(help_guide.REPO_URL))
        help_menu.addAction(self.github_action)
        help_menu.addSeparator()
        self.debug_log_action = QAction("&Debug Log", self)
        self.debug_log_action.setToolTip("What the app has logged this session, for bug reports")
        self.debug_log_action.triggered.connect(self._show_debug_log)
        help_menu.addAction(self.debug_log_action)
        # No settings.get_/set_ backing on purpose (draw-perf plan Step 1): a
        # debug toggle that defaults off each session is the right behavior
        # anyway, and it sidesteps QSettings' string-to-bool read trap.
        self.perf_trace_action = QAction(
            "Perf &Trace", self, checkable=True, checked=perf_trace.is_enabled()
        )
        self.perf_trace_action.setToolTip(
            "Traces per-stroke drag-paint latency by phase, one-shot edits, pan/zoom repaints, level builds "
            "and event-loop stalls to Help > Debug Log. "
            "Also settable via the DESCAPE_PERF_TRACE=1 environment variable."
        )
        self.perf_trace_action.toggled.connect(perf_trace.enable)
        self.perf_trace_action.toggled.connect(viewer_canvas.set_stall_watchdog)
        self.perf_trace_action.toggled.connect(perf_trace.set_gc_hook)
        # toggled doesn't fire for the initial checked state (DESCAPE_PERF_TRACE=1).
        if perf_trace.is_enabled():
            viewer_canvas.set_stall_watchdog(True)
            perf_trace.set_gc_hook(True)
        help_menu.addAction(self.perf_trace_action)
        help_menu.addSeparator()
        self.about_action = QAction("&About", self)
        self.about_action.setToolTip("Version, license and credits")
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)

        # GH #55: QMenu hides action tooltips unless asked; submenus are menu-bar descendants too.
        for menu in menu_bar.findChildren(QMenu):
            menu.setToolTipsVisible(True)

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

        self.elevation_actions: dict[str, QAction] = {}
        for style in TERRAIN_STYLES:
            action_id = f"elevation_{style}"
            action = QAction(settings.get_action_label(action_id), self)
            action.triggered.connect(lambda _=False, s=style: self._select_elevation_view(s))
            self.addAction(action)
            self.elevation_actions[action_id] = action

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

        # GH #211: the trigger move shortcuts live on the trigger tree and the
        # entry tree, never on the window. Window-wide, Alt+Down would be taken
        # from every combo box in the trigger form, where it opens the popup.
        # No auto-repeat: each press is an undo record and a signature pass
        # over every trigger, and the five-place key covers distance.
        self.trigger_move_actions: dict[str, QAction] = {}
        for action_id, direction, far, edge in (
            ("trigger_move_up", -1, False, False),
            ("trigger_move_down", 1, False, False),
            ("trigger_move_up_far", -1, True, False),
            ("trigger_move_down_far", 1, True, False),
            ("trigger_move_top", -1, False, True),
            ("trigger_move_bottom", 1, False, True),
        ):
            action = QAction(settings.get_action_label(action_id), self)
            action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
            action.setAutoRepeat(False)
            action.triggered.connect(
                lambda _=False, d=direction, f=far, e=edge: self.trigger_panel.request_move(d, far=f, edge=e)
            )
            self.trigger_panel.tree.addAction(action)
            self.trigger_panel.entry_tree.addAction(action)
            self.trigger_move_actions[action_id] = action

        # Every tool's toolbar action reused directly -- no separate action
        # needed, just give it a shortcut too. Looked up via getattr rather
        # than hand-listed so this stays in sync with settings.TOOLS
        # automatically: a tool registered there without a matching
        # self.<tool_id>_action (built in _build_toolbar, before this runs)
        # is a wiring bug, and this is deliberately not defensive about it
        # -- getattr with no default raises immediately, which is the point
        # (see settings.ToolDef's own docstring).
        self._keybind_actions = {
            "file_new": self.new_map_action,
            "file_new_default": self.new_action,
            "file_open": self.open_action,
            "file_close": self.close_action,
            "file_save": self.save_action,
            "file_save_as": self.save_as_action,
            "file_recover_autosave": self.recover_autosave_action,
            "file_exit": self.exit_action,
            "edit_undo": self.undo_action,
            "edit_redo": self.redo_action,
            "edit_cut": self.cut_action,
            "edit_copy": self.copy_action,
            "edit_paste": self.paste_action,
            "edit_flip_horizontal": self.flip_horizontal_action,
            "edit_flip_vertical": self.flip_vertical_action,
            "edit_rotate_group_ccw": self.rotate_group_ccw_action,
            "edit_rotate_group_cw": self.rotate_group_cw_action,
            "edit_select_all": self.select_all_action,
            "edit_deselect": self.deselect_action,
            "edit_clipboard_history": self.clipboard_history_action,
            "edit_history": self.history_action,
            "edit_scatter_units": self.scatter_action,
            "edit_find_replace": self.find_action,
            "edit_settings": self.settings_action,
            "map_resize": self.resize_action,
            "map_mirror": self.mirror_action,
            "map_disables": self.disables_action,
            "map_swap_players": self.swap_players_action,
            "analysis_run": self.analysis_action,
            "view_isometric": self.iso_action,
            "view_distance_ticks": self.distance_ticks_action,
            "view_zoom_native": self.zoom_native_action,
            "view_show_sprites": self.show_sprites_action,
            "view_stack_badges": self.stack_badges_action,
            "view_grid_overlay": self.grid_action,
            "view_grid_follow": self.grid_follow_action,
            "view_footprint_outlines": self.footprint_action,
            "view_footprint_merged": self.footprint_merged_action,
            "view_footprint_by_owner": self.footprint_by_owner_action,
            "view_selection_owner_colour": self.selection_owner_colour_action,
            "view_range_rings": self.range_rings_action,
            "view_range_rings_by_owner": self.range_rings_by_owner_action,
            "view_range_rings_triggers": self.range_rings_triggers_action,
            "view_range_rings_buildings": self.range_rings_buildings_action,
            "view_range_rings_revealers": self.range_rings_revealers_action,
            "view_range_rings_selected_revealers": self.range_rings_selected_revealers_action,
            "view_player_cameras": self.player_cameras_action,
            "view_trigger_overlay": self.trigger_overlay_action,
            "view_trigger_ghosts": self.trigger_ghosts_action,
            "view_instruction_overlay": self.instruction_overlay_action,
            "view_panel_dock": self.panel_dock_action,
            "view_log_dock": self.log_dock.toggleViewAction(),
            "view_reset_layout": self.reset_layout_action,
            **{f"view_layer_{lid}": action for lid, action in self.layer_actions.items()},
            "view_pan_up": self.view_pan_up_action,
            "view_pan_down": self.view_pan_down_action,
            "view_pan_left": self.view_pan_left_action,
            "view_pan_right": self.view_pan_right_action,
            "help_guide": self.help_guide_action,
            "help_keys": self.help_keys_action,
            "help_web_guide": self.web_guide_action,
            "help_release_notes": self.release_notes_action,
            "help_report_bug": self.report_bug_action,
            "help_github": self.github_action,
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
            "filter_show_buildings": self.show_buildings_action,
            "filter_show_eye_candy": self.show_eye_candy_action,
            "filter_show_obstacles": self.show_obstacles_action,
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
            "unit_wall_pick": self.wall_pick_action,
            **self.trigger_move_actions,
            "trigger_variables": self.trigger_panel.variables_action,
            **self.trigger_panel.variable_actions,
            **self.elevation_actions,
        }
        for tool in settings.TOOLS:
            self._keybind_actions[f"tool_{tool.tool_id}"] = getattr(self, f"{tool.tool_id}_action")
        for pid, action in self.player_select_actions.items():
            self._keybind_actions[f"player_select_{pid}"] = action
        for pid, action in self.player_actions.items():
            self._keybind_actions[f"filter_player_{pid}"] = action
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
        # A mode-scoped action holds its key only in its own modes, so a key it shares with
        # another mode's action never sits on two live shortcuts (GH #267, _apply_mode_shortcuts).
        live = key_text if settings.keybind_live_in(action_id, self.mode) else ""
        action.setShortcut(QKeySequence(live) if live else QKeySequence())
        tool_id = action_id.removeprefix("tool_")
        if action_id.startswith("tool_") and tool_id in _TOOL_LABELS:
            # An icon-only button shows no name, so the tooltip leads with it and the key.
            base = self._tool_base_tips.setdefault(tool_id, action.toolTip())
            key = QKeySequence(key_text).toString(QKeySequence.NativeText) if key_text else ""
            label = _TOOL_LABELS[tool_id]
            action.setToolTip(f"{label} ({key}): {base}" if key else f"{label}: {base}")

    def _apply_mode_shortcuts(self) -> None:
        """Re-applies every mode-scoped keybind for the current self.mode, so each
        holds its key only where it is live. Run on every mode change."""
        for action_id in self._keybind_actions:
            if settings.keybind_modes(action_id):
                self.apply_keybind(action_id)

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
                who = "GAIA" if player_id == GAIA_PLAYER_ID else f"Player {player_id}"
                self._log_status(f"Players: switched to {who}")
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

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            f"DEscape {__version__} by Combinebobnt\n\n"
            "External map editor for Age of Empires 2: Definitive Edition scenarios.\n\n"
            "Copyright (C) 2026 Combinebobnt\n"
            "Licensed under the GNU General Public License v3.0 or later.\n\n"
            f"{PARSER_CREDIT}",
        )

    def _open_help_url(self, url: str) -> None:
        help_window.open_url(url, self._log_status)

    def _show_help(self, topic_id: str | None = None) -> None:
        """One shared, non-modal guide window; with no topic, the current mode's."""
        if self._help_window is None:
            self._help_window = help_window.HelpWindow(self, self._log_status)
        self._help_window.select_topic(topic_id or help_guide.topic_for_mode(self.mode).topic_id)
        self._help_window.show()
        self._help_window.raise_()
        self._help_window.activateWindow()

    def _show_settings(self) -> None:
        self._pick_up_mod_changes()  # TASK-438: its mod lists start current
        dialog = SettingsDialog(self)
        dialog.exec_()

    def _show_debug_log(self) -> None:
        # Before the dialog snapshots the log: a just-finished op isn't written until idle.
        perf_trace.flush_pending_op()
        dialog = DebugLogDialog(self)
        dialog.exec_()

    def _show_mirror_dialog(self) -> None:
        if self.scenario is None:
            return
        dialog = MirrorDialog(self, last=self._mirror_defaults)
        if dialog.exec_() == QDialog.Accepted:
            self._mirror_defaults = dialog.state()
