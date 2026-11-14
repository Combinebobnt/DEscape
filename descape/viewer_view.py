"""ViewerWindow's view options: unit filters, sprites, layers and mod art, the colour mood, grid,
footprint outlines, badges, range rings, cameras and the overlay toggles."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QToolButton,
)

from descape import (
    asset_source,
    colour_mood,
    mod_graphics,
    option_fields,
    settings,
    unit_sprites,
    view_layers,
)
from descape.options_model import (
    COLOUR_MOOD_FIELD_ID,
)
from descape.unit_filter import MAX_PLAYER_ID, UnitFilter
from descape.viewer_common import (
    KeepOpenMenu,
)

MOD_LIST_PICKED_UP = "Picked up a change in the game's mod list"


class _ViewerViewMixin:
    """ViewerWindow's view toggles and filters (viewer.py's docstring has the split's rules)."""

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

        menu = KeepOpenMenu(self.filters_button)
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

        self.show_buildings_action = QAction("Show Buildings", self, checkable=True, checked=True)
        self.show_buildings_action.setToolTip(
            "Every building except walls and gates -- up to 1,113 in one example file. "
            "Trebuchets and other packable siege count as units and stay visible"
        )
        self.show_buildings_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_buildings_action)

        self.show_eye_candy_action = QAction("Show Eye Candy", self, checkable=True, checked=True)
        self.show_eye_candy_action.setToolTip(
            "Grass, flowers, stumps and barrels -- up to 2,445 in one example file. "
            "Gold, stone and berry bushes are resources, not eye candy, and stay visible"
        )
        self.show_eye_candy_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_eye_candy_action)

        self.show_obstacles_action = QAction("Show Obstacles", self, checkable=True, checked=True)
        self.show_obstacles_action.setToolTip(
            "Rocks, ruins, statues and mountains that units cannot walk through -- "
            "up to 521 in one example file"
        )
        self.show_obstacles_action.toggled.connect(self._on_filter_changed)
        menu.addAction(self.show_obstacles_action)

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
            action = QAction(self._player_labels.text[player_id], self, checkable=True, checked=True)
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
            "Check every entry above -- GAIA, Trees, Walls, Buildings, Eye Candy, Obstacles, Invisible Objects, "
            "Garrisoned Units, and all players"
        )
        self.filter_show_all_action.triggered.connect(lambda: self._set_all_filters(True))
        menu.addAction(self.filter_show_all_action)
        self.filter_hide_all_action = QAction("Hide All", self)
        self.filter_hide_all_action.setToolTip(
            "Uncheck every entry above -- GAIA, Trees, Walls, Buildings, Eye Candy, Obstacles, Invisible Objects, "
            "Garrisoned Units, and all players"
        )
        self.filter_hide_all_action.triggered.connect(lambda: self._set_all_filters(False))
        menu.addAction(self.filter_hide_all_action)
        for bulk_action in (
            self.filter_all_players_action,
            self.filter_no_players_action,
            self.filter_show_all_action,
            self.filter_hide_all_action,
        ):
            menu.keep_open(bulk_action)

        self.filters_button.setMenu(menu)
        toolbar.addWidget(self.filters_button)
        # Show Garrisoned ships unchecked, so the app's default filter is no
        # longer UnitFilter(): re-read the menu now, or _unit_filter claims
        # occupants are shown until the first toggle and a freshly loaded
        # file draws them stacked on their hosts.
        self._unit_filter = self._current_unit_filter()

    def _refresh_selection_after_filter(self) -> None:
        """A filter toggle can hide currently selected units. Rebuild the
        index, then drop every selection key that no longer resolves --
        leaving a highlight around a unit that no longer paints would be a
        cue pointing at nothing.

        The index rebuild also covers the footprint overlay, which is index-
        driven and live outside Units mode. Outside Units the selection's
        keys and highlight reconcile without the inspector (TASK-359). With
        nothing needing the index, it is dropped rather than left built under
        the old filter (it is current or None)."""
        if not self._needs_unit_index():
            self._stack_cycle = None
            self.map_view.set_unit_index(None)
            return
        self._rebuild_unit_index()
        if self.mode != "units":
            self._reconcile_selection()
            self._drop_unit_index_if_unneeded()
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
        """Same as _set_all_players, but also the eight kind toggles -- the
        menu's "Show All"/"Hide All" shortcuts at the bottom."""
        for action in (
            self.show_gaia_action,
            self.show_trees_action,
            self.show_walls_action,
            self.show_buildings_action,
            self.show_eye_candy_action,
            self.show_obstacles_action,
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
            show_buildings=self.show_buildings_action.isChecked(),
            show_eye_candy=self.show_eye_candy_action.isChecked(),
            show_obstacles=self.show_obstacles_action.isChecked(),
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

        _cancel_warms() comes first, for the reason _on_layer_toggled()
        gives: a warm in flight was built against the old filter, and a
        margin-warm tick would rebuild a whole stale level inside itself.
        """
        self._unit_filter = self._current_unit_filter()
        self._cancel_warms()
        if self.scenario is None or self._cache is None:
            return
        self._cache.set_unit_filter(self._unit_filter)
        canvas_w, canvas_h = self._cache.canvas_dims(0)
        self.map_view.invalidate_region((0, 0, canvas_w, canvas_h))
        self._refresh_selection_after_filter()
        self._rearm_pending_warm()
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
        # Before the processEvents() below: the placeholder watchdog re-arms on _busy in that turn.
        was_busy = self._busy
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            # Again: a poll fired in that turn re-arms on the pre-toggle cache and uses up the first fire.
            self._cancel_warms()
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
            # Farms drape only with sprites on, so their outlines change shape here.
            self.map_view.refresh_drape_outlines()
            # GH #167: the trigger ghosts follow the toggle; set_source() is not on this path.
            self.map_view.refresh_trigger_ghosts()
            elapsed = time.perf_counter() - t0
        finally:
            self._busy = was_busy
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
        self._rearm_pending_warm()
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
        settings.set_view_layer(layer_id, on)
        spec = next(s for s in view_layers.LAYERS if s.layer_id == layer_id)
        label = spec.label.replace("&", "")
        if layer_id == "mod_art":
            self._apply_mod_art_change(f"{label}: {'on' if on else 'off'}")
            return
        if layer_id == "colour_mood":
            # Draw-only: no wait cursor and no eviction, and the cache's copy keeps up.
            if self._cache is not None:
                self._cache.set_layers(self._layers)
            self._refresh_colour_mood_preview()
            self._log_status(f"{label}: {'on' if on else 'off'}")
            return
        if spec.scene_overlay:
            self._apply_scene_overlay_layers(f"{label}: {'on' if on else 'off'}")
            return
        self._apply_layers(f"{label}: {'on' if on else 'off'}", drape_outlines=layer_id == "farm_overlay")
        if layer_id in ("terrain_textures", "terrain_blend"):
            # Terrain Blending greys with Terrain Textures; the Target tooltip names both.
            self._update_tool_enabled()

    def _colour_mood_value(self) -> str | None:
        """The effective stored mood the preview follows: the pending edit,
        else the file's own. None with the layer off, no document or no field."""
        if self.scenario is None or not self._layers.colour_mood:
            return None
        spec = next((s for s in option_fields.specs_for(self.scenario) if s.field_id == COLOUR_MOOD_FIELD_ID), None)
        if spec is None:
            return None
        pending = {} if self.option_edits is None else self.option_edits.pending_values()
        return pending.get(COLOUR_MOOD_FIELD_ID, option_fields.current_value(self.scenario, spec))

    def _refresh_colour_mood_preview(self) -> None:
        value = self._colour_mood_value()
        self.map_view.set_colour_grade(None if value is None else colour_mood.grade_for(value))

    def _apply_scene_overlay_layers(self, status: str) -> None:
        """Pushes self._layers' scene-overlay rows (Fish Outlines, TASK-499) to
        MapView. Nothing reaches the chunk cache, whose render_key() ignores
        them, so there is no wait cursor and no eviction. A non-None index is
        current (_rebuild_unit_index()'s contract), so only a missing one is built."""
        if self._layers.fish_outlines and self.map_view._unit_index is None:
            self._rebuild_unit_index()
        self.map_view.set_fish_outlines(self._layers.fish_outlines)
        self._log_status(status)

    def set_wall_opacity_pct(self, pct: int) -> None:
        """Settings > Appearance's wall opacity (GH #35), already persisted by
        the caller. With Translucent Walls off the drawn wall_alpha is
        unchanged, so only the window's copy updates: set_layers() would miss
        its equality guard on the raw pct and evict the whole canvas for no
        visible change."""
        old_alpha = self._layers.wall_alpha
        self._layers = dataclasses.replace(self._layers, wall_opacity_pct=pct)
        if self._layers.wall_alpha != old_alpha:
            self._apply_layers(f"Wall opacity: {pct}%")

    def _apply_layers(self, status: str, drape_outlines: bool = False) -> None:
        """Pushes self._layers into the live cache and repaints, logging
        `status` with the time it took. See _on_layer_toggled() for the order.
        drape_outlines re-derives the draped farms' per-tile outlines (Farm Overlay)."""
        self._cancel_warms()
        if self.scenario is None or self._cache is None:
            self._update_tool_enabled()
            return
        # Before the processEvents() below: the placeholder watchdog re-arms on _busy in that turn.
        was_busy = self._busy
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            QApplication.processEvents()
            # Again: a poll fired in that turn re-arms on the pre-toggle cache and uses up the first fire.
            self._cancel_warms()
            t0 = time.perf_counter()
            self._cache.set_layers(self._layers)
            self.map_view.invalidate_region((0, 0, *self._cache.canvas_dims(0)))
            if drape_outlines:
                # A draped farm outlines per tile in Stepped and Sloped (TASK-015).
                self.map_view.refresh_drape_outlines()
            # GH #167: tree scale, hero glow and wall alpha reach the trigger ghosts' sprites too.
            self.map_view.refresh_trigger_ghosts()
            elapsed = time.perf_counter() - t0
        finally:
            self._busy = was_busy
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
        self._rearm_pending_warm()
        self._log_status(f"{status} (applied in {elapsed:.2f}s)")

    def _pick_up_mod_changes(self, *, loading: bool = False) -> bool:
        """A mod enabled, disabled or reordered in the game's mod list since the
        scan (TASK-438), checked on a scenario open and on opening Settings:
        redraws when what draws changed and drops the AI scan. True when it found
        one. While loading, the load's own render draws it and its own Players show lists it."""
        if self._busy and not loading:
            return False
        changed = mod_graphics.refresh()
        if not changed:
            return False
        if mod_graphics.ART in changed:
            self._apply_mod_art_change(MOD_LIST_PICKED_UP, picked_up=True)
        else:
            self._log_status(MOD_LIST_PICKED_UP)  # a mod without art: nothing to redraw
        if loading:
            self._drop_ai_scan()
        else:
            self.invalidate_ai_choices()
        return True

    def _apply_mod_art_change(self, status: str, *, rescan: bool = False, picked_up: bool = False) -> bool:
        """The one path for a Mod Art toggle, a Settings opt-out or opt-in and a
        profile folder change (TASK-378): mod_graphics takes the window's state, every
        art cache drops, and the map, swatches and thumbnails redraw from the
        files that now win. rescan also forgets the scanned mod folders; without
        it, a change in the game's mod list is picked up first (TASK-438), and
        the result says whether there was one. picked_up: _pick_up_mod_changes() already did."""
        if rescan:
            mod_graphics.clear()
        elif not picked_up and mod_graphics.refresh():
            picked_up = True
            status = f"{MOD_LIST_PICKED_UP}. {status}"
            self.invalidate_ai_choices()  # a mod enabled in the game offers its AIs
        toggled = mod_graphics.set_enabled(self._layers.mod_art)
        mod_graphics.set_excluded(settings.get_mod_art_excluded())
        mod_graphics.set_included(settings.get_mod_art_included())
        if not (toggled or mod_graphics.enabled()):
            # Off before and after (an opt-out, opt-in or profile change): nothing drawn changes.
            self._update_tool_enabled()
            self._log_status(status)
            return picked_up
        self._cancel_warms()
        asset_source.clear_art_caches()
        # Starts the background walk of every active mod file (GH #272) before the redraw below.
        unit_sprites.sprite_reach()
        self.refresh_map()
        self.terrain_panel.view.refresh_swatches()
        self.units_panel.refresh_thumbnails()
        self.trigger_panel.refresh_portraits()
        if self.scenario is None:
            self._update_tool_enabled()  # refresh_map() re-gates when a map is open
        pending = " (measuring its reach in the background)" if unit_sprites.reach_pending() else ""
        self._log_status(f"{status}. {self._mod_art_status() or 'Mod art: none drawn'}{pending}")
        return picked_up

    def _mod_art_status(self) -> str:
        """"Mod art: N mods" while any mod's art can draw, else "", so a
        mod-derived render never goes unsaid in the log."""
        if not mod_graphics.enabled():
            return ""
        count = sum(mod.drawn for mod in mod_graphics.mods_with_art())
        return f"Mod art: {count} mod{'s' if count != 1 else ''}" if count else ""

    def _has_mod_art(self) -> bool:
        """Whether any scanned mod ships art, cached per mod_graphics.generation()."""
        generation = mod_graphics.generation()
        if self._mod_art_available is None or self._mod_art_available[0] != generation:
            self._mod_art_available = (generation, bool(mod_graphics.mods_with_art()))
        return self._mod_art_available[1]

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
        if not self._unit_filter.show_buildings:
            parts.append("buildings hidden")
        if not self._unit_filter.show_eye_candy:
            parts.append("eye candy hidden")
        if not self._unit_filter.show_obstacles:
            parts.append("obstacles hidden")
        if not self._unit_filter.show_invisible:
            parts.append("invisible objects hidden")
        if not self._unit_filter.show_garrisoned:
            parts.append("garrisoned units hidden")
        if self._unit_filter.players is not None:
            shown = sorted(self._unit_filter.players)
            parts.append(f"players {shown}" if shown else "no players")
        return ", ".join(parts)

    def _on_distance_ticks_toggled(self, checked: bool) -> None:
        settings.set_distance_ticks(checked)
        self.map_view.set_edge_ticks(checked)

    def _on_grid_overlay_toggled(self, checked: bool) -> None:
        settings.set_grid_overlay(checked)
        self._apply_grid_change(lambda: self.map_view.set_grid_overlay(checked))

    def _on_grid_follow_toggled(self, checked: bool) -> None:
        settings.set_grid_follow_elevation(checked)
        self._apply_grid_change(lambda: self.map_view.set_grid_follow_elevation(checked))

    def _on_grid_major_every(self, tiles: int) -> None:
        settings.set_grid_major_every(tiles)
        self._apply_grid_change(lambda: self.map_view.set_grid_major_every(tiles))

    def _apply_grid_change(self, change: Callable[[], bool]) -> None:
        """Every View > Grid change that can re-bake the grid into the chunk
        cache: the toggle, Follow Terrain Elevation, the stronger-line spacing, and both ends of a
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

    def _on_footprint_merged_toggled(self, checked: bool) -> None:
        settings.set_footprint_merged(checked)
        self.map_view.set_footprint_merged(checked)

    def _on_footprint_by_owner_toggled(self, checked: bool) -> None:
        settings.set_footprint_by_owner(checked)
        self.map_view.set_footprint_by_owner(checked)

    def _on_stack_badges_toggled(self, checked: bool) -> None:
        settings.set_stack_badges(checked)
        self.map_view.set_stack_badges(checked)

    def _on_selection_owner_colour_toggled(self, checked: bool) -> None:
        settings.set_selection_by_owner(checked)
        self.map_view.set_selection_by_owner(checked)

    def _on_range_rings_toggled(self, checked: bool) -> None:
        settings.set_range_rings(checked)
        self.map_view.set_range_rings(checked)

    def _on_range_rings_by_owner_toggled(self, checked: bool) -> None:
        settings.set_range_rings_by_owner(checked)
        self.map_view.set_range_rings_by_owner(checked)

    def _on_range_rings_triggers_toggled(self, checked: bool) -> None:
        settings.set_range_rings_triggers(checked)
        self.map_view.set_range_rings_triggers(checked)

    def _on_range_rings_selected_revealers_toggled(self, checked: bool) -> None:
        settings.set_range_rings_selected_revealers(checked)
        self.map_view.set_range_rings_selected_revealers(checked)

    def _on_range_rings_buildings_toggled(self, checked: bool) -> None:
        settings.set_range_rings_buildings(checked)
        if checked and self._needs_unit_index():
            self._rebuild_unit_index()
        self.map_view.set_ring_layer_buildings(checked)

    def _on_range_rings_revealers_toggled(self, checked: bool) -> None:
        settings.set_range_rings_revealers(checked)
        if checked and self._needs_unit_index():
            self._rebuild_unit_index()
        self.map_view.set_ring_layer_revealers(checked)

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
        ref = self._trigger_overlay_ref
        if checked and ref is not None and len(ref[0]) > 1:
            # A union is not derived while the overlay is off (_refresh_trigger_overlay()).
            self._refresh_trigger_overlay()
        self._push_area_edit_target()

    def _on_trigger_ghosts_toggled(self, checked: bool) -> None:
        settings.set_trigger_ghosts(checked)
        self.map_view.set_trigger_ghosts_enabled(checked)
        if checked:
            # A cap said before the toggle speaks again: a kept key would stay silent.
            self._trigger_ghost_cap_logged = None
            if self._trigger_overlay_ref is not None:
                # A union is not derived while both are off, and a cap's line comes only from a derive.
                self._refresh_trigger_overlay()

    def _on_instruction_overlay_toggled(self, checked: bool) -> None:
        settings.set_instruction_overlay(checked)
        self._push_instruction_overlay()

    def _on_instruction_view(self, view) -> None:
        self._instruction_view = view
        self._push_instruction_overlay()

    def _push_instruction_overlay(self) -> None:
        """TASK-550: the one funnel to the map's instruction box. Every site that
        changes the mode or the document calls it, so a stale box cannot survive one."""
        shown = self.scenario is not None and self.mode == "triggers" and self.instruction_overlay_action.isChecked()
        self.map_view.show_instruction(self._instruction_view if shown else None)

    def _on_distance_tick_interval(self, tiles: int) -> None:
        settings.set_distance_tick_interval(tiles)
        self.map_view.set_edge_tick_interval(tiles)
