"""ViewerWindow's tool state: enabled state, brush sync, the terrain target, tool options and defaults,
tool and view selection, terrain style."""

from __future__ import annotations

from PyQt5.QtCore import QSignalBlocker

from descape import (
    asset_source,
    brush,
    debug_log,
    perf_trace,
    settings,
    terrain_target,
    tool_options,
    view_layers,
)
from descape.beach_edges import ring_mode
from descape.terrain_palette import name_for_terrain_id
from descape.terrain_style import label_for, style_for_label
from descape.viewer_common import (
    _SCATTER_TIP,
    _TOOL_PARAM,
    _TOOL_SHAPE,
    _WALL_TOOLS,
    GATE_WALLS_TOOLS,
    OFF_GRID_TOOLS,
    SKIP_OCCUPIED_TOOLS,
    UNRESTRICTED_TOOLS,
    brush_applicable,
    tool_applicable,
)

# GH #88: the tools that lay an auto-beach ring (Draw's stroke, the two shape commits).
_AUTO_BEACH_TOOLS = ("draw", "draw_line", "draw_rect")


_SCATTER_NO_AREA_TIP = "Select an area first: drag one with Select in Terrain mode, or select units in Units mode"


class _ViewerToolStateMixin:
    """ViewerWindow's tool state and tool options (viewer.py's docstring has the split's rules)."""

    def _update_rotate_enabled(self) -> None:
        """Rotate's gate: Units mode plus a target, i.e. a selection or (GH
        #172, GH #355) Place Unit or the Unit Brush armed
        (_place_rotate_armed()), whose ghost it turns. The Edit menu's four
        (and Flip) also open for an armed pasted region (GH #197).

        Re-evaluated on tool change and catalog pick, never per hover: with
        the cursor off the map an armed Rotate logs "nothing to rotate"
        rather than greying, so the buttons don't flicker as the ghost comes
        and goes. Not narrowed to rotatable consts either, for the same reason:
        a selection's per-unit skips are reported in the status line.

        Flip and Rotate Group (TASK-256) open for any _transform_target() in
        a document that can be edited (GH #67), the Place preview included:
        there they log that nothing changed."""
        actions = getattr(self, "_rotate_toolbar_actions", None)
        if actions is None:
            return  # a catalog signal during construction, before the toolbar exists
        unit_editable = self.scenario is not None and self.mode == "units"
        enabled = unit_editable and (bool(self._selection) or self._place_rotate_armed())
        # GH #197 and TASK-256: the menu's four also turn a pasted or a committed region;
        # the toolbar pair stays units-only.
        committed = self._region_rotate_target() and self._region_write_refusal("") is None
        region = self._region_turn_target() or committed
        for action in self._rotate_actions:
            action.setEnabled(enabled or region)
        for action in actions:
            action.setEnabled(enabled)
        unlocked = self.scenario is not None and self.game_data_lock_reason() is None
        target = self._transform_target() if unlocked else None
        if target == "region" and self._region_write_refusal("") is not None:
            target = None
        for action in (*self._flip_actions, *self._rotate_group_actions):
            action.setEnabled(target is not None)

    def _paste_armed(self) -> bool:
        """A pure read: a paste is armed and still the next thing undo takes."""
        state = self._paste_move
        return state is not None and self.edit_history.peek_undo() is state.record

    def _region_turn_target(self) -> bool:
        """Whether Rotate turns the armed pasted region (GH #197): one is armed,
        outside Triggers mode, and Units mode has no selection or Place target,
        which keep today's Rotate. The wheel's rotate binding asks the same
        (GH #245) and does nothing where this says no. A pure read with no side
        effect: it never runs the validity sweep, since _set_paste_move()
        re-gates through here."""
        if not self._paste_armed() or self.mode == "triggers":
            return False
        return not (self.mode == "units" and (self._selection or self._place_rotate_armed()))

    def _update_zoom_native_action(self) -> None:
        """Greys View > Zoom to Native Sprite Size where the live ladder has no
        native level (TASK-599), with a tooltip saying why."""
        available = self.map_view.native_zoom_factor() is not None
        self.zoom_native_action.setEnabled(available)
        quality = settings.get_graphics_quality()
        if available:
            tip = "Zoom so one tile spans 96 screen pixels: unit sprites drawn at their native size, pixel for pixel"
        elif self.scenario is None:
            tip = "Open a map first"
        elif quality < settings.GRAPHICS_QUALITY_DEFAULT:
            label = settings.GRAPHICS_QUALITY_LABELS[quality]
            tip = (
                f"Needs Graphics quality Default or Enhanced (Settings > Appearance); it is {label}, "
                "which never draws sprites at native size"
            )
        else:
            tip = "Not in top-down Flat: turn on Isometric View, or use Elevation View Stepped or Sloped"
        self.zoom_native_action.setToolTip(tip)

    def _zoom_to_native(self) -> None:
        self._update_zoom_native_action()
        if not self.map_view.zoom_to_native():
            self.statusBar().showMessage(self.zoom_native_action.toolTip(), 5000)

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
        # GH #67: a document shown in the other game's data is read-only.
        unlocked = self.game_data_lock_reason() is None
        write_ok = has_map and self.scenario.terrain_write_supported and unlocked
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
        unit_editable = has_map and self.mode == "units" and unlocked
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
        self._regate_swap()
        # Read-only, so any loaded map; each check reports what it can't run.
        self.analysis_action.setEnabled(has_map)
        self.find_action.setEnabled(has_map)
        # Same gate as Draw/Paint Can (has_map and write_ok, no squareness
        # requirement): a cliff placement never touches the elevation
        # recursion elevation_ok exists for.
        self.cliff_action.setEnabled(has_map and write_ok)
        # Same gate as Draw/Paint Can/Cliff: a pick never calls
        # set_elevation() either, so squareness is irrelevant to it too.
        # In Units mode it writes UI state only, so no terrain gate applies.
        self.eyedropper_action.setEnabled(has_map and (write_ok or self.mode == "units"))
        # Same gate as Eyedropper in Terrain mode: Select only reads (Copy)
        # and its own write (Paste) is gated separately below, so squareness
        # is irrelevant to the tool itself, only to whether the Elevation
        # paste checkbox may be checked, gated further down. In Units mode it
        # picks units, which no terrain gate applies to.
        self.select_action.setEnabled(has_map and (write_ok or self.mode == "units"))
        # Place Unit's gate: whether the units are writable is found lazily.
        self.move_action.setEnabled(unit_editable)
        self.place_unit_action.setEnabled(unit_editable)
        self.unit_line_action.setEnabled(unit_editable)
        self.unit_shape_action.setEnabled(unit_editable)
        # GH #125's wall keybind: Place Unit's own gate. Its button is gated with the params below.
        self.wall_pick_action.setEnabled(unit_editable)
        self.convert_action.setEnabled(unit_editable)
        self.unit_brush_action.setEnabled(unit_editable)
        # GH #184: has_map only, in both its modes; ToolDef.modes hides it elsewhere.
        self.eraser_action.setEnabled(has_map and unlocked)
        # Deliberately coarse: this runs on mode changes, not entry-tree
        # clicks, so the template itself is validated per click instead.
        trigger_editable = has_map and self.mode == "triggers" and self.trigger_panel._editable and unlocked
        self.create_objects_action.setEnabled(trigger_editable)
        self._area_edit_editable = trigger_editable
        # Cycle Variant needs a selection as well as Units mode -- it acts on
        # what is already selected rather than on a click, so with nothing
        # selected there is no target and the button should say so by being
        # grey rather than by logging a refusal. Rotate's own gate is in
        # _update_rotate_enabled().
        for action in self._variant_actions:
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
        self._update_rotate_enabled()
        self.close_action.setEnabled(has_map)
        self.save_action.setEnabled(write_ok)
        self.save_as_action.setEnabled(write_ok)
        self.resize_action.setEnabled(elevation_ok)  # square-only, see scenario_resize
        self.terrain_style_combo.setEnabled(has_map)
        for action in self.elevation_actions.values():
            action.setEnabled(self.terrain_style_combo.isEnabled())

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
        self._update_zoom_native_action()

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
                spec, style=self._render_style, sprites_enabled=self._sprites_enabled, has_map=has_map,
                terrain_textures=self._layers.terrain_textures,
                # Only asked with a map open, so the first mod-folder scan never lands on startup.
                has_mod_art=has_map and self._has_mod_art(),
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
        # Terrain mode only: in Units mode it picks a unit and shows no params.
        if self._current_tool == "eyedropper" and self.mode == "terrain":
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
        unit_brush_param_ok = param == "unit_brush" and self.unit_brush_action.isEnabled()
        unit_shape_param_ok = param == "unit_shape" and current_action is not None and current_action.isEnabled()
        eraser_param_ok = param == "eraser" and self.eraser_action.isEnabled()
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
        # Terrain mode); it still gates the Target combo, and Trees, Eye candy and auto beach below.
        # The Target combo needs a file that stores a layer; without one it resets to Terrain.
        layer_ok = self._layer_write_ok()
        if not layer_ok and self.terrain_target_combo.currentIndex() != 0:
            with QSignalBlocker(self.terrain_target_combo):
                self.terrain_target_combo.setCurrentIndex(0)
        target_param_ok = terrain_param_ok and layer_ok
        self.terrain_target_label_action.setVisible(target_param_ok)
        self.terrain_target_param_action.setVisible(target_param_ok)
        self.terrain_target_combo.setEnabled(target_param_ok)
        # Only while the combo shows: the note can read the install's blend masks.
        visibility_note = self._layer_visibility_note() if target_param_ok else ""
        self.terrain_target_combo.setToolTip(
            "Which terrain the tool writes. Terrain: the tile's base terrain, which also clears "
            "its layered terrain. Layer: the layered terrain drawn over the base. Clear layer: "
            "remove the layered terrain"
            + (f". Right now layered terrain is {visibility_note}" if visibility_note else "")
        )
        # Auto beach, Trees and Eye candy read a base-terrain edit only, so they hide off Terrain.
        base_param_ok = terrain_param_ok and self._terrain_target() == terrain_target.TERRAIN
        # Auto beach. The tool term is load-bearing: Paint Can shares
        # param_widget="terrain" but lays no ring, so the box must not show for it.
        auto_beach_ok = (
            base_param_ok
            and self._current_tool in _AUTO_BEACH_TOOLS
            and ring_mode(self.terrain_panel.terrain_id()) is not None
        )
        self.auto_beach_param_action.setVisible(base_param_ok and self._current_tool in _AUTO_BEACH_TOOLS)
        self.auto_beach_check.setEnabled(auto_beach_ok)
        self.auto_beach_check.setToolTip(
            "Paint a beach shoreline between water and the terrain this stroke, line or rectangle lays down"
            if auto_beach_ok
            else "A beach, bridge, non-navigable shore or obsolete terrain lays no shoreline"
        )
        # The combo and width follow the checkbox, so a hidden setting can
        # never sit live behind the scenes.
        beach_param_ok = auto_beach_ok and self.auto_beach_check.isChecked()
        self.beach_param_action.setVisible(beach_param_ok)
        self.beach_combo.setEnabled(beach_param_ok)
        self.beach_width_param_action.setVisible(beach_param_ok)
        self.beach_width_spin.setEnabled(beach_param_ok)
        self.paint_trees_param_action.setVisible(base_param_ok)
        self.paint_trees_check.setEnabled(base_param_ok)
        self.paint_eye_candy_param_action.setVisible(base_param_ok)
        self.paint_eye_candy_check.setEnabled(base_param_ok)
        self.level_param_label_action.setVisible(level_param_ok)
        self.level_param_spin_action.setVisible(level_param_ok)
        self.elevation_level_spin.setEnabled(level_param_ok)
        self.convert_sources_param_action.setVisible(convert_param_ok)
        self.convert_sources_button.setEnabled(convert_param_ok)
        self.unit_brush_density_label_action.setVisible(unit_brush_param_ok)
        self.unit_brush_density_param_action.setVisible(unit_brush_param_ok)
        self.unit_brush_density_spin.setEnabled(unit_brush_param_ok)
        self.unit_brush_random_param_action.setVisible(unit_brush_param_ok)
        self.unit_brush_random_check.setEnabled(unit_brush_param_ok)
        # Shape and Filled under Unit Shape alone; Spacing and Random under both.
        unit_shape_box_ok = unit_shape_param_ok and self._current_tool == "unit_shape"
        self.unit_shape_kind_param_action.setVisible(unit_shape_box_ok)
        self.unit_shape_kind_combo.setEnabled(unit_shape_box_ok)
        self.unit_shape_filled_param_action.setVisible(unit_shape_box_ok)
        self.unit_shape_filled_check.setEnabled(unit_shape_box_ok)
        self.unit_shape_spacing_label_action.setVisible(unit_shape_param_ok)
        self.unit_shape_spacing_param_action.setVisible(unit_shape_param_ok)
        self.unit_shape_spacing_spin.setEnabled(unit_shape_param_ok)
        self.unit_shape_random_param_action.setVisible(unit_shape_param_ok)
        self.unit_shape_random_check.setEnabled(unit_shape_param_ok)
        self.eraser_targets_param_action.setVisible(eraser_param_ok)
        self.eraser_targets_button.setEnabled(eraser_param_ok)
        # A v1.21 file has no `layer` field to clear; the stroke ignores the tick too.
        self.eraser_target_actions["overlay"].setEnabled(self._erase_overlay_ok())
        self.rect_fill_param_action.setVisible(rect_param_ok)
        self.rect_fill_combo.setEnabled(rect_param_ok)
        self.brush_param_label_action.setVisible(brush_ok)
        self.brush_size_spin_action.setVisible(brush_ok)
        self.brush_size_spin.setEnabled(brush_ok)
        self.brush_shape_combo_action.setVisible(brush_ok)
        self.brush_shape_combo.setEnabled(brush_ok)
        thickness_ok = brush_ok and self.brush_shape_combo.currentData() in brush.OUTLINE_SHAPES
        self.brush_thickness_label_action.setVisible(thickness_ok)
        self.brush_thickness_spin_action.setVisible(thickness_ok)
        self.brush_thickness_spin.setEnabled(thickness_ok)
        self._refresh_brush_shape_labels()
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
        self._update_clipboard_actions()
        if self.mode == "triggers":
            self.select_all_action.setEnabled(self.trigger_panel.tree.topLevelItemCount() > 0)
            self.deselect_action.setEnabled(bool(self.trigger_panel.selected_trigger_indices()))
            self.select_all_action.setToolTip("Select every trigger")
            self.deselect_action.setToolTip("Clear the trigger selection")
        elif self._units_select_active():
            self.select_all_action.setToolTip("Select every unit the filters show")
            self.deselect_action.setToolTip(
                "Clear the unit selection" if self._selection else "Clear the selected region"
            )
            self.select_all_action.setEnabled(has_map)
            self.deselect_action.setEnabled(has_map and (bool(self._selection) or self._region is not None))
        else:
            self.select_all_action.setToolTip("Select the whole map as a region")
            self.deselect_action.setToolTip("Clear the selected region")
            self.select_all_action.setEnabled(has_map)
            self.deselect_action.setEnabled(has_map and self._region is not None)
        # A region enables it in any mode (Copy/Paste's reason above); so does a
        # unit selection where it is live, its bounding box the area (GH #196, TASK-359).
        scatter_ok = has_map and (self._region is not None or (self._selection_live() and bool(self._selection)))
        self.scatter_action.setEnabled(scatter_ok)
        self.scatter_action.setToolTip(_SCATTER_TIP if scatter_ok or not has_map else _SCATTER_NO_AREA_TIP)

        # The paste filter checkboxes: visible only while Select is active in
        # Terrain mode (in Units mode it picks units, not a region), same
        # convention as every other tool param above. Elevation is
        # additionally gated on elevation_ok (map_is_square) -- paste_region()
        # would otherwise hand set_tiles_elevation a non-square map and hit
        # MapManager.get_tile's ValueError mid-stroke. _close_stroke_on_error()
        # keeps that from wedging EditHistory, but the paste would still fail.
        select_param_ok = (
            self._current_tool == "select" and self.mode == "terrain" and self.select_action.isEnabled()
        )
        self.paste_terrain_param_action.setVisible(select_param_ok)
        self.paste_terrain_check.setEnabled(select_param_ok)
        self.paste_elevation_param_action.setVisible(select_param_ok)
        self.paste_elevation_check.setEnabled(select_param_ok and elevation_ok)
        self.paste_units_param_action.setVisible(select_param_ok)
        self.paste_units_check.setEnabled(select_param_ok)

        # D2's off-grid toggle -- same tool-param convention as every
        # group above, derived from OFF_GRID_TOOLS rather than naming
        # Place Unit here, so a second off-grid tool needs only a ToolDef
        # field. It goes in the separator roll-up below too: a new param group
        # that forgets that line leaves the separator hidden while its own
        # widget shows.
        # GH #176: both placement boxes show under Move too, where unit drags run.
        tool_action_ok = current_action is not None and current_action.isEnabled()
        unrestricted_ok = self._current_tool in UNRESTRICTED_TOOLS and tool_action_ok
        self.unrestricted_param_action.setVisible(unrestricted_ok)
        self.unrestricted_check.setEnabled(unrestricted_ok)
        off_grid_ok = self._current_tool in OFF_GRID_TOOLS and tool_action_ok
        self.off_grid_param_action.setVisible(off_grid_ok)
        self.off_grid_check.setEnabled(off_grid_ok)
        # GH #124's toggle, the same convention, and in the roll-up below too.
        skip_occupied_ok = (
            self._current_tool in SKIP_OCCUPIED_TOOLS
            and current_action is not None
            and current_action.isEnabled()
        )
        self.skip_occupied_param_action.setVisible(skip_occupied_ok)
        self.skip_occupied_check.setEnabled(skip_occupied_ok)
        # GH #194's toggle, under the tool that drags units (Move, Units mode only).
        gate_walls_ok = self._current_tool in GATE_WALLS_TOOLS and unit_editable
        self.gate_walls_param_action.setVisible(gate_walls_ok)
        self.gate_walls_check.setEnabled(gate_walls_ok)
        # TASK-287: in both modes the Eyedropper applies in.
        eyedropper_return_ok = self._current_tool == "eyedropper" and self.eyedropper_action.isEnabled()
        self.eyedropper_return_param_action.setVisible(eyedropper_return_ok)
        self.eyedropper_return_check.setEnabled(eyedropper_return_ok)
        # GH #190: the wall button only beside the tools that place walls, never Convert's row.
        wall_pick_ok = self._current_tool in _WALL_TOOLS and tool_action_ok
        self.wall_pick_param_action.setVisible(wall_pick_ok)
        self.wall_pick_button.setEnabled(wall_pick_ok)

        area_param_ok = self._area_options_shown()
        for action, _check in self._area_option_params:
            action.setVisible(area_param_ok)
        self._regate_area_options()

        ruler_param_ok = self._current_tool == "ruler" and self.ruler_action.isEnabled()
        self.clear_rulers_param_action.setVisible(ruler_param_ok)
        self.ruler_hint_param_action.setVisible(ruler_param_ok)
        self._on_pinned_rulers_changed(self.map_view.pinned_ruler_count())

        self.tool_param_separator_action.setVisible(
            terrain_param_ok or level_param_ok or convert_param_ok or unit_brush_param_ok or eraser_param_ok
            or unit_shape_param_ok or cliff_param_ok or brush_ok or select_param_ok or rect_param_ok
            or beach_param_ok or unrestricted_ok or off_grid_ok or ruler_param_ok or skip_occupied_ok
            or wall_pick_ok or gate_walls_ok or area_param_ok or eyedropper_return_ok
        )

        # Stage 3: mode may have just changed which tools are applicable,
        # and the active tool may have just changed which one the More
        # Tools button's own label should track -- both are this function's
        # business already, so recompute overflow membership here too.
        self._apply_toolbar_overflow()
        self._push_area_edit_target()

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
            self.map_view.set_brush(
                self.brush_size_spin.value(), self.brush_shape_combo.currentData(), self.brush_thickness_spin.value()
            )
        else:
            self.map_view.set_brush(brush.BRUSH_SIZE_MIN, brush.BRUSH_SHAPE_SQUARE, brush.BRUSH_THICKNESS_MIN)

    def _brush_footprint(self, x: int, y: int) -> list[tuple[int, int]]:
        """The toolbar brush's tiles at (x, y), clipped to the map: every brush tool's stroke reads this."""
        mm = self.scenario.map_manager
        return brush.brush_tiles(
            x, y, self.brush_size_spin.value(), self.brush_shape_combo.currentData(),
            mm.map_width, mm.map_height, self.brush_thickness_spin.value(),
        )

    def _refresh_brush_shape_labels(self) -> None:
        """The two line shapes are named by how they run on screen, which flips between the
        isometric views and top-down Flat (brush.shape_label())."""
        isometric = self._render_style != "flat"
        for index in range(self.brush_shape_combo.count()):
            shape = self.brush_shape_combo.itemData(index)
            if shape in (brush.BRUSH_SHAPE_LINE_X, brush.BRUSH_SHAPE_LINE_Y):
                label = brush.shape_label(shape, isometric)
                if self.brush_shape_combo.itemText(index) != label:
                    self.brush_shape_combo.setItemText(index, label)

    def _rect_filled(self) -> bool:
        """Draw Rectangle's Fill/Outline state. A method rather than a plain
        attribute so the combo stays the single source of truth -- nothing
        mirrors it into a field that could drift."""
        return bool(self.rect_fill_combo.currentData())

    def _layer_write_ok(self) -> bool:
        """Whether a layered terrain can be written: the file stores `layer` and its terrain is writable."""
        scenario = self.scenario
        return scenario is not None and scenario.terrain_has_layer and scenario.terrain_write_supported

    def _terrain_target(self) -> str:
        """The Target combo's state (a terrain_target constant). Always
        TERRAIN on a file with no layer field, where a layer would never save."""
        if not self._layer_write_ok():
            return terrain_target.TERRAIN
        return self.terrain_target_combo.currentData()

    def _layer_visibility_note(self) -> str:
        """Why a layered terrain does not draw right now, as a short clause
        ("not drawn while ..."), or "" when it does. A layer edit made while
        this is non-empty changes the file and no pixel."""
        from descape import render_paint

        if render_paint.blend_effective(self._layers):
            return ""
        if not self._layers.terrain_textures:
            return "not drawn while View > Layers > Terrain Textures is off"
        if not self._layers.terrain_blend:
            return "not drawn while View > Layers > Terrain Blending is off"
        return "not drawn without an AoE2:DE install configured"

    def _layer_edit_status(self, label: str, target: str, terrain_id: int, changed: int, total: int | None) -> str:
        """The status line for a Draw, line or rectangle edit off the Terrain
        target: what was done to how many tiles, plus the visibility note."""
        count = f"{changed} tiles" if total is None else f"{changed} of {total} tiles"
        if target == terrain_target.LAYER:
            line = f"{label}: {name_for_terrain_id(terrain_id)} layered on {count}"
        else:
            line = f"{label}: cleared layered terrain on {count}"
        note = self._layer_visibility_note()
        return f"{line} ({note})" if note else line

    def _on_terrain_target_changed(self) -> None:
        # Re-gates the row: Auto beach, Trees and Eye candy show for the Terrain target only.
        self._update_tool_enabled()
        line = f"Target: {self.terrain_target_combo.currentText()}"
        note = self._layer_visibility_note()
        if note and self._terrain_target() != terrain_target.TERRAIN:
            line += f" (layered terrain is {note})"
        self._log_status(line)

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

    def _on_brush_shape_changed(self) -> None:
        # Re-gates the row as well: Thickness shows for an outline shape only.
        self._update_tool_enabled()
        self._on_brush_changed()

    def _on_terrain_changed(self, terrain_id: int) -> None:
        """The sidebar picker's hookup, for a user pick and the Eyedropper's
        set_terrain() alike: re-gates the param row (auto beach needs a pick
        beach_edges.ring_mode() accepts) and logs the new terrain."""
        self._update_tool_enabled()
        self._log_status(f"Terrain type: {name_for_terrain_id(terrain_id)}")

    def _on_auto_beach_toggled(self, checked: bool) -> None:
        # Re-gates the combo and width, which are visible only while the box
        # is ticked.
        self._update_tool_enabled()

    # -- Tool Options defaults (Settings > Tools) --------------------------------
    #
    # The row holds the session value and Settings > Tools the default
    # (settings.get_tool_option_defaults()). Nothing on the row writes the store:
    # no persistence hook is ever connected to a row widget.

    def _apply_tool_options(self, values: dict, *, changed_only: bool) -> list[str]:
        """Puts `values` on the row in schema order and returns the names that did
        not apply. The Cliff trio is a chain: a family that did not apply skips piece
        and frame, and a piece that did not apply skips frame, so a frame never lands
        on the wrong piece. `changed_only` skips a value equal to its schema default,
        which the row already shows at construction."""
        skipped: list[str] = []
        chain_broken = False
        for spec in tool_options.OPTIONS:
            name = spec.name
            if name not in values:
                continue
            if chain_broken and name in ("cliff_piece", "cliff_frame"):
                skipped.append(name)
                continue
            if changed_only and tool_options.is_default(name, values[name]):
                continue
            if not self.tool_option_bindings[name].apply(values[name]):
                skipped.append(name)
                chain_broken = chain_broken or name in ("cliff_family", "cliff_piece")
        return skipped

    def _restore_tool_options(self) -> None:
        """Once, at construction. A default this session's game data can't show
        leaves its control at the schema default, and the stored value stays."""
        skipped = self._apply_tool_options(settings.get_tool_option_defaults(), changed_only=True)
        if skipped:
            debug_log.warning(f"Tool defaults not shown this session: {', '.join(skipped)}")

    def apply_tool_option_default(self, name: str, value) -> bool:
        """Settings > Tools' live apply: an edited default reaches its row control at once."""
        return self.tool_option_bindings[name].apply(value)

    def tool_option_values(self) -> dict[str, object]:
        """Every row control's session value, canonical, in schema order."""
        return {name: binding.read() for name, binding in self.tool_option_bindings.items()}

    def reset_tool_options_to(self, values: dict) -> list[str]:
        """Every value onto the row, defaults included; the names that did not apply."""
        return self._apply_tool_options(values, changed_only=False)

    def _on_tool_selected(self, tool: str) -> None:
        with perf_trace.op("tool-switch"):
            self._select_tool(tool)

    def _select_tool(self, tool: str) -> None:
        if tool != "pan":
            # Pick from map lives on Pan; any other tool takes the clicks back.
            self.disarm_unit_picker()
        # set_tool() first: it can synchronously close a dangling stroke via
        # _end_stroke() -> on_edit_stroke_end(), which labels the undo record
        # from self._current_tool -- that must still read the OUTGOING tool.
        self.map_view.set_tool(tool)
        if tool == "eyedropper" and self._current_tool != "eyedropper":
            # TASK-287: where Switch back after a pick returns.
            self._eyedropper_return_tool = self._current_tool
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

    def _select_elevation_view(self, style: str) -> None:
        if self.terrain_style_combo.isEnabled():
            self.terrain_style_combo.setCurrentText(label_for(style))

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
        settings.set_terrain_style(style)  # after the _busy refusal, so a refused switch never persists
        self._sync_rotation_dial_projection()
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
                elapsed, tile_px = self._render_current(reset_view=False, op_label="style-switch")
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
        self._sync_rotation_dial_projection()
        # Re-gate before the guarded render, not after: this action is one of
        # the two inputs to self._render_style, so flipping it moves Flat
        # between "really Flat" and "renders through IsoChunkCache" -- and
        # View > Layers' Small Trees and Hero Glow rows are gated on exactly
        # that. Placed above the early returns so the busy/no-map paths re-gate too; with no map
        # every row is greyed anyway.
        self._update_tool_enabled()
        if self.scenario is None or self._busy or self._terrain_style != "flat":
            return
        self._busy = True
        try:
            elapsed, tile_px = self._render_current(reset_view=False, op_label="style-switch")
            state = "on" if checked else "off"
            self._log_status(f"Isometric View {state} (tile_px={tile_px}) prepared in {elapsed:.2f}s")
        finally:
            self._busy = False

    def _sync_rotation_dial_projection(self) -> None:
        """TASK-023: the inspector's facing dial follows the projection the map renders."""
        self.units_panel.set_rotation_projection(self._render_style == "flat")
