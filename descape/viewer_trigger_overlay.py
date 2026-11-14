"""ViewerWindow's trigger side of the map: area and location edit, unit references, the tile and unit
pickers and trigger overlay drawing."""

from __future__ import annotations

from descape import (
    facet_preview,
    library_compat,
    trigger_geometry,
    trigger_status,
    unit_references,
)
from descape.render_cache_splice import UnitSplice
from descape.render_units import created_unit_point
from descape.scenario_io import (
    parse_triggers,
)
from descape.trigger_panel_fields import GROUP_AREA, GROUP_OBJECTS, PICK_POINT, PICK_RECT, PICK_TYPE
from descape.unit_filter import UnitFilter
from descape.unit_model import (
    new_unit,
)
from descape.viewer_common import (
    _entries_of,
    _unit_name,
)

# GH #168: the most shapes a several-trigger overlay draws. Every unit edit re-derives
# the union, and 300 is what fits a 25 ms refresh on the corpus file with the most shapes.
MAX_OVERLAY_SHAPES = 300
# GH #167: past this many trigger-created units every ghost draws as its mark, never a mix by count.
MAX_SPRITE_GHOSTS = 500
# With the overlay off, a several-trigger union derived for its ghosts alone stops past this many.
MAX_UNION_GHOSTS = 300


class _ViewerTriggerOverlayMixin:
    """ViewerWindow's trigger pickers and overlay (viewer.py's docstring has the split's rules)."""

    def _on_area_edit_toggled(self, checked: bool) -> None:
        self._update_tool_enabled()
        if (
            checked
            and self.mode == "triggers"
            and self._area_edit_target is None
            and self._location_edit_target is None
        ):
            self._log_status("Select one condition or effect with an area or location to edit it on the map")

    def _area_options_shown(self) -> bool:
        """Edit Area's four checkboxes: under Pan in Triggers mode."""
        return self.mode == "triggers" and self._current_tool == "pan"

    def _area_edit_on(self) -> bool:
        return self.area_edit_check.isChecked() and self.area_edit_check.isEnabled()

    def _regate_area_options(self) -> None:
        self.area_edit_check.setEnabled(self._area_options_shown() and self._area_edit_editable)
        on = self._area_edit_on()
        self.area_move_check.setEnabled(on)
        self.area_resize_check.setEnabled(on)
        # It only shapes the resize border.
        self.area_tile_borders_check.setEnabled(on and self.area_resize_check.isChecked())

    def _on_area_options_toggled(self) -> None:
        self._regate_area_options()
        self.map_view.set_area_edit_options(
            self.area_move_check.isChecked(), self.area_resize_check.isChecked(), self.area_tile_borders_check.isChecked()
        )

    def _push_area_edit_target(self) -> None:
        """GH #229: hands the map view the one area and location Edit Area may
        drag, each or None. Live only with the box ticked, in Triggers mode with the
        overlay shown and Pan active, and for the form's one entry where that group
        is set and on the map. A location its location object overrides gets no
        handle, as the game ignores it. The map view stands down while a pick is armed."""
        target = location = origin = None
        ref = self._trigger_overlay_ref
        loaded = self.scenario
        if (
            self._area_edit_on()
            and self.mode == "triggers"
            and self.trigger_overlay_action.isChecked()
            and self._current_tool == "pan"
            and loaded is not None
            and ref is not None
            and len(ref[0]) == 1
            and len(ref[1]) == 1
        ):
            (index,), (entry_ref,) = ref[0], tuple(ref[1])
            mm = loaded.map_manager
            entry_shapes = [
                s for s in self._trigger_overlay_shapes if s.trigger_index == index and s.entry_ref == entry_ref
            ]
            for shape in entry_shapes:
                if shape.fields == trigger_geometry.AREA_FIELDS:
                    x1, y1, x2, y2 = shape.coords
                    if 0 <= x1 <= x2 < mm.map_width and 0 <= y1 <= y2 < mm.map_height:
                        target = (index, *entry_ref, shape.coords)
                elif shape.fields == trigger_geometry.LOCATION_FIELDS:
                    x, y = shape.coords
                    on_map = 0 <= x < mm.map_width and 0 <= y < mm.map_height
                    if on_map and not trigger_geometry.location_superseded(entry_shapes):
                        location = (index, *entry_ref, shape.coords)
                        origin = trigger_geometry.run_origin(entry_shapes)
        previous = self._area_edit_target or self._location_edit_target
        current = target or location
        self._area_edit_target, self._location_edit_target = target, location
        if previous is not None and current is not None and previous[:3] != current[:3]:
            # Another entry with the same coords: a live drag belongs to the old one.
            self.map_view.set_area_edit(None)
            self.map_view.set_location_edit(None)
        self.map_view.set_area_edit(None if target is None else target[3])
        self.map_view.set_location_edit(None if location is None else location[3], origin)

    def _on_area_edited(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """GH #229's release: one "Move Area" or "Resize Area" record, through GH #138's funnel."""
        target = self._area_edit_target
        if target is None or not self.trigger_panel.is_pick_target_current(target):
            return
        old = target[3]
        moved = (x2 - x1, y2 - y1) == (old[2] - old[0], old[3] - old[1])
        values = dict(zip(trigger_geometry.AREA_FIELDS, (x1, y1, x2, y2), strict=True))
        self.set_entry_field_group(target[0], (target[1], target[2]), values, "Move Area" if moved else "Resize Area")

    def _on_location_edited(self, x: int, y: int) -> None:
        """GH #41's release: one "Move Location" record, through GH #138's funnel."""
        target = self._location_edit_target
        if target is None or not self.trigger_panel.is_pick_target_current(target):
            return
        values = dict(zip(trigger_geometry.LOCATION_FIELDS, (x, y), strict=True))
        self.set_entry_field_group(target[0], (target[1], target[2]), values, "Move Location")

    def _on_trigger_selection_changed(self) -> None:
        """The panel's selection-set callback: Copy gating, and the overlay,
        since a multi-selection or a deselect never reaches the entry form."""
        self._update_tool_enabled()
        self._on_trigger_focus_changed()

    def _on_trigger_focus_changed(self) -> None:
        """The panel's trigger or entry selection moved: re-read it and redraw.
        One trigger emphasises its selected entries, or the current one when
        none is selected. Several triggers draw their union with no emphasis,
        as the panel shows no form for them (GH #168)."""
        panel = self.trigger_panel
        index = panel.current_trigger_index()
        selected = panel.selected_trigger_indices()
        if len(selected) > 1:
            self._trigger_overlay_ref = (tuple(selected), frozenset())
        elif index is None:
            self._trigger_overlay_ref = None
        else:
            # The property form's own fallback (TriggerPanel._form_refs_now()).
            refs = panel.selected_entry_refs()
            current = panel.current_entry_ref()
            if not refs and current is not None:
                refs = [current]
            self._trigger_overlay_ref = ((index,), frozenset(refs))
        # A compare, not a plain disarm: a list pick's own write re-enters here.
        for target in (self._unit_picker, self._tile_picker):
            if target is not None and self._trigger_overlay_ref != (
                (target[0],),
                frozenset({(target[1], target[2])}),
            ):
                self.disarm_unit_picker()
                break
        self._refresh_trigger_overlay()
        # The window hears entry selection only through here (GH #137's Cut/Copy gating).
        self._update_clipboard_actions()

    # -- trigger unit references ---------------------------------------------

    def _unit_reference_index(self) -> unit_references.ReferenceIndex:
        if self.scenario is None:
            return unit_references.EMPTY_INDEX
        if self._unit_ref_index is None:
            self._unit_ref_index = unit_references.build_reference_index(self.scenario)
        return self._unit_ref_index

    def _describe_unit_reference(self, ref_id) -> str:
        return unit_references.describe(self._unit_reference_index(), ref_id)

    def _patch_unit_ref_index(self, changed: list[UnitSplice] | None) -> None:
        """Keeps the trigger reference index in step with a unit edit: patched
        per splice when `changed` is given, else dropped for a lazy rebuild
        (as is one unit_references.patch_reference_index() refuses)."""
        index = self._unit_ref_index
        if index is None:
            return
        if changed is None or not unit_references.patch_reference_index(index, self.scenario, changed):
            self._unit_ref_index = None

    def _edited_trigger_indices(self) -> frozenset[int]:
        """The triggers that may differ from their parsed bytes, for the panel's
        status memo. None edited before the first edit builds the model."""
        if self.trigger_edits is None:
            return frozenset()
        return frozenset(self.trigger_edits.dirty_indices())

    def _trigger_status_context(self, vocabulary, trigger_ids) -> trigger_status.StatusContext:
        index = self._unit_reference_index()
        return trigger_status.StatusContext(
            vocabulary, trigger_ids, unit_exists=lambda ref_id: index.get(ref_id) is not None
        )

    def _on_unit_references_moved(self) -> None:
        """A unit mutation can move, remove or reassign what a trigger names.
        The index itself is already patched or dropped (_patch_unit_ref_index()).
        Statuses recompute only in Triggers mode; mode entry repopulates otherwise."""
        self.trigger_panel.refresh_unit_reference_labels(recompute_status=self.mode == "triggers")
        ref = self._trigger_overlay_ref
        if ref is not None and len(ref[0]) > 1 and not self.trigger_overlay_action.isChecked():
            # A union with the overlay off draws only ghosts, read off trigger fields alone.
            return
        # The held ref, not a panel read: undo is global, and the panel may sit unrefreshed.
        self._refresh_trigger_overlay()

    def _on_pick_unit_requested(self, target) -> None:
        """The panel's Pick, Set and eyedropper buttons: a target arms, None
        disarms. A sixth slot is a tile pick's kind (GH #138) or PICK_TYPE, which
        picks a unit (GH #276)."""
        if target is None:
            self.disarm_unit_picker()
        elif target[5:] in ((PICK_POINT,), (PICK_RECT,)):
            self.arm_tile_picker(target)
        else:
            self.arm_unit_picker(target)

    def arm_tile_picker(self, target: tuple) -> None:
        """GH #138's Set Location (one click) or Set Area (one drag), one-shot:
        the pick writes the group as one record and disarms."""
        if self.scenario is None:
            return
        # Every other picker off first, the Garrison block's included.
        self.disarm_unit_picker()
        self.pan_action.setChecked(True)
        self._tile_picker = target
        self.map_view.set_tile_picker(target[5])
        self.trigger_panel.set_pick_armed(target)
        self._focus_map_view()
        if target[5] == PICK_RECT:
            self._log_status("Drag a rectangle to set the area; Esc to cancel")
        else:
            self._log_status("Click a tile to set the location; Esc to cancel")

    def _disarm_tile_picker(self) -> None:
        if self._tile_picker is None:
            return
        self._tile_picker = None
        self.map_view.set_tile_picker(None)
        self.trigger_panel.set_pick_armed(None)

    def _on_tile_picked(self, tile: tuple[int, int]) -> None:
        self._write_tile_pick(PICK_POINT, dict(zip(trigger_geometry.LOCATION_FIELDS, tile, strict=True)), "Set Location")

    def _on_rect_picked(self, x1: int, y1: int, x2: int, y2: int) -> None:
        # Min into x1/y1 and max into x2/y2: the order the game and a re-parse store.
        values = dict(zip(trigger_geometry.AREA_FIELDS, (x1, y1, x2, y2), strict=True))
        self._write_tile_pick(PICK_RECT, values, "Set Area")

    def _write_tile_pick(self, kind: str, values: dict, label: str) -> None:
        """Disarm first (the write re-enters the focus compare), then write
        only if the armed entry is still the one the form shows."""
        target = self._tile_picker
        if target is None or target[5] != kind:
            return
        self._disarm_tile_picker()
        if not self.trigger_panel.is_pick_target_current(target):
            return
        self.set_entry_field_group(target[0], (target[1], target[2]), values, label)

    def _unit_reference_tile(self, ref_id) -> tuple[int, int] | None:
        """A placed unit's own tile, or None when the id names nothing on the map."""
        ref = self._unit_reference_index().get(ref_id)
        return None if ref is None or ref.bounds is None else ref.own_tile

    def go_to_trigger_group(self, target: tuple) -> None:
        """GH #138's Go to: centre the map on an entry's location, area centre
        or selected objects' centroid, read off a fresh parse_triggers() like
        the overlay, so it works on a read-only file. Clamped to the map."""
        trigger_index, kind, entry_index, group = target
        loaded = self.scenario
        if loaded is None or loaded.trigger_read_supported is not True:
            return
        manager = parse_triggers(loaded)
        if (
            manager is None
            or not library_compat.vocabulary_is_available(loaded.scenario_version)
            or not 0 <= trigger_index < len(manager.triggers)
        ):
            return
        entries = _entries_of(manager.triggers[trigger_index], kind)
        if not 0 <= entry_index < len(entries):
            return
        entry = entries[entry_index]
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        if kind == "condition":
            definitions, presentation, type_attribute = (
                vocabulary.conditions, vocabulary.condition_presentation, "condition_type"
            )
        else:
            definitions, presentation, type_attribute = vocabulary.effects, vocabulary.effect_presentation, "effect_type"
        shapes = trigger_geometry.shapes_for_entry(
            entry,
            definitions.get(getattr(entry, type_attribute, None)),
            trigger_index=trigger_index,
            entry_kind=kind,
            entry_index=entry_index,
            references=self._unit_reference_index(),
            presentation=presentation,
        )
        tile = None
        if group == GROUP_OBJECTS:
            tile = trigger_geometry.tile_centroid(
                [s.anchor for s in shapes if s.fields == (trigger_geometry.SELECTED_OBJECTS_FIELD,) and s.anchor]
            )
        else:
            wanted = trigger_geometry.SHAPE_AREA if group == GROUP_AREA else trigger_geometry.SHAPE_LOCATION
            shape = next((s for s in shapes if s.shape == wanted), None)
            if shape is not None:
                tile = trigger_geometry.area_centre(shape.coords) if group == GROUP_AREA else shape.coords
        if tile is None:
            self._log_status(f"Go to: this entry's {group} is not set")
            return
        mm = loaded.map_manager
        x = max(0, min(mm.map_width - 1, tile[0]))
        y = max(0, min(mm.map_height - 1, tile[1]))
        self.map_view.center_on_tile(x, y)
        self._log_status(f"Centred on the {'selected objects' if group == GROUP_OBJECTS else group} ({x}, {y})")

    def arm_unit_picker(self, target: tuple) -> None:
        """Pick from map for one Unit/Unit[] field. Leaves self._selection
        alone: the field's current units are drawn through MapView's selection
        layer only, and set back to nothing on disarm."""
        if self.scenario is None:
            return
        self._disarm_garrison_picker()
        self._disarm_tile_picker()
        self.pan_action.setChecked(True)
        self._unit_picker = target
        self.map_view.set_unit_picker(True)
        if self.map_view._unit_index is None:
            self._rebuild_unit_index()
        self.trigger_panel.set_pick_armed(target)
        self._refresh_picker_highlight()
        self._focus_map_view()
        if target[5:] == (PICK_TYPE,):
            hint = f"Click a unit to copy its type into {self.trigger_panel.type_pick_caption(target)}; Esc to cancel"
        else:
            verb = "add or remove units in" if target[4] else "set"
            hint = f"Click a unit to {verb} {target[3].replace('_', ' ')}; Esc to finish"
        if self._unit_filter != UnitFilter():
            hint += " (units hidden by Filters can't be picked)"
        self._log_status(hint)

    def disarm_unit_picker(self) -> None:
        # Every disarm site (Esc, right-click, mode switch, load) ends the Garrison and tile picks too.
        self._disarm_garrison_picker()
        self._disarm_tile_picker()
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
        self._push_trigger_range_effects()
        self.map_view.set_unit_selection(entries)

    def _on_unit_picked(self, key: tuple[int, int]) -> None:
        """MapView's pick: written through the panel's funnel as one undo record."""
        target = self._unit_picker
        if target is None:
            return
        if target[5:] == (PICK_TYPE,):
            self._on_unit_type_picked(target, key)
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

    def _on_unit_type_picked(self, target: tuple, key: tuple[int, int]) -> None:
        """GH #276's eyedropper, one-shot: the clicked unit's const, read before
        the disarm can drop the index. The unit itself is not touched."""
        index = self.map_view._unit_index
        entry = None if index is None else index.entry_for_key(key)
        if entry is None:
            self.disarm_unit_picker()
            return
        const = entry.unit.unit_const
        written = not self._refuse_locked_edit() and self.trigger_panel.apply_picked_type(target, const)
        if self._unit_picker is None:
            return
        self.disarm_unit_picker()
        if written:
            self._log_status(f"Picked {_unit_name(const)} from ({entry.own_x}, {entry.own_y})")

    def _rederive_trigger_overlay(self) -> None:
        """After a trigger edit or undo: the snapshot is plain ints, so it is
        stale the moment a coordinate or an entry index changes. Outside
        Triggers mode the panel is not refreshed, so the held ref is used."""
        if self.mode == "triggers":
            self._on_trigger_focus_changed()
        else:
            self._refresh_trigger_overlay()

    def _refresh_trigger_overlay(self) -> None:
        self._trigger_overlay_shapes = self._draw_trigger_overlay()
        self._push_area_edit_target()

    def _draw_trigger_overlay(self) -> list[trigger_geometry.TriggerShape]:
        """Re-derives the overlay from a fresh parse_triggers(), never a held
        manager (the library's field gating is process-global). Read-only, so
        it never goes through TriggerEditModel.

        Several triggers past MAX_OVERLAY_SHAPES draw nothing, never a partial
        union, and drop the ref so a unit edit does not re-derive them. One
        trigger is never capped. With the overlay off a union is not derived
        at all (TASK-352): the ref is kept and turning it on derives it.

        GH #167's ghosts come from the same loop and parse. With the overlay
        off and the ghosts on, a union is derived for its ghosts alone and
        draws none past MAX_UNION_GHOSTS, and a unit edit does not re-derive
        it (_on_unit_references_moved()); with both off, not at all. Past
        MAX_SPRITE_GHOSTS every ghost is a mark."""
        ref = self._trigger_overlay_ref
        loaded = self.scenario
        # trigger_read_supported first: a ref only exists once the panel parsed.
        if ref is None or loaded is None or loaded.trigger_read_supported is not True:
            self._clear_trigger_drawing()
            return []
        union = len(ref[0]) > 1
        ghosts_only = union and not self.trigger_overlay_action.isChecked()
        if ghosts_only and not self.trigger_ghosts_action.isChecked():
            self._clear_trigger_drawing()
            return []
        manager = parse_triggers(loaded)
        indices, emphasis = ref
        if (
            manager is None
            or not library_compat.vocabulary_is_available(loaded.scenario_version)
            or not all(0 <= index < len(manager.triggers) for index in indices)
        ):
            self._trigger_overlay_ref = None
            self._clear_trigger_drawing()
            return []
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        references = None if ghosts_only else self._unit_reference_index()
        shapes = []
        records: list[trigger_geometry.CreatedUnit] = []
        for index in indices:
            trigger = manager.triggers[index]
            records.extend(trigger_geometry.created_units_for_trigger(trigger, vocabulary, trigger_index=index))
            if ghosts_only:
                if len(records) > MAX_UNION_GHOSTS:
                    self._clear_trigger_drawing()
                    self._log_trigger_ghost_cap(
                        ("union", indices),
                        f"Trigger-created units hidden: the {len(indices)} selected triggers create more than "
                        f"{MAX_UNION_GHOSTS} units to draw",
                    )
                    return []
                continue
            # Walls are never drawn (MapView.show_trigger_overlay()), so they do not count.
            shapes.extend(
                shape
                for shape in trigger_geometry.shapes_for_trigger(
                    trigger, vocabulary, trigger_index=index, references=references
                )
                if shape.shape != trigger_geometry.SHAPE_WALL
            )
            # Checked per trigger: Select All on a huge list stops here instead of deriving it all.
            if union and len(shapes) > MAX_OVERLAY_SHAPES:
                self._trigger_overlay_ref = None
                self._clear_trigger_drawing()
                self._log_status(
                    f"Trigger overlay hidden: the {len(indices)} selected triggers have more than "
                    f"{MAX_OVERLAY_SHAPES} areas, locations and units to draw"
                )
                return []
        if ghosts_only:
            self.map_view.clear_trigger_overlay()
        else:
            self.map_view.show_trigger_overlay(shapes, emphasis or None)
        sprites = len(records) <= MAX_SPRITE_GHOSTS
        self.map_view.show_trigger_ghosts(records, emphasis or None, sprites)
        if not sprites and self.trigger_ghosts_action.isChecked() and self._ghost_sprites_on():
            self._log_trigger_ghost_cap(
                ("sprites", indices, len(records)),
                f"Trigger-created units drawn as marks: {len(records)} is more than the "
                f"{MAX_SPRITE_GHOSTS} drawn as sprites",
            )
        else:
            self._trigger_ghost_cap_logged = None
        return shapes

    def _clear_trigger_drawing(self) -> None:
        self.map_view.clear_trigger_overlay()
        self.map_view.clear_trigger_ghosts()

    def _log_trigger_ghost_cap(self, key: tuple, message: str) -> None:
        """A ghost cap's status line, once per (selection, count): a re-derive repeats it. Turning the ghosts on says it again."""
        if key != self._trigger_ghost_cap_logged:
            self._trigger_ghost_cap_logged = key
            self._log_status(message)

    def _resolve_trigger_ghost(self, record: trigger_geometry.CreatedUnit, sprites: bool) -> tuple[list, tuple | None]:
        """MapView's resolver for one trigger-created unit (GH #167): its
        sprite draws, else its coloured mark, where the game puts it
        (render_units.created_unit_point(), GH #369), built by new_unit() and
        never add()ed. Nothing for an off-map tile or a player
        with no colour slot. `sprites` False is the count cap: the mark.
        No depoison() of its own: MapView calls prepare_trigger_ghost_resolve() once per rebuild."""
        loaded = self.scenario
        if loaded is None:
            return [], None
        mm = loaded.map_manager
        x, y = record.tile
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height) or not 0 <= record.player < len(loaded.player_colors):
            return [], None
        rotation = facet_preview.ghost_rotation(record.unit_const, record.facet)
        gx, gy = created_unit_point(record.unit_const, x, y)
        ghost = new_unit(
            loaded._scenario.uuid, record.player, record.unit_const, gx, gy, reference_id=-1, rotation=rotation, depoison=False
        )
        draws = self._ghost_sprite_draws(record.player, ghost) if sprites else []
        if draws:
            return draws, None
        return [], self._ghost_mark(record.player, ghost)
