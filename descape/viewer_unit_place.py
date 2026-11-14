"""ViewerWindow's unit placement: the placement point, Place Unit, gate plans, the place preview and
its ghost, scatter, wall menus, the Cliff tool, Unit Shape and the wall run."""

from __future__ import annotations

import dataclasses
import weakref

from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QPixmap,
)
from PyQt5.QtWidgets import (
    QDialog,
)

from descape import (
    cliff_catalog,
    cliff_chain,
    debug_log,
    game_dataset,
    garrison,
    gate_orientation,
    object_catalog,
    perf_trace,
    place_rotation,
    placement_rules,
    scatter,
    settings,
    unit_brush,
    unit_kind,
    unit_pick,
    unit_rotation,
    unit_shape,
    unit_sprites,
    unit_variant,
    wall_run,
)
from descape.constant_picker import preview_pixmap
from descape.render_cache_splice import UnitSplice
from descape.render_units import NON_BUILDING_SPAN, _unit_gen_cached, occupied_tiles_for, snapped_anchor
from descape.scatter_dialog import ScatterDialog, units_region
from descape.terrain_palette import tile_span
from descape.unit_filter import GAIA_PLAYER_ID
from descape.unit_model import (
    new_unit,
)
from descape.viewer_common import (
    _DEFAULT_WALL_CONST,
    _TOOL_LABELS,
    _WALL_TOOLS,
    OFF_GRID_FALLBACK_MESSAGE,
    _ghost_entry,
    _unit_name,
)

# TASK-327: wall_run.wall_scene() per scenario and unit_gen, which a gate's hover ghost asks per move.
_GATE_SCENE_MEMO: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


@game_dataset.cache(maxsize=1)
def _wall_consts() -> frozenset[int]:
    """Place Unit's wall-run consts (GH #98). Lazy: the graphic map it
    reads is not needed to import this module."""
    return frozenset(unit_sprites.wall_family_consts())


# The Cliff hover keeps one ghost's parts per tile crossed; past this many it starts over.
_CLIFF_HOVER_PARTS_MAX = 64


# TASK-288: Place Unit and Scatter coerced a GAIA-only object to GAIA.
_GAIA_ONLY_PLACED_SUFFIX = " (GAIA-only, so placed for GAIA)"


class _ViewerUnitPlaceMixin:
    """ViewerWindow's unit placement, walls and cliffs (viewer.py's docstring has the split's rules)."""

    def _placement_point(self, pos, modifiers, unit_const: int | None = None) -> tuple[float, float] | None:
        """Where a click puts a unit -- the one place the snapped/off-grid choice
        is made, so Place Unit and Move Unit cannot disagree about it.

        Snapped (the default) is `render_units.snapped_anchor` for `unit_const`'s
        span: `tile + 0.5` on an odd-span axis, which is every axis of a 1x1
        unit and of a const-less call, and the whole number nearest the
        cursor on an even-span one, so a House centres on the cursor on the
        game's grid (GH #206). `_pick_tile()` resolves the
        clicked SCREEN pixel to an integer tile in all three render styles
        (Flat's division, Stepped's screen_to_tile, Sloped's pick-plane
        lookup). Off-grid is `_pick_map_point()`, the continuous inverse added by
        off-grid placement's Stage 2 -- which supersedes the "Sloped has no
        analytic inverse" reason this feature used to be deferred for. That
        claim is true of the TERRAIN surface and was never true of the
        point-placement one, which inverts closed-form.

        Off-grid is reached by the toolbar checkbox or by holding Alt for a
        one-off (D2's modifier), either one. Falls back to the snapped point
        whenever the inverse returns None -- a degenerate denominator or a
        failed round-trip, both reachable on a steep Sloped ramp -- rather
        than refusing the click: a placement that silently does nothing is
        worse than one that snaps.

        **The fallback says so.** Without a status line a tester who asked
        for off-grid placement and got a grid point cannot tell "the inverse
        refused this pixel" from "the checkbox didn't apply" from "the window
        manager ate the Alt", and those want three different responses."""
        point, fell_back = self._resolve_placement_point(pos, modifiers, unit_const=unit_const)
        if fell_back:
            self._log_status(OFF_GRID_FALLBACK_MESSAGE)
        return point

    def _off_grid_requested(self, modifiers) -> bool:
        """The Off-grid toolbar checkbox, or a held Alt for a one-off."""
        # `modifiers` is None from a caller that has no real event behind it
        # (a test, or a programmatic place), so it must not be bit-tested.
        alt = modifiers is not None and bool(modifiers & Qt.AltModifier)
        return self.off_grid_check.isChecked() or alt

    def _restrictions_lifted(self) -> bool:
        """The Free placement checkbox (GH #176): Place Unit and Move skip
        placement_rules' terrain and object rules. It has no modifier."""
        return self.unrestricted_check.isChecked()

    def _resolve_placement_point(
        self, pos, modifiers, unit_const: int | None = None
    ) -> tuple[tuple[float, float] | None, bool]:
        """`_placement_point` without the status line -- (point, fell_back).

        Split out for the mid-drag move preview, which resolves this same
        point on EVERY mouse-move: on a steep Sloped ramp the inverse can
        refuse pixel after pixel, and logging each one would fill the status
        log with one line per frame. The preview reads `fell_back` and ignores
        it; the two committing callers log it, exactly as before."""
        tile = self.map_view._pick_tile(pos)
        if tile is None:
            return None, False
        off_grid = self._off_grid_requested(modifiers)
        span = (1, 1) if unit_const is None else tile_span(unit_const, NON_BUILDING_SPAN)
        # The snapped anchor needs the cursor point only on an even-span axis.
        point = self.map_view._pick_map_point(pos) if off_grid or span[0] % 2 == 0 or span[1] % 2 == 0 else None
        if off_grid and point is not None:
            return point, False
        mm = self.scenario.map_manager
        return snapped_anchor(tile, point, span, mm.map_width, mm.map_height), off_grid

    def on_unit_place(self, pos, modifiers) -> None:
        """MapView's seventh injected callable -- a left click with the
        Place Unit tool active (b1.4).

        Snapped by default, off-grid with the param toolbar's checkbox or a held
        Alt -- see `_placement_point`, which is where that choice lives.

        Refused, with nothing added and no undo record, where
        placement_rules says the game would refuse it (GH #176), unless Free
        placement is ticked. The walls a gate replaces never block it.

        Pointed at a host, with Free placement off, an object that can go
        inside something goes inside it or nowhere (TASK-549,
        _place_into_garrison()); the placement rules then play no part."""
        if self.scenario is None:
            return
        object_id = self.units_panel.selected_object_const()
        point = self._placement_point(pos, modifiers, unit_const=object_id)
        if point is None:
            return
        if object_id is None:
            self._log_status("Place Unit: choose an object first")
            return
        # TASK-288: GAIA-only objects always go to GAIA, whatever the combo says.
        chosen = self.units_panel.owner_id()
        player = unit_kind.placement_owner(object_id, chosen)
        # GH #172: the same value on_place_preview's ghost was built with.
        rotation = place_rotation.rotation_for(self._place_rotation, object_id, player)
        # TASK-549: pointed at a host that can hold anything, the object goes inside or nowhere.
        host = self._place_garrison_host(pos, object_id)
        if host is not None:
            self._place_into_garrison(host, object_id, player, chosen, rotation)
            return
        # The same label as _unit_edit's own op, so the lazy model build lands inside it.
        with perf_trace.op("Place unit"):
            model = self._ensure_unit_edits()
            if model is None:
                return
            x, y = point
            gate_plan, gate_walls = self._gate_plan(model, object_id, x, y)
            if not self._restrictions_lifted():
                reason = placement_rules.refusal(self.scenario, object_id, x, y, exclude=gate_walls)
                if reason is not None:
                    self._log_status(f"Place Unit: {reason}. Tick Free placement to place it anyway.")
                    return
            if gate_plan:
                # GH #159: removals shift list indices, so no splice; one wholesale record.
                with self._unit_edit(model, "Place unit", wall_run.touched_players(player, gate_plan)):
                    unit = wall_run.apply_gate_plan(model, player, object_id, x, y, gate_plan)
                    self._selection = [unit_pick.unit_key(player, unit)]
            else:
                splices: list[UnitSplice] = []
                with self._unit_edit(model, "Place unit", [player], splices):
                    unit = model.add(player, object_id, x, y, rotation=rotation)
                    # Placed units become the selection (same reasoning as
                    # reassign's own key update): the id just chosen is the one worth
                    # showing in the inspector next, not whatever was selected before.
                    self._selection = [unit_pick.unit_key(player, unit)]
                    new_own, new_tiles = self._unit_footprint(unit)
                    idx = self._unit_list_index(player, unit)
                    splices.append(UnitSplice(player, idx, unit, None, new_own, (), new_tiles))
                    index = self.map_view._unit_index
                    if index is not None:
                        with perf_trace.phase("unit_index_patch"):
                            unit_pick.patch_index_for_add(self.scenario, index, player, unit, self._unit_filter)
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        status = f"Placed {_unit_name(object_id)} for {owner_text} at ({x:g}, {y:g})"
        if rotation:
            status += f", {self._placement_rotation_text(object_id, rotation)}"
        if gate_plan:
            if gate_plan.removals:
                count = len(gate_plan.removals)
                status += f", replacing {count} wall{'' if count == 1 else 's'}"
            if gate_plan.rewrites:
                status += f", reshaping {len(gate_plan.rewrites)} adjacent"
        if player != chosen:
            status += _GAIA_ONLY_PLACED_SUFFIX
        # The same gap the cliff tool surfaces for Show GAIA, generalized:
        # placing something the live filter hides looks exactly like the tool
        # doing nothing. Asked of the filter rather than of a const set, so it
        # covers walls, gates, eye candy, obstacles, trees and owner alike, and picks up
        # whatever gate the next one adds for free.
        if not self._unit_filter.matches(player, unit):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)
        # GH #164: a random pick rolls the next const, and the ghost shows it before the next click.
        self.units_panel.reroll_pending()

    def _place_into_garrison(self, host_entry, object_id: int, player: int, chosen: int, rotation: float) -> None:
        """Place Unit's click on a host (TASK-549): a new `object_id` at the
        host's point, inside it, in one "Place unit" record, or nothing with a
        status line when garrison.refusal() refuses. The host becomes the
        selection, so its Garrison list shows the new row. A gate never gets
        here (garrison.can_garrison()), so there is no gate plan."""
        message = self._place_garrison_refusal(host_entry, object_id)
        if message is not None:
            self._log_status(
                f"Place Unit: {message.removeprefix('Garrison: ')}. Tick Free placement to place it anyway."
            )
            return
        host = host_entry.unit
        with perf_trace.op("Place unit"):
            model = self._ensure_unit_edits()
            if model is None:
                return
            splices: list[UnitSplice] = []
            with self._unit_edit(model, "Place unit", [player], splices):
                unit = model.add(
                    player,
                    object_id,
                    host.x,
                    host.y,
                    getattr(host, "z", 0.0),
                    rotation=rotation,
                    garrisoned_in_id=host.reference_id,
                )
                new_own, new_tiles = self._unit_footprint(unit)
                idx = self._unit_list_index(player, unit)
                splices.append(UnitSplice(player, idx, unit, None, new_own, (), new_tiles))
                index = self.map_view._unit_index
                if index is not None:
                    unit_pick.patch_index_for_add(self.scenario, index, player, unit, self._unit_filter)
                # A filter-hidden host is synthesized with order -1 and its key resolves to nothing.
                self._selection = [] if host_entry.order == -1 else [unit_pick.unit_key(host_entry.player_id, host)]
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        held = len(self._garrison_occupants(host))
        status = (
            f"Placed {_unit_name(object_id)} for {owner_text} inside {_unit_name(host.unit_const)}"
            f" ({held}/{garrison.capacity(host.unit_const)})"
        )
        if player != chosen:
            status += _GAIA_ONLY_PLACED_SUFFIX
        # No Filters note: Show Garrisoned off hides every occupant, and the host's list shows it.
        self._log_status(status)
        self.units_panel.reroll_pending()

    def _gate_plan(self, model, object_id: int, x: float, y: float) -> tuple[wall_run.GatePlan | None, list]:
        """(plan, walls) for a gate click at (x, y): the walls it replaces or
        reshapes (GH #159), or None for a non-gate const or an off-map
        footprint, and every wall under its footprint.

        `walls` is taken before the plan drops the ones remove_many() would
        refuse, and is what the placement rules exclude (GH #176): the gate
        replaces the walls it can and still sits over a referenced one. It is
        _gate_walls(), which the hover ghost asks too (TASK-327)."""
        if not gate_orientation.is_gate(object_id):
            return None, []
        mm = self.scenario.map_manager
        gate_tiles = occupied_tiles_for(object_id, x, y, mm.map_width, mm.map_height)
        if gate_tiles is None:
            return None, []
        existing_tiles, existing_walls = self._gate_wall_scene()
        plan = wall_run.plan_gate_over_walls(
            gate_tiles, existing_tiles=existing_tiles, existing_walls=existing_walls
        )
        walls = self._gate_walls(object_id, x, y)
        # Keep a wall remove_many() would refuse (a referenced or -1 reference_id), as Delete does.
        removals = [(p, u) for p, u in plan.removals if not model.referencing(u)]
        if len(removals) != len(plan.removals):
            plan = dataclasses.replace(plan, removals=removals)
        return plan, walls

    def _gate_wall_scene(self):
        """wall_run.wall_scene() of the live map, read once per unit edit and
        game dataset (TASK-327): a data mod or game-data switch changes the wall
        consts without bumping unit_gen."""
        scenario = self.scenario
        mm = scenario.map_manager
        active, mod = game_dataset.active(), game_dataset.data_mod()
        seen = getattr(self, "_gate_scene_dataset", None)
        if seen is None or seen[0] != active or seen[1] is not mod:
            _GATE_SCENE_MEMO.clear()
            self._gate_scene_dataset = (active, mod)
        return _unit_gen_cached(
            _GATE_SCENE_MEMO, scenario, lambda: wall_run.wall_scene(scenario, mm.map_width, mm.map_height)
        )

    def _gate_walls(self, object_id: int, x: float, y: float) -> list:
        """Every wall under a gate's footprint at (x, y), which the placement
        rules do not count (GH #159): the model-free half of _gate_plan(), so
        Place Unit's hover asks it without building the unit model. [] for a
        non-gate const or an off-map footprint."""
        if not gate_orientation.is_gate(object_id):
            return []
        mm = self.scenario.map_manager
        gate_tiles = occupied_tiles_for(object_id, x, y, mm.map_width, mm.map_height)
        if gate_tiles is None:
            return []
        _tiles, existing_walls = self._gate_wall_scene()
        return [unit for _player_id, unit in wall_run.walls_under(gate_tiles, existing_walls)]

    def _move_plan(self, model, moves, *, gate_walls: bool, consts=None) -> wall_run.MovePlan | None:
        """What moving `moves`, [(entry, new_x, new_y), ...], changes in the
        walls (wall_run.scenario_move_plan()): each wall carried onto another
        tile takes its new place's shape and the walls beside where it left
        and landed reshape (GH #266), and with `gate_walls` and "Moving a gate
        updates walls" ticked, a gate's move swaps walls too (GH #194). None
        with neither a wall nor such a gate among them, or none changing
        tiles. The box is read even while hidden, as Free placement is for a
        drag. `consts` is scenario_move_plan()'s, for a gate a drag turns.

        A falsy plan can still carry `blocked` for the status line."""
        gate_walls = gate_walls and self.gate_walls_check.isChecked()
        # Checked before the scene walk, so a move with no wall or gate in it pays nothing.
        if not any(
            unit_sprites.rotation_variant_eligible(entry.unit.unit_const)
            or (gate_walls and gate_orientation.is_gate(entry.unit.unit_const))
            for entry, _x, _y in moves
        ):
            return None
        mm = self.scenario.map_manager
        return wall_run.scenario_move_plan(
            self.scenario,
            mm.map_width,
            mm.map_height,
            [(entry.unit, x, y) for entry, x, y in moves],
            gate_walls=gate_walls,
            removable=lambda unit: not model.referencing(unit),
            consts=consts,
        )

    def _gate_change_plan(self, model, changes, *, keep_ids=frozenset()) -> wall_run.GateMovePlan | None:
        """What giving units a gate const changes in the walls (GH #339):
        `changes`, [(unit, new_const, new_x, new_y), ...], from an orientation
        cycle or Replace into a gate const. The walls under each new
        footprint go and the walls beside the old and new tiles re-derive,
        always (a gate never stands on walls, as Place's GH #159), with no
        fill. None when no `new_const` is a gate, before any scene walk. A
        wall in `keep_ids` (a Replace row the pre-flight refused) or one
        remove_many() would refuse stays as a neighbour."""
        if not any(gate_orientation.is_gate(const) for _unit, const, _x, _y in changes):
            return None
        mm = self.scenario.map_manager
        return wall_run.scenario_gate_changes(
            self.scenario,
            mm.map_width,
            mm.map_height,
            changes,
            removable=lambda unit: not model.referencing(unit) and id(unit) not in keep_ids,
            fill=False,
        )

    def _apply_wall_reshapes(self, model, plan: wall_run.MovePlan, splices: list) -> None:
        """`plan`'s wall shapes (GH #266), inside an open fields_only move
        edit after every set_position(). A carried wall already has its move
        splice; each other rewritten wall gets an own-equals-own one, the
        Rotation field's shape, unless it moved within its own tile and so
        has one too. Never for a plan that is `wholesale`."""
        for _player_id, unit, variant in plan.moved:
            model.set_wall_variant(unit, variant)
        spliced = {id(splice.unit) for splice in splices}
        for player_id, unit, variant in plan.rewrites:
            model.set_wall_variant(unit, variant)
            if id(unit) in spliced:
                continue
            own, tiles = self._unit_footprint(unit)
            splices.append(UnitSplice(player_id, self._unit_list_index(player_id, unit), unit, own, own, tiles, tiles))

    @staticmethod
    def _gate_move_replaced(plan: wall_run.MovePlan | None) -> list:
        """The walls `plan` removes, which placement_rules must not count as
        blockers: the gate replaces them (GH #194). A wall the model would
        refuse to remove is not among them, so it still blocks."""
        return [] if plan is None else [wall for _player_id, wall in plan.removals]

    @staticmethod
    def _gate_move_clauses(plan: wall_run.MovePlan | None, movers=()) -> str:
        """The status-line tail for a move's plan, in on_unit_place's wording.
        Only walls that stay count as reshaped: a carried wall's own shape,
        and that of one of `movers` moved within its tile, are the user's move
        (GH #266)."""
        if plan is None:
            return ""
        text = ""
        if plan.removals:
            count = len(plan.removals)
            text += f", replacing {count} wall{'' if count == 1 else 's'}"
        if plan.fills:
            text += f", filling {len(plan.fills)}"
        moving = {id(unit) for unit in movers}
        reshaped = sum(1 for _player_id, unit, _variant in plan.rewrites if id(unit) not in moving)
        if reshaped:
            text += f", reshaping {reshaped} adjacent"
        if plan.blocked:
            text += f" ({plan.blocked} occupied tiles left open)"
        return text

    def on_place_preview(self, pos, modifiers) -> None:
        """MapView's place-preview hook (GH #128): the hover ghost, a
        translucent copy of the picked object where a click would put it, for
        Place Unit and the Unit Brush (GH #270).

        Resolved through on_unit_place's own resolver and new_unit(), the
        constructor add() uses, so point and look cannot drift from the click.
        The Unit Brush resolves its point through _unit_brush_hover_point()
        instead, and shows one sample of what a dab places.
        Never builds the lazy unit model: hovering must not pay for it. A
        filtered-out object still ghosts; the click's status line says why it
        then vanishes. Place Unit's walls never get here (it drags them).

        Drawn refused where the click would refuse it (TASK-327), through
        placement_rules.refused(), which builds no message.

        Over a host the click would put it inside (TASK-549,
        _place_garrison_host()), the placement rules are skipped: the ghost is
        drawn in the garrison look and the host outlined when it takes the
        object, refused when it is full or the wrong type. The outline is set
        or cleared once, before the memo, so a memo hit keeps it."""
        view = self.map_view
        const = None if self.scenario is None else self.units_panel.selected_object_const()
        point = None
        if const is not None:
            if self._current_tool == "unit_brush":
                point = self._unit_brush_hover_point(pos, const)
            else:
                point, _fell_back = self._resolve_placement_point(pos, modifiers, unit_const=const)
        if point is None:
            self._place_ghost_memo = None
            view.set_unit_drop_target(None)
            view.set_unit_ghost()
            return
        host = self._place_garrison_host(pos, const)
        host_refusal = None if host is None else self._place_garrison_refusal(host, const)
        view.set_unit_drop_target(host if host is not None and host_refusal is None else None)
        owner = unit_kind.placement_owner(const, self.units_panel.owner_id())
        # GH #172: through new_unit(), not rotation_override, so stored_rotation's GAIA rule applies as on the click.
        rotation = place_rotation.rotation_for(self._place_rotation, const, owner)
        lifted = self._restrictions_lifted()
        key = (
            point,
            const,
            owner,
            rotation,
            self.scenario.player_colors[owner],
            self._sprites_enabled,
            view._terrain_style,
            self._layers.tree_scale,
            self._layers.hero_glow,
            self._layers.wall_alpha,
            id(view._iso_elevations),
            # The rule's inputs; a terrain edit drops the memo in _apply_dirty().
            self.scenario.unit_gen,
            lifted,
            # GH #281: a Fish Outlines toggle with the cursor still adds or drops the ring.
            view._fish_enabled,
            # TASK-549: the cursor can reach a host's art with the snapped point unchanged.
            None if host is None else unit_pick.unit_key(host.player_id, host.unit),
        )
        # Snapped, the point only changes on a tile crossing (half-tile on an even span).
        # The item check drops the memo whenever MapView cleared or replaced the ghost.
        memo = self._place_ghost_memo
        if memo is not None and memo[0] == key and memo[1] is view._unit_ghost_item:
            return
        if host is not None:
            refused = host_refusal is not None
        else:
            # on_unit_place's own rule: the walls a gate replaces never block it.
            refused = not lifted and placement_rules.refused(
                self.scenario, const, *point, exclude=self._gate_walls(const, *point)
            )
        inside = host is not None and not refused
        ghost = new_unit(self.scenario._scenario.uuid, owner, const, *point, reference_id=-1, rotation=rotation)
        ghost_entry = _ghost_entry(owner, ghost)
        draws = self._ghost_sprite_draws(owner, ghost)
        if draws:
            view.set_unit_ghost(draws=draws, refused=refused, garrison=inside)
        else:
            polygons, color = self._ghost_mark(owner, ghost, ghost_entry)
            view.set_unit_ghost(polygons=polygons, color=color, refused=refused, garrison=inside)
        item = view._unit_ghost_item
        if item is not None:
            view.set_unit_ghost_ring(ghost_entry)
        self._place_ghost_memo = None if item is None else (key, item)

    def _unit_brush_hover_point(self, pos, const: int) -> tuple[float, float] | None:
        """Where a dab on the cursor tile stands `const`, or None where a press
        paints nothing whatever the boxes say: a refused pick or a footprint
        past the map edge. The whole pick counts, as _begin_unit_brush_stroke()'s
        refusal does, not just the const rolled to stand for it."""
        if self._unit_brush_pick_refused():
            return None
        tile = self.map_view._pick_tile(pos)
        if tile is None:
            return None
        mm = self.scenario.map_manager
        return unit_brush.hover_anchor(const, tile, mm.map_width, mm.map_height)

    def _unit_brush_pick_refused(self) -> bool:
        """Whether any const of the Units panel's pick is one the Unit Brush refuses (a wall, gate or cliff)."""
        choices = self.units_panel.selected_object_choices() or ()
        return any(unit_brush.is_refused_const(choice) for choice in choices)

    def _refresh_place_ghost(self, *_args) -> None:
        """A catalog pick, owner, Off-grid or Free placement change with no
        mouse move, for Place Unit's or the Unit Brush's ghost."""
        # GH #172: a half notch turned for the old pick must not fire on the new one.
        self.map_view.reset_place_wheel()
        self._update_rotate_enabled()
        self.map_view.refresh_place_ghost()

    def _place_rotate_armed(self) -> bool:
        """In Units mode with an object picked, Place Unit is the tool and the
        pick is not a wall, or the Unit Brush is (GH #355) and a press would
        paint the pick: no const of it is one the brush refuses."""
        if self.scenario is None or self.mode != "units" or self.units_panel.selected_object_const() is None:
            return False
        if self._current_tool == "place_unit":
            return not self._place_shape()
        return self._current_tool == "unit_brush" and not self._unit_brush_pick_refused()

    def _place_ghost_active(self) -> bool:
        """GH #172: whether a rotate input turns Place Unit's or (GH #355) the
        Unit Brush's ghost rather than the selection. True only while the
        ghost is showing, i.e. the cursor is over the map."""
        return self._place_rotate_armed() and self.map_view._unit_ghost_item is not None

    def rotate_place_ghost(self, steps: int | None = None, coarse_direction: int | None = None) -> None:
        """Turns the pending placement rotation (GH #172) by `steps`, or by a
        quarter turn in `coarse_direction`, and redraws the ghost.

        A gate turns by picking its orientation sibling in the catalog
        (pick_object, GH #125's precedent), so the catalog shows what a click
        places and ghost and click keep their single const source.

        Under the Unit Brush (GH #355) the same pending state turns, so a
        switch to Place Unit keeps it. A turn there unticks Random
        facing/variant, so the next stroke paints the turned facing."""
        brush = self._current_tool == "unit_brush"
        prefix = "Unit Brush" if brush else "Placement"
        const = None if self.scenario is None else self.units_panel.selected_object_const()
        if const is None:
            self._log_status(f"{prefix}: choose an object first")
            return
        if steps is None:
            steps = place_rotation.steps_for_coarse(const, coarse_direction or 1)
        owner = unit_kind.placement_owner(const, self.units_panel.owner_id())
        state, new_const, outcome = place_rotation.step(self._place_rotation, const, owner, steps)
        self._place_rotation = state
        name = _unit_name(const)
        if outcome == place_rotation.GATE:
            had_filter = bool(self.units_panel.catalog_view.filter_edit.text())
            if not self.units_panel.pick_object(new_const):
                self._log_status(f"{prefix}: couldn't pick {_unit_name(new_const)} in the catalog")
                return
            status = f"{prefix}: {_unit_name(new_const)}"
            if had_filter and not self.units_panel.catalog_view.filter_edit.text():
                status += " (cleared the catalog filter to show it)"
        elif outcome == place_rotation.NOTHING:
            status = f"{prefix}: {name} has no facing or variant to rotate"
        else:
            rotation = place_rotation.rotation_for(state, const, owner)
            status = f"{prefix}: {name} {self._placement_rotation_text(const, rotation)}"
        if brush and outcome != place_rotation.NOTHING and self.unit_brush_random_check.isChecked():
            self.unit_brush_random_check.setChecked(False)
            status += " (Random facing/variant unticked)"
        self.map_view.refresh_place_ghost()
        self._log_status(status)

    @staticmethod
    def _placement_rotation_text(const: int, rotation: float) -> str:
        """"facing 6/16" or "variant 5/42", in the units _rotate_status and the
        Rotation field show, so the numbers match the inspector (GH #61)."""
        angle_count = unit_rotation.angle_count_for(const)
        if unit_rotation.rotation_is_angle(const):
            return f"facing {unit_rotation.rotation_to_facing(rotation, angle_count)}/{angle_count}"
        count = unit_variant.variant_count_for(const)
        return f"variant {unit_variant.variant_of(rotation, angle_count, count)}/{count}"

    def scatter_units_in_region(self) -> None:
        """Edit > Scatter Units in Region…: N randomized copies of the Units
        panel's chosen object across an area, as one undo step. The area is
        the committed Select region or, with none, the Units-mode selection's
        bounding box grown by the dialog's Grow (GH #196); the dialog picks.

        The object and the default owner come from the Units panel, which is
        a persistent widget whose readers work in every mode, so a region
        needs no mode gate (see _update_tool_enabled). The dialog's other mode
        moves the Units-mode selection instead (GH #105,
        _scatter_selected_units).
        """
        if self.scenario is None:
            return
        object_id = self.units_panel.selected_object_const()
        selected = self._selected_unit_entries()
        mm = self.scenario.map_manager
        if self._region is None and units_region([e.unit for e in selected], mm.map_width, mm.map_height) is None:
            self._log_status(
                "Scatter: no area yet. Drag one with Select in Terrain mode, "
                "or select units in Units mode to scatter around them"
            )
            return
        if object_id is None and not selected:
            # Place Unit's own rule, so a scatter with nothing chosen says so
            # rather than opening a dialog that cannot be completed.
            self._log_status("Scatter: choose an object in the Units panel first, or select units to move")
            return
        # GH #164: a random pick scatters a mix, rolled per copy; object_id is its rolled const.
        choices = self.units_panel.selected_object_choices()
        random_label = self.units_panel.selected_object_label() if choices and len(choices) > 1 else None
        dialog = ScatterDialog(
            self.scenario,
            self._region,
            object_id,
            self.units_panel.owner_id(),
            parent=self,
            last=self._scatter_defaults,
            labels=self._player_labels,
            colors=self.scenario.player_colors,
            selected_units=[e.unit for e in selected],
            object_label=random_label,
            unit_consts=choices if random_label is not None else None,
        )
        # Before _ensure_unit_edits(), so cancelling never pays for the lazy
        # unit-model build.
        if dialog.exec_() != QDialog.Accepted:
            return
        self._scatter_defaults = dialog.state()
        params = dialog.params()
        if params.move_selected:
            self._scatter_selected_units(selected, params)
            return
        if params.unit_const is None:
            self._log_status("Scatter: choose an object in the Units panel first")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        player = unit_kind.placement_owner(params.unit_const, params.player)
        requested = params.requested(len(params.tiles))
        name = random_label if params.unit_consts else _unit_name(params.unit_const)
        # The dialog already subtracted occupied tiles, hence avoid_occupied=False.
        specs = scatter.plan_scatter_units(
            self.scenario,
            params.tiles,
            params.unit_consts or params.unit_const,
            count=params.count,
            density=params.density,
            seed=params.seed,
            jitter=params.jitter,
            min_spacing=params.min_spacing,
            avoid_occupied=False,
        )
        if not specs:
            # Every multi-tile pick overhung the eligible tiles (TASK-277): no edit, so no empty undo step.
            self._log_status(f"Scatter: no {name} fits the eligible tiles, so nothing was placed")
            return
        # splices=None, so one wholesale _after_unit_mutation() for the whole
        # scatter, the same trade the wall run makes.
        with self._unit_edit(model, "Scatter units", [player]):
            placed = model.add_many(player, specs)
            # Inside the block, so the exit reconciles the selection once.
            # Only in Units mode: elsewhere the placed units are left unselected,
            # and a kept selection is left as it was (TASK-359).
            if self.mode == "units":
                self._selection = [unit_pick.unit_key(player, unit) for unit in placed]
        owner_text = "GAIA" if player == GAIA_PLAYER_ID else f"Player {player}"
        status = f"Scattered {len(placed)} x {name} for {owner_text}"
        if requested is not None and len(placed) < requested:
            status += f" (asked for {requested}; clamped by eligible tiles, spacing or object size)"
        if player != params.player:
            status += _GAIA_ONLY_PLACED_SUFFIX
        # on_unit_place()'s clause: a scatter the live filter hides looks
        # exactly like the action doing nothing.
        if placed and not self._unit_filter.matches(player, placed[0]):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)

    def _selected_unit_entries(self) -> list:
        """The unit selection as index entries, sorted by unit key so a
        seeded scatter does not depend on click order. Empty in a mode where
        the selection is dormant (TASK-359)."""
        index = self.map_view._unit_index
        if not self._selection_live() or index is None:
            return []
        entries = [e for e in (index.entry_for_key(k) for k in self._selection) if e is not None]
        return sorted(entries, key=lambda e: unit_pick.unit_key(e.player_id, e.unit))

    def _scatter_selected_units(self, entries, params) -> None:
        """Scatter's move mode (GH #105): every selected unit to a random
        eligible tile of the area (the Select region or the selection's grown
        box), in one undo record. Only x/y change
        (set_position, z passed through), the same write a group move makes,
        plus the wall shapes that move re-derives, as a group move's (GH #266,
        with no gate part).

        A host's garrison rides to the host's new point (GH #42), and a
        selected occupant rides with its host rather than scattering alone."""
        if not entries:
            self._log_status("Scatter: select the units to move in Units mode first")
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        riders = {id(e.unit): self._garrison_occupant_entries(model, [e]) for e in entries}
        rider_ids = {id(r.unit) for rows in riders.values() for r in rows}
        movers = [e for e in entries if id(e.unit) not in rider_ids]
        plan = scatter.scatter_existing_units(
            self.scenario, [e.unit for e in movers], params.tiles, seed=params.seed, jitter=params.jitter
        )
        moves = [(e, pos) for e, pos in zip(movers, plan.positions, strict=True) if pos is not None]
        stayed = len(movers) - len(moves)
        if not moves:
            # Nothing changes, so no _unit_edit: entering one would record a phantom undo step.
            self._log_status(f"Scatter: none of the {len(movers)} selected units fits in the area; nothing moved")
            return
        touched = [e for e, _ in moves] + [r for e, _ in moves for r in riders[id(e.unit)]]
        writes = []
        moved_ids: set[int] = set()
        for entry, (x, y) in moves:
            for row in [entry, *riders[id(entry.unit)]]:
                if id(row.unit) in moved_ids:
                    continue
                moved_ids.add(id(row.unit))
                writes.append((row, x, y))
        wall_plan = self._move_plan(model, writes, gate_walls=False)
        players = sorted({e.player_id for e in touched} | (wall_run.move_plan_players(wall_plan) if wall_plan else set()))
        wholesale = wall_plan is not None and wall_plan.wholesale
        splices: list[UnitSplice] = []
        with self._unit_edit(
            model, "Scatter selected units", players, None if wholesale else splices, fields_only=not wholesale
        ):
            for row, x, y in writes:
                self._move_occupant_to(model, row, x, y, splices)
            if wholesale:
                wall_run.apply_move_plan(model, wall_plan)
            elif wall_plan:
                self._apply_wall_reshapes(model, wall_plan, splices)
        status = f"Scattered {len(moves)} selected units across the area"
        status += self._gate_move_clauses(wall_plan, [row.unit for row, _x, _y in writes])
        if plan.reused:
            status += f" ({plan.reused} share a tile: the area has too few free tiles)"
        if stayed:
            status += f" ({stayed} did not fit and stayed put)"
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
        self._focus_map_view()

    def _arm_wall(self, const: int) -> None:
        """GH #125's wall button, menu and keybind: pick `const` in the Units
        catalog and switch to Place Unit, unless a wall tool is already active."""
        if not self.place_unit_action.isEnabled():
            self._log_status("Wall: Place Unit is unavailable here")
            return
        if not self.units_panel.pick_object(const):
            self._log_status(f"Wall: could not pick {_unit_name(const)} in the catalog", level=debug_log.WARNING)
            return
        if self._current_tool not in _WALL_TOOLS:
            self.place_unit_action.setChecked(True)
        self._set_last_wall(const)

    def _set_last_wall(self, const: int) -> None:
        self._last_wall_const = const
        self._refresh_wall_pick_button()

    def _refresh_wall_menu(self) -> None:
        """The wall button's menu: the active dataset's wall family, less what its
        editor hides (GH #67: none in AoE2; 119, 370, 788 and 1062 in Return of
        Rome). The last-used wall falls back to the default when it is gone."""
        listed = {entry.id for entry in object_catalog.objects() if not entry.hidden}
        consts = [const for const in unit_sprites.wall_family_consts() if const in listed]
        menu = self.wall_pick_button.menu()
        menu.clear()
        for const in consts:
            action = menu.addAction(_unit_name(const))
            action.setData(const)
            action.triggered.connect(lambda _=False, c=const: self._arm_wall(c))
        if self._last_wall_const not in consts:
            preferred = settings.get_tool_option_default("last_wall")
            self._last_wall_const = preferred if preferred in consts else _DEFAULT_WALL_CONST
        self._refresh_wall_pick_button()

    def _refresh_unit_tool_menus(self) -> None:
        """GH #67: the wall menu and the cliff families follow the active game dataset."""
        self._refresh_wall_menu()
        self._populate_cliff_families()

    def _refresh_wall_pick_button(self) -> None:
        # _unit_name(), as the wall-run status line: object_name() reads "WALL2" with no install.
        self.wall_pick_button.setText(f"Wall: {_unit_name(self._last_wall_const)}")

    def _populate_cliff_families(self) -> None:
        """Fills the Family combo, at construction and on every game dataset
        change (GH #67: Return of Rome has no Short Marble). Keeps the picked
        family and piece when the new lists still have them."""
        current = self.cliff_family_combo.currentData()
        piece = self.cliff_piece_combo.currentData()
        self._cliff_families = cliff_catalog.families()
        self.cliff_family_combo.blockSignals(True)
        self.cliff_family_combo.clear()
        for file_name in sorted(self._cliff_families, key=cliff_catalog.family_label):
            self.cliff_family_combo.addItem(cliff_catalog.family_label(file_name), file_name)
        self.cliff_family_combo.setCurrentIndex(max(0, self.cliff_family_combo.findData(current)))
        self.cliff_family_combo.blockSignals(False)
        self._on_cliff_family_changed(self.cliff_family_combo.currentIndex())
        if piece is not None and self.cliff_piece_combo.findData(piece) >= 0:
            self.cliff_piece_combo.setCurrentIndex(self.cliff_piece_combo.findData(piece))

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
        self.map_view.refresh_cliff_ghost()

    # -- Cliff tool (Track B Stage 1 + 2) ---------------------------------
    #
    # Accumulate-then-commit, exactly like the Convert brush above and for
    # the same two reasons: UnitEditModel.begin_unit_edit(players) wants the
    # complete touched-player set up front, and here a node's own piece
    # isn't even decidable mid-drag (its neighbour set isn't complete until
    # the node after it exists). One undo record per stroke either way.
    #
    # A press-release without moving is a one-tile stroke, so Stage 1's
    # click-to-place gesture survives the widening: a lone node places the
    # Piece/Frame picker's own piece and frame, unless it touches an
    # existing cliff, whose contact shapes it (and may reshape that cliff).
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
        # Elevation and style can change between strokes, never during one.
        self._cliff_ghost_parts = {}
        if self.scenario is None:
            return
        unit_const = self.cliff_piece_combo.currentData()
        if unit_const is None:
            self._log_status("Cliff: choose a family and piece first")
            return
        mm = self.scenario.map_manager
        self._cliff_stroke_existing, index = self._cliff_existing()
        # A join rewrites live units: a mid-drag undo or document swap would leave them stale.
        # unit_gen restarts per loaded file, so the scenario's identity is kept too.
        self._cliff_stroke_gen = self.scenario.unit_gen
        self._cliff_stroke_scenario = weakref.ref(self.scenario)
        self._cliff_stroke = cliff_chain.ChainStroke(
            map_width=mm.map_width,
            map_height=mm.map_height,
            existing=self._cliff_stroke_existing,
            fallback_const=unit_const,
            fallback_frame=self.cliff_frame_spin.value(),
            existing_index=index,
        )

    def _cliff_existing(self) -> tuple[list, object]:
        """The map's cliffs and their tile index, scanned once per
        (unit_gen, map size): the hover builds a stroke per tile it crosses,
        and a scan each would cost a pass over every unit."""
        scenario = self.scenario
        mm = scenario.map_manager
        key = (scenario.unit_gen, mm.map_width, mm.map_height)
        memo = self._cliff_existing_memo
        if memo is None or memo[0]() is not scenario or memo[1] != key:
            existing = cliff_chain.existing_cliffs(scenario, mm.map_width, mm.map_height)
            memo = self._cliff_existing_memo = (weakref.ref(scenario), key, existing, cliff_chain.index_cliffs(existing))
        return memo[2], memo[3]

    def _cliff_stroke_tile(self, x: int, y: int) -> None:
        if self._cliff_stroke is not None:
            self._cliff_stroke.add_cursor_tile(x, y)

    def _cliff_ghost(self, unit_const: int, rotation: int, x: float, y: float) -> tuple[list, tuple | None]:
        """(sprite draws, mark) of one cliff a release would place, on GAIA."""
        ghost = new_unit(
            self.scenario._scenario.uuid, GAIA_PLAYER_ID, unit_const, x, y, reference_id=-1,
            rotation=float(rotation), initial_animation_frame=rotation,
        )
        draws = self._ghost_sprite_draws(GAIA_PLAYER_ID, ghost)
        return draws, (None if draws else self._ghost_mark(GAIA_PLAYER_ID, ghost))

    def _show_cliff_run(self, stroke: cliff_chain.ChainStroke, existing: list, parts: dict) -> None:
        """Ghosts what a release would do right now: every node of the run,
        and each existing cliff a join reshapes at its new frame (same const,
        same place). Memoized per ghost in `parts`, and the highlight follows
        the run's head."""
        keys = [(node.unit_const, node.rotation, node.x, node.y) for node in stroke.nodes()]
        for rewrite in stroke.rewrites():
            cliff = existing[rewrite.existing_index]
            if cliff.unit is not None:
                keys.append((cliff.unit_const, rewrite.rotation, cliff.unit.x, cliff.unit.y))
        draws: list = []
        marks: list = []
        for key in keys:
            found = parts.get(key)
            if found is None:
                found = parts[key] = self._cliff_ghost(*key)
            if found[0]:
                draws.extend(found[0])
            elif found[1] is not None:
                marks.append(found[1])
        self.map_view.set_unit_ghosts(draws=draws, marks=marks)
        head = stroke.head_bounds()
        self.map_view.set_cliff_footprint(
            None if head is None else [(x, y) for x in range(head[0], head[1]) for y in range(head[2], head[3])]
        )

    def on_cliff_preview(self, tile) -> None:
        """MapView's Cliff hover hook: what a click on `tile` would do, as a
        one-tile stroke resolves it. That is the picked piece centred on the
        tile, or the piece and frame a neighbouring cliff makes of it, plus
        the new frame of an end cap it joins. Nothing on an existing cliff (a
        press there starts a run, a click places nothing) or off the map."""
        view = self.map_view
        const = None if self.scenario is None or tile is None else self.cliff_piece_combo.currentData()
        if const is None:
            self._cliff_hover_memo = None
            view.set_cliff_footprint(None)
            view.set_unit_ghost()
            return
        frame = self.cliff_frame_spin.value()
        look = (
            self._sprites_enabled, view._terrain_style, id(view._iso_elevations),
            self.scenario.player_colors[GAIA_PLAYER_ID],
        )
        key = (tile, const, frame, self.scenario.unit_gen, *look)
        memo = self._cliff_hover_memo
        # The empty result is memoized too: crossing a cliff rebuilds no stroke per pixel.
        if memo is not None and memo[0] == key and memo[1] is view._unit_ghost_item:
            return
        if look != self._cliff_hover_look or len(self._cliff_hover_parts) > _CLIFF_HOVER_PARTS_MAX:
            self._cliff_hover_look = look
            self._cliff_hover_parts = {}
        mm = self.scenario.map_manager
        existing, index = self._cliff_existing()
        stroke = cliff_chain.ChainStroke(
            map_width=mm.map_width,
            map_height=mm.map_height,
            existing=existing,
            fallback_const=const,
            fallback_frame=frame,
            existing_index=index,
        )
        stroke.add_cursor_tile(*tile)
        self._show_cliff_run(stroke, existing, self._cliff_hover_parts)
        self._cliff_hover_memo = (key, view._unit_ghost_item)

    def _end_cliff_stroke(self) -> None:
        """Commits the whole run through the same UnitEditModel.add() write
        path Place Unit uses.

        Always GAIA (status quo for every real cliff, and the AGENTS.md hard
        rule on GAIA rotation being a variant index, not an angle, is exactly
        why `rotation` is passed through verbatim below rather than via
        set_rotation(), which raises for a variant const by design).

        An existing end cap or lone piece the run joins gets its new frame in
        the same edit, through set_cliff_frame() (AGENTS.md's Cliff drag
        exception): one undo record for the run and its joins.
        """
        stroke = self._cliff_stroke
        existing = self._cliff_stroke_existing
        self._cliff_stroke = None
        self._cliff_stroke_existing = []
        self.map_view.set_unit_ghost()
        self.map_view.set_cliff_footprint(None)
        self._cliff_ghost_parts = {}
        # The stroke's ghost replaced the hover's: its memo no longer says what is shown.
        self._cliff_hover_memo = None
        if stroke is None or self.scenario is None:
            return
        try:
            self._commit_cliff_stroke(stroke, existing)
        finally:
            # The hover ghost comes back at the cursor without a mouse move.
            self.map_view.refresh_cliff_ghost()

    def _commit_cliff_stroke(self, stroke: cliff_chain.ChainStroke, existing: list) -> None:
        pressed_on = self._cliff_stroke_scenario
        if pressed_on is None or pressed_on() is not self.scenario or self.scenario.unit_gen != self._cliff_stroke_gen:
            # An undo, redo or document swap mid-drag: the stroke's view of
            # the map's cliffs is stale, and a rewrite would raise on it.
            self._log_status("Cliff: the map's units changed during the drag, so nothing was placed")
            return
        nodes = stroke.nodes()
        if not nodes:
            return
        model = self._ensure_unit_edits()
        if model is None:
            return
        rewrites = [(existing[r.existing_index], r.rotation) for r in stroke.rewrites()]
        rewrites = [(cliff, frame) for cliff, frame in rewrites if cliff.unit is not None]
        label = "Place cliff" if len(nodes) == 1 else f"Place {len(nodes)} cliffs"
        # Selection is deliberately left untouched, unlike on_unit_place():
        # Terrain mode has no selection-visual concept, and _after_unit_
        # mutation() only rebuilds the pick index _selection resolves
        # against when self.mode == "units" (see its own docstring) -- so
        # pointing _selection at a placed cliff here would set it against
        # an index that this mutation doesn't rebuild.
        # TASK-278: the trees under the new cliffs go, bar one a garrison names (remove_many() would refuse it).
        trees = cliff_chain.trees_under(self.scenario, [node.bounds for node in nodes])
        trees = [(player, unit) for player, unit in trees if not model.referencing(unit)]
        players = sorted({GAIA_PLAYER_ID, *(cliff.player for cliff, _ in rewrites), *(player for player, _ in trees)})
        with self._unit_edit(model, label, players):
            # Rewrites first: an abort does not roll back, so a raise there must precede every add.
            for cliff, frame in rewrites:
                model.set_cliff_frame(cliff.unit, frame)
            # The garrison filter above is why this cannot raise after a rewrite.
            if trees:
                model.remove_many([unit for _player, unit in trees])
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
        # re-resolving the name a second, worse way. A run's pieces come from
        # the drag direction and their neighbours, so only a lone node still
        # on the combo's own piece is named.
        if len(nodes) == 1 and nodes[0].unit_const == self.cliff_piece_combo.currentData():
            status = f"Placed {self.cliff_piece_combo.currentText()} at ({nodes[0].x:g}, {nodes[0].y:g})"
        else:
            status = (
                "Placed a cliff from the picked family, shaped to join the cliff beside it"
                if len(nodes) == 1
                else f"Placed a run of {len(nodes)} cliffs from the picked family, "
                "shaped by the drag direction and their neighbours"
            )
            unresolved = sum(1 for node in nodes if not node.resolved)
            if unresolved:
                status += f" ({unresolved} at a junction with no matching piece)"
        if rewrites:
            status += f", {len(rewrites)} existing {'cliff' if len(rewrites) == 1 else 'cliffs'} reshaped to join"
        if trees:
            status += f", removed {len(trees)} tree(s) under {'it' if len(nodes) == 1 else 'them'}"
        unjoined = stroke.unjoined()
        if unjoined:
            left = "1 left as it was" if unjoined == 1 else f"{unjoined} left as they were"
            status += f", {left} (a join there needs a different-size piece)"
        # A5's own known gap, surfaced here rather than only in the tooltip --
        # placing a cliff with GAIA hidden looks exactly like the tool doing
        # nothing.
        if not self.show_gaia_action.isChecked():
            status += " (Show GAIA is off, so it won't be visible)"
        self._log_status(status)

    # -- Place Unit's wall branch (GH #98; was the Wall Run tool) and the
    # -- wall branch of Unit Line and Unit Shape (GH #365) -----------------
    #
    # A drag_shape path, not a stroke: MapView owns the rubber band and calls
    # on_shape_commit() once at release with the exact tile path, so there is
    # no per-tile half here at all. That matters beyond tidiness -- a wall's
    # variant index is a function of its neighbours, and a node's neighbour
    # set is not complete until the path is.

    def _place_shape(self) -> str:
        """MapView's place-shape query: "wall" when the Units catalog's
        picked const is one of the 9 wall-family consts, else ""."""
        return "wall" if self.units_panel.selected_object_const() in _wall_consts() else ""

    def _blocked_for_walls(self) -> frozenset[tuple[int, int]]:
        """GH #124: the tiles a wall run must leave empty, or none with
        "Walls skip occupied tiles" off. Also MapView's blocked query."""
        if self.scenario is None or not self.skip_occupied_check.isChecked():
            return frozenset()
        mm = self.scenario.map_manager
        return frozenset(wall_run.blocked_tiles(self.scenario, mm.map_width, mm.map_height))

    def _shape_press(self) -> unit_shape.ShapeLatch | None:
        """MapView's press query under Unit Line and Unit Shape (GH #365): the
        pick and the options read once, as the latch MapView holds for the
        drag and hands back at release, or None (after logging why) for a
        pick the tools refuse, so no drag starts."""
        tool_name = _TOOL_LABELS.get(self._current_tool, "Unit Shape")
        if self.scenario is None:
            return None
        consts = tuple(self.units_panel.selected_object_choices() or ())
        reason = unit_shape.refusal(consts)
        if reason is not None:
            self._log_status(f"{tool_name}: {reason}")
            return None
        kind = unit_shape.LINE if self._current_tool == "unit_line" else self.unit_shape_kind_combo.currentData()
        latch = unit_shape.latch_for(
            consts,
            self.units_panel.owner_id(),
            kind,
            filled=self.unit_shape_filled_check.isChecked(),
            spacing=self.unit_shape_spacing_spin.value(),
            randomize=self.unit_shape_random_check.isChecked(),
            rules=not self._restrictions_lifted(),
            pending=self._place_rotation,
        )
        return dataclasses.replace(latch, label=self.units_panel.selected_object_label() or "")

    def _commit_wall_run(self, tiles, unit_const, owner: int | None = None) -> None:
        """Places a whole wall run (any tile list: a Place Unit drag or a
        Unit Shape outline) as one undo record, with `unit_const` from the
        Units catalog and `owner` the chosen owner (read now when None; a
        Unit Line or Unit Shape drag passes the one latched at its press).

        Follows _end_cliff_stroke()'s shape, with one structural difference
        forced by the junction rewrites: the touched-player set has to be
        complete BEFORE begin_unit_edit(), and it includes the owner of every
        pre-existing wall the run reshapes. add()/add_many() are refused
        inside a fields_only edit, so both halves run as one
        fields_only=False edit. wall_run.touched_players() is that set.

        GAIA is allowed: render_units.stored_rotation() keeps a GAIA wall's index
        (all 9 wall-family consts are rotation_is_variant), so it draws shaped.
        """
        tool_name = _TOOL_LABELS.get(self._current_tool, "Place Unit")
        # For Place Unit the pick can change mid-drag after the press latched "wall".
        if unit_const not in _wall_consts():
            self._log_status(f"{tool_name}: pick a wall in the catalog")
            return
        player = self.units_panel.owner_id() if owner is None else owner
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
            blocked_tiles=self._blocked_for_walls(),
        )
        if not plan.nodes and not plan.rewrites:
            if not plan.blocked:
                reason = f"those {plan.skipped} tiles already hold a wall"
            elif not plan.skipped:
                reason = f"those {plan.blocked} tiles are occupied"
            else:
                reason = f"those tiles already hold a wall ({plan.skipped}) or are occupied ({plan.blocked})"
            self._log_status(f"{tool_name}: {reason}")
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
        if plan.blocked:
            status += f" ({plan.blocked} occupied tiles skipped)"
        # on_unit_place()'s clause: a run the live filter hides (Show Walls
        # off) looks exactly like the tool doing nothing.
        if units and not self._unit_filter.matches(player, units[0]):
            status += " (a Filters toggle is hiding it, so it won't be visible)"
        self._log_status(status)
        self._set_last_wall(unit_const)
