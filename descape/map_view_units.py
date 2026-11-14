"""MapView's units: the unit index, stack badges, hover, selection, range rings, marquee, drag, ghosts and footprint outlines."""

from __future__ import annotations

from typing import ClassVar

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QGraphicsPathItem,
)

from descape import (
    game_dataset,
    outline_chunks,
    perf_trace,
    range_overlay,
    render_units,
    trigger_range,
    unit_kind,
    unit_pick,
)

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- the render modules import no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.unit_filter import GAIA_PLAYER_ID
from descape.viewer_canvas import (
    ChunkedPathItem,
    UnitGhostItem,
)
from descape.viewer_common import (
    TOOL_MOVE,
    TOOL_SELECT,
)


def _add_closed_polygons(path, polygons) -> None:
    """Appends each polygon to `path` as its own CLOSED subpath.
    addPolygon() leaves the subpath open, so stroking it draws only 3 of a
    diamond's 4 edges -- invisible wherever a fill dominates, and plain wrong
    for an outline-only cue."""
    for points in polygons:
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        path.closeSubpath()


class _MapViewUnitsMixin:
    """MapView's unit display, selection and drag (map_view.py's docstring has the split's rules)."""

    def set_unit_index(self, index) -> None:
        """Swaps the pick index (a filter change rebuilds it) and drops any
        hover/selection keys that the new index can no longer resolve."""
        self._unit_index = index
        self._eyedrop_cycle = None  # the stacks it indexed are rebuilt below
        self.refresh_stack_groups()
        self._clear_unit_hover()
        self.refresh_unit_highlight()
        self._refresh_unit_outline_overlays()
        self.refresh_ring_layer()

    def refresh_drape_outlines(self) -> None:
        """For a Show sprites or Farm Overlay toggle: a draped farm's outline
        shape follows both (TASK-015), so the hover, the selection and the
        footprint outlines are redrawn (the fish rings with them, on the same
        geometry). The hover memo is dropped too, or a cursor resting on the
        farm keeps its old path."""
        self._clear_unit_hover()
        self.refresh_unit_highlight()
        self._refresh_unit_outline_overlays()

    def refresh_after_index_patch(self) -> None:
        """For a caller that patched _unit_index in place rather than
        swapping it: everything derived from the index, which is the stacks,
        the footprint and fish outlines and the always-on ring layer."""
        self.refresh_stack_groups()
        self._refresh_unit_outline_overlays()
        self.refresh_ring_layer()

    def refresh_stack_groups(self) -> None:
        """Recomputes the stacks from the current index -- for a caller that
        patched that index in place rather than swapping it.

        Skips the badge rebuild (~8 ms in Sloped on old-allies) when every
        stack tile and member count is unchanged and the badges are live: an
        anchor depends only on its tile and elevation, and a unit edit changes
        neither. refresh_elevation_overlays() rebuilds them unconditionally."""
        old_counts = {tile: len(members) for tile, members in self._stack_groups.items()}
        self._stack_groups = {} if self._unit_index is None else unit_pick.stack_groups(self._unit_index)
        new_counts = {tile: len(members) for tile, members in self._stack_groups.items()}
        if new_counts == old_counts and not self._stack_badges_stale:
            return
        self._rebuild_stack_badges()

    def stack_group_at(self, tile: tuple[int, int]) -> list | None:
        """The stack members (top-first) anchored at `tile`, or None."""
        return self._stack_groups.get(tile)

    def _stack_badges_shown(self) -> bool:
        return self._stack_badges_enabled and self._mode == "units" and bool(self._stack_groups)

    def _rebuild_stack_badges(self) -> None:
        """Skipped while hidden (Sloped anchors cost ~0.1 ms per stack, per elevation
        touch); _sync_stack_badges_visible rebuilds on show."""
        if self._stack_badge_item is None:
            return
        if not self._stack_badges_shown():
            self._stack_badges_stale = True
            self._stack_badge_item.setVisible(False)
            return
        self._stack_badges_stale = False
        badges = []
        for (tx, ty), members in self._stack_groups.items():
            anchors = self._stack_badge_anchor(tx, ty)
            if anchors is None:
                continue
            badges.append((anchors[0], anchors[1], len(members)))
        self._stack_badge_item.set_placement(self._stack_badge_position != "above")
        self._stack_badge_item.set_badges(badges)
        self._sync_stack_badges_visible()

    # GH #100: position id -> the _tile_edge_points side it tucks against.
    # The sides are grid-neighbour names that are also the iso screen quadrants.
    _STACK_BADGE_SIDES: ClassVar[dict[str, str]] = {
        "bottom_right": "right", "bottom_left": "left", "top_right": "up_right", "top_left": "up_left",
    }

    def _stack_badge_anchor(self, tile_x: int, tile_y: int) -> tuple[QPointF, QPointF] | None:
        """(anchor, centre) in scene coords for a badge on this tile. Iso on
        screen: the named side's midpoint (midpoints survive Flat-iso's linear
        view transform). Square: the side's first point, its named corner.
        `above`: the screen top vertex in iso, the top-edge centre in square."""
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return None
        centre = polygon.boundingRect().center()
        iso = self._isometric or self._terrain_style in _ELEVATED_STYLES
        position = self._stack_badge_position
        if position == "above":
            points = self._tile_edge_points(tile_x, tile_y, "up_left")
            if points is None:
                return None
            anchor = QPointF(points[-1]) if iso else (points[0] + points[-1]) / 2.0
            return anchor, centre
        points = self._tile_edge_points(tile_x, tile_y, self._STACK_BADGE_SIDES[position])
        if points is None:
            return None
        anchor = (points[0] + points[-1]) / 2.0 if iso else QPointF(points[0])
        return anchor, centre

    def set_stack_badge_position(self, position: str) -> None:
        """Settings > Appearance's badge position: stores it and re-anchors the live item."""
        self._stack_badge_position = position
        self._rebuild_stack_badges()

    def _sync_stack_badges_visible(self) -> None:
        if self._stack_badge_item is None:
            return
        shown = self._stack_badges_shown()
        if shown and self._stack_badges_stale:
            # Rebuilds, then calls back here with the flag cleared.
            self._rebuild_stack_badges()
            return
        self._stack_badge_item.setVisible(shown)

    def set_stack_badges(self, enabled: bool) -> None:
        """Shows or hides the stacked-unit badges. Nothing is baked into
        chunk pixels, so there is nothing to evict."""
        self._stack_badges_enabled = enabled
        self._sync_stack_badges_visible()

    def set_team_colors(self, team_colors) -> None:
        """Stores each player's unit_sprites.TEAM_COLORS tint, indexed by
        player_id, or None: the owner colour of the selection (TASK-115),
        footprint outlines and range rings (TASK-449). Stores only: callers
        redraw explicitly, via set_unit_index(), refresh_unit_highlight() and
        the refresh_*_owner_colors() pair."""
        self._team_colors = None if team_colors is None else tuple(team_colors)

    def set_selection_by_owner(self, enabled: bool) -> None:
        """View > Colour Selection by Owner; redraws a live selection."""
        self._selection_by_owner = enabled
        self.refresh_unit_highlight()

    def _unit_polygons_for(self, entry):
        """One place decides which elevations and corner_rise a unit's
        geometry is read from, so the hover cue and the footprint overlay can
        never outline against different height fields."""
        cache = self._sloped_cache()
        return unit_pick.unit_polygons(
            entry,
            self._terrain_style,
            self._tile_pixels or 1,
            self._map_width or 0,
            self._map_height or 0,
            self._iso_elevations,
            self._iso_proj,
            corner_rise=None if cache is None else cache.corner_rise,
            farms_draped=self._farms_draped(),
            fold=None if cache is None else cache.fold,
        )

    def _unit_path(self, entry) -> QPainterPath | None:
        polygons = self._unit_polygons_for(entry)
        if not polygons:
            return None
        path = QPainterPath()
        _add_closed_polygons(path, polygons)
        return path

    def _update_unit_hover(self, entry) -> None:
        key = unit_pick.unit_key(entry.player_id, entry.unit)
        if key == self._unit_hover_key:
            return
        self._unit_hover_key = key
        path = self._unit_path(entry)
        if path is None:
            self._clear_unit_hover()
            return
        if self._unit_hover_item is None:
            self._unit_hover_item = self.scene().addPath(path, self._unit_hover_pen)
            self._unit_hover_item.setZValue(self.UNIT_HOVER_Z)
        else:
            self._unit_hover_item.setPath(path)

    def set_unit_drop_target(self, entry) -> None:
        """GH #115: outline the host a unit drag would garrison into, or clear
        it with None. The hover outline, which a drag otherwise leaves empty."""
        if entry is None:
            self._clear_unit_hover()
        else:
            self._update_unit_hover(entry)

    def _clear_unit_hover(self) -> None:
        self._unit_hover_key = None
        if self._unit_hover_item is not None:
            self.scene().removeItem(self._unit_hover_item)
            self._unit_hover_item = None

    def set_unit_selection(self, entries) -> None:
        """entries: an iterable of UnitEntry (b2.2 generalizes this from a
        single entry to N), or None/empty to clear. ViewerWindow's
        self._selection is the authoritative list of keys; this is purely
        "here is what to draw right now" -- MapView keeps its own key list
        only so refresh_unit_highlight() can re-resolve after an index swap
        with no fresh push from the caller."""
        if not entries:
            self._clear_unit_selection()
            return
        self._unit_select_keys = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
        self._draw_unit_selection(entries)

    def _owner_group_key(self, player_id: int, by_owner: bool):
        """An owner-coloured overlay's group key: the owner's sprite tint, so
        the selection, footprint outlines and range rings match the unit they
        mark (TASK-115, TASK-449), or None for the configured colour (`by_owner`
        off, GAIA, or no colours pushed). Each overlay has its own toggle."""
        colors = self._team_colors
        if not by_owner or colors is None or player_id == GAIA_PLAYER_ID:
            return None
        if not 0 < player_id < len(colors):
            return None
        return tuple(colors[player_id])

    def _selection_group_key(self, player_id: int):
        return self._owner_group_key(player_id, self._selection_by_owner)

    def _selection_ink(self, key) -> tuple[QPen, QColor]:
        """(outline pen, fill colour) for one selection group."""
        if key is None:
            return self._unit_select_pen, self._unit_select_fill_color
        pen = QPen(QColor(*key), self.UNIT_SELECT_PEN_WIDTH)
        fill = QColor(*key)
        fill.setAlpha(self.UNIT_SELECT_FILL_ALPHA)
        return pen, fill

    def _add_selection_under_item(self, path: QPainterPath) -> QGraphicsPathItem:
        item = self.scene().addPath(
            path, QPen(QColor(*self.UNIT_SELECT_UNDERSTROKE_RGBA), self.UNIT_SELECT_UNDERSTROKE_WIDTH)
        )
        item.setZValue(self.UNIT_SELECT_UNDER_Z)
        # Here, not at the caller: a live group gains this when Colour Selection by Owner turns on.
        item.setOpacity(self._overlay_opacity("unit_select"))
        return item

    def _draw_unit_selection(self, entries) -> None:
        """One unioned QPainterPath per owner colour group rather than one
        item set per unit, so the cost is bounded by the owner count (at most
        9), not the selection size. Each group is a fill, an under-stroke
        (only while colouring by owner) and an outline, on the explicit Z
        sub-layers above. Groups are updated in place, added, or removed."""
        paths: dict = {}
        for entry in entries:
            entry_path = self._unit_path(entry)
            if entry_path is not None:
                paths.setdefault(self._selection_group_key(entry.player_id), QPainterPath()).addPath(entry_path)
        if not paths:
            self._clear_unit_selection()
            return
        want_under = self._selection_by_owner
        for key in [k for k in self._unit_select_groups if k not in paths]:
            self._remove_selection_group(key)
        for key, path in paths.items():
            group = self._unit_select_groups.get(key)
            if group is None:
                pen, fill = self._selection_ink(key)
                fill_item = self.scene().addPath(path, QPen(Qt.NoPen), QBrush(fill))
                fill_item.setZValue(self.UNIT_SELECT_Z)
                under_item = self._add_selection_under_item(path) if want_under else None
                outline_item = self.scene().addPath(path, pen)
                outline_item.setZValue(self.UNIT_SELECT_OUTLINE_Z)
                opacity = self._overlay_opacity("unit_select")
                fill_item.setOpacity(opacity)
                outline_item.setOpacity(opacity)
                self._unit_select_groups[key] = (fill_item, under_item, outline_item)
                continue
            fill_item, under_item, outline_item = group
            fill_item.setPath(path)
            outline_item.setPath(path)
            if under_item is not None and not want_under:
                self.scene().removeItem(under_item)
                under_item = None
            elif under_item is None and want_under:
                under_item = self._add_selection_under_item(path)
            elif under_item is not None:
                under_item.setPath(path)
            self._unit_select_groups[key] = (fill_item, under_item, outline_item)
        # Hung off the selection rather than off ViewerWindow, so the rings
        # inherit index-swap refresh and mode-change teardown for free.
        self._draw_unit_range_rings(entries)

    def _remove_selection_group(self, key) -> None:
        for item in self._unit_select_groups.pop(key):
            if item is not None:
                self.scene().removeItem(item)

    def _clear_unit_selection(self) -> None:
        self._unit_select_keys = []
        for key in list(self._unit_select_groups):
            self._remove_selection_group(key)
        self._clear_unit_range_rings()

    def _range_ring_rise_px(self, entry) -> int | None:
        """The canvas-pixel height the whole ring is DRAWN at, from the
        building's own tile rather than per sample -- unit_pick.unit_polygons'
        asymmetry-2 rule, a multi-tile building being a flat slab at one
        height. The ring's DISTANCE is still ground-plane (see
        range_overlay's header).

        Sloped reads the same corner_rise the pixels were painted from; on a
        cache miss it falls back to the Stepped-style rise rather than
        dropping the ring."""
        cache = self._sloped_cache()
        if cache is not None:
            return unit_pick.unit_rise_px_for(entry, cache.corner_rise, cache.fold)
        if self._iso_elevations is None or self._iso_proj is None:
            return None
        return int(self._iso_elevations[entry.own_y, entry.own_x]) * self._iso_proj.elev_step

    def _range_ring_points(self, entry, radius: float | None):
        """One entry's ring polygon in scene space at `radius` tiles, or None
        when it draws no ring: no radius, or no geometry to project with.

        The centre is the drawn footprint's own (render_units.unit_footprint_centre),
        not the stored coordinate: the two differ by half a tile for a
        building stored off its span's parity, e.g. a 4x4 at `.5` (GH #200).
        The ring follows the footprint, by decision, wherever that is drawn.
        For a 1x1 such as a map revealer the two are the same point."""
        if radius is None:
            return None
        anchor = self._range_ring_anchor(entry)
        if anchor is None:
            return None
        return self._range_ring_points_at(radius, *anchor)

    def _range_ring_anchor(self, entry) -> tuple[float, float, int] | None:
        """(cx, cy, rise_px) one entry's ring is drawn from, rise 0 in Flat,
        or None with no geometry to project with."""
        cx, cy = render_units.unit_footprint_centre(entry.unit)
        if self._terrain_style == "flat":
            return cx, cy, 0
        if self._iso_proj is None:
            return None
        rise_px = self._range_ring_rise_px(entry)
        if rise_px is None:
            return None
        return cx, cy, rise_px

    def _range_ring_points_at(self, radius: float, cx: float, cy: float, rise_px: int):
        if self._terrain_style == "flat":
            return range_overlay.ring_points(cx, cy, radius, "flat", tile_px=self._tile_pixels or 1)
        return range_overlay.ring_points(cx, cy, radius, self._terrain_style, proj=self._iso_proj, rise_px=rise_px)

    def _draw_unit_range_rings(self, entries) -> None:
        """One QPainterPath per owner colour group unioning that group's
        qualifying rings, the shape _draw_unit_selection() already uses for N
        units. Colour by Owner off is the one configured group (GH #156).

        The ring draws OVER the sprites, unlike the game's own, which
        composites its circle into the terrain so a palisade occludes it.
        That is an accepted deviation: drawing under the sprites means
        painting into cached chunk pixels, a different feature.

        Trigger Changes (GH #49 slice 2) adds the dashed rings in their own
        items, evaluated per draw from the live entry against the pushed index,
        so a move out of an effect's area needs no extra push. Both path sets
        are built first and reconciled separately: a building with base range
        0 (a House) has a dashed ring and no solid one."""
        if not (self._range_rings_enabled or self._range_rings_selected_revealers):
            self._clear_unit_range_rings()
            return
        index = self._trigger_range_index if self._range_rings_enabled and self._trigger_rings_enabled else None
        paths: dict = {}
        trigger_paths: dict = {}
        for entry in entries:
            radius = range_overlay.selection_ring_radius(
                entry.unit.unit_const,
                buildings=self._range_rings_enabled,
                revealers=self._range_rings_selected_revealers,
            )
            key = self._owner_group_key(entry.player_id, self._range_rings_by_owner)
            points = self._range_ring_points(entry, radius)
            if points:
                _add_closed_polygons(paths.setdefault(key, QPainterPath()), [points])
            if index is None:
                continue
            radii = range_overlay.trigger_ring_radii(
                index,
                player_id=entry.player_id,
                unit_const=entry.unit.unit_const,
                reference_id=entry.unit.reference_id,
                own_tile=(entry.own_x, entry.own_y),
            )
            anchor = self._range_ring_anchor(entry) if radii else None
            if anchor is not None:
                rings = [self._range_ring_points_at(r, *anchor) for r in radii]
                _add_closed_polygons(trigger_paths.setdefault(key, QPainterPath()), rings)
        self._reconcile_ring_items(self._range_ring_items, paths, self._range_ring_pen, None)
        self._reconcile_ring_items(
            self._trigger_ring_items, trigger_paths, self._range_ring_trigger_pen, self.RANGE_RING_TRIGGER_DASH
        )

    def _reconcile_ring_items(self, items: dict, paths: dict, configured_pen: QPen, dash) -> None:
        """Brings one ring item dict (colour group -> item) in line with
        `paths`: drops groups no longer drawn, adds or re-paths the rest."""
        for key in [k for k in items if k not in paths]:
            self.scene().removeItem(items.pop(key))
        for key, path in paths.items():
            item = items.get(key)
            if item is None:
                # Cosmetic width 0 for an owner group too: see _range_ring_pen.
                pen = configured_pen
                if key is not None:
                    pen = QPen(QColor(*key), 0)
                    if dash is not None:
                        pen.setDashPattern(dash)
                item = items[key] = self.scene().addPath(path, pen)
                item.setZValue(self.RANGE_RING_Z)
            else:
                item.setPath(path)

    def _clear_unit_range_rings(self) -> None:
        for item in (*self._range_ring_items.values(), *self._trigger_ring_items.values()):
            self.scene().removeItem(item)
        self._range_ring_items = {}
        self._trigger_ring_items = {}

    def set_range_rings_triggers(self, enabled: bool) -> None:
        """View > Range Rings > Trigger Changes (GH #49 slice 2): the dashed
        trigger rings beside Selected Buildings' solid ones. Scene items only."""
        self._trigger_rings_enabled = enabled
        self.refresh_unit_highlight()

    def set_trigger_range_effects(self, effects) -> None:
        """The viewer's push of trigger_range.collect_range_effects() output,
        or None/UNREAD/UNREADABLE for none. The index is built once per new
        tuple, never per draw; the same object again is a no-op."""
        if effects is self._trigger_range_effects:
            return
        self._trigger_range_effects = effects
        self._trigger_range_index = trigger_range.index_effects(effects) if isinstance(effects, tuple) else None
        self.refresh_unit_highlight()

    def set_range_rings(self, enabled: bool) -> None:
        """Shows or hides View > Range Rings. Like set_footprint_outlines,
        this evicts no chunk cache: the ring is a scene item, never baked into
        canvas pixels. Redraws through the selection, which is the only thing
        a ring ever hangs off."""
        self._range_rings_enabled = enabled
        if not enabled:
            self._clear_unit_range_rings()
        self.refresh_unit_highlight()

    def set_range_rings_selected_revealers(self, enabled: bool) -> None:
        """View > Range Rings > Selected Map Revealers: the selection's rings
        also cover map revealers, at their line of sight. Scene items only."""
        self._range_rings_selected_revealers = enabled
        if not (enabled or self._range_rings_enabled):
            self._clear_unit_range_rings()
        self.refresh_unit_highlight()

    def set_range_rings_by_owner(self, enabled: bool) -> None:
        """GH #156: each ring in its owner's team colour, GAIA in the configured one.
        Applies to the always-on layer too."""
        self._range_rings_by_owner = enabled
        self.refresh_unit_highlight()
        self.refresh_ring_layer()

    def set_ring_layer_revealers(self, enabled: bool) -> None:
        """GH #153: View > Range Rings > Map Revealers. Like set_range_rings,
        this evicts no chunk cache: the rings are scene items only."""
        self._ring_layer_revealers = enabled
        self.refresh_ring_layer()

    def set_ring_layer_buildings(self, enabled: bool) -> None:
        """GH #155: View > Range Rings > All Buildings. Scene items only, as above."""
        self._ring_layer_buildings = enabled
        self.refresh_ring_layer()

    def refresh_ring_layer_owner_colors(self) -> None:
        """For a player colour edit, after set_team_colors(): only
        owner-coloured rings have anything to recolour."""
        if self._range_rings_by_owner:
            self.refresh_ring_layer()

    def _ring_layer_wanted(self) -> bool:
        return (self._ring_layer_revealers or self._ring_layer_buildings) and self._unit_index is not None

    def refresh_ring_layer(self) -> None:
        """Rebuilds the always-on rings (GH #153 / #155) from the live index,
        in every mode, the way refresh_footprint_overlay() does the outlines.
        Iterates index.entries, so it inherits the unit filter: Show Invisible
        Objects off hides the revealer rings.

        A selected building with All Buildings on also draws in the selection's
        own ring item: same path, same pen, identical pixels, not deduplicated.

        Each ring's polygon is kept between rebuilds, keyed on everything it
        depends on but the projection, which invalidates the whole set: this
        runs synchronously on every unit edit, and sampling 1300 rings from
        scratch measured ~275 ms against the footprint layer's 45-80 ms.
        Only rings still drawn are kept, so the set never outgrows the map."""
        if not self._ring_layer_wanted():
            self._clear_ring_layer()
            return
        projection = (self._terrain_style, self._tile_pixels, self._iso_proj)
        previous = self._ring_layer_polygons if projection == self._ring_layer_projection else {}
        polygons: dict = {}
        paths: dict = {}
        for entry in self._unit_index.entries:
            radius = range_overlay.persistent_ring_radius(
                entry.unit.unit_const, revealers=self._ring_layer_revealers, buildings=self._ring_layer_buildings
            )
            if radius is None:
                continue
            anchor = self._range_ring_anchor(entry)
            if anchor is None:
                continue
            ring_key = (radius, *anchor)
            polygon = polygons.get(ring_key) or previous.get(ring_key)
            if polygon is None:
                polygon = QPolygonF([QPointF(x, y) for x, y in self._range_ring_points_at(radius, *anchor)])
            polygons[ring_key] = polygon
            path = paths.setdefault(self._owner_group_key(entry.player_id, self._range_rings_by_owner), QPainterPath())
            path.addPolygon(polygon)
            path.closeSubpath()
        self._ring_layer_polygons = polygons
        self._ring_layer_projection = projection
        for key in [k for k in self._ring_layer_items if k not in paths]:
            self.scene().removeItem(self._ring_layer_items.pop(key))
        for key, path in paths.items():
            item = self._ring_layer_items.get(key)
            if item is None:
                # Cosmetic width 0 for an owner group too: see _range_ring_pen.
                pen = self._range_ring_pen if key is None else QPen(QColor(*key), 0)
                item = self._ring_layer_items[key] = self.scene().addPath(path, pen)
                item.setZValue(self.RANGE_RING_Z)
            else:
                item.setPath(path)

    def _clear_ring_layer(self) -> None:
        for item in self._ring_layer_items.values():
            self.scene().removeItem(item)
        self._ring_layer_items = {}
        self._ring_layer_polygons = {}

    def refresh_unit_highlight(self) -> None:
        """Re-entry point for state changing WITHOUT the mouse moving --
        mirroring refresh_highlight(tile). Two live cases: a filter toggle
        rebuilds the index (some, none or all of the selected units may
        survive it), and set_unit_index() itself calls this after every
        index swap.

        The elevation-edit case comes through _flush_footprint_refresh():
        the selection stays drawn in Terrain mode (TASK-359), so an
        elevation edit made there moves it, deferred to the event loop like
        the footprint outlines. Phase 3.5 gives this a further caller: a
        unit mutation."""
        if not self._unit_select_keys:
            return
        entries = []
        if self._unit_index is not None:
            for key in self._unit_select_keys:
                entry = self._unit_index.entry_for_key(key)
                if entry is not None:
                    entries.append(entry)
        if not entries:
            self._clear_unit_selection()
        else:
            self._unit_select_keys = [unit_pick.unit_key(e.player_id, e.unit) for e in entries]
            self._draw_unit_selection(entries)

    def _update_marquee(self, start_pos: QPointF, current_pos: QPointF) -> None:
        """b2.3's rubber-band rectangle, in DEVICE (widget) coordinates
        converted to scene space at each call -- the same mapToScene()
        every other MapView geometry method already uses, so the rectangle
        tracks correctly through pan/zoom mid-drag. Reuses the selection's
        own blue rather than inventing a third unit-mode colour."""
        rect = QRectF(self.mapToScene(start_pos), self.mapToScene(current_pos)).normalized()
        if self._marquee_item is None:
            self._marquee_item = self.scene().addRect(
                rect, self._unit_select_pen, QBrush(self._unit_select_fill_color)
            )
            self._marquee_item.setZValue(self.UNIT_SELECT_Z)
            self._marquee_item.setOpacity(self._overlay_opacity("unit_select"))
        else:
            self._marquee_item.setRect(rect)

    def _clear_marquee(self) -> None:
        if self._marquee_item is not None:
            self.scene().removeItem(self._marquee_item)
            self._marquee_item = None

    def set_unit_drag_hooks(self, on_move_press, is_group_drag) -> None:
        """The Move tool's two viewer hooks, a setter like set_place_shape_query:
        on_move_press(scene_pos, modifiers) returns the key a press drags (or
        None), and is_group_drag(key) says the drag moves a whole selection (GH #75)."""
        self._on_move_press = on_move_press
        self._is_group_drag = is_group_drag

    def set_unit_escape_hook(self, on_unit_escape) -> None:
        """Units Select's Escape (TASK-325): on_unit_escape() clears the unit
        selection and returns True, or returns False to fall through to the region."""
        self._on_unit_escape = on_unit_escape

    def set_lift_hooks(self, on_apply, on_revert) -> None:
        """A live flip or turn of a selection (TASK-256): on_apply() keeps it
        (Enter) and on_revert() undoes it (Escape), each returning True when
        it acted, or False to fall through."""
        self._on_lift_apply = on_apply
        self._on_lift_revert = on_revert

    def region_drag_live(self) -> bool:
        """Whether a Select-tool region drag or pasted-region move is mid-gesture."""
        return self._select_anchor is not None or self._move_anchor is not None

    def unit_drag_live(self) -> bool:
        """Whether a unit drag has passed the threshold and not ended: what the
        Rotate inputs turn from then on (GH #280). Keyed on state, not the tool,
        so a keybound tool switch mid-drag still turns the drag."""
        return self._unit_drag_key is not None and self._unit_drag_hover is not None

    def _unit_drag_moved(self, viewport_pos, modifiers) -> None:
        """One position of a unit drag: the preview, or nothing inside the
        threshold or off the map. The mouse and refresh_unit_drag_ghost() both
        come here, so a turn redraws exactly what the next move would."""
        press_pos = self._unit_drag_press_pos
        # Only once the drag is real -- the same threshold mouseReleaseEvent
        # uses to tell a click from a drag. A ghost appearing on 1px of jitter
        # during a plain selection click would read as a flicker bug.
        past = (
            press_pos is not None
            and (viewport_pos - press_pos).manhattanLength() > self.UNIT_DRAG_THRESHOLD_PX
        )
        if past or self._unit_drag_hover is not None:
            self._unit_drag_hover = (viewport_pos, modifiers)
        pos = self.mapToScene(viewport_pos)
        if past and (self._pos_on_map(pos) or self._is_group_drag(self._unit_drag_key)):
            self._on_unit_drag_preview(self._unit_drag_key, pos, modifiers)
        else:
            self._clear_unit_hover()
            self._clear_unit_ghost()

    def refresh_unit_drag_ghost(self) -> None:
        """Re-runs a live drag's last position (GH #280), through mapToScene
        again, since the view may have zoomed since."""
        if self.unit_drag_live():
            self._unit_drag_moved(*self._unit_drag_hover)

    def _units_select(self) -> bool:
        return self._mode == "units" and self._tool == TOOL_SELECT

    def _units_move(self) -> bool:
        return self._mode == "units" and self._tool == TOOL_MOVE

    def _marquee_engaged(self, viewport_pos, tile) -> bool:
        """Whether a Select-tool marquee is a rectangle yet rather than a click:
        past UNIT_DRAG_THRESHOLD_PX and, for a press on a unit, off its tile
        (an unresolved tile differs from a resolved one). The move branch draws and the release decides by it."""
        start = self._marquee_start_pos
        if start is None or (viewport_pos - start).manhattanLength() <= self.UNIT_DRAG_THRESHOLD_PX:
            return False
        return not self._marquee_press_on_unit or tile != self._marquee_press_tile

    def _reset_unit_gesture(self) -> None:
        """Drops whatever unit drag or marquee an earlier press left armed: a
        release swallowed by another branch (Ruler, shapes) must not misroute this one."""
        self._unit_drag_key = None
        self._unit_drag_press_pos = None
        self._marquee_start_pos = None
        self._marquee_press_tile = None
        self._clear_marquee()
        self._pan_click_pos = None

    def _clamped_to_map_rect(self, pos: QPointF) -> QPointF:
        """pos pulled inside _map_rect, so a group release past the edge still commits."""
        rect = self._map_rect
        if rect is None:
            return pos
        x = min(max(pos.x(), rect.left()), rect.right() - 1.0)
        y = min(max(pos.y(), rect.top()), rect.bottom() - 1.0)
        return QPointF(x, y)

    def ground_map_point(self, pos: QPointF) -> tuple[float, float] | None:
        """The continuous map point under pos on the elevation-0 plane, with no
        on-map check: where an off-map group drag is pointing (GH #75)."""
        if self._terrain_style == "flat":
            if not self._tile_pixels:
                return None
            return pos.x() / self._tile_pixels, pos.y() / self._tile_pixels
        proj = self._iso_proj
        if proj is None:
            return None
        u = pos.x() - proj.origin_x - proj.half_w
        v = pos.y() - proj.origin_y - proj.half_h
        return (u / proj.half_w - v / proj.half_h) / 2 + 0.5, (u / proj.half_w + v / proj.half_h) / 2 + 0.5

    def _ensure_unit_ghost_item(self) -> UnitGhostItem:
        """The live ghost item, created on first use with the refusal and garrison colours."""
        item = self._unit_ghost_item
        if item is None:
            item = UnitGhostItem()
            item.setZValue(self.UNIT_GHOST_Z)
            item.set_refusal_color(self._unit_ghost_refused_color)
            item.set_garrison_color(self._unit_ghost_garrison_color)
            self.scene().addItem(item)
            self._unit_ghost_item = item
        return item

    def set_unit_ghosts(self, draws=(), marks=(), refused_draws=(), refused_marks=()) -> None:
        """A group ghost (GH #75): every member's sprite draws, plus a
        (polygons, color) mark per member whose sprite did not resolve. The
        refused_ pair holds the members placement refuses (TASK-327)."""
        item = self._ensure_unit_ghost_item()
        if not item.set_content(
            draws,
            [(p, c) for p, c in marks if p and c is not None],
            refused_draws,
            [(p, c) for p, c in refused_marks if p and c is not None],
        ):
            self._clear_unit_ghost()

    def set_unit_ghost(
        self, draws=None, polygons=None, color=None, refused: bool = False, garrison: bool = False
    ) -> None:
        """The mid-drag move preview's content, resolved by ViewerWindow and
        pushed here -- `set_unit_selection`'s shape, for the same reason (see
        this class's `_on_unit_drag_preview` comment).

        `draws` are render_sprites.unit_sprite_draws_at()'s (draw, px, py) triples;
        `polygons`/`color` are the coloured-mark fallback. Passing neither, or
        a sprite resolve that comes back empty with no mark behind it, clears
        the ghost rather than leaving a stale one on screen: an empty
        sprite_pieces_for(), and a None from unit_polygons() on a ghost entry
        at a destination with no elevations/proj/corner_rise, both mean "no
        ghost this frame" -- never a crash, never a partial draw.

        `refused` draws it in the refused look (TASK-327), `garrison` in the
        garrison look (TASK-549, Place Unit over a host that takes it)."""
        item = self._ensure_unit_ghost_item()
        if draws and item.set_sprites(draws, refused=refused, garrison=garrison):
            return
        if polygons and color is not None and item.set_mark(polygons, color, refused=refused, garrison=garrison):
            return
        self._clear_unit_ghost()

    def _clear_unit_ghost(self) -> None:
        # Leave, tool and mode switches all land here, so a leftover half notch never fires later.
        self._place_wheel_accum = 0
        if self._unit_ghost_item is not None:
            self.scene().removeItem(self._unit_ghost_item)
            self._unit_ghost_item = None
        self._clear_ghost_fish_ring()

    def set_unit_ghost_ring(self, entry) -> None:
        """The place ghost's fish ring (GH #281): `entry` is the ghost's
        unit_pick.UnitEntry, or None. Ringed only while Fish Outlines is on and
        the ghost is a fish, as a click there would show it; refused or not."""
        if entry is None or not self._fish_enabled or entry.unit.unit_const not in unit_kind.fish_consts():
            self._clear_ghost_fish_ring()
            return
        path = self._fish_ring_path((entry,))
        if path.isEmpty():
            self._clear_ghost_fish_ring()
            return
        item = self._ghost_fish_item
        if item is None:
            item = self._ghost_fish_item = self.scene().addPath(path, self._fish_pen)
            item.setZValue(self.GHOST_FISH_Z)
        else:
            item.setPath(path)

    def _clear_ghost_fish_ring(self) -> None:
        if self._ghost_fish_item is not None:
            self.scene().removeItem(self._ghost_fish_item)
            self._ghost_fish_item = None

    def _flat_scene_rect_to_tile_rect(self, rect: QRectF) -> tuple[int, int, int, int] | None:
        """Flat only: the exact half-open tile rect under `rect`, clamped to the map."""
        if self._map_width is None or self._map_height is None:
            return None
        tp = self._tile_pixels or 1
        tx0 = max(0, int(rect.left() // tp))
        ty0 = max(0, int(rect.top() // tp))
        tx1 = min(self._map_width, int(rect.right() // tp) + 1)
        ty1 = min(self._map_height, int(rect.bottom() // tp) + 1)
        return (tx0, ty0, tx1, ty1) if tx0 < tx1 and ty0 < ty1 else None

    def _units_in_scene_rect(self, rect: QRectF) -> list[tuple[int, int]]:
        """The (player_id, reference_id) keys of every unit the marquee
        `rect` (scene/screen pixel space) covers -- already filter-
        respecting, since unit_pick.build_index() never inserted a filtered
        unit into the index in the first place.

        Flat selects any unit with a footprint tile in the box. Stepped and
        Sloped (Flat + Isometric View included) select by on-screen anchor,
        GH #114: a tile bbox is a diamond on screen there."""
        if self._unit_index is None:
            return []
        if self._terrain_style == "flat":
            tile_rect = self._flat_scene_rect_to_tile_rect(rect)
            if tile_rect is None:
                return []
            entries = unit_pick.units_in_rect(self._unit_index, *tile_rect)
        else:
            cache = self._sloped_cache()
            entries = unit_pick.units_in_screen_rect(
                self._unit_index,
                self._terrain_style,
                rect.left(),
                rect.top(),
                rect.right(),
                rect.bottom(),
                elevations=self._iso_elevations,
                proj=self._iso_proj,
                corner_rise=None if cache is None else cache.corner_rise,
                fold=None if cache is None else cache.fold,
            )
        return [unit_pick.unit_key(e.player_id, e.unit) for e in entries]

    def pick_unit_at(self, pos: QPointF, tile=unit_pick.UNRESOLVED_TILE):
        """The unit under scene-space pos, or None -- in every style since
        Track C5's Step 3, so this mirrors _pick_tile again.

        corner_rise comes from the live SlopedChunkCache, for the same Risk
        #6 reason _pick_tile and _tile_polygon read it: that object
        composited the pixels on screen, so hit-testing and the outline see
        the SAME corner_rise those pixels were painted from. terrain_tile is
        resolved here rather than inside unit_pick because Sloped's terrain
        has no analytic inverse -- it is a pick-plane lookup, which is
        exactly the cache reference unit_pick stays free of.

        `tile` is that terrain tile, for a caller that has already resolved
        it: mouseMoveEvent picks one per move for the hover cue, and
        recomputing it here cost that pixel a second plane lookup (Sloped)
        or a second screen_to_tile (Stepped) on every move. Omitting it
        resolves one here, which is what the click paths do."""
        found = self.pick_unit_cover_at(pos, tile)
        return None if found is None else found[0]

    def pick_unit_cover_at(self, pos: QPointF, tile=unit_pick.UNRESOLVED_TILE):
        """pick_unit_at() plus the footprint tile the winner was covering,
        as (entry, (x, y)) -- the click path's key into stack_group_at()."""
        if self._unit_index is None or self._tile_pixels is None:
            return None
        if tile is unit_pick.UNRESOLVED_TILE:
            tile = self._pick_tile(pos)
        cache = self._sloped_cache()
        return unit_pick.pick_unit_cover(
            self._unit_index,
            self._terrain_style,
            int(pos.x()),
            int(pos.y()),
            self._tile_pixels,
            self._map_width or 0,
            self._map_height or 0,
            self._iso_elevations,
            self._iso_proj,
            corner_rise=None if cache is None else cache.corner_rise,
            terrain_tile=tile,
            farms_draped=self._farms_draped(),
            sprites_on=self._objects_after_terrain(),
            fold=None if cache is None else cache.fold,
        )

    def _farms_draped(self) -> bool:
        """Whether the live cache drapes farms as terrain, in any style: the
        render's own three gates (units shown, sprites on, Farm Overlay
        layer), so pick and highlight follow the paint (TASK-015, GH #150)."""
        if self._canvas_item is None:
            return False
        cache = self._canvas_item._cache
        return cache is not None and cache.farms_draped()

    def _objects_after_terrain(self) -> bool:
        """Whether the live cache paints objects after all terrain (TASK-133),
        which pick's terrain occlusion has to follow (TASK-319)."""
        cache = None if self._canvas_item is None else self._canvas_item._cache
        return cache is not None and cache.with_units and cache.sprites_enabled

    def set_footprint_outlines(self, enabled: bool) -> None:
        """Shows or hides View > Footprint Outlines. Like set_edge_ticks,
        this evicts no chunk cache: the outlines are a scene item, never
        baked into canvas pixels."""
        self._footprint_enabled = enabled
        if not enabled:
            self._footprint_chunks.reset()
        for item in self._footprint_items.values():
            item.setVisible(enabled)
        self.refresh_footprint_overlay()

    def set_footprint_scope(self, scope: str) -> None:
        self._footprint_scope = scope
        self.refresh_footprint_overlay()

    def set_footprint_merged(self, enabled: bool) -> None:
        """GH #143: one ring per footprint (unit_pick.merge_footprint_polygons)."""
        self._footprint_merged = enabled
        self.refresh_footprint_overlay()

    def set_footprint_by_owner(self, enabled: bool) -> None:
        """GH #143: each outline in its owner's team colour, GAIA in the configured one."""
        self._footprint_by_owner = enabled
        self.refresh_footprint_overlay()

    def refresh_footprint_owner_colors(self) -> None:
        """For a player colour edit, after set_team_colors(): only
        owner-coloured outlines have anything to recolour."""
        if self._footprint_by_owner:
            self.refresh_footprint_overlay()

    def _add_footprint_item(self, pen: QPen) -> ChunkedPathItem:
        # Not an opacity group, unlike the selection: the configured pen has always drawn opaque.
        item = ChunkedPathItem(pen)
        item.setZValue(self.FOOTPRINT_Z)
        item.setVisible(self._footprint_enabled)
        self.scene().addItem(item)
        return item

    def _outline_context(self) -> tuple:
        """Every input of unit_polygons() that is not per unit or a height field (outline_chunks)."""
        return (
            self._terrain_style,
            self._tile_pixels,
            self._map_width,
            self._map_height,
            self._iso_proj,
            self._farms_draped(),
            game_dataset.generation(),
        )

    def _outline_heights(self) -> tuple:
        """The height fields _unit_polygons_for() reads in the live style."""
        if self._terrain_style == "stepped" and self._iso_elevations is not None:
            return (outline_chunks.HeightField(self._iso_elevations),)
        cache = self._sloped_cache()
        if cache is None or cache.corner_rise is None:
            return ()
        fields = [outline_chunks.HeightField(cache.corner_rise, corners=True)]
        if cache.fold is not None:
            fields.append(outline_chunks.HeightField(cache.fold))
        return tuple(fields)

    def _outline_plan(self, planner, entries, context) -> outline_chunks.OutlinePlan:
        index = self._unit_index if self._unit_index is not None else unit_pick.UnitIndex()
        return planner.plan(entries, index, (context, self._outline_context()), self._outline_heights())

    def refresh_footprint_overlay(self) -> None:
        """Rebuilds the outlines of the chunks an edit made stale, from the
        live index and height field (outline_chunks says what makes a chunk
        stale); a toggle, scope, merge, colour or style change rebuilds all.

        While enabled it runs in every mode: it is live in Terrain mode while
        the elevations under it change, as the kept unit selection now is
        too (TASK-359; both ride _flush_footprint_refresh()).

        While disabled it is skipped, and enabling rebuilds every chunk.

        Colour by Owner splits the path into one item per owner colour (at
        most 9), added and removed here like _draw_unit_selection's groups;
        the None group stays for as long as the map does."""
        if not self._footprint_items:
            return
        # Safe to skip: _footprint_enabled's only writers are __init__ and
        # set_footprint_outlines(), which rebuilds right after enabling.
        if not self._footprint_enabled:
            return
        entries = []
        if self._unit_index is not None:
            entries = unit_pick.footprint_entries(self._unit_index, self._footprint_scope)
        by_owner, merged = self._footprint_by_owner, self._footprint_merged
        colours = self._team_colors if by_owner else None
        plan = self._outline_plan(self._footprint_chunks, entries, (self._footprint_scope, merged, by_owner, colours))
        items = self._footprint_items
        if plan.full:
            for item in items.values():
                item.clear()
        updates: dict = {key: {} for key in items}
        for chunk in plan.emptied:
            for key in items:
                updates[key][chunk] = None
        owner_key = self._owner_group_key
        for chunk, members in plan.dirty.items():
            paths: dict = {}
            for entry in members:
                polygons = self._unit_polygons_for(entry)
                if not polygons:
                    continue
                if merged:
                    polygons = unit_pick.merge_footprint_polygons(polygons)
                key = owner_key(entry.player_id, True) if by_owner else None
                path = paths.get(key)
                if path is None:
                    path = paths[key] = QPainterPath()
                _add_closed_polygons(path, polygons)
            for key, group in updates.items():
                group[chunk] = paths.pop(key, None)
            for key, path in paths.items():
                updates[key] = {chunk: path}
        for key, group in updates.items():
            item = items.get(key)
            if item is None:
                item = items[key] = self._add_footprint_item(QPen(QColor(*key), 0))
            item.set_chunks(group)
        for key in [k for k, item in items.items() if k is not None and not item.chunk_count()]:
            self.scene().removeItem(items.pop(key))

    def set_fish_outlines(self, enabled: bool) -> None:
        """Shows or hides View > Layers > Fish Outlines (TASK-499). A scene
        item like set_footprint_outlines()'s, so no chunk cache is evicted."""
        self._fish_enabled = enabled
        if not enabled:
            self._fish_chunks.reset()
        if self._fish_item is not None:
            self._fish_item.setVisible(enabled)
        self.refresh_fish_overlay()
        # The hover ghost's ring follows with no mouse move (GH #281).
        self.refresh_place_ghost()

    def _add_fish_item(self) -> ChunkedPathItem:
        item = ChunkedPathItem(self._fish_pen)
        item.setZValue(self.FISH_OUTLINE_Z)
        item.setVisible(self._fish_enabled)
        self.scene().addItem(item)
        return item

    def refresh_fish_overlay(self) -> None:
        """Rebuilds the fish rings of the stale chunks, as refresh_footprint_overlay()
        does: one merged ring per unit_kind.fish_consts() member the filter shows,
        whatever Footprint Outlines' scope and Merge tiles say. Skipped while
        disabled, like the footprint path; set_fish_outlines() rebuilds on enable."""
        if self._fish_item is None or not self._fish_enabled:
            return
        entries = []
        if self._unit_index is not None:
            fish = unit_kind.fish_consts()
            entries = [entry for entry in self._unit_index.entries if entry.unit.unit_const in fish]
        plan = self._outline_plan(self._fish_chunks, entries, "fish")
        if plan.full:
            self._fish_item.clear()
        updates: dict = dict.fromkeys(plan.emptied)
        for chunk, members in plan.dirty.items():
            updates[chunk] = self._fish_ring_path(members)
        self._fish_item.set_chunks(updates)

    def _fish_ring_path(self, entries) -> QPainterPath:
        """One merged ring per entry, from the live height field: the placed
        fish's rings and a place ghost's (GH #281) share it, so they cannot disagree."""
        path = QPainterPath()
        for entry in entries:
            polygons = self._unit_polygons_for(entry)
            if polygons:
                _add_closed_polygons(path, unit_pick.merge_footprint_polygons(polygons))
        return path

    def _refresh_unit_outline_overlays(self) -> None:
        """Every outline overlay built from the unit index: footprints and fish."""
        self.refresh_footprint_overlay()
        self.refresh_fish_overlay()

    def schedule_footprint_refresh(self) -> None:
        """Queues one rebuild of the footprint and fish outlines, the always-on
        ring layer and a drawn unit selection (TASK-359), whichever is live."""
        if self._footprint_refresh_pending:
            return
        # Hidden: nothing to queue per drag step (refresh_footprint_overlay's invariant).
        footprints = bool(self._footprint_items) and self._footprint_enabled
        fish = self._fish_item is not None and self._fish_enabled
        if not footprints and not fish and not self._ring_layer_wanted() and not self._unit_select_keys:
            return
        self._footprint_refresh_pending = True
        QTimer.singleShot(0, self._flush_footprint_refresh)

    def _flush_footprint_refresh(self) -> None:
        self._footprint_refresh_pending = False
        # scene().clear() can have destroyed the item between the schedule
        # and this call (File > Close, a style switch), so re-check.
        if self._footprint_items:
            with perf_trace.phase("footprint_refresh"):
                self.refresh_footprint_overlay()
        if self._fish_item is not None and self._fish_enabled:
            with perf_trace.phase("fish_refresh"):
                self.refresh_fish_overlay()
        if self._ring_layer_wanted():
            with perf_trace.phase("ring_layer_refresh"):
                self.refresh_ring_layer()
        # A timer can fire between every drag step; rebuild a large kept selection only at stroke end.
        if self._unit_select_keys and not self._stroke_active:
            with perf_trace.phase("selection_refresh"):
                self.refresh_unit_highlight()
