"""MapView's trigger side: unit and tile pickers, area and location edit, the trigger overlay, its ghosts, ring paths and run labels."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from functools import lru_cache
from typing import NamedTuple

import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsSimpleTextItem,
)

from descape import (
    iso_geometry,
    ruler,
    settings,
    trigger_geometry,
    unit_pick,
)
from descape.instruction_overlay import InstructionOverlay

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- render.py imports no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.viewer_canvas import (
    UnitGhostItem,
    map_overlay_font,
)


class _RingEntry(NamedTuple):
    path: QPainterPath
    indices: tuple[np.ndarray, ...] = ()
    values: np.ndarray | None = None
    rise: np.ndarray | None = None


@lru_cache(maxsize=1024)
def _sloped_edge_points(tile_px: int, side: str, nw: int, ne: int, sw: int, se: int) -> np.ndarray:
    """One Sloped tile edge's tile-local points as a read-only float64 (N, 2)
    array, keyed on the -d_min-normalized corners like _sloped_outline_polygon
    (TASK-355). Never empty: half_w >= 2, so each side keeps a column."""
    points = np.asarray(iso_geometry.sloped_tile_edge_outline(tile_px, side, nw, ne, sw, se), dtype=np.float64)
    points = np.ascontiguousarray(points.reshape(-1, 2))
    points.setflags(write=False)
    return points


def _polygon_from_points(points: np.ndarray) -> QPolygonF:
    """A QPolygonF holding a float64 (N, 2) array's rows, written in one copy
    through the polygon's own buffer (the sip voidptr route zoom_preframe.py
    uses for QImage bits). Relies on qreal being double, as on every desktop
    PyQt5 build."""
    points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = QPolygonF(len(points))
    if len(points):
        buffer = polygon.data()
        buffer.setsize(points.nbytes)
        np.frombuffer(buffer, dtype=np.float64).reshape(-1, 2)[:] = points
    return polygon


def _points_from_polygon(polygon: QPolygonF) -> np.ndarray:
    """The reverse of _polygon_from_points: a float64 (N, 2) copy of the points."""
    count = polygon.size()
    if not count:
        return np.zeros((0, 2), dtype=np.float64)
    buffer = polygon.data()
    buffer.setsize(count * 2 * np.dtype(np.float64).itemsize)
    return np.frombuffer(buffer, dtype=np.float64).reshape(count, 2).copy()


def _ring_path_from_points(points: np.ndarray) -> QPainterPath:
    """One closed subpath through `points`, consecutive duplicates dropped as
    the old per-point loop's currentPosition() check did. No points, empty path."""
    path = QPainterPath()
    if not len(points):
        return path
    # One complex per row: an exact (x, y) compare in a single ufunc call.
    rows = np.ascontiguousarray(points, dtype=np.float64).view(np.complex128).ravel()
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = rows[1:] != rows[:-1]
    path.addPolygon(_polygon_from_points(points[keep]))
    path.closeSubpath()
    return path


class _MapViewTriggersMixin:
    """MapView's trigger pickers and overlay (map_view.py's docstring has the split's rules)."""

    def set_unit_picker(self, active: bool) -> None:
        """Trigger Pick from map: left clicks pick a unit (on_unit_pick) and
        hover outlines one, in any mode. The window forces Pan first."""
        active = bool(active)
        if active == self._unit_picker_active:
            return
        self._unit_picker_active = active
        self._picker_cycle = None
        self._cancel_area_drag()
        if not active:
            self._clear_unit_hover()
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def unit_picker_active(self) -> bool:
        return self._unit_picker_active

    def set_tile_picker(self, kind: str | None) -> None:
        """GH #138: "point" picks one tile (on_tile_pick), "rect" one dragged
        rectangle (on_rect_pick), None disarms. The window forces Pan first."""
        self._cancel_area_drag()
        self._tile_picker_kind = kind
        self._tile_pick_anchor = None
        self._clear_tile_pick_preview()
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def tile_picker_kind(self) -> str | None:
        return self._tile_picker_kind

    def set_area_edit(self, coords: tuple[int, int, int, int] | None) -> None:
        """GH #229: the inclusive area (x1, y1, x2, y2) a Pan drag may move or
        resize, or None. Pushed by the window; a change drops a live drag."""
        if coords == self._area_edit:
            return
        self._area_edit = coords
        self._cancel_area_drag()

    def area_edit(self) -> tuple[int, int, int, int] | None:
        return self._area_edit

    def set_location_edit(self, tile: tuple[int, int] | None, origin: tuple[int, int] | None = None) -> None:
        """GH #41: the location a Pan drag may move, or None, and the run origin
        its preview measures from. Pushed by the window; a change drops a live drag."""
        if (tile, origin) == (self._location_edit, self._location_run_origin):
            return
        self._location_edit, self._location_run_origin = tile, origin
        self._cancel_area_drag()

    def location_edit(self) -> tuple[int, int] | None:
        return self._location_edit

    def set_area_edit_options(self, move: bool, resize: bool, tile_borders: bool) -> None:
        """Edit Area's options. Resize off makes the whole area the move zone, Move
        off leaves only the edges and corners, Tile borders makes the border at
        least one tile wide on an axis 3+ tiles long. A change drops a live drag."""
        options = (move, resize, tile_borders)
        if options == (self._area_move, self._area_resize, self._area_tile_borders):
            return
        self._area_move, self._area_resize, self._area_tile_borders = options
        self._cancel_area_drag()

    def _area_edit_live(self) -> bool:
        return (
            (
                (self._area_edit is not None and (self._area_move or self._area_resize))
                or (self._location_edit is not None and self._area_move)
            )
            and self._tool == "pan"
            and self._tile_picker_kind is None
            and not self._unit_picker_active
        )

    def _map_axes_on_screen(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """One tile along map x and along map y as viewport-pixel vectors, through
        the whole view transform (an iso view is squashed vertically)."""
        if self._terrain_style == "flat" or self._iso_proj is None:
            step = self._tile_pixels or 1
            axes = ((step, 0.0), (0.0, step))
        else:
            hw, hh = self._iso_proj.half_w, self._iso_proj.half_h
            axes = ((hw, -hh), (hw, hh))
        t = self.transform()
        return tuple((x * t.m11() + y * t.m21(), x * t.m12() + y * t.m22()) for x, y in axes)

    def _area_band_tiles(self) -> float:
        """AREA_EDIT_BAND_PX in tiles, across the narrower of a tile's two edge spacings."""
        (ax, ay), (bx, by) = self._map_axes_on_screen()
        area = abs(ax * by - ay * bx)
        across = area / max(math.hypot(ax, ay), math.hypot(bx, by)) if area else 1.0
        return self.AREA_EDIT_BAND_PX / across

    def _area_hit(self, pos: QPointF) -> str | None:
        """The live grab zone under scene-space pos, or None. The location mark
        draws above the area, so it wins: its own half-extent, floored on screen."""
        if not self._area_edit_live():
            return None
        point = self._pick_map_point(pos) or self.ground_map_point(pos)
        if point is None:
            return None
        if self._area_move and self._location_edit is not None:
            reach = max(self.TRIGGER_LOCATION_SCALE / 2, self._area_band_tiles())
            if trigger_geometry.location_hit(self._location_edit, point, reach):
                return trigger_geometry.ZONE_LOCATION
        if self._area_edit is None:
            return None
        # Resize off means no border at all, Tile borders included.
        band = self._area_band_tiles() if self._area_resize else 0.0
        tile_borders = self._area_tile_borders and self._area_resize
        zone = trigger_geometry.area_hit_zone(self._area_edit, point, band, tile_borders)
        return None if zone == trigger_geometry.ZONE_MOVE and not self._area_move else zone

    def _area_zone_cursor(self, zone: str) -> Qt.CursorShape:
        """The resize cursor along the zone's ON-SCREEN drag direction, so an
        iso map's diagonal edges read right."""
        if zone in (trigger_geometry.ZONE_MOVE, trigger_geometry.ZONE_LOCATION):
            return Qt.SizeAllCursor
        ax, ay = self._AREA_ZONE_AXES[zone]
        (ex, ey), (fx, fy) = self._map_axes_on_screen()
        sx, sy = ax * ex + ay * fx, ax * ey + ay * fy
        return self._SIZE_CURSORS[round(math.degrees(math.atan2(-sy, sx)) / 45) % 4]

    def _set_area_hover(self, zone: str | None) -> None:
        if zone == self._area_hover:
            return
        self._area_hover = zone
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _cancel_area_drag(self) -> None:
        """Drops a GH #229 drag with no write, and the hover zone with it."""
        if self._area_drag is not None:
            self._area_drag = None
            self._area_drag_end = None
            self._clear_tile_pick_preview()
        self._area_hover = None
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _dragged_area(self) -> tuple[int, int, int, int]:
        zone, start, coords = self._area_drag
        return trigger_geometry.dragged_area(coords, zone, start, self._area_drag_end, self._map_width, self._map_height)

    def _update_area_drag_preview(self) -> None:
        if self._area_drag[0] == trigger_geometry.ZONE_LOCATION:
            self._update_location_drag_preview(self._dragged_area()[:2])
            return
        rect = self._trigger_area_rect(self._dragged_area())
        self._show_tile_pick_path(
            self._rect_ring_path(rect) if rect is not None else QPainterPath(),
            self._trigger_outline_pens[True],
            QBrush(self._trigger_fill_colors[True]),
            self.TRIGGER_AREA_Z,
            self._overlay_opacity("trigger_area"),
        )

    def _update_location_drag_preview(self, tile: tuple[int, int]) -> None:
        """The mark at `tile`, plus the run line and its live distance when the
        entry has an origin on the map, all in the point pick's look."""
        path = self._location_mark_path(tile)
        origin = self._location_run_origin
        start = self._ruler_anchor(origin) if origin is not None and self._on_map(origin) else None
        end = self._ruler_anchor(tile)
        if start is not None and end is not None:
            path.moveTo(start)
            path.lineTo(end)
            self._show_tile_pick_label(end, ruler.format_measurement(ruler.measure(origin, tile)))
        self._show_tile_pick_path(path, self._trigger_run_pens[True], QBrush(Qt.NoBrush), self.TRIGGER_MARK_Z, 1.0)

    def _show_tile_pick_label(self, anchor: QPointF, text: str) -> None:
        label = self._tile_pick_label
        if label is None:
            label = self._add_run_label(anchor, text)
            label.setData(0, "tile_pick_preview")
            self._tile_pick_label = label
        else:
            label.setText(text)
            self._place_run_label(label, anchor)

    def _tile_pick_sample(self, pos: QPointF) -> tuple[int, int] | None:
        """A rect drag's tile under scene-space pos: _pick_tile on the map, else
        the elevation-0 point clamped to the map edge (past it, or a skirt face)."""
        if self._map_width is None or self._map_height is None:
            return None
        # _pos_on_map first: Flat's _pick_tile has no bounds check.
        tile = self._pick_tile(pos) if self._pos_on_map(pos) else None
        if tile is None:
            point = self.ground_map_point(pos)
            if point is None:
                return None
            tile = (math.floor(point[0]), math.floor(point[1]))
        return (min(max(tile[0], 0), self._map_width - 1), min(max(tile[1], 0), self._map_height - 1))

    def _update_tile_pick_preview(self, tile: tuple[int, int] | None) -> None:
        """The pick's preview in the overlay's strong colours: the anchor-to-tile
        rectangle (a hovered 1x1 before the press), or a point's location mark."""
        scene = self.scene()
        if scene is None or tile is None:
            self._clear_tile_pick_preview()
            return
        if self._tile_picker_kind == "rect":
            path = QPainterPath()
            anchor = self._tile_pick_anchor or tile
            rect = self._trigger_area_rect((anchor[0], anchor[1], tile[0], tile[1]))
            if rect is not None:
                path = self._rect_ring_path(rect)
            pen, brush = self._trigger_outline_pens[True], QBrush(self._trigger_fill_colors[True])
            z, opacity = self.TRIGGER_AREA_Z, self._overlay_opacity("trigger_area")
        else:
            path = self._location_mark_path(tile)
            pen, brush, z, opacity = self._trigger_run_pens[True], QBrush(Qt.NoBrush), self.TRIGGER_MARK_Z, 1.0
        self._show_tile_pick_path(path, pen, brush, z, opacity)

    def _show_tile_pick_path(self, path: QPainterPath, pen, brush: QBrush, z: float, opacity: float) -> None:
        scene = self.scene()
        if scene is None:
            return
        item = self._tile_pick_item
        if item is None:
            item = scene.addPath(path, pen, brush)
            item.setData(0, "tile_pick_preview")
            self._tile_pick_item = item
        else:
            item.setPath(path)
            item.setPen(pen)
            item.setBrush(brush)
        item.setZValue(z)
        item.setOpacity(opacity)

    def _clear_tile_pick_preview(self) -> None:
        if self._tile_pick_item is not None:
            self.scene().removeItem(self._tile_pick_item)
            self._tile_pick_item = None
        if self._tile_pick_label is not None:
            self.scene().removeItem(self._tile_pick_label)
            self._tile_pick_label = None

    def _cycle_pick(self, pos: QPointF, cycle_attr: str):
        """One pick: (entry, cover tile) for the unit under pos, walking down a
        stack on repeat clicks, or None on empty ground. `cycle_attr` names the
        caller's own (cover tile, stack index) attribute."""
        found = self.pick_unit_cover_at(pos)
        if found is None:
            return None
        entry, cover_tile = found
        previous = getattr(self, cycle_attr)
        entry, index = unit_pick.stack_cycle_step(
            self.stack_group_at(cover_tile), entry, previous[1] if previous and previous[0] == cover_tile else None
        )
        setattr(self, cycle_attr, (cover_tile, index) if index >= 0 else None)
        return entry, cover_tile

    def _picker_click(self, pos: QPointF) -> None:
        """Trigger Pick from map's click: reports the picked unit's key."""
        found = self._cycle_pick(pos, "_picker_cycle")
        if found is not None:
            entry, _cover_tile = found
            self.on_unit_pick(unit_pick.unit_key(entry.player_id, entry.unit))

    def show_trigger_overlay(self, shapes: Sequence, emphasis: Collection[tuple[str, int]] | None) -> None:
        """GH #41: the selected triggers' areas, locations and runs, from
        trigger_geometry.TriggerShape snapshots. `emphasis` is the selected
        entries' (kind, index) refs (GH #168): they draw at full strength and
        every other entry dimmed; None or empty draws everything at full
        strength. Walls are not drawn. Referenced units ("unit" shapes) are
        outlined, never filled."""
        drawn = (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_LOCATION, trigger_geometry.SHAPE_UNIT)
        self._trigger_shapes = [s for s in shapes if s.shape in drawn]
        # Empty folds into None, so "everything strong" has one representation.
        self._trigger_emphasis = frozenset(emphasis) if emphasis else None
        self._rebuild_trigger_overlay()

    def clear_trigger_overlay(self) -> None:
        """No trigger selected, or its document went away. Idempotent."""
        self._trigger_shapes = []
        self._trigger_emphasis = None
        self._rebuild_trigger_overlay()

    def set_trigger_overlay_enabled(self, enabled: bool) -> None:
        """View > Trigger Overlay: visibility only, the snapshot is kept."""
        self._trigger_overlay_enabled = bool(enabled)
        for item in self._trigger_items:
            item.setVisible(self._trigger_overlay_enabled)

    def trigger_overlay_items(self) -> list[QGraphicsItem]:
        """The live scene items, for tests and the review pack."""
        return list(self._trigger_items)

    def show_instruction(self, view) -> None:
        """TASK-550: one Display Instructions box over the map, or None to hide it."""
        self._instruction_overlay.place(self.viewport().geometry())
        self._instruction_overlay.set_view(view)

    def instruction_overlay(self) -> InstructionOverlay:
        """The widget, for tests and the review pack."""
        return self._instruction_overlay

    def show_trigger_ghosts(self, records: Sequence, emphasis: Collection[tuple[str, int]] | None, sprites: bool) -> None:
        """GH #167: a see-through copy of each unit the selected triggers'
        Create Object and Place Foundation effects would place
        (trigger_geometry.CreatedUnit records). `emphasis` follows
        show_trigger_overlay()'s rule; `sprites` False draws every ghost as its mark."""
        self._trigger_ghost_records = list(records)
        self._trigger_ghost_emphasis = frozenset(emphasis) if emphasis else None
        self._trigger_ghost_sprites = bool(sprites)
        self._rebuild_trigger_ghosts()

    def clear_trigger_ghosts(self) -> None:
        """No trigger selected, or its document went away. Idempotent."""
        self._trigger_ghost_records = []
        self._trigger_ghost_emphasis = None
        self._rebuild_trigger_ghosts()

    def set_trigger_ghosts_enabled(self, enabled: bool) -> None:
        """View > Show Trigger-Created Units: off removes the items and keeps the records and memo."""
        self._trigger_ghosts_enabled = bool(enabled)
        self._rebuild_trigger_ghosts()

    def refresh_trigger_ghosts(self) -> None:
        """A look input set_source() does not cover changed (sprites, layers, player colours): re-resolve."""
        self._trigger_ghost_parts = {}
        self._rebuild_trigger_ghosts()

    def trigger_ghost_items(self) -> list[UnitGhostItem]:
        """The live ghost items, strong and dim, for tests and the review pack."""
        return list(self._trigger_ghost_items)

    def _forget_trigger_ghost_items(self) -> None:
        """After scene().clear(): the items are gone, and the memo holds the old projection."""
        self._trigger_ghost_items = []
        self._trigger_ghost_parts = {}

    def _rebuild_trigger_ghosts(self) -> None:
        """Rebuilds both ghost items from the records. Only a record whose
        (payload, sprites) is not in the memo is resolved, so an emphasis
        change or a unit edit with the same records resolves nothing. Items
        never touch the chunk cache."""
        scene = self.scene()
        if scene is None:
            return
        for item in self._trigger_ghost_items:
            scene.removeItem(item)
        self._trigger_ghost_items = []
        if not self._trigger_ghost_records:
            self._trigger_ghost_parts = {}
            return
        if not self._trigger_ghosts_enabled or self._map_width is None:
            return
        sprites = self._trigger_ghost_sprites
        parts = self._trigger_ghost_parts
        live: dict[tuple, tuple[list, tuple | None]] = {}
        content: dict[bool, tuple[list, list]] = {True: ([], []), False: ([], [])}
        prepared = False
        # Row-major by tile, the order the composite walk paints units in.
        for record in sorted(self._trigger_ghost_records, key=lambda r: (r.tile[1], r.tile[0])):
            key = (record.payload, sprites)
            hit = live.get(key) or parts.get(key)
            if hit is None:
                if not prepared:
                    self.prepare_trigger_ghost_resolve()
                    prepared = True
                hit = self.resolve_trigger_ghost(record, sprites)
            live[key] = hit
            draws, mark = hit
            strong = self._trigger_ghost_emphasis is None or record.entry_ref in self._trigger_ghost_emphasis
            content[strong][0].extend(draws)
            if not draws and mark is not None:
                content[strong][1].append(mark)
        # Only what this selection draws stays cached.
        self._trigger_ghost_parts = live
        for strong in (False, True):
            draws, marks = content[strong]
            if not draws and not marks:
                continue
            item = UnitGhostItem()
            item.set_content(draws, marks)
            item.setOpacity(UnitGhostItem.GHOST_OPACITY if strong else self.TRIGGER_GHOST_DIM_OPACITY)
            item.setZValue(self.TRIGGER_GHOST_Z)
            item.setData(0, "ghosts_strong" if strong else "ghosts_dim")
            scene.addItem(item)
            self._trigger_ghost_items.append(item)

    def _schedule_trigger_ghost_refresh(self) -> None:
        """An elevation edit moves the ghosts' ground: drop the memo and queue
        one re-resolve on the event loop, coalesced like schedule_footprint_refresh()."""
        if not self._trigger_ghost_records:
            return
        self._trigger_ghost_parts = {}
        if self._trigger_ghost_refresh_pending:
            return
        self._trigger_ghost_refresh_pending = True
        QTimer.singleShot(0, self._flush_trigger_ghost_refresh)

    def _flush_trigger_ghost_refresh(self) -> None:
        self._trigger_ghost_refresh_pending = False
        # A timer can fire between drag steps: rebuild once at stroke end (_end_stroke()).
        if self._stroke_active:
            self._trigger_ghosts_stale = True
            return
        self._rebuild_trigger_ghosts()

    def _rect_ring_path(self, rect: tuple[int, int, int, int]) -> QPainterPath:
        """A half-open tile rect's outer boundary as one closed ring that
        follows each border tile's own edge, so it climbs Stepped risers and
        Sloped warping. O(perimeter): only the border is visited.

        Built in numpy (TASK-355) and element-identical to walking
        _tile_edge_points() over _region_edge_walk(), which stays the spec
        (tests/test_trigger_overlay_viewer.py pins the two together). Empty
        before a projection snapshot (and, for Sloped, a cache) exists."""
        points = self._rect_ring_points(rect)
        return QPainterPath() if points is None else _ring_path_from_points(points)

    def _memo_ring_path(self, rect: tuple[int, int, int, int]) -> QPainterPath | None:
        entry = self._trigger_ring_memo.get(rect)
        return None if entry is None else entry.path

    def _trigger_ring_path(self, rect: tuple[int, int, int, int]) -> QPainterPath:
        entry = self._trigger_ring_memo.get(rect)
        if entry is not None:
            return entry.path
        path = self._rect_ring_path(rect)
        if path.isEmpty():
            return path
        entry = _RingEntry(path)
        if self._terrain_style in _ELEVATED_STYLES:
            chains = self._region_edge_chains(*rect)
            xs = np.concatenate([c[0] for c in chains])
            ys = np.concatenate([c[1] for c in chains])
            rise = self._sloped_cache().corner_rise if self._terrain_style == "sloped" else self._iso_elevations
            nw = ys * rise.shape[1] + xs
            indices = (nw, nw + 1, nw + rise.shape[1], nw + rise.shape[1] + 1) if self._terrain_style == "sloped" else (nw,)
            # An elevation edit changes a ring only through these border values.
            values = np.take(rise, np.concatenate(indices))
            entry = _RingEntry(path, indices, values, rise if self._terrain_style == "sloped" else None)
        self._trigger_ring_memo[rect] = entry
        return path

    def _ring_fingerprint_matches(self, entries: dict[tuple[int, int, int, int], _RingEntry]) -> set[tuple[int, int, int, int]]:
        """Valid rects, with one batched height read for all entries needing a check."""
        if self._terrain_style == "flat":
            return set(entries)
        sloped = self._terrain_style == "sloped"
        cache = self._sloped_cache() if sloped else None
        rise = (None if cache is None else cache.corner_rise) if sloped else self._iso_elevations
        if rise is None:
            return set()
        valid = {rect for rect, entry in entries.items() if sloped and entry.rise is rise}
        check = [(rect, entry) for rect, entry in entries.items() if rect not in valid]
        if check:
            current = np.take(rise, np.concatenate([index for _rect, entry in check for index in entry.indices]))
            expected = np.concatenate([entry.values for _rect, entry in check])
            starts = np.cumsum([0, *(entry.values.size for _rect, entry in check[:-1])])
            matches = np.add.reduceat(current != expected, starts) == 0
            for (rect, entry), match in zip(check, matches, strict=True):
                if match:
                    valid.add(rect)
                    if sloped:
                        entries[rect] = entry._replace(rise=rise)
        return valid

    def _rect_ring_points(self, rect: tuple[int, int, int, int]) -> np.ndarray | None:
        """_rect_ring_path()'s float64 (N, 2) points, before its consecutive-
        duplicate drop, or None with no geometry to place them on."""
        if self._terrain_style == "sloped":
            return self._sloped_ring_points(rect)
        parts = []
        if self._terrain_style == "stepped":
            proj, elevations = self._iso_proj, self._iso_elevations
            if proj is None or elevations is None:
                return None
            local = np.array(unit_pick.diamond_points(0, 0, proj.half_w, proj.half_h), dtype=np.float64)
            table = self._STEPPED_EDGE_VERTICES
            for xs, ys, side in self._region_edge_chains(*rect):
                ox, oy = iso_geometry.tile_screen_origin(xs, ys, elevations[ys, xs].astype(np.int64), proj)
                parts.append((side, np.stack([ox, oy], axis=1)))
        else:
            tp = self._tile_pixels
            local = np.array([(0, 0), (tp, 0), (tp, tp), (0, tp)], dtype=np.float64)
            table = self._FLAT_EDGE_VERTICES
            for xs, ys, side in self._region_edge_chains(*rect):
                parts.append((side, np.stack([xs * tp, ys * tp], axis=1)))
        # Each side's two local points are constant, so a chain is one broadcast add.
        return np.concatenate(
            [
                (local[list(table[side])][None, :, :] + origins[:, None, :]).reshape(-1, 2)
                for side, origins in parts
            ]
        )

    def _sloped_ring_points(self, rect: tuple[int, int, int, int]) -> np.ndarray | None:
        """Sloped's ring points: per border tile, the cached tile-local edge for
        its normalized corners (_sloped_edge_points), offset exactly as
        _sloped_tile_base() places the tile."""
        cache = self._sloped_cache()
        proj = self._iso_proj
        if cache is None or proj is None:
            return None
        rise = cache.corner_rise
        tile_px = self._tile_pixels
        chains = self._region_edge_chains(*rect)
        xs = np.concatenate([c[0] for c in chains])
        ys = np.concatenate([c[1] for c in chains])
        sides = [side for chain_xs, _ys, side in chains for _ in range(len(chain_xs))]
        corners = np.stack([rise[ys, xs], rise[ys, xs + 1], rise[ys + 1, xs], rise[ys + 1, xs + 1]]).astype(np.int64)
        d_min = corners.min(axis=0)
        nw, ne, sw, se = (corners - d_min).tolist()
        # The per-tile part: one cache lookup per border tile, in _region_edge_walk()'s order.
        arrays = [_sloped_edge_points(tile_px, *key) for key in zip(sides, nw, ne, sw, se, strict=True)]
        ox, oy = iso_geometry.tile_screen_origin(xs, ys, 0, proj)
        offsets = np.stack([ox, oy - d_min], axis=1).astype(np.float64)
        return np.concatenate(arrays) + np.repeat(offsets, [len(a) for a in arrays], axis=0)

    def _trigger_area_rect(self, coords) -> tuple[int, int, int, int] | None:
        """An inclusive trigger area as a half-open rect clamped to the map, or
        None when none of it is on the map. A typed area_x2 of 500 must not
        index past the elevation grid."""
        if self._map_width is None or self._map_height is None:
            return None
        x1, y1, x2, y2 = coords
        tx0, tx1 = max(0, min(x1, x2)), min(self._map_width, max(x1, x2) + 1)
        ty0, ty1 = max(0, min(y1, y2)), min(self._map_height, max(y1, y2) + 1)
        if tx0 >= tx1 or ty0 >= ty1:
            return None
        return tx0, ty0, tx1, ty1

    def _on_map(self, tile: tuple[int, int]) -> bool:
        return (
            self._map_width is not None
            and self._map_height is not None
            and 0 <= tile[0] < self._map_width
            and 0 <= tile[1] < self._map_height
        )

    def _location_mark(self, tile: tuple[int, int]) -> QPolygonF | None:
        """The named tile's own polygon shrunk about its centre, so the mark
        sits inside that tile at its real height in every style."""
        polygon = self._tile_polygon(*tile)
        if polygon is None:
            return None
        centre = polygon.boundingRect().center()
        c = np.array([centre.x(), centre.y()], dtype=np.float64)
        # Same operation order as the per-point centre + (p - centre) * s, so bit-identical.
        return _polygon_from_points(c + (_points_from_polygon(polygon) - c) * self.TRIGGER_LOCATION_SCALE)

    def _location_mark_path(self, tile: tuple[int, int]) -> QPainterPath:
        path = QPainterPath()
        polygon = self._location_mark(tile)
        if polygon is not None:
            path.addPolygon(polygon)
            path.closeSubpath()
        return path

    def _forget_trigger_items(self) -> None:
        """After scene().clear(): the C++ items are gone, only the refs remain."""
        self._trigger_items = []

    def _rebuild_trigger_overlay(self) -> None:
        """Rebuilds every trigger item from the snapshot against the CURRENT
        projection: on a push, after set_source(), on an elevation edit and on
        a colour change. Items never touch the chunk cache.
        Rings use a per-rect memo, cleared by set_source() and checked against border heights on every rebuild."""
        scene = self.scene()
        if scene is None:
            return
        for item in self._trigger_items:
            scene.removeItem(item)
        self._trigger_items = []
        if not self._trigger_shapes:
            return
        rects = {
            rect for shape in self._trigger_shapes
            if shape.shape in (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_UNIT)
            and (rect := self._trigger_area_rect(shape.coords)) is not None
        }
        if rects:
            entries = {rect: entry for rect, entry in self._trigger_ring_memo.items() if rect in rects}
            valid = self._ring_fingerprint_matches(entries)
            # Only what this selection draws stays cached; deselecting keeps it warm.
            self._trigger_ring_memo = {rect: entries[rect] for rect in valid}
        rings = {True: QPainterPath(), False: QPainterPath()}
        units = {True: QPainterPath(), False: QPainterPath()}
        marks = {True: QPainterPath(), False: QPainterPath()}
        runs: list[tuple[QPointF, str]] = []
        # Keyed on the trigger too: two triggers' ("effect", 1) must never pair into one run (GH #168).
        by_entry: dict[tuple[int, tuple[str, int]], list] = {}
        seen: set[tuple[bool, bool, tuple[int, int, int, int]]] = set()
        for shape in self._trigger_shapes:
            by_entry.setdefault((shape.trigger_index, shape.entry_ref), []).append(shape)
        for key, shapes in by_entry.items():
            _trigger_index, entry_ref = key
            strong = self._trigger_emphasis is None or entry_ref in self._trigger_emphasis
            # A location the entry's location object overrides draws dimmed: the hint that it is not the target.
            superseded = trigger_geometry.location_superseded(shapes)
            for shape in shapes:
                if shape.shape in (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_UNIT):
                    # A unit's footprint is a tile rect too: the same O(perimeter) ring, outline only.
                    rect = self._trigger_area_rect(shape.coords)
                    is_area = shape.shape == trigger_geometry.SHAPE_AREA
                    # Once per bucket: a repeat adds no fill, only a darker antialiased outline.
                    if rect is not None and (is_area, strong, rect) not in seen:
                        seen.add((is_area, strong, rect))
                        target = rings if is_area else units
                        target[strong].addPath(self._trigger_ring_path(rect))
                elif self._on_map(shape.coords):
                    polygon = self._location_mark(shape.coords)
                    if polygon is not None:
                        mark = marks[strong and not superseded]
                        mark.addPolygon(polygon)
                        mark.closeSubpath()
            run = trigger_geometry.run_for_entry(shapes)
            if run is None or not (self._on_map(run.a) and self._on_map(run.b)):
                continue
            a, b = self._ruler_anchor(run.a), self._ruler_anchor(run.b)
            if a is None or b is None:
                continue
            marks[strong].moveTo(a)
            marks[strong].lineTo(b)
            # Labels only on full-strength runs: a dimmed sibling's number is clutter.
            if strong:
                runs.append((b, ruler.format_measurement(run)))
        for strong in (False, True):
            suffix = "strong" if strong else "dim"
            ring = rings[strong]
            if not ring.isEmpty():
                ring.setFillRule(Qt.WindingFill)
                fill = scene.addPath(ring, QPen(Qt.NoPen), QBrush(self._trigger_fill_colors[strong]))
                outline = scene.addPath(ring, self._trigger_outline_pens[strong])
                for role, item in ((f"fill_{suffix}", fill), (f"outline_{suffix}", outline)):
                    item.setZValue(self.TRIGGER_AREA_Z)
                    # Areas only (GH #129): marks, runs, unit refs and labels stay full strength.
                    item.setOpacity(self._overlay_opacity("trigger_area"))
                    item.setData(0, role)
                    self._trigger_items.append(item)
            if not units[strong].isEmpty():
                item = scene.addPath(units[strong], self._trigger_unit_pens[strong])
                item.setZValue(self.TRIGGER_AREA_Z)
                item.setData(0, f"units_{suffix}")
                self._trigger_items.append(item)
            if not marks[strong].isEmpty():
                item = scene.addPath(marks[strong], self._trigger_run_pens[strong])
                item.setZValue(self.TRIGGER_MARK_Z)
                item.setData(0, f"marks_{suffix}")
                self._trigger_items.append(item)
        for anchor, text in runs:
            label = self._add_run_label(anchor, text)
            label.setData(0, "label")
            self._trigger_items.append(label)
        for item in self._trigger_items:
            item.setVisible(self._trigger_overlay_enabled)

    def _add_run_label(self, anchor: QPointF, text: str) -> QGraphicsSimpleTextItem:
        label = self.scene().addSimpleText(text)
        font = map_overlay_font(settings.get_ruler_label_font_px())
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(self._trigger_run_pens[True].color()))
        label.setPen(QPen(self._ruler_label_outline, 0))
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label.setZValue(self.TRIGGER_LABEL_Z)
        self._place_run_label(label, anchor)
        return label

    def _place_run_label(self, label: QGraphicsSimpleTextItem, anchor: QPointF) -> None:
        """Centred over anchor, measured from the label's current text."""
        label.setPos(anchor)
        box = label.boundingRect()
        label.setTransform(QTransform.fromTranslate(-box.width() / 2.0, -box.height() - self.RULER_LABEL_GAP_PX))
