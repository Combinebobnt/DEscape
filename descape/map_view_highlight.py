"""MapView's edit highlight: cliff and place ghosts, tile geometry, the brush highlight, shape preview and pan highlight."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import lru_cache
from typing import ClassVar

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import (
    QBrush,
    QPainterPath,
    QPen,
    QPolygonF,
)

from descape import (
    brush,
    iso_geometry,
    shape_tools,
    unit_pick,
    unit_shape,
)
from descape.viewer_common import (
    EDIT_TOOLS,
    SHAPE_TOOLS,
)

# GH #365's drag shapes: the press asks set_shape_press_query() for a latch, and they snap in unit_shape.
_LATCHED_SHAPES = frozenset({"unit_line", "unit_shape"})


# The tools with a hover ghost of the picked object (GH #128, GH #270).
_HOVER_GHOST_TOOLS = frozenset({"place_unit", "unit_brush"})


@lru_cache(maxsize=1024)
def _sloped_outline_polygon(tile_px: int, nw: int, ne: int, sw: int, se: int, coarse: bool) -> QPolygonF:
    """One Sloped tile's tile-local outline as a QPolygonF, keyed on its
    -d_min-normalized corners: the outline depends on nothing else, and a
    map has few distinct shapes (TASK-031.38). Callers must never mutate
    the returned polygon; _tile_polygon hands out a translated() copy."""
    outline_fn = iso_geometry.sloped_tile_outline_coarse if coarse else iso_geometry.sloped_tile_outline
    return QPolygonF([QPointF(px, py) for px, py in outline_fn(tile_px, nw, ne, sw, se)])


class _MapViewHighlightMixin:
    """MapView's edit highlight and tile geometry (map_view.py's docstring has the split's rules)."""

    def set_cliff_preview(self, fn: Callable[[tuple[int, int] | None], None]) -> None:
        """`fn(tile)` resolves the Cliff tool's hover ghost for the piece a
        click on `tile` would place, `fn(None)` clears it. Mid-stroke the
        drag's own run is the ghost, pushed by the stroke handler instead."""
        self._on_cliff_preview = fn

    def stroke_button(self):
        """The mouse button holding the open stroke, None between strokes."""
        return self._stroke_button

    def refresh_cliff_ghost(self) -> None:
        """Re-runs the Cliff hover preview at the last hovered tile, for a
        picker change, a release or an undo with no mouse move. A no-op unless
        Cliff is the active tool, no stroke is open and the cursor is over the map."""
        if self._cliff_hover is not None and self._tool == "cliff" and not self._stroke_active:
            self._on_cliff_preview(self._cliff_hover)

    @property
    def _cliff_hover(self) -> tuple[int, int] | None:
        """The tile the Cliff tool last hovered, None off the map or once
        another tool, mode or scene owns the cursor."""
        return self._cliff_hover_tile

    @_cliff_hover.setter
    def _cliff_hover(self, tile: tuple[int, int] | None) -> None:
        self._cliff_hover_tile = tile
        if tile is None:
            # A footprint belongs to a hover: every site that drops one drops the other.
            self._cliff_footprint = None

    def set_cliff_footprint(self, tiles) -> None:
        """The tiles the Cliff tool's highlight covers in place of the brush
        diamond: what a click would place on hover, the run's head mid-drag.
        None (nothing placeable there) is the cursor tile, as for any brush."""
        tiles = tuple(tiles) if tiles else None
        if tiles == self._cliff_footprint:
            return
        self._cliff_footprint = tiles
        if self._tool != "cliff":
            return
        if tiles is not None:
            self._update_highlight(*tiles[0])
        elif self._cliff_hover is not None:
            self._update_highlight(*self._cliff_hover)

    def refresh_place_ghost(self) -> None:
        """Re-runs the hover preview at the last hovered point with no mouse
        move: refresh_highlight()'s analogue, for a catalog pick, owner,
        off-grid or Alt change. A no-op unless Place Unit or the Unit Brush is
        the active tool and the cursor is over the view, and during a brush
        stroke, whose painted units are the preview."""
        if self._place_hover is None or self._mode != "units" or self._tool not in _HOVER_GHOST_TOOLS:
            return
        if self._stroke_active:
            return
        if self._is_shape_tool():
            # A wall pick makes Place Unit a shape drag: no ghost, and its moves
            # never update _place_hover, so the stored point goes stale.
            self._place_hover = None
            self._clear_unit_ghost()
            self._clear_unit_hover()
            return
        pos, modifiers = self._place_hover
        self._show_place_ghost(pos, modifiers, self._pos_on_map(pos))

    def _show_place_ghost(self, pos: QPointF, modifiers, on_map: bool) -> None:
        if on_map:
            self._on_place_preview(pos, modifiers)
        else:
            self._clear_unit_ghost()
            # TASK-549: the garrison host outline goes with the ghost.
            self._clear_unit_hover()

    def _is_shape_tool(self) -> bool:
        """Whether a press now would start a shape drag: a drag_shape tool,
        or Place Unit with a wall const picked."""
        return self._tool in SHAPE_TOOLS or (
            self._tool == "place_unit" and self._place_shape_query() == "wall"
        )

    def _shape_drag_live(self) -> bool:
        """Move/release routing: a live drag stays a shape drag even if the
        catalog selection changed under it since the press."""
        return self._shape_anchor is not None or self._is_shape_tool()

    def refresh_highlight(self, tile: tuple[int, int] | None) -> None:
        """Rebuilds the edit-mode hover highlight for `tile` right now,
        using whatever brush state set_brush() last set -- for a caller
        (ViewerWindow's brush size/shape widgets) that changes brush state
        without the mouse having moved, so the preview doesn't sit stale
        until the next mouseMoveEvent. tile is typically ViewerWindow's own
        _hover_tile, last reported by on_hover(). A None tile, or a tool
        with no highlight of this kind, just clears it -- same as if the
        mouse had left the map."""
        # A live shape drag owns the highlight: this method is reached from
        # the brush widgets' own change handler, so ]/[ mid-drag would
        # otherwise replace the rubber band with a brush footprint. The
        # preview is rebuilt with the new brush instead.
        if self._shape_anchor is not None:
            self._update_shape_preview()
            return
        if tile is not None and self._tool in EDIT_TOOLS:
            self._update_highlight(*tile)
        else:
            self._clear_highlight()

    def _sloped_tile_base(
        self, tile_x: int, tile_y: int
    ) -> tuple[int, int, tuple[int, int, int, int]] | None:
        """Sloped mode's per-tile placement: the screen origin to translate a
        tile-local outline by, plus its four normalized corner rises. Shared by
        _tile_polygon and _tile_edge_points so a whole-tile outline and a
        single-edge one can never disagree about where a tile sits.

        corner_rise lives on the cache rather than on MapView because it is the
        same array the cache composited these pixels from -- reading it from
        anywhere else would risk outlining against different geometry than the
        one on screen. None before a projection snapshot and a cache exist."""
        cache = self._sloped_cache()
        if self._iso_proj is None or cache is None:
            return None
        corner_rise = cache.corner_rise
        d_nw = int(corner_rise[tile_y, tile_x])
        d_ne = int(corner_rise[tile_y, tile_x + 1])
        d_sw = int(corner_rise[tile_y + 1, tile_x])
        d_se = int(corner_rise[tile_y + 1, tile_x + 1])
        ox, oy = iso_geometry.tile_screen_origin(tile_x, tile_y, 0, self._iso_proj)
        oy -= min(d_nw, d_ne, d_sw, d_se)
        return ox, oy, (d_nw, d_ne, d_sw, d_se)

    # Which two _tile_polygon vertices bound each side, per style. Stepped's
    # polygon is unit_pick.diamond_points' [N, E, S, W]; Flat's is
    # [TL, TR, BR, BL]. Both orders put the side facing grid neighbor
    # (x, y-1) first and then run screen-clockwise, so consecutive sides of a
    # region walk chain end-to-start.
    _STEPPED_EDGE_VERTICES: ClassVar[dict[str, tuple[int, int]]] = {
        "up_left": (3, 0), "up_right": (0, 1), "right": (1, 2), "left": (2, 3),
    }
    _FLAT_EDGE_VERTICES: ClassVar[dict[str, tuple[int, int]]] = {
        "up_left": (0, 1), "up_right": (1, 2), "right": (2, 3), "left": (3, 0),
    }

    def _tile_edge_points(self, tile_x: int, tile_y: int, side: str) -> list[QPointF] | None:
        """Just the one side of tile (tile_x, tile_y)'s screen outline that
        faces grid neighbor `side` (iso_geometry.tile_edge_indices' vocabulary),
        in screen-clockwise order.

        Flat and Stepped index _tile_polygon's own corners rather than
        rebuilding any geometry, so there is no second construction to drift.
        Sloped has no corners to index -- its silhouette is a pair of warped
        staircases -- so it goes through iso_geometry.sloped_tile_edge_outline,
        translated exactly as _tile_polygon translates the full outline.

        None propagates from _tile_polygon / _sloped_tile_base (defensive:
        before a projection snapshot exists)."""
        if self._terrain_style == "sloped":
            base = self._sloped_tile_base(tile_x, tile_y)
            if base is None:
                return None
            ox, oy, corners = base
            points = iso_geometry.sloped_tile_edge_outline(self._tile_pixels, side, *corners)
            return [QPointF(ox + px, oy + py) for px, py in points]
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return None
        table = (
            self._STEPPED_EDGE_VERTICES
            if self._terrain_style == "stepped"
            else self._FLAT_EDGE_VERTICES
        )
        first, second = table[side]
        # Copies: an indexed QPolygonF point references the polygon's own memory,
        # which is freed when this local goes out of scope.
        return [QPointF(polygon[first]), QPointF(polygon[second])]

    def _tile_polygon(self, tile_x: int, tile_y: int, *, coarse: bool = False) -> QPolygonF | None:
        """The on-screen footprint of tile (tile_x, tile_y) as a polygon --
        a plain axis-aligned square in Flat mode, the tile's real projected,
        elevation-displaced diamond in Stepped mode, its warped per-column
        silhouette in Sloped. Shared by the edit-mode pulsing highlight and
        Pan mode's static outline below, so the two can never disagree about
        where a tile actually is on screen. None in Stepped/Sloped mode
        before a projection snapshot (and, for Sloped, a cache) exists
        (shouldn't happen once an image is loaded -- defensive only).

        coarse (draw-perf plan item 3): Sloped only, and only for
        _update_highlight's brush footprint -- selects sloped_tile_outline_
        coarse() instead of the exact staircase. Every other caller (Pan's
        static outline, ruler) leaves this False and must keep getting the
        pixel-exact outline. Region select no longer calls this: its fill is
        its ring (_rect_ring_path), which is pinned element for element to
        _tile_edge_points, which indexes this outline's exact corners."""
        if self._terrain_style == "stepped":
            if self._iso_elevations is None or self._iso_proj is None:
                return None
            elevation = int(self._iso_elevations[tile_y, tile_x])
            ox, oy = iso_geometry.tile_screen_origin(tile_x, tile_y, elevation, self._iso_proj)
            half_w, half_h = self._iso_proj.half_w, self._iso_proj.half_h
            # The 4-point construction lives in unit_pick.diamond_points() so
            # tile highlights and unit highlights can never drift apart about
            # what shape a diamond is.
            return QPolygonF(
                [QPointF(x, y) for x, y in unit_pick.diamond_points(ox, oy, half_w, half_h)]
            )
        if self._terrain_style == "sloped":
            # Deliberately NOT unit_pick.diamond_points: a sloped tile's
            # silhouette is a pair of warped staircases, not a diamond (see
            # iso_geometry.sloped_tile_outline).
            base = self._sloped_tile_base(tile_x, tile_y)
            if base is None:
                return None
            ox, oy, corners = base
            d_min = min(corners)
            nw, ne, sw, se = (d - d_min for d in corners)
            # Integer offsets, so translated() gives the same doubles as QPointF(ox + px, oy + py).
            return _sloped_outline_polygon(self._tile_pixels, nw, ne, sw, se, coarse).translated(ox, oy)
        tp = self._tile_pixels
        x0, y0 = tile_x * tp, tile_y * tp
        return QPolygonF(
            [QPointF(x0, y0), QPointF(x0 + tp, y0), QPointF(x0 + tp, y0 + tp), QPointF(x0, y0 + tp)]
        )

    def _update_highlight(self, tile_x: int, tile_y: int) -> None:
        """The edit-mode hover highlight, covering the WHOLE brush footprint
        centered on (tile_x, tile_y) -- not just that one tile -- so it
        reads as exactly the tiles a stroke would paint from here, per-tile
        borders included (one addPolygon() per tile into a single
        QPainterPath, deliberately not .simplified(): a bounding outline
        would misrepresent a circle brush's actual footprint). Memoized on
        (tile_x, tile_y, size, shape, thickness) since this runs on every pixel of
        mouseMoveEvent -- see _highlight_key's own comment in __init__.

        Passes coarse=True to _tile_polygon (draw-perf plan item 3): in
        Sloped this brush footprint was measured at 5.6ms at brush size 9,
        almost entirely _tile_polygon's per-column exact staircase times up
        to 81 tiles. Each shape's polygon is now cached (TASK-031.38), ~0.2ms
        here; coarse still halves the path the pulse repaints. The
        approximate outline is fine here since this path only ever feeds
        display, never a pick -- unlike _update_pan_highlight below, which
        stays exact."""
        if self._tool == "cliff" and self._cliff_footprint is not None:
            # The Cliff tool's own footprint, whichever tile the cursor is on:
            # a per-move repaint or a revisited tile cannot put the diamond back.
            key = ("cliff", self._cliff_footprint)
            if key != self._highlight_key:
                self._set_highlight_tiles(list(self._cliff_footprint), key)
            return
        key = (tile_x, tile_y, self._brush_size, self._brush_shape, self._brush_thickness)
        if key == self._highlight_key:
            return
        if self._map_width is None or self._map_height is None:
            tiles = [(tile_x, tile_y)]
        else:
            tiles = brush.brush_tiles(
                tile_x, tile_y, self._brush_size, self._brush_shape, self._map_width, self._map_height,
                self._brush_thickness,
            )
        self._set_highlight_tiles(tiles, key)

    def _set_highlight_tiles(self, tiles: list[tuple[int, int]], key) -> None:
        """Paints `tiles` as the pulsing edit highlight, memoized on `key`.
        Split out of _update_highlight above so the shape tools' rubber band
        can reuse the item management and path build wholesale -- the two
        differ only in which tiles they name and how the memo key is
        composed."""
        self._highlight_key = key
        path = QPainterPath()
        for tx, ty in tiles:
            polygon = self._tile_polygon(tx, ty, coarse=True)
            if polygon is not None:
                path.addPolygon(polygon)
                # addPolygon() leaves the subpath OPEN, so the outline pen
                # drew 3 of each tile's 4 edges. Invisible until now only
                # because the fill below covers the same path.
                path.closeSubpath()
        if path.isEmpty():
            self._clear_highlight()
            return
        if self._highlight_outline_item is None:
            self._highlight_outline_item = self.scene().addPath(path, self._highlight_outline_pen)
            self._highlight_fill_item = self.scene().addPath(
                path, QPen(Qt.NoPen), QBrush(self._highlight_fill_color)
            )
        else:
            self._highlight_outline_item.setPath(path)
            self._highlight_fill_item.setPath(path)
        # A fill item created while the pulse is paused would otherwise sit
        # at Qt's default opacity of 1.0 until the stroke ends.
        self._apply_highlight_opacity()

    def _shape_span(self) -> tuple[int, int, int, int]:
        """The drag's anchor and endpoint in tile space, with Shift already
        applied. Tile space, never screen space: in Sloped the projection
        warps per corner with elevation, so a screen angle is not even
        well-defined along the line's own length, and the same drag would
        paint different tiles per view style."""
        ax, ay = self._shape_anchor
        ex, ey = self._shape_end
        # GH #365's shapes snap once, in unit_shape.lattice_cells(), on the cell delta.
        if self._shape_modifiers & Qt.ShiftModifier and self._drag_shape not in _LATCHED_SHAPES:
            dx, dy = ex - ax, ey - ay
            shape = self._drag_shape
            if shape == "line":
                dx, dy = shape_tools.snap_line_delta(dx, dy)
            elif shape == "wall":
                # 8 directions, not the line tool's 16: a wall run has no
                # shallow-angle form to snap to (see snap_wall_delta).
                dx, dy = shape_tools.snap_wall_delta(dx, dy)
            else:
                dx, dy = shape_tools.snap_square_delta(dx, dy)
            ex, ey = ax + dx, ay + dy
        return ax, ay, ex, ey

    def _brush_union(self, tiles: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Every brush footprint along `tiles`, deduped, order preserved.
        dict.fromkeys rather than a set: the committed order is what
        EditHistory records, and a set would make it arbitrary per run."""
        if self._brush_size <= 1 or self._map_width is None or self._map_height is None:
            return tiles
        out: dict[tuple[int, int], None] = {}
        for tx, ty in tiles:
            for tile in brush.brush_tiles(
                tx, ty, self._brush_size, self._brush_shape, self._map_width, self._map_height, self._brush_thickness
            ):
                out[tile] = None
        return list(out)

    def _shape_tiles(self, *, preview: bool) -> list[tuple[int, int]]:
        """The tile set this drag names. `preview=True` may return a cheaper
        approximation; the COMMITTED set (preview=False) is always exact.

        The approximation matters: one rebuild of a 480-square filled
        rectangle is ~230k _tile_polygon() calls into a single QPainterPath,
        each reading four corner_rise values in Sloped, which freezes the
        drag. A filled rectangle previews its perimeter ring instead, which
        is what the user reads as "the rectangle" anyway."""
        if self._shape_anchor is None or self._shape_end is None:
            return []
        if self._map_width is None or self._map_height is None:
            return []
        x0, y0, x1, y1 = self._shape_span()
        width, height = self._map_width, self._map_height
        shape = self._drag_shape
        if shape == "wall":
            # No approximation: a wall run
            # is bounded by the drag's own longer axis, so it can never
            # reach the tile counts a filled rectangle can. No brush union
            # either -- the tool has supports_brush=False, since a wall's
            # shape is derived from tile adjacency and a dilated piece has
            # no meaning.
            # GH #124: the preview drops blocked tiles; the commit keeps the
            # full path so the viewer can count and report them.
            return self._drop_blocked(shape_tools.wall_path_tiles(x0, y0, x1, y1, width, height), preview)
        if shape in _LATCHED_SHAPES:
            # Must stay above the terrain-rectangle fallthrough below, which
            # would pick up Draw Rectangle's Filled state and the brush union.
            return self._latched_shape_tiles((x0, y0), (x1, y1), preview)
        if shape == "line":
            core = shape_tools.line_tiles(x0, y0, x1, y1, width, height)
            if preview and len(core) * self._brush_size**2 > self.SHAPE_PREVIEW_TILE_LIMIT:
                return core
            return self._brush_union(core)
        if self._rect_filled:
            if preview:
                return shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, width, height)
            return shape_tools.rect_tiles(x0, y0, x1, y1, width, height, filled=True)
        ring = shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, width, height)
        if preview and len(ring) * self._brush_size**2 > self.SHAPE_PREVIEW_TILE_LIMIT:
            return ring
        return self._brush_union(ring)

    def _latched_shape_tiles(self, anchor, end, preview: bool) -> list[tuple[int, int]]:
        """GH #365: the latch's cells. A wall branch's are its tiles, blocked
        ones dropped from the preview as for a wall run. A unit branch
        commits the cells' low corners and previews their W x H blocks: a
        Filled shape its outline, as Draw Rectangle does, and past
        SHAPE_PREVIEW_TILE_LIMIT the cells' corner tiles alone."""
        latch = self._shape_latch
        if latch is None:
            return []
        shift = bool(self._shape_modifiers & Qt.ShiftModifier)
        width, height = self._map_width, self._map_height
        if latch.branch == unit_shape.WALL:
            return self._drop_blocked(unit_shape.lattice_cells(latch, anchor, end, shift, width, height), preview)
        if not preview:
            return unit_shape.lattice_cells(latch, anchor, end, shift, width, height)
        outline = dataclasses.replace(latch, filled=False) if latch.filled else latch
        cells = unit_shape.lattice_cells(outline, anchor, end, shift, width, height)
        if len(cells) * latch.W * latch.H > self.SHAPE_PREVIEW_TILE_LIMIT:
            return [cell for cell in cells if 0 <= cell[0] < width and 0 <= cell[1] < height]
        tiles: dict[tuple[int, int], None] = {}
        for cell in cells:
            tiles.update(dict.fromkeys(unit_shape.cell_block(latch, cell, width, height)))
        return list(tiles)

    def _drop_blocked(self, tiles: list[tuple[int, int]], preview: bool) -> list[tuple[int, int]]:
        """A wall drag's preview without the tiles latched in _drag_blocked, path order kept."""
        if not preview or not self._drag_blocked:
            return tiles
        return [tile for tile in tiles if tile not in self._drag_blocked]

    def _update_shape_preview(self) -> None:
        """Repaints the rubber band. Memoized on the same key shape
        _update_highlight uses, for the same reason: this runs on every
        pixel of a drag, and a rebuild is only warranted when the tile set
        could actually have changed."""
        if self._shape_anchor is None:
            return
        key = (
            *self._shape_span(),
            self._brush_size,
            self._brush_shape,
            self._brush_thickness,
            self._rect_filled,
            self._tool,
            # The latched shapes snap after _shape_span(), so Shift with no move must still rebuild.
            bool(self._shape_modifiers & Qt.ShiftModifier),
        )
        if key == self._highlight_key:
            return
        self._set_highlight_tiles(self._shape_tiles(preview=True), key)

    def _cancel_shape(self) -> None:
        """Drops a live drag without committing. Every exit that is not a
        left release lands here -- Escape, right button, tool/mode switch,
        a new image -- because a shape tool mutates, so "still armed after
        you switched away" is a hazard rather than the convenience it is for
        the Ruler."""
        if self._shape_anchor is None:
            return
        self._shape_anchor = None
        self._shape_end = None
        self._shape_modifiers = Qt.NoModifier
        self._drag_shape = ""
        self._drag_blocked = frozenset()
        self._shape_latch = None
        self._clear_highlight()

    def _clear_highlight(self) -> None:
        # Always resets _highlight_key too, not just on a state change that
        # routes through set_brush()/set_source() -- leaveEvent()/set_tool()
        # call this directly, and without the reset, moving back onto the
        # exact same tile with the same brush afterwards would match the
        # stale key and skip rebuilding the (now-removed) scene items.
        self._highlight_key = None
        if self._highlight_outline_item is not None:
            self.scene().removeItem(self._highlight_outline_item)
            self.scene().removeItem(self._highlight_fill_item)
            self._highlight_outline_item = None
            self._highlight_fill_item = None

    def _update_pan_highlight(self, tile_x: int, tile_y: int) -> None:
        """Pan mode's own hover cue -- a thin static outline (default black),
        no fill, no pulse (see self._pan_highlight_pen). Deliberately much
        quieter than _update_highlight()'s edit-mode glow: Pan can't mutate
        anything, so it shouldn't compete for attention the way a live
        "this is about to paint" cursor needs to."""
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return
        if self._pan_highlight_item is None:
            self._pan_highlight_item = self.scene().addPolygon(polygon, self._pan_highlight_pen)
        else:
            self._pan_highlight_item.setPolygon(polygon)

    def _clear_pan_highlight(self) -> None:
        if self._pan_highlight_item is not None:
            self.scene().removeItem(self._pan_highlight_item)
            self._pan_highlight_item = None
