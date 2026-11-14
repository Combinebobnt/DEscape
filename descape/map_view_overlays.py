"""MapView's overlays: overlay ink, region select, the mirror overlay, the ruler, grid and edge ticks, camera and analysis markers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PyQt5.QtCore import QLineF, QPointF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QGraphicsItem,
)

from descape import (
    grid_overlay,
    perf_trace,
    region_clipboard,
    ruler,
    settings,
)
from descape.viewer_canvas import (
    AnalysisMarkerItem,
    CameraMarkerItem,
    map_overlay_font,
)
from descape.viewer_common import (
    EDIT_TOOLS,
)


class _MapViewOverlaysMixin:
    """MapView's map overlays (map_view.py's docstring has the split's rules)."""

    def _rebuild_overlay_ink(self) -> None:
        """Builds every tool-overlay pen/color from settings.get_overlay_color,
        replacing what used to be class constants of the same name. Pen
        widths, cosmetic flags and the region ants' dash pattern are NOT
        settings (see settings.OVERLAY_COLORS's own RGB-only-scope comment),
        so they're reapplied here from the same literals the old class
        constants held -- miss one and it silently reverts to Qt's defaults
        (e.g. a non-cosmetic region outline that scales with zoom)."""

        def _pen(color_id: str, width: int, *, cosmetic: bool = False, dash=None) -> QPen:
            pen = QPen(QColor(settings.get_overlay_color(color_id)), width)
            if cosmetic:
                pen.setCosmetic(True)
            if dash is not None:
                pen.setDashPattern(dash)
            return pen

        self._highlight_outline_pen = _pen("highlight_outline", 0)
        self._highlight_fill_color = QColor(settings.get_overlay_color("highlight_fill"))
        self._pan_highlight_pen = _pen("pan_highlight", 0)
        self._unit_hover_pen = _pen("unit_hover", 0)
        # Deliberately NOT cosmetic -- see UNIT_SELECT_PEN_WIDTH's own comment.
        self._unit_select_pen = _pen("unit_select", self.UNIT_SELECT_PEN_WIDTH)
        self._unit_select_fill_color = QColor(settings.get_overlay_color("unit_select_fill"))
        self._unit_select_fill_color.setAlpha(self.UNIT_SELECT_FILL_ALPHA)
        self._ruler_pen = _pen("ruler_line", 0)
        self._ruler_label_color = QColor(settings.get_overlay_color("ruler_label"))
        self._ruler_label_outline = QColor(settings.get_overlay_color("ruler_label_outline"))
        self._ruler_label_outline.setAlpha(self.RULER_LABEL_OUTLINE_ALPHA)
        self._ruler_glow_color = QColor(settings.get_overlay_color("ruler_line"))
        self._region_fill_color = QColor(settings.get_overlay_color("region_fill"))
        self._region_fill_color.setAlpha(self.REGION_SELECT_FILL_ALPHA)
        self._region_outline_pen = _pen("region_outline", self.REGION_SELECT_PEN_WIDTH, cosmetic=True)
        self._region_ants_pen = _pen(
            "region_ants", self.REGION_SELECT_PEN_WIDTH, cosmetic=True, dash=self.REGION_SELECT_ANT_DASH
        )
        self._mirror_overlay_outline_pen = _pen("mirror_overlay", 0)
        self._mirror_overlay_fill_color = QColor(settings.get_overlay_color("mirror_overlay"))
        self._mirror_overlay_fill_color.setAlpha(self.MIRROR_OVERLAY_FILL_ALPHA)
        self._mirror_axis_pen = _pen("mirror_overlay", 0)
        # TASK-327: pushed into the ghost item when it is created; apply_overlay_colors() drops a live one.
        self._unit_ghost_refused_color = QColor(settings.get_overlay_color("unit_ghost_refused"))
        self._unit_ghost_garrison_color = QColor(settings.get_overlay_color("unit_ghost_garrison"))
        self._footprint_pen = _pen("footprint_outline", 0)
        self._fish_pen = _pen("fish_outline", self.FISH_OUTLINE_PEN_WIDTH, cosmetic=True)
        # Cosmetic width 0 like the footprint outline: Flat's squash
        # transform would thin a real-width line unevenly around the ellipse.
        self._range_ring_pen = _pen("range_ring", 0)
        self._range_ring_trigger_pen = _pen("range_ring", 0, dash=self.RANGE_RING_TRIGGER_DASH)
        # Keyed on "full strength": True for the emphasised entry, False for its siblings.
        self._trigger_outline_pens = {}
        self._trigger_run_pens = {}
        self._trigger_unit_pens = {}
        self._trigger_fill_colors = {}
        for strong in (True, False):
            width = self.TRIGGER_PEN_WIDTH if strong else 1
            for pens, color_id in (
                (self._trigger_outline_pens, "trigger_area_outline"),
                (self._trigger_run_pens, "trigger_run"),
                (self._trigger_unit_pens, "trigger_unit_ref"),
            ):
                pen = _pen(color_id, width, cosmetic=True)
                if not strong:
                    color = pen.color()
                    color.setAlpha(self.TRIGGER_DIM_PEN_ALPHA)
                    pen.setColor(color)
                pens[strong] = pen
            fill = QColor(settings.get_overlay_color("trigger_area_fill"))
            fill.setAlpha(self.TRIGGER_FILL_ALPHA if strong else self.TRIGGER_DIM_FILL_ALPHA)
            self._trigger_fill_colors[strong] = fill

    def apply_overlay_colors(self) -> None:
        """Settings > Appearance's re-entry point for a color change: persist
        then push, mirroring set_edge_tick_interval's shape. Every push below
        is guarded on self.scene() AND the relevant item(s) being alive --
        the Settings dialog is reachable with no map open, and scene().clear()
        (File > Close) destroys every scene item while leaving these Python
        refs pointing at the dangling C++ object (the same hazard
        _forget_ruler_items exists for). Checking one representative item per
        atomically-created/-cleared group is enough: every group here is
        created and forgotten together, the same invariant _clear_highlight's
        own single-item check already relies on."""
        self._rebuild_overlay_ink()
        if self.scene() is None:
            return
        if self._pan_highlight_item is not None:
            self._pan_highlight_item.setPen(self._pan_highlight_pen)
        if self._unit_hover_item is not None:
            self._unit_hover_item.setPen(self._unit_hover_pen)
        # Only the configured group: owner groups don't depend on overlay settings.
        configured = self._unit_select_groups.get(None)
        if configured is not None:
            fill_item, _under_item, outline_item = configured
            outline_item.setPen(self._unit_select_pen)
            fill_item.setBrush(QBrush(self._unit_select_fill_color))
        if self._marquee_item is not None:
            self._marquee_item.setPen(self._unit_select_pen)
            self._marquee_item.setBrush(QBrush(self._unit_select_fill_color))
        if self._ruler_line_item is not None:
            self._ruler_line_item.setPen(self._ruler_pen)
            for item in self._ruler_end_items:
                item.setPen(self._ruler_pen)
            for item in self._ruler_glow_items:
                item.setBrush(QBrush(self._ruler_glow_color))
            self._ruler_label_item.setBrush(QBrush(self._ruler_label_color))
            self._ruler_label_item.setPen(QPen(self._ruler_label_outline, 0))
        if self._pinned_ruler_items:
            self._rebuild_pinned_rulers()
        if self._region_fill_item is not None:
            self._region_fill_item.setBrush(QBrush(self._region_fill_color))
            self._region_outline_item.setPen(self._region_outline_pen)
            # Carries the ants' current phase forward -- a rebuilt pen would
            # otherwise reset dashOffset to 0 and visibly restart the crawl.
            ants_pen = QPen(self._region_ants_pen)
            ants_pen.setDashOffset(self._region_ants_item.pen().dashOffset())
            self._region_ants_item.setPen(ants_pen)
        if self._stack_badge_item is not None:
            self._stack_badge_item.set_color(QColor(settings.get_overlay_color("unit_stack")))
            self._stack_badge_item.set_background_color(QColor(settings.get_overlay_color("unit_stack_background")))
        if self._analysis_marker_item is not None:
            self._analysis_marker_item.set_color(QColor(settings.get_overlay_color("analysis_marker")))
        if self._trigger_items:
            # Rebuilt rather than re-inked: at most a handful of items, two pen strengths.
            self._rebuild_trigger_overlay()
        if None in self._footprint_items:
            # Owner groups draw in player colours, which this setting does not reach.
            self._footprint_items[None].setPen(self._footprint_pen)
        if self._fish_item is not None:
            self._fish_item.setPen(self._fish_pen)
        if None in self._range_ring_items:
            # Owner groups draw in player colours, which this setting does not reach.
            self._range_ring_items[None].setPen(self._range_ring_pen)
        if None in self._trigger_ring_items:
            self._trigger_ring_items[None].setPen(self._range_ring_trigger_pen)
        if None in self._ring_layer_items:
            # The layer may hold only owner groups, hence the membership test.
            self._ring_layer_items[None].setPen(self._range_ring_pen)
        if self._mirror_overlay_outline_item is not None:
            self._mirror_overlay_outline_item.setPen(self._mirror_overlay_outline_pen)
            self._mirror_overlay_fill_item.setBrush(QBrush(self._mirror_overlay_fill_color))
            for item in self._mirror_axis_items:
                item.setPen(self._mirror_axis_pen)
        if self._unit_ghost_item is not None:
            # The next move or hover rebuilds it through _ensure_unit_ghost_item(), in the new refusal colour.
            self._clear_unit_ghost()
        # The edit highlight has no in-place update path worth writing (it
        # rebuilds on every hover move anyway) -- clearing it here just forces
        # that rebuild to happen with the new ink instead of leaving a stale
        # color showing until the mouse next moves.
        self._clear_highlight()

    def _overlay_opacity(self, group_id: str) -> float:
        """One settings.OVERLAY_OPACITIES group's multiplier, 0.1-1.0: a live
        slider override if one is pending, else the persisted percent."""
        pct = self._overlay_opacity_overrides.get(group_id)
        if pct is None:
            pct = settings.get_overlay_opacity(group_id)
        return pct / 100.0

    def apply_overlay_opacity(self, overrides: dict[str, int] | None = None) -> None:
        """Settings > Appearance's opacity sliders (GH #129). `overrides` maps a
        group to an unpersisted percent mid-drag; None means "persisted", and
        clears them. Sets opacity in place on every live item of the three
        groups, never rebuilding, with apply_overlay_colors()'s guards: the
        refs below are None whenever their items are gone."""
        self._overlay_opacity_overrides = dict(overrides) if overrides else {}
        if self.scene() is None:
            return
        unit = self._overlay_opacity("unit_select")
        for group in self._unit_select_groups.values():
            for item in group:
                if item is not None:
                    item.setOpacity(unit)
        if self._marquee_item is not None:
            self._marquee_item.setOpacity(unit)
        if self._region_fill_item is not None:
            region = self._overlay_opacity("region")
            for item in (self._region_fill_item, self._region_outline_item, self._region_ants_item):
                item.setOpacity(region)
        trigger = self._overlay_opacity("trigger_area")
        for item in self._trigger_items:
            if str(item.data(0)).startswith(("fill_", "outline_")):
                item.setOpacity(trigger)

    def apply_ruler_label_font(self) -> None:
        """Settings > Appearance's re-entry point for the ruler label
        font-size spinbox -- apply_overlay_colors()'s sibling, same
        deleted-item guard (the Settings dialog is reachable with no map
        open). Pinned rulers are rebuilt; the live half is a no-op with no
        live label: _create_ruler_items() already reads
        settings.get_ruler_label_font_px() directly, so the next measurement
        picks up the new size with no extra plumbing needed."""
        if self.scene() is None:
            return
        # Before the live-label guard: pinned labels exist without a live one.
        self._rebuild_pinned_rulers()
        if self._ruler_label_item is None:
            return
        # Rebuilt from map_overlay_font rather than the item's own font, so
        # this stays the carve-out font even if a live app-font change ever
        # reaches the item some other way. Bold is re-applied, not inherited.
        font = map_overlay_font(settings.get_ruler_label_font_px())
        font.setBold(True)
        self._ruler_label_item.setFont(font)
        self._update_ruler()

    def apply_distance_tick_font(self) -> None:
        """Settings > Appearance's re-entry point for the distance-tick
        label font-size spinbox -- set_edge_tick_interval's persist-then-push
        shape. Unlike apply_ruler_label_font, the pad also has to be
        recomputed (edge_ticks.scene_pad depends on font_px), so this
        re-runs _repad_edge_ticks() rather than duplicating its
        floor/current-scale min() here."""
        if self._edge_tick_item is None:
            return
        self._edge_tick_item.set_label_font_px(settings.get_distance_tick_font_px())
        self._repad_edge_ticks()

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        """The one mutator for the committed region -- called from MapView's
        own drag-release/Escape-clear paths, and by ViewerWindow directly for
        Select All/Deselect, so there is exactly one place that rebuilds the
        overlay from a new value. Idempotent: setting the same region twice
        just rebuilds the same path again."""
        self._region = region
        self._update_region_overlay()

    def set_region_movable(self, movable: bool) -> None:
        """The window's word on whether the committed region is the result of
        a paste that can still be re-placed. MapView never decides this
        itself -- it only turns the flag into press-inside vs press-outside."""
        self._region_movable = bool(movable)
        if not self._region_movable and self._move_anchor is not None:
            self._move_anchor = None
            self._move_current = None
            # Only a move drag ends here: a paste disarmed mid-select-drag keeps that drag's trace.
            self._close_region_drag_trace()
        self._update_region_overlay()

    def _tile_in_region(self, tile) -> bool:
        if tile is None or self._region is None:
            return False
        tx0, ty0, tx1, ty1 = self._region
        return tx0 <= tile[0] < tx1 and ty0 <= tile[1] < ty1

    def _region_edge_walk(self, tx0: int, ty0: int, tx1: int, ty1: int):
        """The half-open rectangle's OUTER boundary as (tile_x, tile_y, side)
        triples, screen-clockwise in one closed ring -- only the edges facing
        outside the region, never an interior seam between two boundary tiles.

        Four chains, each a straight run along one map axis emitting the one
        side that faces off-region there:

            row ty0,     x ascending  -> up_left   (faces y-1)
            col tx1-1,   y ascending  -> up_right  (faces x+1)
            row ty1-1,   x descending -> right     (faces y+1)
            col tx0,     y descending -> left      (faces x-1)

        The ring closes exactly. Within a chain, consecutive tiles' emitted
        endpoints coincide in Flat and share screen-x in Stepped (differing in
        y only by the elevation step, which is precisely the vertical riser a
        stepped silhouette wants). Chain-to-chain transitions land on the SAME
        tile's shared vertex -- chain 1 ends at (tx1-1, ty0)'s N, chain 2
        starts there -- so they are zero-length. Sloped abuts to within the 1px
        apex overlap sloped_tile_edge_outline documents.

        Degenerate rects need no special case: 1x1 emits all four sides of the
        one tile, and 1xN / Nx1 emit two long sides plus two caps. A tile
        appearing in more than one chain under DIFFERENT sides is the point,
        which is why this has none of the old _region_boundary_tiles'
        de-duplication guards.

        Still O(perimeter), never O(area): a Select region can span the whole
        map, unlike a brush footprint."""
        for x in range(tx0, tx1):
            yield x, ty0, "up_left"
        for y in range(ty0, ty1):
            yield tx1 - 1, y, "up_right"
        for x in range(tx1 - 1, tx0 - 1, -1):
            yield x, ty1 - 1, "right"
        for y in range(ty1 - 1, ty0 - 1, -1):
            yield tx0, y, "left"

    @staticmethod
    def _region_edge_chains(tx0: int, ty0: int, tx1: int, ty1: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
        """_region_edge_walk()'s four chains as (xs, ys, side) int64 arrays, in
        the same order; see its docstring for the order and why the ring closes."""
        xs = np.arange(tx0, tx1, dtype=np.int64)
        ys = np.arange(ty0, ty1, dtype=np.int64)
        return [
            (xs, np.full_like(xs, ty0), "up_left"),
            (np.full_like(ys, tx1 - 1), ys, "up_right"),
            (xs[::-1], np.full_like(xs, ty1 - 1), "right"),
            (np.full_like(ys, tx0), ys[::-1], "left"),
        ]

    def _region_render_rect(self) -> tuple[int, int, int, int] | None:
        """What the overlay should show right now: the live anchor-to-current
        preview during a drag, else the committed region -- so starting a new
        drag previews without disturbing self._region until the drag actually
        commits (Escape-cancel then just re-renders the untouched committed
        value)."""
        if self._move_anchor is not None and self._move_current is not None and self._region is not None:
            return region_clipboard.translated_region(
                self._region,
                self._move_current[0] - self._move_anchor[0],
                self._move_current[1] - self._move_anchor[1],
                self._map_width or 0,
                self._map_height or 0,
            )
        if self._select_anchor is not None and self._select_current is not None:
            return region_clipboard.normalize_region(
                self._select_anchor, self._select_current, self._map_width or 0, self._map_height or 0
            )
        return self._region

    def _update_region_overlay(self) -> None:
        """Rebuilds the region's outline/ants/fill from _region_render_rect().
        Outline and ants share one QPainterPath: a single continuous ring
        built in numpy by _rect_ring_path() in _region_edge_walk()'s order, so it conforms to
        Stepped columns and Sloped warping while landing ink only on the
        selection's true silhouette. Adding each perimeter tile's WHOLE
        polygon instead would stroke every seam between two adjacent boundary
        tiles, which is what the ants used to crawl along. Still O(perimeter),
        never O(area).

        The fill is that same ring under Qt.WindingFill (TASK-031.37), so it
        is O(perimeter) to build and to paint, which matters because every 80
        ms ants tick repaints the ring's bounding rect. It is the trigger-area
        previews' existing look: in Stepped it tints the risers inside the
        outline evenly, where a per-tile union under OddEvenFill punched holes
        wherever neighbouring diamonds overlap. WindingFill keeps every lobe
        tinted if extreme elevation makes the ring cross itself."""
        rect = self._region_render_rect()
        boundary_path = QPainterPath() if rect is None else self._rect_ring_path(rect)
        if boundary_path.isEmpty():
            self._clear_region_overlay()
            return
        boundary_path.setFillRule(Qt.WindingFill)
        fill_path = boundary_path
        if self._region_fill_item is None:
            scene = self.scene()
            self._region_fill_item = scene.addPath(fill_path, QPen(Qt.NoPen), QBrush(self._region_fill_color))
            self._region_outline_item = scene.addPath(boundary_path, self._region_outline_pen)
            self._region_ants_item = scene.addPath(boundary_path, self._region_ants_pen)
            region_opacity = self._overlay_opacity("region")
            for item in (self._region_fill_item, self._region_outline_item, self._region_ants_item):
                item.setZValue(self.REGION_SELECT_Z)
                item.setOpacity(region_opacity)
        else:
            self._region_fill_item.setPath(fill_path)
            self._region_outline_item.setPath(boundary_path)
            self._region_ants_item.setPath(boundary_path)
        # The items sit at the scene origin untransformed, so the path's own
        # bounds are already scene-space.
        self._region_scene_rect = boundary_path.boundingRect()
        self._region_overlay_rect = rect
        self._sync_region_ant_timer()

    def _clear_region_overlay(self) -> None:
        if self._region_fill_item is not None:
            scene = self.scene()
            scene.removeItem(self._region_fill_item)
            scene.removeItem(self._region_outline_item)
            scene.removeItem(self._region_ants_item)
            self._region_fill_item = None
            self._region_outline_item = None
            self._region_ants_item = None
        self._region_scene_rect = None
        self._region_overlay_rect = None
        self._region_ant_timer.stop()

    def _refresh_live_region(self) -> None:
        """A live Select drag's move: rebuilds only when the drawn rect changed
        (a move clamped at the map edge draws the same rect for several tiles),
        as one Perf Trace step. Every other caller rebuilds unconditionally."""
        if self._region_render_rect() == self._region_overlay_rect:
            return
        self._traced_region_rebuild()

    def _traced_region_rebuild(self) -> None:
        with perf_trace.span("step"), perf_trace.phase("region_overlay"):
            self._update_region_overlay()
        perf_trace.step()

    def _close_region_drag_trace(self) -> None:
        """Ends a live Select drag's Perf Trace drag once; a no-op with none open."""
        label = self._region_drag_label
        if label is not None:
            self._region_drag_label = None
            perf_trace.end_drag(label)

    def _region_on_screen(self) -> bool:
        """Whether the region overlay's own scene rect intersects the visible
        viewport rect. Deliberately a rect test, not a per-tile visibility
        one: a region straddling the edge keeps its ants running, which is
        what the visible part of the outline needs.

        Reads the remembered _region_scene_rect rather than asking the item
        for its own bounds. This runs from _note_viewport_changed(), which
        clear_image()/set_source() both reach in the window between
        scene().clear() destroying the item and the Python-side reference
        being dropped, where any call on the item raises."""
        if self._region_ants_item is None or self._region_scene_rect is None:
            return False
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        return self._region_scene_rect.intersects(visible)

    def _sync_region_ant_timer(self) -> None:
        """The ants only cost anything while a region is actually shown, and
        only earn it while that region is on screen. An off-screen one
        dirties a scene rect every 80ms that nothing can see (perf findings
        item 21). Re-checked from _note_viewport_changed(), so scrolling the
        region back into view starts them again."""
        if self._region_on_screen():
            if not self._region_ant_timer.isActive():
                self._region_ant_timer.start()
        else:
            self._region_ant_timer.stop()

    def _advance_region_ants(self) -> None:
        total = sum(self.REGION_SELECT_ANT_DASH)
        self._region_ant_offset = (self._region_ant_offset + self.REGION_SELECT_ANT_STEP_PX) % total
        if self._region_ants_item is not None:
            pen = self._region_ants_item.pen()
            pen.setDashOffset(self._region_ant_offset)
            self._region_ants_item.setPen(pen)

    def show_mirror_overlay(
        self, tiles: list[tuple[int, int]], axes: list[tuple[QPointF, QPointF]]
    ) -> None:
        """MirrorDialog's live preview: shades `tiles` (the exact source
        slice -- MirrorPlan.source_indices, not an approximation) via
        _tile_polygon, same one-QPainterPath-per-call shape as
        _update_highlight(), plus zero or more straight `axes` line items
        (Flat style only -- see MirrorDialog._axis_lines' docstring for why
        Stepped/Sloped skip the line and rely on the shading alone).
        Rebuilt wholesale on every call rather than diffed in place, since
        MirrorDialog calls this on every option change and a whole-dialog
        rebuild is cheap next to plan_mirror() itself."""
        self.clear_mirror_overlay()
        path = QPainterPath()
        for tx, ty in tiles:
            polygon = self._tile_polygon(tx, ty)
            if polygon is not None:
                path.addPolygon(polygon)
                # addPolygon() leaves the subpath OPEN -- see _set_highlight_tiles.
                path.closeSubpath()
        if not path.isEmpty():
            self._mirror_overlay_outline_item = self.scene().addPath(
                path, self._mirror_overlay_outline_pen
            )
            self._mirror_overlay_fill_item = self.scene().addPath(
                path, QPen(Qt.NoPen), QBrush(self._mirror_overlay_fill_color)
            )
            self._mirror_overlay_outline_item.setZValue(self.MIRROR_OVERLAY_Z)
            self._mirror_overlay_fill_item.setZValue(self.MIRROR_OVERLAY_Z)
        for a, b in axes:
            item = self.scene().addLine(a.x(), a.y(), b.x(), b.y(), self._mirror_axis_pen)
            item.setZValue(self.MIRROR_OVERLAY_Z)
            self._mirror_axis_items.append(item)

    def clear_mirror_overlay(self) -> None:
        if self._mirror_overlay_outline_item is not None:
            self.scene().removeItem(self._mirror_overlay_outline_item)
            self.scene().removeItem(self._mirror_overlay_fill_item)
            self._mirror_overlay_outline_item = None
            self._mirror_overlay_fill_item = None
        for item in self._mirror_axis_items:
            self.scene().removeItem(item)
        self._mirror_axis_items = []

    def _ruler_anchor(self, tile: tuple[int, int]) -> QPointF | None:
        """The point a Ruler endpoint pins to: the centre of that tile's real
        on-screen footprint, so it sits at the elevation the user actually
        clicked in Stepped and Sloped.

        Deliberately NOT the elevation-0 ground plane edge_ticks anchors to.
        The ticks want a straight edge that no border tile can make ragged; an
        endpoint wants to be under the cursor, and one pinned to elevation 0
        would float below it on raised terrain and read as a bug. The measured
        distance is the other half of that split and stays purely 2D, per
        descape.ruler's module docstring.

        Going through _tile_polygon rather than tile_screen_origin is what
        makes this work in all three Elevation Views with no geometry of its
        own, Sloped's warped silhouette included."""
        polygon = self._tile_polygon(*tile)
        return None if polygon is None else polygon.boundingRect().center()

    def _create_ruler_items(self) -> None:
        """Four stock scene items, no QGraphicsItem subclass. Only the label
        needs ItemIgnoresTransformations: it draws in device coordinates, so
        it survives Flat+Isometric's scale(1, 0.5) + rotate(-45) upright and
        at a constant size, where scene-space text would be sheared and
        sub-pixel at fit-to-view."""
        scene = self.scene()
        self._ruler_line_item = scene.addLine(QLineF(), self._ruler_pen)
        self._ruler_line_item.setZValue(self.RULER_Z)
        self._ruler_end_items = []
        self._ruler_glow_items = []
        for _ in range(2):
            glow = scene.addPolygon(QPolygonF(), QPen(Qt.NoPen), QBrush(self._ruler_glow_color))
            glow.setZValue(self.RULER_GLOW_Z)
            # Seeded from the live phase, so a new measurement joins the
            # breath already in progress instead of flashing at full alpha
            # for one frame.
            glow.setOpacity(self._ruler_pulse_alpha())
            self._ruler_glow_items.append(glow)
            item = scene.addPolygon(QPolygonF(), self._ruler_pen)
            item.setZValue(self.RULER_Z)
            self._ruler_end_items.append(item)
        label = scene.addSimpleText("")
        # map_overlay_font, not label.font(): the latter is the app chrome
        # font, so the family silently followed apply_ui_font()'s setting.
        font = map_overlay_font(settings.get_ruler_label_font_px())
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(self._ruler_label_color))
        label.setPen(QPen(self._ruler_label_outline, 0))
        label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        label.setZValue(self.RULER_LABEL_Z)
        self._ruler_label_item = label

    def _update_ruler(self) -> None:
        """Rebuilds the overlay from the session's two tiles. Items are
        updated in place via setLine/setPolygon/setText rather than removed
        and re-added, the same convention _update_highlight follows."""
        ends = self._ruler.endpoints
        if ends is None:
            self._clear_ruler()
            return
        anchor_a = self._ruler_anchor(ends[0])
        anchor_b = self._ruler_anchor(ends[1])
        if anchor_a is None or anchor_b is None:
            # No projection snapshot yet, so there is nowhere to draw. The
            # session is left alone: the tiles are still valid.
            return
        if self._ruler_line_item is None:
            self._create_ruler_items()
            self._sync_pulse_timer()
        self._ruler_line_item.setLine(QLineF(anchor_a, anchor_b))
        for i in range(2):
            polygon = self._tile_polygon(*ends[i])
            if polygon is not None:
                self._ruler_end_items[i].setPolygon(polygon)
                self._ruler_glow_items[i].setPolygon(polygon)
        self._ruler_label_item.setText(ruler.format_measurement(self._ruler.measurement))
        self._position_ruler_label(anchor_b)
        self._on_ruler_changed(self._ruler.measurement)

    def _position_ruler_label(self, anchor_b: QPointF) -> None:
        """setPos places the label's origin at `anchor_b` in SCENE space; the
        translation is then applied in the label's own coordinate system,
        which ItemIgnoresTransformations keeps in DEVICE pixels. That is what
        lets a device-pixel gap sit the text above the point at any zoom, and
        why boundingRect() here is already in device units.

        Anchored to `anchor_b` (the tile the mouse is currently on, or the
        one it last set the measurement to) rather than the segment midpoint:
        on a small window a long measurement's midpoint can sit off-screen
        with no readout visible at all, where `anchor_b` is always the point
        the user was just looking at."""
        self._ruler_label_item.setPos(anchor_b)
        rect = self._ruler_label_item.boundingRect()
        self._ruler_label_item.setTransform(
            QTransform.fromTranslate(-rect.width() / 2.0, -rect.height() - self.RULER_LABEL_GAP_PX)
        )

    def _report_ruler(self, previous_state: str) -> None:
        """Fires the host's callback only on the edge into DONE, so a finished
        measurement logs exactly once whichever gesture completed it, a still
        pending one logs nothing, and starting a fresh measurement from a
        finished one does not re-log the old result."""
        if previous_state != ruler.STATE_DONE and self._ruler.state == ruler.STATE_DONE:
            measurement = self._ruler.measurement
            self._on_ruler_measured(measurement)
            # GH #108/#109: the finished measurement becomes a pinned ruler and
            # the live session goes back to IDLE. Zero length is logged, not pinned.
            index = len(self._pinned_rulers)
            pinned = self._pinned_rulers.add(measurement)
            self._remove_live_ruler()
            if pinned:
                self._pinned_rulers_changed()
                self.on_pinned_rulers_edited("Add ruler", ((index, measurement),), ())
            else:
                self._emit_ruler_status()

    def _remove_live_ruler(self) -> None:
        """Drops the live session and its items, leaving pinned rulers and the
        status bar alone."""
        self._ruler.clear()
        if self._ruler_line_item is not None:
            self.scene().removeItem(self._ruler_line_item)
            for item in self._ruler_end_items + self._ruler_glow_items:
                self.scene().removeItem(item)
            self.scene().removeItem(self._ruler_label_item)
        self._forget_ruler_items()

    def _clear_ruler(self) -> None:
        """Drops the live measurement AND its items; pinned rulers stay. Removes
        from the scene explicitly, unlike the clear_image/set_source paths where
        scene().clear() has already destroyed the C++ objects and only the
        Python-side references need forgetting."""
        self._remove_live_ruler()
        self._emit_ruler_status()

    def _emit_ruler_status(self) -> None:
        """The status-bar rule in one place: the live measurement while one is
        pending, otherwise the newest pinned one, otherwise None."""
        if self._ruler.state != ruler.STATE_IDLE:
            self._on_ruler_changed(self._ruler.measurement)
        else:
            self._on_ruler_changed(self._pinned_rulers.newest)

    def _pinned_rulers_changed(self) -> None:
        self._rebuild_pinned_rulers()
        self._emit_ruler_status()
        self.on_pinned_rulers_changed(len(self._pinned_rulers))

    def pinned_ruler_count(self) -> int:
        return len(self._pinned_rulers)

    def clear_rulers(self) -> None:
        """The Clear rulers button: the live measurement and every pinned one."""
        self._remove_live_ruler()
        before = tuple(self._pinned_rulers)
        self._pinned_rulers.clear()
        if before:
            self._pinned_rulers_changed()
            self.on_pinned_rulers_edited("Clear rulers", (), tuple(enumerate(before)))
        else:
            self._emit_ruler_status()

    def apply_ruler_delta(self, remove, insert) -> None:
        """Undo/redo's way in (GH #108): PinnedRulers.apply_delta, without
        firing on_pinned_rulers_edited."""
        self._pinned_rulers.apply_delta(remove, insert)
        self._pinned_rulers_changed()

    def _drop_pinned_rulers(self) -> None:
        """For a new document or File > Close, after scene().clear() destroyed the items."""
        self._pinned_ruler_items = []
        if len(self._pinned_rulers):
            self._pinned_rulers.clear()
            self.on_pinned_rulers_changed(0)

    def _rebuild_pinned_rulers(self) -> None:
        """Rebuilds every pinned ruler against the CURRENT projection, the
        _rebuild_trigger_overlay pattern: one path for every line and endpoint
        outline, one device-space label per ruler anchored at its b end."""
        scene = self.scene()
        if scene is None:
            return
        for item in self._pinned_ruler_items:
            scene.removeItem(item)
        self._pinned_ruler_items = []
        if not len(self._pinned_rulers):
            return
        path = QPainterPath()
        labels: list[tuple[QPointF, str]] = []
        for m in self._pinned_rulers:
            # Guards a future map resize: a ruler off the map is skipped, not drawn.
            if not (self._on_map(m.a) and self._on_map(m.b)):
                continue
            anchor_a, anchor_b = self._ruler_anchor(m.a), self._ruler_anchor(m.b)
            if anchor_a is None or anchor_b is None:
                continue
            path.moveTo(anchor_a)
            path.lineTo(anchor_b)
            for tile in (m.a, m.b):
                polygon = self._tile_polygon(*tile)
                if polygon is not None:
                    path.addPolygon(polygon)
                    path.closeSubpath()
            labels.append((anchor_b, ruler.format_measurement(m)))
        if not path.isEmpty():
            item = scene.addPath(path, self._ruler_pen)
            item.setZValue(self.PINNED_RULER_Z)
            self._pinned_ruler_items.append(item)
        font = map_overlay_font(settings.get_ruler_label_font_px())
        font.setBold(True)
        for anchor, text in labels:
            label = scene.addSimpleText(text)
            label.setFont(font)
            label.setBrush(QBrush(self._ruler_label_color))
            label.setPen(QPen(self._ruler_label_outline, 0))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            label.setZValue(self.PINNED_RULER_LABEL_Z)
            label.setPos(anchor)
            box = label.boundingRect()
            label.setTransform(QTransform.fromTranslate(-box.width() / 2.0, -box.height() - self.RULER_LABEL_GAP_PX))
            self._pinned_ruler_items.append(label)
        # Pinned items never take a click: Units-mode selection and the Ruler's own presses go through.
        for item in self._pinned_ruler_items:
            item.setAcceptedMouseButtons(Qt.NoButton)

    def _forget_ruler_items(self) -> None:
        """The one choke point every clear gesture funnels through, which is
        why the pulse timer is re-synced here rather than per gesture: a timer
        still ticking after scene().clear() would call setOpacity() on a
        destroyed C++ object."""
        self._ruler_line_item = None
        self._ruler_end_items = []
        self._ruler_glow_items = []
        self._ruler_label_item = None
        self._sync_pulse_timer()

    def _on_pulse_tick(self) -> None:
        # Phase advances unconditionally, so the animation is free-running
        # wall-clock rather than resuming mid-breath where it left off.
        self._pulse_phase_ms = (self._pulse_phase_ms + self.HIGHLIGHT_PULSE_TICK_MS) % (
            self.HIGHLIGHT_PULSE_PERIOD_MS
        )
        self._apply_highlight_opacity()
        self._apply_ruler_glow_opacity()

    def _sync_pulse_timer(self) -> None:
        """The pulse runs while an edit tool is active AND no stroke is in
        progress. Every tick dirties the highlight's scene rect, which
        re-enters MapCanvasItem.paint (perf findings item 21), so during a
        stroke it competes with the edit work for the same frames. The
        cursor is not resting there to be found anyway.

        Paused rather than merely slowed, so the highlight has to hold a
        steady value while it waits: see _pulse_alpha()."""
        editing = self._tool in EDIT_TOOLS and not self._stroke_active
        if editing or self._ruler_glow_items:
            if not self._pulse_timer.isActive():
                self._pulse_timer.start(self.HIGHLIGHT_PULSE_TICK_MS)
        else:
            self._pulse_timer.stop()
        self._apply_highlight_opacity()
        self._apply_ruler_glow_opacity()

    def _pulse_alpha(self) -> float:
        """The fill opacity for the current phase, or the steady mid-pulse
        value while the pulse is paused. A paused highlight must read as a
        solid cursor, never as a frozen fade-out at whichever phase the
        press happened to interrupt."""
        return self._phase_alpha(self.HIGHLIGHT_PULSE_MIN_ALPHA, self.HIGHLIGHT_PULSE_MAX_ALPHA)

    def _ruler_pulse_alpha(self) -> float:
        return self._phase_alpha(self.RULER_PULSE_MIN_ALPHA, self.RULER_PULSE_MAX_ALPHA)

    def _phase_alpha(self, min_alpha: float, max_alpha: float) -> float:
        mid = (min_alpha + max_alpha) / 2
        if not self._pulse_timer.isActive():
            return mid
        amplitude = (max_alpha - min_alpha) / 2
        t = self._pulse_phase_ms / self.HIGHLIGHT_PULSE_PERIOD_MS
        return mid + amplitude * math.sin(2 * math.pi * t)

    def _apply_highlight_opacity(self) -> None:
        if self._highlight_fill_item is not None:
            self._highlight_fill_item.setOpacity(self._pulse_alpha())

    def _apply_ruler_glow_opacity(self) -> None:
        alpha = self._ruler_pulse_alpha()
        for item in self._ruler_glow_items:
            item.setOpacity(alpha)

    def _repad_edge_ticks(self) -> None:
        """Hands the tick and grid overlays the smallest scale the view can currently
        reach, which is what its bounding-rect pad has to cover.

        min() of the floor and the CURRENT scale, and neither term subsumes
        the other: resizeEvent re-runs _capture_zoom_baseline, and a larger
        viewport raises _min_linear_scale without rescaling the transform, so
        maximising the window while zoomed to the old floor leaves the
        current scale below the new minimum. wheelEvent needs no hook of its
        own: zooming in only raises the scale, and zooming out is refused
        below _min_linear_scale, which this already covers."""
        current = abs(self.transform().determinant()) ** 0.5
        floor = self._min_linear_scale
        scale = min(floor, current) if floor is not None else current
        if self._edge_tick_item is not None:
            self._edge_tick_item.set_min_view_scale(scale)
        if self._grid_item is not None:
            self._grid_item.set_min_view_scale(scale)
        if self._analysis_marker_item is not None:
            self._analysis_marker_item.set_min_view_scale(scale)

    def set_edge_ticks(self, enabled: bool) -> None:
        """Shows or hides the map-edge distance ruler. Never rebuilds a
        chunk cache: unlike Show sprites, these marks are not baked into
        canvas pixels, so there is nothing to evict."""
        self._edge_ticks_enabled = enabled
        if self._edge_tick_item is not None:
            self._edge_tick_item.setVisible(enabled)

    def grid_bake_live(self) -> bool:
        """Whether View > Grid is drawn inside the chunk composite (under the
        sprites) rather than by GridItem. Follow Terrain Elevation off in an
        iso style is a flat elevation-0 lattice, not per-tile geometry, so it
        stays an overlay; Flat ignores the setting and always bakes."""
        return self._grid_enabled and (self._terrain_style == "flat" or self._grid_follow_elevation)

    def grid_spec(self) -> grid_overlay.GridBake:
        if not self.grid_bake_live() or self._grid_previewing:
            return grid_overlay.DEFAULT_GRID
        return grid_overlay.grid_bake(True, self._grid_blend, self._grid_thickness, self._grid_major_every)

    def grid_change_evicts(self) -> bool:
        """Whether the state just stored would change the cache's baked spec,
        i.e. whether applying it costs a full-canvas eviction."""
        cache = self._canvas_item._cache if self._canvas_item is not None else None
        return cache is not None and cache.grid != self.grid_spec()

    def _apply_grid(self) -> bool:
        """Pushes the grid state to both halves: the cache's baked spec and
        GridItem's visibility. Returns whether the cache evicted. The bake
        costs one full-canvas eviction per real change, so every caller that
        can reach one goes through ViewerWindow._apply_grid_change()."""
        if self._grid_item is not None:
            self._grid_item.setVisible(self._grid_enabled and (self._grid_previewing or not self.grid_bake_live()))
        if not self.grid_change_evicts():
            return False
        cache = self._canvas_item._cache
        cache.set_grid(self.grid_spec())
        self.invalidate_region((0, 0, *cache.canvas_dims(0)))
        return True

    def set_grid_overlay(self, enabled: bool) -> bool:
        """Shows or hides View > Grid. Evicts the chunk cache whenever the
        bake is (or was) live; returns whether it did."""
        self._grid_enabled = enabled
        return self._apply_grid()

    def set_grid_follow_elevation(self, enabled: bool) -> bool:
        """In an iso style this swaps between the bake and the overlay, so
        it is an eviction site like the toggle itself."""
        self._grid_follow_elevation = enabled
        return self._apply_grid()

    def set_grid_major_every(self, major_every: int) -> bool:
        """View > Grid's stronger-line spacing, for both halves. Evicts only
        when the bake is live; returns whether it did."""
        self._grid_major_every = major_every
        if self._grid_item is not None:
            self._grid_item.set_major_every(major_every)
        return self._apply_grid()

    def begin_grid_preview(self) -> bool:
        """Enters a slider drag: the bake is pulled (one eviction) and
        GridItem previews the appearance instead, so each tick is two QPens
        and an update() rather than an eviction. Accepted limitation: with
        elevation on the map, the preview lattice sits at elevation 0 while
        the applied grid drapes; the sliders set colour and weight, not
        placement. A no-op when the bake is not live, since GridItem is
        already the live thing there."""
        if self._grid_previewing or not self.grid_bake_live():
            return False
        self._grid_previewing = True
        return self._apply_grid()

    def end_grid_preview(self) -> bool:
        """Leaves a slider drag: pushes the real spec (the second and last
        eviction of the gesture) and hides GridItem again."""
        if not self._grid_previewing:
            return False
        self._grid_previewing = False
        return self._apply_grid()

    def grid_previewing(self) -> bool:
        return self._grid_previewing

    def set_camera_markers(
        self, markers: Sequence[tuple[int, int, int, QColor]], emphasised: int | None = None
    ) -> None:
        """View > Player Cameras' one push: (player_id, tile_x, tile_y,
        colour) per player with a view set, plus the player to emphasise
        (the one selected in Players mode, or None everywhere else).

        Rebuilt wholesale rather than diffed, like show_mirror_overlay: this
        is at most 8 items and ViewerWindow pushes the whole list whenever
        any part of it could have changed."""
        self._camera_markers = [(int(p), int(x), int(y), QColor(c)) for p, x, y, c in markers]
        self._camera_marker_emphasised = emphasised
        self._rebuild_camera_markers()

    def set_camera_markers_enabled(self, enabled: bool) -> None:
        """Shows or hides the layer, mirroring set_footprint_outlines: the
        markers are scene items, never baked into canvas pixels, so nothing
        is evicted here either."""
        self._camera_markers_enabled = enabled
        self._rebuild_camera_markers()

    def camera_marker_items(self) -> list[CameraMarkerItem]:
        """The live items, for tests and the eyeball tool."""
        return list(self._camera_marker_items)

    def _clear_camera_markers(self) -> None:
        for item in self._camera_marker_items:
            self.scene().removeItem(item)
        self._camera_marker_items = []

    def _rebuild_camera_markers(self) -> None:
        """Re-anchors every marker against the CURRENT projection. Called on
        a push, on a toggle, after set_source() (the items are destroyed
        with the scene) and from refresh_elevation_overlays()."""
        self._clear_camera_markers()
        if not self._camera_markers_enabled:
            return
        if self._map_width is None or self._map_height is None:
            return
        for player_id, tile_x, tile_y, color in self._camera_markers:
            # Bounds-checked here, not left to _ruler_anchor: Flat's
            # _tile_polygon does no check at all and Stepped's would index
            # the elevation array out of range (or, negatively, wrap).
            if not (0 <= tile_x < self._map_width and 0 <= tile_y < self._map_height):
                continue
            anchor = self._ruler_anchor((tile_x, tile_y))
            if anchor is None:
                # No projection snapshot yet, or the tile is off this map --
                # skipped rather than drawn at a guessed point.
                continue
            item = CameraMarkerItem(player_id, color, (tile_x, tile_y))
            emphasised = player_id == self._camera_marker_emphasised
            item.set_emphasised(emphasised)
            item.setPos(anchor)
            # The emphasised one sits a hair above its neighbours inside the
            # same layer: real files share one view tile across several
            # players (five of York's six sit on the same tile), and the
            # marker the Players panel is pointing at must not end up under
            # another player's.
            item.setZValue(self.CAMERA_MARKER_Z + (0.01 if emphasised else 0.0))
            self.scene().addItem(item)
            self._camera_marker_items.append(item)

    def set_analysis_markers(self, anchors: Sequence[tuple[tuple[int, int], str, int]]) -> None:
        """Tools > Map Analysis' one push, from map_analysis.marker_anchors():
        replaces any previous markers and drops the focus."""
        self._analysis_markers = [((int(t[0]), int(t[1])), str(s), int(n)) for t, s, n in anchors]
        self._analysis_focus = None
        self._rebuild_analysis_markers()

    def clear_analysis_markers(self) -> None:
        """The dialog closed, or its document went away. Idempotent."""
        self._analysis_markers = []
        self._analysis_focus = None
        self._rebuild_analysis_markers()

    def set_analysis_marker_focus(self, tile: tuple[int, int] | None) -> None:
        """Rings the marker on `tile` (the selected dialog row's), or none."""
        self._analysis_focus = None if tile is None else (int(tile[0]), int(tile[1]))
        if self._analysis_marker_item is not None:
            self._analysis_marker_item.set_focus(self._analysis_anchor_point(self._analysis_focus))

    def analysis_marker_item(self) -> AnalysisMarkerItem | None:
        """The live item, for tests and the eyeball tool."""
        return self._analysis_marker_item

    def _analysis_anchor_point(self, tile: tuple[int, int] | None) -> QPointF | None:
        """A tile's bounding-rect top-centre (the top vertex in Stepped/Sloped)."""
        if tile is None or self._map_width is None or self._map_height is None:
            return None
        tile_x, tile_y = tile
        # Flat's _tile_polygon does no bounds check; Stepped's would index out of range.
        if not (0 <= tile_x < self._map_width and 0 <= tile_y < self._map_height):
            return None
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return None
        rect = polygon.boundingRect()
        return QPointF(rect.center().x(), rect.top())

    def _rebuild_analysis_markers(self) -> None:
        """Re-anchors every marker against the CURRENT projection: on a push,
        after set_source() and from refresh_elevation_overlays(). No markers
        means no item at all, so a closed dialog leaves nothing in the scene."""
        if self.scene() is None:
            return
        points = []
        for tile, severity, count in self._analysis_markers:
            point = self._analysis_anchor_point(tile)
            if point is not None:
                points.append((point, severity, count))
        if not points:
            if self._analysis_marker_item is not None:
                self.scene().removeItem(self._analysis_marker_item)
                self._analysis_marker_item = None
            return
        if self._analysis_marker_item is None:
            self._analysis_marker_item = AnalysisMarkerItem(QColor(settings.get_overlay_color("analysis_marker")))
            self._analysis_marker_item.setZValue(self.ANALYSIS_MARKER_Z)
            self.scene().addItem(self._analysis_marker_item)
            # Created after _capture_zoom_baseline() ran, so it needs its first pad here.
            self._repad_edge_ticks()
        self._analysis_marker_item.set_markers(points)
        self._analysis_marker_item.set_focus(self._analysis_anchor_point(self._analysis_focus))

    def set_grid_appearance(self, blend: int, thickness: int) -> bool:
        """Stores the appearance for both halves. Evicts only when the bake
        is live and no preview is in progress; returns whether it did."""
        self._grid_blend = blend
        self._grid_thickness = thickness
        if self._grid_item is not None:
            self._grid_item.set_appearance(blend, thickness)
        return self._apply_grid()

    def set_edge_tick_interval(self, interval: int) -> None:
        self._edge_tick_interval = interval
        if self._edge_tick_item is not None:
            self._edge_tick_item.set_interval(interval)
