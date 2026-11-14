"""MapView's input: mouse, wheel and key handlers, tile picking, stroke bookkeeping and key panning."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable
from typing import ClassVar

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QKeySequence,
    QMouseEvent,
)
from PyQt5.QtWidgets import (
    QApplication,
)

from descape import (
    brush,
    gc_hold,
    iso_geometry,
    level_warm,
    perf_trace,
    region_clipboard,
    ruler,
    settings,
    trigger_geometry,
    unit_shape,
    zoom_preframe,
    zoom_stops,
)
from descape.map_view_highlight import _HOVER_GHOST_TOOLS, _LATCHED_SHAPES

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- render.py imports no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.viewer_common import (
    _TOOL_SHAPE,
    CLICK_TOOLS,
    EDIT_TOOLS,
    TOOL_EYEDROPPER,
    TOOL_RULER,
    TOOL_SELECT,
    brush_applicable,
)

# Stroke tools _touch_tile gap-fills. Cliff too: each path tile feeds the chain walk
# in order, so a fast drag lays the same run as a slow one.
_INTERPOLATED_STROKE_TOOLS = frozenset({"draw", "elevation", "set_level", "convert", "eraser", "cliff", "unit_brush"})


# Units-mode brushes: a drag mutates, so they skip the selection click and hover cue.
_UNIT_BRUSH_TOOLS = frozenset({"convert", "eraser", "unit_brush"})


class _MapViewInputMixin:
    """MapView's mouse, wheel and key handling (map_view.py's docstring has the split's rules)."""

    def set_wheel_bindings(self, slots: dict[str, str]) -> None:
        """Settings > Keybinds' wheel rows: slot -> action, as
        settings.get_wheel_bindings() returns them. Live on the next wheel event."""
        self._wheel_bindings = {
            modifiers: slots[slot] for slot, modifiers in self._WHEEL_SLOT_MODIFIERS.items() if slot in slots
        }

    def set_wheel_brush_hooks(self, query: Callable[[], bool], step: Callable[[int], None]) -> None:
        """`query()` says whether the active tool has a brush size, and
        `step(notches)` steps it, +1 per wheel-up notch, clamping at its bounds."""
        self._wheel_brush_query = query
        self._on_wheel_brush_step = step

    def set_wheel_paste_hooks(self, query: Callable[[], bool], turn: Callable[[int], None]) -> None:
        """GH #245: `query()` says whether a just-pasted region would turn, and
        `turn(direction)` turns it a quarter turn, +1 per wheel-up notch."""
        self._wheel_paste_query = query
        self._on_wheel_paste_turn = turn

    def set_wheel_selection_query(self, query: Callable[[], bool]) -> None:
        """GH #356: `query()` says whether the Units-mode selection would turn,
        for the "_selection" wheel choices. The turn itself goes through
        set_place_rotate()'s hook, the Rotate keys' router."""
        self._wheel_selection_query = query

    def _sloped_cache(self):
        """The live SlopedChunkCache, or None before one is wired up.

        Reached through the canvas item (matching _coarsest_mip_scale's own
        access) rather than held as a second MapView attribute on purpose:
        this is the object that actually composited the pixels on screen, so
        both hit-testing and the outline read the SAME corner_rise those
        pixels were painted from. A parallel copy on MapView is exactly the
        stale-snapshot drift Risk #6 exists to prevent."""
        if self._canvas_item is None or self._terrain_style != "sloped":
            return None
        return self._canvas_item._cache

    def _pick_tile(self, pos: QPointF) -> tuple[int, int] | None:
        """The tile under scene-space pos, or None if pos doesn't land on
        any tile (Stepped mode only -- Flat mode's flat division is always
        defined, matching its pre-Phase-3 behavior exactly, including for
        out-of-map/overscroll positions, which callers already bounds-check
        themselves). Stepped mode delegates to iso_geometry.screen_to_tile
        against the exact elevation snapshot and IsoProjection the current
        image was rendered from (set together, in set_source() -- see Risk #6
        in the parent plan). Returns None on a skirt-face pixel too (no
        exact analytic inverse there, per Risk #5) -- deferred, not solved,
        matching Phase 2's own accepted behavior."""
        if self._terrain_style == "stepped":
            if self._iso_elevations is None or self._iso_proj is None:
                return None
            return iso_geometry.screen_to_tile(
                int(pos.x()), int(pos.y()), self._iso_elevations, self._iso_proj
            )
        if self._terrain_style == "sloped":
            # Sloped has no analytic inverse -- but not for the reason this
            # comment used to give ("a bilinear-blended ramp surface"). Post
            # seam-resample the cuts come from corner values on the shared
            # tile edge by integer arithmetic; the terrain surface is no
            # longer a per-pixel bilinear field at all. The real obstacle is
            # that a per-column integer RESAMPLE has no closed form to
            # invert, so Track C4 answers by lookup instead: an id plane
            # rasterized by the same gathers the compositor paints through.
            #
            # Better than Stepped's branch above, not worse: screen_to_tile
            # returns None on every skirt-face pixel (a documented accepted
            # residual). Sloped paints no skirts, so it has no such hole.
            cache = self._sloped_cache()
            if cache is None:
                return None
            return cache.pick_tile(int(pos.x()), int(pos.y()))
        return int(pos.x()) // self._tile_pixels, int(pos.y()) // self._tile_pixels

    def _pick_map_point(self, pos: QPointF) -> tuple[float, float] | None:
        """The CONTINUOUS map point under scene-space pos, or None -- off-grid
        placement's Stage 2, and `_pick_tile`'s fractional counterpart.

        A separate method rather than a widened `_pick_tile`, deliberately:
        `_pick_tile` has many callers (`_pos_on_map`, `mouseMoveEvent`,
        `pick_unit_at`, `on_unit_place`, `on_unit_move`) and every one of
        them wants an integer tile.

        The float is carried past the `int()` truncation the tile path does,
        which is what actually caps precision at one pixel -- Sloped still
        hands `cache.pick_tile` integers, since the id plane is integer-indexed,
        but the solve itself gets the unrounded position. corner_rise comes
        through `_sloped_cache()`, the same accessor `_pick_tile` and
        `_tile_polygon` use and for the same reason: it is the object that
        actually composited the pixels."""
        if self._terrain_style == "stepped":
            if self._iso_elevations is None or self._iso_proj is None:
                return None
            return iso_geometry.screen_to_map_point(
                int(pos.x()), int(pos.y()), "stepped", self._iso_proj,
                elevations=self._iso_elevations,
            )
        if self._terrain_style == "sloped":
            cache = self._sloped_cache()
            if cache is None or self._iso_proj is None:
                return None
            tile = cache.pick_tile(int(pos.x()), int(pos.y()))
            if tile is None:
                return None
            return iso_geometry.screen_to_map_point(
                int(pos.x()), int(pos.y()), "sloped", self._iso_proj,
                corner_rise=cache.corner_rise, tile=tile, fold=cache.fold,
            )
        return iso_geometry.screen_to_map_point(
            int(pos.x()), int(pos.y()), "flat", tile_px=self._tile_pixels
        )

    def _pos_on_map(self, pos: QPointF) -> bool:
        """Flat mode's old _map_rect.contains(pos) generalizes in Stepped
        mode to "does this pixel unproject to a real tile" -- the map's
        silhouette there is a diamond inscribed in _map_rect (the full
        pixmap bounding box), with real background pixels in the corners
        _map_rect.contains(pos) alone would wrongly call on-map. Sloped
        answers the same question the same way, via its own pick plane (see
        _pick_tile) -- a background pixel there resolves to no tile."""
        if self._terrain_style in _ELEVATED_STYLES:
            return self._pick_tile(pos) is not None
        return self._map_rect is not None and self._map_rect.contains(pos)

    def _touch_tile(self, tile_x: int, tile_y: int, modifiers) -> None:
        key = (tile_x, tile_y)
        # Qt's motion compression lands a slow stroke's cursor tiles apart: fill the
        # gap, in one call. Off-map moves keep the last tile, so re-entry spans a skirt.
        last = self._stroke_last_tile
        if last is None or self._tool not in _INTERPOLATED_STROKE_TOOLS:
            path = [key]
        else:
            path = brush.line_tiles(*last, *key)
        self._stroke_last_tile = key
        tiles = [t for t in path if t not in self._stroke_touched]
        if not tiles:
            return
        self._stroke_touched.update(tiles)
        # Right-click strokes lower elevation -- the opposite of left-click's
        # raise. Synthesized as the same ShiftModifier bit Shift+left-click
        # already used for "lower" (kept working, not replaced) rather than
        # widening on_stroke_tiles's signature with a separate direction
        # argument the "draw"/"set_level" tools would just ignore.
        if self._stroke_button == Qt.RightButton:
            modifiers = modifiers | Qt.ShiftModifier
        with perf_trace.span("step"):
            self._on_stroke_tiles(tiles, modifiers)
        perf_trace.step()

    def _pause_gc_for_stroke(self) -> None:
        """A full collection walks everything built since the load-time
        gc.freeze() (preloaded levels' sprite layers, grids): 20-45 ms that
        otherwise lands mid-stroke. Refcounting still frees acyclic garbage;
        cycles wait until release. Leaves gc alone if someone else disabled it."""
        if gc.isenabled():
            gc.disable()
            self._stroke_gc_paused = True

    def resume_stroke_gc(self) -> None:
        """Undoes _pause_gc_for_stroke(); idempotent, so every stroke-closing
        path (release, leave, File > Close, window close) can call it."""
        if self._stroke_gc_paused:
            self._stroke_gc_paused = False
            gc.enable()

    def _end_stroke(self) -> None:
        # The release span: the drag line's wall covers it up to the stroke-end flush.
        with perf_trace.span("release"):
            with perf_trace.phase("gc_resume"):
                self.resume_stroke_gc()
            self._stroke_active = False
            self._stroke_touched = set()
            self._stroke_last_tile = None
            self._stroke_button = None
            # Resumes the highlight pulse the press paused. See
            # _sync_pulse_timer() for why a stroke stops it.
            self._sync_pulse_timer()
            if self._trigger_ghosts_stale:
                self._trigger_ghosts_stale = False
                self._schedule_trigger_ghost_refresh()
            try:
                self._on_stroke_end()
                # Also for a stroke that mutated nothing, so _after_unit_mutation() never refreshed.
                if self._tool == "unit_brush":
                    self.refresh_place_ghost()
            finally:
                perf_trace.end_drag(self._tool)

    def mousePressEvent(self, event) -> None:
        self._press_tool = self._tool
        if event.button() == Qt.MiddleButton:
            self._middle_drag_active = True
            self._middle_drag_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        # GH #138's tile pick, the same priority as the unit pick below. A point
        # commits on press; a rect press only anchors, the release commits.
        if self._tile_picker_kind is not None:
            if event.button() == Qt.RightButton:
                self.on_tile_picker_cancel()
            elif event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                tile = self._pick_tile(pos) if self._pos_on_map(pos) else None
                if tile is not None and self._on_map(tile):
                    if self._tile_picker_kind == "rect":
                        self._tile_pick_anchor = tile
                        self._update_tile_pick_preview(tile)
                    else:
                        self.on_tile_pick(tile)
            return
        # Trigger Pick from map, above every tool and mode branch: arming forces
        # Pan, and a pick must never fall through to a selection or a pan.
        if self._unit_picker_active:
            if event.button() == Qt.RightButton:
                self.on_unit_picker_cancel()
            elif event.button() == Qt.LeftButton and self._map_rect is not None:
                pos = self.mapToScene(event.pos())
                if self._map_rect.contains(pos):
                    self._picker_click(pos)
            return
        # GH #229: with Pan, a left press on the live area's middle, edge or corner, or
        # its location mark, starts a drag; elsewhere it pans. A right press cancels it.
        if self._area_drag is not None:
            if event.button() == Qt.RightButton:
                self._cancel_area_drag()
            return
        if event.button() == Qt.LeftButton and self._area_edit_live():
            pos = self.mapToScene(event.pos())
            zone = self._area_hit(pos)
            start = self._tile_pick_sample(pos) if zone is not None else None
            if start is not None:
                location = zone == trigger_geometry.ZONE_LOCATION
                self._area_drag = (zone, start, self._location_edit * 2 if location else self._area_edit)
                self._area_drag_end = start
                self._apply_drag_mode()
                self._apply_tool_cursor()
                self._update_area_drag_preview()
                return
        # No single tool arms every unit gesture field, so a stale drag or marquee
        # from a swallowed release dies at the next left press, whichever tool presses.
        if self._mode == "units" and event.button() == Qt.LeftButton:
            self._reset_unit_gesture()
        # The Ruler, deliberately ABOVE the Units branch below rather than
        # beside the tool checks under it: that branch is independent of the
        # active tool, so it would otherwise swallow every ruler click in
        # Units mode. Right button cancels a pending measurement, or else
        # removes the pinned ruler whose endpoint it hits; either button mutates nothing.
        if self._tool == TOOL_RULER:
            if event.button() == Qt.RightButton:
                pos = self.mapToScene(event.pos())
                if self._ruler.state != ruler.STATE_IDLE:
                    self._clear_ruler()
                elif self._pos_on_map(pos):
                    tile = self._pick_tile(pos)
                    before = tuple(self._pinned_rulers)
                    gone = self._pinned_rulers.remove_at(tile) if tile is not None else None
                    if gone is not None:
                        index = next(i for i, m in enumerate(before) if m is gone)
                        self._pinned_rulers_changed()
                        self.on_pinned_rulers_edited("Remove ruler", (), ((index, gone),))
            elif event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                # _pos_on_map, NOT "_pick_tile is not None": Flat's integer
                # division never returns None, so a press out in the
                # OVERSCROLL_FRACTION void would otherwise yield a negative or
                # past-the-edge tile that _tile_polygon happily projects and
                # that Stepped would use to index _iso_elevations.
                if self._pos_on_map(pos):
                    previous = self._ruler.state
                    self._ruler.press(self._pick_tile(pos))
                    self._update_ruler()
                    self._report_ruler(previous)
            return
        # The Select tool's region drag, is_edit_tool=False like the Ruler
        # above and given the same early priority, in every mode but Units,
        # where Select picks units in the Units branch below. Left button
        # only: a rectangle drag has no meaningful right-button inverse.
        if self._tool == TOOL_SELECT and self._mode != "units":
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                if self._pos_on_map(pos):
                    tile = self._pick_tile(pos)
                    # A press left open by a lost release closes first, so drags never nest.
                    self._close_region_drag_trace()
                    if self._region_movable and self._tile_in_region(tile):
                        self._move_anchor = tile
                        self._move_current = tile
                        self._region_drag_label = "select-move"
                    else:
                        self._select_anchor = tile
                        self._select_current = tile
                        self._region_drag_label = "select"
                    # A Perf Trace drag like a stroke's: the press is its first step.
                    perf_trace.begin_drag(None)
                    self._traced_region_rebuild()
            return
        # The Eyedropper in Units mode picks a unit, above the Units branch, which would place one on
        # any click_only tool's click. Left button, inside the canvas; Terrain mode goes on below.
        if self._tool == TOOL_EYEDROPPER and self._mode == "units":
            if event.button() == Qt.LeftButton and self._map_rect is not None:
                pos = self.mapToScene(event.pos())
                if self._map_rect.contains(pos):
                    found = self._cycle_pick(pos, "_eyedrop_cycle")
                    if found is not None:
                        self.on_unit_eyedrop(*found, event.modifiers())
            return
        # Draw Line / Draw Rectangle, above the EDIT_TOOLS stroke branch
        # below (they are edit tools, so they would otherwise open a stroke)
        # and beside the Ruler/Select branches, whose press-drag-release
        # grammar they share. Gated on _pos_on_map, NOT "_pick_tile is not
        # None" -- see the Ruler's own note above for why those differ in
        # Flat. Right button cancels rather than drawing an inverse: there
        # is no meaningful inverse of painting a shape, and _touch_tile ORs
        # in ShiftModifier for right-button strokes, which this path would
        # read as a phantom constrain.
        # Place Unit with a wall const picked joins them (GH #98), which is
        # why this sits above the Units-mode branch below.
        if self._is_shape_tool():
            if event.button() == Qt.RightButton:
                self._cancel_shape()
            elif event.button() == Qt.LeftButton and self._map_rect is not None:
                pos = self.mapToScene(event.pos())
                if self._pos_on_map(pos):
                    tile = self._pick_tile(pos)
                    shape = _TOOL_SHAPE.get(self._tool) or "wall"
                    latch = None
                    if shape in _LATCHED_SHAPES:
                        # A refused pick (logged by the viewer) starts no drag.
                        latch = self._shape_press_query()
                        if latch is None:
                            return
                    self._drag_shape = shape
                    self._shape_latch = latch
                    walls = shape == "wall" or (latch is not None and latch.branch == unit_shape.WALL)
                    # Gated on the shape: the query walks every unit, and Draw Line needs none of it.
                    self._drag_blocked = frozenset(self._place_blocked_query()) if walls else frozenset()
                    self._shape_anchor = tile
                    self._shape_end = tile
                    self._shape_modifiers = event.modifiers()
                    self._update_shape_preview()
            return
        # Units mode's left press, ABOVE the CLICK_TOOLS/EDIT_TOOLS checks,
        # gated by tool (TASK-252): Select picks, Move drags, Pan falls through
        # to the hand drag. The Ruler and the Eyedropper are handled above;
        # Convert (b2.5) and the Eraser are excluded so they reach the generic
        # stroke branch further down, since their drag is a mutation with its
        # own undo record, not a selection gesture. Left button only.
        if (
            self._mode == "units"
            and event.button() == Qt.LeftButton
            and self._map_rect is not None
            and self._tool not in _UNIT_BRUSH_TOOLS
        ):
            pos = self.mapToScene(event.pos())
            # Place Unit (b1.4) is click_only, so it's in CLICK_TOOLS -- and
            # unambiguously so here: the Eyedropper, the other click_only tool
            # reachable in Units mode, returned from its own branch above, and
            # Paint Can and Create Objects gate on other modes. A place click
            # is a distinct action from selection, not an additional effect of it.
            if self._tool in CLICK_TOOLS:
                if self._pos_on_map(pos):
                    self._on_unit_place(pos, event.modifiers())
                return
            # The ghost of a drag that never released; Place Unit keeps its hover ghost above.
            self._clear_unit_ghost()
            if self._units_select():
                # The click fires at press; a press-then-jitter-then-release
                # must still select. A press on a unit arms the marquee too,
                # but it only engages once the cursor leaves that unit's tile.
                # on_click_select's return IS the press's one unit pick.
                selected_key = self._on_click_select(pos, event.modifiers())
                self._marquee_start_pos = event.pos()
                self._marquee_press_on_unit = selected_key is not None
                self._marquee_press_tile = self._pick_tile(pos) if selected_key is not None else None
                return
            if self._units_move():
                # b1.5's move-by-drag: the key to drag (None on empty ground)
                # and where, so mouseReleaseEvent can tell a click from a drag.
                self._unit_drag_key = self._on_move_press(pos, event.modifiers())
                self._unit_drag_press_pos = event.pos()
                self._unit_drag_hover = None
                return
            if self._tool == "pan":
                # Falls through to the hand drag; the release says whether it was a click.
                self._pan_click_pos = event.pos()
        # Click-only tools (Paint Can) never open a stroke: one edit per
        # press, nothing per drag-entered tile. Handled here, above the
        # stroke block, so the shared begin_stroke/commit_stroke pair below
        # stays exactly one-to-one -- routing a one-shot edit through those
        # handlers instead (suppressing after the first stroke tile) needs a
        # latch flag and conditional commits in all three shared handlers,
        # and has a real wedge bug: _on_tool_selected sets self._current_tool
        # before calling set_tool() (which closes any dangling stroke), so a
        # mid-drag tool switch can leave a stroke's begin_stroke() snapshot
        # uncommitted. Left button only: a fill has no meaningful inverse for
        # the right button to carry (unlike Elevate's raise/lower), and a
        # stray right-click silently rewriting an entire region is a costly
        # surprise.
        if self._tool in CLICK_TOOLS and event.button() == Qt.LeftButton and self._map_rect is not None:
            pos = self.mapToScene(event.pos())
            if self._pos_on_map(pos):
                self._on_click_edit(*self._pick_tile(pos), event.modifiers())
            return
        # Right button starts a stroke exactly like left does -- the only
        # difference is _touch_tile()'s elevation-lowering synthesis above,
        # which the "draw"/"set_level" tools simply don't look at, and the
        # Cliff tool's erase, which reads stroke_button() at start. Click
        # tools are excluded here too (not just above): the CLICK_TOOLS
        # branch above only returns for a LEFT press, so without this guard
        # a right-click on Paint Can would fall through and open a real
        # stroke instead of being the no-op its own tooltip/behavior promise.
        if (
            self._tool in EDIT_TOOLS
            and self._tool not in CLICK_TOOLS
            and event.button() in (Qt.LeftButton, Qt.RightButton)
            and self._map_rect is not None
        ):
            if self._stroke_active:
                # The other button pressed while one is already down mid-
                # drag (e.g. right held, then left also pressed) -- ignored,
                # not a second stroke start. Letting this through would
                # start a second begin_stroke() on edit_history without an
                # intervening commit_stroke(), and mouseReleaseEvent's own
                # button check would end the still-physically-held stroke
                # early. See mouseReleaseEvent for the matching half of
                # this fix.
                return
            # The press span: begin_drag() runs inside it, and its wall counts from here.
            with perf_trace.span("press"):
                pos = self.mapToScene(event.pos())
                if self._pos_on_map(pos):
                    # The stroke's painted units are the preview; the release restores the ghost.
                    if self._tool == "unit_brush":
                        self._clear_unit_ghost()
                    self._stroke_active = True
                    self._stroke_touched = set()
                    self._stroke_last_tile = None
                    self._stroke_button = event.button()
                    # Pauses the pulse for the duration of the stroke, which is
                    # the one time its 40ms tick competes with real edit work.
                    self._sync_pulse_timer()
                    self._pause_gc_for_stroke()
                    perf_trace.begin_drag(
                        f"{self._brush_size} {self._brush_shape}"
                        + (f" t{self._brush_thickness}" if self._brush_shape in brush.OUTLINE_SHAPES else "")
                        if brush_applicable(self._tool, rect_filled=self._rect_filled)
                        else None
                    )
                    self._on_stroke_start()
                    self._touch_tile(*self._pick_tile(pos), event.modifiers())
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        # QGraphicsView's default swallows the second press of a fast pair,
        # which would drop a repeat click on a stack (or a second edit click).
        if self._press_tool not in (None, self._tool):
            # TASK-287: the first press switched tools (an Eyedropper switch-back); the second belongs to no gesture.
            return
        self.mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._middle_drag_active:
            self._middle_drag_active = False
            self._middle_drag_last_pos = None
            self._apply_tool_cursor()
            return
        # GH #138's rect pick commits here, clamped to the map like every drag sample.
        if self._tile_picker_kind is not None:
            anchor = self._tile_pick_anchor
            if event.button() == Qt.LeftButton and anchor is not None:
                end = self._tile_pick_sample(self.mapToScene(event.pos())) or anchor
                self._tile_pick_anchor = None
                self._clear_tile_pick_preview()
                self.on_rect_pick(
                    min(anchor[0], end[0]), min(anchor[1], end[1]), max(anchor[0], end[0]), max(anchor[1], end[1])
                )
            return
        # GH #229's drag writes once, here, and only a real change.
        if self._area_drag is not None:
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                end = self._tile_pick_sample(pos)
                if end is not None:
                    self._area_drag_end = end
                zone, coords, area = self._area_drag[0], self._area_drag[2], self._dragged_area()
                self._cancel_area_drag()
                if area != coords:
                    if zone == trigger_geometry.ZONE_LOCATION:
                        self.on_location_edit(area[0], area[1])
                    else:
                        self.on_area_edit(*area)
                self._set_area_hover(self._area_hit(pos))
            return
        # A left release only ever DECIDES for the Ruler; the endpoint was
        # already placed by mouseMoveEvent, so there is nothing to redraw. A
        # release inside the start tile deliberately leaves the session
        # pending, which is what makes click-move-click work.
        if self._tool == TOOL_RULER:
            if event.button() == Qt.LeftButton:
                previous = self._ruler.state
                self._ruler.release()
                self._report_ruler(previous)
            return
        # The Select tool's matching release: commits self._select_anchor/
        # self._select_current (the latter frozen at its last on-map value if
        # the release itself is off-map, same "hold the last valid tile"
        # convention as the Ruler's endpoint) into a real region via
        # set_region(), then announces it -- the one path
        # ViewerWindow.on_region_selected() shares with Select All/Deselect.
        # Not in Units mode, where a Select release ends a marquee below.
        if self._tool == TOOL_SELECT and self._mode != "units":
            if event.button() == Qt.LeftButton and self._move_anchor is not None:
                anchor, current = self._move_anchor, self._move_current
                self._move_anchor = None
                self._move_current = None
                # Before the callback, so the move's undo and paste keep their own op lines.
                self._close_region_drag_trace()
                self._update_region_overlay()
                dx, dy = current[0] - anchor[0], current[1] - anchor[1]
                # A zero-delta press-release inside the selection is a no-op
                # that KEEPS the region, deliberately not a deselect.
                if dx or dy:
                    self._on_region_move(dx, dy)
                return
            if event.button() == Qt.LeftButton and self._select_anchor is not None:
                anchor, current = self._select_anchor, self._select_current
                self._select_anchor = None
                self._select_current = None
                self._close_region_drag_trace()
                region = region_clipboard.normalize_region(
                    anchor, current, self._map_width or 0, self._map_height or 0
                )
                self.set_region(region)
                self._on_region_selected(region)
            return
        # The shape tools' sole commit point. One on_shape_commit() call per
        # drag, with the exact (never the preview-approximated) tile set;
        # the drag state is cleared BEFORE the callback so a re-entrant
        # repaint during the commit can't see a half-live drag. A release
        # arriving with no anchor -- a drag closed mid-flight by set_tool(),
        # say -- falls through to a no-op rather than committing.
        if self._shape_drag_live():
            if event.button() == Qt.LeftButton and self._shape_anchor is not None:
                tiles = self._shape_tiles(preview=False)
                latch = self._shape_latch
                self._cancel_shape()
                if tiles and latch is not None:
                    self._on_shape_commit(tiles, latch=latch)
                elif tiles:
                    self._on_shape_commit(tiles)
            return
        # b1.5's move-by-drag: the matching half of the press-time bookkeeping
        # above. Cleared unconditionally either way -- a drag is decided once,
        # here, never left pending for a later event. Keyed on the state, not
        # the tool, so a drag begun under Move still commits after a key switch.
        if event.button() == Qt.LeftButton and self._unit_drag_key is not None:
            key = self._unit_drag_key
            press_pos = self._unit_drag_press_pos
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            # Unconditionally, before either outcome below: the ghost is a
            # preview of a pending move, and both a committed move and a
            # below-threshold click end the gesture it was previewing. So is
            # the drop host's outline (GH #115).
            self._clear_unit_ghost()
            self._clear_unit_hover()
            moved = (
                press_pos is not None
                and (event.pos() - press_pos).manhattanLength() > self.UNIT_DRAG_THRESHOLD_PX
            )
            if moved and self._map_rect is not None:
                pos = self.mapToScene(event.pos())
                if self._is_group_drag(key):
                    # GH #75: a group clamps at the edge instead of cancelling.
                    self._on_unit_move(key, self._clamped_to_map_rect(pos), event.modifiers())
                elif self._pos_on_map(pos):
                    self._on_unit_move(key, pos, event.modifiers())
            return
        # b2.3's marquee: the matching half of the press-time bookkeeping in
        # mousePressEvent, same decide-once-here shape as b1.5's move-drag
        # above (and mutually exclusive with it). A release that never
        # engaged was already a plain click, handled by mousePressEvent's
        # unconditional on_click_select() call; nothing further to do here.
        if event.button() == Qt.LeftButton and self._marquee_start_pos is not None:
            engaged = self._marquee_engaged(event.pos(), self._pick_tile(self.mapToScene(event.pos())))
            start_pos = self._marquee_start_pos
            self._marquee_start_pos = None
            self._marquee_press_tile = None
            self._clear_marquee()
            if engaged and self._map_rect is not None:
                rect = QRectF(self.mapToScene(start_pos), self.mapToScene(event.pos())).normalized()
                keys = self._units_in_scene_rect(rect)
                self._on_marquee_select(keys, event.modifiers())
            return
        # Gated on the specific button that started the stroke, not "any
        # left/right release" -- with the other button ignored at press
        # time (see mousePressEvent), the only release that should ever end
        # a stroke is the one matching self._stroke_button; anything else
        # (e.g. releasing a left click that mousePressEvent already ignored
        # while a right-button stroke was active) must fall through and be
        # a no-op, not end the still-held stroke early.
        if event.button() == self._stroke_button and self._stroke_active:
            self._end_stroke()
            return
        # A Pan press in Units mode that never became a pan: a click, which
        # the window answers with a one-time hint. At release, so a pan drag
        # that starts on a unit says nothing.
        if event.button() == Qt.LeftButton and self._pan_click_pos is not None:
            press_pos, self._pan_click_pos = self._pan_click_pos, None
            if (
                self._mode == "units"
                and self._tool == "pan"
                and (event.pos() - press_pos).manhattanLength() <= self.UNIT_DRAG_THRESHOLD_PX
            ):
                self.on_pan_click(self.mapToScene(event.pos()))
        super().mouseReleaseEvent(event)

    # Arrow-key nudge deltas, in tiles -- Qt.Key -> (dx, dy). This MAPPING is
    # fixed, not a REBINDABLE_ACTIONS set (see this method's own docstring on
    # Escape, which the same reasoning applies to). The same four physical
    # keys are also the view_pan_* defaults, which is a separate, rebindable
    # binding; keyPressEvent below is where the two are arbitrated.
    _UNIT_NUDGE_KEYS: ClassVar[dict[int, tuple[int, int]]] = {
        Qt.Key_Left: (-1, 0),
        Qt.Key_Right: (1, 0),
        Qt.Key_Up: (0, -1),
        Qt.Key_Down: (0, 1),
    }

    def keyPressEvent(self, event) -> None:
        """Escape cancels the nearest live thing, in order: tile picker, unit
        picker, area drag, shape, ruler, unit drag, marquee, region move,
        region drag, a live flip or turn (TASK-256, which Enter keeps), unit
        selection (Units Select only), committed region. In
        Units mode, arrow keys nudge the selection (b1.5) and Delete removes it
        (b1.6); anything bound to view_pan_* starts a held-key pan.

        Escape/Delete/the nudge mapping are not REBINDABLE_ACTIONS rows: they
        are platform conventions rather than commands, and every added row is
        one more hand-audited entry in a keybind namespace with no runtime
        collision detection. Pan is, and ships defaulted to the same four
        arrow keys.

        **Arrow-key ownership.** An arrow nudges when Units mode has a
        selection the nudge can actually act on, and pans in every other case
        -- including Units mode with nothing selected. on_unit_nudge's own
        return value is the discriminator (it already refuses on no scenario,
        no unit index, no resolvable entries and no edit model), so there is
        one authority for "did the nudge happen" rather than two that can
        disagree. Consequences worth knowing: rebinding pan to WASD makes W
        pan even with a selection, since W is not a nudge key, and Shift+Up
        with a selection is always the whole-tile nudge, never a pan.

        Accepted limitation: all of these only fire while the view holds
        keyboard focus, so after clicking away to the status log they do
        nothing. A selection click always grants that focus (QGraphicsView's
        default focus policy is StrongFocus), so the ordinary click-then-
        nudge-or-delete flow works; Right-click and starting the next ruler
        measurement both still clear it.
        """
        # Any key: the short level-warm budget, as wheelEvent; _on_pan_tick covers a held pan (TASK-031.80).
        level_warm.note_input()
        # A pending coalesced move lands first, so the key sees the last cursor tile.
        self._flush_pending_move()
        # Pick from map first: arming forced Pan, so no other Escape owner is live.
        if event.key() == Qt.Key_Escape and self._tile_picker_kind is not None:
            self.on_tile_picker_cancel()
            return
        if event.key() == Qt.Key_Escape and self._unit_picker_active:
            self.on_unit_picker_cancel()
            return
        if event.key() == Qt.Key_Escape and self._area_drag is not None:
            self._cancel_area_drag()
            return
        # Above the Ruler's own Escape below: the two are never live at once
        # (different tools), so order is arbitrary, but a mutating tool's
        # cancel reads better first.
        if event.key() == Qt.Key_Escape and self._shape_anchor is not None:
            self._cancel_shape()
            return
        if event.key() == Qt.Key_Escape and self._ruler.state != ruler.STATE_IDLE:
            self._clear_ruler()
            return
        # Cancels a unit drag outright. Not optional polish: a visible preview
        # invites a cancel gesture, and before this a started drag could only
        # be completed. Dropping _unit_drag_key is what makes the eventual
        # release commit nothing -- mouseReleaseEvent's unit branch is gated on
        # exactly that key being set.
        if event.key() == Qt.Key_Escape and self._unit_drag_key is not None:
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            self._clear_unit_ghost()
            return
        # Cancels a live Units marquee, never the selection its press already made (TASK-325).
        if event.key() == Qt.Key_Escape and self._marquee_start_pos is not None:
            self._marquee_start_pos = None
            self._marquee_press_tile = None
            self._clear_marquee()
            return
        # Select tool's Escape: a live gesture first, then the unit selection, then the
        # committed region, so Escape mid-gesture never destroys an earlier region (daubED's rule).
        if event.key() == Qt.Key_Escape and self._move_anchor is not None:
            self._move_anchor = None
            self._move_current = None
            self._close_region_drag_trace()
            self._update_region_overlay()
            return
        if event.key() == Qt.Key_Escape and self._select_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._close_region_drag_trace()
            self._update_region_overlay()
            return
        # TASK-256: a live flip or turn reverts before the selection or region it acted on clears.
        if event.key() == Qt.Key_Escape and self._on_lift_revert is not None and self._on_lift_revert():
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._on_lift_apply is not None and self._on_lift_apply():
            return
        if (
            event.key() == Qt.Key_Escape
            and self._units_select()
            and self._on_unit_escape is not None
            and self._on_unit_escape()
        ):
            return
        if event.key() == Qt.Key_Escape and self._region is not None:
            self.set_region(None)
            self._on_region_selected(None)
            return
        # Live Shift: mouseMoveEvent already carries event.modifiers(), so a
        # shape drag gets snap-on-next-move for free -- but pressing Shift
        # without moving the mouse is exactly the gesture a constrain
        # modifier is reached for, and it has to answer immediately.
        # keyReleaseEvent below is the other half; MapView is StrongFocus,
        # so both actually arrive during a drag.
        if event.key() == Qt.Key_Shift and self._shape_anchor is not None:
            self._shape_modifiers = self._shape_modifiers | Qt.ShiftModifier
            self._update_shape_preview()
            return
        # Held Alt is off-grid placement's one-off, so the hover ghost answers it
        # without a move. Falls through: Alt alone is no other binding here.
        if event.key() == Qt.Key_Alt and self._place_hover is not None:
            pos, modifiers = self._place_hover
            self._place_hover = (pos, modifiers | Qt.AltModifier)
            self.refresh_place_ghost()
        if self._mode == "units":
            if event.key() in self._UNIT_NUDGE_KEYS:
                dx, dy = self._UNIT_NUDGE_KEYS[event.key()]
                if self._on_unit_nudge(dx, dy, event.modifiers()):
                    # Ownership of this key just moved from pan to nudge --
                    # clicking a unit mid-hold is the real case. Ending the
                    # pan here rather than at a release the pan path no
                    # longer owns is what stops a runaway timer.
                    self._release_pan_key(event.key())
                    event.accept()
                    return
                # Falls through to the pan branches below: "no selection to
                # nudge" is exactly the case the user asked to pan.
            elif event.key() == Qt.Key_Delete:
                self._on_unit_delete(event.modifiers())
                return
        # Conditional, not a blanket auto-repeat swallow: the nudge above
        # deliberately fires once per OS repeat, so short-circuiting every
        # repeat here would kill held-arrow nudging.
        pan_key = (event.key(), self._pan_modifiers(event.modifiers()))
        if event.isAutoRepeat() and pan_key in self._pan_held:
            event.accept()
            return
        direction = self._pan_bindings.get(pan_key)
        if direction is not None:
            self._pan_held[pan_key] = direction
            if not self._pan_timer.isActive():
                self._pan_elapsed.start()
                self._pan_timer.start(self.PAN_TICK_MS)
            event.accept()
            return
        # Suppresses QAbstractScrollArea's native single-step arrow scroll,
        # so panning is governed only by the keybind system above: rebind pan
        # to WASD and a bare arrow pans nothing.
        if self._pan_modifiers(event.modifiers()) == 0 and event.key() in self._UNIT_NUDGE_KEYS:
            event.accept()
            return
        super().keyPressEvent(event)

    def set_pan_binding(self, action_id: str, key_sequence_text: str) -> None:
        """ViewerWindow.apply_keybind's target for the four view_pan_* ids --
        they never reach a QAction, since a shortcut consumes the press and
        never reports the release a hold timer needs.

        An empty sequence unbinds. A multi-chord sequence is treated as
        unbound too: a chord cannot be "held"."""
        direction = self.PAN_DIRECTIONS[action_id]
        for key, bound in list(self._pan_bindings.items()):
            if bound == direction:
                del self._pan_bindings[key]
                self._release_pan_key(key[0])
        if not key_sequence_text:
            return
        seq = QKeySequence(key_sequence_text)
        if seq.isEmpty() or seq.count() != 1:
            return
        packed = int(seq[0])
        key = packed & ~self._KEY_MODIFIER_MASK
        self._pan_bindings[(key, self._pan_modifiers(packed & self._KEY_MODIFIER_MASK))] = direction

    @classmethod
    def _pan_modifiers(cls, modifiers) -> int:
        """The modifier half a pan binding is matched on, with KeypadModifier
        masked off. Not optional tidiness: on macOS Qt sets KeypadModifier on
        the ARROW keys themselves, so an unmasked comparison would leave the
        shipped arrow defaults matching nothing there -- no pan, and the bare
        arrow falling through to the native single-step jump this feature
        replaces. Masking it also makes the numpad arrows behave like the
        cursor arrows, which is what a pan key should do anyway."""
        return int(modifiers) & ~int(Qt.KeypadModifier)

    def _release_pan_key(self, key: int) -> None:
        """Drops a held key and stops the timer once nothing is held. Safe to
        call for a key that was never held -- a nudged arrow, or any
        unrelated release.

        Matched on the Qt key ALONE, never on (key, modifiers): releasing
        Ctrl before Up on a Ctrl+Up binding delivers the Up release with the
        modifier already gone, and a tuple match would miss it and leave the
        timer running forever."""
        released = [k for k in self._pan_held if k[0] == key]
        if not released:
            return
        for k in released:
            del self._pan_held[k]
        if not self._pan_held:
            self._stop_pan()

    def _stop_pan(self) -> None:
        self._pan_timer.stop()
        self._pan_accum_x = 0.0
        self._pan_accum_y = 0.0

    def _on_pan_tick(self) -> None:
        # A held-key pan holds no mouse button: defer the gen-2 collect past it (TASK-031.81), and keep
        # the short level-warm budget before auto-repeat starts (TASK-031.80).
        gc_hold.note_input()
        level_warm.note_input()
        dt_ms = min(float(self._pan_elapsed.restart()), self.PAN_MAX_DT_MS)
        self._pan_step(dt_ms)

    def _pan_step(self, dt_ms: float) -> None:
        """Split out from _on_pan_tick so a test can drive a deterministic
        dt. Stops the timer when the view is clamped in every held direction
        and nothing actually moved."""
        if self._middle_drag_active:
            return  # two mechanisms, one pair of scrollbars
        if not self._pan_held:
            self._stop_pan()
            return
        dx = sum(d[0] for d in self._pan_held.values())
        dy = sum(d[1] for d in self._pan_held.values())
        # No diagonal normalization, deliberately: two keys held gives
        # ~1.41x, exactly as middle-drag does not normalize either.
        px = settings.get_pan_speed() * dt_ms / 1000.0
        self._pan_accum_x += dx * px
        self._pan_accum_y += dy * px
        step_x = int(self._pan_accum_x)
        step_y = int(self._pan_accum_y)
        self._pan_accum_x -= step_x
        self._pan_accum_y -= step_y
        if step_x == 0 and step_y == 0:
            return
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        before = (h.value(), v.value())
        self._scroll_by(step_x, step_y)
        if (h.value(), v.value()) == before:
            self._stop_pan()

    def focusOutEvent(self, event) -> None:
        """Alt+Tab mid-hold never delivers a release, so the timer would
        otherwise run forever."""
        self._pan_held.clear()
        self._stop_pan()
        super().focusOutEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """Shift during a shape drag (the matching half of keyPressEvent's
        live-constrain branch), Alt over Place Unit's hover ghost, and the end
        of a held pan.

        X11 emits release+press pairs while a key is held, both flagged
        isAutoRepeat(), so an auto-repeat release must not end the hold. A
        release for a key that nudged, or was never held at all, falls
        through silently."""
        self._flush_pending_move()
        if not event.isAutoRepeat():
            self._release_pan_key(event.key())
        if event.key() == Qt.Key_Shift and self._shape_anchor is not None:
            self._shape_modifiers = self._shape_modifiers & ~Qt.ShiftModifier
            self._update_shape_preview()
            return
        if event.key() == Qt.Key_Alt and self._place_hover is not None:
            pos, modifiers = self._place_hover
            self._place_hover = (pos, modifiers & ~Qt.AltModifier)
            self.refresh_place_ghost()
        super().keyReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        # Before the unconditional _clear_highlight() below, which would
        # otherwise nuke the rubber band. Cancelled only when NO button is
        # held: dragging a rectangle past the viewport edge on a zoomed view
        # is ordinary, and Qt's implicit grab still routes the release back
        # here. leaveEvent gets a bare QEvent with no button state of its
        # own, hence the QApplication query.
        shape_held = self._shape_anchor is not None and QApplication.mouseButtons() != Qt.NoButton
        if self._shape_anchor is not None and not shape_held:
            self._cancel_shape()
        # A drag that leaves without a release (focus lost mid-drag) must not leave a half-open stroke.
        # Closed before the clears below: a Cliff stroke's close repaints the highlight and hover ghost.
        if self._stroke_active:
            self._end_stroke()
        if not shape_held:
            self._clear_highlight()
        self._clear_pan_highlight()
        # Hover only -- a selection deliberately survives the cursor leaving
        # the map, the same way it survives moving onto empty ground.
        self._clear_unit_hover()
        # A tile pick's hover goes; a held rect or area drag does not, its release still commits.
        if self._tile_pick_anchor is None and self._area_drag is None:
            self._clear_tile_pick_preview()
        if self._area_drag is None:
            self._set_area_hover(None)
        # Defensive, same reasoning as the stroke close above: a marquee
        # drag that exits the widget without a release must not leave a
        # dangling rectangle on screen with no pending selection to resolve
        # it into. Cancelled outright rather than resolved here -- the
        # press-time on_click_select() call already set the "plain click"
        # selection state, and there is no natural rect to finalize against
        # a cursor that has left the view.
        if self._marquee_start_pos is not None:
            self._marquee_start_pos = None
            self._marquee_press_tile = None
            self._clear_marquee()
        # The ghost only, not _unit_drag_key: Qt's implicit grab still routes
        # the eventual release back here, so a drag that merely crosses the
        # viewport edge must still be able to commit its move. What must not
        # survive is a preview drawn at a destination the cursor has left.
        # Place Unit's hover ghost (GH #128) goes with it, and its hover point.
        self._clear_unit_ghost()
        self._place_hover = None
        self._cliff_hover = None
        # Defensive, same reasoning as the marquee above: cancelled outright,
        # not committed -- there is no natural release position to resolve
        # against. The committed region (if any) is untouched, exactly like
        # an Escape-cancel.
        if self._select_anchor is not None or self._move_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._move_anchor = None
            self._move_current = None
            self._close_region_drag_trace()
            self._update_region_overlay()

    def _end_wheel_gesture(self) -> None:
        """Drops the sub-notch residual, the running direction and the
        current gesture's mip budget, so the next wheel event starts a
        gesture of its own."""
        self._wheel_accum = 0
        self._wheel_dir = 0
        self._wheel_gesture_mip = None
        self._wheel_last_t = 0.0
        self._wheel_budget_t = 0.0

    def _wheel_zoom_factor(self, notches: int) -> float | None:
        """The zoom factor for a pending notch count, reduced until it is
        inside the floor/ceiling AND within the budget
        (_crosses_wheel_mip_budget()). Each notch moves one point along the level lattice
        (_wheel_lattice(), TASK-600), so every level's stop is a landing. A
        step that would overshoot a bound lands exactly on it instead, so the
        readout can reach 50% and 6400%. None refuses the whole event: already
        at the bound, or every step crosses the budget."""
        current = abs(self.transform().determinant()) ** 0.5
        lattice = self._wheel_lattice()
        while notches != 0:
            if lattice is not None:
                points, per = lattice
                device = zoom_stops.notch_target(current * per, notches, points)
                target = math.inf if notches > 0 else 0.0
                if device is not None:
                    target = device / per
            else:
                target = current * zoom_stops.NOTCH_STEP**notches
            factor = target / current
            too_small = notches < 0 and self._min_linear_scale is not None and target < self._min_linear_scale
            too_big = notches > 0 and self._max_linear_scale is not None and target > self._max_linear_scale
            if too_small or too_big:
                factor = (self._min_linear_scale if too_small else self._max_linear_scale) / current
                # 1e-9: float drift after a clamp must not read as room for one more sliver.
                moves = factor > 1.0 + 1e-9 if notches > 0 else factor < 1.0 - 1e-9
                if not moves:
                    return None
            elif not math.isfinite(factor) or factor <= 0:
                return None
            if not self._crosses_wheel_mip_budget(factor):
                return factor
            notches += -1 if notches > 0 else 1
        return None

    def _crosses_wheel_mip_budget(self, factor: float) -> bool:
        """Whether zooming by `factor` would land more than one mip level
        from the gesture's budget anchor on a level that would build first
        (TASK-614: a resident level is no placeholder and no sync build, so a
        roll passes it unrefused). False with no canvas item (nothing selects
        a level) and on a single-level ladder, where mip_for_scale() clamps
        every scale to the same answer."""
        start = self._wheel_gesture_mip
        if start is None:
            return False
        mip = self._mip_for_zoom_factor(factor)
        if mip is None or abs(mip - start) <= 1:
            return False
        return not self._canvas_item._cache.paints_without_build(mip)

    def _wheel_notches(self, accum: int, delta: int) -> tuple[int, int]:
        """(whole notches, new residual) for a non-zoom wheel action's
        accumulator: one step per whole WHEEL_NOTCH_UNITS, high-res deltas
        accumulated, a reversal dropping the stale residual as the zoom does."""
        total = accum + delta
        if accum and (accum > 0) != (delta > 0):
            total = delta
        notches = int(total / self.WHEEL_NOTCH_UNITS)
        return notches, total - notches * self.WHEEL_NOTCH_UNITS

    def _place_wheel_rotate(self, delta: int) -> None:
        """The wheel's rotate binding (Ctrl+wheel by default) over Place Unit's
        or the Unit Brush's ghost (GH #172, GH #355), a live unit drag (GH
        #280) or, under the "_selection" choices, the Units-mode selection
        (GH #356), one fine step per notch.
        Scroll up is PLACE_WHEEL_DIRECTION."""
        notches, self._place_wheel_accum = self._wheel_notches(self._place_wheel_accum, delta)
        if notches:
            self._on_place_rotate(notches * self.PLACE_WHEEL_DIRECTION)

    def _brush_wheel_step(self, delta: int) -> None:
        """The wheel's brush-size binding: scroll up is one size bigger per
        notch. At either bound the step does nothing, and never zooms instead."""
        notches, self._brush_wheel_accum = self._wheel_notches(self._brush_wheel_accum, delta)
        if notches:
            self._on_wheel_brush_step(notches)

    def _paste_wheel_turn(self, delta: int) -> None:
        """The wheel's rotate binding over a just-pasted region (GH #245): one
        quarter turn per whole notch, scroll up the ghost's way round."""
        notches, self._paste_wheel_accum = self._wheel_notches(self._paste_wheel_accum, delta)
        for _ in range(abs(notches)):
            if not self._wheel_paste_query():
                break  # the last turn lapsed the paste, so the next one would only log a refusal
            self._on_wheel_paste_turn(1 if notches > 0 else -1)

    def _wheel_pan(self, action: str, delta: int) -> None:
        """The wheel's pan bindings: WHEEL_PAN_NOTCH_PX per notch, scroll up
        moving the view up (pan_vertical) or left (pan_horizontal)."""
        if self._middle_drag_active:
            return  # as _pan_step: two mechanisms, one pair of scrollbars
        px = -round(delta * self.WHEEL_PAN_NOTCH_PX / self.WHEEL_NOTCH_UNITS)
        if action == "pan_vertical":
            self._scroll_by(0, px)
        else:
            self._scroll_by(px, 0)

    def _wheel_delta(self, event) -> int:
        """The event's vertical angleDelta, or its sideways one when Alt or
        Shift is held and the vertical is 0: Qt 5's xcb and Windows plugins
        deliver Alt+wheel transposed, and macOS turns Shift+wheel into a
        sideways scroll (TASK-363, fixed unchecked on a Mac). A sideways tilt
        with neither held stays 0."""
        angle = event.angleDelta()
        if angle.y() == 0 and event.modifiers() & (Qt.AltModifier | Qt.ShiftModifier):
            return angle.x()
        return angle.y()

    def _wheel_action(self, modifiers) -> str:
        """The settings.WHEEL_ACTIONS id a wheel event runs, with "tool"
        resolved, or the internal "paste_turn". The slot is the exact modifier
        set, so any other combination zooms. A tool action with nothing to act
        on is "none", so only a slot bound to zoom zooms (GH #245, user
        2026-10-25): rotate needs a live unit drag, Place Unit's or the Unit
        Brush's ghost showing or a just-pasted region, brush_size a tool with
        a brush. Under
        "tool" the brush wins over the paste, so a resize right after a paste
        still resizes. A live drag comes first: it is keyed on state, not the
        tool, so a keybound switch to Place Unit mid-drag still turns the drag.
        The Unit Brush's ghost turns under "rotate" only (GH #355): under
        "tool" its brush size wins, as above.
        "rotate_selection" and "tool_selection" (GH #356) resolve as "rotate"
        and "tool", then fall back to turning the Units-mode selection through
        the same hook as the Rotate keys. A committed region never turns by
        wheel under any choice."""
        action = self._wheel_bindings.get(int(modifiers) & self._WHEEL_MODIFIER_MASK, "zoom")
        rotates = action in ("rotate", "rotate_selection")
        tools = action in ("tool", "tool_selection")
        if (rotates or tools) and self.unit_drag_live():
            return "rotate"
        ghost_slot = rotates or (tools and self._tool == "place_unit")
        if ghost_slot and self._tool in _HOVER_GHOST_TOOLS and self._place_ghost_query():
            return "rotate"
        if rotates and self._wheel_paste_query():
            return "paste_turn"
        if (tools or action == "brush_size") and self._wheel_brush_query():
            return "brush_size"
        if tools and self._wheel_paste_query():
            return "paste_turn"
        if action in ("rotate_selection", "tool_selection") and self._wheel_selection_query():
            return "rotate"
        if rotates or tools or action == "brush_size":
            return "none"
        return action

    def wheelEvent(self, event):
        """Runs the wheel action bound to the event's modifiers (Settings >
        Keybinds, see _wheel_action). The rest of this is the zoom action: it
        zooms by whole WHEEL_NOTCH_UNITS notches, at most one mip level per
        gesture.

        Accumulating angleDelta rather than taking every event as a full step
        is what keeps a high-resolution wheel or a trackpad, which sends many
        small deltas, to one zoom step per real detent instead of one per
        event, and each step is a full-viewport repaint. A coarse wheel
        sends exactly one notch per detent, so it behaves as it always has,
        including the ceiling/floor refusal.

        A GESTURE is a run of same-direction events with no
        WHEEL_GESTURE_GAP_S pause between them: one flick of the wheel.
        Its budget is one mip level from an anchor: notches past the next
        boundary are dropped rather than banked, so a hard flick lands one
        level away instead of two. Two would land outside
        level_warm.neighbour_mips()'s +/-1 warm set: a placeholder frame from
        a resident level while that level warms (TASK-031.25), or, where no
        placeholder applies, a synchronous level build inside paint().
        A level already resident costs neither, so it spends no budget
        (TASK-614). Pausing (or reversing) starts a fresh
        gesture. A roll that keeps going re-anchors the budget once
        WHEEL_GESTURE_GAP_S has passed since the last crossing: a rate limit
        of one level per gap, never a stop (it used to refuse until the wheel
        went idle, 2026-09-30 stress log)."""
        # Level-warm ticks run on the short input budget while the user drives the canvas (TASK-031.80).
        level_warm.note_input()
        delta = self._wheel_delta(event)
        if delta == 0:
            return
        # Every wheel action, refused notches included, defers the gen-2 collect (TASK-031.81).
        gc_hold.note_input()
        action = self._wheel_action(event.modifiers())
        # Every other action returns before the gesture state, so it spends no zoom budget (GH #172).
        if action != "zoom":
            if action == "rotate":
                self._place_wheel_rotate(delta)
            elif action == "paste_turn":
                self._paste_wheel_turn(delta)
            elif action == "brush_size":
                self._brush_wheel_step(delta)
            elif action in ("pan_vertical", "pan_horizontal"):
                self._wheel_pan(action, delta)
            event.accept()
            return
        now = time.monotonic()
        # Direction is tracked separately from the residual, which is 0 after
        # a whole detent: rolling in then straight back out is the common
        # reversal, and it must not spend one budget in both directions.
        flipped = self._wheel_dir != 0 and (delta > 0) != (self._wheel_dir > 0)
        if flipped or now - self._wheel_last_t >= self.WHEEL_GESTURE_GAP_S:
            self._end_wheel_gesture()
        self._wheel_dir = 1 if delta > 0 else -1
        # Stamped on every event, including one whose notches are all trimmed
        # away: otherwise a refused event would let the next detent open a
        # fresh gesture and step around the one-mip budget.
        self._wheel_last_t = now
        total = self._wheel_accum + delta
        magnitude = abs(total) // self.WHEEL_NOTCH_UNITS
        notches = magnitude if total > 0 else -magnitude
        # Only sub-notch delta carries over; whole notches the trim refuses
        # are dropped, not banked for the next event.
        self._wheel_accum = total - notches * self.WHEEL_NOTCH_UNITS
        if notches == 0:
            return
        if self._wheel_gesture_mip is None:
            self._wheel_gesture_mip = self._mip_for_zoom_factor(1.0)
            self._wheel_budget_t = now
        factor = self._wheel_zoom_factor(notches)
        if factor is None and now - self._wheel_budget_t >= self.WHEEL_GESTURE_GAP_S:
            self._wheel_gesture_mip = self._mip_for_zoom_factor(1.0)
            self._wheel_budget_t = now
            factor = self._wheel_zoom_factor(notches)
        if factor is None:
            return
        mip_before = self._mip_for_zoom_factor(1.0)
        anchor = event.position().toPoint() if self._zoom_on_cursor else self.viewport().rect().center()
        self._last_zoom_anchor = anchor
        # The repaint this schedules lands on the next `perf view` line, with its mip change.
        with perf_trace.op("zoom"):
            # Shared with the pre-rendered frames' twin, which predicts this transform (TASK-031.71).
            zoom_preframe.zoom_about(self, factor, anchor)
            if self.at_level_stop():
                self._snap_to_device_px()
            self._preframe_dir = 1 if factor > 1.0 else -1
            self._note_viewport_changed()
            self._on_zoom_changed()
        # A crossing restarts the gap before the budget may re-anchor past it.
        if self._mip_for_zoom_factor(1.0) != mip_before:
            self._wheel_budget_t = now

    def viewportEvent(self, event) -> bool:
        """Coalesces mouse moves to at most one handled move per paint
        (GH #179). Qt5's Wayland plugin delivers every motion event, so a drag
        whose step outlasts the motion interval otherwise queues steps faster
        than it drains them; xcb compresses them for us, this does it on
        every platform.

        A move is stashed (newest wins) and handed to mouseMoveEvent by
        _flush_pending_move(): on the next loop turn once the viewport has
        painted since the last handled move, or after MOVE_PAINT_WAIT_MS when
        nothing paints. Press, release, double-click, wheel and leave handle
        a pending move first, as do the key handlers, so they always see the
        last position. Middle-drag and hand-drag pans are delta-based, so
        dropping the moves in between loses nothing. mouseMoveEvent itself is
        unchanged, and direct callers (tests, tools) stay synchronous."""
        kind = event.type()
        if kind == QEvent.MouseMove and self._move_flush_timer is not None:
            if not self.isEnabled():
                # Ahead of QWidget::event's own disabled check, so Qt drops it as before.
                return super().viewportEvent(event)
            self._move_stash = (
                QPointF(event.localPos()),
                QPointF(event.windowPos()),
                QPointF(event.screenPos()),
                event.buttons(),
                event.modifiers(),
            )
            perf_trace.note_move_stashed()
            if self._move_gate_open:
                self._move_flush_timer.start()
            elif not self._move_fallback_timer.isActive():
                self._move_fallback_timer.start()
            event.accept()
            return True
        try:
            if kind in self._MOVE_FLUSH_FIRST_EVENTS:
                self._flush_pending_move()
            elif kind == QEvent.Hide:
                self._drop_pending_move()
        finally:
            # A raising move handler must not also cost the press/release behind it.
            result = super().viewportEvent(event)
        if kind == QEvent.Paint and self._move_flush_timer is not None:
            self._move_gate_open = True
            self._move_fallback_timer.stop()
            if self._move_stash is not None:
                self._move_flush_timer.start()
        return bool(result)

    def event(self, event) -> bool:
        # A shortcut fires from ShortcutOverride, ahead of keyPressEvent's own flush.
        if event.type() == QEvent.ShortcutOverride:
            self._flush_pending_move()
        return super().event(event)

    def flush_pending_move(self) -> None:
        """For window-level actions that read the hovered tile (Paste
        Region): lands a coalesced move first, since a QAction shortcut
        never passes through this view's key handlers when it lacks focus."""
        self._flush_pending_move()

    def _flush_pending_move(self) -> None:
        """Handles the stashed move, if any. The gate closes before the
        handler, so a paint inside it reopens the gate rather than being
        overwritten; the fallback is armed after it."""
        if self._move_stash is None:
            return
        self._move_flush_timer.stop()
        self._move_fallback_timer.stop()
        local, window, screen, buttons, modifiers = self._move_stash
        self._move_stash = None
        if not self.isEnabled():
            return
        self._move_gate_open = False
        perf_trace.note_move_handled()
        self.mouseMoveEvent(QMouseEvent(QEvent.MouseMove, local, window, screen, Qt.NoButton, buttons, modifiers))
        self._move_fallback_timer.start()

    def _on_move_fallback(self) -> None:
        """No paint within MOVE_PAINT_WAIT_MS of the last handled move (a
        hover that changed nothing, or an occluded Wayland window): reopen."""
        self._move_gate_open = True
        self._flush_pending_move()

    def _drop_pending_move(self) -> None:
        """Discards a pending move without handling it: the scene it was
        aimed at is being torn down or replaced."""
        self._move_stash = None
        self._move_gate_open = True
        if self._move_flush_timer is not None:
            self._move_flush_timer.stop()
            self._move_fallback_timer.stop()

    def mouseMoveEvent(self, event):
        if self._zoom_on_cursor:
            self.retarget_preframes()  # the next notch's anchor moved
        if self._middle_drag_active:
            delta = event.pos() - self._middle_drag_last_pos
            self._middle_drag_last_pos = event.pos()
            self._scroll_by(-delta.x(), -delta.y())
            return
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        with perf_trace.phase("pick"):
            tile = self._pick_tile(pos)
        self._on_hover(tile)
        # In Stepped mode tile is not None already *is* the on-map test
        # (that's exactly what _pick_tile/_pos_on_map do there) -- avoid
        # re-running screen_to_tile a second time for the same pixel on
        # every mouse move by not routing through _pos_on_map here too.
        # Sloped shares this branch for the same reason, and since C4 landed
        # it does so with a real backend: its _pick_tile answers from the
        # pick plane, so hover/outline work here with no change of its own.
        if self._terrain_style in _ELEVATED_STYLES:
            on_map = tile is not None
        else:
            on_map = self._map_rect is not None and self._map_rect.contains(pos)
        # GH #138's tile pick preview: a held rect drag clamps, a hover shows only on-map tiles.
        if self._tile_picker_kind is not None:
            self._clear_highlight()
            self._clear_pan_highlight()
            self._clear_unit_hover()
            if self._tile_pick_anchor is not None:
                self._update_tile_pick_preview(self._tile_pick_sample(pos))
            else:
                self._update_tile_pick_preview(tile if on_map and tile is not None and self._on_map(tile) else None)
            return
        # GH #229: a live drag redraws its preview, clamped like a rect pick; else the hover zone.
        # A move with the button up means its release was swallowed: the drag dies unwritten.
        if self._area_drag is not None and not event.buttons() & Qt.LeftButton:
            self._cancel_area_drag()
        if self._area_drag is not None:
            self._clear_highlight()
            self._clear_pan_highlight()
            end = self._tile_pick_sample(pos)
            if end is not None:
                self._area_drag_end = end
            self._update_area_drag_preview()
            return
        if self._area_edit is not None or self._location_edit is not None:
            self._set_area_hover(self._area_hit(pos))
        # Pick from map's hover, the Units-mode unit cue in whatever mode is showing.
        if self._unit_picker_active:
            self._clear_highlight()
            self._clear_pan_highlight()
            in_canvas = self._map_rect is not None and self._map_rect.contains(pos)
            entry = self.pick_unit_at(pos, tile) if in_canvas else None
            if entry is None:
                self._clear_unit_hover()
            else:
                self._update_unit_hover(entry)
            return
        # The Ruler, above the mode branch for the same reason its press
        # handler is: the Units branch below clears both tile cues, so a ruler
        # drag in Units mode would lose the hover outline showing which tile
        # the next click snaps to. Reuses Pan's quiet cue rather than the edit
        # tools' pulsing gold, which is reserved for "about to paint".
        if self._tool == TOOL_RULER:
            self._clear_highlight()
            self._clear_unit_hover()
            if on_map:
                self._update_pan_highlight(*tile)
                if self._ruler.move(tile):
                    self._update_ruler()
            else:
                # The endpoint deliberately holds its last valid tile rather
                # than collapsing the line, which is also what keeps a drag
                # readable across a Stepped skirt face (_pick_tile returns
                # None there, a documented residual).
                self._clear_pan_highlight()
            return
        # A sibling of the Ruler/Select branches, NOT nested inside the
        # generic EDIT_TOOLS branch below -- that branch's own
        # _update_highlight(*tile) call is exactly what would overwrite the
        # rubber band on every move.
        if self._shape_drag_live():
            self._clear_pan_highlight()
            self._clear_unit_hover()
            if self._shape_anchor is not None:
                # An off-map move mid-drag holds the last valid endpoint
                # rather than collapsing the shape -- same convention as the
                # Ruler above, and the only option available, since there is
                # no tile there to clamp to.
                if tile is not None:
                    self._shape_end = tile
                self._shape_modifiers = event.modifiers()
                self._update_shape_preview()
            elif on_map:
                # Before the press these tools show the ordinary brush hover
                # cue, so the cursor still says what a press would paint.
                with perf_trace.phase("highlight"):
                    self._update_highlight(*tile)
            else:
                self._clear_highlight()
            return
        if self._tool == TOOL_SELECT and self._mode != "units":
            self._clear_highlight()
            self._clear_pan_highlight()
            self._clear_unit_hover()
            if self._move_anchor is not None and tile is not None:
                # Same "hold the last valid tile" convention as the select
                # drag below -- a move legitimately runs off the map edge and
                # back, and the block is re-pasted at an unclipped anchor.
                self._move_current = tile
                self._refresh_live_region()
            elif self._select_anchor is not None and tile is not None:
                # Same "hold the last valid tile" convention as the Ruler
                # above: an off-map move mid-drag must not collapse the
                # preview, since the drag legitimately runs off the edge.
                self._select_current = tile
                self._refresh_live_region()
            # Advertises the gesture. _apply_tool_cursor() stays the
            # tool/mode-level default and is deliberately not touched.
            if self._move_anchor is not None or (self._region_movable and self._tile_in_region(tile)):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor)
            return
        # Gated on MODE, so a unit pick costs nothing per pixel in
        # View/Terrain/Triggers. Both TILE highlights are cleared here: Units
        # mode addresses units, and leaving a tile cue on screen would suggest
        # the click does something tile-shaped. Pan is the exception: it only
        # moves the camera, so with no drag or marquee still live from another
        # tool it takes the final branch's tile cue, as in every other mode.
        if self._mode == "units" and (
            self._tool != "pan" or self._unit_drag_key is not None or self._marquee_start_pos is not None
        ):
            self._clear_highlight()
            self._clear_pan_highlight()
            # Convert (b2.5) and the Eraser are mutating brushes like Draw/Elevate, not a
            # selection gesture -- it gets their gold pulsing "about to
            # paint" cue via the same _update_highlight()/_touch_tile() pair
            # the generic EDIT_TOOLS branch below uses, just reached from
            # here since Convert lives in Units mode (mousePressEvent's own
            # units-branch exclusion is the other half of this routing).
            # A drag or marquee begun before a keybound switch keeps its own branch below (TASK-562).
            if self._tool in _UNIT_BRUSH_TOOLS and self._unit_drag_key is None and self._marquee_start_pos is None:
                self._clear_unit_hover()
                if on_map:
                    with perf_trace.phase("highlight"):
                        self._update_highlight(*tile)
                    if self._stroke_active:
                        self._touch_tile(*tile, event.modifiers())
                else:
                    self._clear_highlight()
                # GH #270. The point stays live mid-stroke, for the release.
                if self._tool == "unit_brush":
                    self._place_hover = (pos, event.modifiers())
                    if not self._stroke_active:
                        self._show_place_ghost(pos, event.modifiers(), on_map)
                return
            # GH #128: Place Unit's hover ghost. A place click never selects, so
            # the hover outline is the viewer's: the host it would garrison into
            # (TASK-549), set or cleared by on_place_preview or the off-map clear.
            # A drag or marquee begun before a keybound switch to Place Unit keeps its own branch.
            if self._tool == "place_unit" and self._unit_drag_key is None and self._marquee_start_pos is None:
                self._place_hover = (pos, event.modifiers())
                self._show_place_ghost(pos, event.modifiers(), on_map)
                return
            if self._unit_drag_key is not None:
                # A unit drag is in progress (b1.5): the hover cue would
                # misleadingly highlight whatever the cursor happens to be
                # crossing, so it is cleared here the way every sibling branch
                # clears it. Before the mid-drag preview this branch returned
                # without clearing, which froze the cyan outline over the
                # source unit for the whole drag. Past the threshold the
                # preview owns it instead: the drop host, or nothing (GH #115).
                self._unit_drag_moved(event.pos(), event.modifiers())
                return
            if self._marquee_start_pos is not None:
                # b2.3: a marquee drag is in progress -- update the rubber
                # band instead of the hover cue, for the same "no live
                # preview of the eventual result" reason as the unit-drag
                # branch above (the candidate set is resolved once, at
                # release, not recomputed every frame). Drawn only while it
                # engages, the same test the release decides by.
                self._clear_unit_hover()
                if self._marquee_engaged(event.pos(), tile):
                    self._update_marquee(self._marquee_start_pos, event.pos())
                else:
                    self._clear_marquee()
                return
            # Canvas containment, NOT `on_map`: in Stepped, on_map is
            # "screen_to_tile resolved a tile", which is False on a skirt
            # face -- and a unit standing at the top of a skirt is visible
            # exactly there. Gating on on_map would make those pixels
            # unhoverable. See pick_unit()'s documented skirt residual.
            in_canvas = self._map_rect is not None and self._map_rect.contains(pos)
            # This same pixel's tile, resolved at the top of this handler;
            # nothing since has touched the geometry it came from.
            entry = self.pick_unit_at(pos, tile) if in_canvas else None
            if entry is None:
                self._clear_unit_hover()
            else:
                self._update_unit_hover(entry)
        elif self._tool in EDIT_TOOLS:
            self._clear_pan_highlight()
            if self._tool == "cliff" and self._stroke_active:
                # Kept current mid-stroke, so the ghost returns here at release.
                # The backing field: the run's head footprint stays off the map.
                self._cliff_hover_tile = tile if on_map else None
            if on_map:
                with perf_trace.phase("highlight"):
                    self._update_highlight(*tile)
                if self._stroke_active:
                    self._touch_tile(*tile, event.modifiers())
            else:
                self._clear_highlight()
            if self._tool == "cliff" and not self._stroke_active:
                self._cliff_hover = tile if on_map else None
                self._on_cliff_preview(self._cliff_hover)
        else:
            self._clear_highlight()
            self._clear_unit_hover()  # Pan in Units mode outlines no unit
            if on_map:
                self._update_pan_highlight(*tile)
            else:
                self._clear_pan_highlight()
