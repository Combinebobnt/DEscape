"""MapView: the QGraphicsView the map is panned, zoomed and edited in.

Dumb the same way TriggerPanel is -- it reports what the user did
through the callbacks ViewerWindow constructs it with and never
reaches back up into the window."""

from __future__ import annotations


import math

import numpy as np
from PyQt5.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


from descape import (
    brush,
    iso_geometry,
    perf_trace,
    ruler,
    settings,
    unit_pick,
)
from descape.render import SMALL_MAP_TILE_PIXELS
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)
from descape.viewer_canvas import EdgeTickItem, MapCanvasItem
from descape.viewer_common import CLICK_TOOLS, EDIT_TOOLS, TOOL_RULER

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Deliberately local: the same pair appears in
# render.py and viewer.py too, but render.py imports no PyQt5 at all, so a
# shared home has to be Qt-free -- viewer_common.py is not it.
_ELEVATED_STYLES = ("stepped", "sloped")


class MapView(QGraphicsView):
    # How far past each map edge the view is allowed to scroll, as a fraction of
    # that edge's length -- e.g. 0.4 lets you pan 40% of the map's width past its
    # left/right edges. Comfortable for working near border tiles without the
    # view refusing to scroll further, like the in-game editor allows.
    OVERSCROLL_FRACTION = 0.4
    OUTSIDE_MAP_COLOR = QColor(35, 35, 35)

    # AoE2's camera isn't a plain 45deg rotation -- it's a dimetric projection:
    # rotate 45deg, then squash vertically to roughly a 2:1 diamond (the classic
    # RTS "tile is twice as wide as it is tall" look). Since the rendered image
    # already stores tile (x, y) directly as pixel (x, y), that whole projection
    # is just this one linear transform applied to the existing pixmap -- no
    # per-tile re-rendering needed.
    # 45deg gives the correct diamond shape; the extra -90deg (Qt's rotate() is
    # clockwise on screen, so -90 = 90deg counterclockwise) matches AoE2's actual
    # camera facing, which is rotated 90deg from the "obvious" 45deg choice.
    ISO_ROTATION_DEGREES = 45 - 90
    ISO_VERTICAL_SQUASH = 0.5
    # How much of the viewport the isometric fit leaves the diamond, so its
    # corners don't sit flush against the edges. A class constant rather than
    # set_isometric()'s old local, because _fit_baseline_scale() has to
    # reproduce that method's arithmetic exactly and the two must not drift.
    ISO_FIT_MARGIN = 0.92

    # How far past "whole map fits in view" wheelEvent() allows zooming out or
    # in -- see _capture_zoom_baseline(). Zoom-out past this aliases badly
    # (real per-tile texture detail is too high-frequency for
    # QPainter.SmoothPixmapTransform's plain bilinear filtering to minify
    # cleanly without mipmapping); zoom-in past this just shows the same
    # finite per-tile texture resolution increasingly blurry/blocky, with no
    # more real detail to reveal.
    # Phase B-D-d divides the zoom-out half of this by the coarsest available
    # mip level's scale, and (2026-08-27) the zoom-in half by the finest
    # available mip level's scale, both in _capture_zoom_baseline() -- so the
    # ceiling actually reaches the finest enumerated level instead of
    # capping the user on the reference level with the finer levels
    # unreachable dead weight in the ladder.
    MIN_ZOOM_FRACTION_OF_FIT = 0.5
    MAX_ZOOM_MULTIPLE_OF_FIT = 64.0

    # Hover-highlight convention for every brush-style tool (Terrain,
    # Elevate, Set Elevation): an outline plus a translucent gold
    # fill that pulses steadily, so it reads as a live cursor following the
    # mouse rather than a static "you clicked here" marker. Follows on hover,
    # not just on click.
    HIGHLIGHT_OUTLINE_PEN = QPen(QColor(255, 215, 0), 0)
    HIGHLIGHT_FILL_COLOR = QColor(255, 215, 0)
    HIGHLIGHT_PULSE_MIN_ALPHA = 0.25
    HIGHLIGHT_PULSE_MAX_ALPHA = 0.55
    HIGHLIGHT_PULSE_PERIOD_MS = 500
    HIGHLIGHT_PULSE_TICK_MS = 40

    # Pan mode's own hover cue -- deliberately much quieter than the edit
    # highlight above (a static thin black outline, no fill, no pulse): Pan
    # isn't about to mutate anything, so it shouldn't compete visually with
    # the "this is live and about to paint" signal edit tools need. Cosmetic
    # pen width 0 keeps it a true 1-device-pixel line regardless of zoom,
    # same convention HIGHLIGHT_OUTLINE_PEN already uses.
    PAN_HIGHLIGHT_PEN = QPen(QColor(0, 0, 0), 0)

    # Units mode's two cues, phase 3's P3-d. Deliberately NOT the pulsing
    # gold HIGHLIGHT_OUTLINE_PEN reserves for "live and about to paint":
    # phase 3 mutates nothing, so it must not claim that signal. Hover is
    # thin and quiet like PAN_HIGHLIGHT_PEN; selection is solid plus a
    # translucent fill, in a blue distinct from gold. Neither pulses.
    UNIT_HOVER_PEN = QPen(QColor(255, 255, 255), 0)
    UNIT_SELECT_PEN = QPen(QColor(80, 170, 255), 2)
    UNIT_SELECT_FILL_COLOR = QColor(80, 170, 255, 70)

    # The codebase's FIRST setZValue use -- everything else stacks by scene
    # INSERTION order, and the existing highlight items only land on top
    # because they're created lazily on first hover, after set_source() has
    # added the canvas item. Two independently-lazily-created unit items
    # would stack in whichever order the user happened to trigger first, so
    # these are explicit rather than inheriting that latent ordering bug.
    UNIT_HOVER_Z = 10.0
    UNIT_SELECT_Z = 11.0

    # b1.5: how far a press must travel (screen pixels) before release counts
    # as a unit move rather than a plain select click -- a physically-held
    # mouse rarely lands at the exact press pixel, so a bare "did it move at
    # all" test would misfire on ordinary clicks.
    UNIT_DRAG_THRESHOLD_PX = 4

    # The Ruler's line, its two endpoint outlines and its label. Cosmetic
    # pen for the same zoom-invariance reason PAN_HIGHLIGHT_PEN gives, and
    # a fixed colour rather than a themed one: MapView paints its scene
    # background unconditionally to OUTSIDE_MAP_COLOR, so apply_theme never
    # reaches any of this. Orange is unclaimed here; gold, blue, white and
    # black already mean edit, selection, unit hover and pan.
    RULER_PEN = QPen(QColor(255, 130, 40), 0)
    RULER_LABEL_COLOR = QColor(255, 190, 110)
    # A dark cosmetic outline around the glyphs, so the label survives
    # both the near-black void and bright terrain without a backing rect.
    RULER_LABEL_OUTLINE = QColor(0, 0, 0, 230)
    RULER_LABEL_FONT_PX = 18
    RULER_LABEL_GAP_PX = 6.0
    # Above UNIT_SELECT_Z: a measurement is a deliberate act, and should
    # not be occluded by the hover cue it was drawn on top of.
    RULER_Z = 12.0
    RULER_LABEL_Z = 13.0

    def __init__(
        self,
        on_hover,
        on_stroke_start,
        on_stroke_tile,
        on_stroke_end,
        on_click_edit,
        on_click_select,
        on_unit_place,
        on_unit_move,
        on_unit_nudge,
        on_unit_delete,
        on_marquee_select,
        on_ruler_measured,
        on_ruler_changed,
        on_zoom_changed,
    ):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.scene().setBackgroundBrush(self.OUTSIDE_MAP_COLOR)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        # Holds the QGraphicsItem itself, not the chunk cache directly
        # (ViewerWindow keeps that reference; this class only ever needs to
        # forward paint-invalidation to the item -- see invalidate_region()).
        # Both terrain styles use this since Phase B-E (Flat previously used
        # a separate _pixmap_item/QGraphicsPixmapItem, retired in that
        # phase). Set by set_source(), cleared by clear_image().
        self._canvas_item: MapCanvasItem | None = None
        # Both halves of View > Distance Ticks live on MapView, which is
        # built once per ViewerWindow and outlives every set_source(), so
        # unlike _sprites_enabled they need no re-application in
        # _render_current(). If MapView ever becomes per-document, this
        # state moves up to ViewerWindow.
        self._edge_tick_item: EdgeTickItem | None = None
        self._edge_ticks_enabled = settings.get_distance_ticks()
        self._edge_tick_interval = settings.get_distance_tick_interval()
        # The Ruler's grammar lives in the Qt-free descape.ruler; this
        # class only turns its two tiles into scene items. Like the tick
        # state above it sits on MapView, which outlives every set_source().
        self._ruler = ruler.RulerSession()
        # Fired once per COMPLETED measurement, never per drag frame; see
        # _report_ruler for the edge it triggers on.
        self._on_ruler_measured = on_ruler_measured
        # Fired on every live update (including mid-drag) and with None on
        # every clear, unlike on_ruler_measured above -- this is what backs
        # a status-bar readout that has to track the drag, not just its end.
        self._on_ruler_changed = on_ruler_changed
        # Fired whenever the fit-relative zoom percentage may have changed --
        # see zoom_percent_of_fit() and its three call sites (wheelEvent,
        # _capture_zoom_baseline, clear_image) -- so a status-bar readout can
        # stay live without polling the transform on a timer.
        self._on_zoom_changed = on_zoom_changed
        self._ruler_line_item: QGraphicsLineItem | None = None
        self._ruler_end_items: list[QGraphicsPolygonItem] = []
        self._ruler_label_item: QGraphicsSimpleTextItem | None = None
        self._on_hover = on_hover
        # A "stroke" is one drag with an edit tool active, from press to
        # release (or a defensive close on leaveEvent) -- see mousePressEvent/
        # mouseMoveEvent/mouseReleaseEvent/leaveEvent below. on_stroke_start()
        # takes no args; on_stroke_tile(tile_x, tile_y, modifiers) fires once
        # per newly-entered tile (deduped within the stroke -- see
        # _touch_tile); on_stroke_end() closes it. ViewerWindow uses this to
        # keep exactly one descape.edit_history.EditHistory record per stroke
        # while still repainting live as the drag progresses.
        self._on_stroke_start = on_stroke_start
        self._on_stroke_tile = on_stroke_tile
        self._on_stroke_end = on_stroke_end
        # CLICK_TOOLS (Paint Can, and any future tool with no meaningful
        # drag semantics) never open a stroke at all -- on_click_edit(tile_x,
        # tile_y, modifiers) fires once per left press instead, handled by a
        # dedicated branch in mousePressEvent above the stroke block, so the
        # begin_stroke/commit_stroke pairing above stays exactly one-to-one
        # with no special-casing.
        self._on_click_edit = on_click_edit
        # Units mode's selection click (phase 3's P3-d), the sixth injected
        # callable. MapView emits no pyqtSignal anywhere -- plain callables
        # are this repo's established MapView -> ViewerWindow convention, and
        # a sixth follows it rather than introducing a second mechanism.
        self._on_click_select = on_click_select
        # Phase 3.5b's b1.4/b1.5, the seventh and eighth injected callables.
        # on_unit_place(scene_pos, modifiers) fires on a Place-tool click;
        # on_unit_move(key, scene_pos, modifiers) fires when a press-drag-
        # release on an already-picked unit moves past _UNIT_DRAG_THRESHOLD_PX
        # (see mousePressEvent/mouseReleaseEvent) -- key is the (player_id,
        # reference_id) pair picked at press time, resolved late by
        # ViewerWindow exactly like on_click_select's own selection key.
        self._on_unit_place = on_unit_place
        self._on_unit_move = on_unit_move
        # b1.5/b1.6, the ninth and tenth injected callables.
        # on_unit_nudge(dx_tiles, dy_tiles, modifiers) fires on an arrow key
        # in Units mode; on_unit_delete(modifiers) fires on Delete. Both are
        # keyboard-only (no MapView state to track), unlike place/move above.
        self._on_unit_nudge = on_unit_nudge
        self._on_unit_delete = on_unit_delete
        # b2.3, the eleventh injected callable. on_marquee_select(keys,
        # modifiers) fires once a completed marquee drag (a press-drag-
        # release on EMPTY ground past UNIT_DRAG_THRESHOLD_PX, Pan tool,
        # Units mode) is resolved to tile space -- keys are the (player_id,
        # reference_id) pairs of every unit units_in_rect() found, already
        # filter-respecting. A drag that never exceeds the threshold is a
        # plain click, already handled by the on_click_select() call
        # mousePressEvent makes unconditionally at press time (see that
        # method's own comment on why the click fires regardless).
        self._on_marquee_select = on_marquee_select
        # MapView has historically known only about TOOLS, never modes. Units
        # mode needs a top-level branch in mousePressEvent/mouseMoveEvent, so
        # the mode has to be mirrored here, set by ViewerWindow.on_mode_
        # changed() alongside its own self.mode update.
        self._mode = "view"
        # The pick index for the current scenario+filter, set by set_source()
        # in the SAME call that sets _iso_elevations/_iso_proj -- see that
        # method's docstring for why (Risk #6, the stale-snapshot class of
        # bug); the index must join that discipline rather than sit outside it.
        self._unit_index = None
        self._unit_hover_item: QGraphicsPathItem | None = None
        self._unit_select_item: QGraphicsPathItem | None = None
        self._unit_select_fill_item: QGraphicsPathItem | None = None
        # Memoized on the picked (player_id, reference_id) key, the same way
        # _highlight_key memoizes the tile highlight: mouseMoveEvent runs a
        # pick on every pixel, and rebuilding a 25-diamond path per pixel
        # would be pure waste.
        self._unit_hover_key: tuple[int, int] | None = None
        # b2.2 generalizes this from a single key to a list -- see
        # set_unit_selection()'s own docstring for why MapView keeps a copy
        # at all rather than only ever being pushed one by ViewerWindow.
        self._unit_select_keys: list[tuple[int, int]] = []
        # b1.5's move-by-drag: the unit key mousePressEvent picked (if any),
        # and the press position, both set in the Units-mode branch and
        # consumed (cleared either way) by the matching mouseReleaseEvent --
        # see _UNIT_DRAG_THRESHOLD_PX for why a press+release with no real
        # drag is a plain click, not a move.
        self._unit_drag_key: tuple[int, int] | None = None
        self._unit_drag_press_pos: QPointF | None = None
        # b2.3's marquee: set only when a Units-mode press with the Pan tool
        # lands on EMPTY ground (a press that hits a unit is b1.5's move-drag
        # above, never both at once). Holds the PRESS-TIME device position;
        # mouseReleaseEvent decides click vs. drag off the same
        # UNIT_DRAG_THRESHOLD_PX b1.5 already uses, then converts start/end
        # into a scene rect and hands it to _units_in_scene_rect().
        self._marquee_start_pos: QPointF | None = None
        self._marquee_item: QGraphicsRectItem | None = None
        self._stroke_active = False
        # Keyed on the CURSOR tile, not the painted tile -- a cheap early
        # out only (skip re-entering _touch_tile/on_stroke_tile for a mouse
        # move that hasn't left the current cursor tile), not a correctness
        # guarantee. With a brush bigger than one tile, one painted tile
        # falls under many distinct cursor tiles during a drag, so the
        # tile-level dedupe that actually matters (e.g. Elevate's
        # accumulating +/-1 must apply once per stroke, not once per cursor
        # tile that overlapped it) is ViewerWindow._stroke_painted instead.
        self._stroke_touched: set[tuple[int, int]] = set()
        # Brush footprint for the hover-preview highlight -- ViewerWindow
        # keeps the authoritative (size, shape) and pushes it here via
        # set_brush() whenever the toolbar spinbox/combo changes. Size 1 /
        # square makes brush_tiles() degrade to exactly the old single-tile
        # footprint, so a tool with supports_brush=False (or before any map
        # is loaded) previews correctly with no special-casing.
        self._brush_size = brush.BRUSH_SIZE_MIN
        self._brush_shape = brush.BRUSH_SHAPE_SQUARE
        # Set by set_source() alongside each render, cleared by clear_image()
        # -- needed to clip the brush footprint the same way
        # ViewerWindow.on_edit_stroke_tile clips a single tile today.
        self._map_width: int | None = None
        self._map_height: int | None = None
        # Memoizes the last highlight built, so mouseMoveEvent's per-pixel
        # calls into _update_highlight() don't rebuild an up-to-81-tile
        # QPainterPath every time the mouse moves within the same tile.
        self._highlight_key: tuple[int, int, int, str] | None = None
        self._map_rect: QRectF | None = None
        # Set by set_source() just before it calls set_isometric(), consumed
        # (and cleared) by set_isometric() itself -- see set_source()'s
        # reset_view parameter. None means "reset to fit", matching every
        # pre-existing caller's behavior.
        self._pending_view_restore: tuple[float, QPointF] | None = None
        # Real value is set by set_source() alongside each render. This default
        # only matters before any scenario has loaded, when mouseMoveEvent
        # still computes a (meaningless, discarded) tile coordinate for the
        # hover callback -- ViewerWindow.on_hover() no-ops while
        # self.scenario is None, so an unset/default value here is harmless.
        self._tile_pixels = SMALL_MAP_TILE_PIXELS
        self._isometric = True
        # "flat" (today's existing view-transform trick, plain terrain
        # color/texture with no elevation cue at all) or "stepped" (Phase
        # 1-5's real per-tile Z-height compositor -- Phase B-C composites it
        # lazily, chunk by chunk, via a MapCanvasItem/IsoChunkCache, not
        # baked into one pixmap up front -- see set_source()). Stepped is
        # the default (ViewerWindow's own default matches -- see its
        # Elevation View toolbar combo). Set for real by set_source();
        # _iso_elevations/_iso_proj carry the exact (h, w) elevation snapshot
        # and IsoProjection descape.render.elevations_and_proj()/
        # sloped_elevations_and_proj() computed for the current source from
        # (None in Flat, which has neither) -- Risk #6: these must never
        # drift from what's actually on screen, so they're only ever set
        # together with the source itself, in set_source(), never patched
        # independently.
        self._terrain_style = "stepped"
        self._iso_elevations: np.ndarray | None = None
        self._iso_proj: iso_geometry.IsoProjection | None = None
        self._tool = "pan"
        # QGraphicsPathItem, not QGraphicsPolygonItem: the edit-mode
        # highlight covers the whole brush footprint (up to
        # brush.BRUSH_SIZE_MAX**2 tiles), built as one QPainterPath with one
        # addPolygon() per tile -- see _update_highlight(). Still exactly
        # two scene items regardless of footprint size, so _on_pulse_tick()
        # (which only ever calls setOpacity() on _highlight_fill_item)
        # needs no change.
        self._highlight_outline_item: QGraphicsPathItem | None = None
        self._highlight_fill_item: QGraphicsPathItem | None = None
        # Pan mode's own quieter highlight -- see PAN_HIGHLIGHT_PEN above.
        # Always exactly one tile (Pan has no brush), so this stays a plain
        # QGraphicsPolygonItem.
        self._pan_highlight_item: QGraphicsPolygonItem | None = None
        # Which button started the active stroke (Qt.LeftButton raises
        # elevation, Qt.RightButton lowers it -- see _touch_tile). None
        # when no stroke is active.
        self._stroke_button: int | None = None
        self._pulse_phase_ms = 0
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._on_pulse_tick)
        # Middle-button pan is independent of the active tool/dragMode
        # (Qt's ScrollHandDrag only ever responds to the left button) --
        # handled manually via the scrollbars, see mouse{Press,Move,Release}Event.
        self._middle_drag_active = False
        self._middle_drag_last_pos = None
        # Set for real by _capture_zoom_baseline(), called from
        # set_isometric() once a map is loaded and a fit-to-view scale exists.
        self._min_linear_scale: float | None = None
        self._max_linear_scale: float | None = None
        self.set_zoom_anchor_mode(settings.get_zoom_centered_on_cursor())
        self.setMouseTracking(True)

    def set_zoom_anchor_mode(self, centered_on_cursor: bool) -> None:
        # QGraphicsView.scale() zooms around whatever transformationAnchor is
        # currently set to -- AnchorViewCenter (the default, "zoom in overall")
        # or AnchorUnderMouse ("zoom in centered on mouse cursor location").
        # Setting this once here means wheelEvent's plain self.scale(...) call
        # doesn't need to know or care which mode is active.
        self.setTransformationAnchor(
            QGraphicsView.AnchorUnderMouse if centered_on_cursor else QGraphicsView.AnchorViewCenter
        )

    def set_brush(self, size: int, shape: str) -> None:
        """Updates the brush the hover-preview highlight (and, via
        ViewerWindow reading these same values back at stroke time, the
        actual edit) uses. Only the preview needs to react here -- the
        active stroke's own footprint is computed fresh per cursor tile in
        ViewerWindow.on_edit_stroke_tile, not cached."""
        self._brush_size = size
        self._brush_shape = shape
        self._highlight_key = None  # force the next _update_highlight to rebuild

    def refresh_highlight(self, tile: tuple[int, int] | None) -> None:
        """Rebuilds the edit-mode hover highlight for `tile` right now,
        using whatever brush state set_brush() last set -- for a caller
        (ViewerWindow's brush size/shape widgets) that changes brush state
        without the mouse having moved, so the preview doesn't sit stale
        until the next mouseMoveEvent. tile is typically ViewerWindow's own
        _hover_tile, last reported by on_hover(). A None tile, or a tool
        with no highlight of this kind, just clears it -- same as if the
        mouse had left the map."""
        if tile is not None and self._tool in EDIT_TOOLS:
            self._update_highlight(*tile)
        else:
            self._clear_highlight()

    def set_mode(self, mode: str) -> None:
        """Mirrors ViewerWindow's own self.mode -- phase 3's P3-d.

        Modes and tools are orthogonal in this class (Pan and every edit tool
        exist in every mode; only reachability differs), and Units mode is
        deliberately a MODE rather than a fifth tool: mousePressEvent branches
        on self._tool through a CLICK_TOOLS/EDIT_TOOLS/fall-through-to-Pan
        split, and threading a "select" tool into that split would fight its
        structure for no benefit. A top-level mode branch is the clean cut.
        """
        if mode == self._mode:
            return
        self._mode = mode
        # Same reasoning as the tool switch: a measurement made in one mode
        # has no owner in the next.
        self._clear_ruler()
        if mode != "units":
            self._clear_unit_hover()
            self._clear_unit_selection()
            # A pending drag has no owner in the next mode either -- defensive
            # only, since a mode switch mid-press is not reachable through the
            # toolbar/combo UI, but leaving a stale key would misroute the
            # eventual release.
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            self._marquee_start_pos = None
            self._clear_marquee()
        # Units mode forces NoDrag regardless of the active tool (Pan is the
        # only tool reachable there) -- without this a left click would fall
        # through to ScrollHandDrag and pan instead of selecting.
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _apply_drag_mode(self) -> None:
        # The Ruler needs NoDrag for the same reason Units mode does, and is
        # named explicitly because it is NOT in EDIT_TOOLS: left-dragging is
        # how a measurement is made, so leaving ScrollHandDrag on would pan
        # the view instead and the tool would never work at all.
        if self._mode == "units" or self._tool in EDIT_TOOLS or self._tool == TOOL_RULER:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_unit_index(self, index) -> None:
        """Swaps the pick index (a filter change rebuilds it) and drops any
        hover/selection keys that the new index can no longer resolve."""
        self._unit_index = index
        self._clear_unit_hover()
        self.refresh_unit_highlight()

    def _unit_path(self, entry) -> QPainterPath | None:
        cache = self._sloped_cache()
        polygons = unit_pick.unit_polygons(
            entry,
            self._terrain_style,
            self._tile_pixels or 1,
            self._map_width or 0,
            self._map_height or 0,
            self._iso_elevations,
            self._iso_proj,
            corner_rise=None if cache is None else cache.corner_rise,
        )
        if not polygons:
            return None
        path = QPainterPath()
        for points in polygons:
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
            # addPolygon() leaves the subpath OPEN, so stroking it draws only
            # 3 of a diamond's 4 edges. Invisible wherever a fill dominates,
            # but the hover cue is outline-only -- confirmed by rendering it
            # offscreen and looking at the image, not by reading the docs.
            path.closeSubpath()
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
            self._unit_hover_item = self.scene().addPath(path, self.UNIT_HOVER_PEN)
            self._unit_hover_item.setZValue(self.UNIT_HOVER_Z)
        else:
            self._unit_hover_item.setPath(path)

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

    def _draw_unit_selection(self, entries) -> None:
        """One QPainterPath unioning every selected entry's own highlight,
        rather than one scene-item pair per unit -- cheap regardless of
        selection size, and the existing add/update-in-place shape below
        needs no change to become N-way."""
        path = QPainterPath()
        any_path = False
        for entry in entries:
            entry_path = self._unit_path(entry)
            if entry_path is not None:
                path.addPath(entry_path)
                any_path = True
        if not any_path:
            self._clear_unit_selection()
            return
        if self._unit_select_item is None:
            self._unit_select_fill_item = self.scene().addPath(
                path, QPen(Qt.NoPen), QBrush(self.UNIT_SELECT_FILL_COLOR)
            )
            self._unit_select_item = self.scene().addPath(path, self.UNIT_SELECT_PEN)
            self._unit_select_fill_item.setZValue(self.UNIT_SELECT_Z)
            self._unit_select_item.setZValue(self.UNIT_SELECT_Z)
        else:
            self._unit_select_item.setPath(path)
            self._unit_select_fill_item.setPath(path)

    def _clear_unit_selection(self) -> None:
        self._unit_select_keys = []
        if self._unit_select_item is not None:
            self.scene().removeItem(self._unit_select_item)
            self.scene().removeItem(self._unit_select_fill_item)
            self._unit_select_item = None
            self._unit_select_fill_item = None

    def refresh_unit_highlight(self) -> None:
        """Re-entry point for state changing WITHOUT the mouse moving --
        mirroring refresh_highlight(tile). Two live cases: a filter toggle
        rebuilds the index (some, none or all of the selected units may
        survive it), and set_unit_index() itself calls this after every
        index swap.

        Deliberately NOT the elevation-edit case, though the mirrored
        refresh_highlight() exists partly for that: terrain edits are only
        reachable in Terrain mode, and leaving Units mode drops the
        selection outright (see ViewerWindow.on_mode_changed), so a
        selection can never be live across an elevation edit. Phase 3.5
        gives this a second caller -- a unit mutation."""
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
                rect, self.UNIT_SELECT_PEN, QBrush(self.UNIT_SELECT_FILL_COLOR)
            )
            self._marquee_item.setZValue(self.UNIT_SELECT_Z)
        else:
            self._marquee_item.setRect(rect)

    def _clear_marquee(self) -> None:
        if self._marquee_item is not None:
            self.scene().removeItem(self._marquee_item)
            self._marquee_item = None

    def _scene_rect_to_tile_rect(self, rect: QRectF) -> tuple[int, int, int, int] | None:
        """A tile-space bounding box covering `rect`, half-open
        ([tx0, tx1), [ty0, ty1)), clamped to the map -- the marquee's own
        screen-to-tile conversion, kept here rather than in unit_pick.py
        because it is real per-style geometry (see units_in_rect()'s own
        docstring on why that module stays style-agnostic).

        Flat's is exact -- its tile grid is an axis-aligned pixel division,
        same arithmetic _pick_tile() already uses. Stepped/Sloped have no
        closed-form rectangle inverse (the same reason _pick_tile's own
        docstring gives for why even a single POINT needs a numeric
        approach there), so this samples _pick_tile() on a grid across
        `rect` at tile-pixel spacing and takes the bounding box of whatever
        resolves. A marquee is a convenience gesture, not a precision hit-
        test, and grid density here matches the per-pixel cost
        mouseMoveEvent already pays continuously while hovering -- this
        just does it for one release event instead of every frame.
        """
        if self._map_width is None or self._map_height is None:
            return None
        if self._terrain_style == "flat":
            tp = self._tile_pixels or 1
            tx0 = max(0, int(rect.left() // tp))
            ty0 = max(0, int(rect.top() // tp))
            tx1 = min(self._map_width, int(rect.right() // tp) + 1)
            ty1 = min(self._map_height, int(rect.bottom() // tp) + 1)
            return (tx0, ty0, tx1, ty1) if tx0 < tx1 and ty0 < ty1 else None

        step = max(1, self._tile_pixels or 1)
        left, top = int(rect.left()), int(rect.top())
        right, bottom = int(rect.right()), int(rect.bottom())
        xs = list(range(left, right, step)) + [right]
        ys = list(range(top, bottom, step)) + [bottom]
        tiles = [
            tile
            for y in ys
            for x in xs
            if (tile := self._pick_tile(QPointF(x, y))) is not None
        ]
        if not tiles:
            return None
        txs = [t[0] for t in tiles]
        tys = [t[1] for t in tiles]
        return (min(txs), min(tys), max(txs) + 1, max(tys) + 1)

    def _units_in_scene_rect(self, rect: QRectF) -> list[tuple[int, int]]:
        """The (player_id, reference_id) keys of every unit the marquee
        `rect` (scene/screen pixel space) covers -- already filter-
        respecting, since unit_pick.build_index() never inserted a filtered
        unit into the index in the first place."""
        if self._unit_index is None:
            return []
        tile_rect = self._scene_rect_to_tile_rect(rect)
        if tile_rect is None:
            return []
        entries = unit_pick.units_in_rect(self._unit_index, *tile_rect)
        return [unit_pick.unit_key(e.player_id, e.unit) for e in entries]

    def pick_unit_at(self, pos: QPointF):
        """The unit under scene-space pos, or None -- in every style since
        Track C5's Step 3, so this mirrors _pick_tile again.

        Sloped's two extra arguments both come from the live
        SlopedChunkCache, for the same Risk #6 reason _pick_tile and
        _tile_polygon read it: that object composited the pixels on screen,
        so hit-testing and the outline see the SAME corner_rise those pixels
        were painted from. terrain_tile is resolved here rather than inside
        unit_pick because Sloped's terrain has no analytic inverse -- it is
        a pick-plane lookup, which is exactly the cache reference unit_pick
        stays free of."""
        if self._unit_index is None or self._tile_pixels is None:
            return None
        cache = self._sloped_cache()
        return unit_pick.pick_unit(
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
            terrain_tile=None if cache is None else self._pick_tile(pos),
        )

    def set_tool(self, tool: str) -> None:
        # Switching tools mid-drag (e.g. a keyboard shortcut fired while the
        # mouse button is still down) must not leave a stroke dangling --
        # close it out first, same as a normal release would.
        if self._stroke_active and tool != self._tool:
            self._end_stroke()
        self._tool = tool
        # Whichever highlight belonged to the tool being left is no longer
        # valid -- mouseMoveEvent's next move re-creates the right one for
        # the new tool, but a stale one from before the switch (e.g. Pan's
        # black outline still showing right after switching into an edit
        # tool) must not linger until then.
        self._clear_highlight()
        self._clear_pan_highlight()
        # A measurement belongs to the Ruler, so leaving the tool drops it
        # rather than stranding an inert line on the map that nothing on
        # screen explains.
        if tool != TOOL_RULER:
            self._clear_ruler()
        if tool in EDIT_TOOLS:
            self._pulse_timer.start(self.HIGHLIGHT_PULSE_TICK_MS)
        else:
            self._pulse_timer.stop()
        # Routed through _apply_drag_mode() rather than set here directly, so
        # Units mode's NoDrag can't be undone by a tool switch.
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _apply_tool_cursor(self) -> None:
        if self._middle_drag_active:
            return  # middle-drag's closed-hand cursor takes priority for now
        # Above the mode check, unlike every other tool: Units mode's
        # pointing hand otherwise wins and the Ruler loses its aiming cursor
        # in the one mode where precision matters most.
        if self._tool == TOOL_RULER:
            self.setCursor(Qt.CrossCursor)
        elif self._mode == "units":
            self.setCursor(Qt.PointingHandCursor)
        elif self._tool in EDIT_TOOLS:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

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
        if key in self._stroke_touched:
            return
        self._stroke_touched.add(key)
        # Right-click strokes lower elevation -- the opposite of left-click's
        # raise. Synthesized as the same ShiftModifier bit Shift+left-click
        # already used for "lower" (kept working, not replaced) rather than
        # widening on_stroke_tile's signature with a separate direction
        # argument the "draw"/"set_level" tools would just ignore.
        if self._stroke_button == Qt.RightButton:
            modifiers = modifiers | Qt.ShiftModifier
        self._on_stroke_tile(tile_x, tile_y, modifiers)
        perf_trace.step()

    def _end_stroke(self) -> None:
        self._stroke_active = False
        self._stroke_touched = set()
        self._stroke_button = None
        self._on_stroke_end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._middle_drag_active = True
            self._middle_drag_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        # The Ruler, deliberately ABOVE the Units branch below rather than
        # beside the tool checks under it: that branch is independent of the
        # active tool, so it would otherwise swallow every ruler click in
        # Units mode. Right button cancels, a gesture that needs no keyboard
        # focus; either button mutates nothing.
        if self._tool == TOOL_RULER:
            if event.button() == Qt.RightButton:
                self._clear_ruler()
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
        # Units mode's selection click, deliberately ABOVE the CLICK_TOOLS/
        # EDIT_TOOLS checks and independent of whatever tool is active (the
        # Ruler is handled above; Pan and Place Unit are the only other tools
        # reachable in Units mode -- Convert (b2.5) is excluded below so it
        # falls through to the generic stroke branch further down instead,
        # since a Convert drag is a mutation with its own undo record, not a
        # selection gesture). Left button only: selection has no meaningful
        # right-button inverse, and phase 3 mutates nothing for one to undo.
        if (
            self._mode == "units"
            and event.button() == Qt.LeftButton
            and self._map_rect is not None
            and self._tool != "convert"
        ):
            pos = self.mapToScene(event.pos())
            # Place Unit (b1.4) is click_only, so it's in CLICK_TOOLS -- and
            # unambiguously so here, since no OTHER click_only tool is ever
            # reachable in Units mode (Paint Can, the only other member, gates
            # on Terrain mode). A place click is a distinct action from
            # selection, not an additional effect of it.
            if self._tool in CLICK_TOOLS:
                if self._pos_on_map(pos):
                    self._on_unit_place(pos, event.modifiers())
                return
            # b1.5's move-by-drag: remember what (if anything) this press hit
            # and where, so mouseReleaseEvent can tell a plain click from a
            # drag. The selection click itself still fires unconditionally --
            # a press-then-tiny-jitter-then-release must still select.
            entry = self.pick_unit_at(pos)
            self._on_click_select(pos, event.modifiers())
            self._unit_drag_key = None if entry is None else unit_pick.unit_key(entry.player_id, entry.unit)
            self._unit_drag_press_pos = event.pos()
            # b2.3's marquee: only a candidate when the press hit nothing --
            # a hit is already b1.5's move-drag above, and the two must never
            # both be live (mouseReleaseEvent checks _unit_drag_key first).
            self._marquee_start_pos = event.pos() if entry is None else None
            return
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
        # which the "draw"/"set_level" tools simply don't look at. Click
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
            pos = self.mapToScene(event.pos())
            if self._pos_on_map(pos):
                self._stroke_active = True
                self._stroke_touched = set()
                self._stroke_button = event.button()
                self._on_stroke_start()
                self._touch_tile(*self._pick_tile(pos), event.modifiers())
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton and self._middle_drag_active:
            self._middle_drag_active = False
            self._middle_drag_last_pos = None
            self._apply_tool_cursor()
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
        # b1.5's move-by-drag: the matching half of the press-time bookkeeping
        # above. Cleared unconditionally either way -- a drag is decided once,
        # here, never left pending for a later event.
        if event.button() == Qt.LeftButton and self._unit_drag_key is not None:
            key = self._unit_drag_key
            press_pos = self._unit_drag_press_pos
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            moved = (
                press_pos is not None
                and (event.pos() - press_pos).manhattanLength() > self.UNIT_DRAG_THRESHOLD_PX
            )
            if moved and self._map_rect is not None:
                pos = self.mapToScene(event.pos())
                if self._pos_on_map(pos):
                    self._on_unit_move(key, pos, event.modifiers())
            return
        # b2.3's marquee: the matching half of the press-time bookkeeping in
        # mousePressEvent, same decide-once-here shape as b1.5's move-drag
        # above (and mutually exclusive with it -- see that branch's own
        # comment). A drag that never exceeded the threshold was already a
        # plain click, handled by mousePressEvent's unconditional
        # on_click_select() call; nothing further to do here for that case.
        if event.button() == Qt.LeftButton and self._marquee_start_pos is not None:
            start_pos = self._marquee_start_pos
            self._marquee_start_pos = None
            self._clear_marquee()
            moved = (event.pos() - start_pos).manhattanLength() > self.UNIT_DRAG_THRESHOLD_PX
            if moved and self._map_rect is not None:
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
        super().mouseReleaseEvent(event)

    # Arrow-key nudge deltas, in tiles -- Qt.Key -> (dx, dy). Not a
    # REBINDABLE_ACTIONS set (see this method's own docstring on Escape,
    # which the same reasoning applies to).
    _UNIT_NUDGE_KEYS = {
        Qt.Key_Left: (-1, 0),
        Qt.Key_Right: (1, 0),
        Qt.Key_Up: (0, -1),
        Qt.Key_Down: (0, 1),
    }

    def keyPressEvent(self, event) -> None:
        """Escape clears a live measurement; in Units mode, arrow keys nudge
        the selection (b1.5) and Delete removes it (b1.6).

        None of these are REBINDABLE_ACTIONS rows: Escape/Delete/arrows are
        platform conventions rather than commands, and every added row is one
        more hand-audited entry in a keybind namespace with no runtime
        collision detection.

        Accepted limitation: all of these only fire while the view holds
        keyboard focus, so after clicking away to the status log they do
        nothing. A selection click always grants that focus (QGraphicsView's
        default focus policy is StrongFocus), so the ordinary click-then-
        nudge-or-delete flow works; Right-click and starting the next ruler
        measurement both still clear it.
        """
        if event.key() == Qt.Key_Escape and self._ruler.state != ruler.STATE_IDLE:
            self._clear_ruler()
            return
        if self._mode == "units":
            if event.key() in self._UNIT_NUDGE_KEYS:
                dx, dy = self._UNIT_NUDGE_KEYS[event.key()]
                self._on_unit_nudge(dx, dy, event.modifiers())
                return
            if event.key() == Qt.Key_Delete:
                self._on_unit_delete(event.modifiers())
                return
        super().keyPressEvent(event)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._clear_highlight()
        self._clear_pan_highlight()
        # Hover only -- a selection deliberately survives the cursor leaving
        # the map, the same way it survives moving onto empty ground.
        self._clear_unit_hover()
        # Defensive: a drag that exits the widget bounds without a release
        # (Qt usually still routes the eventual release back here via its
        # implicit mouse grab during a press-drag, but this covers the case
        # where it doesn't -- e.g. focus lost mid-drag) must not leave a
        # half-open stroke with no matching commit.
        if self._stroke_active:
            self._end_stroke()
        # Defensive, same reasoning as the stroke close above: a marquee
        # drag that exits the widget without a release must not leave a
        # dangling rectangle on screen with no pending selection to resolve
        # it into. Cancelled outright rather than resolved here -- the
        # press-time on_click_select() call already set the "plain click"
        # selection state, and there is no natural rect to finalize against
        # a cursor that has left the view.
        if self._marquee_start_pos is not None:
            self._marquee_start_pos = None
            self._clear_marquee()

    def _tile_polygon(self, tile_x: int, tile_y: int) -> QPolygonF | None:
        """The on-screen footprint of tile (tile_x, tile_y) as a polygon --
        a plain axis-aligned square in Flat mode, the tile's real projected,
        elevation-displaced diamond in Stepped mode, its warped per-column
        silhouette in Sloped. Shared by the edit-mode pulsing highlight and
        Pan mode's static outline below, so the two can never disagree about
        where a tile actually is on screen. None in Stepped/Sloped mode
        before a projection snapshot (and, for Sloped, a cache) exists
        (shouldn't happen once an image is loaded -- defensive only)."""
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
            # iso_geometry.sloped_tile_outline). corner_rise lives on the
            # cache rather than on MapView because it is the same array the
            # cache composited these pixels from -- reading it from anywhere
            # else would risk outlining against different geometry than the
            # one on screen.
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
            points = iso_geometry.sloped_tile_outline(self._tile_pixels, d_nw, d_ne, d_sw, d_se)
            return QPolygonF([QPointF(ox + px, oy + py) for px, py in points])
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
        (tile_x, tile_y, size, shape) since this runs on every pixel of
        mouseMoveEvent -- see _highlight_key's own comment in __init__."""
        key = (tile_x, tile_y, self._brush_size, self._brush_shape)
        if key == self._highlight_key:
            return
        self._highlight_key = key
        if self._map_width is None or self._map_height is None:
            tiles = [(tile_x, tile_y)]
        else:
            tiles = brush.brush_tiles(
                tile_x, tile_y, self._brush_size, self._brush_shape, self._map_width, self._map_height
            )
        path = QPainterPath()
        for tx, ty in tiles:
            polygon = self._tile_polygon(tx, ty)
            if polygon is not None:
                path.addPolygon(polygon)
        if path.isEmpty():
            self._clear_highlight()
            return
        if self._highlight_outline_item is None:
            self._highlight_outline_item = self.scene().addPath(path, self.HIGHLIGHT_OUTLINE_PEN)
            self._highlight_fill_item = self.scene().addPath(
                path, QPen(Qt.NoPen), QBrush(self.HIGHLIGHT_FILL_COLOR)
            )
        else:
            self._highlight_outline_item.setPath(path)
            self._highlight_fill_item.setPath(path)

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
        """Pan mode's own hover cue -- a thin static black outline, no fill,
        no pulse (see PAN_HIGHLIGHT_PEN). Deliberately much quieter than
        _update_highlight()'s edit-mode gold glow: Pan can't mutate
        anything, so it shouldn't compete for attention the way a live
        "this is about to paint" cursor needs to."""
        polygon = self._tile_polygon(tile_x, tile_y)
        if polygon is None:
            return
        if self._pan_highlight_item is None:
            self._pan_highlight_item = self.scene().addPolygon(polygon, self.PAN_HIGHLIGHT_PEN)
        else:
            self._pan_highlight_item.setPolygon(polygon)

    def _clear_pan_highlight(self) -> None:
        if self._pan_highlight_item is not None:
            self.scene().removeItem(self._pan_highlight_item)
            self._pan_highlight_item = None

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
        self._ruler_line_item = scene.addLine(QLineF(), self.RULER_PEN)
        self._ruler_line_item.setZValue(self.RULER_Z)
        self._ruler_end_items = []
        for _ in range(2):
            item = scene.addPolygon(QPolygonF(), self.RULER_PEN)
            item.setZValue(self.RULER_Z)
            self._ruler_end_items.append(item)
        label = scene.addSimpleText("")
        font = label.font()
        font.setPixelSize(self.RULER_LABEL_FONT_PX)
        font.setBold(True)
        label.setFont(font)
        label.setBrush(QBrush(self.RULER_LABEL_COLOR))
        label.setPen(QPen(self.RULER_LABEL_OUTLINE, 0))
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
        self._ruler_line_item.setLine(QLineF(anchor_a, anchor_b))
        for i in range(2):
            polygon = self._tile_polygon(*ends[i])
            if polygon is not None:
                self._ruler_end_items[i].setPolygon(polygon)
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
            self._on_ruler_measured(self._ruler.measurement)

    def _clear_ruler(self) -> None:
        """Drops the measurement AND its items. Removes from the scene
        explicitly, unlike the clear_image/set_source paths where
        scene().clear() has already destroyed the C++ objects and only the
        Python-side references need forgetting."""
        self._ruler.clear()
        if self._ruler_line_item is not None:
            self.scene().removeItem(self._ruler_line_item)
            for item in self._ruler_end_items:
                self.scene().removeItem(item)
            self.scene().removeItem(self._ruler_label_item)
        self._forget_ruler_items()
        self._on_ruler_changed(None)

    def _forget_ruler_items(self) -> None:
        self._ruler_line_item = None
        self._ruler_end_items = []
        self._ruler_label_item = None

    def _on_pulse_tick(self) -> None:
        if self._highlight_fill_item is None:
            return
        self._pulse_phase_ms = (self._pulse_phase_ms + self.HIGHLIGHT_PULSE_TICK_MS) % (
            self.HIGHLIGHT_PULSE_PERIOD_MS
        )
        t = self._pulse_phase_ms / self.HIGHLIGHT_PULSE_PERIOD_MS
        mid = (self.HIGHLIGHT_PULSE_MIN_ALPHA + self.HIGHLIGHT_PULSE_MAX_ALPHA) / 2
        amplitude = (self.HIGHLIGHT_PULSE_MAX_ALPHA - self.HIGHLIGHT_PULSE_MIN_ALPHA) / 2
        alpha = mid + amplitude * math.sin(2 * math.pi * t)
        self._highlight_fill_item.setOpacity(alpha)

    def clear_image(self) -> None:
        """Undoes set_source() -- back to the pre-load empty state (used by
        File > Close). Mirrors set_source()'s own scene().clear() + highlight
        reset, since QGraphicsScene.clear() destroys the highlight items (and
        the canvas item) too; leaving the Python-side reference pointing at
        it would dangle."""
        self.scene().clear()
        self._canvas_item = None
        self._map_rect = None
        self._map_width = None
        self._map_height = None
        self._highlight_outline_item = None
        self._highlight_fill_item = None
        self._pan_highlight_item = None
        self._highlight_key = None
        # Mandatory, not tidiness: scene().clear() destroys the C++ object,
        # and resizeEvent still reaches _capture_zoom_baseline with no map
        # loaded. File > Close then a resize would call a method on a deleted
        # object and raise RuntimeError.
        self._edge_tick_item = None
        # Same hazard as the tick item above, and the same fix: forget the
        # destroyed items, and drop the measurement itself, since File > Close
        # leaves no map for it to refer to.
        self._ruler.clear()
        self._forget_ruler_items()
        self._on_ruler_changed(None)
        # _capture_zoom_baseline() is not called here (that would recompute
        # bounds against a since-cleared map) -- but the readout still has to
        # fall back to "--", so fire the notification directly.
        self._on_zoom_changed()
        # scene().clear() destroyed these too -- drop the Python-side
        # references and the memo keys, or the next hover matches a stale key
        # and skips rebuilding an item that no longer exists.
        self._unit_hover_item = None
        self._unit_select_item = None
        self._unit_select_fill_item = None
        self._unit_hover_key = None
        self._unit_select_keys = []
        self._marquee_item = None
        self._marquee_start_pos = None
        self._unit_index = None
        self._stroke_active = False
        self._stroke_touched = set()

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Phase B-C's replacement for the old update_region_bbox()/
        update_region() (which read the whole QPixmap out, QPainter-patched
        one rect into it, and wrote the whole thing back -- ~200ms for a
        single-tile edit on this project's biggest real map, dominated by
        that whole-pixmap round trip, not the actual patch). The chunk cache
        is assumed already patched by the caller (see
        ViewerWindow._apply_dirty: dirty_screen_bbox_iso()/per-tile rects ->
        cache.patch()/patch_rects() -> here) -- this method is purely
        "please repaint this rect," forwarded to MapCanvasItem.
        invalidate_region(), which schedules a Qt repaint; the next paint()
        call pulls the already-patched chunk pixels straight from the cache,
        no pixmap read/write at all. Originally Stepped-only (Phase B-C);
        Flat reaches this too as of Phase B-E, once it stopped holding a
        single pre-baked pixmap to patch in place."""
        if self._canvas_item is None:
            return
        self._canvas_item.invalidate_region(bbox)

    def refresh_canvas_dims(self) -> None:
        """Re-syncs the canvas item's boundingRect AND this view's own scene
        rect/overscroll margin to the cache's CURRENT canvas_dims() -- Track
        P3-g6. Needed only when a cache's canvas_dims() answer can change
        after set_source() already ran; today that is SlopedChunkCache
        toggling sprites at a low elev_step_pct stop (see its canvas_dims()
        docstring). Every other caller's canvas_dims() never moves once
        set_source() has run, so nothing before this needed a resync path --
        invalidate_region() alone was always enough. Skipping this after
        such a toggle leaves the newly-widened strip composited in the
        cache but never reaching the screen: MapCanvasItem.boundingRect()
        and this view's own setSceneRect() both stay at their set_source()-
        time size regardless of what the cache reports afterwards."""
        if self._canvas_item is None:
            return
        self._canvas_item.refresh_canvas_dims()
        w, h = self._canvas_item._cache.canvas_dims()
        self._map_rect = QRectF(0, 0, w, h)
        margin_x = w * self.OVERSCROLL_FRACTION
        margin_y = h * self.OVERSCROLL_FRACTION
        self.setSceneRect(self._map_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))

    def set_source(
        self,
        tile_pixels: int,
        terrain_style: str = "flat",
        cache: "IsoChunkCache | FlatChunkCache | SlopedChunkCache | None" = None,
        elevations: np.ndarray | None = None,
        proj: "iso_geometry.IsoProjection | None" = None,
        unit_index=None,
        reset_view: bool = True,
    ) -> None:
        """Phase B-C rename/generalization of the old set_image(); Phase B-E
        drops that method's img parameter entirely -- both styles now paint
        lazily from a chunk cache via MapCanvasItem (see
        ViewerWindow._render_current), never a single pre-baked array/pixmap.
        elevations/proj (Stepped only) are set together with cache so
        _pick_tile's hit-testing can never point at a stale elevation
        snapshot (Risk #6 in the parent plan); both stay None for Flat.
        unit_index (phase 3) joins that same discipline for exactly the same
        reason: a pick index paired with the wrong elevations resolves clicks
        against a map that isn't on screen any more.

        reset_view (default True, matching every caller that means "a
        different document is now on screen" -- a fresh load_scenario()):
        whether the fit-to-view/pan reset at the end of this method is
        appropriate. False for "the same document, just redrawn or
        re-viewed" -- the full-redraw fallback ViewerWindow._apply_dirty()
        takes when an edit's incremental patch path declines, a settings-
        triggered refresh_map(), or an Elevation View (terrain style)
        switch -- where the caller wants set_source()'s side effects
        without losing the zoom/pan the user was already looking at. A
        terrain style switch changes the projection's shape (Flat's plain
        grid vs Stepped/Sloped's isometric one), so what's restored there is
        an approximation of the same map area (same zoom-relative-to-fit,
        same fractional position in the new map_rect), not pixel-exact.

        Kept as one method (dispatching on terrain_style), not two, matching
        this class's existing internal-dispatch convention (_pick_tile,
        _pos_on_map, set_isometric already branch on self._terrain_style
        rather than being split into per-style methods) -- callers still
        have exactly one entry point to call regardless of style."""
        view_restore = None if reset_view else self._capture_view_state()
        assert cache is not None, f"set_source(terrain_style={terrain_style!r}) requires cache"
        assert cache.style == terrain_style, (
            f"set_source(terrain_style={terrain_style!r}) got a cache built for "
            f"style={cache.style!r} -- Flat/Stepped must never cross-wire"
        )
        self._tile_pixels = tile_pixels
        self._terrain_style = terrain_style
        self._iso_elevations = elevations
        self._iso_proj = proj
        # Tile-count dims (not cache.canvas_dims()'s pixel dims) for clipping
        # the hover-preview brush footprint the same way
        # ViewerWindow.on_edit_stroke_tile clips a stroke's own footprint.
        mm = cache.scenario.map_manager
        self._map_width, self._map_height = mm.map_width, mm.map_height
        self.scene().clear()
        self._highlight_outline_item = None
        self._highlight_fill_item = None
        self._pan_highlight_item = None
        self._highlight_key = None
        # The Ruler's items went with scene().clear(). The measurement is
        # dropped rather than re-anchored for the same reason the selection
        # below is: set_source() means a new scenario or a style switch, and
        # neither guarantees the old tiles still mean anything.
        self._ruler.clear()
        self._forget_ruler_items()
        self._on_ruler_changed(None)
        # scene().clear() destroyed the unit items too. Selection is dropped
        # rather than re-resolved: set_source() means a new scenario or a
        # style switch, and neither guarantees the old key still addresses
        # anything.
        self._unit_hover_item = None
        self._unit_select_item = None
        self._unit_select_fill_item = None
        self._unit_hover_key = None
        self._unit_select_keys = []
        self._marquee_item = None
        self._marquee_start_pos = None
        self._unit_index = unit_index

        self._canvas_item = MapCanvasItem(cache)
        self.scene().addItem(self._canvas_item)
        w, h = cache.canvas_dims()

        self._map_rect = QRectF(0, 0, w, h)
        margin_x = w * self.OVERSCROLL_FRACTION
        margin_y = h * self.OVERSCROLL_FRACTION
        self.setSceneRect(self._map_rect.adjusted(-margin_x, -margin_y, margin_x, margin_y))

        # Outline the actual map extent so it stays visually distinct from
        # the overscroll margin once you've panned past an edge. In Stepped
        # and Sloped modes the extent is a diamond (the grid's own
        # silhouette, ignoring elevation -- see ground_outline_corners'
        # docstring), not the full pixmap rect, which is mostly background
        # there.
        if terrain_style in _ELEVATED_STYLES and proj is not None and elevations is not None:
            map_h, map_w = elevations.shape
            corners = iso_geometry.ground_outline_corners(map_w, map_h, proj)
            self.scene().addPolygon(
                QPolygonF([QPointF(x, y) for x, y in corners]), QPen(QColor(200, 200, 200), 0)
            )
        else:
            self.scene().addRect(self._map_rect, QPen(QColor(200, 200, 200), 0))

        # Always constructed, then hidden when off, so paint order is settled
        # here rather than by whenever the user first reaches the menu. Added
        # after the canvas and the extent outline, so it stacks above both by
        # scene insertion order.
        iso_proj = proj if terrain_style in _ELEVATED_STYLES else None
        self._edge_tick_item = EdgeTickItem(
            self._map_width,
            self._map_height,
            self._edge_tick_interval,
            self._map_rect,
            proj=iso_proj,
            tile_px=tile_pixels,
        )
        self._edge_tick_item.setVisible(self._edge_ticks_enabled)
        self.scene().addItem(self._edge_tick_item)

        # Ends with set_isometric(), which funnels into
        # _capture_zoom_baseline() and hands the item its first pad.
        # view_restore rides along via _pending_view_restore rather than a
        # set_isometric() parameter, since set_isometric() is also a public
        # entry point (the View > Isometric View checkbox) that must keep
        # its own no-argument fit-to-view behavior.
        self._pending_view_restore = view_restore
        self.set_isometric(self._isometric)

    def _capture_view_state(self) -> tuple[float, QPointF] | None:
        """Snapshot of the current view, as a zoom multiple of the CURRENT
        fit-to-view baseline plus the viewport's centered scene point
        expressed as a FRACTION of the current _map_rect, for
        set_isometric() to restore instead of resetting to fit -- see
        set_source()'s reset_view parameter. Both readings are relative
        (fit multiple, map-rect fraction), not absolute (scale, scene
        point): reset_view=False callers include a graphics-quality change,
        which rescales tile_pixels and therefore the WHOLE scene coordinate
        system (a smaller/larger map_rect in the same map, still 0..1
        across it) -- an absolute center point captured before that rescale
        would land somewhere else on the new, differently-scaled map.

        None when there's no established fit yet (nothing on screen to be
        relative to) -- reset_view=False callers only ever call this while a
        map is already showing, so this is a defensive fallback, not an
        expected path."""
        fit = self._fit_baseline_scale()
        if not fit or self._map_rect is None:
            return None
        if self._map_rect.width() <= 0 or self._map_rect.height() <= 0:
            return None
        current_scale = abs(self.transform().determinant()) ** 0.5
        center = self.mapToScene(self.viewport().rect().center())
        frac = QPointF(
            (center.x() - self._map_rect.left()) / self._map_rect.width(),
            (center.y() - self._map_rect.top()) / self._map_rect.height(),
        )
        return current_scale / fit, frac

    def _restore_view(self, restore: tuple[float, QPointF]) -> None:
        """Applies a captured _capture_view_state() reading on top of the
        transform set_isometric() has already established at the call site
        (squash+rotate for isometric Flat, untouched/identity otherwise),
        instead of that method's usual fit-to-view. Falls back to leaving
        the current transform alone if no fit baseline can be computed (e.g.
        a zero-size viewport) -- same "nothing sensible to do" case
        _capture_view_state() itself declines to capture."""
        relative_scale, frac = restore
        new_fit = self._fit_baseline_scale()
        if new_fit is None or self._map_rect is None:
            return
        target = relative_scale * new_fit
        if self._terrain_style in _ELEVATED_STYLES or not self._isometric:
            self.scale(target, target)
        else:
            k = target / math.sqrt(self.ISO_VERTICAL_SQUASH)
            self.scale(k, k)
        center = QPointF(
            self._map_rect.left() + frac.x() * self._map_rect.width(),
            self._map_rect.top() + frac.y() * self._map_rect.height(),
        )
        self.centerOn(center)

    def set_isometric(self, enabled: bool) -> None:
        self._isometric = enabled
        self.resetTransform()
        restore = self._pending_view_restore
        self._pending_view_restore = None
        # Stepped's (and Sloped's) projection is already baked into the
        # pixels -- this whole method (below this point) is Flat mode's own
        # QTransform trick (see the class comment above) and has nothing
        # left to do for a Stepped/Sloped image beyond the same plain
        # "whole canvas fits the viewport" fit Flat's own "unchecked" case
        # already uses. ViewerWindow greys out the View > Isometric View
        # checkbox whenever Terrain Style != Flat, so this method only ever
        # reaches the enabled=True rotate/squash path below while Flat is
        # active.
        if self._terrain_style in _ELEVATED_STYLES or not enabled:
            if self._map_rect is not None:
                if restore is not None:
                    self._restore_view(restore)
                else:
                    self.fitInView(self._map_rect, Qt.KeepAspectRatio)
                self._capture_zoom_baseline()
            return

        # Order matters: QTransform.scale() called before rotate() applies the
        # squash to already-rotated (diamond) coordinates, giving the correct
        # 2:1 bounding box; the other order gives a square bounding box instead
        # (verified directly against QTransform.mapRect()).
        self.scale(1.0, self.ISO_VERTICAL_SQUASH)
        self.rotate(self.ISO_ROTATION_DEGREES)

        if self._map_rect is None:
            return
        if restore is not None:
            self._restore_view(restore)
            self._capture_zoom_baseline()
            return
        # fitInView() is intentionally not used here: its internal normalization
        # step (dividing out the transformed unit square's own bounding box)
        # interacts badly with an anisotropic transform and silently squashes
        # the 2:1 diamond back down to ~1:1 (verified empirically). Computing
        # the fit scale by hand and applying it as one final *uniform* scale
        # sidesteps that -- a uniform scale can't introduce shear, so it can't
        # distort the ratio no matter how Qt composes it with what came before.
        mapped_rect = self.transform().mapRect(self._map_rect)
        viewport_rect = self.viewport().rect()
        if mapped_rect.width() <= 0 or mapped_rect.height() <= 0:
            return
        fit_scale = self.ISO_FIT_MARGIN * min(
            viewport_rect.width() / mapped_rect.width(),
            viewport_rect.height() / mapped_rect.height(),
        )
        self.scale(fit_scale, fit_scale)
        self.centerOn(self._map_rect.center())
        self._capture_zoom_baseline()

    def _coarsest_mip_scale(self) -> float:
        """How many scene units one pixel of the COARSEST enumerated mip
        level covers -- i.e. how much coarser than the reference the level
        set actually goes. Always >= 1.0, and exactly 1.0 both when no
        canvas item exists yet and when the level set enumerates nothing
        below the reference (a normal case, e.g. elev_step_pct=10, where it
        makes _capture_zoom_baseline's floor identical to the pre-mip
        one)."""
        if self._canvas_item is None:
            return 1.0
        cache = self._canvas_item._cache
        return cache.mip_scale(cache.mip_levels()[0])

    def _finest_mip_scale(self) -> float:
        """Mirrors _coarsest_mip_scale() at the other end of the ladder: how
        many scene units one pixel of the FINEST enumerated mip level covers
        -- i.e. how much finer than the reference the level set actually
        goes. Always <= 1.0, and exactly 1.0 both when no canvas item exists
        yet and when the level set enumerates nothing above the reference."""
        if self._canvas_item is None:
            return 1.0
        cache = self._canvas_item._cache
        return cache.mip_scale(cache.mip_levels()[-1])

    def _fit_baseline_scale(self) -> float | None:
        """The sqrt-determinant scale this view WOULD have at fit-to-view,
        derived from geometry rather than from the current transform.

        Returns None when no source is set: with no map there is no fit, so
        the quantity this computes does not exist to be computed.

        The transform reading this replaces was only ever a fit baseline
        because both of _capture_zoom_baseline's callers ran immediately
        after a fit. Recomputing on resize needs a form that doesn't depend
        on that -- otherwise a resize taken while zoomed to 3x fit re-anchors
        the bounds to that zoom and strands the user above fit, unable to
        zoom back out to the whole map.
        """
        if self._map_rect is None:
            return None
        viewport_rect = self.viewport().rect()
        if viewport_rect.width() <= 0 or viewport_rect.height() <= 0:
            return None
        if self._map_rect.width() <= 0 or self._map_rect.height() <= 0:
            return None
        if self._terrain_style in _ELEVATED_STYLES or not self._isometric:
            # fitInView's own arithmetic, including its hardcoded 2 px per side
            # (measured against the real transform, three aspect ratios).
            return min(
                (viewport_rect.width() - 4) / self._map_rect.width(),
                (viewport_rect.height() - 4) / self._map_rect.height(),
            )
        # Flat + isometric: squash, rotate, then one uniform fit scale -- see
        # set_isometric(). Rotation's determinant is 1, so the baseline is that
        # uniform scale times sqrt(ISO_VERTICAL_SQUASH).
        iso = QTransform()
        iso.scale(1.0, self.ISO_VERTICAL_SQUASH)
        iso.rotate(self.ISO_ROTATION_DEGREES)
        mapped_rect = iso.mapRect(self._map_rect)
        if mapped_rect.width() <= 0 or mapped_rect.height() <= 0:
            return None
        fit_scale = self.ISO_FIT_MARGIN * min(
            viewport_rect.width() / mapped_rect.width(),
            viewport_rect.height() / mapped_rect.height(),
        )
        return fit_scale * math.sqrt(self.ISO_VERTICAL_SQUASH)

    def zoom_percent_of_fit(self) -> float | None:
        """Current zoom as a percentage of fit-to-view -- the same quantity
        wheelEvent's ceiling/floor checks compare against, times 100. Backs
        the status-bar readout; reuses both expressions rather than
        re-deriving either, since the readout is worthless for calibrating
        MAX_ZOOM_MULTIPLE_OF_FIT if its arithmetic can drift from the clamp's.

        None means "no map loaded" (_fit_baseline_scale's own None case),
        not the identity-transform fallback _capture_zoom_baseline uses --
        reporting a fabricated 100% for an empty view would be wrong.
        """
        fit = self._fit_baseline_scale()
        if fit is None or fit <= 0:
            return None
        current = abs(self.transform().determinant()) ** 0.5
        return current / fit * 100.0

    def _capture_zoom_baseline(self) -> None:
        # Real per-tile texture detail (see render.py) is high-frequency
        # enough that zooming out past a few times the "whole map fits in
        # view" scale minifies it beyond what QPainter.SmoothPixmapTransform's
        # plain bilinear filtering can represent without mipmapping -- the
        # result is visible moire/banding artifacts, not a rendering bug in
        # the usual sense. Zooming in past a similar multiple just upscales
        # that same finite per-tile texture resolution further, revealing no
        # more real detail, only blur/blockiness. Rather than fix either with
        # real mipmapping (a much bigger rendering change), wheelEvent() just
        # refuses to zoom past floor/ceiling values relative to this
        # fit-to-view baseline -- which is also just more usable regardless
        # (nothing useful past "whole map visible" in one direction or "one
        # tile fills the view" in the other).
        # determinant() gives the transform's area scale factor independent
        # of rotation, so this stays valid even under the isometric
        # transform's rotation + anisotropic squash.
        #
        # Phase B-D-d: mipmapping now exists below the reference level, so
        # the zoom-out floor gets divided by exactly how much coarser the
        # ladder goes. The property that buys, stated so it can't drift into
        # an overclaim: at the floor the residual minification is UNCHANGED
        # from the pre-mip one (the floor scales by 1/S while the level
        # painted there is S times coarser), and the reachable zoom-out range
        # grows by exactly the mip depth. Both halves are one line of algebra
        # and neither depends on window size. At a fixed scale the coarser
        # level does alias S times less, but that is not what this floor is
        # claiming -- see PLAN_MIPS.md's correction entry for the version of
        # this comparison that got the direction backwards.
        #
        # 2026-08-27: the zoom-in ceiling gets the same treatment, divided by
        # _finest_mip_scale() (<= 1.0, so dividing raises the ceiling). B-D-d
        # left this half alone on the theory that mip levels finer than the
        # reference were a pure detail win with no reachability question --
        # wrong in practice: MAX_ZOOM_MULTIPLE_OF_FIT was calibrated against
        # the pre-mip reference tile size, and on real maps that ceiling sat
        # BELOW the scale needed to ever select the finest enumerated level,
        # so the sharper level the mip track added was dead code behind the
        # wheel-zoom limit (measured: reachable at fit-multiple ~48, capped
        # at 40). Same algebra as the floor, mirrored: at the new ceiling the
        # residual magnification is unchanged from the pre-mip one, and the
        # reachable zoom-in range grows by exactly the mip depth.
        #
        # 2026-08-28: mip_for_scale's rule flipped from floor(log2(scale)) to
        # ceil, which does not change either formula below -- both derivations
        # above only used mip_scale()/_coarsest_mip_scale()/_finest_mip_scale(),
        # never the selection rule's rounding direction. What it does change:
        # the floor's clamp-binding and residual-unchanged property (the
        # paragraph above this one) survives verbatim; the ceiling's
        # reachability property survives too and gets slightly MORE robust
        # (the finest level now engages at half the scale it used to, so
        # "reachable at fit-multiple ~48" above is now reachable at ~24). Net
        # effect on the two clamped-beyond-the-ladder bands: the over-
        # minification band below the floor shrinks by one octave (less blur
        # at heavy zoom-out) and the over-magnification band above the
        # ceiling grows by one octave (one more octave of zoom-in past native
        # density before the ceiling has to intervene). Deliberately not
        # compensated further -- same reasoning as the note above about why
        # this isn't an absolute bound.
        #
        # Fit-RELATIVE rather than an absolute scale bound, which is what the
        # plan originally specified: an absolute floor has no notion of how
        # big the canvas is in the window, so it lands above fit-to-view on a
        # big map (measured, 480x480 at a 480px window: fit is scale 0.0413,
        # an absolute floor of 1/(4 * 4) = 0.0625 would refuse zoom-out
        # everywhere and strand the user above fit). mip_scale >= 1 always,
        # so dividing by it is never tighter than the pre-mip floor, and a
        # level set with nothing below the reference (elev_step_pct=10
        # enumerates {0, 1}) reduces it to exactly the pre-mip expression
        # rather than needing a special case.
        #
        # 2026-08-30: the widened value is now capped back to the pre-mip
        # MIN_ZOOM_FRACTION_OF_FIT/MAX_ZOOM_MULTIPLE_OF_FIT bound with
        # max()/min() below, reversing the "never tighter than the pre-mip
        # bound" property above on user request -- a fixed 50%-6400%-of-fit
        # range was wanted more than always being able to wheel-zoom to the
        # sharpest/coarsest enumerated mip level. Concretely, on Flat/Stepped's
        # typical 4-level ladder this pulled the reachable range in from
        # 12.5%-12800% (measured on the 120x120 fixture) back to 50%-6400%.
        # The mip-scale division above still runs and stays correct on its
        # own terms -- it is what this clamp is capping -- so the two
        # invariants coexist rather than one being deleted in favour of the
        # other; only the enumerated-level-reachability guarantee is gone.
        #
        # resizeEvent below re-runs this on every viewport size change, so
        # neither bound goes stale after a resize any more. That is only safe
        # because the baseline comes from _fit_baseline_scale() -- geometry --
        # rather than from the current transform: reading the transform while
        # the user is zoomed in would re-anchor both bounds to that zoom.
        fit = self._fit_baseline_scale()
        # No map set means no fit scale exists to derive, and the current
        # transform is the only reading available. Not a compatibility shim:
        # it is the defined behaviour for a view with no source. resizeEvent
        # reaches this before any scenario is loaded, where it sets the bounds
        # from the identity transform (0.5 and 64) rather than leaving them
        # None -- a no-op either way, since there is nothing on screen to zoom,
        # and set_isometric() recomputes both the moment a source arrives.
        baseline = fit if fit is not None else abs(self.transform().determinant()) ** 0.5
        raw_min = self.MIN_ZOOM_FRACTION_OF_FIT * baseline
        raw_max = self.MAX_ZOOM_MULTIPLE_OF_FIT * baseline
        self._min_linear_scale = max(raw_min / self._coarsest_mip_scale(), raw_min)
        self._max_linear_scale = min(raw_max / self._finest_mip_scale(), raw_max)
        self._repad_edge_ticks()
        # Covers every path that reaches here: set_isometric()/set_source()
        # resetting the transform (readout returns to ~100%), and resizeEvent
        # -- which changes the percentage with no zoom action at all, since
        # _fit_baseline_scale() reads the viewport rect.
        self._on_zoom_changed()

    def _repad_edge_ticks(self) -> None:
        """Hands the tick overlay the smallest scale the view can currently
        reach, which is what its bounding-rect pad has to cover.

        min() of the floor and the CURRENT scale, and neither term subsumes
        the other: resizeEvent re-runs _capture_zoom_baseline, and a larger
        viewport raises _min_linear_scale without rescaling the transform, so
        maximising the window while zoomed to the old floor leaves the
        current scale below the new minimum. wheelEvent needs no hook of its
        own: zooming in only raises the scale, and zooming out is refused
        below _min_linear_scale, which this already covers."""
        if self._edge_tick_item is None:
            return
        current = abs(self.transform().determinant()) ** 0.5
        floor = self._min_linear_scale
        self._edge_tick_item.set_min_view_scale(min(floor, current) if floor is not None else current)

    def set_edge_ticks(self, enabled: bool) -> None:
        """Shows or hides the map-edge distance ruler. Never rebuilds a
        chunk cache: unlike Show sprites, these marks are not baked into
        canvas pixels, so there is nothing to evict."""
        self._edge_ticks_enabled = enabled
        if self._edge_tick_item is not None:
            self._edge_tick_item.setVisible(enabled)

    def set_edge_tick_interval(self, interval: int) -> None:
        self._edge_tick_interval = interval
        if self._edge_tick_item is not None:
            self._edge_tick_item.set_interval(interval)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Recompute the bounds, but deliberately no re-fit: that would fight
        # the zoom and pan the user chose. The new log-splitter handle makes a
        # viewport resize a one-drag action rather than a window-resize-only
        # one, which is what made the stale bounds worth fixing.
        self._capture_zoom_baseline()

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current_scale = abs(self.transform().determinant()) ** 0.5
        if factor < 1.0 and self._min_linear_scale is not None and current_scale * factor < self._min_linear_scale:
            return
        if factor > 1.0 and self._max_linear_scale is not None and current_scale * factor > self._max_linear_scale:
            return
        self.scale(factor, factor)
        self._on_zoom_changed()

    def mouseMoveEvent(self, event):
        if self._middle_drag_active:
            delta = event.pos() - self._middle_drag_last_pos
            self._middle_drag_last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
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
        # Gated on MODE, not tool, so a unit pick costs nothing per pixel in
        # View/Terrain/Triggers. Both TILE highlights are cleared here: Units
        # mode addresses units, and leaving a tile cue on screen would suggest
        # the click does something tile-shaped.
        if self._mode == "units":
            self._clear_highlight()
            self._clear_pan_highlight()
            # Convert (b2.5) is a mutating brush like Draw/Elevate, not a
            # selection gesture -- it gets their gold pulsing "about to
            # paint" cue via the same _update_highlight()/_touch_tile() pair
            # the generic EDIT_TOOLS branch below uses, just reached from
            # here since Convert lives in Units mode (mousePressEvent's own
            # units-branch exclusion is the other half of this routing).
            if self._tool == "convert":
                self._clear_unit_hover()
                if on_map:
                    with perf_trace.phase("highlight"):
                        self._update_highlight(*tile)
                    if self._stroke_active:
                        self._touch_tile(*tile, event.modifiers())
                else:
                    self._clear_highlight()
                return
            if self._unit_drag_key is not None:
                # A unit drag is in progress (b1.5): the hover cue would
                # misleadingly highlight whatever the cursor happens to be
                # crossing, and there is no live drag preview to update here
                # either -- the move is computed once, at release.
                return
            if self._marquee_start_pos is not None:
                # b2.3: a marquee drag is in progress -- update the rubber
                # band instead of the hover cue, for the same "no live
                # preview of the eventual result" reason as the unit-drag
                # branch above (the candidate set is resolved once, at
                # release, not recomputed every frame).
                self._clear_unit_hover()
                self._update_marquee(self._marquee_start_pos, event.pos())
                return
            # Canvas containment, NOT `on_map`: in Stepped, on_map is
            # "screen_to_tile resolved a tile", which is False on a skirt
            # face -- and a unit standing at the top of a skirt is visible
            # exactly there. Gating on on_map would make those pixels
            # unhoverable. See pick_unit()'s documented skirt residual.
            in_canvas = self._map_rect is not None and self._map_rect.contains(pos)
            entry = self.pick_unit_at(pos) if in_canvas else None
            if entry is None:
                self._clear_unit_hover()
            else:
                self._update_unit_hover(entry)
        elif self._tool in EDIT_TOOLS:
            self._clear_pan_highlight()
            if on_map:
                with perf_trace.phase("highlight"):
                    self._update_highlight(*tile)
                if self._stroke_active:
                    self._touch_tile(*tile, event.modifiers())
            else:
                self._clear_highlight()
        else:
            self._clear_highlight()
            if on_map:
                self._update_pan_highlight(*tile)
            else:
                self._clear_pan_highlight()
