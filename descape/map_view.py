"""MapView: the QGraphicsView the map is panned, zoomed and edited in.

Dumb the same way TriggerPanel is -- it reports what the user did
through the callbacks ViewerWindow constructs it with and never
reaches back up into the window."""

from __future__ import annotations


import math
from collections.abc import Callable

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
    region_clipboard,
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
from descape.viewer_canvas import EdgeTickItem, MapCanvasItem, _max_axis_scale, level_rect_for
from descape.viewer_common import CLICK_TOOLS, EDIT_TOOLS, TOOL_EYEDROPPER, TOOL_RULER, TOOL_SELECT

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

    # Repeating, self-stopping poll interval backing the viewport-changed hook
    # (maintainer plan 2026-09-07's A2.4) -- PROVISIONAL, to be tuned in the
    # in-app pass the same way level_warm.BUDGET_MS is. Bounds how often a
    # margin ring gets rebuilt/re-sorted during a drag (once per this many ms
    # of motion, not once per mouse-move event, which arrive every ~8ms) while
    # still giving the ring's leading-side ordering a real displacement to
    # diff against between fires.
    VIEWPORT_POLL_MS = 150

    # Hover-highlight convention for every brush-style tool (Terrain,
    # Elevate, Set Elevation): an outline plus a translucent fill (default
    # gold) that pulses steadily, so it reads as a live cursor following the
    # mouse rather than a static "you clicked here" marker. Follows on hover,
    # not just on click. Colors are user-settable (Settings > Appearance,
    # settings.OVERLAY_COLORS) and live in self._highlight_outline_pen/
    # self._highlight_fill_color, rebuilt by _rebuild_overlay_ink() -- these
    # are just the pulse behaviour, which isn't a color and stays fixed.
    HIGHLIGHT_PULSE_MIN_ALPHA = 0.25
    HIGHLIGHT_PULSE_MAX_ALPHA = 0.55
    HIGHLIGHT_PULSE_PERIOD_MS = 500
    HIGHLIGHT_PULSE_TICK_MS = 40

    # Units mode's two cues, phase 3's P3-d. Deliberately NOT the pulsing
    # highlight above reserves for "live and about to paint": phase 3
    # mutates nothing, so it must not claim that signal. Hover defaults thin
    # and quiet like Pan's; selection is solid plus a translucent fill, in a
    # color distinct from the edit highlight by default. Neither pulses.
    # Colors live in self._unit_hover_pen/self._unit_select_pen/
    # self._unit_select_fill_color; UNIT_SELECT_FILL_ALPHA is the one part of
    # the fill that ISN'T user-settable (see settings.OVERLAY_COLORS's own
    # comment on RGB-only scope), and UNIT_SELECT_PEN_WIDTH is this
    # codebase's only non-cosmetic pen width, deliberately not 0.
    UNIT_SELECT_FILL_ALPHA = 70
    UNIT_SELECT_PEN_WIDTH = 2

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

    # The Ruler's line, its two endpoint outlines and its label. A fixed
    # (user-settable) colour rather than a themed one: MapView paints its
    # scene background unconditionally to OUTSIDE_MAP_COLOR, so apply_theme
    # never reaches any of this. Colors live in self._ruler_pen/
    # self._ruler_label_color/self._ruler_label_outline;
    # RULER_LABEL_OUTLINE_ALPHA is the one part of the outline that isn't
    # user-settable, so the label keeps surviving both the near-black void
    # and bright terrain without a backing rect regardless of the chosen hue.
    RULER_LABEL_OUTLINE_ALPHA = 230
    RULER_LABEL_GAP_PX = 6.0
    # Above UNIT_SELECT_Z: a measurement is a deliberate act, and should
    # not be occluded by the hover cue it was drawn on top of.
    RULER_Z = 12.0
    RULER_LABEL_Z = 13.0

    # Phase 2.8's Select tool. Teal by default, distinct from the unit
    # marquee's default blue or the edit highlight's default gold -- this
    # needs to read unambiguously as a third, distinct thing: "this region is
    # selected", though a user override can collapse that distinction
    # deliberately. Marching ants (a dark solid pass plus a lighter dashed
    # pass, both cosmetic so the dashes stay the same device-pixel size at
    # every zoom) are what makes it read as a selection rather than a static
    # highlight -- see _advance_region_ants(). Colors live in
    # self._region_fill_color/self._region_outline_pen/self._region_ants_pen;
    # REGION_SELECT_FILL_ALPHA is the one part of the fill that isn't
    # user-settable, and REGION_SELECT_PEN_WIDTH/REGION_SELECT_ANT_DASH are
    # not colors at all, so both stay fixed literals reapplied by
    # _rebuild_overlay_ink() on every rebuild.
    REGION_SELECT_FILL_ALPHA = 50
    REGION_SELECT_PEN_WIDTH = 2
    REGION_SELECT_ANT_DASH = [4.0, 4.0]  # device pixels, since cosmetic
    REGION_SELECT_ANT_STEP_PX = 1.0
    REGION_SELECT_ANT_TICK_MS = 80
    # Above UNIT_SELECT_Z (a region can carry units, so its outline must read
    # on top of the units inside it), below RULER_Z (a measurement is a
    # deliberate act and should never be occluded).
    REGION_SELECT_Z = 11.5

    # Map mirroring's (Stage 1) live preview overlay in MirrorDialog: the
    # shaded source slice and, in Flat mode only, the mode's own symmetry
    # axis line(s). Above every other overlay here -- a deliberate,
    # momentary dialog-driven state that should read as on top of
    # everything while showing, the same reasoning RULER_Z sits above
    # UNIT_SELECT_Z for. Colors (default cyan) are user-settable like every
    # other overlay above (settings.OVERLAY_COLORS's "mirror_overlay" row)
    # and live in self._mirror_overlay_outline_pen/_fill_color/_axis_pen,
    # rebuilt by _rebuild_overlay_ink(); MIRROR_OVERLAY_FILL_ALPHA is the one
    # part of the fill that isn't user-settable, matching REGION_SELECT_FILL_ALPHA.
    MIRROR_OVERLAY_FILL_ALPHA = 60
    MIRROR_OVERLAY_Z = 14.0

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
        on_region_selected,
        on_ruler_measured,
        on_ruler_changed,
        on_zoom_changed,
        on_viewport_changed=lambda: None,
    ):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.scene().setBackgroundBrush(self.OUTSIDE_MAP_COLOR)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        # Populates every self._*_pen/self._*_color attribute _update_highlight
        # et al. read below -- must run before anything else in __init__ could
        # plausibly touch them. apply_overlay_colors() re-runs this later.
        self._rebuild_overlay_ink()
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
        # A2's viewport-changed hook (maintainer plan 2026-09-07) -- a
        # twelfth injected callable, defaulting to a no-op lambda (unlike
        # every callable above, which every real caller must supply): most
        # callers reassign this straight to a real target after construction
        # (ViewerWindow does, see the on_zoom_changed placeholder's own
        # comment for why a direct constructor argument doesn't work there
        # either), and every other MapView() call site in the test suite
        # that doesn't care about margin warming needs no changes at all.
        # Public (not `_on_viewport_changed`) to mirror how it's reassigned.
        self.on_viewport_changed = on_viewport_changed
        # Repeating, self-stopping (see _on_viewport_poll_tick) rather than a
        # restarted single-shot -- a restart-on-every-move debounce would
        # never fire during a continuous drag (mouse-moves arrive every
        # ~8ms, faster than any sane debounce interval), which is exactly
        # the starvation A2.4 exists to avoid. Started by
        # _note_viewport_changed(), never directly.
        self._viewport_poll_timer = QTimer(self)
        self._viewport_poll_timer.setInterval(self.VIEWPORT_POLL_MS)
        self._viewport_poll_timer.timeout.connect(self._on_viewport_poll_tick)
        # The target recorded at the last poll fire (or None before the
        # first one) -- compared against on every subsequent fire to decide
        # whether to notify on_viewport_changed() or stop the timer.
        self._last_viewport_target: tuple[int, int, int, int, int] | None = None
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
        # Phase 2.8's Select tool, the twelfth injected callable.
        # on_region_selected(region | None) fires at the two moments a
        # COMMITTED region changes from user input MapView itself sees: a
        # completed drag, and an Escape that clears an existing region with
        # no drag in progress. ViewerWindow's Select All/Deselect actions
        # change the same committed value from the other direction and reach
        # this same handler directly rather than through a callback -- see
        # set_region()'s own docstring for the split.
        self._on_region_selected = on_region_selected
        # The anchor tile from mousePressEvent, and the last tile a move
        # resolved to (frozen, like the Ruler's endpoint, if the drag runs
        # off-map) -- both None outside an in-progress drag. The COMMITTED
        # region lives in self._region below, mirroring the anchor/committed
        # split ruler.RulerSession keeps internally.
        self._select_anchor: tuple[int, int] | None = None
        self._select_current: tuple[int, int] | None = None
        # The committed region, half-open tile-space (tx0, ty0, tx1, ty1).
        # Set by set_region() -- MapView's own drag-release/Escape-clear
        # paths, and ViewerWindow's Select All/Deselect, both funnel through
        # it, so there is exactly one place that rebuilds the overlay.
        self._region: tuple[int, int, int, int] | None = None
        self._region_fill_item: QGraphicsPathItem | None = None
        self._region_outline_item: QGraphicsPathItem | None = None
        self._region_ants_item: QGraphicsPathItem | None = None
        self._region_ant_offset = 0.0
        self._region_ant_timer = QTimer(self)
        self._region_ant_timer.setInterval(self.REGION_SELECT_ANT_TICK_MS)
        self._region_ant_timer.timeout.connect(self._advance_region_ants)
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
        # Pan mode's own quieter highlight -- see self._pan_highlight_pen,
        # built by _rebuild_overlay_ink(). Always exactly one tile (Pan has
        # no brush), so this stays a plain QGraphicsPolygonItem.
        self._pan_highlight_item: QGraphicsPolygonItem | None = None
        # MirrorDialog's live preview overlay -- see show_mirror_overlay()/
        # clear_mirror_overlay(). Same QPainterPath-of-tile-polygons shape as
        # the edit highlight above, plus optional straight axis line items.
        self._mirror_overlay_outline_item: QGraphicsPathItem | None = None
        self._mirror_overlay_fill_item: QGraphicsPathItem | None = None
        self._mirror_axis_items: list[QGraphicsLineItem] = []
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
        if self._unit_select_item is not None:
            self._unit_select_item.setPen(self._unit_select_pen)
            self._unit_select_fill_item.setBrush(QBrush(self._unit_select_fill_color))
        if self._marquee_item is not None:
            self._marquee_item.setPen(self._unit_select_pen)
            self._marquee_item.setBrush(QBrush(self._unit_select_fill_color))
        if self._ruler_line_item is not None:
            self._ruler_line_item.setPen(self._ruler_pen)
            for item in self._ruler_end_items:
                item.setPen(self._ruler_pen)
            self._ruler_label_item.setBrush(QBrush(self._ruler_label_color))
            self._ruler_label_item.setPen(QPen(self._ruler_label_outline, 0))
        if self._region_fill_item is not None:
            self._region_fill_item.setBrush(QBrush(self._region_fill_color))
            self._region_outline_item.setPen(self._region_outline_pen)
            # Carries the ants' current phase forward -- a rebuilt pen would
            # otherwise reset dashOffset to 0 and visibly restart the crawl.
            ants_pen = QPen(self._region_ants_pen)
            ants_pen.setDashOffset(self._region_ants_item.pen().dashOffset())
            self._region_ants_item.setPen(ants_pen)
        if self._mirror_overlay_outline_item is not None:
            self._mirror_overlay_outline_item.setPen(self._mirror_overlay_outline_pen)
            self._mirror_overlay_fill_item.setBrush(QBrush(self._mirror_overlay_fill_color))
            for item in self._mirror_axis_items:
                item.setPen(self._mirror_axis_pen)
        # The edit highlight has no in-place update path worth writing (it
        # rebuilds on every hover move anyway) -- clearing it here just forces
        # that rebuild to happen with the new ink instead of leaving a stale
        # color showing until the mouse next moves.
        self._clear_highlight()

    def apply_ruler_label_font(self) -> None:
        """Settings > Appearance's re-entry point for the ruler label
        font-size spinbox -- apply_overlay_colors()'s sibling, same
        deleted-item guard (the Settings dialog is reachable with no map
        open). No-op with no live label: _create_ruler_items() already reads
        settings.get_ruler_label_font_px() directly, so the next measurement
        picks up the new size with no extra plumbing needed."""
        if self.scene() is None or self._ruler_label_item is None:
            return
        font = self._ruler_label_item.font()
        font.setPixelSize(settings.get_ruler_label_font_px())
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
        if (
            self._mode == "units"
            or self._tool in EDIT_TOOLS
            or self._tool == TOOL_RULER
            or self._tool == TOOL_EYEDROPPER
        ):
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
            farms_draped=False if cache is None else (cache.with_units and cache.sprites_enabled),
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
            self._unit_hover_item = self.scene().addPath(path, self._unit_hover_pen)
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
                path, QPen(Qt.NoPen), QBrush(self._unit_select_fill_color)
            )
            self._unit_select_item = self.scene().addPath(path, self._unit_select_pen)
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
                rect, self._unit_select_pen, QBrush(self._unit_select_fill_color)
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
            farms_draped=False if cache is None else (cache.with_units and cache.sprites_enabled),
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
        # An in-progress select drag belongs to the Select tool the same way
        # a measurement belongs to the Ruler -- but the COMMITTED region does
        # not: it must survive a tool switch (Copy/Paste act on it
        # regardless of which tool is active), so only the anchor is
        # cancelled here, not self._region.
        if tool != TOOL_SELECT and self._select_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._update_region_overlay()
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
        if self._tool in (TOOL_RULER, TOOL_EYEDROPPER, TOOL_SELECT):
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
        # The Select tool, is_edit_tool=False like the Ruler above and given
        # the same early, mode-independent priority (it only ever applies in
        # Terrain mode in practice, via ToolDef.modes, but nothing here
        # depends on that). Left button only: a rectangle drag has no
        # meaningful right-button inverse.
        if self._tool == TOOL_SELECT:
            if event.button() == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                if self._pos_on_map(pos):
                    tile = self._pick_tile(pos)
                    self._select_anchor = tile
                    self._select_current = tile
                    self._update_region_overlay()
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
        # The Select tool's matching release: commits self._select_anchor/
        # self._select_current (the latter frozen at its last on-map value if
        # the release itself is off-map, same "hold the last valid tile"
        # convention as the Ruler's endpoint) into a real region via
        # set_region(), then announces it -- the one path
        # ViewerWindow.on_region_selected() shares with Select All/Deselect.
        if self._tool == TOOL_SELECT:
            if event.button() == Qt.LeftButton and self._select_anchor is not None:
                anchor, current = self._select_anchor, self._select_current
                self._select_anchor = None
                self._select_current = None
                region = region_clipboard.normalize_region(
                    anchor, current, self._map_width or 0, self._map_height or 0
                )
                self.set_region(region)
                self._on_region_selected(region)
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
        # Select tool's Escape: cancel an in-progress drag first, only clear
        # a committed region when there is no drag -- same ordering rule
        # daubED's canvas_widget.py follows, so releasing Escape mid-drag
        # can never destroy a PREVIOUSLY committed region the drag hadn't
        # replaced yet.
        if event.key() == Qt.Key_Escape and self._select_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._update_region_overlay()
            return
        if event.key() == Qt.Key_Escape and self._region is not None:
            self.set_region(None)
            self._on_region_selected(None)
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
        # Defensive, same reasoning as the marquee above: cancelled outright,
        # not committed -- there is no natural release position to resolve
        # against. The committed region (if any) is untouched, exactly like
        # an Escape-cancel.
        if self._select_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._update_region_overlay()

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
        coarse() instead of the exact staircase. Every other caller
        (Pan's static outline, ruler, region select) leaves this False and
        must keep getting the pixel-exact outline."""
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
            outline_fn = (
                iso_geometry.sloped_tile_outline_coarse if coarse else iso_geometry.sloped_tile_outline
            )
            points = outline_fn(self._tile_pixels, d_nw, d_ne, d_sw, d_se)
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
        mouseMoveEvent -- see _highlight_key's own comment in __init__.

        Passes coarse=True to _tile_polygon (draw-perf plan item 3): in
        Sloped this brush footprint was measured at 5.6ms at brush size 9,
        almost entirely _tile_polygon's per-column exact staircase times up
        to 81 tiles. The approximate outline is fine here since this path
        only ever feeds display, never a pick -- unlike _update_pan_
        highlight below, which stays exact."""
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
            polygon = self._tile_polygon(tx, ty, coarse=True)
            if polygon is not None:
                path.addPolygon(polygon)
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

    def set_region(self, region: tuple[int, int, int, int] | None) -> None:
        """The one mutator for the committed region -- called from MapView's
        own drag-release/Escape-clear paths, and by ViewerWindow directly for
        Select All/Deselect, so there is exactly one place that rebuilds the
        overlay from a new value. Idempotent: setting the same region twice
        just rebuilds the same path again."""
        self._region = region
        self._update_region_overlay()

    def _region_boundary_tiles(self, tx0: int, ty0: int, tx1: int, ty1: int) -> list[tuple[int, int]]:
        """The perimeter ring of the half-open rectangle, one tile deep --
        NOT every tile in it. A Select region can span the whole map, unlike
        a brush footprint, so _update_region_overlay() below only ever walks
        O(perimeter) tiles, never O(area)."""
        tiles = [(x, ty0) for x in range(tx0, tx1)]
        if ty1 - ty0 > 1:
            tiles.extend((x, ty1 - 1) for x in range(tx0, tx1))
        for y in range(ty0 + 1, ty1 - 1):
            tiles.append((tx0, y))
            if tx1 - tx0 > 1:
                tiles.append((tx1 - 1, y))
        return tiles

    def _region_render_rect(self) -> tuple[int, int, int, int] | None:
        """What the overlay should show right now: the live anchor-to-current
        preview during a drag, else the committed region -- so starting a new
        drag previews without disturbing self._region until the drag actually
        commits (Escape-cancel then just re-renders the untouched committed
        value)."""
        if self._select_anchor is not None and self._select_current is not None:
            return region_clipboard.normalize_region(
                self._select_anchor, self._select_current, self._map_width or 0, self._map_height or 0
            )
        return self._region

    def _update_region_overlay(self) -> None:
        """Rebuilds the region's outline/ants/fill from _region_render_rect().
        Outline and ants share one QPainterPath built from _tile_polygon()
        per BOUNDARY tile (see _region_boundary_tiles()) -- the way
        _update_highlight() assembles the brush cursor, restricted to the
        perimeter so the outline conforms to Stepped columns and Sloped
        warping without costing O(area) on a region the size of the whole
        map. The fill is a separate O(area) path over every tile in the
        rect -- TODO(descape#region-select-perf): revisit if this lags on a
        whole-map Select All."""
        rect = self._region_render_rect()
        boundary_path = QPainterPath()
        fill_path = QPainterPath()
        if rect is not None:
            for tx, ty in self._region_boundary_tiles(*rect):
                polygon = self._tile_polygon(tx, ty)
                if polygon is not None:
                    boundary_path.addPolygon(polygon)
            tx0, ty0, tx1, ty1 = rect
            for ty in range(ty0, ty1):
                for tx in range(tx0, tx1):
                    polygon = self._tile_polygon(tx, ty)
                    if polygon is not None:
                        fill_path.addPolygon(polygon)
        if boundary_path.isEmpty():
            self._clear_region_overlay()
            return
        if self._region_fill_item is None:
            scene = self.scene()
            self._region_fill_item = scene.addPath(fill_path, QPen(Qt.NoPen), QBrush(self._region_fill_color))
            self._region_outline_item = scene.addPath(boundary_path, self._region_outline_pen)
            self._region_ants_item = scene.addPath(boundary_path, self._region_ants_pen)
            for item in (self._region_fill_item, self._region_outline_item, self._region_ants_item):
                item.setZValue(self.REGION_SELECT_Z)
        else:
            self._region_fill_item.setPath(fill_path)
            self._region_outline_item.setPath(boundary_path)
            self._region_ants_item.setPath(boundary_path)
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
        self._region_ant_timer.stop()

    def _sync_region_ant_timer(self) -> None:
        """The ants only cost anything while a region is actually shown --
        gated the same way _pulse_timer is gated on an edit tool being
        active."""
        if self._region_ants_item is not None:
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
        for _ in range(2):
            item = scene.addPolygon(QPolygonF(), self._ruler_pen)
            item.setZValue(self.RULER_Z)
            self._ruler_end_items.append(item)
        label = scene.addSimpleText("")
        font = label.font()
        font.setPixelSize(settings.get_ruler_label_font_px())
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
        # scene().clear() destroyed these too -- see show_mirror_overlay()'s
        # own docstring for why MirrorDialog must re-create rather than
        # reuse after any re-render.
        self._mirror_overlay_outline_item = None
        self._mirror_overlay_fill_item = None
        self._mirror_axis_items = []
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
        # Same reasoning: nothing left to warm a margin around, but
        # viewport_chunk_target() returning None here (no canvas item) still
        # needs a fire so a stale pre-close target isn't left recorded. Reset
        # the baseline first (see set_source()'s matching comment) so the
        # NEXT document's first poll fire can't be silently swallowed by a
        # coincidentally-equal leftover target.
        self._last_viewport_target = None
        self._note_viewport_changed()
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
        # scene().clear() destroyed these too. Dropped outright, unlike
        # set_source()'s own region handling below: File > Close leaves no
        # map for a region to refer to, so there is no "survives" case here.
        self._select_anchor = None
        self._select_current = None
        self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_ant_timer.stop()
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

    def set_paint_timed_callback(self, callback) -> None:
        """Installs or (with None) removes the current canvas item's paint
        stopwatch. See set_source()'s on_paint_timed. Silently no-ops with
        no canvas item, matching invalidate_region() above: File > Close
        drops the item, and a caller clearing a callback it installed
        shouldn't have to care whether the document is still open."""
        if self._canvas_item is None:
            return
        self._canvas_item._on_paint_timed = callback

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

    def viewport_chunk_target(self) -> tuple[int, int, int, int, int] | None:
        """(mip, cx0, cy0, cx1, cy1) -- the mip a real paint would select
        right now, and the inclusive chunk-index range the current viewport
        covers at it -- or None with no canvas item or a degenerate
        viewport. The paint-free equivalent of what MapCanvasItem.paint()
        knows: A2's hook for driving a margin warm (descape.margin_warm)
        off the viewport between paints, since option.exposedRect only
        exists inside an actual Qt paint call.

        Mip selection mirrors MapCanvasItem._select_mip() exactly but reads
        the ITEM's deviceTransform(viewportTransform) rather than a live
        QPainter's deviceTransform() (there is none here).
        QGraphicsItem.deviceTransform(viewportTransform) is used rather
        than self.viewportTransform() alone so a future item-level
        transform can't silently desynchronise this reading from what a
        real paint selects; multiplying by devicePixelRatioF() folds in
        DPR the same way ViewerWindow._start_level_warm already does, for
        the same reason -- scaling a transform by a uniform `s` scales its
        singular values by `s`, so this is exact, not an approximation.

        Scene rect: the viewport mapped to scene space via
        mapToScene(...).boundingRect(), intersected with the item's own
        boundingRect(). Under Flat's rotate+squash view transform, the
        bounding box of that mapped polygon is precisely the shape a real
        option.exposedRect already is.

        Correctness is pinned by a test, not by inspection (tests/
        test_margin_warm.py): after a real paint cycle, this method's mip
        must equal MapCanvasItem._last_mip, and its chunk range must
        contain every chunk that paint's own exposed rect resolved to."""
        item = self._canvas_item
        if item is None:
            return None
        device_transform = item.deviceTransform(self.viewportTransform())
        scale = _max_axis_scale(device_transform) * self.devicePixelRatioF()
        cache = item._cache
        mip = cache.mip_for_scale(scale)
        scene_rect = self._current_scene_rect()
        if scene_rect is None:
            return None
        level_rect = level_rect_for(cache, mip, scene_rect)
        if level_rect is None:
            return None
        x0, y0, x1, y1 = level_rect
        cx0, cy0, cx1, cy1 = cache.chunk_index_range(mip, x0, y0, x1, y1)
        return mip, cx0, cy0, cx1, cy1

    def _current_scene_rect(self) -> QRectF | None:
        """The scene-space rect the viewport currently covers, intersected
        with the canvas item's own boundingRect() -- the mip-independent
        half of viewport_chunk_target()'s projection, shared with
        viewport_chunk_target_at() below (2026-09-07 plan's load-time
        margin warm, Step 1) so there is only one copy of this
        mapToScene(...).boundingRect().intersected(...) line. None with no
        canvas item or a degenerate (empty) viewport."""
        item = self._canvas_item
        if item is None:
            return None
        rect = self.mapToScene(self.viewport().rect()).boundingRect().intersected(item.boundingRect())
        return None if rect.isEmpty() else rect

    def viewport_chunk_target_at(self, mip: int) -> tuple[int, int, int, int] | None:
        """(cx0, cy0, cx1, cy1) -- the chunk range level `mip` would show if
        the viewport were looking at it right now, projecting the SAME
        on-screen scene rect viewport_chunk_target() uses for the live mip
        (2026-09-07 plan's load-time margin warm, Step 1). Unlike that
        method, this never reads the device transform or picks a mip
        itself -- `mip` is the caller's, so this is safe to call for a
        level the viewport isn't actually showing (a neighbour level
        that hasn't been visited yet).

        None with no canvas item or a degenerate/empty projection at that
        mip. `mip` itself must be one this cache's ladder actually
        enumerates (mip_scale()/canvas_dims() raise KeyError otherwise) --
        every real caller gets `mip` from level_warm.neighbour_mips(),
        which already clamps to the enumerated ladder, so this is never
        asked about a mip outside it."""
        item = self._canvas_item
        if item is None:
            return None
        cache = item._cache
        scene_rect = self._current_scene_rect()
        if scene_rect is None:
            return None
        level_rect = level_rect_for(cache, mip, scene_rect)
        if level_rect is None:
            return None
        x0, y0, x1, y1 = level_rect
        return cache.chunk_index_range(mip, x0, y0, x1, y1)

    def viewport_chunk_span(self, cache, mip: int) -> tuple[int, int]:
        """(chunks_wide, chunks_tall) -- how many chunks a REAL viewport at
        `mip` would cover, sized from THIS view's own on-screen dimensions
        (device pixels) rather than from whatever scene-space rect happens
        to be visible right now. At least 1x1.

        This is deliberately independent of the live zoom/mip:
        viewport_chunk_target_at()'s raw projection reuses the CURRENT
        scene rect verbatim, which at a fit-to-view zoom (every
        load_scenario() open) is the whole map -- margin_warm.
        bounded_chunk_range() is what turns that raw projection plus this
        span into an actual viewport-sized patch; see its own docstring
        for why the two are split rather than done in one method here."""
        dpr = self.devicePixelRatioF()
        rect = self.viewport().rect()
        chunks_w = max(1, math.ceil(rect.width() * dpr / cache.chunk_px))
        chunks_h = max(1, math.ceil(rect.height() * dpr / cache.chunk_px))
        return chunks_w, chunks_h

    def _note_viewport_changed(self) -> None:
        """Starts the viewport-changed poll if it isn't already running --
        NEVER restarts an already-active one. That restart guard is the
        entire mechanism a restart-on-every-move debounce would be missing:
        see VIEWPORT_POLL_MS's own comment and _on_viewport_poll_tick() for
        why a restarted timer would starve during a continuous drag.

        Called from every path that can change what viewport_chunk_target()
        would return: scrollContentsBy (every pan source at once -- left
        ScrollHandDrag, middle-drag's direct scrollbar writes, keyboard
        scrolling, centerOn, and the scrollbar shifts a scale() call
        induces), wheelEvent (a zoom whose scrollbars happen not to move
        still changes the mip), _capture_zoom_baseline (covers
        resizeEvent), and set_source()/clear_image() (a new document)."""
        if not self._viewport_poll_timer.isActive():
            self._viewport_poll_timer.start()

    def _on_viewport_poll_tick(self) -> None:
        """One poll fire: notify on_viewport_changed() if the target has
        moved since the last fire, else stop -- the "self-stopping poll,
        not a restart-on-change debounce" shape A2.4 spends real space
        justifying. A REPEATING timer (not single-shot) firing at a bounded
        cadence throughout a drag, delivering the final target one fire
        after motion ends, then switching itself off: no trailing-edge
        special case, and a static view costs nothing once it stops."""
        target = self.viewport_chunk_target()
        if target == self._last_viewport_target:
            self._viewport_poll_timer.stop()
            return
        self._last_viewport_target = target
        self.on_viewport_changed()

    def set_source(
        self,
        tile_pixels: int,
        terrain_style: str = "flat",
        cache: "IsoChunkCache | FlatChunkCache | SlopedChunkCache | None" = None,
        elevations: np.ndarray | None = None,
        proj: "iso_geometry.IsoProjection | None" = None,
        unit_index=None,
        reset_view: bool = True,
        on_paint_timed: Callable[[float, int], None] | None = None,
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

        on_paint_timed (default None, i.e. untimed) is handed straight to the
        new MapCanvasItem, which then calls it with (elapsed_seconds, mip) on
        every paint until it is cleared again via
        set_paint_timed_callback(None). ViewerWindow uses it to report the
        deferred first-paint composite a load's own timings can't see.

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
        self._mirror_overlay_outline_item = None
        self._mirror_overlay_fill_item = None
        self._mirror_axis_items = []
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
        # scene().clear() destroyed the region's items too, but -- unlike the
        # ruler/unit selection above -- the region ITSELF is only dropped on
        # reset_view=True ("a different document is now on screen"; the old
        # tile rect may not even be on the new map). A same-document style
        # switch keeps it: the map's own tile grid hasn't changed, so a
        # previously selected rectangle is still exactly as meaningful,
        # just re-projected. Rebuilt (not merely kept) at the end of this
        # method, once _map_width/_map_height/_iso_elevations/_iso_proj all
        # match the new render.
        self._select_anchor = None
        self._select_current = None
        if reset_view:
            self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_ant_timer.stop()

        self._canvas_item = MapCanvasItem(cache)
        self._canvas_item._on_paint_timed = on_paint_timed
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
        # Not read at construction time by EdgeTickItem itself (it defaults
        # to edge_ticks.LABEL_FONT_PX), so a persisted non-default setting
        # needs this explicit push -- otherwise it would only take effect
        # after the user next touched the Appearance spinbox in the same
        # session, not on the next map open.
        self._edge_tick_item.set_label_font_px(settings.get_distance_tick_font_px())
        self.scene().addItem(self._edge_tick_item)

        # Ends with set_isometric(), which funnels into
        # _capture_zoom_baseline() and hands the item its first pad.
        # view_restore rides along via _pending_view_restore rather than a
        # set_isometric() parameter, since set_isometric() is also a public
        # entry point (the View > Isometric View checkbox) that must keep
        # its own no-argument fit-to-view behavior.
        self._pending_view_restore = view_restore
        self.set_isometric(self._isometric)
        # Rebuild (not merely keep) the region overlay now that the new
        # canvas item/projection exist -- see this method's own region
        # comment above for why self._region itself survives a style switch
        # while its scene items do not.
        self._update_region_overlay()
        # Forces the next poll fire to notify regardless of what it finds:
        # without this, a cache swap whose new viewport_chunk_target()
        # happens to equal the OLD document's last-recorded one (same mip,
        # same chunk range -- not far-fetched right after a reset_view=False
        # graphics-quality change) would see "no change" on its first fire
        # and stop silently, skipping that document's first margin ring
        # until the next scroll.
        self._last_viewport_target = None
        self._note_viewport_changed()

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
        # claiming -- an earlier version of this comparison got the
        # direction backwards.
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
        # Same coverage argument for the viewport-changed poll: resizeEvent
        # changes what viewport_chunk_target() would return with no scroll or
        # zoom action of its own, so it needs its own fire here rather than
        # relying on scrollContentsBy/wheelEvent.
        self._note_viewport_changed()

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

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        """The one Qt override that catches every pan source at once: left
        ScrollHandDrag, middle-drag's direct scrollbar writes (see
        mouseMoveEvent), keyboard scrolling, centerOn(), and the scrollbar
        shifts a scale() call induces -- all of them move the scrollbars,
        and this is what Qt calls whenever they do. A pair of scrollbar
        valueChanged connections would be equivalent but two objects
        instead of one."""
        super().scrollContentsBy(dx, dy)
        self._note_viewport_changed()

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current_scale = abs(self.transform().determinant()) ** 0.5
        if factor < 1.0 and self._min_linear_scale is not None and current_scale * factor < self._min_linear_scale:
            return
        if factor > 1.0 and self._max_linear_scale is not None and current_scale * factor > self._max_linear_scale:
            return
        self.scale(factor, factor)
        self._note_viewport_changed()
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
        if self._tool == TOOL_SELECT:
            self._clear_highlight()
            self._clear_pan_highlight()
            self._clear_unit_hover()
            if self._select_anchor is not None and tile is not None:
                # Same "hold the last valid tile" convention as the Ruler
                # above: an off-map move mid-drag must not collapse the
                # preview, since the drag legitimately runs off the edge.
                self._select_current = tile
                self._update_region_overlay()
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
