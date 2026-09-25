"""MapView: the QGraphicsView the map is panned, zoomed and edited in.

Dumb the same way TriggerPanel is -- it reports what the user did
through the callbacks ViewerWindow constructs it with and never
reaches back up into the window."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
from PyQt5.QtCore import QElapsedTimer, QLineF, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt5.QtWidgets import (
    QApplication,
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
    grid_overlay,
    iso_geometry,
    perf_trace,
    range_overlay,
    region_clipboard,
    ruler,
    settings,
    shape_tools,
    trigger_geometry,
    unit_pick,
)
from descape.render import SMALL_MAP_TILE_PIXELS
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- render.py imports no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.unit_filter import GAIA_PLAYER_ID
from descape.viewer_canvas import (
    AnalysisMarkerItem,
    CameraMarkerItem,
    EdgeTickItem,
    GridItem,
    MapCanvasItem,
    StackBadgeItem,
    UnitGhostItem,
    _max_axis_scale,
    level_rect_for,
    map_overlay_font,
)
from descape.viewer_common import (
    _TOOL_SHAPE,
    CLICK_TOOLS,
    EDIT_TOOLS,
    SHAPE_TOOLS,
    TOOL_EYEDROPPER,
    TOOL_RULER,
    TOOL_SELECT,
)

# Stroke tools _touch_tile gap-fills. Cliff (per-tile chain semantics) and Convert
# (own stroke path) are left out on purpose: still one tile per touch.
_INTERPOLATED_STROKE_TOOLS = frozenset({"draw", "elevation", "set_level"})


def _add_closed_polygons(path, polygons) -> None:
    """Appends each polygon to `path` as its own CLOSED subpath.
    addPolygon() leaves the subpath open, so stroking it draws only 3 of a
    diamond's 4 edges -- invisible wherever a fill dominates, and plain wrong
    for an outline-only cue."""
    for points in polygons:
        path.addPolygon(QPolygonF([QPointF(x, y) for x, y in points]))
        path.closeSubpath()


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

    # One mouse-wheel detent in QWheelEvent.angleDelta() units (Qt's own
    # documented value: eighths of a degree, 15 degrees per detent). A
    # high-resolution wheel or a trackpad sends fractions of this many units
    # per event, so wheelEvent() accumulates rather than treating every event
    # as a full zoom step. See wheelEvent's own docstring.
    WHEEL_NOTCH_UNITS = 120
    # Idle gap that ends a wheel GESTURE, so the next event starts a fresh
    # one with a fresh +/-1 mip budget. PROVISIONAL, in the same sense as
    # VIEWPORT_POLL_MS: long enough that one flick of a real wheel stays one
    # gesture, short enough that a deliberate second roll isn't refused.
    WHEEL_GESTURE_GAP_S = 0.25

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
    # mouse rather than a static "you clicked here" marker. A GOLD pulse is
    # specifically "a brush is live and about to mutate"; the Ruler's
    # endpoints pulse too (RULER_PULSE_MIN_ALPHA below), in its own orange,
    # meaning "an active measurement". Follows on hover, not just on click. Colors are user-settable (Settings > Appearance,
    # settings.OVERLAY_COLORS) and live in self._highlight_outline_pen/
    # self._highlight_fill_color, rebuilt by _rebuild_overlay_ink() -- these
    # are just the pulse behaviour, which isn't a color and stays fixed.
    HIGHLIGHT_PULSE_MIN_ALPHA = 0.25
    HIGHLIGHT_PULSE_MAX_ALPHA = 0.55
    HIGHLIGHT_PULSE_PERIOD_MS = 500
    HIGHLIGHT_PULSE_TICK_MS = 40

    # Held-key pan: ~60 fps, and the dt clamp that stops a stalled event loop
    # from teleporting the view on the tick that finally lands.
    PAN_TICK_MS = 16
    PAN_MAX_DT_MS = 100.0
    # Referenced from viewer.py as MapView.PAN_DIRECTIONS (that module does
    # `from descape.map_view import MapView`, so a bare module-level name
    # would not be reachable there).
    PAN_DIRECTIONS: ClassVar[dict[str, tuple[int, int]]] = {
        "view_pan_up": (0, -1),
        "view_pan_down": (0, 1),
        "view_pan_left": (-1, 0),
        "view_pan_right": (1, 0),
    }
    # QKeySequence packs a key WITH its modifier bits into one int; this
    # masks the modifier half off. Qt.KeyboardModifierMask isn't exposed by
    # every PyQt5 build, hence the literal.
    _KEY_MODIFIER_MASK = 0xFE000000

    # Units mode's two cues, phase 3's P3-d. Deliberately NOT the pulsing
    # highlight above: a selection is a resting state, where both pulses
    # above mean something is live right now. Hover defaults thin
    # and quiet like Pan's; selection is solid plus a translucent fill. Neither
    # pulses. By default (View > Colour Selection by Owner) each owner's units
    # take that player's colour over a dark under-stroke; GAIA, the toggle
    # off, or no known colours use the configured colour, which is the only
    # one kept distinct from the edit highlight by default.
    # Colors live in self._unit_hover_pen/self._unit_select_pen/
    # self._unit_select_fill_color; UNIT_SELECT_FILL_ALPHA is the one part of
    # the fill that ISN'T user-settable (see settings.OVERLAY_COLORS's own
    # comment on RGB-only scope), and UNIT_SELECT_PEN_WIDTH is this
    # codebase's only non-cosmetic pen width, deliberately not 0 (the
    # under-stroke is non-cosmetic for the same reason).
    UNIT_SELECT_FILL_ALPHA = 70
    UNIT_SELECT_PEN_WIDTH = 2
    UNIT_SELECT_UNDERSTROKE_WIDTH = 4
    UNIT_SELECT_UNDERSTROKE_RGBA = (10, 10, 10, 180)

    # The codebase's FIRST setZValue use -- everything else stacks by scene
    # INSERTION order, and the existing highlight items only land on top
    # because they're created lazily on first hover, after set_source() has
    # added the canvas item. Two independently-lazily-created unit items
    # would stack in whichever order the user happened to trigger first, so
    # these are explicit rather than inheriting that latent ordering bug.
    # View > Footprint Outlines. Below UNIT_HOVER_Z, explicitly rather than
    # by insertion order, so hover, selection and the ruler stay legible over
    # a map-wide outline layer. Colour is settings.OVERLAY_COLORS'
    # "footprint_outline", rebuilt by _rebuild_overlay_ink().
    FOOTPRINT_Z = 5.0

    # View > Range Rings. Above the footprint layer and below the hover cue,
    # so the two cues that say "this one, right now" both read over it: the
    # ring is ambient context about an already-selected building. Colour is
    # settings.OVERLAY_COLORS' "range_ring", rebuilt by _rebuild_overlay_ink().
    RANGE_RING_Z = 9.0

    UNIT_HOVER_Z = 10.0
    # The selection is one fill/under-stroke/outline trio per owner colour.
    # Explicit sub-layers so no group's fill can wash over another's edge,
    # whichever group was created first. The marquee sits at UNIT_SELECT_Z.
    UNIT_SELECT_Z = 11.0
    UNIT_SELECT_UNDER_Z = 11.1
    UNIT_SELECT_OUTLINE_Z = 11.2

    # b1.5: how far a press must travel (screen pixels) before release counts
    # as a unit move rather than a plain select click -- a physically-held
    # mouse rarely lands at the exact press pixel, so a bare "did it move at
    # all" test would misfire on ordinary clicks.
    UNIT_DRAG_THRESHOLD_PX = 4

    # Above this many tiles, a shape preview drops its brush union and
    # previews the bare rasterized path instead. The committed set is
    # unaffected. A size-9 brush unioned along a 680-tile line is ~55k tiles
    # before dedupe, each one a _tile_polygon() call -- in Sloped that reads
    # four corner_rise values per tile, into a single QPainterPath, which
    # freezes the drag.
    SHAPE_PREVIEW_TILE_LIMIT = 4000

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
    # The two endpoint tiles' pulsing fill, a sibling of each static outline
    # rather than a property of it: pulsing the outline itself would fade the
    # crisp shape that says which tile was picked. Its own min/max pair, not
    # the highlight's: it sits over terrain the edit highlight never covers.
    # Colour is the ruler's own, so recolouring the ruler recolours the glow.
    RULER_PULSE_MIN_ALPHA = 0.15
    RULER_PULSE_MAX_ALPHA = 0.50
    # Above UNIT_SELECT_Z: a measurement is a deliberate act, and should
    # not be occluded by the hover cue it was drawn on top of.
    # Under RULER_Z so the endpoint outline still reads over its own glow.
    RULER_GLOW_Z = 11.7
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
    REGION_SELECT_ANT_DASH: ClassVar[list[float]] = [4.0, 4.0]  # device pixels, since cosmetic
    REGION_SELECT_ANT_STEP_PX = 1.0
    REGION_SELECT_ANT_TICK_MS = 80
    # Above UNIT_SELECT_Z (a region can carry units, so its outline must read
    # on top of the units inside it), below RULER_Z (a measurement is a
    # deliberate act and should never be occluded).
    REGION_SELECT_Z = 11.5
    # Stacked-unit count badges: above the region outline (a badge is small
    # and must stay readable over a selection), below the ruler.
    UNIT_STACK_Z = 11.6

    # View > Player Cameras (GH #22): one camera glyph per player whose
    # starting view is set. Above the unit sprites, the selection (11.0) and
    # the region outline (11.5), and just above the stack badge, since both
    # are small glyphs and the marker is the rarer, deliberately-enabled one.
    # Below the ruler (11.7+), which is a transient tool that must read over
    # every ambient overlay.
    CAMERA_MARKER_Z = 11.65

    # The mid-drag move preview's ghost. Above REGION_SELECT_Z (a unit dragged
    # across a selected region must read on top of that region's outline, since
    # the ghost IS the thing the gesture is about), below UNIT_STACK_Z so a
    # stack badge stays readable through a drag passing under it.
    UNIT_GHOST_Z = 11.55

    # Tools > Map Analysis' markers, one layer item while the results dialog
    # is open. Above RULER_LABEL_Z (13.0), so a finding never hides under a
    # measurement label; below MIRROR_OVERLAY_Z (14.0), which stays on top of
    # everything. Colour is settings.OVERLAY_COLORS' "analysis_marker".
    ANALYSIS_MARKER_Z = 13.5

    # View > Trigger Overlay (GH #41): the selected trigger's areas, location
    # marks, runs and run labels. Area and marks sit above the selection trio
    # (11.0-11.2) and below REGION_SELECT_Z (an active region selection wins);
    # labels above RULER_LABEL_Z and below the analysis markers. Colours are
    # settings.OVERLAY_COLORS' trigger_* rows; the selected entry draws at full
    # strength, its siblings at the DIM alphas in the same hue.
    TRIGGER_AREA_Z = 11.3
    TRIGGER_MARK_Z = 11.4
    TRIGGER_LABEL_Z = 13.25
    TRIGGER_PEN_WIDTH = 2
    TRIGGER_FILL_ALPHA = 55
    TRIGGER_DIM_FILL_ALPHA = 20
    TRIGGER_DIM_PEN_ALPHA = 110
    # A location mark is its tile's own polygon shrunk to this fraction.
    TRIGGER_LOCATION_SCALE = 0.6

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
        on_stroke_tiles,
        on_stroke_end,
        on_click_edit,
        on_shape_commit,
        on_click_select,
        on_unit_place,
        on_unit_move,
        on_unit_nudge,
        on_unit_delete,
        on_marquee_select,
        on_region_selected,
        on_region_move,
        on_unit_drag_preview,
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
        # View > Grid, same lifetime as the tick state above.
        self._grid_item: GridItem | None = None
        self._grid_enabled = settings.get_grid_overlay()
        self._grid_blend = settings.get_grid_blend()
        self._grid_thickness = settings.get_grid_thickness()
        self._grid_follow_elevation = settings.get_grid_follow_elevation()
        # A Settings > Appearance slider drag in progress: see begin_grid_preview().
        self._grid_previewing = False
        # View > Footprint Outlines, same lifetime again.
        self._footprint_item = None
        self._footprint_enabled = settings.get_footprint_outlines()
        self._footprint_scope = settings.get_footprint_scope()
        self._footprint_refresh_pending = False
        # Stacked-unit badges: groups recomputed on every index swap, item
        # rebuilt per set_source() like _edge_tick_item.
        self._stack_badge_item: StackBadgeItem | None = None
        self._stack_badges_enabled = settings.get_stack_badges()
        self._stack_groups: dict = {}
        # View > Player Cameras: the pushed marker list survives a
        # set_source() (it is scenario state, not scene state), the items do
        # not -- scene().clear() destroys them, so they are rebuilt from the
        # list against the new projection.
        self._camera_markers: list[tuple[int, int, int, QColor]] = []
        self._camera_marker_emphasised: int | None = None
        self._camera_marker_items: list[CameraMarkerItem] = []
        self._camera_markers_enabled = settings.get_camera_markers()
        # Tools > Map Analysis markers: (tile, severity, count) anchors live
        # as long as the dialog; the item is created lazily and, like the
        # camera markers, rebuilt against the new projection after set_source().
        self._analysis_markers: list[tuple[tuple[int, int], str, int]] = []
        self._analysis_focus: tuple[int, int] | None = None
        self._analysis_marker_item: AnalysisMarkerItem | None = None
        # View > Trigger Overlay: the pushed TriggerShape snapshot survives a
        # same-document set_source() like self._region; the items do not.
        self._trigger_shapes: list = []
        self._trigger_emphasis: tuple[str, int] | None = None
        self._trigger_items: list[QGraphicsItem] = []
        self._trigger_overlay_enabled = settings.get_trigger_overlay()
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
        # Trigger Pick from map: a picked unit's (player, ref) key, and a
        # cancel (Escape or right-click). Assigned by the window, like the above.
        self.on_unit_pick = lambda key: None
        self.on_unit_picker_cancel = lambda: None
        self._unit_picker_active = False
        # (cover tile, stack index) of the last pick, so a repeat click walks the stack.
        self._picker_cycle: tuple[tuple[int, int], int] | None = None
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
        self._ruler_glow_items: list[QGraphicsPolygonItem] = []
        self._ruler_label_item: QGraphicsSimpleTextItem | None = None
        self._on_hover = on_hover
        # A "stroke" is one drag with an edit tool active, from press to
        # release (or a defensive close on leaveEvent) -- see mousePressEvent/
        # mouseMoveEvent/mouseReleaseEvent/leaveEvent below. on_stroke_start()
        # takes no args; on_stroke_tiles(tiles, modifiers) fires once per
        # mouse event that entered new tiles, with those tiles in path order
        # (deduped within the stroke, and gap-filled for the interpolated
        # tools -- see _touch_tile); on_stroke_end() closes it. ViewerWindow
        # uses this to keep exactly one descape.edit_history.EditHistory
        # record per stroke while still repainting live as the drag progresses.
        self._on_stroke_start = on_stroke_start
        self._on_stroke_tiles = on_stroke_tiles
        self._on_stroke_end = on_stroke_end
        # CLICK_TOOLS (Paint Can, and any future tool with no meaningful
        # drag semantics) never open a stroke at all -- on_click_edit(tile_x,
        # tile_y, modifiers) fires once per left press instead, handled by a
        # dedicated branch in mousePressEvent above the stroke block, so the
        # begin_stroke/commit_stroke pairing above stays exactly one-to-one
        # with no special-casing.
        self._on_click_edit = on_click_edit
        # SHAPE_TOOLS (Draw Line, Draw Rectangle): on_shape_commit(tiles)
        # fires exactly once, at release, with the final tile list. Neither
        # the stroke trio nor on_click_edit fits -- a stroke cannot un-paint
        # a tile the rubber band has since shrunk off, so nothing is mutated
        # until the button comes up. The tile list is computed here rather
        # than re-derived by ViewerWindow so the committed set can never
        # disagree with the one the preview just showed.
        self._on_shape_commit = on_shape_commit
        # The live shape drag: anchor tile at press, endpoint at the last
        # move, and the modifiers last seen (Shift recomputes the preview
        # from them). Plain attributes rather than a session object like the
        # Ruler's -- a shape drag never outlives its press, so there is no
        # pending state to model, and an armed-across-tool-switch MUTATING
        # tool is a hazard the Ruler can afford and this cannot.
        self._shape_anchor: tuple[int, int] | None = None
        self._shape_end: tuple[int, int] | None = None
        self._shape_modifiers = Qt.NoModifier
        # The live drag's shape ("line" | "rect" | "wall"), latched at press
        # so a mid-drag catalog change can't switch it. "" when no drag.
        self._drag_shape = ""
        # GH #98: returns "wall" when Place Unit should drag a wall run. A
        # setter rather than a constructor callable, which would grow the
        # already long injected list (a known merge-collision point).
        self._place_shape_query: Callable[[], str] = lambda: ""
        # GH #75's unit-drag hooks, set by set_unit_drag_hooks() for the same reason.
        self._on_unit_click_release: Callable[[tuple[int, int]], None] = lambda key: None
        self._is_group_drag: Callable[[tuple[int, int]], bool] = lambda key: False
        # Draw Rectangle's Fill/Outline state, pushed down from the toolbar
        # combo by ViewerWindow.set_rect_filled() the same way set_brush()
        # pushes brush size/shape.
        self._rect_filled = True
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
        # on_region_move(dx, dy), the thirteenth: a committed region the
        # window has marked movable (set_region_movable) was dragged by whole
        # tiles and released somewhere else. Fires only on a non-zero delta,
        # and only reports the delta -- what a move MEANS is entirely the
        # window's business, since MapView knows nothing about pastes.
        self._on_region_move = on_region_move
        # on_unit_drag_preview(key, scene_pos, modifiers), the fourteenth: one
        # mouse-move of an in-progress unit drag, past UNIT_DRAG_THRESHOLD_PX.
        # MapView cannot resolve what the ghost should look like -- it holds no
        # scenario, so it has no player_colors, no team_indices and no sprites
        # toggle -- so it reports the gesture and ViewerWindow pushes the
        # resolved content back through set_unit_ghost(), the same split
        # on_click_select/set_unit_selection already uses.
        self._on_unit_drag_preview = on_unit_drag_preview
        # The ghost overlay itself, created lazily on the first move past the
        # threshold and cleared on every exit from the drag (release, leave,
        # Escape) -- _update_marquee/_clear_marquee's lifecycle exactly.
        self._unit_ghost_item: UnitGhostItem | None = None
        # The anchor tile from mousePressEvent, and the last tile a move
        # resolved to (frozen, like the Ruler's endpoint, if the drag runs
        # off-map) -- both None outside an in-progress drag. The COMMITTED
        # region lives in self._region below, mirroring the anchor/committed
        # split ruler.RulerSession keeps internally.
        self._select_anchor: tuple[int, int] | None = None
        self._select_current: tuple[int, int] | None = None
        # The move-a-pasted-region drag. MapView deliberately knows nothing
        # about what a paste is: the window sets _region_movable, and this
        # class decides only press-inside vs press-outside.
        self._region_movable = False
        self._move_anchor: tuple[int, int] | None = None
        self._move_current: tuple[int, int] | None = None
        # The committed region, half-open tile-space (tx0, ty0, tx1, ty1).
        # Set by set_region() -- MapView's own drag-release/Escape-clear
        # paths, and ViewerWindow's Select All/Deselect, both funnel through
        # it, so there is exactly one place that rebuilds the overlay.
        self._region: tuple[int, int, int, int] | None = None
        self._region_fill_item: QGraphicsPathItem | None = None
        self._region_outline_item: QGraphicsPathItem | None = None
        self._region_ants_item: QGraphicsPathItem | None = None
        # The overlay's scene-space bounds, remembered whenever the boundary
        # path is rebuilt. See _region_on_screen(), which gates the ants.
        self._region_scene_rect: QRectF | None = None
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
        # Group key (an owner's RGB, or None for the configured colour) ->
        # (fill, under-stroke or None, outline). See _draw_unit_selection().
        self._unit_select_groups: dict = {}
        # scenario.player_colors, pushed by ViewerWindow (MapView holds no scenario).
        self._selection_player_colors: tuple | None = None
        self._selection_by_owner = settings.get_selection_by_owner()
        # One item for every selected building's ring, created lazily on the
        # first draw like the selection groups above.
        self._range_ring_item: QGraphicsPathItem | None = None
        self._range_rings_enabled = settings.get_range_rings()
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
        # out only (skip re-entering _touch_tile/on_stroke_tiles for a mouse
        # move that hasn't left the current cursor tile), not a correctness
        # guarantee. With a brush bigger than one tile, one painted tile
        # falls under many distinct cursor tiles during a drag, so the
        # tile-level dedupe that actually matters (e.g. Elevate's
        # accumulating +/-1 must apply once per stroke, not once per cursor
        # tile that overlapped it) is ViewerWindow._stroke_painted instead.
        self._stroke_touched: set[tuple[int, int]] = set()
        # Cursor tile of the stroke's last _touch_tile, where the next gap-fill starts.
        self._stroke_last_tile: tuple[int, int] | None = None
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
        # Held-key pan (view_pan_*). Its own timer, deliberately not
        # _pulse_timer above: that one only runs in EDIT_TOOLS and no-ops
        # without a highlight item, i.e. it is stopped in exactly the
        # Pan-mode case that matters here.
        #
        # _pan_bindings maps (key, modifiers) -> (dx, dy) and is pushed in by
        # ViewerWindow.apply_keybind; _pan_held is the subset currently down.
        # The accumulators carry the sub-pixel remainder between ticks --
        # setValue() takes an int, so a slow speed at 16 ms ticks would
        # otherwise truncate to zero every tick and never move.
        self._pan_bindings: dict[tuple[int, int], tuple[int, int]] = {}
        self._pan_held: dict[tuple[int, int], tuple[int, int]] = {}
        self._pan_accum_x = 0.0
        self._pan_accum_y = 0.0
        self._pan_timer = QTimer(self)
        self._pan_timer.timeout.connect(self._on_pan_tick)
        self._pan_elapsed = QElapsedTimer()
        # Middle-button pan is independent of the active tool/dragMode
        # (Qt's ScrollHandDrag only ever responds to the left button) --
        # handled manually via the scrollbars, see mouse{Press,Move,Release}Event.
        self._middle_drag_active = False
        self._middle_drag_last_pos = None
        # Set for real by _capture_zoom_baseline(), called from
        # set_isometric() once a map is loaded and a fit-to-view scale exists.
        self._min_linear_scale: float | None = None
        self._max_linear_scale: float | None = None
        # Wheel-gesture state, all reset together by _end_wheel_gesture():
        # leftover sub-notch angleDelta, the direction the gesture is
        # running in, the mip it started on (None while idle), and when the
        # last wheel event arrived.
        self._wheel_accum = 0
        self._wheel_dir = 0
        self._wheel_gesture_mip: int | None = None
        self._wheel_last_t = 0.0
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
        self._footprint_pen = _pen("footprint_outline", 0)
        # Cosmetic width 0 like the footprint outline: Flat's squash
        # transform would thin a real-width line unevenly around the ellipse.
        self._range_ring_pen = _pen("range_ring", 0)
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
        if self._analysis_marker_item is not None:
            self._analysis_marker_item.set_color(QColor(settings.get_overlay_color("analysis_marker")))
        if self._trigger_items:
            # Rebuilt rather than re-inked: at most a handful of items, two pen strengths.
            self._rebuild_trigger_overlay()
        if self._footprint_item is not None:
            self._footprint_item.setPen(self._footprint_pen)
        if self._range_ring_item is not None:
            self._range_ring_item.setPen(self._range_ring_pen)
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

    def set_rect_filled(self, filled: bool) -> None:
        """Draw Rectangle's Fill/Outline state, pushed down from the toolbar
        combo. Same shape as set_brush() above, including the memo reset: the
        preview's tile set depends on this, so a stale key would keep the old
        one on screen until the cursor crossed a tile boundary."""
        self._rect_filled = filled
        self._highlight_key = None

    def set_place_shape_query(self, fn: Callable[[], str]) -> None:
        """GH #98: `fn()` returns "wall" when Place Unit's picked const is a
        wall, which turns Place Unit into a wall-run drag tool."""
        self._place_shape_query = fn

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
            self._clear_unit_ghost()
            self._marquee_start_pos = None
            self._clear_marquee()
        # Defensive, mirroring the _unit_drag_key reset above: a mode
        # switch mid-drag reaches the tool through _update_tool_enabled's
        # forced-back-to-Pan path (which calls set_tool, which cancels
        # already), so this is a second belt on the same trousers.
        self._cancel_shape()
        self._sync_stack_badges_visible()
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
            or self._unit_picker_active
            or self._tool in EDIT_TOOLS
            or self._tool == TOOL_RULER
            or self._tool == TOOL_EYEDROPPER
        ):
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_unit_picker(self, active: bool) -> None:
        """Trigger Pick from map: left clicks pick a unit (on_unit_pick) and
        hover outlines one, in any mode. The window forces Pan first."""
        active = bool(active)
        if active == self._unit_picker_active:
            return
        self._unit_picker_active = active
        self._picker_cycle = None
        if not active:
            self._clear_unit_hover()
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def unit_picker_active(self) -> bool:
        return self._unit_picker_active

    def _picker_click(self, pos: QPointF) -> None:
        """One pick: the unit under pos, walking down a stack on repeat clicks."""
        found = self.pick_unit_cover_at(pos)
        if found is None:
            return
        entry, cover_tile = found
        previous = self._picker_cycle
        entry, index = unit_pick.stack_cycle_step(
            self.stack_group_at(cover_tile), entry, previous[1] if previous and previous[0] == cover_tile else None
        )
        self._picker_cycle = (cover_tile, index) if index >= 0 else None
        self.on_unit_pick(unit_pick.unit_key(entry.player_id, entry.unit))

    def set_unit_index(self, index) -> None:
        """Swaps the pick index (a filter change rebuilds it) and drops any
        hover/selection keys that the new index can no longer resolve."""
        self._unit_index = index
        self.refresh_stack_groups()
        self._clear_unit_hover()
        self.refresh_unit_highlight()
        self.refresh_footprint_overlay()

    def refresh_after_index_patch(self) -> None:
        """For a caller that patched _unit_index in place rather than
        swapping it: everything derived from the index, which is the stacks
        and the footprint outlines."""
        self.refresh_stack_groups()
        self.refresh_footprint_overlay()

    def refresh_stack_groups(self) -> None:
        """Recomputes the stacks from the current index -- for a caller that
        patched that index in place rather than swapping it."""
        self._stack_groups = {} if self._unit_index is None else unit_pick.stack_groups(self._unit_index)
        self._rebuild_stack_badges()

    def stack_group_at(self, tile: tuple[int, int]) -> list | None:
        """The stack members (top-first) anchored at `tile`, or None."""
        return self._stack_groups.get(tile)

    def _rebuild_stack_badges(self) -> None:
        if self._stack_badge_item is None:
            return
        badges = []
        for (tx, ty), members in self._stack_groups.items():
            polygon = self._tile_polygon(tx, ty)
            if polygon is None:
                continue
            rect = polygon.boundingRect()
            badges.append((QPointF(rect.center().x(), rect.top()), len(members)))
        self._stack_badge_item.set_badges(badges)
        self._sync_stack_badges_visible()

    def _sync_stack_badges_visible(self) -> None:
        if self._stack_badge_item is not None:
            self._stack_badge_item.setVisible(
                self._stack_badges_enabled and self._mode == "units" and bool(self._stack_groups)
            )

    def set_stack_badges(self, enabled: bool) -> None:
        """Shows or hides the stacked-unit badges. Nothing is baked into
        chunk pixels, so there is nothing to evict."""
        self._stack_badges_enabled = enabled
        self._sync_stack_badges_visible()

    def set_selection_player_colors(self, colors) -> None:
        """Stores scenario.player_colors (or None). Stores only: callers
        redraw explicitly, via set_unit_index() or refresh_unit_highlight()."""
        self._selection_player_colors = None if colors is None else tuple(colors)

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
            farms_draped=False if cache is None else (cache.with_units and cache.sprites_enabled),
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

    def _selection_group_key(self, player_id: int):
        """The owner's RGB, or None for the configured colour (toggle off,
        GAIA, or no colours pushed)."""
        colors = self._selection_player_colors
        if not self._selection_by_owner or colors is None or player_id == GAIA_PLAYER_ID:
            return None
        if not 0 < player_id < len(colors):
            return None
        return tuple(colors[player_id])

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
            return unit_pick.unit_rise_px_for(entry, cache.corner_rise)
        if self._iso_elevations is None or self._iso_proj is None:
            return None
        return int(self._iso_elevations[entry.own_y, entry.own_x]) * self._iso_proj.elev_step

    def _range_ring_points(self, entry):
        """One entry's ring polygon in scene space, or None when it draws no
        ring: not a building, no range, or no geometry to project with.

        The centre is the unit's own stored coordinate, with no span
        arithmetic: every corpus placement sits at `tile + span/2` per axis
        (AGENTS.md), so that coordinate already IS the footprint centre. Span
        enters only through ring_radius_for_const()."""
        radius = range_overlay.ring_radius_for_const(entry.unit.unit_const)
        if radius is None:
            return None
        if self._terrain_style == "flat":
            return range_overlay.ring_points(
                entry.unit.x, entry.unit.y, radius, "flat", tile_px=self._tile_pixels or 1
            )
        if self._iso_proj is None:
            return None
        rise_px = self._range_ring_rise_px(entry)
        if rise_px is None:
            return None
        return range_overlay.ring_points(
            entry.unit.x, entry.unit.y, radius, self._terrain_style, proj=self._iso_proj, rise_px=rise_px
        )

    def _draw_unit_range_rings(self, entries) -> None:
        """One QPainterPath unioning every qualifying entry's ring, the shape
        _draw_unit_selection() already uses for N units.

        The ring draws OVER the sprites, unlike the game's own, which
        composites its circle into the terrain so a palisade occludes it.
        That is an accepted deviation: drawing under the sprites means
        painting into cached chunk pixels, a different feature."""
        if not self._range_rings_enabled:
            self._clear_unit_range_rings()
            return
        polygons = [points for points in (self._range_ring_points(e) for e in entries) if points]
        if not polygons:
            self._clear_unit_range_rings()
            return
        path = QPainterPath()
        _add_closed_polygons(path, polygons)
        if self._range_ring_item is None:
            self._range_ring_item = self.scene().addPath(path, self._range_ring_pen)
            self._range_ring_item.setZValue(self.RANGE_RING_Z)
        else:
            self._range_ring_item.setPath(path)

    def _clear_unit_range_rings(self) -> None:
        if self._range_ring_item is not None:
            self.scene().removeItem(self._range_ring_item)
            self._range_ring_item = None

    def set_range_rings(self, enabled: bool) -> None:
        """Shows or hides View > Range Rings. Like set_footprint_outlines,
        this evicts no chunk cache: the ring is a scene item, never baked into
        canvas pixels. Redraws through the selection, which is the only thing
        a ring ever hangs off."""
        self._range_rings_enabled = enabled
        if not enabled:
            self._clear_unit_range_rings()
        self.refresh_unit_highlight()

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

    def set_unit_drag_hooks(self, on_click_release, is_group_drag) -> None:
        """GH #75's two viewer hooks, a setter like set_place_shape_query:
        on_click_release(key) ends a sub-threshold press on a unit, and
        is_group_drag(key) says the drag moves a whole selection."""
        self._on_unit_click_release = on_click_release
        self._is_group_drag = is_group_drag

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

    def set_unit_ghosts(self, draws=(), marks=()) -> None:
        """A group ghost (GH #75): every member's sprite draws, plus a
        (polygons, color) mark per member whose sprite did not resolve."""
        item = self._unit_ghost_item
        if item is None:
            item = UnitGhostItem()
            item.setZValue(self.UNIT_GHOST_Z)
            self.scene().addItem(item)
            self._unit_ghost_item = item
        if not item.set_content(draws, [(p, c) for p, c in marks if p and c is not None]):
            self._clear_unit_ghost()

    def set_unit_ghost(self, draws=None, polygons=None, color=None) -> None:
        """The mid-drag move preview's content, resolved by ViewerWindow and
        pushed here -- `set_unit_selection`'s shape, for the same reason (see
        this class's `_on_unit_drag_preview` comment).

        `draws` are render.unit_sprite_draws_at()'s (draw, px, py) triples;
        `polygons`/`color` are the coloured-mark fallback. Passing neither, or
        a sprite resolve that comes back empty with no mark behind it, clears
        the ghost rather than leaving a stale one on screen: an empty
        sprite_pieces_for(), and a None from unit_polygons() on a ghost entry
        at a destination with no elevations/proj/corner_rise, both mean "no
        ghost this frame" -- never a crash, never a partial draw."""
        item = self._unit_ghost_item
        if item is None:
            item = UnitGhostItem()
            item.setZValue(self.UNIT_GHOST_Z)
            self.scene().addItem(item)
            self._unit_ghost_item = item
        if draws and item.set_sprites(draws):
            return
        if polygons and color is not None and item.set_mark(polygons, color):
            return
        self._clear_unit_ghost()

    def _clear_unit_ghost(self) -> None:
        if self._unit_ghost_item is not None:
            self.scene().removeItem(self._unit_ghost_item)
            self._unit_ghost_item = None

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
        xs = [*range(left, right, step), right]
        ys = [*range(top, bottom, step), bottom]
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
        if tool != TOOL_SELECT and (self._select_anchor is not None or self._move_anchor is not None):
            self._select_anchor = None
            self._select_current = None
            self._move_anchor = None
            self._move_current = None
            self._update_region_overlay()
        # Cancel, never commit -- a shape the user never released is not a
        # shape they asked to paint. Unconditional rather than gated on the
        # new tool: unlike a region, a live shape drag belongs to the drag,
        # not to the tool, so even switching between Line and Rectangle
        # drops it.
        self._cancel_shape()
        self._sync_pulse_timer()
        # Routed through _apply_drag_mode() rather than set here directly, so
        # Units mode's NoDrag can't be undone by a tool switch.
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _apply_tool_cursor(self) -> None:
        if self._middle_drag_active:
            return  # middle-drag's closed-hand cursor takes priority for now
        if self._unit_picker_active:
            self.setCursor(Qt.PointingHandCursor)
            return
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

    def _pick_map_point(self, pos: QPointF) -> tuple[float, float] | None:
        """The CONTINUOUS map point under scene-space pos, or None -- free
        placement's Stage 2, and `_pick_tile`'s fractional counterpart.

        A separate method rather than a widened `_pick_tile`, deliberately:
        `_pick_tile` has many callers (`_pos_on_map`, `mouseMoveEvent`,
        `_scene_rect_to_tile_rect`, `pick_unit_at`, `on_unit_place`,
        `on_unit_move`) and every one of them wants an integer tile.

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
                corner_rise=cache.corner_rise, tile=tile,
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
        self._on_stroke_tiles(tiles, modifiers)
        perf_trace.step()

    def _end_stroke(self) -> None:
        self._stroke_active = False
        self._stroke_touched = set()
        self._stroke_last_tile = None
        self._stroke_button = None
        # Resumes the highlight pulse the press paused. See
        # _sync_pulse_timer() for why a stroke stops it.
        self._sync_pulse_timer()
        try:
            self._on_stroke_end()
        finally:
            perf_trace.end_drag(self._tool)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._middle_drag_active = True
            self._middle_drag_last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
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
                    if self._region_movable and self._tile_in_region(tile):
                        self._move_anchor = tile
                        self._move_current = tile
                    else:
                        self._select_anchor = tile
                        self._select_current = tile
                    self._update_region_overlay()
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
                    self._drag_shape = _TOOL_SHAPE.get(self._tool) or "wall"
                    self._shape_anchor = tile
                    self._shape_end = tile
                    self._shape_modifiers = event.modifiers()
                    self._update_shape_preview()
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
            #
            # on_click_select's return IS the press's pick result: it returns
            # the key it acted on, or None on empty ground, and it decides that
            # from pick_unit_cover_at(), which pick_unit_at() is a one-line
            # wrapper over. So a separate pick_unit_at() here was a second full
            # unit pick per click (perf item: "a Units-mode CLICK does two full
            # unit picks"), and its `selected_key or unit_key(entry...)`
            # fallback was unreachable, since the two can only be None together.
            # After a repeat click on a stack the key is a lower unit than the
            # top one, which is exactly the one dragging must move.
            selected_key = self._on_click_select(pos, event.modifiers())
            self._unit_drag_key = selected_key
            self._unit_drag_press_pos = event.pos()
            # b2.3's marquee: only a candidate when the press hit nothing --
            # a hit is already b1.5's move-drag above, and the two must never
            # both be live (mouseReleaseEvent checks _unit_drag_key first).
            self._marquee_start_pos = event.pos() if selected_key is None else None
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
                self._stroke_last_tile = None
                self._stroke_button = event.button()
                # Pauses the pulse for the duration of the stroke, which is
                # the one time its 40ms tick competes with real edit work.
                self._sync_pulse_timer()
                perf_trace.begin_drag()
                self._on_stroke_start()
                self._touch_tile(*self._pick_tile(pos), event.modifiers())
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        # QGraphicsView's default swallows the second press of a fast pair,
        # which would drop a repeat click on a stack (or a second edit click).
        self.mousePressEvent(event)

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
            if event.button() == Qt.LeftButton and self._move_anchor is not None:
                anchor, current = self._move_anchor, self._move_current
                self._move_anchor = None
                self._move_current = None
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
                self._cancel_shape()
                if tiles:
                    self._on_shape_commit(tiles)
            return
        # b1.5's move-by-drag: the matching half of the press-time bookkeeping
        # above. Cleared unconditionally either way -- a drag is decided once,
        # here, never left pending for a later event.
        if event.button() == Qt.LeftButton and self._unit_drag_key is not None:
            key = self._unit_drag_key
            press_pos = self._unit_drag_press_pos
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            # Unconditionally, before either outcome below: the ghost is a
            # preview of a pending move, and both a committed move and a
            # below-threshold click end the gesture it was previewing.
            self._clear_unit_ghost()
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
            elif not moved:
                self._on_unit_click_release(key)
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
        """Escape clears a live measurement; in Units mode, arrow keys nudge
        the selection (b1.5) and Delete removes it (b1.6); anything bound to
        view_pan_* starts a held-key pan.

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
        # Pick from map first: arming forced Pan, so no other Escape owner is live.
        if event.key() == Qt.Key_Escape and self._unit_picker_active:
            self.on_unit_picker_cancel()
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
        # Select tool's Escape, in three ordered clauses: cancel a live MOVE,
        # else cancel an in-progress selection drag, and only clear a
        # committed region when neither is live -- the same ordering rule
        # daubED's canvas_widget.py follows, so releasing Escape mid-gesture
        # can never destroy a PREVIOUSLY committed region that gesture hadn't
        # replaced yet.
        if event.key() == Qt.Key_Escape and self._move_anchor is not None:
            self._move_anchor = None
            self._move_current = None
            self._update_region_overlay()
            return
        if event.key() == Qt.Key_Escape and self._select_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._update_region_overlay()
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

    def _scroll_by(self, dx: int, dy: int) -> None:
        """The one place that moves the scrollbars. Shared by middle-drag and
        the held-key pan, so both get setSceneRect()'s overscroll clamping and
        zoom-relative feel identically."""
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def focusOutEvent(self, event) -> None:
        """Alt+Tab mid-hold never delivers a release, so the timer would
        otherwise run forever."""
        self._pan_held.clear()
        self._stop_pan()
        super().focusOutEvent(event)

    def keyReleaseEvent(self, event) -> None:
        """Shift during a shape drag (the matching half of keyPressEvent's
        live-constrain branch), and the end of a held pan.

        X11 emits release+press pairs while a key is held, both flagged
        isAutoRepeat(), so an auto-repeat release must not end the hold. A
        release for a key that nudged, or was never held at all, falls
        through silently."""
        if not event.isAutoRepeat():
            self._release_pan_key(event.key())
        if event.key() == Qt.Key_Shift and self._shape_anchor is not None:
            self._shape_modifiers = self._shape_modifiers & ~Qt.ShiftModifier
            self._update_shape_preview()
            return
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
        if not shape_held:
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
        # The ghost only, not _unit_drag_key: Qt's implicit grab still routes
        # the eventual release back here, so a drag that merely crosses the
        # viewport edge must still be able to commit its move. What must not
        # survive is a preview drawn at a destination the cursor has left.
        self._clear_unit_ghost()
        # Defensive, same reasoning as the marquee above: cancelled outright,
        # not committed -- there is no natural release position to resolve
        # against. The committed region (if any) is untouched, exactly like
        # an Escape-cancel.
        if self._select_anchor is not None or self._move_anchor is not None:
            self._select_anchor = None
            self._select_current = None
            self._move_anchor = None
            self._move_current = None
            self._update_region_overlay()

    def center_on_tile(self, tile_x: int, tile_y: int) -> None:
        """Scroll so tile (tile_x, tile_y) is centred, in every terrain style
        (routed through _tile_polygon for exactly that reason)."""
        poly = self._tile_polygon(tile_x, tile_y)
        if poly is not None:
            self.centerOn(poly.boundingRect().center())

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
            # iso_geometry.sloped_tile_outline).
            base = self._sloped_tile_base(tile_x, tile_y)
            if base is None:
                return None
            ox, oy, corners = base
            outline_fn = (
                iso_geometry.sloped_tile_outline_coarse if coarse else iso_geometry.sloped_tile_outline
            )
            points = outline_fn(self._tile_pixels, *corners)
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
        if self._map_width is None or self._map_height is None:
            tiles = [(tile_x, tile_y)]
        else:
            tiles = brush.brush_tiles(
                tile_x, tile_y, self._brush_size, self._brush_shape, self._map_width, self._map_height
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
        if self._shape_modifiers & Qt.ShiftModifier:
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
                tx, ty, self._brush_size, self._brush_shape, self._map_width, self._map_height
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
            # Same set for preview and commit, no approximation: a wall run
            # is bounded by the drag's own longer axis, so it can never
            # reach the tile counts a filled rectangle can. No brush union
            # either -- the tool has supports_brush=False, since a wall's
            # shape is derived from tile adjacency and a dilated piece has
            # no meaning.
            return shape_tools.wall_path_tiles(x0, y0, x1, y1, width, height)
        if shape == "wall_rect":
            # Must stay above the terrain-rectangle fallthrough below, which
            # would pick up Draw Rectangle's Filled state and the brush union.
            # Same set for preview and commit: the ring is perimeter-bounded.
            return shape_tools.rect_perimeter_tiles(x0, y0, x1, y1, width, height)
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
            self._rect_filled,
            self._tool,
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
        traced edge by edge along _region_edge_walk(), so it conforms to
        Stepped columns and Sloped warping while landing ink only on the
        selection's true silhouette. Adding each perimeter tile's WHOLE
        polygon instead would stroke every seam between two adjacent boundary
        tiles, which is what the ants used to crawl along. Still O(perimeter),
        never O(area).

        The fill is a separate O(area) path over every tile in the rect --
        TODO(descape#region-select-perf): revisit if this lags on a whole-map
        Select All."""
        rect = self._region_render_rect()
        boundary_path = QPainterPath()
        fill_path = QPainterPath()
        if rect is not None:
            boundary_path = self._rect_ring_path(rect)
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
        # The items sit at the scene origin untransformed, so the path's own
        # bounds are already scene-space.
        self._region_scene_rect = boundary_path.boundingRect()
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
        self._region_ant_timer.stop()

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

    def show_trigger_overlay(self, shapes: Sequence, emphasis: tuple[str, int] | None) -> None:
        """GH #41: one trigger's areas, locations and runs, from
        trigger_geometry.TriggerShape snapshots. `emphasis` is the selected
        entry's (kind, index): it draws at full strength and every other entry
        dimmed; None draws everything at full strength. Walls are not drawn.
        Referenced units ("unit" shapes) are outlined, never filled."""
        drawn = (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_LOCATION, trigger_geometry.SHAPE_UNIT)
        self._trigger_shapes = [s for s in shapes if s.shape in drawn]
        self._trigger_emphasis = emphasis
        self._rebuild_trigger_overlay()

    def clear_trigger_overlay(self) -> None:
        """No single trigger selected, or its document went away. Idempotent."""
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

    def _rect_ring_path(self, rect: tuple[int, int, int, int]) -> QPainterPath:
        """A half-open tile rect's outer boundary as one closed ring that
        follows each border tile's own edge, so it climbs Stepped risers and
        Sloped warping. O(perimeter): _region_edge_walk visits only the border."""
        path = QPainterPath()
        started = False
        for tx, ty, side in self._region_edge_walk(*rect):
            points = self._tile_edge_points(tx, ty, side)
            if not points:
                # Defensive only (no projection snapshot yet): break the chain.
                started = False
                continue
            if not started:
                path.moveTo(points[0])
                started = True
            for point in points:
                if point != path.currentPosition():
                    path.lineTo(point)
        if started:
            path.closeSubpath()
        return path

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
        s = self.TRIGGER_LOCATION_SCALE
        return QPolygonF([centre + (p - centre) * s for p in polygon])

    def _forget_trigger_items(self) -> None:
        """After scene().clear(): the C++ items are gone, only the refs remain."""
        self._trigger_items = []

    def _rebuild_trigger_overlay(self) -> None:
        """Rebuilds every trigger item from the snapshot against the CURRENT
        projection: on a push, after set_source(), on an elevation edit and on
        a colour change. Items never touch the chunk cache."""
        scene = self.scene()
        if scene is None:
            return
        for item in self._trigger_items:
            scene.removeItem(item)
        self._trigger_items = []
        if not self._trigger_shapes:
            return
        rings = {True: QPainterPath(), False: QPainterPath()}
        units = {True: QPainterPath(), False: QPainterPath()}
        marks = {True: QPainterPath(), False: QPainterPath()}
        runs: list[tuple[QPointF, str]] = []
        by_entry: dict[tuple[str, int], list] = {}
        for shape in self._trigger_shapes:
            by_entry.setdefault(shape.entry_ref, []).append(shape)
        for ref, shapes in by_entry.items():
            strong = self._trigger_emphasis is None or ref == self._trigger_emphasis
            # A location the entry's location object overrides draws dimmed: the hint that it is not the target.
            superseded = trigger_geometry.location_superseded(shapes)
            for shape in shapes:
                if shape.shape in (trigger_geometry.SHAPE_AREA, trigger_geometry.SHAPE_UNIT):
                    # A unit's footprint is a tile rect too: the same O(perimeter) ring, outline only.
                    rect = self._trigger_area_rect(shape.coords)
                    if rect is not None:
                        target = rings if shape.shape == trigger_geometry.SHAPE_AREA else units
                        target[strong].addPath(self._rect_ring_path(rect))
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
            label = scene.addSimpleText(text)
            font = map_overlay_font(settings.get_ruler_label_font_px())
            font.setBold(True)
            label.setFont(font)
            label.setBrush(QBrush(self._trigger_run_pens[True].color()))
            label.setPen(QPen(self._ruler_label_outline, 0))
            label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            label.setZValue(self.TRIGGER_LABEL_Z)
            label.setData(0, "label")
            label.setPos(anchor)
            box = label.boundingRect()
            label.setTransform(QTransform.fromTranslate(-box.width() / 2.0, -box.height() - self.RULER_LABEL_GAP_PX))
            self._trigger_items.append(label)
        for item in self._trigger_items:
            item.setVisible(self._trigger_overlay_enabled)

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
            self._on_ruler_measured(self._ruler.measurement)

    def _clear_ruler(self) -> None:
        """Drops the measurement AND its items. Removes from the scene
        explicitly, unlike the clear_image/set_source paths where
        scene().clear() has already destroyed the C++ objects and only the
        Python-side references need forgetting."""
        self._ruler.clear()
        if self._ruler_line_item is not None:
            self.scene().removeItem(self._ruler_line_item)
            for item in self._ruler_end_items + self._ruler_glow_items:
                self.scene().removeItem(item)
            self.scene().removeItem(self._ruler_label_item)
        self._forget_ruler_items()
        self._on_ruler_changed(None)

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
        self._grid_item = None
        self._footprint_item = None
        self._stack_badge_item = None
        self._stack_groups = {}
        # Same hazard, and the pushed list goes too: File > Close leaves no
        # scenario for a stored marker to belong to.
        self._camera_marker_items = []
        self._camera_markers = []
        self._camera_marker_emphasised = None
        # Same hazard, and the anchors go too: they describe the closed map.
        self._analysis_marker_item = None
        self._analysis_markers = []
        self._analysis_focus = None
        # Same again for the trigger overlay, snapshot included.
        self._forget_trigger_items()
        self._trigger_shapes = []
        self._trigger_emphasis = None
        # Same hazard: scene().clear() destroyed the ghost's C++ object, so
        # _clear_unit_ghost()'s removeItem() on the next drag exit would raise.
        self._unit_ghost_item = None
        # Same hazard as the tick item above, and the same fix: forget the
        # destroyed items, and drop the measurement itself, since File > Close
        # leaves no map for it to refer to.
        self._ruler.clear()
        self._forget_ruler_items()
        self._on_ruler_changed(None)
        # Same reasoning for a shape drag: File > Close mid-drag leaves no
        # map for the release to commit against.
        self._shape_anchor = None
        self._shape_end = None
        self._shape_modifiers = Qt.NoModifier
        self._drag_shape = ""
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
        self._unit_select_groups = {}
        self._range_ring_item = None
        self._selection_player_colors = None
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
        self._move_anchor = None
        self._move_current = None
        self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_scene_rect = None
        self._region_ant_timer.stop()
        self._stroke_active = False
        self._stroke_touched = set()
        self._stroke_last_tile = None
        # File > Close mid-stroke drops the stroke without a release, so the
        # pulse would otherwise stay paused for the next document.
        self._sync_pulse_timer()

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
        resizeEvent), and set_source()/clear_image() (a new document).

        Also the funnel for the region ants' on-screen gate, since that is
        the same set of paths that changes what is visible. That includes
        the resize on show(), which is what starts the ants for a region set
        while the window had no real viewport yet."""
        self._sync_region_ant_timer()
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
        cache: IsoChunkCache | FlatChunkCache | SlopedChunkCache | None = None,
        elevations: np.ndarray | None = None,
        proj: iso_geometry.IsoProjection | None = None,
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
            f"style={cache.style!r} -- Flat/Stepped/Sloped must never cross-wire"
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
        # scene().clear() destroyed the rubber band's items too, and a
        # shape anchored on the old map's tiles means nothing on the new one.
        self._shape_anchor = None
        self._shape_end = None
        self._shape_modifiers = Qt.NoModifier
        self._drag_shape = ""
        # scene().clear() destroyed the unit items too. Selection is dropped
        # rather than re-resolved: set_source() means a new scenario or a
        # style switch, and neither guarantees the old key still addresses
        # anything.
        self._unit_hover_item = None
        self._unit_select_groups = {}
        self._range_ring_item = None
        # Mandatory, not tidiness: scene().clear() destroyed the C++ objects,
        # so a later removeItem() on one would raise. The pushed list itself
        # survives -- unlike the selection below, a stored camera view is
        # still exactly as meaningful after a style switch -- and the items
        # are rebuilt against the new projection at the end of this method.
        self._camera_marker_items = []
        # Same for the Map Analysis item. Its anchors follow self._region's
        # rule below: kept on a same-document redraw, dropped on reset_view.
        self._analysis_marker_item = None
        if reset_view:
            self._analysis_markers = []
            self._analysis_focus = None
        # The trigger overlay follows the same rule, rebuilt at the end of this method.
        self._forget_trigger_items()
        if reset_view:
            self._trigger_shapes = []
            self._trigger_emphasis = None
        self._unit_hover_key = None
        self._unit_select_keys = []
        self._marquee_item = None
        self._marquee_start_pos = None
        # The ghost went with it too, and the drag behind it means nothing
        # against a new scenario -- dropped, not re-anchored, so the release
        # commits nothing (see keyPressEvent's Escape branch for the other
        # place that pairing matters).
        self._unit_ghost_item = None
        self._unit_drag_key = None
        self._unit_drag_press_pos = None
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
        self._move_anchor = None
        self._move_current = None
        if reset_view:
            self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_scene_rect = None
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

        # Same always-built-then-hidden shape as the ticks. No setZValue: at
        # the default 0.0, insertion order keeps the lazily created brush
        # highlight, pan outline and marquee above the grid. Only the
        # follow-off lattice and the slider preview use it; otherwise the grid
        # is baked into the fresh cache here, before its first paint.
        self._grid_item = GridItem(
            self._map_width,
            self._map_height,
            self._map_rect,
            proj=iso_proj,
            tile_px=tile_pixels,
            blend=self._grid_blend,
            thickness=self._grid_thickness,
        )
        self.scene().addItem(self._grid_item)
        self._apply_grid()

        self._footprint_item = self.scene().addPath(QPainterPath(), self._footprint_pen)
        self._footprint_item.setZValue(self.FOOTPRINT_Z)
        self._footprint_item.setVisible(self._footprint_enabled)

        # Empty until the caller installs an index via set_unit_index().
        tile_extent = 2 * proj.half_w if iso_proj is not None else tile_pixels
        self._stack_groups = {}
        self._stack_badge_item = StackBadgeItem(tile_extent, QColor(settings.get_overlay_color("unit_stack")))
        self._stack_badge_item.setZValue(self.UNIT_STACK_Z)
        self._stack_badge_item.setVisible(False)
        self.scene().addItem(self._stack_badge_item)

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
        # Same rebuild-now-that-the-projection-exists reasoning, for the same
        # reason the markers are not dropped above: a camera view is a
        # property of the scenario, not of this render.
        self._rebuild_camera_markers()
        self._rebuild_analysis_markers()
        self._rebuild_trigger_overlay()
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
        return grid_overlay.grid_bake(True, self._grid_blend, self._grid_thickness)

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

    def set_footprint_outlines(self, enabled: bool) -> None:
        """Shows or hides View > Footprint Outlines. Like set_edge_ticks,
        this evicts no chunk cache: the outlines are a scene item, never
        baked into canvas pixels."""
        self._footprint_enabled = enabled
        if self._footprint_item is not None:
            self._footprint_item.setVisible(enabled)
        self.refresh_footprint_overlay()

    def set_footprint_scope(self, scope: str) -> None:
        self._footprint_scope = scope
        self.refresh_footprint_overlay()

    def refresh_footprint_overlay(self) -> None:
        """Rebuilds the outline path from the live index and height field.

        Runs even while the overlay is hidden, unlike refresh_unit_highlight,
        which may skip elevation edits because terrain edits are only
        reachable in Terrain mode and leaving Units mode drops the selection.
        An always-on overlay voids that argument: it is live in Terrain mode
        while the elevations under it change."""
        if self._footprint_item is None:
            return
        path = QPainterPath()
        if self._unit_index is not None:
            for entry in unit_pick.footprint_entries(self._unit_index, self._footprint_scope):
                polygons = self._unit_polygons_for(entry)
                if polygons:
                    _add_closed_polygons(path, polygons)
        self._footprint_item.setPath(path)

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
        """A tile's top vertex, as _rebuild_stack_badges() anchors badges."""
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

    def viewport_centre_tile(self) -> tuple[int, int] | None:
        """The tile at the centre of what is on screen, for Players mode's
        Set View button. None when that centre is not on the map.

        Flat's own division is always defined (it does no bounds check at
        all), so the bounds check happens here. Stepped and Sloped return
        None off-map and, in Stepped, on a skirt face -- so those fall back
        to the continuous _pick_map_point() solve before giving up."""
        if self._map_width is None or self._map_height is None:
            return None
        pos = self.mapToScene(self.viewport().rect().center())
        tile = self._pick_tile(pos)
        if tile is None:
            point = self._pick_map_point(pos)
            if point is None:
                return None
            tile = (math.floor(point[0]), math.floor(point[1]))
        x, y = tile
        if not (0 <= x < self._map_width and 0 <= y < self._map_height):
            return None
        return x, y

    def refresh_elevation_overlays(self) -> None:
        """The overlays whose geometry follows the terrain, for an elevation
        edit to call once per touched-tile batch.

        The grid needs nothing here: it is baked into the chunks, and the
        edit's own cache.patch() already recomposited them. The footprint
        path is built eagerly, and on the largest example file at All units
        in Sloped that is ~1s. So this defers it to the event loop: one
        8-step Set Elevation drag touches 131 tiles per _apply_dirty's own
        comment, and rebuilding per touch would be unusable."""
        self.schedule_footprint_refresh()
        # Not deferred: at most 8 items, each one _ruler_anchor lookup, and
        # unlike the footprint path there is no per-unit walk behind it.
        self._rebuild_camera_markers()
        # Not deferred either: at most ~1800 _tile_polygon() calls.
        self._rebuild_analysis_markers()
        # _tile_polygon reads the elevations, so Stepped and Sloped rings move. O(perimeter).
        self._rebuild_trigger_overlay()

    def schedule_footprint_refresh(self) -> None:
        if self._footprint_refresh_pending or self._footprint_item is None:
            return
        self._footprint_refresh_pending = True
        QTimer.singleShot(0, self._flush_footprint_refresh)

    def _flush_footprint_refresh(self) -> None:
        self._footprint_refresh_pending = False
        # scene().clear() can have destroyed the item between the schedule
        # and this call (File > Close, a style switch), so re-check.
        if self._footprint_item is not None:
            self.refresh_footprint_overlay()

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

    def _mip_for_zoom_factor(self, factor: float) -> int | None:
        """The mip level a paint would select if this view were scaled by
        `factor` right now; None with no canvas item.

        Same reading as viewport_chunk_target(), and exact rather than
        approximate for a uniform factor: scaling a transform by `s` scales
        its singular values by `s`, which is the same argument that method
        already makes for folding in devicePixelRatioF()."""
        item = self._canvas_item
        if item is None:
            return None
        device_transform = item.deviceTransform(self.viewportTransform())
        scale = _max_axis_scale(device_transform) * self.devicePixelRatioF()
        return item._cache.mip_for_scale(scale * factor)

    def _end_wheel_gesture(self) -> None:
        """Drops the sub-notch residual, the running direction and the
        current gesture's mip budget, so the next wheel event starts a
        gesture of its own."""
        self._wheel_accum = 0
        self._wheel_dir = 0
        self._wheel_gesture_mip = None
        self._wheel_last_t = 0.0

    def _trim_wheel_notches(self, notches: int) -> int:
        """Reduces a pending notch count until the zoom it asks for is inside
        the floor/ceiling AND at most one mip level from where the gesture
        started. 0 means the whole event is refused, which for a single
        over-the-ceiling notch is exactly the refusal this has always been.
        The arithmetic at |notches| == 1 is the pre-accumulation guard
        unchanged."""
        step = 1.25 if notches > 0 else 0.8
        current = abs(self.transform().determinant()) ** 0.5
        while notches != 0:
            factor = step ** abs(notches)
            target = current * factor
            too_small = notches < 0 and self._min_linear_scale is not None and target < self._min_linear_scale
            too_big = notches > 0 and self._max_linear_scale is not None and target > self._max_linear_scale
            if not (too_small or too_big or self._crosses_wheel_mip_budget(factor)):
                return notches
            notches += -1 if notches > 0 else 1
        return 0

    def _crosses_wheel_mip_budget(self, factor: float) -> bool:
        """Whether zooming by `factor` would land more than one mip level
        from where this gesture started. False with no canvas item (nothing
        selects a level) and on a single-level ladder, where mip_for_scale()
        clamps every scale to the same answer."""
        start = self._wheel_gesture_mip
        if start is None:
            return False
        mip = self._mip_for_zoom_factor(factor)
        return mip is not None and abs(mip - start) > 1

    def wheelEvent(self, event):
        """Zooms by whole WHEEL_NOTCH_UNITS notches, at most one mip level
        per gesture.

        Accumulating angleDelta rather than taking every event as a full step
        is what keeps a high-resolution wheel or a trackpad, which sends many
        small deltas, to one zoom step per real detent instead of one per
        event, and each step is a full-viewport repaint. A coarse wheel
        sends exactly one notch per detent, so it behaves as it always has,
        including the ceiling/floor refusal.

        A GESTURE is a run of same-direction events with no
        WHEEL_GESTURE_GAP_S pause between them: one flick of the wheel.
        Notches past the first mip boundary in a gesture are dropped rather
        than banked, so a hard flick lands one level away instead of two.
        Two would land outside level_warm.neighbour_mips()'s +/-1 warm set
        and pay a synchronous level build inside paint(). Pausing (or
        reversing) starts a fresh gesture with a fresh budget."""
        delta = event.angleDelta().y()
        if delta == 0:
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
        notches = self._trim_wheel_notches(notches)
        if notches == 0:
            return
        factor = 1.25**notches if notches > 0 else 0.8 ** -notches
        self.scale(factor, factor)
        self._note_viewport_changed()
        self._on_zoom_changed()

    def mouseMoveEvent(self, event):
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
        if self._tool == TOOL_SELECT:
            self._clear_highlight()
            self._clear_pan_highlight()
            self._clear_unit_hover()
            if self._move_anchor is not None and tile is not None:
                # Same "hold the last valid tile" convention as the select
                # drag below -- a move legitimately runs off the map edge and
                # back, and the block is re-pasted at an unclipped anchor.
                self._move_current = tile
                self._update_region_overlay()
            elif self._select_anchor is not None and tile is not None:
                # Same "hold the last valid tile" convention as the Ruler
                # above: an off-map move mid-drag must not collapse the
                # preview, since the drag legitimately runs off the edge.
                self._select_current = tile
                self._update_region_overlay()
            # Advertises the gesture. _apply_tool_cursor() stays the
            # tool/mode-level default and is deliberately not touched.
            if self._move_anchor is not None or (self._region_movable and self._tile_in_region(tile)):
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.CrossCursor)
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
                # crossing, so it is cleared here the way every sibling branch
                # clears it. Before the mid-drag preview this branch returned
                # without clearing, which froze the cyan outline over the
                # source unit for the whole drag.
                self._clear_unit_hover()
                press_pos = self._unit_drag_press_pos
                # Only once the drag is real -- the same threshold
                # mouseReleaseEvent uses to tell a click from a drag. A ghost
                # appearing on 1px of jitter during a plain selection click
                # would read as a flicker bug.
                if (
                    press_pos is not None
                    and (event.pos() - press_pos).manhattanLength() > self.UNIT_DRAG_THRESHOLD_PX
                    and (self._pos_on_map(pos) or self._is_group_drag(self._unit_drag_key))
                ):
                    self._on_unit_drag_preview(self._unit_drag_key, pos, event.modifiers())
                else:
                    self._clear_unit_ghost()
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
            # This same pixel's tile, resolved at the top of this handler;
            # nothing since has touched the geometry it came from.
            entry = self.pick_unit_at(pos, tile) if in_canvas else None
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
