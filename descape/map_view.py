"""MapView: the QGraphicsView the map is panned, zoomed and edited in.

Dumb the same way TriggerPanel is -- it reports what the user did
through the callbacks ViewerWindow constructs it with and never
reaches back up into the window.

MapView is this module's class plus six mixins, one module per area:

- map_view.py: the class constants (the Z-order ladder among them),
  __init__, tool and brush config, the mode and tool switches, cursors,
  the colour grade, set_source/clear_image and region invalidation.
- map_view_input.py: mouse, wheel and key handlers, tile picking, stroke
  bookkeeping and key panning.
- map_view_units.py: the unit index, stack badges, hover, selection, range
  rings, marquee, drag, ghosts and footprint outlines.
- map_view_overlays.py: overlay ink, region select, the mirror overlay, the
  ruler, grid and edge ticks, camera and analysis markers.
- map_view_viewport.py: zoom anchor and stops, preframes, chunk targets,
  view-state capture and restore, resize and scroll.
- map_view_triggers.py: the trigger unit and tile pickers, area and location
  edit, the trigger overlay, its ghosts, ring paths and run labels.
- map_view_highlight.py: cliff and place ghosts, tile geometry, the brush
  highlight, shape preview and pan highlight.

Rules of the split (tests/test_map_view_split.py checks them):

- A mixin defines no __init__ and no signal: all instance state is created
  in MapView.__init__.
- No name is defined by two of the seven classes.
- The mixins precede QGraphicsView in MapView's bases, and no mixin module
  imports this one.

A branch cut before the split is ported with tools/split_class.py's
`split --like`."""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import numpy as np
from PyQt5.QtCore import QElapsedTimer, QEvent, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QPainter,
    QPen,
    QPolygonF,
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
    colour_mood,
    iso_geometry,
    outline_chunks,
    ruler,
    settings,
    unit_shape,
    zoom_preframe,
)
from descape.instruction_overlay import InstructionOverlay
from descape.map_view_highlight import _HOVER_GHOST_TOOLS, _MapViewHighlightMixin
from descape.map_view_input import _MapViewInputMixin
from descape.map_view_overlays import _MapViewOverlaysMixin
from descape.map_view_triggers import _MapViewTriggersMixin, _RingEntry
from descape.map_view_units import _MapViewUnitsMixin
from descape.map_view_viewport import _MapViewViewportMixin
from descape.render_cache import (
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
)
from descape.render_paint import SMALL_MAP_TILE_PIXELS

# The two non-Flat terrain styles, which share a projected pick plane and a
# diamond ground outline. Aliased to this module's existing private name so
# its use sites read unchanged. The shared home is terrain_style.py, which is
# Qt-free -- the render modules import no PyQt5 at all, so viewer_common.py (which
# does) could never have been it.
from descape.terrain_style import ELEVATED_STYLES as _ELEVATED_STYLES
from descape.viewer_canvas import (
    AnalysisMarkerItem,
    CameraMarkerItem,
    ChunkedPathItem,
    EdgeTickItem,
    GridItem,
    MapCanvasItem,
    StackBadgeItem,
    UnitGhostItem,
)
from descape.viewer_common import (
    EDIT_TOOLS,
    TOOL_EYEDROPPER,
    TOOL_RULER,
    TOOL_SELECT,
)


class MapView(
    _MapViewInputMixin, _MapViewUnitsMixin, _MapViewOverlaysMixin,
    _MapViewTriggersMixin, _MapViewViewportMixin, _MapViewHighlightMixin,
    QGraphicsView,
):
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
    # GH #172: the wheel's rotate binding turns the Place Unit ghost clockwise on wheel up (+1, like "."). Its feel is a human check.
    PLACE_WHEEL_DIRECTION = 1
    # Settings > Keybinds' wheel rows: each settings.WHEEL_SLOTS slot's exact modifier set. Matched
    # under the mask, so KeypadModifier cannot break the lookup (the reason _pan_modifiers exists).
    _WHEEL_SLOT_MODIFIERS: ClassVar[dict[str, int]] = {
        "plain": int(Qt.NoModifier),
        "ctrl": int(Qt.ControlModifier),
        "shift": int(Qt.ShiftModifier),
        "alt": int(Qt.AltModifier),
    }
    _WHEEL_MODIFIER_MASK = int(Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier | Qt.MetaModifier)
    # Viewport pixels one wheel notch moves the view under a pan_vertical/pan_horizontal binding.
    WHEEL_PAN_NOTCH_PX = 120
    # Idle gap that ends a wheel GESTURE, so the next event starts a fresh
    # one with a fresh +/-1 mip budget; also the minimum time between two
    # mip crossings in one continuous roll. PROVISIONAL, in the same sense as
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

    # Mouse-move coalescing (GH #179, see viewportEvent): the most a stashed
    # move waits for a paint before it is handled anyway. Above one 60 Hz
    # frame so a real paint always wins; short enough that a hover that
    # paints nothing still tracks at ~40 Hz.
    MOVE_PAINT_WAIT_MS = 25
    # Viewport events that handle a pending move before their own dispatch.
    _MOVE_FLUSH_FIRST_EVENTS = frozenset(
        {
            QEvent.MouseButtonPress,
            QEvent.MouseButtonRelease,
            QEvent.MouseButtonDblClick,
            QEvent.Wheel,
            QEvent.Leave,
        }
    )
    # Class-level so a viewportEvent fired during QGraphicsView.__init__,
    # before the timers exist, falls straight through to Qt.
    _move_flush_timer: QTimer | None = None
    _move_fallback_timer: QTimer | None = None
    _move_stash: tuple | None = None
    _move_gate_open = True
    # Referenced from viewer_menus.py as MapView.PAN_DIRECTIONS (that module does
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
    # Settings' "unit_select" opacity (GH #129) multiplies the whole trio on top.
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
    # View > Layers > Fish Outlines (TASK-499): just above the footprint layer,
    # so where both are on the fish ring is the one seen. Colour is "fish_outline".
    FISH_OUTLINE_Z = 5.5
    # Cosmetic device pixels, so the ring stays readable over water art at any zoom.
    FISH_OUTLINE_PEN_WIDTH = 2

    # View > Range Rings. Above the footprint layer and below the hover cue,
    # so the two cues that say "this one, right now" both read over it: the
    # ring is ambient context about an already-selected building. Colour is
    # settings.OVERLAY_COLORS' "range_ring", rebuilt by _rebuild_overlay_ink().
    RANGE_RING_Z = 9.0
    # GH #49 slice 2's trigger rings: the ring pen dashed, in device pixels since cosmetic.
    RANGE_RING_TRIGGER_DASH: ClassVar[list[float]] = [6.0, 4.0]

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
    # Pinned rulers (GH #108/#109): just under the live pair, so a pending
    # measurement reads over finished ones. Static, no glow.
    PINNED_RULER_Z = 11.9
    PINNED_RULER_LABEL_Z = 12.9

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
    # Settings' "region" opacity (GH #129) multiplies fill, outline and ants on top.
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
    # The fish ring round a place ghost (GH #281): over the translucent body,
    # below UNIT_STACK_Z for the ghost's own reason. Full opacity, _fish_pen.
    GHOST_FISH_Z = 11.57

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
    # Fixed alphas; Settings' "trigger_area" opacity (GH #129) multiplies the area items only.
    TRIGGER_FILL_ALPHA = 55
    TRIGGER_DIM_FILL_ALPHA = 20
    TRIGGER_DIM_PEN_ALPHA = 110
    # A location mark is its tile's own polygon shrunk to this fraction.
    TRIGGER_LOCATION_SCALE = 0.6
    # GH #167's trigger-created unit ghosts: between the area fill and the location
    # marks, so a mark's outline stays readable over a ghost's feet.
    TRIGGER_GHOST_Z = 11.35
    TRIGGER_GHOST_DIM_OPACITY = 0.3
    # GH #229: the edge and corner grab bands' screen width.
    AREA_EDIT_BAND_PX = 6
    # Each grab zone's drag direction in map axes, for its on-screen cursor.
    _AREA_ZONE_AXES: ClassVar[dict[str, tuple[int, int]]] = {
        "w": (1, 0), "e": (1, 0), "n": (0, 1), "s": (0, 1),
        "nw": (1, 1), "se": (1, 1), "ne": (1, -1), "sw": (1, -1),
    }
    # A screen direction in 45 degree steps counter-clockwise from the x axis.
    _SIZE_CURSORS = (Qt.SizeHorCursor, Qt.SizeBDiagCursor, Qt.SizeVerCursor, Qt.SizeFDiagCursor)

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
        # GH #221: file drops go up to ViewerWindow; the scene would otherwise swallow them.
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
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
        # Kept across set_source(): the viewer sets it before the new item's first paint.
        self._colour_grade: colour_mood.MoodGrade | None = None
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
        self._grid_major_every = settings.get_grid_major_every()
        self._grid_follow_elevation = settings.get_grid_follow_elevation()
        # A Settings > Appearance slider drag in progress: see begin_grid_preview().
        self._grid_previewing = False
        # View > Footprint Outlines, same lifetime again. One ChunkedPathItem per colour group
        # (_owner_group_key); the None group exists exactly while a map is loaded.
        self._footprint_items: dict = {}
        # Which chunks of each outline overlay an edit made stale (TASK-031.42).
        self._footprint_chunks = outline_chunks.OutlineChunks()
        self._fish_chunks = outline_chunks.OutlineChunks()
        self._footprint_enabled = settings.get_footprint_outlines()
        self._footprint_scope = settings.get_footprint_scope()
        self._footprint_merged = settings.get_footprint_merged()
        self._footprint_by_owner = settings.get_footprint_by_owner()
        self._footprint_refresh_pending = False
        # View > Layers > Fish Outlines: one path item while a map is loaded; ViewerWindow pushes the flag.
        self._fish_item: ChunkedPathItem | None = None
        self._fish_enabled = False
        # Stacked-unit badges: groups recomputed on every index swap, item
        # rebuilt per set_source() like _edge_tick_item.
        self._stack_badge_item: StackBadgeItem | None = None
        self._stack_badges_enabled = settings.get_stack_badges()
        self._stack_badge_position = settings.get_stack_badge_position()
        self._stack_groups: dict = {}
        # Anchors skipped while hidden; rebuilt when the layer next shows.
        self._stack_badges_stale = False
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
        self._trigger_emphasis: frozenset[tuple[str, int]] | None = None
        self._trigger_items: list[QGraphicsItem] = []
        self._trigger_ring_memo: dict[tuple[int, int, int, int], _RingEntry] = {}
        self._trigger_overlay_enabled = settings.get_trigger_overlay()
        # GH #167: the trigger_geometry.CreatedUnit records the selected triggers would place,
        # kept like the shapes; the two ghost items (strong, dim) go with every scene().clear().
        self._trigger_ghost_records: list = []
        self._trigger_ghost_emphasis: frozenset[tuple[str, int]] | None = None
        self._trigger_ghost_sprites = True
        self._trigger_ghost_items: list[UnitGhostItem] = []
        self._trigger_ghosts_enabled = settings.get_trigger_ghosts()
        self._instruction_overlay = InstructionOverlay(self)  # TASK-550, a child of this view and never of viewport()
        # {(record.payload, sprites): (draws, mark)}, dropped wherever the look can change.
        self._trigger_ghost_parts: dict[tuple, tuple[list, tuple | None]] = {}
        self._trigger_ghost_refresh_pending = False
        # An elevation edit landed mid-stroke: one rebuild at stroke end.
        self._trigger_ghosts_stale = False
        # The window assigns this ((record, sprites) -> (draws, mark)): MapView holds no scenario.
        self.resolve_trigger_ghost: Callable[[object, bool], tuple[list, tuple | None]] = lambda _r, _s: ([], None)
        # Called once per rebuild, before its first resolve: the window's one depoison() (batch_adds()'s rule).
        self.prepare_trigger_ghost_resolve: Callable[[], None] = lambda: None
        # The Ruler's grammar lives in the Qt-free descape.ruler; this
        # class only turns its two tiles into scene items. Like the tick
        # state above it sits on MapView, which outlives every set_source().
        self._ruler = ruler.RulerSession()
        # Fired once per COMPLETED measurement, never per drag frame; see
        # _report_ruler for the edge it triggers on.
        self._on_ruler_measured = on_ruler_measured
        # Fired on every live update (including mid-drag) and on every clear
        # (with the newest pinned ruler, or None), unlike on_ruler_measured
        # above: see _emit_ruler_status for the rule.
        self._on_ruler_changed = on_ruler_changed
        # GH #108/#109: finished measurements stay on the map until removed.
        self._pinned_rulers = ruler.PinnedRulers()
        self._pinned_ruler_items: list[QGraphicsItem] = []
        # Fired with the pinned count on every change. Assigned by the window, like on_unit_pick.
        self.on_pinned_rulers_changed = lambda count: None
        # GH #108: (label, added, removed) as (index, measurement) pairs, for a user
        # add, remove or Clear rulers; never for undo or a reset.
        self.on_pinned_rulers_edited = lambda label, added, removed: None
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
        # GH #138's Set Location ("point") / Set Area ("rect") pick, or None: a tile,
        # a normalised inclusive rect, and a cancel. Assigned by the window.
        self.on_tile_pick = lambda tile: None
        self.on_rect_pick = lambda x1, y1, x2, y2: None
        self.on_tile_picker_cancel = lambda: None
        self._tile_picker_kind: str | None = None
        self._tile_pick_anchor: tuple[int, int] | None = None
        # Its own scene item, not the trigger overlay, which the View toggle hides.
        self._tile_pick_item: QGraphicsPathItem | None = None
        # A location drag's live distance, shown with the item above.
        self._tile_pick_label: QGraphicsSimpleTextItem | None = None
        # GH #229's Edit Area: the one area the mouse may move or resize (pushed by
        # the window), the zone under the cursor, a live drag's (zone, start tile,
        # coords) and its last tile, and the release's write. The drag previews
        # through the pick's item above; the two are never live together.
        self._area_edit: tuple[int, int, int, int] | None = None
        # Its Move, Resize and Tile borders options: which zones grab, and how wide.
        self._area_move = True
        self._area_resize = True
        self._area_tile_borders = False
        self._area_hover: str | None = None
        self._area_drag: tuple[str, tuple[int, int], tuple[int, int, int, int]] | None = None
        self._area_drag_end: tuple[int, int] | None = None
        self.on_area_edit = lambda x1, y1, x2, y2: None
        # GH #41: the form's entry's location mark, one more zone of that drag, and
        # the run origin its preview measures from. Its own release callback.
        self._location_edit: tuple[int, int] | None = None
        self._location_run_origin: tuple[int, int] | None = None
        self.on_location_edit = lambda x, y: None
        # (cover tile, stack index) of the last pick, so a repeat click walks the stack.
        self._picker_cycle: tuple[tuple[int, int], int] | None = None
        # The Eyedropper's Units-mode pick: on_unit_eyedrop(entry, cover_tile, modifiers),
        # assigned by the window. Its own stack position, never shared with Pick from map's.
        self.on_unit_eyedrop = lambda entry, cover_tile, modifiers: None
        self._eyedrop_cycle: tuple[tuple[int, int], int] | None = None
        # A sub-threshold left click under Pan in Units mode: on_pan_click(scene_pos),
        # assigned by the window for its one-time hint; the press's device pos.
        self.on_pan_click = lambda pos: None
        self._pan_click_pos: QPointF | None = None
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
        # The live drag's shape ("line" | "rect" | "wall" | "unit_line" | "unit_shape"), latched at press
        # so a mid-drag catalog change can't switch it. "" when no drag.
        self._drag_shape = ""
        # GH #98: returns "wall" when Place Unit should drag a wall run. A
        # setter rather than a constructor callable, which would grow the
        # already long injected list (a known merge-collision point).
        self._place_shape_query: Callable[[], str] = lambda: ""
        # GH #124: tiles a wall drag must leave empty, from set_place_blocked_query().
        # Latched at press for a "wall" drag or a wall-branch latch below; empty otherwise.
        self._place_blocked_query: Callable[[], frozenset] = frozenset
        self._drag_blocked: frozenset = frozenset()
        # GH #365: Unit Line / Unit Shape's press query, from set_shape_press_query(),
        # and the latch it returned for the live drag, handed to on_shape_commit().
        # Only MapView keeps it, so a cancelled drag leaves nothing behind.
        self._shape_press_query: Callable[[], unit_shape.ShapeLatch | None] = lambda: None
        self._shape_latch: unit_shape.ShapeLatch | None = None
        # GH #128: the hover ghost's resolver, set by set_place_preview(). _place_hover is
        # the last (scene pos, modifiers) the tool hovered on the map, for refreshes.
        self._on_place_preview: Callable[[QPointF, object], None] = lambda pos, modifiers: None
        self._place_hover: tuple[QPointF, object] | None = None
        # GH #172: the wheel's rotate binding turns that ghost. Set by set_place_ghost_query()/set_place_rotate();
        # _place_wheel_accum is the sub-notch delta, dropped whenever the ghost goes.
        self._place_ghost_query: Callable[[], bool] = lambda: False
        self._on_place_rotate: Callable[[int], None] = lambda steps: None
        self._place_wheel_accum = 0
        # The Cliff tool's hover ghost, set by set_cliff_preview(): fn(tile) or fn(None) off the map.
        self._on_cliff_preview: Callable[[tuple[int, int] | None], None] = lambda tile: None
        # Its highlight's tiles in place of the brush diamond, see set_cliff_footprint().
        self._cliff_footprint: tuple[tuple[int, int], ...] | None = None
        self._cliff_hover = None
        # The Move tool's press and GH #75's group test, set by set_unit_drag_hooks() for the same reason.
        self._on_move_press: Callable = lambda pos, modifiers: None
        self._is_group_drag: Callable[[tuple[int, int]], bool] = lambda key: False
        # TASK-325: True when Escape cleared a unit selection, set by set_unit_escape_hook().
        self._on_unit_escape: Callable[[], bool] | None = None
        # TASK-256: Enter applies and Escape reverts a live flip or turn, set by set_lift_hooks().
        self._on_lift_apply: Callable[[], bool] | None = None
        self._on_lift_revert: Callable[[], bool] | None = None
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
        # on_unit_move(key, scene_pos, modifiers) fires when a Move-tool
        # press-drag-release on a unit moves past UNIT_DRAG_THRESHOLD_PX
        # (see mousePressEvent/mouseReleaseEvent) -- key is the (player_id,
        # reference_id) pair on_move_press chose at press time, resolved late
        # by ViewerWindow exactly like on_click_select's own selection key.
        self._on_unit_place = on_unit_place
        self._on_unit_move = on_unit_move
        # b1.5/b1.6, the ninth and tenth injected callables.
        # on_unit_nudge(dx_tiles, dy_tiles, modifiers) fires on an arrow key
        # in Units mode; on_unit_delete(modifiers) fires on Delete. Both are
        # keyboard-only (no MapView state to track), unlike place/move above.
        self._on_unit_nudge = on_unit_nudge
        self._on_unit_delete = on_unit_delete
        # b2.3, the eleventh injected callable. on_marquee_select(keys,
        # modifiers) fires once a completed marquee drag (a Select-tool
        # press-drag-release in Units mode that engaged, see
        # _marquee_engaged()) is resolved to units. keys are the
        # (player_id, reference_id) pairs of every unit _units_in_scene_rect()
        # found, already filter-respecting. A drag that never engages is a
        # plain click, already handled by the on_click_select() call
        # mousePressEvent makes unconditionally at press time.
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
        # A place ghost's fish ring (GH #281); goes wherever the ghost goes.
        self._ghost_fish_item: QGraphicsPathItem | None = None
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
        # The rect the overlay last drew, for the live-drag memo (_refresh_live_region()).
        self._region_overlay_rect: tuple[int, int, int, int] | None = None
        # The open Perf Trace drag's label ("select" or "select-move"), None with none open.
        self._region_drag_label: str | None = None
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
        # GH #129: a Settings slider's unpersisted percent, per opacity group.
        self._overlay_opacity_overrides: dict[str, int] = {}
        # Each player's sprite tint (TASK-115), pushed by ViewerWindow (MapView holds no scenario):
        # the owner colour of the selection, footprint outlines and range rings (TASK-449).
        self._team_colors: tuple | None = None
        self._selection_by_owner = settings.get_selection_by_owner()
        # One item per colour group (_owner_group_key) for the selected buildings'
        # rings, created lazily on the first draw like the selection groups above.
        self._range_ring_items: dict = {}
        self._range_rings_enabled = settings.get_range_rings()
        self._range_rings_by_owner = settings.get_range_rings_by_owner()
        self._range_rings_selected_revealers = settings.get_range_rings_selected_revealers()
        # GH #49 slice 2: the dashed trigger rings, keyed like _range_ring_items. The
        # viewer pushes the effects (set_trigger_range_effects); the index is built once per push.
        self._trigger_ring_items: dict = {}
        self._trigger_rings_enabled = settings.get_range_rings_triggers()
        self._trigger_range_effects = None
        self._trigger_range_index = None
        # GH #153 / #155: the always-on ring layer, one item per colour group like the
        # selection's rings above, but walking the unit index rather than the selection.
        self._ring_layer_items: dict = {}
        # (radius, cx, cy, rise_px) -> QPolygonF, valid for _ring_layer_projection only. See refresh_ring_layer().
        self._ring_layer_polygons: dict = {}
        self._ring_layer_projection = None
        self._ring_layer_revealers = settings.get_range_rings_revealers()
        self._ring_layer_buildings = settings.get_range_rings_buildings()
        # Memoized on the picked (player_id, reference_id) key, the same way
        # _highlight_key memoizes the tile highlight: mouseMoveEvent runs a
        # pick on every pixel, and rebuilding a 25-diamond path per pixel
        # would be pure waste.
        self._unit_hover_key: tuple[int, int] | None = None
        # b2.2 generalizes this from a single key to a list -- see
        # set_unit_selection()'s own docstring for why MapView keeps a copy
        # at all rather than only ever being pushed one by ViewerWindow.
        self._unit_select_keys: list[tuple[int, int]] = []
        # b1.5's move-by-drag: the unit key a Move-tool press chose (if any),
        # and the press position, both set in the Units-mode branch and
        # consumed (cleared either way) by the matching mouseReleaseEvent --
        # see UNIT_DRAG_THRESHOLD_PX for why a press+release with no real
        # drag is a plain click, not a move.
        self._unit_drag_key: tuple[int, int] | None = None
        self._unit_drag_press_pos: QPointF | None = None
        # GH #280: (viewport pos, modifiers) of the drag's last move, once it passed the threshold.
        # Set from then on, wherever the cursor goes, so a key mid-drag turns it (unit_drag_live()).
        self._unit_drag_hover: tuple | None = None
        # b2.3's marquee: set by every Select-tool press in Units mode, on a
        # unit or not (a Move press arms the drag above instead, never both).
        # Holds the PRESS-TIME device position; _marquee_engaged() decides
        # click vs. rectangle, then mouseReleaseEvent converts start/end into
        # a scene rect for _units_in_scene_rect(). _marquee_press_tile is the
        # tile under a press that hit a unit, else None; _marquee_press_on_unit
        # tells a hit over an unresolved pixel (a Stepped skirt) from a miss.
        self._marquee_start_pos: QPointF | None = None
        self._marquee_press_tile: tuple[int, int] | None = None
        self._marquee_press_on_unit = False
        self._marquee_item: QGraphicsRectItem | None = None
        self._stroke_active = False
        # True while this view's stroke holds gc disabled (_pause_gc_for_stroke).
        self._stroke_gc_paused = False
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
        self._brush_thickness = brush.BRUSH_THICKNESS_MIN
        # Set by set_source() alongside each render, cleared by clear_image()
        # -- needed to clip the brush footprint the same way
        # ViewerWindow.on_edit_stroke_tile clips a single tile today.
        self._map_width: int | None = None
        self._map_height: int | None = None
        # Memoizes the last highlight built, so mouseMoveEvent's per-pixel
        # calls into _update_highlight() don't rebuild an up-to-81-tile
        # QPainterPath every time the mouse moves within the same tile. The
        # Cliff tool keys its footprint instead: ("cliff", tiles).
        self._highlight_key: tuple[int, int, int, str, int] | tuple[str, tuple[tuple[int, int], ...]] | None = None
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
        # and IsoProjection descape.render_iso.elevations_and_proj()/
        # sloped_elevations_and_proj() computed for the current source from
        # (None in Flat, which has neither) -- Risk #6: these must never
        # drift from what's actually on screen, so they're only ever set
        # together with the source itself, in set_source(), never patched
        # independently.
        self._terrain_style = "stepped"
        self._iso_elevations: np.ndarray | None = None
        self._iso_proj: iso_geometry.IsoProjection | None = None
        self._tool = "pan"
        # The tool the last press ran under, so a double-click can tell a switch happened in between.
        self._press_tool: str | None = None
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
        # Move coalescing (viewportEvent): the next-turn flush and the no-paint fallback.
        self._move_flush_timer = QTimer(self)
        self._move_flush_timer.setSingleShot(True)
        self._move_flush_timer.setInterval(0)
        self._move_flush_timer.timeout.connect(self._flush_pending_move)
        self._move_fallback_timer = QTimer(self)
        self._move_fallback_timer.setSingleShot(True)
        self._move_fallback_timer.setInterval(self.MOVE_PAINT_WAIT_MS)
        self._move_fallback_timer.timeout.connect(self._on_move_fallback)
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
        # The viewport QPoint the last wheel zoom anchored at, or None (zoom_anchor_scene()).
        self._last_zoom_anchor = None
        # Pre-rendered frames for the next zoom notch (TASK-031.71): opted into per window
        # (set_preframes_allowed()), rendered SETTLE_MS after the view last moved, nearest the
        # last zoom's direction first. The twin is built on first use.
        self._preframes_allowed = False
        self._preframe_warmer = zoom_preframe.PreframeWarmer()
        self._preframe_twin: zoom_preframe.ZoomTwin | None = None
        self._preframe_dir = 1
        self._preframe_timer = QTimer(self)
        self._preframe_timer.setSingleShot(True)
        self._preframe_timer.setInterval(zoom_preframe.SETTLE_MS)
        self._preframe_timer.timeout.connect(self._on_preframe_settle)
        # Wheel-gesture state, all reset together by _end_wheel_gesture():
        # leftover sub-notch angleDelta, the direction the gesture is
        # running in, the mip its budget is anchored on (None while idle),
        # when the last wheel event arrived, and when that budget was anchored.
        self._wheel_accum = 0
        self._wheel_dir = 0
        self._wheel_gesture_mip: int | None = None
        self._wheel_last_t = 0.0
        self._wheel_budget_t = 0.0
        # (key, points) of the last _wheel_lattice(), keyed on the ladder and the floor/ceiling.
        self._wheel_lattice_cache: tuple[tuple, list[float]] | None = None
        self.set_zoom_anchor_mode(settings.get_zoom_centered_on_cursor())
        # Settings > Keybinds' wheel rows, masked modifiers -> settings.WHEEL_ACTIONS id, and the
        # brush hooks from set_wheel_brush_hooks(). _brush_wheel_accum is the sub-notch delta.
        self._wheel_bindings: dict[int, str] = {}
        self._wheel_brush_query: Callable[[], bool] = lambda: False
        self._on_wheel_brush_step: Callable[[int], None] = lambda steps: None
        self._brush_wheel_accum = 0
        # GH #245: the paste-turn hooks from set_wheel_paste_hooks(), with their own sub-notch delta.
        self._wheel_paste_query: Callable[[], bool] = lambda: False
        self._on_wheel_paste_turn: Callable[[int], None] = lambda direction: None
        self._paste_wheel_accum = 0
        # GH #356: set_wheel_selection_query()'s query; a selection turn shares _place_wheel_accum.
        self._wheel_selection_query: Callable[[], bool] = lambda: False
        self.set_wheel_bindings(settings.get_wheel_bindings())
        self.setMouseTracking(True)

    def set_brush(self, size: int, shape: str, thickness: int = brush.BRUSH_THICKNESS_MIN) -> None:
        """Updates the brush the hover-preview highlight (and, via
        ViewerWindow reading these same values back at stroke time, the
        actual edit) uses. Only the preview needs to react here -- the
        active stroke's own footprint is computed fresh per cursor tile in
        ViewerWindow.on_edit_stroke_tile, not cached. `thickness` counts for
        an outline shape only (brush.OUTLINE_SHAPES)."""
        self._brush_size = size
        self._brush_shape = shape
        self._brush_thickness = thickness
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

    def set_shape_press_query(self, fn: Callable[[], unit_shape.ShapeLatch | None]) -> None:
        """GH #365: `fn()` reads Unit Line / Unit Shape's pick and options at a
        press: a latch held for the drag, or None to start no drag."""
        self._shape_press_query = fn

    def set_place_blocked_query(self, fn: Callable[[], frozenset]) -> None:
        """GH #124: `fn()` returns the tiles a wall drag leaves empty (empty
        with "Walls skip occupied tiles" off). Read once per press."""
        self._place_blocked_query = fn

    def set_place_preview(self, fn: Callable[[QPointF, object], None]) -> None:
        """GH #128: `fn(scene_pos, modifiers)` resolves the Place Unit or Unit
        Brush hover ghost and pushes it back through set_unit_ghost()."""
        self._on_place_preview = fn

    def set_place_ghost_query(self, fn: Callable[[], bool]) -> None:
        """GH #172: `fn()` says whether Place Unit's or the Unit Brush's ghost is showing, so the wheel's rotate binding turns it."""
        self._place_ghost_query = fn

    def set_place_rotate(self, fn: Callable[[int], None]) -> None:
        """GH #172: `fn(steps)` turns the ghost, a live unit drag (GH #280) or the selection (GH #356), +1 per clockwise notch."""
        self._on_place_rotate = fn

    def reset_place_wheel(self) -> None:
        """Drops a half-turned notch of the wheel's rotate binding (a catalog pick change)."""
        self._place_wheel_accum = 0

    def set_mode(self, mode: str) -> None:
        """Mirrors ViewerWindow's own self.mode -- phase 3's P3-d.

        Modes and tools are orthogonal in this class (Pan and every edit tool
        exist in every mode; only reachability differs). Units mode has a
        top-level branch in mousePressEvent/mouseMoveEvent, and inside it the
        tool decides (TASK-252): Select picks, Move drags, Pan pans. The one
        "select" tool means a tile region outside Units mode and units in it.
        """
        if mode == self._mode:
            return
        self._mode = mode
        self._brush_wheel_accum = 0  # a half-stepped brush notch belongs to the mode being left
        self._paste_wheel_accum = 0
        self._eyedrop_cycle = None
        # Same reasoning as the tool switch: a pending measurement has no owner
        # in the next mode. Pinned rulers stay.
        self._clear_ruler()
        if mode != "units":
            self._clear_unit_hover()
            # Not the selection: ViewerWindow keeps it across modes and hides or re-pushes it (TASK-359).
            # A pending drag has no owner in the next mode either -- defensive
            # only, since a mode switch mid-press is not reachable through the
            # toolbar/combo UI, but leaving a stale key would misroute the
            # eventual release.
            self._unit_drag_key = None
            self._unit_drag_press_pos = None
            self._clear_unit_ghost()
            self._place_hover = None
            self._cliff_hover = None
            self._marquee_start_pos = None
            self._marquee_press_tile = None
            self._clear_marquee()
        # Defensive, mirroring the _unit_drag_key reset above: a mode
        # switch mid-drag reaches the tool through _update_tool_enabled's
        # forced-back-to-Pan path (which calls set_tool, which cancels
        # already), so this is a second belt on the same trousers.
        self._cancel_shape()
        self._cancel_area_drag()
        self._sync_stack_badges_visible()
        # Units mode forces NoDrag for every tool but Pan. Without this a
        # Select or Move press would fall through to ScrollHandDrag and pan.
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _apply_drag_mode(self) -> None:
        # The Ruler needs NoDrag for the same reason Select and Move do, and is
        # named explicitly because it is NOT in EDIT_TOOLS: left-dragging is
        # how a measurement is made, so leaving ScrollHandDrag on would pan
        # the view instead and the tool would never work at all.
        if (
            (self._mode == "units" and self._tool != "pan")
            or self._unit_picker_active
            or self._tile_picker_kind is not None
            or self._area_hover is not None
            or self._area_drag is not None
            or self._tool in EDIT_TOOLS
            or self._tool == TOOL_RULER
            or self._tool == TOOL_EYEDROPPER
        ):
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_tool(self, tool: str) -> None:
        # Switching tools mid-drag (e.g. a keyboard shortcut fired while the
        # mouse button is still down) must not leave a stroke dangling --
        # close it out first, same as a normal release would.
        if self._stroke_active and tool != self._tool:
            self._end_stroke()
        # The hover ghost belongs to the tool. Gated on the outgoing tool so
        # a keyboard switch mid-drag keeps the drag's own ghost.
        if self._tool in _HOVER_GHOST_TOOLS and tool != self._tool:
            self._place_hover = None
            self._clear_unit_ghost()
        # The Cliff tool's hover ghost likewise (its stroke closed just above).
        if self._tool == "cliff" and tool != "cliff":
            self._cliff_hover = None
            self._clear_unit_ghost()
        self._tool = tool
        # Whichever highlight belonged to the tool being left is no longer
        # valid -- mouseMoveEvent's next move re-creates the right one for
        # the new tool, but a stale one from before the switch (e.g. Pan's
        # black outline still showing right after switching into an edit
        # tool) must not linger until then.
        self._clear_highlight()
        self._clear_pan_highlight()
        # Units mode's hover outline too: Pan shows none, so a switch to it
        # with the cursor resting on a unit must not leave one behind.
        self._clear_unit_hover()
        # The Eyedropper's stack position belongs to the tool, as its hover outline does.
        self._eyedrop_cycle = None
        # A pending measurement belongs to the Ruler, so leaving the tool
        # drops it. Finished ones are pinned and stay (GH #108/#109).
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
            self._close_region_drag_trace()
            self._update_region_overlay()
        # Cancel, never commit -- a shape the user never released is not a
        # shape they asked to paint. Unconditional rather than gated on the
        # new tool: unlike a region, a live shape drag belongs to the drag,
        # not to the tool, so even switching between Line and Rectangle
        # drops it.
        self._cancel_shape()
        self._brush_wheel_accum = 0  # a half-stepped brush notch belongs to the tool being left
        self._paste_wheel_accum = 0
        self._cancel_area_drag()
        self._sync_pulse_timer()
        # Routed through _apply_drag_mode() rather than set here directly, so
        # Units mode's NoDrag can't be undone by a tool switch.
        self._apply_drag_mode()
        self._apply_tool_cursor()

    def _apply_tool_cursor(self) -> None:
        if self._middle_drag_active:
            return  # middle-drag's closed-hand cursor takes priority for now
        if self._tile_picker_kind is not None:
            self.setCursor(Qt.CrossCursor)
            return
        if self._unit_picker_active:
            self.setCursor(Qt.PointingHandCursor)
            return
        zone = self._area_drag[0] if self._area_drag is not None else self._area_hover
        if zone is not None:
            self.setCursor(self._area_zone_cursor(zone))
            return
        # Above the mode check, unlike every other tool: Units mode's
        # pointing hand otherwise wins and the Ruler loses its aiming cursor
        # in the one mode where precision matters most. Select picks units
        # there, so it takes the pointing hand like the other unit tools.
        if self._tool in (TOOL_RULER, TOOL_EYEDROPPER) or (self._tool == TOOL_SELECT and not self._units_select()):
            self.setCursor(Qt.CrossCursor)
        elif self._units_move():
            self.setCursor(Qt.SizeAllCursor)
        elif self._mode == "units" and self._tool == "pan":
            self.unsetCursor()  # the hand drag's own open/closed hand
        elif self._mode == "units":
            self.setCursor(Qt.PointingHandCursor)
        elif self._tool in EDIT_TOOLS:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

    @property
    def colour_grade(self) -> colour_mood.MoodGrade | None:
        return self._colour_grade

    def set_colour_grade(self, grade: colour_mood.MoodGrade | None) -> None:
        """The map's colour mood, over the canvas and the background around it
        (a scene brush, so scene.render() takes it too). Draw-only: no cache,
        chunk or scenario sees it. None draws both unchanged."""
        if grade == self._colour_grade:
            return
        self._colour_grade = grade
        if self._canvas_item is not None:
            self._canvas_item.grade = grade
        outside = self.OUTSIDE_MAP_COLOR
        if grade is not None:
            outside = QColor(*colour_mood.apply(grade, outside.getRgb()[:3]))
        self.scene().setBackgroundBrush(outside)
        self.viewport().update()

    def clear_image(self) -> None:
        """Undoes set_source() -- back to the pre-load empty state (used by
        File > Close). Mirrors set_source()'s own scene().clear() + highlight
        reset, since QGraphicsScene.clear() destroys the highlight items (and
        the canvas item) too; leaving the Python-side reference pointing at
        it would dangle."""
        self._drop_pending_move()
        self.scene().clear()
        self._canvas_item = None
        self._trigger_ring_memo.clear()
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
        self._footprint_items = {}
        self._fish_item = None
        self._footprint_chunks.reset()
        self._fish_chunks.reset()
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
        # Same again for the trigger overlay, snapshot included, and its ghosts.
        self._forget_trigger_items()
        self._trigger_shapes = []
        self._trigger_emphasis = None
        self._forget_trigger_ghost_items()
        self._trigger_ghost_records = []
        self._trigger_ghost_emphasis = None
        self._trigger_ghosts_stale = False
        # Same hazard: scene().clear() destroyed the ghost's C++ object, so
        # _clear_unit_ghost()'s removeItem() on the next drag exit would raise.
        self._unit_ghost_item = None
        self._ghost_fish_item = None
        self._place_hover = None
        self._cliff_hover = None
        # Same hazard as the tick item above, and the same fix: forget the
        # destroyed items, and drop the measurement itself, since File > Close
        # leaves no map for it to refer to. Pinned rulers go too.
        self._ruler.clear()
        self._forget_ruler_items()
        self._drop_pinned_rulers()
        self._emit_ruler_status()
        # Same reasoning for a shape drag: File > Close mid-drag leaves no
        # map for the release to commit against.
        self._shape_anchor = None
        self._shape_end = None
        self._shape_modifiers = Qt.NoModifier
        self._drag_shape = ""
        self._drag_blocked = frozenset()
        self._shape_latch = None
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
        self._range_ring_items = {}
        self._trigger_ring_items = {}
        self._ring_layer_items = {}
        self._ring_layer_polygons = {}
        self._ring_layer_projection = None
        self._team_colors = None
        self._unit_hover_key = None
        self._unit_select_keys = []
        self._marquee_item = None
        self._marquee_start_pos = None
        # Same for GH #138's pick preview; a held rect drag has no map left to commit against.
        self._tile_pick_item = None
        self._tile_pick_label = None
        self._tile_pick_anchor = None
        self._cancel_area_drag()
        self._unit_index = None
        # scene().clear() destroyed these too. Dropped outright, unlike
        # set_source()'s own region handling below: File > Close leaves no
        # map for a region to refer to, so there is no "survives" case here.
        self._select_anchor = None
        self._select_current = None
        self._move_anchor = None
        self._move_current = None
        self._close_region_drag_trace()
        self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_scene_rect = None
        self._region_overlay_rect = None
        self._region_ant_timer.stop()
        self.resume_stroke_gc()
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
        ViewerWindow._apply_dirty: dirty_screen_bbox_iso()/Flat's block rects
        -> cache.patch()/patch_rects() -> here) -- this method is purely
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

    def set_canvas_placeholders(self, on: bool) -> None:
        """set_paint_timed_callback()'s shape for the canvas item's placeholder
        paints (TASK-031.25): off also settles a pending level."""
        if self._canvas_item is None:
            return
        self._canvas_item.set_placeholders(on)

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
        placeholders: bool = False,
        on_placeholder: Callable[[int], None] | None = None,
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

        placeholders/on_placeholder go to the new MapCanvasItem's constructor
        the same way, since its first paint can fire inside this method
        (TASK-031.25); set_canvas_placeholders() changes the first later.

        Kept as one method (dispatching on terrain_style), not two, matching
        this class's existing internal-dispatch convention (_pick_tile,
        _pos_on_map, set_isometric already branch on self._terrain_style
        rather than being split into per-style methods) -- callers still
        have exactly one entry point to call regardless of style."""
        self._drop_pending_move()
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
        self._trigger_ring_memo.clear()
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
        # neither guarantees the old tiles still mean anything. Pinned rulers
        # follow self._region's rule: kept on a same-document redraw, rebuilt below.
        self._ruler.clear()
        self._forget_ruler_items()
        self._pinned_ruler_items = []
        if reset_view:
            self._drop_pinned_rulers()
        self._emit_ruler_status()
        # scene().clear() destroyed the rubber band's items too, and a
        # shape anchored on the old map's tiles means nothing on the new one.
        self._shape_anchor = None
        self._shape_end = None
        self._shape_modifiers = Qt.NoModifier
        self._drag_shape = ""
        self._drag_blocked = frozenset()
        self._shape_latch = None
        # scene().clear() destroyed the unit items too. Selection is dropped
        # rather than re-resolved: set_source() means a new scenario or a
        # style switch, and neither guarantees the old key still addresses
        # anything.
        self._unit_hover_item = None
        self._unit_select_groups = {}
        self._range_ring_items = {}
        self._trigger_ring_items = {}
        self._ring_layer_items = {}
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
        # The trigger overlay follows the same rule, rebuilt at the end of this method; its ghosts too.
        self._forget_trigger_items()
        self._forget_trigger_ghost_items()
        if reset_view:
            self._trigger_shapes = []
            self._trigger_emphasis = None
            self._trigger_ghost_records = []
            self._trigger_ghost_emphasis = None
        self._unit_hover_key = None
        self._unit_select_keys = []
        self._marquee_item = None
        self._marquee_start_pos = None
        # GH #138's pick preview went too; a held rect drag's anchor is in the old projection.
        self._tile_pick_item = None
        self._tile_pick_label = None
        self._tile_pick_anchor = None
        self._cancel_area_drag()
        # The ghost went with it too, and the drag behind it means nothing
        # against a new scenario -- dropped, not re-anchored, so the release
        # commits nothing (see keyPressEvent's Escape branch for the other
        # place that pairing matters).
        self._unit_ghost_item = None
        self._ghost_fish_item = None
        # A hover point in the old scene's coordinates; the next move sets it again.
        self._place_hover = None
        self._cliff_hover = None
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
        self._close_region_drag_trace()
        if reset_view:
            self._region = None
        self._region_fill_item = None
        self._region_outline_item = None
        self._region_ants_item = None
        self._region_scene_rect = None
        self._region_overlay_rect = None
        self._region_ant_timer.stop()

        self._canvas_item = MapCanvasItem(cache, placeholders=placeholders, on_placeholder=on_placeholder)
        self._canvas_item.grade = self._colour_grade
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
            major_every=self._grid_major_every,
        )
        self.scene().addItem(self._grid_item)
        self._apply_grid()

        self._footprint_items = {None: self._add_footprint_item(self._footprint_pen)}
        self._fish_item = self._add_fish_item()
        self._footprint_chunks.reset()
        self._fish_chunks.reset()

        # Empty until the caller installs an index via set_unit_index().
        tile_extent = 2 * proj.half_w if iso_proj is not None else tile_pixels
        self._stack_groups = {}
        self._stack_badge_item = StackBadgeItem(
            tile_extent,
            QColor(settings.get_overlay_color("unit_stack")),
            QColor(settings.get_overlay_color("unit_stack_background")),
        )
        self._stack_badge_item.set_placement(self._stack_badge_position != "above")
        self._stack_badge_item.setZValue(self.UNIT_STACK_Z)
        self._stack_badge_item.setVisible(False)
        self._stack_badges_stale = True
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
        self._rebuild_trigger_ghosts()
        self._rebuild_pinned_rulers()
        # Forces the next poll fire to notify regardless of what it finds:
        # without this, a cache swap whose new viewport_chunk_target()
        # happens to equal the OLD document's last-recorded one (same mip,
        # same chunk range -- not far-fetched right after a reset_view=False
        # graphics-quality change) would see "no change" on its first fire
        # and stop silently, skipping that document's first margin ring
        # until the next scroll.
        self._last_viewport_target = None
        self._note_viewport_changed()

    def refresh_elevation_overlays(self) -> None:
        """The overlays whose geometry follows the terrain, for an elevation
        edit to call once per touched-tile batch.

        The grid needs nothing here: it is baked into the chunks, and the
        edit's own cache.patch() already recomposited them. The footprint
        and fish outlines rebuild only the chunks under the changed heights,
        and are still deferred to one flush per event-loop turn: one 8-step
        Set Elevation drag touches 131 tiles per _apply_dirty's own comment.
        The always-on ring layer (GH #153 / #155) rides the same deferral."""
        self.schedule_footprint_refresh()
        # Not deferred: at most 8 items, each one _ruler_anchor lookup, and
        # unlike the footprint path there is no per-unit walk behind it.
        self._rebuild_camera_markers()
        # Not deferred either: at most ~1800 _tile_polygon() calls.
        self._rebuild_analysis_markers()
        # _tile_polygon reads the elevations, so Stepped and Sloped rings move. O(perimeter).
        self._rebuild_trigger_overlay()
        # The ghosts re-resolve sprites, so deferred like the footprints, and held to stroke end.
        self._schedule_trigger_ghost_refresh()
        # Same for a pinned ruler's endpoints: at most a handful of items.
        self._rebuild_pinned_rulers()
        # Stack badges anchor on the tile outline too (GH #100). Two outline lookups per
        # stack, but only in Units mode: elevation tools are Terrain-only, so a stroke skips it.
        self._rebuild_stack_badges()
