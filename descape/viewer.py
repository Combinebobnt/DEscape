"""PyQt5 viewer: open an .aoe2scenario file, pan/zoom the rendered map, and read
per-tile / per-unit details. Edit mode's Terrain/Elevate/Set Elevation tools
(v2) paint the map and save out via File > Save As -- see
descape.scenario_write for the write path itself."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPolygonF,
)
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from AoE2ScenarioParser.datasets.terrains import TerrainId

from descape import asset_source, brush, debug_log, iso_geometry, settings
from descape.edit_history import EditHistory, tile_state
from descape.elevation_tools import set_tile_elevation, set_tiles_elevation
from descape.fill_tools import flood_fill_terrain
from descape.render import (
    SMALL_MAP_TILE_PIXELS,
    FlatChunkCache,
    IsoChunkCache,
    SlopedChunkCache,
    dirty_screen_bbox_iso,
    elevations_and_proj,
    sloped_elevations_and_proj,
    tile_pixels_for_map,
)
from descape.scenario_io import (
    BLANK_TEMPLATE_TILES,
    TEMPLATE_DIR,
    LoadedScenario,
    load_map_and_units,
    load_map_and_units_from_bytes,
)
from descape.scenario_new import (
    LARGE_MAP_CONFIRM_TILES,
    MAX_MAP_TILES,
    MIN_MAP_TILES,
    STANDARD_MAP_SIZE_NAMES,
    STANDARD_MAP_SIZES,
    BlankGenerationError,
    MapSizeError,
    blank_scenario_bytes,
)
from descape.scenario_write import WriteBlockedError, write_scenario
from descape.terrain_palette import name_for_terrain_id

# Sourced from iso_geometry, not redefined here, so Set Elevation's spinbox
# range and Stepped mode's canvas sizing (descape.render.elevations_and_proj)
# can never drift apart -- see that constant's own comment for why Phase 4
# needs them to agree exactly.
ELEVATION_LEVEL_MAX = iso_geometry.MAX_ELEVATION

# Above this many dirty tiles, Stepped mode's own _apply_dirty() path (per-
# tile dilation into a Python set, then one patch() over what ends up being
# effectively the whole canvas -- see dirty_screen_bbox_iso's own docstring)
# is slower than just re-rendering: measured with tools/bench_fill_latency.py
# against a full-map Paint Can fill (this project's own worst case for dirty
# tile count) -- 14,400 tiles (blank_120x120) stayed under the dilation path
# at ~105ms, but 230,400 tiles (blank_480x480) cost ~2.3s there versus ~35ms
# for a full _render_current() (Phase B-C's lazy chunk cache means "full
# re-render" no longer means "recomposite every pixel up front" -- see that
# method's own docstring). Set well above the fine case and well below the
# slow one; the exact crossover between them hasn't been measured.
STEPPED_FULL_RERENDER_THRESHOLD = 20_000

# Display-only sentinel assigned to LoadedScenario.path for a File > New map.
# Deliberately relative and non-existent: LoadedScenario.path is only ever read
# for display and for Save As's default name -- write_scenario() takes an
# explicit destination and never reads it (see that field's own comment in
# scenario_io.py).
UNTITLED_NAME = "Untitled.aoe2scenario"
UNTITLED_PATH = Path(UNTITLED_NAME)

# Derived from settings.TOOLS -- the single tool registry -- rather than
# hand-duplicated here, so a new tool added there can't silently miss an
# entry in any of these. Internal tool identifiers (e.g. "elevation",
# "set_level") are unchanged even when a display label is renamed later
# (Elevate / Set Elevation), so settings.get_keybind()'s persisted
# "tool_elevation"/"tool_set_level" keys and every other internal reference
# keep working regardless.
EDIT_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.is_edit_tool)
_TOOL_LABELS = {t.tool_id: t.label for t in settings.TOOLS}
_STROKE_LABELS = {t.tool_id: t.stroke_label for t in settings.TOOLS if t.is_edit_tool}
# One-shot click tools (Paint Can) that must never enter the drag-stroke
# path -- see MapView.mousePressEvent's early CLICK_TOOLS branch and
# ViewerWindow.on_fill.
CLICK_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.click_only)
# Which toolbar param widget ("terrain" | "level" | "") the active tool
# reads -- see ViewerWindow._update_tool_enabled's tool-param visibility
# block.
_TOOL_PARAM = {t.tool_id: t.param_widget for t in settings.TOOLS}
# Tools whose stroke applies across a brush footprint (size + shape) rather
# than always exactly one tile -- see ToolDef.supports_brush's own comment.
BRUSH_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_brush)

STATUS_OK_COLOR = "#4caf50"
STATUS_ERROR_COLOR = "#e05252"


def _max_axis_scale(transform) -> float:
    """The largest singular value of `transform`'s 2x2 linear part -- i.e.
    the greatest factor by which it stretches ANY direction, which is the
    scale a mip level has to keep up with. Phase B-D-c's LOD metric.

    Deliberately NOT QStyleOptionGraphicsItem.levelOfDetailFromTransform():
    measured on Flat's own scale(1, 0.5) + rotate(-45) at 2x zoom, that
    returns 1.5811 where the true max stretch is 2.0 -- floor(log2(...)) of
    0 vs 1, i.e. a different level, one octave of blur along Flat's
    stretched axis. Both numbers are reproduced as an oracle in
    tests/test_mip_viewer.py.

    Qt's m12/m21 naming can't silently transpose this: singular values are
    invariant under transpose, so the metric is identical either way (also
    pinned in that test file)."""
    a, b = transform.m11(), transform.m21()
    c, d = transform.m12(), transform.m22()
    half_sum_sq = (a * a + b * b + c * c + d * d) / 2.0
    det = a * d - b * c
    # max(0, ...) only guards float error at the equal-singular-value
    # (pure uniform scale/rotation) case, where the discriminant is 0.
    disc = math.sqrt(max(0.0, half_sum_sq * half_sum_sq - det * det))
    return math.sqrt(half_sum_sq + disc)


class MapCanvasItem(QGraphicsItem):
    """Qt-side wrapper around a descape.render chunk cache -- Phase B-C of
    Track B, replacing a mode's single monolithic
    QPixmap/array (the whole canvas rendered up front, ~1-2s and hundreds of
    MB for this project's bigger real maps) with a QGraphicsItem whose
    paint() composites only the chunks its own exposedRect actually needs,
    via the Qt-free chunk cache built in Phase B-B. Introduced Stepped-only
    (IsoChunkCache); Phase B-E gives Flat mode the same treatment
    (FlatChunkCache) via this SAME class, unchanged -- its paint() has
    nothing style-specific in it (both caches expose the same canvas_dims()/
    render_rect() contract, see descape.render._ChunkCacheBase), only which
    cache MapView.set_source() constructs differs.

    ItemUsesExtendedStyleOption is the load-bearing flag here: without it
    Qt always reports the item's FULL boundingRect as "exposed" regardless
    of what's actually visible, which would make paint() composite every
    chunk on every repaint -- silently degrading back to full-canvas cost
    with none of the memory/latency upside a chunk cache exists for. See
    tests/test_lazy_viewport.py for this specifically (option.exposedRect
    must come back smaller than boundingRect() under a real Qt paint
    cycle, not just in principle)."""

    # Phase B-E: painting a padded superset per block, tiled at PAINT_BLOCK_PX,
    # rather than one drawImage() over the whole exposedRect -- see paint()'s
    # own comment for why. Both apply identically to Stepped and Flat (this
    # class has nothing style-specific in it -- see the class docstring), and
    # land AFTER Flat was wired to this class (Phase B-E's earlier steps) so a
    # later human-reported Stepped visual artifact still bisects to either
    # Phase B-C or this specific change, not ambiguously to "B-E" as a whole.
    #
    # PAINT_PAD_PX: under Flat's rotated view transform (scale(1,0.5) then
    # rotate(-45), see MapView.set_isometric()), option.exposedRect is the
    # axis-aligned bbox of a ROTATED region, so drawImage()'s destination
    # rect is sampled by QPainter.SmoothPixmapTransform under that rotation
    # -- texels just outside a bare exposedRect-sized block don't exist,
    # which can show as a 1px seam at block edges. Compositing a small
    # padded superset instead gives the filter real neighbor pixels on both
    # sides. Under an identity/axis-aligned transform (Stepped, or Flat
    # unrotated) this padding is a no-op superset -- correctness is
    # unaffected either way, only clipped to canvas bounds like the
    # unpadded block always was.
    #
    # Phase B-D-d note: 2px was justified above (and in PLAN_MIPS.md) partly
    # by "residual magnification in [1, 2) means bilinear reaches at most 1
    # source texel past a block edge", which mip-down's clamp at the coarsest
    # level makes no longer universally true -- below that level the residual
    # is a MINIFICATION. Measured across residuals 0.3/0.5/0.8/0.95 (see
    # tests/test_mip_viewer.py's minifying-band A/B), tiled and single-draw
    # renders still come back byte-identical, so 2px stands unchanged for the
    # minifying band too: bilinear samples a 2x2 texel neighbourhood whatever
    # the ratio, so its reach past a block edge is set by the filter, not by
    # the scale factor.
    #
    # PAINT_BLOCK_PX: at fit-to-view (the whole map exposed at once, e.g.
    # right after opening a file), a single render_rect() call over the
    # entire exposedRect would allocate one full-canvas-sized stitched
    # scratch array (675 MiB on this project's biggest real map) ON TOP OF
    # the persistent warm cache -- a real transient memory regression
    # against the old single-buffer approach this phase is supposed to
    # avoid reintroducing. Tiling a large exposedRect into PAINT_BLOCK_PX
    # sub-blocks (each independently padded and drawn) bounds that scratch
    # to a small, fixed size regardless of how much of the canvas is
    # exposed at once.
    PAINT_PAD_PX = 2
    PAINT_BLOCK_PX = 2048

    def __init__(self, cache: "IsoChunkCache | FlatChunkCache | SlopedChunkCache"):
        super().__init__()
        self._cache = cache
        # Scene space is pinned to the REFERENCE level permanently (plan
        # decision D2), so boundingRect never follows the selected mip --
        # that's what leaves _pick_tile/_pos_on_map/the overscroll scene
        # rect/set_isometric/wheelEvent untouched by this whole track.
        canvas_w, canvas_h = cache.canvas_dims()
        self._bounding_rect = QRectF(0, 0, canvas_w, canvas_h)
        # Write-only, for tests/debugging to see what paint() selected.
        # paint() never reads it back: selection stays a pure function of
        # the painter transform (see _select_mip). It records the LAST
        # paint only, so a reader treating it as "the level this render
        # chose" has to have driven exactly one paint (Phase B-D-d's tests
        # do, via a single scene.render() -- worth keeping in mind before
        # reusing it after a repaint that Qt may have split).
        self._last_mip = 0
        self.setFlag(QGraphicsItem.ItemUsesExtendedStyleOption, True)

    def _select_mip(self, painter: QPainter) -> int:
        """The mip level to paint from, as a pure function of `painter`'s
        transform -- no hysteresis, no stored state. Same view state must
        always yield the same pixels, or every scene-render byte-identity
        check in this track becomes history-dependent.

        Reads deviceTransform(), not worldTransform(): measured, at
        QT_SCALE_FACTOR=2 the device transform's m11 is 6.0 where world's
        is 3.0, so selecting from device gets HiDPI right for free. The
        blit itself stays in logical coordinates, so DPR never enters the
        destination-rect math.

        The rule itself (clamp(floor(log2(scale)), coarsest, finest), plus
        the degenerate scale <= 0 guard) lives on the cache as
        mip_for_scale() and is deliberately not restated here -- Phase
        B-D-c carried a local copy only because it clamped the floor at the
        reference level; Phase B-D-d, which extends selection below the
        reference, has nothing left to add to the cache's own rule.

        floor(log2(scale)) is derived, not tuned: one level-L pixel covers
        mip_scale(L) * scale device pixels, and requiring that be >= 1
        (never minify) is exactly L <= log2(scale). Residual magnification
        then lands in [1, 2) whenever the derived level is actually
        available; below the coarsest enumerated level the clamp binds
        instead and the residual becomes a minification, bounded only by
        how far MapView's zoom-out floor lets the view go (see
        _capture_zoom_baseline)."""
        return self._cache.mip_for_scale(_max_axis_scale(painter.deviceTransform()))

    def boundingRect(self) -> QRectF:
        return self._bounding_rect

    def paint(self, painter: QPainter, option, widget=None) -> None:
        rect = option.exposedRect.intersected(self._bounding_rect)
        if rect.isEmpty():
            return
        # Everything below tiles and pads in LEVEL pixels, deriving each
        # destination rect in scene units by multiplying back by `scale`.
        # Doing it the other way (tiling in scene units) would round block
        # edges independently per block, so adjacent blocks' destination
        # rects could fail to abut -- `scale` is an exact power of two, so
        # this way they always do.
        mip = self._select_mip(painter)
        self._last_mip = mip
        scale = self._cache.mip_scale(mip)
        # Floor/ceil, not round -- the composited block must fully cover
        # exposedRect (a gap at the edge would leave a visible unpainted
        # sliver), matching iso_geometry's own "whole pixels only" integer
        # discipline throughout this project's rendering code.
        x0, y0 = int(math.floor(rect.left() / scale)), int(math.floor(rect.top() / scale))
        x1, y1 = int(math.ceil(rect.right() / scale)), int(math.ceil(rect.bottom() / scale))
        canvas_w, canvas_h = self._cache.canvas_dims(mip)
        x1, y1 = min(x1, canvas_w), min(y1, canvas_h)
        if x1 <= x0 or y1 <= y0:
            return

        # Tile [x0,x1)x[y0,y1) into PAINT_BLOCK_PX-aligned sub-blocks (grid
        # anchored at the canvas origin, not at x0/y0, so repeated partial
        # exposures of the same region always request the same block
        # boundaries -- irrelevant to correctness, since each block is
        # independently correct, but keeps cache chunk-fetch patterns
        # stable across paint calls). Overlapping padded draws between
        # adjacent blocks are harmless: each is an opaque, independently
        # correct overwrite of its own destination rect.
        block_x0 = (x0 // self.PAINT_BLOCK_PX) * self.PAINT_BLOCK_PX
        block_y0 = (y0 // self.PAINT_BLOCK_PX) * self.PAINT_BLOCK_PX
        for by in range(block_y0, y1, self.PAINT_BLOCK_PX):
            for bx in range(block_x0, x1, self.PAINT_BLOCK_PX):
                cx0, cy0 = max(bx, x0), max(by, y0)
                cx1 = min(bx + self.PAINT_BLOCK_PX, x1)
                cy1 = min(by + self.PAINT_BLOCK_PX, y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                px0 = max(0, cx0 - self.PAINT_PAD_PX)
                py0 = max(0, cy0 - self.PAINT_PAD_PX)
                px1 = min(canvas_w, cx1 + self.PAINT_PAD_PX)
                py1 = min(canvas_h, cy1 + self.PAINT_PAD_PX)
                block = self._cache.render_rect(px0, py0, px1, py1, mip=mip)
                if block.size == 0:
                    continue
                h, w = block.shape[:2]
                contiguous = np.ascontiguousarray(block)
                qimg = QImage(contiguous.data, w, h, 3 * w, QImage.Format_RGB888)
                if scale == 1.0:
                    # The point overload is a different QPainter code path
                    # from the scaled one and is not guaranteed
                    # bit-identical to it at unit scale. Keeping it for the
                    # reference level is what makes this phase structurally
                    # incapable of regressing any existing byte-identity
                    # bar, rather than merely empirically not doing so.
                    painter.drawImage(px0, py0, qimg)
                else:
                    painter.drawImage(
                        QRectF(px0 * scale, py0 * scale, w * scale, h * scale),
                        qimg,
                        QRectF(0, 0, w, h),
                    )

    def invalidate_region(self, bbox: tuple[int, int, int, int]) -> None:
        """Schedules a Qt repaint for exactly the given canvas-pixel bbox --
        the Qt-side half of an edit's redraw. Does NOT touch self._cache
        itself (the caller must already have called cache.patch(bbox) --
        see ViewerWindow._apply_dirty); this is purely "please repaint this
        area," so the next paint() call pulls the already-patched pixels.

        Distinct from IsoChunkCache.invalidate_region() (Phase B-B): that
        one evicts cache entries to force a real recomposite; this one
        never touches the cache at all, just Qt's own dirty-region
        tracking. Same name, different class, different layer."""
        x0, y0, x1, y1 = bbox
        self.update(QRectF(x0, y0, x1 - x0, y1 - y0))


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

    # How far past "whole map fits in view" wheelEvent() allows zooming out or
    # in -- see _capture_zoom_baseline(). Zoom-out past this aliases badly
    # (real per-tile texture detail is too high-frequency for
    # QPainter.SmoothPixmapTransform's plain bilinear filtering to minify
    # cleanly without mipmapping); zoom-in past this just shows the same
    # finite per-tile texture resolution increasingly blurry/blocky, with no
    # more real detail to reveal.
    # Phase B-D-d divides the zoom-out half of this by the coarsest available
    # mip level's scale -- see _capture_zoom_baseline(). The zoom-in half is
    # untouched: mip levels finer than the reference are a real-detail win,
    # not a reason to allow more magnification past the finest one.
    MIN_ZOOM_FRACTION_OF_FIT = 0.5
    MAX_ZOOM_MULTIPLE_OF_FIT = 40.0

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

    def __init__(self, on_hover, on_stroke_start, on_stroke_tile, on_stroke_end, on_click_edit):
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
        # _iso_elevations/_iso_proj are Stepped-only, the exact (h, w)
        # elevation snapshot and IsoProjection descape.render.elevations_and_
        # proj() computed for the current source from -- Risk #6: these
        # must never drift from what's actually on screen, so they're only
        # ever set together with the source itself, in set_source(), never
        # patched independently.
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
        if tool in EDIT_TOOLS:
            self.setDragMode(QGraphicsView.NoDrag)
            self._pulse_timer.start(self.HIGHLIGHT_PULSE_TICK_MS)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._pulse_timer.stop()
        self._apply_tool_cursor()

    def _apply_tool_cursor(self) -> None:
        if self._middle_drag_active:
            return  # middle-drag's closed-hand cursor takes priority for now
        if self._tool in EDIT_TOOLS:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()

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
            # Hit-testing is Track C4's job, not this phase's -- no analytic
            # inverse yet for a bilinear-blended ramp surface.
            return None
        return int(pos.x()) // self._tile_pixels, int(pos.y()) // self._tile_pixels

    def _pos_on_map(self, pos: QPointF) -> bool:
        """Flat mode's old _map_rect.contains(pos) generalizes in Stepped
        mode to "does this pixel unproject to a real tile" -- the map's
        silhouette there is a diamond inscribed in _map_rect (the full
        pixmap bounding box), with real background pixels in the corners
        _map_rect.contains(pos) alone would wrongly call on-map. Sloped has
        no hit-testing yet (see _pick_tile), so it's never on-map."""
        if self._terrain_style == "stepped":
            return self._pick_tile(pos) is not None
        if self._terrain_style == "sloped":
            return False
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
        # argument the "terrain"/"set_level" tools would just ignore.
        if self._stroke_button == Qt.RightButton:
            modifiers = modifiers | Qt.ShiftModifier
        self._on_stroke_tile(tile_x, tile_y, modifiers)

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
        # which the "terrain"/"set_level" tools simply don't look at. Click
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

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self._clear_highlight()
        self._clear_pan_highlight()
        # Defensive: a drag that exits the widget bounds without a release
        # (Qt usually still routes the eventual release back here via its
        # implicit mouse grab during a press-drag, but this covers the case
        # where it doesn't -- e.g. focus lost mid-drag) must not leave a
        # half-open stroke with no matching commit.
        if self._stroke_active:
            self._end_stroke()

    def _tile_polygon(self, tile_x: int, tile_y: int) -> QPolygonF | None:
        """The on-screen footprint of tile (tile_x, tile_y) as a polygon --
        a plain axis-aligned square in Flat mode, the tile's real projected,
        elevation-displaced diamond in Stepped mode. Shared by the edit-mode
        pulsing highlight and Pan mode's static outline below, so the two
        can never disagree about where a tile actually is on screen. None
        in Stepped mode before a projection/elevation snapshot exists
        (shouldn't happen once an image is loaded -- defensive only)."""
        if self._terrain_style == "stepped":
            if self._iso_elevations is None or self._iso_proj is None:
                return None
            elevation = int(self._iso_elevations[tile_y, tile_x])
            ox, oy = iso_geometry.tile_screen_origin(tile_x, tile_y, elevation, self._iso_proj)
            half_w, half_h = self._iso_proj.half_w, self._iso_proj.half_h
            return QPolygonF(
                [
                    QPointF(ox + half_w, oy),
                    QPointF(ox + 2 * half_w, oy + half_h),
                    QPointF(ox + half_w, oy + 2 * half_h),
                    QPointF(ox, oy + half_h),
                ]
            )
        if self._terrain_style == "sloped":
            # No highlight/outline polygon until C4 builds real hit-testing.
            return None
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

    def set_source(
        self,
        tile_pixels: int,
        terrain_style: str = "flat",
        cache: "IsoChunkCache | FlatChunkCache | SlopedChunkCache | None" = None,
        elevations: np.ndarray | None = None,
        proj: "iso_geometry.IsoProjection | None" = None,
    ) -> None:
        """Phase B-C rename/generalization of the old set_image(); Phase B-E
        drops that method's img parameter entirely -- both styles now paint
        lazily from a chunk cache via MapCanvasItem (see
        ViewerWindow._render_current), never a single pre-baked array/pixmap.
        elevations/proj (Stepped only) are set together with cache so
        _pick_tile's hit-testing can never point at a stale elevation
        snapshot (Risk #6 in the parent plan); both stay None for Flat.

        Kept as one method (dispatching on terrain_style), not two, matching
        this class's existing internal-dispatch convention (_pick_tile,
        _pos_on_map, set_isometric already branch on self._terrain_style
        rather than being split into per-style methods) -- callers still
        have exactly one entry point to call regardless of style."""
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
        if terrain_style in ("stepped", "sloped") and proj is not None and elevations is not None:
            map_h, map_w = elevations.shape
            corners = iso_geometry.ground_outline_corners(map_w, map_h, proj)
            self.scene().addPolygon(
                QPolygonF([QPointF(x, y) for x, y in corners]), QPen(QColor(200, 200, 200), 0)
            )
        else:
            self.scene().addRect(self._map_rect, QPen(QColor(200, 200, 200), 0))

        self.set_isometric(self._isometric)

    def set_isometric(self, enabled: bool) -> None:
        self._isometric = enabled
        self.resetTransform()
        # Stepped's (and Sloped's) projection is already baked into the
        # pixels -- this whole method (below this point) is Flat mode's own
        # QTransform trick (see the class comment above) and has nothing
        # left to do for a Stepped/Sloped image beyond the same plain
        # "whole canvas fits the viewport" fit Flat's own "unchecked" case
        # already uses. ViewerWindow greys out the View > Isometric View
        # checkbox whenever Terrain Style != Flat, so this method only ever
        # reaches the enabled=True rotate/squash path below while Flat is
        # active.
        if self._terrain_style in ("stepped", "sloped") or not enabled:
            if self._map_rect is not None:
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
        margin = 0.92
        fit_scale = margin * min(
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
        # Fit-RELATIVE rather than an absolute scale bound, which is what the
        # plan originally specified: an absolute floor has no notion of how
        # big the canvas is in the window, so it lands above fit-to-view on a
        # big map (measured, 480x480 at a 480px window: fit is scale 0.0413,
        # an absolute floor of 1/(4 * 4) = 0.0625 would refuse zoom-out
        # everywhere and strand the user above fit). This form cannot do that
        # for anyone: mip_scale >= 1 always, so it is never tighter than the
        # pre-mip floor, and a level set with nothing below the reference
        # (elev_step_pct=10 enumerates {0, 1}) reduces it to exactly the
        # pre-mip expression rather than needing a special case.
        #
        # Both bounds go stale after a window resize: _capture_zoom_baseline
        # is only called from set_isometric() and there is no resizeEvent
        # override. Pre-existing, unchanged by Phase B-D-d.
        baseline = abs(self.transform().determinant()) ** 0.5
        self._min_linear_scale = self.MIN_ZOOM_FRACTION_OF_FIT * baseline / self._coarsest_mip_scale()
        self._max_linear_scale = self.MAX_ZOOM_MULTIPLE_OF_FIT * baseline

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current_scale = abs(self.transform().determinant()) ** 0.5
        if factor < 1.0 and self._min_linear_scale is not None and current_scale * factor < self._min_linear_scale:
            return
        if factor > 1.0 and self._max_linear_scale is not None and current_scale * factor > self._max_linear_scale:
            return
        self.scale(factor, factor)

    def mouseMoveEvent(self, event):
        if self._middle_drag_active:
            delta = event.pos() - self._middle_drag_last_pos
            self._middle_drag_last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        tile = self._pick_tile(pos)
        self._on_hover(tile)
        # In Stepped mode tile is not None already *is* the on-map test
        # (that's exactly what _pick_tile/_pos_on_map do there) -- avoid
        # re-running screen_to_tile a second time for the same pixel on
        # every mouse move by not routing through _pos_on_map here too.
        # Sloped shares this branch: its _pick_tile always returns None (no
        # hit-testing until C4), so on_map stays False there too, safely.
        if self._terrain_style in ("stepped", "sloped"):
            on_map = tile is not None
        else:
            on_map = self._map_rect is not None and self._map_rect.contains(pos)
        if self._tool in EDIT_TOOLS:
            self._clear_pan_highlight()
            if on_map:
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


# Cached the first time apply_theme() runs, before any dark palette is ever
# applied -- the baseline to restore when the user switches back to light.
# Captured from the Fusion style itself (style().standardPalette()), not
# whatever native platform style/palette was active before this app ever
# touched it: apply_theme() always forces Fusion (light or dark), so the
# "light" state should be Fusion's own default, not a different style's
# palette that would look inconsistent switching back and forth.
_LIGHT_PALETTE: QPalette | None = None


def _build_dark_palette() -> QPalette:
    """The standard Fusion dark palette recipe (Window/Base/Text/Button
    darkened, Highlight kept a legible blue, disabled-state colors dimmed
    separately so disabled controls don't just look identically dark) --
    no novel color choices here, this is the well-known combination most
    Fusion-dark Qt apps use."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, QColor(230, 230, 230))
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
    palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))
    return palette


def apply_theme(app: QApplication, dark: bool) -> None:
    """App chrome only -- menus, dialogs, toolbars, the status bar. MapView's
    own colors (real per-tile terrain textures, unit dots, hover highlights)
    are untouched: they represent game data or are already dark, not UI
    styling that should shift with this toggle. Called once at startup
    (main(), from the persisted settings.get_dark_mode()) and live from the
    Appearance settings tab -- safe to call repeatedly, setStyle("Fusion")
    is idempotent and _LIGHT_PALETTE is only ever captured once."""
    global _LIGHT_PALETTE
    app.setStyle("Fusion")
    if _LIGHT_PALETTE is None:
        _LIGHT_PALETTE = app.style().standardPalette()
    app.setPalette(_build_dark_palette() if dark else _LIGHT_PALETTE)


class SettingsDialog(QDialog):
    """Tabbed settings dialog: General (app-wide behavior, including the
    AoE2DE install path -- not its own tab since it's a one-time setup
    step, not a recurring one), Appearance (theme, render quality), and
    Keybinds. Split into tabs so more settings in any category have an
    obvious place to land without reshaping this dialog."""

    def __init__(self, parent: "ViewerWindow"):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 360)
        self._window = parent

        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_appearance_tab(), "Appearance")
        tabs.addTab(self._build_keybinds_tab(), "Keybinds")

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)

    def _build_general_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        zoom_label = QLabel("Mouse wheel zoom:")
        layout.addWidget(zoom_label)
        self.zoom_cursor_radio = QRadioButton("Zoom in centered on mouse cursor location")
        self.zoom_view_center_radio = QRadioButton("Zoom in overall (view center)")
        zoom_group = QButtonGroup(tab)
        zoom_group.addButton(self.zoom_cursor_radio)
        zoom_group.addButton(self.zoom_view_center_radio)
        if settings.get_zoom_centered_on_cursor():
            self.zoom_cursor_radio.setChecked(True)
        else:
            self.zoom_view_center_radio.setChecked(True)
        # Connecting only one radio's toggled is enough: within an exclusive
        # QButtonGroup, this one flips to False exactly when the other flips
        # to True, so both directions are covered by this single signal.
        self.zoom_cursor_radio.toggled.connect(self._on_zoom_mode_toggled)
        layout.addWidget(self.zoom_cursor_radio)
        layout.addWidget(self.zoom_view_center_radio)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)
        layout.addWidget(divider)

        self._add_install_section(layout)

        layout.addStretch(1)
        return tab

    def _on_zoom_mode_toggled(self, centered_on_cursor: bool) -> None:
        settings.set_zoom_centered_on_cursor(centered_on_cursor)
        self._window.map_view.set_zoom_anchor_mode(centered_on_cursor)
        mode = "centered on mouse cursor" if centered_on_cursor else "overall (view center)"
        self._window._log_status(f"Zoom mode: {mode}")

    def _build_appearance_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        dark_checkbox = QCheckBox("Dark mode")
        dark_checkbox.setChecked(settings.get_dark_mode())
        dark_checkbox.setToolTip(
            "App chrome only (menus, dialogs, toolbars) -- the map view's own "
            "colors (real terrain textures, unit dots, hover highlights) are "
            "unaffected, since they represent game data or are already dark."
        )
        dark_checkbox.toggled.connect(self._on_dark_mode_toggled)
        layout.addWidget(dark_checkbox)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Graphics quality:"))
        self.graphics_quality_slider = QSlider(Qt.Horizontal)
        self.graphics_quality_slider.setRange(settings.GRAPHICS_QUALITY_MIN, settings.GRAPHICS_QUALITY_MAX)
        self.graphics_quality_slider.setTickInterval(1)
        self.graphics_quality_slider.setTickPosition(QSlider.TicksBelow)
        self.graphics_quality_slider.setSingleStep(1)
        self.graphics_quality_slider.setPageStep(1)
        self.graphics_quality_slider.setToolTip(
            "Scales map render resolution (see render.tile_pixels_for_map) -- "
            "applies to the next render. Enhanced only doubles resolution on "
            "large maps (already-downscaled); on small maps it's a no-op, "
            "since doubling their base resolution is well past what's been "
            "measured safe for memory."
        )
        self.graphics_quality_slider.setValue(settings.get_graphics_quality())
        self.graphics_quality_label = QLabel()
        self._update_graphics_quality_label(settings.get_graphics_quality())
        self._graphics_quality_apply_timer = QTimer(self)
        self._graphics_quality_apply_timer.setSingleShot(True)
        self._graphics_quality_apply_timer.timeout.connect(self._apply_graphics_quality)
        self.graphics_quality_slider.valueChanged.connect(self._on_graphics_quality_slider_changed)
        quality_row.addWidget(self.graphics_quality_slider, stretch=1)
        quality_row.addWidget(self.graphics_quality_label)
        layout.addLayout(quality_row)

        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("Stepped elevation height:"))
        self.elev_step_slider = QSlider(Qt.Horizontal)
        # Value space is the 1-based stop index, not the pct -- setSingleStep
        # alone only governs arrow keys and the wheel, so a drag would still
        # produce arbitrary off-stop pcts via QStyle::sliderValueFromPosition.
        # Same shape as graphics_quality_slider above.
        self.elev_step_slider.setRange(1, len(settings.ELEV_STEP_PCT_STOPS))
        self.elev_step_slider.setTickInterval(1)
        self.elev_step_slider.setTickPosition(QSlider.TicksBelow)
        self.elev_step_slider.setSingleStep(1)
        self.elev_step_slider.setPageStep(1)
        self.elev_step_slider.setToolTip(
            "Stepped rendering mode only -- how tall one elevation level's "
            "displacement reads on screen. Default is Tall; the range above "
            "it is real headroom, not just cosmetic overshoot -- AoE2's real "
            "elevation transitions are smooth multi-tile ramps rather than a "
            "single hard edge, which is what keeps a taller step from making "
            "tiles hide behind their taller neighbors as easily as a single "
            "sharp step would."
        )
        self.elev_step_slider.setValue(settings.elev_step_index(settings.get_elev_step_pct()))
        self.elev_step_value_label = QLabel()
        self._update_elev_step_label(settings.get_elev_step_pct())
        self._elev_step_apply_timer = QTimer(self)
        self._elev_step_apply_timer.setSingleShot(True)
        self._elev_step_apply_timer.timeout.connect(self._apply_elev_step_pct)
        self.elev_step_slider.valueChanged.connect(self._on_elev_step_slider_changed)
        self.elev_step_slider.sliderReleased.connect(self._apply_elev_step_pct)
        height_row.addWidget(self.elev_step_slider, stretch=1)
        height_row.addWidget(self.elev_step_value_label)
        layout.addLayout(height_row)

        layout.addStretch(1)
        return tab

    def _on_dark_mode_toggled(self, enabled: bool) -> None:
        settings.set_dark_mode(enabled)
        apply_theme(QApplication.instance(), enabled)
        self._window._log_status(f"Dark mode: {'on' if enabled else 'off'}")

    def _update_graphics_quality_label(self, quality: int) -> None:
        self.graphics_quality_label.setText(settings.GRAPHICS_QUALITY_LABELS[quality])

    def _on_graphics_quality_slider_changed(self, value: int) -> None:
        self._update_graphics_quality_label(value)
        # Debounced (not applied on every intermediate value) since a full
        # re-render is expensive and dragging the slider across all 4 stops
        # would otherwise fire one per stop -- see the elev_step slider's
        # matching comment below, same reasoning.
        self._graphics_quality_apply_timer.start(200)

    def _apply_graphics_quality(self) -> None:
        value = self.graphics_quality_slider.value()
        settings.set_graphics_quality(value)
        self._window._log_status(f"Graphics quality: {settings.GRAPHICS_QUALITY_LABELS[value]}")
        self._window.refresh_map()

    def _update_elev_step_label(self, pct: int) -> None:
        suffix = " (Tall, default)" if pct == iso_geometry.ELEV_STEP_DEFAULT_PCT else ""
        self.elev_step_value_label.setText(f"{pct}%{suffix}")

    def _on_elev_step_slider_changed(self, value: int) -> None:
        # value is a stop index -- _update_elev_step_label still takes a pct,
        # so its "(Tall, default)" test against ELEV_STEP_DEFAULT_PCT holds.
        self._update_elev_step_label(settings.elev_step_pct_for_index(value))
        # Each application is a full Stepped-mode re-render
        # (elevations_and_proj + IsoChunkCache rebuild) -- expensive enough
        # on a real map that applying it live per-pixel would make dragging
        # itself laggy. While the mouse is down, just update the label and
        # wait for sliderReleased to apply. Keyboard arrow presses have no
        # press/release pair, so fall back to a short debounce timer for
        # those instead.
        if self.elev_step_slider.isSliderDown():
            return
        self._elev_step_apply_timer.start(200)

    def _apply_elev_step_pct(self) -> None:
        value = settings.elev_step_pct_for_index(self.elev_step_slider.value())
        settings.set_elev_step_pct(value)
        self._window._log_status(f"Stepped elevation height: {value}%")
        self._window.refresh_map()

    # action_id prefix (before the first "_") -> section header text. Covers
    # today's sections (REBINDABLE_ACTIONS's "file_*"/"edit_*"/"view_*"/
    # "help_*"/"mode_*"/"tool_*"/"adjust_*" entries); an unlisted future
    # prefix still gets a section of its own, just titled from the raw
    # prefix instead of a curated name.
    _KEYBIND_SECTION_TITLES = {
        "file": "File",
        "edit": "Edit",  # renamed from "Copy/Paste" -- now covers the whole Edit menu
        "view": "View",
        "help": "Help",
        "mode": "Modes",
        "tool": "Tools",
        "adjust": "Tool Value",
    }

    def _build_keybinds_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)

        rows_widget = QWidget()
        grid = QGridLayout(rows_widget)
        grid.setColumnStretch(1, 1)

        self._keybind_edits: dict[str, QKeySequenceEdit] = {}
        row = 0
        current_section = None
        for action_id, label, _default in settings.REBINDABLE_ACTIONS:
            section = action_id.split("_", 1)[0]
            if section != current_section:
                if current_section is not None:
                    divider = QFrame()
                    divider.setFrameShape(QFrame.HLine)
                    divider.setFrameShadow(QFrame.Sunken)
                    grid.addWidget(divider, row, 0, 1, 4)
                    row += 1
                section_label = QLabel(f"<b>{self._KEYBIND_SECTION_TITLES.get(section, section.title())}</b>")
                grid.addWidget(section_label, row, 0, 1, 4)
                row += 1
                current_section = section

            grid.addWidget(QLabel(label), row, 0)

            edit = QKeySequenceEdit(QKeySequence(settings.get_keybind(action_id)))
            edit.keySequenceChanged.connect(
                lambda seq, aid=action_id: self._on_keybind_changed(aid, seq)
            )
            # QKeySequenceEdit has no setAlignment of its own -- it wraps an
            # internal QLineEdit, which does.
            line_edit = edit.findChild(QLineEdit)
            if line_edit is not None:
                line_edit.setAlignment(Qt.AlignCenter)
            grid.addWidget(edit, row, 1)
            self._keybind_edits[action_id] = edit

            default_btn = QPushButton("Default")
            default_btn.clicked.connect(lambda _checked, aid=action_id: self._reset_keybind(aid))
            grid.addWidget(default_btn, row, 2)

            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda _checked, aid=action_id: self._keybind_edits[aid].clear())
            grid.addWidget(clear_btn, row, 3)

            row += 1

        grid.setRowStretch(row, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rows_widget)
        outer_layout.addWidget(scroll)
        return tab

    def _on_keybind_changed(self, action_id: str, key_sequence: QKeySequence) -> None:
        text = key_sequence.toString()
        settings.set_keybind(action_id, text)
        self._window.apply_keybind(action_id)
        self._window._log_status(f"Keybind changed: {action_id} -> {text or '(cleared)'}")

    def _reset_keybind(self, action_id: str) -> None:
        # QKeySequenceEdit.setKeySequence() emits keySequenceChanged, so
        # _on_keybind_changed() handles the actual persist + apply + log --
        # this only needs to drive the widget.
        self._keybind_edits[action_id].setKeySequence(QKeySequence(settings.get_default_keybind(action_id)))

    def _add_install_section(self, layout: QVBoxLayout) -> None:
        """Appends the AoE2DE install-path controls into the given layout --
        folded into the General tab (moved out of its own "Game Resources"
        tab) since this is a one-time setup step, not a recurring setting
        someone flips back and forth."""
        install_label = QLabel("AoE2DE install path (optional, for real terrain colors):")
        install_label.setWordWrap(True)
        layout.addWidget(install_label)

        install_row = QHBoxLayout()
        self.install_path_edit = QLineEdit()
        existing_install = asset_source.get_install_path()
        if existing_install is not None:
            self.install_path_edit.setText(str(existing_install))
        install_row.addWidget(self.install_path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_install_path)
        install_row.addWidget(browse_btn)
        layout.addLayout(install_row)

        load_install_btn = QPushButton("Load")
        load_install_btn.clicked.connect(self.load_install_path)
        layout.addWidget(load_install_btn)

        self.install_status_label = QLabel("")
        self.install_status_label.setWordWrap(True)
        layout.addWidget(self.install_status_label)

        # Reflects whatever's actually configured right now (dialog-open
        # time), not just the outcome of the last "Load" click in this
        # session -- e.g. a path set via AOE2DE_INSTALL_PATH or a prior
        # session's config.yaml would otherwise leave this blank forever.
        if existing_install is not None:
            ok, message = asset_source.validate_install_path(existing_install)
            self._set_install_status(message, ok=ok)
        else:
            self._set_install_status("No install path configured -- using flat fallback colors.", ok=False)

    def browse_install_path(self) -> None:
        start_dir = self.install_path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select AoE2DE install folder", start_dir)
        if path:
            self.install_path_edit.setText(path)

    def load_install_path(self) -> None:
        text = self.install_path_edit.text().strip()
        if not text:
            self._set_install_status("Enter or browse to an AoE2DE install path first.", ok=False)
            return

        ok, message = asset_source.validate_install_path(Path(text))
        if not ok:
            self._set_install_status(message, ok=False)
            self._window._log_status(f"Install path rejected ({text}): {message}")
            return

        path = Path(text)
        asset_source.set_install_path_override(path)
        asset_source.save_install_path_to_config(path)
        self._set_install_status(message, ok=True)
        self._window._log_status(f"Install path set to {path}: {message}")
        self._window.refresh_map()

    def _set_install_status(self, message: str, ok: bool) -> None:
        color = STATUS_OK_COLOR if ok else STATUS_ERROR_COLOR
        self.install_status_label.setStyleSheet(f"color: {color};")
        self.install_status_label.setText(message)


class DebugLogDialog(QDialog):
    """Read-only viewer over debug_log's in-memory buffer. A snapshot at open
    time (and after Refresh), not a live tail -- simplest thing that's useful
    for "what did the app just do," not meant as a full log console."""

    def __init__(self, parent: "ViewerWindow"):
        super().__init__(parent)
        self.setWindowTitle("Debug Log")
        self.resize(640, 400)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Monospace"))
        self._refresh()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)

        btn_row = QHBoxLayout()
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.text, stretch=1)
        layout.addLayout(btn_row)
        layout.addWidget(buttons)

    def _refresh(self) -> None:
        self.text.setPlainText(debug_log.get_log_text())
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.End)
        self.text.setTextCursor(cursor)

    def _clear(self) -> None:
        debug_log.clear()
        self._refresh()


class ViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        width, height = settings.get_window_size()
        self.resize(width, height)
        self.setMinimumSize(settings.MIN_WINDOW_WIDTH, settings.MIN_WINDOW_HEIGHT)

        self.scenario: LoadedScenario | None = None
        # True while the current scenario came from File > New and has never been
        # saved: its LoadedScenario.path is UNTITLED_PATH (a display-only sentinel,
        # not a real file), so Save As must not use the normal "<stem>_edited"
        # default. Cleared by a successful Save As, by File > Open, and by Close.
        self._untitled = False
        # Holds whichever style's chunk cache is current -- an IsoChunkCache
        # (Stepped), a FlatChunkCache (Flat, since Phase B-E; previously a
        # separate self._img numpy array, retired in that phase), or a
        # SlopedChunkCache (Sloped, since Track C3). Set by
        # _render_current(), matching self._terrain_style/self.map_view's
        # own paired state.
        self._cache: IsoChunkCache | FlatChunkCache | SlopedChunkCache | None = None
        # Stepped mode only, set together with self._cache by
        # _render_current() -- the exact elevation snapshot and
        # IsoProjection self._cache was built from, and the SAME
        # array/object self.map_view holds (passed by reference into
        # MapView.set_source(), never copied) so dirty_screen_bbox_iso()'s
        # in-place elevations mutation is visible to the viewer's own
        # hit-testing, and to self._cache itself, without a separate update
        # step (Risk #6 in the parent plan). None in Flat mode.
        self._iso_elevations: np.ndarray | None = None
        self._iso_proj: iso_geometry.IsoProjection | None = None
        self.mode = "view"
        self._current_tool = "pan"
        # "flat" or "stepped" -- see the Elevation View toolbar combo built
        # in _build_toolbar(). Defaults to "stepped" (Flat alone no longer
        # shows elevation at all -- see render_tile()'s docstring). Persists
        # across a File > Open (a freshly loaded scenario renders in
        # whichever style was already selected), unlike self._current_tool
        # which File > Close resets to "pan".
        self._terrain_style = "stepped"
        # True for the duration of load_scenario()/_render_current()'s
        # blocking work -- an explicit guard against re-entering any of
        # load_scenario/refresh_map/on_terrain_style_changed while one is
        # already running, on top of (not instead of) setEnabled(False):
        # that call only blocks Qt from *delivering new input events* to
        # disabled widgets, it doesn't stop a direct/programmatic call into
        # one of these methods (e.g. from a QTimer callback) from reaching
        # them regardless of enabled state. This flag is what actually
        # makes such a call a no-op rather than a reentrant interleave.
        self._busy = False
        # One history per loaded scenario -- load_scenario()/close_scenario()
        # call .reset() so it never leaks state across files. See
        # descape.edit_history for why this is the single choke point every
        # terrain/elevation edit goes through.
        self.edit_history = EditHistory()
        # v2.7 copy/paste: a single clipboard slot, not a
        # manager -- overwritten on every Copy. None means empty. Tagged by
        # "kind" so copy/paste can dispatch by content rather than by
        # whatever tool happened to be active when it was copied:
        # {"kind": "terrain", "terrain_id": ..., "layer": ...} or
        # {"kind": "elevation", "value": ...}. Set by copy_tile(); read (and
        # its kind checked against the active tool) by paste_tile() and
        # _update_tool_enabled()'s paste-gating.
        self._clipboard: dict | None = None
        # Updated on every mouse move by on_hover() regardless of whether a
        # scenario is loaded -- copy/paste fire from a keyboard shortcut, not
        # a mouse click, so they need "what tile is under the mouse right
        # now" captured at hover time rather than at the moment the key is
        # pressed (MapView has no notion of "current tile" itself).
        self._hover_tile: tuple[int, int] | None = None
        self._update_title()

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()

        self.hover_label = QLabel("Hover the map for tile info")
        self.hover_label.setWordWrap(True)
        left.addWidget(self.hover_label)

        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        left.addWidget(self.info, stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(340)
        content_layout.addWidget(left_widget)

        self.map_view = MapView(
            self.on_hover,
            self.on_edit_stroke_start,
            self.on_edit_stroke_tile,
            self.on_edit_stroke_end,
            self.on_fill,
        )
        content_layout.addWidget(self.map_view, stretch=1)

        # Always-visible short status history, distinct from both the
        # transient single-line QMainWindow.statusBar() message and the
        # Help > Debug Log dialog (a separate popup, not always on screen).
        # _log_status() is the one place that feeds all three destinations
        # that matter for a given message.
        self.status_log = QPlainTextEdit()
        self.status_log.setReadOnly(True)
        line_height = self.status_log.fontMetrics().lineSpacing()
        self.status_log.setFixedHeight(line_height * 4 + 12)
        self.status_log.setMaximumBlockCount(200)

        central = QWidget()
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(content, stretch=1)
        outer_layout.addWidget(self.status_log)

        self.setCentralWidget(central)

        self._build_menu_bar()
        self._build_toolbar()
        self._build_status_bar()
        self._build_keybind_actions()
        self._update_tool_enabled()

        # Connected only now that self.map_view exists, then set to match
        # MapView's own default (isometric on) so the action reflects reality
        # without needing map_view to already exist at widget-creation time.
        self.iso_action.toggled.connect(lambda checked: self.map_view.set_isometric(checked))
        self.iso_action.setChecked(self.map_view._isometric)
        # Has nothing to do while Elevation View = Stepped (today's default
        # -- see self._terrain_style above), same gating on_terrain_style_
        # changed() applies on every later switch.
        self.iso_action.setEnabled(self._terrain_style == "flat")

    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("&File")
        # One action per real AoE2:DE map size (descape.scenario_new.STANDARD_MAP_SIZES
        # -- confirmed empirically against every real scenario in examples/, see that
        # module's docstring), each generated on demand from the shipped 120x120 donor
        # rather than loaded from its own file. new_action stays the 120x120 entry, but
        # Ctrl+N lives on new_custom_action below instead -- picking a size is more
        # useful as the keyboard-driven default than always defaulting to Tiny.
        new_menu = file_menu.addMenu("&New Map")
        self.new_action = QAction(
            f"{BLANK_TEMPLATE_TILES}×{BLANK_TEMPLATE_TILES} ({STANDARD_MAP_SIZE_NAMES[BLANK_TEMPLATE_TILES]})",
            self,
        )
        self.new_action.triggered.connect(lambda: self.new_map(BLANK_TEMPLATE_TILES))
        new_menu.addAction(self.new_action)
        self.new_size_actions = [self.new_action]
        for tiles in STANDARD_MAP_SIZES:
            if tiles == BLANK_TEMPLATE_TILES:
                continue
            action = QAction(f"{tiles}×{tiles} ({STANDARD_MAP_SIZE_NAMES[tiles]})", self)
            # checked=False must come first: QAction.triggered passes a
            # checked: bool positional arg, and PyQt5 binds it into whichever
            # parameter is first available -- a lambda with only `tiles=tiles`
            # gets `checked` bound into `tiles` instead of the captured
            # default, silently passing tiles=0/False (confirmed: hangs the
            # test suite on a QMessageBox.critical() for the resulting
            # nonexistent blank_0x0.aoe2scenario, with nothing able to click
            # "OK" in an offscreen run).
            action.triggered.connect(lambda checked=False, tiles=tiles: self.new_map(tiles))
            new_menu.addAction(action)
            self.new_size_actions.append(action)
        new_menu.addSeparator()
        # Deliberately NOT appended to new_size_actions -- that list means "one
        # action per preset size", and new_map_custom() is the only path that can
        # ever pop the >LARGE_MAP_CONFIRM_TILES confirmation dialog (see its own
        # docstring for why the presets, including 480, never confirm).
        self.new_custom_action = QAction("&Custom size…", self)
        # Same checked=False trap as above -- applies even with no captured value,
        # since QAction.triggered still passes a positional bool.
        self.new_custom_action.triggered.connect(lambda checked=False: self.new_map_custom())
        new_menu.addAction(self.new_custom_action)
        self.open_action = QAction("&Open .aoe2scenario…", self)
        self.open_action.triggered.connect(self.open_file)
        file_menu.addAction(self.open_action)
        self.close_action = QAction("&Close Map", self)
        self.close_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.close_action.triggered.connect(self.close_scenario)
        file_menu.addAction(self.close_action)
        file_menu.addSeparator()
        # Save As only, deliberately -- no in-place Save. v2's write path is
        # shipping this way until in-game round-trip verification passes
        # (currently unconfirmed); a bug that can only ever damage a file
        # the user explicitly named as a new destination is a much smaller
        # blast radius than one that can silently corrupt their only copy
        # of a scenario. Revisit adding a plain Save once that verification
        # passes.
        self.save_as_action = QAction("Save &As…", self)
        self.save_as_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.save_as_action.triggered.connect(self.save_as)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        self.exit_action = QAction("E&xit", self)
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setEnabled(False)  # re-gated by _update_edit_actions()
        self.undo_action.triggered.connect(self.undo)
        edit_menu.addAction(self.undo_action)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setEnabled(False)  # re-gated by _update_edit_actions()
        self.redo_action.triggered.connect(self.redo)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        # v2.7 copy/paste. No setShortcut() here, or anywhere else
        # in this whole menu -- every Edit-menu action's shortcut comes from
        # the settings-backed keybind system instead (see
        # _build_keybind_actions()/apply_keybind()), same as every tool
        # action, so its shortcut comes from settings.REBINDABLE_ACTIONS's
        # matching "edit_*" entry.
        self.copy_action = QAction("&Copy Tile", self)
        self.copy_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.copy_action.triggered.connect(self.copy_tile)
        edit_menu.addAction(self.copy_action)
        self.paste_action = QAction("&Paste Tile", self)
        self.paste_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.paste_action.triggered.connect(self.paste_tile)
        edit_menu.addAction(self.paste_action)
        edit_menu.addSeparator()
        self.settings_action = QAction("&Settings…", self)
        self.settings_action.triggered.connect(self._show_settings)
        edit_menu.addAction(self.settings_action)

        view_menu = menu_bar.addMenu("&View")
        self.iso_action = QAction("&Isometric View (game-style)", self)
        self.iso_action.setCheckable(True)
        self.iso_action.setToolTip(
            "Flat Elevation View's own view rotation. Stepped already renders "
            "its real per-tile projection into the image, so this has "
            "nothing left to do and is disabled while Elevation View = Stepped."
        )
        view_menu.addAction(self.iso_action)

        help_menu = menu_bar.addMenu("&Help")
        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)
        help_menu.addSeparator()
        self.debug_log_action = QAction("&Debug Log", self)
        self.debug_log_action.triggered.connect(self._show_debug_log)
        help_menu.addAction(self.debug_log_action)

    def _build_toolbar(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)

        toolbar.addWidget(QLabel(" Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["View", "Edit"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        toolbar.addWidget(self.mode_combo)
        toolbar.addSeparator()

        # Separate from Mode/the tool group -- Elevation View picks the
        # rendering pipeline (Flat's plain, un-displaced terrain vs.
        # Stepped's real per-tile Z-height compositor -- the only one of the
        # two that actually shows elevation at all now that Phase 3's
        # follow-up removed the old brightness-based elevation hint), an
        # orthogonal axis to View/Edit mode or which brush tool is active.
        # Internal identifiers/log lines still say "terrain style" (the
        # underlying concept -- which renderer is active); only this
        # user-facing label and combo text changed to "Elevation View".
        # Defaults to Stepped, not Flat -- Flat alone no longer conveys
        # elevation at all, so Stepped is the more useful default view.
        toolbar.addWidget(QLabel(" Elevation View: "))
        self.terrain_style_combo = QComboBox()
        self.terrain_style_combo.addItems(["Flat", "Stepped", "Sloped"])
        self.terrain_style_combo.setCurrentText("Stepped")  # before connect(): no spurious signal
        self.terrain_style_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.terrain_style_combo.currentTextChanged.connect(self.on_terrain_style_changed)
        toolbar.addWidget(self.terrain_style_combo)
        toolbar.addSeparator()

        tool_group = QActionGroup(self)
        tool_group.setExclusive(True)

        self.pan_action = QAction("Pan", self)
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.pan_action.setEnabled(False)  # re-enabled by _update_tool_enabled() once a map loads
        self.pan_action.setToolTip("Drag to pan the map")
        self.pan_action.toggled.connect(lambda on: on and self._on_tool_selected("pan"))
        tool_group.addAction(self.pan_action)
        toolbar.addAction(self.pan_action)

        self.terrain_action = QAction("Terrain", self)
        self.terrain_action.setCheckable(True)
        self.terrain_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.terrain_action.setToolTip(
            "Paint the selected terrain type over the brush footprint (Edit mode) -- drag to paint a trail"
        )
        self.terrain_action.toggled.connect(lambda on: on and self._on_tool_selected("terrain"))
        tool_group.addAction(self.terrain_action)
        toolbar.addAction(self.terrain_action)

        self.fill_action = QAction("Paint Can", self)
        self.fill_action.setCheckable(True)
        self.fill_action.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.fill_action.setToolTip(
            "Flood-fill the connected region of matching terrain with the selected "
            "terrain type (Edit mode) -- one click, no drag"
        )
        self.fill_action.toggled.connect(lambda on: on and self._on_tool_selected("fill"))
        tool_group.addAction(self.fill_action)
        toolbar.addAction(self.fill_action)

        self.elevation_action = QAction("Elevate", self)
        self.elevation_action.setCheckable(True)
        self.elevation_action.setEnabled(False)
        self.elevation_action.setToolTip(
            "Left click/drag to raise the brush footprint's elevation by 1; right click/drag "
            "(or Shift+left) to lower (Edit mode, square maps only)"
        )
        self.elevation_action.toggled.connect(lambda on: on and self._on_tool_selected("elevation"))
        tool_group.addAction(self.elevation_action)
        toolbar.addAction(self.elevation_action)

        self.set_level_action = QAction("Set Elevation", self)
        self.set_level_action.setCheckable(True)
        self.set_level_action.setEnabled(False)
        self.set_level_action.setToolTip(
            "Click/drag to set the brush footprint's elevation to the level below "
            "(Edit mode, square maps only)"
        )
        self.set_level_action.toggled.connect(lambda on: on and self._on_tool_selected("set_level"))
        tool_group.addAction(self.set_level_action)
        toolbar.addAction(self.set_level_action)

        # addWidget()/addSeparator() both hand back the QAction Qt actually
        # lays the item out with -- captured here (not discarded) because
        # hiding a toolbar-embedded widget has to go through that action's
        # setVisible(), not the widget's own hide()/setVisible(), or the
        # action's layout slot is left behind. Visibility (which of these
        # three shows at all) and enabled state are both re-gated by
        # _update_tool_enabled() per the active tool's param_widget.
        self.tool_param_separator_action = toolbar.addSeparator()
        self.terrain_param_label_action = toolbar.addWidget(QLabel(" Terrain type: "))
        self.terrain_combo = QComboBox()
        self.terrain_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        for terrain_id in sorted(TerrainId, key=lambda t: t.name):
            self.terrain_combo.addItem(name_for_terrain_id(terrain_id.value), terrain_id.value)
        self.terrain_param_combo_action = toolbar.addWidget(self.terrain_combo)

        self.level_param_label_action = toolbar.addWidget(QLabel(" Level: "))
        self.elevation_level_spin = QSpinBox()
        self.elevation_level_spin.setRange(0, ELEVATION_LEVEL_MAX)
        self.elevation_level_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.level_param_spin_action = toolbar.addWidget(self.elevation_level_spin)

        # Brush size/shape -- shown for any ToolDef.supports_brush tool
        # (Terrain, Elevate, Set Elevation), same visibility/enabled
        # convention as the two params above. Deliberately session-only, not
        # read from or written to settings.py's config: every launch starts
        # at BRUSH_SIZE_MIN/square, so nothing here persists.
        self.brush_param_label_action = toolbar.addWidget(QLabel(" Brush: "))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(brush.BRUSH_SIZE_MIN, brush.BRUSH_SIZE_MAX)
        self.brush_size_spin.setValue(brush.BRUSH_SIZE_MIN)
        self.brush_size_spin.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_size_spin.valueChanged.connect(self._on_brush_changed)
        self.brush_size_spin_action = toolbar.addWidget(self.brush_size_spin)

        self.brush_shape_combo = QComboBox()
        self.brush_shape_combo.addItem("Square", brush.BRUSH_SHAPE_SQUARE)
        self.brush_shape_combo.addItem("Circle", brush.BRUSH_SHAPE_CIRCLE)
        self.brush_shape_combo.setEnabled(False)  # re-gated by _update_tool_enabled()
        self.brush_shape_combo.currentIndexChanged.connect(self._on_brush_changed)
        self.brush_shape_combo_action = toolbar.addWidget(self.brush_shape_combo)

    def _build_status_bar(self) -> None:
        self.mode_status_label = QLabel()
        self.statusBar().addPermanentWidget(self.mode_status_label)
        self._update_mode_status()

    def _update_mode_status(self) -> None:
        tool = _TOOL_LABELS.get(self._current_tool, self._current_tool)
        style = self._terrain_style.capitalize()
        self.mode_status_label.setText(f"  Mode: {self.mode.capitalize()}  |  Tool: {tool}  |  Style: {style}  ")

    def _adjust_tool_value(self, delta: int) -> None:
        """Backs the "]"/"[" tool_value_inc/dec_action shortcuts -- see their
        own comment in _build_keybind_actions for the brush-vs-level
        priority rule. Reads each spinbox's live isEnabled() (kept correct
        by _update_tool_enabled, which gates the actions themselves the same
        way) rather than re-deriving BRUSH_TOOLS/_TOOL_PARAM membership
        here, so there is exactly one place that decides which widget is
        "the active tool's value" at any given moment."""
        if self.brush_size_spin.isEnabled():
            spin = self.brush_size_spin
        elif self.elevation_level_spin.isEnabled():
            spin = self.elevation_level_spin
        else:
            return
        spin.stepUp() if delta > 0 else spin.stepDown()

    def _build_keybind_actions(self) -> None:
        # Mode switching has no QAction of its own (it's a toolbar QComboBox)
        # -- these two exist purely to carry a shortcut. Not added to any
        # menu, just registered on the window (addAction) so Qt still
        # processes the shortcut globally while the window has focus.
        self.mode_view_action = QAction("Switch to View Mode", self)
        self.mode_view_action.triggered.connect(lambda: self.mode_combo.setCurrentText("View"))
        self.addAction(self.mode_view_action)

        self.mode_edit_action = QAction("Switch to Edit Mode", self)
        self.mode_edit_action.triggered.connect(lambda: self.mode_combo.setCurrentText("Edit"))
        self.addAction(self.mode_edit_action)

        # The active tool's own "primary value" -- brush size for any
        # supports_brush tool (Terrain, Elevate, Set Elevation), falling back
        # to Set Elevation's Level spinbox when the active tool has no brush.
        # Brush wins the overlap on Set Elevation, which has both: this is
        # what "]"/"[" -- picked as brush-size-style inc/dec keys, see
        # REBINDABLE_ACTIONS's adjust_increment/adjust_decrement comment in
        # settings.py -- were always meant to drive. Gated the same way the
        # spinboxes themselves are (see _update_tool_enabled) so the
        # shortcut is dead whenever neither would apply. stepUp/stepDown
        # clamp to the target spinbox's own range, same as clicking its
        # arrows would.
        self.tool_value_inc_action = QAction("Increase Tool Value", self)
        self.tool_value_inc_action.triggered.connect(lambda: self._adjust_tool_value(+1))
        self.addAction(self.tool_value_inc_action)

        self.tool_value_dec_action = QAction("Decrease Tool Value", self)
        self.tool_value_dec_action.triggered.connect(lambda: self._adjust_tool_value(-1))
        self.addAction(self.tool_value_dec_action)

        # Every tool's toolbar action reused directly -- no separate action
        # needed, just give it a shortcut too. Looked up via getattr rather
        # than hand-listed so this stays in sync with settings.TOOLS
        # automatically: a tool registered there without a matching
        # self.<tool_id>_action (built in _build_toolbar, before this runs)
        # is a wiring bug, and this is deliberately not defensive about it
        # -- getattr with no default raises immediately, which is the point
        # (see settings.ToolDef's own docstring).
        self._keybind_actions = {
            "file_new": self.new_custom_action,
            "file_open": self.open_action,
            "file_close": self.close_action,
            "file_save_as": self.save_as_action,
            "file_exit": self.exit_action,
            "edit_undo": self.undo_action,
            "edit_redo": self.redo_action,
            "edit_copy": self.copy_action,
            "edit_paste": self.paste_action,
            "edit_settings": self.settings_action,
            "view_isometric": self.iso_action,
            "help_about": self.about_action,
            "help_debug_log": self.debug_log_action,
            "mode_view": self.mode_view_action,
            "mode_edit": self.mode_edit_action,
            "adjust_increment": self.tool_value_inc_action,
            "adjust_decrement": self.tool_value_dec_action,
        }
        for tool in settings.TOOLS:
            self._keybind_actions[f"tool_{tool.tool_id}"] = getattr(self, f"{tool.tool_id}_action")
        for action_id in self._keybind_actions:
            self.apply_keybind(action_id)

    def apply_keybind(self, action_id: str) -> None:
        action = self._keybind_actions.get(action_id)
        if action is None:
            return
        key_text = settings.get_keybind(action_id)
        action.setShortcut(QKeySequence(key_text) if key_text else QKeySequence())

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About",
            "DEscape by Combinebobnt\n\n"
            "External map editor for Age of Empires 2: Definitive Edition scenarios.\n\n"
            "Copyright (C) 2026 Combinebobnt\n"
            "Licensed under the GNU General Public License v3.0 or later. ",
        )

    def on_mode_changed(self, mode_text: str) -> None:
        self.mode = mode_text.lower()
        if self.mode == "view":
            self.pan_action.setChecked(True)
        self._update_tool_enabled()
        self._update_mode_status()
        self._log_status(f"Mode changed to {mode_text}")

    def _update_tool_enabled(self) -> None:
        # Tools (and Close/Save As) have nothing to act on before a map is
        # loaded -- grayed out rather than left clickable-but-pointless.
        # Tool params (Terrain type / Level) go further and hide outright
        # when the active tool doesn't read them at all (see the
        # _TOOL_PARAM block below) -- greying still covers the "wrong tool
        # to use right now" case (no map / View mode / non-square map). The
        # edit tools are additionally gated on Edit mode; Elevation and Set
        # Level further require terrain_write_supported (the load-time
        # terrain-block verification passed -- see scenario_io.py) and
        # map_is_square (MapManager.set_elevation, and even single-tile
        # elevation edits via elevation_tools.set_tile_elevation, raise
        # ValueError on any non-square map -- confirmed against
        # AoE2ScenarioParser's source, not just an assumption). Terrain
        # painting has no such constraint, only terrain_write_supported --
        # Paint Can shares that same gate (write_ok, not elevation_ok): it
        # never calls get_tile()/set_elevation() either, so squareness is
        # irrelevant to it too.
        has_map = self.scenario is not None
        write_ok = has_map and self.scenario.terrain_write_supported
        elevation_ok = write_ok and self.scenario.map_is_square
        # Phase 3's decision #6 scope cut (Stepped had no bounded
        # incremental redraw, so editing there meant a full ~1-2s
        # re-render per touched tile) is lifted as of Phase 4 --
        # refresh_region_iso() gives Stepped its own bounded redraw now,
        # same as Flat's own. Editing is gated the same way in both styles
        # from here on.
        editable = self.mode == "edit"
        # Sloped has no hit-testing yet (Track C4), so none of the paint/
        # elevation tools can resolve a click to a tile there -- disabled
        # outright until C4 lands, regardless of write_ok/elevation_ok.
        sloped_editable = self._terrain_style != "sloped"

        self.pan_action.setEnabled(has_map)
        self.terrain_action.setEnabled(has_map and editable and write_ok and sloped_editable)
        self.fill_action.setEnabled(has_map and editable and write_ok and sloped_editable)
        self.elevation_action.setEnabled(has_map and editable and elevation_ok and sloped_editable)
        self.set_level_action.setEnabled(has_map and editable and elevation_ok and sloped_editable)
        self.close_action.setEnabled(has_map)
        self.save_as_action.setEnabled(write_ok)
        self.terrain_style_combo.setEnabled(has_map)

        # Disabling a checked QAction doesn't uncheck it or notify MapView --
        # e.g. Elevation selected, then a non-square file is opened over it.
        # Left alone, the toolbar button would sit checked-but-greyed-out
        # while MapView still had "elevation" as its live _tool, so a click
        # on the map would still start a stroke and hit set_tile_elevation's
        # ValueError on the new file. Force back to Pan whenever the
        # currently-selected tool is no longer one of the enabled ones.
        current_action = {
            "terrain": self.terrain_action,
            "fill": self.fill_action,
            "elevation": self.elevation_action,
            "set_level": self.set_level_action,
        }.get(self._current_tool)
        if current_action is not None and not current_action.isEnabled():
            self.pan_action.setChecked(True)

        # Tool params (Terrain type / Level): which one shows, if either,
        # depends on the *final* self._current_tool for this call -- same
        # reason the copy/paste block below reads it only after the
        # forced-back-to-Pan block above has had its say. One boolean per
        # param feeds both the widget's visibility and its enabled state
        # (and, for Level, the ]/[ step-value keybinds too) so a hidden
        # param can never be left live behind the scenes.
        param = _TOOL_PARAM.get(self._current_tool, "")
        terrain_param_ok = param == "terrain" and (self.terrain_action.isEnabled() or self.fill_action.isEnabled())
        level_param_ok = param == "level" and self.set_level_action.isEnabled()
        # Brush size/shape -- reuses current_action (computed above for the
        # forced-back-to-Pan check) rather than re-deriving write_ok/
        # elevation_ok: brush_ok should track exactly whether the active
        # tool's own toolbar action is enabled, same as terrain_param_ok/
        # level_param_ok already do via terrain_action/fill_action/
        # set_level_action above. Elevate has no param_widget at all today
        # (param == ""), so this is the first param group it ever shows --
        # see the separator line below, which previously assumed Elevate
        # showed nothing.
        brush_ok = self._current_tool in BRUSH_TOOLS and current_action is not None and current_action.isEnabled()
        self.terrain_param_label_action.setVisible(terrain_param_ok)
        self.terrain_param_combo_action.setVisible(terrain_param_ok)
        self.terrain_combo.setEnabled(terrain_param_ok)
        self.level_param_label_action.setVisible(level_param_ok)
        self.level_param_spin_action.setVisible(level_param_ok)
        self.elevation_level_spin.setEnabled(level_param_ok)
        self.brush_param_label_action.setVisible(brush_ok)
        self.brush_size_spin_action.setVisible(brush_ok)
        self.brush_size_spin.setEnabled(brush_ok)
        self.brush_shape_combo_action.setVisible(brush_ok)
        self.brush_shape_combo.setEnabled(brush_ok)
        self.tool_value_inc_action.setEnabled(level_param_ok or brush_ok)
        self.tool_value_dec_action.setEnabled(level_param_ok or brush_ok)
        self.tool_param_separator_action.setVisible(terrain_param_ok or level_param_ok or brush_ok)

        # v2.7 copy/paste -- deliberately placed after the
        # forced-back-to-Pan block above, not before: that block can flip
        # self._current_tool to "pan" synchronously (pan_action.
        # setChecked(True) -> _on_tool_selected("pan"), which itself calls
        # back into this method -- see that method's own comment) mid-call,
        # and copy/paste's gating needs to see the *final* tool for this
        # call, not whatever was selected when it started.
        #
        # Copy is enabled only for the edit-category tools that have
        # per-tile data to copy (Terrain, Paint Can, Elevate, Set Elevation
        # -- never Pan), and only while that tool's own action is actually
        # enabled (not just selected) -- reusing terrain_action/fill_action/
        # elevation_action/set_level_action's already-computed
        # write_ok/elevation_ok gates above rather than re-deriving them
        # here. Paint Can copies/pastes the same single hovered tile Terrain
        # does -- Paste is not redefined as "fill with the clipboard
        # terrain" -- so it shares Terrain's "terrain" clipboard kind below.
        current_tool_action = {
            "terrain": self.terrain_action,
            "fill": self.fill_action,
            "elevation": self.elevation_action,
            "set_level": self.set_level_action,
        }.get(self._current_tool)
        copy_ok = current_tool_action is not None and current_tool_action.isEnabled()
        self.copy_action.setEnabled(copy_ok)

        # Paste additionally needs a non-empty clipboard whose kind matches
        # what the active tool would produce. Decision (a real open
        # question, not obvious either way): disable Paste outright on a
        # kind mismatch -- e.g. Copy while
        # on Terrain, switch to Elevate, hit Paste -- rather than letting
        # the clipboard's own kind silently override the active tool.
        # Chosen for consistency with every other action this method
        # already gates: all of them fail toward "visibly greyed out with
        # an obvious reason" rather than a behavior that depends on state
        # the toolbar doesn't show. Terrain and Elevate/Set Elevation both
        # copy/paste through the same "elevation" clipboard kind (see
        # copy_tile()), since Elevate and Set Elevation already share the
        # same underlying tile field.
        clipboard_kind = self._clipboard["kind"] if self._clipboard is not None else None
        kind_for_tool = "terrain" if self._current_tool in ("terrain", "fill") else "elevation"
        self.paste_action.setEnabled(copy_ok and clipboard_kind == kind_for_tool)

    def _sync_map_view_brush(self) -> None:
        """Pushes the toolbar's current brush size/shape into MapView's
        hover-preview state -- but only for a tool that actually has one
        (BRUSH_TOOLS); Pan and Paint Can always preview a single tile
        regardless of what size/shape the spinbox/combo were last left at
        for Terrain/Elevate/Set Elevation. Called both when the brush
        widgets change and when the active tool changes, so the preview is
        never stale in either direction."""
        if self._current_tool in BRUSH_TOOLS:
            self.map_view.set_brush(self.brush_size_spin.value(), self.brush_shape_combo.currentData())
        else:
            self.map_view.set_brush(brush.BRUSH_SIZE_MIN, brush.BRUSH_SHAPE_SQUARE)

    def _on_brush_changed(self) -> None:
        self._sync_map_view_brush()
        self.map_view.refresh_highlight(self._hover_tile)

    def _on_tool_selected(self, tool: str) -> None:
        self._current_tool = tool
        self.map_view.set_tool(tool)
        self._sync_map_view_brush()
        self._update_mode_status()
        self._log_status(f"Tool changed to {tool}")
        # v2.7 copy/paste: Copy/Paste's enabled state depends on which tool
        # is active (and, for Paste, whether the clipboard's kind matches
        # it -- see _update_tool_enabled()'s own comment) but this method is
        # the one path that changes self._current_tool without going
        # through _update_tool_enabled() itself (unlike on_mode_changed/
        # on_terrain_style_changed/load_scenario/close_scenario, which all
        # call it already). Without this, switching tools (e.g. Terrain ->
        # Elevate with a terrain-kind clipboard) would leave Paste enabled
        # from the previous tool's evaluation until something else happened
        # to trigger a refresh. Safe against the recursion this method can
        # itself trigger indirectly (_update_tool_enabled() forcing Pan back
        # on via pan_action.setChecked(True) when the current tool becomes
        # disabled, which re-enters this method) -- that re-entry finds Pan
        # already the current tool, whose lookup in _update_tool_enabled()'s
        # forced-Pan check is a no-op, so it terminates rather than looping.
        self._update_tool_enabled()

    def on_terrain_style_changed(self, text: str) -> None:
        style = text.lower()
        if style == self._terrain_style:
            return
        if self._busy:
            # A render is already in progress (reentrant signal -- see
            # load_scenario's comment on self._busy). Refuse the change and
            # put the combo back to what's actually loaded rather than
            # leaving it displaying a style nothing was ever rendered at;
            # blockSignals so this doesn't recurse back into this handler.
            self.terrain_style_combo.blockSignals(True)
            self.terrain_style_combo.setCurrentText(self._terrain_style.capitalize())
            self.terrain_style_combo.blockSignals(False)
            return
        self._terrain_style = style
        # Flat's own view-rotation checkbox has nothing to do while a
        # Stepped image is loaded (its projection is already baked into the
        # pixels) -- see the checkbox's own tooltip.
        self.iso_action.setEnabled(style == "flat")
        self._update_tool_enabled()
        self._update_mode_status()
        if self.scenario is not None:
            self._busy = True
            try:
                elapsed, tile_px = self._render_current()
                self._log_status(f"Elevation view: {text} (tile_px={tile_px}) in {elapsed:.2f}s")
            finally:
                self._busy = False

    # -- Edit-tool strokes (terrain paint / elevation raise-lower / set level) --
    # One stroke (mouse-down through release, or a defensive close on leave --
    # see MapView) is exactly one descape.edit_history.EditHistory record and
    # one undo step, but repaints live as it's dragged. See edit_history.py's
    # module docstring for why nothing here writes to a tile directly.

    def on_edit_stroke_start(self) -> None:
        if self.scenario is None:
            return
        self.edit_history.begin_stroke(self.scenario.map_manager.terrain)
        # index -> the tile state as of the last time we handed that index to
        # _apply_dirty. A dict, not a set of indices: elevation propagation
        # can change one tile SEVERAL times over a single drag, and a
        # membership-only set silently drops every change after the first --
        # see on_edit_stroke_tile's own comment.
        self._stroke_seen_state: dict[int, tuple[int, int, int]] = {}
        # Painted-tile dedupe for this stroke, keyed on the actual tile a
        # brush touched -- see on_edit_stroke_tile's own comment for why
        # this must be separate from MapView._stroke_touched (which is
        # keyed on the CURSOR tile, and is only a cheap early-out, not a
        # correctness guarantee once a brush is bigger than one tile).
        self._stroke_painted: set[tuple[int, int]] = set()

    def on_edit_stroke_tile(self, x: int, y: int, modifiers) -> None:
        if self.scenario is None:
            return
        mm = self.scenario.map_manager
        # Every drag tool reaching this method today (terrain/elevation/
        # set_level) has supports_brush=True, so this is BRUSH_TOOLS in
        # practice -- checked explicitly rather than assumed, so a future
        # is_edit_tool tool with no brush still degrades to its old
        # single-tile behavior instead of silently expanding.
        if self._current_tool in BRUSH_TOOLS:
            footprint = brush.brush_tiles(
                x, y, self.brush_size_spin.value(), self.brush_shape_combo.currentData(), mm.map_width, mm.map_height
            )
        elif 0 <= x < mm.map_width and 0 <= y < mm.map_height:
            footprint = [(x, y)]
        else:
            footprint = []
        # Dedupe against tiles this stroke already painted -- NOT against
        # MapView._stroke_touched's cursor-tile set. With a brush bigger
        # than one tile, a single painted tile falls under many distinct
        # cursor tiles during a drag; without this, Elevate's accumulating
        # +/-1 (see below) would raise/lower the same tile once per cursor
        # tile that overlapped it, not once per stroke.
        footprint = [t for t in footprint if t not in self._stroke_painted]
        if not footprint:
            return

        if self._current_tool == "terrain":
            terrain_id = self.terrain_combo.currentData()
            for tx, ty in footprint:
                tile = mm.get_tile(tx, ty)
                tile.terrain_id = terrain_id
                # Clear a stale double-terrain blend -- render.py doesn't draw
                # `layer`, but the game does, and leaving it set after changing
                # terrain_id would make this tool's own render lie about what
                # the game will actually show.
                tile.layer = -1
        elif self._current_tool == "elevation":
            delta = -1 if modifiers & Qt.ShiftModifier else 1
            # Clamp to the legal range Set Elevation's spinbox already
            # enforces (ELEVATION_LEVEL_MAX) -- Elevate had no clamp at all
            # before Phase 4: going below 0 wrote a negative int that
            # scenario_write.py's raw byte patch (`body[o+1] = tile.elevation`)
            # would crash on at save time, and going above the legal range
            # is exactly the case the fixed-range canvas sizing every Stepped
            # projection uses (iso_geometry.canvas_size_and_origin, see
            # iso_geometry.MAX_ELEVATION) now depends on never happening for
            # Stepped mode's incremental redraw to stay safe. Both real, not
            # hypothetical -- confirmed by reading scenario_write.py and by
            # descape.render.dirty_screen_bbox_iso's own guard below.
            #
            # Every target's current elevation is read here, before ANY tile
            # in this footprint is written -- set_tiles_elevation's
            # propagation pass can still change a not-yet-processed
            # footprint tile's elevation (that's the whole point of a
            # footprint-wide xys), and reading "current" late would apply
            # delta on top of that propagated value instead of the value
            # this cursor tile's drag actually found.
            targets = [
                (tx, ty, max(0, min(ELEVATION_LEVEL_MAX, mm.get_tile(tx, ty).elevation + delta)))
                for tx, ty in footprint
            ]
            set_tiles_elevation(mm, targets)
        elif self._current_tool == "set_level":
            level = self.elevation_level_spin.value()
            set_tiles_elevation(mm, [(tx, ty, level) for tx, ty in footprint])
        else:
            return

        self._stroke_painted.update(footprint)

        # Cumulative dirty set since stroke start, minus what's already been
        # redrawn AT ITS CURRENT STATE this stroke -- avoids repainting the
        # same tile repeatedly as the drag continues over tiles elevation
        # propagation already touched. See EditHistory.stroke_dirty_indices's
        # docstring for the cost of this (a linear scan) at this project's map
        # sizes -- this is exactly why the brush footprint is expanded HERE,
        # once per cursor tile, rather than by calling this method once per
        # brush tile from MapView: doing that would multiply an already-O(map)
        # scan by the brush's area on every mouse-move. Same warning on_fill's
        # own docstring carries for its single full-map fill.
        #
        # Compared on STATE, not on index membership. stroke_dirty_indices is
        # cumulative (everything differing from the stroke-start snapshot), so
        # once a tile appears it stays for the rest of the drag -- and
        # set_tiles_elevation's propagation routinely changes one tile several
        # times as the brush moves over it. Subtracting a plain set of indices
        # therefore synced each tile exactly once, at its FIRST value, and
        # froze it there: _apply_dirty -> dirty_screen_bbox_iso is the only
        # thing that writes MapView._iso_elevations, so that snapshot drifted
        # permanently out of sync with tile.elevation. Measured on one
        # 8-step Set Elevation drag: 131 tiles wrong, 35 of them sitting at
        # elevation 6 while the snapshot still read lower, some off by 2.
        # Top faces still looked right (they render from tile.elevation), but
        # the hover highlight and screen_to_tile hit-testing read the array,
        # and _render_tile_iso mixes the two when it computes skirt/contact-
        # shadow deltas as tile.elevation - elevations[neighbour].
        all_dirty = self.edit_history.stroke_dirty_indices(mm.terrain)
        new_dirty = {i for i in all_dirty if tile_state(mm.terrain[i]) != self._stroke_seen_state.get(i)}
        for i in all_dirty:
            self._stroke_seen_state[i] = tile_state(mm.terrain[i])
        self._apply_dirty(new_dirty)

    def on_edit_stroke_end(self) -> None:
        if self.scenario is None:
            return
        label = _STROKE_LABELS.get(self._current_tool, "Edit")
        self.edit_history.commit_stroke(label, self.scenario.map_manager.terrain)
        self._stroke_seen_state = {}
        self._stroke_painted = set()
        self._update_edit_actions()
        self._update_title()

    # -- Copy/paste (v2.7) -- one clipboard slot, keyed off
    # self._hover_tile (set by on_hover() on every mouse move) rather than a
    # click, since these fire from a keyboard shortcut. Copy reads whatever
    # field the active edit tool cares about straight off the hovered tile
    # (no undo record -- nothing is mutated). Paste is a one-shot,
    # non-interactive edit -- exactly what edit_history.EditHistory.apply()
    # exists for (see its own docstring), unlike the drag-stroke tools above
    # which use begin_stroke/stroke_dirty_indices/commit_stroke directly for
    # live per-tile feedback mid-drag; a keyboard paste has no drag to give
    # feedback during.

    def copy_tile(self) -> None:
        if self.scenario is None or self._hover_tile is None:
            return
        x, y = self._hover_tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        tile = mm.get_tile(x, y)
        if self._current_tool in ("terrain", "fill"):
            # `layer` is captured alongside terrain_id but deliberately never
            # read back on paste (see paste_tile()'s mutate_fn) -- paste
            # always resets layer to -1 instead, matching every other
            # terrain-write path in this tool. Kept in the dict anyway
            # (costs nothing) so it's visible here that this was considered,
            # not overlooked.
            self._clipboard = {"kind": "terrain", "terrain_id": tile.terrain_id, "layer": tile.layer}
            self._log_status(f"Copied terrain ({name_for_terrain_id(tile.terrain_id)}) from ({x}, {y})")
        elif self._current_tool in ("elevation", "set_level"):
            self._clipboard = {"kind": "elevation", "value": tile.elevation}
            self._log_status(f"Copied elevation ({tile.elevation}) from ({x}, {y})")
        else:
            return
        # Paste's enabled state depends on the clipboard's kind (see
        # _update_tool_enabled()'s comment) -- refresh it now rather than
        # waiting for some unrelated event to do so, or a fresh Copy
        # wouldn't visibly enable Paste until then.
        self._update_tool_enabled()

    def paste_tile(self) -> None:
        if self.scenario is None or self._hover_tile is None or self._clipboard is None:
            return
        x, y = self._hover_tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        kind = self._clipboard["kind"]
        # Defensive re-check of the same type-mismatch decision
        # _update_tool_enabled() already encodes in paste_action's enabled
        # state (disable on a kind mismatch -- see that method's comment):
        # this guards paste_tile() itself against being invoked directly
        # (e.g. by a test, or a future caller) bypassing the QAction.
        if kind == "terrain" and self._current_tool not in ("terrain", "fill"):
            return
        if kind == "elevation" and self._current_tool not in ("elevation", "set_level"):
            return

        def mutate() -> None:
            if kind == "terrain":
                t = mm.get_tile(x, y)
                t.terrain_id = self._clipboard["terrain_id"]
                # Same layer-reset every other terrain-write path in this
                # tool applies (the Terrain tool's own click handler in
                # on_edit_stroke_tile(), batch_api.set_terrain) -- a stale
                # double-terrain blend left over from whatever terrain_id
                # used to be there would make this tool's own render lie
                # about what the game will actually show. Preserving the
                # copied tile's own `layer` verbatim was the alternative,
                # but that would make a pasted tile behave differently from
                # one painted with the same terrain_id by any other path in
                # the tool, for no real benefit.
                t.layer = -1
            else:
                # Not a raw `tile.elevation =` write -- goes through the
                # same neighbor-propagation real Elevate/Set Elevation
                # edits already use. No clamping needed here (unlike
                # on_edit_stroke_tile()'s Elevate branch): the copied value
                # was already a legal elevation on its source tile, not a
                # delta that could go out of range.
                set_tile_elevation(mm, x, y, self._clipboard["value"])

        dirty = self.edit_history.apply("Paste", mm.terrain, mutate)
        # Always log, even on a genuine no-op (dirty == [], e.g. pasting the
        # terrain a tile already has) -- EditHistory.apply()/commit_stroke()
        # deliberately push no record for that case (see commit_stroke()'s
        # docstring), but silently doing nothing here would look like the
        # keybind itself was broken. _apply_dirty() already no-ops on an
        # empty dirty_indices, so it's still safe to call unconditionally.
        self._apply_dirty(dirty)
        self._update_edit_actions()
        self._update_title()
        if dirty:
            self._log_status(f"Pasted {kind} to ({x}, {y})")
        else:
            self._log_status(f"Pasted {kind} to ({x}, {y}) (no change)")

    def on_fill(self, x: int, y: int, modifiers) -> None:
        """Paint Can: one flood fill per left click -- MapView routes
        CLICK_TOOLS here directly (see mousePressEvent), never through the
        stroke handlers above. Shaped like paste_tile() just above, not like
        on_edit_stroke_tile(): a one-shot edit_history.apply() rather than
        begin_stroke/stroke_dirty_indices/commit_stroke, since there's no
        drag to give live feedback during and stroke_dirty_indices() is an
        O(map) scan per call -- fine once per touched brush tile, far too
        slow once per filled tile. `modifiers` is accepted only for
        signature symmetry with on_edit_stroke_tile/on_click_edit and is
        deliberately ignored -- a fill has no Shift/right-button inverse.

        Wraps the whole operation in the same wait-cursor/setEnabled(False)/
        self._busy pattern load_scenario()/refresh_map()/_render_current()
        already use: tools/bench_fill_latency.py measured a full-map fill on
        a 480x480 map at ~540ms (Flat) to ~2.4s (Stepped, before
        STEPPED_FULL_RERENDER_THRESHOLD's fallback) end to end -- long
        enough that the window would otherwise sit frozen with no visible
        indication anything is happening, and that a second click dispatched
        via processEvents() mid-fill could reenter this method while
        self.scenario is still being mutated. Applied unconditionally rather
        than only above some measured size, matching every other guarded
        method in this class -- the guard is cheap for the common small-fill
        case too."""
        if self.scenario is None or self._busy:
            return
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            return
        terrain_id = self.terrain_combo.currentData()
        self._busy = True
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.perf_counter()
            dirty = self.edit_history.apply(
                _STROKE_LABELS["fill"], mm.terrain, lambda: flood_fill_terrain(mm, x, y, terrain_id)
            )
            self._apply_dirty(dirty)
            elapsed = time.perf_counter() - t0
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)
            self._busy = False
        self._update_edit_actions()
        self._update_title()
        name = name_for_terrain_id(terrain_id)
        # Always log, including the no-op case, same reasoning as
        # paste_tile()'s own comment above -- a silent no-op reads as a
        # broken keybind, and a click that can rewrite the whole map
        # deserves a record either way.
        if dirty:
            self._log_status(f"Filled {len(dirty)} tiles with {name} from ({x}, {y}) in {elapsed:.2f}s")
        else:
            self._log_status(f"Fill at ({x}, {y}): already {name} (no change)")

    def _apply_dirty(self, dirty_indices) -> None:
        """Repaints exactly the given tile indices -- the incremental path
        used by strokes (above) and undo/redo, instead of a full
        refresh_map() (which reallocates and recomposites the entire map;
        fine once per file open, far too slow per brush touch).

        Stepped mode (Phase B-C, on top of Phase 4/5's terrain+unit
        interleaving): dirty_screen_bbox_iso() plays refresh_region_iso()'s
        old "half 1" role -- mutates self._iso_elevations in place and
        returns the invalidated bbox, without needing an img array at all
        (see descape.render's own docstring for that split, Phase B-B) --
        then self._cache.patch(bbox) recomposites just the touched chunks
        and self.map_view.invalidate_region(bbox) schedules the actual Qt
        repaint, which pulls the freshly-patched pixels straight from the
        cache. dirty_screen_bbox_iso() can decline (return None) if an edit
        pushed a tile's elevation outside the range self._iso_proj was
        sized for -- see that function's own docstring for why this is a
        defensive fallback, not an expected path given viewer.py's own
        elevation tools already clamp to that same range. Falling back to a
        full _render_current() keeps correctness even if that assumption is
        ever violated (e.g. a hand-edited file loaded with an elevation
        already out of range), at the cost of one full re-render instead of
        an instant patch for that one stroke tile.

        Flat mode (Phase B-E): no elevation term and no fallback case --
        the dirty rect is simply the union of the dirty tiles' own pixel
        squares. Unlike Stepped, there's no footprint-expansion step here
        either: composite_rect_flat() redraws terrain AND every overlapping
        unit in full stacking order within whatever rect it's given (see
        that function's own docstring for why that makes the old
        refresh_units_over() fixed-point dirty-tile expansion unnecessary).
        Patches per-tile rects, not one union bbox -- a union over a
        scattered undo set could span the whole map, turning patch() into a
        full recomposite."""
        if not dirty_indices or self.scenario is None:
            return
        mm = self.scenario.map_manager
        tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
        if self._terrain_style == "stepped":
            if self._cache is None:
                return
            # Above STEPPED_FULL_RERENDER_THRESHOLD dirty tiles (a Paint Can
            # fill spanning a large fraction of the map, measured with
            # tools/bench_fill_latency.py -- see that constant's own
            # comment), dirty_screen_bbox_iso's per-tile dilation is slower
            # than just re-rendering. Skips the dilation call entirely
            # rather than running it and discarding the result -- the
            # dilation cost is exactly what this branch exists to avoid.
            if len(dirty_indices) > STEPPED_FULL_RERENDER_THRESHOLD:
                elapsed, _ = self._render_current()
                self._log_status(
                    f"Elevation view: {len(dirty_indices)} tiles dirty, "
                    f"re-rendered full map instead of patching ({elapsed:.2f}s)"
                )
                self._update_title()
                return
            bbox = dirty_screen_bbox_iso(
                self.scenario, dirty_indices, self._iso_elevations, self._iso_proj, with_units=True
            )
            if bbox is None:
                elapsed, _ = self._render_current()
                self._log_status(
                    f"Elevation view: edit exceeded the cached elevation range, "
                    f"re-rendered full map ({elapsed:.2f}s)"
                )
            else:
                self._cache.patch(bbox)
                self.map_view.invalidate_region(bbox)
        else:
            assert self._iso_elevations is None, "Flat mode must never carry a Stepped elevation snapshot"
            if self._cache is None:
                return
            coords = [(mm.terrain[i].x, mm.terrain[i].y) for i in dirty_indices]
            rects = [(x * tile_px, y * tile_px, (x + 1) * tile_px, (y + 1) * tile_px) for x, y in coords]
            self._cache.patch_rects(rects)
            for rect in rects:
                self.map_view.invalidate_region(rect)
        self._update_title()

    def _update_edit_actions(self) -> None:
        can_undo = self.scenario is not None and self.edit_history.can_undo
        can_redo = self.scenario is not None and self.edit_history.can_redo
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)

    def undo(self) -> None:
        if self.scenario is None:
            return
        dirty = self.edit_history.undo(self.scenario.map_manager.terrain)
        self._apply_dirty(dirty)
        self._update_edit_actions()
        self._log_status("Undo")

    def redo(self) -> None:
        if self.scenario is None:
            return
        dirty = self.edit_history.redo(self.scenario.map_manager.terrain)
        self._apply_dirty(dirty)
        self._update_edit_actions()
        self._log_status("Redo")

    def _update_title(self) -> None:
        if self.scenario is None:
            self.setWindowTitle("DEscape")
            return
        marker = "*" if self.edit_history.is_dirty else ""
        self.setWindowTitle(f"{marker}{self.scenario.path.name} — DEscape")

    def _confirm_discard_changes(self) -> bool:
        """True if it's OK to proceed: no scenario loaded, no unsaved edits,
        or the user explicitly chose to discard them. Call before anything
        that would drop the current scenario -- open, close, or window close."""
        if self.scenario is None or not self.edit_history.is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved changes",
            f"{self.scenario.path.name} has unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Discard

    def _save_as_start_path(self) -> str:
        if self._untitled:
            # No "_edited" suffix -- a new map isn't an edit *of* anything -- and
            # never the template's own directory (Path.home() is always writable
            # and never TEMPLATE_DIR).
            return str(Path.home() / UNTITLED_NAME)
        src = self.scenario.path
        default_name = f"{src.stem}_edited{src.suffix}"
        # File > Open can reach the shipped template itself; without this, its
        # own "_edited" default would aim inside TEMPLATE_DIR, which
        # write_scenario() unconditionally refuses to write into.
        parent = Path.home() if src.parent.resolve() == TEMPLATE_DIR.resolve() else src.parent
        return str(parent / default_name)

    def save_as(self) -> None:
        if self.scenario is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scenario as", self._save_as_start_path(), "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        try:
            write_scenario(self.scenario, path)
        except WriteBlockedError as e:
            self._log_status(f"Save blocked: {e}")
            QMessageBox.critical(self, "Save blocked", str(e))
            return
        except Exception as e:
            self._log_status(f"Save failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")
            return
        if self._untitled:
            # An untitled map becomes the file it was just saved to -- otherwise
            # the title would stay "Untitled" with no dirty marker while the map
            # is in fact on disk, and a second Save As would default back to
            # Untitled again. Deliberately NOT done for normally-opened files:
            # that would turn the existing "<stem>_edited" default into
            # "<stem>_edited_edited" and retitle the window away from the file
            # the user actually opened.
            self.scenario.path = Path(path)
            self._untitled = False
            self._update_info()  # the info panel's "File:" line just changed
        self.edit_history.mark_saved()
        self._update_title()
        self._log_status(f"Saved to {path}")

    def closeEvent(self, event) -> None:
        if not self._confirm_discard_changes():
            event.ignore()
            return
        settings.set_window_size(self.width(), self.height())
        event.accept()

    def _show_settings(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec_()

    def _show_debug_log(self) -> None:
        dialog = DebugLogDialog(self)
        dialog.exec_()

    def _log_status(self, message: str) -> None:
        self.status_log.appendPlainText(message)
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
        debug_log.log(message)

    def _render_current(self) -> tuple[float, int]:
        """Renders/prepares self.scenario at the currently selected Terrain
        Style and pushes it to self.map_view -- the render+display step
        shared by load_scenario/refresh_map/on_terrain_style_changed, each
        of which wants a differently-worded status log line, so that part
        stays theirs. Returns (elapsed_seconds, tile_px).

        Neither style blocks on the full canvas anymore as of Phase B-E
        (Flat was the last one still doing so -- render_scenario() always
        compositing the whole map up front, ~1-2s for a real map). Stepped
        (Phase B-C): elevations_and_proj() reads elevations and sizes the
        projection WITHOUT compositing a single pixel. Flat (Phase B-E):
        FlatChunkCache's own construction is similarly cheap -- it only
        precomputes unit_draws (_flat_unit_draws()), not any pixels. Both
        chunk caches then composite lazily, chunk by chunk, only for
        whatever MapCanvasItem's paint() actually asks for.
        This is the whole point of Track B:
        opening/switching to either view no longer pays for the full
        canvas up front.

        setEnabled(False) for the same span, not just the cursor: processEvents()
        re-enters the event loop, which would otherwise let a second click
        (Open, the Elevation View combo, a toolbar tool) dispatch a
        reentrant call into this same render path mid-flight -- e.g. a
        second render starting while self._cache/self.map_view are still
        being written by the first. Now purely a reentrancy guard rather
        than also covering a visible freeze -- the wait cursor still shows
        briefly for the cache construction above, but there's no multi-
        second compositing pass left for the window to actually appear
        frozen during. A disabled window still repaints (the cursor/log
        message stay visible), it just stops accepting input."""
        self.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            t0 = time.time()
            mm = self.scenario.map_manager
            tile_px = tile_pixels_for_map(mm.map_width, mm.map_height)
            if self._terrain_style == "stepped":
                elevations, proj = elevations_and_proj(self.scenario)
                self._iso_elevations, self._iso_proj = elevations, proj
                self._cache = IsoChunkCache(self.scenario, elevations, proj, tile_px)
                self.map_view.set_source(
                    tile_px, terrain_style="stepped", cache=self._cache, elevations=elevations, proj=proj
                )
            elif self._terrain_style == "sloped":
                # No incremental-edit elevation snapshot: Sloped has no live
                # editing yet (Track C4), only Stepped's dirty-patch path
                # (_apply_dirty above) reads self._iso_elevations/_iso_proj.
                elevations, corner_rise, proj = sloped_elevations_and_proj(self.scenario)
                self._iso_elevations, self._iso_proj = None, None
                self._cache = SlopedChunkCache(self.scenario, elevations, corner_rise, proj, tile_px)
                self.map_view.set_source(
                    tile_px, terrain_style="sloped", cache=self._cache, elevations=elevations, proj=proj
                )
            else:
                self._iso_elevations, self._iso_proj = None, None
                self._cache = FlatChunkCache(self.scenario, tile_px)
                self.map_view.set_source(tile_px, terrain_style="flat", cache=self._cache)
            return time.time() - t0, tile_px
        finally:
            QApplication.restoreOverrideCursor()
            self.setEnabled(True)

    def refresh_map(self) -> None:
        # Re-render the currently open map, if any, so a settings change (e.g.
        # a newly-configured install path) takes effect immediately instead of
        # only on the next file open. self._busy guard: see load_scenario's
        # own comment on it -- same reentrancy concern, one shared flag.
        if self.scenario is not None and not self._busy:
            self._busy = True
            try:
                elapsed, tile_px = self._render_current()
                self._log_status(
                    f"Re-rendered map (tile_px={tile_px}, style={self._terrain_style}) in {elapsed:.2f}s"
                )
            finally:
                self._busy = False

    def open_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open scenario", "", "AoE2 Scenario (*.aoe2scenario)"
        )
        if not path:
            return
        self.load_scenario(Path(path))

    def new_map(self, tiles: int = BLANK_TEMPLATE_TILES) -> None:
        """Starts a blank `tiles`x`tiles` map by generating it in memory from
        the shipped 120x120 donor template (descape.scenario_new.
        blank_scenario_bytes -- one of STANDARD_MAP_SIZES via File > New Map's
        size submenu, or any size in [MIN_MAP_TILES, MAX_MAP_TILES] via
        new_map_custom()) and loading the result through the normal load path,
        then marking the document untitled -- see load_scenario()'s `untitled`
        handling. Not built from AoE2ScenarioParser's own from_default():
        that would need a second write path through the library's own
        (unverified, non-byte-stable) serializer instead of the already
        in-game-verified byte-patch path this reuses unchanged.

        The new map inherits the donor's FileHeader verbatim (creator name,
        timestamps) and DataHeader.filename -- the byte-patch splice can't
        rewrite those by design: filename is DataHeader's terminal,
        length-prefixed field, and rewriting it to a different-length name
        would shift every section offset after it. Inherent to reusing a
        donor rather than a defect in it.
        """
        if not self._confirm_discard_changes():
            return
        self._create_new_map(tiles)

    def new_map_custom(self) -> None:
        """File > New Map > Custom size…. Split out from new_map() so the
        size policy (scenario_new.validate_tiles) is testable with no dialog
        at all. Confirms first for anything larger than
        LARGE_MAP_CONFIRM_TILES -- deliberately only on this path, never for
        the labelled preset entries (including 480): a preset is a deliberate
        choice on a size already in-game proven, so a confirm there would
        just be a nag, while a typed-in custom size has no such history.

        Checks _confirm_discard_changes() up front, before either dialog, and
        calls _create_new_map() directly rather than new_map() -- new_map()
        would check discard again, which would mean answering it twice (once
        here, once more inside new_map()) whenever the document is dirty."""
        if not self._confirm_discard_changes():
            return
        tiles, ok = QInputDialog.getInt(
            self,
            "Custom map size",
            f"Map size (tiles, square, {MIN_MAP_TILES}-{MAX_MAP_TILES}):",
            BLANK_TEMPLATE_TILES,
            MIN_MAP_TILES,
            MAX_MAP_TILES,
            1,
        )
        if not ok:
            return
        if tiles > LARGE_MAP_CONFIRM_TILES:
            reply = QMessageBox.question(
                self,
                "Large map",
                f"{tiles}×{tiles} is larger than the {LARGE_MAP_CONFIRM_TILES}×"
                f"{LARGE_MAP_CONFIRM_TILES} standard size. Large maps take "
                "longer to create and render, and use significantly more "
                "memory. Create it anyway?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        self._create_new_map(tiles)

    def _create_new_map(self, tiles: int) -> None:
        """The generate-and-load half of File > New Map, shared by new_map()
        and new_map_custom() -- both of which have already run
        _confirm_discard_changes() themselves before calling this, so it
        does no confirming of its own."""
        try:
            data = blank_scenario_bytes(tiles)
        except (MapSizeError, BlankGenerationError) as e:
            self._log_status(f"New Map failed: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "New Map failed", str(e))
            return
        self.load_scenario(UNTITLED_PATH, untitled=True, data=data)

    def load_scenario(self, path: Path, *, untitled: bool = False, data: bytes | None = None) -> None:
        # Loading + rendering a large map is a multi-second blocking call
        # (see _render_current()'s own docstring) -- the log line and status
        # bar message below are queued but not actually painted until
        # control returns to the event loop, so without this
        # setOverrideCursor()+processEvents() pair the window would just sit
        # frozen with no visible indication anything is happening until the
        # whole load completes. self._busy (see its own comment in
        # __init__) plus setEnabled(False) together guard against the same
        # reentrancy risk _render_current() documents for its own
        # processEvents() call -- e.g. a second Open dispatched from inside
        # this one's processEvents() while self.scenario is still being
        # reassigned. A true no-op, not just queued: checked before
        # anything else runs.
        if self._busy:
            return
        self._busy = True
        try:
            self._log_status(f"Loading scenario: {path}")
            self.statusBar().showMessage(
                "Creating new map..." if untitled else f"Loading {path.name}..."
            )
            self.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            try:
                self.scenario = (
                    load_map_and_units_from_bytes(data, path)
                    if data is not None
                    else load_map_and_units(path)
                )
            except Exception as e:
                self.statusBar().clearMessage()
                self._log_status(f"Failed to load {path}: {type(e).__name__}: {e}")
                QMessageBox.critical(self, "Failed to load", f"{type(e).__name__}: {e}")
                return
            finally:
                QApplication.restoreOverrideCursor()
                self.setEnabled(True)

            # Only reached on a successful load, so a failed File > New can never
            # mislabel whatever was previously open (self.scenario still holds it,
            # untouched, if the try block above returned early).
            self._untitled = untitled
            if untitled:
                self.scenario.path = UNTITLED_PATH
            display_name = self.scenario.path.name

            self.edit_history.reset()

            # Renders at whichever Terrain Style was already selected --
            # File > Open doesn't reset it back to Flat. _render_current()
            # pushes its own wait cursor/setEnabled(False) for this step,
            # nested safely inside this method's own (see both docstrings).
            # The "Loading..." status bar message deliberately stays up
            # through this render step too, not just the parse above --
            # cleared only once everything is actually done, right below.
            elapsed, tile_px = self._render_current()
            mm = self.scenario.map_manager
            self._update_info()
            self._update_tool_enabled()
            self._update_edit_actions()
            self._update_title()
            self.statusBar().clearMessage()
            self._log_status(
                f"{'Created' if untitled else 'Loaded'} {display_name} "
                f"({mm.map_width}x{mm.map_height} tiles, "
                f"{sum(len(u) for u in self.scenario.unit_manager.units):,} units, "
                f"tile_px={tile_px}, style={self._terrain_style}) in {elapsed:.2f}s"
            )
            if not self.scenario.terrain_write_supported:
                self._log_status(
                    f"Warning: {display_name}'s terrain block failed load-time verification -- "
                    "Edit mode's terrain/elevation tools and Save As are disabled for this "
                    "file (still fully viewable)."
                )
            elif not self.scenario.map_is_square:
                # Non-square is NOT actually supported today, despite what this message
                # used to claim: AoE2ScenarioParser's MapManager.map_size raises
                # ValueError("Map is not a square...") from tile.x/tile.y/get_tile(),
                # reached at ~15 call sites across render.py/viewer.py/elevation_tools.py/
                # batch_api.py -- so a non-square map fails during the render this same
                # load_scenario() call triggers below, before this note would ever help.
                # A possible follow-up: deriving tile coordinates from the terrain
                # index instead of these library properties.
                self._log_status(
                    f"Note: {display_name} is a non-square map ({mm.map_width}x{mm.map_height}) -- "
                    "non-square maps are not supported and will likely fail to render."
                )
        finally:
            self._busy = False

    def close_scenario(self) -> None:
        if self.scenario is None:
            return
        if not self._confirm_discard_changes():
            return
        name = self.scenario.path.name
        self.scenario = None
        self._untitled = False
        self._cache = None
        self._iso_elevations, self._iso_proj = None, None
        self.edit_history.reset()
        self.map_view.clear_image()
        # No edit tool has anything to act on with no map open; forcing Pan
        # (rather than just disabling the edit tools) keeps MapView's own
        # tool state in sync too -- same mechanism on_mode_changed already
        # uses when Edit mode becomes unavailable.
        self.pan_action.setChecked(True)
        self.info.setPlainText("")
        self.hover_label.setText("Hover the map for tile info")
        # A stale (x, y) from the just-closed map must not outlive it -- the
        # bounds check in paste_tile()/copy_tile() would likely catch a
        # mismatch against a differently-sized map opened next anyway, but
        # relying on that coincidence is exactly the kind of leak
        # edit_history.reset() above is already here to prevent for edit
        # history. self._clipboard deliberately survives a close (a
        # clipboard outliving the file it was copied from is normal
        # clipboard semantics, and Paste is already disabled with no map
        # loaded via _update_tool_enabled() below) -- only the hover
        # position is map-relative state that needs clearing here.
        self._hover_tile = None
        self._update_tool_enabled()
        self._update_edit_actions()
        self._update_title()
        self._log_status(f"Closed {name}")

    def _update_info(self) -> None:
        s = self.scenario
        mm, um = s.map_manager, s.unit_manager

        terrain_hist = Counter(t.terrain_id for t in mm.terrain)
        lines = [
            f"File: {s.path.name}",
            f"Scenario version: {s.scenario_version}",
            f"Map size: {mm.map_width} x {mm.map_height}",
            f"Trigger tail (not parsed): {len(s.trigger_tail):,} bytes",
            "",
            "Terrain (top 8):",
        ]
        total_tiles = len(mm.terrain)
        for tid, count in terrain_hist.most_common(8):
            lines.append(f"  {name_for_terrain_id(tid):24s} {100 * count / total_tiles:4.1f}%")

        lines += ["", "Units per player:"]
        for player_id, units in enumerate(um.units):
            label = "GAIA" if player_id == 0 else f"Player {player_id}"
            lines.append(f"  {label:10s} {len(units):5d}")
        lines.append(f"  {'Total':10s} {sum(len(u) for u in um.units):5d}")

        self.info.setPlainText("\n".join(lines))

    def on_hover(self, tile: tuple[int, int] | None) -> None:
        self._hover_tile = tile
        if self.scenario is None:
            return
        if tile is None:
            self.hover_label.setText("Hover the map for tile info")
            return
        x, y = tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            self.hover_label.setText("Hover the map for tile info")
            return
        tile = mm.get_tile_safe(x, y)
        if tile is None:
            return
        self.hover_label.setText(
            f"({x}, {y})  {name_for_terrain_id(tile.terrain_id)}  elevation={tile.elevation}"
        )


def main() -> None:
    debug_log.log("Application started")
    app = QApplication(sys.argv)
    apply_theme(app, settings.get_dark_mode())
    window = ViewerWindow()
    window.show()
    ready_file = os.environ.get("DESCAPE_READY_FILE")
    if ready_file:
        Path(ready_file).touch()
    if len(sys.argv) > 1:
        window.load_scenario(Path(sys.argv[1]))
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
