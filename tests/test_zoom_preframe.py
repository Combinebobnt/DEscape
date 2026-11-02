"""Pre-rendered zoom-notch frames (TASK-031.71, descape.zoom_preframe).

A frame serves only under the exact key it was drawn with and only while
every chunk under it still has the pixel stamp it recorded. The stamp half
runs on real caches over test_coarse_reduce.py's edit table: every edit that
changes a level's pixels leaves its recorded stamps stale (or the level not
resident), and an install or eviction elsewhere does not. The paint half
drives real offscreen windows opted in with set_preframes_allowed(True),
the check mode on (conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_coarse_reduce import (  # noqa: F401
    CASES,
    NEW_TERRAIN_ID,
    OPS,
    _coarsest,
    _differs,
    _fresh_like,
    _make,
    _scenario,
    env,
)
from test_native_composite import TERRAIN_IDS, synthetic_textures  # noqa: F401 -- a fixture of env
from test_sprite_edit_bbox import sprite_install  # noqa: F401 -- a fixture of env

from test_unit_sprites import CONST

# Room for every level whole at once, so a stale stamp is never an LRU eviction.
ROOMY = 10**6


def _whole(cache, mip: int) -> tuple[np.ndarray, tuple]:
    """Level `mip` rendered whole, and chunk_stamps() over all of it."""
    w, h = cache.canvas_dims(mip)
    pixels = cache.render_rect(0, 0, w, h, mip=mip)
    stamps = cache.chunk_stamps(mip, *cache.chunk_index_range(mip, 0, 0, w, h))
    assert stamps is not None, "a whole-level read left a chunk uncached"
    return pixels, stamps


def _serves(cache, mip: int, stamps: tuple) -> bool:
    return cache.paints_without_build(mip) and cache.stamps_current(stamps)


# --- Step 2: chunk stamps -------------------------------------------------------


@pytest.mark.parametrize(("style", "op"), CASES)
def test_every_pixel_changing_edit_leaves_a_frames_stamps_stale(env, style, op):  # noqa: F811
    scenario = _scenario()
    cache = _make(style, scenario, max_chunks=ROOMY)
    levels = sorted({0, _coarsest(cache)})
    before = {mip: _whole(cache, mip) for mip in levels}
    assert all(_serves(cache, mip, stamps) for mip, (_px, stamps) in before.items())
    OPS[op](style, scenario, cache)
    fresh = _fresh_like(style, scenario, cache)
    changed = 0
    for mip, (pixels, stamps) in before.items():
        want = fresh.render_rect(0, 0, *fresh.canvas_dims(mip), mip=mip)
        if _differs(want, pixels):
            changed += 1
            assert not _serves(cache, mip, stamps), f"{style} {op}: level {mip} changed under current stamps"
    assert changed, f"{style} {op}: the edit changed no level, so the case proves nothing"


@pytest.mark.parametrize("style", ["stepped", "sloped", "flat"])
def test_installs_and_evictions_elsewhere_keep_a_frame_current(env, style):  # noqa: F811
    cache = _make(style, _scenario(), max_chunks=ROOMY)
    mip = _coarsest(cache)
    _pixels, stamps = _whole(cache, mip)
    _whole(cache, 0)
    assert _serves(cache, mip, stamps), "installing another level's chunks voided the frame"
    for key in [key for key in cache._cache if key[0] == 0][:3]:
        cache._cache_bytes -= cache._cache.pop(key).nbytes
    assert _serves(cache, mip, stamps), "evicting another level's chunks voided the frame"
    key = next(key for key in cache._cache if key[0] == mip)
    cache._cache_bytes -= cache._cache.pop(key).nbytes
    assert not cache.stamps_current(stamps), "a chunk evicted from under the frame kept it current"
    cache.get_chunk(*key)
    assert not cache.stamps_current(stamps), "a chunk re-composited under the frame kept it current"


def test_chunk_stamps_is_none_while_a_chunk_is_uncached(env):  # noqa: F811
    cache = _make("flat", _scenario(), max_chunks=ROOMY)
    w, h = cache.canvas_dims(0)
    rng = cache.chunk_index_range(0, 0, 0, w, h)
    assert cache.chunk_stamps(0, *rng) is None
    _whole(cache, 0)
    assert cache.chunk_stamps(0, *rng) is not None


# --- Step 5: the warmer alone ---------------------------------------------------


def test_a_cancelled_warmer_holds_no_canvas_item():
    """set_source() cancels the warmer; a warmer that kept its item would
    keep the old canvas item, and the cache behind it, alive."""
    import weakref

    from descape import zoom_preframe

    class Item:
        pass

    item = Item()
    ref = weakref.ref(item)
    warmer = zoom_preframe.PreframeWarmer()
    warmer.start(item, (1, -1), lambda direction: None)
    warmer.cancel()
    del item
    assert ref() is None, "a cancelled warmer still holds the canvas item"


# --- Step 5: real windows -------------------------------------------------------


def _pump(seconds: float = 0.0) -> None:
    import time

    from PyQt5.QtWidgets import QApplication

    end = time.perf_counter() + seconds
    QApplication.processEvents()
    while time.perf_counter() < end:
        QApplication.processEvents()
        time.sleep(0.002)


def _drain_warms(window) -> None:
    warmers = (window._level_warmer, window._load_warmer, window._margin_warmer)
    for _ in range(30):
        _pump()
        if not any(w.is_active for w in warmers):
            return
        for w in warmers:
            if w.is_active:
                w.run_to_completion()
    raise AssertionError("the warms never drained")


def _vary(window, sprite_unit: bool) -> None:
    """Gives the blank map structure a misplaced frame cannot hide, even at
    1.7x: TERRAIN_IDS and elevation varying tile by tile and, with
    `sprite_unit`, sprite_install's unit at the centre tile. Then re-renders.
    The unit goes through the window's own UnitEditModel, so the viewer's
    unit edits can build that model over it."""
    scenario = window.scenario
    mm = scenario.map_manager
    for tile in mm.terrain:
        tile.terrain_id = TERRAIN_IDS[(tile.x + 3 * tile.y) % len(TERRAIN_IDS)]
        tile.elevation = (tile.x // 2 + tile.y) % 3
    if sprite_unit:
        window._ensure_unit_edits().add(1, CONST, mm.map_width // 2 + 0.5, mm.map_height // 2 + 0.5)
    window._render_current()


def _preframe_window(size: int = 1200, *, varied: bool = False, sprite_unit: bool = False):
    """A shown Stepped window on the blank fixture (`varied`: _vary()'s
    content), opted in, warms drained. Scrollbars always on, so a notch never
    resizes the viewport (that case is a miss by design, not what these tests
    check)."""
    from PyQt5.QtCore import Qt

    from descape.scenario_io import BLANK_TEMPLATE_PATH

    import conftest

    window = conftest.stepped_window(BLANK_TEMPLATE_PATH, size=size)
    if varied or sprite_unit:
        _vary(window, sprite_unit)
    view = window.map_view
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    view.set_preframes_allowed(True)
    _drain_warms(window)
    return window


def _close(window) -> None:
    import conftest

    conftest.close_window(window)


def _assert_structured(pixels: np.ndarray) -> None:
    """A region a frame drawn under a wrong transform could not match by
    accident: many distinct colours, and unequal to itself one device pixel
    over on either axis."""
    colours = len(np.unique(pixels.reshape(-1, 4)[:, :3], axis=0))
    assert colours >= 64, f"only {colours} distinct colours: too flat to show a misplaced frame"
    assert not np.array_equal(pixels[1:], pixels[:-1]), "rows repeat: a one-row shift would pass"
    assert not np.array_equal(pixels[:, 4:], pixels[:, :-4]), "columns repeat: a one-column shift would pass"


def _cursor_at(monkeypatch, view, anchor) -> None:
    """Points the cursor the warmer anchors on (QCursor.pos()) at viewport point `anchor`."""
    from descape import map_view

    target = view.viewport().mapToGlobal(anchor)
    monkeypatch.setattr(map_view, "QCursor", type("Cursor", (), {"pos": staticmethod(lambda: target)}))


def _render_frames(view, direction: int) -> list:
    """What the settle timer starts, run to completion: frames for one notch
    in `direction` first, then the other."""
    view._preframe_timer.stop()
    view._preframe_dir = direction
    view._on_preframe_settle()
    view.preframe_warmer.run_to_completion()
    return list(view._canvas_item._preframes)


class _PaintLog:
    """Every live canvas paint from here on: (item._last_preframe, the
    viewport's backing-store pixels right after the canvas drew)."""

    def __init__(self, monkeypatch) -> None:
        from descape import viewer_canvas

        self.paints: list[tuple[str | None, np.ndarray | None]] = []
        real = viewer_canvas.MapCanvasItem.paint
        log = self

        def paint(item, painter, option, widget=None):
            item._last_preframe = None
            real(item, painter, option, widget)
            if widget is not None:
                log.paints.append((item._last_preframe, _backing_pixels(painter, widget)))

        monkeypatch.setattr(viewer_canvas.MapCanvasItem, "paint", paint)

    def wait(self, count: int = 1) -> tuple[str | None, np.ndarray | None]:
        import time

        end = time.perf_counter() + 5.0
        while len(self.paints) < count and time.perf_counter() < end:
            _pump(0.01)
        assert len(self.paints) >= count, "no live canvas paint arrived"
        return self.paints[count - 1]


def _backing_pixels(painter, widget) -> np.ndarray | None:
    """The viewport's device pixels in the window's backing store (raster only)."""
    from PyQt5 import sip
    from PyQt5.QtGui import QImage, QPaintEngine

    from descape import zoom_preframe

    engine = painter.paintEngine()
    if engine.type() != QPaintEngine.Raster:
        return None
    store = sip.cast(engine.paintDevice(), QImage)
    r = zoom_preframe.viewport_device_rect(widget, widget.size())
    bits = store.constBits()
    bits.setsize(store.byteCount())
    rows = np.frombuffer(bits, np.uint8).reshape(store.height(), store.bytesPerLine())
    return rows[r.top() : r.bottom() + 1, r.left() * 4 : (r.right() + 1) * 4].copy()


def _wheel(view, anchor, direction: int) -> None:
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtWidgets import QApplication

    view._end_wheel_gesture()
    vp = view.viewport()
    event = QWheelEvent(
        QPointF(anchor),
        QPointF(vp.mapToGlobal(anchor)),
        QPoint(),
        QPoint(0, 120 * direction),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(vp, event)


def _wait_renderable(window, direction: int = 1) -> None:
    """Waits until a notch in `direction` from the centre can render a frame:
    the margin warm, started by the viewport poll, caches the chunks it
    reads, and under load that takes a variable while."""
    import time

    view = window.map_view
    warmer, on_cursor = view.preframe_warmer, view._zoom_on_cursor
    end = time.perf_counter() + 20.0
    try:
        view._zoom_on_cursor = False
        while time.perf_counter() < end:
            _pump(0.05)
            _drain_warms(window)
            warmer._item = view._canvas_item  # _preframe_job() serves the warmer's item only
            if view._preframe_job(direction) is not None:
                return
        raise AssertionError(f"a notch ({direction:+d}) from the centre never became renderable")
    finally:
        warmer.cancel()
        view._zoom_on_cursor = on_cursor


def _set_device_scale(window, scale: float) -> None:
    """Zooms to device `scale` about the map's centre (the scene's corners are
    off the diamond), then _wait_renderable() for a notch in."""
    view = window.map_view
    factor = scale / view.view_device_scale()
    view.scale(factor, factor)
    view.centerOn(view._canvas_item.boundingRect().center())
    view._note_viewport_changed()
    _wait_renderable(window, 1)


def _anchor(view, where: str):
    from PyQt5.QtCore import QPoint

    rect = view.viewport().rect()
    return rect.center() if where == "centre" else QPoint(rect.right() - 2, rect.bottom() - 2)


def _hit_then_direct(monkeypatch, window, where: str, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """Renders the frames, wheels one notch at `where`, and returns (the
    backing store after the hit paint, after a normal paint of the same view)."""
    view = window.map_view
    view._zoom_on_cursor = True
    anchor = _anchor(view, where)
    _cursor_at(monkeypatch, view, anchor)
    frames = _render_frames(view, direction)
    assert frames, "no frame rendered for the predicted notches"
    log = _PaintLog(monkeypatch)
    _wheel(view, anchor, direction)
    outcome, hit = log.wait(1)
    assert outcome == "hit", f"the predicted notch missed: {outcome}"
    assert not view._canvas_item._preframes, "a hit kept its frames"
    view.viewport().update()
    outcome, direct = log.wait(2)
    assert outcome == "miss none"
    return hit, direct


@pytest.mark.gui
@pytest.mark.parametrize(
    ("where", "direction", "start"),
    [("centre", 1, None), ("corner", -1, None), ("centre", 1, (-1, 0.6)), ("corner", 1, (1, 1.7 / 1.25))],
    ids=["fit-in-centre", "fit-out-corner", "in-band-centre", "magnified-1.7-corner"],
)
def test_a_hit_equals_a_direct_paint(monkeypatch, synthetic_textures, where, direction, start):  # noqa: F811
    """Byte for byte against the backing store, on varied, textured content. `start`
    (level, residual) is where the notch starts from: one notch in from
    (-1, 0.6) lands in this task's band, and from (1, 1.36) on 1.7x
    magnification."""
    window = _preframe_window(varied=True)
    try:
        view = window.map_view
        if start is not None:
            mip, residual = start
            _set_device_scale(window, residual / view._canvas_item._cache.mip_scale(mip))
        hit, direct = _hit_then_direct(monkeypatch, window, where, direction)
        assert hit is not None, "not a raster backing store"
        _assert_structured(hit)
        assert np.array_equal(hit, direct), f"{int((hit != direct).any(axis=1).sum())} rows differ"
    finally:
        _close(window)


@pytest.mark.gui
def test_a_hit_takes_the_colour_mood_like_a_direct_paint(monkeypatch, synthetic_textures):  # noqa: F811
    """TASK-238: a frame holds ungraded pixels, so a hit must grade after its
    blit. The hit is measured against a paint of the same view with no grade."""
    from descape.colour_mood import apply, grade_for

    window = _preframe_window(varied=True)
    try:
        view = window.map_view
        grade = grade_for("Night")
        view.set_colour_grade(grade)
        _pump()
        hit, direct = _hit_then_direct(monkeypatch, window, "centre", 1)
        assert hit is not None, "not a raster backing store"
        assert np.array_equal(hit, direct), f"{int((hit != direct).any(axis=1).sum())} rows differ"

        log = _PaintLog(monkeypatch)
        view.set_colour_grade(None)
        _outcome, plain = log.wait(1)
        plain_rgb = plain.reshape(plain.shape[0], -1, 4)[..., 2::-1].astype(int)
        hit_rgb = hit.reshape(hit.shape[0], -1, 4)[..., 2::-1].astype(int)
        assert (plain_rgb != hit_rgb).any(), "the grade changed nothing"
        colours, inverse = np.unique(plain_rgb.reshape(-1, 3), axis=0, return_inverse=True)
        want = np.array([apply(grade, tuple(int(c) for c in colour)) for colour in colours])[inverse.reshape(-1)]
        assert (np.abs(hit_rgb.reshape(-1, 3) - want) <= 2).all()
    finally:
        _close(window)


@pytest.mark.gui
def test_a_walk_of_notches_hits_every_time(monkeypatch):
    """The twin's prediction over a sample of anchors and both directions,
    down the ladder and back: every notch whose frames were rendered for it
    hits, so the twin matched the real wheel zoom exactly each time."""
    from PyQt5.QtCore import QPoint

    window = _preframe_window()
    try:
        view = window.map_view
        view._zoom_on_cursor = True
        rect = view.viewport().rect()
        anchors = [rect.center(), QPoint(3, 3), QPoint(rect.right() - 3, 40), QPoint(200, rect.bottom() - 7)]
        log = _PaintLog(monkeypatch)
        outcomes = []
        for n, direction in enumerate([1] * 6 + [-1] * 6):
            anchor = anchors[n % len(anchors)]
            _cursor_at(monkeypatch, view, anchor)
            _drain_warms(window)
            if not _render_frames(view, direction):
                outcomes.append((n, "no frame"))
                continue
            _wheel(view, anchor, direction)
            outcomes.append((n, log.wait(len(log.paints) + 1)[0]))
        assert all(outcome == "hit" for _n, outcome in outcomes), outcomes
    finally:
        _close(window)


def _frames_then_wheel(monkeypatch, window, between) -> str | None:
    """Renders the frames for a notch in at the centre, runs `between()`, then
    wheels that notch; the paint's outcome."""
    view = window.map_view
    view._zoom_on_cursor = False
    assert _render_frames(view, 1), "no frame rendered"
    between()
    log = _PaintLog(monkeypatch)
    _wheel(view, view.viewport().rect().center(), 1)
    return log.wait(1)[0]


def _scroll_one_pixel(view) -> None:
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().value() + 1)


def _toggle_antialiasing(view) -> None:
    from PyQt5.QtGui import QPainter

    view.setRenderHint(QPainter.Antialiasing, not view.renderHints() & QPainter.Antialiasing)


@pytest.mark.gui
@pytest.mark.parametrize("change", ["scroll", "render hints"])
def test_a_view_change_after_the_render_misses_on_the_key(monkeypatch, change):
    window = _preframe_window()
    try:
        view = window.map_view
        between = {"scroll": lambda: _scroll_one_pixel(view), "render hints": lambda: _toggle_antialiasing(view)}[change]
        assert _frames_then_wheel(monkeypatch, window, between) == "miss key"
    finally:
        _close(window)


class _Viewport:
    """The real viewport with its size, offset in its window or DPR replaced, for _blit_preframe()."""

    def __init__(self, real, size, offset, dpr) -> None:
        self._real, self._size, self._offset, self._dpr = real, size, offset, dpr

    def size(self):
        return self._size

    def mapTo(self, _window, point):
        return self._offset + point

    def devicePixelRatioF(self) -> float:
        return self._dpr

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.gui
def test_each_key_field_changed_alone_at_the_paint_misses():
    """_blit_preframe() itself, handed a rendered frame's own paint state with
    one input changed: each misses on the key and the unchanged state hits,
    so the paint derives every field the prediction keyed on."""
    from PyQt5.QtCore import QPoint, QRectF, QSize
    from PyQt5.QtGui import QImage, QPainter, QTransform

    window = _preframe_window()
    try:
        view = window.map_view
        view._zoom_on_cursor = False
        item = view._canvas_item
        frame = _render_frames(view, 1)[0]
        terms, (x, y, w, h), (vw, vh), (ox, oy), hints, dpr, _bounds = frame.key
        bounding = QRectF(item._bounding_rect)
        one_ulp = list(terms)
        one_ulp[6] = float(np.nextafter(terms[6], np.inf))
        base = {
            "terms": terms, "rect": QRectF(x, y, w, h), "size": QSize(vw, vh), "offset": QPoint(ox, oy),
            "hints": hints, "dpr": dpr, "bounding": bounding, "mip": frame.mip,
        }
        changed = {
            "transform": {"terms": tuple(one_ulp)},
            "rect": {"rect": QRectF(x, y, w, h + 1)},
            "viewport size": {"size": QSize(vw, vh - 1)},
            "viewport offset": {"offset": QPoint(ox + 1, oy)},
            "hints": {"hints": hints ^ int(QPainter.Antialiasing)},
            "dpr": {"dpr": 2.0},
            "bounds": {"bounding": bounding.adjusted(0, 0, 0, 1)},
            "level": {"mip": frame.mip + 1},
        }

        def paint(state) -> str | None:
            image = QImage(16, 16, QImage.Format_RGB32)
            painter = QPainter(image)
            try:
                painter.setRenderHints(painter.renderHints(), False)
                painter.setRenderHints(QPainter.RenderHints(state["hints"]), True)
                painter.setWorldTransform(QTransform(*state["terms"]))
                item._bounding_rect = state["bounding"]
                item._last_preframe = None
                viewport = _Viewport(view.viewport(), state["size"], state["offset"], state["dpr"])
                item._blit_preframe(painter, state["rect"], viewport, state["mip"])
            finally:
                painter.end()
                item._bounding_rect = bounding
            return item._last_preframe

        outcomes = {field: paint({**base, **change}) for field, change in changed.items()}
        assert all(outcome == "miss key" for outcome in outcomes.values()), outcomes
        assert paint(base) == "hit"
    finally:
        _close(window)


def _centre_tile(window) -> tuple[int, int]:
    view = window.map_view
    tile = view._pick_tile(view.mapToScene(view.viewport().rect().center()))
    assert tile is not None, "no tile under the view's centre"
    return tile


def _elevate(window, tile) -> None:
    window._on_tool_selected("elevation")
    window.brush_size_spin.setValue(3)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(*tile, 0)
    window.on_edit_stroke_end()


def _terrain_tool(window) -> None:
    window.mode_combo.setCurrentText("Terrain")
    window._on_tool_selected("draw")
    assert window.terrain_panel.set_terrain(NEW_TERRAIN_ID)
    window.paint_trees_check.setChecked(False)
    window.paint_eye_candy_check.setChecked(False)


def _paint_terrain(window, tile) -> None:
    window.brush_size_spin.setValue(3)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tile(*tile, 0)
    window.on_edit_stroke_end()


def _place_unit(window, monkeypatch, x: float, y: float) -> None:
    """Place Unit's real handler, the click resolved to (x, y): a GAIA
    sprite_install unit, Free placement on (its const is in no catalog)."""
    from PyQt5.QtCore import Qt

    monkeypatch.setattr(window, "_placement_point", lambda _pos, _mods, unit_const=None: (x, y))
    monkeypatch.setattr(window.units_panel, "selected_object_const", lambda: CONST)
    window.units_panel.select_owner(0)
    window.unrestricted_check.setChecked(True)
    gaia = window.scenario.unit_manager.units[0]
    count = len(gaia)
    window.on_unit_place(None, Qt.NoModifier)
    assert len(gaia) == count + 1, "Place Unit placed nothing"


def _eraser_tool(window) -> None:
    window._on_tool_selected("eraser")
    window.brush_size_spin.setValue(1)
    for name, action in window.eraser_target_actions.items():
        action.setChecked(name == "gaia")


def _erase(window, tile) -> None:
    """An Eraser stroke on `tile`, flushed at its release (TASK-326)."""
    gaia = window.scenario.unit_manager.units[0]
    count = len(gaia)
    window.on_edit_stroke_start()
    window.on_edit_stroke_tiles([tile], 0)
    window._erase_refresh_timer.stop()
    assert window._unit_flush_owed(), "the stroke owes no flush, so the case skips the path it is for"
    window.on_edit_stroke_end()
    assert len(gaia) == count - 1, "the stroke erased nothing"


def _drop_free_invalidate_region(monkeypatch) -> None:
    """MapCanvasItem.invalidate_region() without its frame drop, so only the
    chunk stamps can void a frame."""
    from PyQt5.QtCore import QRectF

    from descape import viewer_canvas

    monkeypatch.setattr(
        viewer_canvas.MapCanvasItem,
        "invalidate_region",
        lambda self, bbox: self.update(QRectF(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])),
    )


def _frame_pixels(frame) -> np.ndarray:
    image = frame.image
    bits = image.constBits()
    bits.setsize(image.byteCount())
    return np.frombuffer(bits, np.uint8).reshape(image.height(), image.bytesPerLine())[:, : image.width() * 4].copy()


@pytest.mark.gui
def test_each_viewer_edit_voids_a_frame_through_its_stamps_alone(monkeypatch, sprite_install):  # noqa: F811
    """The viewer's own edit paths, each between a frame's render and its
    notch, with invalidate_region()'s drop gone: a stroke, its undo, a
    terrain stroke, the grid toggle, a Place Unit, an Eraser stroke (whose
    unit edits reach the cache at its own flush) and the sprite toggle each
    change the frame's pixels and miss on stale stamps; a style switch
    replaces the canvas item, so its notch finds no frame. On an unreduced
    level, so a hit's check would see stale content (the check mode is on)."""
    from descape import iso_geometry, zoom_preframe

    _drop_free_invalidate_region(monkeypatch)
    window = _preframe_window(sprite_unit=True)
    try:
        view = window.map_view
        _set_device_scale(window, 0.6 / view._canvas_item._cache.mip_scale(-1))
        # Centred on the sprite's tile, so the sprite toggle changes the frame.
        mm, proj = window.scenario.map_manager, view._iso_proj
        cx, cy = mm.map_width // 2, mm.map_height // 2
        sx, sy = iso_geometry.tile_screen_origin(cx, cy, mm.get_tile(cx, cy).elevation, proj)
        view.centerOn(sx + proj.half_w, sy + proj.half_h)
        view._zoom_on_cursor = False
        tile = _centre_tile(window)
        beside = (tile[0] + 1, tile[1])
        # (name, a tool switch before the render, since it may change the key, the edit, the outcome)
        mutations = [
            ("elevation stroke", None, lambda: _elevate(window, tile), "miss stale"),
            ("undo", None, window.undo, "miss stale"),
            ("terrain stroke", lambda: _terrain_tool(window), lambda: _paint_terrain(window, tile), "miss stale"),
            ("grid toggle", None, lambda: window._on_grid_overlay_toggled(not window.grid_action.isChecked()), "miss stale"),
            ("place unit", None, lambda: _place_unit(window, monkeypatch, beside[0] + 0.5, beside[1] + 0.5), "miss stale"),
            ("eraser stroke", lambda: _eraser_tool(window), lambda: _erase(window, beside), "miss stale"),
            ("sprite toggle", None, lambda: window._on_sprites_toggled(False), "miss stale"),
            ("style switch", None, lambda: window.on_terrain_style_changed("Flat"), "miss none"),
        ]
        log = _PaintLog(monkeypatch)
        failures, direction = [], 1
        for name, prepare, mutate, want in mutations:
            if prepare is not None:
                prepare()
            _wait_renderable(window, direction)
            frames = _render_frames(view, direction)
            assert frames, f"{name}: no frame rendered"
            before = _frame_pixels(frames[0])
            # The edit restarts the settle wait; it must not re-render before the notch.
            view._preframe_timer.setInterval(10**7)
            try:
                mutate()
                _pump(0.2)
                count = len(log.paints)
                _wheel(view, view.viewport().rect().center(), direction)
                outcome, after = log.wait(count + 1)
            finally:
                view._preframe_timer.setInterval(zoom_preframe.SETTLE_MS)
            if outcome != want:
                failures.append(f"{name}: {outcome}, want {want}")
            if after is None or np.array_equal(before, after):
                failures.append(f"{name}: the frame's pixels did not change, so the case proves nothing")
            if zoom_preframe.check_failures:
                failures.append(f"{name}: {zoom_preframe.check_failures}")
                zoom_preframe.check_failures.clear()
            direction = -direction
        assert not failures, failures
    finally:
        _close(window)


@pytest.mark.gui
def test_the_check_mode_catches_an_edit_that_skips_its_stamp(monkeypatch):
    """The inject: with _stamp() a no-op and invalidate_region()'s drop gone,
    an edit leaves a stale frame current, and the hit's check records it. On
    a level paint() reads unreduced: TASK-031.70's reduced entries are keyed
    on the same stamps, so there the check's fresh render would be stale too."""
    from descape import zoom_preframe
    from descape.render_cache import _ChunkCacheBase

    assert zoom_preframe.CHECK, "the suite runs with DESCAPE_PREFRAME_CHECK=1 (conftest.py)"
    window = _preframe_window()
    try:
        view = window.map_view
        _set_device_scale(window, 0.6 / view._canvas_item._cache.mip_scale(-1))
        assert view._canvas_item.reduction_at(-1, view.view_device_scale()) == 0
        tile = _centre_tile(window)

        def inject():
            monkeypatch.setattr(_ChunkCacheBase, "_stamp", lambda self, key: None)
            _drop_free_invalidate_region(monkeypatch)
            _elevate(window, tile)
            # The stroke's patch lands on the event loop; the settle timer must not re-render first.
            view._preframe_timer.setInterval(10**7)
            _pump(0.2)
            assert view._canvas_item._preframes, "the edit dropped the frames"

        assert _frames_then_wheel(monkeypatch, window, inject) == "hit"
        assert zoom_preframe.check_failures, "a stale hit passed the check"
        zoom_preframe.check_failures.clear()
    finally:
        _close(window)


@pytest.mark.gui
def test_the_check_mode_catches_a_key_that_leaves_out_a_paint_input(monkeypatch):
    """The inject: paint_key() without the render hints, and smooth scaling
    turned off between the render and the notch. The frame hits on the
    narrowed key; the check renders from the live paint's own hints and
    records the difference."""
    from PyQt5.QtGui import QPainter

    from descape import zoom_preframe

    real = zoom_preframe.paint_key
    monkeypatch.setattr(
        zoom_preframe,
        "paint_key",
        lambda device, rect, viewport, offset, hints, dpr, bounds: real(device, rect, viewport, offset, 0, dpr, bounds),
    )
    window = _preframe_window(varied=True)
    try:
        view = window.map_view
        _set_device_scale(window, 0.6 / view._canvas_item._cache.mip_scale(-1))
        assert view.renderHints() & QPainter.SmoothPixmapTransform
        outcome = _frames_then_wheel(monkeypatch, window, lambda: view.setRenderHint(QPainter.SmoothPixmapTransform, False))
        assert outcome == "hit"
        assert zoom_preframe.check_failures, "a hit under other render hints passed the check"
        zoom_preframe.check_failures.clear()
    finally:
        _close(window)


@pytest.mark.gui
def test_the_warmer_backs_off_while_a_button_is_held_and_retargets_on_a_cursor_move(monkeypatch):
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication

    window = _preframe_window()
    try:
        view = window.map_view
        view._zoom_on_cursor = False  # the offscreen cursor is off the viewport, which refuses a job
        warmer = view.preframe_warmer
        held = {"value": True}
        monkeypatch.setattr(QApplication, "mouseButtons", staticmethod(lambda: Qt.LeftButton if held["value"] else Qt.NoButton))
        view._on_preframe_settle()
        assert not warmer.is_active and view._preframe_timer.isActive(), "a held button let the warm start"
        held["value"] = False
        view._on_preframe_settle()
        assert warmer.is_active
        warmer.tick()
        job = warmer._job
        assert job is not None
        held["value"] = True
        for _ in range(5):
            warmer.tick()
        assert warmer._job is job and job.next_step == "fetch" and not job._blocks, "a tick ran with a button held"
        held["value"] = False
        view._zoom_on_cursor = True
        move = QMouseEvent(QEvent.MouseMove, QPointF(QPoint(40, 40)), Qt.NoButton, Qt.NoButton, Qt.NoModifier)
        view.mouseMoveEvent(move)
        assert not warmer.is_active, "a cursor move left the old anchor's job running"
        assert view._preframe_timer.isActive(), "a cursor move did not restart the settle wait"
    finally:
        _close(window)


@pytest.mark.gui
def test_closing_the_window_leaves_no_frame_timer_running():
    """closeEvent()'s _cancel_warms() restarts the settle wait unless the view
    opted out first; a timer left running ticks on a shared event loop."""
    import conftest

    window = _preframe_window()
    view = window.map_view
    view.retarget_preframes()
    conftest.close_window(window)
    assert not view._preframe_timer.isActive(), "the settle wait outlived the window"
    assert not view.preframe_warmer.is_active


@pytest.mark.gui
def test_a_refused_notch_renders_once_the_margin_warm_drains(monkeypatch):
    """The notch out refused once at the settle (as when its chunks are not
    cached yet; test_a_frame_never_composites_a_chunk has the real refusal),
    then a real margin warm over evicted ring chunks: its drain renders it."""
    window = _preframe_window()
    try:
        view = window.map_view
        view._zoom_on_cursor = False
        _set_device_scale(window, 0.7)
        real_job, refusals = view._preframe_job, {-1: 1}

        def job(direction):
            if refusals.get(direction):
                refusals[direction] -= 1
                return None
            return real_job(direction)

        monkeypatch.setattr(view, "_preframe_job", job)
        frames = _render_frames(view, 1)
        assert len(frames) == 1 and view.preframe_warmer.refused == [-1], (len(frames), view.preframe_warmer.refused)
        from descape import margin_warm

        mip, cx0, cy0, cx1, cy1 = view.viewport_chunk_target()
        cache = view._canvas_item._cache
        # The whole ring, cached or not: ring_chunks() drops resident chunks, and the warms may hold them all.
        with monkeypatch.context() as m:
            m.setattr(cache, "has_chunk", lambda *_args: False)
            ring = margin_warm.ring_chunks(cache, mip, cx0, cy0, cx1, cy1)
        assert ring, "no ring around the view to warm"
        for key in [(mip, *c) for c in ring if (mip, *c) in cache._cache]:
            cache._cache_bytes -= cache._cache.pop(key).nbytes
        window._on_viewport_changed()
        _drain_warms(window)
        view.preframe_warmer.run_to_completion()
        assert len(view._canvas_item._preframes) == 2, "the refused notch was never rendered"
        assert view.preframe_warmer.refused == []
    finally:
        _close(window)


@pytest.mark.gui
def test_a_frame_never_composites_a_chunk(monkeypatch):
    """With the view's chunks evicted, no frame renders and nothing is inserted."""
    from descape.render_cache import _ChunkCacheBase

    window = _preframe_window()
    try:
        view = window.map_view
        cache = view._canvas_item._cache
        for key in list(cache._cache):
            cache._cache_bytes -= cache._cache.pop(key).nbytes
        inserts = []
        real = _ChunkCacheBase._insert
        monkeypatch.setattr(_ChunkCacheBase, "_insert", lambda self, *a, **k: inserts.append(a) or real(self, *a, **k))
        assert _render_frames(view, 1) == []
        assert inserts == []
    finally:
        _close(window)


@pytest.mark.gui
def test_strips_reassemble_the_one_clipped_draw(monkeypatch):
    """Step 1b's finding: a frame drawn in STRIP_ROWS strips equals one draw
    clipped to the viewport, at 1.7x magnification (where an unclipped draw
    differs near the edge)."""
    from descape import zoom_preframe

    window = _preframe_window()
    try:
        view = window.map_view
        cache = view._canvas_item._cache
        _set_device_scale(window, 1.7 / cache.mip_scale(cache.mip_levels()[-1]))
        view._zoom_on_cursor = False
        view.preframe_warmer._item = view._canvas_item  # _preframe_job() serves the warmer's item only
        images = []
        for rows in (37, 10**6):
            monkeypatch.setattr(zoom_preframe, "STRIP_ROWS", rows)
            job = view._preframe_job(1)
            assert job is not None
            images.append(job.run_to_completion().image)
        a, b = images
        assert a.format() == b.format() and a.size() == b.size()
        assert a == b
    finally:
        _close(window)


def _offset_viewport(window, odd: bool) -> None:
    """Viewport margins that put the viewport at an even offset in its window
    with an even size (`odd`: at an odd offset), then the warms drained."""
    from PyQt5.QtCore import QPoint

    view = window.map_view
    view.setViewportMargins(0, 0, 0, 0)
    _pump(0.05)
    vp = view.viewport()
    offset = vp.mapTo(vp.window(), QPoint())
    left, top = (offset.x() + odd) % 2, (offset.y() + odd) % 2
    view.setViewportMargins(left, top, (vp.width() - left) % 2, (vp.height() - top) % 2)
    _pump(0.05)
    _drain_warms(window)


_DPR_SCRIPT = """
import functools, sys, tempfile
from pathlib import Path
sys.path[:0] = ["tests", "."]
from testkit import qt_window, settings_isolation
qt_window.ensure_qapp()
settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="preframe_dpr_")))
import pytest, numpy as np
from PyQt5.QtCore import QPoint
import test_zoom_preframe as t
from test_native_composite import _texture
from descape import asset_source, zoom_preframe
mp = pytest.MonkeyPatch()
# The synthetic_textures fixture's patch, on its own MonkeyPatch: mp.undo() runs per case.
textures = pytest.MonkeyPatch()
textures.setattr(asset_source, "get_terrain_texture_array", functools.lru_cache(_texture))
window = t._preframe_window(size=900, varied=True)
try:
    view = window.map_view
    vp = view.viewport()
    print("DPR", vp.devicePixelRatioF())
    for odd in (False, True):
        t._offset_viewport(window, odd)
        aligned = zoom_preframe.device_aligned(vp.mapTo(vp.window(), QPoint()), vp.size(), vp.devicePixelRatioF())
        for where, direction in (("centre", 1), ("corner", -1)):
            if aligned:
                hit, direct = t._hit_then_direct(mp, window, where, direction)
                t._assert_structured(hit)
                ok = np.array_equal(hit, direct)
            else:
                # Refused, and by the alignment check alone: with it passed, the same notch renders.
                view._zoom_on_cursor = False
                ok = t._render_frames(view, direction) == []
                mp.setattr(zoom_preframe, "device_aligned", lambda *_args: True)
                ok = ok and t._render_frames(view, direction) != []
                view._canvas_item.drop_preframes()
            print("RESULT", "odd" if odd else "even", "aligned" if aligned else "refused", where, ok)
            mp.undo()
    print("CHECK", len(zoom_preframe.check_failures))
finally:
    mp.undo()
    t._close(window)
    textures.undo()
"""


@pytest.mark.gui
@pytest.mark.parametrize("scale", ["2", "1.5"])
def test_a_hit_equals_a_direct_paint_at_hidpi(scale):
    """QT_SCALE_FACTOR needs its own process (test_mip_viewer.py's HiDPI
    probe). At DPR 2 every viewport offset lands on whole device pixels, so
    both offsets hit byte for byte; at 1.5 an odd one would not, and is
    refused by zoom_preframe.device_aligned() (with it patched True, the
    same notch renders)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    child_env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=scale, DESCAPE_PREFRAME_CHECK="1")
    child_env.pop("QT_AUTO_SCREEN_SCALE_FACTOR", None)
    proc = subprocess.run(
        [sys.executable, "-c", _DPR_SCRIPT], cwd=root, env=child_env, capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = proc.stdout.splitlines()
    assert f"DPR {float(scale)}" in lines, proc.stdout
    odd_kind = "aligned" if scale == "2" else "refused"
    want = [f"RESULT {odd} {kind} {where} True" for odd, kind in (("even", "aligned"), ("odd", odd_kind)) for where in ("centre", "corner")]
    assert [ln for ln in lines if ln.startswith("RESULT")] == want, proc.stdout
    assert "CHECK 0" in lines, proc.stdout
