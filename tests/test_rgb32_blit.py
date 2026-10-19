"""MapCanvasItem.paint()'s RGB32 blit gate (descape.viewer_canvas._rgb32_blit_ok).

paint() hands Qt an RGB32 copy of each block for the smooth scaled
drawImage() only where that is byte-identical to the RGB888 draw: an
axis-aligned device transform at a residual of 0.5 or more. The paint cases
render through the real paint() with the gate as shipped, forced off
("never") and forced on ("always"), and compare pixels:

- gate taken (residuals 0.5-1.0, the finest-level magnification, DPR 2):
  the shipped render equals the never render byte for byte, in RGB32.
- gate not taken (a residual just under 0.5, Flat's rotated transform): the
  shipped render equals never, and always differs, so each case is one where
  converting would really change pixels rather than a vacuous one.
- the scale-1.0 point overload: never converts, even with the gate forced on.

Those run over a noise cache, not a scenario: the blank fixture renders too
smooth (no game textures under pytest) for the under-0.5 and rotated draws
to differ at all, which would make "always differs" unprovable. One last
test drives a real window's backing-store paint for the gate's wiring.

Everything is gui-marked: the whole file needs PyQt5."""

from __future__ import annotations

import math

import numpy as np
import pytest

from descape.render_cache import _ChunkCacheBase
from descape.scenario_io import BLANK_TEMPLATE_PATH as FIXTURE_PATH
from testkit import qt_capture

import conftest

TX_SCALE = 0x02  # QTransform.TxScale
NOISE_CANVAS_PX = 4096
# Interior scene rect, sized like test_mip_viewer's minifying-band probe.
PROBE = (300.0, 300.0, 960.0, 960.0)
MAGNIFY_PROBE = (300.0, 300.0, 240.0, 240.0)


class _NoiseCache:
    """The cache members paint() touches, over a hashed-noise canvas with
    levels -2..1, so every level has full-range texel-to-texel contrast."""

    mip_levels = _ChunkCacheBase.mip_levels
    mip_scale = _ChunkCacheBase.mip_scale
    mip_for_scale = _ChunkCacheBase.mip_for_scale

    def __init__(self) -> None:
        self._mip_tile_px = {level: int(64 * 2.0**level) for level in (-2, -1, 0, 1)}

    def canvas_dims(self, mip: int = 0) -> tuple[int, int]:
        side = math.ceil(NOISE_CANVAS_PX / self.mip_scale(mip))
        return side, side

    def render_rect(self, x0, y0, x1, y1, mip=0) -> np.ndarray:
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.uint32)
        h = (xs * np.uint32(2654435761)) ^ (ys * np.uint32(2246822519)) ^ np.uint32(97 * (mip + 8))
        h ^= h >> np.uint32(15)
        h *= np.uint32(2246822519)
        h ^= h >> np.uint32(13)
        return np.stack([h & 255, (h >> 8) & 255, (h >> 16) & 255], axis=-1).astype(np.uint8)


@pytest.fixture
def noise_scene():
    from PyQt5.QtWidgets import QGraphicsScene

    from descape.viewer_canvas import MapCanvasItem

    conftest.ensure_qapp()
    scene = QGraphicsScene()
    item = MapCanvasItem(_NoiseCache())
    scene.addItem(item)
    yield scene, item
    scene.removeItem(item)


def _gate_spy(monkeypatch, mode: str) -> list[tuple[int, float, bool]]:
    """Replaces the gate with one that records (transform type, x residual,
    verdict) per paint() and returns the shipped verdict, False or True."""
    from descape import viewer_canvas

    real = viewer_canvas._rgb32_blit_ok
    calls: list[tuple[int, float, bool]] = []

    def gate(device, scale):
        verdict = {"real": real(device, scale), "never": False, "always": True}[mode]
        calls.append((int(device.type()), abs(device.m11()) * scale, verdict))
        return verdict

    monkeypatch.setattr(viewer_canvas, "_rgb32_blit_ok", gate)
    return calls


def _render(scene, zoom: float, probe=PROBE, pre=None, dpr: float = 1.0) -> np.ndarray:
    """One paint() of `probe` into an RGB32 target (the raster backing
    store's format), `zoom` logical pixels per scene unit, under an optional
    extra painter transform `pre` and target device pixel ratio `dpr`."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    rect = QRectF(*probe)
    tw, th = round(rect.width() * zoom), round(rect.height() * zoom)
    image = QImage(round(tw * dpr), round(th * dpr), QImage.Format_RGB32)
    image.setDevicePixelRatio(dpr)
    image.fill(0)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    if pre is not None:
        painter.setTransform(pre)
    scene.render(painter, QRectF(0, 0, tw, th), rect)
    painter.end()
    return qt_capture.qimage_rgb888_to_array(image.convertToFormat(QImage.Format_RGB888))


def _format_name(fmt) -> str | None:
    from PyQt5.QtGui import QImage

    return {QImage.Format_RGB32: "RGB32", QImage.Format_RGB888: "RGB888"}.get(fmt)


def _renders(monkeypatch, item, scene, modes, **kw):
    """Per mode: (pixels, gate calls, selected mip, last block's format name)."""
    out = {}
    for mode in modes:
        with monkeypatch.context() as m:
            item._last_blit_format = None
            calls = _gate_spy(m, mode)
            pixels = _render(scene, **kw)
        out[mode] = (pixels, calls, item._last_mip, _format_name(item._last_blit_format))
    return out


def _max_diff(a: np.ndarray, b: np.ndarray) -> int:
    assert a.shape == b.shape
    return int(np.abs(a.astype(int) - b.astype(int)).max())


# ---------------------------------------------------------------------------
# The gate as a pure function of the transform.
# ---------------------------------------------------------------------------


def _t(sx=1.0, sy=1.0, rot=0.0):
    from PyQt5.QtGui import QTransform

    t = QTransform()
    t.translate(13.0, -7.0)
    if rot:
        t.rotate(rot)
    t.scale(sx, sy)
    return t


@pytest.mark.gui
@pytest.mark.parametrize(
    ("transform", "scale", "expected"),
    [
        ((0.125, 0.125), 4.0, True),  # residual 0.5 exactly
        ((0.1249, 0.1249), 4.0, False),  # just under
        ((0.3, 0.3), 2.0, True),  # 0.6
        ((0.6, 0.6), 4.0, True),  # 2.4, finest-level magnification
        ((0.6, 0.1), 4.0, False),  # y under 0.5
        ((0.1, 0.6), 4.0, False),  # x under 0.5
        ((-0.3, 0.3), 2.0, True),  # mirrored, measured identical
    ],
)
def test_gate_on_axis_aligned_transforms(transform, scale, expected) -> None:
    from descape.viewer_canvas import _rgb32_blit_ok

    assert _rgb32_blit_ok(_t(*transform), scale) is expected


@pytest.mark.gui
def test_gate_refuses_flat_s_isometric_transform_and_a_30_degree_turn() -> None:
    from PyQt5.QtGui import QTransform

    from descape.map_view import MapView
    from descape.viewer_canvas import _rgb32_blit_ok

    flat = QTransform()
    flat.scale(2.0, 2.0 * MapView.ISO_VERTICAL_SQUASH)
    flat.rotate(MapView.ISO_ROTATION_DEGREES)
    assert not _rgb32_blit_ok(flat, 1.0)
    assert not _rgb32_blit_ok(_t(0.8, 0.8, rot=30.0), 1.0)


# ---------------------------------------------------------------------------
# Real paint() cycles over the noise cache.
# ---------------------------------------------------------------------------


@pytest.mark.gui
@pytest.mark.parametrize("residual", [0.5, 0.61, 0.8, 1.0])
def test_converted_blit_is_byte_identical_at_residual_half_and_up(monkeypatch, noise_scene, residual) -> None:
    from descape.viewer_canvas import MapCanvasItem

    scene, item = noise_scene
    # Several padded, overlapping blocks per draw, each converted on its own.
    monkeypatch.setattr(MapCanvasItem, "PAINT_BLOCK_PX", 64)
    out = _renders(monkeypatch, item, scene, ("real", "never"), zoom=residual / 4.0)

    pixels, calls, mip, fmt = out["real"]
    assert mip == -2
    assert calls and all(c[0] == TX_SCALE and c[2] for c in calls), calls
    assert all(c[1] == pytest.approx(residual, rel=1e-2) for c in calls), calls
    assert fmt == "RGB32"
    assert out["never"][3] == "RGB888"
    assert _max_diff(pixels, out["never"][0]) == 0


@pytest.mark.gui
@pytest.mark.parametrize("zoom", [2.5, 3.7])
def test_converted_blit_is_byte_identical_when_magnifying_past_the_finest_level(monkeypatch, noise_scene, zoom) -> None:
    scene, item = noise_scene
    out = _renders(monkeypatch, item, scene, ("real", "never"), zoom=zoom, probe=MAGNIFY_PROBE)

    pixels, calls, mip, fmt = out["real"]
    assert mip == 1
    assert calls and all(c[2] and c[1] > 1.0 for c in calls), calls
    assert fmt == "RGB32"
    assert _max_diff(pixels, out["never"][0]) == 0


@pytest.mark.gui
@pytest.mark.parametrize("residual", [0.45, 0.49])
def test_blit_stays_rgb888_just_under_residual_half(monkeypatch, noise_scene, residual) -> None:
    scene, item = noise_scene
    out = _renders(monkeypatch, item, scene, ("real", "never", "always"), zoom=residual / 4.0)

    pixels, calls, mip, fmt = out["real"]
    assert mip == -2
    assert calls and all(not c[2] and c[1] < 0.5 for c in calls), calls
    assert fmt == "RGB888"
    assert _max_diff(pixels, out["never"][0]) == 0
    # Converting here would change pixels, so this case is not vacuous.
    assert out["always"][3] == "RGB32"
    assert _max_diff(out["always"][0], out["never"][0]) > 0


@pytest.mark.gui
@pytest.mark.parametrize("flat", [True, False], ids=["flat-iso", "rotate-30"])
def test_blit_stays_rgb888_under_a_rotated_transform(monkeypatch, noise_scene, flat) -> None:
    """Flat's own squash-then-rotate halves one axis, so its residual check
    already refuses it; a plain 30-degree turn at residual 0.7 leaves the
    axis-aligned check as the only thing refusing it."""
    from PyQt5.QtGui import QTransform

    from descape.map_view import MapView

    scene, item = noise_scene
    pre = QTransform()
    pre.translate(0.0, 120.0)
    if flat:
        pre.scale(1.0, MapView.ISO_VERTICAL_SQUASH)
        pre.rotate(MapView.ISO_ROTATION_DEGREES)
    else:
        pre.rotate(-30.0)
    out = _renders(monkeypatch, item, scene, ("real", "never", "always"), zoom=0.7 / 4.0, pre=pre)

    pixels, calls, mip, fmt = out["real"]
    assert mip == -2
    assert calls and all(c[0] > TX_SCALE and not c[2] for c in calls), calls
    if not flat:
        assert all(c[1] >= 0.5 for c in calls), calls
    assert fmt == "RGB888"
    assert _max_diff(pixels, out["never"][0]) == 0
    assert _max_diff(out["always"][0], out["never"][0]) > 0


@pytest.mark.gui
def test_dpr_2_converts_on_device_residual_and_stays_identical(monkeypatch, noise_scene) -> None:
    """A logical residual of 0.3 at DPR 2 is 0.6 device pixels: taken, and a
    gate reading the world transform would refuse it."""
    scene, item = noise_scene
    out = _renders(monkeypatch, item, scene, ("real", "never"), zoom=0.3 / 4.0, dpr=2.0)

    pixels, calls, mip, fmt = out["real"]
    assert mip == -2
    assert calls and all(c[2] and c[1] == pytest.approx(0.6, rel=1e-2) for c in calls), calls
    assert fmt == "RGB32"
    assert _max_diff(pixels, out["never"][0]) == 0


@pytest.mark.gui
@pytest.mark.parametrize("zoom", [1.0, 0.7])
def test_scale_one_point_overload_never_converts(monkeypatch, noise_scene, zoom) -> None:
    """Level 0 draws with the point overload, at unit device scale or under a
    residual scale alike; forcing the gate on must change neither."""
    scene, item = noise_scene
    out = _renders(monkeypatch, item, scene, ("real", "always"), zoom=zoom, probe=MAGNIFY_PROBE)

    for mode in ("real", "always"):
        _, calls, mip, fmt = out[mode]
        assert mip == 0
        assert calls == [], f"{mode}: the point overload consulted the gate"
        assert fmt == "RGB888"
    assert _max_diff(out["real"][0], out["always"][0]) == 0


# ---------------------------------------------------------------------------
# The real window's backing-store paint.
# ---------------------------------------------------------------------------


@pytest.mark.gui
def test_a_real_viewport_paint_converts_only_where_gated() -> None:
    """Fit-to-view (residual under 0.5 at the coarsest level) stays RGB888, a
    zoom into the 0.5-1.0 band converts, and level 0 stays on the point
    overload."""
    from PyQt5.QtWidgets import QApplication

    window = conftest.stepped_window(FIXTURE_PATH, size=300)
    try:
        view = window.map_view
        item = view._canvas_item
        if item._cache.mip_levels()[0] >= 0:
            pytest.skip("fixture enumerates no level coarser than the reference")

        def paint_once():
            item._last_blit_format = None
            view.viewport().update()
            QApplication.processEvents()
            device = view.viewportTransform()
            resid = device.m11() * item._cache.mip_scale(item._last_mip)
            return item._last_mip, resid, _format_name(item._last_blit_format)

        mip, resid, fmt = paint_once()
        assert resid < 0.5, f"fit-to-view residual {resid} is not the under-0.5 case"
        assert (mip, fmt) == (item._cache.mip_levels()[0], "RGB888")

        # Each step doubles the device scale, so some step lands in [0.5, 1)
        # at a non-reference level.
        seen = []
        for _ in range(6):
            view.scale(2.0, 2.0)
            mip, resid, fmt = paint_once()
            seen.append((mip, round(resid, 3), fmt))
            if mip == 0:
                assert fmt == "RGB888", seen
            elif resid >= 0.5:
                assert fmt == "RGB32", seen
        assert any(f == "RGB32" for _, _, f in seen), seen
    finally:
        window.edit_history.mark_saved()
        window.close()
