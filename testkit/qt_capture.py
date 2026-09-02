"""Rasterizing a QGraphicsScene rect to a numpy array, once.

Five call sites across tests/ and tools/ each hand-rolled the same
QGraphicsScene.render() -> QImage -> numpy dance. Four of them were
character-identical; the fifth (tests/test_mip_viewer.py's
_render_scene_rect) deliberately renders into a target `zoom` times
larger and enables smooth filtering, so it keeps its own sizing and
render-hint policy and only shares the stride extraction below.

The split is load-bearing, not stylistic. A one-size-fits-all helper
taking `zoom=1.0` would NOT reproduce the 1:1 path: this module's
scene_rect_to_array() sizes with math.ceil() while the mip variant
truncates via int(rect.width() * zoom). On a fractional rect those
disagree by a pixel (measured: a 96.4-wide rect gives 97 vs 96), and
the 1:1 callers depend on exact sizing for byte-identity.
"""

from __future__ import annotations

import math


def qimage_rgb888_to_array(image):
    """A Format_RGB888 QImage's pixels as an (h, w, 3) uint8 array.

    QImage rows are 4-byte aligned, so bytesPerLine() != w*3 in general;
    the reshape/slice/reshape here is what strips that padding back out.
    The .copy() matters -- the array otherwise aliases the QImage's own
    buffer, which dies with the QImage.
    """
    import numpy as np

    w, h = image.width(), image.height()
    bits = image.constBits()
    bits.setsize(image.bytesPerLine() * h)
    return np.array(bits).reshape(h, image.bytesPerLine())[:, : w * 3].reshape(h, w, 3).copy()


def scene_rect_to_array(scene, rect):
    """A fixed scene-space rect's actual displayed pixels, as an (h, w, 3)
    uint8 array, via QGraphicsScene.render().

    Item-agnostic by design: it goes through the real paint dispatch
    rather than reading a specific item's pixmap, so it works the same
    whether the pixels come from a QGraphicsPixmapItem (Flat mode) or a
    custom-painted QGraphicsItem like MapCanvasItem (Stepped mode).

    **1:1 ONLY.** The target QImage is exactly `rect`'s own size, so Qt
    hands MapCanvasItem.paint() a unit device transform -> mip 0 -> the
    unscaled drawImage(point, ...) overload -- what makes byte-identity
    with descape.render.render_terrain_iso() reachable at all. Any other
    target size (a zoomed capture, a non-integer rect) selects a
    different mip and exactness silently degrades to a plausible
    near-match instead of failing loudly. Callers that need a specific
    zoom should use tests/test_mip_viewer.py's own `_render_scene_rect`
    instead -- that's what it exists for.

    2026-08-28: `rect` must have INTEGER width/height, not just "close to
    the target size". Under mip_for_scale's floor-era rule a unit device
    scale had a full octave of margin (any scale in [1.0, 2.0) selected
    mip 0), so a fractional rect that rounded up by less than a pixel was
    harmless. Under the current ceil(log2(scale)) rule mip 0 is selected
    only exactly AT scale 1.0 -- `w, h = ceil(rect.width()), ceil(rect.
    height())` below means a fractional rect yields a device scale
    fractionally ABOVE 1.0, which now silently selects mip 1 instead of
    failing loudly, the same "degrades to a near-match" failure mode the
    paragraph above warns about for a zoomed capture.
    """
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    w, h = int(math.ceil(rect.width())), int(math.ceil(rect.height()))
    image = QImage(w, h, QImage.Format_RGB888)
    image.fill(0)
    painter = QPainter(image)
    scene.render(painter, QRectF(0, 0, w, h), rect)
    painter.end()
    return qimage_rgb888_to_array(image)
