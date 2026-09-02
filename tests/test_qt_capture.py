"""Guards testkit/qt_capture.py's two-function split.

The five offscreen-capture copies this module replaced were not all the
same function, and the difference is invisible until it silently corrupts
a byte-identity comparison. These tests pin the two properties that make
collapsing them into one `zoom=1.0` helper wrong, so a future tidy-up
fails loudly here instead of quietly degrading a render assertion
somewhere else.

Deliberately built on a bare QGraphicsScene rather than a real
ViewerWindow: nothing here is about terrain, so there's no reason to pay
for a scenario load.
"""

from __future__ import annotations

import pytest

import conftest
from testkit import qt_capture

pytestmark = pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")


def _scene():
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QBrush, QColor
    from PyQt5.QtWidgets import QGraphicsScene

    scene = QGraphicsScene()
    scene.setSceneRect(QRectF(0, 0, 200, 200))
    scene.addRect(QRectF(0, 0, 200, 200), brush=QBrush(QColor(10, 20, 30)))
    scene.addRect(QRectF(20, 20, 40, 40), brush=QBrush(QColor(200, 100, 50)))
    return scene


@pytest.mark.gui
def test_padded_rows_are_stripped_not_reinterpreted() -> None:
    """The branch qimage_rgb888_to_array owns for all five callers.

    At width 97 a Format_RGB888 row is 291 bytes of pixels in a 292-byte
    stride, so a reshape that trusts w*3 would shear the image by a byte
    per row. Widths that happen to be 4-aligned (96, 240) never exercise
    this -- including, as it turns out, a real 240x240 map's full rect.
    """
    import numpy as np
    from PyQt5.QtGui import QColor, QImage

    conftest.ensure_qapp()
    w, h = 97, 8
    image = QImage(w, h, QImage.Format_RGB888)
    assert image.bytesPerLine() != w * 3, "width 97 no longer pads; pick another to keep this honest"

    # Every pixel encodes its own (x, y), so a stride mishandling shears
    # the pattern progressively instead of landing on a value that still
    # happens to look plausible. A uniform fill would pass either way.
    expected = np.empty((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            r, g, b = (x % 251), (y * 31) % 251, ((x + y) * 7) % 251
            image.setPixelColor(x, y, QColor(r, g, b))
            expected[y, x] = (r, g, b)

    arr = qt_capture.qimage_rgb888_to_array(image)
    assert arr.shape == (h, w, 3)
    assert np.array_equal(arr, expected), "row padding leaked into the pixel data"


@pytest.mark.gui
def test_one_to_one_capture_rounds_up_where_the_mip_variant_truncates() -> None:
    """Why tests/test_mip_viewer.py's _render_scene_rect stays separate.

    scene_rect_to_array sizes with math.ceil(); _render_scene_rect uses
    int(rect.width() * zoom). On a fractional rect they disagree by a
    pixel EVEN AT zoom 1.0, so a single helper defaulting zoom to 1.0
    would silently change every 1:1 caller's output shape.
    """
    from PyQt5.QtCore import QRectF

    conftest.ensure_qapp()
    scene = _scene()

    assert qt_capture.scene_rect_to_array(scene, QRectF(10, 10, 96.4, 96.4)).shape == (97, 97, 3)
    assert qt_capture.scene_rect_to_array(scene, QRectF(10, 10, 96.0, 96.0)).shape == (96, 96, 3)


@pytest.mark.gui
def test_capture_reflects_the_requested_rect_not_the_whole_scene() -> None:
    """scene_rect_to_array maps `rect` onto the target 1:1, so two
    different scene rects over differently-coloured regions must not come
    back identical -- the failure mode if the source rect were ignored."""
    from PyQt5.QtCore import QRectF

    conftest.ensure_qapp()
    scene = _scene()

    inside = qt_capture.scene_rect_to_array(scene, QRectF(25, 25, 30, 30))
    outside = qt_capture.scene_rect_to_array(scene, QRectF(120, 120, 30, 30))

    assert inside.shape == outside.shape
    assert tuple(outside[0, 0]) == (10, 20, 30), "background rect not sampled where expected"
    assert tuple(inside[0, 0]) == (200, 100, 50), "foreground rect not sampled where expected"
