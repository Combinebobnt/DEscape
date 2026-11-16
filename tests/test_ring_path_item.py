"""TASK-508's RingPathItem on a bare scene and view: culled paints are
byte-equal to a plain QGraphicsPathItem of the exact path, the shared memo
hits and misses by its key, every coarser level stays inside its error bound
both ways, and the bounds never stroke the path."""

from __future__ import annotations

import math

import numpy as np
import pytest

import conftest

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

VIEW_W, VIEW_H = 320, 240


def _circle(cx: float, cy: float, r: float, n: int = 900, start: float = 0.0) -> np.ndarray:
    angles = start + np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    # Half-pixel rounding makes a staircase of short steps, like a Sloped ring.
    return np.round(np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1) * 2.0) / 2.0


def _band(x0: float, x1: float, y: float, step: float = 3.0) -> np.ndarray:
    """A thin closed loop starting at (x0, y): right along y, back along y + 6."""
    xs = np.arange(x0, x1, step)
    return np.vstack([np.stack([xs, np.full_like(xs, y)], axis=1), np.stack([xs[::-1], np.full_like(xs, y + 6.0)], axis=1)])


def _pens():
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QBrush, QColor, QPen

    strong = QPen(QColor(255, 0, 255), 2)
    strong.setCosmetic(True)
    dim = QPen(QColor(255, 0, 255, 110), 1)
    dim.setCosmetic(True)
    return [
        ("strong", strong, QBrush(Qt.NoBrush)),
        ("dim", dim, QBrush(Qt.NoBrush)),
        ("fill", QPen(Qt.NoPen), QBrush(QColor(0, 160, 255, 90))),
    ]


class _Stage:
    """A shown scene and view holding a RingPathItem and its plain twin per pen."""

    def __init__(self, rings: list[np.ndarray], deviation: float = 0.0, chords=None) -> None:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication, QGraphicsPathItem, QGraphicsScene, QGraphicsView

        from descape.ring_path_item import RingGeometry, RingPathItem, RingSet
        from testkit import qt_window

        qt_window.ensure_qapp()
        chords = chords or rings
        geometries = [RingGeometry(r, c, deviation) for r, c in zip(rings, chords, strict=True)]
        self.ring_set = RingSet(geometries, 4.0)
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-6000, -6000, 12000, 12000)
        self.pairs = []
        for _name, pen, brush in _pens():
            ours = RingPathItem(self.ring_set, pen, brush)
            plain = QGraphicsPathItem(self.ring_set.exact_path)
            plain.setPen(pen)
            plain.setBrush(brush)
            for item in (ours, plain):
                self.scene.addItem(item)
            self.pairs.append((ours, plain))
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.resize(VIEW_W, VIEW_H)
        self.view.show()
        QApplication.processEvents()

    def close(self) -> None:
        self.view.close()

    def place(self, transform) -> None:
        from PyQt5.QtWidgets import QApplication

        self.view.setTransform(transform)
        self.view.centerOn(0.0, 0.0)
        QApplication.processEvents()

    def show_only(self, index: int, ours: bool) -> None:
        for i, (a, b) in enumerate(self.pairs):
            a.setVisible(i == index and ours)
            b.setVisible(i == index and not ours)

    def captures(self) -> list[bytes]:
        """A viewport grab (exposed = the view) plus view.render() of sub-rects."""
        from PyQt5.QtCore import QRect, QRectF
        from PyQt5.QtGui import QImage, QPainter

        out = [_bits(self.view.viewport().grab().toImage())]
        for rect in (QRect(150, 110, 24, 24), QRect(0, 0, 120, 90), QRect(200, 140, 120, 100), QRect(30, 115, 260, 12)):
            image = QImage(rect.width(), rect.height(), QImage.Format_ARGB32)
            image.fill(0)
            painter = QPainter(image)
            self.view.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)
            painter.end()
            out.append(_bits(image))
        return out


def _bits(image) -> bytes:
    from PyQt5.QtGui import QImage

    image = image.convertToFormat(QImage.Format_ARGB32)
    ptr = image.constBits()
    ptr.setsize(image.byteCount())
    return bytes(ptr)


def _transforms():
    from PyQt5.QtGui import QTransform

    iso = QTransform().scale(2.0, 2.0).scale(1.0, 0.5).rotate(-45)
    return [("x1", QTransform()), ("x3", QTransform().scale(3.0, 3.0)), ("flat iso x2", iso)]


def _assert_byte_equal(rings: list[np.ndarray]) -> None:
    stage = _Stage(rings)
    try:
        for label, transform in _transforms():
            stage.place(transform)
            for index, (name, _pen, _brush) in enumerate(_pens()):
                stage.show_only(index, ours=False)
                plain = stage.captures()
                stage.show_only(index, ours=True)
                ours = stage.captures()
                assert ours == plain, f"{label} {name}: RingPathItem differs from the plain item"
    finally:
        stage.close()


CASES = {
    "partly inside": [_circle(150, 0, 100)],
    "encircles the view, no chunk visible": [_circle(0, 0, 900, n=1800)],
    "start inside, the rest out": [_band(0, 4000, -20)],
    "wholly outside": [_circle(3000, 3000, 200), _circle(20, 10, 30, n=120)],
    "two overlapping": [_circle(-150, 0, 200), _circle(150, 0, 200), _circle(-40, 0, 60), _circle(40, 0, 60)],
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_culled_ring_is_byte_equal_to_the_plain_item(case: str) -> None:
    _assert_byte_equal(CASES[case])


def test_a_chunk_boundary_point_stays_exact_where_its_edge_crosses_the_view() -> None:
    """Point 64 opens a chunk far off screen, and the edge into it from point
    63 crosses the view at a shallow slope. As the last point of a visible
    chunk too it must stay exact: clamped, that edge would steepen on screen."""
    from descape.ring_path_item import RING_CHUNK_POINTS

    n = RING_CHUNK_POINTS
    inside = np.stack([np.linspace(-100.0, 50.0, n), np.full(n, 10.0)], axis=1)
    away = np.stack([np.linspace(3000.0, 3600.0, 2 * n), np.linspace(70.0, 3000.0, 2 * n)], axis=1)
    back = np.stack([np.linspace(3600.0, -3000.0, 2 * n), np.full(2 * n, 3000.0)], axis=1)
    home = np.stack([np.full(2 * n, -3000.0), np.linspace(3000.0, 10.0, 2 * n)], axis=1)
    ring = np.vstack([inside, away, back, home])
    _assert_byte_equal([ring])


def _diagonal_exit() -> np.ndarray:
    """Chunk 0 sits far off screen and leaves through a long diagonal
    edge across the view into chunk 1, also off screen: only that edge's after
    point puts chunk 0's box over the view."""
    from descape.ring_path_item import RING_CHUNK_POINTS

    n = RING_CHUNK_POINTS
    far = np.stack([np.linspace(-3000.0, -2900.0, n), np.full(n, -2300.0)], axis=1)
    near = np.stack([np.linspace(2900.0, 3000.0, n), np.full(n, 2300.0)], axis=1)
    return np.vstack([far, near, [[3000.0, -2600.0]]])


def _closing_edge_crosses() -> np.ndarray:
    """Chunk 0 is off screen, and the ring's closing edge back to point 0
    crosses the view at a shallow slope: point 0 must stay exact."""
    from descape.ring_path_item import RING_CHUNK_POINTS

    n = RING_CHUNK_POINTS
    down = np.stack([np.full(n, -3000.0), np.linspace(50.0, 3000.0, n)], axis=1)
    over = np.stack([np.linspace(-2900.0, 3000.0, n), np.full(n, 3000.0)], axis=1)
    up = np.stack([np.full(n, 3000.0), np.linspace(2900.0, -60.0, n)], axis=1)
    return np.vstack([down, over, up])


EDGE_CASES = {
    "diagonal exit": [_diagonal_exit()],
    "closing edge crosses": [_closing_edge_crosses()],
    "several rings, mixed chunks": CASES["two overlapping"] + [_closing_edge_crosses(), _circle(150, 0, 100)],
}


@pytest.mark.parametrize("case", list(EDGE_CASES))
def test_a_culled_edge_case_is_byte_equal_to_the_plain_item(case: str) -> None:
    _assert_byte_equal(EDGE_CASES[case])


def _render(path, world, pen, brush) -> bytes:
    from PyQt5.QtGui import QImage, QPainter

    image = QImage(VIEW_W, VIEW_H, QImage.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    painter.setWorldTransform(world)
    painter.setPen(pen)
    painter.setBrush(brush)
    painter.drawPath(path)
    painter.end()
    return _bits(image)


@pytest.mark.parametrize("case", list(CASES) + list(EDGE_CASES))
def test_the_clamp_alone_is_exact_at_the_tight_margin(case: str) -> None:
    """_Level.clamped() against the view grown by just the margin, with no memo
    cell snapping to push the clamp far off screen: the clamped path paints
    byte-equal to the exact one."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QTransform

    from descape.ring_path_item import RingGeometry, RingSet, min_axis_scale
    from testkit import qt_window

    qt_window.ensure_qapp()
    rings = (CASES | EDGE_CASES)[case]
    ring_set = RingSet([RingGeometry(r, r, 0.0) for r in rings], 4.0)
    level = ring_set._level(0)
    for label, transform in _transforms():
        world = transform * QTransform.fromTranslate(VIEW_W / 2, VIEW_H / 2)
        view = world.inverted()[0].mapRect(QRectF(0, 0, VIEW_W, VIEW_H))
        grow = ring_set.margin_px / min_axis_scale(world)
        clamped = level.clamped(view.left() - grow, view.top() - grow, view.right() + grow, view.bottom() + grow)
        for name, pen, brush in _pens():
            exact = _render(level.path, world, pen, brush)
            assert _render(clamped, world, pen, brush) == exact, f"{label} {name}: the clamp changed pixels"


def test_the_clamp_rect_holds_the_view_grown_by_the_margin() -> None:
    from PyQt5.QtCore import QRectF

    from descape.ring_path_item import RingGeometry, RingSet, min_axis_scale

    rings = RingSet([RingGeometry(_circle(0, 0, 50), _circle(0, 0, 50), 0.0)], 4.0)
    for _label, transform in _transforms():
        world = transform.scale(0.7, 0.7)
        # Each edge sits just inside a 64 px cell line, so snapping alone cannot cover the margin.
        exposed = QRectF(-127.9, -63.9, 255.8, 127.8)
        _index, (x0, y0, x1, y1) = rings.paint_key(world, exposed)
        grow = rings.margin_px / min_axis_scale(world)
        assert x0 <= exposed.left() - grow and y0 <= exposed.top() - grow
        assert x1 >= exposed.right() + grow and y1 >= exposed.bottom() + grow


def test_the_memo_hits_and_misses_by_its_key() -> None:
    """Decision 4: fill and outline share one memo; a hover repaint and a pan
    or zoom inside the same snapped cells hit, a long pan and a level change miss."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QApplication

    from descape.viewer_canvas import _max_axis_scale

    stage = _Stage([_circle(0, 0, 300, n=3000), _band(-2000, 2000, 40)])
    view, rings = stage.view, stage.ring_set
    try:
        stage.place(QTransform().scale(2.0, 2.0))
        for ours, plain in stage.pairs:
            plain.setVisible(False)
            ours.setVisible(True)

        def key():
            world = view.viewportTransform()
            return rings.paint_key(world, view.mapToScene(view.viewport().rect()).boundingRect(), view.viewport())

        def step(action) -> tuple[int, int, tuple, tuple]:
            before, hits, misses = key(), rings.memo_hits, rings.memo_misses
            action()
            view.viewport().repaint()
            QApplication.processEvents()
            return rings.memo_hits - hits, rings.memo_misses - misses, before, key()

        def assert_paints_its_level(label: str) -> None:
            """Each item's paint equals a plain item of the picked level's whole path."""
            from PyQt5.QtWidgets import QGraphicsPathItem

            index = rings.level_index_for(_max_axis_scale(view.viewportTransform()))
            for ours, _plain in stage.pairs:
                ours.setVisible(False)
            for ours, _plain in stage.pairs:
                ours.setVisible(True)
                ours_bits = _bits(view.viewport().grab().toImage())
                ours.setVisible(False)
                twin = QGraphicsPathItem(rings._level(index).path)
                twin.setPen(ours.pen())
                twin.setBrush(ours.brush())
                stage.scene.addItem(twin)
                twin_bits = _bits(view.viewport().grab().toImage())
                stage.scene.removeItem(twin)
                assert ours_bits == twin_bits, f"{label}: {ours.brush().style()} paints other than its level"
            for ours, _plain in stage.pairs:
                ours.setVisible(True)

        view.viewport().repaint()
        hits, misses, _before, _after = step(lambda: None)
        assert misses == 0 and hits >= 3, "a repeat full repaint must hit for all three items"
        assert_paints_its_level("full repaint")
        hits, misses, _before, _after = step(lambda: view.viewport().repaint(QRect(100, 100, 64, 64)))
        assert misses == 0, "a hover repaint inside the view keeps the view's key"

        def nudge():
            view.centerOn(0.5, 0.0)

        hits, misses, before, after = step(nudge)
        assert before == after and misses == 0, "a half-pixel pan stays in the same snapped cells and hits"
        assert_paints_its_level("small pan")

        def far():
            view.centerOn(400.0, 0.0)

        hits, misses, before, after = step(far)
        assert before != after and misses == 1, "a pan past a cell misses once for all three items"
        assert_paints_its_level("long pan")

        def zoom_out():
            stage.place(QTransform().scale(0.2, 0.2))

        hits, misses, before, after = step(zoom_out)
        assert before[0] != after[0] and misses == 1, "crossing a level threshold misses"
        assert before[0] == 0 and after[0] > 0, "the zoom out must reach a coarser level"
        assert_paints_its_level("coarser level")
        hits, misses, before, after = step(lambda: view.scale(1.01, 1.01))
        assert before == after and misses == 0, "a 1% zoom stays in the same level and cells and hits"
        assert_paints_its_level("small zoom")
    finally:
        stage.close()


def test_a_power_of_two_cell_lets_a_small_zoom_keep_its_key() -> None:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QTransform

    from descape.ring_path_item import RingGeometry, RingSet

    rings = RingSet([RingGeometry(_circle(0, 0, 50), _circle(0, 0, 50), 0.0)], 4.0)
    exposed = QRectF(-100, -75, 200, 150)
    a = rings.paint_key(QTransform().scale(4.0, 4.0), exposed)
    b = rings.paint_key(QTransform().scale(4.02, 4.02), QRectF(-99.5, -74.6, 199, 149.2))
    assert a == b


# -- level bounds ---------------------------------------------------------------


def _sloped_rings():
    """Single-tile Sloped rings over several corner shapes, at tile_px 64, side by side."""
    from descape import iso_geometry
    from descape.map_view_triggers import _sloped_edge_chord_deviation

    shapes = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 2, 1, 0), (2, 1, 0, 1), (0, 0, 2, 2), (3, 0, 1, 2)]
    rings, chords, deviation = [], [], 0.0
    for i, corners in enumerate(shapes):
        sides = [
            np.array(iso_geometry.sloped_tile_edge_outline(64, side, *corners), dtype=np.float64).reshape(-1, 2)
            for side in ("up_left", "up_right", "right", "left")
        ]
        offset = np.array([i * 150.0, 0.0])
        rings.append(np.vstack(sides) + offset)
        chords.append(np.vstack([s[[0, -1]] for s in sides]) + offset)
        deviation = max(deviation, *(_sloped_edge_chord_deviation(64, side, *corners) for side in ("up_left", "up_right", "right", "left")))
    return rings, chords, deviation


def _distance_to_polyline(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Each point's distance to the closed polyline through `ring`, point to segment."""
    a = ring
    b = np.roll(ring, -1, axis=0)
    d = b - a
    rel = points[:, None, :] - a[None, :, :]
    length_sq = np.einsum("ij,ij->i", d, d)
    u = np.clip(np.einsum("pij,ij->pi", rel, d) / np.where(length_sq > 0, length_sq, 1.0), 0.0, 1.0)
    gap = rel - u[:, :, None] * d[None, :, :]
    return np.sqrt(np.einsum("pij,pij->pi", gap, gap)).min(axis=1)


def _densified(ring: np.ndarray, step: float = 0.25) -> np.ndarray:
    """The polyline's vertices plus points at most `step` apart along every segment."""
    a, b = ring, np.roll(ring, -1, axis=0)
    out = [a]
    for p, q, n in zip(a, b, np.ceil(np.hypot(*(b - a).T) / step).astype(int), strict=True):
        if n > 1:
            out.append(p + (q - p) * (np.arange(1, n)[:, None] / n))
    return np.vstack(out)


def test_every_level_stays_within_its_bound_both_ways() -> None:
    from descape.ring_path_item import RingGeometry, RingSet

    rings, chords, deviation = _sloped_rings()
    assert deviation > 0, "the shapes must give the chord level something to bound"
    ring_set = RingSet([RingGeometry(r, c, deviation) for r, c in zip(rings, chords, strict=True)], 4.0)
    for index, error in ring_set.errors[1:]:
        for exact, level in zip(ring_set.level_rings(0), ring_set.level_rings(index), strict=True):
            assert len(level) >= 2
            assert _distance_to_polyline(exact, level).max() <= error + 1e-9, f"level {index}: an exact point strays"
            assert _distance_to_polyline(_densified(level), exact).max() <= error + 1e-9, f"level {index}: the level strays"
    sizes = [sum(len(r) for r in ring_set.level_rings(i)) for i, _e in ring_set.errors]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] < sizes[1], f"coarser levels must not add points: {sizes}"


def test_dedupe_drops_repeats_within_a_ring_never_across_rings() -> None:
    """A ring that ends where the next starts keeps both points; empty rings stay empty."""
    from descape.ring_path_item import RingGeometry, RingSet, simplify_rings

    rings = [
        np.array([[0, 0], [1, 0], [1, 0], [5, 5]], float),
        np.array([[5, 5], [5, 5], [6, 6], [0, 9]], float),
        np.zeros((0, 2)),
        np.array([[0, 9]], float),
    ]
    want = [[[0, 0], [1, 0], [5, 5]], [[5, 5], [6, 6], [0, 9]], [], [[0, 9]]]
    got = RingSet([RingGeometry(r, r, 0.0) for r in rings], 4.0).level_rings(0)
    assert [g.tolist() for g in got] == want
    assert [len(r) for r in simplify_rings(got, 0.5)] == [3, 3, 0, 1]


DP_RINGS = {
    # (-30, 0) runs back past the (0, 0)-(200, 0) span's start: 0 from its line, 30 from the segment.
    "hairpin past a segment end": np.array([[0, 0], [50, 0], [-30, 0], [20, 0], [200, 0], [200, 200], [0, 200]], float),
    "sub-tolerance ring": np.array([[0, 0], [0.4, 0.1], [0.5, 0.5], [0.1, 0.4]], float),
    "revisits a vertex": np.array([[0, 0], [100, 0], [100, 100], [0, 0], [-100, 100], [-100, 0]], float),
}


@pytest.mark.parametrize("case", list(DP_RINGS))
def test_dp_stays_within_its_tolerance_both_ways(case: str) -> None:
    from descape.ring_path_item import RING_DP_TOLERANCES, simplify_rings

    ring = DP_RINGS[case]
    for tolerance in RING_DP_TOLERANCES:
        (level,) = simplify_rings([ring], tolerance)
        assert len(level) >= 2, f"t={tolerance}: a ring must keep a point besides its first"
        assert np.array_equal(level[0], ring[0])
        assert _distance_to_polyline(ring, level).max() <= tolerance + 1e-9, f"t={tolerance}: an exact point strays"
        assert _distance_to_polyline(_densified(level), ring).max() <= tolerance + 1e-9, f"t={tolerance}: the level strays"


def test_the_pick_uses_the_largest_singular_value_not_levelofdetail() -> None:
    from PyQt5.QtGui import QTransform
    from PyQt5.QtWidgets import QStyleOptionGraphicsItem

    from descape.ring_path_item import RING_LOD_TOLERANCE_PX, RingGeometry, RingSet
    from descape.viewer_canvas import _max_axis_scale

    rings, chords, deviation = _sloped_rings()
    ring_set = RingSet([RingGeometry(r, c, deviation) for r, c in zip(rings, chords, strict=True)], 4.0)
    errors = dict(ring_set.errors)
    straddles = 0
    for zoom in np.geomspace(0.002, 2.0, 200):
        transform = QTransform().scale(zoom, zoom).scale(1.0, 0.5).rotate(-45)
        scale = _max_axis_scale(transform)
        index = ring_set.level_index_for(scale)
        assert errors[index] * scale <= RING_LOD_TOLERANCE_PX
        coarser = [i for i, e in ring_set.errors if e > errors[index]]
        assert all(errors[i] * scale > RING_LOD_TOLERANCE_PX for i in coarser)
        lod = QStyleOptionGraphicsItem.levelOfDetailFromTransform(transform)
        straddles += ring_set.level_index_for(lod) != index
    assert straddles, "no zoom straddled a level, so the test shows nothing"


def test_a_sloped_ring_paints_byte_equal_where_only_exact_fits() -> None:
    from PyQt5.QtGui import QTransform

    from descape.viewer_canvas import _max_axis_scale

    rings, chords, deviation = _sloped_rings()
    rings = [r - np.array([300.0, 0.0]) for r in rings]
    chords = [c - np.array([300.0, 0.0]) for c in chords]
    stage = _Stage(rings, deviation, chords)
    try:
        for zoom in (1.0, 2.5):
            transform = QTransform().scale(zoom, zoom)
            assert stage.ring_set.level_index_for(_max_axis_scale(transform)) == 0
            stage.place(transform)
            for index in range(len(stage.pairs)):
                stage.show_only(index, ours=False)
                plain = stage.captures()
                stage.show_only(index, ours=True)
                assert stage.captures() == plain
    finally:
        stage.close()


# -- bounds and shape -----------------------------------------------------------


def test_bounds_contain_the_plain_items_and_never_stroke() -> None:
    from PyQt5.QtWidgets import QGraphicsPathItem

    from descape.ring_path_item import RingGeometry, RingPathItem, RingSet

    calls = []

    class Counting(RingPathItem):
        def shape(self):
            calls.append(1)
            return super().shape()

    ring = _circle(0, 0, 80)
    ring_set = RingSet([RingGeometry(ring, ring, 0.0), RingGeometry(_band(-50, 400, 30), _band(-50, 400, 30), 0.0)], 4.0)
    for name, pen, brush in _pens():
        item = Counting(ring_set, pen, brush)
        calls.clear()
        bounds = item.boundingRect()
        # Qt's own QGraphicsPathItem::boundingRect() strokes through the virtual shape() for a pen.
        assert not calls, f"{name}: boundingRect() went through shape()"
        plain = QGraphicsPathItem(ring_set.exact_path)
        plain.setPen(pen)
        plain.setBrush(brush)
        assert bounds.contains(plain.boundingRect()), name
        assert item.shape() == item.path(), name
