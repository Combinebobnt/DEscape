"""Thumbnails for the Units panel's catalog and its two unit cards (GH #68).

A portrait decode is about 54 ms (256 px DDS, times 4 on tester hardware), so
portraits decode on one worker thread; sprites are cheap (median 0.2 ms) and
share unit_sprites' unlocked LRUs with the map renderer, so they stay on the
GUI thread. ThumbLoader keeps one decode in flight and a GUI-side pending
list that each request() replaces, so rows that scrolled away are dropped
rather than drained, and a fast scroll wastes at most one decode.

Two rules carried from GOTCHAS: the install path is resolved on the GUI
thread at request() and handed to the worker, never read lazily there; and a
result from before asset_source.clear_install_caches() or clear_art_caches()
(the install generation moved) is discarded, never shown or memoized.

View-agnostic: the catalog browser and the cards consume the same `ready`
signal. Nothing here builds a widget.
"""

from __future__ import annotations

import functools
import itertools
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap

from descape import asset_source, object_catalog, unit_sprites
from descape.viewer_common import pixmap_from_rgba

# Neutral, as constant_picker's preview: GAIA's team index, and art at the preview's own scale.
_TEAM_INDEX = 0
_HALF_W = 32
# The memo's byte budget for its RGBA thumbnails (a 48 px one is 9 KB).
MEMO_BYTES_CAP = 32 * 1024 * 1024

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
# Set by shutdown_thumb_pool(): no new pool after it, or interpreter exit would wait on its worker.
_shut_down = False


def _thumb_pool() -> ThreadPoolExecutor | None:
    """The one-worker pool, or None once shutdown_thumb_pool() has run."""
    global _pool
    with _pool_lock:
        if _pool is None and not _shut_down:
            _pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="catalog-thumb")
        return _pool


def shutdown_thumb_pool() -> None:
    """For app exit, beside asset_source.shutdown_prefetch_pool(): cancels a
    queued decode and stops the worker without waiting on a running one.
    A request after it decodes nothing."""
    global _pool, _shut_down
    with _pool_lock:
        executor, _pool = _pool, None
        _shut_down = True
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def sprite_thumb(const: int, px: int) -> np.ndarray | None:
    """const's sprite turned toward the viewer (ANGLE consts only, see
    unit_sprites.rotation_facing_viewer), every piece composited on
    transparent and fitted into px x px, as an (h, w, 4) uint8 array; or None
    with no sprite. GUI thread only: the sprite caches are unlocked."""
    rotation = unit_sprites.rotation_facing_viewer(const)
    pieces = unit_sprites.sprite_pieces_for(const, 0.0 if rotation is None else rotation, _TEAM_INDEX, _HALF_W)
    if not pieces:
        return None
    boxes = []
    for piece in pieces:
        height, width = piece.draw.rgba.shape[:2]
        x0, y0 = piece.dx - piece.draw.hotspot_x, piece.dy - piece.draw.hotspot_y
        boxes.append((x0, y0, x0 + width, y0 + height))
    min_x, min_y = min(b[0] for b in boxes), min(b[1] for b in boxes)
    canvas = np.zeros((max(b[3] for b in boxes) - min_y, max(b[2] for b in boxes) - min_x, 4), dtype=np.uint8)
    for piece, (x0, y0, _x1, _y1) in zip(pieces, boxes, strict=True):
        unit_sprites._source_over(canvas, y0 - min_y, x0 - min_x, piece.draw.rgba)
    from PIL import Image

    image = Image.fromarray(canvas)
    image.thumbnail((px, px), Image.LANCZOS)
    return np.array(image)


def _decode_portrait(install, icon: int, building: bool, px: int):
    """The worker's half: the install's portrait fitted into px, or None. Never touches a sprite."""
    return asset_source.read_unit_icon_in(install, icon, building, px)


def _icon(arr: np.ndarray | None, px: int) -> QIcon | None:
    """Centred on a square px canvas: a narrow sprite would otherwise shift its row's text left."""
    if arr is None:
        return None
    fitted = pixmap_from_rgba(arr, px)
    square = QPixmap(px, px)
    square.fill(Qt.transparent)
    painter = QPainter(square)
    painter.drawPixmap((px - fitted.width()) // 2, (px - fitted.height()) // 2, fitted)
    painter.end()
    return QIcon(square)


def team_filled(icon: QIcon | None, source: str | None, rgb: tuple[int, int, int], px: int) -> QIcon | None:
    """An "icon"-source thumbnail over a solid `rgb` square: DE's portraits
    leave the player-colour region fully transparent for the game to fill
    (479 of the install's 716 portraits). Every portrait is a 256 px square,
    so the fill never shows outside one. A sprite, or no icon, passes through."""
    if icon is None or source != "icon":
        return icon
    pixmap = QPixmap(px, px)
    pixmap.fill(QColor(*rgb))
    painter = QPainter(pixmap)
    painter.drawPixmap(0, 0, icon.pixmap(px, px))
    painter.end()
    return QIcon(pixmap)


class _Relay(QObject):
    """Carries each worker result to the GUI thread, for the app's lifetime.
    The worker holds a token, never a loader: if a worker dropped a loader's
    last reference, the QObject would be deleted off its own thread, which
    corrupts Qt state (a later segfault in an unrelated test)."""

    decoded = pyqtSignal(object)


_relay: _Relay | None = None
_tokens = itertools.count(1)
# token -> its live loader; a dropped loader leaves, so a late result has nowhere to land.
_loaders: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _gui_relay() -> _Relay:
    """Created on the GUI thread by the first loader, so its queued slot runs there."""
    global _relay
    if _relay is None:
        _relay = _Relay()
        _relay.decoded.connect(_deliver, Qt.QueuedConnection)
    return _relay


def _deliver(payload) -> None:
    token, key, arr, cancelled = payload
    loader = _loaders.get(token)
    if loader is not None:
        loader._on_decoded(key, arr, cancelled)


def _post(relay: _Relay, token: int, key, future) -> None:
    """Worker (or caller) thread: hand the result to the GUI thread by token.
    A cancelled decode is reported too, so its loader never waits on it."""
    cancelled = future.cancelled()
    arr = None
    if not cancelled:
        try:
            arr = future.result()
        except Exception:  # noqa: BLE001 -- a decode failure is just "no portrait"
            arr = None
    relay.decoded.emit((token, key, arr, cancelled))


class ThumbLoader(QObject):
    """Portrait-first thumbnails, sprite fallback, None when neither exists
    (the consumer draws its own fallback). `ready(const, px, icon)` fires once
    per resolved request; cached() answers a memo hit without one.

    One loader serves the panel's list and both cards. Each `client` (the
    list, each card) has its own pending list, which its next request()
    replaces, so one card never drops another's request. A `priority` client
    (a card) is served before every other, so a card never waits behind a
    viewport of rows, even one the list asked for later (a scroll after the
    pick); among equals the most recently requested goes first."""

    ready = pyqtSignal(int, int, object)

    def __init__(self, parent: QObject | None = None, executor=None):
        super().__init__(parent)
        self._relay = _gui_relay()
        self._token = next(_tokens)
        _loaders[self._token] = self
        # A test passes its own; the app shares the module's one-worker pool.
        self._executor = executor
        # id(client) -> (priority, its pending (const, px) served from the end); dict order is recency.
        # Keyed by id, never the client: a held card and its own loader would form a GC cycle.
        self._pending: dict[int, tuple[bool, list[tuple[int, int]]]] = {}
        self._in_flight: tuple[int, int, int] | None = None
        self._install = None
        self._request_generation = asset_source.install_generation()
        # (const, px) -> (icon, source): source "icon", "sprite" or None.
        self._memo: OrderedDict[tuple[int, int], tuple[QIcon | None, str | None]] = OrderedDict()
        self._memo_sizes: dict[tuple[int, int], int] = {}
        self._memo_bytes = 0
        self._memo_generation = self._request_generation

    # -- the API ------------------------------------------------------------------

    def cached(self, const: int, px: int) -> tuple[bool, QIcon | None]:
        """(hit, icon) from the memo of the current install."""
        self._check_generation()
        key = (const, px)
        if key not in self._memo:
            return False, None
        self._memo.move_to_end(key)
        return True, self._memo[key][0]

    def source(self, const: int, px: int) -> str | None:
        """Where a memoized thumbnail came from: "icon" (the install's
        portrait), "sprite", or None (nothing, or not resolved yet)."""
        found = self._memo.get((const, px))
        return None if found is None else found[1]

    def request(self, consts, px: int, client: object = None, *, priority: bool = False) -> None:
        """Replaces `client`'s pending list with `consts` at `px`, served in
        the given order; a `priority` client goes before every other. A memo
        hit is not re-requested; read it with cached()."""
        self._check_generation()
        self._install = asset_source.get_install_path()
        self._request_generation = asset_source.install_generation()
        wanted = list(dict.fromkeys(c for c in consts if (c, px) not in self._memo))
        self._pending.pop(id(client), None)
        if wanted:
            self._pending[id(client)] = (priority, [(const, px) for const in reversed(wanted)])
        self._pump()

    def idle(self) -> bool:
        """Nothing in flight and nothing pending."""
        return self._in_flight is None and not self._pending

    def clear(self) -> None:
        """Forgets the memo and every pending list (after an install change)."""
        self._pending = {}
        self._memo.clear()
        self._memo_sizes.clear()
        self._memo_bytes = 0

    # -- internals --------------------------------------------------------------

    def _check_generation(self) -> None:
        generation = asset_source.install_generation()
        if generation != self._memo_generation:
            self._memo_generation = generation
            self.clear()

    def _next(self) -> tuple[int, int] | None:
        if not self._pending:
            return None
        keys = [key for key, (priority, _queue) in self._pending.items() if priority] or list(self._pending)
        key = keys[-1]
        queue = self._pending[key][1]
        item = queue.pop()
        if not queue:
            del self._pending[key]
        return item

    def _pump(self) -> None:
        while self._in_flight is None:
            if self._request_generation != asset_source.install_generation():
                self._pending = {}
                return
            item = self._next()
            if item is None:
                return
            const, px = item
            if (const, px) in self._memo:
                continue
            icon, building = object_catalog.portrait_for(const)
            if icon is None or self._install is None:
                # No portrait to decode: the sprite now, on this thread.
                self._finish(const, px, sprite_thumb(const, px) if self._install is not None else None, "sprite")
                continue
            executor = self._executor or _thumb_pool()
            if executor is None:  # app exit: shutdown_thumb_pool() has run
                self._pending = {}
                return
            key = (const, px, self._request_generation)
            self._in_flight = key
            future = executor.submit(_decode_portrait, self._install, icon, building, px)
            # A partial over the relay and the token: the worker must not hold this loader.
            future.add_done_callback(functools.partial(_post, self._relay, self._token, key))

    def _on_decoded(self, key, arr, cancelled: bool = False) -> None:
        const, px, generation = key
        if self._in_flight == key:
            self._in_flight = None
        if not cancelled and generation == asset_source.install_generation():
            if arr is None:
                self._finish(const, px, sprite_thumb(const, px), "sprite")
            else:
                self._finish(const, px, arr, "icon")
        self._pump()

    def _finish(self, const: int, px: int, arr: np.ndarray | None, source: str) -> None:
        icon = _icon(arr, px)
        key = (const, px)
        if key in self._memo:
            self._memo_bytes -= self._memo_sizes[key]
        size = 0 if arr is None else int(arr.nbytes)
        self._memo[key] = (icon, None if arr is None else source)
        self._memo.move_to_end(key)
        self._memo_sizes[key] = size
        self._memo_bytes += size
        while self._memo_bytes > MEMO_BYTES_CAP and len(self._memo) > 1:
            old, _value = self._memo.popitem(last=False)
            self._memo_bytes -= self._memo_sizes.pop(old)
        self.ready.emit(const, px, icon)
