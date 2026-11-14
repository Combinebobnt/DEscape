"""Thumbnails for the Units panel's catalog and its two unit cards (GH #68).

A portrait decode is about 54 ms (256 px DDS, times 4 on tester hardware), so
portraits decode on one worker thread; sprites are cheap (median 0.2 ms) and
share unit_sprites' unlocked LRUs with the map renderer, so they stay on the
GUI thread. ThumbLoader keeps one decode in flight and a GUI-side pending
list that each request() replaces, so rows that scrolled away are dropped
rather than drained, and a fast scroll wastes at most one decode.

Two rules carried from GOTCHAS: the install path and the portrait layers
(asset_source.portrait_layers(), GH #363) are resolved on the GUI thread at
request() and handed to the worker, never read lazily there; and a
result from before asset_source.clear_install_caches() or clear_art_caches()
(the install generation moved), or from before a data-mod switch
(game_dataset.generation() moved, TASK-478), is discarded, never shown or memoized.

View-agnostic: the catalog browser and the cards consume the same `ready`
signal. Nothing here builds a widget.

A portrait follows its owner's civ (GH #372): every key is (const, px, civ),
civ folded to None by object_catalog.portrait_civ_key() when the const has no
portrait of that civ's own, so such a const keeps one entry for every owner.
"""

from __future__ import annotations

import functools
import itertools
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap

from descape import asset_source, game_dataset, object_catalog, unit_sprites
from descape.viewer_common import pixmap_from_rgba

# Neutral, as constant_picker's preview: GAIA's team index, and art at the preview's own scale.
_TEAM_INDEX = 0
_HALF_W = 32
# The memo's byte budget: each icon's RGBA pixels (a 48 px one is 9 KB) plus a portrait's kept array.
MEMO_BYTES_CAP = 32 * 1024 * 1024

_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()
# Set by shutdown_thumb_pool(): no new pool after it, or interpreter exit would wait on its worker.
_shut_down = False


def art_generation() -> tuple[int, int]:
    """What a memo entry or decode is keyed to: the install's art and the tables a const resolves through (TASK-478)."""
    return (asset_source.install_generation(), game_dataset.generation())


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


def _decode_portrait(install, layers, icon: int, building: bool, px: int):
    """The worker's half: the portrait fitted into px, or None. Never touches a sprite."""
    return asset_source.read_unit_icon_in(install, icon, building, px, layers)


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


def tinted(arr: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    """A portrait with its player-colour region in `rgb`, opaque. DE leaves the
    region transparent (479 of the install's 716 portraits) but keeps its
    shading in the colour channels, and the game tints that shading with the
    player colour. The tint is fitted to the game's own portraits (GH #372: a
    red East Asian monk, then checked on a blue generic monk, within about 10
    of 255 where a plain multiply was 40 to 70 too dark). A region with no
    shading at all (every fully transparent pixel black, as in some mods'
    portraits) gets the flat `rgb`, as team_filled() without an array does."""
    pixels = arr.astype(np.float32) / 255
    weight = pixels[..., 3:4]
    shade = (pixels[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    colour = np.asarray(rgb, np.float32) / 255
    clear = arr[..., 3] == 0
    if clear.any() and not arr[..., :3][clear].any():
        region = np.broadcast_to(colour, pixels[..., :3].shape)
    else:
        region = colour * (0.93 * shade + 0.23) + 0.19 * shade + 0.03
    out = np.empty((*arr.shape[:2], 4), np.uint8)
    out[..., :3] = np.clip((pixels[..., :3] * weight + region * (1 - weight)) * 255 + 0.5, 0, 255)
    out[..., 3] = 255
    return out


def team_filled(
    icon: QIcon | None, source: str | None, rgb: tuple[int, int, int], px: int, art: np.ndarray | None = None
) -> QIcon | None:
    """An "icon"-source thumbnail with its player-colour region in `rgb`:
    tinted() from `art`, the loader's array for it (ThumbLoader.art()), else
    the icon over a solid `rgb` square. A portrait is square (the install's
    256 px, a data mod's often 36 px, scaled up), so a solid fill shows outside
    one only as a pixel's strip round an odd one (GH #363). A sprite, or no icon, passes through."""
    if icon is None or source != "icon":
        return icon
    if art is not None:
        return _icon(tinted(art, rgb), px)
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
    # A closed window can delete the loader's QObject while Python still holds its wrapper.
    if loader is not None and not sip.isdeleted(loader):
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
    (the consumer draws its own fallback). `ready(const, px, icon, civ)` fires
    once per resolved request, civ already folded; cached() answers a memo hit
    without one. The sprite ignores civ (GH #284).

    One loader serves the panel's list and both cards. Each `client` (the
    list, each card) has its own pending list, which its next request()
    replaces, so one card never drops another's request. A `priority` client
    (a card) is served before every other, so a card never waits behind a
    viewport of rows, even one the list asked for later (a scroll after the
    pick); among equals the most recently requested goes first."""

    ready = pyqtSignal(int, int, object, object)

    def __init__(self, parent: QObject | None = None, executor=None):
        super().__init__(parent)
        self._relay = _gui_relay()
        self._token = next(_tokens)
        _loaders[self._token] = self
        # A test passes its own; the app shares the module's one-worker pool.
        self._executor = executor
        # id(client) -> (priority, its pending (const, px, civ) served from the end); dict order is recency.
        # Keyed by id, never the client: a held card and its own loader would form a GC cycle.
        self._pending: dict[int, tuple[bool, list[tuple[int, int, int | None]]]] = {}
        self._in_flight: tuple[int, int, int | None, int] | None = None
        self._install = None
        self._layers: tuple = ()
        self._request_generation = art_generation()
        # (const, px, civ) -> (icon, source, array): source "icon", "sprite" or None; the array tinted() reads.
        self._memo: OrderedDict[tuple[int, int, int | None], tuple[QIcon | None, str | None, np.ndarray | None]] = OrderedDict()
        self._memo_sizes: dict[tuple[int, int, int | None], int] = {}
        self._memo_bytes = 0
        self._memo_generation = self._request_generation

    # -- the API ------------------------------------------------------------------

    def cached(self, const: int, px: int, civ: int | None = None) -> tuple[bool, QIcon | None]:
        """(hit, icon) from the memo of the current install, for an owner of `civ`."""
        self._check_generation()
        key = (const, px, object_catalog.portrait_civ_key(const, civ))
        if key not in self._memo:
            return False, None
        self._memo.move_to_end(key)
        return True, self._memo[key][0]

    def source(self, const: int, px: int, civ: int | None = None) -> str | None:
        """Where a memoized thumbnail came from: "icon" (a portrait, the
        install's or a data mod's), "sprite", or None (nothing, or not resolved yet)."""
        found = self._memo.get((const, px, object_catalog.portrait_civ_key(const, civ)))
        return None if found is None else found[1]

    def art(self, const: int, px: int, civ: int | None = None) -> np.ndarray | None:
        """A memoized portrait's array, its player-colour shading kept, for team_filled(); None for a sprite or a miss."""
        found = self._memo.get((const, px, object_catalog.portrait_civ_key(const, civ)))
        return found[2] if found is not None and found[1] == "icon" else None

    def request(self, consts, px: int, client: object = None, *, priority: bool = False, civ: int | None = None) -> None:
        """Replaces `client`'s pending list with `consts` at `px` for an owner
        of `civ`, served in the given order; a `priority` client goes before
        every other. A memo hit is not re-requested; read it with cached()."""
        self._check_generation()
        self._install = asset_source.get_install_path()
        self._layers = asset_source.portrait_layers()
        self._request_generation = art_generation()
        keys = ((c, px, object_catalog.portrait_civ_key(c, civ)) for c in consts)
        wanted = list(dict.fromkeys(key for key in keys if key not in self._memo))
        self._pending.pop(id(client), None)
        if wanted:
            self._pending[id(client)] = (priority, wanted[::-1])
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
        generation = art_generation()
        if generation != self._memo_generation:
            self._memo_generation = generation
            self.clear()

    def _next(self) -> tuple[int, int, int | None] | None:
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
            if self._request_generation != art_generation():
                self._pending = {}
                return
            item = self._next()
            if item is None:
                return
            if item in self._memo:
                continue
            const, px, civ = item
            icon, building = object_catalog.portrait_for(const, civ)
            if icon is None or self._install is None:
                # No portrait to decode: the sprite now, on this thread.
                self._finish(item, sprite_thumb(const, px) if self._install is not None else None, "sprite")
                continue
            executor = self._executor or _thumb_pool()
            if executor is None:  # app exit: shutdown_thumb_pool() has run
                self._pending = {}
                return
            key = (*item, self._request_generation)
            self._in_flight = key
            future = executor.submit(_decode_portrait, self._install, self._layers, icon, building, px)
            # A partial over the relay and the token: the worker must not hold this loader.
            future.add_done_callback(functools.partial(_post, self._relay, self._token, key))

    def _on_decoded(self, key, arr, cancelled: bool = False) -> None:
        const, px, civ, generation = key
        if self._in_flight == key:
            self._in_flight = None
        if not cancelled and generation == art_generation():
            if arr is None:
                self._finish((const, px, civ), sprite_thumb(const, px), "sprite")
            else:
                self._finish((const, px, civ), arr, "icon")
        self._pump()

    def _finish(self, key: tuple[int, int, int | None], arr: np.ndarray | None, source: str) -> None:
        const, px, civ = key
        icon = _icon(arr, px)
        if key in self._memo:
            self._memo_bytes -= self._memo_sizes[key]
        # Only a portrait keeps its array (art()); the cap counts it beside the icon's pixels.
        kept = arr if arr is not None and source == "icon" else None
        size = (0 if icon is None else px * px * 4) + (0 if kept is None else int(kept.nbytes))
        self._memo[key] = (icon, None if arr is None else source, kept)
        self._memo.move_to_end(key)
        self._memo_sizes[key] = size
        self._memo_bytes += size
        while self._memo_bytes > MEMO_BYTES_CAP and len(self._memo) > 1:
            old, _value = self._memo.popitem(last=False)
            self._memo_bytes -= self._memo_sizes.pop(old)
        self.ready.emit(const, px, icon, civ)
