"""descape/catalog_thumbs.py (GH #68, TASK-075 S3): the Units catalog's
thumbnail loader. Portraits decode on one worker, sprites stay on the GUI
thread, one decode is in flight, a new request drops what the old one left,
and a result from an older install is never shown."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from descape import asset_source, catalog_thumbs, object_catalog, unit_sprites

import conftest
from test_unit_sprites import build_sld

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

_ARCHER = 4  # icon 17: the fake install's red unit portrait
_CASTLE = 82  # icon 7, a building: the fake install's green one
_MILITIA = 74  # icon 8, no file in the fake install
_SKIRMISHER = 75  # icon 10, no file either
_TREE = 349


_ManualExecutor = conftest.ManualExecutor

def _loader(executor=None):
    conftest.ensure_qapp()
    loader = catalog_thumbs.ThumbLoader(executor=executor)
    got = []
    loader.ready.connect(lambda const, px, icon: got.append((const, px, icon)))
    return loader, got


def _pump(until=lambda: True, timeout=5.0) -> None:
    from PyQt5.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    QApplication.processEvents()
    while not until() and time.monotonic() < deadline:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()


def _no_icon_const() -> int:
    return next(entry.id for entry in object_catalog.objects() if object_catalog.icon_for(entry.id) is None)


def test_the_worker_never_runs_the_sprite_path(fake_install, monkeypatch) -> None:
    main = threading.current_thread()
    sprite_threads, decode_threads = [], []
    sprite_thumb, decode = catalog_thumbs.sprite_thumb, catalog_thumbs._decode_portrait

    def recording_sprite(*args):
        sprite_threads.append(threading.current_thread())
        return sprite_thumb(*args)

    def recording_decode(*args):
        decode_threads.append(threading.current_thread())
        return decode(*args)

    monkeypatch.setattr(catalog_thumbs, "sprite_thumb", recording_sprite)
    monkeypatch.setattr(catalog_thumbs, "_decode_portrait", recording_decode)
    loader, got = _loader()
    consts = [_ARCHER, _MILITIA, _no_icon_const()]
    loader.request(consts, 32)
    _pump(lambda: len(got) == 3)
    assert sorted(const for const, _px, _icon in got) == sorted(consts)
    assert decode_threads and all(thread is not main for thread in decode_threads)
    # The militia's missing portrait and the icon-less const both fell back to the sprite path.
    assert len(sprite_threads) == 2 and all(thread is main for thread in sprite_threads)
    icons = {const: icon for const, _px, icon in got}
    assert icons[_ARCHER] is not None and icons[_ARCHER].pixmap(32).width() == 32


def test_a_new_request_drops_what_the_old_one_left(fake_install) -> None:
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    loader.request([_ARCHER, _MILITIA, _SKIRMISHER], 32)
    assert executor.icons() == [17], "one decode in flight"
    loader.request([_CASTLE, _MILITIA], 32)
    assert executor.icons() == [17]
    executor.run(0)
    _pump(lambda: len(executor.submitted) == 2)
    assert executor.icons() == [17, 7]
    executor.run(1)
    _pump(lambda: len(executor.submitted) == 3)
    assert executor.icons() == [17, 7, 8], "the skirmisher was superseded, never decoded"
    assert [const for const, _px, _icon in got] == [_ARCHER, _CASTLE]


def test_one_client_never_drops_anothers_request(fake_install) -> None:
    """The list and the two cards share a loader: a card's pick is served
    first and does not drop the rows the list still wants."""
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    list_client, card_client = object(), object()
    loader.request([_ARCHER, _MILITIA, _SKIRMISHER], 32, client=list_client)
    loader.request([_CASTLE], 64, client=card_client)
    for index in range(4):
        _pump(lambda index=index: len(executor.submitted) > index)
        executor.run(index)
    _pump(lambda: len(got) == 4)
    assert executor.icons() == [17, 7, 8, 10], "the card's castle jumps the list's queue, nothing is lost"
    assert loader.idle()


def test_the_loader_takes_a_buildings_portrait_from_the_buildings_set(fake_install) -> None:
    """Icon 28 is the Trebuchet's in units/ and the Town Center's in buildings/
    (TASK-191); the worker path must keep the building flag."""
    from PIL import Image

    ingame = asset_source.get_install_path() / "widgetui" / "textures" / "ingame"
    Image.new("RGBA", (32, 32), (1, 2, 3, 255)).save(ingame / "units" / "028_50730.DDS", format="DDS")
    Image.new("RGBA", (32, 32), (4, 5, 6, 255)).save(ingame / "buildings" / "028_town_center.DDS", format="DDS")
    loader, got = _loader()
    for const in (109, 42):  # Town Center, Trebuchet
        loader.request([const], 32, client=const)
    _pump(lambda: len(got) == 2)
    centres = {const: icon.pixmap(32, 32).toImage().pixelColor(16, 16).getRgb() for const, _px, icon in got}
    assert centres == {109: (4, 5, 6, 255), 42: (1, 2, 3, 255)}


def test_a_priority_client_goes_before_a_later_ordinary_one(fake_install) -> None:
    executor = _ManualExecutor()
    loader, _got = _loader(executor)
    list_client, card_client = object(), object()
    loader.request([_MILITIA], 32, client=list_client)  # in flight
    loader.request([_CASTLE], 64, client=card_client, priority=True)
    loader.request([_ARCHER, _SKIRMISHER], 32, client=list_client)  # a scroll after the pick
    for index in range(3):
        _pump(lambda index=index: len(executor.submitted) > index)
        executor.run(index)
    assert executor.icons() == [8, 7, 17], "the card's castle before the list's later rows"


def test_a_result_from_an_older_install_is_dropped(fake_install) -> None:
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    loader.request([_ARCHER, _MILITIA], 32)
    asset_source.clear_install_caches()
    executor.run(0)
    _pump()
    assert got == []
    assert loader.cached(_ARCHER, 32) == (False, None)
    assert len(executor.submitted) == 1, "nothing from the old request is decoded after the change"


def test_the_install_is_captured_at_request_and_never_read_by_the_worker(fake_install, monkeypatch) -> None:
    executor = _ManualExecutor()
    loader, _got = _loader(executor)
    install = asset_source.get_install_path()
    loader.request([_ARCHER], 32)
    assert executor.submitted[0][1][0] == install

    def no_lookup():
        raise AssertionError("the worker read the install itself")

    no_lookup.cache_clear = asset_source.get_install_path.cache_clear
    monkeypatch.setattr(asset_source, "get_install_path", no_lookup)
    executor.run(0)
    assert executor.submitted[0][2].result() is not None


def test_a_memo_hit_is_not_requested_again(fake_install) -> None:
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    loader.request([_ARCHER], 32)
    executor.run(0)
    _pump(lambda: bool(got))
    hit, icon = loader.cached(_ARCHER, 32)
    assert hit and icon is not None
    loader.request([_ARCHER], 32)
    assert len(executor.submitted) == 1


def test_with_no_install_everything_resolves_to_none_at_once() -> None:
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    loader.request([_ARCHER, _TREE], 32)
    assert executor.submitted == []
    assert got == [(_ARCHER, 32, None), (_TREE, 32, None)]


def test_a_dropped_loader_is_never_destroyed_on_the_worker(fake_install, monkeypatch) -> None:
    """The worker must not hold the last reference to a loader: a QObject
    deleted off its own thread corrupted Qt state, and an unrelated later test
    (test_themes' apply_theme) segfaulted in the default tier."""
    import gc
    import weakref

    started, gate = threading.Event(), threading.Event()

    def slow_decode(*_args):
        started.set()
        gate.wait(5)

    monkeypatch.setattr(catalog_thumbs, "_decode_portrait", slow_decode)
    conftest.ensure_qapp()
    loader = catalog_thumbs.ThumbLoader()
    finalized = []
    weakref.finalize(loader, lambda: finalized.append(threading.current_thread()))
    loader.request([_ARCHER], 32)
    assert started.wait(5)
    del loader
    gc.collect()
    gate.set()
    catalog_thumbs._thumb_pool().submit(lambda: None).result(timeout=5)  # the one worker has run the callback
    _pump()
    assert finalized == [threading.main_thread()]


def test_a_cancelled_decode_does_not_leave_its_loader_waiting(fake_install) -> None:
    executor = _ManualExecutor()
    loader, got = _loader(executor)
    loader.request([_ARCHER, _CASTLE], 32)
    executor.submitted[0][2].cancel()
    _pump(lambda: len(executor.submitted) == 2)
    assert executor.icons() == [17, 7], "the next row is decoded"
    assert got == []


def test_shutdown_cancels_the_queue(monkeypatch) -> None:
    # Restored after the test, so later tests on this worker still get a pool.
    monkeypatch.setattr(catalog_thumbs, "_shut_down", False)
    pool = catalog_thumbs._thumb_pool()
    started, gate = threading.Event(), threading.Event()

    def hold():
        started.set()
        return gate.wait(5)

    running = pool.submit(hold)
    assert started.wait(5)
    queued = pool.submit(lambda: 1)
    catalog_thumbs.shutdown_thumb_pool()
    assert queued.cancelled()
    gate.set()
    running.result(timeout=5)
    assert catalog_thumbs._pool is None


def test_a_request_after_shutdown_starts_no_new_worker(fake_install, monkeypatch) -> None:
    """A fresh pool after shutdown would be a non-daemon worker that interpreter exit waits on."""
    # This worker's real pool is set aside and restored, never shut down here.
    monkeypatch.setattr(catalog_thumbs, "_pool", None)
    monkeypatch.setattr(catalog_thumbs, "_shut_down", False)
    catalog_thumbs.shutdown_thumb_pool()
    loader, got = _loader()
    loader.request([_ARCHER, _CASTLE], 32)
    _pump()
    assert catalog_thumbs._pool is None, "no new pool after shutdown"
    assert catalog_thumbs._thumb_pool() is None
    assert loader.idle() and got == []


# -- the sprite thumbnail ---------------------------------------------------------


@pytest.fixture
def facing_install(tmp_path, monkeypatch):
    """The archer as a 16-frame sprite with a different colour per frame."""
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    name = "t_facing_archer_x1"
    (graphics / f"{name}.sld").write_bytes(build_sld(16, canvas=16, playercolor=False))
    entry = {"graphic_id": 1, "file_name": name, "angle_count": 16, "mirroring_mode": 0, "frame_count": 1}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: {_ARCHER: entry})
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


def test_the_sprite_thumb_faces_the_viewer_and_fits(facing_install, monkeypatch) -> None:
    conftest.ensure_qapp()
    facing = catalog_thumbs.sprite_thumb(_ARCHER, 12)
    assert facing is not None and max(facing.shape[:2]) <= 12 and facing.shape[2] == 4
    monkeypatch.setattr(unit_sprites, "rotation_facing_viewer", lambda const: None)
    unturned = catalog_thumbs.sprite_thumb(_ARCHER, 12)
    assert not np.array_equal(facing, unturned), "the facing must pick another frame"


def test_a_narrow_thumb_is_centred_on_a_square_icon() -> None:
    """Every row's icon is the same width, so the names line up (measured in the eyeball pass)."""
    conftest.ensure_qapp()
    narrow = np.full((32, 10, 4), 255, dtype=np.uint8)
    image = catalog_thumbs._icon(narrow, 32).pixmap(32, 32).toImage()
    assert (image.width(), image.height()) == (32, 32)
    opaque = [x for x in range(32) if image.pixelColor(x, 16).alpha() > 0]
    assert opaque[0] > 5 and 31 - opaque[-1] > 5, "centred, not left-aligned"


def test_no_sprite_gives_no_thumb(facing_install) -> None:
    assert catalog_thumbs.sprite_thumb(_TREE, 32) is None
