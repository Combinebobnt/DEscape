"""The first paint's sprite prefetch (TASK-031.47, descape/sprite_prefetch.py).

The prefetch only moves work: a level walked after it must draw the same
pieces, leave unit_sprites' caches in the same key order, open each file
once, and leave every failure to today's synchronous path and its log line.
A synthetic install covers every dispatch branch the walk can take: angle,
variant, wall (overrides and Translucent Walls), composite, seeded, garrison
flag and frame_offset consts, plus a player the unit filter hides.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from descape import (
    asset_source,
    composite_backend,
    debug_log,
    game_dataset,
    mod_graphics,
    perf_trace,
    render,
    sld_decoder,
    sprite_prefetch,
    unit_sprites,
)
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.unit_filter import UnitFilter

from test_unit_sprites import build_sld

ANGLE, VARIANT, FISH, COMPOSITE, COMPOSITE_FRONT = 900_001, 900_002, 900_003, 900_004, 900_005
PASTURE, PASTURE_FENCE, HOST, FLAG, HIDDEN = 900_006, 900_007, 900_008, 900_009, 900_010
WALL = 117  # a real wall-family const, so the 8-neighbour overrides and Translucent Walls reach it
FILES = {
    "t_angle_x1": 4, "t_variant_x1": 5, "t_fish_x1": 2, "t_tc_main_x1": 4, "t_tc_front_x1": 4,
    "t_post_x1": 1, "t_fence_x1": 4, "t_host_x1": 1, "t_flag_x1": 1, "t_wall_x1": 5, "t_hidden_x1": 1,
}
HIDDEN_PLAYER = 3
SHOWN = UnitFilter(players=frozenset({1, 2}))


def _entry(file_name: str, angle_count: int, **extra) -> dict:
    return {"graphic_id": 1, "file_name": file_name, "angle_count": angle_count, "mirroring_mode": 6,
            "frame_count": 1, **extra}


def _piece(unit_id: int, file_name: str, angle_count: int, dx: int, **extra) -> dict:
    return {"unit_id": unit_id, "file_name": file_name, "angle_count": angle_count, "frame_count": 1,
            "dx": dx, "dy": 0, **extra}


GRAPHIC_MAP = {
    ANGLE: _entry("t_angle_x1", 4),
    VARIANT: _entry("t_variant_x1", 5, rotation_is_variant=True),
    # GH #178's fish: frame 0 is empty, so the body is frame_offset 1, drawn at a body offset.
    FISH: _entry("t_fish_x1", 1, frame_offset=1, dx=0, dy=-2),
    COMPOSITE: _entry("t_tc_main_x1", 4, pieces=[
        _piece(COMPOSITE, "t_tc_main_x1", 4, 0, parent=True), _piece(COMPOSITE_FRONT, "t_tc_front_x1", 4, 3),
    ]),
    PASTURE: _entry("t_post_x1", 1, pieces=[
        _piece(PASTURE, "t_post_x1", 1, 0, parent=True), _piece(PASTURE_FENCE, "t_fence_x1", 4, 2, seeded=True),
    ]),
    HOST: _entry("t_host_x1", 1, garrison_flags=[_piece(FLAG, "t_flag_x1", 1, 0)]),
    WALL: _entry("t_wall_x1", 5, rotation_is_variant=True),
    HIDDEN: _entry("t_hidden_x1", 1),
}


@dataclass
class Unit:
    x: float
    y: float
    unit_const: int
    rotation: float = 0.0
    reference_id: int = 0
    garrisoned_in_id: int = -1


@pytest.fixture
def install(tmp_path, monkeypatch):
    graphics = tmp_path / unit_sprites.GRAPHICS_SUBPATH
    graphics.mkdir(parents=True)
    for base, (name, frames) in enumerate(FILES.items()):
        (graphics / f"{name}.sld").write_bytes(build_sld(frames, canvas=16, colour_base=base * 3))
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: GRAPHIC_MAP)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield graphics
    sprite_prefetch.discard()
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


@pytest.fixture
def native():
    if not composite_backend.available():
        pytest.skip(composite_backend.unavailable_reason)
    with composite_backend.use_backend("native"):
        yield


class _InlineExecutor:
    """Runs a job at submit() and hands back its finished Future."""

    def submit(self, fn, *args):
        future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future


class _HeldExecutor:
    """Queues every job and never runs one: each stays cancellable."""

    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append(future)
        return future


class _SlowExecutor:
    """Marks each job running at submit and finishes it on a timer thread, so
    a consumer finds it running and has to wait."""

    def __init__(self, delay: float = 0.03):
        self.delay = delay
        self.timers = []

    def submit(self, fn, *args):
        future = Future()
        future.set_running_or_notify_cancel()
        timer = threading.Timer(self.delay, lambda: future.set_result(fn(*args)))
        timer.start()
        self.timers.append(timer)
        return future


@pytest.fixture(params=["inline", "threads"])
def pool(request, monkeypatch, native):
    executor = _InlineExecutor() if request.param == "inline" else ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(sprite_prefetch, "_executor", executor)
    yield executor
    if request.param == "threads":
        executor.shutdown(wait=True)


def _scenario():
    scenario = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    scenario.player_art = None
    units = scenario.unit_manager.units
    units[1] += [
        Unit(10.5, 10.5, ANGLE, 0.0, 1), Unit(12.5, 10.5, ANGLE, 1.6, 2), Unit(14.5, 10.5, VARIANT, 3.0, 3),
        Unit(16.5, 10.5, FISH, 0.0, 4), Unit(18.5, 10.5, COMPOSITE, 3.2, 5),
        Unit(20.5, 10.5, PASTURE, 0.0, 6), Unit(22.5, 10.5, PASTURE, 0.0, 7),
        Unit(24.5, 10.5, HOST, 0.0, 8), Unit(24.5, 10.5, ANGLE, 0.0, 9, garrisoned_in_id=8),
    ]
    # A run of walls the overrides reshape from their neighbours, plus a lone one (the tower).
    units[2] += [Unit(30.5 + k, 20.5, WALL, 0.0, 20 + k) for k in range(3)] + [Unit(40.5, 30.5, WALL, 1.0, 30)]
    units[HIDDEN_PLAYER] += [Unit(50.5, 50.5, HIDDEN, 0.0, 40)]
    return scenario


def _walk(scenario, unit_filter=SHOWN, wall_alpha=1.0):
    elevations, proj = render.elevations_and_proj(scenario)
    return render.sprite_draws_by_anchor(scenario, proj, elevations, unit_filter, wall_alpha=wall_alpha)


def _normalized(layer: render.SpriteLayer):
    return {
        key: [(d.rgba.shape, d.rgba.tobytes(), d.hotspot_x, d.hotspot_y, x, y, rise) for d, x, y, rise in draws]
        for key, draws in layer.by_anchor.items()
    }, layer.bboxes


def _state():
    return list(unit_sprites._native_cache), list(unit_sprites._sld_index_cache)


def _build(scenario, enabled: bool, unit_filter=SHOWN, wall_alpha=1.0, before=None):
    """A cold walk with the prefetch on or off; (layer, cache key orders, files submitted)."""
    unit_sprites.clear_caches()
    if before is not None:
        before()
    sprite_prefetch.ENABLED = enabled
    try:
        submitted = sprite_prefetch.prefetch_level_sprites(scenario, unit_filter)
    finally:
        sprite_prefetch.ENABLED = True
    if isinstance(sprite_prefetch._executor, ThreadPoolExecutor):
        # Real threads, finished first, so served counts are exact: a job still queued when the
        # walk reaches it is cancelled (D3), which the held and slow executors pin.
        futures_wait([future for _gen, future in unit_sprites._prefetch_pending.values()])
    layer = _walk(scenario, unit_filter, wall_alpha)
    return _normalized(layer), _state(), submitted


@pytest.fixture
def opened(monkeypatch):
    """Every .sld read, on any thread: the GUI's load_sld() and fresh reads, and the workers'."""
    reads, lock = [], threading.Lock()
    real = Path.read_bytes

    def counted(self):
        if self.suffix == ".sld":
            with lock:
                reads.append(self.stem)
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", counted)
    return reads


@pytest.mark.parametrize("wall_alpha", [1.0, 0.5])
def test_a_prefetched_walk_draws_and_caches_exactly_as_the_serial_one(install, pool, wall_alpha):
    scenario = _scenario()
    want_layer, want_state, off = _build(scenario, False, wall_alpha=wall_alpha)
    got_layer, got_state, on = _build(scenario, True, wall_alpha=wall_alpha)
    assert off == 0 and on == len(FILES) - 1
    assert got_layer == want_layer
    assert got_state == want_state
    stats = sprite_prefetch.stats()
    assert stats["files_served"] == on and stats["frames_served"] > 0


def test_each_file_is_opened_once_and_every_frame_the_walk_needs_was_hinted(install, pool, opened):
    _build(_scenario(), True)
    stats = sprite_prefetch.stats()
    assert stats.get("unhinted", 0) == 0
    assert stats.get("frames_unserved", 0) == 0
    assert sorted(opened) == sorted(name for name in FILES if name != "t_hidden_x1")
    # Non-vacuity: every branch's frame came from the pool.
    assert stats["frames_served"] == stats["frames_hinted"] >= len(FILES) - 1
    assert "t_hidden_x1" not in unit_sprites._sld_index_cache


def test_the_filtered_out_player_is_not_hinted(install, native):
    hints, _ = sprite_prefetch._hints(_scenario(), SHOWN)
    assert "t_hidden_x1" not in hints
    everything, _ = sprite_prefetch._hints(_scenario(), UnitFilter())
    assert "t_hidden_x1" in everything


def test_an_indexed_file_is_not_submitted_and_keeps_its_lru_place(install, native, monkeypatch):
    executor = _HeldExecutor()
    monkeypatch.setattr(sprite_prefetch, "_executor", executor)
    # Indexed in the reverse of the hints' first-need order, so any reordering lookup shows.
    unit_sprites._sld_index("t_variant_x1")
    unit_sprites._sld_index("t_angle_x1")
    submitted = sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN)
    assert submitted == len(FILES) - 3
    assert "t_angle_x1" not in unit_sprites._prefetch_pending
    assert list(unit_sprites._sld_index_cache) == ["t_variant_x1", "t_angle_x1"]


def test_an_indexed_files_frames_are_todays_decodes_not_hint_misses(install, pool):
    """In the app, cliff_catalog walks the cliff files before any load, so D8
    leaves them to today's path: their decodes are counted apart from
    `unhinted`, which then means hint drift alone."""
    scenario = _scenario()
    want_layer, want_state, _ = _build(scenario, False, before=lambda: unit_sprites._sld_index("t_wall_x1"))
    got_layer, got_state, submitted = _build(scenario, True, before=lambda: unit_sprites._sld_index("t_wall_x1"))
    assert got_layer == want_layer and got_state == want_state
    stats = sprite_prefetch.stats()
    assert submitted == len(FILES) - 2
    assert stats["frames_indexed"] > 0 and stats.get("unhinted", 0) == 0


def test_a_queued_job_is_cancelled_and_its_file_walked_synchronously(install, native, monkeypatch):
    scenario = _scenario()
    want_layer, want_state, _ = _build(scenario, False)
    executor = _HeldExecutor()
    monkeypatch.setattr(sprite_prefetch, "_executor", executor)
    got_layer, got_state, submitted = _build(scenario, True)
    assert got_layer == want_layer and got_state == want_state
    stats = sprite_prefetch.stats()
    assert stats["files_cancelled"] == submitted
    assert stats["frames_unserved"] > 0 and stats.get("unhinted", 0) == 0
    assert all(future.cancelled() for future in executor.jobs)


def test_a_running_job_is_waited_on_and_served(install, native, monkeypatch):
    scenario = _scenario()
    want_layer, want_state, _ = _build(scenario, False)
    executor = _SlowExecutor()
    monkeypatch.setattr(sprite_prefetch, "_executor", executor)
    got_layer, got_state, submitted = _build(scenario, True)
    assert got_layer == want_layer and got_state == want_state
    stats = sprite_prefetch.stats()
    assert stats["files_served"] == submitted and stats["wait_ms"] > 0
    for timer in executor.timers:
        timer.join()


def _log_lines() -> list[str]:
    return [line.split("] ", 1)[-1] for line in debug_log.get_log_text().splitlines() if "sld" in line or "sprite" in line]


def test_an_unreadable_file_is_walked_synchronously_with_todays_log_line(install, pool):
    (install / "t_fence_x1.sld").write_bytes(b"not an sld at all")
    scenario = _scenario()
    debug_log.clear()
    want_layer, want_state, _ = _build(scenario, False)
    want_log = _log_lines()
    debug_log.clear()
    got_layer, got_state, _ = _build(scenario, True)
    assert any("t_fence_x1.sld not readable" in line for line in want_log)
    assert _log_lines() == want_log
    assert got_layer == want_layer and got_state == want_state
    assert sprite_prefetch.stats()["files_unserved"] == 1


def test_a_failing_job_is_walked_synchronously(install, native, monkeypatch):
    scenario = _scenario()
    want_layer, want_state, _ = _build(scenario, False)

    def boom(path, hints):
        raise RuntimeError("worker bug")

    monkeypatch.setattr(sprite_prefetch, "_job", boom)
    monkeypatch.setattr(sprite_prefetch, "_executor", _InlineExecutor())
    got_layer, got_state, submitted = _build(scenario, True)
    assert got_layer == want_layer and got_state == want_state
    assert sprite_prefetch.stats()["files_unserved"] == submitted


def _declining(real, refuse_walk_of: bytes):
    """The real kernel, refusing to walk one file and to decode one layer kind."""

    def sld_walk(data, *args):
        return -1 if bytes(data[:64]) == refuse_walk_of[:64] else real.sld_walk(data, *args)

    def sld_decode_layer(data, layers, chain, layer_id, bc1, canvas):
        return False if not bc1 else real.sld_decode_layer(data, layers, chain, layer_id, bc1, canvas)

    names = {name: getattr(real, name) for name in dir(real) if not name.startswith("__")}
    return SimpleNamespace(**{**names, "sld_walk": sld_walk, "sld_decode_layer": sld_decode_layer})


def test_a_declining_kernel_is_left_to_the_synchronous_path_and_its_disagreement_log(install, native, monkeypatch):
    """The worker never runs the Python fallback (D2): a walk or a layer the
    kernel declines is unserved, and the GUI thread's own walk and decode
    count and log the disagreement exactly as with the prefetch off."""
    scenario = _scenario()
    refused = (install / "t_variant_x1.sld").read_bytes()
    monkeypatch.setattr(composite_backend, "native", _declining(composite_backend.native, refused))
    monkeypatch.setattr(sprite_prefetch, "_executor", ThreadPoolExecutor(max_workers=2))
    threads = []
    real_disagreed = sld_decoder._native_disagreed
    monkeypatch.setattr(
        sld_decoder, "_native_disagreed",
        lambda what: (threads.append(threading.current_thread()), real_disagreed(what))[1],
    )

    debug_log.clear()
    before = sld_decoder.native_disagreements
    want_layer, want_state, _ = _build(scenario, False)
    want = sld_decoder.native_disagreements - before, _log_lines()
    debug_log.clear()
    before = sld_decoder.native_disagreements
    got_layer, got_state, _ = _build(scenario, True)
    got = sld_decoder.native_disagreements - before, _log_lines()
    sprite_prefetch._executor.shutdown(wait=True)

    assert want[0] > 1, "the fixture declined nothing, so this proves nothing"
    assert set(threads) == {threading.main_thread()}, "a worker ran the Python fallback"
    assert got == want
    assert got_layer == want_layer and got_state == want_state
    stats = sprite_prefetch.stats()
    assert stats["files_unserved"] == 1 and stats["frames_unserved"] > 0


@pytest.mark.parametrize("change", ["clear_caches", "mod_art", "data_mod"])
def test_a_change_between_submit_and_consume_discards_the_prefetch(install, native, monkeypatch, change):
    scenario = _scenario()
    want_layer, want_state, _ = _build(scenario, False)

    def bump():
        if change == "clear_caches":
            unit_sprites.clear_caches()
        elif change == "mod_art":
            monkeypatch.setattr(mod_graphics, "_generation", mod_graphics.generation() + 1)
        else:
            monkeypatch.setattr(game_dataset, "_generation", game_dataset.generation() + 1)

    monkeypatch.setattr(sprite_prefetch, "_executor", _InlineExecutor())
    unit_sprites.clear_caches()
    submitted = sprite_prefetch.prefetch_level_sprites(scenario, SHOWN)
    bump()
    got_layer = _normalized(_walk(scenario))
    assert submitted
    assert got_layer == want_layer and _state() == want_state
    stats = sprite_prefetch.stats()
    assert stats.get("files_served", 0) == 0 and stats.get("frames_served", 0) == 0
    assert not unit_sprites._prefetch_stash


def test_clear_caches_forgets_pending_jobs(install, native, monkeypatch):
    executor = _HeldExecutor()
    monkeypatch.setattr(sprite_prefetch, "_executor", executor)
    assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN)
    unit_sprites.clear_caches()
    assert not unit_sprites._prefetch_pending and not unit_sprites._prefetch_active
    assert all(future.cancelled() for future in executor.jobs)


def test_the_numpy_backend_and_the_switch_off_submit_nothing(install, monkeypatch):
    monkeypatch.setattr(sprite_prefetch, "_executor", _InlineExecutor())
    with composite_backend.use_backend("numpy"):
        assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN) == 0
    monkeypatch.setattr(sprite_prefetch, "ENABLED", False)
    assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN) == 0
    assert not unit_sprites._prefetch_pending


def test_mod_shipped_and_particle_stems_are_not_hinted(install, native, monkeypatch):
    graphic_map = {**GRAPHIC_MAP, ANGLE: {**GRAPHIC_MAP[ANGLE], "particle": True}}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: graphic_map)
    unit_sprites._particle_stems.cache_clear()
    real = mod_graphics.resolve_sprite
    monkeypatch.setattr(
        mod_graphics, "resolve_sprite",
        lambda stem: (Path("mod/t_variant_x1.sld"), "sld") if stem == "t_variant_x1" else real(stem),
    )
    hints, _ = sprite_prefetch._hints(_scenario(), SHOWN)
    assert "t_angle_x1" not in hints and "t_variant_x1" not in hints
    assert "t_fish_x1" in hints


def test_an_opening_level_that_may_draw_x2_submits_nothing(install, native, monkeypatch):
    monkeypatch.setattr(sprite_prefetch, "_executor", _HeldExecutor())
    unit_sprites.set_hi_res(True)
    try:
        assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN) == 0
        # The fit's half_w bound puts the level under 1:1: x1 draws, so it prefetches.
        assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN, max_fit_half_w=11.0) > 0
        assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN, max_fit_half_w=24.0) == 0
    finally:
        unit_sprites.set_hi_res(False)
    assert sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN, max_fit_half_w=500.0) > 0


def test_the_dispatch_the_worker_runs_is_the_gui_threads(install, native):
    """Every hint's frame, dispatched with the file's own count as a worker
    does, equals _frame_for() of its piece's own const on the GUI thread."""
    unit_ids = {}
    for const, entry in GRAPHIC_MAP.items():
        unit_ids[id(entry)] = const
        for part in (*entry.get("pieces", ()), *entry.get("garrison_flags", ())):
            unit_ids[id(part)] = part["unit_id"]
    hints, _ = sprite_prefetch._hints(_scenario(), UnitFilter())
    checked = 0
    for file_name, file_hints in hints.items():
        count = unit_sprites.sld_frame_count(file_name)
        for entry, rotation, offset, seed, piece_index, is_variant in file_hints:
            got = unit_sprites._dispatch(entry, rotation, offset, seed, piece_index, is_variant, lambda n=count: n)
            want = unit_sprites._frame_for(unit_ids[id(entry)], entry, rotation, offset, seed, piece_index)
            assert got == want, file_name
            checked += 1
    assert checked >= len(FILES)


# -- Perf Trace --------------------------------------------------------------


@pytest.fixture
def tracing():
    perf_trace.reset()
    perf_trace.enable(True)
    yield
    perf_trace.enable(False)
    perf_trace.reset()


def _prefetch_field() -> str | None:
    return next((e for e in perf_trace._take_extras() if e.startswith("sprite prefetch")), None)


def test_perf_trace_names_the_prefetch_on_the_next_line(install, pool, tracing):
    _build(_scenario(), True)
    stats = sprite_prefetch.stats()
    field = _prefetch_field()
    assert field == (
        f"sprite prefetch +{len(FILES) - 1} files (wait {stats.get('wait_ms', 0.0):.1f} ms, "
        f"served {int(stats['frames_served'])} frames, unhinted 0)"
    )
    assert _prefetch_field() is None, "taken once"


def test_perf_trace_puts_the_file_count_on_the_line_its_frames_land_on(install, pool, tracing):
    """The viewer submits right after the parse, before any op opens, and a
    `perf view` line can be written before the first paint: the count waits
    for the first take, so it shares the paint's line with the waits."""
    scenario = _scenario()
    unit_sprites.clear_caches()
    submitted = sprite_prefetch.prefetch_level_sprites(scenario, SHOWN)
    assert _prefetch_field() is None
    _walk(scenario)
    assert _prefetch_field().startswith(f"sprite prefetch +{submitted} files")


def test_perf_trace_counts_a_discarded_prefetchs_files(install, native, monkeypatch, tracing):
    monkeypatch.setattr(sprite_prefetch, "_executor", _HeldExecutor())
    submitted = sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN)
    sprite_prefetch.discard()
    assert _prefetch_field() == f"sprite prefetch +{submitted} files (wait 0.0 ms, served 0 frames, unhinted 0)"


def test_perf_trace_has_no_prefetch_field_without_one(install, native, tracing):
    _build(_scenario(), False)
    assert _prefetch_field() is None


def test_perf_trace_records_nothing_while_off_and_reset_clears_it(install, pool):
    perf_trace.reset()
    _build(_scenario(), True)
    assert not perf_trace._sprite_prefetch_raw
    perf_trace.enable(True)
    try:
        _build(_scenario(), True)
        assert perf_trace._sprite_prefetch_raw
        perf_trace.reset()
        assert not perf_trace._sprite_prefetch_raw and perf_trace._sprite_prefetch_pending.text() is None
    finally:
        perf_trace.enable(False)
        perf_trace.reset()


def test_the_stash_holds_only_install_frames_cropped_to_ink(install, native, monkeypatch):
    """A worker's value is _decoded_install_frame()'s, compared directly."""
    monkeypatch.setattr(sprite_prefetch, "_executor", _InlineExecutor())
    sprite_prefetch.prefetch_level_sprites(_scenario(), SHOWN)
    sld = unit_sprites._sld_index("t_variant_x1")
    stashed = {k: v for k, v in unit_sprites._prefetch_stash.items() if k[0] == "t_variant_x1"}
    assert stashed
    directory = unit_sprites._graphics_dir()
    for (_name, index), value in stashed.items():
        want = unit_sprites._decoded_install_frame(sld, directory, "t_variant_x1", index)
        assert value is not None and want is not None
        assert np.array_equal(value[0], want[0]) and np.array_equal(value[1], want[1]) and value[2:] == want[2:]
