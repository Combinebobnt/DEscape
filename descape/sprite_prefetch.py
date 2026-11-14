"""Unit sprite frames decoded on a thread pool while the first paint's level
builds (TASK-031.47).

A cold coarsest-level build reads, walks, decodes and ink-crops every .sld
frame its units draw, one at a time on the GUI thread as the sprite walk
reaches them. The walk and the decode are the nogil kernel's and the read
releases the GIL, so this module predicts the frames a load's opening level
needs (cheap: ~1,400 distinct groups on a 13,000-unit map) and computes them
on a small pool ahead of the walk, as asset_source.prefetch_terrain_textures()
does for terrain textures.

The rules that keep it pixel-identical by construction (D1-D3, D8):
- A worker computes only what _decoded_install_frame() would for the same
  key, from the same function on the same bytes, and touches no shared
  state: no cache, no debug_log, no game_dataset or mod_graphics lookup, no
  load_sld(). The path is resolved on the GUI thread at submit. Anything off
  the happy path marks the file or frame unserved and the GUI thread runs
  today's code for it, so every log line and negative-cache entry is today's.
- Only the GUI thread writes unit_sprites' caches, at the moment it does
  today (unit_sprites._sld_index() and _native_frame()). A served file's
  bytes never enter _sld_bytes_cache (D6).
- The GUI thread waits only on a running job: a queued one is cancelled and
  done synchronously, so it never waits behind other files.
- Only files with no walked index are submitted (`in`, which keeps the LRU
  order), since a hint's frame index can need the file's real frame count.

A wrong or missing hint costs a decode, never a pixel. Scope (D4): x1
install art, Stepped and Sloped, native backend, a load's opening level.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from descape import composite_backend, mod_graphics, render_units, unit_kind, unit_sprites
from descape.sld_decoder import LayerKind, SLDFile
from descape.unit_filter import UnitFilter

# The bench's interleaved on/off switch and the tests'; not a user setting (D7).
ENABLED = True
# Sized like asset_source.PREFETCH_WORKERS. Created on first use; tests swap in an inline executor.
PREFETCH_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))
_executor = None
_executor_lock = threading.Lock()

_KINDS = (LayerKind.MAIN, LayerKind.PLAYERCOLOR)
_UNKNOWN = object()


def _pool():
    global _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=PREFETCH_WORKERS, thread_name_prefix="sprite-prefetch")
        return _executor


def shutdown_pool() -> None:
    """For app exit, beside asset_source.shutdown_prefetch_pool(): cancels the
    queued jobs and stops the pool without waiting on a running one."""
    global _executor
    unit_sprites.prefetch_discard()
    with _executor_lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)


def discard() -> None:
    """Drops what is left of the current prefetch (D6): its queued jobs and
    the frames no build has taken yet."""
    unit_sprites.prefetch_discard()


def stats() -> dict[str, float]:
    return unit_sprites.prefetch_stats()


def x2_may_draw(max_fit_half_w: float | None) -> bool:
    """Whether the opening level could draw x2 art (D4), where every x1
    decode would be wasted. `max_fit_half_w` bounds the fit view's on-screen
    tile half-width; the opening level's own half_w is under twice it, since
    the level picked for a scale is at most one ladder step (2x) above it."""
    if not unit_sprites.hi_res():
        return False
    return max_fit_half_w is None or unit_sprites.sprite_scale(2 * max_fit_half_w) >= 1.0


def prefetch_level_sprites(
    scenario, unit_filter: UnitFilter = UnitFilter(), max_fit_half_w: float | None = None,
) -> int:
    """Starts decoding the x1 frames a load's opening Stepped/Sloped level
    draws, one pool job per .sld file in first-need order. Returns the number
    of files submitted. The previous prefetch is dropped first."""
    unit_sprites.prefetch_discard()
    if not ENABLED or composite_backend.native is None or x2_may_draw(max_fit_half_w):
        return 0
    directory = unit_sprites._graphics_dir()
    if directory is None:
        return 0
    t0 = time.perf_counter()
    hints, overrides_ms = _hints(scenario, unit_filter)
    t1 = time.perf_counter()
    generation = unit_sprites.prefetch_generation()
    # D8: plain `in`, never get_or_none(), which would reorder the LRU.
    indexed = {file_name for file_name in hints if file_name in unit_sprites._sld_index_cache}
    unit_sprites._prefetch_begin(indexed)
    pool = _pool()
    submitted = 0
    for file_name, file_hints in hints.items():
        if file_name in indexed:
            continue
        path = unit_sprites._sld_path(directory, file_name)
        unit_sprites._prefetch_register(file_name, generation, pool.submit(_job, path, tuple(file_hints)))
        submitted += 1
    unit_sprites._prefetch_count("hint_ms", (t1 - t0) * 1000 - overrides_ms)
    unit_sprites._prefetch_count("overrides_ms", overrides_ms)
    unit_sprites._prefetch_count("submit_ms", (time.perf_counter() - t1) * 1000)
    return submitted


def _hints(scenario, unit_filter: UnitFilter) -> tuple[dict[str, list[tuple]], float]:
    """{file_name: [dispatch args]} in the walk's first-need order, plus the
    ms of the wall overrides' computation (moved cost: memoized per unit
    generation, so the walk's own call reuses it). Mirrors
    render_sprites._resolve_unit_sprite() -> unit_sprites.sprite_pieces_for() and
    garrison_flag_pieces_for(): the same filter, bounds, rotation, art,
    pieces and seeds, at the iso zero point."""
    t0 = time.perf_counter()
    overrides = render_units.wall_variant_rotation_overrides(scenario)
    overrides_ms = (time.perf_counter() - t0) * 1000
    flag_hosts = render_units.garrison_host_ids(scenario)
    mm = scenario.map_manager
    tile_w, tile_h = mm.map_width, mm.map_height
    skipped_stems = unit_sprites._particle_stems()
    stem_ok: dict[str, bool] = {}
    by_file: dict[str, list[tuple]] = {}
    offset = unit_sprites.ANGLE_ZERO_OFFSET_DEG

    def add(unit_id: int, entry: dict, rotation: float, seed, piece_index: int) -> None:
        file_name = str(entry["file_name"])
        ok = stem_ok.get(file_name)
        if ok is None:
            # Particle stems have no .sld; a stem a cosmetic mod ships draws the mod's file.
            ok = stem_ok[file_name] = file_name not in skipped_stems and mod_graphics.resolve_sprite(file_name) is None
        if ok:
            hint = (entry, rotation, offset, seed, piece_index, unit_sprites.rotation_is_variant(unit_id))
            by_file.setdefault(file_name, []).append(hint)

    for player_id, units in enumerate(scenario.unit_manager.units):
        art = render_units.player_art_for(scenario, player_id)
        # const -> (entry, whether a piece is seeded), or None for a const that draws no .sld.
        entries: dict[int, tuple[dict, bool] | None] = {}
        # Per player, stored_rotation() is a function of (const, raw rotation), so a group is resolved once.
        seen: set[tuple] = set()
        for i, unit in enumerate(units):
            if not unit_filter.matches(player_id, unit):
                continue
            const = unit.unit_const
            known = entries.get(const, _UNKNOWN)
            if known is _UNKNOWN:
                entry = None if unit_kind.invisible_category(const) is not None else unit_sprites._entry_for_art(const, art)
                known = entries[const] = None if entry is None else (
                    entry, any(piece.get("seeded") for piece in entry.get("pieces") or ())
                )
            if known is None:
                continue
            entry, seeded = known
            override = overrides.get((player_id, i)) if overrides else None
            reference_id = getattr(unit, "reference_id", None)
            host = reference_id in flag_hosts
            # A seeded piece's frame follows the unit's reference_id; nothing else's does.
            group = (const, unit.rotation, override, host, reference_id if seeded else None)
            if group in seen:
                continue
            # The walk skips an off-map unit; an on-map one later in the group still counts.
            if render_units.unit_tile_bounds(unit, tile_w, tile_h) is None:
                continue
            seen.add(group)
            rotation = override if override is not None else render_units.stored_rotation(player_id, unit)
            pieces = entry.get("pieces")
            if not pieces:
                add(const, entry, rotation, None, 0)
            else:
                for index, piece in enumerate(pieces):
                    add(piece["unit_id"], piece, rotation, reference_id, index)
            if host:
                for flag in entry.get("garrison_flags", ()):
                    add(flag["unit_id"], flag, 0.0, None, 0)
    return by_file, overrides_ms


def _job(path: Path, hints: tuple[tuple, ...]):
    """One file on a worker (D2): read, native-only walk, the hinted frames'
    native-only decode and ink crop. (walked SLDFile holding no bytes,
    {frame_index: value or None (no ink) or unit_sprites._UNSERVED}), or None
    when the file is unserved. Exactly unit_sprites._decoded_install_frame()'s
    computation; touches no shared state."""
    try:
        data = path.read_bytes()
        sld = SLDFile(data, keep_data=False, native_only=True)
    except Exception:  # noqa: BLE001 -- unserved: the GUI thread's load_sld() logs it
        return None
    count = sld.frame_count
    frames: dict[int, object] = {}
    for entry, rotation, offset, seed, piece_index, is_variant in hints:
        try:
            index = unit_sprites._dispatch(entry, rotation, offset, seed, piece_index, is_variant, lambda: count)
        except Exception:  # noqa: BLE001, S112 -- a worker never logs (D2); a missed hint is only a decode
            continue
        if index in frames or not 0 <= index < count:
            continue
        try:
            frame = sld.decode_frame(index, data, kinds=_KINDS, window_to_main=True, native_only=True)
            frames[index] = None if frame.main is None else unit_sprites._cropped_to_ink(
                frame.main, frame.playercolor, frame.hotspot_x, frame.hotspot_y
            )
        except Exception:  # noqa: BLE001 -- unserved: the GUI thread decodes it and logs as today
            frames[index] = unit_sprites._UNSERVED
    return sld, frames
