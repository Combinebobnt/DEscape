"""The chunk caches' shared constants and job types: layer fields, warm, stamp and reduce constants, box_reduce2, the jobs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from descape import (
    composite_backend,
    debug_log,
)

# LayerState fields that are baked into a SpriteLayer's own contents when it
# is BUILT, rather than read off self.layers at paint time. Flipping one needs
# the unit sources rebuilding, not just the chunks evicting -- see
# _ChunkCacheBase.set_layers(). A new build-time row belongs here; a paint-time
# one deliberately does not.
# Translucent Walls is compared through its derived float, so an opacity change
# with Translucent Walls off rebuilds nothing.
# Mod Art's art comes from mod_graphics' module state; the viewer's toggle replaces the whole
# cache (refresh_map), so this entry only guards a future set_layers() caller.
_BUILD_TIME_LAYER_FIELDS = ("farm_overlay", "small_trees", "hero_glow", "wall_alpha", "mod_art")
# LayerState fields no chunk's pixels depend on: set_layers() stores them and evicts nothing.
_DRAW_ONLY_LAYER_FIELDS = ("colour_mood",)


# Unit-pack tiles derived per pack_warm_job() step. Measured 2026-09-27 at
# 0.9-1.5us per occupied tile (Dos Pilas the densest), so a slice is ~1-1.5ms dev.
PACK_WARM_TILES = 1024


# Byte bound of the reduced store (render_reduced_rect()), apart from max_bytes: a k=1 chunk is 768 KiB.
REDUCED_MAX_BYTES = 64 << 20
# _chunk_stamp entries for uncached keys are dropped once per this many stamps.
STAMP_PRUNE_EVERY = 1024
# A reduced entry's stamp for a parent chunk that was not cached when it was built: never valid.
_NEVER_STAMP = -1

# Relative slack _ChunkCacheBase.mip_for_scale() allows: a view landed on a
# level's own ratio by a float multiply can sit an ulp or two above it.
# MapView's level stop tests use the same one, so "at the stop" picks it.
MIP_SCALE_SLACK = 1e-12


def _box_reduce2_numpy(src: np.ndarray) -> np.ndarray:
    """box_reduce2()'s numpy path, and the native kernel's byte-identity oracle."""
    h, w = src.shape[:2]
    if h == 0 or w == 0:
        return np.zeros(((h + 1) // 2, (w + 1) // 2, *src.shape[2:]), dtype=np.uint8)
    if h % 2:
        src = np.concatenate([src, src[-1:]], axis=0)
    if w % 2:
        src = np.concatenate([src, src[:, -1:]], axis=1)
    acc = src[0::2, 0::2].astype(np.uint16)
    acc += src[1::2, 0::2]
    acc += src[0::2, 1::2]
    acc += src[1::2, 1::2]
    acc += 2
    return (acc >> 2).astype(np.uint8)


def box_reduce2(src: np.ndarray) -> np.ndarray:
    """A 2x2 box reduce of an (h, w, c) uint8 array to (ceil(h/2), ceil(w/2), c):
    (a + b + c + d + 2) >> 2 per channel, an odd last row or column replicated.
    Output pixel i covers input pixels 2i and 2i+1 (TASK-031.70). The native
    kernel when active (read at call time), else numpy, byte-identical."""
    kernel = composite_backend.native
    if kernel is None:
        return _box_reduce2_numpy(src)
    h, w = src.shape[:2]
    out = np.empty(((h + 1) // 2, (w + 1) // 2, *src.shape[2:]), dtype=np.uint8)
    if h and w:
        kernel.box_reduce2(np.ascontiguousarray(src), out)
    return out


@dataclass
class _Reduced:
    """One reduced chunk (render_reduced_rect()): its pixels, the stamp of every
    REAL coarsest-level chunk under it when it was built, and that level's
    canvas_dims() then."""

    pixels: np.ndarray
    parents: tuple[tuple[tuple[int, int, int], int], ...]
    dims: tuple[int, int]


@dataclass
class LevelWarmJob:
    """One mip level's warm, handed to level_warm.LevelWarmer: a resumable
    walk plus the closure that installs its result on THIS cache. The walk
    builds the level's sprite layer (level_warm_job) or derives its unit
    pack's pending tiles in place (pack_warm_job, whose install is a no-op).

    The split is what keeps Qt out of this module. The driver owns pacing
    (a QTimer and a wall-clock budget) and knows nothing about what a level
    is; the cache owns the walk and the install semantics -- including the
    validity predicate, which differs per style and is the whole correctness
    argument for installing a layer built across several event-loop turns.

    install() returns True if it landed, False if the job was revalidated
    away (the source mutated under the warm, or a real paint already built
    the level). A False is a normal outcome, not an error."""

    gen: object
    install: object
    # Perf Trace's label for this job's steps in a level-warm tick.
    kind: str = "walk"


@dataclass
class ChunkJob:
    """One chunk's composite with every Python step already done on the GUI
    thread, so run() is a single nogil kernel call safe on a worker thread
    (Batch F T2, margin_warm). The cache installs the result only through
    install_chunk(), which rejects it if anything mutated since `epoch`."""

    key: tuple[int, int, int]
    epoch: int
    kernel: object
    scratch: np.ndarray
    args: tuple
    # Owners of memory args reach only by address (the unit pack's sprites).
    holds: tuple = ()

    def run(self) -> bool:
        """Paints scratch. False: the kernel declined and painted nothing."""
        return self.kernel(self.scratch, *self.args) is not False


@dataclass
class RectJob(ChunkJob):
    """A ChunkJob over a sub-rect of a CACHED chunk, for a pooled patch
    (TASK-031.67). scratch is the slice's pixels; install_slice() writes them
    into `chunk`, the array that was cached at prepare."""

    chunk: np.ndarray | None = None
    # The slice, LEVEL pixels (ix0, iy0, ix1, iy1).
    slice: tuple[int, int, int, int] = (0, 0, 0, 0)


def join_chunk_jobs(submitted: list) -> list[bool]:
    """Waits for every (job, future) in `submitted`, in submit order, and
    returns whether each painted (TASK-031.64). The calling thread is one
    more consumer: it cancels the newest job still unstarted and runs it
    inline, then looks again, so the workers drain from the head while it
    takes from the tail. A pool is FIFO, so once a future will not cancel
    every older one has started and the rest are waited on head to tail.
    A raise, inline or on a worker, is logged and counts as declined; a None
    future (the submit itself failed) is declined too."""
    painted = [False] * len(submitted)
    tail = len(submitted) - 1
    while tail >= 0:
        job, future = submitted[tail]
        if future is not None and not future.cancel():
            break
        if future is not None:
            try:
                painted[tail] = job.run()
            except Exception as exc:  # noqa: BLE001
                debug_log.log(f"pooled paint: chunk {job.key} raised inline ({exc!r})", level=debug_log.ERROR)
        tail -= 1
    for i in range(tail + 1):
        job, future = submitted[i]
        if future is None:
            continue
        try:
            painted[i] = future.result() is True
        except Exception as exc:  # noqa: BLE001
            debug_log.log(f"pooled paint: chunk {job.key} raised on a worker ({exc!r})", level=debug_log.ERROR)
    return painted
