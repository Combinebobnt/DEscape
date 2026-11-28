"""The Spray Can's scatter kernel: an airbrush on a tile grid (GH #295).

A stroke is a run of dabs, one per timer tick while the button is held. Each
dab adds `flow * weight` to the coverage of every footprint tile, the weight
falling from 1 in the brush's hard core to WEIGHT_FLOOR at its rim, and a
tile paints once its coverage exceeds its own hashed threshold. Holding still
therefore fills the footprint in, and a fast drag leaves it sparse. The
threshold is a pure function of (seed, x, y), so a stroke is reproducible
from its seed and its dab sequence alone.

Qt-free and imports only `brush`, so a later unit spray can reuse it: call
`dab()` and place one unit per returned tile. That is not the Unit Brush's
model, which rolls each tile once per stroke at a fixed density.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import cache

from descape import brush

SPRAY_TICK_MS = 40
# The rim's weight: every footprint tile fills eventually, however soft the brush.
WEIGHT_FLOOR = 0.1

_MASK = (1 << 64) - 1


def _mix(z: int) -> int:
    """splitmix64's finaliser, masked to 64 bits so it is identical on every platform."""
    z = (z + 0x9E3779B97F4A7C15) & _MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def tile_threshold(seed: int, x: int, y: int) -> float:
    """The coverage tile (x, y) must exceed to paint, in [0, 1). The seed is
    mixed first, so seeds s and s + 1 give unrelated thresholds."""
    h = _mix(_mix(seed & _MASK) ^ (((x & 0xFFFFFFFF) << 32) | (y & 0xFFFFFFFF)))
    return (h >> 11) / float(1 << 53)


@cache
def weights(size: int, shape: str, thickness: int, hardness: float) -> dict[tuple[int, int], float]:
    """Each brush offset's weight: 1 out to `hardness` of the way from the box
    centre to the shape's farthest tile, then linear down to WEIGHT_FLOOR there."""
    offsets = brush.brush_offsets(size, shape, thickness)
    lo, hi = brush._box(brush.clamp_brush_size(size))
    c = (lo + hi) / 2
    radii = {(dx, dy): math.hypot(dx - c, dy - c) for dx, dy in offsets}
    r_min = 0.0 if (hi - lo) % 2 == 0 else math.sqrt(0.5)
    r_far = max(radii.values())
    if hardness >= 1 or r_far <= r_min:
        return dict.fromkeys(offsets, 1.0)
    out = {}
    for offset, r in radii.items():
        r_n = (r - r_min) / (r_far - r_min)
        w = 1.0 if r_n <= hardness else 1.0 - (1.0 - WEIGHT_FLOOR) * (r_n - hardness) / (1.0 - hardness)
        out[offset] = min(1.0, max(WEIGHT_FLOOR, w))
    return out


@dataclass
class SprayStroke:
    seed: int
    flow: float  # 0.01-1, coverage a full-weight tile gains per dab
    hardness: float  # 0-1, the fraction of the brush radius at full weight
    coverage: dict[tuple[int, int], float] = field(default_factory=dict)


def dab(
    stroke: SprayStroke, tiles, cx: int, cy: int, size: int, shape: str, thickness: int
) -> list[tuple[int, int]]:
    """One dab at cursor tile (cx, cy) over `tiles` (the clipped footprint,
    tiles already painted left out): adds each tile's share to its coverage
    and returns, in input order, the tiles that now exceed their threshold."""
    table = weights(size, shape, thickness, stroke.hardness)
    coverage = stroke.coverage
    seed = stroke.seed
    painted = []
    for tile in tiles:
        x, y = tile
        value = coverage.get(tile, 0.0) + stroke.flow * table[(x - cx, y - cy)]
        coverage[tile] = value
        if value > tile_threshold(seed, x, y):
            painted.append(tile)
    return painted


def new_seed() -> int:
    """A fresh stroke seed, from the OS: a test seeding `random` cannot make two strokes alike."""
    return random.SystemRandom().getrandbits(64)
