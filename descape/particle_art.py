"""Frames of a particle effect, for the units whose standing graphic is one
(GH #20: Butterflies 1608-1610; GH #67: Seagulls 303; GH #256: Smoke 1308 and
Flames 1333-1336). None of them draws from an .sld.

Leaf module like unit_sprites: numpy, Pillow and stdlib, no Qt. Pixels come
from the user's own install at runtime and nothing is shipped.

An effect is `<PARTICLE_SUBPATH>/<name>.json`: `AtlasFile` (a .dds beside a
same-named TexturePacker .json), the frame run `ImageFirst`/`ImageCount`
inside that atlas, and a `Scale`. The Scale is NOT applied here. It rides on
the graphic-map entry as `art_scale` and multiplies into the one resample
unit_sprites already does, because a second nearest-neighbour pass over a
frame of 1 px specks would drop ink.

`FlipH: true` draws the frame run mirrored, so one atlas run serves two
effects: FLAME4 1336 (`fire_large_right`) is FLAME3 1335's run flipped. The
mirror is about the pivot, so the hotspot's distance from the frame's left edge
becomes its distance from the right. Only a real bool counts, as with the
generator's check. That the game mirrors about the pivot too is an inference,
like the pivot itself below.

An atlas frame is TexturePacker's "JSON array" record:

- `frame` is where it sits in the atlas. Its w/h are the UNROTATED size, so a
  `rotated` frame occupies frame.h columns by frame.w rows. Measured on
  butterflies.dds: read that way its 271 distinct rects tile the atlas with
  no overlap, read unswapped 14 leave it and 87191 px overlap.
- A `rotated` frame is stored turned 90 degrees clockwise, so np.rot90's
  default counter-clockwise turn restores it. Measured, since the two turns
  give the same shape: a restored frame matches the unrotated frame beside it
  in the loop better than its half-turn does on 38 of butterflies_2's 40 such
  pairs (tests/test_particle_art.py re-measures it).
- `spriteSourceSize` is where the trimmed frame sits in the untrimmed
  `sourceSize` canvas, and `pivot` is a fraction of that canvas. The pivot is
  taken as the unit's ground anchor, which is an inference: nothing in the
  files says where the effect's origin sits against the tile.

Everything returns None on failure rather than raising, like unit_sprites.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from descape import asset_source, debug_log

_FAILURES = (OSError, ValueError, KeyError, TypeError, IndexError, NotImplementedError)


def clear_caches() -> None:
    """Drops everything read from the install; unit_sprites.clear_caches() calls this."""
    _effect.cache_clear()
    _atlas_frames.cache_clear()
    _atlas_pixels.cache_clear()


def _particles_dir() -> Path | None:
    root = asset_source.get_install_path()
    return None if root is None else root / asset_source.PARTICLE_SUBPATH


@lru_cache(maxsize=64)
def _effect(directory: Path, name: str) -> dict | None:
    """The effect's JSON, or None when it is absent, unreadable or not a plain
    atlas run. `ImageAngles` (a frame run per facing, the waterfalls) is not
    read, so an effect carrying it draws nothing rather than the wrong frame."""
    try:
        effect = json.loads((directory / f"{name}.json").read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(effect, dict) or not isinstance(effect.get("AtlasFile"), str):
        return None
    return None if "ImageAngles" in effect else effect


@lru_cache(maxsize=8)
def _atlas_frames(path: Path) -> tuple | None:
    """The atlas .json's frame records, in atlas index order."""
    try:
        frames = json.loads(path.read_text())["frames"]
    except _FAILURES:
        return None
    return tuple(frames) if isinstance(frames, list) else None


# One atlas at a time: butterflies.dds decodes to 16 MB of RGBA, and each cut
# frame lands in unit_sprites' native cache, so the pixels are needed once.
@lru_cache(maxsize=1)
def _atlas_pixels(path: Path):
    """The atlas .dds as a read-only (h, w, 4) uint8 RGBA array."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            pixels = np.array(image.convert("RGBA"))
    except _FAILURES:
        return None
    pixels.flags.writeable = False
    return pixels


def _cut(pixels: np.ndarray, record: dict):
    """(rgba, hotspot_x, hotspot_y) for one atlas record, upright and still
    trimmed, or None when its rect leaves the atlas."""
    rect, placed, source, pivot = (
        record["frame"], record["spriteSourceSize"], record["sourceSize"], record["pivot"]
    )
    x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
    rotated = bool(record.get("rotated"))
    cols, rows = (h, w) if rotated else (w, h)
    if min(x, y, w, h) < 0 or w == 0 or h == 0 or y + rows > pixels.shape[0] or x + cols > pixels.shape[1]:
        return None
    rgba = pixels[y : y + rows, x : x + cols]
    if rotated:
        rgba = np.rot90(rgba)
    hotspot_x = round(float(pivot["x"]) * int(source["w"])) - int(placed["x"])
    hotspot_y = round(float(pivot["y"]) * int(source["h"])) - int(placed["y"])
    return np.ascontiguousarray(rgba), hotspot_x, hotspot_y


def native_frame(effect_name: str, index: int):
    """(main, None, hotspot_x, hotspot_y) for frame `index` of the effect's
    own run, at the atlas's native resolution, or None. unit_sprites.
    _native_frame()'s contract before its crop to ink: there is no
    player-colour layer, and the effect's Scale is the caller's to apply."""
    directory = _particles_dir()
    if directory is None:
        return None
    effect = _effect(directory, effect_name)
    if effect is None:
        return None
    try:
        first, count = int(effect.get("ImageFirst", 0)), int(effect["ImageCount"])
        if not (0 <= index < count) or first < 0:
            return None
        atlas = directory.joinpath(*[part for part in re.split(r"[\\/]", effect["AtlasFile"]) if part])
        frames = _atlas_frames(atlas.with_suffix(".json"))
        if frames is None or first + index >= len(frames):
            return None
        pixels = _atlas_pixels(atlas)
        if pixels is None:
            return None
        cut = _cut(pixels, frames[first + index])
    except _FAILURES as exc:
        debug_log.log(f"particle: {effect_name} frame {index} failed to read: {exc}")
        return None
    if cut is None:
        return None
    rgba, hotspot_x, hotspot_y = cut
    if effect.get("FlipH") is True:
        rgba = np.ascontiguousarray(np.fliplr(rgba))
        hotspot_x = rgba.shape[1] - hotspot_x
    return rgba, None, hotspot_x, hotspot_y
