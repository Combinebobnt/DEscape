"""The unit_state review pack: units whose Unit State is not Alive draw faded.

Registered in gen_review_pack.py's PACKS; this module holds its scenes,
frames, checks, injects and --check. Five frames, sprites on, at 64 px tiles
but rows_b at 48, where the rows' diagonal would not fit the band:

- rows_a (Stepped) and rows_b (Flat): rows 1-3 of
  tools/gen_unit_state_pair.py's states file (Almost Alive, Alive, Dead),
  the first six objects of each row (Player 1's Villager, House, Oak and
  Stone Wall, Player 2's Villager and House). The middle row is Alive.
- forest_a (Stepped) and forest_b (Flat): a forest of
  examples/8tp3w9j.aoe2scenario with a 6x6-tile block of its trees set to
  Remove in memory, the trees around it Alive.
- rows_c (Stepped): rows_a's crop of the pair's alive file, every unit
  Alive: the specificity frame, where nothing can draw faded.

Injects: `opaque` sets unit_state.GHOST_ALPHA to 1.0 (nothing fades) and
`near-invisible` to 0.05. Both reach every ghosted unit, so neither is a
localized inject; --check measures each one's reach and holds the
discriminators the see and make-out checks rest on.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np

from descape import asset_source, batch_api, render_iso, render_sprites, render_units, unit_state
from descape import iso_geometry as ig
from descape.render_cache import FlatChunkCache, IsoChunkCache
from testkit import review_pack as rp

ROOT = Path(__file__).resolve().parent.parent
FOREST_MAP = ROOT / "examples" / "8tp3w9j.aoe2scenario"
TILE_PX = 64
REMOVE = 7
# The forest's ghosted block, inclusive tile ranges, wholly inside the wood.
FOREST_BLOCK = ((11, 16), (187, 192))
FOREST_CROP = (1200, 800)
ROWS = (1, 2, 3)
ROW_OBJECTS = 6
PAD_PX = 24
SPECIFICITY = "rows_c"
# frame id -> (scene, style, tile px)
FRAMES = {
    "rows_a": ("states", "stepped", TILE_PX),
    "rows_b": ("states", "flat", 48),
    "forest_a": ("forest", "stepped", TILE_PX),
    "forest_b": ("forest", "flat", TILE_PX),
    "rows_c": ("alive", "stepped", TILE_PX),
}
INJECTS = {"opaque": 1.0, "near-invisible": 0.05}
# Summed |dRGB| a pixel must differ from the bare ground by to count as visible ink.
INK_DELTA = 30
# Of the ink an Alive unit lays on its ground, the share a faded one must keep visible, and an
# invisible one must not (measured by --check; see its printout).
CLEAN_VISIBLE_MIN = 0.6
FAINT_VISIBLE_MAX = 0.15

_SCENES: dict[str, object] = {}


def prerequisites() -> str | None:
    """Why the pack cannot render here, or None."""
    if asset_source.get_install_path() is None:
        return "no AoE2:DE install visible (set AOE2DE_INSTALL_PATH)"
    if not FOREST_MAP.is_file():
        return f"examples/{FOREST_MAP.name} is missing"
    return None


def _pair_tool():
    import gen_unit_state_pair

    return gen_unit_state_pair


def _scene(name: str) -> tuple[object, list[tuple[int, object]]]:
    """(loaded, [(player, unit)] the frame is about), built once per process."""
    if name not in _SCENES:
        if name == "forest":
            loaded = batch_api.load(FOREST_MAP)
            (x0, x1), (y0, y1) = FOREST_BLOCK
            picked = [
                (player, u)
                for player, units in enumerate(loaded.unit_manager.units)
                for u in units
                if x0 <= int(u.x) <= x1 and y0 <= int(u.y) <= y1
            ]
            for _player, unit in picked:
                unit.status = REMOVE
            loaded.unit_gen += 1
        else:
            loaded, _units, placed = _pair_tool().build(states=name == "states")
            rows = [(spot, unit) for spot, unit in placed if spot.state in ROWS]
            keep = {id(unit) for i, (_spot, unit) in enumerate(rows) if i % 12 < ROW_OBJECTS}
            picked = [(spot.owner, unit) for spot, unit in rows if id(unit) in keep]
            for units in loaded.unit_manager.units:
                units[:] = [u for u in units if id(u) in keep]
            loaded.unit_gen += 1
        _SCENES[name] = (loaded, picked)
    return _SCENES[name]


@contextmanager
def _variant(loaded, picked, how: str | None):
    """`how` None: as built. "alive": every picked unit at Alive. "absent": every ghosted picked unit removed."""
    saved_status = [(u, u.status) for _p, u in picked]
    saved_lists = [list(units) for units in loaded.unit_manager.units]
    if how == "alive":
        for _p, u in picked:
            u.status = unit_state.ALIVE
    elif how == "absent":
        gone = {id(u) for _p, u in picked if unit_state.is_ghosted(u)}
        for units in loaded.unit_manager.units:
            units[:] = [u for u in units if id(u) not in gone]
    loaded.unit_gen += 1
    try:
        yield
    finally:
        for u, status in saved_status:
            u.status = status
        for units, before in zip(loaded.unit_manager.units, saved_lists, strict=True):
            units[:] = before
        loaded.unit_gen += 1


@contextmanager
def _alpha(value: float | None):
    real = unit_state.GHOST_ALPHA
    if value is not None:
        unit_state.GHOST_ALPHA = value
    try:
        yield
    finally:
        unit_state.GHOST_ALPHA = real


def frame_tile_px(frame_id: str) -> int:
    return FRAMES[frame_id][2]


def _cache(loaded, frame_id: str):
    """A fresh sprites-on cache for the frame's style and tile size, settings-free."""
    _scene_name, style, tile_px = FRAMES[frame_id]
    if style == "flat":
        return FlatChunkCache(loaded, tile_px, sprites=True)
    elevations, _proj = render_iso.elevations_and_proj(loaded)
    mm = loaded.map_manager
    proj = ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, tile_px, ig.MIN_ELEVATION, ig.MAX_ELEVATION, elev_step_pct=ig.ELEV_STEP_DEFAULT_PCT,
    )
    return IsoChunkCache(loaded, elevations, proj, tile_px, sprites=True)


def _unit_boxes(cache, loaded, picked, style: str) -> list[tuple[int, int, int, int]]:
    """Level-0 canvas box of each picked unit's drawing: sprite ink in Stepped, its footprint in Flat."""
    mm = loaded.map_manager
    tile_px = cache.tile_px
    overrides = render_units.wall_variant_rotation_overrides(loaded)
    boxes = []
    for player, unit in picked:
        if style == "flat":
            x0, x1, y0, y1 = render_units.unit_tile_bounds(unit, mm.map_width, mm.map_height)
            boxes.append((x0 * tile_px, y0 * tile_px, x1 * tile_px, y1 * tile_px))
            continue
        index = next(i for i, u in enumerate(loaded.unit_manager.units[player]) if u is unit)
        override = overrides.get((player, index))
        for draw, px, py in render_sprites.unit_sprite_draws_at(
            loaded, cache.proj, cache.elevations, None, player, unit, override
        ):
            h, w = draw.rgba.shape[:2]
            x, y = px - draw.hotspot_x, py - draw.hotspot_y
            boxes.append((x, y, x + w, y + h))
    return boxes


def _box(frame_id: str) -> tuple[int, int, int, int]:
    """The frame's crop: the rows' drawings plus a pad, or a fixed window round the forest block."""
    key = f"box:{frame_id}"
    if key not in _SCENES:
        scene, style, _tile_px = FRAMES[frame_id]
        loaded, picked = _scene("states" if scene == "alive" else scene)
        cache = _cache(loaded, frame_id)
        if scene == "forest":
            boxes = _unit_boxes(cache, loaded, picked, style)
            cx = (min(b[0] for b in boxes) + max(b[2] for b in boxes)) // 2
            cy = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) // 2
            w, h = FOREST_CROP
            _SCENES[key] = (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)
        else:
            with _variant(loaded, picked, "alive"):
                boxes = _unit_boxes(cache, loaded, picked, style)
            _SCENES[key] = (
                min(b[0] for b in boxes) - PAD_PX, min(b[1] for b in boxes) - PAD_PX,
                max(b[2] for b in boxes) + PAD_PX, max(b[3] for b in boxes) + PAD_PX,
            )
    return _SCENES[key]


def _render(frame_id: str, how: str | None = None) -> np.ndarray:
    loaded, picked = _scene(FRAMES[frame_id][0])
    with _variant(loaded, picked, how):
        return _cache(loaded, frame_id).render_rect(*_box(frame_id), mip=0).copy()


def frame_images(inject: str | None) -> dict[str, np.ndarray]:
    """Every frame at 1:1 under `inject` (None: as shipped)."""
    with _alpha(INJECTS[inject] if inject else None):
        return {frame_id: _render(frame_id) for frame_id in FRAMES}


_SEE_CAPTION = {
    "rows_a": "Rows of objects on open grass, seen at an angle from above.",
    "rows_b": "Rows of objects on open grass, seen straight down from above.",
    "forest_a": "Part of a forest on a map, seen at an angle from above.",
    "forest_b": "Part of a forest on a map, seen straight down from above.",
    "rows_c": "Rows of objects on open grass, seen at an angle from above.",
}


def spec() -> rp.PackSpec:
    """Five frames; a see-through check on each, make-out checks on the rows, gap checks on the forest."""
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=_SEE_CAPTION[fid]) for fid in FRAMES)

    def see(fid: str) -> rp.Check:
        return rp.Check(
            id=f"see_{fid}",
            question=(
                f"Look at frame {fid}. First, in your own words, describe the objects in it and how each one is "
                "drawn.\n\n"
                "Then the closed question. Does at least one object look see-through, the ground texture showing "
                "through its body, while at least one other object looks solid (SOME)? Or does every object look "
                "equally solid (NONE)? Shadows on the ground and gaps between an object's parts, such as the grass "
                "seen between a tree's branches, do not count: only an object's own body showing the ground "
                "through it does."
            ),
            frames=(fid,),
            answers=("SOME", "NONE", "CANNOT-TELL"),
        )

    def make_out(fid: str) -> rp.Check:
        return rp.Check(
            id=f"make_out_{fid}",
            question=(
                f"Frame {fid} holds three parallel lines of objects, one beside the other. Each line holds the "
                "same six objects in the same order: a small person, a house, a tree, a stone tower, a second "
                "small person and a second house. First describe each line in your own words.\n\n"
                "Then the closed question. In every line, can you find all six objects and tell what each one "
                "is, faint ones included (ALL)? Or is at least one of them missing, or too faint to tell what it "
                "is, in some line (SOME-MISSING)?"
            ),
            frames=(fid,),
            answers=("ALL", "SOME-MISSING", "CANNOT-TELL"),
        )

    def gap(fid: str) -> rp.Check:
        return rp.Check(
            id=f"gap_{fid}",
            question=(
                f"Frame {fid} shows part of a forest. First describe the trees and the ground between them in "
                "your own words.\n\n"
                "Then the closed question. Is there a patch inside the wooded area, several trees wide, where the "
                "trees seem to be missing, so the forest floor shows with no tree standing there at all (GAP)? Or "
                "does every part of the wooded area hold trees, faint ones included (NO-GAP)? The forest's outer "
                "edge and any open ground beyond it do not count."
            ),
            frames=(fid,),
            answers=("GAP", "NO-GAP", "CANNOT-TELL"),
        )

    return rp.PackSpec(
        id="unit_state",
        title="Map objects review",
        frames=frames,
        checks=(
            *(see(fid) for fid in FRAMES),
            make_out("rows_a"), make_out("rows_b"),
            gap("forest_a"), gap("forest_b"),
        ),
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, enlarged by "
            "a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def _visible_share(img: np.ndarray, alive: np.ndarray, absent: np.ndarray) -> float:
    """Of the px an Alive drawing of the ghosted units changes from bare ground, the share `img` still changes."""
    ink = np.abs(alive.astype(int) - absent.astype(int)).sum(axis=2) > INK_DELTA
    seen = np.abs(img.astype(int) - absent.astype(int)).sum(axis=2) > INK_DELTA
    return float(np.count_nonzero(seen & ink)) / max(1, int(np.count_nonzero(ink)))


def _inside(mask: np.ndarray, frame_id: str) -> np.ndarray:
    """The px of `mask` that fall outside every ghosted unit's drawing box."""
    scene, style, _tile_px = FRAMES[frame_id]
    loaded, picked = _scene(scene)
    ghosted = [(p, u) for p, u in picked if unit_state.is_ghosted(u)]
    x0, y0, _x1, _y1 = _box(frame_id)
    allowed = np.zeros(mask.shape, dtype=bool)
    with _variant(loaded, picked, "alive"):
        boxes = _unit_boxes(_cache(loaded, frame_id), loaded, ghosted, style)
    for bx0, by0, bx1, by1 in boxes:
        allowed[max(by0 - y0, 0):max(by1 - y0, 0), max(bx0 - x0, 0):max(bx1 - x0, 0)] = True
    return mask & ~allowed


def check() -> None:
    """--check: spec coupling, no leak words, every frame in the band; only the
    ghosted units' boxes differ from an all-Alive render, and only on the four
    ghosted frames; opaque equals the all-Alive render; each inject's reach,
    none on the specificity frame; and the visible-ink shares the see and
    make-out checks rest on: clean keeps at least CLEAN_VISIBLE_MIN of an
    Alive unit's ink visible, near-invisible at most FAINT_VISIBLE_MAX."""
    if reason := prerequisites():
        raise SystemExit(f"unit_state: {reason}")
    failures: list[str] = []
    pack = spec()
    try:
        rp.validate_spec(pack)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += [f"leak: {leak}" for leak in rp.leaks(pack)]

    clean = frame_images(None)
    injected = {name: frame_images(name) for name in INJECTS}
    for frame_id, img in clean.items():
        try:
            scaled, _factor = rp.frame_to_band(img)
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame_id)
        except rp.PackError as exc:
            failures.append(str(exc))
        alive = _render(frame_id, "alive")
        changed = rp.changed_pixels(img, alive)
        print(f"ghost px: {frame_id} {int(np.count_nonzero(changed))} of {img.shape[1]}x{img.shape[0]}")
        if (frame_id == SPECIFICITY) == bool(np.any(changed)):
            failures.append(f"{frame_id}: {int(np.count_nonzero(changed))} px differ from the all-Alive render")
        stray = int(np.count_nonzero(_inside(changed, frame_id))) if frame_id != SPECIFICITY else 0
        if stray:
            failures.append(f"{frame_id}: {stray} changed px fall outside every ghosted unit's box")
        if not np.array_equal(injected["opaque"][frame_id], alive):
            failures.append(f"{frame_id}: opaque does not equal the all-Alive render")
        if frame_id != SPECIFICITY:
            absent = _render(frame_id, "absent")
            shares = {
                "clean": _visible_share(img, alive, absent),
                "near-invisible": _visible_share(injected["near-invisible"][frame_id], alive, absent),
            }
            print(f"visible ink share: {frame_id} " + ", ".join(f"{k} {v:.2f}" for k, v in shares.items()))
            if shares["clean"] < CLEAN_VISIBLE_MIN:
                failures.append(f"{frame_id}: clean keeps {shares['clean']:.2f} of the ink visible")
            if shares["near-invisible"] > FAINT_VISIBLE_MAX:
                failures.append(f"{frame_id}: near-invisible keeps {shares['near-invisible']:.2f} visible")

    for name in INJECTS:
        touched = {}
        for frame_id in FRAMES:
            mask = rp.changed_pixels(clean[frame_id], injected[name][frame_id])
            if np.any(mask):
                touched[frame_id] = (int(np.count_nonzero(mask)), rp.largest_component_px(mask))
        print(f"reach: {name} -> " + ", ".join(f"{f} {px}px/{comp}max" for f, (px, comp) in sorted(touched.items())))
        if SPECIFICITY in touched:
            failures.append(f"inject {name} reaches the specificity frame {SPECIFICITY}")
        if set(touched) != set(FRAMES) - {SPECIFICITY}:
            failures.append(f"inject {name} touches {sorted(touched)}, not every ghosted frame")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: unit_state frames couple to its checks and land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px, its text "
        f"leaks nothing, only ghosted units' boxes differ from the all-Alive render and never on {SPECIFICITY}, "
        "opaque equals that render, both injects reach every ghosted frame, and the visible-ink shares hold"
    )
