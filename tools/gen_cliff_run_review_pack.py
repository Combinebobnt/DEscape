#!/usr/bin/env python3
"""Review pack for the Cliff tool's chain drag (GH #44): does a run the tool
lays read as one continuous wall, does it keep one facing from end to end,
and does a run continued off an existing cliff join it without a bump.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_blocker_review_pack.py. The runs come from
`cliff_chain.ChainStroke`, the drag's own resolver and layout, and render
off-engine through the app's Stepped chunk cache with sprites on, on flat
grass. Needs a configured AoE2:DE install for the cliff art and terrain.

Six scenes: a straight drawn east (front-facing) and one drawn west
(back-facing), one L drawn south then east and the same L drawn back (west
then north), a run anchored on the end of an existing run, and a lone piece
as the frame where no join exists to judge.

Two injects. `wrong-side-leg` dresses every edge after a run's first turn for
the other side, so one leg of an L is the rock face and the other the low
ridge. `unrewritten-cap` leaves an existing end cap on its cap frame when a
run joins it, so the old end still shows inside the wall.

Run:
    tools/gen_cliff_run_review_pack.py --out-dir /tmp/somewhere
    tools/gen_cliff_run_review_pack.py --replica 1
    tools/gen_cliff_run_review_pack.py --inject wrong-side-leg
    tools/gen_cliff_run_review_pack.py --check      # writes nothing

Writes <out-dir>/<pack id>/<opaque token>/ (default build/review_pack/, gitignored).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, cliff_chain, cliff_connectivity, iso_geometry, render, render_cache, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "objects_r"
OUT_DIR = grp.OUT_DIR
PICKED = 264  # Cliff (Default) 01, the picker's default piece
PICKED_FRAME = 3
GRASS = 0
N, S, W, E = (0, -1), (0, 1), (-1, 0), (1, 0)
_CARDINALS = (N, S, W, E)


@dataclass(frozen=True)
class _Scene:
    """One frame's strokes: `before` (press, steps) lays the cliffs already on
    the map, then `press` and `steps` are the run under review."""

    press: tuple[int, int]
    steps: tuple
    before: tuple | None = None


_RUNS = {
    "run_a": _Scene((54, 60), (E, E, E, E)),
    "run_b": _Scene((66, 60), (W, W, W, W)),
    "run_c": _Scene((58, 56), (S, S, E, E)),
    "run_d": _Scene((64, 62), (W, W, N, N)),
    # A press on the south end of a three-piece run, continued in line.
    "run_e": _Scene((61, 61), (S, S), before=((61, 55), (S, S))),
    "run_f": _Scene((62, 60), ()),
}
_STRAIGHTS = ("run_a", "run_b")
_TURNS = ("run_c", "run_d")
_JOIN = "run_e"
_LONE = "run_f"
_PAD_PX = 20


@dataclass(frozen=True)
class _Cliff:
    unit_const: int
    rotation: int
    x: float
    y: float
    bounds: tuple[int, int, int, int]


def _stroke(existing, press, steps) -> cliff_chain.ChainStroke:
    stroke = cliff_chain.ChainStroke(
        map_width=120, map_height=120, existing=existing, fallback_const=PICKED, fallback_frame=PICKED_FRAME
    )
    stroke.set_path(press, list(steps))
    return stroke


def _cliffs(frame_id: str) -> list[_Cliff]:
    """Every cliff in the scene once its strokes are released: the earlier
    run, with the frames the reviewed run's joins give it, then that run."""
    scene = _RUNS[frame_id]
    first = _stroke([], *scene.before).nodes() if scene.before else []
    existing = [cliff_chain.ExistingCliff(n.unit_const, n.bounds, rotation=n.rotation) for n in first]
    stroke = _stroke(existing, scene.press, scene.steps)
    reshaped = {r.existing_index: r.rotation for r in stroke.rewrites()}
    out = [_Cliff(n.unit_const, reshaped.get(i, n.rotation), n.x, n.y, n.bounds) for i, n in enumerate(first)]
    return out + [_Cliff(n.unit_const, n.rotation, n.x, n.y, n.bounds) for n in stroke.nodes()]


def _scenario(cliffs):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GRASS
    units = [[] for _ in base.unit_manager.units]
    for i, cliff in enumerate(cliffs):
        units[0].append(SimpleNamespace(
            unit_const=cliff.unit_const, x=cliff.x, y=cliff.y, z=0.0, rotation=float(cliff.rotation),
            initial_animation_frame=cliff.rotation, reference_id=7400 + i, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


def _cache(scn):
    mm = scn.map_manager
    tile_px = render.tile_pixels_for_map(mm.map_width, mm.map_height)
    elevations, proj = render.elevations_and_proj(scn)
    return render_cache.IsoChunkCache(scn, elevations, proj, tile_px, sprites=True)


def _full(frame_id: str, box) -> np.ndarray:
    return _cache(_scenario(_cliffs(frame_id)))._composite_rect(0, *box).copy()


def _search_box(frame_id: str):
    """A generous screen rect around every run variant's tiles, for locating the ink."""
    cache = _cache(_scenario([]))
    proj = cache._levels[0].proj
    tiles = []
    for inject in (None, *INJECTS):
        with _patched(inject):
            for cliff in _cliffs(frame_id):
                x0, x1, y0, y1 = cliff.bounds
                tiles += [(x, y) for x in range(x0 - 2, x1 + 2) for y in range(y0 - 2, y1 + 2)]
    x0, y0, x1, y1 = grp._tiles_bbox_px(proj, tiles, 0)
    return max(0, x0), max(0, y0 - 6 * proj.half_h), x1, y1


def _boxes() -> dict[str, tuple[int, int, int, int]]:
    """Per frame, the crop: the union over clean and every inject of where
    the run's pixels differ from bare ground, padded. Shared by every run so
    no frame's size tells a reviewer which run it is."""
    out = {}
    for frame_id in _RUNS:
        search = _search_box(frame_id)
        bare = _cache(_scenario([]))._composite_rect(0, *search)
        union = np.zeros(bare.shape[:2], dtype=bool)
        for inject in (None, *INJECTS):
            with _patched(inject):
                union |= rp.changed_pixels(bare, _full(frame_id, search))
        x0, y0, x1, y1 = rp.feature_bbox(union, _PAD_PX, union.shape)
        out[frame_id] = (search[0] + x0, search[1] + y0, search[0] + x1, search[1] + y1)
    return out


# ----------------------------------------------------------------- injects


def _inject_wrong_side_leg():
    """Coarse: every edge after a run's first turn is typed for the other
    side, so the second leg's pieces are the wrong size for the first leg's
    facing. The corner then has one leg of each type on arms no piece has,
    and keeps the picked piece and frame. A straight has no turn to reach."""
    real = cliff_chain._path_arms

    def path_arms(steps, side):
        turn = next((i for i in range(1, len(steps)) if steps[i] != steps[0]), len(steps))
        arms = real(steps, side)
        flipped = real(steps, -side)
        for i in range(turn, len(steps)):
            arms[i][steps[i]] = flipped[i][steps[i]]
            back = (-steps[i][0], -steps[i][1])
            arms[i + 1][back] = flipped[i + 1][back]
        return arms

    return [(cliff_chain, "_path_arms", path_arms)]


def _inject_unrewritten_cap():
    """Localized: a join reshapes nothing, so an existing end cap the run
    continues stays on its end-cap frame inside the wall. One piece's frame,
    and only where a scene has a cliff on the map before the run."""
    return [(cliff_chain.ChainStroke, "rewrites", lambda _self: [])]


INJECTS = {"wrong-side-leg": _inject_wrong_side_leg, "unrewritten-cap": _inject_unrewritten_cap}
COARSE_INJECT = "wrong-side-leg"
LOCALIZED_INJECT = "unrewritten-cap"


@contextmanager
def _patched(inject: str | None):
    patches = [] if inject is None else INJECTS[inject]()
    saved = [(mod, name, getattr(mod, name)) for mod, name, _new in patches]
    for mod, name, new in patches:
        setattr(mod, name, new)
    unit_sprites.clear_caches()
    try:
        yield
    finally:
        for mod, name, old in saved:
            setattr(mod, name, old)
        unit_sprites.clear_caches()


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    caption = "Rock formations on flat grass, seen at an angle."
    frames = tuple(rp.Frame(fid, f"{fid}.png", "raw", caption=caption) for fid in _RUNS)
    wall = (
        "Each frame shows rock placed on flat grass. Small loose stones and pebbles scattered on the grass "
        "around the rock's base are part of the art, not rock formations: ignore them throughout. "
    )
    joined = (
        wall + "First, for each frame, describe the large rock in it: how many separate rock masses you see, and "
        "for each place where grass shows between two large rock masses, where it is. Then: ONE if in every "
        "frame all the large rock forms a single continuous mass, joined end to end, with no strip of grass "
        "running all the way through it; SPLIT if in any frame grass separates it into two or more masses."
    )
    walls = (*_STRAIGHTS, *_TURNS)
    checks = (
        rp.Check(id="q1", question=joined, frames=(*walls, _JOIN), answers=("ONE", "SPLIT", "CANNOT-TELL")),
        rp.Check(
            id="q2",
            question=(
                wall + "In each frame the rock forms one long wall, straight or with a single bend. A wall can be "
                "one of two kinds: a tall rock face, standing several times higher above the grass than it is "
                "deep, or a low ridge, a narrow strip standing only a little above the grass. First, for each "
                "frame, follow the wall from one free end to the other and say which kind each stretch is. Then: "
                "SAME if in every frame the whole wall is one kind from end to end; MIXED if in any frame one "
                "stretch is a tall rock face and another stretch of the same wall is a low ridge. The blocks at "
                "the two free ends and at a bend are shaped differently from the stretches between them: judge "
                "only the stretches, not those blocks."
            ),
            frames=walls,
            answers=("SAME", "MIXED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                wall + "The rock in this frame forms one straight wall. First, follow its upper outline from one "
                "free end to the other and describe it: where it runs level, and any place where it drops, "
                "rounds off or steps before the wall carries on. Then: THROUGH if the wall carries on evenly "
                "from one free end to the other, its only rounded-off ends being the two free ends; ENDED if "
                "somewhere between the two free ends the wall rounds off or tapers as a free end does and then "
                "starts again, as two walls set end to end would."
            ),
            frames=(_JOIN,),
            answers=("THROUGH", "ENDED", "CANNOT-TELL"),
        ),
        rp.Check(id="q4", question=joined, frames=(_LONE,), answers=("ONE", "SPLIT", "CANNOT-TELL")),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


def _frame_images(inject: str | None, boxes) -> dict[str, np.ndarray]:
    with _patched(inject):
        return {frame_id: _full(frame_id, boxes[frame_id]) for frame_id in _RUNS}


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_ID / grp._run_token(PACK_ID, inject, replica)
    pack_dir.mkdir(parents=True, exist_ok=True)
    images = _frame_images(inject, _boxes())
    written, rendered = [], {}
    for frame in spec.frames:
        scaled, factor = rp.frame_to_band(images[frame.id])
        path = pack_dir / frame.filename
        seam_eyeball._save_png(scaled, path)
        written.append(path)
        rendered[frame.id] = (scaled.shape[1], scaled.shape[0], factor)
    review = pack_dir / "REVIEW.md"
    review.write_text(grp._review_markdown(spec, rendered, pack_dir))
    written.append(review)
    return written


def _walls(cliffs) -> int:
    """4-connected sprite-ink components at least one ground diamond in area
    (smaller ones are each piece's loose debris), pasted on flat ground."""
    proj = iso_geometry.canvas_size_and_origin(120, 120, 64, 0, 0)
    pastes = []
    for cliff in cliffs:
        ax, ay = iso_geometry.map_point_to_screen(cliff.x, cliff.y, 0, proj)
        pastes.extend(
            (ax + p.dx - p.draw.hotspot_x, ay + p.dy - p.draw.hotspot_y, p.draw.rgba[..., 3] > 0)
            for p in unit_sprites.sprite_pieces_for(cliff.unit_const, float(cliff.rotation), 0, proj.half_w)
        )
    x0, y0 = min(p[0] for p in pastes), min(p[1] for p in pastes)
    x1, y1 = max(p[0] + p[2].shape[1] for p in pastes), max(p[1] + p[2].shape[0] for p in pastes)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for px, py, a in pastes:
        mask[py - y0 : py - y0 + a.shape[0], px - x0 : px - x0 + a.shape[1]] |= a
    labels = np.zeros(mask.shape, dtype=np.int32)
    walls, height, width = 0, *mask.shape
    for sy, sx in zip(*np.nonzero(mask), strict=True):
        if labels[sy, sx]:
            continue
        labels[sy, sx] = 1
        stack, size = [(int(sy), int(sx))], 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not labels[yy, xx]:
                    labels[yy, xx] = 1
                    stack.append((yy, xx))
        walls += size >= 2 * proj.half_w * proj.half_h
    return walls


# Decoded ink height at the art's own scale (a tile 96 px wide): a front
# straight's frames measure 192 px, a back straight's 85 to 90.
_NATIVE_HALF_W = 48
_FRONT_MIN_INK_PX = 150
_BACK_MAX_INK_PX = 110


def _arms(cliff, cliffs) -> set:
    found = {cliff_connectivity.adjacency_dir(cliff.bounds, other.bounds) for other in cliffs if other is not cliff}
    return found & set(_CARDINALS)


def _mixed_facing(cliffs) -> bool:
    """Whether the wall's straight stretches (pieces with two opposite
    neighbours) hold both a rock face and a low ridge."""
    front = back = False
    for cliff in cliffs:
        arms = _arms(cliff, cliffs)
        if arms not in ({N, S}, {W, E}):
            continue
        (piece,) = unit_sprites.sprite_pieces_for(cliff.unit_const, float(cliff.rotation), 0, _NATIVE_HALF_W)
        rows = np.nonzero((piece.draw.rgba[..., 3] > 0).any(axis=1))[0]
        height = int(rows[-1] - rows[0] + 1)
        front = front or height >= _FRONT_MIN_INK_PX
        back = back or height <= _BACK_MAX_INK_PX
    return front and back


def _cap_inside(cliffs) -> bool:
    """Whether a piece with two opposite neighbours is on an end-cap frame (16..23)."""
    return any(cliff.rotation >= 16 and _arms(cliff, cliffs) in ({N, S}, {W, E}) for cliff in cliffs)


# What each answer option rests on, per run: (walls > 1, mixed facing, an end
# cap inside the wall). Measured, then pinned: a render or resolver change that
# stops an inject producing its symptom fails --check instead of emptying a check.
_CLEAN = dict.fromkeys(_RUNS, (False, False, False))
_TOPOLOGY = {
    None: _CLEAN,
    "wrong-side-leg": {**_CLEAN, "run_c": (False, True, False), "run_d": (False, True, False)},
    "unrewritten-cap": {**_CLEAN, "run_e": (False, False, True)},
}


def _measured(frame_id: str) -> tuple[bool, bool, bool]:
    cliffs = _cliffs(frame_id)
    return _walls(cliffs) > 1, _mixed_facing(cliffs), _cap_inside(cliffs)


def _topology_failures() -> list[str]:
    out = []
    for inject, expected in _TOPOLOGY.items():
        with _patched(inject):
            for fid, want in expected.items():
                got = _measured(fid)
                print(f"topology: {inject or 'clean'} {fid} (split, mixed, cap inside) {got}")
                if got != want:
                    out.append(f"{inject or 'clean'} {fid}: (split, mixed, cap inside) {got}, expected {want}")
    return out


def check() -> None:
    """Spec coupling, frame band, determinism, and each inject's reach, printed per frame."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    boxes = _boxes()
    clean = _frame_images(None, boxes)
    again = _frame_images(None, boxes)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean captures differ")
        try:
            scaled, factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
            print(f"frame {frame.id}: {clean[frame.id].shape[1]}x{clean[frame.id].shape[0]} at {factor}x")
        except rp.PackError as exc:
            failures.append(str(exc))
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name, boxes)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        if _LONE in touched:
            failures.append(f"inject {name}: reached the lone-piece frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    # Each inject reaches its own frames and no other check's.
    for name, frames in ((COARSE_INJECT, _TURNS), (LOCALIZED_INJECT, (_JOIN,))):
        failures.extend(f"inject {name}: did not reach {fid}" for fid in frames if fid not in reach[name])
        failures.extend(f"inject {name}: reached {fid}" for fid in reach[name] if fid not in frames)
    failures.extend(_topology_failures())
    coarse = max((comp for _px, comp in reach[COARSE_INJECT].values()), default=0)
    for fid, (_px, comp) in reach[LOCALIZED_INJECT].items():
        if comp >= coarse:
            failures.append(f"inject {LOCALIZED_INJECT}: {fid} component {comp}px is not under the coarse inject's {coarse}px")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of one run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install for the cliff art and terrain textures")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
