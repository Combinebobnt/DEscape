#!/usr/bin/env python3
"""Review pack for the Cliff tool's chain drag (TASK-113, GH #44): does a run
the tool lays read as one continuous wall, and does each end piece match the
wall it ends.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_blocker_review_pack.py. The runs come from
`cliff_chain.ChainStroke`, the drag's own resolver and layout, and render
off-engine through the app's Stepped chunk cache with sprites on, on flat
grass. Needs a configured AoE2:DE install for the cliff art and terrain.

Run:
    tools/gen_cliff_run_review_pack.py --out-dir /tmp/somewhere
    tools/gen_cliff_run_review_pack.py --replica 1
    tools/gen_cliff_run_review_pack.py --inject one-suffix
    tools/gen_cliff_run_review_pack.py --check      # writes nothing

Writes <out-dir>/<pack id>/<opaque token>/ (default build/review_pack/, gitignored).
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
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

# frame id -> (press tile, steps, cursor tiles); steps None means a drag over the cursor tiles.
_RUNS = {
    "run_a": ((60, 56), [S, S, E, E], None),
    "run_b": ((58, 56), [S, S, E, E, N, N], None),
    "run_c": (None, None, [(56, 56), (65, 65)]),
    "run_d": ((62, 60), [], None),
}
_PAD_PX = 20


def _nodes(frame_id: str) -> list[cliff_chain.ChainNode]:
    press, steps, cursor = _RUNS[frame_id]
    stroke = cliff_chain.ChainStroke(
        map_width=120, map_height=120, existing=[], fallback_const=PICKED, fallback_frame=PICKED_FRAME
    )
    if cursor is None:
        stroke.set_path(press, steps)
    else:
        for tile in cursor:
            stroke.add_cursor_tile(*tile)
    return stroke.nodes()


def _scenario(nodes):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GRASS
    units = [[] for _ in base.unit_manager.units]
    for i, node in enumerate(nodes):
        units[0].append(SimpleNamespace(
            unit_const=node.unit_const, x=node.x, y=node.y, z=0.0, rotation=float(node.rotation),
            initial_animation_frame=node.rotation, reference_id=7400 + i, status=2, garrisoned_in_id=-1,
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
    return _cache(_scenario(_nodes(frame_id)))._composite_rect(0, *box).copy()


def _search_box(frame_id: str):
    """A generous screen rect around every run variant's tiles, for locating the ink."""
    cache = _cache(_scenario([]))
    proj = cache._levels[0].proj
    tiles = []
    for inject in (None, *INJECTS):
        with _patched(inject):
            for node in _nodes(frame_id):
                x0, x1, y0, y1 = node.bounds
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


class _Tables(cliff_connectivity.ChainTables):
    """The measured tables with each node's candidates narrowed by `pick`."""

    def __init__(self, data, pick) -> None:
        super().__init__(data)
        self._pick = pick

    def candidates(self, dirset, allowed):
        real = super().candidates(dirset, allowed)
        return self._pick(self, dirset, real)


def _tables_patch(pick):
    data = cliff_connectivity._data()
    return [(cliff_connectivity, "chain_tables", lambda: _Tables(data, pick))]


def _inject_one_suffix():
    """Coarse: one piece size for the whole stroke, the picked 01 (3x3) at
    every node, the resolver this walk replaced. A node whose shape has no
    01 row keeps the picked piece and frame."""
    return _tables_patch(lambda _t, _d, real: [1])


def _inject_majority():
    """Each node on its own commonest suffix for its shape, ignoring its
    neighbours (the 2026-09-20 replan whose gate failed)."""

    def pick(t, dirset, real):
        if not real:
            return real
        return [max(real, key=lambda s: (t.node_n.get((dirset, s), 0), -s))]

    return _tables_patch(pick)


INJECTS = {"one-suffix": _inject_one_suffix, "per-node-majority": _inject_majority}
COARSE_INJECT = "one-suffix"


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
    frames = (
        rp.Frame("run_a", "run_a.png", "raw", caption="Rock formations on flat grass, seen at an angle."),
        rp.Frame("run_b", "run_b.png", "raw", caption="Rock formations on flat grass, seen at an angle."),
        rp.Frame("run_c", "run_c.png", "raw", caption="Rock formations on flat grass, seen at an angle."),
        rp.Frame("run_d", "run_d.png", "raw", caption="Rock formations on flat grass, seen at an angle."),
    )
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
    checks = (
        rp.Check(id="q1", question=joined, frames=("run_a", "run_b", "run_c"), answers=("ONE", "SPLIT", "CANNOT-TELL")),
        rp.Check(
            id="q2",
            question=(
                wall + "In each frame the rock forms a long wall; find its two free ends. First, for each frame "
                "and each end, describe the last block of rock there and the stretch of wall leading up to it: is "
                "the end block about as thick as that stretch, or clearly thicker (a bulky lump capping a thinner "
                "wall)? Then: EVEN if at every free end in every frame the last block is no thicker than the wall "
                "leading to it; BULGE if at any free end it is clearly thicker. Judge only the ends: a wall may "
                "change thickness at a bend, so ignore thickness changes anywhere but the last block at a free end."
            ),
            frames=("run_a", "run_b", "run_c"),
            answers=("EVEN", "BULGE", "CANNOT-TELL"),
        ),
        rp.Check(id="q3", question=joined, frames=("run_d",), answers=("ONE", "SPLIT", "CANNOT-TELL")),
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


def _walls(nodes) -> int:
    """4-connected sprite-ink components at least one ground diamond in area
    (smaller ones are each piece's loose debris), pasted on flat ground."""
    proj = iso_geometry.canvas_size_and_origin(120, 120, 64, 0, 0)
    pastes = []
    for node in nodes:
        ax, ay = iso_geometry.map_point_to_screen(node.x, node.y, 0, proj)
        pastes.extend(
            (ax + p.dx - p.draw.hotspot_x, ay + p.dy - p.draw.hotspot_y, p.draw.rgba[..., 3] > 0)
            for p in unit_sprites.sprite_pieces_for(node.unit_const, float(node.rotation), 0, proj.half_w)
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


def _end_bulges(nodes) -> bool:
    """Whether either end node is thicker (shorter footprint side) than its path neighbour."""
    def thick(node):
        x0, x1, y0, y1 = node.bounds
        return min(x1 - x0, y1 - y0)

    return len(nodes) > 1 and (thick(nodes[0]) > thick(nodes[1]) or thick(nodes[-1]) > thick(nodes[-2]))


# What each answer option rests on, per run: (walls > 1, an end bulges). run_d is the lone piece.
_TOPOLOGY = {
    None: {"run_a": (False, False), "run_b": (False, False), "run_c": (False, False), "run_d": (False, False)},
    "one-suffix": {"run_a": (True, False), "run_b": (True, False), "run_c": (True, False), "run_d": (False, False)},
    "per-node-majority": {
        "run_a": (True, True), "run_b": (True, True), "run_c": (False, True), "run_d": (False, False),
    },
}


def _topology_failures() -> list[str]:
    out = []
    for inject, expected in _TOPOLOGY.items():
        with _patched(inject):
            for fid, want in expected.items():
                nodes = _nodes(fid)
                got = (_walls(nodes) > 1, _end_bulges(nodes))
                if got != want:
                    out.append(f"{inject or 'clean'} {fid}: (split, bulge) {got}, expected {want}")
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
        if "run_d" in touched:
            failures.append(f"inject {name}: reached the lone-piece frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    for name, frames in (("one-suffix", ("run_a", "run_b", "run_c")), ("per-node-majority", ("run_a", "run_b"))):
        failures.extend(f"inject {name}: did not reach {fid}" for fid in frames if fid not in reach[name])
    failures.extend(_topology_failures())
    coarse = reach[COARSE_INJECT]
    for fid, (_px, comp) in reach["per-node-majority"].items():
        if fid in coarse and comp >= coarse[fid][1]:
            failures.append(f"inject per-node-majority: {fid} component {comp}px is not under the coarse inject's")
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
