#!/usr/bin/env python3
"""Review pack for GH #151: top-down Flat (Elevation View Flat, Isometric View
off) draws walls and gates along the square grid, through
unit_sprites.flat_art().

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_ship_sails_review_pack.py. Qt-free: Flat's own
oracle render, render.render_terrain() plus render.overlay_units(
with_sprites=True), on one synthetic scenario, cropped per frame and enlarged
by an integer nearest-neighbour factor. Needs a configured AoE2:DE install.

The PACK_ID and every frame/check name are neutral on purpose: frame paths
appear in REVIEW.md, and nothing a reviewer reads may name the subject.

Frames, one feature each: a stone wall run along x with a post at each end
(a), one along y (b), one down each diagonal (c, d), a stone gate set into an
x-run (e), a y-run (f) and each diagonal run (g, h), and the specificity frame
(i): lone posts and a house, which no inject can reach.

Injects:
    unturned        coarse: flat_art() is the identity, i.e. the render before
                    GH #151, for every wall and gate.
    one-piece       localized: one interior piece of frame a's run unturned.
    gates-unturned  the gate check's own: gates unturned, walls still turned.

Run:
    tools/gen_flat_structures_review_pack.py --out-dir <neutral dir>
    tools/gen_flat_structures_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_flat_structures_review_pack.py --out-dir <neutral dir> --inject one-piece
    tools/gen_flat_structures_review_pack.py --check      # writes nothing

Writes <out-dir>/top_view_objects/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import statistics
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
import verify_flat_facing as vf

from descape import asset_source, gate_orientation, render, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

PACK_ID = "top_view_objects"
OUT_DIR = grp.OUT_DIR
GRASS = 0
PLAYER = 1
WALL = 117  # Stone Wall
HOUSE = 70
GATE_A, GATE_C, GATE_B, GATE_D = gate_orientation.groups()[("2", "A")]  # stone gate, closed
DIAGONAL_TOLERANCE_DEG = 20.0

# frame id -> (crop tiles x0, y0, x1, y1 half-open, walls, gates as (const, low corner tile)).
_X_RUN = [(x, 20) for x in range(20, 28)]
_Y_RUN = [(40, y) for y in range(20, 28)]
_FALL_RUN = [(20 + i, 40 + i) for i in range(7)]
_RISE_RUN = [(40 + i, 46 - i) for i in range(7)]
_FRAMES = {
    "a": ((19, 18, 29, 23), _X_RUN, []),
    "b": ((38, 19, 43, 29), _Y_RUN, []),
    "c": ((19, 39, 28, 48), _FALL_RUN, []),
    "d": ((39, 39, 48, 48), _RISE_RUN, []),
    "e": ((59, 58, 69, 63), [(60, 60), (61, 60), (66, 60), (67, 60)], [(GATE_A, (62, 60))]),
    "f": ((78, 59, 83, 69), [(80, 60), (80, 61), (80, 66), (80, 67)], [(GATE_B, (80, 62))]),
    "g": ((77, 17, 87, 27), [(78, 18), (79, 19), (84, 24), (85, 25)], [(GATE_C, (80, 20))]),
    "h": ((77, 37, 87, 47), [(78, 45), (79, 44), (84, 39), (85, 38)], [(GATE_D, (80, 40))]),
    "i": ((59, 78, 69, 83), [(60, 80), (62, 80), (64, 80)], [(HOUSE, (66, 79))]),
}
RUN_FRAMES = ("a", "b", "c", "d")
GATE_FRAMES = ("e", "f", "g", "h")
CONTROL_FRAME = "i"
# Each run frame's Flat screen axis (degrees clockwise from right, y down) and tolerance.
_RUN_AXES = {"a": (0.0, vf.TOLERANCE_DEG), "b": (90.0, vf.TOLERANCE_DEG),
             "c": (45.0, DIAGONAL_TOLERANCE_DEG), "d": (135.0, DIAGONAL_TOLERANCE_DEG)}
_GATE_AXES = {GATE_A: (0.0, vf.TOLERANCE_DEG), GATE_B: (90.0, vf.TOLERANCE_DEG),
              GATE_C: (45.0, DIAGONAL_TOLERANCE_DEG), GATE_D: (135.0, DIAGONAL_TOLERANCE_DEG)}
# The one-piece inject's target: an interior piece of frame a's run.
ONE_PIECE_TILE = (23, 20)


def _ref(tile: tuple[int, int]) -> int:
    return 10_000 + tile[1] * 1000 + tile[0]


def _units() -> list[tuple[int, int, int, float]]:
    """(const, tile x, tile y, rotation) for every unit, walls carrying the
    stored shape a Place Unit wall run writes (neighbours8, a lone piece 2)."""
    span = render.tile_span
    connectors: set[tuple[int, int]] = set()
    gates = []
    for _box, walls, placed in _FRAMES.values():
        connectors.update(walls)
        for const, (tx, ty) in placed:
            sx, sy = span(const, (1, 1))
            x, y = render.span_anchor(tx, ty, sx, sy)
            gates.append((const, x, y))
            if gate_orientation.is_gate(const):
                unit = SimpleNamespace(unit_const=const, x=x, y=y)
                connectors.update(render.unit_occupied_tiles(unit, 120, 120))
    out = []
    for _box, walls, _placed in _FRAMES.values():
        for tx, ty in walls:
            mask = unit_sprites.neighbour_mask(tx, ty, connectors, diagonals=True)
            shape = unit_sprites.wall_variant_from_neighbours8(mask)
            out.append((WALL, tx + 0.5, ty + 0.5, float(2 if shape is None else shape)))
    return out + [(const, x, y, 0.0) for const, x, y in gates]


def _scenario():
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GRASS
    placed = [[] for _ in base.unit_manager.units]
    for const, x, y, rotation in _units():
        placed[PLAYER].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=rotation,
            reference_id=_ref((int(x), int(y))), status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=placed),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


# ----------------------------------------------------------------- injects


def _identity(unit_const, rotation):
    return unit_const, rotation


def _gates_identity(real):
    def flat_art(unit_const, rotation):
        if gate_orientation.is_gate(unit_const):
            return unit_const, rotation
        return real(unit_const, rotation)
    return flat_art


def _one_piece_icon_for(real_icon_for, real_flat_art):
    """icon_for() with flat_art() the identity for one reference_id only."""
    target = _ref(ONE_PIECE_TILE)

    def icon_for(*args, **kwargs):
        seed = kwargs.get("seed", args[5] if len(args) > 5 else None)
        if seed != target:
            return real_icon_for(*args, **kwargs)
        unit_sprites.flat_art = _identity
        try:
            return real_icon_for(*args, **kwargs)
        finally:
            unit_sprites.flat_art = real_flat_art
    return icon_for


INJECTS = ("unturned", "one-piece", "gates-unturned")
COARSE_INJECT = "unturned"
# Which frames each inject may change. The control frame is in none.
INJECT_REACH = {
    "unturned": frozenset(RUN_FRAMES) | frozenset(GATE_FRAMES),
    "one-piece": frozenset({"a"}),
    "gates-unturned": frozenset(GATE_FRAMES),
}
# one-piece's bound: one 64 px tile's worth of ink at most.
ONE_PIECE_BOUND = rp.LocalizedBound(max_changed_fraction=0.02, max_component_px=64 * 64)


@contextmanager
def _patched(inject: str | None):
    real_flat_art = unit_sprites.flat_art
    real_icon_for = unit_sprites.icon_for
    unit_sprites.clear_caches()
    if inject == "unturned":
        unit_sprites.flat_art = _identity
    elif inject == "gates-unturned":
        unit_sprites.flat_art = _gates_identity(real_flat_art)
    elif inject == "one-piece":
        unit_sprites.icon_for = _one_piece_icon_for(real_icon_for, real_flat_art)
    try:
        yield
    finally:
        unit_sprites.flat_art = real_flat_art
        unit_sprites.icon_for = real_icon_for
        unit_sprites.clear_caches()


# --------------------------------------------------------------- rendering


def _tile_px(scn) -> int:
    return render.tile_pixels_for_map(scn.map_manager.map_width, scn.map_manager.map_height)


def _crop(img: np.ndarray, box, tile_px: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(img[y0 * tile_px : y1 * tile_px, x0 * tile_px : x1 * tile_px, :3])


def _full_images(inject: str | None, scn=None, terrain=None) -> tuple[np.ndarray, np.ndarray]:
    """(terrain only, terrain plus Flat icons) for the whole map."""
    scn = scn or _scenario()
    terrain = render.render_terrain(scn) if terrain is None else terrain
    with _patched(inject):
        full = render.overlay_units(terrain, scn, with_sprites=True)
    return terrain, full


def _frame_images(inject: str | None, scn=None, terrain=None) -> dict[str, np.ndarray]:
    scn = scn or _scenario()
    _terrain, full = _full_images(inject, scn, terrain)
    tile_px = _tile_px(scn)
    return {fid: _crop(full, box, tile_px) for fid, (box, _w, _g) in _FRAMES.items()}


# --------------------------------------------------------------- measurement


def _footprint_axis(full: np.ndarray, terrain: np.ndarray, const: int, x: float, y: float, tile_px: int):
    """The principal axis (and elongation) of what one unit painted over the terrain."""
    unit = SimpleNamespace(unit_const=const, x=x, y=y)
    tx0, tx1, ty0, ty1 = render.unit_tile_bounds(unit, 120, 120)
    sl = np.s_[ty0 * tile_px : ty1 * tile_px, tx0 * tile_px : tx1 * tile_px]
    mask = rp.changed_pixels(full[sl], terrain[sl]).astype(np.uint8) * 255
    if not mask.any():
        return None
    return vf.principal_axis_deg(mask), vf.axis_elongation(mask)


def _axes(full: np.ndarray, terrain: np.ndarray, tile_px: int) -> dict[str, list[tuple[str, float, float]]]:
    """Per frame: (label, axis, elongation) for each interior run piece and each gate."""
    stored = {(int(x), int(y)): (c, x, y, r) for c, x, y, r in _units()}
    out: dict[str, list[tuple[str, float, float]]] = {}
    for fid, (_box, walls, gates) in _FRAMES.items():
        rows = []
        for tile in walls:
            const, x, y, _r = stored[tile]
            mask = unit_sprites.neighbour_mask(*tile, set(walls), diagonals=True)
            if fid in RUN_FRAMES and bin(mask).count("1") != 2:
                continue  # an end post
            got = _footprint_axis(full, terrain, const, x, y, tile_px)
            if got is not None:
                rows.append((f"wall {tile}", *got))
        for const, (tx, ty) in gates:
            sx, sy = render.tile_span(const, (1, 1))
            x, y = render.span_anchor(tx, ty, sx, sy)
            got = _footprint_axis(full, terrain, const, x, y, tile_px)
            if got is not None:
                rows.append((f"const {const}", *got))
        out[fid] = rows
    return out


def _run_axis_failures(axes, *, turned: bool) -> list[str]:
    """turned: every run frame's median interior piece axis is within
    tolerance of its Flat axis; not turned: every one is outside it."""
    failures = []
    for fid in RUN_FRAMES:
        want, tol = _RUN_AXES[fid]
        median = statistics.median(a for label, a, _e in axes[fid] if label.startswith("wall"))
        gap = vf.axis_gap_deg(median, want)
        if (gap <= tol) != turned:
            failures.append(f"frame {fid}: median piece axis {median:.1f} vs {want} (off {gap:.1f}, turned={turned})")
    return failures


def _gate_axis_failures(axes, *, turned: bool) -> list[str]:
    """The same for every gate in the gate frames."""
    failures = []
    for fid in GATE_FRAMES:
        for label, axis, _e in axes[fid]:
            if not label.startswith("const"):
                continue
            want, tol = _GATE_AXES[int(label.split()[1])]
            gap = vf.axis_gap_deg(axis, want)
            if (gap <= tol) != turned:
                failures.append(f"frame {fid} {label}: axis {axis:.1f} vs {want} (off {gap:.1f}, turned={turned})")
    return failures


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = (
        rp.Frame("a", "a.png", "raw", caption="Grass seen from directly above, with one line of joined wall pieces."),
        rp.Frame("b", "b.png", "raw", caption="Grass seen from directly above, with one line of joined wall pieces."),
        rp.Frame("c", "c.png", "raw", caption="Grass seen from directly above, with one line of joined wall pieces."),
        rp.Frame("d", "d.png", "raw", caption="Grass seen from directly above, with one line of joined wall pieces."),
        *(
            rp.Frame(fid, f"{fid}.png", "raw",
                     caption="Grass seen from directly above, with one wall line and a gatehouse in it.")
            for fid in GATE_FRAMES
        ),
        rp.Frame(CONTROL_FRAME, f"{CONTROL_FRAME}.png", "raw",
                 caption="Grass seen from directly above, with a few separate objects."),
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                "Each frame holds one straight line of joined stone wall pieces with a square post at each "
                "end. First, for each frame, say which way the line runs across the image, then for each "
                "piece between the two end posts say which way its own long side points. Then the closed "
                "question, ignoring the end posts. ALIGNED if in both frames every piece points along the "
                "line it belongs to. MIXED if the pieces mostly point along their line but at least one "
                "piece points clearly another way from its neighbours. CROSSED if in either frame most "
                "pieces point clearly at an angle to the line they form."
            ),
            frames=("a", "b"),
            answers=("ALIGNED", "MIXED", "CROSSED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                "Each frame holds one line of stone wall pieces running diagonally, corner to corner, with "
                "a post at each end. First, for each frame, say which way the line slopes, then which way "
                "the pieces between the end posts lean. Then, ignoring the end posts: ALIGNED if in both "
                "frames those pieces lean the same way as their line slopes, even if less steeply. "
                "CROSSED if in either frame most of those pieces lie straight across the image "
                "(horizontal or vertical) or lean against the slope of their line."
            ),
            frames=("c", "d"),
            answers=("ALIGNED", "CROSSED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "Each wall line in these frames has one gatehouse in it: a larger piece standing in a gap "
                "between the wall pieces. First, for each gatehouse, describe its outline, which way its "
                "long side points, and how much of the gap between the wall pieces on either side it "
                "fills. Then: ALONG if every gatehouse's long side points along the wall line it sits in. "
                "ACROSS if any gatehouse points clearly at an angle to its wall line, however much of the "
                "gap it fills."
            ),
            frames=GATE_FRAMES,
            answers=("ALONG", "ACROSS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                "First describe each object in the frame: its outline, its size and which way any long side "
                "points. Then: NORMAL if every object reads as one whole, upright object. ODD if any object "
                "is drawn lying on its side, stretched, split in two or cut off."
            ),
            frames=(CONTROL_FRAME,),
            answers=("NORMAL", "ODD", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game, drawn as a flat top-down "
            "grid: each square of grass is one map tile, and each object is a small picture fitted inside "
            "the tiles it stands on. Each frame is a crop, enlarged by a whole-number factor with no "
            "smoothing. Judge only what you can actually see."
        ),
        injects=INJECTS,
    )


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / PACK_ID / grp._run_token(PACK_ID, inject, replica)
    pack_dir.mkdir(parents=True, exist_ok=True)
    images = _frame_images(inject)
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


def check() -> None:
    """Spec coupling, frame band, determinism, each inject's reach per frame
    against INJECT_REACH, the localized inject's bound, and the per-piece axes
    the clean and coarse runs' answers rest on."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    scn = _scenario()
    tile_px = _tile_px(scn)
    terrain = render.render_terrain(scn)
    bare = {fid: _crop(terrain, box, tile_px) for fid, (box, _w, _g) in _FRAMES.items()}
    _t, clean_full = _full_images(None, scn, terrain)
    clean = {fid: _crop(clean_full, box, tile_px) for fid, (box, _w, _g) in _FRAMES.items()}
    again = _frame_images(None, scn, terrain)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean renders differ")
        if not rp.changed_pixels(clean[frame.id], bare[frame.id]).any():
            failures.append(f"frame {frame.id}: nothing is drawn on the grass")
        try:
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))

    clean_axes = _axes(clean_full, terrain, tile_px)
    for fid in (*RUN_FRAMES, *GATE_FRAMES):
        shown = ", ".join(f"{label} {a:.1f}/{e:.1f}" for label, a, e in clean_axes[fid])
        print(f"measure: clean {fid} axis/elongation {shown}")
    failures += _run_axis_failures(clean_axes, turned=True) + _gate_axis_failures(clean_axes, turned=True)

    coarse_largest: dict[str, int] = {}
    for name in INJECTS:
        _t, injected_full = _full_images(name, scn, terrain)
        injected = {fid: _crop(injected_full, box, tile_px) for fid, (box, _w, _g) in _FRAMES.items()}
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if set(touched) != INJECT_REACH[name]:
            failures.append(f"inject {name}: changed frames {sorted(touched)}, expected {sorted(INJECT_REACH[name])}")
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
        axes = _axes(injected_full, terrain, tile_px)
        if name == COARSE_INJECT:
            coarse_largest = {fid: comp for fid, (_px, comp) in touched.items()}
            for fid in (*RUN_FRAMES, *GATE_FRAMES):
                shown = ", ".join(f"{label} {a:.1f}" for label, a, _e in axes[fid])
                print(f"measure: {name} {fid} axis {shown}")
            failures += [
                f"inject {name}: {f}"
                for f in _run_axis_failures(axes, turned=False) + _gate_axis_failures(axes, turned=False)
            ]
        elif name == "one-piece":
            mask = rp.changed_pixels(clean["a"], injected["a"])
            failures += ONE_PIECE_BOUND.failures(mask, "inject one-piece frame a")
            if touched.get("a", (0, 0))[1] >= coarse_largest.get("a", 0):
                failures.append("inject one-piece: not smaller than the coarse inject on frame a")
            moved = [
                (label, a) for (label, a, _e), (_l, c, _ce) in zip(axes["a"], clean_axes["a"], strict=True)
                if vf.axis_gap_deg(a, c) > vf.TOLERANCE_DEG
            ]
            print(f"measure: {name} a pieces off their clean axis: {moved}")
            if [label for label, _a in moved] != [f"wall {ONE_PIECE_TILE}"]:
                failures.append(f"inject one-piece: moved {moved}, expected only wall {ONE_PIECE_TILE}")
        elif name == "gates-unturned":
            for fid in GATE_FRAMES:
                rows = [r for r in axes[fid] if r[0].startswith("const")]
                print(f"measure: {name} {fid} gate axis " + ", ".join(f"{label} {a:.1f}" for label, a, _e in rows))
            failures += [
                f"inject {name}: {f}"
                for f in _gate_axis_failures(axes, turned=False) + _run_axis_failures(axes, turned=True)
            ]
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite their own frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of one run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: the pack judges sprite art")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
