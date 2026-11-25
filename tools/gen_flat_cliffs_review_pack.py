#!/usr/bin/env python3
"""Review pack for TASK-350: top-down Flat (Elevation View Flat, Isometric View
off) de-projects the thin cliff straights (unit_sprites.
FLAT_DEPROJECT_CLIFF_FRAMES) so a thin run lies along the square grid.

The GH #151 sibling of tools/gen_flat_structures_review_pack.py, kept apart
so neither spec grows crowded. Same protocol (tools/REVIEW_PACK.md), same
Qt-free oracle render: render_flat.render_terrain() plus render_flat.
overlay_units(with_sprites=True) on one synthetic scenario, cropped per frame
and enlarged by an integer nearest-neighbour factor. Needs a configured
AoE2:DE install. PACK_ID and every frame/check name stay neutral.

Frames, one feature each, all the Default cliff family: four 3x1 pieces end
to end along x on frames 10/11 (a), four 1x3 pieces along y on frames 7/8
(b), and the specificity frame (c): three 3x3 pieces along x on frames 4/5,
which no inject reaches since tall frames are never de-projected.

Injects:
    undeprojected   coarse: no frame is de-projected, i.e. the render before
                    TASK-350, every thin piece contain-fitted.
    one-piece       localized: one interior piece of frame a's run contain-fitted.

Run:
    tools/gen_flat_cliffs_review_pack.py --out-dir <neutral dir>
    tools/gen_flat_cliffs_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_flat_cliffs_review_pack.py --out-dir <neutral dir> --inject one-piece
    tools/gen_flat_cliffs_review_pack.py --check      # writes nothing

Writes <out-dir>/top_view_objects_r/<opaque token>/ (build/review_pack/ by default, gitignored).
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

from descape import asset_source, cliff_catalog, render_flat, render_paint, render_units, unit_sprites
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from descape.terrain_palette import tile_span
from testkit import review_pack as rp

PACK_ID = "top_view_objects_r"
OUT_DIR = grp.OUT_DIR
GRASS = 0
GAIA = 0
TALL = 264  # Cliff (Default) 01, 3x3; its family's suffix 2 (1x3) and 3 (3x1) pieces below

# frame id -> (crop tiles x0, y0, x1, y1 half-open, suffix, [(low corner tile, frame)]).
_FRAMES = {
    "a": ((18, 18, 34, 23), 3, [((20 + 3 * i, 20), (10, 11)[i % 2]) for i in range(4)]),
    "b": ((38, 18, 43, 34), 2, [((40, 20 + 3 * i), (8, 7)[i % 2]) for i in range(4)]),
    "c": ((18, 38, 33, 45), 1, [((20 + 3 * i, 40), (4, 5)[i % 2]) for i in range(3)]),
}
RUN_FRAMES = ("a", "b")
CONTROL_FRAME = "c"
_RUN_AXES = {"a": 0.0, "b": 90.0}
# The one-piece inject's target: an interior piece of frame a's run.
ONE_PIECE_TILE = (26, 20)


def _ref(tile: tuple[int, int]) -> int:
    return 20_000 + tile[1] * 1000 + tile[0]


def _const(suffix: int) -> int:
    const = TALL if suffix == 1 else cliff_catalog.piece_for_suffix(TALL, suffix)
    assert const is not None, f"the Default cliff family has no suffix {suffix} piece"
    return const


def _units() -> list[tuple[str, int, tuple[int, int], int, float, float]]:
    """(frame id, const, low corner tile, cliff frame, x, y) for every cliff."""
    out = []
    for fid, (_box, suffix, pieces) in _FRAMES.items():
        const = _const(suffix)
        sx, sy = tile_span(const, (1, 1))
        for tile, frame in pieces:
            out.append((fid, const, tile, frame, *render_units.span_anchor(*tile, sx, sy)))
    return out


def _scenario():
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = GRASS
    placed = [[] for _ in base.unit_manager.units]
    for _fid, const, tile, frame, x, y in _units():
        placed[GAIA].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=float(frame), initial_animation_frame=frame,
            reference_id=_ref(tile), status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=placed),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


# ----------------------------------------------------------------- injects


def _one_piece_icon_for(real_icon_for, real_frames):
    """icon_for() with no frame de-projected for one reference_id only."""
    target = _ref(ONE_PIECE_TILE)

    def icon_for(*args, **kwargs):
        seed = kwargs.get("seed", args[5] if len(args) > 5 else None)
        if seed != target:
            return real_icon_for(*args, **kwargs)
        unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES = frozenset()
        try:
            return real_icon_for(*args, **kwargs)
        finally:
            unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES = real_frames
    return icon_for


INJECTS = ("undeprojected", "one-piece")
COARSE_INJECT = "undeprojected"
# Which frames each inject may change. The control frame is in none.
INJECT_REACH = {"undeprojected": frozenset(RUN_FRAMES), "one-piece": frozenset({"a"})}


def _one_piece_bound(tile_px: int) -> rp.LocalizedBound:
    """One 3x1 piece's footprint at most."""
    return rp.LocalizedBound(max_changed_fraction=0.3, max_component_px=3 * tile_px * tile_px)


@contextmanager
def _patched(inject: str | None):
    real_frames = unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES
    real_icon_for = unit_sprites.icon_for
    unit_sprites.clear_caches()
    if inject == "undeprojected":
        unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES = frozenset()
    elif inject == "one-piece":
        unit_sprites.icon_for = _one_piece_icon_for(real_icon_for, real_frames)
    try:
        yield
    finally:
        unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES = real_frames
        unit_sprites.icon_for = real_icon_for
        unit_sprites.clear_caches()


# --------------------------------------------------------------- rendering


def _tile_px(scn) -> int:
    return render_paint.tile_pixels_for_map(scn.map_manager.map_width, scn.map_manager.map_height)


def _crop(img: np.ndarray, box, tile_px: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.ascontiguousarray(img[y0 * tile_px : y1 * tile_px, x0 * tile_px : x1 * tile_px, :3])


def _full_images(inject: str | None, scn=None, terrain=None) -> tuple[np.ndarray, np.ndarray]:
    """(terrain only, terrain plus Flat icons) for the whole map."""
    scn = scn or _scenario()
    terrain = render_flat.render_terrain(scn) if terrain is None else terrain
    with _patched(inject):
        full = render_flat.overlay_units(terrain, scn, with_sprites=True)
    return terrain, full


def _frame_images(inject: str | None, scn=None, terrain=None) -> dict[str, np.ndarray]:
    scn = scn or _scenario()
    _terrain, full = _full_images(inject, scn, terrain)
    tile_px = _tile_px(scn)
    return {fid: _crop(full, box, tile_px) for fid, (box, _s, _p) in _FRAMES.items()}


# --------------------------------------------------------------- measurement


def _piece_rows(full: np.ndarray, terrain: np.ndarray, tile_px: int) -> dict[str, list[tuple]]:
    """Per frame: (low corner tile, axis, share of the footprint's long side inked) for each piece."""
    out: dict[str, list[tuple]] = {fid: [] for fid in _FRAMES}
    for fid, const, tile, _frame, x, y in _units():
        unit = SimpleNamespace(unit_const=const, x=x, y=y)
        tx0, tx1, ty0, ty1 = render_units.unit_tile_bounds(unit, 120, 120)
        sl = np.s_[ty0 * tile_px : ty1 * tile_px, tx0 * tile_px : tx1 * tile_px]
        mask = rp.changed_pixels(full[sl], terrain[sl])
        if not mask.any():
            out[fid].append((tile, None, 0.0))
            continue
        long_axis = 0 if mask.shape[1] >= mask.shape[0] else 1
        reach = mask.any(axis=long_axis).mean()
        out[fid].append((tile, vf.principal_axis_deg(mask.astype(np.uint8) * 255), float(reach)))
    return out


def _run_failures(rows, *, along: bool) -> list[str]:
    """along: every run frame's median piece is within tolerance of its run's
    axis and every piece inks at least 0.9 of its long side; not along: the
    median piece is outside it. The median is of the gaps, since axes wrap at 180."""
    failures = []
    for fid in RUN_FRAMES:
        want = _RUN_AXES[fid]
        gaps = [90.0 if a is None else vf.axis_gap_deg(a, want) for _t, a, _r in rows[fid]]
        gap = statistics.median(gaps)
        if (gap <= vf.TOLERANCE_DEG) != along:
            failures.append(f"frame {fid}: median piece is {gap:.1f} off its run's axis {want} (along={along})")
        reach = min(r for _t, _a, r in rows[fid])
        if along and reach < 0.9:
            failures.append(f"frame {fid}: a piece inks only {reach:.2f} of its long side")
    return failures


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    run_caption = "Grass seen from directly above, with one line of rock pieces placed end to end."
    frames = (
        rp.Frame("a", "a.png", "raw", caption=run_caption),
        rp.Frame("b", "b.png", "raw", caption=run_caption),
        rp.Frame(CONTROL_FRAME, f"{CONTROL_FRAME}.png", "raw",
                 caption="Grass seen from directly above, with a few large rock pieces in a row."),
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                "Each frame holds one straight line of rock pieces placed end to end across the grass. "
                "First, for each frame, say which way the line of pieces runs across the image, then for "
                "each piece describe its outline, which way its own long side points, and how much of its "
                "stretch of the line it covers. Then the closed question. ALONG if in both frames every "
                "piece lies along the line, so the pieces read as one band of rock. MIXED if most pieces "
                "lie along the line but at least one is a short slanted mark at an angle to it. SLANTED if "
                "in either frame most pieces are short slanted marks at an angle to the line they form."
            ),
            frames=RUN_FRAMES,
            answers=("ALONG", "MIXED", "SLANTED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                "First describe each object in the frame: its outline, its size and which way any long side "
                "points. Then: NORMAL if every object reads as one whole block of rock. ODD if any object "
                "is drawn stretched, sheared into a thin slab, split in two or cut off."
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
            "grid: each square of grass is one map tile, and each object is a picture drawn inside the "
            "tiles it stands on. Each frame is a crop, enlarged by a whole-number factor with no "
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
    and run coverage the clean and coarse runs' answers rest on."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    for fid, (_box, suffix, pieces) in _FRAMES.items():
        frames = {frame for _tile, frame in pieces}
        if (frames <= unit_sprites.FLAT_DEPROJECT_CLIFF_FRAMES) != (fid in RUN_FRAMES):
            failures.append(f"frame {fid}: suffix {suffix} frames {sorted(frames)} against the de-projected set")
    scn = _scenario()
    tile_px = _tile_px(scn)
    terrain = render_flat.render_terrain(scn)
    bare = {fid: _crop(terrain, box, tile_px) for fid, (box, _s, _p) in _FRAMES.items()}
    _t, clean_full = _full_images(None, scn, terrain)
    clean = {fid: _crop(clean_full, box, tile_px) for fid, (box, _s, _p) in _FRAMES.items()}
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

    clean_rows = _piece_rows(clean_full, terrain, tile_px)
    for fid in _FRAMES:
        shown = ", ".join(f"{t} {a:.1f}/{r:.2f}" for t, a, r in clean_rows[fid] if a is not None)
        print(f"measure: clean {fid} axis/long-side reach {shown}")
    failures += _run_failures(clean_rows, along=True)

    coarse_largest: dict[str, int] = {}
    for name in INJECTS:
        _t, injected_full = _full_images(name, scn, terrain)
        injected = {fid: _crop(injected_full, box, tile_px) for fid, (box, _s, _p) in _FRAMES.items()}
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
        rows = _piece_rows(injected_full, terrain, tile_px)
        if name == COARSE_INJECT:
            coarse_largest = {fid: comp for fid, (_px, comp) in touched.items()}
            for fid in RUN_FRAMES:
                shown = ", ".join(f"{t} {a:.1f}/{r:.2f}" for t, a, r in rows[fid])
                print(f"measure: {name} {fid} axis/long-side reach {shown}")
            failures += [f"inject {name}: {f}" for f in _run_failures(rows, along=False)]
        elif name == "one-piece":
            mask = rp.changed_pixels(clean["a"], injected["a"])
            failures += _one_piece_bound(tile_px).failures(mask, "inject one-piece frame a")
            if touched.get("a", (0, 0))[1] >= coarse_largest.get("a", 0):
                failures.append("inject one-piece: not smaller than the coarse inject on frame a")
            moved = [
                (t, a) for (t, a, _r), (_ct, c, _cr) in zip(rows["a"], clean_rows["a"], strict=True)
                if vf.axis_gap_deg(a, c) > vf.TOLERANCE_DEG
            ]
            print(f"measure: {name} a pieces off their clean axis: {moved}")
            if [t for t, _a in moved] != [ONE_PIECE_TILE]:
                failures.append(f"inject one-piece: moved {moved}, expected only the piece at {ONE_PIECE_TILE}")
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
