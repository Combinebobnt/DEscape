#!/usr/bin/env python3
"""Review pack for TASK-222: the waterfalls (896 WFALL, 1635 its background
twin) draw their particle effect's art, each placement's variant choosing one
of the effect's ImageAngles runs.

Protocol: tools/REVIEW_PACK.md. Rendered off-engine (render_cache.IsoChunkCache,
Stepped, 64 px tiles, sprites on) from examples/R4_LeLoi_4.aoe2scenario, the
only corpus file that places waterfalls: 896 at (219, 37) variant 0, and a
row at y 129 (1635 at x 186, 896 at 187 and 188) at variant 1. PACK_ID and
every frame/check name stay neutral.

Frames: row (the three-wide row), lone (the 896 at (219, 37)), and the
specificity frame ledge (the same cliff beside the row, with no waterfall,
which no inject reaches). The plan's butterfly and seagull control is left to
tests/test_particle_art.py and the release render diff: at this zoom their art
is a few faint specks.

Injects:
    one-size   coarse (the largest delta): the row's 1635 at twice its art_scale.
    slight     localized: the row's 1635 at 1.3 times its art_scale, bounded.
    marks      no waterfall has a table entry, so each draws as a coloured
               mark, the render before TASK-222.
    facing     every waterfall drawn at variant 0.

The plan's "foam pool on its own tile" check is not posed: where a sprite's
anchor lands is countable and belongs in an assertion (not yet written).

Run:
    tools/gen_waterfall_review_pack.py --out-dir <neutral dir>
    tools/gen_waterfall_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_waterfall_review_pack.py --out-dir <neutral dir> --inject facing
    tools/gen_waterfall_review_pack.py --check      # writes nothing

Needs examples/ and a configured AoE2:DE install.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from descape import asset_source, render_iso, unit_sprites
from descape import iso_geometry as ig
from descape.scenario_io import load_map_and_units
from testkit import review_pack as rp

PACK_ID = "scenery_k"
OUT_DIR = grp.OUT_DIR
SCENARIO = ROOT / "examples" / "R4_LeLoi_4.aoe2scenario"
TILE_PX = 64
WATERFALLS = (896, 1635)
ONE_SIZE_CONST = 1635  # only the row's left fall is this const
# frame id -> (anchor tile, crop box in canvas px around that tile's centre).
_FRAMES = {
    "row": ((187, 129), (-160, -150, 80, 30)),
    "lone": ((219, 37), (-160, -140, 80, 40)),
    "ledge": ((187, 129), (90, -230, 330, -50)),
}
SPECIFICITY = "ledge"
INJECTS = ("marks", "facing", "one-size", "slight")
COARSE_INJECT = "one-size"
INJECT_REACH = {"marks": {"row", "lone"}, "facing": {"row"}, "one-size": {"row"}, "slight": {"row"}}
LOCALIZED_INJECT = "slight"
SCALE = {"one-size": 2.0, "slight": 1.3}
# Measured 2026-11-27: slight changes 6872 px of the row frame (43200), largest component 6783.
LOCALIZED_BOUND = rp.LocalizedBound(max_changed_fraction=0.2, max_component_px=8000)


@contextmanager
def _patched(inject: str | None):
    """The run's graphic table, every sprite cache dropped on both sides."""
    real_table = unit_sprites.graphic_map
    if inject in ("marks", *SCALE):
        table = dict(real_table())
        if inject == "marks":
            for const in WATERFALLS:
                del table[const]
        else:
            entry = table[ONE_SIZE_CONST]
            table[ONE_SIZE_CONST] = {**entry, "art_scale": SCALE[inject] * entry["art_scale"]}
        unit_sprites.graphic_map = lambda: table
    unit_sprites.clear_caches()
    try:
        yield
    finally:
        unit_sprites.graphic_map = real_table
        unit_sprites.clear_caches()


def _scenario(inject: str | None):
    loaded = load_map_and_units(str(SCENARIO))
    falls = [u for units in loaded.unit_manager.units for u in units if u.unit_const in WATERFALLS]
    if len(falls) != 4:
        raise SystemExit(f"{SCENARIO.name}: expected 4 waterfalls, found {len(falls)}")
    if inject == "facing":
        for unit in falls:
            unit.rotation = 0.0
    return loaded


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    from descape.render_cache import IsoChunkCache
    from descape.unit_filter import UnitFilter

    loaded = _scenario(inject)
    elevations, proj = render_iso.elevations_and_proj(loaded)
    images = {}
    with _patched(inject):
        cache = IsoChunkCache(loaded, elevations, proj, TILE_PX, unit_filter=UnitFilter(show_garrisoned=False), sprites=True)
        for fid, ((tx, ty), (x0, y0, x1, y1)) in _FRAMES.items():
            sx, sy = ig.tile_screen_origin(tx, ty, int(elevations[ty, tx]), proj)
            ax, ay = sx + proj.half_w, sy + proj.half_h
            images[fid] = np.ascontiguousarray(cache.render_rect(ax + x0, ay + y0, ax + x1, ay + y1, mip=0)[..., :3])
    return images


def _spec() -> rp.PackSpec:
    ignore = (
        "Ignore the small pale square tiles marked with a cross: they are editor markers, not scenery. "
    )
    caption = "A rocky ledge above water."
    frames = (
        rp.Frame("row", "row.png", "raw", caption=caption),
        rp.Frame("lone", "lone.png", "raw", caption=caption),
        rp.Frame(SPECIFICITY, f"{SPECIFICITY}.png", "raw", caption=caption),
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                ignore + "First, for each frame, describe everything drawn on or against the face of the "
                "ledge, other than rock and plants: its outline, its colours and what it looks like. Then the "
                "closed question. WATER if in both frames each such thing is a picture of water pouring over "
                "the ledge. MARK if in either frame any such thing is a plain coloured shape or symbol rather "
                "than a picture. NONE if neither frame has anything there but rock and plants."
            ),
            frames=("row", "lone"),
            answers=("WATER", "MARK", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                ignore + "First, for each frame, say whether any water is drawn pouring over the ledge, and "
                "if so describe it: how wide it is against its height, and which way its face and its spray "
                "point across the image. Then the closed question, comparing the first frame with the "
                "second. DIFFERENT if both have pouring water and the two are drawn as clearly different "
                "pictures (a different shape, width or direction), SAME if both have it and it is the same "
                "picture repeated, only placed differently, NONE if either frame has no pouring water."
            ),
            frames=("row", "lone"),
            answers=("DIFFERENT", "SAME", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                ignore + "First say whether any water is drawn pouring over the ledge, and if so describe "
                "each stretch of it from left to right: its width, its height and where its top and bottom "
                "sit. Then: EVEN if there are several stretches and all are drawn at about the same size, "
                "UNEVEN if any one is clearly larger or smaller than the others, NONE if no water pours "
                "over the ledge. Count a difference only in the size of the water itself, not in its "
                "brightness or in the rock around it."
            ),
            frames=("row",),
            answers=("EVEN", "UNEVEN", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                ignore + "First describe what lies along the face of the ledge, from left to right: the "
                "rock, any plants, and anything else drawn on or against it. Then: BARE if nothing but "
                "rock and plants is drawn there, OTHER if any water pours over the ledge or any plain "
                "coloured shape or symbol sits on its face."
            ),
            frames=(SPECIFICITY,),
            answers=("BARE", "OTHER", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map scenery review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game, drawn in an angled "
            "three-quarter view where each diamond of ground is one map tile. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
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
    """Spec coupling and leaks, frame band, determinism, each inject's reach per frame against
    INJECT_REACH, and the localized inject staying under the coarse one."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += [f"leak: {word}" for word in rp.leaks(spec)]
    clean, again = _frame_images(None), _frame_images(None)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean renders differ")
        scaled, factor = rp.frame_to_band(clean[frame.id])
        try:
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
        print(f"frame {frame.id}: {clean[frame.id].shape[1]}x{clean[frame.id].shape[0]} x{factor}")
    largest: dict[str, dict[str, int]] = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for fid in _FRAMES:
            mask = rp.changed_pixels(clean[fid], injected[fid])
            if mask.any():
                touched[fid] = (int(mask.sum()), rp.largest_component_px(mask))
        largest[name] = {fid: px for fid, (px, _c) in touched.items()}
        print(f"reach: {name} -> " + ", ".join(f"{f} {px}px/{c}max" for f, (px, c) in touched.items()))
        if set(touched) != INJECT_REACH[name]:
            failures.append(f"inject {name}: changed frames {sorted(touched)}, expected {sorted(INJECT_REACH[name])}")
    for name in INJECTS:
        if name == COARSE_INJECT:
            continue
        for fid in largest[name].keys() & largest[COARSE_INJECT].keys():
            px, coarse = largest[name][fid], largest[COARSE_INJECT][fid]
            if px >= coarse:
                failures.append(f"inject {name}: {px} px on {fid}, not under the coarse inject's {coarse}")
    slight = rp.changed_pixels(clean["row"], _frame_images(LOCALIZED_INJECT)["row"])
    failures += LOCALIZED_BOUND.failures(slight, f"inject {LOCALIZED_INJECT} frame row")
    if failures:
        raise SystemExit("FAILED:\n  " + "\n  ".join(failures))
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite only their own frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of one run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None or not SCENARIO.exists():
        raise SystemExit(f"needs a configured AoE2:DE install and {SCENARIO.relative_to(ROOT)}")
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
