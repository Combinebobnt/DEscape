#!/usr/bin/env python3
"""Review pack for TASK-221: a unit standing beside a building's front faces
draws in front of it, because each screen column of a multi-tile sprite paints
at its own front footprint tile (render._sprite_columns()).

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_clipped_sprite_review_pack.py, whose scenario
and cache helpers this mirrors. Renders off-engine through the app's own
Stepped chunk cache with units and sprites on and the native kernel live.
Needs a configured AoE2:DE install (unit sprites, terrain textures) and the
built kernel. Sloped ordering is pinned by tests/test_sprite_columns.py and
tests/test_composite_depth_real_assets.py, not by this pack.

Run:
    tools/gen_building_faces_review_pack.py --out-dir /tmp/review
    tools/gen_building_faces_review_pack.py --out-dir /tmp/review --inject single-anchor
    tools/gen_building_faces_review_pack.py --out-dir /tmp/review --replica 1
    tools/gen_building_faces_review_pack.py --check      # writes nothing

Writes <out-dir>/<PACK_ID>/<opaque token>/ (default build/review_pack,
gitignored). Pass --out-dir whenever the checkout's own path names the work:
every frame path is printed in REVIEW.md.

Both injects run in the kernel: they change which tile each column run is
keyed on, upstream of every painter.
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

from descape import asset_source, composite_backend, render, render_cache, settings, unit_sprites
from descape import iso_geometry as ig
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
from testkit import review_pack as rp

# In every frame path REVIEW.md prints, so it must not name the artifact.
PACK_ID = "figures_buildings"
OUT_DIR = grp.OUT_DIR
BARRACKS, TOWN_CENTRE, GATE = 12, 109, 64  # 3x3, 4x4 and a 4x1 gate
ARCHER, VILLAGER, SPEARMAN = 4, 83, 93
GRASS, SAND, ALT_GRASS = 0, 14, 47
PLAYER = 1
RAISED = 2
# The corpus tests' scale (tests/test_composite_depth_real_assets.py), where the
# single anchor's cover was measured; the blank map's own 64 px leaves figures tiny.
TILE_PX = 2 * render.SMALL_MAP_TILE_PIXELS
_PAD = 16
# A figure frame's window at TILE_PX: about two tiles each way round the figure.
_FIGURE_W, _FIGURE_H = 288, 224

# Footprints: Barracks 60..62 on both axes, town centre 58..61, gate x 60..63 on y 60.
_BARRACKS = (BARRACKS, 61.0, 61.0)
_TOWN_CENTRE = (TOWN_CENTRE, 60.0, 60.0)
_GATE = (GATE, 62.0, 60.5)


def _at(const: int, tile: tuple[int, int]) -> tuple[int, float, float]:
    return const, tile[0] + 0.5, tile[1] + 0.5


def _flat(_x: int, _y: int) -> tuple[int, int]:
    return 0, GRASS


def _left_rise(x: int, y: int) -> tuple[int, int]:
    """Raised sand in front of the Barracks' front-left face, inside its footprint's
    own screen columns (x + y >= 120): art overhanging left of those keeps the
    single anchor by design (_sprite_columns()), which is not this frame's subject."""
    return (RAISED, SAND) if x <= 59 and x + y >= 120 else (0, GRASS)


# frame id -> (building, figure or None, ground). Ids are neutral: REVIEW.md prints them.
# a: front-left face, a spearman: an archer's dark hair on the dark wall there read to a
# reviewer as a missing head; b: the town
# centre's front-right face (191 px); c: behind the Barracks (covered by design);
# d: in front of the gate's tiles right of its front one (its front-right face); e: raised ground at the front-left face
# (no figure); f: a figure clear of the building (the specificity frame).
_FRAMES = {
    "a": (_BARRACKS, _at(SPEARMAN, (59, 61)), _flat),
    "b": (_TOWN_CENTRE, _at(VILLAGER, (60, 62)), _flat),
    "c": (_BARRACKS, _at(ARCHER, (63, 59)), _flat),
    "d": (_GATE, _at(ARCHER, (62, 61)), _flat),
    "e": (_BARRACKS, None, _left_rise),
    "f": (_BARRACKS, _at(ARCHER, (58, 60)), _flat),
}


def _scenario(placed, ground, terrain_id: int | None = None):
    """`terrain_id` repaints the low ground only (the figure-opacity probe)."""
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation, tile.terrain_id = ground(tile.x, tile.y)
        if terrain_id is not None and tile.elevation == 0:
            tile.terrain_id = terrain_id
    units = [[] for _ in base.unit_manager.units]
    for ref, (const, x, y) in enumerate(placed):
        units[PLAYER].append(SimpleNamespace(
            unit_const=const, x=x, y=y, z=0.0, rotation=0.0,
            reference_id=8000 + ref, status=2, garrisoned_in_id=-1,
        ))
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        _base=base,
    )


def _elevations_and_proj(scn):
    """Pinned to TILE_PX and the default step, so the user's settings never move a frame."""
    elevations, _proj = render.elevations_and_proj(scn)
    mm = scn.map_manager
    proj = ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, TILE_PX, ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=ig.ELEV_STEP_DEFAULT_PCT,
    )
    return elevations, proj


def _cache(scn):
    """The app's own Stepped chunk cache, sprites on, default layers."""
    elevations, proj = _elevations_and_proj(scn)
    return render_cache.IsoChunkCache(scn, elevations, proj, TILE_PX, sprites=True)


def _sprite_box(unit, ground) -> tuple[int, int, int, int]:
    """One unit's sprite bbox, the union of its column slices."""
    scn = _scenario([unit], ground)
    elevations, proj = _elevations_and_proj(scn)
    boxes = list(render.sprite_draws_by_anchor(scn, proj, elevations).bboxes.values())
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _box(frame_id: str) -> tuple[int, int, int, int]:
    """A figure frame is a fixed window centred on the figure, so the place it
    meets the building fills the frame. Frame e is the building plus three
    tiles at its front-left face (two of them not raised; they only pad the
    crop). Neither moves under an inject."""
    building, figure, ground = _FRAMES[frame_id]
    if figure is not None:
        x0, y0, x1, y1 = _sprite_box(figure, ground)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        return cx - _FIGURE_W // 2, cy - _FIGURE_H // 2, cx + _FIGURE_W // 2, cy + _FIGURE_H // 2
    x0, y0, x1, y1 = _sprite_box(building, ground)
    _elev, proj = _elevations_and_proj(_scenario((), ground))
    tx0, ty0, tx1, ty1 = grp._tiles_bbox_px(proj, [(58, 60), (59, 63), (59, 60)], RAISED)
    return min(x0, tx0) - _PAD, min(y0, ty0) - _PAD, max(x1, tx1) + _PAD, max(y1, ty1) + _PAD


def _render(frame_id: str, keep: str = "both", terrain_id: int | None = None) -> np.ndarray:
    """`keep` is "both", "building", "figure" or "none"."""
    building, figure, ground = _FRAMES[frame_id]
    placed = {"both": (building, figure), "building": (building,), "figure": (figure,), "none": ()}[keep]
    cache = _cache(_scenario([u for u in placed if u is not None], ground, terrain_id))
    return cache._composite_rect(0, *_box(frame_id))


# ----------------------------------------------------------------- injects


def _inject_single_anchor():
    """COARSE: every multi-tile sprite paints whole at its front corner, the
    render from before TASK-221."""
    return render, "SPRITE_COLUMN_SLICING", False


def _rekeyed(back_on_key):
    """render._sprite_columns() with each screen strip s (tile_screen_origin():
    [origin_x + s*half_w, origin_x + (s+1)*half_w)) put back on the piece's
    own key wherever back_on_key(s, key) holds."""
    real = render._sprite_columns

    def columns(occupied, key, x0, w, proj):
        runs: list[list] = []
        for anchor, c0, c1, _moved in real(occupied, key, x0, w, proj):
            a = c0
            while a < c1:
                strip = (x0 + a - proj.origin_x) // proj.half_w
                b = min(c1, proj.origin_x + (strip + 1) * proj.half_w - x0)
                run_key = key if back_on_key(strip, key) else anchor
                if runs and runs[-1][0] == run_key and runs[-1][2] == a:
                    runs[-1][2] = b
                else:
                    runs.append([run_key, a, b])
                a = b
        return [(anchor, c0, c1, anchor != key) for anchor, c0, c1 in runs]

    return render, "_sprite_columns", columns


def _inject_left_face_only():
    """FACE-SPECIFIC: strips right of a piece's own key tile (it covers strips
    kx+ky and kx+ky+1) keep that key, so only the front-left face's columns
    move. Reproduces the real defect at its real size on the right faces only."""
    return _rekeyed(lambda strip, key: strip >= key[0] + key[1] + 2)


def _inject_alternate_strips():
    """LOCALIZED: every odd strip keeps the piece's own key, so each face's
    defect comes back at about half its size, one strip in two."""
    return _rekeyed(lambda strip, _key: strip % 2 == 1)


# name -> builder returning one (module, attribute, replacement)
INJECTS = {
    "single-anchor": _inject_single_anchor,
    "left-face-only": _inject_left_face_only,
    "alternate-strips": _inject_alternate_strips,
}
COARSE_INJECT = "single-anchor"
# Held strictly under the coarse one's largest component wherever they share a frame.
LOCALIZED_INJECT = "alternate-strips"
SPECIFICITY_FRAME = "f"


@contextmanager
def _patched(inject: str | None):
    # The chunk cache reads the elevation step from settings itself; pinned with proj.
    patches = [(settings, "get_elev_step_pct", lambda: ig.ELEV_STEP_DEFAULT_PCT)]
    if inject is not None:
        patches.append(INJECTS[inject]())
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

_ORDER = (
    "Look at where the person and the building meet. First describe what you see there: which parts of the "
    "person are visible, and whether any part of the person's outline is cut off along a straight vertical or "
    "diagonal edge where the building begins. Then decide, using only the place where the two actually overlap "
    "on screen: is the person drawn over the building there (the person's whole figure is visible on top of the "
    "building's wall or roof), or is the building drawn over the person (part of the person is hidden by the "
    "building)? Do not count shadows, and do not count any gap of plain ground between them. PERSON-IN-FRONT if "
    "the person is drawn over the building, BUILDING-IN-FRONT if the building hides part of the person, "
    "NO-OVERLAP if the two do not overlap on screen at all."
)
_ORDER_ANSWERS = ("PERSON-IN-FRONT", "BUILDING-IN-FRONT", "NO-OVERLAP", "CANNOT-TELL")


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "A building on level grass, with a person standing on the grass near it."),
            ("b", "A large building on level grass, with a person standing on the grass near it."),
            ("c", "A building on level grass, with a person standing on the grass near it."),
            ("d", "A gateway on level grass, with a person standing on the grass near it."),
            ("e", "A building on grass, with an area of raised sandy ground beside it."),
            ("f", "A building on level grass, with a person standing on the grass near it."),
        )
    )
    order_checks = tuple(
        rp.Check(id=f"q{n}", question=_ORDER, frames=(fid,), answers=_ORDER_ANSWERS)
        for n, fid in enumerate(("a", "b", "c", "d", "f"), start=1)
    )
    raised = rp.Check(
        id="q6",
        question=(
            "Part of the ground here is raised: sandy, standing higher than the grass, with a darker steep side "
            "where it drops down to the grass. The building stands on the lower grass. First describe what you "
            "see where the building meets the raised sandy ground. Then: does any part of the building's lower "
            "half, near the ground, lie drawn on top of the raised sand or its steep side, or does the raised "
            "ground pass unbroken in front of the building's lower edge there? Do not count the building's upper "
            "parts (its roof, towers or flags) rising above and behind the raised ground, and do not count "
            "shadows. ON-RAISED if any of the building's lower half lies over the raised ground, BEHIND if none "
            "does."
        ),
        frames=("e",),
        answers=("ON-RAISED", "BEHIND", "CANNOT-TELL"),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Buildings and figures review",
        frames=frames,
        checks=(*order_checks, raised),
        preamble=(
            "These are offscreen renders from a map editor for a strategy game, seen from above at an angle: "
            "the ground is a grid of diamond-shaped tiles, and the bottom of a frame is nearest the viewer. Each "
            "frame is a crop, enlarged by a whole-number factor with no smoothing. Judge only what you can "
            "actually see."
        ),
        injects=tuple(INJECTS),
    )


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    with _patched(inject):
        return {frame_id: _render(frame_id) for frame_id in _FRAMES}


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


# ------------------------------------------------------------------- check


def _eroded(mask: np.ndarray) -> np.ndarray:
    """3x3 erosion: drops the antialiased rim."""
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _figure_cover(frame_id: str, full: np.ndarray) -> tuple[int, int]:
    """(covered, overlap): the figure's opaque ink the building paints over in
    `full`, and the part of that ink lying over building ink at all."""
    bare = _render(frame_id, "none")
    alone = _render(frame_id, "figure")
    opaque = ~rp.changed_pixels(alone, _render(frame_id, "figure", ALT_GRASS))
    figure = _eroded(rp.changed_pixels(alone, bare) & opaque)
    building = rp.changed_pixels(_render(frame_id, "building"), bare)
    covered = figure & rp.changed_pixels(full, alone)
    return int(covered.sum()), int((figure & building).sum())


def _raised_cover(full: np.ndarray) -> tuple[int, int]:
    """(area, raw): building ink drawn over raised ground in frame e. `area`
    is after one erosion: the clean render leaves two thin diagonal strips
    (2-4 px) where the wall's foot meets the raised edge, not wall over sand."""
    bare = _render("e", "none")
    # Two erosions drop the raised edge's antialiased rim, where the clean render's wall foot meets it.
    raised = _eroded(_eroded(rp.changed_pixels(bare, _flat_e())))
    ink = rp.changed_pixels(full, bare) & raised
    return int(_eroded(ink).sum()), int(ink.sum())


def _flat_e() -> np.ndarray:
    cache = _cache(_scenario((), _flat))
    return cache._composite_rect(0, *_box("e"))


# frame -> expected figure cover per run: NONE (PERSON-IN-FRONT), SOME (the
# localized inject's cut, at least a few px, BUILDING-IN-FRONT) or MOST (BUILDING-IN-FRONT).
NONE, SOME, MOST = 0, 1, 2
_COVER = {
    None: {"a": NONE, "b": NONE, "c": MOST, "d": NONE},
    "single-anchor": {"a": MOST, "b": MOST, "c": MOST, "d": MOST},
    "left-face-only": {"a": NONE, "b": MOST, "c": MOST, "d": MOST},
    "alternate-strips": {"a": SOME, "b": SOME, "c": MOST, "d": SOME},
}
_RAISED_COVER = {None: False, "single-anchor": True, "left-face-only": False, "alternate-strips": True}
# Bounds each answer option rests on, from the 2026-10-26 measurements: overlap
# 103-333 px; full cover 100% of it, alternate strips 18-149 px; raised clean 28 px, injected 880-1,407.
_MIN_OVERLAP_PX = 80
_MOST_SHARE = 0.5
_SOME_MIN_PX = 15
_RAISED_MAX_CLEAN_PX = 60
_RAISED_MIN_PX = 200


def check() -> None:
    """Spec coupling, frame band, the kernel against the numpy path on every
    clean frame, each inject's reach per frame, and the cover each answer
    option rests on, measured per run."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    with composite_backend.use_backend("numpy"):
        reference = _frame_images(None)
    for frame in spec.frames:
        if not np.array_equal(clean[frame.id], reference[frame.id]):
            failures.append(f"{frame.id}: the kernel and the numpy path disagree")
        try:
            scaled, factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
            print(f"frame {frame.id}: {clean[frame.id].shape[1]}x{clean[frame.id].shape[0]} x{factor}")
        except rp.PackError as exc:
            failures.append(str(exc))
    runs = {None: clean}
    reach = {}
    for name in INJECTS:
        runs[name] = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], runs[name][frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask), mask)
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        if SPECIFICITY_FRAME in touched:
            failures.append(f"inject {name}: reached the specificity frame")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp, _m) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    # Every other inject's change is a part of the coarse one's, so none can drift coarse;
    # the localized one is also strictly smaller per shared frame (REVIEW_PACK.md).
    coarse = reach[COARSE_INJECT]
    for name in INJECTS:
        if name == COARSE_INJECT:
            continue
        for fid, (_px, comp, mask) in reach[name].items():
            if fid not in coarse:
                failures.append(f"inject {name}: reached {fid}, which the coarse one does not")
            elif np.any(mask & ~coarse[fid][2]):
                failures.append(f"inject {name}: changed {fid} outside the coarse one's change")
            elif name == LOCALIZED_INJECT and comp >= coarse[fid][1]:
                failures.append(f"inject {name}: {fid} component {comp} not under the coarse one's {coarse[fid][1]}")
    for run, images in runs.items():
        label = run or "clean"
        for fid, expect in _COVER[run].items():
            covered, overlap = _figure_cover(fid, images[fid])
            print(f"cover: {label} {fid} {covered} of {overlap} px")
            if overlap < _MIN_OVERLAP_PX:
                failures.append(f"{label} {fid}: the figure overlaps only {overlap} px of building ink")
            if expect == MOST and covered < _MOST_SHARE * overlap:
                failures.append(f"{label} {fid}: covered {covered} of {overlap} px, expected most of it")
            if expect == SOME and covered < _SOME_MIN_PX:
                failures.append(f"{label} {fid}: covered {covered} of {overlap} px, expected a visible cut")
            if expect == NONE and covered:
                failures.append(f"{label} {fid}: covered {covered} px, expected none")
        covered, overlap = _figure_cover(SPECIFICITY_FRAME, images[SPECIFICITY_FRAME])
        if covered or overlap:
            failures.append(f"{label} {SPECIFICITY_FRAME}: the figure overlaps the building ({overlap} px)")
        raised, raw = _raised_cover(images["e"])
        print(f"raised: {label} e {raised} px of building ink over raised ground ({raw} before erosion)")
        if _RAISED_COVER[run] and raised < _RAISED_MIN_PX:
            failures.append(f"{label} e: {raised} px over raised ground, expected at least {_RAISED_MIN_PX}")
        if not _RAISED_COVER[run] and raised > _RAISED_MAX_CLEAN_PX:
            failures.append(f"{label} e: {raised} px over raised ground, expected at most {_RAISED_MAX_CLEAN_PX}")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(f"OK: {PACK_ID} frames couple to checks, land in band, all {len(INJECTS)} injects bite as registered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--inject", choices=sorted(INJECTS))
    parser.add_argument("--replica", type=int, default=0, help="a second or third opaque directory of the same run")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if asset_source.get_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: unit sprites and terrain textures")
    if composite_backend.native is None:
        raise SystemExit(f"needs the native kernel live: {composite_backend.describe()}")
    # The settings pin for every render, the --check probes outside a run included.
    with _patched(None):
        if args.check:
            check()
            return
        written = generate(args.out_dir, args.inject, args.replica)
    for path in written:
        print(f"wrote {path}")
    print(
        f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
        f"{grp._run_token(PACK_ID, args.inject, args.replica)} -- record that pairing OUTSIDE the pack"
    )


if __name__ == "__main__":
    main()
