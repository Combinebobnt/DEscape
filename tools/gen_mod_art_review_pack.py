#!/usr/bin/env python3
"""Review pack for TASK-378: the player's enabled cosmetic mods draw their own
art. Needs the AoE2:DE install and a profile with 0xDB Chunky Tree Tiles and
Age of Mandala's Tower range mod enabled, named in AOE2DE_PROFILE_PATH.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), Qt-free
like it: full off-engine renders with sprites on, cropped to the objects.

Frames, all on flat checkered ground where every object's own tile is a third,
pale terrain, so a reviewer can see which cell an object belongs to:
- a, b, c: four oak trees (const 349) on separate tiles, Mod Art on, in Flat,
  Stepped and Sloped.
- d: the same trees in Stepped with Mod Art off (the install's trees).
- e: a player-1 Watch Tower (const 79), Stepped, Mod Art on.
- f: checkered ground only, frame b's rect moved along +y: the specificity frame.

Injects, each the one defect its check asks about:
- `mod-off` (coarse): no mod resolves, so a, b, c draw the install's trees.
- `wrong-palette`: every .smx frame decodes against palette 28 and 30 swapped.
- `section-swap` (localized): a 4plus1 chunk's palette sections read in the
  reverse bit order, the openage-spec order the files disproved.
- `no-tint`: an .smx frame's player-colour layer is dropped.
- `shifted`: every .smx frame's hotspot moved one native tile width, and every
  Flat icon one tile, so the art lands beside its own tile.

The reach inject the plan listed ("reach left at the constants") is replaced
by tests/test_mod_graphics.py's synthetic wide-mod dirty-bbox test: no active
mod's drawn art reaches past the constants here (decided 2026-10-15).

Run:
    AOE2DE_PROFILE_PATH=<profile> tools/gen_mod_art_review_pack.py --out-dir <neutral dir>
    ... --replica 1 | --inject section-swap | --check (writes nothing)

Writes <out-dir>/objects/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import numpy as np

from testkit import review_pack as rp
from testkit import settings_isolation

PACK_ID = "objects"
OUT_DIR = grp.OUT_DIR
TREE = 349  # n_tree_oak_x1, which Chunky Tree Tiles replaces
TOWER = 79  # b_west_tower_age2_x1, which the Tower range mod replaces
CHECKER = (14, 0)  # DESERT_SAND, GRASS_1: one diamond cell per tile
OWN_CELL = 32  # SNOW: the pale cell under each object
TREE_TILES = ((50, 50), (52, 50), (50, 52), (52, 52))
TOWER_TILE = (70, 50)
GROUND_SHIFT_TILES = 30
PAD_PX = 24
STYLES = {"a": "flat", "b": "stepped", "c": "sloped"}


def _scenario(with_trees: bool, with_tower: bool, pale: bool = True):
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    own = (set(TREE_TILES) | {TOWER_TILE}) if pale else set()
    for tile in loaded.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = OWN_CELL if (tile.x, tile.y) in own else CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    if with_trees:
        for x, y in TREE_TILES:
            model.add(0, TREE, x + 0.5, y + 0.5)
    if with_tower:
        model.add(1, TOWER, TOWER_TILE[0] + 0.5, TOWER_TILE[1] + 0.5)
    return loaded


# ----------------------------------------------------------------- injects


def _inject_mod_off():
    """Coarse: no mod resolves, every sprite is the install's."""
    from descape import mod_graphics

    return [(mod_graphics, "resolve_sprite", lambda _name: None)]


def _inject_wrong_palette():
    """Every .smx frame on palettes 28 (b_scen) and 30 (n_trees) reads the other one."""
    from descape import mod_graphics

    real = mod_graphics.palettes

    class _Swapped:
        def __init__(self, inner):
            self._inner = inner

        def get(self, number, default=None):
            return self._inner.get({28: 30, 30: 28}.get(number, number), default)

    return [(mod_graphics, "palettes", lambda: _Swapped(real()))]


def _inject_section_swap():
    """A 4plus1 chunk's four 2-bit palette sections in the reverse order."""
    from descape import smx_decoder

    return [(smx_decoder, "_SECTION_SHIFTS_4PLUS1", np.array([6, 4, 2, 0], dtype=np.uint8))]


def _inject_no_tint():
    """An .smx frame keeps its grey player pixels but loses the player-colour layer."""
    from descape import smx_decoder

    real = smx_decoder._decode_main

    def untinted(*args, **kwargs):
        main, _player = real(*args, **kwargs)
        return main, None

    return [(smx_decoder, "_decode_main", untinted)]


def _inject_shifted():
    """Every .smx frame's hotspot one tile width to the right (native px), so its art lands
    one tile left; Flat places icons by their rect, not a hotspot, so each icon moves a tile too."""
    import dataclasses

    from descape import render, smx_decoder, unit_sprites

    real = smx_decoder.SMXFile.decode_frame
    real_draw = render._draw_unit

    def moved(self, *args, **kwargs):
        decoded = real(self, *args, **kwargs)
        return dataclasses.replace(decoded, hotspot_x=decoded.hotspot_x + unit_sprites.NATIVE_TILE_W)

    def draw_left(img, unit, color, tile_w, tile_h, tile_px, icon=None):
        if icon is not None:
            # Transparent columns on the right move a centred icon's ink tile_px left.
            pad = np.zeros((icon.rgba.shape[0], 2 * tile_px, 4), dtype=icon.rgba.dtype)
            icon = dataclasses.replace(icon, rgba=np.concatenate([icon.rgba, pad], axis=1))
        return real_draw(img, unit, color, tile_w, tile_h, tile_px, icon)

    return [(smx_decoder.SMXFile, "decode_frame", moved), (render, "_draw_unit", draw_left)]


INJECTS = {
    "mod-off": _inject_mod_off,
    "wrong-palette": _inject_wrong_palette,
    "section-swap": _inject_section_swap,
    "no-tint": _inject_no_tint,
    "shifted": _inject_shifted,
}
COARSE_INJECT = "mod-off"


@contextmanager
def _patched(patches):
    from descape import asset_source

    saved = [(owner, name, getattr(owner, name)) for owner, name, _new in patches]
    for owner, name, new in patches:
        setattr(owner, name, new)
    asset_source.clear_art_caches()
    try:
        yield
    finally:
        for owner, name, old in saved:
            setattr(owner, name, old)
        asset_source.clear_art_caches()


# ----------------------------------------------------------------- capture

_READY = False
_BOXES = None


def _setup() -> None:
    global _READY
    if _READY:
        return
    if not os.environ.get("AOE2DE_PROFILE_PATH"):
        raise SystemExit("set AOE2DE_PROFILE_PATH to the profile whose enabled mods should draw")
    if settings_isolation.pin_install_path() is None:
        raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
    settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="objects_pack_")))
    from descape import mod_graphics

    titles = [m.title for m in mod_graphics.mods_with_art()]
    for needed in ("Chunky Tree Tiles", "Tower range mod"):
        if not any(needed in t for t in titles):
            raise SystemExit(f"no enabled mod named like {needed!r} in this profile")
    _READY = True


def _render(scenario, style: str) -> np.ndarray:
    from descape import render

    if style == "flat":
        return render.overlay_units(render.render_terrain(scenario), scenario, with_sprites=True)
    if style == "stepped":
        return render.render_terrain_iso_with_proj(scenario, with_sprites=True)[0]
    return render.render_terrain_sloped_with_proj(scenario, with_sprites=True)[0]


def _full_images(inject: str | None, extra=()) -> dict[str, np.ndarray]:
    """Uncropped renders per frame id (f shares b's canvas)."""
    from descape import asset_source, mod_graphics

    _setup()
    trees, tower = _scenario(True, False), _scenario(False, True)
    patches = ([] if inject is None else INJECTS[inject]()) + list(extra)
    out = {}
    with _patched(patches):
        mod_graphics.set_enabled(True)
        asset_source.clear_art_caches()
        for fid, style in STYLES.items():
            out[fid] = _render(trees, style)
        out["e"] = _render(tower, "stepped")
        mod_graphics.set_enabled(False)
        asset_source.clear_art_caches()
        out["d"] = _render(trees, "stepped")
    out["f"] = out["b"]
    return out


FRAME_STYLES = {**STYLES, "d": "stepped", "e": "stepped"}


def _grounds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per style, the uncropped empty ground and the mask of its pale own cells."""
    empty, plain = _scenario(False, False), _scenario(False, False, pale=False)
    out = {}
    for style in set(FRAME_STYLES.values()):
        ground = _render(empty, style)
        out[style] = (ground, rp.changed_pixels(ground, _render(plain, style)))
    return out


def _boxes() -> dict[str, tuple[int, int, int, int]]:
    """Crop boxes shared by every run, so no frame's size names its run: per frame, the union
    over the clean run and every inject of the px that differ from empty ground, padded
    (b and d share one, the union of both)."""
    global _BOXES
    if _BOXES is None:
        from descape import asset_source, mod_graphics

        _setup()
        grounds = _grounds()
        masks = {fid: np.zeros(grounds[style][0].shape[:2], dtype=bool) for fid, style in FRAME_STYLES.items()}
        for inject in (None, *INJECTS):
            full = _full_images(inject)
            for fid, style in FRAME_STYLES.items():
                masks[fid] |= rp.changed_pixels(full[fid], grounds[style][0])
        masks["b"] = masks["d"] = masks["b"] | masks["d"]
        boxes = {fid: rp.feature_bbox(mask, PAD_PX, mask.shape) for fid, mask in masks.items()}
        empty = _scenario(False, False)
        x0, y0, x1, y1 = boxes["b"]
        from descape import render

        half_h = render.render_terrain_iso_with_proj(empty, with_units=False)[2].half_h
        boxes["f"] = (x0, y0 + 2 * GROUND_SHIFT_TILES * half_h, x1, y1 + 2 * GROUND_SHIFT_TILES * half_h)
        mod_graphics.set_enabled(False)
        asset_source.clear_art_caches()
        _BOXES = boxes
    return _BOXES


def _frame_images(inject: str | None, extra=()) -> dict[str, np.ndarray]:
    boxes = _boxes()
    full = _full_images(inject, extra)
    return {fid: rp.crop(img, boxes[fid]).copy() for fid, img in full.items()}


def _cropped_grounds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per frame a..e, the empty ground and its pale-cell mask, cropped like the frame."""
    boxes, grounds = _boxes(), _grounds()
    return {
        fid: tuple(rp.crop(layer, boxes[fid]) for layer in grounds[style])
        for fid, style in FRAME_STYLES.items()
    }


# --------------------------------------------------------------- measurement


def _foreign_px(img: np.ndarray) -> np.ndarray:
    """Px in hues the oak art never uses: red, near-white, blue, magenta or cyan (orange,
    yellow, green and brown are its own)."""
    r, g, b = (img[..., i].astype(int) for i in range(3))
    red = (r > 140) & (g < 90) & (b < 90)
    white = np.minimum(np.minimum(r, g), b) > 200
    blue = (b > r + 40) & (b > g + 20)
    magenta = (r > 140) & (b > 140) & (g < 100)
    cyan = (g > 140) & (b > 140) & (r < 100)
    return red | white | blue | magenta | cyan


def _measure(images: dict[str, np.ndarray]) -> dict[str, float]:
    """What the answers rest on, per frame: the share of the pale own cells the art covers
    (q2), foreign-hue px inside the art (q3/q4), saturated blue on the tower (q5)."""
    out = {}
    for fid, (ground, pale) in _cropped_grounds().items():
        art = rp.changed_pixels(images[fid], ground)
        out[f"{fid} own-cell cover"] = float((art & pale).sum() / pale.sum())
        out[f"{fid} foreign px"] = float((art & _foreign_px(images[fid])).sum())
    e = images["e"].astype(int)
    out["e blue px"] = float(((e[..., 2] > 100) & (e[..., 0] < 60) & (e[..., 1] < 60)).sum())  # band ~(0, 0, 140)
    return out


# Which frames each inject must change, and only those: d (Mod Art off) and f (bare ground) never.
EXPECTED_REACH = {
    "mod-off": {"a", "b", "c", "e"},
    "wrong-palette": {"a", "b", "c", "e"},
    "section-swap": {"a", "b", "c"},
    "no-tint": {"e"},
    "shifted": {"a", "b", "c", "e"},
}
BLUE_MIN_PX = 20  # clean e has 43; no-tint must leave none
SHIFT_COVER_RATIO = 0.6  # of clean's own-cell share; shifted measured 0.39 (a), 0.46 (b, c), 0.06 (e)
SPECKLE_RATIO = 3  # section-swap's foreign px in b over clean's; measured 12x (192 against 16)
SECTION_SWAP_FRACTION = 0.10  # of the frame


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("a", "A map area, top-down view."),
            ("b", "The same map area, angled view."),
            ("c", "The same map area, angled view, a second ground renderer."),
            ("d", "The same map area, angled view."),
            ("e", "A second map area, angled view."),
            ("f", "A third map area, angled view, same crop size and zoom as the second frame."),
        )
    )
    ground = (
        "The ground is a checkerboard of two alternating ground textures, one cell per map tile, and some "
        "cells are a third, pale (white) ground. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                ground + "First describe the objects in each frame: their height compared with one checker cell, "
                "their outline, and whether a single trunk is visible. Then, per frame: TALL if each object is "
                "one tree with a visible trunk whose crown rises well above its cell, LOW if each object is a "
                "low patch or clump of many small plants lying roughly within one cell, NONE if there are no "
                "objects."
            ),
            frames=("a", "b", "c", "d"),
            answers=("TALL", "LOW", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                ground + "First describe where each object's base or footprint sits relative to the pale cells: "
                "which cells it stands on, and whether it is centred in a pale cell, straddles a cell edge or "
                "stands on the checkered cells. Then, per frame: ON-CELL if every object stands within a pale "
                "cell, OFFSET if the objects stand beside the pale cells, on a cell edge or on the checkered "
                "cells, NONE if there are no objects."
            ),
            frames=("a", "b", "c"),
            answers=("ON-CELL", "OFFSET", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "First describe the overall colour of each surface of the object's artwork, ignoring the "
                "ground and ignoring single pixels and small dots: the foliage as a mass, bark, stone, roof, "
                "wood. Then, per frame: NATURAL if every surface's overall colour is plausible for what it "
                "depicts (foliage greens, yellows and oranges, bark browns, stone greys, wood browns), "
                "FALSE-COLOUR if a whole surface reads as a colour it should not be (green or purple stone, a "
                "foliage mass that reads dark red or brown-black rather than leafy)."
            ),
            frames=("b", "e"),
            answers=("NATURAL", "FALSE-COLOUR", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                "First describe, surface by surface, how the colour varies from pixel to pixel inside the "
                "object's artwork, ignoring the ground, and name any single pixels or small dots whose hue "
                "differs sharply from the pixels around them. Then, per frame: SMOOTH if every surface's pixels "
                "stay within its own colour family (greens, yellows and oranges in foliage, browns in bark, "
                "greys in stone), apart from leaf edges, shading and highlights, SPECKLED if dots of hues "
                "foreign to their surface (red, white, blue, magenta or cyan in foliage, say) are scattered "
                "through it. Count such dots even where they could pass for berries or flowers."
            ),
            frames=("b", "e"),
            answers=("SMOOTH", "SPECKLED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q5",
            question=(
                "First describe every colour on the building, including any band, trim or banner. Then: does any "
                "part of it carry a strong saturated blue (a team colour), as opposed to only stone, wood, foliage "
                "and neutral greys? BLUE-PRESENT or NO-BLUE."
            ),
            frames=("e",),
            answers=("BLUE-PRESENT", "NO-BLUE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q6",
            question=(
                "First describe everything drawn in this frame. Then: is there any object, plant, building or "
                "coloured mark in it, or only the checkered ground? OBJECTS-PRESENT or GROUND-ONLY."
            ),
            frames=("f",),
            answers=("OBJECTS-PRESENT", "GROUND-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object artwork review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen renders from a map editor for a strategy game. Each frame is a crop, "
            "enlarged by a whole-number factor with no smoothing. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
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
    """Spec coupling, frame band, determinism, each inject's reach per frame,
    and the measurements each run's answers rest on."""
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    clean = _frame_images(None)
    again = _frame_images(None)
    for frame in spec.frames:
        if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
            failures.append(f"frame {frame.id}: two clean captures differ")
        try:
            scaled, factor = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
            print(f"frame {frame.id}: {clean[frame.id].shape[1]}x{clean[frame.id].shape[0]} at {factor}x")
        except rp.PackError as exc:
            failures.append(str(exc))
    measured = {"clean": _measure(clean)}
    grounds = _cropped_grounds()
    reach, masks = {}, {}
    for name in INJECTS:
        injected = _frame_images(name)
        measured[name] = _measure(injected)
        touched = {}
        for frame in spec.frames:
            mask = masks[name, frame.id] = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if set(touched) != EXPECTED_REACH[name]:
            failures.append(f"inject {name}: reached {sorted(touched)}, expected {sorted(EXPECTED_REACH[name])}")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    for key in measured["clean"]:
        print(f"measure: {key}: " + ", ".join(f"{run} {m[key]:.3g}" for run, m in measured.items()))

    coarse = reach[COARSE_INJECT]
    for fid in EXPECTED_REACH["section-swap"]:
        # One tree's own art is the most a section-order fault can reach in one piece.
        one_tree = rp.largest_component_px(rp.changed_pixels(clean[fid], grounds[fid][0]))
        bound = rp.LocalizedBound(max_changed_fraction=SECTION_SWAP_FRACTION, max_component_px=one_tree)
        failures += bound.failures(masks["section-swap", fid], f"inject section-swap on {fid}")
        comp = reach["section-swap"].get(fid, (0, 0))[1]
        if comp >= coarse.get(fid, (0, 0))[1]:
            failures.append(f"inject section-swap: {fid} component {comp}px is not under the coarse inject's")
    for run, m in measured.items():
        if run == "no-tint":
            if m["e blue px"] > 0:
                failures.append(f"no-tint: {m['e blue px']:.0f} saturated blue px left on the tower")
        elif m["e blue px"] < BLUE_MIN_PX:
            failures.append(f"{run}: {m['e blue px']:.0f} saturated blue px on the tower, under {BLUE_MIN_PX}")
    for fid in ("a", "b", "c", "e"):
        key = f"{fid} own-cell cover"
        if measured["shifted"][key] > SHIFT_COVER_RATIO * measured["clean"][key]:
            failures.append(f"shifted: {key} {measured['shifted'][key]:.3f} is not under "
                            f"{SHIFT_COVER_RATIO} x clean's {measured['clean'][key]:.3f}")
    if measured["section-swap"]["b foreign px"] < SPECKLE_RATIO * max(1.0, measured["clean"]["b foreign px"]):
        failures.append(f"section-swap: b foreign px {measured['section-swap']['b foreign px']:.0f} not "
                        f"{SPECKLE_RATIO}x clean's {measured['clean']['b foreign px']:.0f}")
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
    if args.check:
        check()
        return
    for path in generate(args.out_dir, args.inject, args.replica):
        print(f"wrote {path}")
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
