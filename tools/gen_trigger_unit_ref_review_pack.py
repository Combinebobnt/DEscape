#!/usr/bin/env python3
"""Review pack for the trigger overlay's referenced-unit outline: does the
outline a Unit/Unit[] trigger field draws sit on the footprint of the unit it
names, in Stepped and Sloped, where that footprint crosses raised ground?

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md) and the
shape of tools/gen_trigger_overlay_review_pack.py: a real offscreen
ViewerWindow, scene rects rasterized 1:1 with testkit.qt_capture. The units
are real placed units, resolved through descape.unit_references and
trigger_geometry exactly as the window does, but hidden with Show GAIA off:
the resolver ignores filters by design, so the outline still draws, and the
footprint tiles are painted a third terrain by the scenario itself. The
reviewer compares the outline against ground the overlay code never touched.
Needs a configured AoE2:DE install (terrain textures).

Run:
    tools/gen_trigger_unit_ref_review_pack.py
    tools/gen_trigger_unit_ref_review_pack.py --replica 1
    tools/gen_trigger_unit_ref_review_pack.py --inject low-corner
    tools/gen_trigger_unit_ref_review_pack.py --check      # writes nothing

Writes build/review_pack/trigger_unit_ref/<opaque token>/, gitignored.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_review_pack as grp
import gen_seam_eyeball as seam_eyeball
import gen_trigger_overlay_review_pack as overlay_pack
import numpy as np

from testkit import review_pack as rp
from testkit import settings_isolation

PACK_ID = "trigger_unit_ref"
OUT_DIR = grp.OUT_DIR
CHECKER = overlay_pack.CHECKER
PATCH = overlay_pack.PATCH
PLATEAU = overlay_pack.PLATEAU  # x 55-66, y 44-67 at elevation 3
_GAIA = 0
# A 4x4 Castle whose footprint (x 53-56, y 50-53) straddles the plateau's
# west edge, anchored at tile + span/2 like every corpus building.
CASTLE = SimpleNamespace(const=82, ref=-1, x=55.0, y=52.0)
# A 1x1 Archer on flat ground, x != y so a swapped axis cannot land by luck.
ARCHER = SimpleNamespace(const=4, ref=-1, x=24.5, y=82.5)
_STYLES = {"stepped": "Stepped", "sloped": "Sloped"}


def _footprints():
    from descape.render import unit_tile_bounds

    out = {}
    for unit in (CASTLE, ARCHER):
        probe = SimpleNamespace(unit_const=unit.const, x=unit.x, y=unit.y)
        out[unit.const] = unit_tile_bounds(probe, 120, 120)  # (x0, x1, y0, y1), half-open
    return out


def _in_footprint(bounds, x: int, y: int) -> bool:
    x0, x1, y0, y1 = bounds
    return x0 <= x < x1 and y0 <= y < y1


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    footprints = _footprints().values()
    for tile in loaded.map_manager.terrain:
        tile.elevation = overlay_pack._elevation(tile.x, tile.y)
        patch = any(_in_footprint(b, tile.x, tile.y) for b in footprints)
        tile.terrain_id = PATCH if patch else CHECKER[(tile.x + tile.y) % 2]
    model = UnitEditModel(loaded)
    for unit in (CASTLE, ARCHER):
        unit.ref = model.add(_GAIA, unit.const, unit.x, unit.y).reference_id
    out = tmp / "unit_refs.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=model)
    return out


def _shapes(window):
    """The window's own path, resolved against the live scenario's reference
    index: a Destroy Object naming the castle and a Remove Object selecting the
    archer. Neither has a destination, so no run line crosses either frame."""
    from descape import library_compat, trigger_geometry, unit_references

    vocabulary = library_compat.load_vocabulary(window.scenario.scenario_version)
    index = unit_references.build_reference_index(window.scenario)
    shapes = []
    for kind, definitions, presentation, name, fields in (
        ("condition", vocabulary.conditions, vocabulary.condition_presentation, "destroy_object",
         {"unit_object": CASTLE.ref}),
        ("effect", vocabulary.effects, vocabulary.effect_presentation, "remove_object",
         {"selected_object_ids": [ARCHER.ref]}),
    ):
        definition = next(d for d in definitions.values() if d.name == name)
        shapes += trigger_geometry.shapes_for_entry(
            SimpleNamespace(**fields),
            definition,
            trigger_index=0,
            entry_kind=kind,
            entry_index=0,
            references=index,
            presentation=presentation,
        )
    assert {s.shape for s in shapes} == {trigger_geometry.SHAPE_UNIT}, shapes
    return shapes


# ----------------------------------------------------------------- injects


def _inject_low_corner():
    """The unit's own tile read as the footprint's low corner, not its middle:
    a 4x4 building's outline lands two tiles along +x and +y. A 1x1 unit's is
    unchanged, which is what makes the archer frame this inject's control."""
    from descape import trigger_geometry

    def low_corner(ref):
        x0, x1, y0, y1 = ref.bounds
        ax, ay = ref.own_tile
        return (ax, ay, ax + (x1 - x0) - 1, ay + (y1 - y0) - 1)

    return trigger_geometry, "_unit_corners", low_corner


def _inject_grow():
    """Every outline drawn two tiles larger on every side."""
    from descape import trigger_geometry

    def grown(ref):
        x0, x1, y0, y1 = ref.bounds
        return (x0 - 2, y0 - 2, x1 + 1, y1 + 1)

    return trigger_geometry, "_unit_corners", grown


def _inject_oversize():
    """The half-open far corner kept as inclusive: one tile too far on the far x and y sides."""
    from descape import trigger_geometry

    def oversize(ref):
        x0, x1, y0, y1 = ref.bounds
        return (x0, y0, x1, y1)

    return trigger_geometry, "_unit_corners", oversize


INJECTS = {"low-corner": _inject_low_corner, "grow": _inject_grow, "oversize": _inject_oversize}
COARSE_INJECT = "grow"
LOCALIZED_INJECT = "oversize"


@contextmanager
def _patched(inject: str | None):
    patches = [] if inject is None else [INJECTS[inject]()]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _new in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


# ----------------------------------------------------------------- capture


def _footprint_bounds(mv, bounds):
    """Scene bounds of the footprint's own tiles, from the overlay-independent tile polygons."""
    x0, x1, y0, y1 = bounds
    rect = None
    for tx, ty in ((x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)):
        tile = mv._tile_polygon(tx, ty).boundingRect()
        rect = tile if rect is None else rect.united(tile)
    return rect


def _capture(window, inject: str | None) -> dict[str, np.ndarray]:
    mv = window.map_view
    footprints = _footprints()
    out: dict[str, np.ndarray] = {}
    for style, label in _STYLES.items():
        window.terrain_style_combo.setCurrentText(label)
        with _patched(inject):
            shapes = _shapes(window)
        mv.show_trigger_overlay(shapes, None)
        frames = (("a", CASTLE),) if style == "sloped" else (("a", CASTLE), ("b", ARCHER))
        for suffix, unit in frames:
            b = _footprint_bounds(mv, footprints[unit.const])
            one_tile = mv._tile_polygon(footprints[unit.const][0], footprints[unit.const][2]).boundingRect().width()
            # Wide margins: the injects move or grow an outline by whole tiles.
            pad = max(b.width() * 0.6, 3 * one_tile)
            rect = overlay_pack._int_rect(b.left() - pad, b.top() - pad * 1.6, b.right() + pad, b.bottom() + pad)
            out[f"{style}_{suffix}"] = overlay_pack._grab(mv, rect)
    mv.clear_trigger_overlay()
    return out


_WINDOW = None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="trigger_unit_ref_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
        _WINDOW.load_scenario(_scenario_file(tmp))
        if _WINDOW.scenario is None:
            raise SystemExit("the generated unit scenario failed to load")
        # Hides both units: the footprint patch, not a unit box, is what the outline is judged against.
        _WINDOW.show_gaia_action.setChecked(False)
    return _WINDOW


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    return _capture(_window(), inject)


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("stepped_a", "A square patch of pale ground on a checkered map, seen at an angle; some ground is raised."),
            ("sloped_a", "The same patch in a second map view style with smooth slopes."),
            ("stepped_b", "A single pale tile on a checkered map, seen at an angle."),
        )
    )
    ground = (
        "The ground is a checkerboard of two alternating ground textures, one cell per map tile, "
        "except for a patch of pale ground. A green line is drawn over the map. "
    )
    exclusion = (
        "Ignore the line's own thickness. A departure means the green line leaves the patch's edge "
        "by a visible gap, runs through the checkered ground outside the patch, cuts across the pale "
        "patch itself, encloses a different area, or is missing. "
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                ground + "In each frame, first describe where the green line runs relative to the pale "
                "patch's edge along each of the patch's four sides, including where the edge crosses "
                "raised ground. Then: does the green line run along the pale patch's edge on all four "
                "sides in EVERY frame? " + exclusion + "FOLLOWS if it does everywhere, DEPARTS if it "
                "leaves the edge anywhere."
            ),
            frames=("stepped_a", "sloped_a"),
            answers=("FOLLOWS", "DEPARTS", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                ground + "Here the patch is a single tile. First describe where the green line runs "
                "relative to that tile's four edges. Then: does the green line run along the pale "
                "tile's edge on all four sides? " + exclusion + "FOLLOWS or DEPARTS."
            ),
            frames=("stepped_b",),
            answers=("FOLLOWS", "DEPARTS", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map outline review",
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
    """Spec coupling, frame band, determinism, and each inject's reach per frame."""
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
            scaled, _f = rp.frame_to_band(clean[frame.id])
            rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
        except rp.PackError as exc:
            failures.append(str(exc))
    reach = {}
    for name in INJECTS:
        injected = _frame_images(name)
        touched = {}
        for frame in spec.frames:
            mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
            n = int(np.count_nonzero(mask))
            if n:
                touched[frame.id] = (n, rp.largest_component_px(mask))
        if not touched:
            failures.append(f"inject {name}: changed no pixels")
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    if "stepped_b" in reach["low-corner"]:
        failures.append("inject low-corner reached stepped_b, the 1x1 control")
    failures.extend(
        f"inject low-corner did not reach {fid}" for fid in ("stepped_a", "sloped_a") if fid not in reach["low-corner"]
    )
    coarse, local = reach[COARSE_INJECT], reach[LOCALIZED_INJECT]
    shared = [fid for fid in local if fid in coarse]
    if not shared:
        failures.append(f"{LOCALIZED_INJECT} shares no frame with {COARSE_INJECT}")
    failures.extend(
        f"{LOCALIZED_INJECT} is not smaller than {COARSE_INJECT} on {fid}"
        for fid in shared
        if local[fid][1] >= coarse[fid][1]
    )
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
