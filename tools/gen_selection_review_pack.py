#!/usr/bin/env python3
"""Review pack for the Units-mode selection coloured by owner: each owner's
outline in its player colour over a dark under-stroke, GAIA and the marquee in
the configured colour.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as
its own script like tools/gen_marker_review_pack.py. Unlike those, the feature
is a set of MapView scene items rather than chunk pixels, so this one drives a
real offscreen ViewerWindow and rasterizes scene rects 1:1 with
testkit.qt_capture. Needs a configured AoE2:DE install (sprites and terrain).

Run:
    tools/gen_selection_review_pack.py
    tools/gen_selection_review_pack.py --replica 1
    tools/gen_selection_review_pack.py --inject no-under
    tools/gen_selection_review_pack.py --check      # writes nothing

Writes build/review_pack/selection/<opaque token>/, gitignored.
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

PACK_ID = "selection"
OUT_DIR = grp.OUT_DIR
HOUSE = 70
GOLD_MINE = 66
# One screen-horizontal row (x - y constant): P2 house, P3 house, GAIA gold mine.
_ROW = [(2, HOUSE, 57.0, 57.0), (3, HOUSE, 61.0, 61.0), (0, GOLD_MINE, 64.5, 64.5)]
_STYLES = {"stepped": "Stepped", "flat": "Flat"}
_ROW_FRAME = {"stepped": "row_a", "flat": "row_b"}


def _scenario_file(tmp: Path) -> Path:
    from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units
    from descape.scenario_write import write_scenario
    from descape.unit_model import UnitEditModel

    loaded = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    model = UnitEditModel(loaded)
    for player, const, x, y in _ROW:
        model.add(player, const, x, y)
    out = tmp / "row.aoe2scenario"
    write_scenario(loaded, out, units=model, backup=False)
    return out


# ----------------------------------------------------------------- injects


def _inject_no_owner():
    from descape.map_view import MapView

    return MapView, "_selection_group_key", lambda self, player_id: None


def _inject_no_under():
    from descape.map_view import MapView

    return MapView, "UNIT_SELECT_UNDERSTROKE_RGBA", (10, 10, 10, 0)


def _inject_swap():
    """P2 and P3 draw in each other's colour."""
    from descape.map_view import MapView

    real = MapView._selection_group_key

    def swapped(self, player_id):
        return real(self, {2: 3, 3: 2}.get(player_id, player_id))

    return MapView, "_selection_group_key", swapped


def _inject_gaia_owner():
    """GAIA takes its own resolved player colour instead of the configured one."""
    from descape.map_view import MapView

    real = MapView._selection_group_key

    def gaia(self, player_id):
        if player_id == 0 and self._selection_player_colors is not None:
            return tuple(self._selection_player_colors[0])
        return real(self, player_id)

    return MapView, "_selection_group_key", gaia


INJECTS = {
    "no-owner": _inject_no_owner,
    "no-under": _inject_no_under,
    "swap": _inject_swap,
    "gaia-owner": _inject_gaia_owner,
}
COARSE_INJECT = "no-owner"


@contextmanager
def _patched(inject: str | None):
    patches = [] if inject is None else [INJECTS[inject]()]
    saved = [(obj, name, obj.__dict__[name]) for obj, name, _new in patches]
    for obj, name, new in patches:
        setattr(obj, name, new)
    try:
        yield
    finally:
        for obj, name, old in saved:
            setattr(obj, name, old)


# ----------------------------------------------------------------- capture


def _int_rect(x0: float, y0: float, x1: float, y1: float):
    from PyQt5.QtCore import QRectF

    left, top = int(np.floor(x0)), int(np.floor(y0))
    return QRectF(left, top, int(np.ceil(x1)) - left, int(np.ceil(y1)) - top)


def _grab(mv, rect) -> np.ndarray:
    from PyQt5.QtWidgets import QApplication

    from testkit.qt_capture import scene_rect_to_array

    QApplication.processEvents()
    return scene_rect_to_array(mv.scene(), rect)


def _entries_by_player(mv) -> dict[int, object]:
    return {e.player_id: e for e in mv._unit_index.entries}


def _select_row(window) -> dict[int, object]:
    """Selects all three in Units mode; returns the entry per player."""
    mv = window.map_view
    window.mode_combo.setCurrentText("Units")
    entries = _entries_by_player(mv)
    window._selection = [(p, entries[p].unit.reference_id) for p, *_rest in _ROW]
    mv.set_unit_selection([entries[p] for p, *_rest in _ROW])
    return entries


def _path_bounds(mv, entry):
    return mv._unit_path(entry).boundingRect()


def _capture(window, inject: str | None) -> dict[str, np.ndarray]:
    from PyQt5.QtCore import QPointF

    mv = window.map_view
    out: dict[str, np.ndarray] = {}
    with _patched(inject):
        for style, label in _STYLES.items():
            window.mode_combo.setCurrentText("View")
            window.terrain_style_combo.setCurrentText(label)
            entries = _select_row(window)
            row = None
            for p, *_rest in _ROW:
                b = _path_bounds(mv, entries[p])
                row = b if row is None else row.united(b)
            # Headroom above for the house sprites, which stand above their footprints.
            head = row.height() * (1.4 if style == "stepped" else 0.3)
            pad = row.height() * 0.3
            row_rect = _int_rect(row.left() - pad, row.top() - head, row.right() + pad, row.bottom() + pad)
            out[_ROW_FRAME[style]] = _grab(mv, row_rect)
            if style != "stepped":
                continue
            for fid, player in (("close_a", 2), ("close_b", 0)):
                b = _path_bounds(mv, entries[player])
                # The footprint's lower half: both front edges and the near corner.
                m = b.height() * 0.15
                out[fid] = _grab(mv, _int_rect(b.left() - m, b.center().y(), b.right() + m, b.bottom() + m))
            mv.set_unit_selection(None)
            window._selection = []
            out["row_c"] = _grab(mv, row_rect)
            # The marquee alone over bare ground, well clear of the row.
            clear = row_rect.translated(0, row_rect.height() * 2)
            mv.centerOn(clear.center())
            c = clear.center()
            a = mv.mapFromScene(c - QPointF(row.width() * 0.3, row.height() * 0.3))
            mv._update_marquee(a, mv.mapFromScene(c + QPointF(row.width() * 0.3, row.height() * 0.3)))
            box = mv._marquee_item.rect()
            out["rect"] = _grab(mv, _int_rect(box.left() - 20, box.top() - 20, box.right() + 20, box.bottom() + 20))
            mv._clear_marquee()
    window.mode_combo.setCurrentText("View")
    return out


_WINDOW = None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        tmp = Path(tempfile.mkdtemp(prefix="selection_pack_"))
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(tmp)
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(1400, 1000)
        _WINDOW.show()
        _WINDOW.load_scenario(_scenario_file(tmp))
        if _WINDOW.scenario is None:
            raise SystemExit("the generated row scenario failed to load")
    return _WINDOW


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    return _capture(_window(), inject)


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    frames = tuple(
        rp.Frame(fid, f"{fid}.png", "raw", caption=caption)
        for fid, caption in (
            ("row_a", "Three objects in a row on grass, seen at an angle."),
            ("row_b", "The same three objects in a second map view style."),
            ("close_a", "Close-up of the lower part of the left-hand object's ground outline, angled view."),
            ("close_b", "Close-up of the lower part of the right-hand object's ground outline, angled view."),
            ("rect", "A rectangle drawn over bare grass."),
            ("row_c", "The three objects in a row on grass, angled view, a second render."),
        )
    )
    checks = (
        rp.Check(
            id="q1",
            question=(
                "Three objects stand in a row: left, middle and right. In these frames each one has a "
                "diamond-shaped outline drawn on the ground around its base. First, for each object, name "
                "the colour of its ground outline, and the colour of the object's own coloured markings "
                "(flags, banners, cloth or roof accents on a building). Then, for the LEFT and MIDDLE objects only: "
                "is each one's outline the same hue as that object's own markings? MATCH if both are, "
                "MISMATCH if either outline is clearly a different hue from its own object's markings. "
                "Ignore the right-hand object for this answer, and ignore brightness or transparency "
                "differences within one hue."
            ),
            frames=("row_a", "row_b"),
            answers=("MATCH", "MISMATCH", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q2",
            question=(
                "First describe the colour of the ground outline around the RIGHT-hand object of the row in "
                "the angled frame, and the colour of the rectangle's outline in the other frame. Then: are "
                "those two outline colours the same hue? SAME or DIFFERENT. Judge the thin outline lines "
                "themselves, not the translucent fill inside them."
            ),
            frames=("row_a", "rect"),
            answers=("SAME", "DIFFERENT", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q3",
            question=(
                "Each close-up shows part of a coloured outline drawn over the ground and part of an object. "
                "First describe, in each frame, what lies immediately on each side of the coloured line along "
                "its length. Then: in EVERY frame, is the coloured line bordered, on at least one side, by a "
                "thin near-black line that follows it continuously along its whole visible length? Exclude "
                "dark pixels that belong to the object's own art, its shadow or the ground texture: those are "
                "irregular and do not run parallel to the coloured line. BORDERED if every frame's line is, "
                "UNBORDERED if any frame's line is not."
            ),
            frames=("close_a", "close_b"),
            answers=("BORDERED", "UNBORDERED", "CANNOT-TELL"),
        ),
        rp.Check(
            id="q4",
            question=(
                "First describe everything drawn in this frame. Then: is there any straight-edged coloured "
                "outline or line drawn on the ground around any object, that is not part of the object's own "
                "art or the grass texture? LINES or NONE."
            ),
            frames=("row_c",),
            answers=("LINES", "NONE", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map object outline review",
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
        failures.extend(f"inject {name}: reached the {c} frame" for c in ("row_c", "rect") if c in touched)
        reach[name] = touched
        shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
        print(f"reach: {name} -> {shown or 'nothing'}")
    coarse = reach[COARSE_INJECT]
    shared = [fid for fid in reach["no-under"] if fid in coarse]
    if not shared:
        failures.append(f"no-under shares no frame with {COARSE_INJECT}")
    failures.extend(
        f"no-under is not smaller than {COARSE_INJECT} on {fid}"
        for fid in shared
        if reach["no-under"][fid][1] >= coarse[fid][1]
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
