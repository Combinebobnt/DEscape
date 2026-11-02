#!/usr/bin/env python3
"""Review pack for TASK-550: a Display Instructions box drawn over a real, busy map.

Same protocol as tools/gen_review_pack.py (see tools/REVIEW_PACK.md), kept as its
own script like tools/gen_doodad_review_pack.py, because the box is a child widget
of MapView: it never reaches scene.render(), so this drives a real offscreen
ViewerWindow and crops map_view.grab(). gen_review_pack.py itself stays Qt-free.
Needs examples/old-allies-final-v2.aoe2scenario and a configured AoE2:DE install.

Each frame shows that map's own effects, selected in the real trigger panel at
1:1 zoom, with nothing written back: one effect per panel slot, each with a
leading colour tag and a portrait, over a forest, a wall with cavalry and an
autumn wood. The window is 1800x1080, a viewport of 1440x850, so the box is the
394 px wide it would be at about that size on screen.

Frames:
- view_1: the whole map view, the TOP effect in place.
- panel_1, panel_2, panel_3: a crop around each slot's box, fixed from the clean
  capture, so a box moved off its place leaves the crop rather than dragging it.
- panel_4: panel_1's crop with View > Show Instruction on Map off. The specificity
  frame: no box is possible there.

Injects, every one a patch of descape.instruction_overlay's painting except
plain-text:
- box-offset (coarse): the box 340 px right and 300 px down, out of every panel crop.
- clipped-text (localized): the BOTTOM box lays its text out without wrapping, so
  the first line stops at the text column's edge through a letter and the rest
  of the message is gone.
- see-through: the box's fill gone, its outline and text left over the bare map.
- icon-slip: the MIDDLE box's portrait moved out through its frame.
- plain-text: no colour tag reaches the text.
- broken-outline: a stretch of the top and bottom outline left out.

The PACK_ID goes into the opaque token only. Frame and check names are neutral on
purpose, and the directory REVIEW.md's paths run through is OUT_SUBDIR: nothing a
reviewer reads may name the subject.

Run:
    tools/gen_instruction_box_review_pack.py --out-dir <neutral dir>
    tools/gen_instruction_box_review_pack.py --out-dir <neutral dir> --replica 1
    tools/gen_instruction_box_review_pack.py --out-dir <neutral dir> --inject clipped-text
    tools/gen_instruction_box_review_pack.py --check      # writes nothing

Writes <out-dir>/views/<opaque token>/ (build/review_pack/ by default, gitignored).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
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

PACK_ID = "instruction_box"
OUT_SUBDIR = "views"
OUT_DIR = grp.OUT_DIR
MAP_PATH = ROOT / "examples" / "old-allies-final-v2.aoe2scenario"
WINDOW_SIZE = (1800, 1080)
# frame id -> (trigger, effect, its slot, the scene px centred, the tag's colour name).
_SHOTS = {
    "panel_1": (58, 0, 0, (9400, 5600), "AQUA"),
    "panel_2": (70, 0, 1, (9500, 6200), "ORANGE"),
    "panel_3": (25, 1, 2, (10200, 6390), "PURPLE"),
}
_PANELS = tuple(_SHOTS)
_FULL_VIEW = "view_1"
_SPECIFICITY = "panel_4"
_CROP_PAD_PX = 26
OFFSET_PX = (340, 300)
CLIPPED_SLOT, SLIPPED_SLOT = 2, 1
ICON_SLIP = (70, 12)  # canvas units; little downward, so the portrait stays clear of the bottom outline
# Canvas units along the box: where broken-outline leaves the line out, top and bottom.
GAP_X0, GAP_X1 = 350, 650
MIN_BUSY_STD = 25.0

_WINDOW = None
_CROPS: dict[str, tuple[int, int, int, int]] = {}


# ----------------------------------------------------------------- injects


def _overlay_module():
    from descape import instruction_overlay

    return instruction_overlay


def _inject_box_offset():
    """Coarse: the box placed off its slot, after the real placement."""
    over = _overlay_module().InstructionOverlay
    real = over._fit

    def moved(self) -> None:
        real(self)
        self.move(self.x() + OFFSET_PX[0], self.y() + OFFSET_PX[1])

    return [(over, "_fit", moved)]


def _inject_clipped_text():
    """Localized: the BOTTOM box's document laid out three times too wide; _compose() still clips to the column."""
    io = _overlay_module()
    plain = io.QTextDocument

    class Unwrapped(plain):
        def setTextWidth(self, width: float) -> None:
            super().setTextWidth(width * 3)

    real = io.InstructionOverlay._compose

    def compose(self) -> None:
        io.QTextDocument = Unwrapped if self._view.slot == CLIPPED_SLOT else plain
        real(self)

    return [(io, "QTextDocument", plain), (io.InstructionOverlay, "_compose", compose)]


def _inject_see_through():
    """The fill gone from inside the outline, so the text sits on the map itself."""
    from PyQt5.QtCore import QRectF, Qt
    from PyQt5.QtGui import QPainter

    io = _overlay_module()
    real = io.InstructionOverlay.__dict__["_paint_art"]

    def art(painter, height: int) -> None:
        real.__func__(painter, height)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        painter.fillRect(QRectF(0, 0, io.PANEL_WIDTH, height).adjusted(4, 4, -4, -4), Qt.transparent)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    return [(io.InstructionOverlay, "_paint_art", staticmethod(art))]


def _inject_icon_slip():
    """The MIDDLE box's portrait drawn ICON_SLIP past its place, its frame left where it was."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor

    io = _overlay_module()
    real = io.InstructionOverlay.__dict__["_paint_icon"]

    def icon(painter, view) -> None:
        if view.slot != SLIPPED_SLOT:
            real.__func__(painter, view)
            return
        left, top, width, height = io.ICON_RECT
        pixmap = view.icon
        painter.drawPixmap(
            left + (width - pixmap.width()) // 2 + ICON_SLIP[0], top + (height - pixmap.height()) // 2 + ICON_SLIP[1], pixmap
        )
        painter.fillPath(io._ring(QRectF(left, top, width, height), io.ICON_FRAME), QColor(*view.tint))

    return [(io.InstructionOverlay, "_paint_icon", staticmethod(icon))]


def _inject_plain_text():
    """No colour tag reaches the text: every run drawn in the default colour."""
    from descape import message_markup

    real = message_markup.to_html

    def to_html(runs, default):
        return real([run._replace(color=None) for run in runs], default)

    return [(message_markup, "to_html", to_html)]


def _inject_broken_outline():
    """GAP_X0..GAP_X1 of the top and bottom outline repainted as plain fill."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QPainter

    io = _overlay_module()
    real = io.InstructionOverlay.__dict__["_paint_art"]

    def art(painter, height: int) -> None:
        real.__func__(painter, height)
        painter.setCompositionMode(QPainter.CompositionMode_Source)
        fill = QColor(*io.ART_FILL, io.ART_ALPHA)
        for y in (0, height - 4):
            painter.fillRect(QRectF(GAP_X0, y, GAP_X1 - GAP_X0, 4), fill)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

    return [(io.InstructionOverlay, "_paint_art", staticmethod(art))]


INJECTS = {
    "box-offset": _inject_box_offset,
    "clipped-text": _inject_clipped_text,
    "see-through": _inject_see_through,
    "icon-slip": _inject_icon_slip,
    "plain-text": _inject_plain_text,
    "broken-outline": _inject_broken_outline,
}
COARSE_INJECT = "box-offset"
REACH = {
    "box-offset": {*_PANELS, _FULL_VIEW},
    "clipped-text": {"panel_3"},
    "see-through": {*_PANELS, _FULL_VIEW},
    "icon-slip": {"panel_2"},
    "plain-text": {*_PANELS, _FULL_VIEW},
    "broken-outline": {*_PANELS, _FULL_VIEW},
}
# Measured 2026-10-31: 1200 px of panel_3's 446x115 (2.3%), 177 in its largest component.
CLIPPED_BOUND = rp.LocalizedBound(max_changed_fraction=0.04, max_component_px=300)


@contextmanager
def _patched(patches):
    saved = [(owner, name, owner.__dict__[name]) for owner, name, _new in patches]
    for owner, name, new in patches:
        setattr(owner, name, new)
    try:
        yield
    finally:
        for owner, name, old in saved:
            setattr(owner, name, old)


# ----------------------------------------------------------------- capture


def prerequisites() -> str | None:
    from descape import asset_source

    if not MAP_PATH.is_file():
        return f"{MAP_PATH.name} is missing from examples/"
    if asset_source.get_install_path() is None:
        return "no AoE2:DE install visible (set AOE2DE_INSTALL_PATH)"
    return None


def _window():
    global _WINDOW
    if _WINDOW is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        if settings_isolation.pin_install_path() is None:
            raise SystemExit("needs a configured AoE2:DE install: sprites and terrain textures")
        settings_isolation.isolate_settings(Path(tempfile.mkdtemp(prefix="views_pack_")))
        from testkit.qt_window import ensure_qapp

        ensure_qapp()
        from PyQt5.QtWidgets import QApplication

        from descape.viewer import ViewerWindow

        _WINDOW = ViewerWindow()
        _WINDOW.resize(*WINDOW_SIZE)
        _WINDOW.show()
        QApplication.processEvents()
        _WINDOW.load_scenario(MAP_PATH)
        if _WINDOW.scenario is None:
            raise SystemExit(f"{MAP_PATH.name} failed to load")
        _WINDOW.mode_combo.setCurrentText("Triggers")
        QApplication.processEvents()
        one_to_one = 1.0 / _WINDOW.map_view.transform().m11()
        _WINDOW.map_view.scale(one_to_one, one_to_one)
    return _WINDOW


def close_window() -> None:
    """Drops the window with nothing to save. A test calls this; a tool run simply exits."""
    global _WINDOW
    if _WINDOW is not None:
        _WINDOW.edit_history.mark_saved()
        _WINDOW.close()
        _WINDOW = None
        _CROPS.clear()


def _grab() -> np.ndarray:
    from PyQt5.QtGui import QImage

    from testkit.qt_capture import qimage_rgb888_to_array

    return qimage_rgb888_to_array(_window().map_view.grab().toImage().convertToFormat(QImage.Format_RGB888))


def _settled_grab() -> np.ndarray:
    """The view once three grabs in a row agree, so a chunk still painting cannot land in a frame."""
    from PyQt5.QtWidgets import QApplication

    previous, agreeing = None, 0
    for _attempt in range(200):
        QApplication.processEvents()
        current = _grab()
        agreeing = agreeing + 1 if previous is not None and np.array_equal(previous, current) else 0
        if agreeing == 2:
            return current
        previous = current
        time.sleep(0.02)
    raise SystemExit("the map view never settled")


def _show(frame_id: str) -> np.ndarray:
    """The whole view with `frame_id`'s effect selected and its spot centred."""
    from PyQt5.QtWidgets import QApplication

    window = _window()
    trigger, effect, _slot, centre, _tag = _SHOTS[frame_id]
    panel = window.trigger_panel
    panel.select_trigger(trigger)
    panel.refresh_entries(select=("effect", effect))
    QApplication.processEvents()
    window.map_view.centerOn(*centre)
    return _settled_grab()


def _crops() -> dict[str, tuple[int, int, int, int]]:
    """Each panel frame's crop in view px, from the clean box, so every run crops the same view px."""
    if not _CROPS:
        overlay = _window().map_view.instruction_overlay()
        for frame_id in _PANELS:
            _show(frame_id)
            box = overlay.geometry()
            pad = _CROP_PAD_PX
            _CROPS[frame_id] = (box.left() - pad, box.top() - pad, box.right() + 1 + pad, box.bottom() + 1 + pad)
    return _CROPS


def _bare(frame_id: str) -> np.ndarray:
    """The view at `frame_id`'s spot with the toggle off."""
    action = _window().instruction_overlay_action
    action.setChecked(False)
    try:
        return _show(frame_id)
    finally:
        action.setChecked(True)


def _frame_images(inject: str | None) -> dict[str, np.ndarray]:
    crops = _crops()
    images = {}
    with _patched([] if inject is None else INJECTS[inject]()):
        for frame_id in _PANELS:
            view = _show(frame_id)
            if frame_id == "panel_1":
                images[_FULL_VIEW] = view
            images[frame_id] = rp.crop(view, crops[frame_id])
        images[_SPECIFICITY] = rp.crop(_bare("panel_1"), crops["panel_1"])
    return images


# --------------------------------------------------------------- measurement


def _pixmap_rgba() -> np.ndarray:
    """The box as composed, (h, w, 4), straight RGBA, for the slot last selected."""
    from PyQt5.QtGui import QImage

    image = _window().map_view.instruction_overlay()._pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    bits = image.constBits()
    bits.setsize(image.bytesPerLine() * image.height())
    return np.array(bits).reshape(image.height(), image.bytesPerLine() // 4, 4).copy()


def _measure(frame_id: str) -> dict[str, float]:
    """What a box's answers rest on, off its own composed pixmap with `frame_id` selected. Text is
    opaque over the art's alpha, so the text's ink is the opaque px right of the portrait's column."""
    io = _overlay_module()
    _show(frame_id)
    rgba = _pixmap_rgba()
    scale = _window().map_view.instruction_overlay()._scale
    frame_right = round((io.ICON_RECT[0] + io.ICON_RECT[2]) * scale) + 1
    text_x0 = round(io.TEXT_RECT[0] * scale) - 1
    opaque = rgba[..., 3] > 220
    text = opaque.copy()
    text[:, :text_x0] = False
    rows = np.nonzero(text.any(axis=1))[0]
    cols = np.nonzero(text.any(axis=0))[0]
    return {
        "box_w": rgba.shape[1], "box_h": rgba.shape[0], "text_top": int(rows[0]), "text_bottom": int(rows[-1]) + 1,
        "text_left": int(cols[0]), "text_right": int(cols[-1]) + 1, "column_right": (io.TEXT_RECT[0] + io.TEXT_RECT[2]) * scale,
        "lines": len(np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1)),
        "gap_ink": int(opaque[:, frame_right:text_x0].sum()),  # the empty strip between portrait frame and text
        "fill_alpha": int(rgba[rgba.shape[0] // 2, round(990 * scale), 3]),  # the margin right of the text column
        "colour": tuple(float(v) for v in rgba[text][:, :3].mean(axis=0)),
    }


def _busy(frame_id: str) -> tuple[float, float]:
    """(std of the bare map's luma under the box, mean luma there)."""
    view = _bare(frame_id)
    x0, y0, x1, y1 = _crops()[frame_id]
    pad = _CROP_PAD_PX
    luma = view[y0 + pad : y1 - pad, x0 + pad : x1 - pad].astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    return float(luma.std()), float(luma.mean())


# --------------------------------------------------------------- the spec


def _spec() -> rp.PackSpec:
    panels = ", ".join(_PANELS)
    frames = (
        rp.Frame(_FULL_VIEW, f"{_FULL_VIEW}.png", "raw", caption="The whole map view."),
        *(rp.Frame(fid, f"{fid}.png", "raw", caption="A crop of the map view.") for fid in _PANELS),
        rp.Frame(_SPECIFICITY, f"{_SPECIFICITY}.png", "control", caption="A crop of the map view."),
    )
    worst = " If the frames differ, answer for the worst one. Answer NONE if a frame shows no such box."
    checks = (
        rp.Check(
            id="words",
            question=(
                f"Frames {panels} should each show a dark rectangular message box lying over a busy map. First, for "
                "each frame, write out the box's text word for word as you read it, marking any word you cannot make "
                "out with [?], and say what lies behind the text.\n\n"
                "Then the closed question. READABLE if every word can be read without effort, its letters standing "
                "clearly apart from what is behind them. STRUGGLE if the letters of at least one word merge into or "
                "compete with the map detail behind them, so that the word has to be worked out. Not counted: small "
                "type alone, and words missing because a line stops short, which the next check asks about." + worst
            ),
            frames=_PANELS,
            answers=("READABLE", "STRUGGLE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="text_edges",
            question=(
                f"Frames {panels} again. First, for each box, say where its lines of text start and stop against the "
                "box, and how its last word ends.\n\n"
                "Then the closed question. FITS if all the text lies inside the box and the message ends on a whole "
                "word, with every letter complete. CUT if any letter is sliced through, or a line stops part way "
                "through a word with no continuation on the next line. Not counted: text wrapping normally on to a "
                "further line, and an empty space left in the box under the last line." + worst
            ),
            frames=_PANELS,
            answers=("FITS", "CUT", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="outline",
            question=(
                f"Frames {panels} again. A thin pale line runs around the edge of each box. First, for each box, "
                "describe that line along each of the four sides.\n\n"
                "Then the closed question. WHOLE if the line runs unbroken along all four sides of every box. BROKEN "
                "if any side has a stretch where the line is missing, so that the box's edge has no line along it "
                "there. Not counted: the line looking fainter or brighter where the map behind it changes, and the "
                "box's corners being square." + worst
            ),
            frames=_PANELS,
            answers=("WHOLE", "BROKEN", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="portrait",
            question=(
                f"Frames {panels} again. Each box has a small square picture at its left end, with a coloured square "
                "frame drawn around it. First, for each box, describe where the picture sits against its coloured "
                "frame.\n\n"
                "Then the closed question. INSIDE if in every box the picture fills its coloured frame and stays "
                "within it on all sides. SPILLS if in any box part of the picture sticks out past the frame, or a "
                "bare strip shows inside the frame because the picture has slid. Not counted: the frame's colour, and "
                "dark areas that are part of the picture itself." + worst
            ),
            frames=_PANELS,
            answers=("INSIDE", "SPILLS", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="text_colour",
            question=(
                f"Frames {panels} again. First, name the colour of the text in each box.\n\n"
                "Then the closed question. COLOURED if the text in every box is a clear colour rather than white or "
                "pale grey. PLAIN if the text in at least one box is plain white or pale grey. Not counted: the "
                "colour of the portrait's frame." + worst
            ),
            frames=_PANELS,
            answers=("COLOURED", "PLAIN", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="box_place",
            question=(
                f"Frame {_FULL_VIEW} is the whole map view. First, describe where the dark message box sits in the "
                "frame, if there is one.\n\n"
                "Then the closed question. TOP-CENTRE if the box is centred left to right and lies in the top quarter "
                "of the frame. ELSEWHERE if it sits anywhere else. NONE if there is no such box."
            ),
            frames=(_FULL_VIEW,),
            answers=("TOP-CENTRE", "ELSEWHERE", "NONE", "CANNOT-TELL"),
        ),
        rp.Check(
            id="panel_contents",
            question=(
                f"First, in your own words, describe everything visible in frame {_SPECIFICITY}.\n\n"
                "Then the closed question. BOX-PRESENT if a dark rectangle, a panel, a line of text or a small framed "
                "picture is drawn over the map. MAP-ONLY if the frame shows only the map itself: ground, trees, "
                "cliffs, buildings and units."
            ),
            frames=(_SPECIFICITY,),
            answers=("BOX-PRESENT", "MAP-ONLY", "CANNOT-TELL"),
        ),
    )
    return rp.PackSpec(
        id=PACK_ID,
        title="Map view review",
        frames=frames,
        checks=checks,
        preamble=(
            "These are offscreen captures from a map editor for an isometric strategy game. Each frame is the map view or "
            "a crop of it, enlarged by a whole-number factor with no smoothing, so one original pixel appears as a square "
            "block of pixels. Judge only what you can actually see."
        ),
        injects=tuple(INJECTS),
    )


# ---------------------------------------------------------------- generate


def generate(out_dir: Path, inject: str | None, replica: int = 0) -> list[Path]:
    if reason := prerequisites():
        raise SystemExit(f"{PACK_ID}: {reason}")
    spec = _spec()
    rp.validate_spec(spec)
    pack_dir = out_dir / OUT_SUBDIR / grp._run_token(PACK_ID, inject, replica)
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


def _check_measures(failures: list[str]) -> None:
    """Each check's discriminator, off the composed boxes: text inside its column and the box, clean;
    clipped-text reaching the column's edge in one line; the slipped portrait past its frame; the see-
    through fill gone; each tag's colour in the text, and none of it under plain-text."""
    from descape import message_markup

    colours = message_markup.text_colors()
    clean = {}
    for frame_id in _PANELS:
        got = clean[frame_id] = _measure(frame_id)
        std, mean = _busy(frame_id)
        print(
            f"measure: {frame_id} box {got['box_w']}x{got['box_h']} text x {got['text_left']}..{got['text_right']} (column "
            f"ends {got['column_right']:.0f}) y {got['text_top']}..{got['text_bottom']}, {got['lines']} line(s), fill alpha "
            f"{got['fill_alpha']}, map under it luma std {std:.0f} mean {mean:.0f}"
        )
        if got["text_right"] > got["column_right"] + 1 or got["text_bottom"] > got["box_h"]:
            failures.append(f"{frame_id}: clean text leaves its column or box")
        if std < MIN_BUSY_STD:
            failures.append(f"{frame_id}: the map under the box is not busy, luma std {std:.0f}")
        tag = colours[_SHOTS[frame_id][4]]
        distance = max(abs(a - b) for a, b in zip(got["colour"], tag, strict=True))
        print(f"measure: {frame_id} text colour {[round(c) for c in got['colour']]} against its tag {tag}")
        if distance > 40:
            failures.append(f"{frame_id}: text colour is {distance:.0f} from its tag's")
    with _patched(_inject_clipped_text()):
        got = _measure("panel_3")
    print(f"measure: clipped-text panel_3 text ends {got['text_right']}, column {got['column_right']:.0f}, {got['lines']} line(s)")
    if got["text_right"] < got["column_right"] - 1 or got["lines"] >= clean["panel_3"]["lines"]:
        failures.append("clipped-text no longer cuts a line at the column's edge")
    with _patched(_inject_icon_slip()):
        got = _measure("panel_2")
    print(f"measure: icon-slip panel_2 opaque px past the portrait's frame {got['gap_ink']}, clean {clean['panel_2']['gap_ink']}")
    if clean["panel_2"]["gap_ink"] or not got["gap_ink"]:
        failures.append("icon-slip no longer separates: the portrait past its frame")
    with _patched(_inject_see_through()):
        got = _measure("panel_1")
    print(f"measure: see-through panel_1 fill alpha {got['fill_alpha']}")
    if got["fill_alpha"] != 0:
        failures.append("see-through leaves a fill")
    with _patched(_inject_plain_text()):
        for frame_id in _PANELS:
            got = _measure(frame_id)
            spread = max(got["colour"]) - min(got["colour"])
            print(f"measure: plain-text {frame_id} text colour {[round(c) for c in got['colour']]}")
            if spread > 30:
                failures.append(f"plain-text leaves a tint on {frame_id}: {got['colour']}")


def check() -> None:
    """Spec coupling, reviewer text, frame band, determinism, each inject reaching exactly its
    REACH frames and never the specificity frame, the localized one bounded and under the coarse one,
    then the discriminators (_check_measures()). Closes its window, so a test can call it."""
    if reason := prerequisites():
        raise SystemExit(f"{PACK_ID}: {reason}")
    failures: list[str] = []
    spec = _spec()
    try:
        rp.validate_spec(spec)
    except rp.PackError as exc:
        failures.append(f"spec: {exc}")
    failures += [f"leak: {leak}" for leak in rp.leaks(spec)]
    try:
        clean = _frame_images(None)
        again = _frame_images(None)
        for frame in spec.frames:
            if rp.changed_pixels(clean[frame.id], again[frame.id]).any():
                failures.append(f"frame {frame.id}: two clean captures differ")
            try:
                scaled, factor = rp.frame_to_band(clean[frame.id])
                rp.validate_frame_size(scaled.shape[1], scaled.shape[0], frame.id)
                print(f"frame: {frame.id} {clean[frame.id].shape[1]}x{clean[frame.id].shape[0]} x{factor}")
            except rp.PackError as exc:
                failures.append(str(exc))
        reach = {}
        for name in INJECTS:
            injected = _frame_images(name)
            touched = {}
            for frame in spec.frames:
                mask = rp.changed_pixels(clean[frame.id], injected[frame.id])
                if mask.any():
                    touched[frame.id] = (int(np.count_nonzero(mask)), rp.largest_component_px(mask))
                    if name == "clipped-text":
                        failures += CLIPPED_BOUND.failures(mask, f"inject {name} on {frame.id}")
            reach[name] = touched
            shown = ", ".join(f"{fid} {px}px/{comp}max" for fid, (px, comp) in sorted(touched.items()))
            print(f"reach: {name} -> {shown or 'nothing'}")
            if set(touched) != REACH[name]:
                failures.append(f"inject {name} reaches {sorted(touched)}, not {sorted(REACH[name])}")
        coarse = reach[COARSE_INJECT]
        for name, touched in reach.items():
            if name == COARSE_INJECT:
                continue
            for frame_id, (_px, comp) in sorted(touched.items()):
                if comp >= coarse.get(frame_id, (0, 0))[1]:
                    failures.append(f"inject {name} on {frame_id}: largest delta {comp} px is not under {COARSE_INJECT}'s")
        _check_measures(failures)
    finally:
        close_window()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(f"{len(failures)} check(s) failed")
    print(
        f"OK: {PACK_ID} frames couple to its checks and land in {rp.MIN_LONG_EDGE}-{rp.MAX_LONG_EDGE} px, its text leaks "
        f"nothing, clean captures agree, every inject reaches exactly its frames and none reaches {_SPECIFICITY}, "
        f"clipped-text stays localized, and each check's discriminator measures as registered"
    )


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
    close_window()
    print(f"\nrun `{args.inject or 'clean'}` replica {args.replica} is directory "
          f"{grp._run_token(PACK_ID, args.inject, args.replica)}")


if __name__ == "__main__":
    main()
