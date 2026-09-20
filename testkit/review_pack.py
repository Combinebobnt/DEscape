"""Declarative review packs: the Qt-free half of the agent visual review.

A pack is a PackSpec -- a set of Frames (one rendered PNG each) and a set of
Checks (one question each) -- plus the framing rules that decide whether a
rendered image is judgeable at all. Nothing here renders or imports Qt, so
the default test tier can exercise the whole validation surface cheaply;
tools/gen_review_pack.py supplies the rendering.

The framing rules are measured, not preference. Reading real captures out of
build/ found a 144x112 seam thumbnail literally unjudgeable, a 1317x844
full-window grab useless for render detail (the map is a ~20px smudge), and a
576x448 amplified A/B diff the most productive class of all. Hence: long edge
900-1500 px, integer nearest-neighbour upscale only (a smooth resample turns a
1px line into a gradient, and this project's mip selection is
exactness-sensitive -- see testkit/qt_capture.py), never a downscale, and one
feature per frame.

The coupling rules exist because the failure this whole mechanism targets is
a half-checked render reading as fully checked: a Check naming no Frame, or a
Frame no Check names, is a validation failure rather than a silent pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Below ~600 px on the long edge there is nothing to judge; above ~1500 the
# vision pipeline downsamples and the detail is lost anyway.
MIN_LONG_EDGE = 900
MAX_LONG_EDGE = 1500

# What kind of image answers a question. A seam/contour question answered off
# a texture-dominated raw crop is the mistake this vocabulary prevents.
FRAME_KINDS = ("raw", "control", "diff")

VERDICTS = ("PASS", "FAIL", "CANNOT-TELL")


class PackError(ValueError):
    """A pack that cannot be reviewed as declared."""


@dataclass(frozen=True)
class Frame:
    """One rendered PNG in a pack.

    `upscale` is recorded rather than inferred so a reviewer reading the
    manifest knows a 1px feature is being shown 4px wide. `kind` is what the
    frame IS, not what it is for: "raw" is the shipped render, "control" is
    the same render with the feature neutralised, "diff" is the amplified
    difference of the two.
    """

    id: str
    filename: str
    kind: str
    upscale: int = 1
    caption: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FRAME_KINDS:
            raise PackError(f"frame {self.id!r}: kind must be one of {FRAME_KINDS}, got {self.kind!r}")
        if self.upscale < 1:
            raise PackError(f"frame {self.id!r}: upscale must be >= 1, got {self.upscale}")


@dataclass(frozen=True)
class Check:
    """One question, answered per frame id.

    `question` must be phrased without naming the artifact it hunts -- ask the
    open question before the closed one, or the reviewer agrees with whichever
    phrasing it was handed. `answers`, when set, is a forced choice; it is the
    right shape for a symptom-disambiguation question, where PASS/FAIL would
    itself presuppose which symptom is present.
    """

    id: str
    question: str
    frames: tuple[str, ...]
    answers: tuple[str, ...] = VERDICTS

    def __post_init__(self) -> None:
        if not self.frames:
            raise PackError(f"check {self.id!r}: names no frame")
        if len(self.answers) < 2:
            raise PackError(f"check {self.id!r}: needs at least two allowed answers")


@dataclass(frozen=True)
class PackSpec:
    id: str
    title: str
    frames: tuple[Frame, ...]
    checks: tuple[Check, ...]
    preamble: str = ""
    injects: tuple[str, ...] = field(default_factory=tuple)

    def frame(self, frame_id: str) -> Frame:
        for f in self.frames:
            if f.id == frame_id:
                return f
        raise PackError(f"no frame {frame_id!r} in pack {self.id!r}")


def validate_spec(spec: PackSpec) -> None:
    """Every check names a frame that exists; every frame is named by a check.

    The second half is the load-bearing one: an unnamed frame is a property
    nobody wrote down before looking, which is exactly how "verified visually"
    came to cover a render only half of which was ever checked.
    """
    frame_ids = [f.id for f in spec.frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise PackError(f"pack {spec.id!r}: duplicate frame ids")
    check_ids = [c.id for c in spec.checks]
    if len(set(check_ids)) != len(check_ids):
        raise PackError(f"pack {spec.id!r}: duplicate check ids")

    named: set[str] = set()
    for check in spec.checks:
        for frame_id in check.frames:
            if frame_id not in frame_ids:
                raise PackError(f"check {check.id!r} names frame {frame_id!r}, which the pack does not declare")
            named.add(frame_id)
    orphans = sorted(set(frame_ids) - named)
    if orphans:
        raise PackError(f"pack {spec.id!r}: frames no check asks about: {', '.join(orphans)}")


def nn_upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Integer nearest-neighbour upscale. Never a smooth resample: this
    project's renders are exactness-sensitive, and interpolation would turn
    the 1px features under review into gradients."""
    if factor < 1:
        raise PackError(f"upscale factor must be >= 1, got {factor}")
    if factor == 1:
        return img
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def fit_upscale_factor(width: int, height: int) -> int:
    """The largest integer factor keeping the long edge inside the band.

    Raises rather than downscaling an oversized render: build/iso_preview/
    holds 12800x6512 full-map renders, and fitting one to 1500 makes every
    defect in this pack disappear. An oversized frame is a cropping bug in
    the pack, and must be fixed there.
    """
    long_edge = max(width, height)
    if long_edge <= 0:
        raise PackError("empty frame")
    if long_edge > MAX_LONG_EDGE:
        raise PackError(f"frame long edge {long_edge} exceeds {MAX_LONG_EDGE}; crop it, never downscale")
    factor = MAX_LONG_EDGE // long_edge
    if factor * long_edge < MIN_LONG_EDGE:
        raise PackError(
            f"frame long edge {long_edge} cannot reach {MIN_LONG_EDGE} at any integer factor "
            f"without passing {MAX_LONG_EDGE}; widen or tighten the crop"
        )
    return factor


def validate_frame_size(width: int, height: int, frame_id: str = "") -> None:
    long_edge = max(width, height)
    if not (MIN_LONG_EDGE <= long_edge <= MAX_LONG_EDGE):
        where = f"frame {frame_id!r}: " if frame_id else ""
        raise PackError(f"{where}long edge {long_edge} outside the judgeable band {MIN_LONG_EDGE}-{MAX_LONG_EDGE}")


def frame_to_band(img: np.ndarray) -> tuple[np.ndarray, int]:
    """(upscaled image, factor used). The one entry point a pack should use,
    so no frame reaches a reviewer at a size nothing could be judged at."""
    height, width = img.shape[0], img.shape[1]
    factor = fit_upscale_factor(width, height)
    out = nn_upscale(img, factor)
    validate_frame_size(out.shape[1], out.shape[0])
    return out, factor


def changed_pixels(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise PackError(f"shape mismatch {a.shape} vs {b.shape}")
    return np.any(a != b, axis=-1)


def delta_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def feature_bbox(mask: np.ndarray, pad_px: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """The changed region plus a margin, clipped to the frame.

    "The feature must own the frame" is a rule a hand-picked tile bbox keeps
    failing at: a pyramid's terraces sit in the top half of their own tile
    extent, so cropping to the tiles leaves most of the image empty ground
    and shrinks the feature to a fraction of what the reviewer is looking at.
    Cropping to where the feature actually landed fixes that without anyone
    having to re-tune a crop by eye each time a fixture changes.
    """
    box = delta_bbox(mask)
    if box is None:
        raise PackError("nothing changed: cannot crop to a feature that is not there")
    height, width = shape
    x0, y0, x1, y1 = box
    return (
        max(0, x0 - pad_px),
        max(0, y0 - pad_px),
        min(width, x1 + pad_px),
        min(height, y1 + pad_px),
    )


def crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1]


def components(mask: np.ndarray) -> int:
    """8-connected component count, iterative flood fill (no scipy here).
    Same routine as tools/gen_contact_shadow_eyeball.py's."""
    return _flood(mask)[0]


def largest_component_px(mask: np.ndarray) -> int:
    return _flood(mask)[1]


def _flood(mask: np.ndarray) -> tuple[int, int]:
    seen = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    count, largest = 0, 0
    for y0, x0 in zip(*np.nonzero(mask), strict=True):
        if seen[y0, x0]:
            continue
        count += 1
        seen[y0, x0] = True
        stack = [(int(y0), int(x0))]
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        largest = max(largest, size)
    return count, largest


@dataclass(frozen=True)
class LocalizedBound:
    """The bound a LOCALIZED inject must stay inside.

    A negative control is only worth running if the two injects differ in
    size: a reviewer that catches "the band is gone" has proved nothing about
    whether it resolves a 2px tick. Without this assertion a localized inject
    can drift coarse and the sensitivity control silently becomes a second
    sanity control -- which would be this whole mechanism's own headline
    failure, one property checked and silently generalized, recurring inside
    its own validation.

    Bounded on the largest connected delta rather than on a bounding box: a
    defect that repeats at every tile apex along a run spans the whole frame
    by bbox while each individual mark stays 2px, so a bbox ceiling cannot
    tell the two injects apart here.
    """

    max_changed_fraction: float
    max_component_px: int

    def failures(self, mask: np.ndarray, where: str = "") -> list[str]:
        prefix = f"{where}: " if where else ""
        changed = int(np.count_nonzero(mask))
        if changed == 0:
            return [f"{prefix}inject changed no pixels at all"]
        out = []
        fraction = changed / mask.size
        if fraction > self.max_changed_fraction:
            out.append(
                f"{prefix}changed {fraction:.3%} of the frame, over the "
                f"{self.max_changed_fraction:.1%} ceiling -- no longer localized"
            )
        largest = largest_component_px(mask)
        if largest > self.max_component_px:
            out.append(
                f"{prefix}largest connected delta is {largest} px, over the "
                f"{self.max_component_px} px ceiling -- no longer localized"
            )
        return out
