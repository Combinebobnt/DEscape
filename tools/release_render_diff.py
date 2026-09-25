#!/usr/bin/env python3
"""Pixel-diffs this checkout's map render against a release tag's (GH #87:
"everything looks like v0.6").

Renders each scenario at both commits, off-engine and full-canvas, in Flat,
Stepped and Sloped with sprites off and on, and reports every pixel that
differs outside the changes the releases since then made on purpose. The
release's `descape/` is extracted with `git archive` into a scratch directory
and rendered by the same Python in its own subprocess, with the install and
the Isometric elevation height pinned through a throwaway `config.yaml`, so
neither side reads your real settings.

What is neutralized or masked, and why. Everything else must match exactly.

- **Sub-tile unit positions** (v0.7: "Units render at the position their file
  actually stores"). This checkout renders with the pre-0.7 placement put
  back: `render.unit_paint_offset()` returns (0, 0) and a span-1 sprite axis
  centres on its tile again. The Sloped rise is still sampled at the stored
  sub-tile point, as it was in v0.6. Only this checkout is patched: v0.6
  painted from `int(unit.x)`, which the patch does not change.
- **Stepped contact shadows** (v0.7: shadows scale with step height, no tick
  marks, no break at inside corners). Masked: the screen bounds, over their
  3x3 neighbourhood's height range, of every tile with a neighbour at a
  different height. `iso_geometry.tile_screen_bounds_over` already includes
  the shadow passes' upward reach.
- **Sprite art changed on purpose**, sprites-on runs only. These units are
  removed from the scenario on both sides before rendering:
  walls and gates (shapes derived from their neighbours, palisade and sea
  wall banners and bases, GH #51), heroes (hero glow, GH #39), invisible
  objects (editor markers, GH #53), and buildings drawn from per-piece depth
  slots (town centres, pastures; GH #66). Their look is judged by their own
  review packs (tools/REVIEW_PACK.md); `--frames-dir` writes v0.6 / HEAD /
  amplified-diff crops of them for a side-by-side look.

Sprites-off runs keep every unit. Each Stepped sprites-on run also asserts that
both sides resolve the same number of units to sprite art, and at least one, so
a missing install cannot pass as "no difference".

Run:
    tools/release_render_diff.py                         # the quick corpus in examples/
    tools/release_render_diff.py examples/foo.aoe2scenario --elev-step-pct 25 200
    tools/release_render_diff.py --frames-dir build/v06_frames
    tools/release_render_diff.py --inject repaint-tile   # a control: must exit 1

Needs git, the release tag, and an AoE2:DE install (--install,
AOE2DE_INSTALL_PATH, or the configured one). Exits 1 on any unmasked
difference. The default composite backend is numpy on both sides, since v0.6
has no native kernel; `--composite native` renders this checkout through it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE_REF = "v0.6"
STYLES = ("flat", "stepped", "sloped")
DEFAULT_PCTS = (50, 200)
# The same four files as tests/conftest.py's QUICK_CORPUS_NAMES.
QUICK_CORPUS = (
    "0_June_Event_Scenario.aoe2scenario",
    "2_Joan_coop_1_v0_13.aoe2scenario",
    "C2_ElCid_coop_1_v0_16.aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
)
# Controls: each must make the comparison fail. The last three drop one neutralization each, proving it is load-bearing.
INJECTS = ("repaint-tile", "no-placement-patch", "no-exclusions", "no-mask")
STRIP_ROWS = 1024
# Review frames: crop CROP_W x CROP_H canvas px, upscaled nearest-neighbour to 1200x900 (REVIEW_PACK framing rules).
CROP_W, CROP_H, UPSCALE = 240, 180, 5
DIFF_GAIN = 4


# ---------------------------------------------------------------------------
# Worker: runs in a fresh interpreter whose descape/ is the tree being rendered.


def _legacy_placement(render) -> None:
    """Put the pre-0.7 placement back: marks at the tile centre, span-1 sprite axes centred on their tile."""
    for name in ("unit_paint_offset", "_sprite_axis_centre", "_span_start"):
        if not hasattr(render, name):
            raise SystemExit(f"render.{name} is gone: update _legacy_placement() to match the refactor")
    render.unit_paint_offset = lambda unit: (0.0, 0.0)
    render._sprite_axis_centre = lambda coord, span: render._span_start(coord, span) + span / 2


def _terrace_mask(iso_geometry, elevations: np.ndarray, proj, shape: tuple[int, int]) -> np.ndarray:
    h, w = elevations.shape
    pad = np.pad(elevations, 1, mode="edge")
    around = np.stack([pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    lo, hi = around.min(axis=0), around.max(axis=0)
    ys, xs = np.nonzero(lo != hi)
    x0, y0, x1, y1 = iso_geometry.tile_screen_bounds_over(xs, ys, lo[ys, xs], hi[ys, xs], proj)
    mask = np.zeros(shape, dtype=bool)
    canvas_h, canvas_w = shape
    for top, bottom, left, right in zip(y0.tolist(), y1.tolist(), x0.tolist(), x1.tolist(), strict=True):
        mask[max(top, 0) : min(bottom, canvas_h), max(left, 0) : min(right, canvas_w)] = True
    return mask


def _repaint_one_flat_tile(scenario) -> None:
    """The regression control: retexture one unit-free tile whose 3x3 neighbourhood is flat, so no mask covers it."""
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    tiles = {(t.x, t.y): t for t in mm.terrain}
    occupied = {(int(u.x), int(u.y)) for units in scenario.unit_manager.units for u in units}
    for y in range(h // 2, h - 1):
        for x in range(w // 2, w - 1):
            around = [tiles[(x + dx, y + dy)] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            if len({t.elevation for t in around}) == 1 and not occupied & {(t.x, t.y) for t in around}:
                tile = tiles[(x, y)]
                tile.terrain_id = 0 if tile.terrain_id != 0 else 1
                return
    raise SystemExit("no flat, unit-free tile to repaint")


def _worker(spec: dict) -> None:
    sys.path.insert(0, spec["root"])
    from descape import iso_geometry, render
    from descape.scenario_io import load_map_and_units

    scenario = load_map_and_units(Path(spec["scenario"]))
    if spec.get("repaint_tile"):
        _repaint_one_flat_tile(scenario)
    excluded = frozenset(spec["exclude"])
    removed = 0
    for units in scenario.unit_manager.units:
        kept = [u for u in units if u.unit_const not in excluded]
        removed += len(units) - len(kept)
        units[:] = kept
    if spec["legacy_placement"]:
        _legacy_placement(render)
    style, sprites = spec["style"], spec["sprites"]
    meta: dict = {"removed_units": removed}
    if style == "flat":
        img = render.overlay_units(render.render_terrain(scenario), scenario, with_sprites=sprites)
    elif style == "stepped":
        img, elevations, proj = render.render_terrain_iso_with_proj(scenario, with_units=True, with_sprites=sprites)
        if spec.get("mask_out"):
            np.save(spec["mask_out"], _terrace_mask(iso_geometry, elevations, proj, img.shape[:2]))
        if sprites:
            meta["sprite_units"] = len(render.sprite_draws_by_anchor(scenario, proj, elevations).skip_ids)
    else:
        img = render.render_terrain_sloped_with_proj(scenario, with_units=True, with_sprites=sprites)[0]
    np.save(spec["out"], img)
    Path(spec["meta_out"]).write_text(json.dumps(meta))


# ---------------------------------------------------------------------------
# Driver.


@dataclass(frozen=True)
class Job:
    scenario: Path
    style: str
    sprites: bool
    pct: int
    full: bool = False  # a review-frames render: nothing excluded

    @property
    def key(self) -> str:
        return f"{self.scenario.stem}__{self.style}_{'sprites' if self.sprites else 'marks'}_{self.pct}{'_full' if self.full else ''}"


@dataclass
class PairResult:
    job: Job
    diff_px: int = 0
    unmasked_px: int = 0
    mask_frac: float = 0.0
    removed_units: int = 0
    sprite_units: tuple[int, int] | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def line(self) -> str:
        j = self.job
        tag = "ok  " if self.ok else "FAIL"
        mask = f" masked {self.mask_frac:.1%}" if j.style == "stepped" else ""
        sprites = f" sprite units {self.sprite_units[0]}/{self.sprite_units[1]}" if self.sprite_units else ""
        removed = f" removed {self.removed_units}" if self.removed_units else ""
        detail = "; ".join(self.problems)
        return (
            f"{tag} {j.scenario.name} {j.style:7} {'sprites' if j.sprites else 'marks  '} pct {j.pct:3}: "
            f"diff {self.diff_px} px, unmasked {self.unmasked_px}{mask}{removed}{sprites}{' -- ' + detail if detail else ''}"
        )


def intended_change_consts() -> dict[str, frozenset[int]]:
    """The consts whose sprite art changed on purpose since v0.6, by reason."""
    from descape import render, unit_kind
    from descape.terrain_palette import HERO_GLOW_CONSTS

    graphics = json.loads((ROOT / "descape" / "unit_graphic_map.json").read_text())["graphics"]
    slotted = frozenset(int(c) for c, e in graphics.items() if any("slot" in p for p in e.get("pieces", ())))
    return {
        "walls and gates (neighbour shapes, banners, sea wall base)": unit_kind.wall_consts(),
        "heroes (hero glow)": frozenset(HERO_GLOW_CONSTS),
        "invisible objects (editor markers)": unit_kind.invisible_consts(),
        "depth-slotted buildings (town centres, pastures)": slotted | render.DRAPED_SPRITE_CONSTS,
    }


def extract_base(ref: str, dest: Path) -> Path:
    """`git archive <ref> descape` into dest/; raises CalledProcessError if the ref is unknown."""
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", ref, "descape"], check=True, capture_output=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar, check=True)
    return dest


def _write_config(config_home: Path, install: Path, pct: int) -> None:
    app_dir = config_home / "DEscape"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "config.yaml").write_text(f"aoe2de_install: {json.dumps(str(install))}\nelev_step_pct: {pct}\n")


def _launch(spec: dict, config_home: Path, install: Path, composite: str) -> None:
    env = dict(os.environ, XDG_CONFIG_HOME=str(config_home), AOE2DE_INSTALL_PATH=str(install), DESCAPE_COMPOSITE=composite)
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)],
        env=env, capture_output=True, text=True, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"render failed for {spec['out']}:\n{proc.stderr[-4000:]}")


def _compare(a_path: Path, b_path: Path, mask_path: Path | None) -> tuple[int, int, float, np.ndarray | None]:
    """(diff px, diff px outside the mask, masked fraction, the diff outside the mask), strip-wise to fit a 480x480 map."""
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    if a.shape != b.shape:
        raise ValueError(f"canvas shape {a.shape} != {b.shape}")
    mask = np.load(mask_path, mmap_mode="r") if mask_path else None
    diff = np.zeros(a.shape[:2], dtype=bool)
    for top in range(0, a.shape[0], STRIP_ROWS):
        rows = slice(top, top + STRIP_ROWS)
        diff[rows] = (a[rows] != b[rows]).any(axis=-1)
    total = int(diff.sum())
    if mask is None:
        return total, total, 0.0, diff
    outside = diff & ~mask
    return total, int(outside.sum()), float(mask.mean()), outside


def _write_frames(result_job: Job, a_path: Path, b_path: Path, diff: np.ndarray, out_dir: Path, per_config: int) -> list[str]:
    """Crops the densest clusters of `diff` (already outside the contact-shadow mask) as a/b/amplified-diff PNGs."""
    from PIL import Image

    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    h, w = diff.shape
    rows, cols = h // CROP_H, w // CROP_W
    counts = diff[: rows * CROP_H, : cols * CROP_W].reshape(rows, CROP_H, cols, CROP_W).sum(axis=(1, 3))
    lines = []
    for rank, flat in enumerate(np.argsort(counts, axis=None)[::-1][:per_config]):
        r, c = divmod(int(flat), cols)
        if counts[r, c] == 0:
            break
        ys, xs = np.nonzero(diff[r * CROP_H : (r + 1) * CROP_H, c * CROP_W : (c + 1) * CROP_W])
        cy, cx = r * CROP_H + int(ys.mean()), c * CROP_W + int(xs.mean())
        top, left = min(max(cy - CROP_H // 2, 0), h - CROP_H), min(max(cx - CROP_W // 2, 0), w - CROP_W)
        crop = (slice(top, top + CROP_H), slice(left, left + CROP_W))
        stem = f"{result_job.key}_{rank:02d}"
        amplified = np.clip(np.abs(a[crop].astype(np.int16) - b[crop].astype(np.int16)).max(axis=-1) * DIFF_GAIN, 0, 255)
        for suffix, pixels in (("a_base", a[crop]), ("b_head", b[crop]), ("c_diff", amplified.astype(np.uint8))):
            img = Image.fromarray(np.ascontiguousarray(pixels))
            img.resize((CROP_W * UPSCALE, CROP_H * UPSCALE), Image.NEAREST).save(out_dir / f"{stem}_{suffix}.png")
        lines.append(f"| {stem} | canvas x {left}..{left + CROP_W}, y {top}..{top + CROP_H} | {int(counts[r, c])} |")
    return lines


def _pair_result(job: Job, out: Path, diff_px: int, unmasked: int, frac: float) -> PairResult:
    meta_a = json.loads((out / f"{job.key}_a.json").read_text())
    meta_b = json.loads((out / f"{job.key}_b.json").read_text())
    result = PairResult(job, diff_px, unmasked, frac, meta_b["removed_units"])
    if unmasked:
        result.problems.append(f"{unmasked} px differ outside the intended changes")
    if meta_a["removed_units"] != meta_b["removed_units"]:
        result.problems.append(f"removed {meta_a['removed_units']} vs {meta_b['removed_units']} units")
    if "sprite_units" in meta_b:
        result.sprite_units = (meta_a["sprite_units"], meta_b["sprite_units"])
        if 0 in result.sprite_units:
            result.problems.append("a side resolved no sprite art: is the install visible?")
        elif result.sprite_units[0] != result.sprite_units[1]:
            result.problems.append("the two sides resolve a different number of units to sprite art")
    return result


def run(
    scenarios: list[Path],
    work_dir: Path,
    install: Path,
    *,
    base_ref: str = BASE_REF,
    pcts: tuple[int, ...] = DEFAULT_PCTS,
    styles: tuple[str, ...] = STYLES,
    composite: str = "numpy",
    jobs: int = 4,
    frames_dir: Path | None = None,
    frames_per_config: int = 4,
    inject: str | None = None,
    keep_renders: bool = False,
    log=print,
) -> list[PairResult]:
    """Renders and compares every (scenario, style, sprites, pct); returns one PairResult per comparison.

    Each pair is compared as soon as both sides finish and its .npy canvases are then deleted
    (unless keep_renders), so disk peaks at about `jobs` pairs rather than the whole run.
    """
    if inject is not None and inject not in INJECTS:
        raise ValueError(f"inject must be one of {INJECTS}, got {inject!r}")
    base_root = extract_base(base_ref, work_dir / "base")
    groups = intended_change_consts()
    exclude = sorted(set().union(*groups.values()))
    out = work_dir / "renders"
    out.mkdir(parents=True, exist_ok=True)
    homes = {pct: work_dir / f"config_{pct}" for pct in pcts}
    for pct, home in homes.items():
        _write_config(home, install, pct)

    todo: list[Job] = []
    for scenario in scenarios:
        for pct in pcts:
            # Flat never reads the elevation step, so it runs at the first pct only.
            for style in styles if pct == pcts[0] else tuple(s for s in styles if s != "flat"):
                for sprites in (False, True):
                    todo.append(Job(scenario, style, sprites, pct))
                    if sprites and frames_dir is not None:
                        todo.append(Job(scenario, style, sprites, pct, full=True))

    def render_pair(job: Job) -> None:
        excluded = exclude if job.sprites and not job.full and inject != "no-exclusions" else []
        for side, root in (("a", base_root), ("b", ROOT)):
            spec = {
                "root": str(root),
                "scenario": str(job.scenario),
                "style": job.style,
                "sprites": job.sprites,
                "exclude": excluded,
                "legacy_placement": side == "b" and inject != "no-placement-patch",
                "repaint_tile": side == "b" and inject == "repaint-tile",
                "out": str(out / f"{job.key}_{side}.npy"),
                "meta_out": str(out / f"{job.key}_{side}.json"),
                "mask_out": str(out / f"{job.key}_mask.npy") if side == "b" and job.style == "stepped" else None,
            }
            _launch(spec, homes[job.pct], install, "numpy" if side == "a" else composite)

    def finish(job: Job) -> tuple[PairResult | None, list[str]]:
        a_path, b_path = out / f"{job.key}_a.npy", out / f"{job.key}_b.npy"
        mask_path = out / f"{job.key}_mask.npy" if job.style == "stepped" and inject != "no-mask" else None
        diff_px, unmasked, frac, diff = _compare(a_path, b_path, mask_path)
        frames: list[str] = []
        if job.full and frames_dir is not None and unmasked:
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames = _write_frames(job, a_path, b_path, diff, frames_dir, frames_per_config)
        if not keep_renders:
            # By name, not mask_path: the no-mask control still writes the mask.
            for side in ("a", "b", "mask"):
                (out / f"{job.key}_{side}.npy").unlink(missing_ok=True)
        if job.full:
            return None, frames
        return _pair_result(job, out, diff_px, unmasked, frac), frames

    # Compared serially on this thread as each pair lands: one full-canvas diff in memory at a time.
    finished: dict[Job, tuple[PairResult | None, list[str]]] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(render_pair, job): job for job in todo}
        for future in as_completed(futures):
            future.result()
            finished[futures[future]] = finish(futures[future])

    results: list[PairResult] = []
    index: list[str] = []
    for job in todo:
        result, frames = finished[job]
        index += frames
        if result is not None:
            results.append(result)
            log(result.line())
    if frames_dir is not None and index:
        header = [
            f"# {base_ref} vs HEAD: the intended sprite-art changes, side by side",
            "",
            f"Each frame is a/b/c: the base release, this checkout, and |a-b| x{DIFF_GAIN}.",
            f"Crops are {CROP_W}x{CROP_H} canvas px, upscaled x{UPSCALE}.",
            "Units at sub-tile positions are drawn at their tile centre on both sides. Excluded here, compared in full:",
            *(f"- {reason}" for reason in groups),
            "",
            "| frame | where | diff px in crop |",
            "|---|---|---|",
        ]
        (frames_dir / "INDEX.md").write_text("\n".join(header + index) + "\n")
    return results


def _default_install() -> Path | None:
    from descape import asset_source

    return asset_source.get_install_path()


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]))
        return 0
    # Only the driver imports this checkout's descape; a worker must see its own tree's.
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenarios", type=Path, nargs="*", help="Scenario files (default: the quick corpus in examples/)")
    parser.add_argument("--base", default=BASE_REF, help=f"Release ref to compare against (default {BASE_REF})")
    parser.add_argument("--elev-step-pct", type=int, nargs="+", default=list(DEFAULT_PCTS), help="Isometric elevation height stops")
    parser.add_argument("--style", choices=STYLES, nargs="+", default=list(STYLES))
    parser.add_argument("--install", type=Path, default=None, help="AoE2:DE install (default: AOE2DE_INSTALL_PATH or the configured one)")
    parser.add_argument("--composite", choices=("numpy", "native"), default="numpy", help="This checkout's composite backend")
    parser.add_argument("--jobs", type=int, default=4, help="Concurrent render pairs (a 480x480 Flat canvas is ~700 MB)")
    parser.add_argument("--work-dir", type=Path, default=None, help="Work here (default: a temporary directory)")
    parser.add_argument("--keep-renders", action="store_true", help="Keep each pair's .npy canvases (default: deleted once compared)")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Also write side-by-side crops of the excluded units")
    parser.add_argument("--frames-per-config", type=int, default=4)
    parser.add_argument("--inject", choices=INJECTS, default=None, help="A control run that must fail")
    args = parser.parse_args()

    scenarios = args.scenarios or [ROOT / "examples" / name for name in QUICK_CORPUS]
    missing = [str(p) for p in scenarios if not p.is_file()]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    install = args.install or _default_install()
    if install is None:
        parser.error("no AoE2:DE install: pass --install or set AOE2DE_INSTALL_PATH")

    def go(work_dir: Path) -> list[PairResult]:
        return run(
            scenarios, work_dir, install, base_ref=args.base, pcts=tuple(args.elev_step_pct), styles=tuple(args.style),
            composite=args.composite, jobs=args.jobs, frames_dir=args.frames_dir, frames_per_config=args.frames_per_config,
            inject=args.inject, keep_renders=args.keep_renders,
        )

    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        results = go(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="release_render_diff_") as tmp:
            results = go(Path(tmp))
    failed = [r for r in results if not r.ok]
    print(f"{len(results) - len(failed)}/{len(results)} comparisons match {args.base} outside the intended changes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
