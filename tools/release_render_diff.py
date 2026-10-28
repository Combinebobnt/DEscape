#!/usr/bin/env python3
"""Pixel-diffs this checkout's map render against the last release's (GH #87).

Renders each scenario at both commits, off-engine and full-canvas, in Flat,
Stepped and Sloped with sprites off and on, and reports every pixel that
differs outside the changes made on purpose since that release. The
release's `descape/` is extracted with `git archive` into a scratch directory
and rendered by the same Python in its own subprocess, with the install and
the Isometric elevation height pinned through a throwaway `config.yaml`, so
neither side reads your real settings.

The baseline rolls. BASE_REF is the latest release tag and moves to each new
tag once it is cut, so the registry below only ever holds the changes since
the last release. At a bump, set BASE_REF, run `--attribute`, and delete each
entry whose change the release shipped: check_registry() refuses any entry
still measured against the old tag.

What is neutralized or masked, and why, is the NEUTRALIZATIONS registry below
(`--help` lists it). Everything else must match exactly. Each entry is one
change made on purpose since BASE_REF, kept out of the comparison in one of
three ways, in order of preference:

- **A patch** puts this checkout's code back to the release's behaviour, in
  this checkout's render subprocess only. The change's units and pixels stay
  under test.
- **An exclusion** removes a group of unit consts from the scenario on both
  sides before rendering, in sprites-on runs or in every run. For per-unit
  art with no small patch surface. `--frames-dir` writes base / HEAD /
  amplified-diff crops of the excluded units for a side-by-side look.
- **A mask** ignores a screen region in Stepped. The last resort, with its
  canvas fraction pinned per file by MASK_MEASURED. `_terrace_mask` is the
  only shape the code supports: the screen bounds, over their 3x3
  neighbourhood's height range, of every tile with a neighbour at a
  different height. That band covers every skirt, so a mask entry fails the
  `shift-skirts` control below by design: a new mask needs a shape that
  leaves the skirts uncovered, never a weaker control.

Every entry has a control, `--inject no-<name>`, which drops that one entry
and must then fail on the entry's own `must_fail` configs of its
`control_file` and nowhere else. Two controls stay across every bump:
`--inject repaint-tile` retextures one tile and must fail everywhere, and
`--inject shift-skirts` samples every Stepped skirt from another part of its
tile's texture and must fail both Stepped cells of ElCid, so a mask that
swallows the skirts fails a control instead of passing quietly.

Sprites-off runs keep every unit an every-run exclusion does not name. Each
Stepped sprites-on run also asserts that both sides resolve the same number of
units to sprite art, and at least one, so a missing install cannot pass as "no
difference".

Run:
    tools/release_render_diff.py                         # the quick corpus in examples/
    tools/release_render_diff.py examples/foo.aoe2scenario --elev-step-pct 25 200
    tools/release_render_diff.py --attribute             # name the tiles, terrains and units under each diff
    tools/release_render_diff.py --frames-dir build/release_frames
    tools/release_render_diff.py --inject repaint-tile   # a control: must exit 1

Needs git, the release tag, and an AoE2:DE install (--install,
AOE2DE_INSTALL_PATH, or the configured one). Exits 1 on any unmasked
difference. The default composite backend is numpy on both sides;
`--composite native` renders this checkout through the native kernel.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE_REF = "v0.10"
STYLES = ("flat", "stepped", "sloped")
DEFAULT_PCTS = (50, 200)
# The same four files as tests/conftest.py's QUICK_CORPUS_NAMES.
# No examples/ file places town centres 71, 141 or 142, so their art is not compared (decided 2026-10-20, record only).
QUICK_CORPUS = (
    "0_June_Event_Scenario.aoe2scenario",
    "2_Joan_coop_1_v0_13.aoe2scenario",
    "C2_ElCid_coop_1_v0_16.aoe2scenario",
    "F7_2_Dos Pilas (648).aoe2scenario",
)
# A comparison config is (style, sprites on). These name the sets the controls must fail.
ALL_CONFIGS = frozenset((style, sprites) for style in STYLES for sprites in (False, True))
SPRITES_ON = frozenset((style, True) for style in STYLES)
SPRITES_OFF = ALL_CONFIGS - SPRITES_ON
ISO_CONFIGS = frozenset((style, sprites) for style, sprites in ALL_CONFIGS if style != "flat")
ELCID = "C2_ElCid_coop_1_v0_16.aoe2scenario"
JOAN = "2_Joan_coop_1_v0_13.aoe2scenario"
OLD_ALLIES = "old-allies-final-v2.aoe2scenario"
LELOI = "R4_LeLoi_4.aoe2scenario"
DOS_PILAS = "F7_2_Dos Pilas (648).aoe2scenario"
# Matched by the release gate beside the quick corpus whatever entries exist (tests/test_release_render_diff.py).
EXTRA_MATCH_FILES = (OLD_ALLIES, LELOI)
STRIP_ROWS = 1024
# Review frames: crop CROP_W x CROP_H canvas px, upscaled nearest-neighbour to 1200x900 (REVIEW_PACK framing rules).
CROP_W, CROP_H, UPSCALE = 240, 180, 5
DIFF_GAIN = 4
# --attribute: how many terrains, consts and tiles each report line names, and how far "near" reaches in tiles.
ATTRIBUTE_TOP = 8
ATTRIBUTE_NEAR = 2


# ---------------------------------------------------------------------------
# Patches: each runs in this checkout's worker, before it renders.


def _require(module, *names: str) -> None:
    for name in names:
        if not hasattr(module, name):
            raise SystemExit(f"{module.__name__}.{name} is gone: update its patch in {Path(__file__).name} to match the refactor")


def _unconformed_blend_edges() -> None:
    """Put the pre-TASK-284 blend masks back: each blendomatic tile keeps its own non-bleeding edges."""
    from descape import terrain_blend

    _require(terrain_blend, "conform_edges")
    terrain_blend.conform_edges = lambda alpha, tile_index: alpha


def _unturned_flat_walls() -> None:
    """Put the pre-GH #151 Flat art back: a wall draws its stored shape and a gate its own const's art."""
    from descape import unit_sprites

    _require(unit_sprites, "flat_art")
    unit_sprites.flat_art = lambda unit_const, rotation: (unit_const, rotation)


def _single_anchor_sprites() -> None:
    """Put the pre-TASK-221 depth order back: a multi-tile sprite paints whole at its front corner."""
    from descape import render

    _require(render, "SPRITE_COLUMN_SLICING")
    render.SPRITE_COLUMN_SLICING = False


def _no_garrison_flags() -> None:
    """Put the pre-GH #203 hosts back: a unit holding a garrison draws no garrison flag."""
    from descape import unit_sprites

    _require(unit_sprites, "garrison_flag_pieces_for")
    unit_sprites.garrison_flag_pieces_for = lambda *args, **kwargs: []


# ---------------------------------------------------------------------------
# The registry.


@dataclass(frozen=True)
class Neutralization:
    """One change made on purpose since BASE_REF, and how the comparison is kept blind to it.

    Exactly one of `patch`, `consts` and `mask` is set. `since` is the BASE_REF the entry was
    measured against: run() refuses a registry holding any other, so a BASE_REF bump has to
    revisit every entry. `--inject no-<name>` is the entry's control and must fail exactly the
    `must_fail` (style, sprites) configs of `control_file`.
    """

    name: str
    change: str  # GH/TASK ids and what changed
    since: str
    control_file: str
    must_fail: frozenset[tuple[str, bool]]
    patch: Callable[[], None] | None = None
    consts: Callable[[], frozenset[int]] | None = None
    every_run: bool = False  # consts only: removed from sprites-off runs too
    mask: bool = False

    @property
    def kind(self) -> str:
        if self.patch is not None:
            return "patch"
        if self.consts is not None:
            return "exclusion, every run" if self.every_run else "exclusion, sprites-on runs"
        return "mask"


NEUTRALIZATIONS: tuple[Neutralization, ...] = (
    Neutralization(
        "blend-seams",
        "TASK-284: terrain blending has no square steps in coastlines and water inlets",
        since="v0.10", control_file=ELCID, must_fail=ALL_CONFIGS,
        patch=_unconformed_blend_edges,
    ),
    Neutralization(
        "flat-walls",
        "GH #151 (TASK-227): walls and gates lie along the square grid in top-down Flat",
        since="v0.10", control_file=OLD_ALLIES, must_fail=frozenset({("flat", True)}),
        patch=_unturned_flat_walls,
    ),
    Neutralization(
        "face-columns",
        "TASK-221 (GH #20): units beside a building's front faces draw in front of it, not behind",
        since="v0.10", control_file=DOS_PILAS,
        must_fail=frozenset({("stepped", True), ("sloped", True)}),
        patch=_single_anchor_sprites,
    ),
    Neutralization(
        "garrison-flags",
        "GH #203: a building, ram, siege tower or transport ship holding units draws its garrison flag",
        since="v0.10", control_file=OLD_ALLIES,
        must_fail=frozenset({("stepped", True), ("sloped", True)}),
        patch=_no_garrison_flags,
    ),
)

# A mask entry's measured Stepped canvas fraction against BASE_REF, by file and elev_step_pct (empty while none exists).
# A run over measured + MASK_MARGIN fails: a mask grown to cover the canvas would pass every Stepped cell and its control.
MASK_MEASURED: dict[str, dict[int, float]] = {}
MASK_MARGIN = 0.005


def mask_ceiling(scenario_name: str, pct: int) -> float | None:
    """The largest Stepped masked fraction this file may report at this pct, or None where none is pinned."""
    measured = MASK_MEASURED.get(scenario_name, {}).get(pct)
    return None if measured is None else measured + MASK_MARGIN


REPAINT_TILE = "repaint-tile"
SHIFT_SKIRTS = "shift-skirts"


def controls() -> list[tuple[str, str, frozenset[tuple[str, bool]]]]:
    """(inject, control file, the configs it must fail) for every control run."""
    fixed = [(REPAINT_TILE, ELCID, ALL_CONFIGS), (SHIFT_SKIRTS, ELCID, frozenset({("stepped", False), ("stepped", True)}))]
    return fixed + [(f"no-{n.name}", n.control_file, n.must_fail) for n in NEUTRALIZATIONS]


INJECTS = tuple(inject for inject, _file, _configs in controls())


def check_registry() -> None:
    """Raises ValueError for a registry that would prove nothing: see Neutralization."""
    names = [n.name for n in NEUTRALIZATIONS]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate neutralization names in {names}")
    for n in NEUTRALIZATIONS:
        if sum((n.patch is not None, n.consts is not None, n.mask)) != 1:
            raise ValueError(f"{n.name}: set exactly one of patch, consts and mask")
        if not n.must_fail or not n.must_fail <= ALL_CONFIGS:
            raise ValueError(f"{n.name}: must_fail is empty or names an unknown config, so its control proves nothing")
        if n.since != BASE_REF:
            raise ValueError(
                f"{n.name} was measured against {n.since}, but BASE_REF is {BASE_REF}: delete the entry if the "
                f"release shipped its change, else re-measure it (--attribute) and update its `since`"
            )
    if sum(n.mask for n in NEUTRALIZATIONS) > 1:
        raise ValueError("more than one mask entry: _terrace_mask is the only mask shape, so their controls could not tell them apart")


def intended_change_consts(active: tuple[Neutralization, ...] = NEUTRALIZATIONS) -> dict[str, frozenset[int]]:
    """The consts the exclusion entries remove, by the change that needs them out."""
    return {n.change: n.consts() for n in active if n.consts is not None}


# ---------------------------------------------------------------------------
# Worker: runs in a fresh interpreter whose descape/ is the tree being rendered.


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


def _shift_skirts() -> None:
    """The skirt control: every Stepped skirt samples its tile's texture half a tile down and a third across."""
    from descape import iso_geometry

    _require(iso_geometry, "skirt_quad_indices", "index_extent")
    skirt = iso_geometry.skirt_quad_indices

    @functools.lru_cache(maxsize=256)
    def shifted(tile_px: int, drop_px: int, side: str):
        dst_y, dst_x, src_y, src_x = skirt(tile_px, drop_px, side)
        return dst_y, dst_x, (src_y + tile_px // 2) % tile_px, (src_x + tile_px // 3) % tile_px

    # Render's per-tile paint reads the producer through the module attribute, on both backends.
    iso_geometry.skirt_quad_indices = shifted
    iso_geometry.index_extent.cache_clear()


def _load_without(spec: dict):
    """(scenario with spec's excluded consts removed, how many units that removed)."""
    from descape.scenario_io import load_map_and_units

    scenario = load_map_and_units(Path(spec["scenario"]))
    excluded = frozenset(spec["exclude"])
    removed = 0
    for units in scenario.unit_manager.units:
        kept = [u for u in units if u.unit_const not in excluded]
        removed += len(units) - len(kept)
        units[:] = kept
    return scenario, removed


def _worker(spec: dict) -> None:
    sys.path.insert(0, spec["root"])
    if "attribute_out" in spec:
        _attribute(spec)
        return
    from descape import iso_geometry, render

    patches = {n.name: n.patch for n in NEUTRALIZATIONS if n.patch is not None}
    for name in spec["patches"]:
        patches[name]()
    if spec.get("shift_skirts"):
        _shift_skirts()
    scenario, removed = _load_without(spec)
    if spec.get("repaint_tile"):
        _repaint_one_flat_tile(scenario)
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


def _diff_px_by_tile(spec: dict, scenario, diff: np.ndarray) -> tuple[Counter, int]:
    """(diff px per (x, y) tile, px no tile claims). Flat is exact (px // tile_px). Stepped and Sloped
    sample one diff pixel per half-tile screen cell through iso_geometry.screen_to_tile, which reads
    flat tile tops: a skirt pixel resolves to no tile, and a Sloped tile is found by its Stepped diamond."""
    from descape import iso_geometry, render

    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    px: Counter = Counter()
    if spec["style"] == "flat":
        tile_px = render.tile_pixels_for_map(w, h)
        for top in range(0, diff.shape[0], STRIP_ROWS):
            ys, xs = np.nonzero(diff[top : top + STRIP_ROWS])
            ids, counts = np.unique(((ys + top) // tile_px) * w + xs // tile_px, return_counts=True)
            for tile_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
                px[(tile_id % w, tile_id // w)] += count
        return px, 0
    if spec["style"] == "stepped":
        elevations, proj = render.elevations_and_proj(scenario)
    else:
        elevations, _corner_rise, proj = render.sloped_elevations_and_proj(scenario)
    cell_w, cell_h = proj.half_w, proj.half_h
    cols = -(-diff.shape[1] // cell_w)
    strip = cell_h * max(1, STRIP_ROWS // cell_h)
    unresolved = 0
    for top in range(0, diff.shape[0], strip):
        ys, xs = np.nonzero(diff[top : top + strip])
        if not ys.size:
            continue
        cells = ((ys + top) // cell_h) * cols + xs // cell_w
        order = np.argsort(cells, kind="stable")
        _ids, starts, counts = np.unique(cells[order], return_index=True, return_counts=True)
        # Each cell's middle diff pixel in row-major order stands for the whole cell.
        mid = order[starts + counts // 2]
        for sx, sy, count in zip(xs[mid].tolist(), (ys[mid] + top).tolist(), counts.tolist(), strict=True):
            tile = iso_geometry.screen_to_tile(sx, sy, elevations, proj)
            if tile is None:
                unresolved += count
            else:
                px[(int(tile[0]), int(tile[1]))] += count
    return px, unresolved


def _attribute(spec: dict) -> None:
    """Names what sits under one comparison's unmasked diff: per tile, its terrain id and the consts of
    the units whose footprint covers it, else of those within ATTRIBUTE_NEAR tiles (a sprite overhangs)."""
    from descape import object_catalog, render, terrain_palette

    scenario, _removed = _load_without(spec)
    mm = scenario.map_manager
    w, h = mm.map_width, mm.map_height
    px_by_tile, unresolved = _diff_px_by_tile(spec, scenario, np.load(spec["diff"], mmap_mode="r"))

    on_tile: dict[tuple[int, int], set[int]] = {}
    for units in scenario.unit_manager.units:
        for unit in units:
            bounds = render.unit_tile_bounds(unit, w, h)
            if bounds is None:
                continue
            x0, x1, y0, y1 = bounds
            for ty in range(y0, y1):
                for tx in range(x0, x1):
                    on_tile.setdefault((tx, ty), set()).add(unit.unit_const)

    reach = range(-ATTRIBUTE_NEAR, ATTRIBUTE_NEAR + 1)
    by_terrain, by_const, by_near, bare = {}, {}, {}, [0, 0]
    tiles = []
    for (x, y), count in px_by_tile.items():
        terrain_id = mm.terrain[y * w + x].terrain_id
        consts = sorted(on_tile.get((x, y), ()))
        near = [] if consts else sorted(set().union(*(on_tile.get((x + dx, y + dy), ()) for dy in reach for dx in reach)))
        tiles.append((count, x, y, terrain_id, consts, near))
        for table, keys in ((by_terrain, [terrain_id]), (by_const, consts), (by_near, near)):
            for key in keys:
                entry = table.setdefault(key, [0, 0])
                entry[0] += count
                entry[1] += 1
        if not consts and not near:
            bare[0] += count
            bare[1] += 1

    def ranked(table: dict, name) -> list:
        top = sorted(table.items(), key=lambda item: -item[1][0])[:ATTRIBUTE_TOP]
        return [[key, name(key), px, n] for key, (px, n) in top]

    report = {
        "px": sum(px_by_tile.values()) + unresolved,
        "tiles": len(px_by_tile),
        "unresolved_px": unresolved,
        "terrains": ranked(by_terrain, terrain_palette.name_for_terrain_id),
        "consts": ranked(by_const, object_catalog.display_name),
        "near_consts": ranked(by_near, object_catalog.display_name),
        "bare": bare,
        "top_tiles": [list(t) for t in sorted(tiles, reverse=True)[:ATTRIBUTE_TOP]],
    }
    Path(spec["attribute_out"]).write_text(json.dumps(report))


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
    attribution: list[str] = field(default_factory=list)

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
        raise RuntimeError(f"worker failed for {spec.get('out') or spec.get('attribute_out')}:\n{proc.stderr[-4000:]}")


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
    """Crops the densest clusters of `diff` (already outside any mask) as a/b/amplified-diff PNGs."""
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


def _attribution_lines(report: dict) -> list[str]:
    def named(rows: list) -> str:
        return "; ".join(f"{key} {name} {px} px/{tiles} tiles" for key, name, px, tiles in rows) or "none"

    tiles = "; ".join(
        f"({x}, {y}) terrain {terrain} consts {consts or 'none'}{f' near {near}' if near else ''} {px} px"
        for px, x, y, terrain, consts, near in report["top_tiles"]
    )
    return [
        f"     attribute: {report['px']} px on {report['tiles']} tiles ({report['unresolved_px']} px on no tile top)",
        f"       terrain ids: {named(report['terrains'])}",
        f"       unit consts on the tile: {named(report['consts'])}",
        f"       unit consts within {ATTRIBUTE_NEAR} tiles of a unit-free tile: {named(report['near_consts'])}",
        f"       no unit within {ATTRIBUTE_NEAR} tiles: {report['bare'][0]} px/{report['bare'][1]} tiles",
        f"       densest tiles: {tiles or 'none'}",
    ]


def _pair_result(job: Job, out: Path, diff_px: int, unmasked: int, frac: float) -> PairResult:
    meta_a = json.loads((out / f"{job.key}_a.json").read_text())
    meta_b = json.loads((out / f"{job.key}_b.json").read_text())
    result = PairResult(job, diff_px, unmasked, frac, meta_b["removed_units"])
    if unmasked:
        result.problems.append(f"{unmasked} px differ outside the intended changes")
    ceiling = mask_ceiling(job.scenario.name, job.pct) if job.style == "stepped" else None
    if ceiling is not None and frac > ceiling:
        result.problems.append(f"the mask covers {frac:.2%} of the canvas, over its {ceiling:.2%} ceiling")
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
    attribute: bool = False,
    log=print,
) -> list[PairResult]:
    """Renders and compares every (scenario, style, sprites, pct); returns one PairResult per comparison.

    Each pair is compared as soon as both sides finish and its .npy canvases are then deleted
    (unless keep_renders), so disk peaks at about `jobs` pairs rather than the whole run.
    attribute adds --attribute's report lines to each failing result.
    """
    check_registry()
    if inject is not None and inject not in INJECTS:
        raise ValueError(f"inject must be one of {INJECTS}, got {inject!r}")
    active = tuple(n for n in NEUTRALIZATIONS if inject != f"no-{n.name}")
    base_root = extract_base(base_ref, work_dir / "base")
    patches = [n.name for n in active if n.patch is not None]
    masked = any(n.mask for n in active)
    groups = intended_change_consts(active)
    exclude_always = set().union(*(n.consts() for n in active if n.consts is not None and n.every_run))
    exclude_sprites = set().union(*(n.consts() for n in active if n.consts is not None and not n.every_run))
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

    def excluded_for(job: Job) -> list[int]:
        if job.full:
            return []
        return sorted(exclude_always | exclude_sprites if job.sprites else exclude_always)

    def render_pair(job: Job) -> None:
        for side, root in (("a", base_root), ("b", ROOT)):
            spec = {
                "root": str(root),
                "scenario": str(job.scenario),
                "style": job.style,
                "sprites": job.sprites,
                "exclude": excluded_for(job),
                "patches": patches if side == "b" else [],
                "repaint_tile": side == "b" and inject == REPAINT_TILE,
                "shift_skirts": side == "b" and inject == SHIFT_SKIRTS,
                "out": str(out / f"{job.key}_{side}.npy"),
                "meta_out": str(out / f"{job.key}_{side}.json"),
                "mask_out": str(out / f"{job.key}_mask.npy") if side == "b" and job.style == "stepped" else None,
            }
            _launch(spec, homes[job.pct], install, "numpy" if side == "a" else composite)

    def attribution(job: Job, diff: np.ndarray) -> list[str]:
        diff_path, report_path = out / f"{job.key}_diff.npy", out / f"{job.key}_attribute.json"
        np.save(diff_path, diff)
        spec = {
            "root": str(ROOT), "scenario": str(job.scenario), "style": job.style, "exclude": excluded_for(job),
            "diff": str(diff_path), "attribute_out": str(report_path),
        }
        _launch(spec, homes[job.pct], install, "numpy")
        diff_path.unlink()
        return _attribution_lines(json.loads(report_path.read_text()))

    def finish(job: Job) -> tuple[PairResult | None, list[str]]:
        a_path, b_path = out / f"{job.key}_a.npy", out / f"{job.key}_b.npy"
        mask_path = out / f"{job.key}_mask.npy" if job.style == "stepped" and masked else None
        diff_px, unmasked, frac, diff = _compare(a_path, b_path, mask_path)
        frames: list[str] = []
        if job.full and frames_dir is not None and unmasked:
            frames_dir.mkdir(parents=True, exist_ok=True)
            frames = _write_frames(job, a_path, b_path, diff, frames_dir, frames_per_config)
        notes = attribution(job, diff) if attribute and unmasked and not job.full else []
        if not keep_renders:
            # By name, not mask_path: a run without the mask entry still writes the mask.
            for side in ("a", "b", "mask"):
                (out / f"{job.key}_{side}.npy").unlink(missing_ok=True)
        if job.full:
            return None, frames
        result = _pair_result(job, out, diff_px, unmasked, frac)
        result.attribution = notes
        return result, frames

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
            for note in result.attribution:
                log(note)
    if frames_dir is not None and index:
        header = [
            f"# {base_ref} vs HEAD: the intended sprite-art changes, side by side",
            "",
            f"Each frame is a/b/c: the base release, this checkout, and |a-b| x{DIFF_GAIN}.",
            f"Crops are {CROP_W}x{CROP_H} canvas px, upscaled x{UPSCALE}.",
            "Excluded from the comparison, compared in full here:",
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


def _registry_help() -> str:
    lines = [f"Neutralized against {BASE_REF} (name, kind, change, control file):"]
    lines += [f"  {n.name} ({n.kind}): {n.change}. Control on {n.control_file}." for n in NEUTRALIZATIONS]
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]))
        return 0
    # Only the driver imports this checkout's descape; a worker must see its own tree's.
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=_registry_help(), formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument(
        "--attribute", action="store_true",
        help="Under each failing comparison, name the tiles, terrain ids and unit consts its diff sits on",
    )
    args = parser.parse_args()

    scenarios = args.scenarios or [ROOT / "examples" / name for name in QUICK_CORPUS]
    missing = [str(p) for p in scenarios if not p.is_file()]
    if missing:
        parser.error(f"not found: {', '.join(missing)}")
    install = args.install or _default_install()
    if install is None:
        parser.error("no AoE2:DE install: pass --install or set AOE2DE_INSTALL_PATH")
    if args.base != BASE_REF:
        print(f"warning: --base {args.base} is not BASE_REF ({BASE_REF}), which the neutralizations were measured against", file=sys.stderr)

    def go(work_dir: Path) -> list[PairResult]:
        return run(
            scenarios, work_dir, install, base_ref=args.base, pcts=tuple(args.elev_step_pct), styles=tuple(args.style),
            composite=args.composite, jobs=args.jobs, frames_dir=args.frames_dir, frames_per_config=args.frames_per_config,
            inject=args.inject, keep_renders=args.keep_renders, attribute=args.attribute,
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
