"""GH #20: units drawn from a particle effect's atlas (Butterflies 1608-1610; Seagulls 303
since GH #67; Smoke 1308 and Flames 1333-1336 since GH #256; their hidden twins and the
waterfalls' ImageAngles runs since GH #330).

Two halves. The synthetic one builds a tiny TexturePacker atlas in tmp_path
and pins what descape/particle_art.py reads out of it: which rect a frame is
cut from (a `rotated` frame's is the swapped one), which way it is turned
back, where the hotspot lands, and that the effect's Scale reaches the draw
exactly once. The real-asset one settles the one thing a synthetic atlas
cannot, because the fixture would only restate the code's own convention:
which way the game's packer turned its rotated frames.

The real-asset tests need AOE2DE_INSTALL_PATH, like
tests/test_wall_variants_real_assets.py: conftest._isolated_settings hides
config.yaml. Not `corpus`-marked: one 2048 px atlas decode and a few dozen cuts.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from descape import asset_source, game_dataset, games, iso_geometry, particle_art, render_paint, render_units, unit_sprites

CONST = 990001
BLOCK_CONST = 990002
EFFECT = "fx_low"
BLOCK_EFFECT = "fx_block"
ATLAS_FILE = "textures\\atlases\\fx.dds"

NATIVE_HALF_W = unit_sprites.NATIVE_TILE_W // 2


def _pattern(rows: int, cols: int, base: int) -> np.ndarray:
    """An opaque sprite whose every pixel is distinct, so a wrong rect, a
    wrong turn and a flip each compare unequal."""
    index = np.arange(rows * cols, dtype=np.uint8).reshape(rows, cols)
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    rgba[..., 0] = base + index
    rgba[..., 1] = 250 - index
    rgba[..., 2] = base
    rgba[..., 3] = 255
    return rgba


# Frame 0 of EFFECT: upright in the atlas, with a clear left column and bottom
# row so the crop to ink has something to trim.
UPRIGHT = _pattern(4, 6, 40)
UPRIGHT[:, 0, 3] = 0
UPRIGHT[3, :, 3] = 0
# Frame 1 of EFFECT: stored turned 90 degrees clockwise, 3 columns by 7 rows.
TURNED = _pattern(3, 7, 120)
# Frame 0 of BLOCK_EFFECT: even-sized, so halving it is exact.
BLOCK = _pattern(8, 12, 10)

RECORDS = [
    # Atlas index 0 belongs to no effect: ImageFirst has to skip it.
    {"frame": {"x": 0, "y": 0, "w": 2, "h": 2}, "rotated": False,
     "spriteSourceSize": {"x": 0, "y": 0, "w": 2, "h": 2},
     "sourceSize": {"w": 4, "h": 4}, "pivot": {"x": 0.5, "y": 0.5}},
    {"frame": {"x": 5, "y": 3, "w": 6, "h": 4}, "rotated": False,
     "spriteSourceSize": {"x": 7, "y": 12, "w": 6, "h": 4},
     "sourceSize": {"w": 40, "h": 60}, "pivot": {"x": 0.25, "y": 0.5}},
    # w/h are the UPRIGHT size; the atlas region is 3 wide by 7 tall.
    {"frame": {"x": 20, "y": 10, "w": 7, "h": 3}, "rotated": True,
     "spriteSourceSize": {"x": 3, "y": 5, "w": 7, "h": 3},
     "sourceSize": {"w": 40, "h": 60}, "pivot": {"x": 0.5, "y": 0.5}},
    {"frame": {"x": 2, "y": 20, "w": 12, "h": 8}, "rotated": False,
     "spriteSourceSize": {"x": 12, "y": 26, "w": 12, "h": 8},
     "sourceSize": {"w": 40, "h": 60}, "pivot": {"x": 0.5, "y": 0.5}},
]


def _atlas_pixels() -> np.ndarray:
    # A faint non-zero fill, so a cut that strays off its rect picks up ink.
    atlas = np.full((32, 32, 4), (3, 3, 3, 9), dtype=np.uint8)
    atlas[0:2, 0:2] = (9, 9, 9, 255)
    atlas[3:7, 5:11] = UPRIGHT
    atlas[10:17, 20:23] = np.rot90(TURNED, -1)
    atlas[20:28, 2:14] = BLOCK
    return atlas


def _entry(effect: str, **extra) -> dict:
    return {"graphic_id": 1, "file_name": effect, "angle_count": 1, "mirroring_mode": 0,
            "frame_count": 1, "particle": True, "art_scale": 0.5, **extra}


@pytest.fixture
def particles(tmp_path, monkeypatch):
    """A tmp install holding one atlas and two effects over it, and a graphic
    map naming them. Yields the particle folder and the live map dict."""
    folder = tmp_path / asset_source.PARTICLE_SUBPATH
    (folder / "textures" / "atlases").mkdir(parents=True)
    Image.fromarray(_atlas_pixels(), "RGBA").save(folder / "textures" / "atlases" / "fx.dds")
    (folder / "textures" / "atlases" / "fx.json").write_text(json.dumps({"frames": RECORDS}))
    (folder / f"{EFFECT}.json").write_text(json.dumps(
        {"AtlasFile": ATLAS_FILE, "ImageFirst": 1, "ImageCount": 2, "Scale": 0.5}
    ))
    (folder / f"{BLOCK_EFFECT}.json").write_text(json.dumps(
        {"AtlasFile": ATLAS_FILE, "ImageFirst": 3, "ImageCount": 1, "Scale": 0.5}
    ))
    table = {CONST: _entry(EFFECT), BLOCK_CONST: _entry(BLOCK_EFFECT)}
    monkeypatch.setattr(unit_sprites, "graphic_map", lambda: table)
    asset_source.set_install_path_override(tmp_path)
    unit_sprites.clear_caches()
    yield folder, table
    asset_source.set_install_path_override(None)
    unit_sprites.clear_caches()


# --- the cut ------------------------------------------------------------


def test_an_upright_frame_is_cut_from_its_own_rect(particles):
    main, playercolor, hx, hy = particle_art.native_frame(EFFECT, 0)
    assert np.array_equal(main, UPRIGHT)
    assert playercolor is None
    # pivot (0.25, 0.5) of the 40x60 source is (10, 30); the frame sits at (7, 12).
    # Not the trimmed rect's own pivot, which would be (2, 2).
    assert (hx, hy) == (3, 18)


def test_a_rotated_frame_is_cut_from_the_swapped_rect_and_turned_back(particles):
    main, _playercolor, hx, hy = particle_art.native_frame(EFFECT, 1)
    assert main.shape == TURNED.shape, "the atlas region is frame.h wide by frame.w tall"
    assert np.array_equal(main, TURNED), "turned the wrong way, or cut from the unswapped rect"
    assert (hx, hy) == (20 - 3, 30 - 5)


def test_the_frame_index_counts_from_the_effects_own_first_image(particles):
    """ImageFirst 1 skips atlas record 0, and the run ends at ImageCount."""
    assert particle_art.native_frame(EFFECT, 0)[0][0, 1, 0] == UPRIGHT[0, 1, 0]
    assert particle_art.native_frame(EFFECT, 2) is None
    assert particle_art.native_frame(EFFECT, -1) is None
    assert np.array_equal(particle_art.native_frame(BLOCK_EFFECT, 0)[0], BLOCK)


def _flip_effect(folder, flip, image_first: int) -> str:
    """One frame of the atlas as an effect carrying `FlipH: flip`."""
    (folder / "fx_flip.json").write_text(json.dumps(
        {"AtlasFile": ATLAS_FILE, "ImageFirst": image_first, "ImageCount": 1, "Scale": 0.5, "FlipH": flip}
    ))
    return "fx_flip"


@pytest.mark.parametrize(
    ("image_first", "frame"),
    [(3, BLOCK), (2, TURNED)],
    ids=["upright", "rotated-is-turned-back-first"],
)
def test_fliph_mirrors_the_frame_and_its_hotspot_about_the_pivot(particles, image_first, frame):
    folder, _table = particles
    plain = particle_art.native_frame(_flip_effect(folder, False, image_first), 0)
    particle_art.clear_caches()
    main, _playercolor, hx, hy = particle_art.native_frame(_flip_effect(folder, True, image_first), 0)
    assert np.array_equal(plain[0], frame)
    assert np.array_equal(main, np.fliplr(frame))
    assert main.flags.c_contiguous, "unit_sprites and the kernel take a contiguous frame"
    assert (hx, hy) == (frame.shape[1] - plain[2], plain[3])
    assert hx != plain[2], "this frame's hotspot is centred, so it cannot show the mirror"


@pytest.mark.parametrize("flip", ["true", 1, None])
def test_only_a_real_bool_fliph_mirrors(particles, flip):
    folder, _table = particles
    main, _playercolor, hx, _hy = particle_art.native_frame(_flip_effect(folder, flip, 3), 0)
    assert np.array_equal(main, BLOCK)
    assert hx == 8


def _hide(path) -> None:
    path.rename(path.with_name(path.name + ".hidden"))


@pytest.mark.parametrize(
    "break_it",
    [
        lambda folder: _hide(folder / f"{EFFECT}.json"),
        lambda folder: (folder / f"{EFFECT}.json").write_text("{not json"),
        lambda folder: (folder / f"{EFFECT}.json").write_text(json.dumps({"ImageCount": 2})),
        # An ImageAngles that does not divide ImageCount: nothing, not frame 0.
        lambda folder: (folder / f"{EFFECT}.json").write_text(json.dumps(
            {"AtlasFile": ATLAS_FILE, "ImageFirst": 1, "ImageCount": 2, "ImageAngles": 3}
        )),
        lambda folder: _hide(folder / "textures" / "atlases" / "fx.dds"),
        lambda folder: (folder / "textures" / "atlases" / "fx.dds").write_bytes(b"not a dds"),
        lambda folder: _hide(folder / "textures" / "atlases" / "fx.json"),
        lambda folder: (folder / "textures" / "atlases" / "fx.json").write_text(json.dumps(
            {"frames": [RECORDS[0], {**RECORDS[1], "frame": {"x": 30, "y": 3, "w": 6, "h": 4}}]}
        )),
        lambda folder: (folder / "textures" / "atlases" / "fx.json").write_text(json.dumps(
            {"frames": [RECORDS[0], {"rotated": False}]}
        )),
    ],
    ids=["no-effect", "bad-effect-json", "no-atlas-file", "image-angles", "no-dds", "bad-dds",
         "no-atlas-json", "rect-leaves-atlas", "record-missing-keys"],
)
def test_every_failure_is_none_not_an_exception(particles, break_it):
    folder, _table = particles
    break_it(folder)
    unit_sprites.clear_caches()
    assert particle_art.native_frame(EFFECT, 0) is None
    # And so the unit falls back to its coloured mark.
    assert unit_sprites.sprite_pieces_for(CONST, 0.0, 0, NATIVE_HALF_W) == []


def test_no_install_is_none(particles):
    asset_source.set_install_path_override(None)
    assert particle_art.native_frame(EFFECT, 0) is None


# --- ImageAngles runs (GH #330, the waterfalls) -------------------------------

RUNS_CONST = 990003
RUNS_EFFECT = "fx_runs"


def _runs_effect(folder, **extra) -> str:
    """EFFECT's two frames as two ImageAngles runs of one frame each."""
    (folder / f"{RUNS_EFFECT}.json").write_text(json.dumps(
        {"AtlasFile": ATLAS_FILE, "ImageFirst": 1, "ImageCount": 2, "Scale": 0.5, "ImageAngles": 2, **extra}
    ))
    return RUNS_EFFECT


def test_an_image_angles_run_starts_at_its_share_of_the_count(particles):
    """Run 1's first frame is index ImageCount / ImageAngles, across the whole run."""
    folder, _table = particles
    stem = _runs_effect(folder)
    assert np.array_equal(particle_art.native_frame(stem, 0)[0], UPRIGHT)
    assert np.array_equal(particle_art.native_frame(stem, 1)[0], TURNED)


@pytest.mark.parametrize(
    "extra",
    [{"ImageAngles": 3}, {"ImageAngles": 0}, {"ImageAngles": -2}, {"ImageAngles": "2"},
     {"ImageAngles": 2.0}, {"ImageAngles": True}, {"ImageCount": "2"}],
    ids=["not-a-divisor", "zero", "negative", "str", "float", "bool", "str-count"],
)
def test_a_malformed_image_angles_draws_nothing(particles, extra):
    folder, _table = particles
    stem = _runs_effect(folder, **extra)
    assert particle_art.native_frame(stem, 0) is None
    assert particle_art.native_frame(stem, 1) is None


def test_the_variant_dispatch_picks_an_image_angles_run(particles):
    """A waterfall-shaped entry (angle_count = runs, frame_count = run length)
    lands on each run's first frame, install-free."""
    folder, table = particles
    table[RUNS_CONST] = _entry(
        _runs_effect(folder), art_scale=1.0, angle_count=2, rotation_is_variant=True, variant_count=2
    )
    unit_sprites.clear_caches()
    runs = [unit_sprites.sprite_pieces_for(RUNS_CONST, r, 0, NATIVE_HALF_W) for r in (0.0, 1.0)]
    assert [len(pieces) for pieces in runs] == [1, 1]
    assert np.array_equal(runs[0][0].draw.rgba, UPRIGHT[0:3, 1:6])
    assert np.array_equal(runs[1][0].draw.rgba, TURNED)


def test_a_particle_stem_has_no_drawn_frame_count_and_reads_no_file(monkeypatch):
    """drawn_frame_count() is the dispatch's frame_count_of; a particle stem
    returns None before the .sld and mod lookups, so the variant branch falls
    back to angle_count with no miss lookup per draw."""
    unit_sprites.clear_caches()
    touched = []
    monkeypatch.setattr(unit_sprites, "_sld_index", touched.append)
    monkeypatch.setattr(unit_sprites, "_mod_only_index", touched.append)
    assert unit_sprites.drawn_frame_count("waterfall") is None
    assert touched == []
    unit_sprites.drawn_frame_count("not_a_particle_x1")
    assert touched, "the spy saw no lookup at all, so it proves nothing"


# --- the unit_sprites seam ------------------------------------------------


def test_native_frame_routes_a_particle_stem_and_crops_it_to_ink(particles):
    main, playercolor, hx, hy = unit_sprites._native_frame(EFFECT, 0)
    assert np.array_equal(main, UPRIGHT[0:3, 1:6]), "the clear column and row are trimmed"
    assert playercolor is None
    assert (hx, hy) == (3 - 1, 18)


def test_a_particle_miss_is_cached_like_an_sld_miss(particles):
    assert unit_sprites._native_frame(EFFECT, 2) is None
    assert unit_sprites._native_cache[(EFFECT, 2)] is unit_sprites._MISS


def test_a_stem_is_a_particle_only_when_its_entry_says_so(particles):
    """Without the flag the stem goes to the .sld reader, which has no such file."""
    _folder, table = particles
    del table[CONST]["particle"]
    unit_sprites.clear_caches()
    assert unit_sprites._native_frame(EFFECT, 0) is None


# --- art_scale --------------------------------------------------------------


def test_art_scale_is_applied_exactly_once(particles):
    """At sprite_scale() == 1.0 the 8x12 block draws at 4x6: not 8x12 (the
    effect's Scale skipped) and not 2x3 (applied to the frame and again to the
    draw). The hotspot scales with it."""
    assert unit_sprites.sprite_scale(NATIVE_HALF_W) == 1.0
    (piece,) = unit_sprites.sprite_pieces_for(BLOCK_CONST, 0.0, 0, NATIVE_HALF_W)
    assert piece.draw.rgba.shape[:2] == (4, 6)
    # Native hotspot (20 - 12, 30 - 26) = (8, 4), halved.
    assert (piece.draw.hotspot_x, piece.draw.hotspot_y) == (4, 2)
    assert (piece.dx, piece.dy) == (0, 0)
    single = unit_sprites.sprite_for(BLOCK_CONST, 0.0, 0, NATIVE_HALF_W)
    assert single.rgba.shape[:2] == (4, 6)
    # Twice the projection scale undoes the 0.5 exactly.
    (doubled,) = unit_sprites.sprite_pieces_for(BLOCK_CONST, 0.0, 0, 2 * NATIVE_HALF_W)
    assert np.array_equal(doubled.draw.rgba, BLOCK)


def test_two_art_scales_of_one_file_do_not_collide_in_the_scaled_cache(particles):
    _folder, table = particles
    (half,) = unit_sprites.sprite_pieces_for(BLOCK_CONST, 0.0, 0, NATIVE_HALF_W)
    table[BLOCK_CONST] = _entry(BLOCK_EFFECT, art_scale=1.0)
    (full,) = unit_sprites.sprite_pieces_for(BLOCK_CONST, 0.0, 0, NATIVE_HALF_W)
    assert half.draw.rgba.shape[:2] == (4, 6) and full.draw.rgba.shape[:2] == (8, 12)


def test_a_pieces_offset_scales_with_that_pieces_own_art_scale(particles):
    """A piece's dx/dy follow the scale its art was drawn at, as they do for
    tree_scale. No committed entry has such a piece today."""
    _folder, table = particles
    piece = {"unit_id": BLOCK_CONST, "file_name": BLOCK_EFFECT, "angle_count": 1, "frame_count": 1}
    table[BLOCK_CONST] = {**_entry(BLOCK_EFFECT), "pieces": [
        {**piece, "dx": 0, "dy": 0, "parent": True},
        {**piece, "dx": 10, "dy": -20, "art_scale": 0.5},
        {**piece, "dx": 10, "dy": -20},
    ]}
    parent, scaled, plain = unit_sprites.sprite_pieces_for(BLOCK_CONST, 0.0, 0, NATIVE_HALF_W)
    assert (scaled.dx, scaled.dy) == (5, -10) and scaled.draw.rgba.shape[:2] == (4, 6)
    assert (plain.dx, plain.dy) == (10, -20) and plain.draw.rgba.shape[:2] == (8, 12)
    assert parent.draw.rgba.shape[:2] == (8, 12)


def test_the_flat_icon_fits_its_footprint_whatever_the_art_scale(particles):
    """icon_for() contain-fits the ink, so art_scale has nothing to multiply."""
    icon = unit_sprites.icon_for(BLOCK_CONST, 0.0, 0, 24, 24)
    assert icon.rgba.shape[:2] == (16, 24)


# --- the real atlas -----------------------------------------------------------

BUTTERFLIES = {1608: "butterflies_1_low", 1609: "butterflies_2_low", 1610: "butterflies_3_low"}
SMOKE = 1308
FLAME1, FLAME2, FLAME3, FLAME4 = 1333, 1334, 1335, 1336
# GH #256. FLAME4 is FLAME3's frame run with FlipH.
SMOKE_AND_FLAMES = {
    SMOKE: "smoke_chimney",
    FLAME1: "fire_small_left",
    FLAME2: "fire_medium_left",
    FLAME3: "fire_large_left",
    FLAME4: "fire_large_right",
}
# The hidden SMOKE 902 and FLAME1-4 939-942 name the same effects (GH #330).
HIDDEN_TWINS = {902: SMOKE, 939: FLAME1, 940: FLAME2, 941: FLAME3, 942: FLAME4}
# GH #330: const -> (effect stem, ImageAngles runs, frames per run).
WATERFALLS = {896: ("waterfall", 3, 60), 1635: ("waterfall_background", 3, 60), 2598: ("waterfall_small", 2, 30)}
# Every particle entry the committed AoE2 table carries, so the seagulls (GH #67) too.
PARTICLE_STEMS = {
    **BUTTERFLIES, **SMOKE_AND_FLAMES, 303: "dock_seagulls",
    **{const: SMOKE_AND_FLAMES[twin] for const, twin in HIDDEN_TWINS.items()},
    **{const: stem for const, (stem, _runs, _n) in WATERFALLS.items()},
}
SOURCE = 400

# Opaque px of each effect's first frame at native size, measured 2026-10-12.
NATIVE_INK = {1608: 149, 1609: 495, 1610: 305}


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible. Set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )


def test_every_particle_stem_names_an_effect_this_install_has():
    """A particle entry's file_name is joined to the install's particle folder,
    not the graphics one, and its committed art_scale is that effect's Scale."""
    _require_install()
    entries = [e for e in unit_sprites.graphic_map().values() if e.get("particle")]
    assert {e["file_name"] for e in entries} == set(PARTICLE_STEMS.values()), "the particle entries are missing, so this proves nothing"
    folder = asset_source.get_install_path() / asset_source.PARTICLE_SUBPATH
    for entry in entries:
        stem = entry["file_name"]
        effect = json.loads((folder / f"{stem}.json").read_text())
        assert effect["Scale"] == entry["art_scale"], stem
        # A plain atlas run, or ImageAngles runs the entry's angle_count / frame_count split.
        assert effect["AtlasFile"], stem
        assert effect.get("ImageAngles", 1) == entry["angle_count"], stem
        if entry["angle_count"] > 1:
            assert entry["angle_count"] * entry["frame_count"] == effect["ImageCount"], stem
        assert particle_art.native_frame(stem, 0) is not None, stem


def _blurred(plane: np.ndarray, radius: int = 8) -> np.ndarray:
    """A box blur, so specks a few px apart in two loop frames still overlap."""
    size = 2 * radius + 1
    sums = np.pad(np.cumsum(np.cumsum(np.pad(plane, radius), axis=0), axis=1), ((1, 0), (1, 0)))
    return sums[size:, size:] - sums[:-size, size:] - sums[size:, :-size] + sums[:-size, :-size]


def _match(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _blurred(a), _blurred(b)
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def _in_source(rgba: np.ndarray, hx: int, hy: int, size: int = SOURCE) -> np.ndarray:
    """The frame's alpha on a `size` px canvas, placed by its hotspot at the
    centre (the pivot is the canvas centre on every butterflies record)."""
    plane = np.zeros((size, size), dtype=np.float32)
    x0, y0 = size // 2 - hx, size // 2 - hy
    plane[y0 : y0 + rgba.shape[0], x0 : x0 + rgba.shape[1]] = rgba[..., 3] / 255.0
    return plane


def _restored_wins(effect: str, pairs, size: int = SOURCE) -> int:
    """How many (rotated, unrotated neighbour) pairs the restored frame
    matches better than its half turn does."""
    wins = 0
    for index, neighbour in pairs:
        rgba, _pc, hx, hy = particle_art.native_frame(effect, index)
        beside, _pc, beside_hx, beside_hy = particle_art.native_frame(effect, neighbour)
        reference = _in_source(beside, beside_hx, beside_hy, size)
        restored = _match(_in_source(rgba, hx, hy, size), reference)
        half_turn = _match(_in_source(np.rot90(rgba, 2), hx, hy, size), reference)
        wins += restored > half_turn
    return wins


def test_a_rotated_butterfly_frame_is_turned_back_the_right_way():
    """np.rot90 one way and the other differ by a half turn and give the same
    shape, so only the art can say which is right: a restored frame must look
    like the unrotated frame next to it in the loop more than its half turn
    does. Butterflies 2, whose several insects make a frame least symmetric.
    Measured 2026-10-12: 38 of 40 pairs, mean match 0.71 against 0.18."""
    _require_install()
    effect = "butterflies_2_low"
    folder = asset_source.get_install_path() / asset_source.PARTICLE_SUBPATH
    first = particle_art._effect(folder, effect)["ImageFirst"]
    records = particle_art._atlas_frames(folder / "textures" / "atlases" / "butterflies.json")
    # The loop's second half is 180 blank 3 px records.
    live = [i for i in range(180) if records[first + i]["frame"]["w"] > 3]
    pairs = [
        (i, n) for i in live if records[first + i]["rotated"]
        for n in (i - 1, i + 1) if n in live and not records[first + n]["rotated"]
    ]
    assert len(pairs) >= 30, f"only {len(pairs)} rotated frames have an unrotated neighbour"
    wins = _restored_wins(effect, pairs)
    assert wins >= 0.9 * len(pairs), (
        f"the restored frame beat its half turn on only {wins} of {len(pairs)} pairs: "
        "np.rot90 is turning rotated atlas frames the wrong way"
    )


@pytest.mark.parametrize("unit_const", sorted(BUTTERFLIES))
def test_every_butterfly_draws_specks_near_its_anchor(unit_const):
    """At the default zoom (tile_px 64) each butterfly is a sprite, not the
    coloured mark, and its ink stays within a tile of the anchor."""
    _require_install()
    entry = unit_sprites.graphic_map()[unit_const]
    assert entry["file_name"] == BUTTERFLIES[unit_const]
    native = unit_sprites._native_frame(entry["file_name"], 0)
    assert native is not None and native[1] is None
    assert int((native[0][..., 3] > 0).sum()) == NATIVE_INK[unit_const]

    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    pieces = unit_sprites.sprite_pieces_for(unit_const, 0.0, 0, half_w)
    assert len(pieces) == 1, f"{unit_const} falls back to the coloured mark"
    (piece,) = pieces
    ys, xs = np.nonzero(piece.draw.rgba[..., 3] > 0)
    # Measured 14 / 61 / 36 px here against 149 / 495 / 305 native.
    assert len(ys) >= 10, (unit_const, len(ys))
    assert np.abs(xs - piece.draw.hotspot_x).max() <= 2 * half_w, unit_const
    assert np.abs(ys - piece.draw.hotspot_y).max() <= 2 * half_w, unit_const
    # Scale 0.5 of the 400 px source at this zoom's 2/3: no draw can exceed 134 px.
    assert max(piece.draw.rgba.shape[:2]) <= 134, piece.draw.rgba.shape


@pytest.mark.parametrize("unit_const", sorted(SMOKE_AND_FLAMES))
def test_smoke_and_flames_draw_and_rise_from_their_anchor(unit_const):
    """At the default zoom each is a sprite, not the coloured mark, with a real
    share of its ink above the anchor. Measured 2026-10-31: 0.26 for FLAME1,
    whose pivot sits near the frame's centre, and 0.68-0.73 for the others."""
    _require_install()
    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    pieces = unit_sprites.sprite_pieces_for(unit_const, 0.0, 0, half_w)
    assert len(pieces) == 1, f"{unit_const} falls back to the coloured mark"
    (piece,) = pieces
    ys, _xs = np.nonzero(piece.draw.rgba[..., 3] > 0)
    above = int((ys < piece.draw.hotspot_y).sum())
    assert above >= 0.2 * len(ys), (unit_const, above, len(ys))


def test_return_of_rome_draws_smoke_and_flames_from_aoe2s_effects():
    """GH #311: the RoR table names AoE2's effects. Drawing always reads the
    install's AoE2 particle folder while the generator reads RoR's over it, so
    an effect RoR carries itself would make the two disagree."""
    _require_install()
    game_dataset.set_active(game_dataset.ROR)
    ror_folder = games.ROR.root(asset_source.get_install_path()) / asset_source.PARTICLE_SUBPATH
    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    for unit_const, stem in SMOKE_AND_FLAMES.items():
        assert unit_sprites.graphic_map().get(unit_const, {}).get("file_name") == stem, unit_const
        assert particle_art.native_frame(stem, 0) is not None, stem
        assert not (ror_folder / f"{stem}.json").exists(), f"RoR carries its own {stem}"
        assert len(unit_sprites.sprite_pieces_for(unit_const, 0.0, 0, half_w)) == 1, unit_const


def test_flame4_is_flame3_mirrored_about_its_hotspot():
    """FLAME3 and FLAME4 share frames 60-119, so this pins that FlipH is honoured
    on real data. Whether the game mirrors about the pivot too is a watch step."""
    _require_install()
    for read in (particle_art.native_frame, unit_sprites._native_frame):
        left, _pc, left_hx, left_hy = read(SMOKE_AND_FLAMES[FLAME3], 0)
        right, _pc, right_hx, right_hy = read(SMOKE_AND_FLAMES[FLAME4], 0)
        assert np.array_equal(right, np.fliplr(left)), read
        assert (right_hx, right_hy) == (left.shape[1] - left_hx, left_hy), read
        assert not np.array_equal(left, right), "a symmetric frame cannot show the flip"


@pytest.mark.parametrize(
    ("unit_const", "floor_tile_px"),
    [*((const, 32) for const in sorted(BUTTERFLIES)),
     (SMOKE, 16), (FLAME1, 16), (FLAME2, 16), (FLAME3, 16), (FLAME4, 16)],
)
def test_particle_art_keeps_ink_down_to_its_floor(unit_const, floor_tile_px):
    """One nearest-neighbour pass keeps roughly ink * scale^2 px. Butterflies,
    measured 2026-10-12 at tile_px 128 / 64 / 32: 70 / 14 / 4, 214 / 61 / 11 and
    125 / 36 / 6. At the 16 px floor level 0 / 4 / 1 remain: the specks are
    gone there, which is accepted (a butterfly is under a pixel at that zoom).
    FLAME1, the smallest of GH #256's, measured 451 / 115 / 25 / 3 at 128 / 64 /
    32 / 16 (2026-10-31, pre-registered 450 / 113 / 28 / 7), so it and the larger
    ones hold to the 16 px floor."""
    _require_install()
    for tile_px in (tile_px for tile_px in (128, 64, 32, 16) if tile_px >= floor_tile_px):
        half_w, _half_h = iso_geometry.half_dims(tile_px)
        (piece,) = unit_sprites.sprite_pieces_for(unit_const, 0.0, 0, half_w)
        assert int((piece.draw.rgba[..., 3] > 0).sum()) >= 1, (unit_const, tile_px)


@pytest.mark.parametrize("unit_const", sorted({*BUTTERFLIES, *SMOKE_AND_FLAMES, *WATERFALLS}))
def test_every_particle_unit_has_a_flat_icon(unit_const):
    _require_install()
    icon = unit_sprites.icon_for(unit_const, 0.0, 0, 64, 64)
    assert icon is not None and int((icon.rgba[..., 3] > 0).sum()) >= 10, unit_const


# --- the waterfalls and the hidden twins (GH #330) -------------------------------


@pytest.mark.parametrize("unit_const", sorted(WATERFALLS))
def test_every_waterfall_run_is_a_distinct_waterfall_with_ink_above_its_anchor(unit_const):
    """Each ImageAngles run's first frame is non-empty and differs from the
    others, and at the default zoom it draws with ink above the anchor.
    Measured 2026-11-25, share above: 0.76 / 0.79 / 0.79 for the large runs,
    0.42 / 0.12 for SWFALL, whose pivot sits on the fall's edge."""
    _require_install()
    stem, runs, per_run = WATERFALLS[unit_const]
    entry = unit_sprites.graphic_map()[unit_const]
    assert (entry["file_name"], entry["angle_count"], entry["frame_count"]) == (stem, runs, per_run)
    natives = [particle_art.native_frame(stem, k * per_run) for k in range(runs)]
    assert all(native is not None and (native[0][..., 3] > 0).sum() > 1000 for native in natives), stem
    for a in range(runs):
        for b in range(a + 1, runs):
            assert not np.array_equal(natives[a][0], natives[b][0]), (stem, a, b)
    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    floor = 0.5 if runs == 3 else 0.05
    for variant in range(runs):
        (piece,) = unit_sprites.sprite_pieces_for(unit_const, float(variant), 0, half_w)
        ys, _xs = np.nonzero(piece.draw.rgba[..., 3] > 0)
        assert int((ys < piece.draw.hotspot_y).sum()) >= floor * len(ys), (unit_const, variant)


def _in_run_pairs(atlas: str, per_run: int) -> list[tuple[int, int]]:
    """(rotated, unrotated neighbour) index pairs inside one ImageAngles run."""
    folder = asset_source.get_install_path() / asset_source.PARTICLE_SUBPATH
    records = particle_art._atlas_frames(folder / "textures" / "atlases" / f"{atlas}.json")
    return [
        (i, n) for i, record in enumerate(records) if record["rotated"]
        for n in (i - 1, i + 1)
        if 0 <= n < len(records) and n // per_run == i // per_run and not records[n]["rotated"]
    ]


def test_a_rotated_waterfall_frame_is_turned_back_the_right_way():
    """The butterflies' measure over waterfall.dds, the atlas whose drawn frames
    (runs 1 and 2's first) are rotated: 146 of its 180 records are. Its 8
    in-run rotated/unrotated neighbour pairs, read off waterfall.json."""
    _require_install()
    pairs = _in_run_pairs("waterfall", 60)
    assert len(pairs) >= 8, pairs
    assert particle_art._atlas_frames(
        asset_source.get_install_path() / asset_source.PARTICLE_SUBPATH / "textures" / "atlases" / "waterfall.json"
    )[60]["rotated"], "run 1's first frame is no longer rotated, so this pins nothing it draws"
    wins = _restored_wins("waterfall", pairs, size=1400)
    assert wins >= 0.9 * len(pairs), f"restored beat its half turn on {wins} of {len(pairs)} pairs"


@pytest.mark.parametrize("unit_const", sorted(HIDDEN_TWINS))
def test_a_hidden_twin_draws_pixel_identical_to_its_visible_const(unit_const):
    _require_install()
    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    (twin,) = unit_sprites.sprite_pieces_for(unit_const, 0.0, 0, half_w)
    (visible,) = unit_sprites.sprite_pieces_for(HIDDEN_TWINS[unit_const], 0.0, 0, half_w)
    assert np.array_equal(twin.draw.rgba, visible.draw.rgba), unit_const
    assert (twin.draw.hotspot_x, twin.draw.hotspot_y) == (visible.draw.hotspot_x, visible.draw.hotspot_y)


def _drawn(unit_const: int, rotation: float) -> np.ndarray:
    half_w, _half_h = iso_geometry.half_dims(render_paint.SMALL_MAP_TILE_PIXELS)
    (piece,) = unit_sprites.sprite_pieces_for(unit_const, rotation, 0, half_w)
    return piece.draw.rgba


def test_a_waterfalls_rotation_picks_its_run():
    """896's literal 0, 1, 2 draw three different runs, and the radian form
    2*pi/3 draws run 1. SWFALL 2598 has 2 runs, so its literal 2 wraps to run
    0, variant_index()'s rule for an exact integer past the count."""
    _require_install()
    runs = [_drawn(896, float(r)) for r in range(3)]
    assert not np.array_equal(runs[0], runs[1]) and not np.array_equal(runs[1], runs[2])
    assert not np.array_equal(runs[0], runs[2])
    assert np.array_equal(_drawn(896, 2 * math.pi / 3), runs[1])
    small = [_drawn(2598, float(r)) for r in range(3)]
    assert not np.array_equal(small[0], small[1])
    assert np.array_equal(small[2], small[0])


def test_the_reach_scan_measures_every_waterfall_run():
    """tools/scan_sprite_reach.py takes each run's first frame, not frame 0
    alone: run 2 reaches furthest down. Measured 2026-11-25 (left, up, right,
    down): 87/127/76/46 for the large waterfall against run 0's 87/127/76/32."""
    _require_install()
    import conftest

    scan = conftest.load_verify_module("scan_sprite_reach")
    assert scan.particle_scales()["waterfall"] == (0.53, 60, 3)
    assert scan.particle_reach("waterfall", *scan.particle_scales()["waterfall"]) == (87, 127, 76, 46)
    assert scan.particle_reach("waterfall", 0.53, 60, 1) == (87, 127, 76, 32)


def test_a_gaia_waterfall_keeps_its_stored_rotation():
    """render_units.stored_rotation() zeroes a GAIA const that is neither VARIANT
    nor ANGLE; a waterfall's rotation_is_variant keeps its stored run."""
    for unit_const in WATERFALLS:
        unit = SimpleNamespace(unit_const=unit_const, rotation=1.0)
        assert render_units.stored_rotation(0, unit) == 1.0, unit_const
