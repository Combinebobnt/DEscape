"""Composite depth slots on real art: a unit standing among a town centre's or
a pasture's pieces paints after the pieces its depth slots put behind it and
before the ones they put in front of it.

Each check renders terrain only (T), building only (B), unit only (U), both
(BU), and the building reduced to just its back (K) or front (F) pieces. On the
unit's ink, BU must equal B where a front piece covers it and U where only a
back piece does. The review pack's `no-slots` inject is the control.

Needs AOE2DE_INSTALL_PATH: conftest._isolated_settings hides the configured
install from the suite.
"""

from __future__ import annotations

import copy
import functools
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from descape import asset_source, render, unit_sprites
from descape import iso_geometry as ig
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

import conftest

pytestmark = [
    pytest.mark.corpus,
    # gen_pasture_review_pack imports the Qt capture helpers at module level.
    pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable"),
]

TOWN_CENTRE = 109
WAR_ELEPHANT = 239
PASTURE = 1897
CORNER_POST = 1888
PLAYER = 1
# DESERT_SAND: contrasts with the pasture's green drape.
GROUND_TERRAIN = 14
# BLACK: a unit pixel that looks the same over both grounds is opaque.
ALT_GROUND_TERRAIN = 47
# Twice the app's small-map tile: a corner post is ~3 px wide at 64 and the
# 1 px mask erosion would leave almost nothing of it.
TILE_PX = 2 * render.SMALL_MAP_TILE_PIXELS
ELEV_STEP_PCT = 50
# Both buildings are 4x4 and centred here, so their footprint is tiles 58..61.
CENTRE = 60.0
# A unit this far inside or outside a pasture corner overlaps that corner's post.
CORNER_NUDGE = 0.1
# Where inside its tile the town-centre unit stands (measured): from here the
# elephant overlaps both the main piece and the front pieces.
TC_UNIT_OFFSET = 0.1
# Seeded pieces pick their variant from the reference id, so it must not vary
# with a unit's position in the list.
REFERENCE_IDS = {TOWN_CENTRE: 9101, WAR_ELEPHANT: 9102, PASTURE: 9133}
# A file_name no install has: a piece pointed at it resolves to nothing and is
# skipped, while every other piece keeps its list index (seeded variants use it).
_HIDDEN_FILE = "descape_test_hidden_piece"


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip(
            "no AoE2:DE install visible: set AOE2DE_INSTALL_PATH to one. The configured "
            "config.yaml install is deliberately hidden by conftest._isolated_settings."
        )
    table = unit_sprites.graphic_map()
    for const in (TOWN_CENTRE, WAR_ELEPHANT, PASTURE):
        if const not in table:
            pytest.skip(f"unit_const {const} is not in this install's graphic map")


@pytest.fixture(autouse=True, scope="module")
def _drop_caches():
    yield
    _render.cache_clear()
    _oracle.cache_clear()
    _base.cache_clear()


@functools.cache
def _no_slots():
    return conftest.load_verify_module("gen_pasture_review_pack").INJECTS["no-slots"][0]


# ----------------------------------------------------------------- rendering


@functools.cache
def _base(terrain_id: int = GROUND_TERRAIN):
    base = load_map_and_units(str(BLANK_TEMPLATE_PATH))
    for tile in base.map_manager.terrain:
        tile.elevation = 0
        tile.terrain_id = terrain_id
    return base


def _unit(const: int, x: float, y: float):
    return SimpleNamespace(
        unit_const=const, x=x, y=y, z=0.0, rotation=0.0,
        reference_id=REFERENCE_IDS[const], status=2, garrisoned_in_id=-1,
    )


def _scenario(placed, terrain_id: int):
    base = _base(terrain_id)
    units = [[] for _ in base.unit_manager.units]
    units[PLAYER] = [_unit(*p) for p in placed]
    return SimpleNamespace(
        map_manager=base.map_manager,
        unit_manager=SimpleNamespace(units=units),
        team_indices=base.team_indices,
        player_colors=base.player_colors,
        # The library's tiles look their scenario up by uuid, so it must stay alive.
        _base=base,
    )


@functools.cache
def _projection(style: str):
    mm = _base().map_manager
    return ig.canvas_size_and_origin(
        mm.map_width, mm.map_height, TILE_PX, ig.MIN_ELEVATION, ig.MAX_ELEVATION,
        elev_step_pct=ELEV_STEP_PCT, corner_headroom_steps=1 if style == "sloped" else 0,
    )


@functools.cache
def _box(style: str):
    """Screen rect over tiles 56..63 plus three tiles of headroom for sprites."""
    if style == "flat":
        return (56 * TILE_PX, 56 * TILE_PX, 64 * TILE_PX, 64 * TILE_PX)
    proj = _projection(style)
    xs, ys = [], []
    for tx in (56, 63):
        for ty in (56, 63):
            sx, sy = ig.tile_screen_origin(tx, ty, 0, proj)
            xs += [sx, sx + 2 * proj.half_w]
            ys += [sy, sy + 2 * proj.half_h]
    return min(xs), min(ys) - 6 * proj.half_h, max(xs), max(ys)


@contextmanager
def _patched(const: int, transform):
    """gen_pasture_review_pack._patched for any const: `transform` rewrites its pieces."""
    real_map = unit_sprites.graphic_map
    unit_sprites._scaled_cache.clear()
    if transform is not None:
        table = copy.deepcopy(real_map())
        table[const]["pieces"] = transform(table[const]["pieces"])
        unit_sprites.graphic_map = lambda: table
    try:
        yield
    finally:
        unit_sprites.graphic_map = real_map
        unit_sprites._scaled_cache.clear()


def _keep_only(keep: frozenset[int]):
    def transform(pieces):
        out = []
        for index, piece in enumerate(pieces):
            if index in keep:
                out.append(piece)
            else:
                hidden = {k: v for k, v in piece.items() if k != "parent"}
                out.append({**hidden, "file_name": _HIDDEN_FILE})
        return out
    return transform


def _transform(patch):
    if patch is None:
        return None, None
    kind, const, *rest = patch
    if kind == "keep":
        return const, _keep_only(rest[0])
    assert kind == "no-slots"
    return const, _no_slots()


# Bounded: each render is ~2 MB and the module makes ~140 distinct ones.
@functools.lru_cache(maxsize=48)
def _render(style: str, placed: tuple, patch: tuple | None = None, terrain_id: int = GROUND_TERRAIN) -> np.ndarray:
    """One style's composite over _box(style). `placed` is ((const, x, y), ...)
    in list order; `patch` is ("keep", const, indices) or ("no-slots", const)."""
    const, transform = _transform(patch)
    scn = _scenario(placed, terrain_id)
    mm = scn.map_manager
    box = _box(style)
    with _patched(const, transform):
        if style == "flat":
            draws = render._flat_unit_draws(scn, TILE_PX)
            icons, _rows = render._flat_icon_layer(scn, TILE_PX)
            return render.composite_rect_flat(scn, *box, TILE_PX, unit_draws=draws, icons=icons)
        proj = _projection(style)
        elevations = np.zeros((mm.map_height, mm.map_width), dtype=np.int64)
        corner_rise = (
            ig.corner_rise_px(elevations, proj, rule=render.SLOPE_CORNER_RULE) if style == "sloped" else None
        )
        sprites = render.sprite_draws_by_anchor(scn, proj, elevations, corner_rise=corner_rise)
        units_by_tile = render._units_by_tile(scn)
        bboxes = render.merge_sprite_bboxes(
            render._building_bboxes_iso(scn, mm.map_width, mm.map_height, proj, elevations), sprites
        )
        if style == "sloped":
            return render.composite_rect_sloped(
                scn, *box, corner_rise, proj, TILE_PX, units_by_tile, bboxes, with_units=True, sprites=sprites
            )
        return render.composite_rect_iso(
            scn, *box, elevations, proj, TILE_PX, units_by_tile, bboxes, with_units=True, sprites=sprites
        )


# ------------------------------------------------------------------- oracle


def _changed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.any(a != b, axis=2)


def _eroded(mask: np.ndarray) -> np.ndarray:
    """3x3 erosion: drops the antialiased rim, where blending makes BU match neither side."""
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = mask.copy()
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy:dy + h, dx:dx + w]
    return out


def _unit_mask(style: str, unit: tuple) -> np.ndarray:
    """The unit's opaque ink: changed from bare ground, and the same over both grounds."""
    u_img = _render(style, (unit,))
    opaque = ~_changed(u_img, _render(style, (unit,), terrain_id=ALT_GROUND_TERRAIN))
    return _eroded(_changed(u_img, _render(style, ())) & opaque)


def _depth_key(tile: tuple[int, int]) -> tuple[int, int]:
    # The depth walk's (d = y - x, x) order, as unit_sprites.sprite_anchor_tile sorts.
    return tile[1] - tile[0], tile[0]


def _unit_tile(const: int, x: float, y: float) -> tuple[int, int]:
    mm = _base().map_manager
    return unit_sprites.sprite_anchor_tile(render.unit_occupied_tiles(_unit(const, x, y), mm.map_width, mm.map_height))


def _slot_tiles(const: int) -> list[tuple[int, int]]:
    """Each piece's depth-slot tile for a building centred on CENTRE, from the committed slots."""
    mm = _base().map_manager
    occupied = render.unit_occupied_tiles(_unit(const, CENTRE, CENTRE), mm.map_width, mm.map_height)
    x0, y0 = min(t[0] for t in occupied), min(t[1] for t in occupied)
    return [(x0 + p["slot"][0], y0 + p["slot"][1]) for p in unit_sprites.graphic_map()[const]["pieces"]]


def _piece_ink(style: str, const: int, keep: frozenset[int]) -> np.ndarray:
    """Every pixel the `keep` pieces touch, rim included (not eroded)."""
    building = ((const, CENTRE, CENTRE),)
    if not keep:
        return np.zeros(_render(style, ()).shape[:2], dtype=bool)
    # A pasture's drape stays when its pieces are hidden, so the drape alone is the baseline.
    ground = (
        _render(style, building, ("keep", const, frozenset()))
        if const in render.DRAPED_SPRITE_CONSTS
        else _render(style, ())
    )
    return _changed(_render(style, building, ("keep", const, keep)), ground)


@functools.lru_cache(maxsize=16)
def _oracle(style: str, const: int, unit: tuple, unit_first: bool, inject: str | None = None):
    """(front, back, bad_front, bad_back, front_ink) for one unit beside one building.
    Pieces sharing the unit's slot tile are in neither set: list order decides those."""
    building = (const, CENTRE, CENTRE)
    placed = (unit, building) if unit_first else (building, unit)
    patch = None if inject is None else (inject, const)
    unit_key = _depth_key(_unit_tile(*unit))
    keys = [_depth_key(t) for t in _slot_tiles(const)]
    front_ids = frozenset(i for i, k in enumerate(keys) if k > unit_key)
    back_ids = frozenset(i for i, k in enumerate(keys) if k < unit_key)
    tie_ids = frozenset(i for i, k in enumerate(keys) if k == unit_key)

    b_img, u_img = _render(style, (building,)), _render(style, (unit,))
    bu_img = _render(style, placed, patch)
    unit_mask = _unit_mask(style, unit)
    front_ink = _piece_ink(style, const, front_ids)
    tie_ink = _piece_ink(style, const, tie_ids)
    front = unit_mask & _eroded(front_ink) & ~tie_ink
    back = unit_mask & _eroded(_piece_ink(style, const, back_ids)) & ~front_ink & ~tie_ink
    return (
        front, back,
        front & _changed(bu_img, b_img),
        back & _changed(bu_img, u_img),
        front_ink,
    )


# ------------------------------------------------------ town centre + unit


def _tc_unit_tiles() -> list[tuple[int, int]]:
    """Footprint tiles strictly between the town centre's back and front slots."""
    slots = _slot_tiles(TOWN_CENTRE)
    keys = sorted({_depth_key(t) for t in slots})
    assert len(keys) >= 2, f"town centre slots collapsed onto one tile: {slots}"
    mm = _base().map_manager
    footprint = render.unit_occupied_tiles(_unit(TOWN_CENTRE, CENTRE, CENTRE), mm.map_width, mm.map_height)
    return sorted(t for t in footprint if keys[0] < _depth_key(t) < keys[-1] and _depth_key(t) not in keys)


def _tc_unit(tile: tuple[int, int]) -> tuple:
    return (WAR_ELEPHANT, tile[0] + TC_UNIT_OFFSET, tile[1] + TC_UNIT_OFFSET)


def test_a_unit_inside_a_town_centre_paints_between_its_back_and_front_pieces():
    _require_install()
    tiles = _tc_unit_tiles()
    assert tiles, "no footprint tile sorts between the town centre's back and front slots"
    for tile in tiles:
        unit = _tc_unit(tile)
        assert _unit_tile(*unit) == tile
        for unit_first in (True, False):
            front, back, bad_front, bad_back, _ = _oracle("stepped", TOWN_CENTRE, unit, unit_first)
            where = f"unit on {tile}, {'unit' if unit_first else 'town centre'} listed first"
            assert front.any() and back.any(), (
                f"{where}: overlap front {int(front.sum())} px, back {int(back.sum())} px; the check needs both"
            )
            assert not bad_front.any(), f"{where}: {int(bad_front.sum())} px of it paint over a front piece"
            assert not bad_back.any(), f"{where}: {int(bad_back.sum())} px of it hidden by a back piece"


def test_the_no_slots_inject_breaks_the_town_centre_oracle():
    """The control: with every piece back on the single anchor tile, the main
    piece paints over the unit, so the back check must catch it."""
    _require_install()
    for tile in _tc_unit_tiles():
        _front, back, bad_front, bad_back, _ = _oracle(
            "stepped", TOWN_CENTRE, _tc_unit(tile), True, inject="no-slots"
        )
        assert back.any()
        assert bad_front.any() or bad_back.any(), f"unit on {tile}: the no-slots inject left the oracle green"


# ---------------------------------------------------------- pasture corners

# Corner name -> (sign of the post's mx, sign of its my).
CORNERS = {"back": (1, -1), "right": (1, 1), "left": (-1, -1), "front": (-1, 1)}
# A unit sharing a post's slot tile sorts by list order, not by where it stands.
_TIE_BY_LIST_ORDER = "a unit on a corner post's own slot tile sorts by list order, not depth"
_KNOWN_TIE_FAILURES = {("back", "inside", "unit-first"), ("front", "inside", "pasture-first")}


def _post(corner: str) -> tuple[int, dict]:
    pieces = unit_sprites.graphic_map()[PASTURE]["pieces"]
    sx, sy = CORNERS[corner]
    hits = [
        (i, p) for i, p in enumerate(pieces)
        if p["unit_id"] == CORNER_POST and np.sign(p["mx"]) == sx and np.sign(p["my"]) == sy
    ]
    assert len(hits) == 1, f"expected one {corner} corner post in 1897's pieces, found {len(hits)}"
    return hits[0]


def _corner_unit(corner: str, side: str) -> tuple:
    _index, post = _post(corner)
    sx, sy = CORNERS[corner]
    step = CORNER_NUDGE if side == "outside" else -CORNER_NUDGE
    return (WAR_ELEPHANT, CENTRE + post["mx"] + sx * step, CENTRE + post["my"] + sy * step)


def _post_in_front(corner: str, unit: tuple) -> bool:
    """Slot order, or on a shared slot tile, which of the two stands nearer the camera."""
    index, post = _post(corner)
    post_key = _depth_key(_slot_tiles(PASTURE)[index])
    unit_key = _depth_key(_unit_tile(*unit))
    if post_key != unit_key:
        return post_key > unit_key
    _const, ux, uy = unit
    return (uy - ux) < post["my"] - post["mx"]


def _corner_cases():
    for style in ("stepped", "sloped"):
        for corner in CORNERS:
            for side in ("inside", "outside"):
                for order in ("unit-first", "pasture-first"):
                    marks = (
                        [pytest.mark.xfail(strict=True, reason=_TIE_BY_LIST_ORDER)]
                        if (corner, side, order) in _KNOWN_TIE_FAILURES
                        else []
                    )
                    yield pytest.param(style, corner, side, order, marks=marks, id=f"{style}-{corner}-{side}-{order}")


def _post_check(style: str, corner: str, side: str, unit_first: bool, inject: str | None = None):
    """(overlap, wrong): the unit's ink over the corner post, and the part of
    it where BU matches the wrong side for _post_in_front()."""
    unit = _corner_unit(corner, side)
    index, _piece = _post(corner)
    building = (PASTURE, CENTRE, CENTRE)
    placed = (unit, building) if unit_first else (building, unit)
    patch = None if inject is None else (inject, PASTURE)
    b_img, u_img = _render(style, (building,)), _render(style, (unit,))
    bu_img = _render(style, placed, patch)
    overlap = _unit_mask(style, unit) & _eroded(_piece_ink(style, PASTURE, frozenset({index})))
    if _post_in_front(corner, unit):
        return overlap, overlap & _changed(bu_img, b_img)
    # Other pieces on the post's own slot tile also sort by list order; leave them out.
    slots = _slot_tiles(PASTURE)
    others = frozenset(i for i, t in enumerate(slots) if t == slots[index] and i != index)
    front_ink = _oracle(style, PASTURE, unit, unit_first)[4]
    return overlap, overlap & ~front_ink & ~_piece_ink(style, PASTURE, others) & _changed(bu_img, u_img)


@pytest.mark.parametrize(("style", "corner", "side", "order"), list(_corner_cases()))
def test_a_unit_at_a_pasture_corner_sorts_against_that_corner_post(style, corner, side, order):
    _require_install()
    unit_first = order == "unit-first"
    _front, _back, bad_front, bad_back, _ = _oracle(style, PASTURE, _corner_unit(corner, side), unit_first)
    assert not bad_front.any(), f"{int(bad_front.sum())} px of the unit paint over a front piece"
    assert not bad_back.any(), f"{int(bad_back.sum())} px of the unit hidden by a back piece"
    overlap, wrong = _post_check(style, corner, side, unit_first)
    assert overlap.any(), f"the unit {side} the {corner} corner does not overlap its post"
    assert not wrong.any(), f"{int(wrong.sum())} of {int(overlap.sum())} px sort the wrong way against the post"


@pytest.mark.parametrize("style", ["stepped", "sloped"])
def test_the_no_slots_inject_breaks_the_pasture_oracle(style):
    """The control. Under no-slots every piece paints on the footprint's last
    tile, so the back post and the pieces behind an inside unit move over it."""
    _require_install()
    broken = []
    for corner in CORNERS:
        for side in ("inside", "outside"):
            for unit_first in (True, False):
                *_masks, bad_front, bad_back, _ = _oracle(
                    style, PASTURE, _corner_unit(corner, side), unit_first, inject="no-slots"
                )
                _overlap, wrong = _post_check(style, corner, side, unit_first, inject="no-slots")
                if bad_front.any() or bad_back.any() or wrong.any():
                    broken.append((corner, side, unit_first))
    assert ("back", "inside", False) in broken, f"the no-slots inject broke only {broken}"
    assert ("front", "inside", True) in broken, f"the no-slots inject broke only {broken}"


def test_flat_draws_the_pasture_and_every_corner_unit():
    """Flat has no depth walk, so only presence is checked."""
    _require_install()
    building = (PASTURE, CENTRE, CENTRE)
    terrain, b_img = _render("flat", ()), _render("flat", (building,))
    x0, y0 = _box("flat")[:2]

    def tile_view(img, tx, ty):
        return img[ty * TILE_PX - y0:(ty + 1) * TILE_PX - y0, tx * TILE_PX - x0:(tx + 1) * TILE_PX - x0]

    footprint = (slice(58 * TILE_PX - y0, 62 * TILE_PX - y0), slice(58 * TILE_PX - x0, 62 * TILE_PX - x0))
    assert _changed(b_img[footprint], terrain[footprint]).any(), "the pasture paints nothing in Flat"
    for corner in CORNERS:
        for side in ("inside", "outside"):
            unit = _corner_unit(corner, side)
            tx, ty = _unit_tile(*unit)
            bu_img = _render("flat", (building, unit))
            assert _changed(tile_view(bu_img, tx, ty), tile_view(b_img, tx, ty)).any(), (
                f"the unit {side} the {corner} corner is missing in Flat"
            )
