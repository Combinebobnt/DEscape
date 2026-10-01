"""render.patch_bystander_grid() against build_bystander_grid() of the same
dict, over random bboxes: negative coordinates, edges exactly on a cell
boundary, multi-cell spans, and batches mixing add, remove, move and
unchanged keys. Entry order within a cell is compared as sorted, since the
patch appends where the build walks in dict order (see the patch docstring)."""

from __future__ import annotations

import random

import pytest

from descape import render

CELL = 64


def grid_state(grid: render.BystanderGrid | None) -> dict:
    """A grid's cells with entry order within a cell sorted away. Tuples, not
    frozensets, so a duplicated entry still fails. The splice oracles use it too."""
    return {} if grid is None else {cell: tuple(sorted(entries)) for cell, entries in grid.cells.items()}


def _rand_bbox(rng: random.Random) -> tuple[int, int, int, int]:
    """Edges snap to a cell boundary half the time, so ux1 == k * CELL is common."""

    def edge() -> int:
        return rng.randrange(-3, 8) * CELL if rng.random() < 0.5 else rng.randrange(-3 * CELL, 8 * CELL)

    x0, y0 = edge(), edge()
    x1 = max(x0 + 1, edge()) if rng.random() < 0.5 else x0 + rng.choice((1, CELL, CELL + 1, 3 * CELL))
    y1 = max(y0 + 1, edge()) if rng.random() < 0.5 else y0 + rng.choice((1, CELL, CELL + 1, 3 * CELL))
    return x0, y0, x1, y1


def _mutate(rng: random.Random, bboxes: dict) -> dict:
    """Applies one random batch to bboxes in place, returning old_bboxes as
    _reanchor_units() records it: every touched key, None if it was absent."""
    old: dict = {}
    keys = list(bboxes)
    for _ in range(rng.randrange(1, 12)):
        op = rng.choice(("add", "remove", "move", "unchanged"))
        if op == "add" or not keys:
            key = (rng.randrange(-5, 40), rng.randrange(-5, 40))
        else:
            key = rng.choice(keys)
        old.setdefault(key, bboxes.get(key))
        if op == "remove":
            bboxes.pop(key, None)
        elif op in ("add", "move"):
            bboxes[key] = _rand_bbox(rng)
    return old


@pytest.mark.parametrize("seed", range(200))
def test_patch_matches_a_full_build(seed):
    rng = random.Random(seed)
    bboxes = {(rng.randrange(40), rng.randrange(40)): _rand_bbox(rng) for _ in range(rng.randrange(0, 40))}
    grid = render.build_bystander_grid(bboxes, CELL)
    for _ in range(5):
        before = grid_state(grid)
        old = _mutate(rng, bboxes)
        patched = render.patch_bystander_grid(grid, old, bboxes)
        assert grid_state(grid) == before, "the patch mutated its input grid"
        assert patched.cell_px == CELL
        assert all(patched.cells.values()), "an empty cell survived the patch"
        assert grid_state(patched) == grid_state(render.build_bystander_grid(bboxes, CELL))
        grid = patched


def test_an_all_unchanged_batch_returns_the_same_grid():
    bboxes = {(1, 1): (-10, -10, CELL, CELL), (2, 2): (0, 0, 1, 1)}
    grid = render.build_bystander_grid(bboxes, CELL)
    old = {(1, 1): bboxes[(1, 1)], (2, 2): bboxes[(2, 2)], (3, 3): None}
    assert render.patch_bystander_grid(grid, old, bboxes) is grid


def test_removing_a_cells_only_entry_deletes_the_cell():
    bboxes = {(1, 1): (5 * CELL, 5 * CELL, 5 * CELL + 1, 5 * CELL + 1)}
    grid = render.build_bystander_grid(bboxes, CELL)
    old = {(1, 1): bboxes.pop((1, 1))}
    assert render.patch_bystander_grid(grid, old, bboxes).cells == {}
