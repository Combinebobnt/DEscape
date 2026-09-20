"""Pins descape.unit_kind's two derived const sets -- GH #65's Show Walls /
Show Eye Candy.

Both sets are derived from the committed .dat tables rather than transcribed,
so the thing worth testing is not their contents but their BOUNDARIES: that
`class == 27` alone was not used (it drags in 27 invisible scaffolding
consts), that gates are in, that trees / resources / cliffs are out, and that
the one deliberate divergence from unit_sprites.wall_connector_consts() is
exactly Aqueduct. Each of those is a decision the module's docstring argues
for; a test here is what stops a later session quietly re-deciding it.

The corpus-marked test at the bottom re-measures the coverage claim against
real files, the same shape tests/test_unit_rotation.py uses for the same
reason: the sets stay pinned to data rather than to one session's judgement.
"""

from __future__ import annotations

import pytest

from descape import unit_kind
from descape.terrain_palette import TREE_UNIT_IDS

# Hand-picked representatives, each named for why it is in this list.
WALL2 = 117  # the corpus's most-placed wall, 4437 placements
AQUEDUCT = 231  # class 27, angle_count 5 -- the connector-set divergence
FENCE = 1062  # the low-placement-count wall, in on structure not on volume
STONE_GATE_CLOSED = 64  # one of the 96 gate consts, orientation lives here
TWAL = 208  # class 27, NO graphic entry, 0 placements -- stays out
SHEEP_ANNEX1 = 1694  # class 27 scaffolding, exactly what angle_count filters
EMPTY_TC_ANNEX = 890  # class 27 with corpus placements, still not a wall

GRASS_GREEN = 1358  # class 14, type 10 -- the TODO entry's correct example
GRASS_DRY = 1359
HRICH_D = 647  # the TODO entry's WRONG example: class 11, type 30, 0 placements
GOLD_MINE = 66  # GOLDM -- a resource, the thing the user is looking *for*
STONE_MINE = 102  # STONM, class 8
BERRY_BUSH = 59  # FORAG, class 7
FLARE = 112  # class 30, type 10 -- in, and deliberately so


def test_wall_consts_is_the_nine_walls_plus_ninety_six_gates() -> None:
    """105 is not a magic number: it is 9 real walls (class 27 at
    angle_count 5) plus gate_orientation's own 24 complete groups of 4."""
    from descape import gate_orientation

    walls = unit_kind.wall_consts()
    gates = {const for group in gate_orientation.groups().values() for const in group}
    assert len(gates) == 96
    assert gates <= walls
    assert len(walls) == 105
    assert walls - gates == {72, 117, 119, 155, 231, 370, 788, 1062, 2678}


def test_class_27_alone_is_not_the_basis() -> None:
    """The angle_count == 5 half of the derivation is load-bearing: without
    it the set drags in 27 invisible scaffolding consts. Sheep annex1 is one
    of them; Empty TC annex is the one that actually has corpus placements,
    which is what makes the omission checkable rather than theoretical."""
    walls = unit_kind.wall_consts()
    assert SHEEP_ANNEX1 not in walls
    assert EMPTY_TC_ANNEX not in walls


def test_twal_has_no_graphic_entry_and_stays_out() -> None:
    """208 is class 27, so a class-only derivation would include it, but it
    has no unit_graphic_map.json entry at all and zero corpus placements."""
    assert TWAL not in unit_kind.wall_consts()


def test_the_only_divergence_from_the_connector_set_is_aqueduct() -> None:
    """unit_sprites.wall_connector_consts() answers a different question --
    which consts get a stored index RE-DERIVED from a neighbour mask, a
    write-path concern -- and AGENTS.md carries an open [NEEDS DECISION] on
    whether Aqueduct belongs there. That question does not transfer: hiding
    an aqueduct when you asked to hide walls is what you want. Pinning the
    delta in both directions is what keeps the divergence asserted rather
    than accidental, and stops a later session "fixing" either set to match
    the other."""
    from descape import unit_sprites

    walls = unit_kind.wall_consts()
    connectors = unit_sprites.wall_connector_consts()
    assert walls - connectors == {AQUEDUCT}
    assert connectors - walls == set()


def test_wall_consts_matches_unit_sprites_rotation_variant_consts_plus_aqueduct() -> None:
    """The 8-const frozenset unit_sprites hand-keeps is re-derived here
    rather than imported (that module pulls numpy and a configured install,
    which would take unit_filter.py off the leaf list). This is the check
    that the two cannot silently drift apart."""
    from descape import gate_orientation, unit_sprites

    gates = {const for group in gate_orientation.groups().values() for const in group}
    assert unit_kind.wall_consts() - gates - {AQUEDUCT} == set(unit_sprites._ROTATION_VARIANT_CONSTS)


def test_eye_candy_consts_excludes_trees_resources_and_cliffs() -> None:
    import json
    from pathlib import Path

    candy = unit_kind.eye_candy_consts()
    assert len(candy) == 395
    assert candy.isdisjoint(TREE_UNIT_IDS), "show_trees owns those"

    objects = json.loads((Path(unit_kind.__file__).parent / "object_catalog.json").read_text())["objects"]
    classes = {int(c): e.get("class") for c, e in objects.items()}
    assert all(classes.get(c) not in unit_kind._RESOURCE_CLASSES for c in candy)
    assert all(classes.get(c) != unit_kind._CLIFF_CLASS for c in candy)
    assert GOLD_MINE not in candy


def test_the_todo_entrys_const_pointer_is_corrected() -> None:
    """The TODO entry said "grass, flowers and shrubs (consts 647/1358/1359)".
    1358/1359 are right; 647 is HRICH_D, class 11, **type 30**, hide_in_editor,
    with zero corpus placements and no graphic entry. It does not classify as
    eye candy and must not be hard-included to make that list true."""
    candy = unit_kind.eye_candy_consts()
    assert GRASS_GREEN in candy
    assert GRASS_DRY in candy
    assert HRICH_D not in candy


def test_flares_are_in_and_that_is_deliberate() -> None:
    """The one arguable consequence of the measured derivation. Flares are
    cosmetic markers, so hiding them under "eye candy" is right; carving them
    out by hand would break the measured-not-judged property. The real flags
    are type 20 and never in the set at all."""
    candy = unit_kind.eye_candy_consts()
    assert FLARE in candy
    assert 1150 not in candy and 1151 not in candy and 1307 not in candy


def test_the_two_sets_are_non_empty_and_disjoint() -> None:
    """A wall is never eye candy, so no unit can be hidden by both toggles
    for two different reasons -- which keeps _filter_summary()'s per-toggle
    wording honest."""
    walls = unit_kind.wall_consts()
    candy = unit_kind.eye_candy_consts()
    assert walls and candy
    assert walls.isdisjoint(candy)


def test_the_module_stays_a_stdlib_only_leaf() -> None:
    """unit_filter.py imports this, and view_layers.py:6-8 records that
    unit_filter staying a leaf is load-bearing. numpy or asset_source
    arriving here through unit_sprites is the regression to catch."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path(unit_kind.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            if node.module.startswith("descape."):
                imported.add(node.module)
            elif node.module == "descape":
                # `from descape import gate_orientation` -- node.module is
                # bare "descape", so the branch above never sees the sibling
                # being imported. Without this the whole check is vacuous.
                imported.update(f"descape.{a.name}" for a in node.names)
    assert "numpy" not in imported
    assert not {m for m in imported if m.startswith("descape.")} - {
        "descape.terrain_palette",
        "descape.gate_orientation",
    }


@pytest.mark.corpus
def test_the_wall_set_covers_the_corpus_class_27_placements(corpus_files) -> None:
    """Re-measures the plan's coverage claim rather than trusting it: over
    the corpus, wall_consts() must account for essentially every class-27
    placement, and the residue must be only the named non-walls.

    Measured 2026-09-20 over the 20-file examples/ corpus: 8204 class-27
    placements, 8193 covered (99.87%), residue exactly Empty TC annex (10)
    and Mole under construction (1)."""
    import json
    from collections import Counter
    from pathlib import Path

    from descape.scenario_io import load_map_and_units

    objects = json.loads((Path(unit_kind.__file__).parent / "object_catalog.json").read_text())["objects"]
    class_27 = {int(c) for c, e in objects.items() if e.get("class") == unit_kind._WALL_CLASS}
    walls = unit_kind.wall_consts()

    placed: Counter[int] = Counter()
    for path in corpus_files:
        loaded = load_map_and_units(path)
        for unit in loaded.unit_manager.get_all_units():
            if unit.unit_const in class_27:
                placed[unit.unit_const] += 1

    assert placed, "no class-27 placements found -- this test would prove nothing"
    total = sum(placed.values())
    covered = sum(n for const, n in placed.items() if const in walls)
    assert covered / total > 0.99, f"wall_consts() covered only {covered}/{total} class-27 placements"
    assert set(placed) - walls <= {EMPTY_TC_ANNEX, 2421}, "an unexpected class-27 const is uncovered"


@pytest.mark.corpus
def test_the_eye_candy_set_hides_decoratives_and_keeps_resources(corpus_files) -> None:
    """The other half of the measured claim: every corpus file that carries
    units at all carries a substantial eye-candy population.

    Measured 2026-09-20 over the 20-file examples/ corpus: 18 files hold units,
    and of those, 17 hold 224-3093 eye-candy placements. The 18th,
    ring75_v0_scx_resaved, is an .scx resave holding 1843 units and zero of
    either kind -- which is why the bound is per-file-conditional rather than a
    flat minimum. The GH #65 plan's "1030-3093 per file" was measured over a
    subset and is too narrow.
    """
    from collections import Counter

    from descape.scenario_io import load_map_and_units

    candy = unit_kind.eye_candy_consts()
    measured = {}
    for path in corpus_files:
        loaded = load_map_and_units(path)
        counts: Counter[int] = Counter(u.unit_const for u in loaded.unit_manager.get_all_units())
        if not counts:
            continue  # blank_map.aoe2scenario, the empty template fixture
        measured[path.name] = sum(n for const, n in counts.items() if const in candy)

    assert measured, "no corpus file carried units -- this test would prove nothing"
    populated = [n for n in measured.values() if n]
    assert populated, f"no file carried any eye candy at all: {measured}"
    assert max(populated) > 200, f"the set may have collapsed: {measured}"
    assert len(populated) >= 0.8 * len(measured), f"most files should carry eye candy: {measured}"
    # Resources are type 10 too and must survive the subtraction.
    for resource in (GOLD_MINE, STONE_MINE, BERRY_BUSH):
        assert resource not in candy
