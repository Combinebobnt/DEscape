"""AoE2:DE's importer encodings for pre-DE (header-2) scenarios, Qt-free.

A legacy file shown with DE data stores two values in an older encoding:
starting age and tree variants. DEscape reads them the way AoE2:DE's own
importer rewrites them, for the bands with a measured DE-import pair only
(a legacy file next to the game's own resave of it). Stored values never
change: ages are translated at read (player_fields.shown_value()), trees
through a draw-only rotation override beside the wall shapes
(render_units.wall_variant_rotation_overrides()). The one write path that
uses them is File > Convert to .aoe2scenario (legacy_convert), which writes
a new DE document and applies them to every band, reporting a band outside
MEASURED_BANDS as unmeasured.
"""

from __future__ import annotations

from descape import scenario_format

# ScenarioFormat.game bands whose import pair has been measured. A band joins
# only alongside its pair in tests/test_legacy_oracle.py's PAIRS:
# AOC: examples/atilla_1.scn and ring75_v0.scx against their
# *_resaved.aoe2scenario (2026-11-23).
MEASURED_BANDS = frozenset({scenario_format.AOC})

U32_MINUS_ONE = 0xFFFFFFFF
LEGACY_AGES = range(5)  # Dark .. Post-Imperial
DE_AGE_OFFSET = 2

# Bamboo, oak, pine, palm, oak (forest), snow pine: every changed unit in the
# AoC pairs went from v to one of 3v, 3v+1, 3v+2.
TREE_CONSTS = frozenset({348, 349, 350, 351, 411, 413})
LEGACY_TREE_VARIANTS = 14


def applies(loaded) -> bool:
    """Whether `loaded` is a legacy file of a measured band."""
    fmt = getattr(loaded, "source_format", None)
    return bool(fmt is not None and fmt.legacy and fmt.game in MEASURED_BANDS)


def starting_age_to_de(raw):
    """A legacy StartingAge as DE stores it: -1 / u32 -1 ("none") is 0, 0..4
    is 2..6. Anything else passes through, so the map is not injective
    outside those values (to_de(4) == to_de(6) == 6)."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return raw
    if raw in (-1, U32_MINUS_ONE):
        return 0
    if raw in LEGACY_AGES:
        return raw + DE_AGE_OFFSET
    return raw


def starting_age_from_de(value):
    """The inverse for a legacy write: 0 is the stored u32 -1, 2..6 is 0..4.
    Anything else passes through."""
    if isinstance(value, bool) or not isinstance(value, int):
        return value
    if value == 0:
        return U32_MINUS_ONE
    if value - DE_AGE_OFFSET in LEGACY_AGES:
        return value - DE_AGE_OFFSET
    return value


def tree_variant(rotation, reference_id: int) -> float | None:
    """The DE variant a stored legacy tree variant draws as: 3v plus
    reference_id % 3, one of the three values the importer picks from at
    random. None (draw the stored value) unless `rotation` is an exact
    integer in 0..13."""
    try:
        v = float(rotation)
    except (TypeError, ValueError):
        return None
    if not v.is_integer() or not 0 <= v < LEGACY_TREE_VARIANTS:
        return None
    return float(3 * int(v) + int(reference_id) % 3)


def drawn_rotation_overrides(scenario) -> dict[tuple[int, int], float]:
    """Per-(player_id, index) drawn rotation for every legacy tree, keyed like
    the wall overrides. {} unless applies()."""
    if not applies(scenario):
        return {}
    overrides: dict[tuple[int, int], float] = {}
    for player_id, units in enumerate(scenario.unit_manager.units):
        for i, unit in enumerate(units):
            if unit.unit_const not in TREE_CONSTS:
                continue
            drawn = tree_variant(unit.rotation, unit.reference_id)
            if drawn is not None:
                overrides[(player_id, i)] = drawn
    return overrides
