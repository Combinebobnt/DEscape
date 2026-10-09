"""Corpus tier for descape.trigger_status (GH #166): every condition and
effect in the real examples/ corpus is evaluated against the committed
requirements table. Nothing may raise, and no type with at least 20 uses may
have more than 5% of its entries flagged, apart from the entries listed
below, each checked by hand.

The listed entries must also still be flagged, and the EXPECTED_UNCHECKED
ones still UNCHECKED, so the lists cannot go stale silently. The full-corpus false-positive audit is
tools/census_trigger_fields.py --status; this test covers examples/ only.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from descape import library_compat, trigger_status
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.unit_references import build_reference_index

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
MIN_USES = 20
MAX_FLAGGED_SHARE = 0.05

# (file, trigger index, "conditions"/"effects", entry index) -> why it is broken.
_8TP = "8tp3w9j.aoe2scenario"
_HALF_AREA = "area x1=x2=19 but y1=y2=-1: half set"
CONFIRMED_BROKEN = {
    (_8TP, 8, "effects", 0): "type 0, no effect chosen",
    (_8TP, 8, "effects", 11): "type 0, no effect chosen",
    (_8TP, 8, "effects", 22): "type 0, no effect chosen",
    (_8TP, 30, "effects", 2): "type 0, no effect chosen",
    (_8TP, 9, "effects", 0): _HALF_AREA,
    (_8TP, 9, "effects", 1): _HALF_AREA,
    (_8TP, 14, "effects", 4): _HALF_AREA,
    (_8TP, 14, "effects", 5): _HALF_AREA,
    (_8TP, 14, "effects", 6): _HALF_AREA,
    (_8TP, 25, "conditions", 0): "unit 8664 was deleted (below the file's next unit id)",
    (_8TP, 29, "effects", 0): "unit 44174 never existed in this file (above its next unit id), and " + _HALF_AREA,
    (_8TP, 29, "effects", 1): "unit 44174 never existed in this file (above its next unit id), and " + _HALF_AREA,
    ("F7_2_Dos Pilas (648).aoe2scenario", 19, "conditions", 0): "unit 43279 was deleted",
    ("old-allies-final-v2.aoe2scenario", 57, "effects", 2): "patrol with no location; its 5 sibling patrols have one",
}

# Incomplete on purpose: a divider's condition that can never fire. Only the
# divider row itself is unstyled; its entry rows are styled like any other.
INCOMPLETE_BY_DESIGN = {
    (_8TP, 13, "conditions", 0): "trigger_active with no trigger, in the begin-scenario divider",
    ("old-allies-final-v2.aoe2scenario", 2, "conditions", 0): "destroy_object with no unit, in a divider",
}

LISTED = {**CONFIRMED_BROKEN, **INCOMPLETE_BY_DESIGN}

# An unknown type id is UNCHECKED, not flagged (user decision, 2026-10-07).
EXPECTED_UNCHECKED = {
    ("R4_LeLoi_4.aoe2scenario", 63, "effects", 0): (
        "effect type 91, which no shipped vocabulary lists, with message ACHIEVEMENT_NOT_THE_VIPER; "
        "it looks deliberate (an achievement hook), unverified in-game"
    ),
}


def _sweep(files: list[Path]) -> tuple[Counter, Counter, set, set]:
    """(uses, flagged-and-unlisted counts per (kind key, type), flagged keys, unchecked keys)."""
    uses: Counter = Counter()
    unlisted: Counter = Counter()
    flagged: set = set()
    unchecked: set = set()
    for path in files:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
        if manager is None or not library_compat.vocabulary_is_available(loaded.scenario_version):
            continue
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        index = build_reference_index(loaded)
        context = trigger_status.StatusContext(
            vocabulary=vocabulary,
            trigger_ids=frozenset(range(len(manager.triggers))),
            unit_exists=lambda ref_id, by_id=index.by_id: ref_id in by_id,
        )
        for trigger_index, trigger in enumerate(manager.triggers):
            try:
                result = trigger_status.evaluate_trigger(trigger, context)
            except Exception as e:  # noqa: BLE001 -- name the trigger that raised
                pytest.fail(f"{path.name} trigger {trigger_index}: {type(e).__name__}: {e}")
            for kind, kind_key in ((trigger_status.CONDITION, "conditions"), (trigger_status.EFFECT, "effects")):
                entries = list(getattr(trigger, kind_key))
                for entry_index, (entry, status) in enumerate(zip(entries, getattr(result, kind_key), strict=True)):
                    definition = trigger_status.definition_for(kind, entry, vocabulary)
                    type_id = trigger_status.read_value(entry, f"{kind}_type")
                    name = definition.name if definition is not None else f"type {type_id}"
                    uses[(kind_key, name)] += 1
                    key = (path.name, trigger_index, kind_key, entry_index)
                    if status.status is trigger_status.Status.UNCHECKED:
                        unchecked.add(key)
                    if status.status is not trigger_status.Status.PROBLEM:
                        continue
                    flagged.add(key)
                    if key not in LISTED:
                        unlisted[(kind_key, name)] += 1
    return uses, unlisted, flagged, unchecked


@pytest.mark.corpus
def test_examples_entry_status_never_raises_and_rarely_flags() -> None:
    files = sorted(EXAMPLES.glob("*.aoe2scenario"))
    if not files:
        pytest.skip(f"no .aoe2scenario files in {EXAMPLES}")
    uses, unlisted, flagged, unchecked = _sweep(files)
    assert uses, "no file in examples/ had a parseable trigger section"

    present = {path.name for path in files}
    stale = [key for key in LISTED if key[0] in present and key not in flagged]
    assert not stale, f"listed entries no longer flagged, update the lists: {stale}"
    not_unchecked = [key for key in EXPECTED_UNCHECKED if key[0] in present and key not in unchecked]
    assert not not_unchecked, f"expected UNCHECKED, update the list: {not_unchecked}"

    over = {
        type_key: f"{unlisted[type_key]}/{count}"
        for type_key, count in uses.items()
        if count >= MIN_USES and unlisted[type_key] > MAX_FLAGGED_SHARE * count
    }
    assert not over, f"types flagged on more than {MAX_FLAGGED_SHARE:.0%} of their uses: {over}"
