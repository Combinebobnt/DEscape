"""GH #315's oracle: AoE2:DE's own editor import. Each pair is a pre-DE
original and the DE editor's resave of it, in examples/ (gitignored, so this
skips elsewhere). The converted original must match the resave wherever the
importer is deterministic, and fall in the same rule set (legacy_convert.
unit_rule()) wherever it picks at random. Every difference DEscape does not
reproduce is listed here, never waved through.
"""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from descape import (
    ai_scripts,
    diplomacy_fields,
    game_dataset,
    games,
    legacy_convert,
    messages_fields,
    option_fields,
    player_fields,
    unit_sprites,
)
from descape.scenario_io import load_map_and_units

pytestmark = pytest.mark.corpus

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
# (original, DE resave, its Messages the resave replaced with DE campaign text).
PAIRS = {
    "atilla_1": ("atilla_1.scn", "atilla_1_scn_resaved.aoe2scenario", {"ascii_scouts"}),
    "ring75_v0": ("ring75_v0.scx", "ring75_v0_scx_resaved.aoe2scenario", set()),
}
# Walls DE stores as its neighbours give them (the 26 Attila misses are palisade runs stored as towers).
WALL_FLOOR = {"atilla_1": 514, "ring75_v0": 0}
# Type-80 units whose DE rotation the rule does not give (0): 805 is a dock variant, one tower 79.
ROTATION_EXCEPTIONS = {805: {7.0}, 79: {2.0}}
TRIBE_RENAMES = {("atilla_1", 5): "The Eastern Roman Empire"}
_NEXT_UNIT_ID = struct.Struct("<i")


@pytest.fixture(scope="module", params=sorted(PAIRS))
def pair(request):
    original, resave, replaced = PAIRS[request.param]
    if not (EXAMPLES / original).is_file() or not (EXAMPLES / resave).is_file():
        pytest.skip(f"no {original} / {resave} pair in examples/")
    legacy = load_map_and_units(EXAMPLES / original)
    new, report = legacy_convert.load_converted(legacy, rng=random.Random(7))
    return request.param, legacy, new, load_map_and_units(EXAMPLES / resave), replaced, report


def test_terrain_units_and_counter_match_the_resave(pair) -> None:
    _name, legacy, new, de, _replaced, report = pair
    assert not new.source_format.legacy
    tiles = [(t.terrain_id, t.layer, t.elevation) for t in new.map_manager.terrain]
    assert tiles == [(t.terrain_id, t.layer, t.elevation) for t in de.map_manager.terrain]

    def placed(loaded):
        fields = ("reference_id", "unit_const", "x", "y", "z", "status", "garrisoned_in_id")
        return [[tuple(getattr(u, f) for f in fields) for u in us] for us in loaded.unit_manager.units]

    assert placed(new) == placed(de)
    counter = _NEXT_UNIT_ID.unpack_from(new.decompressed_body, 0)[0]
    ids = [u.reference_id for us in new.unit_manager.units for u in us]
    assert counter >= _NEXT_UNIT_ID.unpack_from(legacy.decompressed_body, 0)[0] and counter > max(ids)
    assert report.triggers_left_out == legacy_convert.legacy_trigger_count(legacy)


def _in_rule_set(rule, const, rotation, frame) -> bool:
    frames = int((unit_sprites.graphic_map().get(const) or {}).get("frame_count") or 0)
    if rule == "trees":
        return rotation == frame
    if rule == "fish":
        return rotation == frame and 0 <= frame < frames
    if rule == "buildings":
        return (rotation == 0 or rotation in ROTATION_EXCEPTIONS.get(const, ())) and 0 <= frame < frames
    return True


def test_unit_images_fall_in_the_importers_rule_sets(pair) -> None:
    name, legacy, new, de, _replaced, _report = pair
    old = {u.reference_id: u for us in legacy.unit_manager.units for u in us}
    theirs = {u.reference_id: u for us in de.unit_manager.units for u in us}
    walls = 0
    diffs = []
    with game_dataset.activated(games.default().id):
        for units in new.unit_manager.units:
            for u in units:
                d, rule = theirs[u.reference_id], legacy_convert.unit_rule(u.unit_const)
                ours, want = (u.rotation, u.initial_animation_frame), (d.rotation, d.initial_animation_frame)
                if rule == "walls":
                    walls += unit_sprites.variant_index(u.rotation, 5) == unit_sprites.variant_index(d.rotation, 5)
                    ok = u.initial_animation_frame == d.initial_animation_frame
                elif rule == "trees":
                    ok = int(u.rotation) // 3 == int(d.rotation) // 3 == int(old[u.reference_id].rotation)
                elif rule in (None, "gates"):
                    ok = ours == want or (u.rotation == d.rotation and rule == "gates")
                else:
                    ok = True
                ok = ok and _in_rule_set(rule, u.unit_const, *ours) and _in_rule_set(rule, d.unit_const, *want)
                if not ok:
                    diffs.append((rule, u.unit_const, ours, want))
    assert not diffs, diffs[:20]
    assert walls >= WALL_FLOOR[name]


def test_players_options_and_diplomacy_match_the_resave(pair) -> None:
    name, _legacy, new, de, _replaced, _report = pair
    diffs = []
    for player_id in range(9):
        theirs = {s.field_id: s for s in player_fields.specs_for_player(de, player_id)}
        for spec in player_fields.specs_for_player(new, player_id):
            if spec.field_id == "personality" or spec.field_id not in theirs:
                continue
            a = player_fields.current_value(new, spec, player_id)
            b = player_fields.current_value(de, theirs[spec.field_id], player_id)
            if a != b and not (spec.field_id == "tribe_name" and TRIBE_RENAMES.get((name, player_id)) == b):
                diffs.append(f"p{player_id}.{spec.field_id}: ours {a!r}, DE {b!r}")
    ours = player_fields.stored_personalities(new, player_fields.personality_layout(new))
    want = player_fields.stored_personalities(de, player_fields.personality_layout(de))
    for player_id, a in ours.items():
        b = want[player_id]
        # DE writes its own RandomGame stub for a standard AI, DEscape its built-in standard one.
        same = a.ai_type == b.ai_type and (a.ai_type == ai_scripts.AI_TYPE_STANDARD or (a.stored_name, a.text) == (b.stored_name, b.text))
        if not same:
            diffs.append(f"p{player_id} AI: ours {a.stored_name!r}/{a.ai_type}, DE {b.stored_name!r}/{b.ai_type}")
    theirs = {s.field_id: s for s in option_fields.specs_for(de)}
    for spec in option_fields.specs_for(new):
        a, b = option_fields.current_value(new, spec), option_fields.current_value(de, theirs[spec.field_id])
        if a != b:
            diffs.append(f"{spec.field_id}: ours {a!r}, DE {b!r}")
    for row in range(1, 9):
        diffs.extend(
            f"stance {row}->{col}"
            for col in range(1, 9)
            if diplomacy_fields.stance_value(new, row, col) != diplomacy_fields.stance_value(de, row, col)
        )
        if diplomacy_fields.allied_victory_value(new, row) != diplomacy_fields.allied_victory_value(de, row):
            diffs.append(f"allied victory {row}")
    assert not diffs, diffs


def test_cameras_and_messages_match_the_resave(pair) -> None:
    _name, _legacy, new, de, replaced, _report = pair
    ours, theirs = player_fields.camera_fields(new), player_fields.camera_fields(de)
    pd3 = len(new._scenario.sections["Units"].retriever_map["player_data_3"].data) * 4
    # DE resets the first player_data_3 entry (its editor view) and fills initial views outside P1..P8.
    checked = [i for i in range(len(ours)) if 4 <= i < pd3 or pd3 + 2 <= i < pd3 + 18]
    assert [ours[i].value for i in checked] == [theirs[i].value for i in checked]
    nm = new._scenario.sections["Messages"].retriever_map
    dm = de._scenario.sections["Messages"].retriever_map
    diffs = []
    for spec in messages_fields.present_fields(nm):
        a, b = nm[spec.retriever].data, dm[spec.retriever].data
        if b and a != b and spec.retriever not in replaced:
            diffs.append(f"{spec.retriever}: {a[:60]!r} -> {b[:60]!r}")
        if nm[spec.id_retriever].data != dm[spec.id_retriever].data:
            diffs.append(f"{spec.id_retriever}: {nm[spec.id_retriever].data} -> {dm[spec.id_retriever].data}")
    assert not diffs, diffs
