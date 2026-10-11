"""TASK-061: the six 1.41 C2_ElCid_coop_* files read and write their Map
flags and Point of View at the offsets library_compat.fix_structure() gives
them. The library's own 1.41 structure frames the flag byte before
collide_and_correct; on these files it sits after the 16 views.

Corpus tier, parametrized over the six names directly rather than through
QUICK_CORPUS_NAMES, so every one runs without --corpus-full. The default-tier
shape and wiring tests live in tests/test_library_compat.py.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from descape import option_fields, player_fields
from descape.options_model import OptionsEditModel, field_offsets, verify_options_block
from descape.scenario_io import load_map_and_units
from descape.scenario_resize import Anchor, resize_scenario
from descape.scenario_write import write_scenario

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# P1 and P2's stored views, measured by decoding the Map bytes both ways.
ELCID_VIEWS = {
    "C2_ElCid_coop_1_v0_16.aoe2scenario": ((119, 101), (114, 100)),
    "C2_ElCid_coop_2_v0_13.aoe2scenario": ((81, 11), (81, 11)),
    "C2_ElCid_coop_3_v0_12.aoe2scenario": ((7, 13), (7, 14)),
    "C2_ElCid_coop_4_v0_13.aoe2scenario": ((166, 19), (166, 20)),
    "C2_ElCid_coop_5_v0_14.aoe2scenario": ((6, 5), (5, 6)),
    "C2_ElCid_coop_6_v0_13.aoe2scenario": ((86, 35), (86, 35)),
}
_VIEW = struct.Struct("<ii")


def _path(name: str) -> Path:
    path = EXAMPLES / name
    if not path.is_file():
        pytest.skip(f"corpus file not present: {path}")
    return path


def _flag(loaded, field_id: str) -> int:
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == field_id)
    return option_fields.current_value(loaded, spec)


def _views(loaded) -> list[tuple[int, int]]:
    specs = {s.field_id: s for s in player_fields.specs_for(loaded)}
    x_spec, y_spec = specs[player_fields.POV_X_FIELD], specs[player_fields.POV_Y_FIELD]
    return [
        (player_fields.current_value(loaded, x_spec, p), player_fields.current_value(loaded, y_spec, p))
        for p in range(9)
    ]


def _all_raw_views(loaded) -> list[tuple[int, int]]:
    views = loaded._scenario.sections["Map"].retriever_map["initial_player_views"].data
    return [(v.retriever_map["location_x"].data, v.retriever_map["location_y"].data) for v in views]


def _collide_offset(loaded) -> int:
    spec = next(s for s in option_fields.specs_for(loaded) if s.field_id == "collide_and_correct")
    offsets, ok = field_offsets(loaded, [spec])
    assert ok
    return offsets[spec].offset


@pytest.mark.corpus
@pytest.mark.parametrize("name", sorted(ELCID_VIEWS))
def test_v141_map_structure_reads_the_real_flags_and_views(name: str) -> None:
    loaded = load_map_and_units(_path(name))
    assert loaded.scenario_version == "1.41"
    assert _flag(loaded, "collide_and_correct") == 1
    assert _flag(loaded, "villager_force_drop") == 0
    assert _flag(loaded, "lock_coop_alliances") == 1
    views = _views(loaded)
    assert (views[1], views[2]) == ELCID_VIEWS[name]
    # The raw bytes agree, independent of any offset this code computes:
    # cc, vfd, 16 views, lock, in that order.
    at = _collide_offset(loaded)
    body = loaded.decompressed_body
    assert body[at : at + 2] == b"\x01\x00"
    raw = [_VIEW.unpack_from(body, at + 2 + 8 * i) for i in range(16)]
    assert raw == _all_raw_views(loaded)
    assert body[at + 130] == 1
    assert verify_options_block(loaded, option_fields.specs_for(loaded))
    assert player_fields.verify_player_block(loaded)


@pytest.mark.corpus
@pytest.mark.parametrize("name", sorted(ELCID_VIEWS))
def test_v141_map_structure_unchanged_save_is_byte_identical(name: str, tmp_path) -> None:
    original = load_map_and_units(_path(name))
    out = tmp_path / name
    write_scenario(original, out, backup=False)
    assert load_map_and_units(out).decompressed_body == original.decompressed_body


@pytest.mark.corpus
@pytest.mark.parametrize("name", sorted(ELCID_VIEWS))
def test_v141_map_structure_edit_writes_exactly_its_own_bytes(name: str, tmp_path) -> None:
    """The regression check proper: before the override, Collide & Correcting
    and the views were patched one byte off, into villager_force_drop's slot."""
    original = load_map_and_units(_path(name))
    w, h = original.map_manager.map_width, original.map_manager.map_height
    target = (min(w, h) // 4, min(w, h) // 2 + 3)  # in-map, x != y, clear of every stored view
    assert target[0] != target[1] and target != ELCID_VIEWS[name][0]
    before_views = _all_raw_views(original)
    vfd, lock = _flag(original, "villager_force_drop"), _flag(original, "lock_coop_alliances")
    at = _collide_offset(original)

    loaded = load_map_and_units(_path(name))
    model = OptionsEditModel(loaded)
    model.set_value(player_fields.player_field_id(player_fields.POV_X_FIELD, 1), target[0])
    model.set_value(player_fields.player_field_id(player_fields.POV_Y_FIELD, 1), target[1])
    model.set_value("collide_and_correct", 0)
    out = tmp_path / name
    write_scenario(loaded, out, options=model, backup=False)

    reloaded = load_map_and_units(out)
    assert _flag(reloaded, "collide_and_correct") == 0
    assert _flag(reloaded, "villager_force_drop") == vfd
    assert _flag(reloaded, "lock_coop_alliances") == lock
    after_views = _all_raw_views(reloaded)
    assert after_views[1] == target
    assert after_views[:1] + after_views[2:] == before_views[:1] + before_views[2:]

    expected = bytearray(original.decompressed_body)
    expected[at] = 0
    _VIEW.pack_into(expected, at + 2 + 8 * 1, *target)
    changed = [i for i, (a, b) in enumerate(zip(original.decompressed_body, expected, strict=True)) if a != b]
    assert changed and set(changed) <= {at, *range(at + 10, at + 18)}
    assert reloaded.decompressed_body == bytes(expected)


@pytest.mark.corpus
def test_v141_map_structure_resize_clamps_the_views(tmp_path) -> None:
    name = "C2_ElCid_coop_1_v0_16.aoe2scenario"
    loaded = load_map_and_units(_path(name))
    fields = player_fields.camera_fields(loaded)
    assert fields is not None
    view_fields = [f for f in fields if f.target.codec == "s32"]
    assert len(view_fields) == 32  # 16 views x 2 axes
    w = loaded.map_manager.map_width
    result = resize_scenario(loaded, w - 24, w - 24, Anchor.BOTTOM_RIGHT)
    out = tmp_path / name
    write_scenario(result.loaded, out, units=result.unit_edits, triggers=result.trigger_edits, backup=False)
    reloaded = load_map_and_units(out)
    assert _views(reloaded)[1] == (95, 95)
    # The clamp stayed inside the views: the flags either side are untouched.
    assert [_flag(reloaded, f) for f in ("collide_and_correct", "villager_force_drop", "lock_coop_alliances")] == [1, 0, 1]
