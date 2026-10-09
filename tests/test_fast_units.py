"""The Units fast path's oracle, fallback and write-path tests: Unit objects
built off the raw players_units bytes (scenario_io._load_units_fast()) must
equal the library walk's (fast_units=False) attribute for attribute, with the
same spans, byte lengths and class poisoning, and the write path must produce
the same bytes from either load.

Covers every tests/fixtures file (units_120x120's non-empty caption,
v159_units_triggers' capture_flag), plus corpus-marked variants over
examples/ and the 25 real v1.21 Workshop files.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from AoE2ScenarioParser.objects.data_objects.unit import Unit
from AoE2ScenarioParser.objects.data_objects.units.player_units import PlayerUnits
from AoE2ScenarioParser.objects.managers.unit_manager import UnitManager
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario

from descape import gate_orientation, library_compat, scenario_io, unit_rotation, unit_sprites, unit_variant, unlinked_fields
from descape.scenario_io import load_map_and_units, load_map_and_units_from_bytes
from descape.scenario_write import write_scenario
from descape.terrain_units import UnitAddSpec
from descape.unit_model import UnitEditModel, _encode_unit_values, _player_units_link, _serialize_unit, _unit_values
from testkit.scenario_targets import collect_files

FIXTURES = sorted((Path(__file__).resolve().parent / "fixtures").glob("*.aoe2scenario"))
UNITS_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"
V159_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "v159_units_triggers.aoe2scenario"
_WITH_UNITS_NAMES = {"units_120x120.aoe2scenario", "v159_units_triggers.aoe2scenario", "triggers_120x120.aoe2scenario"}
WITH_UNITS = [p for p in FIXTURES if p.name in _WITH_UNITS_NAMES]
WORKSHOP_DIR = Path("~/games/steam/steamapps/workshop/content/221380").expanduser()

# Captured at import, before this module loads anything: the fast path must
# never relink these, so a later commit is the library's own.
_LINK_LISTS = {cls: cls._link_list for cls in (Unit, UnitManager, PlayerUnits)}
_MANAGER_STATE = {cls: dict(vars(cls)) for cls in (UnitManager, PlayerUnits)}


def _v121_files() -> list[Path]:
    return collect_files([WORKSHOP_DIR], pytest.fail) if WORKSHOP_DIR.is_dir() else []


def _plain(value):
    # Floats as their f32 bytes, so a NaN compares equal to itself.
    return struct.pack("<f", value) if isinstance(value, float) else value


def _unit_dicts(loaded) -> list[list[dict]]:
    uuid = loaded._scenario.uuid
    out = []
    for units in loaded.unit_manager.units:
        player = []
        for unit in units:
            fields = {name: _plain(value) for name, value in vars(unit).items()}
            assert fields.pop("_uuid") == uuid
            player.append(fields)
        out.append(player)
    return out


def _poison_state() -> tuple:
    links = tuple((link.name, link.disabled) for link in library_compat._iter_links(Unit._link_list))
    return library_compat.class_state_delta(Unit), links


class _Spy:
    def __init__(self) -> None:
        self.results: list = []


@pytest.fixture
def fast_spy(monkeypatch) -> _Spy:
    spy = _Spy()
    original = scenario_io._load_units_fast

    def wrapped(scenario, data_igen):
        result = original(scenario, data_igen)
        spy.results.append(result)
        return result

    monkeypatch.setattr(scenario_io, "_load_units_fast", wrapped)
    return spy


def _load_both(spy: _Spy, loader, *args):
    """(library, fast) loads, each with the Unit poisoning it left behind."""
    library = loader(*args, fast_units=False)
    library_poison = _poison_state()
    assert spy.results == [], "fast_units=False must not try the fast walk"
    fast = loader(*args)
    fast_poison = _poison_state()
    assert len(spy.results) == 1 and spy.results[0] is not None, "the fast walk was refused"
    return library, library_poison, fast, fast_poison


def _assert_units_agree(library, library_poison, fast, fast_poison) -> None:
    assert fast_poison == library_poison
    for attr in ("units_block_offset", "units_section_end", "players_units_end", "units_write_supported",
                 "number_of_unit_sections", "unit_spans", "terrain_write_supported"):
        assert getattr(fast, attr) == getattr(library, attr), attr
    lib_units = library._scenario.sections["Units"]
    fast_units = fast._scenario.sections["Units"]
    assert fast_units.byte_length == lib_units.byte_length
    lib_entries = lib_units.retriever_map["players_units"].data
    fast_entries = fast_units.retriever_map["players_units"].data
    assert [e.byte_length for e in fast_entries] == [e.byte_length for e in lib_entries]
    assert [e.retriever_map["unit_count"].data for e in fast_entries] == [
        e.retriever_map["unit_count"].data for e in lib_entries
    ]
    assert all(e.retriever_map["units"].data == [] for e in fast_entries)
    # The spans are the library entries' own byte_lengths, not just equal to a
    # derivation of them.
    assert [[length for _, length in player] for player in fast.unit_spans] == [
        [unit.byte_length for unit in e.retriever_map["units"].data] for e in lib_entries
    ]
    lib_manager, fast_manager = library.unit_manager, fast.unit_manager
    assert type(fast_manager.units) is type(lib_manager.units)
    assert [type(units) for units in fast_manager.units] == [type(units) for units in lib_manager.units]
    assert set(vars(fast_manager)) == set(vars(lib_manager))
    assert fast_manager._instance_number_history == lib_manager._instance_number_history
    assert (
        fast_manager.reference_id_generator.gi_frame.f_locals["start_id"]
        == lib_manager.reference_id_generator.gi_frame.f_locals["start_id"]
    )
    fast_dicts, lib_dicts = _unit_dicts(fast), _unit_dicts(library)
    assert [len(p) for p in fast_dicts] == [len(p) for p in lib_dicts]
    for player, (a, b) in enumerate(zip(fast_dicts, lib_dicts, strict=True)):
        mismatched = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        if mismatched:
            i = mismatched[0]
            pytest.fail(f"player {player}: {len(mismatched)} units differ, first {i}: {a[i]} != {b[i]}")


def test_fast_units_oracle_covers_the_named_fixtures() -> None:
    names = {path.name for path in FIXTURES}
    assert {"units_120x120.aoe2scenario", "v159_units_triggers.aoe2scenario"} <= names
    assert len(WITH_UNITS) == 3


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_fast_units_match_the_library_walk(path: Path, fast_spy) -> None:
    _assert_units_agree(*_load_both(fast_spy, load_map_and_units, path))


def test_the_non_empty_caption_and_capture_flag_come_through() -> None:
    """The two fields the fixtures carry on purpose, read off the fast load."""
    import conftest

    fast = load_map_and_units(UNITS_FIXTURE)
    assert "Fixture caption" in {u.caption_string for units in fast.unit_manager.units for u in units}
    v159 = load_map_and_units(V159_FIXTURE)
    flags = {u.reference_id: u.capture_flag for units in v159.unit_manager.units for u in units}
    assert flags == conftest.load_verify_module("gen_v159_fixture").CAPTURE_FLAGS


def test_fast_units_on_a_new_map(fast_spy) -> None:
    from descape.scenario_new import blank_scenario_bytes

    data = blank_scenario_bytes(144)
    fast_spy.results.clear()  # the donor's own load
    _assert_units_agree(*_load_both(fast_spy, load_map_and_units_from_bytes, data, "blank.aoe2scenario"))


def test_fast_load_leaves_unit_links_pristine() -> None:
    """No relink: the fast path builds Units and the UnitManager through their
    constructors, so the manager.commit() of serialize()'s fallback
    (_serialize_via_commit()) runs the library's own link lists."""
    load_map_and_units(UNITS_FIXTURE)
    for cls, links in _LINK_LISTS.items():
        assert cls._link_list is links, cls.__name__
    for cls, state in _MANAGER_STATE.items():
        assert dict(vars(cls)) == state, cls.__name__
    # Only the library's own poisoning: this 1.58 fixture predates capture_flag.
    assert library_compat.class_state_delta(Unit) == (["capture_flag"], [])


def test_a_link_an_earlier_load_left_disabled_gets_none_like_the_library(monkeypatch) -> None:
    """With depoison() skipped, a link an older file disabled stays disabled,
    and the library's pull hands Unit None for it even on a newer file. The
    fast path follows the link's state, not just the version."""
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    depoison = library_compat.depoison
    link = next(link for link in library_compat._iter_links(Unit._link_list) if link.name == "caption_string_id")

    def _get(self_):
        raise UnsupportedAttributeError("left by an earlier load")

    def _set(self_, value):
        if value is not None:
            raise UnsupportedAttributeError("left by an earlier load")

    monkeypatch.setattr(library_compat, "depoison", lambda: None)
    results = []
    try:
        for fast in (False, True):
            depoison()
            link.disabled = True
            Unit.caption_string_id = property(_get, _set)
            loaded = load_map_and_units(UNITS_FIXTURE, fast_units=fast)
            results.append((_unit_dicts(loaded), _poison_state()))
    finally:
        monkeypatch.undo()
        depoison()
    assert results[0] == results[1]
    assert all("caption_string_id" not in unit for units in results[1][0] for unit in units)


@pytest.mark.corpus
def test_fast_units_match_the_library_walk_corpus(scenario_path, fast_spy) -> None:
    _assert_units_agree(*_load_both(fast_spy, load_map_and_units, scenario_path))


@pytest.mark.corpus
@pytest.mark.skipif(not WORKSHOP_DIR.is_dir(), reason="the real v1.21 Workshop corpus only exists on this machine")
@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_fast_units_match_the_library_walk_v121(path: Path, fast_spy) -> None:
    _assert_units_agree(*_load_both(fast_spy, load_map_and_units, path))


# -- the format -----------------------------------------------------------------

_DE_UNIT_STRUCT = {
    "x": {"type": "f32"},
    "y": {"type": "f32"},
    "z": {"type": "f32"},
    "reference_id": {"type": "s32"},
    "unit_const": {"type": "u16"},
    "status": {"type": "u8"},
    "rotation": {"type": "f32"},
    "initial_animation_frame": {"type": "u16"},
    "garrisoned_in_id": {"type": "s32"},
    "capture_flag": {"type": "s8"},
    "caption_string_id": {"type": "s32"},
    "caption_string": {"type": "str32"},
}


def test_unit_struct_format_is_derived_from_the_model() -> None:
    fmt = scenario_io._unit_struct_format(_DE_UNIT_STRUCT)
    assert (fmt.fixed.format, fmt.fixed.size, fmt.caption) == ("<fffiHBfHibi", 34, "caption_string")
    assert fmt.names == tuple(_DE_UNIT_STRUCT)
    old = {k: v for k, v in _DE_UNIT_STRUCT.items() if k not in {"capture_flag", "caption_string_id", "caption_string"}}
    fmt = scenario_io._unit_struct_format(old)
    assert (fmt.fixed.size, fmt.caption) == (29, None)


@pytest.mark.parametrize(
    "change",
    [
        {"status": {"type": "u8", "repeat": 2}},
        {"status": {"type": "u8", "dependencies": {"on_construct": {"action": "REFRESH_SELF"}}}},
        {"status": {"type": "1"}},
        {"status": {"type": "c1"}},
        {"status": {"type": "str16"}},
        {"caption_string": {"type": "str32"}, "after": {"type": "u8"}},
        {"status": {"type": "str32"}},
    ],
    ids=["repeat", "dependency", "raw-bytes", "char-type", "str16", "str32-not-last", "two-str32"],
)
def test_unit_struct_format_refuses_anything_else(change: dict) -> None:
    assert scenario_io._unit_struct_format({**_DE_UNIT_STRUCT, **change}) is None


def _units_structure(unit_struct: dict) -> dict:
    return {
        "retrievers": {"players_units": {"type": "struct:PlayerUnitsStruct"}},
        "structs": {
            "PlayerUnitsStruct": {
                "retrievers": {"unit_count": {"type": "u32"}, "units": {"type": "struct:UnitStruct"}},
                "structs": {"UnitStruct": {"retrievers": unit_struct}},
            }
        },
    }


def test_unit_codec_carries_every_field() -> None:
    codec = scenario_io._unit_codec(_units_structure(_DE_UNIT_STRUCT), "1.59")
    assert codec.attrs == ()
    assert "capture_flag" in codec.kwargs and codec.unsupported == ()
    old = {k: v for k, v in _DE_UNIT_STRUCT.items() if k not in {"capture_flag", "caption_string_id", "caption_string"}}
    codec = scenario_io._unit_codec(_units_structure(old), "1.37")
    assert set(codec.unsupported) == {"capture_flag", "caption_string_id", "caption_string"}


def _with_field_before_caption(name: str) -> dict:
    fields = {k: v for k, v in _DE_UNIT_STRUCT.items() if k != "caption_string"}
    return {**fields, name: {"type": "u8"}, "caption_string": {"type": "str32"}}


@pytest.mark.parametrize(
    ("unit_struct", "version"),
    [
        (_with_field_before_caption("mystery"), "1.59"),
        ({k: v for k, v in _DE_UNIT_STRUCT.items() if k != "status"}, "1.59"),
        (_DE_UNIT_STRUCT, "1.54"),
    ],
    ids=["uncarried-field", "supported-link-missing", "field-of-an-unsupported-link"],
)
def test_unit_codec_refuses_what_it_cannot_carry(unit_struct: dict, version: str) -> None:
    assert scenario_io._unit_codec(_units_structure(unit_struct), version) is None


def test_str32_decode_matches_the_library() -> None:
    from AoE2ScenarioParser.helper.bytes_conversions import bytes_to_str

    for raw in (b"", b"\x00", b"abc\x00", b"abc", b"abc\x00\x00", "café\x00".encode(), b"\xff\xfe\x00"):
        assert scenario_io._decode_str32(raw) == bytes_to_str(raw), raw


# -- fallback -------------------------------------------------------------------


def _spy_sections(monkeypatch) -> list[str]:
    from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

    seen: list[str] = []
    original = AoE2Scenario._create_and_load_section

    def spy(self, name, igen):
        seen.append(name)
        return original(self, name, igen)

    monkeypatch.setattr(AoE2Scenario, "_create_and_load_section", spy)
    return seen


def test_a_refused_unit_model_falls_back_to_the_library_walk(monkeypatch) -> None:
    """A structure copy whose u8 `status` is a one-byte char field instead
    reads the same bytes, is refused by the format, and loads through the
    library walk, equal to a forced library load of the same structure. The
    edit model then takes the entry gate, which still passes."""
    original = AoE2DEScenario._load_structure

    def load_structure(self):
        original(self)
        units = self.structure["Units"]["structs"]["PlayerUnitsStruct"]["structs"]["UnitStruct"]["retrievers"]
        units["status"] = {"type": "c1", "default": "\x02"}

    monkeypatch.setattr(AoE2DEScenario, "_load_structure", load_structure)
    seen = _spy_sections(monkeypatch)
    loaded = load_map_and_units(UNITS_FIXTURE)
    assert "Units" in seen, "the Units section never went through the library walk"
    assert scenario_io.unit_codec(loaded) is None
    library = load_map_and_units(UNITS_FIXTURE, fast_units=False)
    assert _unit_dicts(loaded) == _unit_dicts(library)
    assert loaded.unit_spans == library.unit_spans and loaded.units_write_supported
    UnitEditModel(loaded)


class _UnitsHandoff(Exception):
    def __init__(self, progress: int) -> None:
        self.progress = progress


def test_a_unit_block_past_the_body_falls_back_to_the_library_walk(monkeypatch) -> None:
    """Player 1's unit_count inflated so its units would run past the body:
    the fast walk rewinds to Units' start and hands Units to the library walk.
    Stopped at the handoff, since the library would only fail its own way
    after printing every struct it read."""
    from AoE2ScenarioParser.scenarios.aoe2_scenario import AoE2Scenario

    from descape.scenario_write import _compress_bytes

    donor = load_map_and_units(UNITS_FIXTURE)
    body = bytearray(donor.decompressed_body)
    count_offset = donor.unit_spans[1][0][0] - 4
    struct.pack_into("<I", body, count_offset, 1_000_000)
    data = donor.header_bytes + _compress_bytes(bytes(body))

    ends: dict[str, int] = {}
    original = AoE2Scenario._create_and_load_section

    def spy(self, name, igen):
        if name == "Units":
            raise _UnitsHandoff(igen.progress)
        original(self, name, igen)
        ends[name] = igen.progress

    monkeypatch.setattr(AoE2Scenario, "_create_and_load_section", spy)
    with pytest.raises(_UnitsHandoff) as handoff:
        load_map_and_units_from_bytes(data, "inflated.aoe2scenario", fast_terrain=False)
    assert handoff.value.progress == ends["Map"]


def test_walk_units_refuses_a_block_past_the_body() -> None:
    fmt = scenario_io._unit_struct_format(_DE_UNIT_STRUCT)
    one = fmt.fixed.pack(1.5, 2.5, 0.0, 7, 4, 2, 0.0, 0, -1, -1, -1) + struct.pack("<i", 4) + b"abc\x00"
    values, spans, end = scenario_io._walk_units(fmt, one, 0, 1)
    assert (values[0][-1], spans, end) == ("abc", [(0, len(one))], len(one))
    assert scenario_io._walk_units(fmt, one[:-1], 0, 1) is None
    assert scenario_io._walk_units(fmt, one[: fmt.fixed.size + 2], 0, 1) is None
    assert scenario_io._walk_units(fmt, one, 0, 2) is None
    negative = fmt.fixed.pack(1.5, 2.5, 0.0, 7, 4, 2, 0.0, 0, -1, -1, -1) + struct.pack("<i", -3)
    values, spans, end = scenario_io._walk_units(fmt, negative, 0, 1)
    assert (values[0][-1], end) == ("", len(negative))


# -- write path -----------------------------------------------------------------


def _dirty_every_unit(loaded) -> UnitEditModel:
    model = UnitEditModel(loaded)
    for units in loaded.unit_manager.units:
        for unit in units:
            model.set_position(unit, unit.x, unit.y, unit.z)
    return model


def _edit(loaded) -> UnitEditModel:
    """One of each kind of unit edit, deterministic across two loads."""
    model = UnitEditModel(loaded)
    all_units = [u for units in loaded.unit_manager.units for u in units]
    first, last = all_units[0], all_units[-1]
    model.set_position(first, first.x + 1.0, first.y, first.z)
    model.add(player=2, unit_const=83, x=15.5, y=15.5, capture_flag=0)
    model.reassign(last, 3 if int(last.player) != 3 else 4)
    if len(all_units) > 2:
        model.remove(all_units[1])
    return model


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_an_unedited_fast_load_saves_byte_identical(path: Path, tmp_path: Path) -> None:
    loaded = load_map_and_units(path)
    out = tmp_path / "same.aoe2scenario"
    write_scenario(loaded, out, backup=False, units=UnitEditModel(loaded))
    assert out.read_bytes() == path.read_bytes()


def _dirty_save_reproduces(path: Path, tmp_path: Path, via_commit: bool = False) -> None:
    """`via_commit` routes the same write path through _serialize_via_commit(),
    which on a fast load builds each slot from the model's defaults first."""
    loaded = load_map_and_units(path)
    out = tmp_path / "dirty.aoe2scenario"
    model = _dirty_every_unit(loaded)
    if via_commit:
        model.serialize = model._serialize_via_commit
    write_scenario(loaded, out, backup=False, units=model)
    assert load_map_and_units(out).decompressed_body == loaded.decompressed_body


@pytest.mark.parametrize("via_commit", [False, True], ids=["encoder", "commit"])
@pytest.mark.parametrize("path", WITH_UNITS, ids=lambda p: p.name)
def test_a_fast_load_dirty_save_of_every_unit_is_byte_identical(path: Path, tmp_path: Path, via_commit: bool) -> None:
    _dirty_save_reproduces(path, tmp_path, via_commit)


def _edited_saves_agree(path: Path, tmp_path: Path) -> None:
    outs = []
    for fast in (True, False):
        loaded = load_map_and_units(path, fast_units=fast)
        out = tmp_path / f"edited_{fast}.aoe2scenario"
        write_scenario(loaded, out, backup=False, units=_edit(loaded))
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


@pytest.mark.parametrize("path", WITH_UNITS, ids=lambda p: p.name)
def test_an_edited_fast_save_equals_the_same_edit_on_a_library_load(path: Path, tmp_path: Path) -> None:
    _edited_saves_agree(path, tmp_path)


def _encoder_serializer_and_spans_agree(path: Path) -> None:
    """Per unit: the encoder's bytes (from the span's decoded values and from
    the library Unit's own), _serialize_unit() of the library entry and the
    raw span are identical; and a commit of the fast-path Units reproduces
    the spans byte for byte through _serialize_unit()."""
    library = load_map_and_units(path, fast_units=False)
    if not library.units_write_supported:
        pytest.skip(f"{path.name}: units are not editable")
    codec = scenario_io.unit_codec(library)
    assert codec is not None
    body = library.decompressed_body
    entries = [pu.retriever_map["units"].data for pu in library._scenario.sections["Units"].retriever_map["players_units"].data]
    for units, player_entries, spans in zip(library.unit_manager.units, entries, library.unit_spans, strict=True):
        for unit, entry, (start, length) in zip(units, player_entries, spans, strict=True):
            raw = body[start : start + length]
            fmt = codec.unit_format
            assert _serialize_unit(entry) == raw, unit.reference_id
            assert _encode_unit_values(fmt, scenario_io.unit_values(fmt, raw)) == raw, unit.reference_id
            assert _encode_unit_values(fmt, _unit_values(codec, unit)) == raw, unit.reference_id

    fast = load_map_and_units(path)
    manager = fast.unit_manager
    library_compat.depoison()
    manager.commit(link_list=[_player_units_link(manager)])
    committed = [pu.retriever_map["units"].data for pu in fast._scenario.sections["Units"].retriever_map["players_units"].data]
    for units, player_entries, spans in zip(manager.units, committed, fast.unit_spans, strict=True):
        unlinked_fields.push(Unit, units, player_entries)
        for entry, (start, length) in zip(player_entries, spans, strict=True):
            assert _serialize_unit(entry) == fast.decompressed_body[start : start + length]


@pytest.mark.parametrize("path", WITH_UNITS, ids=lambda p: p.name)
def test_encoder_serializer_and_spans_agree(path: Path) -> None:
    _encoder_serializer_and_spans_agree(path)


@pytest.mark.corpus
def test_encoder_serializer_and_spans_agree_corpus(scenario_path) -> None:
    _encoder_serializer_and_spans_agree(scenario_path)


@pytest.mark.corpus
@pytest.mark.skipif(not WORKSHOP_DIR.is_dir(), reason="the real v1.21 Workshop corpus only exists on this machine")
@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_encoder_serializer_and_spans_agree_v121(path: Path) -> None:
    _encoder_serializer_and_spans_agree(path)


@pytest.mark.corpus
def test_an_edited_fast_save_equals_the_same_edit_on_a_library_load_corpus(scenario_path, tmp_path: Path) -> None:
    loaded = load_map_and_units(scenario_path, fast_units=False)
    if not loaded.units_write_supported or sum(len(u) for u in loaded.unit_manager.units) < 3:
        pytest.skip(f"{scenario_path.name}: no editable units")
    _edited_saves_agree(scenario_path, tmp_path)


@pytest.mark.corpus
def test_a_fast_load_dirty_save_of_every_unit_is_byte_identical_corpus(scenario_path, tmp_path: Path) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units are not editable")
    _dirty_save_reproduces(scenario_path, tmp_path)


@pytest.mark.corpus
def test_a_fast_load_dirty_save_of_every_unit_through_the_commit_is_byte_identical_corpus(scenario_path, tmp_path: Path) -> None:
    """The fallback writer keeps the full-corpus coverage the encoder took over."""
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported:
        pytest.skip(f"{scenario_path.name}: units are not editable")
    _dirty_save_reproduces(scenario_path, tmp_path, via_commit=True)


# -- the encoder writes a dirty save; the commit is its oracle and fallback ------

_TREE = 349  # cyclable; units_120x120 carries one in GAIA
_WALL = 117  # radian-encoded in units_120x120
_GATE = 64  # stone gate, one of four orientation siblings
_ARCHER = 4


class _Commits:
    def __init__(self) -> None:
        self.count = 0


@pytest.fixture
def commits(monkeypatch) -> _Commits:
    """Counts UnitManager.commit() calls: the fallback's signature."""
    spy = _Commits()
    original = UnitManager.commit

    def wrapped(self, *args, **kwargs):
        spy.count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(UnitManager, "commit", wrapped)
    return spy


def _first_unit(loaded, predicate):
    return next((u for units in loaded.unit_manager.units for u in units if predicate(u.unit_const)), None)


def _edit_battery(loaded) -> UnitEditModel:
    """_edit()'s move/add/reassign/remove, three added trees, then each
    rotation/const exception on a unit the file has, or one added for it."""
    model = _edit(loaded)
    model.add_many(0, [UnitAddSpec(x=30.5 + i, y=31.5, unit_const=_TREE, rotation=float(i), initial_animation_frame=i) for i in range(3)])

    def target(predicate, const: int, x: float):
        return _first_unit(loaded, predicate) or model.add(player=1, unit_const=const, x=x, y=40.5)

    angle = target(unit_rotation.rotation_is_angle, _ARCHER, 40.5)
    model.set_rotation(angle, angle.rotation + 0.5)
    tree = target(unit_variant.is_cyclable, _TREE, 42.5)
    angles, variants = unit_rotation.angle_count_for(tree.unit_const), unit_variant.variant_count_for(tree.unit_const)
    model.set_variant(tree, unit_variant.cycle_step(tree.rotation, angles, variants, 1))
    wall = target(unit_sprites.rotation_variant_eligible, _WALL, 44.5)
    model.set_wall_variant(wall, 1 if wall.rotation == 3.0 else 3)
    gate = target(gate_orientation.is_gate, _GATE, 50.5)
    model.set_unit_const(gate, gate_orientation.cycle_const(gate.unit_const, 1))
    return model


def _encoder_equals_commit(path: Path, commits: _Commits) -> None:
    """Two loads, the same edits: serialize() (which must not commit) and
    _serialize_via_commit() write the same section."""
    outs = []
    for via_commit in (False, True):
        loaded = load_map_and_units(path)
        model = _edit_battery(loaded)
        commits.count = 0
        outs.append(model._serialize_via_commit() if via_commit else model.serialize())
        assert commits.count == int(via_commit)
    assert outs[0] == outs[1]


@pytest.mark.parametrize("path", WITH_UNITS, ids=lambda p: p.name)
def test_the_encoder_and_the_commit_write_the_same_edits(path: Path, commits: _Commits) -> None:
    _encoder_equals_commit(path, commits)


@pytest.mark.corpus
def test_the_encoder_and_the_commit_write_the_same_edits_corpus(scenario_path, commits: _Commits) -> None:
    loaded = load_map_and_units(scenario_path)
    if not loaded.units_write_supported or sum(len(u) for u in loaded.unit_manager.units) < 3:
        pytest.skip(f"{scenario_path.name}: no editable units")
    _encoder_equals_commit(scenario_path, commits)


@pytest.mark.corpus
@pytest.mark.skipif(not WORKSHOP_DIR.is_dir(), reason="the real v1.21 Workshop corpus only exists on this machine")
@pytest.mark.parametrize("path", _v121_files(), ids=lambda p: p.name)
def test_the_encoder_and_the_commit_write_the_same_edits_v121(path: Path, commits: _Commits) -> None:
    loaded = load_map_and_units(path)
    if not loaded.units_write_supported or sum(len(u) for u in loaded.unit_manager.units) < 3:
        pytest.skip(f"{path.name}: no editable units")
    _encoder_equals_commit(path, commits)


def _unit_by_ref(loaded, reference_id: int):
    return next(u for units in loaded.unit_manager.units for u in units if u.reference_id == reference_id)


def _touch(model: UnitEditModel, unit) -> None:
    model.set_position(unit, unit.x, unit.y, unit.z)


def _set_fields(**fields):
    def edit(model, loaded):
        unit = _unit_by_ref(loaded, 201)
        for name, value in fields.items():
            setattr(unit, name, value)
        _touch(model, unit)

    return edit


def _move(x: float, z: float = 0.0):
    def edit(model, loaded):
        unit = _unit_by_ref(loaded, 201)
        model.set_position(unit, x, unit.y, z)

    return edit


def _variant_int(model, loaded) -> None:
    model.set_variant(_unit_by_ref(loaded, 100), 3)


def _caption(value):
    return _set_fields(caption_string=value)


def _added_with_flag(model, loaded) -> None:
    model.add(player=1, unit_const=_ARCHER, x=30.5, y=30.5, capture_flag=5)


def _reassigned_with_flag(model, loaded) -> None:
    unit = _unit_by_ref(loaded, 200)  # capture_flag 2, player 1
    _touch(model, unit)
    model.reassign(unit, 3)
    _touch(model, _unit_by_ref(loaded, 100))  # a commit re-slots every unit


_U16_MAX, _S32_MIN, _S32_MAX = 0xFFFF, -(2**31), 2**31 - 1


@pytest.mark.parametrize(
    ("path", "edit"),
    [
        (UNITS_FIXTURE, _move(12.3)),
        (UNITS_FIXTURE, _move(-0.0, -0.0)),
        (UNITS_FIXTURE, _variant_int),
        (UNITS_FIXTURE, _set_fields(status=255, unit_const=_U16_MAX, initial_animation_frame=_U16_MAX,
                                    garrisoned_in_id=_S32_MIN, caption_string_id=_S32_MAX)),
        (UNITS_FIXTURE, _set_fields(status=0, unit_const=0, initial_animation_frame=0,
                                    garrisoned_in_id=_S32_MAX, caption_string_id=_S32_MIN)),
        (UNITS_FIXTURE, _caption("")),
        (UNITS_FIXTURE, _caption("Archer caption")),
        (UNITS_FIXTURE, _caption("Größe 世界")),
        (UNITS_FIXTURE, _caption(b"\xff\xfe".decode("latin-1"))),
        (UNITS_FIXTURE, _caption(b"\xff\xfe")),
        (V159_FIXTURE, _set_fields(capture_flag=-128)),
        (V159_FIXTURE, _set_fields(capture_flag=127)),
        (V159_FIXTURE, _added_with_flag),
        (V159_FIXTURE, _reassigned_with_flag),
    ],
    ids=[
        "x-not-f32-exact", "negative-zero", "set-variant-int", "int-upper-bounds", "int-lower-bounds",
        "caption-empty", "caption-ascii", "caption-non-ascii", "caption-latin1-decoded", "caption-latin1-bytes",
        "capture-flag-min", "capture-flag-max", "added-capture-flag", "reassigned-capture-flag",
    ],
)
def test_edited_values_encode_as_the_commit_writes_them(path: Path, edit, commits: _Commits) -> None:
    outs = []
    for via_commit in (False, True):
        loaded = load_map_and_units(path)
        model = UnitEditModel(loaded)
        edit(model, loaded)
        commits.count = 0
        outs.append(model._serialize_via_commit() if via_commit else model.serialize())
        assert commits.count == int(via_commit), "the encoder fell back to the commit"
    assert outs[0] == outs[1]


# -- routing: when the commit runs -----------------------------------------------


def test_a_dirty_save_on_a_fast_load_never_commits(commits: _Commits) -> None:
    """The perf guard: a dirty serialize() on a fast load commits zero times,
    so the empty unit slots the fast walk left stay empty."""
    loaded = load_map_and_units(UNITS_FIXTURE)
    players_units = loaded._scenario.sections["Units"].retriever_map["players_units"].data
    model = _edit_battery(loaded)
    section = model.serialize()
    assert commits.count == 0
    assert all(pu.retriever_map["units"].data == [] for pu in players_units)
    assert section[:4] == struct.pack("<I", len(loaded.unit_manager.units[0]))


def test_a_structure_with_no_codec_saves_through_the_commit(monkeypatch, commits: _Commits) -> None:
    def edited(no_codec: bool):
        if no_codec:
            monkeypatch.setattr(scenario_io, "unit_codec", lambda loaded: None)
        loaded = load_map_and_units(UNITS_FIXTURE, fast_units=False)
        return _edit_battery(loaded)

    expected = edited(False)._serialize_via_commit()
    model = edited(True)
    assert model._codec is None
    commits.count = 0
    assert model.serialize() == expected
    assert commits.count == 1


def _inject(name: str, value):
    def inject(loaded) -> UnitEditModel:
        model = UnitEditModel(loaded)
        unit = _unit_by_ref(loaded, 201)
        setattr(unit, name, value)
        _touch(model, unit)
        return model

    return inject


def _outcome(write):
    try:
        return ("bytes", write())
    except Exception as exc:  # noqa: BLE001 -- the outcome is what is compared
        return (type(exc), str(exc))


@pytest.mark.parametrize(
    "inject",
    [_inject("unit_const", np.int64(4)), _inject("rotation", b"\x00\x00\x80\x3f"), _inject("status", None)],
    ids=["numpy-int-const", "bytes-rotation", "none-status"],
)
def test_a_value_the_encoder_refuses_falls_back_to_the_commit(inject, commits: _Commits) -> None:
    """One commit, then either the commit's own bytes or its own exception:
    the reference is a fresh load calling _serialize_via_commit() directly."""
    reference = _outcome(inject(load_map_and_units(UNITS_FIXTURE))._serialize_via_commit)
    model = inject(load_map_and_units(UNITS_FIXTURE))
    commits.count = 0
    got = _outcome(model.serialize)
    assert commits.count == 1
    assert got == reference


# -- poisoning ------------------------------------------------------------------

_PRE_1_55_PATH = Path(__file__).resolve().parent.parent / "examples" / "C2_ElCid_coop_1_v0_16.aoe2scenario"


@pytest.mark.corpus
@pytest.mark.skipif(not _PRE_1_55_PATH.exists(), reason="needs the 1.41 examples/ file")
def test_a_dirty_save_after_an_older_load_poisons_unit_corpus(commits: _Commits) -> None:
    """A real 1.41 load poisons caption_string on the class; the 1.58 model's
    serialize() still encodes, equal to its commit."""
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    outs = []
    try:
        for via_commit in (False, True):
            loaded = load_map_and_units(UNITS_FIXTURE)
            model = _edit_battery(loaded)
            load_map_and_units(_PRE_1_55_PATH)
            with pytest.raises(UnsupportedAttributeError):
                _ = _unit_by_ref(loaded, 300).caption_string
            commits.count = 0
            outs.append(model._serialize_via_commit() if via_commit else model.serialize())
            assert commits.count == int(via_commit)
    finally:
        library_compat.depoison()
    assert outs[0] == outs[1]
