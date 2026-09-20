"""descape/disables_fields.py's codec and its load-time gate -- no Qt.

The load-bearing property, and the reason the gate and the writer are the
same function: re-encoding the region from the parsed counts and id lists
reproduces decompressed_body exactly. Everything else here either builds on
that (an edit re-derives only counts 0..7) or pins a fail-closed path.

The default-tier fixtures both carry empty disable lists, which is what the
corpus says 12 of 20 real files look like too. The interesting shapes --
a nonzero count slot 8, an out-of-enum id -- have no default-tier fixture
and are therefore exercised synthetically, against a stub whose parsed
values this module reads exactly as it would a real file's.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from descape import disables_fields
from descape.scenario_io import BLANK_TEMPLATE_PATH, load_map_and_units

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


# -- stubs ------------------------------------------------------------------
#
# encode_region()/verify_disables_block() read three things off a
# LoadedScenario: the Options section's retriever_map, decompressed_body,
# and the two section-end anchors. A stub supplying exactly those is what
# lets a nonzero count slot 8 -- which no corpus file has -- be tested at
# all, and it keeps that test honest: a stub whose region did NOT round-trip
# would fail the same assertion a real file would.


class _Datatype:
    type = "u32"  # anything but "struct" -- see scenario_io.retriever_length()


class _Retriever:
    def __init__(self, data, size):
        self.data = data
        self._size = size
        self.datatype = _Datatype()

    def get_data_as_bytes(self) -> bytes:
        return b"\x00" * self._size


class _Section:
    def __init__(self, retriever_map):
        self.retriever_map = retriever_map


class _Scenario:
    def __init__(self, sections):
        self.sections = sections


class _StubLoaded:
    """Just enough LoadedScenario for this module: an Options section whose
    retrievers are laid out exactly as a real file's, plus the body those
    retrievers describe."""

    def __init__(self, lists, counts_tail=(0,) * 8, lead=32, tail_all_techs=1):
        retriever_map: dict[str, _Retriever] = {}
        region = bytearray()
        for category in disables_fields.CATEGORIES:
            counts = [len(lists[(category, p)]) for p in range(1, 9)] + list(counts_tail)
            retriever_map[disables_fields.count_retriever_name(category)] = _Retriever(
                list(counts), 4 * len(counts)
            )
            region += struct.pack(f"<{len(counts)}I", *counts)
            for player_id in range(1, 9):
                ids = list(lists[(category, player_id)])
                retriever_map[
                    disables_fields.ids_retriever_name(category, player_id)
                ] = _Retriever(ids, 4 * len(ids))
                region += struct.pack(f"<{len(ids)}I", *ids)
        # combat_mode, naval_mode, all_techs: the two u32s the cross-check
        # steps over, then the anchor it lands on.
        for name, value in (("combat_mode", 0), ("naval_mode", 0), ("all_techs", tail_all_techs)):
            retriever_map[name] = _Retriever(value, 4)
            region += struct.pack("<I", value)

        self.decompressed_body = bytes(lead) + bytes(region)
        self.diplomacy_section_end = lead
        self.options_section_end = len(self.decompressed_body)
        self._scenario = _Scenario({"Options": _Section(retriever_map)})


def _empty_lists() -> dict[tuple[str, int], tuple[int, ...]]:
    return {
        (category, player_id): ()
        for category in disables_fields.CATEGORIES
        for player_id in range(1, 9)
    }


# -- identity round trip ----------------------------------------------------


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH], ids=["blank", "triggers"])
def test_zero_edit_encode_is_byte_identical(path) -> None:
    loaded = load_map_and_units(path)
    span = disables_fields.disables_region_span(loaded)
    assert span is not None
    start, end = span
    assert disables_fields.encode_region(loaded, {}) == loaded.decompressed_body[start:end]


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH], ids=["blank", "triggers"])
def test_gate_passes_on_the_default_tier_fixtures(path) -> None:
    assert disables_fields.verify_disables_block(load_map_and_units(path)) is True


@pytest.mark.corpus
def test_zero_edit_encode_is_byte_identical_corpus(scenario_path) -> None:
    """Measurement 2 of the plan turned into a standing assertion: every
    corpus file's region re-encodes exactly, so a browse-only save of any
    of them is byte-identical."""
    loaded = load_map_and_units(scenario_path)
    assert disables_fields.verify_disables_block(loaded) is True
    start, end = disables_fields.disables_region_span(loaded)
    assert disables_fields.encode_region(loaded, {}) == loaded.decompressed_body[start:end]


# -- the two independent walks agree ----------------------------------------


@pytest.mark.parametrize("path", [BLANK_TEMPLATE_PATH, FIXTURE_PATH], ids=["blank", "triggers"])
def test_forward_walk_and_backward_walk_agree(path) -> None:
    """The forward walk's end plus combat_mode+naval_mode must land on the
    backward walk's all_techs. Asserted here as well as inside the gate so a
    failure says which half broke."""
    loaded = load_map_and_units(path)
    _start, end = disables_fields.disables_region_span(loaded)
    assert end + 8 == disables_fields._all_techs_offset(loaded)


# -- counts -----------------------------------------------------------------


def test_count_slots_8_to_15_survive_a_round_trip_verbatim() -> None:
    """No corpus file stores a nonzero slot 8, so this is the only place the
    pass-it-through-verbatim rule for those slots is actually exercised."""
    tail = (7, 0, 0, 0, 0, 0, 0, 9)
    loaded = _StubLoaded(_empty_lists(), counts_tail=tail)
    assert disables_fields.verify_disables_block(loaded) is True
    start, end = disables_fields.disables_region_span(loaded)
    assert disables_fields.encode_region(loaded, {}) == loaded.decompressed_body[start:end]
    # And an edit elsewhere must not zero them either.
    edited = disables_fields.encode_region(loaded, {("units", 1): (100,)})
    counts = struct.unpack_from("<16I", edited, 16 * 4)  # the units block's own array
    assert counts[8:] == tail


def test_an_edit_rederives_counts_0_to_7_only() -> None:
    loaded = _StubLoaded(_empty_lists())
    edited = disables_fields.encode_region(
        loaded, {("techs", 1): (22, 23, 24), ("techs", 5): (101,)}
    )
    counts = struct.unpack_from("<16I", edited, 0)
    assert counts[:8] == (3, 0, 0, 0, 1, 0, 0, 0)
    assert struct.unpack_from("<3I", edited, 16 * 4) == (22, 23, 24)


def test_an_out_of_enum_id_round_trips() -> None:
    """621 ("Town Center") and 35 ("Battering Ram") are real corpus values
    object_catalog.objects() does not carry -- the codec must not care."""
    lists = _empty_lists()
    lists[("buildings", 2)] = (621,)
    lists[("units", 3)] = (35,)
    loaded = _StubLoaded(lists)
    assert disables_fields.verify_disables_block(loaded) is True
    assert disables_fields.current_ids(loaded, "buildings", 2) == (621,)
    start, end = disables_fields.disables_region_span(loaded)
    assert disables_fields.encode_region(loaded, {}) == loaded.decompressed_body[start:end]


# -- fail-closed paths ------------------------------------------------------


def test_gate_fails_closed_on_a_truncated_body() -> None:
    loaded = _StubLoaded(_empty_lists())
    loaded.decompressed_body = loaded.decompressed_body[:-40]
    loaded.options_section_end = len(loaded.decompressed_body)
    assert disables_fields.verify_disables_block(loaded) is False


def test_gate_fails_closed_on_a_body_that_does_not_match_the_parsed_values() -> None:
    loaded = _StubLoaded(_empty_lists())
    body = bytearray(loaded.decompressed_body)
    body[loaded.diplomacy_section_end] = 3  # a count the id lists do not back
    loaded.decompressed_body = bytes(body)
    assert disables_fields.verify_disables_block(loaded) is False


def test_gate_fails_closed_when_the_two_walks_disagree() -> None:
    """A layout with one extra field between the region and all_techs -- the
    shape a future structure version could take. The region's own bytes stay
    self-consistent, so the identity check still passes and only the
    cross-check catches that all_techs is no longer 8 bytes past the end."""
    loaded = _StubLoaded(_empty_lists())
    retriever_map = loaded._scenario.sections["Options"].retriever_map
    all_techs = retriever_map.pop("all_techs")
    retriever_map["a_future_field"] = _Retriever(0, 4)
    retriever_map["all_techs"] = all_techs
    loaded.decompressed_body += b"\x00" * 4
    loaded.options_section_end = len(loaded.decompressed_body)

    start, end = disables_fields.disables_region_span(loaded)
    assert disables_fields.encode_region(loaded, {}) == loaded.decompressed_body[start:end]
    assert disables_fields.verify_disables_block(loaded) is False


def test_gate_fails_closed_without_a_diplomacy_anchor() -> None:
    loaded = _StubLoaded(_empty_lists())
    loaded.diplomacy_section_end = -1
    assert disables_fields.disables_region_span(loaded) is None
    assert disables_fields.verify_disables_block(loaded) is False


def test_gate_fails_closed_when_combat_mode_is_absent() -> None:
    loaded = _StubLoaded(_empty_lists())
    del loaded._scenario.sections["Options"].retriever_map["combat_mode"]
    assert disables_fields.disables_region_span(loaded) is None
    assert disables_fields.verify_disables_block(loaded) is False


def test_splice_is_none_when_the_gate_fails() -> None:
    loaded = _StubLoaded(_empty_lists())
    loaded.diplomacy_section_end = -1
    assert disables_fields.disables_splice(loaded, {("units", 1): (4,)}) is None


def test_splice_spans_the_whole_region() -> None:
    loaded = _StubLoaded(_empty_lists())
    start, end, replacement = disables_fields.disables_splice(loaded, {("units", 1): (4,)})
    assert (start, end) == disables_fields.disables_region_span(loaded)
    assert len(replacement) == (end - start) + 4


# -- field ids and coercion -------------------------------------------------


def test_field_id_round_trips() -> None:
    assert disables_fields.disables_field_id("buildings", 3) == "disabled:buildings:3"
    assert disables_fields.parse_disables_field_id("disabled:buildings:3") == ("buildings", 3)
    assert len(disables_fields.all_field_ids()) == 24
    assert len(set(disables_fields.all_field_ids())) == 24


@pytest.mark.parametrize("bad", ["stance:1:2", "disabled:heroes:1", "disabled:units", "player:food:3"])
def test_parse_field_id_rejects_a_foreign_id(bad: str) -> None:
    with pytest.raises(ValueError):
        disables_fields.parse_disables_field_id(bad)


def test_coerce_dedupes_and_preserves_order() -> None:
    assert disables_fields.coerce_ids([5, 2, 5, 9, 2]) == (5, 2, 9)


@pytest.mark.parametrize("bad", ["12", 7, None, [1.5], [-1], [0x1_0000_0000], [True]])
def test_coerce_rejects_what_is_not_a_u32_sequence(bad) -> None:
    with pytest.raises(ValueError):
        disables_fields.coerce_ids(bad)
