"""Covers descape/player_fields.py: the Qt-free field-spec module for the
read-only Players mode (Step 1 of the private maintainer plan).

Mirrors test_option_fields.py's shape: stub-based tests pin specs_for()'s
presence/fallback gate directly, then real-load and corpus tests confirm the
layout mapping and current_value() against actual files.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from descape import player_fields, scenario_io
from descape.player_fields import PlayerArrayLayout, current_value, specs_for


class _StubRetriever:
    def __init__(self, raw_bytes: bytes, data=0):
        self._raw = raw_bytes
        self.data = data
        self.datatype = SimpleNamespace(type="u32")

    def get_data_as_bytes(self) -> bytes:
        return self._raw


def _default_entries_for(section_name: str, retriever_name: str):
    """Plain `list(range(16))` for a scalar retriever, or -- when some spec
    reads this (section, retriever) pair through a `struct_field` -- a list
    of struct-shaped stand-ins exposing every such field, matching what
    _retriever_present()'s struct_field check and current_value()'s own
    `entry.retriever_map[struct_field].data` both require. Needed because a
    real struct-array entry (e.g. DataHeader.player_data_1) can be present
    and non-empty while missing a specific struct field on an older
    scenario version -- see _retriever_present()'s own docstring in
    player_fields.py for the corpus file (1.37) that this was measured
    against."""
    struct_fields = {
        s.struct_field
        for s in player_fields._SPECS
        if s.struct_field is not None and s.section == section_name and s.retriever == retriever_name
    }
    if not struct_fields:
        return list(range(16))
    return [
        SimpleNamespace(retriever_map={f: SimpleNamespace(data=i) for f in struct_fields})
        for i in range(16)
    ]


def _stub_loaded(overrides: dict[tuple[str, str], "_StubRetriever"] = None):
    """A LoadedScenario stand-in exposing only what specs_for()/current_value()
    touch. Every primary (section, retriever) pair in player_fields._SPECS
    gets a present, non-empty, repeat-16 stub of P1..P8-then-GAIA-shaped data
    by default; `overrides` replaces or removes individual entries -- pass
    None as the value to simulate a missing retriever. Fallback targets are
    never stubbed in by default (matching the real shape where the fallback
    only matters when the primary is absent), but `overrides` can seed one
    explicitly -- fallback pairs are tracked too, not just primary ones."""
    overrides = overrides or {}
    sections: dict[str, dict[str, _StubRetriever]] = {}

    def seed(section_name: str, retriever_name: str, default: bool) -> None:
        sections.setdefault(section_name, {})
        key = (section_name, retriever_name)
        if key in overrides:
            value = overrides[key]
            if value is not None:
                sections[section_name][retriever_name] = value
        elif default:
            sections[section_name].setdefault(
                retriever_name,
                _StubRetriever(b"\x00" * 4, data=_default_entries_for(section_name, retriever_name)),
            )

    for spec in player_fields._SPECS:
        seed(spec.section, spec.retriever, default=True)
        if spec.fallback is not None:
            seed(spec.fallback[0], spec.fallback[1], default=False)

    section_objs = {
        name: SimpleNamespace(retriever_map=retrievers)
        for name, retrievers in sections.items()
        if retrievers
    }
    return SimpleNamespace(_scenario=SimpleNamespace(sections=section_objs))


# -- PlayerArrayLayout.index_for() --------------------------------------------


def test_p1_to_p8_then_gaia_puts_p1_at_0_and_gaia_at_8() -> None:
    layout = PlayerArrayLayout.P1_TO_P8_THEN_GAIA
    assert [layout.index_for(p) for p in range(1, 9)] == list(range(8))
    assert layout.index_for(0) == 8


def test_gaia_first_puts_gaia_at_0() -> None:
    layout = PlayerArrayLayout.GAIA_FIRST
    assert layout.index_for(0) == 0
    assert [layout.index_for(p) for p in range(1, 9)] == list(range(1, 9))


def test_no_gaia_has_no_slot_for_player_0() -> None:
    layout = PlayerArrayLayout.NO_GAIA
    assert [layout.index_for(p) for p in range(1, 9)] == list(range(8))
    with pytest.raises(ValueError):
        layout.index_for(0)


def test_index_for_rejects_out_of_range_player_id() -> None:
    with pytest.raises(ValueError):
        PlayerArrayLayout.P1_TO_P8_THEN_GAIA.index_for(9)
    with pytest.raises(ValueError):
        PlayerArrayLayout.P1_TO_P8_THEN_GAIA.index_for(-1)


# -- specs_for(): presence + fallback gate ------------------------------------


def test_every_spec_present_by_default() -> None:
    loaded = _stub_loaded()
    assert {s.field_id for s in specs_for(loaded)} == {s.field_id for s in player_fields._SPECS}


def test_a_missing_section_drops_every_spec_in_it() -> None:
    overrides = {
        (s.section, s.retriever): None for s in player_fields._SPECS if s.section == "Map"
    }
    loaded = _stub_loaded(overrides)
    ids = {s.field_id for s in specs_for(loaded)}
    assert "initial_view_x" not in ids
    # pop_limit's primary is also Map -- it survives via its Units fallback
    # only if that fallback is itself present, which this stub doesn't add.
    assert "pop_limit" not in ids
    assert "tribe_name" in ids  # an unrelated section is untouched


def test_a_zero_length_retriever_is_treated_as_absent() -> None:
    loaded = _stub_loaded({("Options", "per_player_starting_age"): _StubRetriever(b"")})
    ids = {s.field_id for s in specs_for(loaded)}
    assert "starting_age" not in ids
    assert "base_priority" in ids


def test_pop_limit_falls_back_when_the_primary_is_absent() -> None:
    """The pre-1.44 corpus shape: Map.per_player_population_cap doesn't
    exist, but Units.player_data_4 does."""
    entries = [SimpleNamespace(retriever_map={"population_limit": SimpleNamespace(data=v)}) for v in range(8)]
    loaded = _stub_loaded({
        ("Map", "per_player_population_cap"): None,
        ("Units", "player_data_4"): _StubRetriever(b"\x00" * 4, data=entries),
    })
    ids = {s.field_id for s in specs_for(loaded)}
    assert "pop_limit" in ids


def test_pop_limit_is_absent_when_neither_primary_nor_fallback_is_present() -> None:
    loaded = _stub_loaded({("Map", "per_player_population_cap"): None})
    ids = {s.field_id for s in specs_for(loaded)}
    assert "pop_limit" not in ids


# -- current_value() -----------------------------------------------------------


def test_current_value_reads_the_stub_data_at_the_layout_index() -> None:
    loaded = _stub_loaded({
        ("Options", "per_player_starting_age"): _StubRetriever(b"\x00" * 4, data=[3, 4, 5, 6, 2, 2, 2, 2, 999]),
    })
    spec = next(s for s in specs_for(loaded) if s.field_id == "starting_age")
    assert current_value(loaded, spec, 1) == 3  # P1 -> index 0
    assert current_value(loaded, spec, 4) == 6  # P4 -> index 3
    assert current_value(loaded, spec, 0) == 999  # GAIA -> index 8


def test_current_value_reads_a_struct_field() -> None:
    entries = [SimpleNamespace(retriever_map={"food": SimpleNamespace(data=v)}) for v in range(16)]
    loaded = _stub_loaded({("PlayerDataTwo", "resources"): _StubRetriever(b"\x00" * 4, data=entries)})
    spec = next(s for s in specs_for(loaded) if s.field_id == "food")
    assert current_value(loaded, spec, 1) == 0
    assert current_value(loaded, spec, 8) == 7  # P8 -> index 7


def test_current_value_reads_via_fallback_and_coerces_a_whole_float() -> None:
    entries = [SimpleNamespace(retriever_map={"population_limit": SimpleNamespace(data=200.0)}) for _ in range(8)]
    loaded = _stub_loaded({
        ("Map", "per_player_population_cap"): None,
        ("Units", "player_data_4"): _StubRetriever(b"\x00" * 4, data=entries),
    })
    spec = next(s for s in specs_for(loaded) if s.field_id == "pop_limit")
    value = current_value(loaded, spec, 1)
    assert value == 200
    assert isinstance(value, int)


def test_current_value_raises_on_a_fractional_float() -> None:
    entries = [SimpleNamespace(retriever_map={"population_limit": SimpleNamespace(data=200.5)}) for _ in range(8)]
    loaded = _stub_loaded({
        ("Map", "per_player_population_cap"): None,
        ("Units", "player_data_4"): _StubRetriever(b"\x00" * 4, data=entries),
    })
    spec = next(s for s in specs_for(loaded) if s.field_id == "pop_limit")
    with pytest.raises(TypeError):
        current_value(loaded, spec, 1)


def test_current_value_prefers_the_map_array_when_the_two_copies_disagree() -> None:
    """Pins the divergence direction confirmed in-game 2026-08-30: opening a
    file where the two copies disagree in the real AoE2:DE Scenario Editor
    showed the Map array's value, not player_data_4's -- so the primary
    (Map) wins when both are present, and player_data_4 is read only when
    Map itself is absent (pre-1.44)."""
    entries = [SimpleNamespace(retriever_map={"population_limit": SimpleNamespace(data=75.0)}) for _ in range(8)]
    loaded = _stub_loaded({
        ("Units", "player_data_4"): _StubRetriever(b"\x00" * 4, data=entries),
        ("Map", "per_player_population_cap"): _StubRetriever(b"\x00" * 4, data=[200] * 16),
    })
    spec = next(s for s in specs_for(loaded) if s.field_id == "pop_limit")
    assert current_value(loaded, spec, 1) == 200


def test_current_value_raises_rather_than_return_a_list() -> None:
    entries = [[1, 2]] * 16
    loaded = _stub_loaded({("Options", "per_player_starting_age"): _StubRetriever(b"\x00" * 4, data=entries)})
    spec = next(s for s in specs_for(loaded) if s.field_id == "starting_age")
    with pytest.raises(TypeError):
        current_value(loaded, spec, 1)


def test_current_value_raises_for_a_player_id_the_layout_has_no_slot_for() -> None:
    loaded = _stub_loaded()
    spec = next(s for s in specs_for(loaded) if s.field_id == "tribe_name")  # NO_GAIA
    with pytest.raises(ValueError):
        current_value(loaded, spec, 0)


# -- against a real load -------------------------------------------------------


def test_specs_for_the_blank_template() -> None:
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    ids = {s.field_id for s in specs_for(loaded)}
    assert ids == {s.field_id for s in player_fields._SPECS}  # 1.58: everything present


def test_every_spec_resolves_to_a_scalar_for_every_p1_to_p8_player() -> None:
    loaded = scenario_io.load_map_and_units(scenario_io.BLANK_TEMPLATE_PATH)
    for spec in specs_for(loaded):
        for player_id in range(1, 9):
            value = current_value(loaded, spec, player_id)
            assert isinstance(value, (int, str)), (spec.field_id, player_id)


def test_ai_files_is_never_read() -> None:
    """The maintainer plan's hard requirement: a single row must not be able
    to dump a multi-megabyte embedded .ai script."""
    assert "ai_files" not in {(s.section, s.retriever) for s in player_fields._SPECS}


def test_never_reads_number_of_players_or_a_swap_players_field() -> None:
    """Tier 3 (derived / not a field) is not a spec at all."""
    ids = {s.field_id for s in player_fields._SPECS}
    assert "number_of_players" not in ids
    assert "swap_players" not in ids


# -- corpus: the layout mapping and the version-gated fallback ---------------


@pytest.mark.corpus
def test_p1_to_p8_then_gaia_specs_put_gaia_where_gaia_player_index_says(scenario_path) -> None:
    """DataHeader.gaia_player_index (only present at 1.53+) is independent,
    library-confirmed ground truth for where GAIA sits in the arrays this
    layout describes -- see player_fields.PlayerArrayLayout's docstring."""
    loaded = scenario_io.load_map_and_units(scenario_path)
    dh = loaded._scenario.sections.get("DataHeader")
    gpi = dh.retriever_map.get("gaia_player_index") if dh else None
    if gpi is None or gpi.data is None:
        pytest.skip(f"{scenario_path.name}: no gaia_player_index (pre-1.53)")
    assert PlayerArrayLayout.P1_TO_P8_THEN_GAIA.index_for(0) == gpi.data


@pytest.mark.corpus
def test_every_spec_element_count_matches_its_retriever_length(scenario_path) -> None:
    """Would catch a future base_priority-shaped surprise: an array whose
    real length disagrees with the 16-vs-8 assumption baked into its
    layout's index_for()."""
    loaded = scenario_io.load_map_and_units(scenario_path)
    for spec in specs_for(loaded):
        if player_fields._retriever_present(loaded, spec.section, spec.retriever):
            section_name, retriever_name, layout = spec.section, spec.retriever, spec.layout
        else:
            section_name, retriever_name, _ = spec.fallback
            layout = PlayerArrayLayout.NO_GAIA
        retriever = loaded._scenario.sections[section_name].retriever_map[retriever_name]
        n = len(retriever.data)
        if layout is PlayerArrayLayout.NO_GAIA:
            assert n >= 8, (scenario_path.name, spec.field_id, n)
        else:
            assert n >= 9, (scenario_path.name, spec.field_id, n)


@pytest.mark.corpus
def test_every_spec_resolves_on_the_corpus_for_every_p1_to_p8_player(scenario_path) -> None:
    loaded = scenario_io.load_map_and_units(scenario_path)
    for spec in specs_for(loaded):
        for player_id in range(1, 9):
            value = current_value(loaded, spec, player_id)
            assert isinstance(value, (int, str)), (scenario_path.name, spec.field_id, player_id)


@pytest.mark.corpus
def test_pop_limit_reads_the_map_array_when_present(scenario_path) -> None:
    """8 of 20 corpus files disagree between the two stored copies (a flat
    200 in Map.per_player_population_cap vs. a real per-player value in
    Units.player_data_4). Confirmed in-game 2026-08-30 on one such file
    (ring75_v0_scx_resaved) that the real AoE2:DE Scenario Editor shows the
    Map array's value, not player_data_4's -- so that is the one this panel
    must show whenever Map is present (1.44+)."""
    loaded = scenario_io.load_map_and_units(scenario_path)
    specs = {s.field_id: s for s in specs_for(loaded)}
    if "pop_limit" not in specs:
        pytest.skip(f"{scenario_path.name}: pop_limit not present")
    map_section = loaded._scenario.sections.get("Map")
    map_retriever = map_section.retriever_map.get("per_player_population_cap") if map_section else None
    if map_retriever is None or not map_retriever.data:
        pytest.skip(f"{scenario_path.name}: pre-1.44, no Map array to read")
    for player_id in range(1, 9):
        expected = int(map_retriever.data[player_id - 1])
        assert current_value(loaded, specs["pop_limit"], player_id) == expected, (
            scenario_path.name, player_id,
        )


@pytest.mark.corpus
def test_civilization_is_int_below_1_56_and_str_from_1_56(scenario_path) -> None:
    loaded = scenario_io.load_map_and_units(scenario_path)
    specs = {s.field_id: s for s in specs_for(loaded)}
    value = current_value(loaded, specs["civilization"], 1)
    if float(loaded.scenario_version) >= 1.56:
        assert isinstance(value, str), scenario_path.name
    else:
        assert isinstance(value, int), scenario_path.name
