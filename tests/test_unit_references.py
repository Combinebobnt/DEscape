"""descape.unit_references: placed-unit reference_ids in trigger fields,
resolved to the unit they name. Default tier, no QApplication; one
corpus-marked test re-measures the plan's counts over examples/."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import library_compat, map_analysis, trigger_fields, unit_references
from descape.library_compat import VocabularyEntry
from descape.scenario_io import load_map_and_units, parse_triggers

import conftest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
_ARCHER = 4
_HOUSE = 70


def _name(const: int) -> str:
    """The display name, which depends on whether an install's strings are configured."""
    return trigger_fields.resolve_reference("UnitInfo", const)


def _unit(ref, const=_ARCHER, x=10.5, y=20.5):
    return SimpleNamespace(reference_id=ref, unit_const=const, x=x, y=y)


def _loaded(players, width=120, height=120):
    """A stand-in LoadedScenario: nine unit lists and a map size."""
    lists = [list(players.get(p, ())) for p in range(9)]
    return SimpleNamespace(
        unit_manager=SimpleNamespace(units=lists),
        map_manager=SimpleNamespace(map_width=width, map_height=height),
    )


def _every_shipped_version() -> list[str]:
    return library_compat.vocabulary_versions()


# -- the vocabulary sweep ---------------------------------------------------


def _field_names(entries, presentation) -> set[str]:
    return {f for d in entries.values() for f in unit_references.unit_reference_fields(d, presentation)}


def test_the_field_set_comes_from_presentation_and_has_the_v136_to_v143_condition_gap() -> None:
    effect_fields = {"location_object_reference", "selected_object_ids"}
    condition_fields = {"unit_object", "next_object"}
    seen = {}
    for version in _every_shipped_version():
        vocabulary = library_compat.load_vocabulary(version)
        seen[version] = (
            _field_names(vocabulary.conditions, vocabulary.condition_presentation),
            _field_names(vocabulary.effects, vocabulary.effect_presentation),
        )
    assert seen, "no shipped vocabulary found"
    for version, (conditions, effects) in seen.items():
        assert effects == effect_fields, version
        major, minor = (int(p) for p in version.split("."))
        # unit_object/next_object carry presentation '' before v1.44, so they are not references there.
        assert conditions == (condition_fields if (major, minor) >= (1, 44) else set()), version


def test_only_the_list_field_is_list_presented() -> None:
    vocabulary = library_compat.load_vocabulary("1.58")
    assert vocabulary.effect_presentation["selected_object_ids"] == unit_references.LIST_PRESENTATION
    assert vocabulary.effect_presentation["location_object_reference"] == "Unit"
    assert vocabulary.condition_presentation["unit_object"] == "Unit"


def test_a_missing_definition_has_no_reference_fields() -> None:
    assert unit_references.unit_reference_fields(None, {"unit_object": "Unit"}) == ()


# -- the index ----------------------------------------------------------------


def test_the_index_holds_plain_ints_owner_tile_and_bounds() -> None:
    house = _unit(200, _HOUSE, 41.0, 63.0)
    index = unit_references.build_reference_index(_loaded({1: [house], 2: [_unit(300, x=50.5, y=55.5)]}))
    ref = index.by_id[200]
    assert (ref.reference_id, ref.player_id, ref.unit_const, ref.own_tile) == (200, 1, _HOUSE, (41, 63))
    # render.unit_tile_bounds() order: (x0, x1, y0, y1), half-open.
    assert ref.bounds == (40, 42, 62, 64)
    assert index.by_id[300].player_id == 2
    assert all(type(v) is int for v in (ref.reference_id, ref.player_id, ref.unit_const, *ref.own_tile, *ref.bounds))


def test_a_duplicate_id_keeps_the_first_unit_and_is_flagged() -> None:
    first, second = _unit(7, x=1.5, y=1.5), _unit(7, _HOUSE, 30.0, 30.0)
    index = unit_references.build_reference_index(_loaded({0: [first], 3: [second]}))
    assert index.by_id[7].player_id == 0 and index.by_id[7].own_tile == (1, 1)
    assert index.duplicates == frozenset({7})
    assert unit_references.describe(index, 7).endswith("(id placed more than once)")


def test_an_off_map_unit_is_indexed_with_no_bounds() -> None:
    index = unit_references.build_reference_index(_loaded({1: [_unit(9, x=130.5, y=4.5)]}))
    assert index.by_id[9].bounds is None
    assert unit_references.describe(index, 9) == f"{_name(_ARCHER)} (4) [P1, off the map]"


def test_the_fixture_index_matches_the_generator() -> None:
    gen = conftest.load_verify_module("gen_trigger_fixture")
    loaded = load_map_and_units(FIXTURE_PATH)
    index = unit_references.build_reference_index(loaded)
    assert set(index.by_id) == set(gen.PLACED_REFERENCE_IDS)
    assert index.duplicates == frozenset()
    assert gen.DANGLING_REFERENCE_ID not in index.by_id


# -- patch_reference_index ------------------------------------------------------


def _splice(player_id, unit, old_own, new_own, old_player_id=None):
    from descape.render_cache import UnitSplice

    # The tile lists are never read by the patch: empty on both sides here.
    return UnitSplice(player_id, 0, unit, old_own, new_own, (), (), old_player_id=old_player_id)


def _own(unit):
    return (int(unit.x), int(unit.y))


def _assert_matches_fresh(index, loaded) -> None:
    fresh = unit_references.build_reference_index(loaded)
    assert dict(index.by_id) == dict(fresh.by_id)
    assert index.duplicates == fresh.duplicates


def test_a_patch_follows_a_move_a_const_swap_an_add_a_remove_and_a_reassign() -> None:
    archer, house, gate = _unit(10), _unit(11, _HOUSE, 41.0, 63.0), _unit(12, x=5.5, y=5.5)
    loaded = _loaded({1: [archer, house], 2: [gate]})
    index = unit_references.build_reference_index(loaded)

    old = _own(archer)
    archer.x, archer.y = 118.5, 3.5
    assert unit_references.patch_reference_index(index, loaded, [_splice(1, archer, old, _own(archer))])
    _assert_matches_fresh(index, loaded)

    old = _own(gate)
    gate.unit_const, gate.x = _HOUSE, 7.0
    assert unit_references.patch_reference_index(index, loaded, [_splice(2, gate, old, _own(gate))])
    _assert_matches_fresh(index, loaded)

    added = _unit(13, x=130.5, y=1.5)  # off the map: no tiles, still an add
    loaded.unit_manager.units[0].append(added)
    loaded.unit_manager.units[1].remove(house)
    changed = [_splice(1, house, _own(house), None), _splice(0, added, None, _own(added))]
    assert unit_references.patch_reference_index(index, loaded, changed)
    _assert_matches_fresh(index, loaded)

    loaded.unit_manager.units[1].remove(archer)
    loaded.unit_manager.units[3].append(archer)
    assert unit_references.patch_reference_index(
        index, loaded, [_splice(3, archer, _own(archer), _own(archer), old_player_id=1)]
    )
    _assert_matches_fresh(index, loaded)


def test_a_removal_then_an_arrival_of_the_same_id_patches_in_list_order() -> None:
    leaving, arriving = _unit(20, x=1.5, y=1.5), _unit(20, x=9.5, y=9.5)
    loaded = _loaded({0: [leaving]})
    index = unit_references.build_reference_index(loaded)
    loaded.unit_manager.units[0][:] = [arriving]
    changed = [_splice(0, leaving, _own(leaving), None), _splice(0, arriving, None, _own(arriving))]
    assert unit_references.patch_reference_index(index, loaded, changed)
    _assert_matches_fresh(index, loaded)


def test_a_patch_refuses_what_depends_on_list_order_or_disagrees_with_the_index() -> None:
    first, second, lone = _unit(7, x=1.5, y=1.5), _unit(7, x=2.5, y=2.5), _unit(8)
    loaded = _loaded({1: [first, lone], 2: [second]})

    def refused(splice) -> bool:
        index = unit_references.build_reference_index(loaded)
        return not unit_references.patch_reference_index(index, loaded, [splice])

    assert refused(_splice(2, second, _own(second), _own(second))), "a duplicated id"
    assert refused(_splice(1, lone, None, _own(lone))), "an add onto a present id"
    assert refused(_splice(1, _unit(99), _own(lone), None)), "a removal of an absent id"
    assert refused(_splice(4, lone, _own(lone), _own(lone))), "an owner the index does not hold"
    assert refused(_splice(4, lone, _own(lone), _own(lone), old_player_id=3)), "a reassign from the wrong owner"
    assert not unit_references.patch_reference_index(unit_references.EMPTY_INDEX, loaded, [])


# -- references_in and describe ----------------------------------------------


def _definition(*attributes: str) -> VocabularyEntry:
    return VocabularyEntry(id=0, name="probe", attributes=attributes, default_attributes={})


_PRESENTATION = {"unit_object": "Unit", "selected_object_ids": "Unit[]", "quantity": ""}


def test_references_in_skips_unset_and_non_reference_fields() -> None:
    definition = _definition("unit_object", "selected_object_ids", "quantity")
    entry = SimpleNamespace(unit_object=12, selected_object_ids=[3, -1, 4], quantity=12)
    assert unit_references.references_in(entry, definition, _PRESENTATION) == {
        "unit_object": (12,),
        "selected_object_ids": (3, 4),
    }
    for unset in (None, -1, []):
        entry = SimpleNamespace(unit_object=unset, selected_object_ids=unset, quantity=5)
        assert unit_references.references_in(entry, definition, _PRESENTATION) == {}, unset


def test_references_in_tolerates_a_raising_version_gated_field() -> None:
    class Gated:
        @property
        def unit_object(self):
            raise AttributeError("not in this version")

    assert unit_references.references_in(Gated(), _definition("unit_object"), _PRESENTATION) == {}


def test_describe_formats_resolved_dangling_and_unset() -> None:
    index = unit_references.build_reference_index(_loaded({1: [_unit(201, x=50.5, y=55.5)]}))
    assert unit_references.describe(index, 201) == f"{_name(_ARCHER)} (4) [P1, X50, Y55]"
    assert unit_references.describe(index, 99999) == "99999: not placed on the map"
    for unset in (None, -1):
        assert unit_references.describe(index, unset) == ""


def test_the_dangling_check_and_describe_agree_on_the_fixture() -> None:
    """One walker: every id the analysis check reports is exactly an id describe() calls unplaced."""
    gen = conftest.load_verify_module("gen_trigger_fixture")
    loaded = load_map_and_units(FIXTURE_PATH)
    result = map_analysis.check_unit_references(loaded)
    assert [f"unit id {gen.DANGLING_REFERENCE_ID}," in f.message for f in result.findings] == [True]
    index = unit_references.build_reference_index(loaded)
    assert unit_references.describe(index, gen.DANGLING_REFERENCE_ID).endswith("not placed on the map")
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    task = manager.triggers[gen.REFERENCE_TRIGGER].effects[gen.TASK_OBJECT_EFFECT]
    definition = vocabulary.effects[task.effect_type]
    refs = unit_references.references_in(task, definition, vocabulary.effect_presentation)
    assert refs == {
        "selected_object_ids": gen.TASK_SELECTED_IDS,
        "location_object_reference": (gen.TASK_TARGET_ID,),
    }


# -- corpus -------------------------------------------------------------------


@pytest.mark.corpus
def test_corpus_unit_references_match_the_plan() -> None:
    """The plan's table: 852 non-empty references, 1 dangling, no duplicate
    ids, none off-map or garrisoned, and the 4 unparseable files degrade."""
    examples = Path(__file__).resolve().parent.parent / "examples"
    files = sorted(examples.glob("*.aoe2scenario"))
    if len(files) != 20:
        pytest.skip(f"expects the 20-file examples/ corpus, found {len(files)}")
    total = dangling = duplicates = off_map = garrisoned = 0
    unparseable = []
    for path in files:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
        if manager is None:
            unparseable.append(path.name)
            assert map_analysis.check_unit_references(loaded).unavailable
            continue
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        index = unit_references.build_reference_index(loaded)
        duplicates += len(index.duplicates)
        inside = {
            u.reference_id
            for _p, u in unit_references.all_units(loaded)
            if getattr(u, "garrisoned_in_id", -1) not in (-1, None, u.reference_id)
        }
        kinds = (
            ("conditions", "condition_type", vocabulary.conditions, vocabulary.condition_presentation),
            ("effects", "effect_type", vocabulary.effects, vocabulary.effect_presentation),
        )
        for trigger in manager.triggers:
            for list_name, type_attribute, definitions, presentation in kinds:
                for entry in getattr(trigger, list_name):
                    definition = definitions.get(getattr(entry, type_attribute))
                    for ids in unit_references.references_in(entry, definition, presentation).values():
                        for ref_id in ids:
                            total += 1
                            ref = index.get(ref_id)
                            dangling += ref is None
                            off_map += ref is not None and ref.bounds is None
                            garrisoned += ref_id in inside
    assert (total, dangling, duplicates, off_map, garrisoned) == (852, 1, 0, 0, 0)
    assert len(unparseable) == 4
