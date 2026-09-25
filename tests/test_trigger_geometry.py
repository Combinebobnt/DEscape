"""descape.trigger_geometry: the Qt-free reader for trigger areas, locations
and wall runs (GH #41). Default tier, no QApplication; one corpus-marked test
re-measures the plan's counts over examples/."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from descape import library_compat, ruler, trigger_geometry
from descape.library_compat import VocabularyEntry
from descape.scenario_io import load_map_and_units, parse_triggers

import conftest

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"


def _definition(*attributes: str) -> VocabularyEntry:
    return VocabularyEntry(id=0, name="probe", attributes=attributes, default_attributes={})


def _shapes(entry, definition):
    return trigger_geometry.shapes_for_entry(entry, definition, trigger_index=7, entry_kind="effect", entry_index=3)


# -- the vocabulary sweep ---------------------------------------------------


def _every_shipped_version() -> list[str]:
    return library_compat.vocabulary_versions()


def test_every_shipped_version_lists_only_whole_coordinate_groups() -> None:
    """coordinate_fields() drops a partly-listed group; this proves none exists
    to drop, so no real field is silently missed by that rule."""
    groups = (trigger_geometry.AREA_FIELDS, trigger_geometry.LOCATION_FIELDS, trigger_geometry.WALL_FIELDS)
    offenders = []
    for version in _every_shipped_version():
        for kind, entries in library_compat.vocabulary_json(version).items():
            for key, value in entries.items():
                if key == "-1":
                    continue
                attributes = set(value.get("attributes", ()))
                for fields in groups:
                    present = sum(f in attributes for f in fields)
                    if 0 < present < len(fields):
                        offenders.append(f"v{version}/{kind}/{value['name']}: {fields}")
    assert offenders == []


def test_the_area_set_is_version_dependent_and_walls_are_v157_build_object_only() -> None:
    area_effects = {}
    wall_effects = {}
    for version in _every_shipped_version():
        vocabulary = library_compat.load_vocabulary(version)
        area_effects[version] = sum(
            trigger_geometry.AREA_FIELDS in trigger_geometry.coordinate_fields(d) for d in vocabulary.effects.values()
        )
        wall_effects[version] = sorted(
            d.name for d in vocabulary.effects.values()
            if trigger_geometry.WALL_FIELDS in trigger_geometry.coordinate_fields(d)
        )
    assert area_effects["1.36"] == 22
    assert area_effects["1.40"] == 30
    assert area_effects["1.54"] == 39
    assert area_effects["1.58"] == 44
    assert {v: names for v, names in wall_effects.items() if names} == {
        "1.57": ["build_object"],
        "1.58": ["build_object"],
        "1.59": ["build_object"],
    }


def test_coordinate_fields_keeps_area_location_wall_order() -> None:
    definition = _definition(*trigger_geometry.WALL_FIELDS, *trigger_geometry.LOCATION_FIELDS,
                             *trigger_geometry.AREA_FIELDS, "quantity")
    assert trigger_geometry.coordinate_fields(definition) == (
        trigger_geometry.AREA_FIELDS,
        trigger_geometry.LOCATION_FIELDS,
        trigger_geometry.WALL_FIELDS,
    )
    assert trigger_geometry.coordinate_fields(None) == ()
    assert trigger_geometry.coordinate_fields(_definition("area_x1", "area_y1", "quantity")) == ()


# -- the reading rules -------------------------------------------------------


def test_none_reads_as_unset_and_yields_no_shape() -> None:
    definition = _definition(*trigger_geometry.AREA_FIELDS, *trigger_geometry.LOCATION_FIELDS)
    entry = SimpleNamespace(area_x1=None, area_y1=None, area_x2=None, area_y2=None, location_x=None, location_y=None)
    assert _shapes(entry, definition) == []


def test_a_field_the_entry_does_not_carry_reads_as_unset() -> None:
    definition = _definition(*trigger_geometry.LOCATION_FIELDS)
    assert _shapes(SimpleNamespace(location_x=4), definition) == []


def test_a_half_set_area_yields_no_shape_in_either_direction() -> None:
    definition = _definition(*trigger_geometry.AREA_FIELDS)
    for coords in ((-1, 5, 12, 9), (3, 5, -1, 9), (3, -1, 12, 9), (3, 5, 12, -1)):
        entry = SimpleNamespace(**dict(zip(trigger_geometry.AREA_FIELDS, coords, strict=True)))
        assert _shapes(entry, definition) == [], coords


def test_coordinates_are_never_re_normalized() -> None:
    """validate_coords() already swapped at parse time; a second pass here
    would double-apply, so a swapped pair is reported verbatim."""
    definition = _definition(*trigger_geometry.AREA_FIELDS)
    entry = SimpleNamespace(area_x1=20, area_y1=9, area_x2=5, area_y2=3)
    (shape,) = _shapes(entry, definition)
    assert shape.coords == (20, 9, 5, 3)


def test_a_shape_is_plain_frozen_ints_with_its_source_fields() -> None:
    definition = _definition(*trigger_geometry.AREA_FIELDS, *trigger_geometry.LOCATION_FIELDS)
    entry = SimpleNamespace(area_x1=1, area_y1=2, area_x2=3, area_y2=4, location_x=5, location_y=6)
    area, location = _shapes(entry, definition)
    assert (area.shape, area.fields, area.coords) == ("area", trigger_geometry.AREA_FIELDS, (1, 2, 3, 4))
    assert (location.shape, location.coords) == ("location", (5, 6))
    assert (area.trigger_index, area.entry_ref) == (7, ("effect", 3))
    assert all(type(c) is int for c in area.coords + location.coords)
    with pytest.raises(AttributeError):
        area.coords = (0, 0, 0, 0)


def test_a_raising_attribute_reads_as_unset() -> None:
    class Gated:
        area_x1 = area_y1 = area_x2 = 1

        @property
        def area_y2(self):
            raise RuntimeError("unsupported in this version")

    assert _shapes(Gated(), _definition(*trigger_geometry.AREA_FIELDS)) == []


# -- the run ------------------------------------------------------------------


def test_run_is_area_centre_to_location_through_the_ruler() -> None:
    definition = _definition(*trigger_geometry.AREA_FIELDS, *trigger_geometry.LOCATION_FIELDS)
    entry = SimpleNamespace(area_x1=30, area_y1=40, area_x2=34, area_y2=46, location_x=60, location_y=50)
    run = trigger_geometry.run_for_entry(_shapes(entry, definition))
    assert run == ruler.measure((32, 43), (60, 50))
    assert ruler.format_measurement(run) == "28.9 tiles  (dx +28, dy +7)"


def test_run_needs_both_an_area_and_a_location() -> None:
    area_only = SimpleNamespace(area_x1=1, area_y1=1, area_x2=2, area_y2=2)
    location_only = SimpleNamespace(location_x=1, location_y=1)
    assert trigger_geometry.run_for_entry(_shapes(area_only, _definition(*trigger_geometry.AREA_FIELDS))) is None
    assert (
        trigger_geometry.run_for_entry(_shapes(location_only, _definition(*trigger_geometry.LOCATION_FIELDS)))
        is None
    )
    assert trigger_geometry.run_for_entry([]) is None


def test_area_centre_is_the_integer_midpoint() -> None:
    assert trigger_geometry.area_centre((0, 0, 119, 119)) == (59, 59)
    assert trigger_geometry.area_centre((5, 5, 5, 5)) == (5, 5)


# -- the fixture --------------------------------------------------------------


GEOMETRY_TRIGGER = 3


def test_the_fixture_geometry_reads_back_as_generated() -> None:
    gen = conftest.load_verify_module("gen_trigger_fixture")
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    index = GEOMETRY_TRIGGER
    shapes = trigger_geometry.shapes_for_trigger(manager.triggers[index], vocabulary, trigger_index=index)
    assert [(s.entry_ref, s.shape, s.coords) for s in shapes] == [
        # condition 0 is the timer, effect 0 the change_variable.
        (("condition", 1), "area", gen.GEOMETRY_CONDITION_AREA),
        (("effect", 1), "area", gen.GEOMETRY_PATROL_AREA),
        (("effect", 1), "location", gen.GEOMETRY_PATROL_LOCATION),
        (("effect", 2), "location", gen.GEOMETRY_CREATE_LOCATION),
        # effect 3 is the half-set area: no shape.
        (("effect", 4), "area", gen.GEOMETRY_WHOLE_MAP_AREA),
        # The task_object's location, the referenced house's own tile.
        (("effect", gen.TASK_OBJECT_EFFECT), "location", gen.HOUSE_TILE),
    ]
    assert all(s.trigger_index == index for s in shapes)
    patrol = [s for s in shapes if s.entry_ref == ("effect", 1)]
    assert trigger_geometry.run_for_entry(patrol) == ruler.measure((32, 43), gen.GEOMETRY_PATROL_LOCATION)
    # The other three triggers carry no coordinates at all.
    for other in range(len(manager.triggers)):
        if other != index:
            shapes = trigger_geometry.shapes_for_trigger(manager.triggers[other], vocabulary, trigger_index=other)
            assert shapes == [], other


# -- corpus -------------------------------------------------------------------


@pytest.mark.corpus
def test_corpus_counts_match_the_plan() -> None:
    """The plan's measurements: 16 of 20 files parse triggers, 1428 areas,
    zero half-set groups, and the 4 unparseable files degrade to no shapes."""
    examples = Path(__file__).resolve().parent.parent / "examples"
    files = sorted(examples.glob("*.aoe2scenario"))
    if len(files) != 20:
        pytest.skip(f"expects the 20-file examples/ corpus, found {len(files)}")
    areas = partial = parsed = 0
    unparseable = []
    for path in files:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
        if manager is None:
            unparseable.append(path.name)
            continue
        parsed += 1
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        for index, trigger in enumerate(manager.triggers):
            shapes = trigger_geometry.shapes_for_trigger(trigger, vocabulary, trigger_index=index)
            areas += sum(s.shape == trigger_geometry.SHAPE_AREA for s in shapes)
            for kind, list_name, type_attribute in trigger_geometry._KINDS:
                definitions = vocabulary.conditions if kind == "condition" else vocabulary.effects
                for entry in getattr(trigger, list_name):
                    for fields in trigger_geometry.coordinate_fields(definitions.get(getattr(entry, type_attribute))):
                        set_count = sum(trigger_geometry._coordinate(entry, f) >= 0 for f in fields)
                        partial += 0 < set_count < len(fields)
    assert (parsed, areas, partial) == (16, 1428, 0)
    assert sorted(unparseable) == [
        "2_Joan_coop_1_v0_13.aoe2scenario",
        "2_Joan_coop_3_v0_14.aoe2scenario",
        "2_Joan_coop_4_v0_13.aoe2scenario",
        "2_Joan_coop_6_v0_14.aoe2scenario",
    ]


# -- referenced units (unit shapes) -------------------------------------------


def _ref(ref_id, own, bounds, player=1, const=70):
    from descape.unit_references import UnitRef

    return UnitRef(ref_id, player, const, own, bounds)


def _index(*refs):
    from types import MappingProxyType

    from descape.unit_references import ReferenceIndex

    return ReferenceIndex(MappingProxyType({r.reference_id: r for r in refs}), frozenset())


_UNIT_PRESENTATION = {"selected_object_ids": "Unit[]", "location_object_reference": "Unit", "unit_object": "Unit"}


def _unit_shapes(entry, definition, index):
    return trigger_geometry.shapes_for_entry(
        entry, definition, trigger_index=7, entry_kind="effect", entry_index=3,
        references=index, presentation=_UNIT_PRESENTATION,
    )


def test_a_unit_shape_is_its_footprint_as_inclusive_corners_in_area_order() -> None:
    """render.unit_tile_bounds() is (x0, x1, y0, y1) half-open; an asymmetric
    3x2 footprint at x != y catches a transposed or half-open conversion."""
    index = _index(_ref(5, (41, 62), (40, 43, 61, 63)))
    (shape,) = _unit_shapes(SimpleNamespace(unit_object=5), _definition("unit_object"), index)
    assert (shape.shape, shape.fields, shape.coords) == ("unit", ("unit_object",), (40, 61, 42, 62))
    assert (shape.anchor, shape.reference_id, shape.entry_ref) == ((41, 62), 5, ("effect", 3))


def test_dangling_off_map_and_unset_references_yield_no_unit_shape() -> None:
    index = _index(_ref(5, (130, 4), None))
    definition = _definition("selected_object_ids", "location_object_reference")
    entry = SimpleNamespace(selected_object_ids=[5, 99, -1], location_object_reference=-1)
    assert _unit_shapes(entry, definition, index) == []


def test_without_an_index_no_unit_shape_is_read() -> None:
    index = _index(_ref(5, (41, 62), (40, 43, 61, 63)))
    entry = SimpleNamespace(unit_object=5)
    assert _shapes(entry, _definition("unit_object")) == []
    assert trigger_geometry.shapes_for_entry(
        entry, _definition("unit_object"), trigger_index=0, entry_kind="effect", entry_index=0, references=index
    ) == [], "no presentation map, no reference fields"


def test_the_run_goes_from_the_selection_centroid_to_the_location_object() -> None:
    index = _index(
        _ref(1, (50, 55), (50, 51, 55, 56)),
        _ref(2, (53, 58), (53, 54, 58, 59)),
        _ref(3, (41, 63), (40, 42, 62, 64)),
    )
    definition = _definition("selected_object_ids", "location_object_reference", *trigger_geometry.LOCATION_FIELDS)
    entry = SimpleNamespace(selected_object_ids=[1, 2], location_object_reference=3, location_x=70, location_y=80)
    shapes = _unit_shapes(entry, definition, index)
    # The location object wins over the stored location, and makes it an authoring copy.
    assert trigger_geometry.run_for_entry(shapes) == ruler.measure((51, 56), (41, 63))
    assert trigger_geometry.location_superseded(shapes)
    entry.location_object_reference = -1
    shapes = _unit_shapes(entry, definition, index)
    assert trigger_geometry.run_for_entry(shapes) == ruler.measure((51, 56), (70, 80))
    assert not trigger_geometry.location_superseded(shapes)


def test_an_area_origin_wins_over_the_selection_and_no_destination_means_no_run() -> None:
    index = _index(_ref(1, (50, 55), (50, 51, 55, 56)), _ref(3, (41, 63), (40, 42, 62, 64)))
    definition = _definition("selected_object_ids", "location_object_reference", *trigger_geometry.AREA_FIELDS)
    entry = SimpleNamespace(
        selected_object_ids=[1], location_object_reference=3, area_x1=10, area_y1=20, area_x2=14, area_y2=30
    )
    assert trigger_geometry.run_for_entry(_unit_shapes(entry, definition, index)) == ruler.measure((12, 25), (41, 63))
    entry = SimpleNamespace(selected_object_ids=[1], location_object_reference=-1, area_x1=-1, area_y1=-1,
                            area_x2=-1, area_y2=-1)
    assert trigger_geometry.run_for_entry(_unit_shapes(entry, definition, index)) is None


def test_the_fixture_reference_entries_read_back_as_unit_shapes() -> None:
    from descape import unit_references

    gen = conftest.load_verify_module("gen_trigger_fixture")
    loaded = load_map_and_units(FIXTURE_PATH)
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    index = unit_references.build_reference_index(loaded)
    trigger = gen.REFERENCE_TRIGGER
    shapes = trigger_geometry.shapes_for_trigger(
        manager.triggers[trigger], vocabulary, trigger_index=trigger, references=index
    )
    units = [(s.entry_ref, s.fields[0], s.reference_id) for s in shapes if s.shape == "unit"]
    assert units == [
        (("condition", gen.DESTROY_CONDITION), "unit_object", gen.REF_HOUSE),
        # The dangling condition draws nothing.
        (("effect", gen.TASK_OBJECT_EFFECT), "location_object_reference", gen.TASK_TARGET_ID),
        *((("effect", gen.TASK_OBJECT_EFFECT), "selected_object_ids", i) for i in gen.TASK_SELECTED_IDS),
        *((("effect", gen.PATROL_IDS_EFFECT), "selected_object_ids", i) for i in gen.PATROL_SELECTED_IDS),
    ]
    house = next(s for s in shapes if s.reference_id == gen.REF_HOUSE)
    assert house.coords == (40, 62, 41, 63), "the 2x2 house at (41.0, 63.0)"
    task = [s for s in shapes if s.entry_ref == ("effect", gen.TASK_OBJECT_EFFECT)]
    centroid = tuple(sum(c) // 2 for c in zip(*(ref for ref in (index.by_id[i].own_tile for i in gen.TASK_SELECTED_IDS)), strict=True))
    assert trigger_geometry.run_for_entry(task) == ruler.measure(centroid, gen.HOUSE_TILE)
    patrol = [s for s in shapes if s.entry_ref == ("effect", gen.PATROL_IDS_EFFECT)]
    assert trigger_geometry.run_for_entry(patrol) is None, "selected ids and no destination: markers only"
