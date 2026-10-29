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


# -- GH #167: the units a trigger creates ------------------------------------

FIND_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "find_triggers_120x120.aoe2scenario"
_ARCHER = 4
_CREATE_OBJECT = 11
_TASK_OBJECT = 12
_PLACE_FOUNDATION = 25


def _created(path: Path) -> dict[int, list]:
    loaded = load_map_and_units(path)
    manager = parse_triggers(loaded)
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    return {
        index: trigger_geometry.created_units_for_trigger(trigger, vocabulary, trigger_index=index)
        for index, trigger in enumerate(manager.triggers)
    }


def test_the_triggers_fixtures_create_object_reads_with_its_unset_player() -> None:
    gen = conftest.load_verify_module("gen_trigger_fixture")
    created = _created(FIXTURE_PATH)
    assert created[GEOMETRY_TRIGGER] == [
        trigger_geometry.CreatedUnit(
            GEOMETRY_TRIGGER, 2, "create_object", _ARCHER, trigger_geometry.UNSET, gen.GEOMETRY_CREATE_LOCATION
        )
    ]
    (record,) = created[GEOMETRY_TRIGGER]
    assert record.entry_ref == ("effect", 2)
    assert record.payload == ("create_object", _ARCHER, -1, (70, 80), -1), "the memo key leaves both indices out"
    assert all(not records for index, records in created.items() if index != GEOMETRY_TRIGGER)


def test_the_find_fixtures_retarget_creates_one_player_one_archer() -> None:
    """Its Task Object, Patrol and Attack Move carry a unit and a location too, and create nothing."""
    gen = conftest.load_verify_module("gen_trigger_fixture")
    created = _created(FIND_FIXTURE_PATH)
    (record,) = created[gen.FIND_RETARGET_TRIGGER]
    assert (record.effect, record.unit_const, record.player, record.tile, record.facet) == (
        "create_object", _ARCHER, 1, (70, 70), -1
    )
    assert record.entry_ref == ("effect", 1)
    assert all(type(v) is int for v in (record.unit_const, record.player, record.facet, *record.tile))


def _vocabulary(*definitions: VocabularyEntry) -> SimpleNamespace:
    return SimpleNamespace(effects={d.id: d for d in definitions})


def _effect_definition(type_id: int, name: str, *attributes: str) -> VocabularyEntry:
    return VocabularyEntry(id=type_id, name=name, attributes=("effect_type", *attributes), default_attributes={})


def test_a_place_foundation_yields_one_record_with_no_facet() -> None:
    """It has no facet field, so even a stray facet attribute on the entry is not read."""
    vocabulary = _vocabulary(
        _effect_definition(_PLACE_FOUNDATION, "place_foundation", "object_list_unit_id", "source_player", "location_x", "location_y")
    )
    entry = SimpleNamespace(
        effect_type=_PLACE_FOUNDATION, object_list_unit_id=82, source_player=2, location_x=12, location_y=30, facet=5
    )
    trigger = SimpleNamespace(effects=[entry])
    assert trigger_geometry.created_units_for_trigger(trigger, vocabulary, trigger_index=4) == [
        trigger_geometry.CreatedUnit(4, 0, "place_foundation", 82, 2, (12, 30), trigger_geometry.UNSET)
    ]


def test_an_unset_const_or_a_half_set_location_creates_nothing_and_a_task_object_never_does() -> None:
    fields = ("object_list_unit_id", "source_player", "location_x", "location_y")
    vocabulary = _vocabulary(
        _effect_definition(_CREATE_OBJECT, "create_object", *fields, "facet"),
        _effect_definition(_TASK_OBJECT, "task_object", *fields),
    )

    def create(**values):
        base = {"effect_type": _CREATE_OBJECT, "object_list_unit_id": _ARCHER, "source_player": 1,
                "location_x": 5, "location_y": 6, "facet": 3}
        return SimpleNamespace(**{**base, **values})

    effects = [
        create(object_list_unit_id=-1),
        create(object_list_unit_id=None),
        create(location_x=-1),
        create(location_y=None),
        create(effect_type=_TASK_OBJECT),
        create(),
    ]
    records = trigger_geometry.created_units_for_trigger(SimpleNamespace(effects=effects), vocabulary, trigger_index=0)
    assert records == [trigger_geometry.CreatedUnit(0, 5, "create_object", _ARCHER, 1, (5, 6), 3)]


# -- corpus -------------------------------------------------------------------


# Per file, (areas, half-set coordinate groups), or None where parse_triggers()
# gives up. The plan measured the 20 files before the 1.59 example joined.
_JOINED_AFTER_THE_PLAN = "8tp3w9j.aoe2scenario"
_GEOMETRY_PER_FILE = {
    "0_June_Event_Scenario.aoe2scenario": (9, 0),
    "2_Joan_coop_1_v0_13.aoe2scenario": None,
    "2_Joan_coop_2_v0_15.aoe2scenario": (36, 0),
    "2_Joan_coop_3_v0_14.aoe2scenario": None,
    "2_Joan_coop_4_v0_13.aoe2scenario": None,
    "2_Joan_coop_5_v0_13.aoe2scenario": (47, 0),
    "2_Joan_coop_6_v0_14.aoe2scenario": None,
    # Its 7 half-set groups are authored area_x set, area_y -1: no shape.
    _JOINED_AFTER_THE_PLAN: (20, 7),
    "C2_ElCid_coop_1_v0_16.aoe2scenario": (80, 0),
    "C2_ElCid_coop_2_v0_13.aoe2scenario": (17, 0),
    "C2_ElCid_coop_3_v0_12.aoe2scenario": (28, 0),
    "C2_ElCid_coop_4_v0_13.aoe2scenario": (63, 0),
    "C2_ElCid_coop_5_v0_14.aoe2scenario": (38, 0),
    "C2_ElCid_coop_6_v0_13.aoe2scenario": (4, 0),
    "F7_2_Dos Pilas (648).aoe2scenario": (14, 0),
    "F7_3_York (865).aoe2scenario": (82, 0),
    "R4_LeLoi_4.aoe2scenario": (38, 0),
    "atilla_1_scn_resaved.aoe2scenario": (77, 0),
    "blank_map.aoe2scenario": (0, 0),
    "old-allies-final-v2.aoe2scenario": (895, 0),
    "ring75_v0_scx_resaved.aoe2scenario": (0, 0),
}


def test_the_per_file_geometry_table_sums_to_the_plan() -> None:
    """The plan's measurements: 16 of 20 files parse triggers, 1428 areas, zero half-set groups."""
    planned = [row for name, row in _GEOMETRY_PER_FILE.items() if name != _JOINED_AFTER_THE_PLAN]
    parsed = [row for row in planned if row is not None]
    assert (len(planned), len(parsed)) == (20, 16)
    assert (sum(a for a, _p in parsed), sum(p for _a, p in parsed)) == (1428, 0)


def _measure_geometry(path: Path) -> tuple[int, int] | None:
    loaded = load_map_and_units(path)
    manager = parse_triggers(loaded)
    if manager is None:
        return None
    areas = partial = 0
    vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
    for index, trigger in enumerate(manager.triggers):
        shapes = trigger_geometry.shapes_for_trigger(trigger, vocabulary, trigger_index=index)
        areas += sum(s.shape == trigger_geometry.SHAPE_AREA for s in shapes)
        for kind, list_name, type_attribute in trigger_geometry._KINDS:
            definitions = vocabulary.conditions if kind == "condition" else vocabulary.effects
            for entry in getattr(trigger, list_name):
                for fields in trigger_geometry.coordinate_fields(definitions.get(getattr(entry, type_attribute))):
                    set_count = sum(trigger_geometry.coordinate(entry, f) >= 0 for f in fields)
                    partial += 0 < set_count < len(fields)
    return areas, partial


@pytest.mark.corpus
def test_corpus_counts_match_the_plan(corpus_files) -> None:
    """Every corpus file against its pinned row; an unpinned file fails with its measured row."""
    measured = {path.name: _measure_geometry(path) for path in corpus_files}
    unpinned = {name: row for name, row in measured.items() if name not in _GEOMETRY_PER_FILE}
    assert not unpinned, f"pin these new files' rows in _GEOMETRY_PER_FILE: {unpinned}"
    assert measured == {name: _GEOMETRY_PER_FILE[name] for name in measured}


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


# -- GH #229: Edit Area's grab zones and drags -----------------------------------

DRAG_AREA = (10, 20, 19, 25)  # 10 x 6 tiles


@pytest.mark.parametrize(
    ("point", "zone"),
    [
        ((15.0, 22.5), "move"),
        ((10.2, 22.5), "w"),
        ((19.8, 22.5), "e"),
        ((15.0, 20.1), "n"),
        ((15.0, 25.9), "s"),
        ((10.1, 20.1), "nw"),
        ((19.9, 20.1), "ne"),
        ((10.1, 25.9), "sw"),
        ((19.9, 25.9), "se"),
        ((9.9, 22.5), None),
        ((20.0, 22.5), None),
        ((15.0, 19.99), None),
        ((15.0, 26.0), None),
    ],
)
def test_area_hit_zone_names_the_part_under_the_point(point, zone) -> None:
    assert trigger_geometry.area_hit_zone(DRAG_AREA, point, 0.5) == zone


def test_a_wide_band_is_capped_so_a_one_tile_area_keeps_a_middle() -> None:
    assert trigger_geometry.area_hit_zone((5, 5, 5, 5), (5.5, 5.5), 4.0) == "move"
    assert trigger_geometry.area_hit_zone((5, 5, 5, 5), (5.1, 5.5), 4.0) == "w"
    # A third of the 6-tile y span: 1.9 tiles in is the middle, 2.1 tiles from the far side too.
    assert trigger_geometry.area_hit_zone(DRAG_AREA, (15.0, 21.9), 5.0) == "n"
    assert trigger_geometry.area_hit_zone(DRAG_AREA, (15.0, 22.1), 5.0) == "move"


@pytest.mark.parametrize(
    ("zone", "end", "expected"),
    [
        ("move", (18, 24), (13, 22, 22, 27)),
        ("w", (12, 99), (7, 20, 19, 25)),
        ("e", (13, 0), (10, 20, 17, 25)),
        ("n", (0, 18), (10, 16, 19, 25)),
        ("s", (99, 26), (10, 20, 19, 29)),
        ("nw", (14, 21), (9, 19, 19, 25)),
        ("se", (17, 23), (10, 20, 21, 26)),
        ("ne", (16, 19), (10, 17, 20, 25)),
        ("sw", (13, 24), (8, 20, 19, 27)),
        # Past the opposite side: stored min/max, like Set Area.
        ("w", (40, 22), (19, 20, 35, 25)),
        ("n", (15, 32), (10, 25, 19, 30)),
        ("move", (15, 22), DRAG_AREA),
    ],
)
def test_dragged_area_moves_or_resizes_by_the_drag_delta(zone, end, expected) -> None:
    """The grabbed sides shift by the drag, not to the cursor, so a press inside a wide band never jumps."""
    assert trigger_geometry.dragged_area(DRAG_AREA, zone, (15, 22), end, 120, 120) == expected


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ((-50, 22), (0, 20, 9, 25)),
        ((500, 22), (110, 20, 119, 25)),
        ((15, -50), (10, 0, 19, 5)),
        ((15, 500), (10, 114, 19, 119)),
    ],
)
def test_a_move_keeps_the_size_and_stops_at_each_map_edge(end, expected) -> None:
    assert trigger_geometry.dragged_area(DRAG_AREA, "move", (15, 22), end, 120, 120) == expected


@pytest.mark.parametrize(
    ("zone", "end", "expected"),
    [
        ("w", (-50, 22), (0, 20, 19, 25)),
        ("e", (500, 22), (10, 20, 119, 25)),
        ("n", (15, -50), (10, 0, 19, 25)),
        ("s", (15, 500), (10, 20, 19, 119)),
        ("se", (500, 500), (10, 20, 119, 119)),
    ],
)
def test_a_resize_clamps_the_grabbed_side_to_the_map(zone, end, expected) -> None:
    assert trigger_geometry.dragged_area(DRAG_AREA, zone, (15, 22), end, 120, 120) == expected


def test_a_non_square_map_clamps_each_axis_to_its_own_size() -> None:
    assert trigger_geometry.dragged_area(DRAG_AREA, "move", (15, 22), (500, 500), 60, 40) == (50, 34, 59, 39)


@pytest.mark.parametrize(
    ("coords", "point", "zone"),
    [
        ((5, 5, 5, 5), (5.5, 5.5), "move"),  # 1x1: thin band, as with tile borders off
        ((5, 5, 6, 6), (6.0, 6.0), "move"),  # 2x2 too
        ((5, 5, 6, 6), (5.1, 6.0), "w"),
        ((5, 5, 7, 7), (5.9, 6.5), "w"),  # 3x3: a one-tile ring and a middle tile
        ((5, 5, 7, 7), (6.5, 6.5), "move"),
        ((5, 5, 7, 7), (7.1, 7.9), "se"),
        ((5, 5, 6, 9), (6.0, 5.9), "n"),  # 2x5: tile ends, a thin long side
        ((5, 5, 6, 9), (5.5, 7.5), "move"),
    ],
)
def test_tile_borders_widen_only_axes_of_three_or_more_tiles(coords, point, zone) -> None:
    assert trigger_geometry.area_hit_zone(coords, point, 0.2, tile_borders=True) == zone


def test_a_wider_pixel_band_with_tile_borders_still_keeps_a_middle() -> None:
    six = (0, 0, 5, 5)
    assert trigger_geometry.area_hit_zone(six, (1.9, 3.0), 3.0, tile_borders=True) == "w", "capped at a third: 2 tiles"
    assert trigger_geometry.area_hit_zone(six, (2.1, 3.0), 3.0, tile_borders=True) == "move"
    assert trigger_geometry.area_hit_zone(six, (0.9, 3.0), 0.5) == "move", "off: today's thin band"
