"""Guards the AoE2ScenarioParser==0.8.3 pin (see requirements.txt's own
comment on why it's pinned, not just listed): descape/scenario_io.py,
descape/elevation_tools.py, descape/trigger_fields.py (the quantity cluster)
and others each reach into several of that library's *private* names. Nothing else asserts these still exist -- an unpinned or
bumped install could silently drop or rename one of them, and the first
sign would be a runtime AttributeError/ImportError deep inside a real
scenario load, not a clear failure here.

A few of the names below (sections, retriever_map, byte_length, Retriever.data)
aren't underscore-prefixed, but are undocumented parse-time internals in the
same sense -- not exposed through MapManager/UnitManager's own public
accessors, and nothing in the library's docs commits to them. Pinned here for
the same reason as the underscore-prefixed ones: scenario_io.py's units-strip
support (units_block_offset etc., used by tools/strip_units.py) reaches into
them directly to derive byte offsets that AoE2FileSection.byte_length tracks
correctly but AoE2ScenarioParser's own get_data_as_bytes() does not (see
scenario_io.py's load_map_and_units() and tools/strip_units.py's docstrings).
"""

from __future__ import annotations

from pathlib import Path

from AoE2ScenarioParser.objects.aoe2_object import AoE2Object
from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.effect import (
    Effect,
    _get_armour_attack_source,
    _is_float_quantity_effect,
)
from AoE2ScenarioParser.objects.data_objects.trigger import Trigger
from AoE2ScenarioParser.objects.data_objects.unit import Unit
from AoE2ScenarioParser.objects.data_objects.variable import Variable
from AoE2ScenarioParser.objects.managers.map_manager import MapManager
from AoE2ScenarioParser.objects.managers.trigger_manager import (
    TriggerManager,
    get_trigger_referencing_ce,
)
from AoE2ScenarioParser.scenarios.aoe2_de_scenario import AoE2DEScenario
from AoE2ScenarioParser.scenarios.aoe2_scenario import (
    AoE2Scenario,
    _decompress_bytes,
    _get_file_version,
    _get_scenario_variant,
    _initialise_version_dependencies,
)
from AoE2ScenarioParser.sections.aoe2_file_section import AoE2FileSection
from AoE2ScenarioParser.sections.retrievers.retriever import Retriever
from AoE2ScenarioParser.sections.retrievers.retriever_object_link import RetrieverObjectLink
from AoE2ScenarioParser.sections.retrievers.retriever_object_link_group import (
    RetrieverObjectLinkGroup,
)

from descape.scenario_io import load_map_and_units, parse_triggers


def test_scenario_io_private_functions_importable() -> None:
    """descape/scenario_io.py imports these four module-level private
    functions directly (see its own import block)."""
    for fn in (_decompress_bytes, _get_file_version, _get_scenario_variant, _initialise_version_dependencies):
        assert callable(fn)


def test_aoe2de_scenario_has_load_structure_and_load_header_section() -> None:
    """descape/scenario_io.py calls scenario._load_structure() and
    scenario._load_header_section(igen) directly on an AoE2DEScenario
    instance."""
    assert callable(getattr(AoE2DEScenario, "_load_structure", None))
    assert callable(getattr(AoE2DEScenario, "_load_header_section", None))


def test_aoe2scenario_base_has_create_and_load_section() -> None:
    """descape/scenario_io.py calls scenario._create_and_load_section(name,
    igen) -- defined on the base AoE2Scenario class the DE subclass
    inherits from, not overridden on AoE2DEScenario itself. Called from two
    places now: the Map/Units walk in _load_map_and_units(), and the lazy
    Triggers walk in parse_triggers()."""
    assert callable(getattr(AoE2Scenario, "_create_and_load_section", None))


def test_trigger_manager_has_construct() -> None:
    """descape/scenario_io.py's parse_triggers() calls
    TriggerManager.construct(uuid) -- a classmethod defined on the base
    AoE2Object, not overridden on TriggerManager itself."""
    assert callable(getattr(TriggerManager, "construct", None))


def test_trigger_classes_keep_their_own_link_lists() -> None:
    """descape/library_compat.py's depoison() walks each poisoned class's
    _link_list to clear `disabled`. Unit is included alongside the four
    trigger-side classes since it joined POISONED_CLASSES once Units editing
    needed the same version-gating fix.

    Asserted as non-empty and correctly-typed rather than merely present:
    AoE2Object declares `_link_list = []` on the base, so a plain
    getattr(cls, "_link_list") check would still pass if a subclass lost its
    own override -- guarding nothing while looking like it guards something.
    """
    for cls in (Trigger, Condition, Effect, Variable, Unit):
        link_list = cls._link_list
        assert isinstance(link_list, list) and link_list, f"{cls.__name__} has no own _link_list"
        assert any(isinstance(entry, RetrieverObjectLinkGroup) for entry in link_list), (
            f"{cls.__name__}._link_list holds no RetrieverObjectLinkGroup; "
            "library_compat._iter_links() assumes the nesting exists"
        )


def test_retriever_object_links_expose_group_and_disabled() -> None:
    """depoison() reads RetrieverObjectLinkGroup.group to reach the leaf
    RetrieverObjectLink objects, and writes .disabled on each. Both are plain
    instance attributes set in __init__, but unlike the Units-section case
    below they need no scenario load: the link objects are built at class
    definition time."""
    flat = []
    for entry in Effect._link_list:
        assert isinstance(entry, RetrieverObjectLinkGroup)
        assert isinstance(entry.group, list) and entry.group
        flat.extend(entry.group)

    assert flat
    for link in flat:
        assert isinstance(link, RetrieverObjectLink)
        assert isinstance(link.disabled, bool)
        # `support` is None for an always-supported field, or a Support range.
        assert hasattr(link, "support")
    assert callable(getattr(RetrieverObjectLink, "overwrite_unsupported_properties", None)), (
        "RetrieverObjectLink.overwrite_unsupported_properties is the method the library's own "
        "load path uses to poison a class -- descape/unit_model.py's tests mimic its exact "
        "getter/setter shape rather than calling it directly, but the method itself is what "
        "the poisoning bug this pins against depends on existing."
    )


def test_trigger_manager_exposes_the_structural_api_the_edit_model_wraps() -> None:
    """descape/trigger_model.py's structural_edit() is a wrapper around these:
    each one renumbers trigger ids and remaps (de)activate-trigger references,
    which is what the blob reconciliation exists to track. get_trigger_
    referencing_ce is a module-level function, not a method, and is what the
    model compares before and after to find the triggers a remap dirtied."""
    for name in (
        "add_trigger",
        "remove_trigger",
        "remove_triggers",
        "reorder_triggers",
        "move_triggers",
        "copy_trigger",
        "import_triggers",
        "add_variable",
        "commit",
    ):
        assert callable(getattr(TriggerManager, name, None)), f"TriggerManager.{name} is gone"
    assert callable(get_trigger_referencing_ce)


def test_trigger_exposes_the_generic_condition_and_effect_constructors() -> None:
    """4b.6b's picker adds a condition or effect by *type id*, through these.

    The public route is trigger.new_condition.<name>() / new_effect.<name>(),
    one hand-written wrapper per type -- but the wrappers do not cover the
    vocabulary. Measured across every shipped versions/DE/v*/ JSON, 71 entries
    name no wrapper at all, from five distinct names: "enable/disable_object"
    and "enable/disable_technology" are not legal Python identifiers, "and" and
    "or" are keywords, and "counts_units_into_variable" is simply absent. A
    picker driven by the vocabulary has to reach the generic entry point the
    wrappers themselves call.
    """
    for name in ("_add_condition", "_add_effect"):
        assert callable(getattr(Trigger, name, None)), f"Trigger.{name} is gone"

    # And the public per-type wrappers really are the incomplete route, rather
    # than this being a preference. Pinned so a library version that closes the
    # gap shows up here as a failure worth acting on.
    from AoE2ScenarioParser.objects.support.new_condition import NewConditionSupport
    from AoE2ScenarioParser.objects.support.new_effect import NewEffectSupport

    from descape import library_compat

    uncovered = [
        (version.name, kind, entry.name)
        for version in sorted(library_compat.VERSIONS_DIR.iterdir())
        if (version / "conditions.json").is_file()
        for kind, entries, support in (
            ("condition", library_compat.load_vocabulary(version.name[1:]).conditions, NewConditionSupport),
            ("effect", library_compat.load_vocabulary(version.name[1:]).effects, NewEffectSupport),
        )
        for entry in entries.values()
        if not hasattr(support, entry.name)
    ]
    assert uncovered, "the public wrappers now cover the vocabulary -- prefer them"
    assert {name for _v, _k, name in uncovered} == {
        "and",
        "counts_units_into_variable",
        "enable/disable_object",
        "enable/disable_technology",
        "or",
    }
    assert len(uncovered) == 71


def test_every_vocabulary_has_type_0_unit_reference_defaults() -> None:
    """trigger_clipboard.reference_fields() takes a Unit/Unit[] field's
    default from the type-0 entry, overlaid with the type's own (v1.58's Task
    Object lists no selected_object_ids default). A version whose type 0 lacked one
    would make a cross-document paste clear nothing and report 0 (GH #3)."""
    from descape import library_compat

    versions = [
        version.name[1:]
        for version in sorted(library_compat.VERSIONS_DIR.iterdir())
        if (version / "conditions.json").is_file()
    ]
    assert {"1.36", "1.44", "1.54", "1.58"} <= set(versions), versions
    for version in versions:
        vocabulary = library_compat.load_vocabulary(version)
        condition_0 = vocabulary.conditions.get(0)
        effect_0 = vocabulary.effects.get(0)
        assert condition_0 is not None and effect_0 is not None, version
        assert condition_0.default_attributes.get("unit_object") == -1, version
        assert effect_0.default_attributes.get("selected_object_ids") == [], version
        assert effect_0.default_attributes.get("location_object_reference") == -1, version
        for base, presentation in (
            (condition_0, vocabulary.condition_presentation),
            (effect_0, vocabulary.effect_presentation),
        ):
            for field, shown_as in presentation.items():
                if shown_as in ("Unit", "Unit[]"):
                    assert base.default_attributes.get(field) is not None, (version, field)


def test_aoe2object_has_get_object_attrs() -> None:
    """tests/test_trigger_write_path.py's field-for-field round-trip snapshot
    is driven off this rather than a hand-listed set of attribute names, so a
    library version that adds a trigger field is compared too."""
    assert callable(getattr(AoE2Object, "_get_object_attrs", None))


def test_effect_exposes_the_raw_quantity_triple() -> None:
    """Effect.quantity is bit-split across these three for armour/attack
    effects. Plan finding 8's drift is an int -1 sentinel re-typed to f32 -1.0
    in one of them, which the public property hides -- so the round-trip test
    compares the raw slots, and the field editor must never write `quantity`
    directly for those effects."""
    # _quantity_int/_quantity_float are class-level properties; `variable` is a
    # plain instance attribute set in __init__, so it needs a real effect.
    for name in ("_quantity_int", "_quantity_float"):
        assert isinstance(getattr(Effect, name, None), property), f"Effect.{name} is not a property"

    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    loaded = load_map_and_units(fixture)
    manager = parse_triggers(loaded)
    assert manager is not None
    effects = [effect for trigger in manager.triggers for effect in trigger.effects]
    assert effects
    for effect in effects:
        for name in ("_quantity_int", "_quantity_float", "variable"):
            assert hasattr(effect, name), f"Effect instance has no {name}"


def test_the_quantity_cluster_gates_and_raw_fields_exist() -> None:
    """trigger_fields.live_quantity_slot() asks these two library gates which
    slot serializes, and cluster_fixup()/cluster_incoherence() read the three
    raw fields, because on a broken pair the public quantity getter raises."""
    assert _get_armour_attack_source(51, 8) == "quantity"
    assert _get_armour_attack_source(79, 8) == "variable"
    assert _get_armour_attack_source(51, 0) is None
    assert callable(_is_float_quantity_effect)

    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    manager = parse_triggers(load_map_and_units(fixture))
    assert manager is not None
    aa, plain = manager.triggers[1].effects
    assert (aa._armour_attack_quantity, aa._armour_attack_class) == (2, 3)
    assert plain._quantity == 45
    assert _is_float_quantity_effect(51, 13) and not _is_float_quantity_effect(51, 0)


def test_triggers_keep_uuid_stamped_inner_lists_the_clipboard_restamps() -> None:
    """trigger_clipboard.paste_into() restamps each pasted trigger's inner
    effect/condition UuidList `_uuid` to the destination manager's, since the
    manager restamps only the entries. Pins those names."""
    from AoE2ScenarioParser.objects.support.uuid_list import UuidList

    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    manager = parse_triggers(load_map_and_units(fixture))
    assert manager is not None and hasattr(manager, "_uuid")
    trigger = manager.triggers[0]
    for name in ("_effects", "_conditions"):
        inner = getattr(trigger, name)
        assert isinstance(inner, UuidList), f"Trigger.{name} is not a UuidList"
        assert inner._uuid == manager._uuid


def test_trigger_section_retrievers_expose_serialization_and_datatypes() -> None:
    """descape/trigger_model.py maps the Triggers section's byte regions from
    its retriever map: per-retriever get_data_as_bytes() for the fixed-width
    fields, datatype.type == "struct" to spot the two struct lists, and each
    struct's own parse-time byte_length for the ones whose re-serialization
    can change length. tools/gen_trigger_fixture.py additionally reads
    datatype.type_and_length to find str fields."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    loaded = load_map_and_units(fixture)
    assert parse_triggers(loaded) is not None

    section = loaded._scenario.sections["Triggers"]
    assert isinstance(section, AoE2FileSection)
    assert isinstance(section.get_data_as_bytes(), bytes)

    for name in (
        "trigger_version",
        "trigger_instruction_start",
        "number_of_triggers",
        "trigger_data",
        "trigger_display_order_array",
        "unknown_bytes",
        "number_of_variables",
        "variable_data",
    ):
        retriever = section.retriever_map[name]
        assert isinstance(retriever, Retriever), f"Triggers.{name} is not a Retriever"
        assert isinstance(retriever.datatype.type, str)
        assert isinstance(retriever.datatype.type_and_length, tuple)
        assert isinstance(retriever.get_data_as_bytes(), bytes)

    entries = section.retriever_map["trigger_data"].data
    assert isinstance(entries, list) and entries
    for entry in entries:
        assert isinstance(entry, AoE2FileSection)
        assert isinstance(entry.byte_length, int)
        assert isinstance(entry.get_data_as_bytes(), bytes)


def test_map_manager_has_elevation_tile_recursion() -> None:
    """descape/elevation_tools.py calls MapManager._elevation_tile_recursion()
    directly to reproduce the library's own neighbor-propagation rule."""
    assert callable(getattr(MapManager, "_elevation_tile_recursion", None))


def test_scenario_units_section_exposes_sections_retriever_map_and_byte_length() -> None:
    """descape/scenario_io.py reads scenario.sections["Units"].retriever_map
    ["players_units"].data, a list of AoE2FileSection whose .byte_length and
    nested retriever_map["unit_count"].data drive units_block_offset -- exact
    parse-time accounting, unlike get_data_as_bytes()'s re-serialization (see
    tools/strip_units.py's docstring). Exercised against a real load rather
    than just checked for existence on the class, since these are plain
    instance attributes set in __init__, not class-level members."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"
    s = load_map_and_units(fixture)
    assert isinstance(s._scenario, AoE2Scenario)

    units_section = s._scenario.sections["Units"]
    assert isinstance(units_section, AoE2FileSection)
    assert isinstance(units_section.byte_length, int)

    players_units = units_section.retriever_map["players_units"].data
    assert isinstance(players_units, list) and players_units
    for section in players_units:
        assert isinstance(section, AoE2FileSection)
        assert isinstance(section.byte_length, int)
        unit_count_retriever = section.retriever_map["unit_count"]
        assert isinstance(unit_count_retriever, Retriever)
        assert isinstance(unit_count_retriever.data, int)


def test_xs_attachment_retrievers_are_reachable() -> None:
    """scenario_io.xs_attachment() reads Map.script_name at load and
    Files.script_file_content once parse_triggers() has walked past Triggers
    into Files -- both through retriever_map, neither via a public manager."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "triggers_120x120.aoe2scenario"
    s = load_map_and_units(fixture)
    script_name = s._scenario.sections["Map"].retriever_map["script_name"]
    assert isinstance(script_name, Retriever)
    assert isinstance(script_name.data, str)

    assert "Files" not in s._scenario.sections
    assert parse_triggers(s) is not None
    content = s._scenario.sections["Files"].retriever_map["script_file_content"]
    assert isinstance(content, Retriever)
    assert isinstance(content.data, str)


def test_map_links_library_compat_rebuilds_are_where_it_expects() -> None:
    """library_compat.adapt_map_links() rebuilds the Map group of
    MapManager._link_list minus `_map_color_mood`, and TerrainTile's group
    minus `layer`, for a structure lacking those fields (v1.21). It relies on
    both links existing under those names, on MapManager.__init__ taking
    `_map_color_mood` as a required argument (so it must be fed a value, not
    dropped), on TerrainTile.__init__ defaulting `layer` to -1 (so it can be
    dropped), and on a group calling pull_from_link/push_to_link on its
    children (the two methods _AbsentFieldLink overrides)."""
    import inspect

    from AoE2ScenarioParser.objects.data_objects.terrain_tile import TerrainTile

    for cls, name in ((MapManager, "_map_color_mood"), (TerrainTile, "layer")):
        assert isinstance(cls._link_list, list) and cls._link_list, f"{cls.__name__} has no own _link_list"
        groups = [e for e in cls._link_list if isinstance(e, RetrieverObjectLinkGroup)]
        assert any(name in [link.name for link in g.group] for g in groups), (
            f"{cls.__name__}._link_list has no group holding {name!r}"
        )

    mm_params = inspect.signature(MapManager.__init__).parameters
    assert "_map_color_mood" in mm_params
    assert mm_params["_map_color_mood"].default is inspect.Parameter.empty
    assert inspect.signature(TerrainTile.__init__).parameters["layer"].default == -1

    group_source = inspect.getsource(RetrieverObjectLinkGroup)
    assert "link.pull_from_link(" in group_source
    assert "link.push_to_link(" in group_source
