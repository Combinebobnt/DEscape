"""trigger_version_compat (GH #317): which blocks paste across scenario versions.

Rule-level tests over synthetic blocks built in the library's own 1.54-1.59
default scenarios, so no `examples/` is needed. Round trips through the real
write path are in tests/test_trigger_clipboard.py; the corpus tier at the end
runs real files' triggers through it.
"""

from __future__ import annotations

import json
import warnings

import pytest
from AoE2ScenarioParser.objects.data_objects.condition import Condition
from AoE2ScenarioParser.objects.data_objects.effect import Effect
from AoE2ScenarioParser.objects.data_objects.trigger import Trigger

from descape import library_compat, trigger_clipboard
from descape import trigger_version_compat as compat
from descape.scenario_io import load_map_and_units, parse_triggers
from descape.scenario_write import write_scenario
from descape.trigger_model import TriggerEditModel

import conftest

# The library's scenario store is weak; new_effect looks the scenario up by uuid.
_KEEP_ALIVE: list = []

_STRUCTS = {Trigger: "TriggerStruct", Effect: "EffectStruct", Condition: "ConditionStruct"}


def _default(version: str):
    loaded = load_map_and_units(library_compat.VERSIONS_DIR / f"v{version}" / "default.aoe2scenario")
    _KEEP_ALIVE.append(loaded)
    return loaded, parse_triggers(loaded)


def _effects(version: str, *names: str, **values):
    """A 1-trigger manager at `version` whose trigger holds one new effect per
    name, each with `values` written straight into its instance dict."""
    _loaded, manager = _default(version)
    trigger = manager.add_trigger("Source")
    for name in names:
        vars(getattr(trigger.new_effect, name)()).update(values)
    return manager


def _entries(version: str, *names: str, **values) -> trigger_clipboard.EntryBlock:
    manager = _effects(version, *names, **values)
    refs = [("effect", i) for i in range(len(names))]
    return trigger_clipboard.copy_entries(manager, 0, refs, doc_id="A", scenario_version=version)


def _struct_names(version: str, cls: type) -> set[str]:
    structure = json.loads((library_compat.VERSIONS_DIR / f"v{version}" / "structure.json").read_text(encoding="utf-8"))
    trigger = structure["Triggers"]["structs"]["TriggerStruct"]
    struct = trigger if cls is Trigger else trigger["structs"][_STRUCTS[cls]]
    return set(struct["retrievers"])


def _structure_versions() -> list[str]:
    return [v for v in library_compat.vocabulary_versions() if (library_compat.VERSIONS_DIR / f"v{v}" / "structure.json").is_file()]


def test_supported_agrees_with_each_versions_structure() -> None:
    """A gated link is supported exactly where the version's struct has its
    retriever, compared by `link` (the struct name), never by `name`."""
    wrong = []
    for version in _structure_versions():
        for cls in compat.CLASSES:
            names = _struct_names(version, cls)
            stored = compat.supported(cls, version)
            wrong.extend((version, cls.__name__, f.name) for f in compat.FIELDS[cls] if (f.name in stored) != (f.name in names))
    assert wrong == []


def test_every_support_name_exists_on_its_class_and_the_stored_keys_are_pinned() -> None:
    """The `__dict__` key each gated link lives under, read off a real parsed
    object: computed properties have none and are never dropped or filled."""
    special = {
        (Effect, "player_color"): "_player_color",
        (Effect, "quantity_float"): None,
        (Effect, "_eff_filler_0"): None,
        (Effect, "_eff_filler_1"): None,
        (Condition, "_cond_filler_0"): None,
    }
    manager = _effects("1.59", "display_instructions")
    trigger = manager.triggers[0]
    trigger.new_condition.timer(timer=5)
    objects = {Trigger: trigger, Effect: trigger.effects[0], Condition: trigger.conditions[0]}
    for cls, fields in compat.FIELDS.items():
        for f in fields:
            assert f.link_name in vars(objects[cls]) or hasattr(cls, f.link_name), (cls.__name__, f.link_name)
            assert f.stored == special.get((cls, f.name), f.name), (cls.__name__, f.name)
            if f.stored is not None:
                assert f.stored in vars(objects[cls]), (cls.__name__, f.stored)


@pytest.mark.parametrize(("value", "refused"), [(-1, False), (0, False), (1, True)])
def test_mutual_diplomacy_minus_one_or_default_is_unset(value, refused) -> None:
    """Real files store -1 where the per-type default is 0: both are unset."""
    block = _entries("1.58", "change_diplomacy", mutual_diplomacy=value)
    reasons = compat.plan_paste(block, "1.56").reasons
    assert bool(reasons) == refused
    if refused:
        assert reasons == ("1 effect set Mutual Diplomacy, which 1.56 does not store",)


@pytest.mark.parametrize(("quantity", "refused"), [(2.5, True), (3.0, False)])
def test_a_fractional_float_quantity_is_refused_below_1_55(quantity, refused) -> None:
    block = _entries("1.55", "modify_attribute", object_attributes=5, _quantity=quantity)
    assert bool(compat.plan_paste(block, "1.54").reasons) == refused


@pytest.mark.parametrize(
    ("aa_class", "quantity", "trigger_version", "refused"),
    [(4, 300, 2.4, True), (4, 30, 2.4, False), (4, 300, 4.1, False), (-1, 65534, 2.4, True), (-1, 65534, 4.9, False)],
)
def test_armour_values_must_fit_the_destinations_packing(aa_class, quantity, trigger_version, refused) -> None:
    """8+8 bits below trigger version 2.5; at 16+16 any value pastes, a
    negative packed one (read as class -1) included. Rule-only: no 2.4
    document is in the quick tier, so the trigger version is passed in."""
    block = _entries("1.55", "change_object_armor", _armour_attack_quantity=quantity, _armour_attack_class=aa_class)
    plan = compat.plan_paste(block, "1.54", trigger_version=trigger_version)
    assert bool(plan.reasons) == refused
    if refused:
        assert "armour/attack value too large" in plan.refusal


def test_an_attribute_left_at_its_default_is_dropped_silently() -> None:
    inert = _entries("1.58", "display_instructions")
    assert compat.plan_paste(inert, "1.57").reasons == ()
    lost = _entries("1.58", "display_instructions", use_tag_color_for_icon=1)
    assert compat.plan_paste(lost, "1.57").reasons == ("1 effect set Use Tag Color For Icon, which 1.57 does not store",)


def test_a_type_the_destination_lacks_or_nobody_defines_is_refused() -> None:
    missing = _entries("1.59", "mirror_diplomacy")
    assert compat.plan_paste(missing, "1.57").reasons == ("Mirror Diplomacy does not exist in 1.57",)
    unknown = _entries("1.56", "display_instructions", _effect_type=91)
    assert compat.plan_paste(unknown, "1.59").reasons == ("1 effect of type 91, which 1.56 does not define",)


@pytest.mark.parametrize(("value", "refused"), [(1, True), (0, False), (-1, False)])
def test_a_set_execute_on_load_is_refused_into_1_54(value, refused) -> None:
    """-1 is unset here only by the rule's own -1: the constructor default is
    0. (Every entry field a lower version drops has a -1, '' or [] default.)"""
    manager = _effects("1.55")
    manager.triggers[0].execute_on_load = value
    block = trigger_clipboard.copy_block(manager, [0], doc_id="A", scenario_version="1.55")
    expected = ("1 trigger set Execute On Load, which 1.54 does not store",) if refused else ()
    assert compat.plan_paste(block, "1.54").reasons == expected


def test_the_1_54_cost_remap_is_refused_when_a_cost_is_set() -> None:
    """1.54 moved Change Object Cost from food/wood/stone/gold to resource_N:
    a set resource_N has nowhere to go in 1.53 (rule only, no 1.53 file)."""
    assert compat.plan_paste(_entries("1.54", "change_object_cost"), "1.53").reasons == ()
    block = _entries("1.54", "change_object_cost", resource_1_quantity=50)
    assert compat.plan_paste(block, "1.53").reasons == ("1 effect set Resource 1 Quantity, which 1.53 does not store",)


def test_the_refusal_names_the_versions_and_caps_its_reasons() -> None:
    block = _entries("1.59", "display_instructions", "change_object_armor", "task_object", "mirror_diplomacy", "modify_object_attribute")
    lost = [("use_tag_color_for_icon", 1), ("max_units_affected", 3), ("issue_group_command", 1)]
    for effect, (name, value) in zip(block.effects, lost, strict=False):
        vars(effect)[name] = value
    plan = compat.plan_paste(block, "1.54")
    assert len(plan.reasons) == 5
    assert plan.refusal.startswith("Copied from a 1.59 scenario; this one is 1.54: ")
    assert plan.refusal.endswith("; and 2 more.")
    assert plan.refusal.count(";") == 4


def test_equal_versions_plan_nothing_and_the_memo_keys_on_the_block_and_destination() -> None:
    block = _entries("1.58", "display_instructions")
    assert compat.plan_paste(block, "1.58") == compat.PastePlan("1.58", "1.58")
    first = compat.plan_paste(block, "1.57")
    assert compat.plan_paste(block, "1.57") is first
    assert compat.plan_paste(block, "1.56") is not first
    again = _entries("1.58", "display_instructions")
    assert compat.plan_paste(again, "1.57") is not first
    triggers = trigger_clipboard.copy_block(_effects("1.58", "display_instructions"), [0], doc_id="A", scenario_version="1.58")
    held = compat.plan_paste(triggers, "1.57")
    compat.plan_paste(again, "1.57")
    assert compat.plan_paste(triggers, "1.57") is held, "the entry clipboard does not evict the trigger one"


def test_stripped_restores_every_popped_attribute_even_on_failure() -> None:
    block = _entries("1.59", "display_instructions", use_tag_color_for_icon=1)
    plan = compat.plan_paste(block, "1.57")
    before = dict(vars(block.effects[0]))
    with pytest.raises(RuntimeError), compat.stripped(plan, compat.block_objects(block)):
        assert "use_tag_color_for_icon" not in vars(block.effects[0])
        raise RuntimeError
    assert vars(block.effects[0]) == before


def test_apply_overwrites_what_the_destination_adds_with_its_defaults() -> None:
    """Rule 3: a field the source lacks can hold None, so the fill
    overwrites whatever is there; a field the destination cannot store
    becomes None, as a native object of that version holds it."""
    block = _entries("1.54", "change_object_armor", max_units_affected=None, object_filter=7)
    plan = compat.plan_paste(block, "1.59")
    effect = block.effects[0]
    compat.apply(plan, (), (), [effect])
    assert (vars(effect)["max_units_affected"], vars(effect)["object_filter"]) == (-1, -1)
    down = _entries("1.59", "display_instructions")
    plan = compat.plan_paste(down, "1.57")
    compat.apply(plan, (), (), down.effects)
    assert vars(down.effects[0])["use_tag_color_for_icon"] is None


def test_paste_refusal_hints_at_the_conversion_only_when_it_helps() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _loaded, manager = _default("1.59")
        trigger = manager.add_trigger("Source")
        trigger.new_condition.object_visible_multiplayer(source_player=1)
    vars(trigger.conditions[0])["allow_in_fog"] = 1  # the one condition type that lists it
    block = trigger_clipboard.copy_block(manager, [0], doc_id="A", scenario_version="1.59")
    refusal = compat.paste_refusal(block, "1.58", 4.9, convertible=True)
    assert "1 condition set Allow In Fog, which 1.58 does not store" in refusal
    assert refusal.endswith("File > Convert to Version 1.59 upgrades this one.")
    assert "File >" not in compat.paste_refusal(block, "1.58", 4.9, convertible=False)
    assert compat.paste_refusal(_entries("1.58", "display_instructions"), "1.59", 5.0, convertible=True) == ""


# -- corpus: real files through the real write path ------------------------------

_EXAMPLES = conftest.ROOT / "examples"


def _example(name: str):
    path = _EXAMPLES / f"{name}.aoe2scenario"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = load_map_and_units(path)
    _KEEP_ALIVE.append(loaded)
    return loaded


def _paste_save_reload(source, destination, tmp_path, skip_effect_types=()):
    """Copy every trigger of `source` (bar those holding a skipped effect
    type) into `destination`, save, reload: (block, result, reloaded pasted triggers, reloaded manager)."""
    manager = parse_triggers(source)
    indices = [i for i, t in enumerate(manager.triggers) if not any(e.effect_type in skip_effect_types for e in t.effects)]
    block = trigger_clipboard.copy_block(manager, indices, doc_id="A", scenario_version=source.scenario_version)
    model = TriggerEditModel(destination)
    before = len(model.manager().triggers)
    results = []
    model.structural_edit(
        lambda m: results.append(trigger_clipboard.paste_into(m, block, doc_id="B", scenario_version=destination.scenario_version))
    )
    out = tmp_path / "pasted.aoe2scenario"
    write_scenario(destination, out, triggers=model)
    reloaded = load_map_and_units(out)
    _KEEP_ALIVE.append(reloaded)
    reloaded_manager = parse_triggers(reloaded)
    assert (reloaded.scenario_version, reloaded.trigger_write_supported) == (destination.scenario_version, True)
    pasted = reloaded_manager.triggers[before:]
    assert [(len(t.conditions), len(t.effects)) for t in pasted] == [(len(t.conditions), len(t.effects)) for t in block.triggers]
    assert [[e.effect_type for e in t.effects] for t in pasted] == [[e.effect_type for e in t.effects] for t in block.triggers]
    return block, results[0], pasted, reloaded_manager


def _units_referenced(entries) -> int:
    return sum(1 for e in entries if getattr(e, "unit_object", -1) not in (-1, None) or getattr(e, "selected_object_ids", []))


@pytest.mark.corpus
def test_a_1_41_block_pastes_into_1_59_clearing_units_and_carrying_variable_names(tmp_path) -> None:
    """1.41's vocabulary gives unit_object no Unit presentation; the 1.59 one
    does, so the paste still clears it. ElCid 5 names two variables."""
    source, destination = _example("C2_ElCid_coop_5_v0_14"), _example("8tp3w9j")
    block, result, pasted, manager = _paste_save_reload(source, destination, tmp_path)
    assert _units_referenced(e for t in block.triggers for e in (*t.conditions, *t.effects)) > 0
    assert result.cleared_unit_refs > 0
    assert _units_referenced(e for t in pasted for e in (*t.conditions, *t.effects)) == 0
    assert block.variable_names
    assert result.named_variables + result.variable_name_conflicts == len(block.variable_names)
    names = {v.variable_id: v.name for v in manager.variables}
    assert sum(names.get(vid) == name for vid, name in block.variable_names) == result.named_variables


@pytest.mark.corpus
def test_a_1_59_conditions_unit_is_cleared_in_a_1_41_destination(tmp_path) -> None:
    """1.41's vocabulary gives a condition's unit_object no Unit presentation;
    the newest vocabulary with the type still says it is one."""
    _loaded, manager = _default("1.59")
    trigger = manager.add_trigger("Source")
    trigger.new_condition.destroy_object(unit_object=1234)
    trigger.new_effect.kill_object(source_player=1, selected_object_ids=[1234])
    block = trigger_clipboard.copy_block(manager, [0], doc_id="A", scenario_version="1.59")
    destination = _example("C2_ElCid_coop_1_v0_16")
    model = TriggerEditModel(destination)
    results = []
    model.structural_edit(lambda m: results.append(trigger_clipboard.paste_into(m, block, doc_id="B", scenario_version="1.41")))
    assert results[0].cleared_unit_refs == 2
    out = tmp_path / "pasted.aoe2scenario"
    write_scenario(destination, out, triggers=model)
    reloaded = load_map_and_units(out)
    _KEEP_ALIVE.append(reloaded)
    pasted = parse_triggers(reloaded).triggers[-1]
    assert (pasted.conditions[0].unit_object, pasted.effects[0].selected_object_ids) == (-1, [])


@pytest.mark.corpus
@pytest.mark.parametrize(
    ("source_name", "destination_name", "skip"),
    [
        ("R4_LeLoi_4", "1.59", range(91, 98)),  # effect 91 is in no vocabulary: that trigger stays behind
        ("8tp3w9j", "atilla_1_scn_resaved", ()),
        ("atilla_1_scn_resaved", "1.54", ()),
    ],
    ids=["1.56-into-1.59", "1.59-into-1.58", "1.58-into-1.54"],
)
def test_a_real_files_triggers_paste_across_versions_and_save(source_name, destination_name, skip, tmp_path) -> None:
    source = _example(source_name)
    destination = _default(destination_name)[0] if destination_name[0].isdigit() else _example(destination_name)
    block, _result, pasted, _manager = _paste_save_reload(source, destination, tmp_path, skip)
    assert len(pasted) == len(block.triggers) > 0


@pytest.mark.corpus
def test_the_effect_91_trigger_is_refused_by_number() -> None:
    source, destination = _example("R4_LeLoi_4"), _default("1.59")[1]
    manager = parse_triggers(source)
    index = next(i for i, t in enumerate(manager.triggers) if any(e.effect_type == 91 for e in t.effects))
    block = trigger_clipboard.copy_block(manager, [index], doc_id="A", scenario_version="1.56")
    with pytest.raises(ValueError, match=r"of type 91, which 1\.56 does not define"):
        trigger_clipboard.paste_into(destination, block, doc_id="B", scenario_version="1.59")


@pytest.mark.corpus
@pytest.mark.parametrize(("quantity", "refused"), [(30, False), (300, True)])
def test_armour_round_trips_into_a_trigger_2_4_document_only_when_it_fits(quantity, refused, tmp_path) -> None:
    """The 1.41 ElCid file's Triggers section is 2.4, packed 8+8."""
    block = _entries("1.54", "change_object_armor", _armour_attack_class=4, _armour_attack_quantity=quantity)
    destination = _example("C2_ElCid_coop_1_v0_16")
    model = TriggerEditModel(destination)
    model.structural_edit(lambda m: m.add_trigger("Destination"))

    def paste(m):
        trigger_clipboard.paste_entries(m, len(m.triggers) - 1, block, doc_id="B", scenario_version="1.41")

    if refused:
        with pytest.raises(ValueError, match=r"armour/attack value too large for 1\.41"):
            paste(model.manager())
        return
    model.structural_edit(paste, content_touched=[len(model.manager().triggers) - 1])
    out = tmp_path / "pasted.aoe2scenario"
    write_scenario(destination, out, triggers=model)
    reloaded = load_map_and_units(out)
    _KEEP_ALIVE.append(reloaded)
    (effect,) = parse_triggers(reloaded).triggers[-1].effects
    assert (reloaded.trigger_write_supported, effect.effect_type) == (True, 31)
    assert (effect.armour_attack_class, effect.armour_attack_quantity) == (4, quantity)
