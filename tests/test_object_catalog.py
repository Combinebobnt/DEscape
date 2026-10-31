"""Coverage for descape/object_catalog.py (phase 4d): the merged object-id
lookup trigger_fields.resolve_reference() depends on, the CatalogEntry
catalogs the picker widgets read from, the document-scoped TriggerId/
VariableId resolvers, and (slice 4) the install-resolved name chain.

Qt-free throughout -- no QApplication needed to run this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from descape import asset_source, object_catalog


@pytest.fixture(autouse=True)
def _no_install_env(monkeypatch):
    """No test here wants the real install, and AOE2DE_INSTALL_PATH outranks
    the config conftest hides, so a shell exporting it would leak one in."""
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def test_combined_object_name_resolves_across_all_four_datasets() -> None:
    """The confirmed slice-0 defect: id 109 is TOWN CENTER in BuildingInfo but
    absent from UnitInfo, and the old per-presentation lookup returned "" for
    it. The merged lookup finds it regardless of which dataset it lives in."""
    assert object_catalog.combined_object_name(109) == "TOWN CENTER"
    assert object_catalog.combined_object_name(486) == "BROWN BEAR"
    assert object_catalog.combined_object_name(4) == "ARCHER"


def test_combined_object_name_is_empty_for_an_unknown_id() -> None:
    assert object_catalog.combined_object_name(999999) == ""


def test_objects_covers_every_dataset_with_no_duplicate_ids() -> None:
    catalog = object_catalog.objects()
    ids = [e.id for e in catalog]
    assert len(ids) == len(set(ids)), "an id appears under more than one category"
    categories = {e.category for e in catalog}
    assert categories == {"Units", "Buildings", "Heroes", "Others"}


@pytest.mark.parametrize("dataset", ["aoe2", "ror"])
def test_every_objects_category_is_an_object_list_type(dataset: str) -> None:
    """GH #41: the form's object list type row offers only OBJECT_CATEGORIES,
    so an entry under any other category could never be narrowed to."""
    from descape import game_dataset

    game_dataset.set_active(dataset)
    categories = {entry.category for entry in object_catalog.objects()}
    assert categories and categories <= set(object_catalog.OBJECT_CATEGORIES)
    assert object_catalog.OBJECT_CATEGORIES == ("Units", "Buildings", "Heroes", "Others"), "the in-game option order"


def test_objects_is_alphabetical_by_name() -> None:
    names = [e.name for e in object_catalog.objects()]
    assert names == sorted(names)


def test_objects_entries_carry_search_text() -> None:
    for entry in object_catalog.objects()[:5]:
        assert entry.name.lower() in entry.search_text
        assert str(entry.id) in entry.search_text


def test_objects_entries_carry_the_committed_dat_fields() -> None:
    """class_id/hidden come from object_catalog.json (slice 4), not from
    install configuration -- present even with no install set."""
    entry = object_catalog.entry(object_catalog.objects(), 109)
    assert entry is not None
    assert entry.class_id == 3
    assert entry.hidden is False


def test_an_entry_the_dat_table_does_not_cover_keeps_the_slice1_placeholders() -> None:
    """object_catalog.json degrades gracefully (missing file, or -- as here
    -- an id genuinely absent from it) to the same None/False shape slice 1
    shipped with, rather than raising."""
    dat_entry = object_catalog._dat_entry("objects", 999999)
    assert dat_entry is None
    entry = object_catalog._entry(999999, "MADE UP", "Units", dat_entry)
    assert entry.class_id is None
    assert entry.hidden is False


# -- slice 4: install-resolved names ------------------------------------------


def _install_with_town_center_string(tmp_path):
    strings_dir = tmp_path / "install" / "resources" / "en" / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text('5164 "Town Center"\n', encoding="utf-8")
    return tmp_path / "install"


def test_resolve_name_prefers_the_install_string(tmp_path) -> None:
    install = _install_with_town_center_string(tmp_path)
    asset_source.set_install_path_override(install)
    try:
        dat_entry = object_catalog._dat_entry("objects", 109)
        assert dat_entry is not None, "object_catalog.json must carry 109"
        assert object_catalog.resolve_name(109, "TOWN CENTER", dat_entry, section="objects") == "Town Center"
    finally:
        asset_source.set_install_path_override(None)


def test_resolve_name_falls_back_to_the_library_name_with_no_install() -> None:
    dat_entry = object_catalog._dat_entry("objects", 109)
    assert object_catalog.resolve_name(109, "TOWN CENTER", dat_entry, section="objects") == "TOWN CENTER"


def test_resolve_name_falls_back_to_the_dat_code_with_no_library_name(tmp_path) -> None:
    install = _install_with_town_center_string(tmp_path)  # has no entry for key 1
    asset_source.set_install_path_override(install)
    try:
        assert object_catalog.resolve_name(109, "", {"string_id": 1, "code": "RTWC"}, section="objects") == "RTWC"
    finally:
        asset_source.set_install_path_override(None)


def test_resolve_name_falls_back_to_unknown_with_nothing_at_all() -> None:
    assert object_catalog.resolve_name(999999, "", None, section="objects") == "UNKNOWN_999999"


def test_combined_object_name_resolves_through_the_install_once_configured(tmp_path) -> None:
    install = _install_with_town_center_string(tmp_path)
    asset_source.set_install_path_override(install)
    try:
        assert object_catalog.combined_object_name(109) == "Town Center"
    finally:
        asset_source.set_install_path_override(None)
    # And reverts once the override is cleared -- clear_caches() must reach
    # _combined_object_names(), not just asset_source's own caches.
    assert object_catalog.combined_object_name(109) == "TOWN CENTER"


def test_objects_and_techs_resolve_through_the_install_once_configured(tmp_path) -> None:
    strings_dir = tmp_path / "install" / "resources" / "en" / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text(
        '5164 "Town Center"\n7427 "Anarchy"\n', encoding="utf-8"
    )
    asset_source.set_install_path_override(tmp_path / "install")
    try:
        assert object_catalog.name_for(object_catalog.objects(), 109) == "Town Center"
        assert object_catalog.name_for(object_catalog.techs(), 16) == "Anarchy"
    finally:
        asset_source.set_install_path_override(None)
    assert object_catalog.name_for(object_catalog.objects(), 109) == "TOWN CENTER"
    assert object_catalog.name_for(object_catalog.techs(), 16) == "ANARCHY"


def test_resolve_name_respects_the_configured_language(tmp_path) -> None:
    strings_dir = tmp_path / "install" / "resources" / "de" / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text('5164 "Stadtzentrum"\n', encoding="utf-8")
    asset_source.CONFIG_PATH.write_text(yaml.safe_dump({"language": "de"}))
    asset_source.set_install_path_override(tmp_path / "install")
    try:
        assert object_catalog.combined_object_name(109) == "Stadtzentrum"
    finally:
        asset_source.set_install_path_override(None)


@pytest.mark.parametrize("builder_name", ["_combined_object_names", "objects", "techs"])
def test_bulk_builders_resolve_the_language_once_per_build(monkeypatch, builder_name) -> None:
    """get_language() parses config.yaml uncached, so a per-row call cost ~4 s
    on a real config; each bulk build must resolve it exactly once."""
    real_get_language = asset_source.get_language
    calls = []

    def counting_get_language() -> str:
        calls.append(1)
        return real_get_language()

    monkeypatch.setattr(asset_source, "get_language", counting_get_language)
    builder = getattr(object_catalog, builder_name)
    builder.cache_clear()
    try:
        builder()
    finally:
        builder.cache_clear()
    assert len(calls) == 1


def test_techs_returns_every_techinfo_member_alphabetically() -> None:
    from AoE2ScenarioParser.datasets.techs import TechInfo

    catalog = object_catalog.techs()
    assert len(catalog) == len(list(TechInfo))
    names = [e.name for e in catalog]
    assert names == sorted(names)
    assert all(e.category == "Techs" for e in catalog)


def test_entry_and_name_for() -> None:
    catalog = object_catalog.objects()
    found = object_catalog.entry(catalog, 109)
    assert found is not None
    assert found.name == "TOWN CENTER"
    assert object_catalog.name_for(catalog, 109) == "TOWN CENTER"


def test_entry_and_name_for_miss() -> None:
    catalog = object_catalog.objects()
    assert object_catalog.entry(catalog, 999999) is None
    assert object_catalog.name_for(catalog, 999999) == ""


# -- document-scoped references ----------------------------------------------


def _manager(triggers=(), variables=()):
    return SimpleNamespace(
        triggers=[SimpleNamespace(trigger_id=tid, name=name) for tid, name in triggers],
        variables=[SimpleNamespace(variable_id=vid, name=name) for vid, name in variables],
    )


def test_trigger_choices_and_name_for() -> None:
    manager = _manager(triggers=[(0, "Setup"), (1, "Attack Wave 1")])
    assert object_catalog.trigger_choices(manager) == (("Setup", 0), ("Attack Wave 1", 1))
    assert object_catalog.trigger_name_for(manager, 1) == "Attack Wave 1"


def test_trigger_name_for_falls_back_to_empty_when_missing() -> None:
    manager = _manager(triggers=[(0, "Setup")])
    assert object_catalog.trigger_name_for(manager, 7) == ""


def test_trigger_choices_labels_an_unnamed_trigger_by_id() -> None:
    manager = _manager(triggers=[(3, "")])
    assert object_catalog.trigger_choices(manager) == (("Trigger 3", 3),)


def test_variable_choices_and_name_for() -> None:
    manager = _manager(variables=[(0, "fixture_var"), (1, "gold_bonus")])
    assert object_catalog.variable_choices(manager) == (
        ("fixture_var", 0),
        ("gold_bonus", 1),
    )
    assert object_catalog.variable_name_for(manager, 0) == "fixture_var"


def test_variable_name_for_falls_back_to_empty_when_missing() -> None:
    manager = _manager(variables=[(0, "fixture_var")])
    assert object_catalog.variable_name_for(manager, 5) == ""


def test_variable_choices_labels_an_unnamed_variable_by_id() -> None:
    manager = _manager(variables=[(2, "")])
    assert object_catalog.variable_choices(manager) == (("Variable 2", 2),)


# -- tech_name: the tech-side mirror of object_name ---------------------------


def test_tech_name_resolves_a_tech_the_enum_carries() -> None:
    assert object_catalog.tech_name(22) == "Loom"


def test_tech_name_resolves_an_internal_tech_the_enum_hides() -> None:
    """1359 is the motivating case from GH #57's plan: a real, disable-able
    tech id whose row object_catalog.json carries but no heuristic separates
    from an ordinary one. Asserted as "resolves at all" rather than against a
    literal, because which of the two names comes back depends on whether an
    install is visible -- conftest deliberately hides the configured one, so
    this is the .dat short code here and the install string in the app."""
    assert object_catalog.tech_name(1359) not in ("", "UNKNOWN_1359")


def test_tech_name_falls_back_to_unknown() -> None:
    assert object_catalog.tech_name(999999) == "UNKNOWN_999999"


def test_tech_name_and_object_name_read_separate_id_spaces() -> None:
    """The reason this is its own function: the same number means different
    things in the two tables, so resolving a tech through the objects table
    would name it after an unrelated unit."""
    assert object_catalog.tech_name(109) != object_catalog.object_name(109)


# -- dat-only objects (the Sept 2026 patch's Jarl, Longhouse, Spruce, ...) -----

# Library ids whose library category derived_category() does not reproduce,
# measured 2026-10-07: PTWC 444 (type 80, class 51, hero_mode 34) is the one
# editor-visible miss; 1639, 1654 and 2171 are editor-hidden helpers.
_DERIVED_CATEGORY_EXCEPTIONS = frozenset({444, 1639, 1654, 2171})


def _library_categories() -> dict[int, str]:
    categories: dict[int, str] = {}
    for dataset, category in object_catalog._OBJECT_DATASETS:
        for member in dataset:
            categories.setdefault(member.ID, category)
    return categories


def test_derived_category_reproduces_the_library_category() -> None:
    """The rule only ever categorizes dat-only ids, so it is re-measured here
    against every id the library does categorize. A new mismatch fails with
    its id listed rather than silently miscategorizing future dat-only ids."""
    mismatches = set()
    for id_, category in _library_categories().items():
        dat_entry = object_catalog._dat_entry("objects", id_)
        assert dat_entry is not None, f"object_catalog.json lacks library id {id_}"
        if object_catalog.derived_category(dat_entry) != category:
            mismatches.add(id_)
    assert mismatches == _DERIVED_CATEGORY_EXCEPTIONS


@pytest.mark.parametrize(
    ("id_", "category"),
    [(2708, "Units"), (2559, "Buildings"), (2735, "Units"), (2731, "Others"), (2721, "Heroes")],
)
def test_objects_lists_the_patchs_dat_only_objects(id_: int, category: str) -> None:
    assert id_ not in _library_categories(), "the case only tests anything for a dat-only id"
    found = object_catalog.entry(object_catalog.objects(), id_)
    assert found is not None, f"{id_} missing from objects()"
    assert found.category == category
    assert found.hidden is False


@pytest.mark.parametrize("id_", [621, 35])
def test_objects_leaves_out_editor_hidden_dat_only_ids(id_: int) -> None:
    dat_entry = object_catalog._dat_entry("objects", id_)
    assert dat_entry is not None and dat_entry["hidden"] is True
    assert id_ not in _library_categories()
    assert object_catalog.entry(object_catalog.objects(), id_) is None


def test_objects_lists_the_gunpowder_keg_as_the_official_editor_does() -> None:
    """GH #208: the .dat hides 2607, but the official editor lists it under
    Other, so the picker does too, without "Show editor-hidden objects"."""
    assert object_catalog._dat_entry("objects", 2607)["hidden"] is True
    assert 2607 not in _library_categories()
    found = object_catalog.entry(object_catalog.objects(), 2607)
    assert found is not None
    assert found.category == "Others"
    assert found.hidden is False
    assert found.class_id == 64


def test_objects_adds_exactly_the_editor_visible_dat_only_ids_and_the_keg() -> None:
    """Plus Invisible Spawner A-J, which the 1.59 editor lists under Other (GH #32)."""
    table = object_catalog._dat_table()["objects"]
    library = _library_categories()
    expected = {int(k) for k, v in table.items() if int(k) not in library and not v["hidden"]}
    assert len(expected) == 64  # measured 2026-09-27, unchanged by the 2026-10-07 regen
    added = {e.id for e in object_catalog.objects()} - set(library)
    assert added == expected | set(object_catalog._EDITOR_LISTED_DAT_ONLY)
    assert set(object_catalog._EDITOR_LISTED_DAT_ONLY) == {2607, 2616, 2617, 2618, 2619, 2620, 2621, 2645, 2646, 2647, 2648}


def test_display_name_names_a_dat_only_const_with_no_install() -> None:
    """The .dat code with no install, where it used to read UNKNOWN_2708. A
    hidden dat-only const (GH #52's barrel, 2607) is named the same way."""
    assert object_catalog.display_name(2708) == "Jarl"
    assert object_catalog.display_name(2607) == "GUNPOWDERKEG"
    assert object_catalog.display_name(999999) == "UNKNOWN_999999"


def test_display_name_names_a_dat_only_const_from_the_install(tmp_path) -> None:
    strings_dir = tmp_path / "install" / "resources" / "en" / "strings" / "key-value"
    strings_dir.mkdir(parents=True)
    (strings_dir / "key-value-strings-utf8.txt").write_text('21390 "Longhouse A"\n', encoding="utf-8")
    asset_source.set_install_path_override(tmp_path / "install")
    try:
        assert object_catalog.display_name(2559) == "Longhouse A"
        assert object_catalog.name_for(object_catalog.objects(), 2559) == "Longhouse A"
    finally:
        asset_source.set_install_path_override(None)
    assert object_catalog.display_name(2559) == "LONGHOUSEA"


def test_combined_object_name_stays_dataset_only() -> None:
    """The trigger formatter's ALL-CAPS contract reads this, so dat-only ids
    must not leak into it."""
    assert object_catalog.combined_object_name(2708) == ""


# --- GH #176: the placement-rule keys -----------------------------------------


def test_objects_carry_the_placement_rule_keys() -> None:
    """Villager 83 walks on land and has a 0.2-tile clearance box; Shore Fish
    69 needs Beach 2 or Ice 35 beside it."""
    objects = object_catalog.dat_objects()
    assert objects[83]["restriction"] == 7
    assert objects[83]["clearance"] == [0.2, 0.2]
    assert "side_terrain" not in objects[83]
    assert objects[69]["side_terrain"] == [2, 35]


def test_the_placement_rule_keys_are_omitted_when_empty() -> None:
    """As `obstruction`: a key is present only when it says something, so a
    reader's `.get()` default is the passable answer."""
    for entry in object_catalog.dat_objects().values():
        assert entry.get("restriction", 1) > 0
        assert any(entry.get("clearance", [1.0, 1.0]))
        assert entry.get("side_terrain", [0])
        assert all(terrain_id >= 0 for terrain_id in entry.get("side_terrain", []))


def test_every_restriction_in_use_has_a_row_of_known_terrains() -> None:
    rows = object_catalog.dat_terrain_restrictions()
    used = {entry["restriction"] for entry in object_catalog.dat_objects().values() if "restriction" in entry}
    assert used, "no object carries a restriction; the catalog was not regenerated"
    # Row 2 is the Beach row: no object uses it, the shore rule reads it.
    assert set(rows) == used | {2}
    assert {2, 37} <= rows[2] and 0 not in rows[2]
    known = set(object_catalog.dat_terrains())
    for restriction_id, passable in rows.items():
        assert passable <= known, f"restriction {restriction_id} names a terrain the catalog lacks"
    # Land units (7) walk on Grass 0 and not on Water, Deep 22; fishing ships (13) the reverse.
    assert 0 in rows[7] and 22 not in rows[7]
    assert 22 in rows[13] and 0 not in rows[13]


# --- GH #169: the command-button keys -----------------------------------------


def test_objects_and_techs_carry_their_command_buttons() -> None:
    """Militia 74 is Barracks 12 button 1, Loom 22 is Town Center 109 button 6,
    and the builder unit 118 trains House 70 on the economic page (2) and
    Barracks on the military one (10)."""
    objects = object_catalog.dat_objects()
    assert objects[74]["train"] == [[12, 1]]
    assert "interface_kind" not in objects[74]
    assert object_catalog.dat_techs()[22]["research"] == [[109, 6]]
    assert [118, 1] in objects[70]["train"] and objects[70]["interface_kind"] == 2
    assert [118, 1] in objects[12]["train"] and objects[12]["interface_kind"] == 10


def test_the_command_button_keys_are_omitted_when_empty() -> None:
    """Every pair names a building and a real button; 0 (no button) is dropped."""
    rows = [*object_catalog.dat_objects().values(), *object_catalog.dat_techs().values()]
    for row in rows:
        for building, button in row.get("train", []) + row.get("research", []):
            assert building >= 0 and button >= 1
        assert row.get("train", [0]) and row.get("research", [0])
    assert all(
        "interface_kind" not in row or any(b == 118 for b, _n in row["train"])
        for row in object_catalog.dat_objects().values()
    )


def test_a_missing_json_still_gives_the_library_only_catalog(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(object_catalog, "_CATALOG_JSON_PATH", tmp_path / "absent.json")
    object_catalog._dat_table.cache_clear()
    object_catalog.clear_caches()
    try:
        assert {e.id for e in object_catalog.objects()} == set(_library_categories())
        assert object_catalog.dat_terrains() == {}
        assert object_catalog.dat_terrain_restrictions() == {}
        assert object_catalog.display_name(2708) == "UNKNOWN_2708"
    finally:
        object_catalog._dat_table.cache_clear()
        object_catalog.clear_caches()


# --- GH #230: which civs can have a trainable unit or a researchable tech -----


def test_civs_lists_the_civs_that_can_have_a_unit() -> None:
    """Mule Cart is Armenian and Georgian, Conquistador Spanish (14), Longbowman Briton (1), Krepost Bulgarian (32)."""
    objects = object_catalog.dat_objects()
    assert objects[1808]["civs"] == [44, 45]
    assert objects[771]["civs"] == [14]
    assert objects[8]["civs"] == [1]
    assert objects[1251]["civs"] == [32]
    assert 14 not in objects[39]["civs"], "Spanish have no Cavalry Archer"


def test_civs_is_omitted_when_every_civ_can_and_only_set_with_a_location() -> None:
    objects, techs = object_catalog.dat_objects(), object_catalog.dat_techs()
    assert "civs" not in objects[74] and "civs" not in techs[22], "Militia and Loom are for everyone"
    assert all("train" in row for row in objects.values() if "civs" in row)
    assert all("research" in row for row in techs.values() if "civs" in row)
    assert all(row["civs"] == sorted(row["civs"]) for row in [*objects.values(), *techs.values()] if "civs" in row)


def test_regional_units_follow_the_civ_trees_re_enables() -> None:
    """Disable Regionals takes camels from every civ; Saracens (9) get them back, Britons (1) do not."""
    camel = object_catalog.dat_objects()[329]["civs"]
    assert 9 in camel and 1 not in camel


def test_gives_lists_the_trainable_units_a_tech_enables_or_upgrades_into() -> None:
    techs = object_catalog.dat_techs()
    assert techs[509]["gives"] == [881], "Elite Kamayuk upgrades Kamayuk into Elite Kamayuk"
    assert "gives" not in techs[22]


def test_age_is_the_age_a_tech_becomes_researchable_in() -> None:
    """Measured: a later start has every generic tech of an earlier age (Loom 2, Bloodlines 3, Long Swordsman 4, Chemistry 5)."""
    techs = object_catalog.dat_techs()
    assert [techs[t]["age"] for t in (22, 435, 207, 47)] == [2, 3, 4, 5]
    assert [techs[t]["age"] for t in (101, 102, 103)] == [2, 3, 4], "each age tech is researched in the age before"
    assert all("age" in row for row in techs.values() if "research" in row)


def test_every_civ_has_a_unit_at_castle_button_1() -> None:
    """A type 2 enable with B = -1 is an enable: seven civs' unique units ("Hearth Troop (make avail)") use it."""
    objects = object_catalog.dat_objects()
    every_civ = {civ for row in objects.values() for civ in row.get("civs", ())}
    assert len(every_civ) >= 62
    trains_there = [row for row in objects.values() if [82, 1] in row.get("train", [])]
    for civ in every_civ:
        assert any(civ in row.get("civs", [civ]) for row in trains_there), civ


def test_unique_units_enabled_with_b_minus_one_are_their_civs_own() -> None:
    objects = object_catalog.dat_objects()
    assert objects[2705]["civs"] == [60], "Hearth Troop is Saxon"
    assert objects[2101]["civs"] == [46], "Immortal is Achaemenid"


def test_other_game_modes_start_techs_do_not_fire() -> None:
    """Disable Spies (Regicide only) leaves Spy Technology to every civ in a standard game."""
    assert "civs" not in object_catalog.dat_techs()[408]


def test_a_shadow_tech_fires_only_where_the_tree_zeroes_its_cost() -> None:
    """Paphos Shadow Tech costs 1 food; only the Chronicles trees zero it, so Armenians (44) and Georgians (45) get no Lembos."""
    assert object_catalog.dat_objects()[2123]["civs"] == [46, 47, 48, 54, 55, 56]


def test_a_unit_only_a_unique_building_trains_is_that_civs() -> None:
    """Nothing enables Konnik or Serjeant; the Krepost (Bulgarians) and Donjon (Sicilians) train them."""
    objects = object_catalog.dat_objects()
    assert objects[1254]["civs"] == [32]
    assert objects[1660]["civs"] == [37]


def test_a_team_bonus_that_zeroes_a_shadow_techs_cost_counts() -> None:
    """Fortified Outpost needs Macedonian Team Bonus 1269, whose cost only the Macedonians' (54) team bonus zeroes."""
    assert object_catalog.dat_objects()[2415]["civs"] == [54]
    assert object_catalog.dat_techs()[1270]["civs"] == [54]


# The park plan's per-family counts (2026-09-02), re-measured on .dat code names 2026-10-29.
_INTERNAL_TECH_COUNTS = {
    "Blank Technology": 40,
    "Placeholder": 10,
    "C-Bonus": 3,
    "(make avail)": 4,
    "Auto upgrade": 1,
    "requirement": 11,
    "Change TC constr time": 2,
    "Static Bonuses": 3,
}


def test_the_internal_tech_families_mark_the_measured_rows() -> None:
    """Each family of the source list matches the plan's count of TechInfo
    rows by .dat code name, and exactly those rows carry `internal`."""
    from collections import Counter

    codes = object_catalog.dat_techs()
    catalog = object_catalog.techs()
    families = {e.id: object_catalog.internal_tech_family(str(codes.get(e.id, {}).get("code") or "")) for e in catalog}
    counts = Counter(families[e.id] for e in catalog)
    assert {family: counts[family] for family in object_catalog.INTERNAL_TECH_FAMILIES} == _INTERNAL_TECH_COUNTS
    assert sum(e.internal for e in catalog) == sum(_INTERNAL_TECH_COUNTS.values()) == 74
    assert all(e.internal == bool(families[e.id]) for e in catalog)
    assert not any(e.internal for e in object_catalog.objects()), "objects never carry it"
