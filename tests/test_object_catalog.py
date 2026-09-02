"""Coverage for descape/object_catalog.py (phase 4d): the merged object-id
lookup trigger_fields.resolve_reference() depends on, the CatalogEntry
catalogs the picker widgets read from, the document-scoped TriggerId/
VariableId resolvers, and (slice 4) the install-resolved name chain.

Qt-free throughout -- no QApplication needed to run this file.
"""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from descape import asset_source, object_catalog


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
        assert object_catalog.resolve_name(109, "TOWN CENTER", dat_entry) == "Town Center"
    finally:
        asset_source.set_install_path_override(None)


def test_resolve_name_falls_back_to_the_library_name_with_no_install() -> None:
    dat_entry = object_catalog._dat_entry("objects", 109)
    assert object_catalog.resolve_name(109, "TOWN CENTER", dat_entry) == "TOWN CENTER"


def test_resolve_name_falls_back_to_the_dat_code_with_no_library_name(tmp_path) -> None:
    install = _install_with_town_center_string(tmp_path)  # has no entry for key 1
    asset_source.set_install_path_override(install)
    try:
        assert object_catalog.resolve_name(109, "", {"string_id": 1, "code": "RTWC"}) == "RTWC"
    finally:
        asset_source.set_install_path_override(None)


def test_resolve_name_falls_back_to_unknown_with_nothing_at_all() -> None:
    assert object_catalog.resolve_name(999999, "", None) == "UNKNOWN_999999"


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
