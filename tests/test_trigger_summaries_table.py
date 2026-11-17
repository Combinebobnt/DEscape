"""descape/trigger_summaries.json against every vocabulary DEscape can serve
(GH #388): every type has a Short and a Full template that parse, every
placeholder is a field the type lists in some version (or a pseudo-field),
and every field a type lists is shown, omitted or left to the appended
extras on purpose. Plus a corpus sweep: every entry in examples/ summarises
without falling back to the generic form."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from descape import library_compat, trigger_summary
from descape.scenario_io import load_map_and_units, parse_triggers

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
_KINDS = (("condition", "conditions"), ("effect", "effects"))
_KEYS = {"short", "full", "omit", "extra", "object_link", "link_as_object"}


def _fields_by_type() -> dict[tuple[str, str], set[str]]:
    """(kind, type name) -> every field it lists in any version."""
    found: dict[tuple[str, str], set[str]] = defaultdict(set)
    for version in library_compat.vocabulary_versions():
        vocabulary = library_compat.load_vocabulary(version)
        for kind, attribute in _KINDS:
            for type_id, definition in getattr(vocabulary, attribute).items():
                if type_id > 0:
                    found[(kind, definition.name)].update(definition.attributes)
    return found


FIELDS = _fields_by_type()
TABLE = trigger_summary.load_table()


def test_every_type_in_every_version_has_both_templates() -> None:
    missing = [
        key for key in FIELDS
        if not {"short", "full"} <= set(TABLE[trigger_summary.KIND_KEYS[key[0]]].get(key[1], {}))
    ]
    assert not missing, missing
    unknown = [
        (kind_key, name) for kind_key, types in TABLE.items() for name in types
        if (next(k for k, v in trigger_summary.KIND_KEYS.items() if v == kind_key), name) not in FIELDS
    ]
    assert not unknown, f"templates for types no vocabulary lists: {unknown}"


@pytest.mark.parametrize("key", sorted(FIELDS), ids=lambda key: f"{key[0]}-{key[1]}")
def test_a_template_parses_names_real_fields_and_covers_every_field(key) -> None:
    kind, name = key
    entry = TABLE[trigger_summary.KIND_KEYS[kind]][name]
    assert set(entry) <= _KEYS, set(entry) - _KEYS
    listed = FIELDS[key]
    named = set(entry.get("omit", ())) | set(entry.get("extra", ()))
    assert named <= listed, f"omit/extra name fields the type never lists: {named - listed}"
    for style in ("short", "full"):
        trigger_summary.parse(entry[style])
        unknown = {p for p in trigger_summary.placeholders(entry[style]) if p not in listed and p not in trigger_summary._PSEUDO_FIELDS}
        assert not unknown, f"{style}: placeholders the type never lists: {unknown}"
        shown = trigger_summary.covered_fields(entry[style]) | named | {trigger_summary.INVERTED, trigger_summary.TYPE_ATTRIBUTES[kind]}
        assert listed <= shown, f"{style}: fields neither shown, omitted nor extra: {sorted(listed - shown)}"


@pytest.mark.corpus
def test_every_examples_entry_summarises_in_both_styles() -> None:
    files = sorted(EXAMPLES.glob("*.aoe2scenario"))
    if not files:
        pytest.skip(f"no .aoe2scenario files in {EXAMPLES}")
    entries = 0
    fallbacks = set()
    for path in files:
        loaded = load_map_and_units(path)
        manager = parse_triggers(loaded)
        if manager is None or not library_compat.vocabulary_is_available(loaded.scenario_version):
            continue
        vocabulary = library_compat.load_vocabulary(loaded.scenario_version)
        for trigger in manager.triggers:
            for kind, attribute in _KINDS:
                definitions = getattr(vocabulary, attribute)
                presentation = getattr(vocabulary, f"{kind}_presentation")
                for entry in getattr(trigger, attribute):
                    definition = definitions.get(getattr(entry, trigger_summary.TYPE_ATTRIBUTES[kind]))
                    if definition is None or definition.id <= 0:  # type 0: no condition or effect chosen
                        continue
                    if definition.name not in TABLE[attribute]:
                        fallbacks.add((kind, definition.name))
                    for style in ("short", "full"):
                        trigger_summary.summarize(
                            kind, entry, definition, presentation, options=trigger_summary.Options(sentence_style=style)
                        )
                    entries += 1
    assert entries, "no file in examples/ had a parseable trigger section"
    assert not fallbacks, f"types read through the generic fallback: {sorted(fallbacks)}"
