"""Covers descape/trigger_organize.py, the Qt-free heuristic behind Phase 4c
(trigger organization). No QApplication and no scenario file: parse_tag()/
is_divider()/sections() take plain names and a plain order.

Mutation-check reminder: the shipped default-tier fixture has 4 triggers and
no dividers, so it cannot discriminate this feature on its own -- these
synthetic cases plus a larger private trigger-name census (not shipped in
this repo, read by an internal tool, never by this test) are what actually
exercise the heuristic.
"""

from __future__ import annotations

import pytest

from descape.trigger_organize import (
    DEFAULT_DIVIDER_FORMAT,
    UNNAMED_SECTION,
    DividerFormat,
    Section,
    divider_format,
    divider_title_error,
    format_divider,
    is_divider,
    leading_divider_run,
    parse_tag,
    retag_name,
    retitle_divider,
    section_end_slot,
    section_header_of,
    section_key,
    section_label,
    sections,
    split_divider,
    split_tag,
    strip_tag,
)

# -- parse_tag ----------------------------------------------------------------


def test_parse_tag_reads_leading_bracket_forms():
    assert parse_tag("[VIP]") == "VIP"
    assert parse_tag("(unused)") == "unused"
    assert parse_tag("<tag>") == "tag"


def test_parse_tag_allows_slash_in_tag_body():
    # 4c's plan: 401 corpus names carry a [tag] prefix, some slash-bearing.
    assert parse_tag("[h/m]") == "h/m"


def test_parse_tag_requires_leading_position():
    assert parse_tag("Escort [VIP]") is None


def test_parse_tag_rejects_empty_and_unbracketed():
    assert parse_tag("[]") is None
    assert parse_tag("Set Scene") is None
    assert parse_tag("") is None


# -- is_divider -----------------------------------------------------------------


def test_is_divider_accepts_real_corpus_shapes():
    assert is_divider("--- Setup ---")
    assert is_divider("---Setup---")
    assert is_divider("--Start--")
    assert is_divider("-" * 20)


def test_is_divider_rejects_near_misses():
    # From the plan: fails the trailing run.
    assert not is_divider("--> Attack")
    # Single '~' delimiters, not a >=2 run.
    assert not is_divider("~Part I - Road to Orleans~")


def test_is_divider_rejects_short_bare_run():
    # Two-character strings are neither a real leading+trailing rule (there's
    # nothing between the two overlapping runs) nor a >=3 bare run.
    assert not is_divider("--")
    assert not is_divider("")
    assert not is_divider("   ")


def test_is_divider_ignores_surrounding_whitespace():
    assert is_divider("  --- Setup ---  ")


def test_leading_divider_run_counts_only_the_leading_run():
    assert leading_divider_run("--- Setup ---") == 3
    assert leading_divider_run("Set Scene") == 0
    assert leading_divider_run("--> Attack") == 2


# -- sections -------------------------------------------------------------------


def test_sections_flat_file_is_one_synthetic_section():
    names = ["Set Scene", "Give Units", "End Game"]
    order = [0, 1, 2]
    assert sections(names, order) == [Section("", None, (0, 1, 2))]


def test_sections_partitions_at_each_divider():
    names = ["Intro", "--- Setup ---", "Give Units", "--- Combat ---", "Spawn Wave"]
    order = [0, 1, 2, 3, 4]
    assert sections(names, order) == [
        Section("", None, (0,)),
        Section("--- Setup ---", 1, (2,)),
        Section("--- Combat ---", 3, (4,)),
    ]


def test_sections_omits_empty_leading_run():
    # The first trigger in display order is itself a divider -- no synthetic
    # leading section should appear before it.
    names = ["--- Setup ---", "Give Units"]
    order = [0, 1]
    assert sections(names, order) == [Section("--- Setup ---", 0, (1,))]


def test_sections_allows_empty_section_between_adjacent_dividers():
    names = ["--- A ---", "--- B ---", "Give Units"]
    order = [0, 1, 2]
    assert sections(names, order) == [
        Section("--- A ---", 0, ()),
        Section("--- B ---", 1, (2,)),
    ]


def test_sections_follows_order_not_list_index():
    # trigger_display_order can differ from raw list order; sections() must
    # partition the order it is given, not range(len(names)).
    names = ["--- Setup ---", "First", "Second"]
    order = [0, 2, 1]
    assert sections(names, order) == [Section("--- Setup ---", 0, (2, 1))]


def test_sections_respects_a_filtered_subset_of_order():
    # sections() makes no assumption that `order` covers every trigger --
    # a caller may hand it a filtered view.
    names = ["--- Setup ---", "Give Units", "End Game"]
    order = [0, 2]
    assert sections(names, order) == [Section("--- Setup ---", 0, (2,))]


# -- split_tag / retag_name / strip_tag -----------------------------------------

# Every shape the census found after a tag's closer, plus synthetic inner
# padding and leading whitespace that parse_tag() accepts but no file uses.
_TAGGED_SHAPES = [
    ("[D0] Spawn", ("[", "D0", "] Spawn")),
    ("[P1]Wolf Sound", ("[", "P1", "]Wolf Sound")),
    ("[D10][S30] P3 Defeated", ("[", "D10", "][S30] P3 Defeated")),
    ("<BLUE>>>>>> go", ("<", "BLUE", ">>>>>> go")),
    ("(DISABLE_LIVE)_test", ("(", "DISABLE_LIVE", ")_test")),
    ("[D1]: X", ("[", "D1", "]: X")),
    ("[ D1 ] X", ("[ ", "D1", " ] X")),
    ("  [D1] X  ", ("  [", "D1", "] X  ")),
    ("[h/m]", ("[", "h/m", "]")),
]


def test_split_tag_cuts_every_shape_around_parse_tags_tag():
    for name, expected in _TAGGED_SHAPES:
        parts = split_tag(name)
        assert parts == expected, name
        assert "".join(parts) == name
        assert parts[1] == parse_tag(name)


def test_split_tag_is_none_exactly_where_parse_tag_is():
    for name in ["", "   ", "Set Scene", "Escort [VIP]", "[]", "[ ]", "[unclosed", "--- Setup ---"]:
        assert parse_tag(name) is None
        assert split_tag(name) is None, name


def test_retag_name_changes_only_the_tag():
    assert retag_name("[D0] Spawn", "Intro") == "[Intro] Spawn"
    assert retag_name("[P1]Wolf Sound", "P2") == "[P2]Wolf Sound"
    assert retag_name("[D10][S30] P3 Defeated", "X") == "[X][S30] P3 Defeated"
    assert retag_name("<BLUE>>>>>> go", "RED") == "<RED>>>>>> go"
    assert retag_name("[ D1 ] X", "Longer tag") == "[ Longer tag ] X"
    assert retag_name("  [D1] X  ", "Y") == "  [Y] X  "


def test_retag_name_refuses_an_untagged_name():
    with pytest.raises(ValueError):
        retag_name("Set Scene", "X")


def test_strip_tag_drops_the_tag_and_its_separator():
    assert strip_tag("[D0] Spawn") == "Spawn"
    assert strip_tag("[P1]Wolf Sound") == "Wolf Sound"
    assert strip_tag("[D1]: X") == ": X"
    assert strip_tag("(DISABLE_LIVE)_test") == "_test"
    assert strip_tag("<BLUE>>>>>> go") == ">>>>> go"
    assert strip_tag("[ D1 ]   X  ") == "X  "
    assert strip_tag("  [D1] X") == "  X"
    assert strip_tag("[D1]") == ""


def test_strip_tag_on_a_chained_name_exposes_the_next_tag():
    stripped = strip_tag("[D10][S30] P3 Defeated")
    assert stripped == "[S30] P3 Defeated"
    assert parse_tag(stripped) == "S30"


def test_strip_tag_refuses_an_untagged_name():
    with pytest.raises(ValueError):
        strip_tag("Escort [VIP]")


# -- section_key / section_label ------------------------------------------------

# The divider sequence of the census file with the most repeated title
# (F2_Dracula_coop_2_v0_04: `--RISK!!!--` ten times), one member under each.
_RISK_DIVIDERS = ["--START RISK!!!--"] + ["--RISK!!!--"] * 10 + ["--END RISK--"]


def _risk_parts() -> list:
    names = ["intro"]
    for divider in _RISK_DIVIDERS:
        names += [divider, f"member of {divider}"]
    return sections(names, list(range(len(names))))


def test_section_key_gives_same_titled_sections_distinct_ordinals():
    parts = _risk_parts()
    keys = [section_key(parts, i) for i in range(len(parts))]
    assert keys[0] == ("", 0), "the synthetic leading section"
    assert keys[1] == ("--START RISK!!!--", 0)
    assert keys[2:12] == [("--RISK!!!--", n) for n in range(10)]
    assert keys[12] == ("--END RISK--", 0)
    assert len(set(keys)) == len(keys)


def test_section_key_ignores_trigger_indices():
    """Same titles in the same order give the same keys whatever the ids."""
    names = ["--RISK!!!--", "a", "--RISK!!!--", "b"]
    forward = sections(names, [0, 1, 2, 3])
    swapped = sections(names, [2, 3, 0, 1])
    assert [section_key(forward, i) for i in range(2)] == [section_key(swapped, i) for i in range(2)]


def test_section_label_names_a_pure_ruler_and_keeps_a_real_title():
    assert section_label("--------") == UNNAMED_SECTION
    assert section_label("-=~*_#-=") == UNNAMED_SECTION
    assert section_label("--RISK!!!--") == "--RISK!!!--"
    assert section_label("--- 2 ---") == "--- 2 ---"


# -- divider formats (section management) -------------------------------------

# The census's four shapes, plus the trailing-space and 3+4 cases seen in
# examples/.
_SHAPES = {
    "--- Setup ---": ("--- ", "Setup", " ---", DividerFormat("-", 3, 3, True)),
    "--Start--": ("--", "Start", "--", DividerFormat("-", 2, 2, False)),
    "---Setup---": ("---", "Setup", "---", DividerFormat("-", 3, 3, False)),
    "---------- front-horde ---------- ": (
        "---------- ",
        "front-horde",
        " ---------- ",
        DividerFormat("-", 10, 10, True),
    ),
    "--- Setup ----": ("--- ", "Setup", " ----", DividerFormat("-", 3, 4, True)),
    "--Tributes- NONE--": ("--", "Tributes- NONE", "--", DividerFormat("-", 2, 2, False)),
}


@pytest.mark.parametrize("name", list(_SHAPES))
def test_split_divider_cuts_every_corpus_shape(name):
    prefix, title, suffix, _ = _SHAPES[name]
    assert split_divider(name) == (prefix, title, suffix)
    assert prefix + title + suffix == name


def test_split_divider_is_none_without_a_title_slot():
    assert split_divider("----------") is None
    assert split_divider("--- ---") is None
    assert split_divider("Setup") is None
    assert split_divider("--> Attack") is None


@pytest.mark.parametrize("name", list(_SHAPES))
def test_divider_format_reads_each_shape(name):
    assert divider_format([name]) == _SHAPES[name][3]


def test_divider_format_takes_the_mode_and_ignores_non_dividers():
    names = ["--A--", "--- B ---", "--C--", "plain", "------", "--D--"]
    assert divider_format(names) == DividerFormat("-", 2, 2, False)


def test_divider_format_tie_prefers_the_default_then_the_smallest_runs():
    assert divider_format(["----X----", "--- Y ---"]) == DEFAULT_DIVIDER_FORMAT
    assert divider_format(["----X----", "--Y--"]) == DividerFormat("-", 2, 2, False)


def test_divider_format_falls_back_to_the_default():
    assert divider_format([]) == DEFAULT_DIVIDER_FORMAT
    assert divider_format(["------", "=====", "Setup"]) == DEFAULT_DIVIDER_FORMAT


def test_divider_format_skips_mixed_character_runs():
    assert divider_format(["-=- X -=-", "--Y--"]) == DividerFormat("-", 2, 2, False)


@pytest.mark.parametrize("fmt", sorted({shape[3] for shape in _SHAPES.values()}, key=repr))
@pytest.mark.parametrize("title", ["Setup", "a", "Wave 3 / late", " "])
def test_format_divider_always_reads_back_as_a_divider(fmt, title):
    name = format_divider(title, fmt)
    assert is_divider(name)
    if title.strip():
        assert split_divider(name)[1] == title
        assert divider_format([name]) == fmt


def test_format_divider_defaults_to_the_spaced_three_run():
    assert format_divider("New") == "--- New ---"


@pytest.mark.parametrize("name", list(_SHAPES))
def test_retitle_divider_keeps_the_decoration_byte_for_byte(name):
    prefix, _, suffix, _ = _SHAPES[name]
    assert retitle_divider(name, "Renamed") == prefix + "Renamed" + suffix


def test_retitle_divider_refuses_a_bare_run():
    with pytest.raises(ValueError):
        retitle_divider("------", "X")


def test_divider_title_error_catches_a_title_merging_into_the_run():
    assert divider_title_error("--- X ---", "X") == ""
    assert divider_title_error("--X--", "")
    assert divider_title_error("--a---", "a-")


def test_section_end_slot_covers_each_section_shape():
    names = ["lead", "--- A ---", "a1", "a2", "--- B ---", "--- C ---", "c1"]
    order = [0, 1, 2, 3, 4, 5, 6]
    assert section_end_slot(names, order, None) == 1
    assert section_end_slot(names, order, 1) == 4
    assert section_end_slot(names, order, 4) == 5, "an empty section ends past its header"
    assert section_end_slot(names, order, 5) == 7
    # A file whose first display slot is a divider has an empty leading section.
    assert section_end_slot(names, [1, 2, 0, 3], None) == 0
    with pytest.raises(ValueError):
        section_end_slot(names, order, 2)


def test_section_end_slot_follows_display_order_not_list_order():
    names = ["--- A ---", "a1", "--- B ---", "b1"]
    assert section_end_slot(names, [2, 3, 0, 1], 2) == 2
    assert section_end_slot(names, [2, 3, 0, 1], 0) == 4


def test_section_header_of_names_the_containing_section():
    names = ["lead", "--- A ---", "a1"]
    assert section_header_of(names, [0, 1, 2], 0) is None
    assert section_header_of(names, [0, 1, 2], 1) == 1
    assert section_header_of(names, [0, 1, 2], 2) == 1


@pytest.mark.corpus
def test_split_divider_agrees_with_is_divider_across_the_corpus(scenario_path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers

    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    for trigger in manager.triggers:
        name = trigger.name or ""
        parts = split_divider(name)
        if parts is None:
            continue
        assert is_divider(name), name
        assert "".join(parts) == name
        assert is_divider(retitle_divider(name, "Renamed")), name


@pytest.mark.corpus
def test_split_tag_agrees_with_parse_tag_across_the_corpus(scenario_path) -> None:
    from descape.scenario_io import load_map_and_units, parse_triggers

    loaded = load_map_and_units(scenario_path)
    manager = parse_triggers(loaded)
    if manager is None:
        pytest.skip(f"{scenario_path.name}: Triggers section does not parse")
    for trigger in manager.triggers:
        name = trigger.name or ""
        parts = split_tag(name)
        if parts is None:
            assert parse_tag(name) is None
            continue
        assert parts[1] == parse_tag(name), name
        assert "".join(parts) == name
