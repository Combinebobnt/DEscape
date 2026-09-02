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

from descape.trigger_organize import Section, is_divider, leading_divider_run, parse_tag, sections

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
