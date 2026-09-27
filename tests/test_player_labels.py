"""GH #130: the shared player label helper. Qt-free; the widgets that show
these labels are covered by tests/test_player_labels_viewer.py."""

from __future__ import annotations

from pathlib import Path

from descape import player_fields, player_labels
from descape.player_labels import (
    DEFAULT_LABELS,
    LABEL_NAME_MAX,
    full_player_label,
    labels_for,
    player_label,
    tribe_names,
)
from descape.scenario_io import load_map_and_units

BLANK_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_blank_120x120.aoe2scenario"


def test_gaia_is_always_gaia() -> None:
    assert player_label(0) == "GAIA"
    assert player_label(0, "Nature") == "GAIA"
    assert full_player_label(0, "Nature") == "GAIA"


def test_an_empty_or_blank_name_is_the_bare_player_number() -> None:
    assert player_label(8) == "P8"
    assert player_label(8, "") == "P8"
    assert player_label(8, "   ") == "P8"
    assert full_player_label(8, " \t") == "P8"


def test_a_name_joins_with_a_single_hyphen() -> None:
    assert player_label(8, "Horde Army") == "P8 - Horde Army"
    assert player_label(8, "  Horde Army ") == "P8 - Horde Army"


def test_a_name_at_the_limit_is_not_cut() -> None:
    name = "x" * LABEL_NAME_MAX
    assert player_label(3, name) == f"P3 - {name}"


def test_a_name_over_the_limit_is_cut_with_an_ellipsis_but_the_full_label_is_not() -> None:
    name = "A" * (LABEL_NAME_MAX + 1)
    label = player_label(3, name)
    assert label == "P3 - " + "A" * (LABEL_NAME_MAX - 1) + "…"
    assert len(label.removeprefix("P3 - ")) == LABEL_NAME_MAX
    assert full_player_label(3, name) == f"P3 - {name}"


def test_labels_for_pairs_cut_text_with_uncut_tooltips() -> None:
    long_name = "B" * 40
    names = ("", "Britons", "", "", "", "", "", "", long_name)
    labels = labels_for(names)
    assert labels.text[:3] == ("GAIA", "P1 - Britons", "P2")
    assert labels.text[8].endswith("…")
    assert labels.full[8] == f"P8 - {long_name}"
    assert DEFAULT_LABELS.text == ("GAIA", *(f"P{n}" for n in range(1, 9)))


def test_tribe_names_merges_pending_over_the_files_values() -> None:
    loaded = load_map_and_units(BLANK_FIXTURE)
    committed = tribe_names(loaded)
    assert len(committed) == 9
    assert committed[0] == ""
    merged = tribe_names(loaded, {2: "  Horde Army ", 5: ""})
    assert merged[2] == "Horde Army"
    assert merged[5] == ""
    assert merged[1] == committed[1]
    assert merged[3:5] == committed[3:5]


def test_tribe_names_is_all_empty_when_the_file_has_no_tribe_name_spec(monkeypatch) -> None:
    loaded = load_map_and_units(BLANK_FIXTURE)
    specs = tuple(
        s for s in player_fields.specs_for(loaded) if s.field_id != player_labels.TRIBE_NAME_FIELD
    )
    monkeypatch.setattr(player_fields, "specs_for", lambda _loaded: specs)
    assert tribe_names(loaded, {2: "ignored"}) == ("",) * 9
