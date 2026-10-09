"""GH #140: descape/message_markup.py, the Display Instructions text markup
(colour tags, trigger variable names, everything else literal), and the
install-backed lookups the preview reads.

Qt-free. The corpus-marked tests need AOE2DE_INSTALL_PATH (conftest hides
config.yaml) and skip without it.
"""

from __future__ import annotations

import json

import pytest

from descape import asset_source, message_markup
from descape.message_markup import LITERAL, TEXT, VARIABLE, Run

BLUE = message_markup.TEXT_COLORS["BLUE"]
GREY = message_markup.TEXT_COLORS["GREY"]
AQUA = message_markup.TEXT_COLORS["AQUA"]


@pytest.fixture
def _no_install_env(monkeypatch):
    monkeypatch.delenv("AOE2DE_INSTALL_PATH", raising=False)
    asset_source.clear_install_caches()


def _install_with_colors(tmp_path, tables: dict):
    install = tmp_path / "AoE2DE"
    (install / "widgetui").mkdir(parents=True)
    (install / "widgetui" / "UIColors.json").write_text(json.dumps({"ColorTables": tables}))
    return install


def test_a_leading_colour_tag_colours_the_whole_message() -> None:
    assert message_markup.parse("<BLUE>Scout: over here") == [Run("Scout: over here", BLUE, TEXT)]


def test_a_mid_string_colour_tag_switches_the_colour_for_the_rest() -> None:
    runs = message_markup.parse("<BLUE>Ally: <GREY>(whispers) go")
    assert runs == [Run("Ally: ", BLUE, TEXT), Run("(whispers) go", GREY, TEXT)]


def test_text_before_any_tag_has_the_default_colour() -> None:
    assert message_markup.parse("plain <BLUE>blue") == [Run("plain ", None, TEXT), Run("blue", BLUE, TEXT)]


def test_colour_tags_are_case_insensitive() -> None:
    assert message_markup.parse("<blue>a<Grey>b") == [Run("a", BLUE, TEXT), Run("b", GREY, TEXT)]


def test_a_trigger_variable_name_becomes_a_variable_run_in_the_current_colour() -> None:
    runs = message_markup.parse("<AQUA>Carts: <Cart-Status> left", ["cart-status", "other"])
    assert runs == [Run("Carts: ", AQUA, TEXT), Run("Cart-Status", AQUA, VARIABLE), Run(" left", AQUA, TEXT)]


def test_unknown_tags_stay_literal() -> None:
    runs = message_markup.parse("<b>bold</b> and <cost> and <cart-status>")
    assert "".join(run.text for run in runs) == "<b>bold</b> and <cost> and <cart-status>"
    assert {run.kind for run in runs if run.text.startswith("<")} == {LITERAL}
    assert all(run.color is None for run in runs)


def test_the_epic_brackets_are_kept_literal() -> None:
    runs = message_markup.parse("<AQUA><<<<<EPIC!>>>>>")
    assert "".join(run.text for run in runs) == "<<<<<EPIC!>>>>>"
    assert all(run.color == AQUA for run in runs)


def test_leading_color_is_the_first_runs_colour_only_when_the_message_opens_with_one() -> None:
    assert message_markup.leading_color(message_markup.parse("<GREY>x <BLUE>y")) == GREY
    assert message_markup.leading_color(message_markup.parse("x <BLUE>y")) is None
    assert message_markup.leading_color([]) is None


def test_to_html_escapes_and_colours() -> None:
    html = message_markup.to_html(message_markup.parse("<BLUE>a & b <c>"), (1, 2, 3))
    assert "&amp;" in html and "&lt;c&gt;" in html
    assert "<c>" not in html
    assert 'color:#6ea6eb' in html  # Blue's Text value, (110,166,235)


def test_to_html_uses_the_default_colour_for_untagged_text() -> None:
    assert 'color:#010203' in message_markup.to_html(message_markup.parse("plain"), (1, 2, 3))


@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
def test_to_html_turns_every_line_break_into_br(newline: str) -> None:
    html = message_markup.to_html(message_markup.parse(f"one{newline}two"), (0, 0, 0))
    assert "one<br>two" in html
    assert "\r" not in html and "\n" not in html


def test_to_html_shows_a_variable_as_a_bracketed_placeholder() -> None:
    html = message_markup.to_html(message_markup.parse("n=<v>", ["v"]), (0, 0, 0))
    assert "[v]" in html


def test_text_colors_falls_back_to_the_table_with_no_install(_no_install_env) -> None:
    assert asset_source.get_install_path() is None
    assert message_markup.text_colors() == message_markup.TEXT_COLORS


def test_text_colors_reads_the_install_and_clears_with_the_install_caches(tmp_path, _no_install_env) -> None:
    install = _install_with_colors(tmp_path, {"Blue": {"Text": [1, 2, 3, 255]}})
    asset_source.set_install_path_override(install)
    try:
        colors = message_markup.text_colors()
        assert colors["BLUE"] == (1, 2, 3)
        # A key the file lacks keeps the fallback.
        assert colors["GREY"] == GREY
        assert message_markup.parse("<BLUE>x")[0].color == (1, 2, 3)
    finally:
        asset_source.set_install_path_override(None)
    assert message_markup.text_colors()["BLUE"] == BLUE, "the colour cache outlived the install"


def test_a_malformed_colour_file_falls_back(tmp_path, _no_install_env) -> None:
    install = tmp_path / "AoE2DE"
    (install / "widgetui").mkdir(parents=True)
    (install / "widgetui" / "UIColors.json").write_text("{not json")
    asset_source.set_install_path_override(install)
    try:
        assert message_markup.text_colors() == message_markup.TEXT_COLORS
    finally:
        asset_source.set_install_path_override(None)


def test_unit_icon_is_none_with_no_install(_no_install_env) -> None:
    assert asset_source.get_unit_icon(64) is None


def test_icon_for_reads_the_catalogs_icon_and_treats_minus_one_as_none() -> None:
    from descape import object_catalog

    assert object_catalog.icon_for(448) == 64  # Scout Cavalry, from the committed catalog
    assert object_catalog.icon_for(0) is None  # icon -1
    assert object_catalog.icon_for(-1) is None
    assert object_catalog.icon_for(999999) is None


def test_unit_icon_matches_the_extension_case_insensitively_and_clears_with_the_install(
    tmp_path, _no_install_env
) -> None:
    from PIL import Image

    install = tmp_path / "AoE2DE"
    units = install / "widgetui" / "textures" / "ingame" / "units"
    units.mkdir(parents=True)
    Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(units / "064_50730.dds", format="DDS")
    asset_source.set_install_path_override(install)
    try:
        icon = asset_source.get_unit_icon(64)
        assert icon is not None and icon.shape == (6, 8, 4)
        assert tuple(icon[0, 0]) == (10, 20, 30, 255)
        assert asset_source.get_unit_icon(65) is None
        assert asset_source.get_unit_icon(-1) is None
    finally:
        asset_source.set_install_path_override(None)
    assert asset_source.get_unit_icon(64) is None, "the icon cache outlived the install"


# -- install-backed (corpus tier) ---------------------------------------------


def _require_install():
    if asset_source.get_install_path() is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")


@pytest.mark.corpus
def test_the_fallback_table_equals_the_installs_text_colours() -> None:
    _require_install()
    raw = json.loads((asset_source.get_install_path() / "widgetui" / "UIColors.json").read_text())
    tables = raw["ColorTables"]
    assert {name.upper() for name in tables} >= set(message_markup.TEXT_COLORS)
    for name, table in tables.items():
        if name.upper() in message_markup.TEXT_COLORS:
            assert message_markup.TEXT_COLORS[name.upper()] == tuple(table["Text"][:3]), name


@pytest.mark.corpus
def test_instruction_preview_icon_for_scout_cavalry_decodes_from_the_install() -> None:
    _require_install()
    from descape import object_catalog

    assert object_catalog.icon_for(448) == 64  # Scout Cavalry
    icon = asset_source.get_unit_icon(64)
    assert icon is not None
    assert icon.shape == (256, 256, 4)


@pytest.mark.corpus
def test_instruction_preview_string_id_resolves_from_the_install() -> None:
    """60014 is a Display Instructions string_id in atilla_1_scn_resaved."""
    _require_install()
    text = asset_source.resource_string(60014)
    assert text is not None and text.startswith("<RED>")
    assert message_markup.parse(text)[0].color == message_markup.TEXT_COLORS["RED"]
