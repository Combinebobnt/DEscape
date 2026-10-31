"""TASK-238: descape/colour_mood.py, the mood table and the draw-only grade.

MOOD_PARAMS is a hardcoded copy of the game's colorcorrection.json, so the
list and order are pinned to the library's ColorMood enum here, and the
values to the install by test_colour_mood_write_path.py's corpus test."""

from __future__ import annotations

import pytest
from AoE2ScenarioParser.datasets.trigger_lists import ColorMood

from descape import colour_mood, option_fields
from descape.colour_mood import MoodGrade, apply, grade_for

import conftest

# GH #165's screenshots of one map, Default then Night: mean grass and water pixels.
DEFAULT_GRASS, NIGHT_GRASS = (127.3, 139.1, 66.2), (70.6, 92.4, 107.5)
DEFAULT_WATER, NIGHT_WATER = (52.8, 100.3, 135.4), (42.3, 79.0, 142.4)


def test_the_list_is_the_librarys_enum_in_value_order() -> None:
    assert [name.upper() for name in option_fields.COLOUR_MOODS] == [m.name for m in sorted(ColorMood, key=int)]


def test_the_spec_offers_the_19_names_as_stored() -> None:
    spec = next(s for s in option_fields._SPECS if s.field_id == "map_color_mood")
    assert spec.choices == tuple((n, n) for n in option_fields.COLOUR_MOODS)
    assert len(spec.choices) == 19


@pytest.mark.parametrize("stored", ["", "Empty", "Default", "default", "默认", "Standard", "Leer", "Bogus"])
def test_a_name_with_no_mood_draws_unchanged(stored) -> None:
    assert grade_for(stored) is None


@pytest.mark.parametrize("stored", ["Night", "night", "NIGHT"])
def test_a_mood_matches_in_any_case(stored) -> None:
    grade = grade_for(stored)
    assert isinstance(grade, MoodGrade) and grade.name == "Night"


def test_every_mood_but_default_has_a_grade_and_a_usable_tint() -> None:
    for name in option_fields.COLOUR_MOODS[1:]:
        grade = grade_for(name)
        assert grade is not None, name
        assert all(0.0 <= c <= 1.0 for c in grade.tint), name
        assert 0.0 <= grade.tint_alpha <= 1.0, name


def test_winters_tint_above_one_is_clamped() -> None:
    assert colour_mood.MOOD_PARAMS["Winter"].tint_color[2] > 1.0
    assert grade_for("Winter").tint[2] == 1.0


def test_a_grade_is_relative_to_default() -> None:
    """A mood at Default's own brightness with white tints would change nothing."""
    default = colour_mood.MOOD_PARAMS["Default"]
    assert grade_for("Murky").multiply[2] == pytest.approx(colour_mood.MOOD_PARAMS["Murky"].brightness / default.brightness)


def test_a_grade_is_frozen() -> None:
    with pytest.raises(AttributeError):
        grade_for("Night").name = "Day"


@pytest.mark.parametrize(
    ("default", "night"), [(DEFAULT_GRASS, NIGHT_GRASS), (DEFAULT_WATER, NIGHT_WATER)], ids=["grass", "water"]
)
def test_the_gh165_night_pair(default, night) -> None:
    """The model against the game: Default's pixels graded land on Night's."""
    got = apply(grade_for("Night"), default)
    assert all(abs(a - b) <= 6 for a, b in zip(got, night, strict=True)), got


def test_apply_saturates_the_multiply_before_the_tint() -> None:
    grade = MoodGrade("probe", (2.0, 2.0, 2.0), (0.0, 0.0, 0.0), 0.5)
    assert apply(grade, (200, 200, 200)) == (128, 128, 128)


# -- the Qt paint ------------------------------------------------------------------

SWATCHES = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "grey 35": (35, 35, 35),
    "grass": (127, 139, 66),
    "water": (53, 100, 135),
    "sand": (214, 188, 128),
    "red": (200, 30, 20),
    "blue": (20, 40, 210),
}


@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("name", option_fields.COLOUR_MOODS[1:])
def test_the_qt_paint_matches_apply(name) -> None:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QImage, QPainter

    from descape.viewer_canvas import paint_colour_grade

    grade = grade_for(name)
    image = QImage(len(SWATCHES), 1, QImage.Format_RGB32)
    for x, rgb in enumerate(SWATCHES.values()):
        image.setPixelColor(x, 0, QColor(*rgb))
    painter = QPainter(image)
    paint_colour_grade(painter, QRectF(0, 0, image.width(), 1), grade)
    assert painter.compositionMode() == QPainter.CompositionMode_SourceOver, "the painter's state leaked"
    painter.end()
    for x, (swatch, rgb) in enumerate(SWATCHES.items()):
        pixel = image.pixelColor(x, 0)
        got = (pixel.red(), pixel.green(), pixel.blue())
        want = apply(grade, rgb)
        assert all(abs(a - b) <= 2 for a, b in zip(got, want, strict=True)), (swatch, got, want)
