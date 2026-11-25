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
        assert all(0.0 < m <= 1.0 for m in grade.multiply), name
        assert 0.0 <= grade.tint_alpha <= 1.0, name


def test_winters_tint_above_one_is_clamped() -> None:
    params = colour_mood.MOOD_PARAMS["Winter"]
    assert params.tint_color[2] > 1.0
    grade = grade_for("Winter")
    assert grade.tint_alpha * grade.tint[2] == pytest.approx(params.tint_factor * 1.0)


def test_a_grade_is_relative_to_default() -> None:
    """Default's own values grade to nothing: the textures already look like Default."""
    grade = colour_mood._grade("probe", colour_mood.MOOD_PARAMS["Default"])
    assert grade.multiply == (1.0, 1.0, 1.0) and grade.tint_alpha == 0.0
    assert all(apply(grade, rgb) == rgb for rgb in SWATCHES.values())
    # Murky has Default's brightness and a white terrain tint, so only its tint moves blue.
    murky = colour_mood.MOOD_PARAMS["Murky"]
    assert apply(grade_for("Murky"), (128, 128, 128))[2] == round(0.9 * 128 + murky.tint_factor * 255 * murky.tint_color[2])


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


def _fitted(name: str, channel: int, level: float) -> float:
    """The module docstring's formula, unrounded and unclamped: what a grade approximates."""
    params, default = colour_mood.MOOD_PARAMS[name], colour_mood.MOOD_PARAMS["Default"]
    slope = (1 - params.tint_factor) * params.brightness / default.brightness * params.terrain_tint_color[channel]
    return slope * level + params.tint_factor * 255 * min(params.tint_color[channel], 1.0)


def test_apply_does_not_saturate_before_the_tint() -> None:
    """Desert brightens: its red at 250 lands on the formula's 249.6, not on a
    clamped 255 under the tint (242), and white stays in range."""
    desert = grade_for("Desert")
    assert _fitted("Desert", 0, 250) == pytest.approx(249.6, abs=0.05)
    assert apply(desert, (250, 250, 250))[0] == 250
    assert all(c <= 255 for c in apply(desert, (255, 255, 255)))
    assert apply(MoodGrade("probe", (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.5), (255, 255, 255)) == (255, 255, 255)


# Per mood: does it need the Multiply fill, and its worst error in levels over 40..200 (TASK-557 Measured 5).
FILLS = {
    "Desert": (False, 0.0),
    "Jungle": (False, 1.6),
    "Winter": (False, 4.0),
    "Autumn": (False, 3.2),
    "Spring": (False, 2.5),
    "Arctic": (False, 9.3),
    "Brumous": (False, 1.6),
    "Misty": (False, 2.5),
    "Murky": (False, 2.7),
    "Summer": (False, 3.4),
    "Night": (True, 0.0),
    "Darkness": (True, 0.0),
    "Evening": (True, 0.0),
    "Twilight": (True, 0.0),
    "Swamp": (True, 0.0),
    "Savannah": (True, 0.0),
    "Rainforest": (True, 3.1),
    "Steppes": (True, 4.0),
}


def test_the_fills_table_covers_every_mood() -> None:
    assert set(FILLS) == set(option_fields.COLOUR_MOODS[1:])


@pytest.mark.parametrize("name", list(FILLS))
def test_each_moods_fills_and_error(name) -> None:
    """The census moods (Desert, Jungle, Winter, Autumn) take one fill, Night two."""
    multiply, bound = FILLS[name]
    grade = grade_for(name)
    assert (min(grade.multiply) < 1.0) is multiply
    # apply() rounds, so half a level on top of the bound.
    worst = max(abs(apply(grade, (c, c, c))[ch] - _fitted(name, ch, c)) for c in range(40, 201) for ch in range(3))
    assert worst <= bound + 0.51, worst


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
@pytest.mark.parametrize("fmt", ["Format_RGB32", "Format_ARGB32_Premultiplied"])
@pytest.mark.parametrize("name", option_fields.COLOUR_MOODS[1:])
def test_the_qt_paint_matches_apply(name, fmt) -> None:
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QColor, QImage, QPainter

    from descape.viewer_canvas import paint_colour_grade

    grade = grade_for(name)
    image = QImage(len(SWATCHES), 1, getattr(QImage, fmt))
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


def _painted_modes(grade: MoodGrade) -> list[int]:
    """The composition modes paint_colour_grade() sets for `grade`, in order."""
    from PyQt5.QtCore import QRectF
    from PyQt5.QtGui import QImage, QPainter

    from descape.viewer_canvas import paint_colour_grade

    class Spy(QPainter):
        def setCompositionMode(self, mode) -> None:
            modes.append(mode)
            super().setCompositionMode(mode)

    modes: list[int] = []
    image = QImage(4, 4, QImage.Format_RGB32)
    image.fill(0x808080)
    painter = Spy(image)
    paint_colour_grade(painter, QRectF(0, 0, 4, 4), grade)
    painter.end()
    return modes


@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize("name", option_fields.COLOUR_MOODS[1:])
def test_no_grade_paints_a_per_pixel_dodge(name) -> None:
    """ColorDodge blends per pixel (TASK-557): every grade is a SourceOver
    tint, after a Multiply only where FILLS says the mood needs one."""
    from PyQt5.QtGui import QPainter

    modes = _painted_modes(grade_for(name))
    assert QPainter.CompositionMode_ColorDodge not in modes
    multiply = [QPainter.CompositionMode_Multiply] if FILLS[name][0] else []
    assert modes == [*multiply, QPainter.CompositionMode_SourceOver]
