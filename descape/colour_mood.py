"""The map's colour moods and their draw-only preview (TASK-238). Qt-free.

MOOD_PARAMS is the install's resources/_common/terrain/colorcorrection_json/
colorcorrection.json, cut down to the four values the preview reads, in the
file's order: that order is the in-game Map > Colour Mood list and the index
Change Color Mood stores. A test re-measures it against the install.

A grade is relative to Default: the textures already look like Default, so
c' = (1 - f) * c * (b / b_default) * t + f * 255 * tint, for brightness b,
terrain_tint_color t, tint_factor f and tint_color tint. Fitted on GH #165's
Default and Night screenshots of one map, within 4.6 per channel. Saturation,
contrast, bloom, fog, vignette and shadows are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

Rgb = tuple[float, float, float]


class MoodParams(NamedTuple):
    brightness: float
    terrain_tint_color: Rgb
    tint_color: Rgb
    tint_factor: float


MOOD_PARAMS: dict[str, MoodParams] = {
    "Default": MoodParams(0.95, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.0),
    "Autumn": MoodParams(1.0, (1.0, 0.95, 0.95), (0.25, 0.0, 0.125), 0.1),
    "Winter": MoodParams(0.97, (1.0, 1.0, 1.0), (0.0, 0.25, 1.5), 0.05),
    "Jungle": MoodParams(1.0, (1.0, 1.025, 1.0), (0.0, 0.125, 0.0), 0.1),
    "Desert": MoodParams(1.0, (1.0, 1.0, 1.0), (0.5, 0.25, 0.25), 0.1),
    "Night": MoodParams(0.75, (0.75, 0.75, 1.0), (0.25, 0.5, 1.0), 0.25),
    "Evening": MoodParams(0.85, (0.85, 0.85, 0.9), (0.25, 0.35, 0.75), 0.15),
    "Spring": MoodParams(0.93, (1.02, 1.02, 1.0), (0.1, 0.15, 0.0), 0.05),
    "Savannah": MoodParams(1.0, (1.05, 1.0, 0.95), (0.2, 0.1, 0.0), 0.15),
    "Arctic": MoodParams(1.025, (1.0, 1.0, 1.0), (0.0, 0.15, 0.75), 0.1),
    "Rainforest": MoodParams(0.95, (0.95, 1.05, 0.95), (0.12, 0.32, 0.17), 0.0475),
    "Swamp": MoodParams(0.92, (1.0, 1.05, 0.9), (0.2, 0.25, 0.2), 0.2),
    "Steppes": MoodParams(1.0, (1.05, 1.0, 0.95), (0.25, 0.15, 0.05), 0.1),
    "Misty": MoodParams(0.9, (0.95, 0.95, 1.0), (0.8, 0.8, 0.9), 0.1),
    "Summer": MoodParams(1.02, (1.02, 1.0, 0.98), (0.15, 0.1, 0.05), 0.08),
    "Murky": MoodParams(0.95, (0.95, 0.95, 1.0), (0.7, 0.7, 0.8), 0.1),
    "Brumous": MoodParams(0.96, (0.975, 0.975, 1.0), (0.35, 0.475, 1.15), 0.075),
    "Twilight": MoodParams(0.85, (0.8, 0.8, 0.9), (0.35, 0.3, 0.7), 0.3),
    "Darkness": MoodParams(0.75, (0.75, 0.75, 1.0), (0.25, 0.5, 1.0), 0.25),
}

_NEUTRAL = "Default"


@dataclass(frozen=True)
class MoodGrade:
    """One mood as paint operations: multiply per channel, then `tint` (each
    0 to 1) over the result at `tint_alpha`."""

    name: str
    multiply: Rgb
    tint: Rgb
    tint_alpha: float


def _grade(name: str, params: MoodParams) -> MoodGrade:
    gain = params.brightness / MOOD_PARAMS[_NEUTRAL].brightness
    return MoodGrade(
        name,
        (gain * params.terrain_tint_color[0], gain * params.terrain_tint_color[1], gain * params.terrain_tint_color[2]),
        (min(params.tint_color[0], 1.0), min(params.tint_color[1], 1.0), min(params.tint_color[2], 1.0)),
        params.tint_factor,
    )


_GRADES = {name.lower(): _grade(name, params) for name, params in MOOD_PARAMS.items() if name != _NEUTRAL}


def grade_for(stored: str) -> MoodGrade | None:
    """The grade for a stored mood name, any case. None draws the map
    unchanged: Default, Empty, empty, and any name the list lacks (a localized
    Default or Empty is stored in the saving client's language)."""
    return _GRADES.get(stored.lower())


def apply(grade: MoodGrade, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """`rgb` under `grade`, the oracle for what the Qt paint produces: the
    multiply saturates at 255 before the tint goes over it, as a fill does."""
    alpha = grade.tint_alpha
    return tuple(
        round((1 - alpha) * min(channel * gain, 255.0) + alpha * 255.0 * tint)
        for channel, gain, tint in zip(rgb, grade.multiply, grade.tint, strict=True)
    )
