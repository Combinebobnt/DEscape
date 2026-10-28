"""The Qt side of descape/tool_options.py: one Binding per schema option,
tying it to the ViewerWindow's Tool Options row widget.

A binding reads the row's session value in canonical form and applies a value
through the widget's own setter (setValue, setChecked, setCurrentIndex,
ViewerWindow._set_last_wall), so the row's handlers run exactly as on a click.
`apply()` returns False, changing nothing, for a value this session's game
data can't show: an id not in the row's list, or a Cliff frame past the
piece's range (setValue would clamp it silently).

A choice or id control's entries are copied from the row's own items, so the
Settings > Tools lists and labels cannot drift from the row's.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QCheckBox, QComboBox, QSpinBox, QWidget

from descape import tool_options

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


@dataclass
class Binding:
    name: str
    read: Callable[[], object]
    apply: Callable[[object], bool]
    widgets: tuple[QWidget, ...]
    # (label, value) pairs for a choice, id or set option, given every option's value
    # (Cliff piece's list depends on the family); None for the other kinds.
    entries: Callable[[dict], list[tuple[str, object]]] | None = field(default=None)


def find_data(combo: QComboBox, value: object) -> int:
    """The index whose data is `value` with the same type, else -1. Not
    QComboBox.findData(), which can't find None and would match True to 1."""
    for index in range(combo.count()):
        data = combo.itemData(index)
        if (data is None and value is None) or (type(data) is type(value) and data == value):
            return index
    return -1


def _spin(name: str, spin: QSpinBox) -> Binding:
    def apply(value) -> bool:
        if not spin.minimum() <= value <= spin.maximum():
            return False
        spin.setValue(value)
        return True

    return Binding(name, spin.value, apply, (spin,))


def _check(name: str, box: QCheckBox) -> Binding:
    def apply(value) -> bool:
        box.setChecked(value)
        return True

    return Binding(name, box.isChecked, apply, (box,))


def _combo(name: str, combo: QComboBox, *, first_is_none: bool = False) -> Binding:
    """`first_is_none`: index 0 reads as None and None applies as index 0, the
    Cliff combos' "first entry", whose data comes from game data."""

    def read():
        if first_is_none and combo.currentIndex() <= 0:
            return None
        return combo.currentData()

    def apply(value) -> bool:
        index = 0 if first_is_none and value is None else find_data(combo, value)
        if index < 0 or index >= combo.count():
            return False
        combo.setCurrentIndex(index)
        return True

    def entries(_values: dict) -> list[tuple[str, object]]:
        return [
            (combo.itemText(i), None if first_is_none and i == 0 else combo.itemData(i)) for i in range(combo.count())
        ]

    return Binding(name, read, apply, (combo,), entries)


def _action_set(name: str, button: QWidget, actions: dict) -> Binding:
    members = tool_options.option(name).members

    def read() -> list:
        return [member for member in members if actions[member].isChecked()]

    def apply(value) -> bool:
        for member, action in actions.items():
            action.setChecked(member in value)
        return True

    def entries(_values: dict) -> list[tuple[str, object]]:
        return [(actions[member].text(), member) for member in members]

    return Binding(name, read, apply, (button,), entries)


def _cliff_pieces(window: ViewerWindow, family: str | None) -> tuple:
    """`family`'s pieces in the row's order; None is the row's first family."""
    if family is None:
        family = window.cliff_family_combo.itemData(0)
    return window._cliff_families.get(family, ())


def cliff_piece_const(window: ViewerWindow, family: str | None, piece: int | None) -> int | None:
    """The unit const a (family, piece) default names, None meaning each one's first entry;
    None when the family has no pieces this session."""
    if piece is not None:
        return piece
    pieces = _cliff_pieces(window, family)
    return pieces[0].unit_const if pieces else None


def _cliff_piece(window: ViewerWindow) -> Binding:
    binding = _combo("cliff_piece", window.cliff_piece_combo, first_is_none=True)

    def entries(values: dict) -> list[tuple[str, object]]:
        pieces = _cliff_pieces(window, values.get("cliff_family"))
        return [(piece.label, None if i == 0 else piece.unit_const) for i, piece in enumerate(pieces)]

    binding.entries = entries
    return binding


def _last_wall(window: ViewerWindow) -> Binding:
    """The wall button's last-used wall: a const from its menu (this dataset's walls)."""

    def listed() -> list:
        return [action for action in window.wall_pick_button.menu().actions() if not action.isSeparator()]

    def apply(value) -> bool:
        if not any(type(action.data()) is type(value) and action.data() == value for action in listed()):
            return False
        window._set_last_wall(value)
        return True

    def entries(_values: dict) -> list[tuple[str, object]]:
        return [(action.text(), action.data()) for action in listed()]

    return Binding("last_wall", lambda: window._last_wall_const, apply, (window.wall_pick_button,), entries)


def build(window: ViewerWindow) -> dict[str, Binding]:
    """One Binding per tool_options.OPTIONS entry, in schema order."""
    made = {
        "brush_size": _spin("brush_size", window.brush_size_spin),
        "brush_shape": _combo("brush_shape", window.brush_shape_combo),
        "brush_thickness": _spin("brush_thickness", window.brush_thickness_spin),
        "auto_beach": _check("auto_beach", window.auto_beach_check),
        "beach_terrain": _combo("beach_terrain", window.beach_combo),
        "beach_width": _spin("beach_width", window.beach_width_spin),
        "paint_trees": _check("paint_trees", window.paint_trees_check),
        "paint_eye_candy": _check("paint_eye_candy", window.paint_eye_candy_check),
        "elevation_level": _spin("elevation_level", window.elevation_level_spin),
        "rect_filled": _combo("rect_filled", window.rect_fill_combo),
        "paste_terrain": _check("paste_terrain", window.paste_terrain_check),
        "paste_elevation": _check("paste_elevation", window.paste_elevation_check),
        "paste_units": _check("paste_units", window.paste_units_check),
        "free_placement": _check("free_placement", window.unrestricted_check),
        "off_grid": _check("off_grid", window.off_grid_check),
        "walls_skip_occupied": _check("walls_skip_occupied", window.skip_occupied_check),
        "gate_walls": _check("gate_walls", window.gate_walls_check),
        "last_wall": _last_wall(window),
        "cliff_family": _combo("cliff_family", window.cliff_family_combo, first_is_none=True),
        "cliff_piece": _cliff_piece(window),
        "cliff_frame": _spin("cliff_frame", window.cliff_frame_spin),
        "unit_brush_density": _spin("unit_brush_density", window.unit_brush_density_spin),
        "unit_brush_random": _check("unit_brush_random", window.unit_brush_random_check),
        "convert_sources": _action_set("convert_sources", window.convert_sources_button, window.convert_source_actions),
        "eraser_targets": _action_set("eraser_targets", window.eraser_targets_button, window.eraser_target_actions),
        "area_edit": _check("area_edit", window.area_edit_check),
        "area_move": _check("area_move", window.area_move_check),
        "area_resize": _check("area_resize", window.area_resize_check),
        "area_tile_borders": _check("area_tile_borders", window.area_tile_borders_check),
    }
    # rect_filled is a bool on a two-item combo: read it as the bool, and the tab shows a checkbox.
    made["rect_filled"].read = window._rect_filled
    made["rect_filled"].entries = None
    return {spec.name: made[spec.name] for spec in tool_options.OPTIONS}
