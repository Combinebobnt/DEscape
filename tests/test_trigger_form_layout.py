"""GH #387: trigger_form_layout's cell placement and Compact split, no window.

Each case is a type the user captured in the game's own trigger editor, keyed
as the property form keys its rows, in the form's row order.
"""

from __future__ import annotations

import pytest

from descape import trigger_form_layout as tfl

LIST_TYPE = tfl.list_type_key("object_list_unit_id")


def _game(keys) -> dict[str, tuple[int, int]]:
    """Each key's (column, row) in the game layout."""
    return {key: cell[:2] for key, cell in tfl.place(keys, tfl.GAME_SLOTS, tfl.GAME_ROWS).items()}


def test_create_object_lands_in_the_games_cells() -> None:
    keys = [
        tfl.HINT, LIST_TYPE, "object_list_unit_id", "source_player", "location_x", "location_y",
        tfl.group_key("location"), "item_id", "facet", tfl.FACET_PREVIEW, "disable_sound",
    ]
    cells = _game(keys)
    assert cells[LIST_TYPE] == (1, 2)
    assert cells["object_list_unit_id"] == (1, 3)
    assert cells["source_player"] == (2, 1)
    assert cells["facet"] == (2, 2)
    assert cells["item_id"] == (2, 3)
    assert cells["disable_sound"] == (5, 1), "the empty game column 5 closes up"
    # The coordinates share their Set row's cell, grown up from (3, 3) through the free cells above.
    full = tfl.place(keys, tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert full["location_x"] == full["location_y"] == full[tfl.group_key("location")] == (3, 1, 3, 1)
    # Unslotted rows take the first free cells, column by column: the tip the game's type cell.
    assert cells[tfl.HINT] == (1, 1)
    assert cells[tfl.FACET_PREVIEW] == (4, 1)


def test_task_object_lands_in_the_games_cells() -> None:
    keys = [
        LIST_TYPE, "object_list_unit_id", "object_group", "object_type", "source_player", "location_x",
        "location_y", tfl.group_key("location"), "location_object_reference", tfl.AREA_GRID, tfl.group_key("area"),
        "action_type", "selected_object_ids", tfl.group_key("objects"), "disable_garrison_unload_sound",
        "max_units_affected", "issue_group_command", "queue_action",
    ]
    cells = _game(keys)
    expected = {
        "object_group": (2, 2), "object_type": (2, 3), tfl.group_key("objects"): (3, 1), "selected_object_ids": (3, 1),
        tfl.group_key("area"): (3, 2), tfl.AREA_GRID: (3, 2), tfl.group_key("location"): (3, 3),
        "issue_group_command": (4, 1), "action_type": (4, 2), "disable_garrison_unload_sound": (4, 3),
        "queue_action": (5, 1),
    }
    assert {key: cells[key] for key in expected} == expected


def test_objects_in_area_and_change_object_armor() -> None:
    cells = _game(["quantity", "object_list", "object_group", "object_type", "source_player", tfl.AREA_GRID,
                   tfl.group_key("area"), "inverted", "object_state", "include_changeable_weapon_objects"])
    assert cells["inverted"] == (3, 3)
    full = tfl.place(["source_player", tfl.AREA_GRID, tfl.group_key("area"), "inverted"], tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert full[tfl.AREA_GRID] == full[tfl.group_key("area")], "one cell"
    assert full[tfl.AREA_GRID][1:] == (1, 2, 1), "the area starts at the top"
    assert (cells["object_state"], cells["quantity"], cells["include_changeable_weapon_objects"]) == ((4, 1), (4, 2), (4, 3))
    cells = _game(["armour_attack_quantity", "armour_attack_class", LIST_TYPE, "object_list_unit_id", "object_group",
                   "object_type", "source_player", tfl.AREA_GRID, tfl.group_key("area"), "operation",
                   "selected_object_ids", tfl.group_key("objects"), "max_units_affected"])
    assert (cells[tfl.group_key("objects")], cells[tfl.group_key("area")], cells["operation"]) == ((3, 1), (3, 2), (3, 3))
    # The game's (4, 2) and (4, 3), moved up into the free (4, 1).
    assert (cells["armour_attack_quantity"], cells["armour_attack_class"]) == ((4, 1), (4, 2))


def test_display_instructions_message_spans_two_columns_and_rows() -> None:
    keys = [LIST_TYPE, "object_list_unit_id", "source_player", "string_id", "display_time",
            "instruction_panel_position", "play_sound", "message", "sound_name", "use_tag_color_for_icon"]
    cells = tfl.place(keys, tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert cells["message"] == (3, 2, 2, 2)
    assert cells["string_id"][:2] == (3, 1) and cells["sound_name"][:2] == (4, 1)
    assert cells["display_time"][:2] == (2, 3) and cells["instruction_panel_position"][:2] == (2, 2)
    assert cells["play_sound"][:2] == (5, 1) and cells["use_tag_color_for_icon"][:2] == (6, 1)
    preview = tfl.place([*keys, tfl.INSTRUCTION_PREVIEW], tfl.GAME_SLOTS, tfl.GAME_ROWS)[tfl.INSTRUCTION_PREVIEW]
    assert preview[:2] == (5, 2), "the game's Object Icon cell"



def test_modify_attributes_one_line_name_stays_in_one_cell() -> None:
    """Quantity and Operation hold two of the cells the message span would cover."""
    keys = ["quantity", "object_list_unit_id", "source_player", "item_id", "operation", "object_attributes", "message"]
    cells = tfl.place(keys, tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert cells["message"][2:] == (1, 1)
    # The game's (3, 2), (3, 3) and (4, 2), each moved up into the free cell above.
    assert cells["message"][:2] == (3, 1) and cells["operation"][:2] == (3, 2) and cells["quantity"][:2] == (4, 1)


def test_change_variable_has_its_own_row() -> None:
    slots = {**tfl.GAME_SLOTS, **tfl.TYPE_SLOTS["change_variable"]}
    cells = tfl.place(["quantity", "operation", "variable", "message"], slots, tfl.GAME_ROWS)
    assert [cells[key][:2] for key in ("variable", "message", "operation", "quantity")] == [(1, 1), (2, 1), (3, 1), (4, 1)]


def test_a_sparse_form_packs_up_and_left() -> None:
    """Timer: the game's (3, 3) Inverse Condition comes up beside the timer, not two columns and rows away."""
    cells = _game(["timer", "inverted"])
    assert cells == {"timer": (1, 1), "inverted": (2, 1)}
    assert _game(["trigger_id"]) == {"trigger_id": (1, 1)}


def test_the_trigger_row() -> None:
    keys = ["name", "short_description", "short_description_string_table_id", "description",
            "description_string_table_id", "enabled", "looping", "header", "description_order",
            "display_as_objective", "display_on_screen", "mute_objectives", "execute_on_load"]
    cells = tfl.place(keys, tfl.TRIGGER_SLOTS, tfl.TRIGGER_ROWS)
    assert cells == {key: tfl.TRIGGER_SLOTS[key] for key in keys}
    assert cells["description"] == (2, 1, 2, 1) and cells["short_description"] == (3, 1, 2, 1)


def test_a_collision_moves_the_later_key_to_the_first_free_cell() -> None:
    """Object Group and Object Attributes share the game's cell but never one type; if they did, both stay."""
    cells = _game(["source_player", "object_group", "object_attributes"])
    assert cells["object_group"] == (2, 2)
    assert cells["object_attributes"] == (1, 1)


def test_an_unknown_key_fills_the_first_free_cell_and_then_new_columns() -> None:
    slotted = [key for key, slot in tfl.GAME_SLOTS.items() if slot[2:] == (1, 1)]
    cells = _game(["mystery", *slotted])
    assert cells["mystery"] == (1, 1), "unslotted keys wait for the slotted ones"
    full = tfl.place([f"extra{n}" for n in range(20)], {}, tfl.GAME_ROWS)
    assert full["extra0"][:2] == (1, 1) and full["extra3"][:2] == (2, 1)
    assert max(cell[0] for cell in full.values()) == 7, "past the game's six columns"


def test_the_button_grid_takes_a_whole_free_column() -> None:
    cells = tfl.place(["source_player", tfl.BUTTON_GRID], tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert cells[tfl.BUTTON_GRID] == (1, 1, tfl.GAME_ROWS, 1)
    cells = tfl.place([LIST_TYPE, tfl.BUTTON_GRID], tfl.GAME_SLOTS, tfl.GAME_ROWS)
    assert cells[tfl.BUTTON_GRID] == (2, 1, tfl.GAME_ROWS, 1)


def test_no_two_keys_overlap_unless_grouped() -> None:
    keys = [*tfl.GAME_SLOTS, "mystery", tfl.BUTTON_GRID, tfl.HINT]
    covered: dict[tuple[int, int], set[str]] = {}
    for key, (column, row, rowspan, colspan) in tfl.place(keys, tfl.GAME_SLOTS, tfl.GAME_ROWS).items():
        for c in range(column, column + colspan):
            for r in range(row, row + rowspan):
                covered.setdefault((c, r), set()).add(tfl.CELL_GROUPS.get(key, key))
                assert 1 <= r <= tfl.GAME_ROWS
    assert all(len(owners) == 1 for owners in covered.values())


@pytest.mark.parametrize(
    ("heights", "available", "expected"),
    [
        ([30, 30, 30, 30], 100, [0, 0, 0, 1]),
        ([30, 30, 30, 30], 60, [0, 0, 1, 1]),
        ([150, 30], 100, [0, 1]),  # a block taller than the column still gets one
        ([30, 150, 30], 100, [0, 1, 2]),
        ([], 100, []),
    ],
)
def test_flow_splits_by_height(heights, available, expected) -> None:
    assert tfl.flow(heights, available) == expected

