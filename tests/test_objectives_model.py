"""GH #391: objectives_model's rows, box lines and Description Order rewrites."""

from __future__ import annotations

from types import SimpleNamespace

from descape.objectives_model import (
    HEADER,
    NEW,
    OBJECTIVE,
    OPTIONAL,
    REQUIRED,
    ObjectiveRow,
    box_lines,
    has_string_id,
    insert_order,
    insert_orders,
    moved_by_step,
    objective_rows,
    orders_after_move,
    renumbered,
    split_marker,
)


def _trigger(name, order=0, objective=False, header=False, screen=False, enabled=True, short="", stid=0):
    return SimpleNamespace(
        name=name, trigger_id=0, description_order=order, display_as_objective=int(objective),
        header=int(header), display_on_screen=int(screen), mute_objectives=0, enabled=int(enabled),
        short_description=short, short_description_stid=stid,
    )


def _manager(triggers, display_order=None):
    for index, trigger in enumerate(triggers):
        trigger.trigger_id = index
    return SimpleNamespace(triggers=triggers, trigger_display_order=display_order or list(range(len(triggers))))


def _row(index, order, header=False, screen=True, enabled=True):
    return ObjectiveRow(index, index, f"t{index}", order, header, True, screen, False, enabled, f"s{index}", 0)


def _apply(rows, changes):
    """The list as the game would sort it after `changes`, ties in the given order."""
    slot = {row.index: position for position, row in enumerate(rows)}
    new = [row._replace(order=changes.get(row.index, row.order)) for row in rows]
    return sorted(new, key=lambda row: (-row.order, slot[row.index]))


def test_rows_keep_objectives_headers_and_on_screen_triggers_sorted_by_order():
    manager = _manager([
        _trigger("plain"),
        _trigger("low", 10, objective=True),
        _trigger("head", 100, header=True),
        _trigger("screen only", 50, screen=True),
        _trigger("string", 50, objective=True, stid=4321),
    ])
    rows = objective_rows(manager)
    assert [row.name for row in rows] == ["head", "screen only", "string", "low"]
    assert rows[0].header and not rows[0].objective
    assert rows[2].short_stid == 4321


def test_ties_keep_display_order_not_list_order():
    manager = _manager([_trigger("a", 5, objective=True), _trigger("b", 5, objective=True)], display_order=[1, 0])
    assert [row.name for row in objective_rows(manager)] == ["b", "a"]


def test_string_ids_zero_and_minus_one_mean_unset():
    assert not has_string_id(0)
    assert not has_string_id(-1)
    assert has_string_id(7)


def test_box_lines_show_on_screen_rows_and_at_start_drops_disabled():
    rows = [_row(0, 100, header=True), _row(1, 99), _row(2, 98, enabled=False), _row(3, 97, screen=False)]
    assert [(line.kind, line.row.index) for line in box_lines(rows, at_start=False)] == [
        (HEADER, 0), (OBJECTIVE, 1), (OBJECTIVE, 2),
    ]
    assert [line.row.index for line in box_lines(rows, at_start=True)] == [0, 1]


def test_a_move_into_a_gap_changes_one_row():
    rows = [_row(0, 100), _row(1, 50), _row(2, 10)]
    changes = orders_after_move(rows, {2}, 1)
    assert changes == {2: 75}
    assert [row.index for row in _apply(rows, changes)] == [0, 2, 1]


def test_a_move_with_no_gap_shifts_the_rows_below_down_to_the_next_gap():
    # The corpus pattern: headers at 100 and 50, objectives counted down under each.
    rows = [_row(0, 100, header=True), _row(1, 99), _row(2, 98), _row(3, 97), _row(4, 50, header=True), _row(5, 49)]
    changes = orders_after_move(rows, {3}, 1)
    assert changes == {3: 99, 1: 98, 2: 97}
    assert [row.index for row in _apply(rows, changes)] == [0, 3, 1, 2, 4, 5]
    assert 4 not in changes and 0 not in changes, "the headers keep 100 and 50"


def test_a_move_renumbers_only_when_shifting_would_go_below_zero():
    rows = [_row(0, 2), _row(1, 1), _row(2, 0)]
    changes = orders_after_move(rows, {2}, 1)
    assert [row.index for row in _apply(rows, changes)] == [0, 2, 1]
    assert min(changes.get(row.index, row.order) for row in rows) >= 0


def test_a_move_to_the_top_never_passes_the_order_limit():
    from descape.objectives_model import MAX_ORDER

    rows = [_row(0, MAX_ORDER), _row(1, 5)]
    changes = orders_after_move(rows, {1}, 0)
    assert [row.index for row in _apply(rows, changes)] == [1, 0]
    assert max(changes.values()) <= MAX_ORDER


def test_a_move_to_the_bottom_never_goes_below_zero():
    rows = [_row(0, 1), _row(1, 0)]
    changes = orders_after_move(rows, {0}, 1)
    assert [row.index for row in _apply(rows, changes)] == [1, 0]
    assert all(value >= 0 for value in changes.values())


def test_a_move_to_the_top_goes_above_the_first_row():
    rows = [_row(0, 100), _row(1, 50)]
    assert orders_after_move(rows, {1}, 0) == {1: 101}


def test_a_block_moves_together_in_its_order():
    rows = [_row(0, 100), _row(1, 90), _row(2, 80), _row(3, 10), _row(4, 0)]
    changes = orders_after_move(rows, {3, 4}, 1)
    assert [row.index for row in _apply(rows, changes)] == [0, 3, 4, 1, 2]
    assert set(changes) == {3, 4}


def test_a_move_that_changes_nothing_writes_nothing():
    rows = [_row(0, 100), _row(1, 50)]
    assert orders_after_move(rows, {0}, 0) == {}


def test_up_and_down_steps_and_edges():
    rows = [_row(0, 100), _row(1, 50), _row(2, 10)]
    assert [row.index for row in _apply(rows, moved_by_step(rows, [2], -1))] == [0, 2, 1]
    assert [row.index for row in _apply(rows, moved_by_step(rows, [0], 1))] == [1, 0, 2]
    assert moved_by_step(rows, [0], -1) == {}
    assert moved_by_step(rows, [2], 1) == {}


def test_renumber_counts_down_by_one_keeping_the_top_value():
    rows = [_row(0, 100), _row(1, 100), _row(2, 3), _row(3, 3), _row(4, 0)]
    changes = renumbered(rows)
    assert [changes.get(row.index, row.order) for row in rows] == [100, 99, 98, 97, 96]
    assert 0 not in changes


def test_renumber_raises_a_top_too_small_for_the_list():
    rows = [_row(0, 1), _row(1, 1), _row(2, 1)]
    assert [renumbered(rows).get(row.index, row.order) for row in rows] == [2, 1, 0]


def test_insert_order_uses_the_gap_under_the_selection():
    rows = [_row(0, 100), _row(1, 50)]
    assert insert_order(rows, 0) == {NEW: 75}
    assert insert_order(rows, None) == {NEW: 24}
    assert insert_order([], None) == {NEW: 0}


def test_insert_order_renumbers_when_there_is_no_room():
    rows = [_row(0, 2), _row(1, 1)]
    changes = insert_order(rows, 0)
    assert changes[NEW] < changes.get(0, 2)
    assert changes[NEW] > changes.get(1, 1)


def test_insert_orders_places_several_in_order():
    rows = [_row(0, 100), _row(1, 50)]
    changes = insert_orders(rows, 0, [7, 8])
    assert 100 > changes[7] > changes[8] > 50


def test_a_leading_plus_or_minus_is_the_marker_not_text():
    assert split_marker("+ Destroy the Invaders") == (REQUIRED, "Destroy the Invaders")
    assert split_marker("-destroy camel") == (OPTIONAL, "destroy camel")
    assert split_marker("  <red> - Defend", ["RED"]) == (OPTIONAL, "<red>Defend")
    assert split_marker("Main Objective(s)") == (None, "Main Objective(s)")
    assert split_marker("<b>+ literal", ["RED"]) == (None, "<b>+ literal"), "only colour tags are skipped"
    assert split_marker("Win +5 gold") == (None, "Win +5 gold")


def test_renumber_with_a_selection_counts_only_those_down_from_the_first():
    rows = [_row(0, 100), _row(1, 68), _row(2, 52), _row(3, 36), _row(4, 20), _row(5, 5)]
    changes = renumbered(rows, [1, 2, 3, 4])
    assert changes == {2: 67, 3: 66, 4: 65}
    assert [row.index for row in _apply(rows, changes)] == [0, 1, 2, 3, 4, 5]


def test_renumber_with_a_gapped_selection_keeps_the_rows_between():
    rows = [_row(0, 68), _row(1, 50), _row(2, 40), _row(3, 20)]
    changes = renumbered(rows, [0, 2])
    assert changes == {2: 49}, "the selected row counts down from the unselected one above it"


def test_renumber_with_a_selection_shifts_colliding_rows_below():
    rows = [_row(0, 10), _row(1, 10), _row(2, 10), _row(3, 9), _row(4, 1)]
    changes = renumbered(rows, [0, 1])
    assert changes == {1: 9, 2: 8, 3: 7}, "tied and colliding rows below move down; the first lower one stops it"
    assert [row.index for row in _apply(rows, changes)] == [0, 1, 2, 3, 4]


def test_renumber_with_one_row_selected_renumbers_the_whole_list():
    rows = [_row(0, 100), _row(1, 50), _row(2, 0)]
    assert renumbered(rows, [1]) == renumbered(rows)


def test_the_renumber_fallback_keeps_a_dense_list_where_it_is():
    rows = [_row(index, 10 - index) for index in range(11)]
    changes = orders_after_move(rows, {0}, 1)
    assert changes == {1: 10, 0: 9}
