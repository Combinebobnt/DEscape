"""Pure coverage for descape/ruler.py: no Qt, no AoE2ScenarioParser, no
QApplication. The golden readouts below are hand-computed rather than
recomputed from the formula under test, matching tests/test_brush.py's
pin-the-shape style, so a change to the distance formula or the label format
fails loudly here instead of agreeing with itself.
"""

from __future__ import annotations

import pytest

from descape.ruler import (
    STATE_DONE,
    STATE_IDLE,
    STATE_PENDING,
    RulerSession,
    format_measurement,
    measure,
)

# (a, b, expected label). Distances hand-computed: the 3-4-5 triangle, three
# exact axis-aligned runs, sqrt(208) = 14.4222 for the 12-by-8 spans, and
# sqrt(2) = 1.41421 for the single diagonal step.
_GOLDEN = [
    ((0, 0), (3, 4), "5.0 tiles  (dx +3, dy +4)"),
    ((3, 4), (15, 12), "14.4 tiles  (dx +12, dy +8)"),
    ((20, 20), (8, 12), "14.4 tiles  (dx -12, dy -8)"),
    ((5, 5), (17, 5), "12.0 tiles  (dx +12, dy +0)"),
    ((10, 10), (10, 3), "7.0 tiles  (dx +0, dy -7)"),
    ((0, 0), (1, 1), "1.4 tiles  (dx +1, dy +1)"),
    ((7, 7), (7, 7), "0.0 tiles  (dx +0, dy +0)"),
]


@pytest.mark.parametrize(("a", "b", "expected"), _GOLDEN)
def test_golden_readouts(a, b, expected) -> None:
    assert format_measurement(measure(a, b)) == expected


def test_zero_components_still_carry_a_sign() -> None:
    """Pinned because "+0" is the odd-looking half of the always-signed
    decision: an unsigned 0 beside a signed -7 would read as two different
    kinds of number."""
    assert "dx +0" in format_measurement(measure((10, 10), (10, 3)))


def test_reversing_the_endpoints_negates_the_components_but_not_the_distance() -> None:
    forward = measure((3, 4), (15, 12))
    backward = measure((15, 12), (3, 4))
    assert backward.dx == -forward.dx
    assert backward.dy == -forward.dy
    assert backward.distance == forward.distance


def test_distance_ignores_the_order_of_magnitude_of_the_map() -> None:
    """A 480x480 corner-to-corner span is the largest real input; pinned so a
    future int/float change can't quietly truncate it."""
    assert measure((0, 0), (479, 479)).distance == pytest.approx(677.4, abs=0.05)


def test_a_fresh_session_has_nothing_to_draw() -> None:
    session = RulerSession()
    assert session.state == STATE_IDLE
    assert session.endpoints is None
    assert session.measurement is None


def test_press_places_both_points_on_the_same_tile() -> None:
    session = RulerSession()
    assert session.press((4, 6)) == STATE_PENDING
    assert session.endpoints == ((4, 6), (4, 6))
    assert session.measurement.distance == 0.0


def test_move_tracks_the_second_point_while_pending() -> None:
    session = RulerSession()
    session.press((4, 6))
    assert session.move((9, 6)) is True
    assert session.endpoints == ((4, 6), (9, 6))


def test_move_reports_no_change_when_the_tile_is_unchanged() -> None:
    """The caller uses this to skip rebuilding scene items on the many
    mouse-move events that land on the tile the line already ends at."""
    session = RulerSession()
    session.press((4, 6))
    session.move((9, 6))
    assert session.move((9, 6)) is False


def test_move_is_ignored_outside_pending() -> None:
    session = RulerSession()
    assert session.move((9, 6)) is False
    assert session.endpoints is None
    session.press((4, 6))
    session.move((9, 6))
    session.release()
    assert session.state == STATE_DONE
    assert session.move((2, 2)) is False
    assert session.endpoints == ((4, 6), (9, 6))


def test_release_inside_the_start_tile_stays_pending() -> None:
    """The whole reason click-move-click works. A machine that completed on
    any release would pass every drag test in this file and still strand the
    user after a plain click."""
    session = RulerSession()
    session.press((4, 6))
    assert session.release() == STATE_PENDING
    assert session.endpoints == ((4, 6), (4, 6))


def test_release_after_a_real_drag_completes() -> None:
    session = RulerSession()
    session.press((4, 6))
    session.move((9, 10))
    assert session.release() == STATE_DONE


def test_click_move_click_completes_on_the_second_press() -> None:
    session = RulerSession()
    session.press((4, 6))
    session.release()
    session.move((9, 10))
    assert session.press((9, 10)) == STATE_DONE
    assert session.endpoints == ((4, 6), (9, 10))


def test_a_deliberate_second_press_on_the_start_tile_completes_at_zero() -> None:
    """Distinct from the release case above: a release there is ambiguous, but
    an explicit second click is not, and must never leave the tool stuck."""
    session = RulerSession()
    session.press((4, 6))
    session.release()
    assert session.press((4, 6)) == STATE_DONE
    assert session.measurement.distance == 0.0


def test_pressing_again_after_a_finished_measurement_starts_a_new_one() -> None:
    session = RulerSession()
    session.press((4, 6))
    session.move((9, 10))
    session.release()
    assert session.press((1, 1)) == STATE_PENDING
    assert session.endpoints == ((1, 1), (1, 1))


def test_clear_returns_to_idle_from_every_state() -> None:
    for finish in (False, True):
        session = RulerSession()
        session.press((4, 6))
        session.move((9, 10))
        if finish:
            session.release()
        session.clear()
        assert session.state == STATE_IDLE
        assert session.endpoints is None
