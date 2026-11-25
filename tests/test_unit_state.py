"""Which Unit States draw faded (descape/unit_state.py), and the mark colour they fade to."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

from descape import unit_fields, unit_state
from testkit.fakes import SyntheticUnit

import conftest


@dataclass
class _Unit:
    status: int


@pytest.mark.parametrize("status", range(9))
def test_every_labelled_state_but_alive_is_ghosted(status: int) -> None:
    assert unit_state.is_ghosted(_Unit(status)) == (status != unit_state.ALIVE)


def test_a_unit_with_no_status_attribute_reads_as_alive() -> None:
    assert not hasattr(SyntheticUnit(1.5, 1.5, 4), "status")
    assert not unit_state.is_ghosted(SyntheticUnit(1.5, 1.5, 4))


@pytest.mark.parametrize("status", [9, 42, 255, -1])
def test_an_unlabelled_raw_byte_draws_normally(status: int) -> None:
    assert not unit_state.is_ghosted(_Unit(status))


def test_the_ghosted_set_is_every_labelled_choice_but_alive() -> None:
    """A label added to UNIT_STATE_CHOICES must be decided here too."""
    labelled = {value for value, _label in unit_fields.UNIT_STATE_CHOICES}
    assert dict(unit_fields.UNIT_STATE_CHOICES)[unit_state.ALIVE] == "Alive"
    assert labelled - {unit_state.ALIVE} == unit_state.GHOSTED_STATES


def test_the_ghost_mark_mixes_halfway_to_grey_and_stays_opaque() -> None:
    assert unit_state.ghost_mark((0, 0, 0)) == (64, 64, 64)
    assert unit_state.ghost_mark((128, 128, 128)) == (128, 128, 128)
    assert unit_state.ghost_mark((255, 0, 100)) == (round(191.5), 64, 114)
    assert all(isinstance(c, int) for c in unit_state.ghost_mark((10, 200, 30)))


def test_the_premise_kit_writes_every_state_row_through_the_write_path(tmp_path, monkeypatch) -> None:
    tool = conftest.load_verify_module("gen_unit_state_pair")
    spots = tool.layout()
    assert len(spots) == 9 * 12
    assert [s.state for s in spots[::12]] == list(range(9)), "rows run in state order"
    monkeypatch.setattr(sys, "argv", ["gen_unit_state_pair.py", "--out", str(tmp_path)])
    assert tool.main() == 0
    states = tool._rows(tmp_path / tool.STATES_NAME)
    alive = tool._rows(tmp_path / tool.ALIVE_NAME)
    assert sorted({row[4] for row in states}) == list(range(9))
    assert {row[4] for row in alive} == {unit_state.ALIVE}
    assert [row[:4] for row in states] == [row[:4] for row in alive], "the pair differs in state alone"
