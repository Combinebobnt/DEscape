"""tools/bench_gui_drag.py's --flow gh180 bookkeeping, Qt-free: the stall
timing and the `gh180 gates:` line its baseline tables are read from.
The flow itself needs a real X server and runs only by hand."""

from __future__ import annotations

import argparse

import pytest

from tools import bench_gui_drag

# Shaped on a real Sloped run's debug log, timestamps stripped.
_SECTIONS = {
    "load": [
        "Loaded old-allies-final-v2.aoe2scenario (240x240 tiles, 13,099 units, tile_px=64, style=sloped) in 0.80s",
        "perf load: repaint: 1 calls, 565ms total (max 564.7) [composite 497.7 x480; blit 11.2 x32; other 51.5]",
        "First paint composited in 0.56s (total 1.36s, mip 0)",
    ],
    "draw": [
        "perf view: repaint: 9 calls, 1ms total (max 0.2) [blit 0.2 x9; other 0.7], composite native",
        "perf drag paint-terrain: 80 steps, 995ms total, 12.4ms/step (max 39.6), wall 529ms, untimed 0ms, composite native",
        "  | stroke_snapshot 0.0 patch 4.4 pick 2.4 highlight 5.0",
        "perf stall 606ms (in repaint; covered 602 of 606: repaint 503, unit_sources 61)",
        "perf view: repaint: 27 calls, 509ms total (max 503.0) [composite 438.2 x480; blit 12.7 x84; other 53.6]",
    ],
    "undo1": [
        "Undo: Paint terrain",
        "perf op undo: 427ms | history_restore 16.0 patch 306.4 unit_sources 93.3 untimed 0.7",
    ],
    "elev": [
        "perf drag set-elevation: 52 steps, 2400ms total, 46.2ms/step (max 152.7), wall 2018ms, untimed 44ms",
    ],
    "undo2": [
        "Sloped view: 22455 tiles dirty, re-rendered full map instead of patching (prepared in 0.40s)",
        "Undo: Set elevation",
        "perf op undo: 349ms | history_restore 6.7 re-render 330.1 untimed 1.0",
    ],
}


def test_the_gates_line_reads_every_gate_from_its_own_section():
    stalls = {"draw": (103.0, 503.0, 607.0), "undo1": (427.0, 503.0, 931.0), "undo2": (349.0, 487.0, 837.0)}
    line = bench_gui_drag.gh180_gates(_SECTIONS, stalls)
    fields = dict(part.split("=", 1) for part in line.removeprefix("gh180 gates: ").split())
    assert fields == {
        "G5_first_paint": "560", "G5_total": "1360",
        "G1_steps": "80", "G1_ms_step": "12.4", "G1_max": "39.6",
        "G2_release": "103", "G2_next_paint": "503", "G2_total": "607",
        "post_stroke_composite": "438x480",
        "G3_steps": "52", "G3_ms_step": "46.2", "G3_max": "152.7",
        "G4_undo1_op": "427", "G4_undo1_next_paint": "503", "G4_undo1_total": "931", "G4_undo1_path": "patch",
        "G4_undo2_op": "349", "G4_undo2_next_paint": "487", "G4_undo2_total": "837",
        "G4_undo2_path": "rerender:22455",
        "G6_max_step": "152.7",
    }


def test_the_post_stroke_composite_is_the_first_after_the_drag_line_not_before():
    sections = {**_SECTIONS, "draw": [_SECTIONS["draw"][4], *_SECTIONS["draw"][:4]]}
    stalls = dict.fromkeys(("draw", "undo1", "undo2"), (0.0, 0.0, 0.0))
    assert "post_stroke_composite" not in bench_gui_drag.gh180_gates(sections, stalls)


def test_a_stall_runs_from_the_handler_to_the_end_of_the_first_paint_after_it():
    paints = [(0.5, 0.6), (1.05, 1.08), (1.20, 1.70), (2.0, 2.1)]
    handler, paint, total = bench_gui_drag._stall_ms([(0.0, 0.1), (1.0, 1.1)], paints)
    assert (round(handler), round(paint), round(total)) == (100, 500, 700)
    assert bench_gui_drag._stall_ms([(5.0, 5.2)], paints) == pytest.approx((200.0, 0.0, 200.0))
    assert bench_gui_drag._stall_ms([], paints) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(("text", "size"), [("1920x1200", (1920, 1200)), ("800X600", (800, 600))])
def test_window_sizes_parse(text, size):
    assert bench_gui_drag._size(text) == size


@pytest.mark.parametrize("text", ["1920", "1920x", "x1200", "axb"])
def test_bad_window_sizes_are_refused(text):
    with pytest.raises(argparse.ArgumentTypeError):
        bench_gui_drag._size(text)
