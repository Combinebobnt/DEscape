"""Drift guard for config.example.yaml -- it's generated from
descape/settings.py by tools/gen_config_example.py, not hand-maintained,
after the example file was found to have gone stale (missing dark_mode/
window_size, missing 16 of 22 keybind entries, one dead tool_pencil key
left over from before the tools list was restructured). This test is the
--check equivalent, run as part of the default tier so a settings.py
change that isn't matched here fails the normal test run.
"""

from __future__ import annotations

from pathlib import Path

import conftest

ROOT = Path(__file__).resolve().parent.parent


def test_config_example_matches_generator() -> None:
    gen = conftest.load_verify_module("gen_config_example")
    on_disk = (ROOT / "config.example.yaml").read_text()
    assert on_disk == gen.render(), "config.example.yaml is stale -- run tools/gen_config_example.py"
