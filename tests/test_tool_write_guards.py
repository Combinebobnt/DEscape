"""Every tool that writes a scenario file without write_scenario() refuses a
path under a Proton compatdata/ prefix, as written or resolved through a
symlink (the AGENTS.md hard rule). Each case aims through a directory
symlink whose own path lacks the marker, so only the resolved check holds.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH, FORBIDDEN_WRITE_MARKER
from descape.scenario_write import WriteBlockedError

import conftest


def _linked_proton_dir(tmp_path: Path) -> tuple[Path, Path]:
    real = tmp_path / FORBIDDEN_WRITE_MARKER / "813780"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert FORBIDDEN_WRITE_MARKER not in str(link)  # else this passes without the resolve
    return real, link


@pytest.mark.parametrize("tool", ["gen_v159_fixture", "gen_units_fixture", "gen_trigger_fixture"])
def test_a_fixture_generator_refuses_a_linked_proton_folder(tmp_path, monkeypatch, tool) -> None:
    real, link = _linked_proton_dir(tmp_path)
    module = conftest.load_verify_module(tool)
    monkeypatch.setattr(sys, "argv", [tool, "--out", str(link / "sub" / "f.aoe2scenario")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []  # neither the file nor its parent's mkdir


def test_gen_blank_maps_refuses_a_linked_proton_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_proton_dir(tmp_path)
    module = conftest.load_verify_module("gen_blank_maps")
    monkeypatch.setattr(sys, "argv", ["gen_blank_maps", "--out", str(link / "out"), "--sizes", "120"])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


def test_gen_resize_pair_refuses_a_linked_proton_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_proton_dir(tmp_path)
    module = conftest.load_verify_module("gen_resize_pair")
    monkeypatch.setattr(
        sys, "argv", ["gen_resize_pair", "--source", str(BLANK_TEMPLATE_PATH), "--out", str(link / "out")]
    )

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


def test_gen_blank_maps_refuses_a_linked_output_file(tmp_path, monkeypatch) -> None:
    """--out itself is clean; only the per-size file inside it links into the prefix."""
    real, _link = _linked_proton_dir(tmp_path)
    target = real / "x.aoe2scenario"
    target.write_bytes(b"keep")
    out = tmp_path / "out"
    out.mkdir()
    (out / "blank_120x120.aoe2scenario").symlink_to(target)
    module = conftest.load_verify_module("gen_blank_maps")
    monkeypatch.setattr(sys, "argv", ["gen_blank_maps", "--out", str(out), "--sizes", "120"])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert target.read_bytes() == b"keep"


def test_strip_units_refuses_a_file_in_a_linked_proton_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_proton_dir(tmp_path)
    shutil.copyfile(BLANK_TEMPLATE_PATH, real / "map.aoe2scenario")
    before = (real / "map.aoe2scenario").read_bytes()
    module = conftest.load_verify_module("strip_units")
    monkeypatch.setattr(sys, "argv", ["strip_units", str(link / "map.aoe2scenario")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert (real / "map.aoe2scenario").read_bytes() == before
    assert [p.name for p in real.iterdir()] == ["map.aoe2scenario"]
