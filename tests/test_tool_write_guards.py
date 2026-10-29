"""Every tool that writes a scenario file without write_scenario() refuses a
path in a subscribed-mods (mods/subscribed/) folder, as written or resolved
through a symlink (the AGENTS.md hard rule). Each case aims through a
directory symlink whose own path lacks those parts, so only the resolved
check holds. A Proton compatdata/ prefix is writable (the game profile).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from descape.scenario_io import BLANK_TEMPLATE_PATH, names_workshop_subscribed
from descape.scenario_write import WriteBlockedError

import conftest


def _linked_subscribed_dir(tmp_path: Path) -> tuple[Path, Path]:
    real = tmp_path / "mods" / "subscribed" / "12345"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    assert not names_workshop_subscribed(link)  # else this passes without the resolve
    return real, link


@pytest.mark.parametrize("tool", ["gen_v159_fixture", "gen_units_fixture", "gen_trigger_fixture", "gen_blank_template"])
@conftest.requires_symlinks
def test_a_fixture_generator_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch, tool) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module(tool)
    monkeypatch.setattr(sys, "argv", [tool, "--out", str(link / "sub" / "f.aoe2scenario")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []  # neither the file nor its parent's mkdir


@conftest.requires_symlinks
def test_gen_test_map_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch) -> None:
    """--no-bisect: without it main() also writes the bisect set into build/, whatever --out says."""
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module("gen_test_map")
    out = link / "sub" / "f.aoe2scenario"
    monkeypatch.setattr(sys, "argv", ["gen_test_map", "--no-bisect", "--out", str(out)])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []
    with pytest.raises(WriteBlockedError):
        module._write(out, b"x")  # the per-file guard behind main()'s early refusal
    assert list(real.iterdir()) == []


@conftest.requires_symlinks
def test_gen_blank_maps_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module("gen_blank_maps")
    monkeypatch.setattr(sys, "argv", ["gen_blank_maps", "--out", str(link / "out"), "--sizes", "120"])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


@conftest.requires_symlinks
def test_gen_resize_pair_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module("gen_resize_pair")
    monkeypatch.setattr(
        sys, "argv", ["gen_resize_pair", "--source", str(BLANK_TEMPLATE_PATH), "--out", str(link / "out")]
    )

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


def test_a_fixture_generator_writes_into_a_compatdata_folder(tmp_path, monkeypatch) -> None:
    profile = tmp_path / "compatdata" / "813780"
    module = conftest.load_verify_module("gen_units_fixture")
    monkeypatch.setattr(sys, "argv", ["gen_units_fixture", "--out", str(profile / "f.aoe2scenario")])

    module.main()  # raises on a refusal or a failed read-back
    assert [p.name for p in profile.iterdir()] == ["f.aoe2scenario"]


@pytest.mark.parametrize(
    "tool", ["gen_building_place_pair", "gen_gh206_pair", "gen_gh187_pair", "gen_ship_sails_pair"]
)
@conftest.requires_symlinks
def test_a_building_pair_generator_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch, tool) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module(tool)
    monkeypatch.setattr(sys, "argv", [tool, "--source", str(BLANK_TEMPLATE_PATH), "--out", str(link / "out")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


@conftest.requires_symlinks
def test_gen_swap_players_probe_make_refuses_a_linked_subscribed_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    module = conftest.load_verify_module("gen_swap_players_probe")
    monkeypatch.setattr(sys, "argv", ["gen_swap_players_probe", "make", "--out", str(link / "out")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert list(real.iterdir()) == []


@conftest.requires_symlinks
def test_gen_blank_maps_refuses_a_linked_output_file(tmp_path, monkeypatch) -> None:
    """--out itself is clean; only the per-size file inside it links into mods/subscribed/."""
    real, _link = _linked_subscribed_dir(tmp_path)
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


@conftest.requires_symlinks
def test_strip_units_refuses_a_file_in_a_linked_subscribed_folder(tmp_path, monkeypatch) -> None:
    real, link = _linked_subscribed_dir(tmp_path)
    shutil.copyfile(BLANK_TEMPLATE_PATH, real / "map.aoe2scenario")
    before = (real / "map.aoe2scenario").read_bytes()
    module = conftest.load_verify_module("strip_units")
    monkeypatch.setattr(sys, "argv", ["strip_units", str(link / "map.aoe2scenario")])

    with pytest.raises(WriteBlockedError):
        module.main()
    assert (real / "map.aoe2scenario").read_bytes() == before
    assert [p.name for p in real.iterdir()] == ["map.aoe2scenario"]
