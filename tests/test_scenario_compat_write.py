"""TASK-057 B2: scenario_compat.write_compat(), the compatibility view's
terrain writer. It patches only the tiles' terrain_id, elevation and layer
bytes in the file's own body, writes every other byte verbatim, and refuses
unless Settings and Units are both trusted. A newer version is simulated as in
test_scenario_compat.py; the written file's c4 is rewritten back to 1.59 so the
normal loader can re-read it."""

from __future__ import annotations

import pytest
from test_scenario_compat import BLANK_V158, UNIT_FIELDS, V159, c4, open_ok, truth_of

from descape import elevation_tools, library_compat, scenario_compat, scenario_io
from descape import format_scan as fs
from descape.scenario_write import WriteBlockedError, WriteResult

SHALLOWS = 4


def edit(loaded) -> set[int]:
    """A 3x3 patch of shallows and one tile raised, through the real models; the changed tile indices."""
    mm = loaded.map_manager
    before = [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain]
    for y in range(4, 7):
        for x in range(4, 7):
            mm.terrain[y * mm.map_width + x].terrain_id = SHALLOWS
    elevation_tools.set_tile_elevation(mm, 20, 30, 1)
    after = [(t.terrain_id, t.elevation, t.layer) for t in mm.terrain]
    return {i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b}


def _tiles(loaded) -> list[tuple[int, int, int]]:
    return [(t.terrain_id, t.elevation, t.layer) for t in loaded.map_manager.terrain]


def _rows(loaded) -> list:
    return [[tuple(getattr(u, f, None) for f in UNIT_FIELDS) for u in units] for units in loaded.unit_manager.units]


def test_write_compat_patches_only_the_edited_tiles(tmp_path) -> None:
    raw = c4(V159.read_bytes())
    proxy = open_ok(raw, "1.59")
    changed = edit(proxy)
    assert len(changed) >= 10
    out = tmp_path / "out.aoe2scenario"
    assert scenario_compat.write_compat(proxy, out, backup=False).wrote
    written, plan = out.read_bytes(), proxy.compat
    assert written[: plan.body.start] == raw[: plan.body.start]
    old, new = plan.body.data, fs.inflate_at(written, plan.body.start).data
    assert len(old) == len(new)
    allowed = {plan.grid.tiles_start + i * plan.grid.stride + k for i in changed for k in (0, 1, 5, 6)}
    diff = {i for i, (a, b) in enumerate(zip(old, new, strict=True)) if a != b}
    assert diff and diff <= allowed
    reread, original = truth_of(c4(written, "1.59")), truth_of(V159.read_bytes())
    assert _tiles(reread) == _tiles(proxy)
    assert _rows(reread) == _rows(original) and sum(map(len, reread.unit_manager.units)) > 0
    assert reread.trigger_tail == original.trigger_tail


def test_write_compat_with_no_edit_writes_the_file_verbatim(tmp_path) -> None:
    raw = c4(V159.read_bytes())
    proxy = open_ok(raw, "1.59")
    out = tmp_path / "out.aoe2scenario"
    assert scenario_compat.write_compat(proxy, out, backup=True) == WriteResult(True, [])
    assert out.read_bytes() == raw
    assert scenario_compat.write_compat(proxy, out, backup=True) == WriteResult(False, [])
    edit(proxy)
    result = scenario_compat.write_compat(proxy, out, backup=True)
    assert result.wrote and result.backups
    assert all(p.read_bytes() == raw for p in result.backups)


def test_write_compat_refuses_the_guarded_destinations(tmp_path) -> None:
    proxy = open_ok(c4(V159.read_bytes()), "1.59")
    edit(proxy)
    subscribed = tmp_path / "mods" / "subscribed" / "x"
    subscribed.mkdir(parents=True)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    for path in (subscribed / "a.aoe2scenario", scenario_io.TEMPLATE_DIR / "a.aoe2scenario", loop / "a.aoe2scenario"):
        with pytest.raises(WriteBlockedError):
            scenario_compat.write_compat(proxy, path, backup=False)
        assert not path.exists()


def test_write_compat_needs_settings_and_units_trusted(tmp_path) -> None:
    units_unknown = open_ok(V159.read_bytes(), "1.58", BLANK_V158.read_bytes())
    raw = (library_compat.VERSIONS_DIR / "v1.56" / "default.aoe2scenario").read_bytes()
    donor = (library_compat.VERSIONS_DIR / "v1.55" / "default.aoe2scenario").read_bytes()
    settings_unknown = open_ok(raw, "1.55", donor)
    out = tmp_path / "out.aoe2scenario"
    for proxy in (units_unknown, settings_unknown):
        assert scenario_compat.terrain_lock_reason(proxy) is not None
        with pytest.raises(WriteBlockedError, match="once DEscape can read"):
            scenario_compat.write_compat(proxy, out, backup=False)
    assert not out.exists()
    with pytest.raises(WriteBlockedError, match="not open in compatibility mode"):
        scenario_compat.write_compat(truth_of(V159.read_bytes()), out, backup=False)


def test_write_compat_refuses_a_tile_list_of_the_wrong_size(tmp_path) -> None:
    proxy = open_ok(c4(V159.read_bytes()), "1.59")
    edit(proxy)
    side = proxy.compat.grid.width - 1  # the library refuses a non-square list
    proxy.map_manager.terrain = proxy.map_manager.terrain[: side * side]
    out = tmp_path / "out.aoe2scenario"
    with pytest.raises(WriteBlockedError, match="refusing to write"):
        scenario_compat.write_compat(proxy, out, backup=False)
    assert not out.exists()
