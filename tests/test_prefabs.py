"""descape/prefabs.py: the saved-prefab file format, its refusals and the
folder operations. Qt-free; prefab_dir() sits under conftest's per-test
config folder."""

from __future__ import annotations

import dataclasses
import functools
import gzip
import json
import math
import struct
from pathlib import Path

import pytest

from descape import prefabs, region_clipboard
from descape.prefabs import PrefabNameTaken, PrefabRefused, PrefabWriteBlocked
from descape.region_clipboard import RegionBlock, RegionUnit
from descape.scenario_io import load_map_and_units

FIXTURE = Path(__file__).parent / "fixtures" / "units_120x120.aoe2scenario"
_NAN_NEG = struct.unpack("<f", bytes.fromhex("0000c0ff"))[0]  # x86's default NaN, sign bit set


@functools.lru_cache(maxsize=1)
def _copied() -> RegionBlock:
    loaded = load_map_and_units(FIXTURE)
    mm = loaded.map_manager
    block = region_clipboard.copy_region(mm, loaded.unit_manager, 0, 0, mm.map_width, mm.map_height)
    assert block.units, "the fixture should carry units"
    return block


def _unit(**kw) -> RegionUnit:
    base = RegionUnit(1, 83, 0.5, 0.5, 0.0, 0.0, 2, 0, -1, "", -1)
    return dataclasses.replace(base, **kw)


def _f32_bytes(block: RegionBlock) -> list[tuple[bytes, bytes]]:
    return [(struct.pack("<f", u.z), struct.pack("<f", u.rotation)) for u in block.units]


def _without_f32(block: RegionBlock) -> RegionBlock:
    return dataclasses.replace(block, units=tuple(dataclasses.replace(u, z=0.0, rotation=0.0) for u in block.units))


def _assert_round_trip(block: RegionBlock, name: str = "piece") -> None:
    path = prefabs.save_block(block, name)
    loaded = prefabs.load_block(path)
    # NaN != NaN, so z and rotation compare as their stored f32 bytes.
    assert _f32_bytes(loaded) == _f32_bytes(block)
    assert _without_f32(loaded) == _without_f32(block)


def _edge_block() -> RegionBlock:
    copied = _copied()
    units = (
        _unit(unit_const=72, rotation=2 * math.pi * 3 / 5),  # radian-encoded wall index
        _unit(unit_const=88, rotation=7.0),  # the gate sentinel
        _unit(player=0, unit_const=349, rotation=53.0),  # GAIA variant index above 2*pi
        _unit(z=_NAN_NEG, rotation=_NAN_NEG),
        _unit(z=math.inf, rotation=-math.inf),
        _unit(caption_string_id=12, caption_string="Héllo, 世界", capture_flag=1),
        _unit(unit_const=82, dx=5.5, dy=5.5, capture_flag=0),
        _unit(dx=5.5, dy=5.5, garrison_slot=len(copied.units) + 6),
        _unit(player=8, status=0, initial_animation_frame=0xFFFF, caption_string_id=-(2**31)),
    )
    layers = tuple(-1 if i % 2 else 9 for i in range(len(copied.layers)))
    return dataclasses.replace(copied, units=copied.units + units, layers=layers)


def test_a_copied_block_round_trips() -> None:
    _assert_round_trip(_copied())


def test_edge_values_round_trip() -> None:
    _assert_round_trip(_edge_block())


def test_a_wide_units_only_block_and_a_ror_block_round_trip() -> None:
    wide = RegionBlock(600, 3, (), (), (), (_unit(dx=599.5), _unit(dx=0.25, dy=2.75)))
    _assert_round_trip(wide, "wide")
    _assert_round_trip(dataclasses.replace(_copied(), dataset="ror"), "ror")
    _assert_round_trip(dataclasses.replace(wide, legacy=True), "legacy")


def test_session_only_fields_are_never_saved() -> None:
    block = dataclasses.replace(_copied(), cut_doc_id="doc-1", units=(_unit(cut_reference_id=41),))
    path = prefabs.save_block(block, "cut")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header, body = json.loads(f.readline()), json.loads(f.readline())
    assert "cut_doc_id" not in header and "cut_doc_id" not in body
    assert "cut_reference_id" not in body["units"][0]
    loaded = prefabs.load_block(path)
    assert loaded.cut_doc_id == "" and loaded.units[0].cut_reference_id == -1


def test_every_region_field_is_saved_or_declared_session_only() -> None:
    """A new RegionUnit/RegionBlock field fails here until prefabs.py places it."""
    unit_saved = set(prefabs.UNIT_INT_FIELDS) | set(prefabs.UNIT_FLOAT_FIELDS) | set(prefabs.UNIT_F32_FIELDS) | set(prefabs.UNIT_STR_FIELDS)
    assert {f.name for f in dataclasses.fields(RegionUnit)} == unit_saved | prefabs.UNIT_SESSION_ONLY
    block_saved = {"width", "height", "terrain_ids", "elevations", "layers", "units", "dataset", "legacy"}
    assert {f.name for f in dataclasses.fields(RegionBlock)} == block_saved | prefabs.BLOCK_SESSION_ONLY


# -- refusals --------------------------------------------------------------


def _write_raw(path: Path, header, body=None, *, raw_lines: list[bytes] | None = None) -> Path:
    lines = raw_lines or [json.dumps(header).encode()] + ([json.dumps(body).encode()] if body is not None else [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(b"\n".join(lines) + b"\n"))
    return path


def _valid_parts(block: RegionBlock | None = None) -> tuple[dict, dict]:
    path = prefabs.save_block(block or _edge_block(), "source")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        parts = json.loads(f.readline()), json.loads(f.readline())
    prefabs.delete(path)
    return parts


def _refused_both(tmp_path: Path, header, body, match: str) -> None:
    path = _write_raw(tmp_path / "prefabs" / f"bad{prefabs.SUFFIX}", header, body)
    with pytest.raises(PrefabRefused, match=match):
        prefabs.load_block(path)
    (info,) = [i for i in prefabs.list_prefabs() if i.path == path]
    assert info.error is not None


def test_a_file_that_is_not_gzip_or_is_truncated_is_refused(tmp_path) -> None:
    folder = prefabs.prefab_dir()
    folder.mkdir(parents=True)
    (folder / f"text{prefabs.SUFFIX}").write_text("hello")
    good = prefabs.save_block(_copied(), "good")
    truncated = folder / f"cut{prefabs.SUFFIX}"
    truncated.write_bytes(good.read_bytes()[:200])
    for path in (folder / f"text{prefabs.SUFFIX}", truncated):
        with pytest.raises(PrefabRefused):
            prefabs.load_block(path)
    rows = {i.path.name: i for i in prefabs.list_prefabs()}
    assert rows[f"text{prefabs.SUFFIX}"].error is not None
    assert rows[good.name].error is None


def test_oversize_lines_are_refused(tmp_path, monkeypatch) -> None:
    header, body = _valid_parts()
    monkeypatch.setattr(prefabs, "HEADER_MAX_BYTES", 64)
    _refused_both(tmp_path, header, body, "header is over")
    monkeypatch.setattr(prefabs, "HEADER_MAX_BYTES", 256 * 1024)
    monkeypatch.setattr(prefabs, "BODY_MAX_BYTES", 1024)
    path = _write_raw(tmp_path / "prefabs" / f"big{prefabs.SUFFIX}", header, body)
    with pytest.raises(PrefabRefused, match="body is over"):
        prefabs.load_block(path)
    assert prefabs.read_info(path).name == "source"  # the list never reads the body


@pytest.mark.parametrize(
    "lines",
    [
        [b"{not json"],
        [b"\xff\xfe"],
        [b"[1, 2]"],
        [b"[" * 100_000],
        [b'{"format": "descape-prefab", "version": ' + b"9" * 5000 + b"}"],
    ],
    ids=["json", "utf8", "array", "deep", "bigint"],
)
def test_unparseable_headers_are_refused(tmp_path, lines) -> None:
    path = _write_raw(tmp_path / "prefabs" / f"bad{prefabs.SUFFIX}", None, raw_lines=lines)
    with pytest.raises(PrefabRefused):
        prefabs.load_block(path)
    assert prefabs.list_prefabs()[0].error is not None


_HEADER_BREAKS = {
    "format": ("format", "descape-theme", "not a DEscape prefab"),
    "newer": ("version", prefabs.VERSION + 1, "newer"),
    "version-bool": ("version", True, "version"),
    "name": ("name", "", "name"),
    "name-control": ("name", "a\x07b", "name"),
    "name-long": ("name", "x" * 81, "name"),
    "saved": ("saved", 5, "saved"),
    "has_tiles": ("has_tiles", 1, "has_tiles"),
    "legacy": ("legacy", "no", "legacy"),
    "dataset": ("dataset", "aoe3", "game"),
    "width": ("width", 0, "width"),
    "width-big": ("width", 2**31, "width"),
    "name-surrogate": ("name", "a\ud800", "name"),
    "height": ("height", 2.0, "height"),
    "unit_count": ("unit_count", -1, "unit count"),
    "thumb-size": ("thumbnail", {"w": 65, "h": 1, "rgb": ""}, "thumbnail"),
    "thumb-rgb": ("thumbnail", {"w": 1, "h": 1, "rgb": "AAAAAA=="}, "thumbnail"),
    "thumb-b64": ("thumbnail", {"w": 1, "h": 1, "rgb": "!!!!"}, "thumbnail"),
}


@pytest.mark.parametrize("case", sorted(_HEADER_BREAKS))
def test_bad_header_fields_are_refused_and_still_listed(tmp_path, case) -> None:
    header, body = _valid_parts()
    key, value, match = _HEADER_BREAKS[case]
    _refused_both(tmp_path, {**header, key: value}, body, match)


def test_a_missing_header_key_is_refused(tmp_path) -> None:
    header, body = _valid_parts()
    del header["has_tiles"]
    _refused_both(tmp_path, header, body, "has_tiles")


def test_unknown_keys_are_ignored(tmp_path) -> None:
    header, body = _valid_parts()
    body["units"][0]["future"] = 1
    path = _write_raw(tmp_path / "prefabs" / f"ok{prefabs.SUFFIX}", {**header, "future": 1}, {**body, "more": []})
    assert len(prefabs.load_block(path).units) == header["unit_count"]


def _body_refused(tmp_path, header, body, match) -> None:
    path = _write_raw(tmp_path / "prefabs" / f"bad{prefabs.SUFFIX}", header, body)
    with pytest.raises(PrefabRefused, match=match):
        prefabs.load_block(path)
    assert prefabs.read_info(path).error is None  # a corrupt body still lists


_UNIT_BREAKS = {
    "bool": ("status", True, "status"),
    "float-int": ("unit_const", 83.0, "unit_const"),
    "string-int": ("player", "1", "player"),
    "player": ("player", 9, "player"),
    "u16": ("unit_const", 0x10000, "unit_const"),
    "s8": ("capture_flag", 128, "capture_flag"),
    "dx-nan": ("dx", float("nan"), "dx"),
    "dx-range": ("dx", 1e39, "dx"),
    "dx-bigint": ("dx", 10**400, "dx"),
    "caption-surrogate": ("caption_string", "\ud800", "caption_string"),
    "z-range": ("z", -1e39, "z"),
    "z-object": ("z", {"f32": "zz"}, "f32"),
    "rotation-string": ("rotation", "0", "rotation"),
    "caption": ("caption_string", 3, "caption_string"),
    "garrison-self": ("garrison_slot", 0, "garrison_slot"),
    "garrison-past": ("garrison_slot", 10_000, "garrison_slot"),
}


@pytest.mark.parametrize("case", sorted(_UNIT_BREAKS))
def test_bad_unit_fields_are_refused(tmp_path, case) -> None:
    header, body = _valid_parts()
    key, value, match = _UNIT_BREAKS[case]
    body["units"][0][key] = value
    _body_refused(tmp_path, header, body, match)


def test_bad_tiles_and_counts_are_refused(tmp_path) -> None:
    header, body = _valid_parts()
    _body_refused(tmp_path, header, {**body, "terrain_ids": body["terrain_ids"][:-1]}, "tiles")
    _body_refused(tmp_path, header, {**body, "elevations": [256, *body["elevations"][1:]]}, "elevations")
    _body_refused(tmp_path, header, {**body, "layers": [True, *body["layers"][1:]]}, "layers")
    _body_refused(tmp_path, {**header, "has_tiles": False}, body, "tiles")
    _body_refused(tmp_path, {**header, "unit_count": header["unit_count"] + 1}, body, "units")
    missing = dict(body)
    del missing["units"]
    _body_refused(tmp_path, header, missing, "units")
    del body["units"][0]["z"]
    _body_refused(tmp_path, header, body, "'z'")


def test_save_refuses_a_block_that_would_not_load_back() -> None:
    with pytest.raises(PrefabRefused, match="cannot be saved"):
        prefabs.save_block(dataclasses.replace(_copied(), units=(_unit(dx=math.nan),)), "nan")
    with pytest.raises(PrefabRefused, match="cannot be saved"):
        prefabs.save_block(dataclasses.replace(_copied(), units=(_unit(caption_string="\ud800"),)), "surrogate")
    with pytest.raises(PrefabRefused, match="garrison_slot"):
        prefabs.save_block(dataclasses.replace(_copied(), units=(_unit(garrison_slot=0),)), "self")
    assert not prefabs.prefab_dir().exists()


# -- names, slugs and folder operations -----------------------------------


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Walled Base", "walled-base"),
        ("  Forêt  #2 ", "foret-2"),
        ("世界", "prefab"),
        ("CON", "con-prefab"),
        ("com1", "com1-prefab"),
        ("x" * 100, "x" * prefabs.SLUG_MAX),
    ],
)
def test_slug_rules(name, slug) -> None:
    assert prefabs.slug_for(name.strip()) == slug


def test_names_are_unique_case_insensitively_and_slugs_get_suffixes() -> None:
    first = prefabs.save_block(_copied(), "Castle")
    second = prefabs.save_block(_copied(), "castle!")
    assert (first.name, second.name) == (f"castle{prefabs.SUFFIX}", f"castle-2{prefabs.SUFFIX}")
    with pytest.raises(PrefabNameTaken) as taken:
        prefabs.save_block(_copied(), " CASTLE ")
    assert taken.value.path == first
    replaced = prefabs.save_block(_edge_block(), "CASTLE", replace_path=first)
    assert replaced == first and prefabs.read_info(first).name == "CASTLE"
    assert len(list(prefabs.prefab_dir().iterdir())) == 2  # no temp file left behind
    with pytest.raises(PrefabRefused, match="control"):
        prefabs.save_block(_copied(), "a\tb")


def test_rename_to_the_same_slug_keeps_one_file_and_its_saved_time() -> None:
    path = prefabs.save_block(_copied(), "Forest")
    saved = prefabs.read_info(path).saved
    assert prefabs.rename(path, "FOREST") == path
    assert [p.name for p in prefabs.prefab_dir().iterdir()] == [path.name]
    info = prefabs.read_info(path)
    assert (info.name, info.saved) == ("FOREST", saved)
    moved = prefabs.rename(path, "Grove")
    assert moved.name == f"grove{prefabs.SUFFIX}" and not path.exists()
    other = prefabs.save_block(_copied(), "Lake")
    with pytest.raises(PrefabNameTaken):
        prefabs.rename(other, "grove")


def test_rename_keeps_the_old_file_when_the_write_fails(monkeypatch) -> None:
    path = prefabs.save_block(_copied(), "Keep")

    def boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(prefabs, "atomic_write", boom)
    with pytest.raises(PrefabRefused, match="disk full"):
        prefabs.rename(path, "Elsewhere")
    assert path.exists() and prefabs.read_info(path).name == "Keep"


def test_delete_and_find_by_name() -> None:
    path = prefabs.save_block(_copied(), "Gone")
    assert prefabs.find_by_name("gone") == path
    prefabs.delete(path)
    assert prefabs.find_by_name("gone") is None and prefabs.list_prefabs() == []


def test_a_subscribed_mods_folder_is_refused_and_nothing_is_created(tmp_path) -> None:
    folder = tmp_path / "mods" / "Subscribed" / "123" / "prefabs"
    with pytest.raises(PrefabWriteBlocked):
        prefabs.save_block(_copied(), "Mod", folder=folder)
    with pytest.raises(PrefabWriteBlocked):
        prefabs.ensure_dir(folder)
    assert not (tmp_path / "mods").exists()


def test_the_header_carries_the_list_columns() -> None:
    block = _copied()
    info = prefabs.read_info(prefabs.save_block(block, "Cols"))
    assert (info.width, info.height, info.unit_count, info.has_tiles) == (block.width, block.height, len(block.units), True)
    assert info.thumbnail.shape == (prefabs.THUMBNAIL_MAX, prefabs.THUMBNAIL_MAX, 3)
    units_only = prefabs.read_info(prefabs.save_block(RegionBlock(2, 2, (), (), (), (_unit(),)), "Units"))
    assert units_only.thumbnail.size == 0 and not units_only.has_tiles


def test_a_damaged_gzip_trailer_or_trailing_data_is_refused(tmp_path) -> None:
    good = prefabs.save_block(_copied(), "Whole").read_bytes()
    bad_crc = bytearray(good)
    bad_crc[-8] ^= 0xFF
    folder = prefabs.prefab_dir()
    for name, data in (("no-trailer", good[:-8]), ("bad-crc", bytes(bad_crc)), ("extra", good + gzip.compress(b"x"))):
        path = folder / f"{name}{prefabs.SUFFIX}"
        path.write_bytes(data)
        with pytest.raises(PrefabRefused):
            prefabs.load_block(path)


@pytest.mark.parametrize(
    ("key", "flood", "match"),
    [("units", [{}] * 2000, "objects"), ("terrain_ids", [[]] * 2000, "lists"), ("layers", [""] * 2000, "strings")],
)
def test_a_body_flooded_with_empty_containers_is_refused_before_parsing(tmp_path, monkeypatch, key, flood, match) -> None:
    header, body = _valid_parts()
    calls = []
    real_loads = prefabs.json.loads
    monkeypatch.setattr(prefabs.json, "loads", lambda text: calls.append(len(text)) or real_loads(text))
    path = _write_raw(tmp_path / "prefabs" / f"flood{prefabs.SUFFIX}", header, {**body, key: flood})
    with pytest.raises(PrefabRefused, match=match):
        prefabs.load_block(path)
    assert len(calls) == 1  # the header only: the body was never handed to json


def test_rename_away_from_a_name_ending_in_digits_takes_the_new_slug() -> None:
    path = prefabs.save_block(_copied(), "Castle 2024")
    assert path.name == f"castle-2024{prefabs.SUFFIX}"
    moved = prefabs.rename(path, "Castle")
    assert moved.name == f"castle{prefabs.SUFFIX}" and not path.exists()
    suffixed = prefabs.save_block(_copied(), "Castle!")
    assert prefabs.rename(suffixed, "castle!!") == suffixed  # a real -2 suffix stays put

