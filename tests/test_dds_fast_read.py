"""asset_source's numpy read of an uncompressed 32-bit DDS (GH #68, TASK-075):
the same pixels Pillow decodes, without Pillow's pure-Python decoder, which
held the GIL for about 60 ms per portrait and froze the GUI from the
catalog's worker thread."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image

from descape import asset_source

_RGBA_MASKS = (0xFF, 0xFF00, 0xFF0000, 0xFF000000)  # the install's own layout
_BGRA_MASKS = (0xFF0000, 0xFF00, 0xFF, 0xFF000000)  # Pillow's writer, and one install file


def _dds(path, pixels: np.ndarray, masks, flags=0x41) -> None:
    """A minimal uncompressed DDS: 128-byte header, then the raw 32-bit pixels."""
    height, width = pixels.shape[:2]
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<7I", header, 4, 124, 0x100F, height, width, width * 4, 0, 1)
    struct.pack_into("<2I4s5I", header, 76, 32, flags, b"\0\0\0\0", 32, *masks)
    path.write_bytes(bytes(header) + pixels.astype(np.uint8).tobytes())


def _pillow(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGBA"))


@pytest.mark.parametrize("masks", [_RGBA_MASKS, _BGRA_MASKS], ids=["rgba", "bgra"])
def test_the_fast_read_matches_pillow(tmp_path, masks) -> None:
    raw = np.random.default_rng(68).integers(0, 256, (17, 23, 4), dtype=np.uint8)
    path = tmp_path / "portrait.dds"
    _dds(path, raw, masks)
    fast = asset_source._read_dds_rgba_fast(path)
    assert fast is not None, "this layout must take the fast path"
    assert np.array_equal(fast, _pillow(path))
    read = asset_source._read_rgba(path)
    assert np.array_equal(read, fast) and not read.flags.writeable


def test_pillow_never_runs_for_the_fast_layout(tmp_path, monkeypatch) -> None:
    path = tmp_path / "portrait.dds"
    _dds(path, np.full((8, 8, 4), 200, dtype=np.uint8), _RGBA_MASKS)

    def refuse(*_args, **_kwargs):
        raise AssertionError("Pillow decoded a layout numpy reads")

    monkeypatch.setattr(Image, "open", refuse)
    assert asset_source._read_rgba(path) is not None


def test_a_compressed_dds_is_left_to_pillow(tmp_path) -> None:
    path = tmp_path / "compressed.dds"
    _dds(path, np.zeros((4, 4, 4), dtype=np.uint8), _RGBA_MASKS, flags=0x4)  # DDPF_FOURCC, not RGB
    assert asset_source._read_dds_rgba_fast(path) is None


def test_a_truncated_file_is_left_to_pillow(tmp_path) -> None:
    path = tmp_path / "short.dds"
    _dds(path, np.zeros((8, 8, 4), dtype=np.uint8), _RGBA_MASKS)
    path.write_bytes(path.read_bytes()[:200])
    assert asset_source._read_dds_rgba_fast(path) is None


def test_the_real_portraits_read_exactly_as_pillow_reads_them() -> None:
    """Needs AOE2DE_INSTALL_PATH (conftest hides config.yaml)."""
    install = asset_source.get_install_path()
    if install is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH")
    fast_count = 0
    for folder in (asset_source.UNIT_ICON_DIR, asset_source.BUILDING_ICON_DIR):
        for path in sorted((install / folder).iterdir())[:60]:
            if path.suffix.lower() != ".dds":
                continue
            fast = asset_source._read_dds_rgba_fast(path)
            if fast is not None:
                fast_count += 1
                assert np.array_equal(fast, _pillow(path)), path.name
    assert fast_count > 30
