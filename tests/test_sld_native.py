"""The native SLD walk and delta-chain decode (_composite_native.pyx's
sld_walk/sld_decode_layer) against the numpy code they replace, byte for byte
and error for error. numpy is the oracle: tests/test_sld_decoder.py pins it
against step-by-step references, and this pins the kernel against it.

Both backends run in one process via composite_backend.use_backend(), which
sld_decoder reads at call time. Skips when the kernel isn't built, unless
DESCAPE_REQUIRE_NATIVE=1, where a missing or stale build fails instead. The
fallback tests at the end fake the kernel, so they run either way.

The kernel answers a failure sentinel and the Python re-runs to raise the
exact SLDError. A refusal the Python then accepts would still draw correct
pixels, just slowly, so every sweep here also asserts
sld_decoder.native_disagreements stays 0: that is the only place that bug shows.
"""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_sld_decoder import (
    Frame,
    Layer,
    _assert_same_frame,
    _walk_fixtures,
    _window_fixtures,
    bc1_block,
    bc4_block,
    build_sld,
    solid_bc1,
    solid_bc4,
    walk_variants,
)

from descape import asset_source, composite_backend, sld_decoder, unit_sprites
from descape.sld_decoder import LayerKind, SLDError, SLDFile

KINDS = ("main", "shadow", "damage", "playercolor")


@pytest.fixture
def native_kernel():
    if composite_backend.available():
        return
    if os.environ.get("DESCAPE_REQUIRE_NATIVE") == "1":
        pytest.fail(f"DESCAPE_REQUIRE_NATIVE=1 but {composite_backend.unavailable_reason}")
    pytest.skip(composite_backend.unavailable_reason)


@pytest.fixture
def no_disagreements():
    before = sld_decoder.native_disagreements
    yield
    assert sld_decoder.native_disagreements == before, "the kernel refused something numpy accepts"


def _index(sld: SLDFile) -> tuple:
    return (sld._frames.tobytes(), sld._layers.tobytes(), tuple(chain.tobytes() for chain in sld._chains))


def _walk(backend: str, data: bytes):
    with composite_backend.use_backend(backend):
        try:
            return "walked", _index(SLDFile(data, keep_data=False))
        except SLDError as exc:
            return "refused", str(exc)


def _decode(backend: str, data: bytes, index: int, **kwargs):
    with composite_backend.use_backend(backend):
        sld = SLDFile(data)
        try:
            frame = sld.decode_frame(index, **kwargs)
        except SLDError as exc:
            return "refused", str(exc)
    return "decoded", frame


def _assert_same_decode(got, want) -> None:
    assert got[0] == want[0]
    if want[0] == "refused":
        assert got[1] == want[1]
    else:
        _assert_same_frame(got[1], want[1])


# -- decode-only failure fixtures -----------------------------------------


def _decode_fixtures() -> dict[str, bytes]:
    """Files that walk cleanly but fail or branch inside the decode: a command
    stream overrunning its grid, draws past EOF, a failure in a chain's root
    reached through a later frame, and delta copies whose box offset is not a
    multiple of 4 in either direction (Python floors, C truncates)."""
    grid = [bc1_block(0x001F * (i + 1) & 0xFFFF, 0x0000, [i % 4] * 16) for i in range(4)]
    return {
        "overrun": build_sld([Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), commands=[(1, 1)], blocks=[solid_bc1()])])]),
        "short_tail": build_sld([Frame([Layer(LayerKind.MAIN, box=(0, 0, 8, 8), commands=[(0, 4)], blocks=[solid_bc1()])])]),
        "bad_root": build_sld(
            [
                Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), commands=[(2, 0)], blocks=[])]),
                Frame([Layer(LayerKind.MAIN, box=(0, 0, 4, 4), flag0=0x80, commands=[(1, 0)], blocks=[])]),
            ]
        ),
        "negative_offset_delta": build_sld(
            [
                Frame([Layer(LayerKind.MAIN, box=(6, 6, 14, 14), blocks=grid)], canvas=(16, 16)),
                Frame(
                    [Layer(LayerKind.MAIN, box=(4, 4, 12, 12), flag0=0x80, commands=[(4, 0)], blocks=[])],
                    canvas=(16, 16),
                ),
            ]
        ),
        "positive_offset_delta": build_sld(
            [
                Frame([Layer(LayerKind.MAIN, box=(2, 4, 10, 12), blocks=grid)], canvas=(16, 16)),
                Frame(
                    [Layer(LayerKind.MAIN, box=(9, 1, 17, 9), flag0=0x80, commands=[(4, 0)], blocks=[])],
                    canvas=(16, 16),
                ),
            ]
        ),
        "bc4_modes": build_sld(
            [
                Frame(
                    [
                        Layer(LayerKind.MAIN, box=(0, 0, 8, 4), blocks=[solid_bc1(), solid_bc1(0x07E0)]),
                        Layer(LayerKind.SHADOW, box=(0, 0, 8, 4), blocks=[bc4_block(40, 200, list(range(8)) * 2), solid_bc4(9)]),
                        Layer(LayerKind.PLAYERCOLOR, blocks=[bc4_block(200, 40, list(range(8)) * 2), bc4_block(7, 7, [6, 7] * 8)]),
                    ]
                )
            ]
        ),
        "bc1_modes": build_sld(
            [
                Frame(
                    [
                        Layer(
                            LayerKind.MAIN,
                            box=(0, 0, 8, 8),
                            blocks=[
                                bc1_block(0xF81F, 0x07E0, [0, 1, 2, 3] * 4),
                                bc1_block(0x07E0, 0xF81F, [3, 2, 1, 0] * 4),
                                bc1_block(0x1234, 0x1234, [0, 1, 2, 3] * 4),
                                bc1_block(0xFFFF, 0x0000, [1, 2, 3, 0] * 4),
                            ],
                        ),
                        Layer(LayerKind.DAMAGE, blocks=[bc1_block(0x0001, 0xFFFE, [2, 3] * 8)] * 4),
                    ],
                    canvas=(6, 10),
                )
            ]
        ),
    }


def _all_fixtures() -> dict[str, bytes]:
    return {**_walk_fixtures(), **_decode_fixtures()}


# -- walk ------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_all_fixtures()))
def test_the_native_walk_builds_the_numpy_walks_index(native_kernel, no_disagreements, name):
    data = _all_fixtures()[name]
    want = _walk("numpy", data)
    assert want[0] == "walked"
    assert _walk("native", data) == want


@pytest.mark.parametrize("name", ["mixed_16", "mixed_14", "every_layer_kind", "variant_14_delta"])
def test_every_truncation_and_corruption_walks_or_fails_as_numpy_does(native_kernel, no_disagreements, name):
    """Every error the walk can raise, reached by the kernel's sentinel and
    worded by the Python re-walk, on every variant test_sld_decoder's
    reference sweep uses."""
    refused = walked = 0
    for variant in walk_variants(_walk_fixtures()[name]):
        want = _walk("numpy", variant)
        assert _walk("native", variant) == want, variant.hex()
        refused += want[0] == "refused"
        walked += want[0] == "walked"
    assert refused and walked


# -- decode ----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_all_fixtures()))
def test_every_frame_decodes_as_numpy_does(native_kernel, no_disagreements, name):
    data = _all_fixtures()[name]
    with composite_backend.use_backend("numpy"):
        count = SLDFile(data).frame_count
    for index in range(count):
        _assert_same_decode(_decode("native", data, index), _decode("numpy", data, index))


@pytest.mark.parametrize("name", sorted({**_all_fixtures(), **_window_fixtures()}))
def test_every_frame_decodes_windowed_as_numpy_does(native_kernel, no_disagreements, name):
    """decode_frame(window_to_main=True), as _native_frame calls it, with and
    without its kinds filter."""
    data = {**_all_fixtures(), **_window_fixtures()}[name]
    with composite_backend.use_backend("numpy"):
        count = SLDFile(data).frame_count
    for index in range(count):
        for kinds in (None, (LayerKind.MAIN, LayerKind.PLAYERCOLOR)):
            want = _decode("numpy", data, index, kinds=kinds, window_to_main=True)
            _assert_same_decode(_decode("native", data, index, kinds=kinds, window_to_main=True), want)


def test_the_decode_fixtures_reach_both_decode_errors_and_real_pixels(native_kernel, no_disagreements):
    """Non-vacuity for the test above: the numpy oracle refuses the three
    failure fixtures with each of its two decode errors, and draws the rest."""
    fixtures = _decode_fixtures()
    assert re.search("fills 2 blocks", _decode("numpy", fixtures["overrun"], 0)[1])
    assert re.search("block data runs past", _decode("numpy", fixtures["short_tail"], 0)[1])
    assert re.search("fills 2 blocks", _decode("numpy", fixtures["bad_root"], 1)[1])
    for name in ("negative_offset_delta", "positive_offset_delta", "bc4_modes", "bc1_modes"):
        outcome = _decode("numpy", fixtures[name], 0)
        assert outcome[0] == "decoded" and outcome[1].main.any(), name


def test_a_delta_copy_floors_a_negative_box_offset(native_kernel, no_disagreements):
    """Python's (4 - 6) // 4 is -1; C's truncating division gives 0. The
    delta layer's block (1, 1) takes the previous layer's block (0, 0), and
    its row and column 0 fall outside the previous layer entirely."""
    data = _decode_fixtures()["negative_offset_delta"]
    for backend in ("numpy", "native"):
        first = _decode(backend, data, 0)[1].main
        delta = _decode(backend, data, 1)[1].main
        assert np.array_equal(delta[8:12, 8:12], first[6:10, 6:10]), backend
        assert not delta[4:8, 4:12].any() and not delta[4:12, 4:8].any(), backend


def test_decoding_every_corrupted_variant_matches_numpy(native_kernel, no_disagreements):
    """Every single-byte corruption of a file mixing all kinds and a delta,
    through both backends' walk and every frame's decode."""
    data = _walk_fixtures()["mixed_16"]
    refused = decoded = 0
    for variant in walk_variants(data):
        if _walk("numpy", variant)[0] != "walked":
            continue
        with composite_backend.use_backend("numpy"):
            count = SLDFile(variant).frame_count
        for index in range(count):
            want = _decode("numpy", variant, index)
            _assert_same_decode(_decode("native", variant, index), want)
            refused += want[0] == "refused"
            decoded += want[0] == "decoded"
    assert refused and decoded


def test_a_released_index_decodes_as_numpy_does_from_passed_bytes(native_kernel, no_disagreements):
    data = _walk_fixtures()["delta_chain"]
    with composite_backend.use_backend("numpy"):
        want = [SLDFile(data).decode_frame(i) for i in range(SLDFile(data).frame_count)]
    with composite_backend.use_backend("native"):
        sld = SLDFile(data, keep_data=False)
        for index, frame in enumerate(want):
            _assert_same_frame(sld.decode_frame(index, bytearray(data)), frame)
        with pytest.raises(SLDError, match="changed on disk"):
            sld.decode_frame(0, data[:-1])


# -- fallback ----------------------------------------------------------------


def _refusing_kernel() -> SimpleNamespace:
    return SimpleNamespace(sld_walk=lambda *_args: -1, sld_decode_layer=lambda *_args: False)


def test_a_kernel_refusal_numpy_accepts_is_counted_and_answered_by_numpy(monkeypatch):
    """The bug class no pixel test sees: numpy's result is used, and the
    counter every sweep above watches goes up."""
    data = _walk_fixtures()["every_layer_kind"]
    with composite_backend.use_backend("numpy"):
        want_index = _index(SLDFile(data))
        want = [SLDFile(data).decode_frame(i) for i in range(2)]
    before = sld_decoder.native_disagreements
    monkeypatch.setattr(composite_backend, "native", _refusing_kernel())

    sld = SLDFile(data)
    assert _index(sld) == want_index
    assert sld_decoder.native_disagreements == before + 1
    for index, frame in enumerate(want):
        _assert_same_frame(sld.decode_frame(index), frame)
    assert sld_decoder.native_disagreements > before + 1


@pytest.mark.parametrize("name", ["mixed_16", "clipped", "off_right"])
def test_a_kernel_refusal_windows_as_numpy_does(monkeypatch, name):
    data = _window_fixtures()[name]
    with composite_backend.use_backend("numpy"):
        sld = SLDFile(data)
        want = [sld.decode_frame(i, window_to_main=True) for i in range(sld.frame_count)]
    monkeypatch.setattr(composite_backend, "native", _refusing_kernel())

    for index, frame in enumerate(want):
        _assert_same_frame(sld.decode_frame(index, window_to_main=True), frame)


def test_a_kernel_refusal_numpy_also_refuses_raises_the_exact_error_uncounted(monkeypatch):
    fixtures = _decode_fixtures()
    with composite_backend.use_backend("numpy"):
        with pytest.raises(SLDError) as walk_error:
            SLDFile(_walk_fixtures()["mixed_16"][:40])
        with pytest.raises(SLDError) as decode_error:
            SLDFile(fixtures["overrun"]).decode_frame(0)
        overrun = SLDFile(fixtures["overrun"])
    before = sld_decoder.native_disagreements
    monkeypatch.setattr(composite_backend, "native", _refusing_kernel())

    with pytest.raises(SLDError, match=re.escape(str(walk_error.value))):
        SLDFile(_walk_fixtures()["mixed_16"][:40])
    with pytest.raises(SLDError, match=re.escape(str(decode_error.value))):
        overrun.decode_frame(0)
    assert sld_decoder.native_disagreements == before


# -- real install ------------------------------------------------------------


@pytest.mark.corpus
def test_real_install_files_walk_and_decode_every_frame_as_numpy_does(native_kernel, no_disagreements):
    """Every frame of a fixed sample of real files, both backends: every
    100th file by name, a variant-14 Stable, and the deep-chain trees."""
    root = asset_source.get_install_path()
    if root is None:
        pytest.skip("no AoE2:DE install visible: set AOE2DE_INSTALL_PATH (conftest hides config.yaml)")
    directory = root / unit_sprites.GRAPHICS_SUBPATH
    names = sorted(path.name for path in directory.glob("*.sld"))
    if not names:
        pytest.skip(f"no .sld files under {directory}")
    wanted = {"b_west_stable_age3_x1.sld", "n_tree_oak_x1.sld", "n_tree_pine_x1.sld"}
    sample = sorted(set(names[::100]) | (wanted & set(names)))

    frames = drawn = windowed = 0
    for file_name in sample:
        data = (directory / file_name).read_bytes()
        want = _walk("numpy", data)
        assert _walk("native", data) == want, file_name
        if want[0] != "walked":
            continue
        with composite_backend.use_backend("numpy"):
            numpy_sld = SLDFile(data)
        with composite_backend.use_backend("native"):
            native_sld = SLDFile(data)
        for index in range(numpy_sld.frame_count):
            with composite_backend.use_backend("numpy"):
                try:
                    expected = ("decoded", numpy_sld.decode_frame(index))
                except SLDError as exc:
                    expected = ("refused", str(exc))
            with composite_backend.use_backend("native"):
                try:
                    got = ("decoded", native_sld.decode_frame(index))
                except SLDError as exc:
                    got = ("refused", str(exc))
            _assert_same_decode(got, expected)
            frames += 1
            drawn += expected[0] == "decoded" and any(getattr(expected[1], kind) is not None for kind in KINDS)
            if expected[0] != "decoded":
                continue
            # The windowed pass _native_frame uses: native == numpy, and both
            # are the whole-canvas decode sliced to the window.
            with composite_backend.use_backend("numpy"):
                window_want = numpy_sld.decode_frame(index, window_to_main=True)
            with composite_backend.use_backend("native"):
                window_got = native_sld.decode_frame(index, window_to_main=True)
            _assert_same_frame(window_got, window_want)
            full = expected[1]
            x0, y0 = full.hotspot_x - window_want.hotspot_x, full.hotspot_y - window_want.hotspot_y
            for kind in KINDS:
                if getattr(full, kind) is not None:
                    sliced = getattr(full, kind)[y0 : y0 + window_want.height, x0 : x0 + window_want.width]
                    assert np.array_equal(getattr(window_want, kind), sliced), (file_name, index, kind)
            windowed += (window_want.width, window_want.height) != (full.width, full.height)
    assert drawn > 0.9 * frames > 0
    assert windowed > 0.5 * drawn
