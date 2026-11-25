"""Area upscale: x1 art enlarged by a non-whole factor at Graphics Quality
Default and above (the 128 px level's 4/3, Small Trees, Flat icons) is
area-resampled like a shrink, so every source pixel keeps an even size.
Whole factors (2x, 3x) stay on the nearest gather, which gives the same
bytes there. Synthetic frames only; the shrink side is pinned in
tests/test_unit_sprites.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from descape import unit_sprites

from test_unit_sprites import (  # noqa: F401 -- fixtures
    CONST,
    _both_backends,
    _random_native,
    _random_rgba,
    _reference_box,
    _tint_spy,
    backend,
    build_sld,
    install,
    native_kernel,
)

ENLARGED, DOUBLED = 64, 96  # half_w: sprite_scale 4/3 (the 128 level) and 2.0 (the 192 level)


def _binary_alpha(rng, h, w):
    """Random RGBA whose alpha is 0 or 255, RGB zeroed under alpha 0, as real
    frames store it: area returns (0, 0, 0, 0) there where the gather keeps RGB."""
    rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    rgba[..., 3] = np.where(rng.random((h, w)) < 0.3, 0, 255)
    rgba[rgba[..., 3] == 0] = 0
    return rgba


# --- the filter ------------------------------------------------------------


@pytest.mark.parametrize(("h", "w", "out_h", "out_w"), [
    (12, 9, 16, 12), (27, 39, 36, 52), (7, 5, 23, 13), (1, 1, 4, 3),  # 4/3, odd sizes, 1-px source
    (12, 18, 16, 9), (18, 12, 9, 16),  # one axis grows while the other shrinks
])
def test_an_upscale_matches_a_float64_area_average(backend, h, w, out_h, out_w):  # noqa: F811
    src = _random_rgba(np.random.default_rng(h * 1000 + w * 10 + out_h), h, w)
    got = unit_sprites._box_resize_rgba(src, out_w, out_h)
    assert got.dtype == np.uint8 and got.shape == (out_h, out_w, 4)
    assert np.abs(got.astype(int) - _reference_box(src, out_w, out_h)).max() <= 1


@pytest.mark.parametrize("seed", range(8))
def test_the_native_upscale_is_byte_identical_to_numpy(native_kernel, seed):  # noqa: F811
    rng = np.random.default_rng(100 + seed)
    h, w = (int(v) for v in rng.integers(1, 60, 2))
    rgba = _random_rgba(rng, h, w)
    for out_h, out_w in ((h * 4 // 3 + 1, w * 4 // 3 + 1), (h + 7, max(1, w // 2)), (max(1, h // 3), w * 5 // 2)):
        native, numpy_out = _both_backends(lambda out_h=out_h, out_w=out_w: unit_sprites._box_resize_rgba(rgba, out_w, out_h))
        assert np.array_equal(native, numpy_out), (h, w, out_h, out_w)


@pytest.mark.parametrize("factor", [2, 3])
def test_a_whole_factor_area_upscale_equals_the_gather(backend, factor):  # noqa: F811
    """Why whole factors may stay nearest: the bytes are the same, given RGB
    zeroed under alpha 0."""
    src = _binary_alpha(np.random.default_rng(factor), 23, 17)
    got = unit_sprites._box_resize_rgba(src, 17 * factor, 23 * factor)
    assert np.array_equal(got, unit_sprites._resize_rgba(src, 17 * factor, 23 * factor))


def test_the_area_path_predicate():
    for scale in (0.67, 0.5):
        assert unit_sprites._area_path(scale, from_x2=True)
        assert not unit_sprites._area_path(scale, from_x2=False)
    for scale in (1.0, 4 / 3, 2.0):
        assert not unit_sprites._area_path(scale, from_x2=False)
    unit_sprites.set_smooth_shrink(True)
    assert unit_sprites._area_path(0.67, from_x2=False) and unit_sprites._area_path(4 / 3, from_x2=False)
    assert unit_sprites._area_path(4 / 3, from_x2=True), "x2 enlarged 4/3 (a 256 level) follows the same rule"
    for scale in (1.0, 2.0, 3.0):
        assert not unit_sprites._area_path(scale, from_x2=False)


# --- sprites ---------------------------------------------------------------


def test_a_4_3_sprite_is_the_area_upscale_of_the_native_tint(install, monkeypatch):  # noqa: F811
    """The 128 level at Default: box(_tinted(native)), _tinted seeing only the
    native frame, the hotspot where the nearest draw puts it."""
    main, pc = _random_native(install, monkeypatch)
    team = unit_sprites.TEAM_COLORS[3]
    off = unit_sprites.sprite_for(CONST, 0.0, 3, ENLARGED)
    width, height = round(main.shape[1] * 4 / 3), round(main.shape[0] * 4 / 3)
    nearest = unit_sprites._resize_rgba(unit_sprites._tinted(main, pc, team), width, height)
    assert np.array_equal(off.rgba, nearest), "smooth off keeps the nearest bytes"
    seen, real_tinted = _tint_spy(monkeypatch)
    unit_sprites.set_smooth_shrink(True)
    on = unit_sprites.sprite_for(CONST, 0.0, 3, ENLARGED)
    assert np.array_equal(on.rgba, unit_sprites._box_resize_rgba(real_tinted(main, pc, team), width, height))
    assert not np.array_equal(on.rgba, off.rgba)
    assert seen == [main.shape]
    assert (on.hotspot_x, on.hotspot_y) == (off.hotspot_x, off.hotspot_y)


def test_the_192_level_is_byte_identical_with_smooth_on_and_off(install, monkeypatch):  # noqa: F811
    """A whole factor stays on the gather even over random RGB under alpha 0,
    where area would zero it: so no 2x draw moves."""
    main, pc = _random_native(install, monkeypatch)
    off = unit_sprites.sprite_for(CONST, 0.0, 3, DOUBLED)
    unit_sprites.set_smooth_shrink(True)
    on = unit_sprites.sprite_for(CONST, 0.0, 3, DOUBLED)
    x2 = np.repeat(np.repeat(unit_sprites._tinted(main, pc, unit_sprites.TEAM_COLORS[3]), 2, axis=0), 2, axis=1)
    assert np.array_equal(off.rgba, x2) and np.array_equal(on.rgba, x2)
    assert (on.hotspot_x, on.hotspot_y) == (off.hotspot_x, off.hotspot_y)


def test_the_smooth_flag_keys_a_4_3_draw_apart(install, monkeypatch):  # noqa: F811
    _random_native(install, monkeypatch)
    draws = []
    for flag in (False, True, False):
        unit_sprites.set_smooth_shrink(flag)
        draws.append(unit_sprites.sprite_for(CONST, 0.0, 3, ENLARGED).rgba)
    assert not np.array_equal(draws[0], draws[1]) and np.array_equal(draws[0], draws[2])


# --- Flat icons ------------------------------------------------------------


def _opaque_native(install, monkeypatch):  # noqa: F811
    """A 30x24 random frame, opaque throughout, so its ink is the whole frame."""
    install.write(build_sld(1))
    install.register(angle_count=1, frame_count=1)
    rng = np.random.default_rng(21)
    main = rng.integers(0, 256, (30, 24, 4), dtype=np.uint8)
    main[..., 3] = 255
    pc = rng.integers(0, 256, (30, 24, 4), dtype=np.uint8)
    monkeypatch.setattr(unit_sprites, "_native_frame", lambda file_name, index: (main, pc, 12, 28))
    unit_sprites.clear_caches()
    return unit_sprites._tinted(main, pc, unit_sprites.TEAM_COLORS[2])


def test_a_single_piece_icon_enlarged_4_3_takes_the_area_path(install, monkeypatch):  # noqa: F811
    tinted = _opaque_native(install, monkeypatch)
    off = unit_sprites.icon_for(CONST, 0.0, 2, 32, 40)
    assert np.array_equal(off.rgba, unit_sprites._resize_rgba(tinted, 32, 40))
    seen, _real = _tint_spy(monkeypatch)
    unit_sprites.set_smooth_shrink(True)
    on = unit_sprites.icon_for(CONST, 0.0, 2, 32, 40)
    assert np.array_equal(on.rgba, unit_sprites._box_resize_rgba(tinted, 32, 40))
    assert not np.array_equal(on.rgba, off.rgba)
    assert seen == [tinted.shape], "tinted once, at native"


def test_a_single_piece_icon_enlarged_2x_stays_nearest(install, monkeypatch):  # noqa: F811
    tinted = _opaque_native(install, monkeypatch)
    unit_sprites.set_smooth_shrink(True)
    calls = []
    real = unit_sprites._box_resize_rgba
    monkeypatch.setattr(unit_sprites, "_box_resize_rgba", lambda *a: calls.append(a[1:]) or real(*a))
    icon = unit_sprites.icon_for(CONST, 0.0, 2, 48, 60)
    assert np.array_equal(icon.rgba, unit_sprites._resize_rgba(tinted, 48, 60))
    assert not calls
