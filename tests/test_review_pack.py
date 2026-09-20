"""The check() equivalent of tests/test_seam_eyeball.py, plus direct unit
coverage of testkit/review_pack.py's own rules.

Qt-free: gen_review_pack.check() never touches PyQt5 or ViewerWindow (see
that tool's docstring for why it renders off-engine throughout), so this
carries no gui marker and runs regardless of PyQt5 availability.

check() renders real crops, so it is the slowest thing here by a wide
margin. It stays in the default tier anyway: a review pack whose framing
rules or inject sizes have silently drifted is worse than no review pack,
because its verdicts still read as evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from testkit import review_pack as rp

import conftest


def test_gen_review_pack_check_passes() -> None:
    gen = conftest.load_verify_module("gen_review_pack")
    gen.check()


# ------------------------------------------------------------ spec coupling


def _frame(frame_id: str) -> rp.Frame:
    return rp.Frame(frame_id, f"{frame_id}.png", "raw")


def _spec(frames, checks) -> rp.PackSpec:
    return rp.PackSpec(id="t", title="t", frames=tuple(frames), checks=tuple(checks))


def test_every_frame_must_be_named_by_some_check() -> None:
    spec = _spec(
        [_frame("a"), _frame("orphan")],
        [rp.Check("only", "?", ("a",))],
    )
    with pytest.raises(rp.PackError, match="orphan"):
        rp.validate_spec(spec)


def test_a_check_may_not_name_a_frame_that_does_not_exist() -> None:
    spec = _spec([_frame("a")], [rp.Check("only", "?", ("a", "ghost"))])
    with pytest.raises(rp.PackError, match="ghost"):
        rp.validate_spec(spec)


def test_a_check_with_no_frame_is_rejected_at_construction() -> None:
    with pytest.raises(rp.PackError, match="names no frame"):
        rp.Check("empty", "?", ())


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(rp.PackError, match="duplicate frame"):
        rp.validate_spec(_spec([_frame("a"), _frame("a")], [rp.Check("c", "?", ("a",))]))
    with pytest.raises(rp.PackError, match="duplicate check"):
        rp.validate_spec(_spec([_frame("a")], [rp.Check("c", "?", ("a",)), rp.Check("c", "?", ("a",))]))


def test_a_valid_spec_passes() -> None:
    rp.validate_spec(_spec([_frame("a"), _frame("b")], [rp.Check("c", "?", ("a", "b"))]))


def test_frame_kind_is_constrained() -> None:
    with pytest.raises(rp.PackError, match="kind"):
        rp.Frame("a", "a.png", "screenshot")


# ------------------------------------------------------------ framing rules


def test_upscale_is_nearest_neighbour_not_interpolation() -> None:
    img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
    out = rp.nn_upscale(img, 3)
    assert out.shape == (3, 6, 3)
    # Every output pixel is one of the two input values: no blend anywhere.
    assert set(np.unique(out).tolist()) == {0, 255}


def test_upscale_factor_lands_inside_the_judgeable_band() -> None:
    for long_edge in (300, 352, 448, 500, 901, 1400):
        factor = rp.fit_upscale_factor(long_edge, long_edge // 2)
        assert rp.MIN_LONG_EDGE <= factor * long_edge <= rp.MAX_LONG_EDGE


def test_an_oversized_frame_is_a_crop_bug_not_a_downscale() -> None:
    with pytest.raises(rp.PackError, match="crop it, never downscale"):
        rp.fit_upscale_factor(12800, 6512)


def test_a_frame_no_integer_factor_can_fit_is_rejected() -> None:
    # 800 -> 1x is 800 (too small), 2x is 1600 (too large). No integer works,
    # and silently accepting either end is how an unjudgeable frame ships.
    with pytest.raises(rp.PackError, match="cannot reach"):
        rp.fit_upscale_factor(800, 400)


def test_frame_to_band_reports_the_factor_it_used() -> None:
    img = np.zeros((208, 352, 3), dtype=np.uint8)
    out, factor = rp.frame_to_band(img)
    assert factor == 4
    assert out.shape[:2] == (208 * 4, 352 * 4)


# --------------------------------------------------------- feature cropping


def test_feature_crop_tightens_onto_the_change_with_a_margin() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:45, 30:36] = True
    assert rp.feature_bbox(mask, 10, (100, 100)) == (20, 30, 46, 55)


def test_feature_crop_clips_to_the_frame() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0, 19] = True
    assert rp.feature_bbox(mask, 5, (20, 20)) == (14, 0, 20, 6)


def test_feature_crop_refuses_an_empty_mask() -> None:
    with pytest.raises(rp.PackError, match="cannot crop to a feature"):
        rp.feature_bbox(np.zeros((10, 10), dtype=bool), 2, (10, 10))


# ------------------------------------------------------------- inject bound


def _mask(spans) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    for y, x0, x1 in spans:
        mask[y, x0:x1] = True
    return mask


def test_a_localized_inject_passes_its_bound() -> None:
    bound = rp.LocalizedBound(max_changed_fraction=0.005, max_component_px=24)
    # Four separate 2px ticks, spread right across the frame: localized by
    # component even though the bounding box covers everything.
    assert bound.failures(_mask([(10, 5, 7), (30, 40, 42), (60, 70, 72), (90, 95, 97)])) == []


def test_an_inject_that_drifted_coarse_fails_its_bound() -> None:
    bound = rp.LocalizedBound(max_changed_fraction=0.005, max_component_px=24)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    failures = bound.failures(mask, "somewhere")
    assert any("no longer localized" in f for f in failures)
    assert all(f.startswith("somewhere: ") for f in failures)


def test_an_inject_that_changes_nothing_fails_loudest() -> None:
    bound = rp.LocalizedBound(max_changed_fraction=1.0, max_component_px=10**9)
    assert bound.failures(np.zeros((10, 10), dtype=bool)) == ["inject changed no pixels at all"]


def test_component_counting_is_eight_connected() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = mask[2, 2] = True  # diagonal touch: one component, not two
    assert rp.components(mask) == 1
    assert rp.largest_component_px(mask) == 2
    mask[4, 4] = True
    assert rp.components(mask) == 2


def test_changed_pixels_requires_matching_shapes() -> None:
    with pytest.raises(rp.PackError, match="shape mismatch"):
        rp.changed_pixels(np.zeros((2, 2, 3), np.uint8), np.zeros((3, 3, 3), np.uint8))
