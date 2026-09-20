"""Default-tier guard for tools/gen_contact_shadow_eyeball.py's own
off-engine path, same shape as tests/test_seam_eyeball.py: check() is
Qt-free, writes nothing, and also asserts every render.py constant it
patches is restored."""

from __future__ import annotations

import conftest


def test_gen_contact_shadow_eyeball_check_passes() -> None:
    gen = conftest.load_verify_module("gen_contact_shadow_eyeball")
    gen.check()
