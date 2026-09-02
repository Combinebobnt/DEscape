"""The check() equivalent of tests/test_config_example.py, run as part of
the default tier so a regression in tools/gen_seam_eyeball.py's own
off-engine rendering path fails the normal test run instead of only
surfacing the next time someone runs `tools/gen_seam_eyeball.py --check`
by hand.

Qt-free -- gen_seam_eyeball.check() never touches PyQt5/ViewerWindow (see
its own docstring for why: the default tier's <=20s budget doesn't
leave room for it here, and tests/test_seam_viewer.py already covers the
through-viewer path byte-identically), so this test carries no gui marker
and runs regardless of PyQt5 availability.
"""

from __future__ import annotations

import conftest


def test_gen_seam_eyeball_check_passes() -> None:
    gen = conftest.load_verify_module("gen_seam_eyeball")
    gen.check()
