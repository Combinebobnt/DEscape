"""Phase 1 adapter (see the migration plan): every check_* function across
tools/verify_*.py, plus verify_roundtrip.py's own main()-level load
assertion (the plan's 42+1+1 reconciliation -- see migration_manifest.py's
own module docstring), wrapped as an individually-named pytest test.

generate_reference_pngs, the reconciliation's other non-check-shaped entry,
no longer has a test here: Ordering step 4 extracted it into
tools/gen_iso_reference_pngs.py, which is what its manifest entry's
migrated_test_nodeid now points at.

This is scaffolding, not the end state: Phase 3 replaces each remaining
test here with a hand-written one in its own file, carrying the original
docstring and (where the plan calls for it) a real fixture instead of the
corpus tier -- see descape/scenario_io.py's BLANK_TEMPLATE_PATH, the shipped
blank template used by every migrated default-tier check so far. This file
and migration_manifest.py's `family`/`tier` columns get deleted once every
entry's migrated_test_nodeid is filled in.
"""

from __future__ import annotations

import pytest

import conftest
from migration_manifest import MANIFEST


def _make_none_test(module_name: str, check_name: str):
    def test() -> None:
        module = conftest.load_verify_module(module_name)
        conftest.run_check(getattr(module, check_name))

    return test


def _make_path_test(module_name: str, check_name: str):
    def test(scenario_path) -> None:
        module = conftest.load_verify_module(module_name)
        conftest.run_check(getattr(module, check_name), scenario_path)

    return test


def _make_gui_path_test(module_name: str, check_name: str):
    def test(scenario_path) -> None:
        if not conftest.PYQT5_AVAILABLE:
            pytest.skip("PyQt5 not importable")
        conftest.ensure_qapp()
        module = conftest.load_verify_module(module_name)
        conftest.run_check(getattr(module, check_name), scenario_path)

    return test


def _make_path_tmp_test(module_name: str, check_name: str):
    def test(scenario_path, tmp_path) -> None:
        module = conftest.load_verify_module(module_name)
        conftest.run_check(getattr(module, check_name), scenario_path, tmp_path)

    return test


def _make_files_test(module_name: str, check_name: str):
    def test(corpus_files) -> None:
        module = conftest.load_verify_module(module_name)
        conftest.run_check(getattr(module, check_name), corpus_files)

    return test


_FAMILY_BUILDERS = {
    "none": _make_none_test,
    "path": _make_path_test,
    "path_tmp": _make_path_tmp_test,
    "files": _make_files_test,
}


def _register(module_name: str, check_name: str, family: str, tier: tuple[str, ...]):
    builder = _make_gui_path_test if "gui" in tier else _FAMILY_BUILDERS[family]
    test_func = builder(module_name, check_name)
    test_func.__name__ = f"test_{module_name.removeprefix('verify_')}__{check_name}"
    test_func.__doc__ = f"Phase 1 adapter for {module_name}.{check_name} -- see migration_manifest.py."
    for mark_name in tier:
        test_func = getattr(pytest.mark, mark_name)(test_func)
    globals()[test_func.__name__] = test_func


for _entry in MANIFEST:
    if _entry.family in _FAMILY_BUILDERS:
        _register(_entry.original_module, _entry.original_check_name, _entry.family, _entry.tier)
del _entry


# -- The one remaining "special" manifest entry that isn't a check_*
# function and isn't already migrated. generate_reference_pngs (the other
# special entry) no longer needs a stand-in here -- it was extracted into
# tools/gen_iso_reference_pngs.py in Ordering step 4, and its manifest
# entry's migrated_test_nodeid points there instead. --


@pytest.mark.corpus
def test_roundtrip__main_load_assertion(scenario_path) -> None:
    """Phase 1 stand-in for verify_roundtrip.py's main(): every file must
    load via load_map_and_units without raising -- a separate assertion
    from check_tail_completeness, not folded into it."""
    from descape.scenario_io import load_map_and_units

    load_map_and_units(scenario_path)
