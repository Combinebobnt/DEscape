"""Guards `testkit.settings_isolation` against the drift that made it exist.

`MEMOIZED_GLOBALS` is hand-maintained, and a global missing from it leaks one
test's or one capture's value into everything later in the same process: silent
contamination, not a failure. Until this list was shared it was copied by hand
into eight `tools/gen_*_eyeball.py` scripts, in four states of staleness (three
were eight names short, three fourteen short, one reset only four names, one
reset none) -- only
`tests/conftest.py`'s copy stayed current, because only that copy had a test
behind it. This file is that test, moved to the shared list (it used to live in
`test_settings.py` as `test_every_memoized_global_is_listed_in_conftest`), plus
a guard that no tool re-grows a private copy.

Discovery is reflective rather than a second hand-written list, which would only
move the same hazard, and runs two ways:

- Annotations, read off the imported module. Every memoized setting is written
  `_name: T | None = None`.
- Names, parsed from settings.py's source: a private module-level assignment
  that isn't a `_CAPS` constant. Value-independent, so a future `_thing: dict =
  {}` memo is caught too, where the annotation rule alone would miss it.

Neither reads live values: by the time this runs, an earlier test in the same
process may already have populated a memo, so "its value is None" is not an
order-independent question. Source-parsing rather than a fresh importlib exec of
settings.py for the same reason the annotation half doesn't reimport it -- that
would rebind the real CONFIG_PATH, outside the isolation conftest sets up.
"""

from __future__ import annotations

import ast
from pathlib import Path

from descape import settings
from testkit.settings_isolation import MEMOIZED_GLOBALS, isolate_settings

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PY = ROOT / "descape" / "settings.py"
TOOLS_DIR = ROOT / "tools"

_FIX_HINT = (
    "Add it to testkit.settings_isolation.MEMOIZED_GLOBALS -- otherwise its value "
    "leaks from one test or one eyeball capture into the next in the same process. "
    "If the name is not a memoized setting, write it as a _CAPS module constant, "
    "which this discovery deliberately skips."
)


def _nullable_annotated_globals() -> set[str]:
    annotations = getattr(settings, "__annotations__", {})
    return {
        name
        for name, annotation in annotations.items()
        if name.startswith("_") and str(annotation).replace(" ", "").endswith("|None")
    }


def _private_module_level_names() -> set[str]:
    tree = ast.parse(SETTINGS_PY.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            bare = target.id.lstrip("_")
            if target.id.startswith("_") and bare and bare != bare.upper():
                found.add(target.id)
    return found


def test_every_memoized_global_is_listed_in_the_shared_reset_list() -> None:
    listed = set(MEMOIZED_GLOBALS)
    discovered = _nullable_annotated_globals() | _private_module_level_names()
    assert discovered, f"discovery found nothing in {SETTINGS_PY.name} -- has the convention decayed?"
    missing = sorted(discovered - listed)
    stale = sorted(listed - discovered)
    assert not missing, f"descape/settings.py memoizes {missing}, absent from the shared reset list. {_FIX_HINT}"
    assert not stale, f"the shared reset list names {stale}, which descape/settings.py no longer declares -- drop them."
    # _DEFAULT_KEYBINDS is the near miss both halves must keep skipping:
    # underscore-prefixed and module-level, but neither nullable nor memoized.
    assert "_DEFAULT_KEYBINDS" not in listed


def test_shared_reset_list_has_no_duplicates() -> None:
    assert len(MEMOIZED_GLOBALS) == len(set(MEMOIZED_GLOBALS))


def test_no_tool_hand_rolls_its_own_reset_list() -> None:
    offenders = [path.name for path in sorted(TOOLS_DIR.glob("*.py")) if "setattr(settings_module" in path.read_text()]
    assert not offenders, (
        f"{offenders} reset descape.settings globals by hand, which is how eight copies drifted -- "
        "call testkit.settings_isolation.isolate_settings() instead."
    )


def test_isolate_settings_redirects_both_paths_and_clears_every_memo(tmp_path: Path, monkeypatch) -> None:
    import descape.asset_source as asset_source_module

    monkeypatch.setattr(settings, "_dark_mode", True)
    monkeypatch.setattr(settings, "_ruler_label_font_px", 99)
    written = isolate_settings(tmp_path)

    assert written == tmp_path / "config.yaml"
    assert written == settings.CONFIG_PATH
    assert written == asset_source_module.CONFIG_PATH
    for name in MEMOIZED_GLOBALS:
        assert getattr(settings, name) is None, name


def test_isolate_settings_can_leave_asset_source_alone(tmp_path: Path) -> None:
    """tools/gen_terrain_browser_eyeball.py depends on this: get_install_path()
    reads asset_source's own CONFIG_PATH, and the real install has to stay
    visible or there are no .dds swatches to look at."""
    import descape.asset_source as asset_source_module

    before = asset_source_module.CONFIG_PATH
    isolate_settings(tmp_path, redirect_asset_source=False)

    assert tmp_path / "config.yaml" == settings.CONFIG_PATH
    assert before == asset_source_module.CONFIG_PATH
