"""Per-file line ceilings for `descape/*.py` and `tests/*.py`.

The big splits (viewer.py, map_view.py, trigger_panel.py, render.py and the
per-area viewer tests) only pay off if the pieces stay small. Before its cut,
`descape/viewer.py` grew 2.7k lines in two weeks with nothing to stop it.

Every file has a ceiling: the default below, or its entry in the table. A
line count is physical lines (newlines in the bytes, same as `wc -l`), so a
file ceiling also bounds the classes inside it.

When this fails, first ask whether the new code belongs in a sibling module
(`tools/split_class.py` moves methods out of a class). If it genuinely
belongs in that file, raise its entry, or add one for a file that crossed the
default, with a dated reason on the line: the failure message prints a line
to paste. Lowering or dropping an entry after a cleanup is always welcome;
the table never tightens itself, so a shrinking file never fails here.
"""

from __future__ import annotations

import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DESCAPE_DEFAULT = 1500
TESTS_DEFAULT = 2500

# Initial ceilings: the size rounded up to the next 100, plus 100.
DESCAPE_CEILINGS = {
    "iso_geometry.py": 2600,  # baseline 2026-11-13
    "map_view.py": 1900,  # baseline 2026-11-13
    "map_view_input.py": 1700,  # baseline 2026-11-13
    "player_fields.py": 2200,  # baseline 2026-11-13
    "render_cache.py": 1700,  # baseline 2026-11-13
    "scenario_io.py": 1800,  # baseline 2026-11-13
    "settings.py": 3500,  # baseline 2026-11-13
    "settings_dialog.py": 2600,  # baseline 2026-11-13
    "trigger_model.py": 1700,  # baseline 2026-11-13
    "unit_model.py": 2200,  # baseline 2026-11-13
    "unit_sprites.py": 3300,  # baseline 2026-11-13
    "units_panel.py": 1700,  # baseline 2026-11-13
    "viewer.py": 1800,  # baseline 2026-11-13
    "viewer_canvas.py": 2200,  # baseline 2026-11-13
    "viewer_options.py": 1700,  # baseline 2026-11-13
    "viewer_region.py": 1900,  # baseline 2026-11-13
    "viewer_triggers.py": 1900,  # baseline 2026-11-13
}

TESTS_CEILINGS = {
    "test_terrain_units_viewer.py": 2900,  # baseline 2026-11-13
    "test_trigger_write_path.py": 3300,  # baseline 2026-11-13
    "test_unit_selection_viewer.py": 2800,  # baseline 2026-11-13
    "test_units_panel.py": 3300,  # baseline 2026-11-13
    "test_wall_run_viewer.py": 3200,  # baseline 2026-11-13
}

_AREAS = (
    ("descape", DESCAPE_DEFAULT, DESCAPE_CEILINGS, "DESCAPE_CEILINGS"),
    ("tests", TESTS_DEFAULT, TESTS_CEILINGS, "TESTS_CEILINGS"),
)


def _line_count(path: Path) -> int:
    return path.read_bytes().count(b"\n")


def _suggested(lines: int) -> int:
    return (lines // 100 + 1) * 100 + 100


def _offenders(folder: str, default: int, table: dict[str, int], table_name: str) -> list[str]:
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    out = []
    for path in sorted((ROOT / folder).glob("*.py")):
        lines = _line_count(path)
        ceiling = table.get(path.name, default)
        if lines <= ceiling:
            continue
        what = "its ceiling" if path.name in table else f"the default {default} (no entry in {table_name})"
        out.append(
            f"{folder}/{path.name}: {lines} lines > {ceiling}, {what}.\n"
            f"  Move code into a sibling module, or if it belongs here, set in {table_name}:\n"
            f'    "{path.name}": {_suggested(lines)},  # {today}: <reason>'
        )
    return out


def _check(area: tuple) -> None:
    offenders = _offenders(*area)
    assert not offenders, (
        "Files over their line ceiling (see tests/test_module_size.py's docstring):\n\n" + "\n\n".join(offenders)
    )


def test_descape_module_sizes():
    _check(_AREAS[0])


def test_test_module_sizes():
    _check(_AREAS[1])


def test_ceiling_tables_are_current():
    problems = []
    for folder, default, table, table_name in _AREAS:
        for name, ceiling in table.items():
            if not (ROOT / folder / name).is_file():
                problems.append(f"{table_name}[{name!r}]: {folder}/{name} does not exist (renamed or deleted? move or drop the entry)")
            if ceiling <= default:
                problems.append(f"{table_name}[{name!r}] = {ceiling} is not above the default {default}; drop the entry")
    assert not problems, "\n".join(problems)
