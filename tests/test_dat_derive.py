"""GH #323: every lenient guard in tools/ names a warning kind the viewer has words for.

tools/_game_root.WARNING_KINDS is the generators' list of kinds, and
descape/data_mod_menu.WARNING_TEXT the user's wording for each plus "other"
(dat_cache's fallback for an item it cannot read). The two live apart because
the generators import nothing from descape/, so this pins both directions and
every refuse() call site."""

from __future__ import annotations

import ast
from pathlib import Path

from descape import dat_cache, dat_derive

TOOLS = Path(dat_derive.__file__).resolve().parent.parent / "tools"


def _refuse_calls() -> list[tuple[str, int, ast.Call]]:
    found = []
    for path in sorted(TOOLS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            func = getattr(node, "func", None)
            if (
                isinstance(node, ast.Call)
                and isinstance(func, ast.Attribute)
                and func.attr == "refuse"
                and isinstance(func.value, ast.Name)
                and func.value.id == "_game_root"
            ):
                found.append((path.name, node.lineno, node))
    return found


def test_every_refuse_call_passes_a_known_literal_kind():
    kinds = dat_derive.warning_kinds()
    calls = _refuse_calls()
    assert calls, "the scan found no _game_root.refuse() call at all"
    bad = []
    for name, line, call in calls:
        kind = next((keyword.value for keyword in call.keywords if keyword.arg == "kind"), None)
        if not (isinstance(kind, ast.Constant) and kind.value in kinds):
            bad.append(f"{name}:{line} kind={ast.unparse(kind) if kind is not None else None}")
    assert bad == [], "each refuse() call needs kind= one of _game_root.WARNING_KINDS, as a literal"


def test_every_kind_has_user_text_and_nothing_else_does():
    from descape import data_mod_menu

    assert dat_cache.OTHER_KIND not in dat_derive.warning_kinds(), "'other' is the viewer's fallback, never a generator's"
    assert set(data_mod_menu.WARNING_TEXT) == set(dat_derive.warning_kinds()) | {dat_cache.OTHER_KIND}
    for kind, text in data_mod_menu.WARNING_TEXT.items():
        assert text.subject in ("unit", "terrain", "none"), kind
        assert "{n}" not in text.one, kind
