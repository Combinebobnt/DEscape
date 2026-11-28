"""Addons (descape/addons.py, addon_api.py, registry.py, addon_menu.py).

The claims, in the order the risk runs:

1. **Consent before code.** An addon is imported only once "Run addon?" was
   answered Yes for its exact bytes; changed bytes ask again, a declined or
   unasked one is never imported, and a script's consent is not an addon's.
2. **A failed addon leaves nothing.** One that registers and then raises, or
   calls sys.exit(), has its registrations and module unwound and is reported
   failed, while an earlier addon stays loaded. KeyboardInterrupt propagates.
3. **No leaks into the process.** Module names are namespaced and sys.path is
   unchanged.
4. **A command row is a script run.** It is one history entry labelled
   `Addon "<title>"`, its refusal greys the row, and a raising command fails
   the run instead of crashing.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from descape import addon_api, addons, scripting, settings
from descape.registry import Registry, ShadowedNameError

import conftest


@pytest.fixture(autouse=True)
def _clean_addon_state(monkeypatch):
    monkeypatch.setattr(settings, "_confirmed_scripts", None)
    monkeypatch.setattr(settings, "_confirmed_addons", None)
    monkeypatch.delenv(settings.ENV_NO_ADDONS, raising=False)
    addon_api.ADDON_COMMANDS.reset()
    addons.clear_last_report()
    yield
    addon_api.ADDON_COMMANDS.reset()
    addons.clear_last_report()
    for name in [name for name in sys.modules if name.startswith(addons.MODULE_PREFIX)]:
        del sys.modules[name]


def _forget() -> None:
    """A fresh process's state, for a second load in one test."""
    addon_api.ADDON_COMMANDS.reset()
    for name in [name for name in sys.modules if name.startswith(addons.MODULE_PREFIX)]:
        del sys.modules[name]


def _write(root: Path, name: str, source: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))
    return path


def _command(name: str, title: str = "") -> str:
    return (
        "from descape.addon_api import ADDON_COMMANDS, AddonCommand\n"
        f'ADDON_COMMANDS.register(AddonCommand({name!r}, {title or name!r}, "tip", lambda editor: None))\n'
    )


class _Asker(list):
    def __init__(self, answer: bool = True):
        super().__init__()
        self.answer = answer

    def __call__(self, path, digest):
        self.append((path, digest))
        return self.answer


def _load(root, ask=None, confirmed=None, argv=(), env=None):
    return addons.load_addons([addon_api.ADDON_COMMANDS], root=root, argv=list(argv), env=env or {}, ask=ask, confirmed=confirmed)


def test_registry_refuses_a_taken_name_and_rolls_back_a_batch():
    class Entry:
        def __init__(self, name):
            self.name = name

    reg = Registry("thing")
    reg.register(Entry("a"))
    mark = reg.generation
    reg.register(Entry("b"))
    with pytest.raises(ShadowedNameError):
        reg.register(Entry("a"))
    with pytest.raises(ValueError):
        reg.register(Entry(""))
    reg.rollback_to(mark)
    assert reg.names() == ("a",) and reg.generation > mark


def test_discovery_takes_py_files_and_packages_and_skips_the_rest(tmp_path):
    for name in ("b.py", "_parked.py", ".hidden.py", "notes.txt", "pkg/__init__.py", "_pkg/__init__.py"):
        _write(tmp_path, name, "")
    (tmp_path / "not_a_package").mkdir()
    assert [path.name for path in addons.discover(tmp_path)] == ["b.py", "pkg"]


def test_consent_is_keyed_on_path_and_digest(tmp_path):
    path = _write(tmp_path, "one.py", _command("one"))
    confirmed: set = set()
    ask = _Asker()
    report = _load(tmp_path, ask, confirmed)
    assert [r.status for r in report.records] == [addons.LOADED]
    assert confirmed == {(str(path.resolve()), scripting.content_digest(path))}
    assert addon_api.ADDON_COMMANDS.names() == ("one",)

    # Same bytes: no question. Changed bytes: asked again, and a No runs nothing.
    _forget()
    _load(tmp_path, ask, confirmed)
    assert len(ask) == 1
    _forget()
    path.write_text(path.read_text() + "\n# edited\n")
    ask.answer = False
    report = _load(tmp_path, ask, confirmed)
    assert len(ask) == 2 and report.declined and not addon_api.ADDON_COMMANDS.names()


def test_the_default_ask_refuses(tmp_path):
    _write(tmp_path, "one.py", _command("one"))
    report = _load(tmp_path)
    assert [r.status for r in report.records] == [addons.DECLINED]
    assert not addon_api.ADDON_COMMANDS.names()


def test_a_script_consent_does_not_confirm_the_addon(tmp_path):
    path = _write(tmp_path, "one.py", _command("one"))
    settings.add_confirmed_script(str(path.resolve()), scripting.content_digest(path))
    report = _load(tmp_path, confirmed=settings.get_confirmed_addons())
    assert report.declined and not addon_api.ADDON_COMMANDS.names()


@pytest.mark.parametrize("ending", ["raise ValueError('boom')", "import sys; sys.exit(3)"])
def test_a_failing_addon_is_unwound_and_an_earlier_one_stays(tmp_path, ending):
    _write(tmp_path, "a_good.py", _command("good"))
    _write(tmp_path, "b_bad.py", _command("bad") + f"\n{ending}\n")
    path_before = list(sys.path)
    report = _load(tmp_path, _Asker())
    assert [(r.name, r.status) for r in report.records] == [("a_good.py", "loaded"), ("b_bad.py", "failed")]
    assert report.failed[0].detail and addon_api.ADDON_COMMANDS.names() == ("good",)
    assert addons.MODULE_PREFIX + "b_bad" not in sys.modules
    assert addons.MODULE_PREFIX + "a_good" in sys.modules and "a_good" not in sys.modules
    assert sys.path == path_before
    assert "1 loaded, 1 failed" in addons.summary_text(report)
    lines = addons.report_lines(report)
    assert any("b_bad.py: failed" in line for line in lines)


def test_a_taken_command_name_fails_only_the_second_addon(tmp_path):
    _write(tmp_path, "a.py", _command("same"))
    _write(tmp_path, "b.py", _command("same", "Other"))
    report = _load(tmp_path, _Asker())
    assert [r.status for r in report.records] == [addons.LOADED, addons.FAILED]
    assert "ShadowedNameError" in report.failed[0].message


def test_a_package_imports_its_own_submodules(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "from . import helper\n" + _command("pkg"))
    _write(tmp_path, "pkg/helper.py", "VALUE = 1\n")
    report = _load(tmp_path, _Asker())
    assert report.loaded and addon_api.ADDON_COMMANDS.names() == ("pkg",)


def test_a_file_runs_the_bytes_that_were_hashed(tmp_path):
    path = _write(tmp_path, "one.py", _command("one"))

    def ask(_path, _digest):
        path.write_text(_command("swapped"))  # saved while the box was open
        return True

    _load(tmp_path, ask)
    assert addon_api.ADDON_COMMANDS.names() == ("one",)


def test_a_package_changed_while_it_loads_fails(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "from . import helper\n" + _command("pkg"))
    helper = _write(tmp_path, "pkg/helper.py", "VALUE = 1\n")

    def ask(_path, _digest):
        helper.write_text("VALUE = 2\n")
        return True

    report = _load(tmp_path, ask)
    assert report.failed and "changed while it loaded" in report.failed[0].message
    assert not addon_api.ADDON_COMMANDS.names()
    assert not [name for name in sys.modules if name.startswith(addons.MODULE_PREFIX)]


def test_a_failing_package_unwinds_its_submodules(tmp_path):
    _write(tmp_path, "pkg/__init__.py", "from . import helper\nraise RuntimeError('late')\n")
    _write(tmp_path, "pkg/helper.py", "VALUE = 1\n")
    report = _load(tmp_path, _Asker())
    assert report.failed
    assert not [name for name in sys.modules if name.startswith(addons.MODULE_PREFIX + "pkg")]


def test_two_addons_with_one_module_name_load_only_the_first(tmp_path):
    _write(tmp_path, "foo/__init__.py", _command("from_package"))
    _write(tmp_path, "foo.py", _command("from_file"))
    _write(tmp_path, "a.b.py", _command("dotted"))
    report = _load(tmp_path, _Asker())
    assert [(r.name, r.status) for r in report.records] == [("a.b.py", "failed"), ("foo", "loaded"), ("foo.py", "failed")]
    assert addon_api.ADDON_COMMANDS.names() == ("from_package",)


def test_a_malformed_command_fails_its_addon(tmp_path):
    _write(
        tmp_path,
        "bad.py",
        "from descape.addon_api import ADDON_COMMANDS, AddonCommand\n"
        "ADDON_COMMANDS.register(AddonCommand('p', 'Paint', lambda editor: None, 'tip'))\n",
    )
    report = _load(tmp_path, _Asker())
    assert report.failed and "TypeError" in report.failed[0].message
    assert not addon_api.ADDON_COMMANDS.names()


def test_consent_is_keyed_on_the_resolved_path(tmp_path):
    real = tmp_path / "real"
    path = _write(real, "one.py", _command("one"))
    link = tmp_path / "link"
    link.symlink_to(real)
    ask = _Asker()
    confirmed: set = set()
    _load(link, ask, confirmed)
    assert confirmed == {(str(path.resolve()), scripting.content_digest(path))}
    assert ask[0][0] == path.resolve()


def test_report_a_bug_carries_the_addon_count_and_no_path(tmp_path):
    from urllib.parse import parse_qs, urlsplit

    from descape import help_guide

    _write(tmp_path, "good.py", _command("good"))
    _write(tmp_path, "bad.py", "raise ValueError\n")
    text = addons.count_text(_load(tmp_path, _Asker()))
    assert text == "1 loaded, 1 failed"
    [body] = parse_qs(urlsplit(help_guide.report_bug_url("9.8", "Plan9", "from source", text)).query)["body"]
    assert "Addons: 1 loaded, 1 failed" in body and str(tmp_path) not in body
    [body] = parse_qs(urlsplit(help_guide.report_bug_url("9.8", "Plan9", "from source")).query)["body"]
    assert "Addons" not in body


def test_keyboard_interrupt_propagates(tmp_path):
    _write(tmp_path, "stop.py", "raise KeyboardInterrupt\n")
    with pytest.raises(KeyboardInterrupt):
        _load(tmp_path, _Asker())


@pytest.mark.parametrize(
    ("argv", "env", "switch"),
    [(["--no-addons"], {}, "--no-addons"), ([], {"DESCAPE_NO_ADDONS": "1"}, "DESCAPE_NO_ADDONS")],
)
def test_the_switches_skip_loading(tmp_path, argv, env, switch):
    _write(tmp_path, "one.py", _command("one"))
    ask = _Asker()
    report = _load(tmp_path, ask, argv=argv, env=env)
    assert report.disabled_by == switch and ask == [] and not report.records
    assert addons.count_text(report) == f"off ({switch})"


def test_addon_root_follows_the_variable(tmp_path):
    assert addons.addon_root({addons.ENV_ADDON_PATH: str(tmp_path)}) == tmp_path
    assert addons.addon_root({}).name == "addons"


# -- the window ---------------------------------------------------------------

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "units_120x120.aoe2scenario"


def _rows(window):
    window.addons_menu.aboutToShow.emit()
    rows = []
    for action in window.addons_menu.actions():
        if action.isSeparator() or action is window.open_addons_folder_action:
            break
        rows.append(action)
    return rows


def _register(name, title, run, refusal=None):
    addon_api.ADDON_COMMANDS.register(addon_api.AddonCommand(name, title, f"{title} tip", run, refusal))


def _paint(editor):
    for x in range(3):
        editor.set_terrain(x, 0, 2)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_a_command_row_runs_as_one_history_entry(monkeypatch):
    from descape import script_menu

    shown: list = []
    monkeypatch.setattr(script_menu, "show_error", lambda _w, *args: shown.append(args))
    _register("paint", "Paint Corner", _paint)
    _register("off", "Never", _paint, refusal=lambda editor: "not today" if editor.width else None)
    _register("boom", "Boom", lambda editor: (_paint(editor), 1 / 0))
    window = conftest.blank_window(load=False)
    try:
        window.load_scenario(_FIXTURE)
        rows = {row.text(): row for row in _rows(window)}
        assert set(rows) == {"Paint Corner", "Never", "Boom"}
        assert not rows["Never"].isEnabled() and "not today" in rows["Never"].toolTip()
        before = [tile.terrain_id for tile in window.scenario.map_manager.terrain[:3]]

        rows["Paint Corner"].trigger()
        [record] = window.edit_history.records
        assert record.label == 'Addon "Paint Corner"'
        assert [tile.terrain_id for tile in window.scenario.map_manager.terrain[:3]] == [2, 2, 2]
        window.undo()
        assert [tile.terrain_id for tile in window.scenario.map_manager.terrain[:3]] == before

        records = list(window.edit_history.records)
        rows = {row.text(): row for row in _rows(window)}
        rows["Boom"].trigger()
        assert len(shown) == 1 and "ZeroDivisionError" in shown[0][1]
        assert [tile.terrain_id for tile in window.scenario.map_manager.terrain[:3]] == before
        assert window.edit_history.records == records
    finally:
        conftest.close_window(window)


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
def test_startup_load_asks_persists_and_summarises(tmp_path, monkeypatch):
    from descape import addon_menu

    root = tmp_path / "addons"
    path = _write(root, "one.py", _command("one", "One"))
    monkeypatch.setenv(addons.ENV_ADDON_PATH, str(root))
    monkeypatch.setattr(sys, "argv", ["descape"])
    monkeypatch.setattr(addon_menu, "_loaded", False)
    asked: list = []
    monkeypatch.setattr(addon_menu, "confirm_load", lambda _w, p: asked.append(p) or True)
    # A script's consent for the same bytes is not an addon's.
    settings.add_confirmed_script(str(path.resolve()), scripting.content_digest(path))
    window = conftest.blank_window(load=False)
    try:
        addon_menu.load_startup(window)
        assert asked == [path.resolve()]
        assert (str(path.resolve()), scripting.content_digest(path)) in settings.get_confirmed_addons()
        rows = _rows(window)
        assert [row.text() for row in rows] == ["One"] and not rows[0].isEnabled()  # no scenario open
        summary = window.addons_menu.actions()[-1]
        assert summary.text().startswith("1 loaded") and not summary.isEnabled()
        # Once per process.
        addon_menu.load_startup(window)
        assert asked == [path.resolve()]
    finally:
        conftest.close_window(window)


def _meddle(editor):
    editor.tile(0, 0).terrain_id = 77
    editor.units(1)[0].rotation = 9.0
    return


@pytest.mark.gui
@pytest.mark.skipif(not conftest.PYQT5_AVAILABLE, reason="PyQt5 not importable")
@pytest.mark.parametrize(
    ("refusal", "why"),
    [
        (lambda editor: 1 / 0, "availability check failed: ZeroDivisionError"),
        (lambda editor: sys.exit(2), "availability check failed: SystemExit"),
        (_meddle, "availability check changed the scenario"),
    ],
)
def test_a_misbehaving_refusal_greys_its_row_and_changes_nothing(refusal, why):
    _register("bad", "Bad Check", _paint, refusal=refusal)
    _register("fine", "Fine", _paint)
    window = conftest.blank_window(load=False)
    try:
        window.load_scenario(_FIXTURE)
        loaded = window.scenario
        tile0 = loaded.map_manager.terrain[0].terrain_id
        image = scripting.capture_image(loaded)
        rows = {row.text(): row for row in _rows(window)}
        assert set(rows) == {"Bad Check", "Fine"} and rows["Fine"].isEnabled()
        assert not rows["Bad Check"].isEnabled() and why in rows["Bad Check"].toolTip()
        assert loaded.map_manager.terrain[0].terrain_id == tile0
        assert not scripting.image_differs(image, loaded)
        assert window.edit_history.records == [] and window.addons_menu.actions()[-1].text()
    finally:
        conftest.close_window(window)
