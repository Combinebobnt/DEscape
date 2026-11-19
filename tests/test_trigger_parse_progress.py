"""The first trigger parse's runner hook, progress, cancel and re-entrancy
(TASK-357, descape/trigger_parse_progress.py and scenario_io.parse_triggers()).

Qt-free: a runner here is a plain bound method. The viewer's own runner and
cancellable sites are tests/test_trigger_parse_progress_viewer.py."""

from __future__ import annotations

import gc
import hashlib

import pytest
from AoE2ScenarioParser.objects.data_objects.trigger import Trigger
from AoE2ScenarioParser.scenarios.scenario_store import getters
from AoE2ScenarioParser.sections.aoe2_file_section import AoE2FileSection

from descape import library_compat, scenario_io
from descape.trigger_parse_progress import TriggerParseCancelled

import conftest

FIXTURE = conftest.ROOT / "tests" / "fixtures" / "triggers_120x120.aoe2scenario"
_SKIP = {"_uuid", "_instance_number_history", "conditions", "effects"}


class _Runner:
    """A first-parse runner calling attempt(report); held weakly by scenario_io, so keep a reference."""

    def __init__(self, report):
        self.report = report

    def run(self, loaded, attempt):
        return attempt(self.report)


@pytest.fixture(autouse=True)
def _no_runner():
    scenario_io.set_first_parse_runner(None)
    yield
    scenario_io.set_first_parse_runner(None)


def _fields(obj) -> tuple:
    from AoE2ScenarioParser.exceptions.asp_exceptions import UnsupportedAttributeError

    names = [name for name in obj._get_object_attrs() if name not in _SKIP]
    names += [name for name in ("_quantity_int", "_quantity_float", "variable") if hasattr(obj, name)]
    values = []
    for name in sorted(set(names)):
        try:
            values.append((name, repr(getattr(obj, name, None))))
        except UnsupportedAttributeError:
            values.append((name, "<unsupported>"))
    return tuple(values)


def _snapshot(loaded) -> list:
    manager = scenario_io.parse_triggers(loaded)
    structs = loaded._scenario.sections["Triggers"].retriever_map["trigger_data"].data or []
    return [
        (
            hashlib.sha256(struct.get_data_as_bytes()).hexdigest(),
            _fields(trigger),
            tuple(_fields(c) for c in trigger.conditions),
            tuple(_fields(e) for e in trigger.effects),
        )
        for trigger, struct in zip(manager.triggers, structs, strict=True)
    ]


def _originals():
    return AoE2FileSection._create_struct, getters.get_scenario_version


def _assert_restored(originals) -> None:
    assert _originals() == originals, "a wrapper or the version memo outlived the parse"
    assert "construct" not in vars(Trigger), "the Trigger.construct wrapper outlived the parse"


def test_a_parse_with_progress_equals_one_without() -> None:
    _check_progress_parse(FIXTURE)


@pytest.mark.test_map
def test_a_parse_with_progress_equals_one_without_on_the_test_map() -> None:
    _check_progress_parse(conftest.TEST_MAP_PATH)


def _check_progress_parse(path) -> None:
    plain = _snapshot(scenario_io.load_map_and_units(str(path)))
    seen: list[tuple[str, int, int]] = []
    runner = _Runner(lambda *event: seen.append(event))
    scenario_io.set_first_parse_runner(runner.run)
    originals = _originals()
    observed = _snapshot(scenario_io.load_map_and_units(str(path)))
    assert observed == plain
    _assert_restored(originals)
    n = len(plain)
    assert n > 0, "vacuous: the fixture has no triggers"
    for phase in ("read", "build"):
        events = [(done, total) for p, done, total in seen if p == phase]
        assert events == [(i, n) for i in range(1, n + 1)], f"{phase} did not count 1..{n}"
    assert [p for p, _, _ in seen] == ["read"] * n + ["build"] * n


@pytest.mark.parametrize("phase", ["read", "build"])
def test_a_cancel_leaves_the_document_and_classes_as_never_parsed(phase, monkeypatch) -> None:
    dumps = []
    real_dump = AoE2FileSection.get_byte_structure_as_string
    monkeypatch.setattr(AoE2FileSection, "get_byte_structure_as_string", lambda s: dumps.append(s) or real_dump(s))

    def report(p, done, total):
        if p == phase and done == 2:
            raise TriggerParseCancelled

    runner = _Runner(report)
    scenario_io.set_first_parse_runner(runner.run)
    originals = _originals()
    loaded = scenario_io.load_map_and_units(str(FIXTURE))
    with pytest.raises(TriggerParseCancelled):
        scenario_io.parse_triggers(loaded)
    assert not dumps, "the library hex-dumped the partial parse on the way out"
    assert loaded.trigger_read_supported is None and loaded._trigger_manager is None
    assert "Triggers" not in loaded._scenario.sections and "Files" not in loaded._scenario.sections
    for cls in library_compat.POISONED_CLASSES:
        assert library_compat.class_state_delta(cls) == ([], []), cls.__name__
    _assert_restored(originals)
    scenario_io.set_first_parse_runner(None)
    assert _snapshot(loaded) == _snapshot(scenario_io.load_map_and_units(str(FIXTURE)))


def test_a_library_failure_keeps_todays_state_and_restores_the_wrappers() -> None:
    runner = _Runner(lambda *event: None)
    scenario_io.set_first_parse_runner(runner.run)
    originals = _originals()
    loaded = scenario_io.load_map_and_units(str(FIXTURE))
    loaded.trigger_tail = loaded.trigger_tail[:200]
    assert scenario_io.parse_triggers(loaded) is None
    assert loaded.trigger_read_supported is False
    assert "Triggers" in loaded._scenario.sections, "the partial section is today's state"
    _assert_restored(originals)


def test_a_parse_from_inside_the_callback_raises_and_the_outer_parse_completes() -> None:
    other = scenario_io.load_map_and_units(str(FIXTURE))
    raised: list[BaseException] = []

    def report(p, done, total):
        if p == "read" and done == 1:
            try:
                scenario_io.parse_triggers(other)
            except RuntimeError as e:
                raised.append(e)

    runner = _Runner(report)
    scenario_io.set_first_parse_runner(runner.run)
    loaded = scenario_io.load_map_and_units(str(FIXTURE))
    observed = _snapshot(loaded)
    assert len(raised) == 1, "the nested parse did not raise RuntimeError"
    assert other.trigger_read_supported is None, "the nested call touched its document"
    scenario_io.set_first_parse_runner(None)
    assert observed == _snapshot(scenario_io.load_map_and_units(str(FIXTURE)))


def test_with_no_runner_the_parse_runs_gc_paused(monkeypatch) -> None:
    enabled: list[bool] = []
    real = scenario_io._parse_trigger_sections

    def recording(loaded, progress):
        enabled.append(gc.isenabled())
        assert progress is None
        return real(loaded, progress)

    monkeypatch.setattr(scenario_io, "_parse_trigger_sections", recording)
    assert gc.isenabled()
    loaded = scenario_io.load_map_and_units(str(FIXTURE))
    assert scenario_io.parse_triggers(loaded) is not None
    assert enabled == [False] and gc.isenabled()


def test_a_dead_runner_is_skipped() -> None:
    runner = _Runner(lambda *event: (_ for _ in ()).throw(AssertionError("a dead runner ran")))
    scenario_io.set_first_parse_runner(runner.run)
    del runner
    gc.collect()
    assert scenario_io.parse_triggers(scenario_io.load_map_and_units(str(FIXTURE))) is not None
