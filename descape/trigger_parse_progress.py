"""Progress, cancel and a version memo for the first trigger parse (TASK-357).

scenario_io.parse_triggers() enters these around its library calls; nothing
here replicates the library's loop, so the parse result cannot differ by
construction. Qt-free: the viewer supplies the callback.

- version_memo(): getters.get_scenario_version() is called millions of times
  by the parse (retriever_object_link's support checks), always for the one
  uuid, so it is swapped for a dict memo for the block. Its other readers are
  on the save path or outside the parse and never see the memo.
- progress_wrappers(report): two scoped wrappers observe the library's own
  loop. AoE2FileSection._create_struct reports ("read", done, total) after
  each trigger struct the top-level Triggers section creates, and
  Trigger.construct reports ("build", done, total) on each call (only
  top-level triggers go through it; conditions and effects have their own
  classes' construct). It yields the guarded report, which scenario_io
  calls once more with ("end", total, total) after the Files section and
  its pull loop, which report nothing of their own (TASK-669).
- TriggerParseCancelled is what a report callback raises to stop the parse.
  It is a BaseException because the library's set_data_from_generator(),
  _create_struct() and _fill_retriever_with_bytes() catch Exception and
  print the hex of everything read so far before re-raising: on a late
  cancel that would format up to 18k triggers.
- ReportFailed carries any other exception a report callback raises, a
  BaseException for the same reason, so a callback bug passes the misparse
  tuple and scenario_io re-raises the original instead of reading the
  document as having unreadable triggers (TASK-669).

Every context manager restores on every exit. A depoison() inside the block
(a re-entrant call would raise before one, but the library may) can already
have deleted the Trigger.construct wrapper, so it is removed only while still
in place.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

from AoE2ScenarioParser.objects.data_objects.trigger import Trigger
from AoE2ScenarioParser.scenarios.scenario_store import getters
from AoE2ScenarioParser.sections.aoe2_file_section import AoE2FileSection, SectionLevel

# (phase, done, total): phase "read" then "build", done rising by one to total, then one "end".
Report = Callable[[str, int, int], None]
# The status line a cancelled first parse leaves.
CANCELLED_MESSAGE = "Reading triggers cancelled"
# The progress window's label and, behind it, the entered mode's panel line.
READING_MESSAGE = "Reading triggers…"


class TriggerParseCancelled(BaseException):
    """Raised by a progress callback to abandon the parse; never caught by the library."""


class ReportFailed(BaseException):
    """A report callback's own exception (`error`), carried out of the parse untouched."""

    def __init__(self, error: Exception) -> None:
        super().__init__(error)
        self.error = error


def _guarded(report: Report) -> Report:
    def call(phase: str, done: int, total: int) -> None:
        try:
            report(phase, done, total)
        except Exception as e:
            raise ReportFailed(e) from e

    return call


@contextlib.contextmanager
def version_memo() -> Iterator[None]:
    original = getters.get_scenario_version
    memo: dict = {}

    def memoized(uuid):
        try:
            return memo[uuid]
        except KeyError:
            value = memo[uuid] = original(uuid)
            return value

    getters.get_scenario_version = memoized
    try:
        yield
    finally:
        getters.get_scenario_version = original


@contextlib.contextmanager
def progress_wrappers(report: Report) -> Iterator[Report]:
    report = _guarded(report)
    original_create = AoE2FileSection._create_struct
    own_construct = vars(Trigger).get("construct")
    inherited = Trigger.construct.__func__
    state = {"read": 0, "build": 0, "total": 0, "struct": None}

    def create_struct(self, model, igenerator):
        struct = original_create(self, model, igenerator)
        if self.level is SectionLevel.TOP_LEVEL and self.name == "Triggers":
            if state["struct"] is None:
                state["struct"] = self.retriever_map["trigger_data"].datatype.get_struct_name()
                state["total"] = self.retriever_map["number_of_triggers"].data or 0
            if model.name == state["struct"]:
                state["read"] += 1
                report("read", state["read"], state["total"])
        return struct

    def construct(cls, *args, **kwargs):
        obj = inherited(cls, *args, **kwargs)
        if cls is Trigger:
            state["build"] += 1
            report("build", state["build"], state["total"])
        return obj

    wrapper = classmethod(construct)
    AoE2FileSection._create_struct = create_struct
    Trigger.construct = wrapper
    try:
        yield report
    finally:
        AoE2FileSection._create_struct = original_create
        if vars(Trigger).get("construct") is wrapper:
            if own_construct is None:
                del Trigger.construct
            else:
                Trigger.construct = own_construct
