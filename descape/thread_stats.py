"""Calling-thread stall attribution, independent of Qt and Perf Trace state."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from itertools import pairwise, product
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

DOMINANT_SHARE = 0.5
SCHEDSTAT = Path("/proc/thread-self/schedstat")


@dataclass(frozen=True)
class ThreadSample:
    wall: float
    cpu_ns: int
    runq_ns: int | None = None
    maj: int | None = None
    min: int | None = None
    vcsw: int | None = None
    ivcsw: int | None = None
    start: float | None = None  # None denotes a zero-span legacy/synthetic sample.
    maj_before: int | None = None


@dataclass(frozen=True)
class Delta:
    """Raw wall envelope with conservative timing skew and in-work fault bounds."""

    wall_ms: float
    cpu_ms: float | None
    runq_ms: float | None = None
    maj: int | None = None
    min: int | None = None
    vcsw: int | None = None
    ivcsw: int | None = None
    sampling_ms: float = 0.0
    maj_bounds: tuple[int | None, int | None] | None = None
    wall_start: float | None = None
    wall_end: float | None = None

    @staticmethod
    def _kind(wall, cpu, runq, maj) -> str:
        if wall <= 0 or cpu is None:
            return "mixed"
        half = wall * DOMINANT_SHARE
        if cpu >= half:
            return "cpu"
        if runq is None:
            return "off-cpu"
        if runq >= half:
            return "runq"
        if max(0.0, wall - cpu - runq) >= half:
            if maj is None:
                return "off-cpu"
            return "fault" if maj > 0 else "blocked"
        return "mixed"

    def nominal_kind(self) -> str:
        return self._kind(self.wall_ms, self.cpu_ms, self.runq_ms, self.maj)

    def possible_kinds(self) -> list[str]:
        def bounds(value):
            if value is None or self.sampling_ms == 0:
                return (value,)
            return (max(0.0, value - self.sampling_ms), value + self.sampling_ms)

        faults = (self.maj,)  # Direct legacy deltas preserve their original nominal classifier.
        if self.maj_bounds is not None:
            lo, hi = self.maj_bounds
            faults = (0, 1) if lo is None or hi is None else ((1,) if lo > 0 else ((0,) if hi == 0 else (0, 1)))
        return sorted({self._kind(w, c, r, m) for w, c, r, m in product(
            bounds(self.wall_ms), bounds(self.cpu_ms), bounds(self.runq_ms), faults
        )})

    def kind(self) -> str:
        kinds = self.possible_kinds()
        return kinds[0] if len(kinds) == 1 else "uncertain"

    def eligible(self) -> bool:
        """Diagnostic class-selection eligibility, not a G2/env scoring exemption."""
        kind = self.kind()
        return (self.wall_ms > 0 and self.cpu_ms is not None and kind not in ("uncertain", "off-cpu")
                and (self.runq_ms is not None or kind == "cpu"))

    def text(self) -> str:
        parts = [f"{name} {value:.1f}ms" for name, value in (("cpu", self.cpu_ms), ("runq", self.runq_ms))
                 if value is not None]
        parts += [f"{name} {getattr(self, name)}" for name in ("maj", "min", "vcsw", "ivcsw")
                  if getattr(self, name) is not None]
        if self.maj_bounds is not None or self.sampling_ms:
            faults = "?..?" if self.maj_bounds is None else "..".join("?" if v is None else str(v) for v in self.maj_bounds)
            parts += [f"sample {self.sampling_ms:.3f}ms maj_bounds {faults} kinds {','.join(self.possible_kinds())}"]
        return " ".join(parts)


def sample(reader=None) -> ThreadSample:
    """Read native-thread schedstat and rusage, or wall/thread CPU only.

    `reader(path)` reads schedstat. The sched_schedstats sysctl does not gate
    sched_info.run_delay; readable counters, including zero, remain available.
    """
    start = time.perf_counter()
    if sys.platform == "linux" and resource is not None and hasattr(resource, "RUSAGE_THREAD"):
        try:
            maj_before = resource.getrusage(resource.RUSAGE_THREAD).ru_majflt
            fields = (reader(SCHEDSTAT) if reader is not None else SCHEDSTAT.read_text()).split()
            cpu, runq = map(int, fields[:2])
            if cpu < 0 or runq < 0:
                raise ValueError("negative schedstat counter")
            usage = resource.getrusage(resource.RUSAGE_THREAD)
            counters = (usage.ru_majflt, usage.ru_minflt, usage.ru_nvcsw, usage.ru_nivcsw)
            return ThreadSample(time.perf_counter(), cpu, runq, *counters, start, maj_before)
        except (OSError, ValueError, AttributeError):
            pass
    cpu = time.thread_time_ns()
    return ThreadSample(time.perf_counter(), cpu, start=start)


def delta(a: ThreadSample, b: ThreadSample) -> Delta:
    """Full wall envelope and bounded endpoint skew; resets are unknown, not zero."""
    def diff(name):
        first, last = getattr(a, name), getattr(b, name)
        return None if first is None or last is None or last < first else last - first

    # schedstat and thread_time are different clocks: don't subtract across a fallback boundary.
    cpu = diff("cpu_ns") if (a.runq_ns is None) == (b.runq_ns is None) else None
    runq = diff("runq_ns")
    start = a.wall if a.start is None else a.start
    end_start = b.wall if b.start is None else b.start
    span = (max(0.0, a.wall - start) + max(0.0, b.wall - end_start)) * 1000
    first_pre = a.maj if a.start is None else a.maj_before
    last_pre = b.maj if b.start is None else b.maj_before
    faults = (first_pre, a.maj, last_pre, b.maj)
    fault_bounds = (None, None)
    if all(v is not None and v >= 0 for v in faults) and all(x <= y for x, y in pairwise(faults)):
        fault_bounds = (max(0, last_pre - a.maj), b.maj - first_pre)
    return Delta(max(0.0, b.wall - start) * 1000, None if cpu is None else cpu / 1e6,
                 None if runq is None else runq / 1e6, *(diff(name) for name in ("maj", "min", "vcsw", "ivcsw")),
                 span, fault_bounds, start, b.wall)
