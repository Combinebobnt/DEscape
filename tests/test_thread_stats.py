"""Native-thread counters and deliberately fixed stall-class boundaries."""

from types import SimpleNamespace

import pytest

from descape import thread_stats as ts


def test_sample_reads_schedstat_and_only_native_thread_rusage(monkeypatch):
    seen = []
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts.time, "perf_counter", lambda: 3.0)
    usage = SimpleNamespace(ru_majflt=2, ru_minflt=7, ru_nvcsw=3, ru_nivcsw=4)
    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: seen.append(scope) or usage))
    got = ts.sample(lambda path: seen.append(path) or "123456789 987654321 42\n")
    assert got == ts.ThreadSample(3.0, 123456789, 987654321, 2, 7, 3, 4, start=3.0, maj_before=2)
    assert seen == [99, ts.SCHEDSTAT, 99]


@pytest.mark.parametrize("contents", ["", "123", "not numbers", "-1 5 0", "1 -5 0"])
def test_malformed_schedstat_uses_portable_fallback(monkeypatch, contents):
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 777)
    got = ts.sample(lambda path: contents)
    assert got.cpu_ns == 777
    assert (got.runq_ns, got.maj, got.min, got.vcsw, got.ivcsw) == (None,) * 5


def test_missing_schedstat_uses_portable_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts, "SCHEDSTAT", tmp_path / "absent")
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 888)
    assert ts.sample().cpu_ns == 888
    assert ts.sample().runq_ns is None


@pytest.mark.parametrize("error", [FileNotFoundError(), PermissionError()], ids=["missing", "unreadable"])
def test_reader_failure_uses_portable_fallback(monkeypatch, error):
    reads = []
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: SimpleNamespace(ru_majflt=0)))
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 1000000)

    def reader(path):
        reads.append(path)
        raise error

    got = ts.sample(reader)
    assert got.cpu_ns == 1000000
    assert (got.runq_ns, got.maj, got.min, got.vcsw, got.ivcsw) == (None,) * 5
    assert reads == [ts.SCHEDSTAT]


def test_readable_schedstat_keeps_zero_and_classifies_runq_without_reading_sysctl(monkeypatch):
    clock = SimpleNamespace(wall=1.0, cpu_ns=1000000, runq_ns=0)
    usage = SimpleNamespace(ru_majflt=0, ru_minflt=0, ru_nvcsw=0, ru_nivcsw=0)
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: usage))
    monkeypatch.setattr(ts.time, "perf_counter", lambda: clock.wall)

    def reader(path):
        assert path == ts.SCHEDSTAT, "sampler must not consult the sched_schedstats sysctl"
        return f"{clock.cpu_ns} {clock.runq_ns} 1\n"

    a = ts.sample(reader)
    assert a.runq_ns == 0 and a.cpu_ns == 1000000
    clock.wall, clock.cpu_ns, clock.runq_ns = 1.1, 2000000, 80000000
    stats = ts.delta(a, ts.sample(reader))
    assert stats.kind() == "runq" and stats.runq_ms == 80


@pytest.mark.parametrize(("first", "last"), [(True, False), (False, True)])
def test_schedstat_availability_transitions_do_not_subtract_different_cpu_clocks(monkeypatch, first, last):
    clock = SimpleNamespace(wall=1.0, available=first)
    usage = SimpleNamespace(ru_majflt=0, ru_minflt=0, ru_nvcsw=0, ru_nivcsw=0)
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: usage))
    monkeypatch.setattr(ts.time, "perf_counter", lambda: clock.wall)
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 1000000)

    def reader(path):
        assert path == ts.SCHEDSTAT
        if not clock.available:
            raise FileNotFoundError
        return "900000000 1000000 1\n"

    a = ts.sample(reader)
    clock.wall, clock.available = 1.1, last
    stats = ts.delta(a, ts.sample(reader))
    assert stats.cpu_ms is None and stats.runq_ms is None and stats.kind() == "mixed"


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_nonlinux_never_reads_proc_or_process_rusage(monkeypatch, platform):
    monkeypatch.setattr(ts.sys, "platform", platform)
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 999)

    def forbidden(*args):
        raise AssertionError("non-Linux sampled Linux counters")

    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=forbidden))
    assert ts.sample(forbidden).cpu_ns == 999
    assert ts.sample(forbidden).maj is None


def test_missing_resource_or_rusage_thread_uses_portable_fallback(monkeypatch):
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts.time, "thread_time_ns", lambda: 55)
    for resource in (None, SimpleNamespace()):
        monkeypatch.setattr(ts, "resource", resource)
        assert ts.sample().cpu_ns == 55
        assert ts.sample().runq_ns is None


def test_delta_subtracts_native_counters_and_formats_units():
    a = ts.ThreadSample(1.0, 1000000, 2000000, 1, 3, 5, 7)
    b = ts.ThreadSample(1.1, 4000000, 82000000, 2, 6, 9, 12)
    got = ts.delta(a, b)
    assert got.wall_ms == pytest.approx(100)
    assert got == ts.Delta(got.wall_ms, 3.0, 80.0, 1, 3, 4, 5, maj_bounds=(1, 1), wall_start=1.0, wall_end=1.1)
    assert got.text() == "cpu 3.0ms runq 80.0ms maj 1 min 3 vcsw 4 ivcsw 5 sample 0.000ms maj_bounds 1..1 kinds runq"
    assert ts.Delta(100, 3).text() == "cpu 3.0ms"


def test_delta_marks_counter_resets_unknown_and_does_not_mix_clock_sources():
    a = ts.ThreadSample(2, 20, 30, 4, 5, 6, 7)
    b = ts.ThreadSample(1, 10, 20, 1, 2, 3, 4)
    reset = ts.delta(a, b)
    assert reset.wall_ms == 0 and reset.cpu_ms is reset.runq_ms is reset.maj is None
    assert reset.maj_bounds == (None, None) and not reset.eligible()
    fallback = ts.ThreadSample(3, 900000000)
    for first, last in ((a, fallback), (fallback, a)):
        got = ts.delta(first, last)
        assert got.cpu_ms is None and got.runq_ms is None and got.maj is None
        assert got.kind() == "mixed"
    assert ts.delta(fallback, ts.ThreadSample(4, 903000000)) == ts.Delta(
        1000, 3, maj_bounds=(None, None), wall_start=3, wall_end=4
    )


@pytest.mark.parametrize("endpoint", ["begin", "end"])
def test_sample_envelope_includes_reader_latency_at_each_endpoint(monkeypatch, endpoint):
    clock = SimpleNamespace(t=1.0, cpu=1000000, delay=0.075 if endpoint == "begin" else 0.0)
    usage = SimpleNamespace(ru_majflt=0, ru_minflt=0, ru_nvcsw=0, ru_nivcsw=0)
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=lambda scope: usage))
    monkeypatch.setattr(ts.time, "perf_counter", lambda: clock.t)

    def reader(path):
        clock.t += clock.delay
        return f"{clock.cpu} 0 1"

    a = ts.sample(reader)
    clock.t += 0.01
    clock.cpu += 1000000
    clock.delay = 0.075 if endpoint == "end" else 0.0
    b = ts.sample(reader)
    stats = ts.delta(a, b)
    assert a.start == 1.0 and b.wall == pytest.approx(1.085)
    assert stats.wall_ms == pytest.approx(85) and stats.sampling_ms == pytest.approx(75)
    assert stats.wall_start == a.start and stats.wall_end == b.wall
    assert stats.kind() == "uncertain" and not stats.eligible()


def test_fallback_encloses_thread_clock_latency(monkeypatch):
    clock = SimpleNamespace(t=1.0)
    monkeypatch.setattr(ts.sys, "platform", "win32")
    monkeypatch.setattr(ts.time, "perf_counter", lambda: clock.t)

    def cpu():
        clock.t += 0.07
        return 1000000

    monkeypatch.setattr(ts.time, "thread_time_ns", cpu)
    got = ts.sample()
    assert got.start == 1.0 and got.wall == pytest.approx(1.07) and got.runq_ns is None


def test_confidence_uses_zero_clipped_full_cartesian_timing_bounds():
    stats = ts.Delta(100, 60, 0, 0, sampling_ms=70, maj_bounds=(0, 0))
    assert stats.nominal_kind() == "cpu"
    assert stats.possible_kinds() == ["blocked", "cpu", "runq"]
    assert stats.kind() == "uncertain" and not stats.eligible()


def test_half_wall_tie_is_only_certain_with_zero_sampling_span():
    assert ts.Delta(100, 50, 0, 0).kind() == "cpu"
    stats = ts.Delta(100, 50, 0, 0, sampling_ms=0.001, maj_bounds=(0, 0))
    assert stats.kind() == "uncertain"
    assert ts.Delta(100, 90, sampling_ms=1, maj_bounds=(None, None)).kind() == "cpu"
    assert ts.Delta(100, 90, sampling_ms=1, maj_bounds=(None, None)).eligible()
    assert not ts.Delta(100, 1, sampling_ms=1, maj_bounds=(None, None)).eligible()
    assert ts.Delta(100, 35, 35, 0, sampling_ms=1, maj_bounds=(0, 0)).kind() == "mixed"


@pytest.mark.parametrize(
    ("first_pre", "first_post", "last_pre", "last_post", "observed", "bounds", "kind"),
    [(0, 0, 0, 1, 1, (0, 1), "uncertain"),  # ending-reader-only major fault
     (0, 1, 1, 1, 0, (0, 1), "uncertain"),  # beginning reader expands the conservative upper bound
     (0, 0, 1, 1, 1, (1, 1), "fault"),
     (0, 0, 0, 0, 0, (0, 0), "blocked"),
     (10, 11, 0, 0, None, (None, None), "uncertain"),
     (2, 1, 3, 3, 2, (None, None), "uncertain"),
     (0, 0, 3, 2, 2, (None, None), "uncertain"),
     (None, 0, 0, 0, 0, (None, None), "uncertain")],
    ids=["ending-reader", "beginning-reader", "in-work", "no-fault", "reset-between", "reset-begin",
         "reset-end", "missing-pre"],
)
def test_fault_confidence_requires_assured_in_work_faults(first_pre, first_post, last_pre, last_post, observed, bounds, kind):
    a = ts.ThreadSample(0.001, 0, 0, first_post, start=0.0, maj_before=first_pre)
    b = ts.ThreadSample(0.200, 2000000, 0, last_post, start=0.199, maj_before=last_pre)
    got = ts.delta(a, b)
    assert got.wall_ms == 200 and got.sampling_ms == pytest.approx(2)
    assert got.maj == observed and got.maj_bounds == bounds and got.kind() == kind
    assert got.eligible() == (kind in ("fault", "blocked"))


def test_sampler_brackets_reader_faults_with_pre_and_post_rusage(monkeypatch):
    clock = SimpleNamespace(t=0.0, maj=0)
    seen = []
    monkeypatch.setattr(ts.sys, "platform", "linux")
    monkeypatch.setattr(ts.time, "perf_counter", lambda: clock.t)

    def usage(scope):
        seen.append(("usage", clock.maj))
        return SimpleNamespace(ru_majflt=clock.maj, ru_minflt=0, ru_nvcsw=0, ru_nivcsw=0)

    def reader(path):
        seen.append(("reader", clock.maj))
        clock.maj += 1
        clock.t += 0.001
        return "2000000 0 1"

    monkeypatch.setattr(ts, "resource", SimpleNamespace(RUSAGE_THREAD=99, getrusage=usage))
    got = ts.sample(reader)
    assert seen == [("usage", 0), ("reader", 0), ("usage", 1)]
    assert got.start == 0.0 and got.wall == 0.001 and got.maj_before == 0 and got.maj == 1


@pytest.mark.parametrize(
    ("cpu", "runq", "maj", "want"),
    [(50, 0, 0, "cpu"), (49.999, 0, 0, "blocked"), (0, 50, 0, "runq"),
     (0, 49.999, 0, "blocked"), (25, 25, 1, "fault"), (25.001, 25, 1, "mixed"),
     (25, 25, 0, "blocked"), (25.001, 25, 0, "mixed"), (50, 50, 1, "cpu"),
     (1, 50, 1, "runq"), (49.999, None, None, "off-cpu"), (50, None, None, "cpu"),
     (1, 1, None, "off-cpu")],
)
def test_kind_exact_half_wall_boundaries(cpu, runq, maj, want):
    assert ts.DOMINANT_SHARE == 0.5
    assert ts.Delta(100, cpu, runq, maj).kind() == want


def test_zero_wall_never_claims_cpu_or_runq_dominance():
    assert ts.Delta(0, 0, 0, 0).kind() == "mixed"
