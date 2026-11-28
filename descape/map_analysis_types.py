"""Map Analysis's shared result types and helpers.

Split out of descape/map_analysis.py so the trigger checks
(descape/map_analysis_triggers.py) can share them without an import cycle.
map_analysis re-exports every name here, so `map_analysis.Finding` and the
rest keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from descape.scenario_io import LoadedScenario
from descape.unit_references import all_units

Severity = Literal["error", "warning", "info"]

# Per-check cap, so a badly broken map can't build a 100k-row dialog.
MAX_FINDINGS_PER_CHECK = 200

_TRIGGERS_UNAVAILABLE = "triggers could not be parsed for this file"


@dataclass(frozen=True)
class Finding:
    message: str
    severity: Severity
    tile: tuple[int, int] | None = None
    unit_key: tuple[int, int] | None = None  # (player_id, reference_id)
    trigger_index: int | None = None  # a trigger list index, what TriggerPanel.select_trigger() takes
    trigger_entry: tuple[str, int] | None = None  # ("condition" or "effect", stored index) in that trigger


@dataclass(frozen=True)
class CheckResult:
    label: str
    findings: tuple[Finding, ...] = ()
    unavailable: str = ""
    # TASK-096: False shows the findings but keeps them out of the headline, where the check counts as clean.
    counted: bool = True

    @property
    def is_clean(self) -> bool:
        return not self.unavailable and not self.findings


@dataclass(frozen=True)
class AnalysisReport:
    results: tuple[CheckResult, ...]

    @property
    def finding_count(self) -> int:
        return sum(len(r.findings) for r in self.results if r.counted)

    @property
    def headline(self) -> str:
        clean = sum(1 for r in self.results if r.is_clean or (not r.counted and not r.unavailable))
        unavailable = sum(1 for r in self.results if r.unavailable)
        findings = self.finding_count
        detail = f"{findings} finding{'' if findings == 1 else 's'}"
        if unavailable:
            detail += f", {unavailable} unavailable"
        return f"Clean on {clean} of {len(self.results)} checks: {detail}"


def _capped(label: str, findings: list[Finding]) -> CheckResult:
    if len(findings) > MAX_FINDINGS_PER_CHECK:
        extra = len(findings) - MAX_FINDINGS_PER_CHECK
        severity = findings[0].severity
        findings = [*findings[:MAX_FINDINGS_PER_CHECK], Finding(f"...and {extra} more not listed", severity)]
    return CheckResult(label, tuple(findings))


def _all_units(loaded: LoadedScenario):
    """(player_id, unit) over the raw nine lists, index 0 = GAIA. Never
    unit_pick.build_index(), which drops off-map units by design."""
    return all_units(loaded)


def _read(obj, attribute: str):
    try:
        return getattr(obj, attribute, None)
    except Exception:  # noqa: BLE001 -- the library's version-gated properties raise
        return None


def _owner(player_id: int) -> str:
    return "GAIA" if player_id == 0 else f"Player {player_id}"


def _trigger_label(trigger) -> str:
    disabled = "" if _read(trigger, "enabled") else ", disabled"
    return f'Trigger {trigger.trigger_id} "{trigger.name}"{disabled}'
