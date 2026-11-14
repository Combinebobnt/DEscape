"""TriggerPanel's status colours (GH #166): StatusStyle, the status icon and tooltip,
and computing and applying each trigger's and entry's status."""

from __future__ import annotations

import contextlib
import dataclasses

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QHeaderView,
)

from descape import (
    trigger_status,
)
from descape.trigger_model import (
    entry_display_order,
)
from descape.trigger_status import EntryStatus, Status
from descape.viewer_dialogs import legible_status_color


@dataclasses.dataclass(frozen=True)
class StatusStyle:
    """How rows show their trigger_status (GH #166). The window pushes the
    settings' values through TriggerPanel.status_style; these defaults are
    the settings' own, for a panel built on its own."""

    marker: str = "color"  # "color", "icon", "both" or "off"
    ok: str = "#4caf50"
    problem: str = "#e05252"
    color_ok_rows: bool = True  # off: an OK row gets neither brush nor icon, and an empty section no dash
    color_sections: bool = False  # TASK-317: a section row's brush; its icon shows either way
    empty: str = "#8a8a8a"  # an empty section's dash, fixed

    @property
    def colours(self) -> bool:
        return self.marker in ("color", "both")

    @property
    def icons(self) -> bool:
        return self.marker in ("icon", "both")


# A PROBLEM row's tooltip lists at most this many reasons, then a count.
_STATUS_TOOLTIP_LINES = 12


def status_icon(status: Status, hex_str: str) -> QIcon:
    """A tick (OK), a cross (PROBLEM) or a dash (EMPTY, a section with no
    entries) in `hex_str`. The shape carries the meaning, so the icon works
    with no colour vision at all."""
    icon = QIcon()
    for size in (16, 32):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(hex_str), size * 0.16, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if status is Status.OK:
            points = ((0.16, 0.55), (0.40, 0.78), (0.84, 0.24))
            painter.drawPolyline(QPolygonF([QPointF(x * size, y * size) for x, y in points]))
        elif status is Status.EMPTY:
            painter.drawLine(QPointF(0.24 * size, 0.5 * size), QPointF(0.76 * size, 0.5 * size))
        else:
            low, high = 0.24 * size, 0.76 * size
            painter.drawLine(QPointF(low, low), QPointF(high, high))
            painter.drawLine(QPointF(high, low), QPointF(low, high))
        painter.end()
        icon.addPixmap(pixmap)
    return icon


def status_tooltip(status: EntryStatus) -> str:
    reasons = list(status.reasons)
    if len(reasons) > _STATUS_TOOLTIP_LINES:
        extra = len(reasons) - _STATUS_TOOLTIP_LINES + 1
        reasons = [*reasons[: _STATUS_TOOLTIP_LINES - 1], f"... and {extra} more"]
    return "\n".join(reasons)


@dataclasses.dataclass(frozen=True, slots=True)
class _StatusMemo:
    """One trigger's statuses, kept across same-document populates (GH #166)."""

    trigger: object  # held, so its id() cannot be recycled while memoized
    statuses: trigger_status.TriggerStatuses
    unit_refs: bool  # an entry holds a placed-unit id: a unit edit can flip it
    trigger_targets: tuple  # the trigger ids its entries name: a count change can flip one
    clean: bool  # unedited when computed (not in edited_trigger_indices())


class _TriggerPanelStatusMixin:
    """TriggerPanel's status colours (trigger_panel.py's docstring has the split's rules)."""

    # -- status colours (GH #166) --------------------------------------------
    #
    # Status never goes into a row's text, so filtering and the tag facet are
    # unaffected. Every path that can change a trigger's completeness
    # recomputes that whole trigger: _refresh_labels() (field edits),
    # refresh_entries() (entry add/delete/cut/paste/retype), refresh_trigger()
    # (the window's group writes), refresh_unit_reference_labels() (unit
    # edits) and show_scenario() (everything else). show_scenario() reuses the
    # memo of every trigger that cannot have changed (_compute_all_statuses()):
    # a full pass is 0.4 s on an 18k-trigger file, measured (0.9 s before per-type plans).

    def _reset_statuses(self) -> None:
        self._status = {}
        self._unit_ref_triggers = set()
        self._status_stale = False
        self._status_memo = {}
        self._status_memo_vocabulary = None
        self._status_memo_count = -1

    def _status_context(self, manager) -> trigger_status.StatusContext:
        return self.status_context(self._vocabulary, frozenset(range(len(manager.triggers))))

    def _compute_all_statuses(self, manager, same_document: bool = False) -> None:
        """Every trigger's statuses, re-evaluating only what can have changed.

        On a same-document populate a trigger keeps its memo when it is the
        same object and was unedited both then and now: TriggerEditModel's save
        invariant is that an unedited trigger still matches its parsed bytes.
        A trigger-count change re-evaluates every trigger naming an id the
        count crossed, and a stale unit index (_status_stale) every one
        holding a unit id.
        """
        edited = self.edited_trigger_indices()
        memo = self._status_memo if same_document and self._status_memo_vocabulary is self._vocabulary else {}
        low, high = sorted((self._status_memo_count, len(manager.triggers)))
        units_stale = self._status_stale
        self._reset_statuses()
        context = None
        for index, trigger in enumerate(manager.triggers):
            old = memo.get(id(trigger))
            if (
                old is None
                or edited is None
                or not old.clean
                or index in edited
                or any(low <= target < high for target in old.trigger_targets)
                or (units_stale and old.unit_refs)
            ):
                if context is None:
                    context = self._status_context(manager)
                self._recompute_trigger(manager, index, context, clean=edited is not None and index not in edited)
                continue
            self._status_memo[id(trigger)] = old
            self._status[index] = old.statuses
            if old.unit_refs:
                self._unit_ref_triggers.add(index)
        self._status_memo_vocabulary = self._vocabulary
        self._status_memo_count = len(manager.triggers)

    def _recompute_trigger(
        self, manager, index: int, context=None, clean: bool = False
    ) -> trigger_status.TriggerStatuses:
        """Re-evaluate one trigger into the cache, its memo and its unit-reference
        membership. `clean` says it is unedited now; False (the safe default)
        makes the next populate re-evaluate it."""
        if context is None:
            context = self._status_context(manager)
        trigger = manager.triggers[index]
        statuses, unit_refs, trigger_targets = trigger_status.evaluate_with_references(trigger, context)
        self._status[index] = statuses
        self._status_memo[id(trigger)] = _StatusMemo(trigger, statuses, unit_refs, trigger_targets, clean)
        if unit_refs:
            self._unit_ref_triggers.add(index)
        else:
            self._unit_ref_triggers.discard(index)
        return statuses

    def _group_statuses(self, trigger, statuses: trigger_status.TriggerStatuses) -> dict[str, EntryStatus]:
        """The Conditions and Effects rows' rollups. An empty Effects group on a
        trigger that needs_effects() carries the trigger's own "has no effects"."""
        if self._vocabulary is None:
            return {"condition": trigger_status.UNCHECKED, "effect": trigger_status.UNCHECKED}
        groups = {}
        for kind, entry_statuses in (("condition", statuses.conditions), ("effect", statuses.effects)):
            entries = list(self._read(trigger, f"{kind}s") or [])
            # GH #213: numbered and listed by display slot, as the rows are.
            order = entry_display_order(trigger, kind) if entries else []
            groups[kind] = trigger_status.group_status(
                (trigger_status.entry_label(kind, slot, entries[i], self._vocabulary), entry_statuses[i])
                for slot, i in enumerate(order)
                if i < len(entry_statuses)
            )
        if not statuses.effects and trigger_status.needs_effects(trigger):
            groups["effect"] = EntryStatus(Status.PROBLEM, ("has no effects",))
        return groups

    def _refresh_status_style(self) -> None:
        style = self.status_style()
        if style != self._status_style or not self._status_icons:
            self._status_style = style
            self._status_icons = {
                Status.OK: status_icon(Status.OK, style.ok),
                Status.PROBLEM: status_icon(Status.PROBLEM, style.problem),
                Status.EMPTY: status_icon(Status.EMPTY, style.empty),
            }
        self.tree.setColumnHidden(self._COL_STATUS, not style.icons)

    @contextlib.contextmanager
    def _batched_trigger_restyle(self):
        """Each setData on the shown trigger tree re-sizes its ResizeToContents
        columns over every row: 2.3 s to restyle old-allies-final-v2's 590 rows,
        measured. Interactive for the batch, then one resize when restored."""
        header = self.tree.header()
        modes = [header.sectionResizeMode(c) for c in range(header.count())]
        for c in range(header.count()):
            header.setSectionResizeMode(c, QHeaderView.Interactive)
        try:
            yield
        finally:
            for c, mode in enumerate(modes):
                header.setSectionResizeMode(c, mode)

    def _apply_status(
        self,
        item,
        status: EntryStatus | None,
        col: int,
        force: bool = False,
        *,
        icon_col: int | None = None,
        divider: bool = False,
    ) -> None:
        """Style one row: the brush on `col`, the icon on `icon_col` (default
        `col`) and a tooltip on both, listing a PROBLEM row's reasons.
        UNCHECKED and NONE stay unstyled, and so does an OK row with
        color_ok_rows off, in every marker mode. An EMPTY section gets a dash
        while OK rows are marked, and never a brush; any other `divider` row
        gets its brush only with color_sections (TASK-317). A row already
        showing `status` is skipped (see _batched_trigger_restyle() for the
        cost); `force` is for a style or theme change, which every row
        follows. The brush is the colour shaded to the floor on its tree's Base."""
        if (
            not force
            and item.data(col, self._STATUS_ROLE) == status
            and bool(item.data(col, self._DIVIDER_ROLE)) == divider
        ):
            return
        item.setData(col, self._STATUS_ROLE, status)
        item.setData(col, self._DIVIDER_ROLE, divider)
        style = self._status_style
        kind = status.status if status is not None else Status.NONE
        marked = kind is Status.PROBLEM or (kind in (Status.OK, Status.EMPTY) and style.color_ok_rows)
        brush = None
        if marked and style.colours and kind is not Status.EMPTY and (not divider or style.color_sections):
            # A trigger row is styled before it is added to self.tree.
            base = (item.treeWidget() or self.tree).palette().color(QPalette.Active, QPalette.Base).name()
            brush = QBrush(QColor(legible_status_color(style.problem if kind is Status.PROBLEM else style.ok, base)))
        item.setData(col, Qt.ForegroundRole, brush)
        icon = self._status_icons.get(kind) if marked and style.icons else None
        item.setIcon(col if icon_col is None else icon_col, icon if icon is not None else QIcon())
        if kind is Status.PROBLEM:
            tip = status_tooltip(status)
        else:
            tip = self._EMPTY_SECTION_TIP if kind is Status.EMPTY else ""
        for tip_col in {col, col if icon_col is None else icon_col}:
            item.setToolTip(tip_col, tip)

    def _style_trigger_item(self, item, index: int) -> None:
        """A trigger-list row from the cache: icon in the status column, brush on the name."""
        statuses = self._status.get(index)
        self._apply_status(
            item,
            statuses.trigger if statuses is not None else None,
            self._COL_NAME,
            icon_col=self._COL_STATUS,
            divider=statuses is not None and statuses.divider,
        )

    def _style_trigger_row(self, index: int) -> None:
        item = self._item_for_index.get(index)
        if item is not None:
            self._style_trigger_item(item, index)

    def _style_entry_tree(self, manager, index: int | None) -> None:
        """The detail tree's Trigger, Conditions, Effects and entry rows, for
        trigger `index`, which the tree must be showing."""
        if manager is None or index is None or not 0 <= index < len(manager.triggers):
            return
        statuses = self._status.get(index)
        if statuses is None:
            statuses = self._recompute_trigger(manager, index)
        groups = self._group_statuses(manager.triggers[index], statuses)
        per_kind = {"condition": statuses.conditions, "effect": statuses.effects}
        for row in range(self.entry_tree.topLevelItemCount()):
            top = self.entry_tree.topLevelItem(row)
            data = top.data(0, Qt.UserRole)
            if not data:
                continue
            if data[0] == "trigger":
                self._apply_status(top, statuses.trigger, 0, divider=statuses.divider)
                continue
            kind = data[1]
            self._apply_status(top, groups.get(kind), 0)
            children = per_kind.get(kind, ())
            for c in range(top.childCount()):
                child = top.child(c)
                entry_index = child.data(0, Qt.UserRole)[1]
                self._apply_status(child, children[entry_index] if 0 <= entry_index < len(children) else None, 0)

    def apply_status_style(self) -> None:
        """Re-read status_style and restyle every row from its stored status,
        with no recompute. The settings dialog pushes this live."""
        self._refresh_status_style()
        with self._batched_trigger_restyle():
            for item in self._item_for_index.values():
                self._apply_status(
                    item,
                    item.data(self._COL_NAME, self._STATUS_ROLE),
                    self._COL_NAME,
                    force=True,
                    icon_col=self._COL_STATUS,
                    divider=bool(item.data(self._COL_NAME, self._DIVIDER_ROLE)),
                )
        for row in range(self.entry_tree.topLevelItemCount()):
            top = self.entry_tree.topLevelItem(row)
            for item in (top, *(top.child(c) for c in range(top.childCount()))):
                divider = bool(item.data(0, self._DIVIDER_ROLE))
                self._apply_status(item, item.data(0, self._STATUS_ROLE), 0, force=True, divider=divider)
        # An icon widens the Item column, which is sized once per populate.
        self._fit_entry_columns()

    def _restyle_for_palette(self) -> None:
        self._status_restyle_pending = False
        self.apply_status_style()

    def status_of(self, trigger_index: int) -> trigger_status.TriggerStatuses | None:
        """The cached statuses of one trigger, for tests and tools."""
        return self._status.get(trigger_index)
