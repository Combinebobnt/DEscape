"""ViewerWindow's Info page: the file and technical sections, player stats and hover text."""

from __future__ import annotations

from collections import Counter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTreeWidgetItem,
)

from descape import (
    map_analysis,
    player_stats,
)
from descape.scenario_io import (
    xs_attachment,
)
from descape.terrain_palette import name_for_terrain_id
from descape.unit_filter import GAIA_PLAYER_ID
from descape.viewer_common import (
    _SCRIPT_NAME_FIELD,
    HOVER_IDLE_TEXT,
    StatRow,
    _set_player_item,
)


def _xs_info_lines(loaded, pending_name: str | None = None) -> list[str]:
    """The info panel's two XS attachment lines. Honest about what is not
    known yet, like the trigger-tail line above them. `pending_name` is an
    unsaved Script filename edit, shown in place of the file's."""
    script_name, embedded = xs_attachment(loaded)
    if pending_name is not None:
        script_name = pending_name
    if script_name is None:
        name_text = "(not stored before scenario 1.40)"
    else:
        name_text = script_name or "(none)"
    if embedded is not None:
        embedded_text = f"{embedded:,} chars" if embedded else "(none)"
    elif loaded.trigger_read_supported is False:
        embedded_text = "(unknown: Triggers section can't be read)"
    else:
        embedded_text = "(unknown until triggers are parsed)"
    return [f"XS script file: {name_text}", f"Embedded XS: {embedded_text}"]


_ALSO_COUNTED = "also counted above"


def _player_stat_rows(
    c: player_stats.PlayerCounts,
    is_gaia: bool,
    resources: player_stats.ResourceTotals | None,
) -> list[StatRow]:
    """The Player section's rows for one owner, before the Triggers row. Own
    side first; a row of the other side (GAIA-side rows for a player, unit
    and building rows for GAIA) only when non-zero. That hide rule is
    _other_side_visible(), keyed by row."""
    unit_rows = [
        ("Units", c.units, "", ""),
        ("Economy", c.economy_units, "", "Units"),
        ("Military", c.military_units, "", "Units"),
        ("Other", c.other_units, "flags, revealers, markers", "Units"),
        ("Water", c.water_units, _ALSO_COUNTED, "Units"),
        ("Unique", c.unique_units, _ALSO_COUNTED, "Units"),
        ("Heroes", c.hero_units, _ALSO_COUNTED, "Units"),
        ("Buildings", c.buildings, "", ""),
        ("Economy", c.economy_buildings, "", "Buildings"),
        ("Military", c.military_buildings, "", "Buildings"),
        ("Walls & gates", c.walls, "", "Buildings"),
        ("Other", c.other_buildings, "", "Buildings"),
    ]
    gaia_rows = [
        ("Resource piles", c.piles, "", ""),
        ("Gold mines", c.gold_mines, "", "Resource piles"),
        ("Stone mines", c.stone_mines, "", "Resource piles"),
        ("Berry bushes", c.berry_bushes, "", "Resource piles"),
        ("Fish", c.fish, "", "Resource piles"),
        ("Other piles", c.other_piles, "", "Resource piles"),
        ("Trees", c.trees, "", ""),
        ("Animals", c.animals, "", ""),
        ("Cliffs", c.cliffs, "", ""),
        ("Relics", c.relics, "", ""),
    ]
    own, other = (gaia_rows, unit_rows) if is_gaia else (unit_rows, gaia_rows)
    rows: list[StatRow] = [("Placements", f"{c.placements:,}", "", "")]
    if is_gaia and resources is not None:
        rows.append(("Resources on map", "", "all owners; game default amounts, triggers not reflected", ""))
        for label, amount in (
            ("Food", resources.food), ("Wood", resources.wood), ("Gold", resources.gold), ("Stone", resources.stone),
        ):
            rows.append((label, f"{amount:,}", "", "Resources on map"))
    rows += [(label, f"{n:,}", note, parent) for label, n, note, parent in own]
    rows += [(label, f"{n:,}", note, parent) for label, n, note, parent in other if _other_side_visible(n)]
    rows.append(("Eye candy", f"{c.eye_candy:,}", "", ""))
    return rows


def _add_info_row(parent: QTreeWidgetItem, label: str, value: str, note: str) -> QTreeWidgetItem:
    """Appends one info-table row. A row with a note but no value is free text
    ("File: name", "Triggers: not counted yet"), spanned across all three
    columns so it isn't squeezed into the Note column; its tooltip is uncut."""
    if note and not value:
        item = QTreeWidgetItem([f"{label}: {note}"])
        item.setToolTip(0, item.text(0))
        parent.addChild(item)
        item.setFirstColumnSpanned(True)
    else:
        item = QTreeWidgetItem([label, value, note])
        # The full label for an elided Stat, and the only route to a note while Note is hidden.
        item.setToolTip(0, f"{label}: {note}" if note else label)
        if note:
            item.setToolTip(2, note)
        parent.addChild(item)
    item.setData(0, Qt.UserRole, label)
    return item


def _other_side_visible(count: int) -> bool:
    """GH #145's hide rule: a row belonging to the other side shows only when
    non-zero, so a player's sheep or relic still shows but Trees never does."""
    return count > 0


class _ViewerInfoMixin:
    """ViewerWindow's Info page and hover text (viewer.py's docstring has the split's rules)."""

    def _update_info(self) -> None:
        """Both halves of page 0: the static File/Terrain/Technical sections,
        then the Player section via _repopulate_stats_players()."""
        s = self.scenario
        mm = s.map_manager

        # Free text goes in the Note column, so Value stays as narrow as its numbers.
        self._set_info_children("File", [
            ("File", "", s.path.name),
            ("Scenario version", str(s.scenario_version), ""),
            ("Map size", "", f"{mm.map_width} x {mm.map_height}"),
        ])
        terrain_hist = Counter(t.terrain_id for t in mm.terrain)
        total_tiles = len(mm.terrain)
        self._set_info_children("Terrain (top 8)", [
            (name_for_terrain_id(tid), f"{100 * count / total_tiles:.1f}%", "")
            for tid, count in terrain_hist.most_common(8)
        ], numeric=True)
        self._update_technical_info()
        # The trigger-tail/XS rows above and the Player section's trigger row flip together.
        self._repopulate_stats_players()

    def _update_technical_info(self) -> None:
        """The info page's Technical rows alone, also refreshed after a Script
        filename edit or its undo, so the XS line shows the pending name."""
        s = self.scenario
        if s is None:
            return
        technical = [("Trigger tail", f"{len(s.trigger_tail):,}", "bytes (not parsed)")]
        pending_name = None
        if self.option_edits is not None:
            pending_name = self.option_edits.pending_values().get(_SCRIPT_NAME_FIELD)
        for line in _xs_info_lines(s, pending_name):
            label, _, value = line.partition(": ")
            technical.append((label, "", value))
        self._set_info_children("Technical", technical)

    def _set_info_children(self, title: str, rows: list[tuple[str, str, str]], numeric: bool = False) -> None:
        """Replaces one static section's rows. numeric right-aligns the Value column."""
        section = self._info_sections[title]
        section.takeChildren()
        for label, value, note in rows:
            item = _add_info_row(section, label, value, note)
            if numeric:
                item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
        self.info_tree.fit_columns()

    def _info_text(self) -> str:
        """The File, Technical and Terrain sections as the plain-text lines the
        old info box showed ("Label: value"; terrain rows indented), for tests."""
        lines = []
        for title in ("File", "Technical"):
            section = self._info_sections[title]
            for i in range(section.childCount()):
                item = section.child(i)
                if item.isFirstColumnSpanned():
                    lines.append(item.text(0))
                else:
                    lines.append(f"{item.text(0)}: {' '.join(t for t in (item.text(1), item.text(2)) if t)}")
        lines += ["", "Terrain (top 8):"]
        section = self._info_sections["Terrain (top 8)"]
        lines += [f"  {section.child(i).text(0):24s} {section.child(i).text(1)}" for i in range(section.childCount())]
        return "\n".join(lines)

    @staticmethod
    def _stats_label_suffix(player_id: int, active: set[int]) -> str:
        return "" if player_id == GAIA_PLAYER_ID or player_id in active else " (inactive)"

    def _repopulate_stats_players(self) -> None:
        """GH #5's combo: GAIA, the defined players as edited, then any
        inactive slot that still owns placements. Keeps the selected player
        when it survives, then re-renders the stats block."""
        combo = self.stats_player_combo
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if self.scenario is not None:
            active = self._current_active_players()
            units = self.scenario.unit_manager.units
            orphaned = [p for p in range(1, len(units)) if units[p] and p not in active]
            active_set = set(active)
            for player_id in [GAIA_PLAYER_ID, *active, *orphaned]:
                combo.addItem("", player_id)
                _set_player_item(
                    combo, combo.count() - 1, player_id, self._player_labels, self.scenario.player_colors,
                    self._stats_label_suffix(player_id, active_set),
                )
            index = combo.findData(previous) if previous is not None else -1
            if index < 0:
                # Player 1 by default, else GAIA.
                index = 1 if combo.count() > 1 else 0
            combo.setCurrentIndex(index)
        combo.setEnabled(combo.count() > 0)
        combo.blockSignals(False)
        self._update_player_stats()

    def _relabel_stats_players(self) -> None:
        """Relabel the stats combo in place (GH #130), keeping its rows and
        selection, then re-render the Player section."""
        if self.scenario is None:
            return
        combo = self.stats_player_combo
        active_set = set(self._current_active_players())
        for i in range(combo.count()):
            pid = combo.itemData(i)
            _set_player_item(
                combo, i, pid, self._player_labels, self.scenario.player_colors,
                self._stats_label_suffix(pid, active_set),
            )
        self._update_player_stats()

    def _update_player_stats(self) -> None:
        """Renders the selected player's breakdown (GH #5, GH #145). Cheap
        enough to run on every unit mutation: a walk of one player's list
        (every owner's, for GAIA's resource totals) plus, only when the
        Triggers section is already parsed, a memoized re-read."""
        player_id = self.stats_player_combo.currentData()
        if self.scenario is None or player_id is None:
            self.player_stats_rows = []
            self._fill_player_section([])
            return
        c = player_stats.counts_for(self.scenario, player_id)
        is_gaia = player_id == GAIA_PLAYER_ID
        resources = player_stats.resources_on_map(self.scenario) if is_gaia else None
        rows = _player_stat_rows(c, is_gaia, resources)
        # A row with no count shows its note alone.
        supported = self.scenario.trigger_read_supported
        if supported is None:
            rows.append(("Triggers", "", "not counted yet (enter Triggers mode or run Map Analysis)", ""))
        elif supported is False:
            rows.append(("Triggers", "", map_analysis._TRIGGERS_UNAVAILABLE, ""))
        else:
            summary = player_stats.trigger_summary(self.scenario)
            if summary is None:
                rows.append(("Triggers", "", "no trigger vocabulary for this scenario version", ""))
            else:
                rows.append((
                    "Triggers",
                    f"{summary.per_player.get(player_id, 0):,}",
                    f"(of {summary.total:,}; {summary.without_player:,} reference no player)",
                    "",
                ))
        self.player_stats_rows = rows
        self._fill_player_section(rows)

    def _fill_player_section(self, rows: list[StatRow]) -> None:
        """Rebuilds the Player section's stat rows only. The header and the
        combo's row stay; a collapsed sub-section stays collapsed."""
        section = self._info_sections["Player"]
        collapsed = {
            section.child(i).data(0, Qt.UserRole)
            for i in range(section.childCount())
            if section.child(i).childCount() and not section.child(i).isExpanded()
        }
        # Child 0 is the combo's row; removing it would delete the combo with it.
        while section.childCount() > 1:
            section.takeChild(1)
        parents: dict[str, QTreeWidgetItem] = {}
        for label, count, note, parent in rows:
            item = _add_info_row(parents[parent] if parent else section, label, count, note)
            item.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
            if not parent:
                parents[label] = item
        for label, item in parents.items():
            item.setExpanded(label not in collapsed)
        self.info_tree.fit_columns()

    def _set_hover_text(self, text: str) -> None:
        # Page 0's label and the Terrain page's one-line copy, so the readout survives the page swap.
        self.hover_label.setText(text)
        self.terrain_panel.set_hover_text(text)

    def on_hover(self, tile: tuple[int, int] | None) -> None:
        self._hover_tile = tile
        self._refresh_hover_text()

    def _refresh_hover_text(self) -> None:
        # Reads, never writes, _hover_tile: Paste Region anchors on it and the clipboard dialog freezes it.
        tile = self._hover_tile
        if self.scenario is None:
            return
        if tile is None:
            self._set_hover_text(HOVER_IDLE_TEXT)
            return
        x, y = tile
        mm = self.scenario.map_manager
        if not (0 <= x < mm.map_width and 0 <= y < mm.map_height):
            self._set_hover_text(HOVER_IDLE_TEXT)
            return
        # Flat indexing, not get_tile_safe(): that returns None for every tile of a non-square map.
        tile = mm.terrain[y * mm.map_width + x]
        # Before elevation: the Terrain page's copy elides on the right.
        layer_part = f"  layer={name_for_terrain_id(tile.layer)}" if tile.layer != -1 else ""
        self._set_hover_text(
            f"({x}, {y})  {name_for_terrain_id(tile.terrain_id)}{layer_part}  elevation={tile.elevation}"
        )
