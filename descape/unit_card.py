"""UnitCard: one object's portrait, name, Unit ID and base stats (GH #148).

The Units panel shows two in one slot at the top of the inspector: the
selection card, for the unit(s) selected on the map, and with nothing
selected the placing card, for the object the next Place click adds
(TASK-337). Everything on a card derives from the const alone (the .dat's
civ-0 values), except the portrait's player-colour region, which the panel
fills with the owner's colour through set_team_color(). Read-only and
callback-free: the panel pushes a const, the card shows it.

`stable` (the placing card; off for the selection card): the card never
changes height, so the inspector does not move as the pick changes.
Every stat cell stays in the grid, present pairs packed first, and the
absent ones and a blank Unit ID row are hidden with setRetainSizeWhenHidden,
so the layout itself holds the five-stat size at any font, with nothing
measured. With an install the portrait slot stays visible when empty;
without one it is hidden, since no portrait or sprite can show (user ruling,
TASK-337). The name is one elided line.

Portraits never decode on the GUI thread (GH #68): a card asks
catalog_thumbs.ThumbLoader, shared with the catalog list, and fills the slot
when the loader answers. The memo is keyed by (const, px), so the two cards
share their 64 px portraits but the list's 32 and 48 px thumbnails never
serve them. A card is a priority client, served before the list's viewport.
A decode costs about 54 ms (4x on tester hardware), and the placing card
changes on every arrow-key step through the list. With no portrait the
loader's sprite fallback faces the viewer, as the list's thumbnails do.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from descape import asset_source, catalog_thumbs, object_catalog, unit_fields
from descape.terrain_palette import PLAYER_COLORS
from descape.unit_stats_table import unit_stats
from descape.viewer_common import pixmap_from_rgba

# The in-game editor's own Units-tab selection panel set (docs/
# INGAME_EDITOR_REFERENCE.md's "Selecting: shows HP/Attack/Armour") plus
# Range, which the original stats request named explicitly. unit_stats()
# keys use genieutils' own British spelling; captions don't have to. The third
# column is the game's own staticons file for it (GH #148, per widgetui/materials.json).
_STAT_FIELDS = (
    ("hp", "Hit points", "hp"),
    ("attack", "Attack", "damage"),
    ("melee_armour", "Melee armour", "armor"),
    ("pierce_armour", "Pierce armour", "range-armor"),
    ("range", "Range", "range"),
)
# Icon+value pairs per stats row: HP/Attack, Melee/Pierce, Range.
_STAT_PAIRS_PER_ROW = 2
PORTRAIT_PX = 64

# GH #148: tooltip-only, on the stats grid; it no longer costs a visible line.
_STATS_NOTE_FULL = (
    "Base values from the game's own unit table: no civ bonuses, tech "
    "upgrades or trigger effects applied. A trigger that changes a building's "
    "range is listed below."
)

_MIXED_TEXT = "(mixed)"
# A group of several types: the Unit ID row counts them (GH #68).
_MIXED_COUNT_TEXT = "(mixed: {count} types)"


class _ElidedLabel(QLabel):
    """One line, elided to whatever width the layout gives it. text() stays
    the full string and the minimum width is zero, so a long name can no
    longer raise the pane's minimum. Not mouse-selectable: custom painting
    breaks the selection drawing."""

    def minimumSizeHint(self) -> QSize:
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        rect = self.contentsRect()
        painter.drawText(rect, int(self.alignment()), self.fontMetrics().elidedText(self.text(), Qt.ElideRight, rect.width()))


def _stat_text(field_id: str, stats: dict) -> str:
    """One stat's value; Range reads "min-max" where the two differ (GH #68: 7 consts, e.g. 2.5-3)."""
    value = stats[field_id]
    if field_id == "range":
        longest = stats.get("max_range", value)
        if longest != value:
            return f"{value:g}-{longest:g}"
    return f"{value:g}"


def _retain_when_hidden(widget: QWidget) -> None:
    policy = widget.sizePolicy()
    policy.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(policy)


class UnitCard(QWidget):
    """GH #148, on the in-game editor's stats preview: the portrait on the
    left; name, Unit ID and the stats as icon+value pairs on the right."""

    def __init__(
        self, *, stable: bool = False, loader: catalog_thumbs.ThumbLoader | None = None, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self._stable = stable
        # The panel shares one between the list and both cards; a lone card gets its own.
        self.loader = loader if loader is not None else catalog_thumbs.ThumbLoader(self)
        self.loader.ready.connect(self._on_thumb_ready)
        # The const show_const() last showed; None after show_mixed() or clear().
        self._shown_const: int | None = None
        # True while the loader owes this card the shown const's portrait.
        self._awaiting = False
        # The portrait as the loader gave it, kept so a new owner colour can refill it.
        self._raw_portrait: tuple[object, str | None] = (None, None)
        self._team_rgb: tuple[int, int, int] = PLAYER_COLORS[0]

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.portrait = QLabel()
        self.portrait.setFixedSize(PORTRAIT_PX, PORTRAIT_PX)
        self.portrait.setAlignment(Qt.AlignCenter)
        row.addWidget(self.portrait, 0, Qt.AlignTop)
        # "icon" (the install's portrait), "sprite" (the map sprite) or None (none shown).
        self.portrait_source: str | None = None
        if stable:
            self._sync_slot()
        else:
            self.portrait.setVisible(False)

        column = QVBoxLayout()
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        name = _ElidedLabel("") if stable else QLabel("")
        font = name.font()
        font.setBold(True)
        name.setFont(font)
        if not stable:
            name.setWordWrap(True)
            name.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.name_label = name
        column.addWidget(name)
        const_row = QHBoxLayout()
        const_row.setContentsMargins(0, 0, 0, 0)
        self.const_caption = QLabel(f"{unit_fields.FIELDS_BY_ID['unit_const'].label}:")
        self.const_label = QLabel("")
        self.const_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        const_row.addWidget(self.const_caption)
        const_row.addWidget(self.const_label, 1)
        column.addLayout(const_row)

        # Derived reference data (empires2_x2_p1.dat, committed as
        # unit_stats.json), never editable, so not in unit_fields.FIELDS.
        # A non-stable card hides it whole when unit_stats() has nothing for the const -- see _apply_stats.
        self.stats_grid = QWidget()
        self.stats_grid.setToolTip(_STATS_NOTE_FULL)
        stats_grid = QGridLayout(self.stats_grid)
        stats_grid.setContentsMargins(0, 0, 0, 0)
        # Packs the pairs to the left whatever the pane width.
        stats_grid.setColumnStretch(2 * _STAT_PAIRS_PER_ROW, 1)
        # Both labels per pair, not just the value: ~1,169 of 2,642 consts
        # have no type_50 combat block at all (a tree, say), so a pair goes
        # outright rather than leaving an icon with nothing beside it.
        # _apply_stats places them; the caption becomes an icon once an install supplies one.
        self.stat_rows: dict[str, tuple[QLabel, QLabel]] = {}
        for field_id, caption_text, _icon in _STAT_FIELDS:
            cap_label = QLabel(f"{caption_text}:")
            cap_label.setToolTip(caption_text)
            cap_label.setAccessibleName(caption_text)
            value_label = QLabel("")
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            # Room for a 4-digit value, so the second pair column lines up across rows.
            value_label.setMinimumWidth(value_label.fontMetrics().horizontalAdvance("0000"))
            self.stat_rows[field_id] = (cap_label, value_label)
        if stable:
            for widget in (self.const_caption, self.const_label, *(w for pair in self.stat_rows.values() for w in pair)):
                _retain_when_hidden(widget)
        # The install the stat icons were loaded for; a sentinel until the first _apply_stats.
        self._stat_icon_install = object()
        column.addWidget(self.stats_grid)
        column.addStretch(1)
        row.addLayout(column, 1)
        if stable:
            self._apply_stats(None)

    # -- what the card shows ----------------------------------------------------

    def show_const(self, const: int, *, prefix: str = "", name: str | None = None, const_text: str | None = None) -> None:
        """Name line `prefix` plus `name` (display_name() by default), then
        Unit ID (`const_text`, or the const), stats and portrait."""
        self._shown_const = const
        self.name_label.setText(prefix + (name or object_catalog.display_name(const)))
        self.const_label.setText(str(const) if const_text is None else const_text)
        self._set_const_row_visible(True)
        self._apply_stats(const)
        self._show_portrait(const)

    def show_mixed(self, count: int | None = None) -> None:
        """A group of `count` different consts: "(mixed)" for the name, the
        count in the Unit ID row ("(mixed)" without one), no stats, no portrait."""
        self._shown_const = None
        self.name_label.setText(_MIXED_TEXT)
        self.const_label.setText(_MIXED_TEXT if count is None else _MIXED_COUNT_TEXT.format(count=count))
        self._set_const_row_visible(True)
        self._apply_stats(None)
        self._show_portrait(None)

    def clear(self, text: str = "") -> None:
        """Name line `text`, a blank Unit ID row, no stats, no portrait."""
        self._shown_const = None
        self.name_label.setText(text)
        self.const_label.setText("")
        self._set_const_row_visible(not self._stable)
        self._apply_stats(None)
        self._show_portrait(None)

    def refresh_thumbnails(self) -> None:
        """Reloads the stat icons and re-requests the shown const's portrait, after an install change."""
        self._stat_icon_install = object()
        self._load_stat_icons()
        if self._stable:
            self._sync_slot()
        self._show_portrait(self._shown_const)

    def set_team_color(self, rgb: tuple[int, int, int]) -> None:
        """The colour behind a portrait's player-colour region; refills the shown one."""
        if tuple(rgb) == self._team_rgb:
            return
        self._team_rgb = tuple(rgb)
        icon, source = self._raw_portrait
        if icon is not None:
            self._set_portrait(icon, source)

    def _set_const_row_visible(self, visible: bool) -> None:
        self.const_caption.setVisible(visible)
        self.const_label.setVisible(visible)

    # -- stats ------------------------------------------------------------------

    def _apply_stats(self, unit_const: int | None) -> None:
        """Shows each stat pair against key presence in unit_stats(unit_const),
        re-packing the present ones _STAT_PAIRS_PER_ROW to a row, so a tree
        shows one pair rather than a gap. A non-stable card hides the whole grid when the dict is
        empty -- an unrecognized const (a modded scenario), or a real one with
        no HP and no combat block at all (roughly a fifth of the table:
        cliffs, rocks, other decoratives); a stable card keeps every cell,
        absent ones last and hidden in place. `unit_const=None` (nothing
        shown) is treated the same as "no stats"."""
        self._load_stat_icons()
        stats = unit_stats(unit_const) if unit_const is not None else {}
        if not self._stable:
            self.stats_grid.setVisible(bool(stats))
        grid = self.stats_grid.layout()
        order = [field_id for field_id in self.stat_rows if field_id in stats]
        if self._stable:
            order += [field_id for field_id in self.stat_rows if field_id not in stats]
        for cap_label, value_label in self.stat_rows.values():
            grid.removeWidget(cap_label)
            grid.removeWidget(value_label)
        for placed, field_id in enumerate(order):
            cap_label, value_label = self.stat_rows[field_id]
            row, pair = divmod(placed, _STAT_PAIRS_PER_ROW)
            grid.addWidget(cap_label, row, 2 * pair)
            grid.addWidget(value_label, row, 2 * pair + 1)
        for field_id, (cap_label, value_label) in self.stat_rows.items():
            present = field_id in stats
            cap_label.setVisible(present)
            value_label.setVisible(present)
            # :g so 4.0 renders as "4" -- a melee unit's range is 0, not "0.0".
            value_label.setText(_stat_text(field_id, stats) if present else "")

    def _load_stat_icons(self) -> None:
        """Each stat caption as the install's icon at the font's line height,
        or its text with no install. Reloads only when the install changes."""
        install = asset_source.get_install_path()
        if install == self._stat_icon_install:
            return
        self._stat_icon_install = install
        for field_id, caption, icon_name in _STAT_FIELDS:
            cap_label = self.stat_rows[field_id][0]
            arr = asset_source.get_stat_icon(icon_name)
            if arr is None:
                cap_label.setText(f"{caption}:")
            else:
                cap_label.setPixmap(pixmap_from_rgba(arr, cap_label.fontMetrics().height()))

    # -- portrait ---------------------------------------------------------------

    def _show_portrait(self, const: int | None) -> None:
        """A memo hit shows at once. A miss blanks the slot and asks the
        loader, which answers through _on_thumb_ready."""
        self._awaiting = False
        if const is None:
            self._set_portrait(None, None)
            return
        hit, icon = self.loader.cached(const, PORTRAIT_PX)
        if hit:
            self._set_portrait(icon, self.loader.source(const, PORTRAIT_PX))
            return
        self._awaiting = True
        self._set_portrait(None, None)
        # A no-install request answers at once, inside this call.
        self.loader.request([const], PORTRAIT_PX, client=self, priority=True)

    def _on_thumb_ready(self, const: int, px: int, icon) -> None:
        if not self._awaiting or px != PORTRAIT_PX or const != self._shown_const:
            return
        self._awaiting = False
        self._set_portrait(icon, self.loader.source(const, PORTRAIT_PX))

    def _set_portrait(self, icon, source: str | None) -> None:
        self.portrait_source = source if icon is not None else None
        self._raw_portrait = (icon, source)
        if icon is None:
            self.portrait.clear()
        else:
            filled = catalog_thumbs.team_filled(icon, source, self._team_rgb, PORTRAIT_PX)
            self.portrait.setPixmap(filled.pixmap(PORTRAIT_PX, PORTRAIT_PX))
        if self._stable:
            self._sync_slot()
        else:
            # Open for a shown const with an install, even empty, so the inspector never jumps; shut otherwise.
            shown = self._shown_const is not None and asset_source.get_install_path() is not None
            self.portrait.setVisible(icon is not None or shown)

    def _sync_slot(self) -> None:
        """A stable card keeps its empty slot only with an install configured:
        without one no portrait or sprite can ever show, and the slot would
        cost 70 px of the pane's minimum width (user ruling, TASK-337)."""
        self.portrait.setVisible(asset_source.get_install_path() is not None)
