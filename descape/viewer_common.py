"""Widget helpers and tool-id lookups shared by the viewer's panels.

Split out of viewer.py so trigger_panel and map_options_panel can share
them without importing viewer.py, which imports both of them. The
settings.TOOLS-derived sets live here as one documented family even
though their consumers are split across map_view*.py and viewer*.py.

They are deliberately not in settings.py next to the TOOLS registry they
derive from: they are view-layer lookups rather than settings (three of the
seven are private by name), and settings.py's import surface is depended on
directly by a lot of tests.

It is also the leaf for the module-level names that viewer.py and the
modules split out of it share (the left-pane page indexes, the unit helpers,
the status colours), so none of those modules imports viewer.py."""

from __future__ import annotations

import dataclasses
import math

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import (
    QColor,
    QFontMetricsF,
    QIcon,
    QImage,
    QPixmap,
    QValidator,
)
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
)

from descape import (
    asset_source,
    composite_backend,
    game_dataset,
    iso_geometry,
    object_catalog,
    settings,
    tool_options,
    unit_pick,
    unit_sprites,
)
from descape.edit_history import (
    UnitDiffRecord,
)
from descape.render_units import NON_BUILDING_SPAN, unit_tile_bounds
from descape.terrain_palette import tile_span
from descape.unit_model import (
    span_low_corner,
)
from descape.viewer_dialogs import status_text_color

# Derived from settings.TOOLS -- the single tool registry -- rather than
# hand-duplicated here, so a new tool added there can't silently miss an
# entry in any of these. Internal tool identifiers (e.g. "elevation",
# "set_level") are unchanged even when a display label is renamed later
# (Elevate / Set Elevation), so settings.get_keybind()'s persisted
# "tool_elevation"/"tool_set_level" keys and every other internal reference
# keep working regardless.
EDIT_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.is_edit_tool)
_TOOL_LABELS = {t.tool_id: t.label for t in settings.TOOLS}
_STROKE_LABELS = {t.tool_id: t.stroke_label for t in settings.TOOLS if t.is_edit_tool}
# One-shot click tools (Paint Can) that must never enter the drag-stroke
# path -- see MapView.mousePressEvent's early CLICK_TOOLS branch and
# ViewerWindow.on_fill.
CLICK_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.click_only)
# Which toolbar param widget ("terrain" | "level" | "") the active tool
# reads -- see ViewerWindow._update_tool_enabled's tool-param visibility
# block.
_TOOL_PARAM = {t.tool_id: t.param_widget for t in settings.TOOLS}
# Tools whose stroke applies across a brush footprint (size + shape) rather
# than always exactly one tile -- see ToolDef.supports_brush's own comment.
BRUSH_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_brush)
# Draw Line / Draw Rectangle: press-drag-preview, commit once at release.
# Members are edit tools but must never reach the per-tile stroke path --
# see MapView's shape branches, which sit above the EDIT_TOOLS ones.
SHAPE_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.drag_shape)
_TOOL_SHAPE = {t.tool_id: t.drag_shape for t in settings.TOOLS}
# Tools that offer the Off-grid (non-snapped) placement checkbox -- see
# ToolDef.supports_off_grid's own comment.
OFF_GRID_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_off_grid)
# Tools that offer "Free placement" (GH #176); see ToolDef.supports_unrestricted.
UNRESTRICTED_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_unrestricted)
# Tools that offer "Walls skip occupied tiles" (GH #124); see
# ToolDef.supports_skip_occupied.
SKIP_OCCUPIED_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_skip_occupied)
# Tools that offer "Moving a gate updates walls" (GH #194); see
# ToolDef.supports_gate_walls.
GATE_WALLS_TOOLS = frozenset(t.tool_id for t in settings.TOOLS if t.supports_gate_walls)
# The tools that are neither an edit tool nor Pan, so they can't be
# recognised by set membership the way the sets above are.
TOOL_RULER = "ruler"
TOOL_EYEDROPPER = "eyedropper"
TOOL_SELECT = "select"
TOOL_MOVE = "move"

# Which mode(s) each tool's toolbar button shows in -- see
# settings.ToolDef.modes.
_TOOL_MODES = {t.tool_id: t.modes for t in settings.TOOLS}


def brush_applicable(tool_id: str, *, rect_filled: bool) -> bool:
    """True if `tool_id` paints across a brush footprint right now.

    BRUSH_TOOLS membership on its own is not the answer for Draw Rectangle:
    it supports a brush in Outline mode (a thicker border) but not in Filled
    mode, where a brush would dilate the rectangle past the bounds the
    preview just showed. That carve-out lives here, in one Qt-free place,
    rather than in the two viewer_tool_state.py call sites plus the TOOLS-derived
    expectations in tests/test_toolbar_params.py -- which is what keeps
    those expectations generated rather than hand-listed.
    """
    if tool_id not in BRUSH_TOOLS:
        return False
    return not (_TOOL_SHAPE.get(tool_id) == "rect" and rect_filled)


def tool_applicable(tool_id: str, mode: str) -> bool:
    """True if `tool_id`'s toolbar button should be visible in `mode`.

    Empty ToolDef.modes means every mode. Used by
    ViewerWindow._update_tool_enabled() to hide mode-inapplicable tools
    instead of just greying them out.
    """
    return settings.modes_include(_TOOL_MODES.get(tool_id, ()), mode)

# How many characters a property-form combo box sizes itself for. Without it a
# combo's minimumSizeHint is its longest *item*, and ObjectAttribute's longest
# is 430 px in a 340 px panel -- so the form sat permanently below its own
# stated minimum, which is what made every attempt to narrow the label column
# tip the whole form into horizontal overflow.
_COMBO_CONTENTS_CHARS = 12


# QFontMetricsF.averageCharWidth() of the font every panel width was measured
# against: the test suite's pinned DejaVu Sans 12pt at 96 DPI.
_BASELINE_AVG_CHAR_PX = 8.109375


class FontScaledWidth:
    """A panel's MIN_USEFUL_WIDTH, read as a class attribute like the plain
    int it replaces. `base_px` is the width measured at the baseline font;
    a larger app font (Windows 150% scaling reports 144 DPI to Qt 5, or a
    bigger Settings > Appearance size) scales it up, since every label and
    field in the panel grows with the font while a pixel constant would not.
    Never scales below `base_px`, so a smaller font keeps today's width."""

    def __init__(self, base_px: int) -> None:
        self.base_px = base_px

    def __get__(self, obj, owner=None) -> int:
        if QApplication.instance() is None:
            return self.base_px
        ratio = QFontMetricsF(QApplication.font()).averageCharWidth() / _BASELINE_AVG_CHAR_PX
        return max(self.base_px, math.ceil(self.base_px * ratio))


def _fit_combo_width(combo: QComboBox) -> None:
    """Size a combo for a fixed character count rather than for its longest
    item -- see _COMBO_CONTENTS_CHARS. The popup still shows every name in
    full. Shared by TriggerPanel and MapOptionsPanel: VictoryCondition and
    SecondaryGameMode are the same class of long-label enum ObjectAttribute
    is."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(_COMBO_CONTENTS_CHARS)


def _swatch_icon(rgb: tuple[int, int, int]) -> QIcon:
    """A 16px player-colour square for a player combo's item icon."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(QColor(*rgb))
    return QIcon(pixmap)


def _player_icon(pid: int, colors) -> QIcon:
    """pid's swatch, or a null icon when there are no colours (no document)."""
    return QIcon() if colors is None else _swatch_icon(colors[pid])


def pixmap_from_rgba(arr, px: int) -> QPixmap:
    """An (h, w, 4) uint8 RGBA array as a QPixmap scaled to fit px x px."""
    import numpy as np

    arr = np.ascontiguousarray(arr)
    height, width = arr.shape[:2]
    image = QImage(arr.data, width, height, 4 * width, QImage.Format_RGBA8888)
    # fromImage copies, so the array may go out of scope.
    return QPixmap.fromImage(image).scaled(px, px, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _set_player_item(combo: QComboBox, index: int, pid: int, labels, colors, suffix: str = "") -> None:
    """Relabel one player row in place (GH #130): text, uncut tooltip, swatch.
    `labels` is a player_labels.PlayerLabels; `colors` a pid-indexed RGB sequence or None."""
    combo.setItemText(index, labels.text[pid] + suffix)
    combo.setItemData(index, labels.full[pid] + suffix, Qt.ToolTipRole)
    combo.setItemIcon(index, _player_icon(pid, colors))


def _add_player_item(combo: QComboBox, pid: int, labels, colors, data=None) -> None:
    """Append a player row carrying `data`, labelled as _set_player_item() does."""
    combo.addItem("", data)
    _set_player_item(combo, combo.count() - 1, pid, labels, colors)


def _relabel_player_rows(combo: QComboBox, labels, colors) -> None:
    """Relabel every row whose itemData is a pid 0..8. Rows carrying None
    ("(mixed)", "(differs)") or an out-of-range value ("unknown (v)") are left alone."""
    for i in range(combo.count()):
        pid = combo.itemData(i)
        if isinstance(pid, int) and 0 <= pid < len(labels.text):
            _set_player_item(combo, i, pid, labels, colors)


class _IndeterminateMixin:
    """A blank "the selection differs" state for a spinbox (GH #60).

    A spinbox has no blank value and -1 is a real one (trigger_fields.UNSET),
    so the box parks on its minimum and renders "" instead. specialValueText
    beats textFromValue() at the minimum, so it is parked too and put back on
    the first real change. The panel clears the state from valueChanged, or
    from editingFinished when the typed value equals the parked one.

    That second case cannot use lineEdit().isModified(): Qt re-renders the
    text before editingFinished, which clears the flag (measured). So the last
    user-typed text is latched from textEdited, which programmatic sets never emit.
    """

    _indeterminate = False
    _parked_special_text = ""
    _typed_text = ""

    def set_indeterminate(self) -> None:
        if self._indeterminate:
            return
        self._indeterminate = True
        self._typed_text = ""
        self._parked_special_text = self.specialValueText()
        self.setSpecialValueText("")
        self.setValue(self.minimum())
        # By hand: setValue() on a box already at its minimum re-renders nothing.
        self.lineEdit().setText("")
        self.lineEdit().textEdited.connect(self._note_typed)

    def _note_typed(self, text: str) -> None:
        if self._indeterminate:
            self._typed_text = text.strip()

    def typed_parked_value(self) -> bool:
        """Whether the user typed the very value the blank box is parked on,
        the one edit that emits no valueChanged."""
        if not self._indeterminate or not self._typed_text:
            return False
        state, _text, _pos = self.validate(self._typed_text, 0)
        return state == QValidator.Acceptable and self.valueFromText(self._typed_text) == self.value()

    def is_indeterminate(self) -> bool:
        return self._indeterminate

    def clear_indeterminate(self) -> None:
        """Leave the blank state and show the value it now holds."""
        if not self._indeterminate:
            return
        self._indeterminate = False
        self.setSpecialValueText(self._parked_special_text)
        # The value that ended the state was rendered blank; Qt will not repaint it.
        value = self.value()
        at_special = value == self.minimum() and bool(self._parked_special_text)
        self.lineEdit().setText(self._parked_special_text if at_special else self.textFromValue(value))

    def textFromValue(self, value):  # Qt override
        return "" if self._indeterminate else super().textFromValue(value)


class IndeterminateSpinBox(_IndeterminateMixin, QSpinBox):
    pass


class IndeterminateDoubleSpinBox(_IndeterminateMixin, QDoubleSpinBox):
    pass


class NoWheelComboBox(QComboBox):
    """A combo the mouse wheel cannot change: the event is ignored, so the
    scroll area the combo sits in scrolls instead of the choice moving."""

    def wheelEvent(self, event) -> None:  # Qt override
        event.ignore()


class FocusWheelComboBox(QComboBox):
    """A combo the mouse wheel changes only once it has focus (TASK-344).
    Unfocused, the wheel goes to the scroll area it sits in. StrongFocus,
    since QComboBox's default WheelFocus would take focus on that same notch."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # Qt override
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class KeepOpenMenu(QMenu):
    """A menu of toggles that stays open after a pick (GH #205): a click or Space
    on a checkable row, or on one opted in with keep_open(), triggers it in place."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self._armed = False
        self._extra: set[QAction] = set()

    def keep_open(self, action: QAction) -> None:
        """Opt a non-checkable row (a bulk "All"/"None" row) in."""
        self._extra.add(action)

    def _keeps_open(self, action: QAction | None) -> bool:
        return (
            action is not None
            and action.isEnabled()
            and action.menu() is None
            and not action.isSeparator()
            and (action.isCheckable() or action in self._extra)
        )

    def showEvent(self, event) -> None:  # Qt override
        self._armed = False
        super().showEvent(event)

    def mousePressEvent(self, event) -> None:  # Qt override
        self._armed = True
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # Qt override
        if event.buttons():  # press on the menu bar, drag onto a row, release
            self._armed = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # Qt override
        # Unarmed: the release of the click that opened an InstantPopup menu.
        armed, self._armed = self._armed, False
        action = self.actionAt(event.pos())
        if armed and self._keeps_open(action):
            # Qt still has to see the release to clear its own press state;
            # a disabled row makes it decline to activate (and so to close).
            action.setEnabled(False)
            try:
                super().mouseReleaseEvent(event)
            finally:
                action.setEnabled(True)
            action.trigger()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # Qt override
        action = self.activeAction()
        if event.key() == Qt.Key_Space and self._keeps_open(action):
            action.trigger()
            event.accept()
            return
        super().keyPressEvent(event)


def _make_spinbox(
    value,
    editable: bool,
    *,
    minimum: int,
    maximum: int,
    special_value_text: str = "",
    indeterminate: bool = False,
) -> QSpinBox:
    """A property-form spinbox with keyboard tracking off.

    `setKeyboardTracking(False)` is the load-bearing one: with it on, typing
    "12" over a 5 emits valueChanged(1) and then valueChanged(12), so a
    two-digit edit records two undo steps and briefly writes a value the user
    never asked for.

    Range is a parameter rather than the hardcoded trigger range this was
    promoted from, because Options.ai_map_type's own minimum is INT32_MIN and
    a -1 floor would silently clamp it. `special_value_text` renders
    `minimum` as text instead of a number whose meaning the user would have to
    know -- TriggerPanel passes it for trigger_fields.UNSET's -1 sentinel, and
    Map Options, which has no unset sentinel, passes nothing.
    """
    spin = IndeterminateSpinBox() if indeterminate else QSpinBox()
    spin.setRange(minimum, maximum)
    if special_value_text:
        spin.setSpecialValueText(special_value_text)
    # Falls back to `minimum` rather than to 0, so TriggerPanel keeps landing
    # on UNSET (and therefore on special_value_text) for a None-valued field.
    spin.setValue(value if isinstance(value, int) and not isinstance(value, bool) else minimum)
    spin.setEnabled(editable)
    spin.setKeyboardTracking(False)
    if indeterminate:
        # Last, so it parks the special text set above.
        spin.set_indeterminate()
    return spin


class LabelFitButton(QPushButton):
    """A push button that may shrink to its own label (GH #213). Fusion
    floors a labelled button's size hint at 80 px, which the entry row cannot
    spare beside its ▲▼ pair. The hint stays; only the minimum drops, to what
    the style gives the same label without that floor. Read off the live font
    and style on every query, so a UI font change needs no refit."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        # Shrinkable: under QPushButton's own Minimum policy the layout takes sizeHint() as the floor.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

    def minimumSizeHint(self) -> QSize:
        hint = super().sizeHint()
        option = QStyleOptionButton()
        self.initStyleOption(option)
        label = self.fontMetrics().size(Qt.TextShowMnemonic, self.text())
        # The style applies its floor only to an option that carries text; frame and margins are the same.
        option.text = ""
        fitted = self.style().sizeFromContents(QStyle.CT_PushButton, option, label, self).width()
        return QSize(min(fitted, hint.width()), hint.height())


# Free-mode headroom below a span-1 unit's far map edge; a float32 coordinate
# at W - 1e-6 rounds back up to W, which is off-map.
_FREE_EDGE_MARGIN = 1e-3


def _axis_move_range(coord: float, low: int, span: int, size: int, whole_tiles: bool) -> tuple[float, float]:
    """(lowest, highest) delta along one axis that keeps a footprint on the map."""
    if whole_tiles:
        lo, hi = -low, size - (low + span)
    elif span > 1:
        # _span_start's half-tile branch: tile + span/2 is the anchor of a footprint starting at tile.
        lo, hi = span / 2 - coord, size - span / 2 - coord
    else:
        lo, hi = -coord, size - _FREE_EDGE_MARGIN - coord
    # A footprint already overhanging an edge doesn't force a move back inward.
    return min(0, lo), max(0, hi)


def clamp_group_delta(units, dx, dy, map_w: int, map_h: int, whole_tiles: bool):
    """(dx, dy, clamped): the group delta pulled into the range every member
    allows, so the whole group stops where its outermost unit reaches the map
    edge (GH #75). Members already off-map do not limit it."""
    lo_x = lo_y = -math.inf
    hi_x = hi_y = math.inf
    for unit in units:
        if unit_tile_bounds(unit, map_w, map_h) is None:
            continue
        span_x, span_y = tile_span(unit.unit_const, NON_BUILDING_SPAN)
        low_x, low_y = span_low_corner(unit)
        ax_lo, ax_hi = _axis_move_range(unit.x, low_x, span_x, map_w, whole_tiles)
        ay_lo, ay_hi = _axis_move_range(unit.y, low_y, span_y, map_h, whole_tiles)
        lo_x, hi_x = max(lo_x, ax_lo), min(hi_x, ax_hi)
        lo_y, hi_y = max(lo_y, ay_lo), min(hi_y, ay_hi)
    cx, cy = min(max(dx, lo_x), hi_x), min(max(dy, lo_y), hi_y)
    return cx, cy, (cx, cy) != (dx, dy)


def _reshaping_clause(reshaped: int) -> str:
    """_report_delete()'s wording for the walls an edit reshaped beside it."""
    return f", reshaping {reshaped} adjacent" if reshaped else ""


# Shown when off-grid placement was asked for and the screen -> map-point inverse
# had no answer for that pixel. A module constant so a test can assert on the
# real string rather than on a paraphrase of it.
OFF_GRID_FALLBACK_MESSAGE = (
    "Off-grid: no exact point under the cursor here, snapped to the grid instead"
)


# Which editor-panel page each mode shows. One dock whose pages swap inside it
# (GH #64), not a dock per panel -- the mode already says which one is
# relevant, so per-panel docks would duplicate the show/hide a stack gets free.
_LEFT_PAGE_INFO = 0
_LEFT_PAGE_TRIGGERS = 1
_LEFT_PAGE_UNITS = 2
# Appended, never renumbered: tests/test_trigger_panel_*.py and
# tests/test_unit_selection_viewer.py assert left_stack.currentIndex() by
# literal integer for pages 0-2.
_LEFT_PAGE_MAP_OPTIONS = 3
_LEFT_PAGE_PLAYERS = 4
# Appended at 5, per the rule above, since Players mode already claimed 4
# by the time Diplomacy mode landed.
_LEFT_PAGE_DIPLOMACY = 5
# Appended at 6, same rule, since Diplomacy mode already claimed 5 by the
# time Messages mode landed.
_LEFT_PAGE_MESSAGES = 6
# Appended at 7, same rule: the Terrain-mode picker page (GH #56).
_LEFT_PAGE_TERRAIN = 7
_LEFT_PAGE_FOR_MODE = {
    "triggers": _LEFT_PAGE_TRIGGERS,
    "units": _LEFT_PAGE_UNITS,
    "map_options": _LEFT_PAGE_MAP_OPTIONS,
    "players": _LEFT_PAGE_PLAYERS,
    "diplomacy": _LEFT_PAGE_DIPLOMACY,
    "messages": _LEFT_PAGE_MESSAGES,
    "terrain": _LEFT_PAGE_TERRAIN,
}


_SCRIPT_NAME_FIELD = "script_name"


# Lives at object_catalog.display_name: units_panel.py needs the same
# display convention, and importing viewer.py from there would invert the
# dependency. Aliased here so this file's own 10 call sites stay untouched.
_unit_name = object_catalog.display_name


def _entries_of(trigger, kind: str):
    """A trigger's live condition or effect list, by kind (not a copy)."""
    return trigger.conditions if kind == "condition" else trigger.effects


def _ghost_entry(player_id: int, ghost) -> unit_pick.UnitEntry:
    """The index entry a ghost unit would have, for geometry only."""
    return unit_pick.UnitEntry(player_id, ghost, int(ghost.x), int(ghost.y), 0)


def _garrison_referrers(model, unit) -> list:
    """UnitEditModel.referencing(unit), minus the one case it deliberately
    keeps: a unit whose own reference_id is -1.

    referencing() preserves the -1 bucket on purpose (its own docstring, so
    a byte-identical answer on a pathological file), and -1 is exactly what
    every unit carries when it is inside nothing. Read literally, a host with
    reference_id -1 therefore "holds" every ungarrisoned unit in the file --
    which would cascade a move or a delete across the whole map, and fill the
    Inspector's Garrison list with it. -1 means "inside nothing" in
    UnitFilter.matches() and in the map-analysis check; it means that here."""
    if unit.reference_id == -1:
        return []
    return model.referencing(unit)


def _team_colors(scenario) -> tuple[tuple[int, int, int], ...]:
    """Each player's sprite tint, indexed by player_id: TEAM_COLORS at the
    file's team index, the lookup the sprite tint itself makes (TASK-115)."""
    team = unit_sprites.TEAM_COLORS
    return tuple(team[index % len(team)] for index in scenario.team_indices)


_DEFAULT_WALL_CONST = tool_options.default("last_wall")  # Stone Wall in AoE2, Medium Wall in Return of Rome


# The wall-placing tools: the wall button shows beside them and keeps them active, and
# on_shape_commit() routes their drags to _commit_wall_run().
_WALL_TOOLS = ("place_unit", "unit_line", "unit_shape")


# GH #196: Edit > Scatter Units in Region's tooltip, and why it is greyed with a map open.
_SCATTER_TIP = (
    "Place random copies of the Units panel's object across the selected region or around the selected units, "
    "or move the selected units there"
)


def _gaia_only_refused_note(count: int) -> str:
    """GH #32's owner-change note: how many GAIA-only objects (or hosts holding
    one) were left with their owner, or "" for none."""
    return f"; {count} object(s) left as they were (GAIA-only or holding one)" if count else ""


# Sourced from iso_geometry, not redefined here, so Set Elevation's spinbox
# range and Stepped mode's canvas sizing (descape.render_iso.elevations_and_proj)
# can never drift apart -- see that constant's own comment for why Phase 4
# needs them to agree exactly.
ELEVATION_LEVEL_MAX = iso_geometry.MAX_ELEVATION


# Above this many tiles, a Paint Can fill with Trees or Eye candy checked
# confirms before writing anything -- a 480x480 map's full-map fill can plan
# >200k units (descape/terrain_units.py), which is not finishable at
# UnitEditModel's current per-op cost. Draw has no equivalent guard: its
# counts are bounded by brush size x drag length, and edit_history.
# abort_stroke() cannot un-paint tiles already written mid-drag, so there is
# no safe way to cancel out of a Draw stroke partway through anyway.
TERRAIN_UNIT_CONFIRM_THRESHOLD = 2000


@dataclasses.dataclass
class _RotationDrag:
    """A compass drag in progress (ViewerWindow._on_rotation_drag(), TASK-548):
    `start` is the dial's value at press, None for "(mixed)", and `record` the
    newest record the drag owns, into which _unit_edit folds each new spoke.

    `pushes` counts records rather than comparing cursors, which stay flat at
    the history's cap. `skipped` holds the members a spoke left alone, for the
    drag's one status line."""

    start: int | None
    record: UnitDiffRecord | None = None
    pushes: int = 0
    skipped: int = 0


STATUS_OK_COLOR = "#4caf50"
STATUS_ERROR_COLOR = "#e05252"


def status_color(ok: bool, *, on_tab_page: bool) -> str:
    """The OK or error hue above, shaded to stay readable under the live theme
    on the label's own background: a tab page, or else the window."""
    return status_text_color(STATUS_OK_COLOR if ok else STATUS_ERROR_COLOR, on_tab_page=on_tab_page)


# The hover label's idle text, shown whenever the cursor isn't over a tile.
HOVER_IDLE_TEXT = "Hover the map for tile info"

# GH #145: one View-page stat row, (label, count, note, parent label or "" at top level).
StatRow = tuple[str, str, str, str]


def _sync_hi_res() -> str:
    """unit_sprites.set_hi_res() from Graphics Quality (GH #241): x2 art at
    Enhanced, each file falling back to x1 where the install has no x2. And
    set_smooth_shrink() (GH #254): box-filtered shrinking at Default and above,
    on the native kernel only. Returns a status-log suffix at Enhanced naming the Enhanced Graphics DLC,
    or the active game's own x2 art where its root ships some (Return of Rome), else ""."""
    quality = settings.get_graphics_quality()
    enhanced = quality == settings.GRAPHICS_QUALITY_MAX
    unit_sprites.set_hi_res(enhanced)
    # On numpy the box shrink costs +46-80% at first paint (TASK-440 bench), so it stays nearest there.
    unit_sprites.set_smooth_shrink(
        quality >= settings.GRAPHICS_QUALITY_DEFAULT and composite_backend.active_backend() == "native"
    )
    if not enhanced:
        return ""
    if asset_source.get_install_path() is None:
        return ""  # no install, no unit art to name
    if unit_sprites.own_x2_art_installed():
        return f" ({game_dataset.game().display_name} high-resolution unit art on)"
    if unit_sprites.x2_art_installed():
        return " (high-resolution unit art on)"
    return " (Enhanced Graphics DLC not installed: regular unit art)"
