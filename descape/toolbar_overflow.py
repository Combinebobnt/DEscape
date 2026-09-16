"""Pure width-partition logic for the "More Tools" toolbar overflow button.

No Qt import, matching how viewer_common.py and settings.py keep Qt-free
logic out of viewer.py: this is a plain arithmetic function ViewerWindow
feeds real pixel widths into, so it can be unit-tested with no QApplication.
"""

from __future__ import annotations

from typing import Sequence


def partition(
    available_px: int, item_widths: Sequence[tuple[str, int]], more_button_px: int
) -> tuple[list[str], list[str]]:
    """(on_bar, overflowed) given ordered (id, width) pairs.

    Two-pass: fit without the More Tools button; if everything fits, no
    button and no overflow. Otherwise re-fit with more_button_px reserved.
    Two passes terminate; a naive single pass oscillates, because adding
    the button consumes the width that made everything fit.

    Overflow is a strict right-hand suffix of `item_widths` in the order
    given -- the first item that doesn't fit, and everything after it,
    overflows. No reordering, so the bar never reshuffles under the cursor
    during a drag-resize.
    """
    total = sum(width for _, width in item_widths)
    if total <= available_px:
        return [item_id for item_id, _ in item_widths], []

    budget = available_px - more_button_px
    on_bar: list[str] = []
    used = 0
    for item_id, width in item_widths:
        if used + width > budget:
            break
        on_bar.append(item_id)
        used += width

    overflowed = [item_id for item_id, _ in item_widths[len(on_bar) :]]
    return on_bar, overflowed
