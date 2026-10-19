"""Drops known-harmless Qt console warnings (GH #236).

Qt 5's Wayland plugin cannot activate a window, so every `activateWindow()`
(DEscape's own dialog openers and Qt's internal calls alike) prints "Wayland
does not support QWindow::requestActivate()" and floods the terminal. The
handler installed here drops only the exact messages in `SUPPRESSED` and
prints everything else as Qt's default handler would.

PyQt5 cannot hand a message back to Qt's default handler (installing one
returns None), so pass-through formats with `qFormatLogMessage`, which keeps
`QT_MESSAGE_PATTERN` working. `QT_LOGGING_RULES` still applies, since Qt
filters by category before calling any handler.
"""

from __future__ import annotations

import sys

SUPPRESSED = frozenset({"Wayland does not support QWindow::requestActivate()"})

_handler = None


def is_suppressed(message: str) -> bool:
    return message in SUPPRESSED


def install() -> None:
    """Installs the filtering handler. Call before the QApplication exists;
    a second call is a no-op."""
    global _handler
    if _handler is not None:
        return
    from PyQt5.QtCore import qFormatLogMessage, qInstallMessageHandler

    def handler(msg_type, context, message):
        if is_suppressed(message):
            return
        formatted = qFormatLogMessage(msg_type, context, message)
        # An empty result means QT_MESSAGE_PATTERN filtered it out; Qt prints nothing then.
        stream = sys.stderr
        if stream is None or (not formatted and message):
            return
        # Qt's own fprintf ignores write errors; a raise here would reach the crash hook.
        try:
            stream.write(formatted + "\n")
            stream.flush()
        except (OSError, ValueError):
            pass

    # Also the install-once flag; PyQt5 keeps its own reference to the callable.
    _handler = handler
    qInstallMessageHandler(handler)
