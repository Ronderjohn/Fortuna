"""Cross-platform helper to prevent the OS from sleeping during long jobs.

Background
----------
On Windows, ``WakeToRun`` on a Scheduled Task is *only* enough to wake the
machine to *start* the task. Without an active power-state lock, Windows can
put the box back to sleep within minutes — even while a Python process is
busy running. We hit this on a real nightly run: the task started at 04:45,
the OS slept again, and the script silently froze for 3.5 hours until the
user touched the keyboard.

This module provides a tiny ``keep_system_awake()`` context manager (and a
``activate()`` convenience for fire-and-forget) that issues
``SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`` on Windows so
the kernel keeps the machine awake until the process exits or
``release()`` is called explicitly.

On non-Windows platforms it is a no-op so callers can use it portably.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


# Win32 constants from ``WinBase.h`` / ``synchapi.h``.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040  # works in modern standby (S0)
_ES_DISPLAY_REQUIRED = 0x00000002


def _set_state(flags: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        prev = ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))
        return prev != 0
    except Exception as exc:  # noqa: BLE001 — keep tolerant on edge OSes.
        logger.warning("[keep_awake] SetThreadExecutionState failed: %s", exc)
        return False


def activate(*, allow_away_mode: bool = True, keep_display: bool = False) -> bool:
    """Request that Windows not sleep until ``release()`` is called.

    Returns ``True`` if the request was accepted. On non-Windows OSes this
    is a no-op that returns ``False`` (callers can ignore the return value).

    Parameters
    ----------
    allow_away_mode:
        Add ``ES_AWAYMODE_REQUIRED`` so the machine stays "awake" even
        when the user is away (works under modern-standby S0 too).
    keep_display:
        Also keep the monitor on. We deliberately default to ``False``
        for nightly batch jobs.
    """
    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    if allow_away_mode:
        flags |= _ES_AWAYMODE_REQUIRED
    if keep_display:
        flags |= _ES_DISPLAY_REQUIRED
    ok = _set_state(flags)
    if ok:
        logger.info(
            "[keep_awake] system-required lock active (away_mode=%s, display=%s)",
            allow_away_mode,
            keep_display,
        )
    return ok


def release() -> bool:
    """Release the previously-issued power-state request."""
    ok = _set_state(_ES_CONTINUOUS)
    if ok:
        logger.info("[keep_awake] power-state lock released")
    return ok


@contextmanager
def keep_system_awake(
    *, allow_away_mode: bool = True, keep_display: bool = False
) -> Iterator[bool]:
    """Context manager wrapping ``activate`` + ``release``."""
    ok = activate(allow_away_mode=allow_away_mode, keep_display=keep_display)
    try:
        yield ok
    finally:
        if ok:
            release()


__all__ = ["activate", "release", "keep_system_awake"]
