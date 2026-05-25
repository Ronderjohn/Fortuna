#!/usr/bin/env python
"""Launch Fortuna Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
app = _ROOT / "app" / "streamlit_app.py"

if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]],
            cwd=str(_ROOT),
        )
    )
