#!/usr/bin/env python
"""Print pre-framework technical readiness checks (informational; not operator preflight)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.framework_readiness import (  # noqa: E402
    render_framework_readiness,
    run_framework_readiness_checks,
)


def main() -> int:
    result = run_framework_readiness_checks()
    print(render_framework_readiness(result))
    return 0 if result.technical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
