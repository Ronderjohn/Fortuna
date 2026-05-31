#!/usr/bin/env python
"""Read-only operator preflight for Fortuna advisory runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.operator_preflight import render_preflight, run_operator_preflight  # noqa: E402
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run operator preflight checks")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--symbol", default="", help="Override symbol for RL pointer checks")
    parser.add_argument("--strict", action="store_true", help="Upgrade key warnings to failures")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    result = run_operator_preflight(
        settings,
        symbol=args.symbol or None,
        strict=args.strict,
    )
    print(render_preflight(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
