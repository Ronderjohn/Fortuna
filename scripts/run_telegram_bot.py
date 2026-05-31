#!/usr/bin/env python
"""Run the Fortuna Telegram polling assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.config.settings import Settings  # noqa: E402
from fortuna.telegram.runtime import TelegramBotRuntime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Fortuna Telegram assistant")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--poll-timeout", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    settings = Settings.from_yaml(Path(args.config) if args.config else None)
    if not settings.telegram_enabled:
        raise SystemExit("FORTUNA_TELEGRAM_ENABLED=1 is required")
    runtime = TelegramBotRuntime(settings)
    runtime.run_forever(
        poll_timeout=int(args.poll_timeout),
        sleep_seconds=float(args.sleep_seconds),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
