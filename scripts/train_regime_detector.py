#!/usr/bin/env python
"""Train the offline market-regime classifier.

Example:
    uv run python scripts/train_regime_detector.py `
        --symbol ICICIBANK.NS --timeframe 5m --days 180 `
        --output models/regime/classifier.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.utils.runtime_env import apply_low_spec_gpu_defaults  # noqa: E402

apply_low_spec_gpu_defaults()

# Warm pre-existing circular: importing calendar first breaks the
# data.manager -> backtesting.standard -> paper -> data.manager cycle.
from fortuna.backtesting.standard.calendar import filter_session_bars  # noqa: E402,F401

from fortuna.config.settings import get_settings  # noqa: E402
from fortuna.data.manager import MarketDataManager  # noqa: E402
from fortuna.rl.inference.regime_detector import RegimeDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Fortuna RegimeDetector")
    parser.add_argument("--symbol", default="ICICIBANK.NS")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--output", default="models/regime/classifier.joblib")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = get_settings()
    mdm = MarketDataManager(settings, data_source=settings.data_source)
    df = mdm.get_ohlcv(args.symbol, args.timeframe, days=args.days)

    det = RegimeDetector()
    meta = det.train(df, test_fraction=args.test_fraction, random_state=args.seed)

    output = Path(args.output)
    det.save(output)

    print(f"\n=== RegimeDetector trained ===")
    print(f"Symbol/TF:      {args.symbol} {args.timeframe}")
    print(f"Sessions:       train={meta.n_train}  test={meta.n_test}")
    print(f"Test accuracy:  {meta.accuracy:.3f}")
    print(f"Saved to:       {output}")
    if meta.n_test > 0:
        print(f"Confusion matrix (rows=true, cols=pred):")
        for row in meta.confusion_matrix:
            print("  ", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
