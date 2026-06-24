#!/usr/bin/env python
"""Build a liquid market-universe shortlist for advisory/training workflows."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fortuna.app.market_universe import build_market_universe  # noqa: E402
from fortuna.config.settings import Settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Fortuna market-universe shortlist")
    parser.add_argument("--config", default="", help="Optional YAML config path")
    parser.add_argument("--limit", type=int, default=15, help="Number of rows to emit")
    parser.add_argument("--timeframe", default="1d", help="OHLCV timeframe for ranking")
    parser.add_argument("--days", type=int, default=30, help="Lookback days for ranking")
    parser.add_argument(
        "--source",
        default="auto",
        choices=("auto", "screener", "registry", "smartapi"),
        help="Candidate seed source (smartapi uses bounded instrument registry pool)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional CSV output path; defaults to stdout only",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    settings = Settings.from_yaml(cfg_path)
    response = build_market_universe(
        settings=settings,
        limit=args.limit,
        timeframe=args.timeframe,
        days=args.days,
        source=args.source,
    )
    if not response.ok:
        message = response.error.message if response.error is not None else "unknown error"
        print(f"[fail] {message}")
        return 1

    print(
        f"[ok] source={response.source} timeframe={response.timeframe} "
        f"days={response.lookback_days} rows={len(response.candidates)}"
    )
    if response.provider_summary:
        print(f"[provider] {response.provider_summary}")
    if response.fundamentals_overlay_summary:
        print(f"[overlay] {response.fundamentals_overlay_summary}")
    if response.fundamentals_overlay_diagnostics:
        for line in response.fundamentals_overlay_diagnostics[:5]:
            print(f"[overlay-diag] {line}")
    header = (
        "symbol,liquidity_score,avg_turnover,avg_volume,trend_pct,last_close,source,"
        "sector,market_cap_bucket,operator_quality_score,fundamentals_overlay_adjustment"
    )
    print(header)
    rows: list[dict[str, object]] = []
    for item in response.candidates:
        row = {
            "symbol": item.symbol,
            "display_name": item.display_name,
            "liquidity_score": item.liquidity_score,
            "avg_turnover": item.avg_turnover,
            "avg_volume": item.avg_volume,
            "trend_pct": item.trend_pct,
            "last_close": item.last_close,
            "source": item.source,
            "sector": item.sector,
            "market_cap_bucket": item.market_cap_bucket,
            "operator_quality_score": item.operator_quality_score,
            "fundamentals_overlay_adjustment": item.fundamentals_overlay_adjustment,
            "notes": "|".join(item.notes),
        }
        rows.append(row)
        print(
            f"{item.symbol},{item.liquidity_score},{item.avg_turnover},"
            f"{item.avg_volume},{item.trend_pct},{item.last_close},{item.source},"
            f"{item.sector},{item.market_cap_bucket},{item.operator_quality_score},"
            f"{item.fundamentals_overlay_adjustment}"
        )

    if args.out:
        out_path = settings.resolve_path(Path(args.out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "symbol",
                    "display_name",
                    "liquidity_score",
                    "avg_turnover",
                    "avg_volume",
                    "trend_pct",
                    "last_close",
                    "source",
                    "sector",
                    "market_cap_bucket",
                    "operator_quality_score",
                    "fundamentals_overlay_adjustment",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"[write] {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
