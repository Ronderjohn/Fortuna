"""Deterministic baseline Sharpe lookup for RL policy comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fortuna.config.settings import Settings, get_settings
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _symbol_stem(symbol: str) -> str:
    return symbol.upper().strip().replace(".", "_")


def _read_learner_sharpe(logs_dir: Path, symbol: str, timeframe: str) -> Optional[float]:
    """Best-effort read from paper league / learner JSON state."""
    stem = f"rl_{_symbol_stem(symbol)}_{timeframe}"
    path = logs_dir / f"{stem}_learning.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("fold_history") or []
        if not history:
            return None
        last = history[-1]
        profit = float(last.get("profit_pct", 0.0))
        trades = int(last.get("total_trades", 0))
        if trades <= 0:
            return None
        return profit / max(trades, 1)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[RL baseline] learner read failed for %s: %s", stem, exc)
        return None


def _read_nightly_report_sharpe(reports_dir: Path, symbol: str) -> Optional[float]:
    """Scan latest nightly JSON for deterministic strategy champion Sharpe."""
    if not reports_dir.is_dir():
        return None
    json_files = sorted(reports_dir.glob("*.json"), reverse=True)
    for path in json_files[:5]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            det = payload.get("deterministic") or payload.get("paper_league") or {}
            if isinstance(det, dict):
                for entry in det.get("results") or det.get("symbols") or []:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("symbol") == symbol:
                        sharpe = entry.get("sharpe") or entry.get("sharpe_ratio")
                        if sharpe is not None:
                            return float(sharpe)
        except Exception:  # noqa: BLE001
            continue
    return None


def baseline_sharpe_for_symbol(
    symbol: str,
    timeframe: str = "5m",
    *,
    settings: Optional[Settings] = None,
) -> Optional[float]:
    """Return deterministic baseline Sharpe for symbol/timeframe if known.

    Lookup order (no network):
    1. Latest nightly report deterministic sweep
    2. ``logs/learner/`` adaptive state for symbol
    """
    settings = settings or get_settings()
    reports_dir = settings.resolve_path(Path("reports/nightly"))
    logs_dir = settings.resolve_path(settings.logs_dir) / "learner"

    sharpe = _read_nightly_report_sharpe(reports_dir, symbol)
    if sharpe is not None:
        return sharpe
    return _read_learner_sharpe(logs_dir, symbol, timeframe)
