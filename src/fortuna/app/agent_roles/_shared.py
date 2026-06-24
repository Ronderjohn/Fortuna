"""Shared helpers for agent role composition."""

from __future__ import annotations

from collections import defaultdict


def universe_blend_note(response) -> str:
    provider_summary = str(getattr(response, "provider_summary", "") or "").strip()
    if provider_summary:
        return provider_summary
    source = str(getattr(response, "source", "") or "auto").strip().lower()
    if source == "screener":
        return "blend=screener_liquidity+ohlcv_activity"
    if source == "smartapi":
        return "blend=smartapi_instruments+ohlcv_activity"
    if source == "registry":
        return "blend=registry_seeds+ohlcv_activity"
    return "blend=market_seeds+ohlcv_activity"


def discovery_alignment(response) -> tuple[str | None, str | None]:
    candidates = tuple(getattr(response, "candidates", ()) or ())
    if not candidates:
        return None, None
    liquidity_ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    activity_ranked = [
        row
        for row in candidates
        if (
            getattr(row, "activity_score", None) is not None
            or getattr(row, "trend_pct", None) is not None
            or getattr(row, "volume_ratio", None) is not None
            or str(getattr(row, "regime", "") or "").strip()
        )
    ]
    activity_ranked = sorted(
        activity_ranked,
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )
    if not liquidity_ranked or not activity_ranked:
        return None, None
    compare_count = min(3, len(liquidity_ranked), len(activity_ranked))
    if compare_count <= 0:
        return None, None
    liquidity_top = tuple(row.symbol for row in liquidity_ranked[:compare_count])
    activity_top = tuple(row.symbol for row in activity_ranked[:compare_count])
    overlap = tuple(symbol for symbol in liquidity_top if symbol in activity_top)
    overlap_count = len(overlap)
    if overlap_count >= compare_count:
        status = "aligned"
    elif overlap_count > 0:
        status = "partial"
    else:
        status = "drifting"
    summary = f"discovery {status} {overlap_count}/{compare_count}"
    overlap_text = ", ".join(overlap) if overlap else None
    return summary, overlap_text


def basket_research_alignment(*, allocation, research) -> tuple[str | None, str | None]:
    if allocation is None or research is None:
        return None, None
    selected_symbols = tuple(getattr(allocation, "selected_symbols", ()) or ())
    if not selected_symbols:
        return None, None
    ml_symbols = {str(symbol) for symbol in getattr(research, "ml_symbols", ()) or ()}
    rl_symbols = {str(symbol) for symbol in getattr(research, "rl_symbols", ()) or ()}
    if not ml_symbols and not rl_symbols:
        return None, None

    target_counts: dict[str, int] = defaultdict(int)
    overlap_count = 0
    for symbol in selected_symbols:
        target = "none"
        in_ml = symbol in ml_symbols
        in_rl = symbol in rl_symbols
        if in_ml and in_rl:
            target = "both"
            overlap_count += 1
        elif in_rl:
            target = "rl"
            overlap_count += 1
        elif in_ml:
            target = "ml"
            overlap_count += 1
        target_counts[target] += 1

    selected_count = len(selected_symbols)
    if overlap_count >= selected_count:
        status = "aligned"
    elif overlap_count > 0:
        status = "partial"
    else:
        status = "drifting"
    summary = f"basket/research {status} {overlap_count}/{selected_count}"
    target_mix = ", ".join(f"{key}={value}" for key, value in sorted(target_counts.items()))
    return summary, target_mix or None


def effective_research_refresh_target(
    *,
    requested_target,
    discovery_target,
) -> str | None:
    normalized = [
        str(target or "").strip().lower()
        for target in (requested_target, discovery_target)
        if str(target or "").strip()
    ]
    if not normalized:
        return None
    concrete = {target for target in normalized if target in {"ml", "rl"}}
    if len(concrete) >= 2:
        return "all"
    if len(concrete) == 1:
        return next(iter(concrete))
    if "all" in normalized:
        return "all"
    return normalized[0]


def targets_mismatch(
    *,
    requested_target,
    discovery_target,
) -> bool:
    requested = str(requested_target or "").strip().lower()
    discovery = str(discovery_target or "").strip().lower()
    if requested not in {"ml", "rl"} or discovery not in {"ml", "rl"}:
        return False
    return requested != discovery
