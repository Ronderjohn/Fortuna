"""Activity scout role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus


def run(response):
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    candidates = tuple(getattr(response, "candidates", ()) or ())
    ranked = [
        row
        for row in candidates
        if (
            getattr(row, "activity_score", None) is not None
            or getattr(row, "trend_pct", None) is not None
            or getattr(row, "volume_ratio", None) is not None
            or str(getattr(row, "regime", "") or "").strip()
        )
    ]
    ranked = sorted(
        ranked,
        key=lambda row: (
            -(float(getattr(row, "activity_score", 0.0) or 0.0)),
            -(abs(float(getattr(row, "trend_pct", 0.0) or 0.0))),
            row.symbol,
        ),
    )
    focus = tuple(row.symbol for row in ranked[:5])
    notes: list[str] = []
    if ranked:
        top = ranked[0]
        regime = str(getattr(top, "regime", "") or "").strip().lower()
        if regime:
            notes.append(f"top_regime={top.symbol}:{regime}")
        volume_ratio = getattr(top, "volume_ratio", None)
        if volume_ratio is not None:
            notes.append(f"top_volume_ratio={top.symbol}:{float(volume_ratio):.2f}")
        trend_pct = getattr(top, "trend_pct", None)
        if trend_pct is not None:
            notes.append(f"top_trend={top.symbol}:{float(trend_pct):+.2f}%")
        notes.append(f"coverage={len(ranked)}/{len(candidates)}")
    headline = (
        f"{len(ranked)} activity/trend names enriched from local OHLCV context"
        if response.ok and ranked
        else "Activity scout awaiting OHLCV enrichment"
    )
    return AgentRoleStatus(
        name="activity_scout",
        ok=bool(response.ok and ranked),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:5]),
        error=response.error if not response.ok else None,
    )
