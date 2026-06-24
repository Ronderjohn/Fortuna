"""Liquidity scout role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus


def run(response):
    return response, compose_role(response)


def compose_role(response) -> AgentRoleStatus:
    candidates = tuple(getattr(response, "candidates", ()) or ())
    ranked = sorted(
        candidates,
        key=lambda row: (
            -(float(getattr(row, "liquidity_score", 0.0) or 0.0)),
            -(float(getattr(row, "avg_turnover", 0.0) or 0.0)),
            row.symbol,
        ),
    )
    focus = tuple(row.symbol for row in ranked[:5])
    top = ranked[0] if ranked else None
    notes: list[str] = []
    if top is not None:
        score = getattr(top, "liquidity_score", None)
        if score is not None:
            notes.append(f"top_liquidity={top.symbol}:{float(score):.2f}")
        turnover = getattr(top, "avg_turnover", None)
        if turnover is not None:
            notes.append(f"top_turnover={top.symbol}:{float(turnover):.0f}")
        avg_volume = getattr(top, "avg_volume", None)
        if avg_volume is not None:
            notes.append(f"top_volume={top.symbol}:{float(avg_volume):.0f}")
        notes.append(f"seed_source={response.source}")
    headline = (
        f"{len(ranked)} liquidity-ranked names prepared from {response.source} seeds"
        if response.ok and ranked
        else "Liquidity scout unavailable"
    )
    return AgentRoleStatus(
        name="liquidity_scout",
        ok=bool(response.ok and ranked),
        headline=headline,
        focus_symbols=focus,
        notes=tuple(notes[:5]),
        error=response.error if not response.ok else None,
    )
