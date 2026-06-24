"""Cross-symbol critique and briefing built on shortlist analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

from fortuna.agentic.contracts import (
    BriefingItem,
    PortfolioCritique,
    PortfolioExposureNote,
    ShortlistBriefingResponse,
)
from fortuna.app.exposure_ranking import advisory_priority_score, rank_with_exposure_penalties
from fortuna.app.shortlist_analysis import analyze_market_shortlist


def build_shortlist_briefing(
    *,
    settings,
    engine_factory=None,
    registry=None,
    universe_limit: int = 10,
    analysis_limit: int = 5,
    timeframe: str = "5m",
    days: int = 30,
    source: str = "auto",
) -> ShortlistBriefingResponse:
    shortlist = analyze_market_shortlist(
        settings=settings,
        engine_factory=engine_factory,
        registry=registry,
        universe_limit=universe_limit,
        analysis_limit=analysis_limit,
        timeframe=timeframe,
        days=days,
        source=source,
    )
    if not shortlist.ok:
        return ShortlistBriefingResponse(
            ok=False,
            source=source,
            timeframe=timeframe,
            lookback_days=days,
            headline="Shortlist briefing unavailable",
            error=shortlist.error,
        )
    account_summary = _account_summary(engine_factory)
    portfolio = critique_portfolio(shortlist.items, account_summary=account_summary)
    ranked_items = _rank_briefing_items(shortlist.items)
    items = tuple(ranked_items)
    candidate_count = sum(1 for item in items if item.verdict == "candidate")
    headline = (
        f"{candidate_count} candidate setups, {len(items)} reviewed"
        if items
        else "No actionable shortlist setups"
    )
    return ShortlistBriefingResponse(
        ok=True,
        source=shortlist.source,
        timeframe=timeframe,
        lookback_days=days,
        headline=headline,
        items=items,
        portfolio=portfolio,
    )


def critique_portfolio(
    items,
    *,
    account_summary: Optional[dict[str, Any]] = None,
) -> PortfolioCritique:
    notes: list[PortfolioExposureNote] = []
    by_action: dict[str, list[str]] = defaultdict(list)
    by_exposure: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in items:
        if item.decision is None:
            continue
        action = item.decision.action
        if action in {"BUY", "SELL"}:
            by_action[action].append(item.symbol)
            by_exposure[(action, _exposure_key(item.symbol))].append(item.symbol)
    for action, symbols in by_action.items():
        if len(symbols) >= 3:
            notes.append(
                PortfolioExposureNote(
                    level="warn",
                    message=f"{len(symbols)} symbols lean {action}; directional crowding risk",
                    symbols=tuple(symbols),
                )
            )
    for (action, exposure_key), symbols in by_exposure.items():
        unique_symbols = tuple(dict.fromkeys(symbols))
        if len(unique_symbols) >= 2:
            notes.append(
                PortfolioExposureNote(
                    level="warn",
                    message=(
                        f"{len(unique_symbols)} shortlisted {action} ideas map to "
                        f"{exposure_key}; overlapping underlying exposure"
                    ),
                    symbols=unique_symbols,
                )
            )
    if account_summary:
        gross_exposure = float(account_summary.get("gross_exposure", 0.0) or 0.0)
        open_positions = int(account_summary.get("open_positions", 0) or 0)
        if open_positions >= 3:
            notes.append(
                PortfolioExposureNote(
                    level="info",
                    message=f"Existing open positions: {open_positions}",
                )
            )
        if gross_exposure > 0:
            notes.append(
                PortfolioExposureNote(
                    level="info",
                    message=f"Current gross exposure is {gross_exposure:.2f}",
                )
            )
        open_symbols = tuple(
            str(sym) for sym in account_summary.get("open_symbols", ()) if str(sym)
        )
        open_sides = {
            str(sym): str(side)
            for sym, side in (account_summary.get("open_position_sides") or {}).items()
        }
        by_open_exposure: dict[str, list[str]] = defaultdict(list)
        for symbol in open_symbols:
            by_open_exposure[_exposure_key(symbol)].append(symbol)
        for item in items:
            if item.decision is None:
                continue
            exposure_key = _exposure_key(item.symbol)
            overlapping = tuple(dict.fromkeys(by_open_exposure.get(exposure_key, ())))
            if not overlapping:
                continue
            action = item.decision.action
            same_side = any(open_sides.get(sym) == action for sym in overlapping)
            note_level = "warn" if same_side else "info"
            side_note = "same-side" if same_side else "cross-side"
            notes.append(
                PortfolioExposureNote(
                    level=note_level,
                    message=(
                        f"{item.symbol} overlaps existing {side_note} exposure in "
                        f"{exposure_key}"
                    ),
                    symbols=overlapping,
                )
            )
    summary = "Portfolio looks balanced enough for shortlist review"
    if notes:
        summary = notes[0].message
    return PortfolioCritique(summary=summary, notes=tuple(notes))


def _to_briefing_item(item) -> BriefingItem:
    decision = item.decision
    critique = item.critique
    return BriefingItem(
        symbol=item.symbol,
        action=decision.action,
        confidence=float(decision.confidence),
        verdict=critique.verdict if critique is not None else "watch",
        summary=(critique.summary if critique is not None else decision.summary),
        priority_score=advisory_priority_score(
            confidence=float(decision.confidence),
            verdict=critique.verdict if critique is not None else "watch",
            liquidity_score=item.liquidity_score,
            action=decision.action,
        ),
        rationale=tuple(decision.reasons[:2]),
        winning_strategy=item.winning_strategy,
        liquidity_score=item.liquidity_score,
        regime=item.regime,
        volume_ratio=item.volume_ratio,
        activity_score=item.activity_score,
    )


def _account_summary(engine_factory) -> Optional[dict[str, Any]]:
    if engine_factory is None:
        return None
    try:
        engine = engine_factory()
    except Exception:
        return None
    account = getattr(engine, "live_account", None)
    if account is None:
        return None
    try:
        summary = dict(account.to_summary())
    except Exception:
        return None
    try:
        positions = getattr(account, "positions", {}) or {}
        summary["open_symbols"] = tuple(str(sym) for sym in positions.keys())
        summary["open_position_sides"] = {
            str(sym): getattr(pos.side, "value", str(pos.side))
            for sym, pos in positions.items()
        }
    except Exception:
        pass
    return summary


def _exposure_key(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return text.split(".", 1)[0]


def _rank_briefing_items(items) -> list[BriefingItem]:
    base_items = [_to_briefing_item(item) for item in items if item.decision is not None]
    ranked = rank_with_exposure_penalties(
        base_items,
        symbol_of=lambda row: row.symbol,
        action_of=lambda row: row.action,
        base_score_of=lambda row: float(row.priority_score or 0.0),
        tie_breaker_of=lambda row: (0 if row.verdict == "candidate" else 1, row.symbol),
    )
    out: list[BriefingItem] = []
    for idx, result in enumerate(ranked, start=1):
        rationale = list(result.item.rationale)
        if result.exposure_penalty > 0:
            rationale.append(
                f"Exposure penalty {result.exposure_penalty:.2f} applied for crowding/overlap"
            )
        out.append(
            BriefingItem(
                symbol=result.item.symbol,
                action=result.item.action,
                confidence=result.item.confidence,
                verdict=result.item.verdict,
                summary=result.item.summary,
                selection_rank=idx,
                priority_score=result.adjusted_score,
                exposure_penalty=result.exposure_penalty,
                rationale=tuple(dict.fromkeys(rationale)),
                winning_strategy=result.item.winning_strategy,
                liquidity_score=result.item.liquidity_score,
                regime=result.item.regime,
                volume_ratio=result.item.volume_ratio,
                activity_score=result.item.activity_score,
            )
        )
    return out
