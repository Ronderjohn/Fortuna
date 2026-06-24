"""Preset prompts for the dashboard assistant surface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantQuickAction:
    label: str
    prompt: str
    caption: str


def build_dashboard_assistant_actions(
    *,
    symbol: str = "RELIANCE",
    timeframe: str = "5m",
    days: int = 20,
    include_expert: bool = False,
) -> tuple[AssistantQuickAction, ...]:
    normalized_symbol = (symbol or "RELIANCE").strip().upper()
    lookback = max(1, int(days))
    tf = (timeframe or "5m").strip()
    actions = [
        AssistantQuickAction(
            label="Search symbol",
            prompt=f"/search {normalized_symbol}",
            caption="Resolve an instrument before asking for a signal.",
        ),
        AssistantQuickAction(
            label="Analyze symbol",
            prompt=f"/analyze {normalized_symbol} {tf} {lookback}d",
            caption="Run the compact signal reply on one instrument.",
        ),
        AssistantQuickAction(
            label="Analyze future",
            prompt=f"/analyze {normalized_symbol} FUT {tf} {lookback}d",
            caption="Check the front-month futures contract.",
        ),
        AssistantQuickAction(
            label="Model health",
            prompt="/health",
            caption="Check runtime and promoted-model readiness.",
        ),
    ]
    if include_expert:
        actions.extend(
            [
                AssistantQuickAction(
                    label="Liquid universe",
                    prompt="/universe 1d 30d limit=10",
                    caption="Rank liquid and active names from the current universe source.",
                ),
                AssistantQuickAction(
                    label="Top setups",
                    prompt=f"/brief {tf} {lookback}d limit=5",
                    caption="Summarize the best shortlisted setups with portfolio notes.",
                ),
                AssistantQuickAction(
                    label="Allocate setups",
                    prompt=f"/allocate {tf} {lookback}d limit=5 maxpos=3 perexp=1 sameside=2",
                    caption=(
                        "Select which shortlisted setups fit together under "
                        "simple portfolio limits."
                    ),
                ),
                AssistantQuickAction(
                    label="ML candidates",
                    prompt=f"/candidates ml {tf} {lookback}d limit=8",
                    caption="Show shortlist-driven ML training candidates.",
                ),
                AssistantQuickAction(
                    label="RL candidates",
                    prompt=f"/candidates rl {tf} {lookback}d limit=8",
                    caption="Show shortlist-driven RL training candidates.",
                ),
                AssistantQuickAction(
                    label="Training research",
                    prompt=f"/research all {tf} {lookback}d limit=8",
                    caption="Inspect the typed ML/RL refresh plan with nightly posture context.",
                ),
            ]
        )
    return tuple(actions)


def build_dashboard_assistant_examples(
    *,
    symbol: str = "RELIANCE",
    timeframe: str = "15m",
    days: int = 20,
    include_expert: bool = False,
) -> tuple[str, ...]:
    normalized_symbol = (symbol or "RELIANCE").strip().upper()
    lookback = max(1, int(days))
    tf = (timeframe or "15m").strip()
    examples = [
        f"/search {normalized_symbol}",
        f"/analyze {normalized_symbol}",
        f"/analyze {normalized_symbol} FUT",
        "Should I enter NIFTY CE 25000 28MAY2026?",
        f"How is {normalized_symbol.title()} looking on {tf} for {lookback}d?",
    ]
    if include_expert:
        examples.extend(
            [
                "/brief 5m 20d limit=5",
                "/allocate 5m 20d limit=5 maxpos=3 perexp=1 sameside=2",
                f"/research all {tf} {lookback}d limit=8",
                "/candidates rl 15m 30d limit=8",
            ]
        )
    return tuple(examples)
