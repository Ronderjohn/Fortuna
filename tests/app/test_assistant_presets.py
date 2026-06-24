from __future__ import annotations

from fortuna.app.assistant_presets import (
    build_dashboard_assistant_actions,
    build_dashboard_assistant_examples,
)


def test_build_dashboard_assistant_actions_includes_multi_agent_workflow():
    actions = build_dashboard_assistant_actions(symbol="SBIN", timeframe="15m", days=30)
    prompts = {row.label: row.prompt for row in actions}
    assert prompts["Analyze symbol"] == "/analyze SBIN 15m 30d"
    assert prompts["Liquid universe"] == "/universe 1d 30d limit=10"
    assert prompts["Top setups"] == "/brief 15m 30d limit=5"
    assert prompts["Allocate setups"] == "/allocate 15m 30d limit=5 maxpos=3 perexp=1 sameside=2"
    assert prompts["ML candidates"] == "/candidates ml 15m 30d limit=8"
    assert prompts["RL candidates"] == "/candidates rl 15m 30d limit=8"
    assert prompts["Training research"] == "/research all 15m 30d limit=8"
    assert prompts["Model health"] == "/health"


def test_build_dashboard_assistant_examples_cover_brief_and_candidates():
    examples = build_dashboard_assistant_examples(symbol="ITC", timeframe="5m", days=10)
    assert "/search ITC" in examples
    assert "/brief 5m 20d limit=5" in examples
    assert "/allocate 5m 20d limit=5 maxpos=3 perexp=1 sameside=2" in examples
    assert "/research all 5m 10d limit=8" in examples
    assert "/candidates rl 15m 30d limit=8" in examples
    assert "How is Itc looking on 5m for 10d?" in examples
