from __future__ import annotations

from fortuna.app.nightly_alignment_view import (
    format_latest_nightly_discovery_context,
    format_latest_nightly_execution_posture,
)


def test_format_latest_nightly_execution_posture_includes_source_and_review():
    text = format_latest_nightly_execution_posture(
        {
            "latest_execution_target": "ml",
            "latest_execution_model_family": "ml",
            "latest_execution_selection_source": "training_research",
            "latest_promotion_review_model_kind": "ml_scorer",
        }
    )

    assert text == (
        "Latest nightly execution: target `ml` | family `ml` | "
        "source `training_research` | review `ml_scorer`"
    )


def test_format_latest_nightly_execution_posture_prefers_plural_review_kinds():
    text = format_latest_nightly_execution_posture(
        {
            "latest_execution_target": "all",
            "latest_execution_model_family": "hybrid",
            "latest_execution_selection_source": "training_research",
            "latest_promotion_review_model_kind": "mixed",
            "latest_promotion_review_model_kinds": ["ml_scorer", "rl_policy"],
        }
    )

    assert text == (
        "Latest nightly execution: target `all` | family `hybrid` | "
        "source `training_research` | reviews `ml_scorer, rl_policy`"
    )


def test_format_latest_nightly_discovery_context_includes_summary_and_target():
    text = format_latest_nightly_discovery_context(
        {
            "latest_training_research_discovery_summary": (
                "preferred=2 | liquidity=TCS.NS,SBIN.NS | posture=rl"
            ),
            "latest_training_research_discovery_recommended_target": "rl",
        }
    )

    assert text == (
        "Latest discovery context: preferred=2 | liquidity=TCS.NS,SBIN.NS | "
        "posture=rl | target `rl`"
    )
