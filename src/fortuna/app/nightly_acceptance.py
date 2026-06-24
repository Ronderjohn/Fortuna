"""Local dry-run harness for candidate-style nightly acceptance."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from unittest.mock import patch

from fortuna.agentic.contracts import (
    AllocationDecision,
    BriefingItem,
    MarketUniverseCandidate,
    MarketUniverseResponse,
    MultiAgentWorkflowResponse,
    NightlyAlignmentStatus,
    NightlyArtifactLinkageSummary,
    PortfolioAllocationResponse,
    ShortlistAnalysisItem,
    ShortlistAnalysisResponse,
    ShortlistBriefingResponse,
    TrainingCandidate,
    TrainingCandidateResponse,
    TrainingResearchPlanResponse,
    TrainingResearchRow,
)
from fortuna.app import nightly_basket
from fortuna.app.acceptance_bundle import (
    AcceptanceBundle,
    export_acceptance_bundle,
    gather_acceptance_bundle,
)
from fortuna.app.agent_roles import (
    activity_scout,
    briefing_agent,
    instrument_analyst,
    liquidity_scout,
    operations_monitor,
    portfolio_critic,
    research_planner,
    universe_scout,
)
from fortuna.app.operator_workflow import (
    WorkflowSnapshot,
    export_workflow_snapshot,
    summarize_multi_agent_workflow,
)
from fortuna.app.promotion_review import (
    build_promotion_review,
    export_promotion_review,
)
from fortuna.app.training_candidates import export_training_candidates
from fortuna.app.training_research import export_training_research_plan
from fortuna.models import PromotionRecord, PromotionStatus, promote
from fortuna.models.metadata import ModelKind
from fortuna.observability.recorder import workflow_boundary
from fortuna.rl.training.checkpoint import OOSMetricsSummary, PolicyCheckpoint

_REPLAY_REQUIRED_ARTIFACTS = frozenset(
    {"nightly_report", "training_candidates", "training_research", "workflow_snapshot"}
)
_REPLAY_OPTIONAL_ARTIFACTS = frozenset({"promotion_review"})


@dataclass(frozen=True)
class NightlyDryRunArtifacts:
    root: str
    training_candidate_manifest_path: str
    training_research_plan_path: str
    workflow_snapshot_path: str
    nightly_report_path: str
    promotion_review_path: str
    live_pointer_path: str
    training_execution_target: Optional[str] = None
    training_execution_model_family: Optional[str] = None
    training_execution_selection_source: Optional[str] = None
    nightly_report_markdown_path: Optional[str] = None
    acceptance_bundle_path: Optional[str] = None
    replay_mode: str = "candidate"
    linkage: Optional[NightlyArtifactLinkageSummary] = None

    def to_dict(self) -> dict[str, str | None]:
        payload = {
            "root": self.root,
            "training_candidate_manifest_path": self.training_candidate_manifest_path,
            "training_research_plan_path": self.training_research_plan_path,
            "workflow_snapshot_path": self.workflow_snapshot_path,
            "nightly_report_path": self.nightly_report_path,
            "nightly_report_markdown_path": self.nightly_report_markdown_path,
            "promotion_review_path": self.promotion_review_path,
            "live_pointer_path": self.live_pointer_path,
            "training_execution_target": self.training_execution_target,
            "training_execution_model_family": self.training_execution_model_family,
            "training_execution_selection_source": self.training_execution_selection_source,
            "acceptance_bundle_path": self.acceptance_bundle_path,
            "replay_mode": self.replay_mode,
        }
        return payload


def build_candidate_nightly_dry_run(
    *,
    settings,
    out_dir: Path | str,
    engine: Any | None = None,
    symbol: str = "RELIANCE.NS",
    timeframe: str = "5m",
    days: int = 20,
    source: str = "screener",
    run_id: str = "dry_run_acceptance",
    export_bundle_path: Optional[Path | str] = None,
    export_bundle_format: str = "md",
) -> tuple[AcceptanceBundle, NightlyDryRunArtifacts]:
    with workflow_boundary(
        settings,
        event_name="nightly_acceptance",
        module="fortuna.app.nightly_acceptance",
        workflow_id="nightly_acceptance",
        symbol=symbol,
        context={"mode": "dry_run", "run_id": run_id},
    ) as span:
        bundle, artifacts = _build_candidate_nightly_dry_run_impl(
            settings=settings,
            out_dir=out_dir,
            engine=engine,
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            source=source,
            run_id=run_id,
            export_bundle_path=export_bundle_path,
            export_bundle_format=export_bundle_format,
        )
        span.set_context(
            bundle_path=artifacts.acceptance_bundle_path,
            nightly_report_path=artifacts.nightly_report_path,
        )
        return bundle, artifacts


def _build_candidate_nightly_dry_run_impl(
    *,
    settings,
    out_dir: Path | str,
    engine: Any | None = None,
    symbol: str = "RELIANCE.NS",
    timeframe: str = "5m",
    days: int = 20,
    source: str = "screener",
    run_id: str = "dry_run_acceptance",
    export_bundle_path: Optional[Path | str] = None,
    export_bundle_format: str = "md",
) -> tuple[AcceptanceBundle, NightlyDryRunArtifacts]:
    root = Path(out_dir)
    acceptance_settings = settings.model_copy(update={"project_root": root})
    training_candidate_path = root / "reports" / "nightly" / "training_candidates.json"
    training_research_path = root / "reports" / "nightly" / "training_research_plan.json"
    workflow_path = root / "reports" / "workflow_snapshot.json"
    nightly_dir = root / "reports" / "nightly"
    models_root = root / "models"
    validated_dir = models_root / "validated" / run_id
    validated_dir.mkdir(parents=True, exist_ok=True)
    training_research_response = _build_training_research_response(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=days,
    )

    _write_workflow_snapshot(
        workflow_path,
        source=source,
        timeframe=timeframe,
        lookback_days=days,
        training_research_response=training_research_response,
    )
    _write_training_candidate_manifest(
        training_candidate_path,
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=days,
    )
    _write_training_research_plan(
        training_research_path,
        response=training_research_response,
    )
    live_ptr = _write_promoted_rl_artifact(
        validated_dir=validated_dir,
        models_root=models_root,
        workflow_snapshot_path=workflow_path,
        symbol=symbol,
        timeframe=timeframe,
        run_id=run_id,
    )
    review = build_promotion_review(
        kind=ModelKind.RL_POLICY,
        models_root=models_root,
        symbol=symbol,
        run_id=run_id,
        settings=settings,
    )
    if review is None:
        raise RuntimeError("failed to build promotion review for dry-run artifact")
    review_path = nightly_dir / "promotions" / f"{symbol.replace('.', '_')}__{run_id}.md"
    export_promotion_review(review, review_path, fmt="md")
    nightly_report_path, nightly_report_markdown_path = _write_realistic_nightly_report(
        nightly_dir,
        training_candidate_path=training_candidate_path,
        training_research_path=training_research_path,
        workflow_snapshot_path=workflow_path,
        promotion_review_path=review_path,
        symbol=symbol,
        source_mode="derived_training_candidates",
        training_research_response=training_research_response,
        execution_target="rl",
        execution_model_family="rl",
        execution_selection_source="training_research",
        promotion_review_model_kind="rl_policy",
    )

    bundle = gather_acceptance_bundle(
        settings=acceptance_settings,
        engine=_engine_for_acceptance(engine, acceptance_settings, symbol),
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        symbol=symbol,
        nightly_report_path=nightly_report_path,
        include_model_health=True,
    )

    bundle_path = None
    if export_bundle_path is not None:
        target = Path(export_bundle_path)
        export_acceptance_bundle(bundle, target, fmt=export_bundle_format)
        bundle_path = str(target)

    artifacts = NightlyDryRunArtifacts(
        root=str(root),
        training_candidate_manifest_path=str(training_candidate_path),
        training_research_plan_path=str(training_research_path),
        workflow_snapshot_path=str(workflow_path),
        nightly_report_path=str(nightly_report_path),
        nightly_report_markdown_path=str(nightly_report_markdown_path),
        promotion_review_path=str(review_path),
        live_pointer_path=str(live_ptr),
        training_execution_target="rl",
        training_execution_model_family="rl",
        training_execution_selection_source="training_research",
        acceptance_bundle_path=bundle_path,
    )
    return bundle, artifacts


def _write_workflow_snapshot(
    path: Path,
    *,
    source: str,
    timeframe: str,
    lookback_days: int,
    training_research_response: TrainingResearchPlanResponse,
) -> None:
    snapshot = _build_replay_workflow_snapshot(
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        training_research_response=training_research_response,
    )
    export_workflow_snapshot(snapshot, path)


def _build_replay_workflow_snapshot(
    *,
    source: str,
    timeframe: str,
    lookback_days: int,
    training_research_response: TrainingResearchPlanResponse,
) -> WorkflowSnapshot:
    primary_symbol = (
        training_research_response.rows[0].symbol
        if training_research_response.rows
        else (
            training_research_response.universe_symbols[0]
            if training_research_response.universe_symbols
            else "RELIANCE.NS"
        )
    )
    universe = MarketUniverseResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        candidates=tuple(
            MarketUniverseCandidate(
                symbol=symbol,
                display_name=display,
                source=source,
                liquidity_score=12.4 - (idx * 0.2),
                regime=regime,
                volume_ratio=1.28 - (idx * 0.02),
                activity_score=1.05 - (idx * 0.03),
                adaptive_score_adjustment=0.1 if idx == 0 else None,
                adaptive_row_count=2 if idx == 0 else 0,
                notes=(("adaptive_boost=+0.10 avg_pnl=+1.25% rows=2",) if idx == 0 else ()),
            )
            for idx, (symbol, display, regime) in enumerate(
                (
                    ("RELIANCE.NS", "Reliance Industries", "TRENDING"),
                    ("TCS.NS", "TCS", "RANGING"),
                    ("SBIN.NS", "State Bank of India", "TRENDING"),
                    ("INFY.NS", "Infosys", "RANGING"),
                    ("ICICIBANK.NS", "ICICI Bank", "VOLATILE"),
                    ("HDFCBANK.NS", "HDFC Bank", "TRENDING"),
                    ("AXISBANK.NS", "Axis Bank", "RANGING"),
                    ("LT.NS", "Larsen & Toubro", "TRENDING"),
                    ("BHARTIARTL.NS", "Bharti Airtel", "RANGING"),
                    ("ITC.NS", "ITC", "RANGING"),
                    ("HINDUNILVR.NS", "Hindustan Unilever", "RANGING"),
                    ("MARUTI.NS", "Maruti Suzuki", "TRENDING"),
                    ("KOTAKBANK.NS", "Kotak Mahindra Bank", "RANGING"),
                    ("TATASTEEL.NS", "Tata Steel", "VOLATILE"),
                    ("BAJFINANCE.NS", "Bajaj Finance", "TRENDING"),
                )
            )
        ),
    )
    shortlist = ShortlistAnalysisResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        items=tuple(
            ShortlistAnalysisItem(
                symbol=symbol,
                winning_strategy=strategy,
                liquidity_score=score,
                regime=regime,
                volume_ratio=1.2,
                activity_score=1.0,
                selection_rank=idx,
                priority_score=score / 10.0,
            )
            for idx, (symbol, strategy, regime, score) in enumerate(
                (
                    ("RELIANCE.NS", "ORB", "TRENDING", 12.4),
                    ("TCS.NS", "MMTS", "RANGING", 11.8),
                    ("SBIN.NS", "VWAP", "TRENDING", 11.3),
                    ("INFY.NS", "ORB", "RANGING", 10.9),
                    ("ICICIBANK.NS", "MMTS", "VOLATILE", 10.5),
                    ("HDFCBANK.NS", "ORB", "TRENDING", 10.2),
                ),
                start=1,
            )
        ),
    )
    briefing = ShortlistBriefingResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        headline="4 candidate setups, 6 reviewed",
        items=tuple(
            BriefingItem(
                symbol=symbol,
                action=action,
                confidence=confidence,
                verdict=verdict,
                summary=summary,
                selection_rank=idx,
                priority_score=score,
                regime=regime,
                volume_ratio=1.2,
                activity_score=1.0,
            )
            for idx, (symbol, action, confidence, verdict, summary, regime, score) in enumerate(
                (
                    (
                        "RELIANCE.NS",
                        "BUY",
                        0.78,
                        "candidate",
                        "Dry-run shortlist leader",
                        "TRENDING",
                        1.2,
                    ),
                    (
                        "SBIN.NS",
                        "SELL",
                        0.8,
                        "candidate",
                        "Dry-run hedge candidate",
                        "TRENDING",
                        1.08,
                    ),
                    (
                        "TCS.NS",
                        "BUY",
                        0.74,
                        "candidate",
                        "Dry-run ranging support",
                        "RANGING",
                        0.94,
                    ),
                    (
                        "INFY.NS",
                        "BUY",
                        0.7,
                        "candidate",
                        "Dry-run secondary support",
                        "RANGING",
                        0.8,
                    ),
                    (
                        "ICICIBANK.NS",
                        "BUY",
                        0.69,
                        "watch",
                        "Dry-run volatile watch",
                        "VOLATILE",
                        0.76,
                    ),
                    (
                        "HDFCBANK.NS",
                        "SELL",
                        0.67,
                        "watch",
                        "Dry-run bank hedge watch",
                        "RANGING",
                        0.73,
                    ),
                ),
                start=1,
            )
        ),
    )
    candidates = TrainingCandidateResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        candidates=(
            TrainingCandidate(
                symbol="RELIANCE.NS",
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="ORB",
                decision_action="BUY",
                decision_confidence=0.78,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.12,
                adaptive_score_adjustment=0.1,
                rationale=("Dry-run training candidate",),
            ),
            TrainingCandidate(
                symbol="TCS.NS",
                shortlist_rank=2,
                selection_rank=2,
                winning_strategy="MMTS",
                decision_action="BUY",
                decision_confidence=0.74,
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=0.88,
                rationale=("Dry-run secondary candidate",),
            ),
            TrainingCandidate(
                symbol="SBIN.NS",
                shortlist_rank=3,
                selection_rank=3,
                winning_strategy="VWAP",
                decision_action="SELL",
                decision_confidence=0.8,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.01,
                rationale=("Dry-run hedge candidate",),
            ),
            TrainingCandidate(
                symbol="INFY.NS",
                shortlist_rank=4,
                selection_rank=4,
                winning_strategy="ORB",
                decision_action="BUY",
                decision_confidence=0.7,
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=False,
                market_regime="RANGING",
                priority_score=0.8,
                rationale=("Dry-run watch candidate",),
            ),
            TrainingCandidate(
                symbol="ICICIBANK.NS",
                shortlist_rank=5,
                selection_rank=5,
                winning_strategy="MMTS",
                decision_action="BUY",
                decision_confidence=0.69,
                critique_verdict="watch",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="VOLATILE",
                priority_score=0.76,
                rationale=("Dry-run volatile candidate",),
            ),
        ),
    )
    allocation = PortfolioAllocationResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        headline="3 selected under dry-run constraints",
        max_positions=3,
        items=(
            AllocationDecision(
                symbol="RELIANCE.NS",
                action="BUY",
                verdict="candidate",
                selected=True,
                reason="selected under dry-run constraints",
                selection_rank=1,
                allocation_rank=1,
                allocation_weight=0.42,
                priority_score=1.2,
                allocation_score=1.24,
                regime="TRENDING",
                risk_bucket="low",
                sizing_hint="core",
                research_target="both",
                research_priority_score=1.24,
            ),
            AllocationDecision(
                symbol="SBIN.NS",
                action="SELL",
                verdict="candidate",
                selected=True,
                reason="selected under dry-run constraints",
                selection_rank=2,
                allocation_rank=2,
                allocation_weight=0.33,
                priority_score=1.05,
                allocation_score=1.08,
                regime="TRENDING",
                risk_bucket="medium",
                sizing_hint="core",
                research_target="rl",
                research_priority_score=1.08,
            ),
            AllocationDecision(
                symbol="TCS.NS",
                action="BUY",
                verdict="watch",
                selected=True,
                reason="selected under dry-run constraints",
                selection_rank=3,
                allocation_rank=3,
                allocation_weight=0.25,
                priority_score=0.92,
                allocation_score=0.94,
                regime="RANGING",
                risk_bucket="medium",
                sizing_hint="starter",
                research_target="ml",
                research_priority_score=0.94,
            ),
            AllocationDecision(
                symbol="INFY.NS",
                action="BUY",
                verdict="watch",
                selected=False,
                reason="dry-run skipped to preserve max_positions=3",
                selection_rank=4,
                priority_score=0.8,
                allocation_score=0.8,
                regime="RANGING",
                risk_bucket="high",
                sizing_hint="blocked",
            ),
        ),
        selected_symbols=("RELIANCE.NS", "SBIN.NS", "TCS.NS"),
        skipped_symbols=("INFY.NS",),
        notes=(
            "portfolio_research_alignment=enabled",
            "allocation_research_targets: both=1, ml=1, rl=1",
        ),
    )
    workflow = MultiAgentWorkflowResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        headline=(
            f"Dry-run multi-agent workflow prepared for {primary_symbol} "
            f"with {len(training_research_response.rows)} research row(s)"
        ),
        roles=(
            universe_scout.compose_role(universe),
            liquidity_scout.compose_role(universe),
            activity_scout.compose_role(universe),
            instrument_analyst.compose_role(shortlist),
            briefing_agent.compose_role(briefing),
            portfolio_critic.compose_role(allocation),
            research_planner.compose_role(training_research_response, allocation=allocation),
            research_planner.compose_training_data_role(training_research_response),
            operations_monitor.compose_role(_dry_run_nightly_alignment()),
        ),
        universe=universe,
        shortlist=shortlist,
        briefing=briefing,
        candidates=candidates,
        allocation=allocation,
        research=training_research_response,
        nightly_alignment=_dry_run_nightly_alignment(),
    )
    return summarize_multi_agent_workflow(workflow)


def _write_training_candidate_manifest(
    path: Path,
    *,
    source: str,
    symbol: str,
    timeframe: str,
    lookback_days: int,
) -> None:
    response = TrainingCandidateResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        candidates=(
            TrainingCandidate(
                symbol=symbol,
                shortlist_rank=1,
                selection_rank=1,
                winning_strategy="ORB",
                decision_action="BUY",
                decision_confidence=0.78,
                critique_verdict="candidate",
                ml_candidate=True,
                rl_candidate=True,
                market_regime="TRENDING",
                priority_score=1.12,
                adaptive_score_adjustment=0.1,
                rationale=(
                    "Dry-run training candidate",
                    "Recent paper outcomes imply a boost (+0.10 score adj)",
                ),
            ),
        ),
    )
    export_training_candidates(response, path)


def _write_promoted_rl_artifact(
    *,
    validated_dir: Path,
    models_root: Path,
    workflow_snapshot_path: Path,
    symbol: str,
    timeframe: str,
    run_id: str,
) -> Path:
    validated_dir.mkdir(parents=True, exist_ok=True)
    (validated_dir / "policy.zip").write_bytes(b"\x00")
    (validated_dir / "normalizer.json").write_text("{}", encoding="utf-8")
    PolicyCheckpoint(
        run_id=run_id,
        symbol=symbol,
        timeframe=timeframe,
        obs_shape=[8, 10],
        policy_type="MlpPolicy",
        total_timesteps=1,
        n_folds=1,
        oos_metrics=OOSMetricsSummary(total_trades=12, sharpe_ratio=1.3, profit_factor=1.25),
        verdict_passed=True,
        verdict_score=0.86,
        verdict_reasons=["dry_run_ready"],
        advisory_ready=True,
    ).write(validated_dir / "metadata.json")
    cp = PolicyCheckpoint.read(validated_dir / "metadata.json")
    record = PromotionRecord.from_rl_checkpoint(cp, validated_dir, status=PromotionStatus.VALIDATED)
    record.workflow_snapshot_path = str(workflow_snapshot_path)
    return promote(record, models_root=models_root, promoted_by="dry_run")


def _write_training_research_plan(
    path: Path,
    *,
    response: TrainingResearchPlanResponse,
) -> None:
    export_training_research_plan(response, path)


def _dry_run_nightly_alignment() -> NightlyAlignmentStatus:
    return NightlyAlignmentStatus(
        report_count=1,
        aligned_reports=1,
        enabled_reports=1,
        latest_execution_target="rl",
        latest_execution_model_family="rl",
        latest_execution_selection_source="training_research",
        latest_target_mix="rl=1",
        latest_promotion_review_model_kind="rl_policy",
        recommended_refresh_target="rl",
        effective_refresh_target="rl",
        latest_training_research_scout_support_target="rl",
        latest_training_research_discovery_recommended_target="rl",
    )


def _build_training_research_response(
    *,
    source: str,
    symbol: str,
    timeframe: str,
    lookback_days: int,
) -> TrainingResearchPlanResponse:
    return TrainingResearchPlanResponse(
        ok=True,
        source=source,
        timeframe=timeframe,
        lookback_days=lookback_days,
        selection_policy="ranked",
        refresh_requested=False,
        refresh_target="all",
        refresh_timeframe=timeframe,
        refresh_lookback_days=lookback_days,
        discovery_preferred_symbols=(symbol,),
        discovery_liquidity_symbols=(symbol, "SBIN.NS", "TCS.NS"),
        discovery_activity_symbols=(symbol, "TCS.NS", "SBIN.NS"),
        discovery_regime_mix="trending=2, ranging=1",
        discovery_summary=(
            f"preferred=1 | liquidity={symbol},SBIN.NS | "
            f"activity={symbol},TCS.NS | regimes=trending=2, ranging=1 | "
            "setup_posture=rl | posture=rl"
        ),
        discovery_recommended_refresh_target="rl",
        effective_refresh_target="rl",
        scout_support_target="rl",
        discovery_follow_up_target="rl",
        discovery_follow_up_summary=(
            "Discovery scouts currently lean RL-focused from rl universe posture."
        ),
        discovery_follow_up_action=(
            "Refresh market universe and shortlist review with RL-focused discovery focus "
            "before the next training cycle."
        ),
        research_follow_up_target="rl",
        research_follow_up_summary=(
            "medium RL remediation pressure on 1 row(s); max=+0.04 total=+0.04"
        ),
        research_follow_up_action=(
            "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
        ),
        execution_follow_up_target="rl",
        execution_follow_up_summary=(
            "Nightly execution drift currently leans RL-focused from rl execution posture."
        ),
        execution_follow_up_action=(
            "Review nightly execution path and retarget execution candidate selection "
            "toward RL-focused before the next promotion or nightly cycle."
        ),
        follow_up_action=(
            "Prioritize RL refresh follow-up from remediation pressure across 1 selected row(s)."
        ),
        universe_symbols=(symbol, "SBIN.NS", "TCS.NS"),
        ml_symbols=(symbol, "TCS.NS"),
        rl_symbols=(symbol, "SBIN.NS"),
        rows=(
            TrainingResearchRow(
                symbol=symbol,
                target="both",
                universe_rank=1,
                shortlist_rank=1,
                selection_rank=1,
                liquidity_score=1.12,
                decision_action="BUY",
                critique_verdict="candidate",
                winning_strategy="ORB",
                market_regime="TRENDING",
                volume_ratio=1.28,
                priority_score=1.12,
                adaptive_score_adjustment=0.1,
                refreshed=False,
                rationale=("Dry-run research plan row",),
            ),
            TrainingResearchRow(
                symbol="SBIN.NS",
                target="rl",
                universe_rank=2,
                shortlist_rank=2,
                selection_rank=2,
                liquidity_score=1.08,
                decision_action="SELL",
                critique_verdict="candidate",
                winning_strategy="VWAP",
                market_regime="TRENDING",
                volume_ratio=1.15,
                priority_score=1.08,
                refreshed=False,
                rationale=("Dry-run RL research row",),
            ),
            TrainingResearchRow(
                symbol="TCS.NS",
                target="ml",
                universe_rank=3,
                shortlist_rank=3,
                selection_rank=3,
                liquidity_score=1.02,
                decision_action="BUY",
                critique_verdict="watch",
                winning_strategy="MMTS",
                market_regime="RANGING",
                volume_ratio=1.05,
                priority_score=0.94,
                refreshed=False,
                rationale=("Dry-run ML research row",),
            ),
        ),
    )


def _write_realistic_nightly_report(
    report_dir: Path,
    *,
    training_candidate_path: Path,
    training_research_path: Path,
    workflow_snapshot_path: Path,
    promotion_review_path: Path,
    symbol: str,
    source_mode: str,
    training_research_response: TrainingResearchPlanResponse,
    execution_target: str,
    execution_model_family: str,
    execution_selection_source: str,
    promotion_review_model_kind: str,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=[symbol],
        overall_status="ok",
    )
    for name, detail in (
        ("basket_resolution", {"mode": source_mode}),
        (
            "training_candidates",
            {
                "path": _repo_relative(report_dir / "placeholder.json", training_candidate_path),
                "count": 1,
                "ml_count": 1,
                "rl_count": 1,
                "selection_policy": "ranked",
            },
        ),
        (
            "training_research",
            {
                "path": _repo_relative(report_dir / "placeholder.json", training_research_path),
                "count": len(training_research_response.rows),
                "ml_count": len(training_research_response.ml_symbols),
                "rl_count": len(training_research_response.rl_symbols),
                "selection_policy": training_research_response.selection_policy,
                "refresh_target": training_research_response.refresh_target,
                "discovery_preferred_count": len(
                    training_research_response.discovery_preferred_symbols
                ),
                "discovery_regime_mix": training_research_response.discovery_regime_mix,
                "discovery_summary": training_research_response.discovery_summary,
                "discovery_recommended_refresh_target": (
                    training_research_response.discovery_recommended_refresh_target
                ),
                "discovery_follow_up_target": training_research_response.discovery_follow_up_target,
                "discovery_follow_up_summary": (
                    training_research_response.discovery_follow_up_summary
                ),
                "discovery_follow_up_action": (
                    training_research_response.discovery_follow_up_action
                ),
                "research_follow_up_target": training_research_response.research_follow_up_target,
                "research_follow_up_summary": (
                    training_research_response.research_follow_up_summary
                ),
                "research_follow_up_action": (
                    training_research_response.research_follow_up_action
                ),
                "execution_follow_up_target": (
                    training_research_response.execution_follow_up_target
                ),
                "execution_follow_up_summary": (
                    training_research_response.execution_follow_up_summary
                ),
                "execution_follow_up_action": (
                    training_research_response.execution_follow_up_action
                ),
                "effective_refresh_target": training_research_response.effective_refresh_target,
                "scout_support_target": training_research_response.scout_support_target,
            },
        ),
        (
            "training_execution_target",
            {
                "target": execution_target,
                "run_ml": execution_model_family in {"ml", "hybrid"},
                "run_rl": execution_model_family in {"rl", "hybrid"},
                "selection_source": execution_selection_source,
                "requested_target": training_research_response.refresh_target,
                "discovery_recommended_refresh_target": (
                    training_research_response.discovery_recommended_refresh_target
                ),
                "discovery_summary": training_research_response.discovery_summary,
            },
        ),
        (
            "workflow_snapshot",
            {
                "path": _repo_relative(report_dir / "placeholder.json", workflow_snapshot_path),
                "universe_count": 15,
                "shortlist_count": 6,
                "briefing_candidates": 4,
                "allocation_research_alignment_enabled": True,
                "allocation_research_target_mix": "both=1, ml=1, rl=1",
                "ml_count": 5,
                "rl_count": 3,
            },
        ),
        (
            "promote_best",
            {
                "count": 1,
                "model_kind": promotion_review_model_kind,
                "workflow_snapshot_path": _repo_relative(
                    report_dir / "placeholder.json",
                    workflow_snapshot_path,
                ),
            },
        ),
        (
            "promotion_reviews",
            {
                "count": 1,
                "format": "md",
                "paths": [_repo_relative(report_dir / "placeholder.json", promotion_review_path)],
                "model_kind": promotion_review_model_kind,
            },
        ),
    ):
        report.add(
            nightly_train.StepResult(
                name=name,
                status="ok",
                started_at="2026-06-03T01:00:01",
                ended_at="2026-06-03T01:00:01",
                duration_s=0.0,
                detail=detail,
            )
        )
    md_path = nightly_train.write_report(report, report_dir)
    return md_path.with_suffix(".json"), md_path


def _repo_relative(report_path: Path, artifact_path: Path) -> str:
    base = report_path.parent
    if len(report_path.parents) >= 3:
        base = report_path.parents[2]
    try:
        return str(artifact_path.relative_to(base))
    except ValueError:
        return str(artifact_path)


def _build_stub_engine(settings: Any, symbol: str) -> Any:
    return SimpleNamespace(
        settings=settings,
        _agentic_store=None,
        _agentic_learning_store=None,
        rl_generator=None,
        state=SimpleNamespace(symbol=symbol),
        _agentic_orchestrator=None,
    )


def _engine_for_acceptance(engine: Any | None, settings: Any, symbol: str) -> Any:
    if engine is None:
        return _build_stub_engine(settings, symbol)
    return SimpleNamespace(
        settings=settings,
        _agentic_store=getattr(engine, "_agentic_store", None),
        _agentic_learning_store=getattr(engine, "_agentic_learning_store", None),
        rl_generator=getattr(engine, "rl_generator", None),
        state=SimpleNamespace(symbol=symbol),
        _agentic_orchestrator=getattr(engine, "_agentic_orchestrator", None),
    )


def _load_nightly_train_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "nightly_train.py"
    spec = importlib.util.spec_from_file_location("fortuna_nightly_train_acceptance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load nightly_train script from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _find_nightly_report_json(
    report_dir: Path,
    *,
    explicit_path: Optional[Path] = None,
) -> Optional[Path]:
    if explicit_path is not None:
        path = Path(explicit_path)
        return path if path.is_file() else None
    skip_names = {
        "training_candidates.json",
        "training_research_plan.json",
        "workflow_snapshot.json",
    }
    for path in sorted(report_dir.glob("*.json"), key=lambda row: row.name, reverse=True):
        if path.name in skip_names:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload.get("steps"), list):
            return path
    return None


def _step_detail_from_report(payload: dict[str, Any], step_name: str) -> Optional[dict[str, Any]]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return None
    for row in steps:
        if not isinstance(row, dict):
            continue
        if str(row.get("name", "")).strip() == step_name:
            detail = row.get("detail")
            return detail if isinstance(detail, dict) else None
    return None


def _resolve_nightly_artifact_path(
    report_path: Path,
    raw_path: str,
    *,
    project_root: Optional[Path] = None,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return Path()
    candidate = Path(text)
    if candidate.is_file():
        return candidate
    bases: list[Path] = [report_path.parent]
    if len(report_path.parents) >= 3:
        bases.append(report_path.parents[2])
    if project_root is not None:
        bases.append(project_root)
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return (report_path.parent / candidate).resolve()


def _path_text(path: Path | str | None) -> Optional[str]:
    if path is None:
        return None
    text = str(path).strip()
    return text or None


def _artifact_presence_map(
    *,
    nightly_report_path: Path | str | None = None,
    training_candidates_path: Path | str | None = None,
    training_research_path: Path | str | None = None,
    workflow_snapshot_path: Path | str | None = None,
    promotion_review_paths: tuple[str, ...] = (),
) -> dict[str, bool]:
    return {
        "nightly_report": bool(_path_text(nightly_report_path)),
        "training_candidates": bool(_path_text(training_candidates_path)),
        "training_research": bool(_path_text(training_research_path)),
        "workflow_snapshot": bool(_path_text(workflow_snapshot_path)),
        "promotion_review": any(str(path or "").strip() for path in promotion_review_paths),
    }


def _replay_run_identifier(payload: dict[str, Any], report_path: Path) -> str:
    for step_name in ("promote_best", "promotion_reviews", "training_execution_target"):
        detail = _step_detail_from_report(payload, step_name)
        if not isinstance(detail, dict):
            continue
        run_id = str(detail.get("run_id", "") or "").strip()
        if run_id:
            return run_id
    return report_path.stem


def _build_linkage_summary(
    *,
    report_path: Path | str | None = None,
    run_identifier: Optional[str] = None,
    training_candidates_path: Path | str | None = None,
    training_research_path: Path | str | None = None,
    workflow_snapshot_path: Path | str | None = None,
    promotion_review_paths: tuple[str, ...] = (),
    missing_artifacts: tuple[str, ...] = (),
    broken_links: tuple[str, ...] = (),
    linkage_warnings: tuple[str, ...] = (),
    summary: str = "",
) -> NightlyArtifactLinkageSummary:
    unique_missing = tuple(sorted({item for item in missing_artifacts if str(item).strip()}))
    unique_broken = tuple(dict.fromkeys(item for item in broken_links if str(item).strip()))
    unique_warnings = tuple(
        dict.fromkeys(item for item in linkage_warnings if str(item).strip())
    )
    required_missing = [item for item in unique_missing if item in _REPLAY_REQUIRED_ARTIFACTS]
    optional_missing = [item for item in unique_missing if item in _REPLAY_OPTIONAL_ARTIFACTS]

    if unique_broken or required_missing:
        status = "broken"
    elif optional_missing or unique_warnings:
        status = "partial"
    else:
        status = "linked"

    if not summary:
        parts: list[str] = []
        if required_missing:
            parts.append("required_missing=" + ",".join(required_missing))
        if optional_missing:
            parts.append("optional_missing=" + ",".join(optional_missing))
        if unique_broken:
            parts.append("broken=" + ",".join(unique_broken))
        if unique_warnings:
            parts.append("warnings=" + " | ".join(unique_warnings))
        if not parts:
            parts.append("nightly artifact chain linked")
        summary = "; ".join(parts)

    return NightlyArtifactLinkageSummary(
        overall_status=status,
        replay_status=status,
        run_identifier=run_identifier,
        nightly_report_path=_path_text(report_path),
        training_candidates_path=_path_text(training_candidates_path),
        training_research_path=_path_text(training_research_path),
        workflow_snapshot_path=_path_text(workflow_snapshot_path),
        promotion_review_paths=promotion_review_paths,
        artifact_presence=_artifact_presence_map(
            nightly_report_path=report_path,
            training_candidates_path=training_candidates_path,
            training_research_path=training_research_path,
            workflow_snapshot_path=workflow_snapshot_path,
            promotion_review_paths=promotion_review_paths,
        ),
        missing_artifacts=unique_missing,
        broken_links=unique_broken,
        linkage_warnings=unique_warnings,
        summary=summary,
        warning=status != "linked",
    )


def verify_nightly_artifact_linkage(
    *,
    report_dir: Path | str,
    project_root: Optional[Path | str] = None,
    nightly_report_path: Optional[Path | str] = None,
    models_root: Optional[Path | str] = None,
) -> NightlyArtifactLinkageSummary:
    del models_root  # reserved for future promotion pointer checks
    root = Path(report_dir)
    explicit = Path(nightly_report_path) if nightly_report_path is not None else None
    report_path = _find_nightly_report_json(root, explicit_path=explicit)
    if report_path is None:
        return _build_linkage_summary(
            missing_artifacts=("nightly_report",),
            summary="No nightly JSON report found in replay directory.",
        )

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _build_linkage_summary(
            report_path=report_path,
            missing_artifacts=("nightly_report",),
            summary="Nightly report JSON could not be loaded.",
        )

    project = Path(project_root) if project_root is not None else None
    missing: list[str] = []
    broken: list[str] = []
    warnings: list[str] = []
    resolved_paths: dict[str, Optional[str]] = {
        "nightly_report_path": str(report_path),
        "training_candidates_path": None,
        "training_research_path": None,
        "workflow_snapshot_path": None,
    }
    review_paths: list[str] = []

    candidate_detail = _step_detail_from_report(payload, "training_candidates")
    if candidate_detail is None:
        missing.append("training_candidates")
    else:
        candidate_path = _resolve_nightly_artifact_path(
            report_path,
            str(candidate_detail.get("path", "") or ""),
            project_root=project,
        )
        if candidate_path.is_file():
            resolved_paths["training_candidates_path"] = str(candidate_path)
        else:
            missing.append("training_candidates")

    research_detail = _step_detail_from_report(payload, "training_research")
    if research_detail is None:
        missing.append("training_research")
    else:
        research_path = _resolve_nightly_artifact_path(
            report_path,
            str(research_detail.get("path", "") or ""),
            project_root=project,
        )
        if research_path.is_file():
            resolved_paths["training_research_path"] = str(research_path)
        else:
            missing.append("training_research")

    workflow_detail = _step_detail_from_report(payload, "workflow_snapshot")
    workflow_resolved: Optional[Path] = None
    if workflow_detail is None:
        missing.append("workflow_snapshot")
    else:
        workflow_resolved = _resolve_nightly_artifact_path(
            report_path,
            str(workflow_detail.get("path", "") or ""),
            project_root=project,
        )
        if workflow_resolved.is_file():
            resolved_paths["workflow_snapshot_path"] = str(workflow_resolved)
            try:
                workflow_payload = json.loads(workflow_resolved.read_text(encoding="utf-8"))
                for section in ("universe", "allocation", "research"):
                    if section not in workflow_payload:
                        broken.append(f"workflow_snapshot missing section={section}")
            except (OSError, json.JSONDecodeError):
                broken.append("workflow_snapshot unreadable")
        else:
            missing.append("workflow_snapshot")

    promote_detail = _step_detail_from_report(payload, "promote_best")
    review_detail = _step_detail_from_report(payload, "promotion_reviews")
    if review_detail is not None:
        raw_paths = review_detail.get("paths")
        if isinstance(raw_paths, list):
            for raw in raw_paths:
                review_path = _resolve_nightly_artifact_path(
                    report_path,
                    str(raw or ""),
                    project_root=project,
                )
                if review_path.is_file():
                    review_paths.append(str(review_path))
                else:
                    missing.append("promotion_review")
        elif int(review_detail.get("count", 0) or 0) > 0:
            missing.append("promotion_review")

    if promote_detail is not None:
        promoted_count = int(promote_detail.get("count", 0) or 0)
        review_count = int(review_detail.get("count", 0) or 0) if review_detail is not None else 0
        if promoted_count > 0 and review_count == 0 and not review_paths:
            missing.append("promotion_review")
            warnings.append("promotion review was not emitted for a promoted replay artifact")

    if (
        promote_detail is not None
        and workflow_resolved is not None
        and workflow_resolved.is_file()
    ):
        promote_workflow = str(promote_detail.get("workflow_snapshot_path", "") or "").strip()
        workflow_step_path = ""
        if workflow_detail is not None:
            workflow_step_path = str(workflow_detail.get("path", "") or "").strip()
        if promote_workflow and workflow_step_path:
            promote_resolved = _resolve_nightly_artifact_path(
                report_path,
                promote_workflow,
                project_root=project,
            )
            if promote_resolved.resolve() != workflow_resolved.resolve():
                broken.append("promote_best workflow_snapshot_path mismatch")

    return _build_linkage_summary(
        report_path=resolved_paths["nightly_report_path"],
        run_identifier=_replay_run_identifier(payload, report_path),
        training_candidates_path=resolved_paths["training_candidates_path"],
        training_research_path=resolved_paths["training_research_path"],
        workflow_snapshot_path=resolved_paths["workflow_snapshot_path"],
        promotion_review_paths=tuple(review_paths),
        missing_artifacts=tuple(missing),
        broken_links=tuple(broken),
        linkage_warnings=tuple(warnings),
    )


def verify_acceptance_replay_chain(
    *,
    settings,
    report_dir: Path | str,
    engine: Any | None = None,
    models_root: Path | str = "models",
    ml_base: Optional[Path | str] = None,
    symbol: Optional[str] = None,
    nightly_report_path: Optional[Path | str] = None,
    include_model_health: bool = True,
) -> tuple[NightlyArtifactLinkageSummary, AcceptanceBundle]:
    root = Path(report_dir)
    project_root = getattr(settings, "project_root", None)
    linkage = verify_nightly_artifact_linkage(
        report_dir=root,
        project_root=Path(project_root) if project_root is not None else None,
        nightly_report_path=nightly_report_path,
        models_root=models_root,
    )
    report_path = linkage.nightly_report_path or nightly_report_path
    workflow_path = linkage.workflow_snapshot_path
    bundle = gather_acceptance_bundle(
        settings=settings,
        engine=engine,
        models_root=models_root,
        ml_base=ml_base,
        symbol=symbol,
        nightly_report_path=report_path,
        workflow_snapshot_path=workflow_path,
        include_model_health=include_model_health,
        linkage_summary=linkage,
    )
    return linkage, bundle


def _replay_emit_args(
    *,
    symbol: str,
    timeframe: str,
    days: int,
    source: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_training_candidates=True,
        candidate_manifest="",
        candidate_target="rl",
        candidate_top_n=0,
        candidate_selection_policy="ranked",
        candidate_universe_limit=15,
        candidate_analysis_limit=8,
        candidate_source=source,
        training_candidate_out="",
        training_research_out="",
        workflow_snapshot_out="",
        timeframe=timeframe,
        days=days,
    )


@contextmanager
def _replay_emit_patches(
    *,
    settings,
    symbol: str,
    source: str,
    timeframe: str,
    lookback_days: int,
) -> Iterator[None]:
    research_response = _build_training_research_response(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=lookback_days,
    )

    def _stub_build_training_candidates(**_kwargs):
        return TrainingCandidateResponse(
            ok=True,
            source=source,
            timeframe=timeframe,
            lookback_days=lookback_days,
            candidates=(
                TrainingCandidate(
                    symbol=symbol,
                    shortlist_rank=1,
                    selection_rank=1,
                    winning_strategy="ORB",
                    decision_action="BUY",
                    decision_confidence=0.78,
                    critique_verdict="candidate",
                    ml_candidate=True,
                    rl_candidate=True,
                    market_regime="TRENDING",
                    priority_score=1.12,
                    adaptive_score_adjustment=0.1,
                    rationale=("Emit-replay training candidate",),
                ),
            ),
        )

    def _stub_build_training_research_plan(**_kwargs):
        return research_response

    def _stub_build_workflow_snapshot(**_kwargs):
        return _build_replay_workflow_snapshot(
            source=source,
            timeframe=timeframe,
            lookback_days=lookback_days,
            training_research_response=research_response,
        )

    with (
        patch.object(nightly_basket, "get_settings", return_value=settings),
        patch.object(
            nightly_basket,
            "build_training_candidates",
            side_effect=_stub_build_training_candidates,
        ),
        patch.object(
            nightly_basket,
            "build_training_research_plan",
            side_effect=_stub_build_training_research_plan,
        ),
        patch.object(
            nightly_basket,
            "build_workflow_snapshot",
            side_effect=_stub_build_workflow_snapshot,
        ),
    ):
        yield


def _build_emit_replay_nightly_report(
    *,
    nightly_dir: Path,
    symbol: str,
    source_mode: str,
    basket_meta: dict[str, object],
    candidate_meta: dict[str, object],
    research_meta: dict[str, object],
    research_response: TrainingResearchPlanResponse,
    execution_plan: dict[str, object],
    workflow_meta: dict[str, object],
    promote_detail: dict[str, object],
    review_detail: dict[str, object],
) -> tuple[Path, Path]:
    nightly_train = _load_nightly_train_module()
    report = nightly_train.NightlyReport(
        started_at="2026-06-03T01:00:00",
        ended_at="2026-06-03T01:10:00",
        total_duration_s=600.0,
        basket=[symbol],
        overall_status="ok",
    )
    research_step_detail = {
        **research_meta,
        "discovery_follow_up_target": research_response.discovery_follow_up_target,
        "discovery_follow_up_summary": research_response.discovery_follow_up_summary,
        "discovery_follow_up_action": research_response.discovery_follow_up_action,
        "research_follow_up_target": research_response.research_follow_up_target,
        "research_follow_up_summary": research_response.research_follow_up_summary,
        "research_follow_up_action": research_response.research_follow_up_action,
        "execution_follow_up_target": research_response.execution_follow_up_target,
        "execution_follow_up_summary": research_response.execution_follow_up_summary,
        "execution_follow_up_action": research_response.execution_follow_up_action,
        "effective_refresh_target": research_response.effective_refresh_target,
        "scout_support_target": research_response.scout_support_target,
    }
    for name, detail in (
        ("basket_resolution", {"mode": source_mode, **basket_meta}),
        ("training_candidates", candidate_meta),
        ("training_research", research_step_detail),
        ("training_execution_target", execution_plan),
        ("workflow_snapshot", workflow_meta),
        ("promote_best", promote_detail),
        ("promotion_reviews", review_detail),
    ):
        report.add(
            nightly_train.StepResult(
                name=name,
                status="ok",
                started_at="2026-06-03T01:00:01",
                ended_at="2026-06-03T01:00:01",
                duration_s=0.0,
                detail=detail,
            )
        )
    md_path = nightly_train.write_report(report, nightly_dir)
    return md_path.with_suffix(".json"), md_path


def build_nightly_emit_replay(
    *,
    settings,
    out_dir: Path | str,
    engine: Any | None = None,
    symbol: str = "RELIANCE.NS",
    timeframe: str = "5m",
    days: int = 20,
    source: str = "screener",
    run_id: str = "dry_run_emit_replay",
    export_bundle_path: Optional[Path | str] = None,
    export_bundle_format: str = "md",
) -> tuple[AcceptanceBundle, NightlyDryRunArtifacts]:
    root = Path(out_dir)
    acceptance_settings = settings.model_copy(update={"project_root": root})
    nightly_dir = root / "reports" / "nightly"
    models_root = root / "models"
    workflow_path = nightly_dir / "workflow_snapshot.json"
    research_response = _build_training_research_response(
        source=source,
        symbol=symbol,
        timeframe=timeframe,
        lookback_days=days,
    )
    args = _replay_emit_args(symbol=symbol, timeframe=timeframe, days=days, source=source)

    with _replay_emit_patches(
        settings=acceptance_settings,
        symbol=symbol,
        source=source,
        timeframe=timeframe,
        lookback_days=days,
    ):
        candidate_meta = nightly_basket.emit_training_candidate_manifest(
            args,
            report_dir=nightly_dir,
        )
        research_meta = nightly_basket.emit_training_research_plan(
            args,
            report_dir=nightly_dir,
        )
        workflow_meta = nightly_basket.emit_workflow_snapshot(
            args,
            report_dir=nightly_dir,
        )

    def _meta_path(meta: Any) -> Optional[Path]:
        if not isinstance(meta, dict):
            return None
        text = str(meta.get("path", "") or "").strip()
        return Path(text) if text else None

    training_candidate_path = _meta_path(candidate_meta)
    training_research_path = _meta_path(research_meta)
    workflow_path = _meta_path(workflow_meta)

    missing_required: list[str] = []
    if training_candidate_path is None:
        missing_required.append("training_candidates")
    if training_research_path is None:
        missing_required.append("training_research")
    if workflow_path is None:
        missing_required.append("workflow_snapshot")

    if missing_required:
        linkage = _build_linkage_summary(
            run_identifier=run_id,
            training_candidates_path=training_candidate_path,
            training_research_path=training_research_path,
            workflow_snapshot_path=workflow_path,
            missing_artifacts=tuple(missing_required),
            linkage_warnings=(
                "emit-replay stopped before nightly report construction "
                "because a required artifact was not emitted",
            ),
        )
        bundle = gather_acceptance_bundle(
            settings=acceptance_settings,
            engine=_engine_for_acceptance(engine, acceptance_settings, symbol),
            models_root=models_root,
            ml_base=models_root / "ml_signal_scorer",
            symbol=symbol,
            workflow_snapshot_path=str(workflow_path) if workflow_path is not None else None,
            include_model_health=True,
            linkage_summary=linkage,
        )

        bundle_path = None
        if export_bundle_path is not None:
            target = Path(export_bundle_path)
            export_acceptance_bundle(bundle, target, fmt=export_bundle_format)
            bundle_path = str(target)

        artifacts = NightlyDryRunArtifacts(
            root=str(root),
            training_candidate_manifest_path=str(training_candidate_path or ""),
            training_research_plan_path=str(training_research_path or ""),
            workflow_snapshot_path=str(workflow_path or ""),
            nightly_report_path="",
            nightly_report_markdown_path=None,
            promotion_review_path="",
            live_pointer_path="",
            acceptance_bundle_path=bundle_path,
            replay_mode="emit",
            linkage=linkage,
        )
        return bundle, artifacts

    validated_dir = models_root / "validated" / run_id
    live_ptr = _write_promoted_rl_artifact(
        validated_dir=validated_dir,
        models_root=models_root,
        workflow_snapshot_path=workflow_path,
        symbol=symbol,
        timeframe=timeframe,
        run_id=run_id,
    )
    review_path: Optional[Path] = None
    try:
        review = build_promotion_review(
            kind=ModelKind.RL_POLICY,
            models_root=models_root,
            symbol=symbol,
            run_id=run_id,
            settings=settings,
        )
        if review is not None:
            review_path = nightly_dir / "promotions" / f"{symbol.replace('.', '_')}__{run_id}.md"
            export_promotion_review(review, review_path, fmt="md")
    except OSError:
        review_path = None

    nightly_train = _load_nightly_train_module()
    basket_meta = {"mode": "derived_training_candidates", "target": "rl"}
    execution_plan = nightly_train._build_training_execution_plan(
        [symbol],
        basket_meta,
        research_meta,
    )
    execution_target = str(execution_plan.get("target", "rl") or "rl")
    promote_detail = {
        "count": 1,
        "model_kind": ModelKind.RL_POLICY.value,
        "run_id": run_id,
        "workflow_snapshot_path": _repo_relative(
            nightly_dir / "placeholder.json",
            workflow_path,
        ),
    }
    review_detail = {
        "count": 1 if review_path is not None else 0,
        "format": "md",
        "paths": (
            [_repo_relative(nightly_dir / "placeholder.json", review_path)]
            if review_path is not None
            else []
        ),
        "model_kind": ModelKind.RL_POLICY.value,
        "run_id": run_id,
    }
    nightly_report_path, nightly_report_markdown_path = _build_emit_replay_nightly_report(
        nightly_dir=nightly_dir,
        symbol=symbol,
        source_mode="derived_training_candidates",
        basket_meta=basket_meta,
        candidate_meta=candidate_meta,
        research_meta=research_meta,
        research_response=research_response,
        execution_plan=execution_plan,
        workflow_meta=workflow_meta,
        promote_detail=promote_detail,
        review_detail=review_detail,
    )

    linkage = verify_nightly_artifact_linkage(
        report_dir=nightly_dir,
        project_root=root,
        nightly_report_path=nightly_report_path,
        models_root=models_root,
    )
    bundle = gather_acceptance_bundle(
        settings=acceptance_settings,
        engine=_engine_for_acceptance(engine, acceptance_settings, symbol),
        models_root=models_root,
        ml_base=models_root / "ml_signal_scorer",
        symbol=symbol,
        nightly_report_path=nightly_report_path,
        workflow_snapshot_path=workflow_path,
        include_model_health=True,
        linkage_summary=linkage,
    )

    bundle_path = None
    if export_bundle_path is not None:
        target = Path(export_bundle_path)
        export_acceptance_bundle(bundle, target, fmt=export_bundle_format)
        bundle_path = str(target)

    artifacts = NightlyDryRunArtifacts(
        root=str(root),
        training_candidate_manifest_path=str(training_candidate_path or ""),
        training_research_plan_path=str(training_research_path or ""),
        workflow_snapshot_path=str(workflow_path or ""),
        nightly_report_path=str(nightly_report_path),
        nightly_report_markdown_path=str(nightly_report_markdown_path),
        promotion_review_path=str(review_path or ""),
        live_pointer_path=str(live_ptr),
        training_execution_target=execution_target,
        training_execution_model_family=execution_target,
        training_execution_selection_source=str(
            execution_plan.get("selection_source") or "training_research"
        ),
        acceptance_bundle_path=bundle_path,
        replay_mode="emit",
        linkage=linkage,
    )
    return bundle, artifacts


__all__ = [
    "NightlyDryRunArtifacts",
    "build_candidate_nightly_dry_run",
    "build_nightly_emit_replay",
    "verify_acceptance_replay_chain",
    "verify_nightly_artifact_linkage",
]
