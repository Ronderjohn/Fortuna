# CLI scripts

**Primary workflow:** [Streamlit dashboard](../README.md#quick-start--the-dashboard)

Fortuna's scripts are grouped around a few practical workflows. Most operators
will use the dashboard, preflight checks, training, and promotion commands.

## Dashboard / runtime

| Script | Purpose |
|---|---|
| `run_dashboard.py` | Start the Streamlit dashboard |
| `run_operator_preflight.py` | Read-only operator readiness check before market open |
| `run_telegram_bot.py` | Start the Telegram polling assistant for search and analysis |
| `build_market_universe.py` | Rank a liquid research/watchlist universe from SmartAPI instrument seeds, Screener CSV, or registry fallback, with optional light adaptive weighting from recent paper outcomes |
| `analyze_market_shortlist.py` | Analyze the top-ranked universe candidates with advisory + critique |
| `prepare_training_candidates.py` | Export shortlist-derived ML/RL candidate manifest and optional backfill; manifests now retain market-context fields such as regime and volume-ratio |
| `build_training_research_plan.py` | Export a typed ML/RL training research plan with separate ML/RL symbol picks and optional targeted data refresh |
| `brief_market_shortlist.py` | Summarize the top shortlisted setups with portfolio/exposure notes |
| `allocate_market_shortlist.py` | Turn shortlisted setups into a constrained basket under simple exposure limits |
| `build_workflow_snapshot.py` | Export a full universe -> shortlist -> briefing -> allocation -> training workflow snapshot |
| `build_acceptance_bundle.py` | Export one higher-level acceptance bundle tying nightly report, workflow snapshot, promotion review, and model-health evidence together; surfaces `refresh_context` (effective target, source, split follow-up lanes, scout support) and `cross_artifact_alignment` |
| `run_nightly_acceptance_dry_run.py` | Local nightly acceptance rehearsal: `--mode candidate` (default synthetic fixtures), `--mode emit-replay` (routes through `nightly_basket.emit_*` + `write_report` and returns typed replay diagnostics), or `--mode verify` (linkage check on an existing nightly report dir with the same typed replay diagnostics) — advisory-only, no network |
| `diagnose_live_feed.py` | Inspect live-feed wiring and runtime state |

## Training

| Script | Purpose |
|---|---|
| `run_rl_train.py` | Train one RL policy checkpoint |
| `nightly_train.py` | Run the nightly RL/ML training and promotion pipeline |
| `train_ml_signal_scorer.py` | Train ML scorer artifacts from agentic learning rows (`--group ml`) |
| `train_regime_detector.py` | Train the regime detector |

## Promotion

| Script | Purpose |
|---|---|
| `promote_model.py` | Promote RL or ML artifacts through the unified registry, optionally linking a workflow snapshot artifact |
| `promote_policy.py` | Promote an RL run, optionally overriding the symbol pointer and linking a workflow snapshot artifact |
| `review_promotion.py` | Review the promoted artifact, key metrics, and linked workflow context in one place |

## Preflight / smoke / diagnostics

| Script | Purpose |
|---|---|
| `test_smartapi_env.py` | Credential and SmartAPI import smoke test |
| `verify_env.ps1` | Quick local Python/package/test sanity check |
| `check_gpu.py` | Report GPU/CuPy availability |
| `smoke_account_sync.py` | Exercise paper-account sync path |

## Backtest / research

| Script | Purpose |
|---|---|
| `run_backtest.py` | Single-strategy backtest |
| `run_strategy_tester_report.py` | Export strategy tester bundle under `logs/strategy_tester` |
| `run_paper_competition.py` | Multi-strategy paper competition |
| `run_institutional_backtest.py` | Walk-forward institutional backtest suite |
| `run_arena.py` / `run_tournament.py` | Strategy search and tournament workflows |
| `smartapi_backfill.py` | Refresh local OHLCV cache |

## Common commands

```powershell
# Operator preflight
uv run python scripts/run_operator_preflight.py

# Start dashboard
uv run python scripts/run_dashboard.py

# Start Telegram assistant
uv run python scripts/run_telegram_bot.py

# Build a liquid shortlist for research/training
uv run python scripts/build_market_universe.py --limit 20 --source auto

# SmartAPI-native discovery (bounded instrument registry pool; optional tradable CSV)
uv run python scripts/build_market_universe.py --limit 20 --source smartapi

# Analyze the best ranked names with critique
uv run python scripts/analyze_market_shortlist.py --analysis-limit 5

# Export shortlist-driven training candidates
uv run python scripts/prepare_training_candidates.py --backfill --force-refresh

# Build a research-prep plan for ML/RL refresh and optionally backfill only the selected names
uv run python scripts/build_training_research_plan.py --source screener --selection-policy diversified --refresh-data --refresh-target rl

# Brief the top setups
uv run python scripts/brief_market_shortlist.py --analysis-limit 5

# Allocate the shortlist under simple portfolio constraints
uv run python scripts/allocate_market_shortlist.py --max-positions 3 --max-per-exposure 1

# Export a full workflow snapshot
uv run python scripts/build_workflow_snapshot.py --source auto --timeframe 5m --days 20

# Export a higher-level acceptance bundle for operator review
uv run python scripts/build_acceptance_bundle.py --workflow-snapshot reports/workflow_snapshot.json

# Or point the acceptance bundle at the latest nightly run evidence directly
uv run python scripts/build_acceptance_bundle.py --nightly-report-dir reports/nightly

# Bundle output includes refresh_context (effective target, source, split follow-up
# lanes, scout support) and cross_artifact_alignment when evidence is linked

# Build a fully local dry-run of the candidate-style nightly acceptance path
# (training candidates, research plan, agent_roles workflow compose, nightly
# report, promotion review, acceptance bundle)
uv run python scripts/run_nightly_acceptance_dry_run.py

# Emit-based replay uses the same nightly_basket emit_* entrypoints as production
# (deterministic stub builders; stops before backfill/training)
uv run python scripts/run_nightly_acceptance_dry_run.py --mode emit-replay

# Verify linkage for an existing nightly report directory
uv run python scripts/run_nightly_acceptance_dry_run.py --mode verify --report-dir reports/nightly

# Suggested local sequence: candidate -> emit-replay -> verify

# Candidate-driven nightly run with automatic training-candidate manifest and
# workflow snapshot emission
uv run --group rl python scripts/nightly_train.py --use-training-candidates --candidate-source auto

# Ask nightly to diversify across action/regime buckets instead of taking only
# the top-ranked candidate list in order
uv run --group rl python scripts/nightly_train.py --use-training-candidates --candidate-source auto --candidate-selection-policy diversified

# The nightly markdown report will reference the emitted
# training_candidates.json, training_research_plan.json, and
# workflow_snapshot.json artifacts and, after successful promotion, nightly can
# also emit promotion review artifacts under <report-dir>/promotions by default

# Promote a validated artifact and carry the workflow snapshot into the promotion audit
uv run python scripts/promote_model.py --kind rl_policy --run-id <run_id> --workflow-snapshot reports/nightly/workflow_snapshot.json

# Promotion commands will also print a compact workflow summary when that
# snapshot is supplied, so review output matches the dashboard model-health view

# Review the current promoted artifact and linked workflow context
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS

# Export a markdown review artifact for the current promoted model
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --out reports/promotion_review.md --format md

# Train RL
uv run --group rl python scripts/run_rl_train.py --help

# Train ML scorer
uv run --group ml python scripts/train_ml_signal_scorer.py --help

# Promote validated artifact
uv run python scripts/promote_model.py --help
```
