# Fortuna Current System Flow

Practical architecture map of what is implemented today in Fortuna. This is the
current-state view for operators and engineers, not a roadmap.

## 1. Runtime advisory pipeline

```text
market data / cache / live bars
  -> deterministic strategy signals
  -> optional RL policy vote
  -> optional ML signal-quality vote
  -> agentic orchestrator
  -> final AgentDecision
  -> decisions.jsonl
  -> optional paper router + learning rows
  -> optional Telegram notification
  -> optional conversational adapter surface (Telegram + dashboard Assistant tab)
```

Key properties:
- Deterministic strategy output is the baseline and survives subsystem failure.
- RL and ML are advisory inputs, not hard overrides.
- Telegram is downstream of final `AgentDecision` only.
- The conversational adapter, when enabled, sits above typed tools and does not
  replace the advisory core.
- Paper learning remains paper-only and does not place live broker orders.

## 2. Training and promotion pipeline

### ML scorer

```text
learning_rows.jsonl
  -> resolved signal examples
  -> enriched OHLCV features
  -> SignalScorer artifact
  -> models/ml_signal_scorer/validated/<run_id>/
  -> promote_model.py
  -> models/ml_signal_scorer/live/live.json
```

### RL policy

```text
historical OHLCV
  -> walk-forward PPO training
  -> validated/rejected checkpoint
  -> promote_model.py or promote_policy.py
  -> models/live/by_symbol/<SYMBOL_KEY>/live.json
```

Promotion commands can optionally carry a workflow snapshot path, so the live
pointer and registry audit can reference the shortlist/briefing/allocation/candidate
artifact that informed the promotion decision.

### Runtime loading

- With `FORTUNA_MODEL_REGISTRY_ENABLED=1` and `FORTUNA_MODEL_PROMOTION_REQUIRED=1`,
  runtime loads only explicitly promoted live pointers.
- If a promoted artifact is missing, Fortuna falls back to deterministic signals.
- Dashboard **Models** surfaces pointer state, advisory readiness, promotions,
  recent paper-learning outcomes, and any linked workflow snapshot path from
  the promotion audit, including a compact summary of the originating universe /
  shortlist / briefing / allocation / training-candidate counts.

## 3. Key artifacts and logs

### Models

- `models/validated/<run_id>/` — RL validated checkpoints
- `models/rejected/<run_id>/` — RL rejected checkpoints
- `models/live/by_symbol/<SYMBOL_KEY>/live.json` — RL live pointers
- `models/ml_signal_scorer/validated/<run_id>/` — ML scorer artifacts
- `models/ml_signal_scorer/live/live.json` — ML scorer live pointer
- `models/registry/promotions.jsonl` — append-only promotion audit log, now able
  to reference a workflow snapshot artifact when promotion was tied to one

### Agentic runtime

- `logs/agentic/decisions.jsonl` — final advisory decisions
- `logs/agentic/notifications.jsonl` — sent/deduped/throttled Telegram audit
- `logs/agentic/learning_rows.jsonl` — durable paper-learning rows and outcomes
- `logs/agentic/learning.jsonl` — paper-learning event stream
- workflow snapshots under `reports/` can now summarize:
  - universe count
  - adaptive universe note/count when recent paper outcomes shifted ranking
  - adaptive score adjustment and supporting row count per universe name
  - shortlist count
  - briefing candidate count
  - allocation selected/skipped counts
  - allocation score / risk / sizing metadata for selected baskets
  - allocation critic metadata when redundant weak setups were penalized or skipped
  - optional research-target metadata when allocation was aligned against the
    typed ML/RL training-research plan
  - ML/RL candidate counts
  - training-research refresh intent and refreshed-row count when the operator
    runs the in-app remediation path
- acceptance bundles under `reports/acceptance/` can now tie together:
  - one workflow snapshot
  - optional nightly report evidence from `reports/nightly/*.json`
  - current RL/ML promotion review evidence
  - model-health visibility of the linked workflow context
  - candidate-driven nightly step-shape checks for basket/workflow/promotion coverage
  - workflow-level allocation research-alignment visibility when the shortlist
    basket was compared against the typed ML/RL training-research plan
  - workflow-level training-research refresh visibility when remediation was
    requested from the dashboard workflow console
- nightly markdown reports can now also surface that allocation
  research-alignment summary directly from the workflow snapshot metadata
- model-health summaries can now scan recent nightly reports and expose a small
  recent-run allocation research-alignment trend beside promotion state
- acceptance bundles can now warn when recent nightly allocation research
  alignment falls below a configured threshold
- promotion review output can now surface that same recent nightly alignment
  warning beside the linked workflow snapshot summary
- acceptance bundles and promotion review can now also warn when the current
  workflow snapshot shows weak basket/research overlap, so the selected basket
  drifting away from the current ML/RL prep plan becomes part of review output
- acceptance-bundle and promotion-review exports now also persist the typed
  remediation guidance for that drift: recommended refresh target,
  force-refresh mode, and suggested command
- the dashboard model-health surface can now show that same recent nightly
  alignment trend and warn when basket/research coherence is slipping
- model-health can now also recommend refreshing the training-research plan and
  candidate-selection policy when that recent alignment trend falls too low
- that recommendation can now include a concrete training-research CLI command
  so the operator has a direct remediation path
- the dashboard workflow console can now load the typed training-research plan
  directly, so the same remediation path is available in-app
- that in-app training-research path can now optionally refresh targeted data
  for ML/RL prep in the same bounded workflow
- the dashboard Assistant tab and Telegram assistant now both expose direct
  training-research inspection commands, so operators can inspect the current
  typed ML/RL refresh plan and nightly posture context without leaving the
  assistant surface
- Telegram `/workflow` can now call the explicit typed multi-agent workflow
  surface directly, so operator chat no longer has to hand-compose the same
  universe/critic/briefing/allocation flow stage by stage
- the dashboard workflow console now refreshes through that same explicit typed
  multi-agent workflow surface, so chat and dashboard share one composed
  universe/critic/briefing/allocation/research backbone
- that shared workflow posture can now also preserve discovery alignment, so
  operators can see whether Screener/liquidity scouting and OHLCV
  activity/trend scouting are converging on the same top names
- the training-research planner can now mildly prefer names in that discovery
  overlap set, so scout agreement has a bounded effect on ML/RL refresh picks
- shortlist analysis can now also treat that overlap as a small, flag-gated
  support signal and mark liquidity-only names as needing stronger activity
  confirmation before they rise in advisory ranking
- the allocator critic can now also penalize same-side names that are not in
  the discovery-overlap set when the current basket already contains a stronger
  scout-reinforced peer
- acceptance bundles and promotion review can now also surface that discovery
  posture from workflow snapshots, so recurring artifacts can show whether the
  final advisory path stayed close to the discovery stack
- those same acceptance and promotion-review artifacts can now also persist
  discovery-side remediation guidance, so weak discovery posture can recommend
  rebuilding the market-universe pass alongside the existing ML/RL refresh
  recovery path
- promotion review and nightly markdown reports can now also surface that
  training-research refresh evidence directly, so remediation attempts remain
  visible in recurring review artifacts
- workflow snapshots, nightly workflow metadata, and model-health alignment
  summaries can now also preserve refreshed allocator research-target mix, so
  recurring runs can show whether the selected basket tracked recently refreshed
  ML/RL prep evidence
- acceptance and promotion review can now also warn when refreshed-alignment
  coverage is weak across recent enabled nightly runs, which makes drift from
  actually refreshed prep evidence part of the higher-level review path
- the allocator can now also read recent nightly training-research execution
  evidence directly, so recurring refreshed/promoted research support can push
  back on weaker same-side basket additions before they are selected
- model-health remediation can now bias the suggested training-research refresh
  target toward `ml`, `rl`, or `all` from the recent refreshed-alignment mix,
  so the recommended next action is more specific to the observed drift
- nightly workflow metadata and model-health summaries now also preserve the
  latest workflow discovery posture, so operator remediation can distinguish
  between "refresh ML/RL prep" and "revisit the market-universe discovery pass"
- when that discovery posture is weak, model health can now suggest rebuilding
  the market universe from SmartAPI instrument seeds, Screener CSV, or registry
  fallback, with a concrete `build_market_universe.py` command beside the
  existing training-research remediation path
- market-universe responses now carry typed provider fields (`seed_source`,
  `scoring_source`, `provider_summary`, `fallback_from`) so operators can see
  which discovery path won and whether fallback occurred
- the explicit runtime team response, Telegram `/workflow`, and the dashboard
  workflow console now also carry that discovery-side follow-up posture, so the
  live operator view can distinguish discovery recovery from ML/RL refresh
  recovery without waiting for nightly review artifacts
- the dashboard workflow console now preselects that inferred refresh target in
  the in-app remediation controls, so the operator action path stays aligned
  with the same guidance unless manually overridden
- when refreshed-alignment drift is severe, those same remediation controls can
  now also prefill `Refresh research data` and sometimes `Force research
  refresh`, so the in-app recovery defaults are a little more proactive
- acceptance bundles now export typed refresh context:
  `effective_refresh_target`, `refresh_target_source`,
  `recommended_follow_up_lane`, `scout_support_target`, and
  `scout_support_summary`; markdown format includes a compact `refresh_context=`
  line
- `src/fortuna/app/acceptance_alignment.py` provides shared helpers for refresh
  context resolution and `CrossArtifactAlignmentSummary` across acceptance
  bundles and promotion review
- the nightly acceptance dry-run (`scripts/run_nightly_acceptance_dry_run.py`)
  supports three modes:
  - `candidate` (default): synthetic fixtures via `summarize_multi_agent_workflow`
    and `agent_roles` compose
  - `emit-replay`: routes through `nightly_basket.emit_*` + `nightly_train.write_report`
    with deterministic stub builders (stops before backfill/training)
  - `verify`: checks `NightlyArtifactLinkageSummary` on an existing nightly report dir
- acceptance bundles can carry `replay_linkage` and a `nightly_replay_linkage` check
- `src/fortuna/app/agent_roles/` packages thin role wrappers
  (`liquidity_scout`, `activity_scout`, `universe_scout`, `instrument_analyst`,
  `briefing_agent`, `portfolio_critic`, `research_planner`,
  `operations_monitor`) consumed by `multi_agent_team.py`

### Market-universe discovery sources

```text
build_market_universe(source=auto|smartapi|screener|registry)
  smartapi:
    tradable_universe.csv (optional) -> InstrumentRegistry.catalog (bounded)
    -> SmartAPI OHLCV ranking -> MarketUniverseResponse
  screener:
    screener_export.csv -> SmartAPI OHLCV ranking
  auto (default):
    screener CSV if present (unless auto_prefer_smartapi)
    else smartapi seeds -> registry fallback
```

### Acceptance artifact chain

```text
nightly report (reports/nightly/*.json)
  + workflow snapshot (workflow_snapshot.json)
  + promotion review (optional)
  + model-health context
  -> acceptance_alignment.resolve_acceptance_refresh_context()
  -> acceptance_alignment.build_cross_artifact_alignment()
  -> acceptance bundle (reports/acceptance/)
  -> promotion review (same cross_artifact_alignment when linked)
```

Local rehearsal:

```text
run_nightly_acceptance_dry_run.py [--mode candidate|emit-replay|verify]
  candidate:
    -> training_candidates.json + training_research_plan.json
    -> multi_agent_team / agent_roles compose
    -> summarize_multi_agent_workflow -> workflow_snapshot.json
  emit-replay:
    -> nightly_basket.emit_training_candidate_manifest
    -> nightly_basket.emit_training_research_plan
    -> nightly_basket.emit_workflow_snapshot
    -> nightly_train.write_report (stub promotion + review)
  verify:
    -> verify_nightly_artifact_linkage(report_dir)
  -> gather_acceptance_bundle (optional replay_linkage check)
```

## 4. Repo structure map

| Area | Purpose |
|---|---|
| `src/fortuna/data` | Data sources, cache, live feed, symbol registry |
| `src/fortuna/indicators` | Canonical indicators |
| `src/fortuna/backtesting` | Backtest engines and validation |
| `src/fortuna/features` | Shared feature/observation transforms |
| `src/fortuna/ml` | Signal scorer training, artifacts, feature rows |
| `src/fortuna/rl` | RL env, training, checkpoints, inference |
| `src/fortuna/agentic` | Orchestrator, decisions, learning rows, notifications |
| `src/fortuna/execution` | Paper broker, router, monitor, reconciliation |
| `src/fortuna/app` | Session engine, model status, dashboard helpers |
| `src/fortuna/app/agent_roles` | Thin role wrappers for multi-agent workflow |
| `src/fortuna/app/acceptance_alignment.py` | Shared refresh context and cross-artifact alignment |
| `scripts` | Operator, training, promotion, diagnostics entrypoints |

## 5. Operator mental model

- **Deterministic baseline:** always available if OHLCV and strategies load.
- **Optional advisory stack:** ML, RL, and agentic orchestration layer on top.
- **Fail-soft behavior:** missing or invalid ML/RL/Telegram components should not
  take down deterministic dashboard usage.
- **Promotion gate:** promoted artifacts are what matter in registry mode, not
  merely the latest trained run.
- **Paper-only downstream actions:** paper router, learning outcomes, and
  notifications are implemented; live order placement is not the normal target.
- **Portfolio shaping:** shortlist allocation now combines simple limits with a
  bounded critic that can push back on same-side, same-regime, watch-grade, and
  high-risk basket crowding before those names become selected advice.

## Related docs

- [Operator runbook](operator_runbook.md)
- [Agentic advisory](agentic_advisory.md)
- [ML signal scorer](ml_signal_scorer.md)
- [RL policy workflow](rl_policy_workflow.md)
