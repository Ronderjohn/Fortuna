# Cursor Phase Prompts for Fortuna

Use these prompts one at a time with Cursor agents. Ask Cursor agents to respond
in Plan mode first. After each phase is implemented, ask Codex to review,
refactor if needed, and update this roadmap.

## How to Run This Workflow

1. Open `C:\dev\Fortuna` as the Cursor workspace root.
2. Start one Cursor agent per phase, or run phases sequentially if you want less
   merge coordination.
3. Paste the Universal Cursor Preamble plus exactly one phase prompt.
4. Require Cursor to stay in Plan mode and return a plan only.
5. After you approve a plan, let Cursor implement that single phase.
6. Ask Cursor to run the phase's focused tests and report results.
7. Return to Codex with the diff for review/refactor before starting the next
   dependent phase.

Suggested order:

1. Phase 1: ML Signal Scorer.
2. Phase 2: Agentic Paper Learning Labels.
3. Phase 3: Agentic Ensemble Upgrade.
4. Phase 4: RL Policy Improvement.
5. Phase 5: Model Promotion and Monitoring.
6. Phase 6: WhatsApp Advisory Hardening.
7. Phase 7: Documentation and Operator Runbook.

Parallelization guidance:

- Phase 1 and Phase 2 can be planned in parallel, but implementation may touch
  shared `agentic/` contracts. Review carefully before merging both.
- Phase 3 should wait until Phase 1 and Phase 2 interfaces are settled.
- Phase 4 can proceed mostly independently of Phases 1-3.
- Phase 5 should wait for Phase 1 and Phase 4 artifact metadata decisions.
- Phase 6 can proceed after Phase 3 stabilizes final `AgentDecision` shape.
- Phase 7 should be last, so docs match actual implementation.

## Universal Cursor Preamble

Paste this before every phase prompt:

```text
You are working in the Fortuna repo at C:\dev\Fortuna.

Read AGENTS.md first and follow it as the shared operating guide.

Stay in Plan mode. Do not edit files yet.

Your task is the phase below only. Inspect the referenced files and produce a
decision-complete implementation plan:
- goal and current-state findings
- files to change
- public APIs/types to add or modify
- data flow
- edge cases and failure modes
- tests to add/run
- risks or assumptions

Preserve deterministic fallback, avoid live broker orders, avoid network calls in
tests, and keep the work behind settings flags where runtime behavior changes.
```

---

## Phase 1 Prompt: ML Signal Scorer

```text
Phase 1: Build the ML Signal Scorer.

Objective:
Create a supervised ML layer that scores deterministic strategy signals for
quality. This model should not replace deterministic or RL signals. It should
produce an explainable vote/score consumed later by the agentic orchestrator.

Current context to inspect:
- AGENTS.md
- src/fortuna/app/live_signals.py
- src/fortuna/backtesting/standard/pipeline.py
- src/fortuna/backtesting/standard/splits.py
- src/fortuna/backtesting/standard/filters.py
- src/fortuna/reporting/strategy_tester/metrics.py
- src/fortuna/search/evaluator.py
- src/fortuna/features/registry.py
- src/fortuna/features/builder.py
- src/fortuna/agentic/models.py
- src/fortuna/agentic/orchestrator.py
- tests/search/
- tests/backtesting/
- tests/agentic/

Design target:
Add a new ML package, likely `src/fortuna/ml/`, with:
- `labels.py`: build supervised examples from historical OHLCV, strategy signals,
  forward returns, trade outcomes, and risk-adjusted outcome labels.
- `features.py`: build tabular ML features from the latest enriched strategy/signal
  context. Reuse existing indicator/feature logic where possible.
- `signal_scorer.py`: train/load/predict a signal-quality model.
- `artifacts.py`: persist model, metadata, feature names, label config, train/OOS
  metrics, and calibration info.
- optional `agent.py`: adapter that converts scorer output into an `AgentVote`.

Recommended dependency:
Use scikit-learn if already available in the `ml` dependency group. If dependency
changes are needed, add them to `pyproject.toml` under the existing `ml` group.

Minimum model:
- LogisticRegression or HistGradientBoostingClassifier is enough for v1.
- Fixed random seed.
- No neural net in this phase.

Labeling requirements:
- Avoid future leakage.
- Each example is generated at a bar where a strategy emitted BUY/SELL/EXIT.
- Use a configurable forward horizon, default 3 bars for 5m data.
- Include cost-adjusted forward return where possible.
- Positive label means the signal had favorable forward outcome after costs.
- Negative label means unfavorable/flat outcome.
- HOLD rows may be included only if useful and sampled to avoid imbalance.
- Store label config in artifact metadata.

Inference requirements:
- If no model artifact exists, scorer must return unavailable/neutral, not fail.
- Prediction output should include:
  - probability/score
  - class label
  - model id/path
  - feature vector metadata
  - reason string suitable for agent votes

Agentic integration requirement:
Do not hard-wire the model into `AgenticOrchestrator` yet unless the interface is
small and optional. Prefer creating an adapter that can be wired in Phase 3.

Tests:
- Unit test label generation on a tiny deterministic OHLCV frame.
- Unit test no-leakage behavior: labels only use future outcome, features only use
  current/past columns.
- Unit test model train/save/load/predict on a tiny synthetic dataset.
- Unit test unavailable artifact returns neutral output.
- Add tests under `tests/ml/`.

Acceptance criteria:
- New `fortuna.ml` package exists with typed, tested surfaces.
- A tiny model can be trained and loaded without network or SmartAPI.
- No runtime dashboard behavior changes yet.
- `uv run pytest tests/ml -q` passes.
- Existing focused tests still pass:
  `uv run pytest tests/agentic tests/execution/test_session_wiring.py -q`
```

---

## Phase 2 Prompt: Agentic Paper Learning Labels

```text
Phase 2: Convert agentic decisions and paper outcomes into a durable learning dataset.

Objective:
The advisory system should learn from its own recommendations. Extend the existing
agentic/paper logging so each final AgentDecision can later be labeled with what
happened after the recommendation.

Current context to inspect:
- AGENTS.md
- src/fortuna/agentic/models.py
- src/fortuna/agentic/store.py
- src/fortuna/app/session_engine.py
- src/fortuna/execution/router.py
- src/fortuna/execution/monitor.py
- src/fortuna/execution/journal.py
- src/fortuna/paper/learner.py
- tests/agentic/
- tests/execution/test_session_wiring.py

Design target:
Extend `src/fortuna/agentic/` with a learning dataset component:
- `learning.py`: resolve outcomes for prior decisions after a horizon or after
  paper execution closes a trade.
- `PaperLearningEvent` may be extended if needed, but preserve backward-compatible
  JSON fields where possible.
- `AgenticDecisionStore` should support reading recent decisions/events, not only
  appending.

Dataset requirements:
Each learning row should include:
- decision hash
- timestamp/bar time
- symbol/timeframe
- final action
- confidence
- current side
- bar close
- agent vote summary
- notification status if available
- paper router submission status if enabled
- later outcome:
  - forward return over configurable horizon
  - MFE/MAE if easy from available OHLCV
  - whether action was directionally correct
  - whether paper trade filled/closed
  - realized P&L if available
  - risk block reason if blocked

Important constraints:
- Do not place live broker orders.
- Do not require SmartAPI credentials in tests.
- Do not make the dashboard depend on this dataset for rendering.
- If outcome cannot be resolved yet, leave row pending and update later.

Public API idea:
- `AgenticLearningStore.append_pending(decision)`
- `AgenticLearningStore.resolve_with_ohlcv(decision_hash, ohlcv, horizon_bars)`
- `AgenticLearningStore.unresolved(limit=...)`
- `build_learning_rows(...) -> list[LearningExample]`

Tests:
- Append/read pending decision.
- Resolve BUY and SELL outcomes from a tiny OHLCV frame.
- Record risk-block/paper-submitted events.
- Ensure JSONL remains readable if optional fields are missing.

Acceptance criteria:
- Learning rows are durable under `logs/agentic/`.
- Tiny deterministic tests prove outcomes are labeled correctly.
- The new dataset is ready for Phase 1 ML scorer consumption.
- `uv run pytest tests/agentic -q` passes.
```

---

## Phase 3 Prompt: Agentic Ensemble Upgrade

```text
Phase 3: Upgrade the agentic orchestrator to consume deterministic, ML, RL,
regime, portfolio, and risk evidence.

Objective:
Fortuna should produce one final advisory decision per symbol with explainable
confidence, source votes, dissent, risk notes, and notification intent.

Current context to inspect:
- AGENTS.md
- src/fortuna/agentic/models.py
- src/fortuna/agentic/agents.py
- src/fortuna/agentic/orchestrator.py
- src/fortuna/ml/ if Phase 1 exists
- src/fortuna/app/live_signals.py
- src/fortuna/rl/inference/signal_generator.py
- src/fortuna/rl/inference/regime_detector.py
- src/fortuna/strategies/regime_router.py
- src/fortuna/app/session_engine.py
- tests/agentic/

Design target:
Make the orchestrator a more explicit internal multi-agent workflow:
- MarketContextAgent: summarizes latest bar/session/regime.
- DeterministicSignalAgent: strategy signal votes.
- MLSignalScorerAgent: optional signal-quality vote from ML scorer.
- RLPolicyAgent: optional RL action vote.
- PortfolioContextAgent: current position/holding state.
- RiskReviewAgent: blocks/penalizes weak, conflicting, or unsafe actions.
- DecisionSynthesizer: final action, confidence, rationale.
- NotificationComposer: WhatsApp-ready message from final decision.

Requirements:
- Every agent must be optional/fail-soft.
- Missing ML/RL artifacts produce neutral votes, not errors.
- Final decision is an `AgentDecision`.
- Votes must be serializable and stored.
- Keep action taxonomy stable:
  BUY, SELL, EXIT_LONG, EXIT_SHORT, HOLD, DO_NOT_ENTER.
- Do not send notifications inside low-level agents; only final decision gets a
  notification intent.

Confidence guidance:
- More independent agreement increases confidence.
- ML/RL disagreement reduces confidence.
- Current open position changes action priority: exits matter more than new entries.
- Risk notes should be visible, not hidden.
- Confidence is advisory, not a guarantee.

Tests:
- Existing deterministic-only behavior still works.
- ML scorer available/unavailable cases.
- RL agrees/disagrees cases.
- Open LONG exits only on exit evidence.
- Flat symbol chooses stronger BUY vs SELL consensus.
- Risk review can downgrade weak entry to DO_NOT_ENTER or low-confidence HOLD.
- Notification intent only appears for actionable decisions or important reversals.

Acceptance criteria:
- `AgenticOrchestrator.decide()` remains the primary API.
- `FortunaSessionEngine` does not need major changes.
- `uv run pytest tests/agentic tests/execution/test_session_wiring.py -q` passes.
```

---

## Phase 4 Prompt: RL Policy Improvement and Promotion Readiness

```text
Phase 4: Improve RL policy quality, evaluation, and promotion readiness.

Objective:
Fortuna already has a Gymnasium trading env, PPO trainer, checkpoint metadata,
and live inference. Tighten the RL pipeline so policies can be trusted as advisory
votes only after strict OOS validation.

Current context to inspect:
- AGENTS.md
- src/fortuna/rl/env/trading_env.py
- src/fortuna/rl/env/reward.py
- src/fortuna/rl/training/trainer.py
- src/fortuna/rl/training/checkpoint.py
- src/fortuna/rl/inference/signal_generator.py
- src/fortuna/features/builder.py
- src/fortuna/features/registry.py
- src/fortuna/backtesting/standard/filters.py
- scripts/nightly_train.py
- tests/rl/

Focus areas:
1. Reward shaping:
   - Detect and discourage hold-locked policies.
   - Penalize overtrading after costs.
   - Reward risk-adjusted terminal performance, not raw P&L only.
   - Preserve square-off behavior.

2. Evaluation:
   - Aggregate OOS metrics across folds.
   - Compare against deterministic baseline for same symbol/timeframe.
   - Store verdict reasons and failure modes in metadata.

3. Per-symbol policy discipline:
   - Prefer `models/live/by_symbol/<SYMBOL_KEY>/`.
   - Do not silently use global policy for unrelated symbols.

4. Checkpoint safety:
   - Feature registry hash mismatch should fail closed.
   - Missing normalizer should be logged and handled intentionally.
   - Invalid model load returns unavailable generator.

5. Tests:
   - Tiny env smoke tests.
   - Reward edge cases.
   - Checkpoint metadata read/write.
   - Inference unavailable fallback.
   - Feature hash mismatch.

Do not:
- Add live execution.
- Make GPU required.
- Add heavyweight training tests.

Acceptance criteria:
- RL inference remains optional and fail-soft.
- Policy metadata clearly explains whether the checkpoint is advisory-ready.
- `uv run pytest tests/rl tests/features -q` passes.
```

---

## Phase 5 Prompt: Model Promotion and Monitoring

```text
Phase 5: Build model promotion gates and monitoring surfaces.

Objective:
Create an auditable promotion workflow for ML/RL models before they influence live
advisory decisions. Add dashboard/reporting surfaces so the operator can understand
which models are active and why.

Current context to inspect:
- AGENTS.md
- scripts/nightly_train.py
- src/fortuna/rl/training/checkpoint.py
- src/fortuna/rl/inference/signal_generator.py
- src/fortuna/ml/ if implemented
- src/fortuna/agentic/store.py
- src/fortuna/app/session_engine.py
- app/streamlit_app.py
- tests/rl/
- tests/agentic/

Design target:
Add promotion metadata and gates:
- model kind: ML scorer / RL policy / regime detector
- symbol/timeframe
- train window and OOS window
- feature schema hash
- metrics and verdict
- baseline comparison
- promotion status: candidate, validated, live, rejected
- reason list

Possible files:
- `src/fortuna/models/registry.py`
- `src/fortuna/models/promotion.py`
- `src/fortuna/models/metadata.py`
- tests under `tests/models/`

Promotion rules:
- No model becomes live advisory unless OOS verdict passes.
- RL policy should beat or complement deterministic baseline.
- ML scorer must show useful calibration/precision on OOS labels.
- Feature schema mismatch blocks promotion.
- Promotion writes a pointer file instead of copying large artifacts where possible.

Dashboard requirements:
- Add or extend a dashboard panel showing active ML/RL model status.
- Show unavailable/degraded state clearly.
- Show latest agentic decision log summary if cheap.
- Do not make the dashboard crash if artifacts are missing.

Tests:
- Promote valid fake metadata.
- Reject failed metadata.
- Missing artifact fails closed.
- Live pointer resolution.
- Dashboard helper renders missing artifacts without error where testable.

Acceptance criteria:
- Model promotion is explicit and auditable.
- Live advisory only loads promoted artifacts.
- Existing deterministic path remains unchanged if no model is live.
```

---

## Phase 6 Prompt: WhatsApp Advisory Hardening

```text
Phase 6: Harden WhatsApp advisory delivery.

Objective:
Make WhatsApp alerts production-safe for advisory use: deduped, throttled,
audited, concise, and downstream of final AgentDecision only.

Current context to inspect:
- AGENTS.md
- src/fortuna/agentic/models.py
- src/fortuna/agentic/notifiers.py
- src/fortuna/agentic/store.py
- src/fortuna/app/session_engine.py
- .env.example
- tests/agentic/test_notifiers.py

Requirements:
- Notification is created only from final `AgentDecision`.
- No low-level strategy/RL/ML module sends messages.
- Dedupe by symbol/action/bar_time/decision_hash.
- Add throttle controls:
  - max messages per symbol per session
  - min seconds between messages per symbol
  - optional quiet hours outside NSE session
- Persist notification attempts/results with decision hash.
- Include enough context:
  - action
  - symbol
  - price
  - confidence
  - top reasons
  - risk notes
  - paper-learning status
  - "advisory only" wording
- Twilio remains the first concrete provider.
- Tests must not call Twilio network.

Possible additions:
- `NotificationPolicy`
- `NotificationAuditStore`
- richer `NotificationResult`
- fake notifier for tests

Tests:
- Dedupe repeat decision.
- Throttle repeated symbol alerts.
- Different action on same symbol can send.
- Twilio payload format.
- Incomplete credentials fail soft.
- Store notification attempts/results.

Acceptance criteria:
- WhatsApp alerts cannot spam on Streamlit refresh.
- Operators can audit what was sent and why.
- Disabled WhatsApp has zero runtime side effects.
```

---

## Phase 7 Prompt: Documentation and Operator Runbook

```text
Phase 7: Document the agentic ML/RL advisory workflow.

Objective:
Create practical docs for running, debugging, and reviewing Fortuna's advisory
pipeline.

Current context to inspect:
- AGENTS.md
- README.md
- Phase_1.md
- Phase_2.md
- docs/
- .env.example

Deliverables:
- `docs/agentic_advisory.md`: architecture, data flow, settings, failure modes.
- `docs/ml_signal_scorer.md`: training, labels, artifacts, validation.
- `docs/rl_policy_workflow.md`: training, checkpoints, promotion, live inference.
- `docs/operator_runbook.md`: dashboard usage, WhatsApp setup, paper learning,
  what to check before market open, what to check after market close.
- Update README with links only; do not paste huge docs into README.

Requirements:
- Be explicit that current system is advisory plus paper learning.
- Explain deterministic fallback.
- Explain how to disable every optional subsystem.
- Include focused commands:
  - run dashboard
  - run tests
  - train ML scorer if implemented
  - train RL if implemented
  - promote model if implemented
- Include troubleshooting for missing SmartAPI, missing model artifacts, and
  WhatsApp credential issues.

Acceptance criteria:
- Docs match actual code and settings.
- A new contributor can understand the phased system without reading every file.
```
