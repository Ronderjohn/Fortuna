# RL Policy Workflow

Fortuna trains PPO policies in a Gymnasium trading environment and exposes them as
optional advisory votes. Policies must pass OOS validation and explicit promotion
before influencing live advisory when the model registry is enabled.

## Key modules

| Path | Role |
|---|---|
| `src/fortuna/rl/env/` | Trading environment, reward shaping, NSE session rules |
| `src/fortuna/rl/training/trainer.py` | Walk-forward PPO training |
| `src/fortuna/rl/training/checkpoint.py` | `PolicyCheckpoint` metadata + `advisory_ready` |
| `src/fortuna/rl/inference/signal_generator.py` | Live inference + safety gates |
| `src/fortuna/models/promotion.py` | Promotion gates + live pointer I/O |

## Training

```powershell
uv run --group rl python scripts/run_rl_train.py --help
```

Checkpoints land under:

```
models/validated/<run_id>/
  policy.zip
  normalizer.json
  metadata.json
```

Failed runs may appear under `models/rejected/<run_id>/`.

## Checkpoint metadata

`PolicyCheckpoint` stores OOS metrics, verdict, feature registry hash, baseline
comparison, fold metrics, and `advisory_ready` (computed via `compute_advisory_ready()`).

Inference fails closed when:

- Feature registry hash mismatches
- Normalizer missing or invalid
- `advisory_ready=false` (blocks `predict()`)

## Per-symbol live policies

Preferred live layout:

```
models/live/by_symbol/<SYMBOL_KEY>/live.json
```

Pointer JSON references the validated artifact directory. Legacy copy-based dirs
(with `policy.zip` directly under `by_symbol/`) still work when registry is off.

Global fallback (`models/live/`) is disabled by default. Enable only for dev:

```
FORTUNA_RL_ALLOW_GLOBAL_POLICY=1
```

## Promotion

Manual per-symbol or global:

```powershell
uv run python scripts/promote_model.py --kind rl_policy --run-id <run_id>
uv run python scripts/promote_policy.py --run-id <run_id> --symbol RELIANCE.NS
uv run python scripts/promote_model.py --kind rl_policy --run-id <run_id> --workflow-snapshot reports/nightly/workflow_snapshot.json
```

Nightly pipeline (`scripts/nightly_train.py`) promotes best `advisory_ready` checkpoint
per symbol via pointer files (not full artifact copies). Candidate-driven
nightly runs can emit a workflow snapshot artifact, and those nightly
promotions now carry that snapshot path into the live pointer payload and
`models/registry/promotions.jsonl` audit.
When a workflow snapshot is supplied on the promotion CLI, the command also
prints a compact summary of the originating universe / shortlist / briefing /
training-candidate counts for quicker manual review.

For a fuller operator review after promotion:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS
```

That review pulls the promoted pointer payload, key OOS metrics, promotion
reasons, and linked workflow summary into one output.

To archive the review as an artifact:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --out reports/promotion_review.md --format md
```

Nightly runs can now do this automatically after successful promotion, placing
review artifacts under the nightly report directory's `promotions/` folder by
default.

When registry is enabled:

```
FORTUNA_MODEL_REGISTRY_ENABLED=1
FORTUNA_MODEL_PROMOTION_REQUIRED=1
```

Missing live pointer → no RL load for that symbol; deterministic signals continue.

## Activation path (trained → promoted → active)

Fortuna surfaces a typed activation stage per lane via `build_lane_activation_summary()`:

1. **Train** — artifact lands under `models/validated/<run_id>/` (or ML `ml_signal_scorer/validated/`).
2. **Promote** — `promote_policy.py` / `promote_model.py` writes the live pointer when registry is strict.
3. **Load** — session engine loads the pointer on dashboard start or **Reload policy** / **Reload ML scorer**.
4. **Active** — loaded generator/scorer with `advisory_ready` metadata participates in agentic votes.

Check current stage without a promotion audit row:

```powershell
uv run python scripts/review_promotion.py --kind rl_policy --symbol RELIANCE.NS --activation-only
```

Stages: `disabled`, `missing_artifact`, `unpromoted`, `promoted_not_advisory_ready`, `not_loaded`, `active`.

## Live inference

`RLSignalGenerator` is wired per symbol in `FortunaSessionEngine._rl_generator_for_symbol()`.
Dashboard **Models** tab shows policy status, advisory-ready flag, and live pointer path.

Reload without restart: **Reload policy** button → `reload_rl_generator()`.

## Disabling RL

With no checkpoint or registry blocking load, RL votes are simply absent. Deterministic
dashboard signals are unchanged.

## Tests

```powershell
uv run --group rl pytest tests/rl tests/models -q
```

## Related

- [Agentic advisory](agentic_advisory.md)
- [Operator runbook](operator_runbook.md)
