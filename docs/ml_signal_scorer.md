# ML Signal Scorer

The ML signal scorer (`src/fortuna/ml/`) scores **deterministic strategy signals**
for quality. It does not replace strategies or RL; it produces an explainable vote
consumed by the agentic orchestrator when enabled.

## Package layout

| Module | Purpose |
|---|---|
| `labels.py` | Supervised examples from historical signals + forward outcomes |
| `features.py` | Tabular features from enriched OHLCV context |
| `signal_scorer.py` | Train / save / load / predict |
| `artifacts.py` | Persist `model.joblib` + `metadata.json` |
| `agent.py` | `MLSignalScorerAgent` adapter → `AgentVote` |

## Labels

- One example per bar where a strategy emitted BUY/SELL/EXIT.
- Configurable forward horizon (default 3 bars on 5m data).
- Cost-adjusted forward return where possible.
- No future leakage: features use current/past data only; labels use forward bars.

Label config is stored in artifact `metadata.json`.

## Training

CLI entrypoint:

```powershell
uv run python scripts/train_ml_signal_scorer.py --help
```

The training script reads resolved agentic learning rows, rebuilds enriched OHLCV
features from cache, trains a scorer, and writes artifacts under
`models/ml_signal_scorer/validated/<run_id>/`.

Programmatic training remains available as well (see `tests/ml/test_signal_scorer.py`):

```python
from pathlib import Path
import numpy as np
from fortuna.ml.features import ML_FEATURE_NAMES
from fortuna.ml.signal_scorer import SignalScorer
from fortuna.ml.types import LabelConfig, ScorerMetadata

X = ...  # shape (n, len(ML_FEATURE_NAMES))
y = ...  # binary labels

scorer = SignalScorer(random_state=42)
meta = ScorerMetadata(
    run_id="my_run_001",
    symbol="RELIANCE.NS",
    timeframe="5m",
    strategy_name="orb",
    label_config=LabelConfig(),
    verdict_passed=True,
)
scorer.fit(X, y, metadata=meta, eval_X=X[:10], eval_y=y[:10])
scorer.save(Path("models/ml_signal_scorer/validated/my_run_001"))
```

## Artifact layout

```
models/ml_signal_scorer/
  validated/<run_id>/
    model.joblib
    metadata.json
  live/
    live.json          # promotion pointer (when promoted)
```

`metadata.json` includes train/OOS metrics, `feature_schema_hash`, `verdict_passed`,
and `advisory_ready`.

## Advisory-ready gate

A scorer is advisory-ready when:

- `verdict_passed` is true
- Feature schema hash matches current registry
- OOS precision ≥ 0.55 and ROC-AUC ≥ 0.55

Computed in `fortuna.models.promotion.compute_ml_advisory_ready()`.

## Live inference

Enable in `.env`:

```
FORTUNA_AGENTIC_ENABLED=1
FORTUNA_AGENTIC_ML_SCORER_ENABLED=1
```

When `FORTUNA_MODEL_REGISTRY_ENABLED=1`, only a **promoted** artifact (via
`models/ml_signal_scorer/live/live.json`) is loaded. With registry off, the latest
mtime under `validated/` is used (legacy behavior).

`SignalScorer.load(..., require_promotion=True)` rejects artifacts that fail gates.

## Promotion

```powershell
uv run python scripts/promote_model.py --kind ml_scorer --run-id my_run_001
```

Writes a live pointer and audit entry under `models/registry/promotions.jsonl`.

## Disabling

```
FORTUNA_AGENTIC_ML_SCORER_ENABLED=0
```

Orchestrator runs without ML votes; deterministic + other agents unchanged.

## Tests

```powershell
uv run pytest tests/ml -q
```
