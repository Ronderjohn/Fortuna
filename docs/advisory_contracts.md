# Advisory contracts

Stable typed request/response shapes for interactive Fortuna advisory surfaces
(Telegram, dashboard helpers, and future tool-calling layers).

## Purpose

Fortuna's orchestration layer uses rich internal types such as `AgentDecision`
in [`src/fortuna/agentic/models.py`](../src/fortuna/agentic/models.py). Those
types include audit metadata, votes, and notification payloads that should not
leak into user-facing interfaces.

Plan 01 introduces a separate contract layer in
[`src/fortuna/agentic/contracts.py`](../src/fortuna/agentic/contracts.py) plus
runtime builders in
[`src/fortuna/app/advisory_service.py`](../src/fortuna/app/advisory_service.py)
and [`src/fortuna/app/model_status.py`](../src/fortuna/app/model_status.py).

Plan 02 adds the callable tool surface in
[`src/fortuna/agentic/tools.py`](../src/fortuna/agentic/tools.py).

## Tool surface

`FortunaAdvisoryTools` is the canonical entrypoint for interactive advisory
operations:

- `search_instruments(query, *, limit=10) -> InstrumentSearchResponse`
- `analyze_instrument(symbol, *, timeframe, days, force_refresh) -> InstrumentAnalysisResponse`
- `get_model_health() -> ModelHealthResponse`
- `get_recent_learning_summary(symbol=None) -> LearningSummaryStatus`

Layering:

```text
parse -> route -> FortunaAdvisoryTools -> UI formatters
```

See [conversation_boundary.md](conversation_boundary.md) for the routing layer.

Obtain tools from a live session via `FortunaSessionEngine.advisory_tools()` or
construct with `build_advisory_tools(settings, engine=...)`.

## Search contracts

`InstrumentSearchResponse` returns typed `InstrumentSearchHitSummary` rows
(symbol, display, segment, tradingsymbol). Search payloads exclude catalog
DataFrames and registry internals.

## Analysis contracts

### Request

`InstrumentAnalysisRequest` captures:

- `symbol` — requested instrument token
- `timeframe` — bar interval (default `5m`)
- `days` — lookback window (default `30`)
- `force_refresh` — optional cache bypass

Telegram maps parsed commands via
`instrument_analysis_request_from_telegram()` in
[`src/fortuna/telegram/formatters.py`](../src/fortuna/telegram/formatters.py)
without changing the parser.

### Response

`InstrumentAnalysisResponse` is the canonical analysis payload:

- `ok` — whether analysis completed
- `instrument` — `InstrumentSnapshot` with resolved symbol, segment label, last bar
- `decision` — optional slim `DecisionSummary` (not full `AgentDecision`)
- `signals` — filtered actionable deterministic signals
- `winning_strategy` — walk-forward winner name when available
- `error` — structured `AdvisoryError` when `ok=False`

Public payloads intentionally exclude:

- OHLCV DataFrames
- batch/runtime objects
- agent votes and notification internals
- enriched ML feature frames

### Errors

`AdvisoryError` uses stable codes:

| Code | Meaning |
|------|---------|
| `invalid_request` | Missing or malformed request |
| `resolve_failed` | Instrument registry could not resolve symbol |
| `load_failed` | Session load failed (data/cache/runtime) |
| `decision_unavailable` | Reserved for explicit no-decision cases |
| `internal` | Unexpected runtime failure |

## Model health contracts

`ModelHealthResponse` wraps typed subsystem status:

- `RlModelStatus`
- `MlModelStatus`
- `RegimeModelStatus`
- `AgenticStatusSummary` (recent decisions + learning summary)

Disabled or missing subsystems return deterministic defaults, for example
`load_error="not_loaded"` for ML and `load_error="no_generator"` for RL.

Dashboard Streamlit panels consume `engine.model_status().to_dict()` for
backward-compatible dict access.

## Formatting boundary

Telegram text formatting lives in
[`src/fortuna/telegram/formatters.py`](../src/fortuna/telegram/formatters.py)
and reads only typed contract fields — not `SessionState` internals.

## Relationship to audit logs

Agentic audit logs continue to persist `AgentDecision.to_dict()` payloads.
`InstrumentAnalysisResponse` wrappers are not written to decision audit stores.

## Evaluation suite

Plan 03 adds a deterministic conversation eval corpus and harness. See
[conversation_eval_suite.md](conversation_eval_suite.md).

Plan 04 adds an explicit parse/route boundary. See
[conversation_boundary.md](conversation_boundary.md).
