# Fortuna Transfer Readme

This workspace snapshot was prepared on 2026-06-24 for moving Fortuna to a new PC.

## Workspace Snapshot

- Local path: `C:\dev\Fortuna`
- Branch: `Phase_2`
- Base commit: `da1cf61d0def7d82297525a0b3f1241532d435c6`
- Remote: `origin https://github.com/Ronderjohn/Fortuna.git`
- Python: `3.12.10`
- uv: `0.11.16`

## Read First

Open these in this order after moving:

1. `AGENTS.md`
2. `README.md`
3. `docs/how_to_use_fortuna.md`
4. `docs/operator_runbook.md`
5. `docs/current_system_flow.md`
6. `docs/first_run.md`

## Current Workstream Shape

The workspace is intentionally not a clean checkout. It contains a large in-progress advisory and Telegram-first expansion with both tracked and untracked changes.

Main areas currently in flight:

- Telegram assistant, routing, request audit, parsing, runtime, and formatting improvements.
- Agentic advisory support including OpenAI routing, chart analysis, redaction, and stronger contracts.
- Market-universe, shortlist, training-research, and workflow snapshot tooling.
- Portfolio allocation, nightly acceptance, runtime readiness, model activation, and operator workflow surfaces.
- Observability scaffolding and related tests.
- Documentation refresh across runbooks, architecture, and first-run guidance.

## Resume Checklist

1. Extract the archive to a writable path on the new PC.
2. Review `git status` before making changes so you know the workspace already has in-progress edits.
3. Recreate the environment with `uv sync`.
4. Confirm secrets in `.env` are still valid for the new machine and rotate them if needed.
5. Run focused checks before new edits:
   - `uv run pytest tests/agentic -q`
   - `uv run pytest tests/execution/test_session_wiring.py -q`
   - `uv run pytest tests/app -q`
   - `uv run ruff check src tests`
6. Break the current work into smaller commits after validating which features are stable enough to keep together.

## Practical Notes

- The archive is meant for continuity of editing and planning, not just source backup.
- `.env` is included in the workspace, so treat the archive as sensitive.
- The Python virtual environment is not required for continuity and may be excluded from the zip to keep the transfer smaller.
- Git history is useful for recovery, so keep the `.git` directory if it is present in the archive.

## Suggested Next Steps

- Start by validating Telegram and agentic focused tests.
- Then verify the new app-layer modules wire cleanly through `session_engine`.
- After that, choose one phase boundary and commit only a coherent slice instead of the whole working tree at once.
