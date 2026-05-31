# Agentic Framework Readiness Plans

These plans break the "make Fortuna ready for a broader agentic framework"
effort into separate, reviewable work packets for Cursor.

They are intentionally about readiness and boundary-setting first, not about
introducing LangGraph/LangChain or another orchestration layer immediately.

Recommended execution order:

1. `01_typed_advisory_contracts.plan.md`
2. `02_agent_tools_surface.plan.md`
3. `03_conversation_eval_suite.plan.md`
4. `04_conversation_boundary_and_router.plan.md`
5. `05_traceability_and_request_audit.plan.md`
6. `06_telegram_interface_hardening.plan.md`
7. `07_framework_adoption_gate.plan.md`

Working rules for Cursor:

- stay in Plan mode first
- inspect the current implementation before proposing edits
- keep deterministic fallback intact
- do not introduce a framework runtime just because a plan mentions future
  framework readiness
- prefer typed Python interfaces and focused tests

Related docs:

- `AGENTS.md`
- `docs/agents.md`
- `docs/cursor_codebase_guide.md`
- `docs/current_system_flow.md`
- `docs/telegram_trading_assistant.md`
- [`docs/framework_adoption_gate.md`](../framework_adoption_gate.md) — adoption yes/no gate (Plan 07)

Plans 01–06 implemented the technical prerequisites. Plan 07 is decision-only:
no framework runtime is added to the repository.
