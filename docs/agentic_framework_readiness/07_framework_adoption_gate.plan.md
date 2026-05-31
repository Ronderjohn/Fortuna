# Plan 07: Framework Adoption Gate

## Objective

Define the explicit criteria Fortuna must satisfy before adding a broader
agentic framework such as LangGraph or LangChain.

This plan is a decision gate, not a commitment to adopt a framework.

## Why this matters

Without an adoption gate, frameworks tend to arrive because they look powerful,
not because the system has earned the complexity.

## In scope

- define readiness criteria
- define what problems a framework must actually solve
- define what remains out of scope even after readiness
- define a small proof-of-concept boundary if the gate is passed

## Out of scope

- implementing the framework
- migrating the typed advisory core into a framework runtime

## Deliverables

1. A written adoption gate with pass/fail criteria
2. A list of valid framework use cases for Fortuna
3. A list of anti-patterns and reasons to reject the integration
4. A narrow proof-of-concept target if the project decides to proceed

## Recommended gate criteria

Fortuna should not adopt a framework until at least these are true:

1. typed advisory/tool contracts are stable
2. Telegram and dashboard share a common analysis tool surface
3. there is a request evaluation suite
4. request tracing exists
5. the team has decided that command-bot behavior is no longer sufficient

## Candidate valid use cases

- conversational research assistant over existing tools
- multi-step clarification for user intent
- structured tool-calling planner above typed analysis tools
- prompt/version observability for an actual LLM layer

## Candidate invalid use cases

- replacing deterministic strategy logic
- replacing typed advisory orchestration
- hiding model-promotion rules behind prompt logic
- routing core decision logic through opaque chat completions

## Acceptance criteria

- there is a written yes/no gate for framework adoption
- the proof-of-concept boundary is narrow and reversible
- the typed advisory core remains explicitly out of scope for migration

## Tests

No primary code tests required unless this plan introduces helper docs or
configuration checks.

## Notes for Cursor

- this plan is about making a good decision, not about forcing a framework in
- if the gate is not met, the correct outcome is "not yet"
