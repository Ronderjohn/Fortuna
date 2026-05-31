"""Read-only checks for pre-framework technical readiness (Plans 01–04 artifacts)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

from fortuna.agentic.contracts import (
    InstrumentAnalysisRequest,
    InstrumentAnalysisResponse,
    InstrumentSearchResponse,
    ModelHealthResponse,
)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class FrameworkReadinessResult:
    checks: list[ReadinessCheck] = field(default_factory=list)

    @property
    def technical_ok(self) -> bool:
        technical = [c for c in self.checks if c.name != "product_decision_manual"]
        return bool(technical) and all(check.passed for check in technical)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(ReadinessCheck(name=name, passed=passed, detail=detail))


def run_framework_readiness_checks() -> FrameworkReadinessResult:
    result = FrameworkReadinessResult()

    _check_import(result, "agentic_contracts", "fortuna.agentic.contracts")
    _check_import(result, "advisory_tools", "fortuna.agentic.tools")
    _check_import(result, "conversation_router", "fortuna.telegram.router")
    _check_import(result, "request_audit", "fortuna.telegram.request_audit")

    try:
        eval_suite = importlib.import_module("fortuna.telegram.eval_suite")
        fixture_path = eval_suite.default_eval_fixture_path()
        exists = fixture_path.is_file()
        result.add(
            "eval_fixture",
            exists,
            str(fixture_path) if exists else f"missing fixture: {fixture_path}",
        )
    except Exception as exc:  # noqa: BLE001
        result.add("eval_fixture", False, f"eval suite import failed: {exc}")

    _check_callable_attr(
        result,
        "route_telegram_request",
        "fortuna.telegram.router",
        "route_telegram_request",
    )
    _check_callable_attr(
        result,
        "audit_store_factory",
        "fortuna.telegram.request_audit",
        "TelegramRequestAuditStore",
    )
    _check_contract_shape(result)
    _check_tool_surface(result)
    _check_session_engine_surface(result)
    _check_eval_cases(result)

    result.add(
        "product_decision_manual",
        False,
        "criterion 5 requires documented team sign-off (see docs/framework_adoption_gate.md)",
    )

    return result


def render_framework_readiness(result: FrameworkReadinessResult) -> str:
    lines = ["Framework technical readiness (criteria 1–4):"]
    for check in result.checks:
        if check.name == "product_decision_manual":
            continue
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{status}] {check.name}: {check.detail}")

    lines.append("")
    lines.append("Manual gate (criterion 5):")
    manual = next((c for c in result.checks if c.name == "product_decision_manual"), None)
    if manual is not None:
        lines.append(f"  [MANUAL] {manual.detail}")

    lines.append("")
    if result.technical_ok:
        lines.append("Technical criteria: PASS")
        lines.append("Adoption outcome: not_yet (until product sign-off)")
    else:
        lines.append("Technical criteria: FAIL")
        lines.append("Adoption outcome: not_yet (fix failing checks first)")
    return "\n".join(lines)


def _check_import(result: FrameworkReadinessResult, name: str, module_path: str) -> None:
    try:
        importlib.import_module(module_path)
        result.add(name, True, module_path)
    except Exception as exc:  # noqa: BLE001
        result.add(name, False, f"{module_path}: {exc}")


def _check_callable_attr(
    result: FrameworkReadinessResult,
    name: str,
    module_path: str,
    attr: str,
) -> None:
    try:
        module = importlib.import_module(module_path)
        obj = getattr(module, attr, None)
        ok = obj is not None
        result.add(name, ok, f"{module_path}.{attr}" if ok else f"missing {module_path}.{attr}")
    except Exception as exc:  # noqa: BLE001
        result.add(name, False, f"{module_path}.{attr}: {exc}")


def _check_contract_shape(result: FrameworkReadinessResult) -> None:
    try:
        request = InstrumentAnalysisRequest(symbol="RELIANCE.NS")
        response = InstrumentAnalysisResponse(ok=True, request=request)
        search = InstrumentSearchResponse(ok=True, query="RELIANCE")
        health_fields = set(ModelHealthResponse.__annotations__)
        ok = (
            request.timeframe == "5m"
            and isinstance(response.to_dict(), dict)
            and isinstance(search.to_dict(), dict)
            and {"registry_enabled", "promotion_required", "rl", "ml", "regime", "agentic"}
            <= health_fields
        )
        detail = "analysis/search/model health contracts serialize cleanly"
        result.add("contract_shape", ok, detail if ok else "contract shape validation failed")
    except Exception as exc:  # noqa: BLE001
        result.add("contract_shape", False, str(exc))


def _check_tool_surface(result: FrameworkReadinessResult) -> None:
    try:
        module = importlib.import_module("fortuna.agentic.tools")
        cls = getattr(module, "FortunaAdvisoryTools", None)
        required = {
            "search_instruments",
            "analyze_instrument",
            "get_model_health",
            "get_recent_learning_summary",
        }
        ok = cls is not None and required <= set(dir(cls))
        detail = "FortunaAdvisoryTools exposes search/analyze/health/learning"
        result.add("tool_surface", ok, detail if ok else "missing advisory tool methods")
    except Exception as exc:  # noqa: BLE001
        result.add("tool_surface", False, str(exc))


def _check_session_engine_surface(result: FrameworkReadinessResult) -> None:
    try:
        module = importlib.import_module("fortuna.app.session_engine")
        cls = getattr(module, "FortunaSessionEngine", None)
        ok = cls is not None and {
            "advisory_tools",
            "model_status",
            "analyze_instrument",
        } <= set(dir(cls))
        detail = "session engine exposes shared advisory tool entrypoints"
        result.add(
            "session_engine_surface",
            ok,
            detail if ok else "missing advisory surface on FortunaSessionEngine",
        )
    except Exception as exc:  # noqa: BLE001
        result.add("session_engine_surface", False, str(exc))


def _check_eval_cases(result: FrameworkReadinessResult) -> None:
    try:
        eval_suite = importlib.import_module("fortuna.telegram.eval_suite")
        fixture_path = eval_suite.default_eval_fixture_path()
        cases = eval_suite.load_eval_cases(fixture_path)
        layers = {case.layer for case in cases}
        expected_layers = {"parse", "route", "tool", "assistant"}
        ok = bool(cases) and expected_layers <= layers
        detail = (
            f"{len(cases)} eval cases loaded from {Path(fixture_path).name}; "
            f"layers={','.join(sorted(layers))}"
        )
        result.add("eval_cases", ok, detail if ok else "missing eval layers or empty corpus")
    except Exception as exc:  # noqa: BLE001
        result.add("eval_cases", False, str(exc))
