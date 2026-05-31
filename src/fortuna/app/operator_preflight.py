"""Read-only operator preflight checks for advisory runtime readiness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fortuna.config.settings import Settings
from fortuna.models.metadata import ModelKind
from fortuna.models.registry import live_pointer_path


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass
class OperatorPreflightResult:
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def ok(self) -> bool:
        return not self.failures

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(PreflightCheck(name=name, status=status, detail=detail))

    def deterministic_only(self) -> bool:
        advisory = _find(self.checks, "agentic")
        ml = _find(self.checks, "ml_scorer")
        rl = _find(self.checks, "rl_policy")
        if advisory is None or advisory.status != "pass":
            return True
        return all(ch is None or ch.status != "pass" for ch in (ml, rl))


def run_operator_preflight(
    settings: Settings,
    *,
    symbol: Optional[str] = None,
    strict: bool = False,
) -> OperatorPreflightResult:
    result = OperatorPreflightResult()
    root = settings.project_root
    models_root = settings.resolve_path(Path("models"))
    ml_base = settings.resolve_path(settings.ml_scorer_artifact_dir)
    ml_root = ml_base.parent if ml_base.name == "validated" else ml_base
    sym = (symbol or settings.default_symbol or "").upper().strip()

    result.add("settings", "pass", f"loaded env={settings.env} config root={root}")
    log_dir = settings.resolve_path(settings.agentic_log_dir)
    if log_dir.parent.exists():
        result.add("agentic_log_dir", "pass", str(log_dir))
    else:
        result.add("agentic_log_dir", "warn", f"parent missing for {log_dir}")

    if settings.agentic_enabled:
        result.add("agentic", "pass", "agentic advisory enabled")
    else:
        result.add("agentic", "warn", "agentic advisory disabled; deterministic-only runtime")

    _check_smartapi(result)
    _check_ml(result, settings, models_root, ml_root)
    _check_rl(result, settings, models_root, sym)
    _check_telegram(result, settings)
    _check_registry(result, settings)

    if strict:
        for check in list(result.checks):
            if check.status != "warn":
                continue
            if check.name == "ml_scorer" and settings.agentic_ml_scorer_enabled:
                result.checks[result.checks.index(check)] = PreflightCheck(
                    name=check.name,
                    status="fail",
                    detail=check.detail,
                )
            elif check.name == "telegram" and settings.telegram_enabled:
                result.checks[result.checks.index(check)] = PreflightCheck(
                    name=check.name,
                    status="fail",
                    detail=check.detail,
                )
            elif check.name == "rl_policy" and settings.agentic_enabled:
                result.checks[result.checks.index(check)] = PreflightCheck(
                    name=check.name,
                    status="fail",
                    detail=check.detail,
                )
            elif (
                check.name == "smartapi_env"
                and getattr(settings, "data_source", "") == "smartapi"
            ):
                result.checks[result.checks.index(check)] = PreflightCheck(
                    name=check.name,
                    status="fail",
                    detail=check.detail,
                )
    return result


def render_preflight(result: OperatorPreflightResult) -> str:
    lines = []
    status = "PASS" if result.ok else "FAIL"
    lines.append(f"Fortuna operator preflight: {status}")
    for check in result.checks:
        badge = check.status.upper()
        lines.append(f"- [{badge}] {check.name}: {check.detail}")
    mode = "deterministic-only" if result.deterministic_only() else "ensemble/advisory"
    lines.append(f"- Runtime expectation: {mode}")
    if result.ok:
        lines.append("- For a deeper API probe, run: uv run python scripts/test_smartapi_env.py")
    return "\n".join(lines)


def _check_smartapi(result: OperatorPreflightResult) -> None:
    try:
        from fortuna.config.smartapi_settings import get_smartapi_settings

        cfg = get_smartapi_settings()
        if getattr(cfg, "configured", False):
            result.add("smartapi_env", "pass", "credentials present")
        else:
            result.add("smartapi_env", "warn", "credentials missing; cache/offline mode only")
    except Exception as exc:  # noqa: BLE001
        result.add("smartapi_env", "warn", f"settings unavailable: {exc}")


def _check_registry(result: OperatorPreflightResult, settings: Settings) -> None:
    if settings.model_registry_enabled:
        detail = "enabled"
        if settings.model_promotion_required:
            detail += ", promotion required"
        result.add("model_registry", "pass", detail)
    else:
        result.add("model_registry", "warn", "disabled; legacy/dev fallback loading allowed")


def _check_ml(
    result: OperatorPreflightResult,
    settings: Settings,
    models_root: Path,
    ml_root: Path,
) -> None:
    if not settings.agentic_ml_scorer_enabled:
        result.add("ml_scorer", "warn", "disabled")
        return
    ptr = live_pointer_path(ModelKind.ML_SCORER, models_root, ml_base=ml_root)
    if settings.model_registry_enabled and settings.model_promotion_required and not ptr.is_file():
        result.add("ml_scorer", "warn", f"promotion required but pointer missing: {ptr}")
        return
    if ptr.is_file():
        result.add("ml_scorer", "pass", f"live pointer present: {ptr}")
        return
    if settings.resolve_path(settings.ml_scorer_artifact_dir).is_dir():
        result.add(
            "ml_scorer",
            "pass",
            f"artifact dir available: {settings.ml_scorer_artifact_dir}",
        )
    else:
        result.add("ml_scorer", "warn", "artifact dir missing")


def _check_rl(
    result: OperatorPreflightResult,
    settings: Settings,
    models_root: Path,
    symbol: str,
) -> None:
    ptr = live_pointer_path(ModelKind.RL_POLICY, models_root, symbol=symbol)
    if settings.model_registry_enabled and settings.model_promotion_required and not ptr.is_file():
        result.add(
            "rl_policy",
            "warn",
            f"promotion required but pointer missing for {symbol}: {ptr}",
        )
        return
    if ptr.is_file():
        result.add("rl_policy", "pass", f"live pointer present for {symbol}: {ptr}")
        return
    if settings.rl_allow_global_policy:
        result.add("rl_policy", "warn", "per-symbol pointer missing; global fallback allowed")
    else:
        result.add("rl_policy", "warn", f"no per-symbol live pointer for {symbol}")


def _check_telegram(result: OperatorPreflightResult, settings: Settings) -> None:
    if not settings.telegram_enabled:
        result.add("telegram", "warn", "disabled")
        return
    required = [
        settings.telegram_bot_token,
        settings.telegram_chat_id,
    ]
    if all(bool(v) for v in required):
        result.add("telegram", "pass", "Telegram bot token and chat id complete")
    else:
        result.add("telegram", "warn", "enabled but Telegram settings are incomplete")


def _find(checks: list[PreflightCheck], name: str) -> Optional[PreflightCheck]:
    for check in checks:
        if check.name == name:
            return check
    return None
