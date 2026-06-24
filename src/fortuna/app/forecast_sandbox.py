"""Sandboxed forecast calculation helpers for conversational signal refinement."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fortuna.agentic.contracts import (
    ForecastTaskRequest,
    ForecastTaskResult,
    SandboxExecutionResult,
)
from fortuna.agentic.redaction import scrub_env
from fortuna.config.settings import Settings

_ALLOWED_IMPORTS = frozenset({"math", "statistics", "json"})
_FORBIDDEN_NAMES = frozenset(
    {
        "__import__",
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
    }
)


def build_generated_code(task_name: str) -> str:
    name = str(task_name or "").strip().lower()
    if name == "realized_volatility":
        return _code_lines(
            "import math",
            "returns = [",
            "    math.log(b / a)",
            "    for a, b in zip(payload['closes'][:-1], payload['closes'][1:])",
            "    if a > 0 and b > 0",
            "]",
            "if not returns:",
            "    result = {",
            "        'metrics': {'volatility': 0.0},",
            "        'summary': 'No return history.',",
            "        'reliability': 0.0,",
            "        'warnings': ['insufficient_history'],",
            "    }",
            "else:",
            "    mean = sum(returns) / len(returns)",
            "    variance = sum((x - mean) ** 2 for x in returns) / max(1, len(returns) - 1)",
            "    vol = math.sqrt(max(0.0, variance)) * math.sqrt(len(returns))",
            "    posture = 'elevated' if vol >= 0.03 else 'contained'",
            "    result = {",
            "        'metrics': {'volatility': round(vol, 6)},",
            "        'summary': f'Realized volatility is {posture}.',",
            "        'reliability': min(0.9, 0.35 + len(returns) / 40.0),",
            "        'warnings': [],",
            "    }",
        )
    if name == "trend_continuation":
        return _code_lines(
            "returns = [b - a for a, b in zip(payload['closes'][:-1], payload['closes'][1:])]",
            "if not returns:",
            "    result = {",
            "        'metrics': {'continuation_score': 0.0},",
            "        'summary': 'No trend history.',",
            "        'reliability': 0.0,",
            "        'warnings': ['insufficient_history'],",
            "    }",
            "else:",
            "    positive = sum(1 for x in returns if x > 0)",
            "    negative = sum(1 for x in returns if x < 0)",
            "    last_leg = returns[-1]",
            "    bias = positive / len(returns) if last_leg >= 0 else negative / len(returns)",
            "    direction = (",
            "        'upside follow-through'",
            "        if last_leg >= 0",
            "        else 'downside follow-through'",
            "    )",
            "    result = {",
            "        'metrics': {'continuation_score': round(bias, 6)},",
            "        'summary': f'Short-horizon {direction} bias is {bias:.0%}.',",
            "        'reliability': min(0.88, 0.4 + len(returns) / 50.0),",
            "        'warnings': [],",
            "    }",
        )
    if name == "support_resistance":
        return _code_lines(
            "window = (",
            "    payload['closes'][-20:]",
            "    if len(payload['closes']) >= 20",
            "    else payload['closes']",
            ")",
            "if not window:",
            "    result = {",
            "        'metrics': {'support': 0.0, 'resistance': 0.0},",
            "        'summary': 'No level history.',",
            "        'reliability': 0.0,",
            "        'warnings': ['insufficient_history'],",
            "    }",
            "else:",
            "    support = min(window)",
            "    resistance = max(window)",
            "    last_close = window[-1]",
            "    nearer = (",
            "        'support'",
            "        if abs(last_close - support) <= abs(resistance - last_close)",
            "        else 'resistance'",
            "    )",
            "    result = {",
            "        'metrics': {",
            "            'support': round(support, 4),",
            "            'resistance': round(resistance, 4),",
            "        },",
            "        'summary': f'Price is trading closer to {nearer}.',",
            "        'reliability': min(0.82, 0.42 + len(window) / 60.0),",
            "        'warnings': [],",
            "    }",
        )
    if name == "atr_posture":
        return _code_lines(
            "highs = payload['highs']",
            "lows = payload['lows']",
            "closes = payload['closes']",
            "trs = []",
            "for idx in range(1, min(len(closes), len(highs), len(lows))):",
            "    trs.append(",
            "        max(",
            "            highs[idx] - lows[idx],",
            "            abs(highs[idx] - closes[idx - 1]),",
            "            abs(lows[idx] - closes[idx - 1]),",
            "        )",
            "    )",
            "if not trs or not closes:",
            "    result = {",
            "        'metrics': {'atr_ratio': 0.0},",
            "        'summary': 'No ATR history.',",
            "        'reliability': 0.0,",
            "        'warnings': ['insufficient_history'],",
            "    }",
            "else:",
            "    atr = sum(trs[-14:]) / min(14, len(trs))",
            "    ratio = atr / closes[-1] if closes[-1] else 0.0",
            "    posture = 'expanding' if ratio >= 0.018 else 'compressed'",
            "    result = {",
            "        'metrics': {'atr_ratio': round(ratio, 6)},",
            "        'summary': f'Range posture is {posture}.',",
            "        'reliability': min(0.85, 0.38 + len(trs) / 50.0),",
            "        'warnings': [],",
            "    }",
        )
    raise ValueError(f"Unsupported forecast task {task_name!r}")


def validate_generated_code(code: str) -> str | None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax_error:{exc.msg}"
    if len(code.splitlines()) > 40:
        return "line_limit_exceeded"
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > 240:
            return "complexity_limit_exceeded"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _ALLOWED_IMPORTS:
                    return f"import_blocked:{alias.name}"
        if isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module not in _ALLOWED_IMPORTS:
                return f"import_from_blocked:{module}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return f"name_blocked:{node.id}"
        if isinstance(node, ast.Attribute) and str(getattr(node, "attr", "")).startswith("__"):
            return "dunder_attribute_blocked"
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                return f"call_blocked:{func.id}"
    return None


def execute_generated_code(
    request: ForecastTaskRequest,
    *,
    code: str,
    settings: Settings,
) -> tuple[SandboxExecutionResult, ForecastTaskResult | None]:
    validation_error = validate_generated_code(code)
    if validation_error is not None:
        return (
            SandboxExecutionResult(
                ok=False,
                mode="sandbox",
                generated_code="",
                validation_error=validation_error,
            ),
            None,
        )
    temp_root = settings.resolve_path(settings.signal_code_execution_temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    workdir = Path(os.path.abspath(tempfile.mkdtemp(prefix="forecast_", dir=str(temp_root))))
    script_path = workdir / "runner.py"
    payload_path = workdir / "payload.json"
    payload_path.write_text(json.dumps(request.to_dict()), encoding="utf-8")
    script_path.write_text(_runner_script(code), encoding="utf-8")
    env = scrub_env(dict(os.environ), preserve=("PATH", "SYSTEMROOT", "TEMP", "TMP", "PYTHONUTF8"))
    env["PYTHONUTF8"] = "1"
    timeout = max(1, int(getattr(settings, "signal_code_execution_timeout_seconds", 4) or 4))
    output_limit = max(
        128,
        int(getattr(settings, "signal_code_execution_max_output_chars", 4000) or 4000),
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path), str(payload_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return (
            SandboxExecutionResult(
                ok=False,
                mode="sandbox",
                runtime_ms=int((time.perf_counter() - start) * 1000),
                validation_error="timeout",
            ),
            None,
        )
    runtime_ms = int((time.perf_counter() - start) * 1000)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if len(stdout) > output_limit or len(stderr) > output_limit:
        return (
            SandboxExecutionResult(
                ok=False,
                mode="sandbox",
                stdout=stdout[:200],
                stderr=stderr[:200],
                runtime_ms=runtime_ms,
                validation_error="output_limit_exceeded",
            ),
            None,
        )
    if proc.returncode != 0:
        return (
            SandboxExecutionResult(
                ok=False,
                mode="sandbox",
                stdout=stdout[:200],
                stderr=stderr[:200],
                runtime_ms=runtime_ms,
                validation_error="runtime_error",
            ),
            None,
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return (
            SandboxExecutionResult(
                ok=False,
                mode="sandbox",
                stdout=stdout[:200],
                stderr=stderr[:200],
                runtime_ms=runtime_ms,
                validation_error="invalid_json",
            ),
            None,
        )
    result = ForecastTaskResult(
        task_name=request.task_name,
        ok=True,
        summary=str(payload.get("summary", "")).strip() or f"{request.task_name} complete.",
        metrics={str(k): float(v) for k, v in dict(payload.get("metrics") or {}).items()},
        warnings=tuple(str(item) for item in list(payload.get("warnings") or [])[:4]),
        reliability=float(payload.get("reliability", 0.0) or 0.0),
        mode="sandbox",
    )
    return (
        SandboxExecutionResult(
            ok=True,
            mode="sandbox",
            runtime_ms=runtime_ms,
        ),
        result,
    )


def cleanup_code_temp_dirs(path: Path, *, retention_minutes: int) -> int:
    if retention_minutes <= 0 or not path.exists():
        return 0
    removed = 0
    cutoff = time.time() - (retention_minutes * 60)
    for child in path.iterdir():
        try:
            if child.stat().st_mtime > cutoff:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed += 1
        except OSError:
            continue
    return removed


def supported_forecast_tasks() -> tuple[str, ...]:
    return ("trend_continuation", "support_resistance", "realized_volatility", "atr_posture")


def _runner_script(code: str) -> str:
    return (
        "import json\n"
        "import math\n"
        "import statistics\n"
        "import sys\n"
        "payload = json.loads(open(sys.argv[1], encoding='utf-8').read())\n"
        f"{code}\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )


def _code_lines(*lines: str) -> str:
    return "\n".join(lines) + "\n"
