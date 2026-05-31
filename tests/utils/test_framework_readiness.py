from __future__ import annotations

from fortuna.utils.framework_readiness import (
    render_framework_readiness,
    run_framework_readiness_checks,
)


def test_framework_readiness_technical_checks_cover_real_surfaces():
    result = run_framework_readiness_checks()
    names = {check.name for check in result.checks}
    assert "contract_shape" in names
    assert "tool_surface" in names
    assert "session_engine_surface" in names
    assert "eval_cases" in names
    assert "product_decision_manual" in names
    technical_checks = [c for c in result.checks if c.name != "product_decision_manual"]
    assert all(check.passed for check in technical_checks)
    assert result.technical_ok is True


def test_framework_readiness_render_mentions_manual_gate():
    result = run_framework_readiness_checks()
    text = render_framework_readiness(result)
    assert "Technical criteria:" in text
    assert "Manual gate" in text
    assert "product sign-off" in text
    assert "not_yet" in text
