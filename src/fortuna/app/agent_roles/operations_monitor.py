"""Operations monitor role wrapper."""

from __future__ import annotations

from fortuna.agentic.contracts import AgentRoleStatus
from fortuna.app.model_status import build_nightly_alignment_status


def run(*, settings):
    response = build_nightly_alignment_status(settings)
    return response, compose_role(response)


def compose_role(nightly_alignment) -> AgentRoleStatus:
    if nightly_alignment is None or int(getattr(nightly_alignment, "report_count", 0) or 0) <= 0:
        return AgentRoleStatus(
            name="operations_monitor",
            ok=False,
            headline="No recent nightly alignment evidence yet",
        )
    enabled_reports = int(getattr(nightly_alignment, "enabled_reports", 0) or 0)
    aligned_reports = int(getattr(nightly_alignment, "aligned_reports", 0) or 0)
    ratio = (
        float(aligned_reports) / float(enabled_reports)
        if enabled_reports > 0
        else 0.0
    )
    notes: list[str] = []
    latest_mix = str(getattr(nightly_alignment, "latest_target_mix", "") or "").strip()
    latest_refreshed = str(
        getattr(nightly_alignment, "latest_refreshed_target_mix", "") or ""
    ).strip()
    recommended_action = str(getattr(nightly_alignment, "recommended_action", "") or "").strip()
    recommended_discovery_action = str(
        getattr(nightly_alignment, "recommended_discovery_action", "") or ""
    ).strip()
    recommended_discovery_command = str(
        getattr(nightly_alignment, "recommended_discovery_cli_command", "") or ""
    ).strip()
    recent_rows = tuple(getattr(nightly_alignment, "recent_rows", ()) or ())
    recent_window = recent_rows[:3]
    if latest_mix:
        notes.append(f"latest={latest_mix}")
    if latest_refreshed:
        notes.append(f"refreshed={latest_refreshed}")
    recent_trend_window = int(getattr(nightly_alignment, "recent_trend_window", 0) or 0)
    recent_trend_enabled = int(getattr(nightly_alignment, "recent_trend_enabled", 0) or 0)
    recent_trend_aligned = int(getattr(nightly_alignment, "recent_trend_aligned", 0) or 0)
    recent_latest_status = str(
        getattr(nightly_alignment, "recent_trend_latest_status", "") or ""
    ).strip()
    recent_latest_basket = int(
        getattr(nightly_alignment, "recent_trend_latest_basket_size", 0) or 0
    )
    if recent_trend_window > 0:
        notes.append(f"recent={recent_trend_aligned}/{recent_trend_enabled}")
        if recent_latest_status or recent_latest_basket:
            notes.append(f"latest_run={recent_latest_status or '—'}:{recent_latest_basket}")
    elif recent_window:
        recent_enabled = sum(
            1 for row in recent_window if getattr(row, "alignment_enabled", False)
        )
        recent_aligned = sum(
            1
            for row in recent_window
            if getattr(row, "alignment_enabled", False)
            and str(getattr(row, "target_mix", "") or "").strip()
        )
        notes.append(f"recent={recent_aligned}/{recent_enabled}")
        latest_row = recent_window[0]
        latest_status = str(getattr(latest_row, "overall_status", "") or "").strip()
        latest_basket = int(getattr(latest_row, "basket_size", 0) or 0)
        if latest_status or latest_basket:
            notes.append(f"latest_run={latest_status or '—'}:{latest_basket}")
    if recommended_action:
        notes.append(recommended_action)
    if getattr(nightly_alignment, "recommended_refresh_target", None):
        notes.append(
            "refresh_target="
            f"{str(getattr(nightly_alignment, 'recommended_refresh_target', '') or '').strip()}"
        )
    if bool(getattr(nightly_alignment, "recommended_force_refresh", False)):
        notes.append("force_refresh=1")
    if recommended_discovery_action:
        notes.append("discovery_follow_up=1")
    if recommended_discovery_command:
        notes.append("discovery_command=1")
    return AgentRoleStatus(
        name="operations_monitor",
        ok=True,
        headline=(
            f"{int(getattr(nightly_alignment, 'report_count', 0) or 0)} nightly reports "
            f"tracked; alignment ratio={ratio:.2f}"
            + ("; discovery follow-up suggested" if recommended_discovery_action else "")
        ),
        notes=tuple(notes[:7]),
    )
