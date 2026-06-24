"""Lightweight formatting helpers for nightly alignment UI surfaces."""

from __future__ import annotations


def format_latest_nightly_execution_posture(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    target = str(summary.get("latest_execution_target") or "").strip()
    family = str(summary.get("latest_execution_model_family") or "").strip()
    source = str(summary.get("latest_execution_selection_source") or "").strip()
    review = str(summary.get("latest_promotion_review_model_kind") or "").strip()
    raw_review_kinds = summary.get("latest_promotion_review_model_kinds")
    review_kinds: list[str] = []
    if isinstance(raw_review_kinds, (list, tuple)):
        review_kinds = [
            str(kind or "").strip()
            for kind in raw_review_kinds
            if str(kind or "").strip()
        ]
    elif isinstance(raw_review_kinds, str):
        review_kinds = [kind.strip() for kind in raw_review_kinds.split(",") if kind.strip()]
    if not any((target, family, source, review, review_kinds)):
        return ""
    parts: list[str] = []
    if target:
        parts.append(f"target `{target}`")
    if family:
        parts.append(f"family `{family}`")
    if source:
        parts.append(f"source `{source}`")
    if review_kinds:
        parts.append(f"reviews `{', '.join(review_kinds)}`")
    elif review:
        parts.append(f"review `{review}`")
    return "Latest nightly execution: " + " | ".join(parts)


def format_latest_nightly_discovery_context(summary: dict) -> str:
    if not isinstance(summary, dict):
        return ""
    discovery_summary = str(
        summary.get("latest_training_research_discovery_summary") or ""
    ).strip()
    discovery_target = str(
        summary.get("latest_training_research_discovery_recommended_target") or ""
    ).strip()
    if not discovery_summary and not discovery_target:
        return ""
    parts: list[str] = []
    if discovery_summary:
        parts.append(discovery_summary)
    if discovery_target:
        parts.append(f"target `{discovery_target}`")
    return "Latest discovery context: " + " | ".join(parts)
