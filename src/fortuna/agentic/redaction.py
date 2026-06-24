"""Helpers for redacting secret-like content from interactive inputs and logs."""

from __future__ import annotations

import re
from typing import Iterable

_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|rk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9]{24,}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:bearer|token|secret|password|totp|api[_-]?key)\s*[:=]\s*\S+\b", re.I),
    re.compile(r"\b(?:SMARTAPI_[A-Z_]+|FORTUNA_[A-Z_]+)\s*=\s*\S+\b"),
)

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+previous\s+instructions", re.I),
    re.compile(r"reveal\s+(?:the\s+)?system\s+prompt", re.I),
    re.compile(r"show\s+(?:me\s+)?(?:your\s+)?secrets", re.I),
    re.compile(r"print\s+the\s+env", re.I),
    re.compile(r"bypass\s+the\s+guardrails", re.I),
)


def redact_secrets(text: str, *, enabled: bool = True) -> str:
    if not enabled:
        return str(text or "")
    redacted = str(text or "")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def contains_secret_like_content(text: str) -> bool:
    payload = str(text or "")
    return any(pattern.search(payload) for pattern in _SECRET_PATTERNS)


def looks_like_prompt_injection(text: str) -> bool:
    payload = str(text or "")
    return any(pattern.search(payload) for pattern in _INJECTION_PATTERNS)


def scrub_env(env: dict[str, str], *, preserve: Iterable[str] = ()) -> dict[str, str]:
    keep = {item for item in preserve}
    blocked = (
        "OPENAI",
        "SMARTAPI",
        "FORTUNA_TELEGRAM",
        "TOTP",
        "PASSWORD",
        "TOKEN",
        "SECRET",
        "API_KEY",
    )
    clean: dict[str, str] = {}
    for key, value in env.items():
        upper = str(key).upper()
        if key in keep:
            clean[key] = value
            continue
        if any(marker in upper for marker in blocked):
            continue
        clean[key] = value
    return clean
