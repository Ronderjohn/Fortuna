"""Parse Telegram user messages into structured Fortuna requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TelegramRequestKind(str, Enum):
    HELP = "help"
    SEARCH = "search"
    ANALYZE = "analyze"
    RISK = "risk"
    UNIVERSE = "universe"
    RESEARCH = "research"
    BRIEF = "brief"
    ALLOCATE = "allocate"
    CANDIDATES = "candidates"
    WORKFLOW = "workflow"
    HEALTH = "health"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TelegramRequest:
    kind: TelegramRequestKind
    raw_text: str
    query: str = ""
    symbol: str = ""
    timeframe: str = "5m"
    days: int = 30
    limit: int = 10
    source: str = "auto"
    target: str = "all"
    max_positions: int = 3
    max_per_exposure: int = 1
    max_same_side: int = 2


def parse_telegram_request(text: str) -> TelegramRequest:
    raw = (text or "").strip()
    if not raw:
        return TelegramRequest(TelegramRequestKind.HELP, raw_text=raw)
    lower = raw.lower()
    if lower in {"/start", "/help", "#help", "help"}:
        return TelegramRequest(TelegramRequestKind.HELP, raw_text=raw)

    for prefix in ("/search", "#search", "search"):
        if lower.startswith(prefix):
            query = raw[len(prefix):].strip()
            return TelegramRequest(TelegramRequestKind.SEARCH, raw_text=raw, query=query)

    for prefix in ("/analyze", "#analyze", "analyze"):
        if lower.startswith(prefix):
            return _parse_analyze(raw, raw[len(prefix):].strip())

    for prefix in ("/risk", "#risk", "risk"):
        if lower.startswith(prefix):
            return _parse_risk(raw, raw[len(prefix):].strip())

    for prefix in ("/universe", "#universe", "universe"):
        if lower.startswith(prefix):
            return _parse_universe(raw, raw[len(prefix):].strip())

    for prefix in ("/research", "#research", "research"):
        if lower.startswith(prefix):
            return _parse_research(raw, raw[len(prefix):].strip())

    for prefix in ("/brief", "#brief", "brief", "/shortlist", "#shortlist", "shortlist"):
        if lower.startswith(prefix):
            return _parse_brief(raw, raw[len(prefix):].strip())

    for prefix in ("/allocate", "#allocate", "allocate", "/portfolio", "#portfolio", "portfolio"):
        if lower.startswith(prefix):
            return _parse_allocate(raw, raw[len(prefix):].strip())

    for prefix in ("/workflow", "#workflow", "workflow", "/pipeline", "#pipeline", "pipeline"):
        if lower.startswith(prefix):
            return _parse_workflow(raw, raw[len(prefix):].strip())

    for prefix in (
        "/candidates",
        "#candidates",
        "candidates",
        "/training",
        "#training",
        "training",
    ):
        if lower.startswith(prefix):
            return _parse_candidates(raw, raw[len(prefix):].strip())

    for prefix in ("/health", "#health", "health"):
        if lower.startswith(prefix):
            return TelegramRequest(TelegramRequestKind.HEALTH, raw_text=raw)

    return TelegramRequest(TelegramRequestKind.UNKNOWN, raw_text=raw, query=raw)


def _parse_analyze(raw_text: str, body: str) -> TelegramRequest:
    tokens = [tok for tok in body.replace(",", " ").split() if tok]
    timeframe = "5m"
    days = 30
    cleaned: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in {"1m", "5m", "15m", "30m", "1h", "1d"}:
            timeframe = tok
            continue
        if low.startswith("days="):
            try:
                days = max(1, int(low.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if low.endswith("d") and low[:-1].isdigit():
            days = max(1, int(low[:-1]))
            continue
        cleaned.append(tok)

    symbol = _symbol_from_tokens(cleaned)
    return TelegramRequest(
        kind=TelegramRequestKind.ANALYZE,
        raw_text=raw_text,
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        query=" ".join(cleaned),
    )


def _parse_risk(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_analyze(raw_text, body)
    return TelegramRequest(
        kind=TelegramRequestKind.RISK,
        raw_text=parsed.raw_text,
        symbol=parsed.symbol,
        timeframe=parsed.timeframe,
        days=parsed.days,
        query=parsed.query,
    )


def _parse_universe(raw_text: str, body: str) -> TelegramRequest:
    tokens = [tok for tok in body.replace(",", " ").split() if tok]
    timeframe = "1d"
    days = 30
    limit = 10
    source = "auto"
    for tok in tokens:
        low = tok.lower()
        if low in {"1m", "5m", "15m", "30m", "1h", "1d"}:
            timeframe = tok
            continue
        if low in {"auto", "screener", "registry"}:
            source = low
            continue
        if low.startswith("days="):
            try:
                days = max(1, int(low.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if low.startswith("limit="):
            try:
                limit = max(1, int(low.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if low.endswith("d") and low[:-1].isdigit():
            days = max(1, int(low[:-1]))
            continue
        if low.endswith("n") and low[:-1].isdigit():
            limit = max(1, int(low[:-1]))
            continue
    return TelegramRequest(
        kind=TelegramRequestKind.UNIVERSE,
        raw_text=raw_text,
        timeframe=timeframe,
        days=days,
        limit=limit,
        source=source,
        query=body.strip(),
    )


def _parse_brief(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_universe(raw_text, body)
    return TelegramRequest(
        kind=TelegramRequestKind.BRIEF,
        raw_text=parsed.raw_text,
        timeframe=parsed.timeframe,
        days=parsed.days,
        limit=parsed.limit,
        source=parsed.source,
        query=parsed.query,
    )


def _parse_research(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_universe(raw_text, body)
    target = "all"
    for tok in [tok.lower() for tok in body.replace(",", " ").split() if tok]:
        if tok in {"ml", "rl", "all"}:
            target = tok
            break
    return TelegramRequest(
        kind=TelegramRequestKind.RESEARCH,
        raw_text=parsed.raw_text,
        timeframe=parsed.timeframe,
        days=parsed.days,
        limit=parsed.limit,
        source=parsed.source,
        target=target,
        query=parsed.query,
    )


def _parse_allocate(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_universe(raw_text, body)
    max_positions = 3
    max_per_exposure = 1
    max_same_side = 2
    for tok in [tok.lower() for tok in body.replace(",", " ").split() if tok]:
        if tok.startswith("maxpos="):
            try:
                max_positions = max(1, int(tok.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if tok.startswith("perexp="):
            try:
                max_per_exposure = max(1, int(tok.split("=", 1)[1]))
                continue
            except ValueError:
                pass
        if tok.startswith("sameside="):
            try:
                max_same_side = max(1, int(tok.split("=", 1)[1]))
                continue
            except ValueError:
                pass
    return TelegramRequest(
        kind=TelegramRequestKind.ALLOCATE,
        raw_text=parsed.raw_text,
        timeframe=parsed.timeframe,
        days=parsed.days,
        limit=parsed.limit,
        source=parsed.source,
        query=parsed.query,
        max_positions=max_positions,
        max_per_exposure=max_per_exposure,
        max_same_side=max_same_side,
    )


def _parse_candidates(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_universe(raw_text, body)
    target = "all"
    for tok in [tok.lower() for tok in body.replace(",", " ").split() if tok]:
        if tok in {"ml", "rl", "all"}:
            target = tok
            break
    return TelegramRequest(
        kind=TelegramRequestKind.CANDIDATES,
        raw_text=parsed.raw_text,
        timeframe=parsed.timeframe,
        days=parsed.days,
        limit=parsed.limit,
        source=parsed.source,
        target=target,
        query=parsed.query,
    )


def _parse_workflow(raw_text: str, body: str) -> TelegramRequest:
    parsed = _parse_universe(raw_text, body)
    return TelegramRequest(
        kind=TelegramRequestKind.WORKFLOW,
        raw_text=parsed.raw_text,
        timeframe=parsed.timeframe,
        days=parsed.days,
        limit=parsed.limit,
        source=parsed.source,
        query=parsed.query,
    )


def _symbol_from_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    joined = ".".join(tokens).upper()
    if ".OPT." in joined or ".FUT" in joined or joined.endswith(".NS"):
        return joined

    upper = [tok.upper() for tok in tokens]
    if len(upper) >= 4 and upper[1] in {"CE", "PE", "C", "P"}:
        expiry = _normalize_expiry(upper[3])
        opt_type = "CE" if upper[1] in {"CE", "C"} else "PE"
        return f"{upper[0]}.OPT.{opt_type}.{upper[2]}.{expiry}"
    if len(upper) >= 2 and upper[1] == "FUT":
        if len(upper) >= 3:
            return f"{upper[0]}.FUT.{_normalize_expiry(upper[2])}"
        return f"{upper[0]}.FUT"
    return upper[0]


def _normalize_expiry(raw: str) -> str:
    text = raw.upper().strip()
    for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%d%b%Y").upper()
        except ValueError:
            continue
    return text
