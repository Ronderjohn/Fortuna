"""Parse Telegram user messages into structured Fortuna requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class TelegramRequestKind(str, Enum):
    HELP = "help"
    SEARCH = "search"
    ANALYZE = "analyze"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TelegramRequest:
    kind: TelegramRequestKind
    raw_text: str
    query: str = ""
    symbol: str = ""
    timeframe: str = "5m"
    days: int = 30


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
