"""Optional OpenAI-backed planner above Fortuna's typed advisory tools."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fortuna.config.settings import Settings
from fortuna.telegram.parser import TelegramRequest, TelegramRequestKind
from fortuna.telegram.router import ConversationRoute, route_telegram_request
from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

OpenAITransport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]

_SYSTEM_PROMPT = (
    "You are a planner for Fortuna, an advisory-first trading analysis system. "
    "Choose exactly one tool call that best maps the user request into Fortuna's typed interface. "
    "Prefer search when the user is discovering symbols, analyze when they want a trading view on "
    "one instrument, universe when they want liquid or active stocks, shortlist briefing when they "
    "want top setups or a watchlist summary, portfolio allocation when they want "
    "to choose which setups should be carried together under simple portfolio "
    "limits, training candidates when they want names for ML/RL data refresh or "
    "model prep, workflow summary when they want the full multi-stage pipeline, "
    "health when they ask about model/runtime state, and help when they ask for commands. Do not "
    "invent symbols."
)


@dataclass(frozen=True)
class OpenAIPlan:
    request: TelegramRequest
    route: ConversationRoute
    tool_name: str
    confidence: float
    rationale: str


class OpenAIPlannerError(RuntimeError):
    pass


class OpenAIConversationPlanner:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: Optional[OpenAITransport] = None,
    ) -> None:
        self.settings = settings
        self._transport = transport or _default_transport

    def available(self) -> bool:
        return bool(getattr(self.settings, "openai_api_key", "").strip())

    def plan(self, text: str) -> OpenAIPlan:
        if not self.available():
            raise OpenAIPlannerError("OpenAI API key is not configured")
        payload = {
            "model": self.settings.openai_model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": _SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            ],
            "tools": _tool_schemas(),
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_output_tokens": int(getattr(self.settings, "openai_max_output_tokens", 250)),
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        raw = self._transport(
            f"{self.settings.openai_base_url.rstrip('/')}/responses",
            payload,
            headers,
            float(getattr(self.settings, "openai_timeout_seconds", 30)),
        )
        tool_name, arguments = _extract_tool_call(raw)
        request = _request_from_tool(tool_name, arguments, text)
        route = route_telegram_request(request)
        confidence = _extract_confidence(raw)
        rationale = _extract_rationale(raw, tool_name)
        return OpenAIPlan(
            request=request,
            route=route,
            tool_name=tool_name,
            confidence=confidence,
            rationale=rationale,
        )


def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise OpenAIPlannerError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIPlannerError(str(exc.reason)) from exc


def _tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "search_instruments",
            "description": (
                "Search symbols or contracts when the user is discovering an instrument."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "analyze_instrument",
            "description": "Analyze one resolved instrument for advisory output.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["symbol", "timeframe", "days"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_market_universe",
            "description": "Rank liquid or active stocks for research or watchlist preparation.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "source": {"type": "string"},
                },
                "required": ["timeframe", "days", "limit", "source"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_shortlist_briefing",
            "description": (
                "Summarize the top shortlisted setups with light "
                "portfolio/exposure notes."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "source": {"type": "string"},
                },
                "required": ["timeframe", "days", "limit", "source"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_portfolio_allocation",
            "description": (
                "Select which shortlisted setups should be carried together "
                "under simple portfolio and exposure constraints."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "source": {"type": "string"},
                    "max_positions": {"type": "integer"},
                    "max_per_exposure": {"type": "integer"},
                    "max_same_side": {"type": "integer"},
                },
                "required": [
                    "timeframe",
                    "days",
                    "limit",
                    "source",
                    "max_positions",
                    "max_per_exposure",
                    "max_same_side",
                ],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_training_candidates",
            "description": (
                "Select shortlist-driven ML or RL training candidates for "
                "data refresh and model preparation."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["timeframe", "days", "limit", "source", "target"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_workflow_summary",
            "description": (
                "Summarize the full liquid-universe to shortlist to briefing to "
                "training-candidate workflow."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {"type": "string"},
                    "days": {"type": "integer"},
                    "limit": {"type": "integer"},
                    "source": {"type": "string"},
                },
                "required": ["timeframe", "days", "limit", "source"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "get_model_health",
            "description": "Show current model and advisory runtime health.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "show_help",
            "description": "Show bot usage and command help.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    ]


def _extract_tool_call(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for item in payload.get("output", []):
        if str(item.get("type", "")).lower() in {"function_call", "tool_call"}:
            name = str(item.get("name") or item.get("tool_name") or "").strip()
            args_raw = item.get("arguments") or "{}"
            if not name:
                continue
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
            return name, arguments
    raise OpenAIPlannerError("No tool call returned by OpenAI planner")


def _extract_confidence(payload: dict[str, Any]) -> float:
    usage = payload.get("usage") or {}
    return 0.85 if usage else 0.8


def _extract_rationale(payload: dict[str, Any], tool_name: str) -> str:
    response_id = payload.get("id")
    if response_id:
        return f"OpenAI planner selected {tool_name} (response {response_id})"
    return f"OpenAI planner selected {tool_name}"


def _request_from_tool(tool_name: str, arguments: dict[str, Any], raw_text: str) -> TelegramRequest:
    name = str(tool_name or "").strip()
    if name == "search_instruments":
        return TelegramRequest(
            TelegramRequestKind.SEARCH,
            raw_text=raw_text,
            query=str(arguments.get("query", "")).strip(),
        )
    if name == "analyze_instrument":
        return TelegramRequest(
            TelegramRequestKind.ANALYZE,
            raw_text=raw_text,
            symbol=str(arguments.get("symbol", "")).strip(),
            query=str(arguments.get("symbol", "")).strip(),
            timeframe=str(arguments.get("timeframe", "5m")).strip() or "5m",
            days=max(1, int(arguments.get("days", 30))),
        )
    if name == "get_market_universe":
        return TelegramRequest(
            TelegramRequestKind.UNIVERSE,
            raw_text=raw_text,
            timeframe=str(arguments.get("timeframe", "1d")).strip() or "1d",
            days=max(1, int(arguments.get("days", 30))),
            limit=max(1, int(arguments.get("limit", 10))),
            source=str(arguments.get("source", "auto")).strip() or "auto",
        )
    if name == "get_shortlist_briefing":
        return TelegramRequest(
            TelegramRequestKind.BRIEF,
            raw_text=raw_text,
            timeframe=str(arguments.get("timeframe", "5m")).strip() or "5m",
            days=max(1, int(arguments.get("days", 30))),
            limit=max(1, int(arguments.get("limit", 5))),
            source=str(arguments.get("source", "auto")).strip() or "auto",
        )
    if name == "get_training_candidates":
        return TelegramRequest(
            TelegramRequestKind.CANDIDATES,
            raw_text=raw_text,
            timeframe=str(arguments.get("timeframe", "5m")).strip() or "5m",
            days=max(1, int(arguments.get("days", 30))),
            limit=max(1, int(arguments.get("limit", 8))),
            source=str(arguments.get("source", "auto")).strip() or "auto",
            target=str(arguments.get("target", "all")).strip() or "all",
        )
    if name == "get_portfolio_allocation":
        return TelegramRequest(
            TelegramRequestKind.ALLOCATE,
            raw_text=raw_text,
            timeframe=str(arguments.get("timeframe", "5m")).strip() or "5m",
            days=max(1, int(arguments.get("days", 30))),
            limit=max(1, int(arguments.get("limit", 5))),
            source=str(arguments.get("source", "auto")).strip() or "auto",
            max_positions=max(1, int(arguments.get("max_positions", 3))),
            max_per_exposure=max(1, int(arguments.get("max_per_exposure", 1))),
            max_same_side=max(1, int(arguments.get("max_same_side", 2))),
        )
    if name == "get_workflow_summary":
        return TelegramRequest(
            TelegramRequestKind.WORKFLOW,
            raw_text=raw_text,
            timeframe=str(arguments.get("timeframe", "5m")).strip() or "5m",
            days=max(1, int(arguments.get("days", 30))),
            limit=max(1, int(arguments.get("limit", 8))),
            source=str(arguments.get("source", "auto")).strip() or "auto",
        )
    if name == "get_model_health":
        return TelegramRequest(TelegramRequestKind.HEALTH, raw_text=raw_text)
    if name == "show_help":
        return TelegramRequest(TelegramRequestKind.HELP, raw_text=raw_text)
    raise OpenAIPlannerError(f"Unsupported tool returned by OpenAI planner: {tool_name}")
