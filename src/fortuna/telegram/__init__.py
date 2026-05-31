"""Telegram bot helpers for interactive Fortuna analysis."""

__all__ = [
    "ClarificationCode",
    "ConversationRoute",
    "InteractionResult",
    "RouteAction",
    "TelegramAnalysisAssistant",
    "TelegramBotClient",
    "TelegramBotRuntime",
    "TelegramRequest",
    "TelegramRequestAuditEntry",
    "TelegramRequestAuditStore",
    "TelegramRequestKind",
    "build_audit_entry",
    "parse_telegram_request",
    "route_telegram_request",
]


def __getattr__(name: str):
    if name in {"InteractionResult", "TelegramAnalysisAssistant"}:
        from fortuna.telegram.assistant import InteractionResult, TelegramAnalysisAssistant

        return {
            "InteractionResult": InteractionResult,
            "TelegramAnalysisAssistant": TelegramAnalysisAssistant,
        }[name]
    if name == "TelegramBotClient":
        from fortuna.telegram.client import TelegramBotClient

        return TelegramBotClient
    if name == "TelegramBotRuntime":
        from fortuna.telegram.runtime import TelegramBotRuntime

        return TelegramBotRuntime
    if name in {"TelegramRequest", "TelegramRequestKind", "parse_telegram_request"}:
        from fortuna.telegram.parser import (
            TelegramRequest,
            TelegramRequestKind,
            parse_telegram_request,
        )

        return {
            "TelegramRequest": TelegramRequest,
            "TelegramRequestKind": TelegramRequestKind,
            "parse_telegram_request": parse_telegram_request,
        }[name]
    if name in {"TelegramRequestAuditEntry", "TelegramRequestAuditStore", "build_audit_entry"}:
        from fortuna.telegram.request_audit import (
            TelegramRequestAuditEntry,
            TelegramRequestAuditStore,
            build_audit_entry,
        )

        return {
            "TelegramRequestAuditEntry": TelegramRequestAuditEntry,
            "TelegramRequestAuditStore": TelegramRequestAuditStore,
            "build_audit_entry": build_audit_entry,
        }[name]
    if name in {"ClarificationCode", "ConversationRoute", "RouteAction", "route_telegram_request"}:
        from fortuna.telegram.router import (
            ClarificationCode,
            ConversationRoute,
            RouteAction,
            route_telegram_request,
        )

        return {
            "ClarificationCode": ClarificationCode,
            "ConversationRoute": ConversationRoute,
            "RouteAction": RouteAction,
            "route_telegram_request": route_telegram_request,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
