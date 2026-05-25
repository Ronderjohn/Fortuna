"""Local HTTP bar-stream server for the TradingView-style chart iframe.

The chart's JavaScript polls this server every couple of seconds and applies
the returned bars / markers via Lightweight Charts' ``series.update()`` API.
Because the iframe is never reloaded the user's zoom / pan state is preserved
across updates — the streaming experience matches TradingView's native
behavior.

This module runs an entirely stdlib-based daemon HTTP server, so there are no
new dependencies and we never block the Streamlit main thread.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)


def _find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


class _BarHandler(BaseHTTPRequestHandler):
    """Per-request handler. Subclassed per-server so we can inject a provider."""

    _provider: Optional[Callable[..., dict[str, Any]]] = None

    def log_message(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        # Silence noisy default access logs — they spam stdout every poll.
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store")

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            parts = urlparse(self.path)
            if parts.path == "/fortuna/health":
                self._send_json({"ok": True})
                return
            if parts.path == "/fortuna/bars":
                params = {k: v[0] for k, v in parse_qs(parts.query).items()}
                provider = type(self)._provider
                if provider is None:
                    self._send_json({"error": "no provider"}, 503)
                    return
                try:
                    since = float(params.get("since", "0") or 0)
                except ValueError:
                    since = 0.0
                payload = provider(
                    symbol=params.get("symbol", ""),
                    timeframe=params.get("tf", ""),
                    strategy=params.get("strat", ""),
                    since=since,
                )
                self._send_json(payload)
                return
            self._send_json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            logger.exception("stream server error: %s", e)
            self._send_json({"error": str(e)}, 500)


_RUNNING: dict[int, ThreadingHTTPServer] = {}
_LOCK = threading.Lock()


def spawn(
    provider: Callable[..., dict[str, Any]], port: Optional[int] = None
) -> int:
    """Start a daemon HTTP server bound to localhost.

    Returns the actual bound port. Pass ``port=0`` (or omit) to auto-pick a
    free one; pass a fixed port to make it stable across reruns.
    """
    with _LOCK:
        if not port or port <= 0:
            port = _find_free_port()
        if port in _RUNNING:
            # Already running; update its provider so live state stays current.
            handler_cls = _RUNNING[port].RequestHandlerClass
            setattr(handler_cls, "_provider", staticmethod(provider))
            return port

        handler_cls = type(
            "_BoundBarHandler",
            (_BarHandler,),
            {"_provider": staticmethod(provider)},
        )
        server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
        t = threading.Thread(
            target=server.serve_forever,
            name="fortuna-bar-stream",
            daemon=True,
        )
        t.start()
        _RUNNING[port] = server
        logger.info("Fortuna bar stream server started on port %d", port)
        return port


def stop(port: int) -> None:
    """Shut down a running server (used in tests)."""
    with _LOCK:
        server = _RUNNING.pop(port, None)
        if server is not None:
            server.shutdown()
            server.server_close()
