"""Optional SSL-verification bypass for TLS-intercepting environments.

Why this exists: certain antivirus products (Avast, AVG, Kaspersky) and many
corporate proxies do **TLS interception** — they replace remote server
certificates with locally signed ones. When the local CA isn't visible in
Python's certifi bundle (or in any trust store ``ssl.enum_certificates``
exposes), every HTTPS call from Python fails with ``CERTIFICATE_VERIFY_FAILED``.

This module provides a single function ``maybe_disable_ssl_verification()``
that, when enabled, monkey-patches ``ssl``, ``requests`` and ``urllib3`` to
skip certificate verification. Call it as early as possible — before any
``requests.Session`` is built and before any ``WebSocketApp`` is opened.

**This is unsafe in general** — it allows MITM attacks. Only enable when:

1. You're on a personal machine where the antivirus already validates the
   cert chain itself (so verification is duplicated, not skipped), AND
2. You can't reasonably disable that antivirus' HTTPS scanning, OR
3. You're behind a managed corporate proxy whose root CA you cannot install.

Toggle: set ``FORTUNA_INSECURE_SSL=true`` in the environment (typically in
``.env``). Default is OFF.
"""

from __future__ import annotations

import os
import ssl
from typing import Any

from fortuna.utils.logging import get_logger

logger = get_logger(__name__)

_ENV_VAR = "FORTUNA_INSECURE_SSL"

_already_applied = False


def _is_enabled() -> bool:
    val = os.environ.get(_ENV_VAR, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


def maybe_disable_ssl_verification(*, force: bool = False) -> bool:
    """Apply the SSL bypass if the env var is set (or ``force=True``).

    Returns ``True`` if the bypass was applied (now or previously), ``False``
    if SSL verification is unchanged.
    """
    global _already_applied
    if _already_applied:
        return True
    if not (force or _is_enabled()):
        return False

    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

    try:
        import urllib3
        from urllib3.exceptions import InsecureRequestWarning

        urllib3.disable_warnings(InsecureRequestWarning)
    except Exception:
        pass

    try:
        import requests
        import requests.adapters as _adapters

        _orig_send = _adapters.HTTPAdapter.send

        def _patched_send(self: Any, request: Any, **kwargs: Any) -> Any:
            kwargs["verify"] = False
            return _orig_send(self, request, **kwargs)

        _adapters.HTTPAdapter.send = _patched_send  # type: ignore[method-assign]
        requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]
    except Exception:
        pass

    _already_applied = True
    logger.warning(
        "FORTUNA_INSECURE_SSL is ON — SSL verification disabled process-wide. "
        "This is acceptable when an antivirus (Avast/AVG/...) already validates "
        "the chain, but is unsafe on untrusted networks."
    )
    return True
