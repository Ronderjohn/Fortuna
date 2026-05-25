"""Append Windows-trusted Avast (and other system) root CAs to certifi's bundle.

Why this exists: Avast Antivirus (and AVG / Kaspersky / many corp proxies) does
TLS interception — every HTTPS server's certificate is replaced with one
locally signed by an "Avast Web/Mail Shield Root" CA that Windows trusts but
Python's ``certifi`` bundle does not. Result: every ``requests``-based HTTPS
call from Python fails with "unable to get local issuer certificate".

This script reads the Windows ROOT and CA stores via ``ssl.enum_certificates``,
filters down to roots that match a small allowlist of TLS-intercepting
products (or ``--all`` to add every Windows root), converts each DER cert to
PEM, and appends the ones not already present to ``certifi.where()``.

Run once. After this, ``requests``/``urllib3``/``websocket-client``/``uv``
will trust those server certs and SmartAPI / PyPI calls will work normally.
"""

from __future__ import annotations

import argparse
import ssl
from pathlib import Path

import certifi


_ALLOWLIST_KEYWORDS = (
    "avast",
    "avg",
    "kaspersky",
    "bitdefender",
    "eset",
    "norton",
    "mcafee",
    "trend micro",
    "fortinet",
    "zscaler",
    "bluecoat",
    "blue coat",
    "cisco umbrella",
    "palo alto",
    "fiddler",
    "charles",
    "burp",
    "zap",
)


def _extract_subject_text(pem: str) -> str:
    """Best-effort 'Subject' string for the PEM (uses Python's ssl decoder)."""
    try:
        info = ssl._ssl._test_decode_cert_pem(pem)  # type: ignore[attr-defined]
    except Exception:
        return ""
    parts: list[str] = []
    for rdn in info.get("subject", ()):
        for k, v in rdn:
            parts.append(f"{k}={v}")
    return " / ".join(parts)


def _gather_windows_roots(include_all: bool) -> list[tuple[str, str]]:
    """Return (subject_text, pem) for every relevant Windows root cert."""
    out: list[tuple[str, str]] = []
    for store in ("ROOT", "CA"):
        try:
            entries = ssl.enum_certificates(store)
        except Exception:
            continue
        for der, encoding, trust in entries:
            if encoding != "x509_asn":
                continue
            if trust is True or (isinstance(trust, set) and trust):
                pem = ssl.DER_cert_to_PEM_cert(der)
                subject = _extract_subject_text(pem)
                if include_all:
                    out.append((subject, pem))
                else:
                    low = subject.lower()
                    if any(kw in low for kw in _ALLOWLIST_KEYWORDS):
                        out.append((subject, pem))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Append ALL Windows root CAs (not just AV/proxy ones).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be appended, don't modify certifi.",
    )
    args = parser.parse_args()

    bundle = Path(certifi.where())
    print(f"certifi bundle: {bundle}")
    bundle_text = bundle.read_text(encoding="utf-8")

    roots = _gather_windows_roots(include_all=args.all)
    if not roots:
        print("No matching Windows roots found.")
        if not args.all:
            print("Try re-running with --all to inspect every Windows root.")
        return 1

    appended = 0
    for subject, pem in roots:
        if pem.strip() in bundle_text:
            continue
        print(f"  + {subject or '<unknown subject>'}")
        if not args.dry_run:
            with bundle.open("a", encoding="utf-8") as f:
                f.write("\n")
                f.write(pem)
            appended += 1

    if args.dry_run:
        print(f"DRY-RUN — {len(roots)} candidate(s); none written.")
    else:
        print(f"Appended {appended} cert(s) to {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
