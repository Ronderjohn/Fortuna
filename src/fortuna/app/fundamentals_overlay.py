"""Optional operator-provided fundamentals/reference CSV overlay for discovery."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from fortuna.agentic.contracts import MarketUniverseCandidate
from fortuna.app.outcome_bias import exposure_key
from fortuna.config.settings import Settings
from fortuna.data.instruments import InstrumentRegistry

_SYMBOL_COLUMNS = ("symbol", "ticker", "stock", "nse code", "nsecode")
_SECTOR_COLUMNS = ("sector", "industry")
_MARKET_CAP_COLUMNS = ("market_cap_bucket", "market cap bucket", "cap_bucket", "cap bucket")
_QUALITY_COLUMNS = ("operator_quality_score", "quality_score", "quality score", "quality")
_EXCLUDE_COLUMNS = ("operator_exclude", "exclude", "skip")
_NOTE_COLUMNS = ("note", "notes", "comment", "comments")
_TRUTHY = {"1", "true", "yes", "y", "on"}
_FALSY = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class FundamentalsOverlayRow:
    symbol: str
    sector: str = ""
    market_cap_bucket: str = ""
    operator_quality_score: Optional[float] = None
    operator_exclude: bool = False
    note: str = ""


@dataclass(frozen=True)
class FundamentalsOverlayLoadResult:
    rows: dict[str, FundamentalsOverlayRow]
    source_path: Optional[str] = None
    loaded_count: int = 0
    skipped_count: int = 0
    diagnostics: tuple[str, ...] = ()
    enabled: bool = False

    @property
    def summary(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.source_path:
            return "disabled"
        return f"loaded={self.loaded_count} skipped={self.skipped_count}"


def load_fundamentals_overlay(
    settings: Settings,
    registry: InstrumentRegistry,
) -> FundamentalsOverlayLoadResult:
    enabled = bool(getattr(settings, "market_universe_fundamentals_overlay_enabled", False))
    if not enabled:
        return FundamentalsOverlayLoadResult(rows={}, enabled=False)

    rel = getattr(
        settings,
        "market_universe_fundamentals_overlay_csv",
        Path("data/universe/fundamentals_overlay.csv"),
    )
    path = settings.resolve_path(Path(rel))
    if not path.is_file():
        return FundamentalsOverlayLoadResult(
            rows={},
            source_path=str(path),
            enabled=True,
            diagnostics=(f"overlay file missing: {path}",),
        )

    rows: dict[str, FundamentalsOverlayRow] = {}
    skipped = 0
    diagnostics: list[str] = []
    line_no = 1
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return FundamentalsOverlayLoadResult(
                    rows={},
                    source_path=str(path),
                    enabled=True,
                    diagnostics=("overlay file has no header row",),
                )
            for raw in reader:
                line_no += 1
                normalized = {str(k or "").strip().lower(): v for k, v in raw.items()}
                symbol_text = _first_value(normalized, _SYMBOL_COLUMNS)
                if not symbol_text:
                    skipped += 1
                    diagnostics.append(f"line {line_no}: missing symbol")
                    continue
                resolved = _resolve_symbol(symbol_text, registry)
                if resolved is None:
                    skipped += 1
                    diagnostics.append(f"line {line_no}: unknown symbol {symbol_text!r}")
                    continue
                quality = _parse_quality(_first_value(normalized, _QUALITY_COLUMNS))
                if _first_value(normalized, _QUALITY_COLUMNS) and quality is None:
                    skipped += 1
                    diagnostics.append(f"line {line_no}: invalid operator_quality_score")
                    continue
                exclude_raw = _first_value(normalized, _EXCLUDE_COLUMNS)
                exclude = _parse_bool(exclude_raw) if exclude_raw is not None else False
                if exclude_raw is not None and exclude is None:
                    skipped += 1
                    diagnostics.append(f"line {line_no}: invalid operator_exclude")
                    continue
                sector = _first_value(normalized, _SECTOR_COLUMNS) or ""
                cap_bucket = _first_value(normalized, _MARKET_CAP_COLUMNS) or ""
                note = _first_value(normalized, _NOTE_COLUMNS) or ""
                key = exposure_key(resolved)
                rows[key] = FundamentalsOverlayRow(
                    symbol=resolved,
                    sector=str(sector).strip(),
                    market_cap_bucket=str(cap_bucket).strip(),
                    operator_quality_score=quality,
                    operator_exclude=bool(exclude),
                    note=str(note).strip(),
                )
    except OSError as exc:
        return FundamentalsOverlayLoadResult(
            rows={},
            source_path=str(path),
            enabled=True,
            diagnostics=(f"overlay read failed: {exc}",),
        )

    return FundamentalsOverlayLoadResult(
        rows=rows,
        source_path=str(path),
        loaded_count=len(rows),
        skipped_count=skipped,
        diagnostics=tuple(diagnostics),
        enabled=True,
    )


def apply_fundamentals_overlay_to_candidate(
    candidate: Optional[MarketUniverseCandidate],
    overlay_row: Optional[FundamentalsOverlayRow],
    *,
    settings: Settings,
) -> Optional[MarketUniverseCandidate]:
    if candidate is None or overlay_row is None:
        return candidate
    if overlay_row.operator_exclude:
        return None

    max_boost = max(
        0.0,
        float(getattr(settings, "market_universe_fundamentals_overlay_max_boost", 0.05)),
    )
    quality_weight = max(
        0.0,
        float(getattr(settings, "market_universe_fundamentals_quality_weight", 1.0)),
    )
    adjustment = 0.0
    notes = list(candidate.notes)
    if overlay_row.operator_quality_score is not None and max_boost > 0:
        centered = (float(overlay_row.operator_quality_score) - 0.5) * 2.0
        adjustment = round(
            max(-max_boost, min(max_boost, centered * max_boost * quality_weight)),
            4,
        )
    if overlay_row.note:
        notes.append(f"overlay_note={overlay_row.note}")
    if adjustment != 0.0:
        notes.append(
            "fundamentals_overlay="
            f"quality={overlay_row.operator_quality_score} adj={adjustment:+.4f}"
        )

    liquidity_score = candidate.liquidity_score
    if liquidity_score is not None and adjustment != 0.0:
        liquidity_score = round(liquidity_score + adjustment, 6)

    return replace(
        candidate,
        sector=overlay_row.sector or candidate.sector,
        market_cap_bucket=overlay_row.market_cap_bucket or candidate.market_cap_bucket,
        operator_quality_score=overlay_row.operator_quality_score,
        fundamentals_overlay_adjustment=adjustment if adjustment != 0.0 else None,
        liquidity_score=liquidity_score,
        notes=tuple(notes),
    )


def _resolve_symbol(text: str, registry: InstrumentRegistry) -> Optional[str]:
    candidate = str(text or "").strip().upper()
    if not candidate:
        return None
    variants = (candidate, f"{candidate}.NS", f"{candidate}-EQ")
    for item in variants:
        try:
            ref = registry.resolve(item)
            if ref.is_future or ref.is_option:
                continue
            return f"{ref.symbol}.NS"
        except Exception:  # noqa: BLE001
            continue
    return None


def _parse_float(value: Any) -> Optional[float]:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_quality(value: Optional[str]) -> Optional[float]:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    if parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in _TRUTHY:
        return True
    if text in _FALSY:
        return False
    return None


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        if key in row and str(row[key] or "").strip():
            return str(row[key]).strip()
    return None
