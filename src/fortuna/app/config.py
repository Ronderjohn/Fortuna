"""Dashboard / Streamlit app configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fortuna.config.settings import Settings, load_settings


@dataclass(frozen=True)
class AppConfig:
    parallel_workers: int = 6
    live_refresh_seconds: int = 5
    strategy_dirs: tuple[str, ...] = (
        "strategies/intraday",
        "strategies/generated",
        "strategies/builtin",
    )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "AppConfig":
        settings = settings or load_settings()
        raw: dict[str, Any] = {}
        import os
        from pathlib import Path

        import yaml

        cfg_path = os.environ.get("FORTUNA_CONFIG", "configs/intraday.yaml")
        path = settings.resolve_path(Path(cfg_path))
        if path.exists():
            with path.open(encoding="utf-8") as f:
                doc = yaml.safe_load(f) or {}
            raw = doc.get("app", {}) or {}

        dirs = raw.get("strategy_dirs")
        if dirs:
            strategy_dirs = tuple(str(d) for d in dirs)
        else:
            strategy_dirs = cls.strategy_dirs

        return cls(
            parallel_workers=int(raw.get("parallel_workers", cls.parallel_workers)),
            live_refresh_seconds=int(raw.get("live_refresh_seconds", cls.live_refresh_seconds)),
            strategy_dirs=strategy_dirs,
        )
