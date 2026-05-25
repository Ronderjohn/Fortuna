"""Discover strategy JSON paths for dashboard batch runs."""

from __future__ import annotations

from pathlib import Path

from fortuna.app.config import AppConfig
from fortuna.config.settings import Settings


def list_strategy_paths(settings: Settings, app_config: AppConfig | None = None) -> list[Path]:
    app_config = app_config or AppConfig.from_settings(settings)
    paths: list[Path] = []
    seen: set[str] = set()
    for rel in app_config.strategy_dirs:
        d = settings.resolve_path(Path(rel))
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name not in seen:
                seen.add(p.name)
                paths.append(p)
    return paths
