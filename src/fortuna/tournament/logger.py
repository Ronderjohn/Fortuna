"""Tournament run logging."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fortuna.tournament.models import TournamentBarResult


class TournamentLogger:
    """Write tournament signals and metadata."""

    def __init__(self, output_dir: Path, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = Path(output_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.signals_path = self.dir / "signals.csv"
        self.meta_path = self.dir / "meta.json"

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def write_signals(self, rows: list[TournamentBarResult]) -> None:
        if not rows:
            return
        fieldnames = list(asdict(rows[0]).keys())
        with self.signals_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
