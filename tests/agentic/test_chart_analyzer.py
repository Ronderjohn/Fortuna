from __future__ import annotations

import json
from pathlib import Path

from fortuna.agentic.chart_analyzer import ChartImageAnalyzer
from fortuna.config.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        project_root=tmp_path,
        data_cache_dir=tmp_path / "cache",
        duckdb_path=tmp_path / "cache" / "fortuna.duckdb",
        openai_api_key="test-key",
        openai_model="gpt-5.4-mini",
    )


def test_chart_analyzer_parses_chart_response(tmp_path: Path):
    captured = {}

    def _transport(url, payload, headers, timeout):
        captured["payload"] = payload
        return {
            "output_text": json.dumps(
                {
                    "intent": "analyze",
                    "symbol": "RELIANCE.FUT",
                    "instrument_family": "future",
                    "timeframe": "15m",
                    "confidence": 0.9,
                    "trend_bias": "bullish",
                    "level_notes": ["support near prior swing"],
                    "summary": "Front-month future chart looks constructive.",
                    "warnings": [],
                }
            )
        }

    analyzer = ChartImageAnalyzer(_settings(tmp_path), transport=_transport)
    result = analyzer.analyze(image_bytes=b"fake-image", caption="Crompton Futures")
    assert result.ok is True
    assert result.symbol == "RELIANCE.FUT"
    assert result.timeframe == "15m"
    assert captured["payload"]["instructions"]
    content = captured["payload"]["input"][0]["content"]
    assert content[0]["type"] == "input_image"


def test_chart_analyzer_rejects_missing_text_output(tmp_path: Path):
    analyzer = ChartImageAnalyzer(_settings(tmp_path), transport=lambda *args: {"output": []})
    result = analyzer.analyze(image_bytes=b"fake-image", caption="")
    assert result.ok is False
    assert result.error is not None
