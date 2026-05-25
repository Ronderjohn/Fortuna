"""Angel One SmartAPI credentials and feed options (SMARTAPI_ env prefix)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SCRIP_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)


class SmartAPISettings(BaseSettings):
    """SmartAPI credentials — use SMARTAPI_* or ANGEL_API_KEY in .env."""

    model_config = SettingsConfigDict(
        env_prefix="SMARTAPI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("SMARTAPI_API_KEY", "ANGEL_API_KEY"),
    )
    client_code: Optional[str] = None
    password: Optional[str] = Field(
        default=None,
        description="Trading PIN used for loginByPassword",
    )
    totp_secret: Optional[str] = None
    use_live_feed: bool = False
    scrip_master_url: str = _DEFAULT_SCRIP_URL
    scrip_cache_path: Path = Field(default=Path("data/smartapi/OpenAPIScripMaster.json"))
    rate_limit_per_sec: float = 3.0

    @model_validator(mode="after")
    def _fill_api_key_alias(self) -> "SmartAPISettings":
        if not self.api_key:
            self.api_key = os.environ.get("ANGEL_API_KEY") or os.environ.get("SMARTAPI_API_KEY")
        return self

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.client_code and self.password and self.totp_secret)

    def resolve_path(self, project_root: Optional[Path] = None) -> Path:
        root = project_root or _PROJECT_ROOT
        if self.scrip_cache_path.is_absolute():
            return self.scrip_cache_path
        return root / self.scrip_cache_path


@lru_cache
def get_smartapi_settings() -> SmartAPISettings:
    load_dotenv(_PROJECT_ROOT / ".env")
    return SmartAPISettings()
