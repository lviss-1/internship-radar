from datetime import time
from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")

NonEmptyStr = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimplifySourceSettings(StrictModel):
    url: str
    max_invalid_ratio: float = Field(ge=0, le=1)


class SourcesSettings(StrictModel):
    simplify: SimplifySourceSettings


class DeliverySettings(StrictModel):
    max_post_age_hours: int = Field(gt=0)
    max_posts_per_run: int = Field(gt=0)


class Thresholds(StrictModel):
    post: int = Field(ge=0, le=100)
    ping: int = Field(ge=0, le=100)


class QuietHours(StrictModel):
    timezone: str
    start: time
    end: time


class ModelSettings(StrictModel):
    cheap: str
    strong: str


class BudgetSettings(StrictModel):
    daily_api_call_cap: int = Field(gt=0)
    monthly_usd_hard_cap: float = Field(gt=0)
    monthly_usd_target: float = Field(gt=0)


class FilterSettings(StrictModel):
    seasons: list[str]
    unconfirmed_season_terms: list[str]
    curated_intern_sources: list[str]
    intern_title_keywords: list[NonEmptyStr]
    non_intern_title_keywords: list[NonEmptyStr]
    excluded_categories: list[NonEmptyStr]
    excluded_title_keywords: list[NonEmptyStr]

    @field_validator(
        "intern_title_keywords",
        "non_intern_title_keywords",
        "excluded_categories",
        "excluded_title_keywords",
        mode="after",
    )
    @classmethod
    def lowercase(cls, values: list[str]) -> list[str]:
        return [value.strip().lower() for value in values]


class LocationSettings(StrictModel):
    in_person_states: list[str]
    in_person_cities: dict[str, list[str]]
    aliases: dict[str, str]
    us_country_names: list[str]


class Settings(StrictModel):
    sources: SourcesSettings
    delivery: DeliverySettings
    thresholds: Thresholds
    quiet_hours: QuietHours
    models: ModelSettings
    budget: BudgetSettings
    watchlist_path: Path
    excluded_companies: list[str]
    filters: FilterSettings
    locations: LocationSettings


class ConfigError(Exception):
    pass


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"settings file {path} is not valid YAML") from exc

    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"settings file {path} is invalid:\n{exc}") from exc
