from datetime import time
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

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
    heartbeat_hour: int = Field(ge=0, le=23)


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


Effort = Literal["low", "medium", "high", "xhigh", "max"]


class ScoringSettings(StrictModel):
    max_description_chars: int = Field(gt=0)
    score_effort: Effort
    score_max_tokens: int = Field(gt=0)
    draft_effort: Effort
    draft_max_tokens: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    worst_case_input_tokens: int = Field(gt=0)


class ModelPrice(StrictModel):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_write: float = Field(ge=0)


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
    us_cities: dict[str, Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]]


class Settings(StrictModel):
    sources: SourcesSettings
    delivery: DeliverySettings
    thresholds: Thresholds
    quiet_hours: QuietHours
    models: ModelSettings
    scoring: ScoringSettings
    pricing: dict[str, ModelPrice]
    budget: BudgetSettings
    watchlist_path: Path
    excluded_companies: list[str]
    filters: FilterSettings
    locations: LocationSettings


class ConfigError(Exception):
    pass


def _check_cross_fields(settings: Settings) -> None:
    if settings.thresholds.ping < settings.thresholds.post:
        raise ConfigError("thresholds.ping must be >= thresholds.post")
    missing = {settings.models.cheap, settings.models.strong} - settings.pricing.keys()
    if missing:
        raise ConfigError(f"pricing is missing models: {sorted(missing)}")


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read settings file {path}: {exc.strerror}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"settings file {path} is not valid YAML") from exc

    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"settings file {path} is invalid:\n{exc}") from exc
    _check_cross_fields(settings)
    return settings
