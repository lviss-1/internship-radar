import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum

from radar.config import FilterSettings, LocationSettings
from radar.normalize import Posting

US_STATES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "GU": "Guam", "VI": "U.S. Virgin Islands",
}  # fmt: skip
_STATE_BY_NAME = {name.lower(): code for code, name in US_STATES.items()}
_TYPO_PERIOD_BEFORE_STATE = re.compile(r"\.\s+(?=[A-Za-z]{2}$)")


class LocationKind(IntEnum):
    """Ordered by preference: when a posting lists several locations, the highest wins."""

    NOT_US = 0
    OTHER_US = 1
    REMOTE_US = 2
    IN_PERSON = 3


class RejectReason(StrEnum):
    INACTIVE = "inactive"
    STALE = "stale"
    EXCLUDED_COMPANY = "excluded_company"
    EXCLUDED_CATEGORY = "excluded_category"
    EXCLUDED_TITLE = "excluded_title"
    NOT_INTERN = "not_intern"
    SEASON = "season"
    LOCATION = "location"


class SeasonStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    MISMATCH = "mismatch"


class Tag(StrEnum):
    SEASON_UNCONFIRMED = "season unconfirmed"
    RELOCATION_UNKNOWN = "relocation unknown"


@dataclass(frozen=True)
class Match:
    posting: Posting
    location_kind: LocationKind
    tags: tuple[Tag, ...] = ()

    @property
    def silent(self) -> bool:
        return self.location_kind is LocationKind.OTHER_US


@dataclass
class FilterResult:
    matches: list[Match] = field(default_factory=list)
    rejected: Counter[RejectReason] = field(default_factory=Counter)


def _keyword_pattern(keywords: list[str]) -> re.Pattern[str] | None:
    if not keywords:
        return None
    ordered = sorted(keywords, key=len, reverse=True)
    alternatives = "|".join(re.escape(keyword) for keyword in ordered)
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


def _normalize_company(name: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", name.casefold()).split())


def _state_code(text: str) -> str | None:
    candidate = text.strip()
    if candidate.upper() in US_STATES:
        return candidate.upper()
    return _STATE_BY_NAME.get(candidate.lower())


class LocationClassifier:
    def __init__(self, settings: LocationSettings) -> None:
        self._aliases = {key.lower(): value for key, value in settings.aliases.items()}
        self._us_names = {name.lower() for name in settings.us_country_names}
        self._in_person_states = {state.upper() for state in settings.in_person_states}
        self._in_person_cities = {
            state.upper(): {city.lower() for city in cities}
            for state, cities in settings.in_person_cities.items()
        }

    def classify(self, raw: str) -> LocationKind:
        text = raw.strip()
        text = self._aliases.get(text.lower(), text)
        text = _TYPO_PERIOD_BEFORE_STATE.sub(", ", text)

        if text.lower().startswith("remote"):
            return self._classify_remote(text)

        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and parts[-1].lower() in self._us_names:
            parts = parts[:-1]
            if not parts:
                return LocationKind.OTHER_US
        if not parts:
            return LocationKind.NOT_US

        state = _state_code(parts[-1])
        if state is None:
            return LocationKind.NOT_US
        if state in self._in_person_states:
            return LocationKind.IN_PERSON
        cities = self._in_person_cities.get(state, set())
        if any(part.lower() in cities for part in parts[:-1]):
            return LocationKind.IN_PERSON
        return LocationKind.OTHER_US

    def _classify_remote(self, text: str) -> LocationKind:
        region = re.sub(r"^remote\s*(?:[-:,(]\s*)?(?:in\s+)?", "", text, flags=re.IGNORECASE)
        region = region.strip(" )")
        if region.lower() in self._us_names or _state_code(region) is not None:
            return LocationKind.REMOTE_US
        return LocationKind.NOT_US

    def best(self, locations: list[str]) -> LocationKind:
        return max((self.classify(location) for location in locations), default=LocationKind.NOT_US)


class PostingFilter:
    """Hard filters that cost nothing to run.

    Check order only decides which reject reason gets counted.
    """

    def __init__(
        self,
        filters: FilterSettings,
        locations: LocationSettings,
        excluded_companies: list[str],
        max_age: timedelta,
    ) -> None:
        self._filters = filters
        self._locations = LocationClassifier(locations)
        self._excluded_companies = [_normalize_company(name) for name in excluded_companies]
        self._max_age = max_age
        self._seasons = {season.lower() for season in filters.seasons}
        self._unconfirmed_terms = {term.lower() for term in filters.unconfirmed_season_terms}
        self._curated_sources = set(filters.curated_intern_sources)
        self._intern_pattern = _keyword_pattern(filters.intern_title_keywords)
        self._non_intern_pattern = _keyword_pattern(filters.non_intern_title_keywords)
        self._excluded_title_pattern = _keyword_pattern(filters.excluded_title_keywords)
        self._excluded_categories = set(filters.excluded_categories)

    def evaluate(self, posting: Posting, now: datetime) -> Match | RejectReason:
        title = posting.title.lower()

        if not posting.active:
            return RejectReason.INACTIVE
        if now - posting.posted_at > self._max_age:
            return RejectReason.STALE
        if self._is_excluded_company(posting.company):
            return RejectReason.EXCLUDED_COMPANY
        if posting.category and posting.category.strip().lower() in self._excluded_categories:
            return RejectReason.EXCLUDED_CATEGORY
        if self._excluded_title_pattern and self._excluded_title_pattern.search(title):
            return RejectReason.EXCLUDED_TITLE
        if not self._is_intern_level(posting.source, title):
            return RejectReason.NOT_INTERN

        tags: list[Tag] = []
        season = self._season_status(posting.terms)
        if season is SeasonStatus.MISMATCH:
            return RejectReason.SEASON
        if season is SeasonStatus.UNCONFIRMED:
            tags.append(Tag.SEASON_UNCONFIRMED)

        location_kind = self._locations.best(posting.locations)
        if location_kind is LocationKind.NOT_US:
            return RejectReason.LOCATION
        if location_kind is LocationKind.OTHER_US:
            tags.append(Tag.RELOCATION_UNKNOWN)

        return Match(posting=posting, location_kind=location_kind, tags=tuple(tags))

    def apply(self, postings: list[Posting], now: datetime) -> FilterResult:
        result = FilterResult()
        for posting in postings:
            outcome = self.evaluate(posting, now)
            if isinstance(outcome, Match):
                result.matches.append(outcome)
            else:
                result.rejected[outcome] += 1
        return result

    def _is_excluded_company(self, company: str) -> bool:
        normalized = _normalize_company(company)
        return any(
            normalized == excluded or normalized.startswith(f"{excluded} ")
            for excluded in self._excluded_companies
        )

    def _is_intern_level(self, source: str, title: str) -> bool:
        if self._non_intern_pattern and self._non_intern_pattern.search(title):
            return False
        if source in self._curated_sources:
            return True
        return bool(self._intern_pattern and self._intern_pattern.search(title))

    def _season_status(self, terms: list[str]) -> SeasonStatus:
        normalized = {term.lower() for term in terms if term.strip()}
        if normalized & self._seasons:
            return SeasonStatus.CONFIRMED
        if normalized <= self._unconfirmed_terms:
            return SeasonStatus.UNCONFIRMED
        return SeasonStatus.MISMATCH
