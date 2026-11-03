import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from radar.ats import parse_ats_url

Source = Literal["simplify", "greenhouse", "lever", "ashby"]

_TERM_IN_TITLE = re.compile(
    r"\b(spring|summer|fall|autumn|winter)\s*(?:'|\u2019)?\s*(20\d{2}|\d{2})\b", re.IGNORECASE
)


def infer_terms(title: str) -> list[str]:
    """Read seasons like "Summer 2027" or "Summer '27" from a job title.

    Board postings have no structured term field, so the title is the only signal; an
    empty result means "unknown", which the filters treat as season unconfirmed.
    """
    terms: list[str] = []
    for season, year in _TERM_IN_TITLE.findall(title):
        name = "Fall" if season.lower() == "autumn" else season.capitalize()
        term = f"{name} {year if len(year) == 4 else '20' + year}"
        if term not in terms:
            terms.append(term)
    return terms


class Posting(BaseModel):
    """The shared shape every source is normalized into."""

    model_config = ConfigDict(frozen=True)

    source: Source
    source_id: str = Field(min_length=1)
    company: str = Field(min_length=1)
    title: str = Field(min_length=1)
    locations: list[str]
    url: HttpUrl
    posted_at: datetime
    description: str = ""
    terms: list[str] = Field(default_factory=list)
    category: str | None = None
    active: bool = True
    board: str | None = None

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def keys(self) -> frozenset[str]:
        """This posting's key plus the key the employer's own board would give the same job,
        so a job found through Simplify and through the board is recognized as one."""
        ref = parse_ats_url(str(self.url))
        alias = ref.job_key if ref else None
        return frozenset(key for key in (self.key, alias) if key)
