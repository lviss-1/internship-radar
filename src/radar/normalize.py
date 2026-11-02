from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

Source = Literal["simplify", "greenhouse", "lever", "ashby"]


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

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}"
