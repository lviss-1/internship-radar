from datetime import timedelta
from typing import Any

import pytest

from radar.config import Settings
from radar.filters import (
    LocationClassifier,
    LocationKind,
    Match,
    PostingFilter,
    RejectReason,
    Tag,
    split_locations,
)
from radar.sources.simplify import parse_listings
from tests.conftest import NOW, make_posting


@pytest.fixture
def posting_filter(settings: Settings) -> PostingFilter:
    return PostingFilter(
        settings.filters,
        settings.locations,
        settings.excluded_companies,
        max_age=timedelta(hours=settings.delivery.max_post_age_hours),
    )


@pytest.fixture
def classifier(settings: Settings) -> LocationClassifier:
    return LocationClassifier(settings.locations)


def outcome_for(posting_filter: PostingFilter, **overrides: Any) -> Match | RejectReason:
    return posting_filter.evaluate(make_posting(**overrides), NOW)


def test_recorded_feed_end_to_end(
    posting_filter: PostingFilter, raw_listings: list[dict[str, Any]]
) -> None:
    postings = parse_listings(raw_listings, max_invalid_ratio=0).postings
    outcomes = {p.company: posting_filter.evaluate(p, NOW) for p in postings}

    def kind(company: str) -> LocationKind:
        outcome = outcomes[company]
        assert isinstance(outcome, Match), f"{company} was rejected: {outcome}"
        return outcome.location_kind

    assert kind("Andreessen Horowitz") is LocationKind.IN_PERSON
    assert kind("Microsoft") is LocationKind.IN_PERSON
    assert kind("Molina Healthcare") is LocationKind.REMOTE_US
    assert kind("Nokia") is LocationKind.OTHER_US
    assert kind("Entrust") is LocationKind.OTHER_US
    assert kind("Elucid") is LocationKind.OTHER_US
    assert kind("KnowBe4") is LocationKind.OTHER_US

    assert outcomes["KION Group"] is RejectReason.LOCATION
    assert outcomes["Qualcomm"] is RejectReason.EXCLUDED_CATEGORY
    assert outcomes["First Citizens BancShares"] is RejectReason.EXCLUDED_CATEGORY
    assert outcomes["TikTok"] is RejectReason.SEASON
    assert outcomes["The New York Times"] is RejectReason.INACTIVE

    microsoft = outcomes["Microsoft"]
    assert isinstance(microsoft, Match)
    assert microsoft.tags == (Tag.SEASON_UNCONFIRMED,)
    nokia = outcomes["Nokia"]
    assert isinstance(nokia, Match)
    assert nokia.tags == (Tag.SEASON_UNCONFIRMED, Tag.RELOCATION_UNKNOWN)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NYC", LocationKind.IN_PERSON),
        ("New York, NY", LocationKind.IN_PERSON),
        ("Brooklyn, NY", LocationKind.IN_PERSON),
        ("Long Island City, Queens, NY", LocationKind.IN_PERSON),
        ("Jersey City, NJ", LocationKind.IN_PERSON),
        ("Newark, NJ, USA", LocationKind.IN_PERSON),
        ("Stamford, CT", LocationKind.IN_PERSON),
        ("New Jersey", LocationKind.IN_PERSON),
        ("Connecticut", LocationKind.IN_PERSON),
        ("White Plains, NY", LocationKind.IN_PERSON),
        ("Tarrytown, NY", LocationKind.IN_PERSON),
        ("Mount Vernon, NY", LocationKind.IN_PERSON),
        ("Mt. Kisco, NY", LocationKind.IN_PERSON),
        ("Hastings-on-Hudson, NY", LocationKind.IN_PERSON),
        ("Pearl River, NY", LocationKind.IN_PERSON),
        ("Plainview, NY", LocationKind.IN_PERSON),
        ("Melville, NY, USA", LocationKind.IN_PERSON),
        ("Rochester, NY", LocationKind.OTHER_US),
        ("Fishkill, NY", LocationKind.OTHER_US),
        ("Bohemia, NY", LocationKind.OTHER_US),
        ("Long Island, NY", LocationKind.OTHER_US),
        ("White Plains, CA", LocationKind.OTHER_US),
        ("Remote in USA", LocationKind.REMOTE_US),
        ("Remote in US", LocationKind.REMOTE_US),
        ("Remote - United States", LocationKind.REMOTE_US),
        ("Remote (US)", LocationKind.REMOTE_US),
        ("Remote in Canada", LocationKind.NOT_US),
        ("Remote in UK", LocationKind.NOT_US),
        ("Remote", LocationKind.NOT_US),
        ("United States", LocationKind.OTHER_US),
        ("Austin, TX", LocationKind.OTHER_US),
        ("Dallas. TX", LocationKind.OTHER_US),
        ("San Jose, Ca", LocationKind.OTHER_US),
        ("SF", LocationKind.OTHER_US),
        ("Texas", LocationKind.OTHER_US),
        ("Washington", LocationKind.OTHER_US),
        ("Washington, DC", LocationKind.OTHER_US),
        ("Toronto, ON, Canada", LocationKind.NOT_US),
        ("Kanata, Ottawa, ON, Canada", LocationKind.NOT_US),
        ("London, UK", LocationKind.NOT_US),
        ("Singapore", LocationKind.NOT_US),
        ("Europe", LocationKind.NOT_US),
        ("Abu Dhabi - United Arab Emirates", LocationKind.NOT_US),
        ("", LocationKind.NOT_US),
    ],
)
def test_location_classification(
    classifier: LocationClassifier, raw: str, expected: LocationKind
) -> None:
    assert classifier.classify(raw) is expected


def test_best_location_wins(classifier: LocationClassifier) -> None:
    assert classifier.best(["London, UK", "Austin, TX", "NYC"]) is LocationKind.IN_PERSON
    assert classifier.best(["Remote in Canada", "Remote in USA"]) is LocationKind.REMOTE_US
    assert classifier.best([]) is LocationKind.NOT_US


@pytest.mark.parametrize(
    "title",
    [
        "Quantitative Trader Intern",
        "Quant Research Intern",
        "Quantitative Developer Intern",
        "FPGA Engineer Intern",
        "Hardware Engineering Intern",
        "Silicon Validation Intern",
    ],
)
def test_excluded_titles(posting_filter: PostingFilter, title: str) -> None:
    assert outcome_for(posting_filter, title=title) is RejectReason.EXCLUDED_TITLE


@pytest.mark.parametrize(
    "title",
    [
        "Quantum Computing Software Intern",
        "Uncertainty Quantification Intern",
        "Software Engineer Intern - Trading Platform",
    ],
)
def test_similar_words_are_not_excluded(posting_filter: PostingFilter, title: str) -> None:
    assert isinstance(outcome_for(posting_filter, title=title), Match)


@pytest.mark.parametrize("category", ["Quant", "Hardware", "Quantitative Finance", " hardware "])
def test_excluded_categories(posting_filter: PostingFilter, category: str) -> None:
    assert outcome_for(posting_filter, category=category) is RejectReason.EXCLUDED_CATEGORY


@pytest.mark.parametrize("company", ["Palantir", "Palantir Technologies", "Anthropic", "ANTHROPIC"])
def test_excluded_companies(posting_filter: PostingFilter, company: str) -> None:
    assert outcome_for(posting_filter, company=company) is RejectReason.EXCLUDED_COMPANY


def test_company_exclusion_needs_whole_word(posting_filter: PostingFilter) -> None:
    assert isinstance(outcome_for(posting_filter, company="Palantirian Labs"), Match)


def test_curated_source_trusts_intern_level(posting_filter: PostingFilter) -> None:
    assert isinstance(outcome_for(posting_filter, title="Student Researcher"), Match)


@pytest.mark.parametrize(
    ("source", "title", "expected_match"),
    [
        ("greenhouse", "Software Engineer", False),
        ("greenhouse", "Software Engineer Intern", True),
        ("lever", "Software Engineering Co-op", True),
        ("simplify", "Software Engineer - New Grad Program", False),
        ("simplify", "Senior Software Engineer", False),
    ],
)
def test_intern_level(
    posting_filter: PostingFilter, source: str, title: str, expected_match: bool
) -> None:
    outcome = outcome_for(posting_filter, source=source, title=title)

    assert isinstance(outcome, Match) is expected_match
    if not expected_match:
        assert outcome is RejectReason.NOT_INTERN


@pytest.mark.parametrize(
    ("terms", "expected"),
    [
        (["Summer 2027"], ()),
        (["summer 2027"], ()),
        (["Fall 2026", "Spring 2027"], ()),
        (["N/A"], (Tag.SEASON_UNCONFIRMED,)),
        ([], (Tag.SEASON_UNCONFIRMED,)),
    ],
)
def test_accepted_seasons(
    posting_filter: PostingFilter, terms: list[str], expected: tuple[Tag, ...]
) -> None:
    outcome = outcome_for(posting_filter, terms=terms)

    assert isinstance(outcome, Match)
    assert outcome.tags == expected


@pytest.mark.parametrize("terms", [["Winter 2027"], ["Summer 2026"], ["N/A", "Summer 2026"]])
def test_rejected_seasons(posting_filter: PostingFilter, terms: list[str]) -> None:
    assert outcome_for(posting_filter, terms=terms) is RejectReason.SEASON


def test_age_window(posting_filter: PostingFilter) -> None:
    fresh = outcome_for(posting_filter, posted_at=NOW - timedelta(hours=71))
    stale = outcome_for(posting_filter, posted_at=NOW - timedelta(hours=73))

    assert isinstance(fresh, Match)
    assert stale is RejectReason.STALE


def test_inactive_is_rejected(posting_filter: PostingFilter) -> None:
    assert outcome_for(posting_filter, active=False) is RejectReason.INACTIVE


def test_only_relocation_roles_post_silently(posting_filter: PostingFilter) -> None:
    local = outcome_for(posting_filter, locations=["NYC"])
    remote = outcome_for(posting_filter, locations=["Remote in USA"])
    elsewhere = outcome_for(posting_filter, locations=["Austin, TX"])

    assert isinstance(local, Match) and not local.silent
    assert isinstance(remote, Match) and not remote.silent
    assert isinstance(elsewhere, Match) and elsewhere.silent
    assert elsewhere.tags == (Tag.RELOCATION_UNKNOWN,)


def test_apply_counts_reject_reasons(posting_filter: PostingFilter) -> None:
    postings = [
        make_posting(source_id="1"),
        make_posting(source_id="2", active=False),
        make_posting(source_id="3", locations=["London, UK"]),
        make_posting(source_id="4", locations=["Toronto, ON, Canada"]),
    ]

    result = posting_filter.apply(postings, NOW)

    assert [m.posting.source_id for m in result.matches] == ["1"]
    assert result.rejected == {RejectReason.INACTIVE: 1, RejectReason.LOCATION: 2}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hybrid - New York, NY", LocationKind.IN_PERSON),
        ("Hybrid - San Francisco, CA", LocationKind.OTHER_US),
        ("San Francisco, CA • New York, NY", LocationKind.IN_PERSON),
        (
            "San Francisco, CA, US; Palo Alto, CA, US; Seattle, WA, US; New York, NY, US",
            LocationKind.IN_PERSON,
        ),
        (
            "San Francisco, CA;  Sunnyvale, CA ; New York City, NY; Seattle, WA",
            LocationKind.IN_PERSON,
        ),
        ("Costa Mesa, California, United States", LocationKind.OTHER_US),
        ("New York, New York", LocationKind.IN_PERSON),
        ("New York, NY (HQ)", LocationKind.IN_PERSON),
        ("New York", LocationKind.IN_PERSON),
        ("New York, Seattle, South San Francisco HQ", LocationKind.IN_PERSON),
        ("San Francisco, New York City, Seattle", LocationKind.IN_PERSON),
        ("Seattle", LocationKind.OTHER_US),
        ("Oakland, CA", LocationKind.OTHER_US),
        ("Remote - US", LocationKind.REMOTE_US),
        ("Toronto, ON, CA", LocationKind.NOT_US),
        ("Toronto, Canada", LocationKind.NOT_US),
        ("Toronto", LocationKind.NOT_US),
        ("Cambridge, UK", LocationKind.NOT_US),
        ("Dublin", LocationKind.NOT_US),
        ("Dublin, IE", LocationKind.NOT_US),
        ("Zurich, CH", LocationKind.NOT_US),
        ("Paris, France", LocationKind.NOT_US),
        ("In-Office", LocationKind.NOT_US),
        ("Hybrid", LocationKind.NOT_US),
    ],
)
def test_board_location_formats(
    classifier: LocationClassifier, raw: str, expected: LocationKind
) -> None:
    assert classifier.best([raw]) is expected


@pytest.mark.parametrize(
    ("raw", "places"),
    [
        ("Hybrid - New York, NY", ["New York, NY"]),
        ("A, CA • B, NY", ["A, CA", "B, NY"]),
        ("New York, NY (HQ)", ["New York, NY"]),
        ("Remote (US)", ["Remote (US)"]),
        ("In-Office", []),
        ("  ;  ", []),
    ],
)
def test_split_locations(raw: str, places: list[str]) -> None:
    assert split_locations(raw) == places
