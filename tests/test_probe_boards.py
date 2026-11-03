import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
import respx

from radar.net import make_client
from tests.conftest import FIXTURES, ROOT

BOARDS = FIXTURES / "boards"
GREENHOUSE = "https://boards-api.greenhouse.io/v1/boards"
LEVER = "https://api.lever.co/v0/postings"
ASHBY = "https://api.ashbyhq.com/posting-api/job-board"


def load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("probe_boards", ROOT / "scripts/probe_boards.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["probe_boards"] = module
    spec.loader.exec_module(module)
    return module


probe = load_probe()


def fixture(name: str) -> Any:
    return json.loads((BOARDS / name).read_text(encoding="utf-8"))


@pytest.fixture
def api() -> Any:
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def run_probe(company: Any, evidence: dict[str, Any] | None = None) -> Any:
    candidates = probe.candidates_for(company, evidence or {})
    with make_client() as client:
        return probe.probe_company(
            company, candidates, probe.BoardClient(client, sleep=lambda _: None)
        )


def fallback_404(api: Any) -> None:
    """Register after specific routes: respx matches in registration order."""
    api.get(url__startswith=GREENHOUSE).mock(return_value=httpx.Response(404))
    api.get(url__startswith=LEVER).mock(return_value=httpx.Response(404))
    api.get(url__startswith=ASHBY).mock(return_value=httpx.Response(404))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://job-boards.greenhouse.io/doordashusa/jobs/7012345",
            ("greenhouse", "doordashusa"),
        ),
        ("https://boards.greenhouse.io/Figma/jobs/123?gh_jid=123", ("greenhouse", "figma")),
        ("https://boards.greenhouse.io/embed/job_app?for=stripe&token=1", ("greenhouse", "stripe")),
        ("https://jobs.lever.co/zoox/abc-123/apply", ("lever", "zoox")),
        ("https://jobs.ashbyhq.com/ramp/c815/application?embed=true", ("ashby", "ramp")),
        ("https://stripe.com/jobs/listing/intern/123?gh_jid=123", None),
        ("https://uber.wd5.myworkdayjobs.com/en-US/job/123", None),
        ("not a url", None),
    ],
)
def test_board_from_url(url: str, expected: tuple[str, str] | None) -> None:
    assert probe.board_from_url(url) == expected


def test_candidates_put_simplify_evidence_first() -> None:
    company = probe.SeedCompany("Scale AI", "extras")
    evidence = probe.simplify_evidence(
        [
            {"company_name": "Scale AI", "url": "https://job-boards.greenhouse.io/scaleai/jobs/1"},
            {"company_name": "Other Co", "url": "https://jobs.lever.co/other/1"},
            "not a record",
        ]
    )

    candidates = probe.candidates_for(company, evidence)

    assert candidates[0] == probe.Candidate(probe.Board.GREENHOUSE, "scaleai", True)
    assert {(c.board, c.slug) for c in candidates[1:]} == {
        (probe.Board.LEVER, "scaleai"),
        (probe.Board.ASHBY, "scaleai"),
        (probe.Board.GREENHOUSE, "scale-ai"),
        (probe.Board.LEVER, "scale-ai"),
        (probe.Board.ASHBY, "scale-ai"),
    }


def test_greenhouse_verified_by_board_and_company_name(api: Any) -> None:
    api.get(f"{GREENHOUSE}/seatgeek").mock(
        return_value=httpx.Response(200, json=fixture("greenhouse_board.json"))
    )
    api.get(f"{GREENHOUSE}/seatgeek/jobs").mock(
        return_value=httpx.Response(200, json=fixture("greenhouse_jobs.json"))
    )
    fallback_404(api)

    result = run_probe(probe.SeedCompany("SeatGeek", "core"))

    assert result.status is probe.Status.VERIFIED
    assert (result.board, result.slug, result.open_jobs) == (probe.Board.GREENHOUSE, "seatgeek", 3)
    assert result.evidence == ["board_name", "job_company_name"]


def test_ashby_verified_by_description_mentions(api: Any) -> None:
    api.get(f"{ASHBY}/ramp").mock(
        return_value=httpx.Response(200, json=fixture("ashby_board.json"))
    )
    fallback_404(api)

    result = run_probe(probe.SeedCompany("Ramp", "core"))

    assert result.status is probe.Status.VERIFIED
    assert (result.board, result.slug) == (probe.Board.ASHBY, "ramp")
    assert result.evidence == ["description_mentions"]


def test_existing_board_without_evidence_needs_review(api: Any) -> None:
    api.get(f"{LEVER}/linkedin").mock(
        return_value=httpx.Response(200, json=fixture("lever_postings.json"))
    )
    fallback_404(api)

    result = run_probe(probe.SeedCompany("LinkedIn", "core"))

    assert result.status is probe.Status.NEEDS_REVIEW
    assert result.board is None
    assert result.others == [{"board": "lever", "slug": "linkedin", "open_jobs": 3}]


def test_no_board_anywhere_is_feed_only(api: Any) -> None:
    fallback_404(api)

    assert run_probe(probe.SeedCompany("Netflix", "core")).status is probe.Status.FEED_ONLY


def test_server_errors_are_not_mistaken_for_feed_only(api: Any) -> None:
    api.get(url__startswith=GREENHOUSE).mock(return_value=httpx.Response(503))
    api.get(url__startswith=LEVER).mock(return_value=httpx.Response(404))
    api.get(url__startswith=ASHBY).mock(return_value=httpx.Response(404))

    result = run_probe(probe.SeedCompany("Netflix", "core"))

    assert result.status is probe.Status.PROBE_FAILED
    assert "HTTP 503" in result.note


def test_excluded_companies_are_never_probed(tmp_path: Path) -> None:
    seed = tmp_path / "seed.yaml"
    seed.write_text("core:\n  - name: Palantir\n  - name: Figma\nextras:\n  - name: anthropic\n")

    companies = probe.load_seed(seed, ["Palantir", "Anthropic"])

    assert [company.name for company in companies] == ["Figma"]


def listing(company: str, url: str, **overrides: Any) -> dict[str, Any]:
    record = {
        "id": f"{company}-{url}",
        "company_name": company,
        "title": "Software Engineer Intern",
        "url": url,
        "locations": ["NYC"],
        "terms": ["Summer 2027"],
        "date_posted": 1700000000,
        "active": False,
        "is_visible": True,
        "category": "Software",
    }
    record.update(overrides)
    return record


@pytest.fixture
def settings() -> Any:
    from radar.config import load_settings

    return load_settings(ROOT / "config" / "settings.yaml")


def test_discovery_ranks_by_matching_listings_and_skips_known(settings: Any) -> None:
    records = [
        listing("Zed", "https://jobs.lever.co/zed/0b1c2d3e-4f50-4a61-8b72-93a4b5c6d7e8"),
        listing("Acme", "https://job-boards.greenhouse.io/acme/jobs/1"),
        listing("Acme", "https://job-boards.greenhouse.io/acme/jobs/2"),
        listing("Known", "https://job-boards.greenhouse.io/known/jobs/3"),
        listing("Toronto Co", "https://job-boards.greenhouse.io/torco/jobs/4",
                locations=["Toronto, ON, Canada"]),
        listing("Chipmaker", "https://job-boards.greenhouse.io/chips/jobs/5", category="Hardware"),
        listing("Palantir", "https://jobs.lever.co/palantir/0b1c2d3e-4f50-4a61-8b72-93a4b5c6d7e9"),
        listing("Workday Co", "https://acme.wd5.myworkdayjobs.com/Careers/job/1"),
    ]  # fmt: skip

    candidates = probe.discovery_candidates(records, settings, {("greenhouse", "known")})

    assert [(c.board, c.slug, c.company, c.listings) for c in candidates] == [
        ("greenhouse", "acme", "Acme", 2),
        ("lever", "zed", "Zed", 1),
    ]


def test_discover_verifies_in_order_until_the_limit(api: Any) -> None:
    candidates = [
        probe.DiscoveryCandidate(probe.Board.GREENHOUSE, "gone", "Gone", 5),
        probe.DiscoveryCandidate(probe.Board.GREENHOUSE, "seatgeek", "SeatGeek", 4),
        probe.DiscoveryCandidate(probe.Board.ASHBY, "ramp", "Ramp", 3),
        probe.DiscoveryCandidate(probe.Board.GREENHOUSE, "never", "Never", 1),
    ]
    api.get(f"{GREENHOUSE}/seatgeek").mock(
        return_value=httpx.Response(200, json=fixture("greenhouse_board.json"))
    )
    api.get(f"{GREENHOUSE}/seatgeek/jobs").mock(
        return_value=httpx.Response(200, json=fixture("greenhouse_jobs.json"))
    )
    api.get(f"{ASHBY}/ramp").mock(
        return_value=httpx.Response(200, json=fixture("ashby_board.json"))
    )
    fallback_404(api)

    with make_client() as client:
        entries, skipped = probe.discover(
            candidates, probe.BoardClient(client, sleep=lambda _: None), limit=2
        )

    assert [(e["board"], e["slug"], e["group"], e["evidence"]) for e in entries] == [
        ("greenhouse", "seatgeek", "discovered", ["simplify_url"]),
        ("ashby", "ramp", "discovered", ["simplify_url"]),
    ]
    assert skipped == {"not_found": 1}


def test_seed_probe_and_discovery_keep_each_others_entries(tmp_path: Path) -> None:
    out = tmp_path / "watchlist.yaml"
    seed = {"name": "Figma", "group": "core", "status": "verified", "board": "greenhouse",
            "slug": "figma"}  # fmt: skip
    old = {"name": "Old", "group": "discovered", "status": "verified", "board": "ashby",
           "slug": "old"}  # fmt: skip
    probe.write_watchlist(out, [seed, old])

    kept = [e for e in probe.read_watchlist(out) if e.get("group") != probe.DISCOVERED_GROUP]
    discovered = [e for e in probe.read_watchlist(out) if e.get("group") == probe.DISCOVERED_GROUP]

    assert kept == [seed]
    assert discovered == [old]
    assert probe.known_boards([seed, {"name": "L", "status": "needs_review",
                                      "candidates": [{"board": "lever", "slug": "l"}]}]) == {
        ("greenhouse", "figma")
    }  # fmt: skip
