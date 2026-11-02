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
