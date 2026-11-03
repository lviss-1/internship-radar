import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from radar.ats import Board, parse_ats_url
from radar.net import make_client
from radar.normalize import Posting, infer_terms
from radar.sources.boards import BoardSource, WatchlistBoard, WatchlistError, load_watchlist
from tests.conftest import FIXTURES, ROOT

BOARDS = FIXTURES / "boards"
GH = "https://boards-api.greenhouse.io/v1/boards"
ASHBY = "https://api.ashbyhq.com/posting-api/job-board"
LEVER = "https://api.lever.co/v0/postings"


def fixture(name: str) -> Any:
    return json.loads((BOARDS / name).read_text(encoding="utf-8"))


def accept_all(source: str, title: str, posted_at: datetime, keys: frozenset[str]) -> bool:
    return True


def interns_only(source: str, title: str, posted_at: datetime, keys: frozenset[str]) -> bool:
    return "intern" in title.lower()


def fetch(boards: list[WatchlistBoard], wants: Any = accept_all) -> Any:
    with make_client() as client:
        return BoardSource(client, 15000, sleep=lambda _: None).fetch(boards, wants)


DATADOG = WatchlistBoard("Datadog", Board.GREENHOUSE, "datadog")
NOTION = WatchlistBoard("Notion", Board.ASHBY, "notion")
ZOOX = WatchlistBoard("Zoox", Board.LEVER, "zoox")


@pytest.mark.parametrize(
    ("url", "board", "slug", "job_key"),
    [
        ("https://job-boards.greenhouse.io/knowbe4/jobs/8919274002", "greenhouse", "knowbe4",
         "greenhouse:8919274002"),
        ("https://a16z.com/about/jobs/?gh_jid=8046730003", "greenhouse", None,
         "greenhouse:8046730003"),
        ("https://boards.greenhouse.io/embed/job_app?for=acme&token=77", "greenhouse", "acme",
         "greenhouse:77"),
        ("https://jobs.lever.co/zoox/0b1c2d3e-4f50-4a61-8b72-93a4b5c6d7e8/apply", "lever", "zoox",
         "lever:0b1c2d3e-4f50-4a61-8b72-93a4b5c6d7e8"),
        ("https://jobs.ashbyhq.com/ramp/not-a-uuid", "ashby", "ramp", None),
    ],
)  # fmt: skip
def test_parse_ats_url(url: str, board: str, slug: str | None, job_key: str | None) -> None:
    ref = parse_ats_url(url)

    assert ref is not None
    assert (ref.board, ref.slug, ref.job_key) == (board, slug, job_key)


@pytest.mark.parametrize(
    "url", ["https://uber.wd5.myworkdayjobs.com/x", "https://example.com/jobs/1", "not a url"]
)
def test_non_ats_urls_have_no_ref(url: str) -> None:
    assert parse_ats_url(url) is None


def test_posting_keys_include_the_board_alias() -> None:
    posting = Posting(
        source="simplify",
        source_id="abc",
        company="a16z",
        title="Research Intern",
        locations=["NYC"],
        url="https://a16z.com/about/jobs/?gh_jid=8046730003",
        posted_at=datetime(2026, 11, 1, tzinfo=UTC),
    )

    assert posting.keys == {"simplify:abc", "greenhouse:8046730003"}


@pytest.mark.parametrize(
    ("title", "terms"),
    [
        ("Software Engineer Intern - Summer 2027", ["Summer 2027"]),
        ("SWE Intern (Summer '27)", ["Summer 2027"]),
        ("Autumn 2027 Co-op", ["Fall 2027"]),
        ("Winter 2027 / Summer 2027 Intern", ["Winter 2027", "Summer 2027"]),
        ("Software Engineer Intern", []),
    ],
)
def test_infer_terms(title: str, terms: list[str]) -> None:
    assert infer_terms(title) == terms


def write_watchlist(tmp_path: Path, companies: list[dict[str, Any]]) -> Path:
    path = tmp_path / "watchlist.yaml"
    path.write_text(json.dumps({"companies": companies}))
    return path


def test_watchlist_takes_verified_and_evidenced_extra_boards(tmp_path: Path) -> None:
    campus = {"board": "greenhouse", "slug": "duolingounirecruitment", "evidence": ["board_name"]}
    unproven = {"board": "lever", "slug": "duo", "open_jobs": 1}
    duolingo = {
        "name": "Duolingo",
        "status": "verified",
        "board": "greenhouse",
        "slug": "duolingo",
        "other_boards": [campus, unproven],
    }
    path = write_watchlist(
        tmp_path,
        [
            duolingo,
            {"name": "LinkedIn", "status": "needs_review", "candidates": [{"board": "lever"}]},
            {"name": "Netflix", "status": "feed_only"},
            {"name": "Palantir", "status": "verified", "board": "lever", "slug": "palantir"},
        ],
    )

    boards = load_watchlist(path, ["Palantir"]).boards

    assert [board.key for board in boards] == [
        "greenhouse/duolingo",
        "greenhouse/duolingounirecruitment",
    ]


def test_real_watchlist_loads() -> None:
    loaded = load_watchlist(ROOT / "config/watchlist.yaml", [])
    keys = {board.key for board in loaded.boards}
    assert loaded.generated_at is not None

    assert {"greenhouse/datadog", "greenhouse/duolingounirecruitment"} <= keys


def test_invalid_watchlist_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "watchlist.yaml"
    path.write_text("companies: not-a-list")

    with pytest.raises(WatchlistError):
        load_watchlist(path, [])


@respx.mock
def test_greenhouse_fetches_detail_only_for_wanted_jobs() -> None:
    listing = fixture("greenhouse_list.json")
    respx.get(f"{GH}/datadog/jobs").mock(return_value=httpx.Response(200, json=listing))
    intern_id = listing["jobs"][0]["id"]
    detail = respx.get(f"{GH}/datadog/jobs/{intern_id}").mock(
        return_value=httpx.Response(200, json=fixture("greenhouse_detail.json"))
    )

    result = fetch([DATADOG], interns_only)

    posting = result.postings[0]
    assert detail.call_count == 1
    assert len(result.postings) == 1
    assert len(result.published) == 3
    assert posting.key == f"greenhouse:{intern_id}"
    assert posting.board == "greenhouse/datadog"
    assert posting.company == "Datadog"
    assert posting.locations == ["Paris, France", "Paris, Paris, France"]
    assert posting.description and "&lt;" not in posting.description
    assert result.polled == ["greenhouse/datadog"]


@respx.mock
def test_greenhouse_detail_failure_still_yields_the_posting() -> None:
    listing = fixture("greenhouse_list.json")
    respx.get(f"{GH}/datadog/jobs").mock(return_value=httpx.Response(200, json=listing))
    respx.get(url__startswith=f"{GH}/datadog/jobs/").mock(return_value=httpx.Response(404))

    result = fetch([DATADOG], interns_only)

    assert len(result.postings) == 1
    assert result.postings[0].description == ""


@respx.mock
def test_ashby_maps_locations_terms_and_remote() -> None:
    respx.get(f"{ASHBY}/notion").mock(
        return_value=httpx.Response(200, json=fixture("ashby_full.json"))
    )

    result = fetch([NOTION])

    intern, multi_site, remote = result.postings
    assert intern.terms == ["Winter 2027"]
    assert intern.description
    assert len(multi_site.locations) >= 2
    assert any(place.startswith("Remote in ") for place in remote.locations)
    assert all(posting.board == "ashby/notion" for posting in result.postings)


@respx.mock
def test_lever_maps_created_at_and_locations() -> None:
    raw = fixture("lever_board.json")
    respx.get(f"{LEVER}/zoox").mock(return_value=httpx.Response(200, json=raw))

    result = fetch([ZOOX])

    first = result.postings[0]
    assert first.posted_at == datetime.fromtimestamp(raw[0]["createdAt"] / 1000, tz=UTC)
    assert first.locations == [raw[0]["categories"]["location"]]
    assert first.key == f"lever:{raw[0]['id']}"


@respx.mock
def test_unwanted_jobs_are_indexed_but_not_built() -> None:
    respx.get(f"{LEVER}/zoox").mock(
        return_value=httpx.Response(200, json=fixture("lever_board.json"))
    )

    result = fetch([ZOOX], interns_only)

    assert result.postings == []
    assert len(result.published) == 3


@respx.mock
def test_one_failing_board_does_not_stop_the_others() -> None:
    respx.get(f"{GH}/datadog/jobs").mock(return_value=httpx.Response(503))
    respx.get(f"{ASHBY}/notion").mock(return_value=httpx.Response(404))
    respx.get(f"{LEVER}/zoox").mock(
        return_value=httpx.Response(200, json=fixture("lever_board.json"))
    )

    result = fetch([DATADOG, NOTION, ZOOX])

    assert result.polled == ["lever/zoox"]
    assert result.failures == {
        "greenhouse/datadog": "HTTP 503 after 3 attempts",
        "ashby/notion": "HTTP 404",
    }


@respx.mock
def test_unexpected_payload_is_a_board_failure() -> None:
    respx.get(f"{GH}/datadog/jobs").mock(
        return_value=httpx.Response(200, json=["not", "a", "dict"])
    )

    result = fetch([DATADOG])

    assert result.failures == {"greenhouse/datadog": "unexpected response (AttributeError)"}


@respx.mock
def test_stale_dates_and_bad_urls_are_skipped_quietly() -> None:
    listing = fixture("greenhouse_list.json")
    listing["jobs"][1]["first_published"] = None
    listing["jobs"][1]["updated_at"] = None
    listing["jobs"][2]["absolute_url"] = "not a url"
    respx.get(f"{GH}/datadog/jobs").mock(return_value=httpx.Response(200, json=listing))
    respx.get(url__startswith=f"{GH}/datadog/jobs/").mock(return_value=httpx.Response(404))

    result = fetch([DATADOG])

    assert len(result.published) == 2
    assert [p.source_id for p in result.postings] == [str(listing["jobs"][0]["id"])]


def test_wants_receives_publish_time_from_the_board() -> None:
    seen: list[datetime] = []

    def record(source: str, title: str, posted_at: datetime, keys: frozenset[str]) -> bool:
        seen.append(posted_at)
        return False

    with respx.mock() as mock:
        mock.get(f"{LEVER}/zoox").mock(
            return_value=httpx.Response(200, json=fixture("lever_board.json"))
        )
        fetch([ZOOX], record)

    assert all(posted_at.tzinfo is not None for posted_at in seen)
    assert all(posted_at < datetime.now(UTC) + timedelta(days=1) for posted_at in seen)
