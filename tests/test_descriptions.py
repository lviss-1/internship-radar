import html

import httpx
import pytest
import respx

from radar.net import make_client
from radar.sources.descriptions import DescriptionFetcher, html_to_text, is_safe_url

LONG_TEXT = "Responsibilities include building services. " * 20


def fetcher(client: httpx.Client, max_chars: int = 15000) -> DescriptionFetcher:
    return DescriptionFetcher(client, max_chars, sleep=lambda _: None)


def test_html_to_text_drops_scripts_and_keeps_structure() -> None:
    markup = (
        "<html><head><title>x</title></head><body><script>var a=1</script>"
        "<h2>About</h2><p>Build &amp; ship.</p><ul><li>Python</li><li>SQL</li></ul></body></html>"
    )

    assert html_to_text(markup) == "About\nBuild & ship.\n- Python\n- SQL"


@pytest.mark.parametrize(
    ("url", "safe"),
    [
        ("https://boards.greenhouse.io/acme/jobs/1", True),
        ("https://careers.example.com/job/1", True),
        ("http://careers.example.com/job/1", False),
        ("https://169.254.169.254/latest/meta-data", False),
        ("https://[::1]/x", False),
        ("https://localhost/x", False),
        ("https://metadata.google.internal/x", False),
        ("https://printer.local/x", False),
        ("https://intranet/x", False),
        ("https://careers.example.com:8443/x", False),
        ("not a url", False),
    ],
)
def test_only_public_https_urls_are_fetched(url: str, safe: bool) -> None:
    assert is_safe_url(url) is safe


@respx.mock
def test_greenhouse_uses_job_api_and_unescapes_content() -> None:
    content = html.escape("<p>Build &amp; ship APIs.</p>")
    route = respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs/123").mock(
        return_value=httpx.Response(200, json={"content": content})
    )

    with make_client() as client:
        result = fetcher(client).fetch("https://job-boards.greenhouse.io/acme/jobs/123?gh_src=x")

    assert route.called
    assert (result.text, result.via) == ("Build & ship APIs.", "greenhouse")


@respx.mock
def test_greenhouse_embed_urls_are_understood() -> None:
    route = respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs/77").mock(
        return_value=httpx.Response(200, json={"content": "&lt;p&gt;Hi&lt;/p&gt;"})
    )

    with make_client() as client:
        fetcher(client).fetch("https://boards.greenhouse.io/embed/job_app?for=acme&token=77")

    assert route.called


@respx.mock
def test_lever_combines_sections() -> None:
    respx.get("https://api.lever.co/v0/postings/acme/abc-123").mock(
        return_value=httpx.Response(
            200,
            json={
                "openingPlain": "Intro",
                "descriptionPlain": "About the role",
                "lists": [{"text": "You will", "content": "<li>Build</li><li>Test</li>"}],
                "additionalPlain": "Benefits",
            },
        )
    )

    with make_client() as client:
        result = fetcher(client).fetch("https://jobs.lever.co/acme/abc-123/apply")

    assert result.text == "Intro\n\nAbout the role\n\nYou will\n\n- Build\n- Test\n\nBenefits"


@respx.mock
def test_ashby_board_is_fetched_once_per_run() -> None:
    route = respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").mock(
        return_value=httpx.Response(
            200,
            json={"jobs": [
                {"id": "j1", "descriptionPlain": "First role"},
                {"id": "j2", "descriptionHtml": "<p>Second role</p>"},
            ]},
        )
    )  # fmt: skip

    with make_client() as client:
        shared = fetcher(client)
        first = shared.fetch("https://jobs.ashbyhq.com/acme/j1/application?embed=true")
        second = shared.fetch("https://jobs.ashbyhq.com/acme/j2")

    assert (first.text, second.text) == ("First role", "Second role")
    assert route.call_count == 1


@respx.mock
def test_workday_uses_json_endpoint_without_locale() -> None:
    route = respx.get(
        "https://acme.wd5.myworkdayjobs.com/wday/cxs/acme/Careers/job/NYC/Intern_R1"
    ).mock(
        return_value=httpx.Response(
            200, json={"jobPostingInfo": {"jobDescription": "<p>Workday role</p>"}}
        )
    )

    with make_client() as client:
        result = fetcher(client).fetch(
            "https://acme.wd5.myworkdayjobs.com/en-US/Careers/job/NYC/Intern_R1"
        )

    assert route.called
    assert result.via == "workday" and result.text == "Workday role"


@respx.mock
def test_structured_failure_falls_back_to_page() -> None:
    url = "https://acme.wd5.myworkdayjobs.com/Careers/job/NYC/Intern_R1"
    respx.get(url__startswith="https://acme.wd5.myworkdayjobs.com/wday/").mock(
        return_value=httpx.Response(404)
    )
    respx.get(url).mock(
        return_value=httpx.Response(
            200, headers={"Content-Type": "text/html"}, text=f"<p>{LONG_TEXT}</p>"
        )
    )

    with make_client() as client:
        result = fetcher(client).fetch(url)

    assert result.via == "page"


@respx.mock
def test_page_follows_safe_redirects_only() -> None:
    respx.get("https://jobs.example.com/a").mock(
        return_value=httpx.Response(302, headers={"Location": "/b"})
    )
    respx.get("https://jobs.example.com/b").mock(
        return_value=httpx.Response(302, headers={"Location": "http://169.254.169.254/"})
    )

    with make_client() as client:
        result = fetcher(client).fetch("https://jobs.example.com/a")

    assert result.found is False


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, headers={"Content-Type": "text/html"}, text="<p>Too short</p>"),
        httpx.Response(200, headers={"Content-Type": "application/pdf"}, content=b"%PDF"),
        httpx.Response(404),
    ],
)
def test_unusable_pages_are_not_found(response: httpx.Response) -> None:
    respx.get("https://jobs.example.com/a").mock(return_value=response)

    with make_client() as client:
        result = fetcher(client).fetch("https://jobs.example.com/a")

    assert (result.found, result.via) == (False, "not_found")


def test_unsafe_url_is_never_requested() -> None:
    with respx.mock(assert_all_called=False) as mock, make_client() as client:
        result = fetcher(client).fetch("http://169.254.169.254/latest")

    assert result.via == "unsafe_url"
    assert mock.calls.call_count == 0


@respx.mock
def test_long_descriptions_are_truncated_and_marked() -> None:
    respx.get("https://jobs.example.com/a").mock(
        return_value=httpx.Response(
            200, headers={"Content-Type": "text/html"}, text=f"<p>{LONG_TEXT * 10}</p>"
        )
    )

    with make_client() as client:
        result = fetcher(client, max_chars=500).fetch("https://jobs.example.com/a")

    assert result.text is not None
    assert result.text.endswith("[description truncated]")
    assert len(result.text) < 540


@respx.mock
def test_unexpected_payload_shapes_fall_back_instead_of_crashing() -> None:
    respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").mock(
        return_value=httpx.Response(200, json=["not", "an", "object"])
    )
    respx.get("https://jobs.ashbyhq.com/acme/j1").mock(return_value=httpx.Response(404))

    with make_client() as client:
        result = fetcher(client).fetch("https://jobs.ashbyhq.com/acme/j1")

    assert result.via == "not_found"
