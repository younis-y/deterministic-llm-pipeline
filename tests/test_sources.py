from __future__ import annotations

import json

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.sources import available, get_source
from rolescan.sources.base import Source, SourceSkipped, strip_html


async def _fetch(entry: SourceEntry) -> list[Job]:
    async with Fetcher(HTTPConfig(max_retries=0)) as f:
        return await get_source(entry, f).fetch()


def test_every_builtin_source_is_registered() -> None:
    assert {
        "smartrecruiters",
        "greenhouse",
        "lever",
        "ashby",
        "workable",
        "workday",
        "adzuna",
    } <= set(available())
    assert all(issubclass(c, Source) for c in available().values())


def test_unknown_kind_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="greenhouse"):
        get_source(SourceEntry(kind="nope", slug="x"), Fetcher())


def test_strip_html_unwraps_entities_and_tags() -> None:
    out = strip_html("<p>Python &amp; SQL</p><ul><li>Energy</li></ul>")
    assert "Python & SQL" in out
    assert "<" not in out


@respx.mock
async def test_greenhouse_parses() -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Energy Data Scientist",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                        "content": "<p>Forecasting &amp; Python</p>",
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                ]
            },
        )
    )
    jobs = await _fetch(SourceEntry(kind="greenhouse", slug="acme", label="Acme"))
    assert len(jobs) == 1
    assert jobs[0].title == "Energy Data Scientist"
    assert jobs[0].description == "Forecasting & Python"
    assert jobs[0].company == "Acme"
    assert jobs[0].posted is not None


@respx.mock
async def test_smartrecruiters_paginates() -> None:
    url = "https://api.smartrecruiters.com/v1/companies/Masdar/postings"
    page1 = {
        "totalFound": 3,
        "content": [
            {
                "id": f"a{i}",
                "name": f"Role {i}",
                "location": {"city": "Abu Dhabi", "country": "United Arab Emirates"},
                "releasedDate": "2026-08-01",
            }
            for i in range(2)
        ],
    }
    page2 = {
        "totalFound": 3,
        "content": [
            {
                "id": "a2",
                "name": "Role 2",
                "location": {"city": "Abu Dhabi", "country": "United Arab Emirates"},
                "jobAd": {
                    "sections": {"jobDescription": {"text": "<p>UAE National</p>"}}
                },
            }
        ],
    }
    respx.get(url).mock(
        side_effect=[httpx.Response(200, json=page1), httpx.Response(200, json=page2)]
    )
    jobs = await _fetch(
        SourceEntry(kind="smartrecruiters", slug="Masdar", label="Masdar")
    )
    assert len(jobs) == 3
    assert jobs[0].location == "Abu Dhabi, United Arab Emirates"
    assert "UAE National" in jobs[-1].description
    assert jobs[-1].url.endswith("/Masdar/a2")


@respx.mock
async def test_lever_converts_epoch_millis() -> None:
    respx.get("https://api.lever.co/v0/postings/vitol").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "x",
                    "text": "Power Trading Analyst",
                    "categories": {"location": "London", "commitment": "Remote"},
                    "hostedUrl": "https://jobs.lever.co/vitol/x",
                    "descriptionPlain": "Day-ahead markets.",
                    "createdAt": 1755648000000,
                }
            ],
        )
    )
    jobs = await _fetch(SourceEntry(kind="lever", slug="vitol", label="Vitol"))
    assert jobs[0].posted is not None
    assert jobs[0].remote is True


@respx.mock
async def test_ashby_and_workable_parse() -> None:
    respx.get("https://api.ashbyhq.com/posting-api/job-board/modo").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": "1",
                        "title": "Analyst",
                        "location": "London",
                        "jobUrl": "https://jobs.ashbyhq.com/modo/1",
                        "descriptionHtml": "<p>Batteries</p>",
                        "publishedAt": "2026-08-11",
                    }
                ]
            },
        )
    )
    respx.get("https://apply.workable.com/api/v1/widget/accounts/enapp").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Data Engineer",
                        "city": "Dubai",
                        "country": "UAE",
                        "url": "https://apply.workable.com/enapp/j/1",
                        "description": "<p>Pipelines</p>",
                        "published_on": "2026-08-12",
                        "telecommuting": True,
                    }
                ]
            },
        )
    )
    ashby = await _fetch(SourceEntry(kind="ashby", slug="modo", label="Modo"))
    workable = await _fetch(SourceEntry(kind="workable", slug="enapp", label="Enapp"))
    assert ashby[0].description == "Batteries"
    assert workable[0].location == "Dubai, UAE"
    assert workable[0].remote is True


@respx.mock
async def test_workday_posts_and_enriches() -> None:
    base = "https://adnoc.wd3.myworkdayjobs.com/wday/cxs/adnoc/Careers"
    respx.post(f"{base}/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "jobPostings": [
                    {
                        "title": "Data Scientist",
                        "externalPath": "/job/AD/DS_1",
                        "locationsText": "Abu Dhabi",
                    }
                ],
            },
        )
    )
    respx.get(f"{base}/job/AD/DS_1").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobPostingInfo": {
                    "jobDescription": "<p>Python and SQL</p>",
                    "startDate": "2026-08-15",
                }
            },
        )
    )
    jobs = await _fetch(
        SourceEntry(
            kind="workday", slug="adnoc", label="ADNOC", site="Careers", host="wd3"
        )
    )
    assert jobs[0].description == "Python and SQL"
    assert jobs[0].posted is not None
    assert "/en-US/Careers/job/AD/DS_1" in jobs[0].url


@respx.mock
async def test_workday_pages_past_the_second_page() -> None:
    """Workday reports `total` on the first page only; later pages say 0.

    Checking `offset >= total` against each page's own total therefore stopped
    after page two, capping every board at 40 jobs. Live 2026-10-06: Shell's
    board said total=140 at offset 0 and total=0 at offsets 20 and 40, and
    Baker Hughes (665 jobs) was read as 40.
    """
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/Careers"

    def page(request: httpx.Request) -> httpx.Response:
        offset = json.loads(request.content)["offset"]
        size = max(0, min(20, 45 - offset))
        return httpx.Response(
            200,
            json={
                "total": 45 if offset == 0 else 0,
                "jobPostings": [
                    {
                        "title": f"Job {offset + i}",
                        "externalPath": f"/job/X/J_{offset + i}",
                    }
                    for i in range(size)
                ],
            },
        )

    respx.post(f"{base}/jobs").mock(side_effect=page)
    jobs = await _fetch(
        SourceEntry(
            kind="workday", slug="acme", site="Careers", host="wd3", details=False
        )
    )
    assert len(jobs) == 45


@respx.mock
async def test_workday_survives_a_failed_detail_call() -> None:
    base = "https://adnoc.wd3.myworkdayjobs.com/wday/cxs/adnoc/Careers"
    respx.post(f"{base}/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "jobPostings": [
                    {
                        "title": "Data Scientist",
                        "externalPath": "/job/AD/DS_1",
                        "locationsText": "Abu Dhabi",
                    }
                ],
            },
        )
    )
    respx.get(f"{base}/job/AD/DS_1").mock(return_value=httpx.Response(500))
    jobs = await _fetch(
        SourceEntry(
            kind="workday", slug="adnoc", label="ADNOC", site="Careers", host="wd3"
        )
    )
    assert len(jobs) == 1, "the listing must survive a dead detail endpoint"
    assert jobs[0].description == ""


async def test_adzuna_skips_unsupported_country() -> None:
    """The UAE is genuinely not covered. It must signal SKIPPED rather than
    return [], which would render as a successful zero-result probe."""
    with pytest.raises(SourceSkipped, match="UAE"):
        await _fetch(
            SourceEntry(
                kind="adzuna",
                slug="ae",
                label="Adzuna AE",
                app_id="x",
                app_key="y",
                queries=["energy"],
            )
        )


async def test_adzuna_without_credentials_signals_skipped() -> None:
    with pytest.raises(SourceSkipped, match="credentials"):
        await _fetch(SourceEntry(kind="adzuna", slug="gb", queries=["energy"]))


@respx.mock
async def test_adzuna_parses() -> None:
    respx.get("https://api.adzuna.com/v1/api/jobs/gb/search/1").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "1",
                        "title": "Energy Analyst",
                        "company": {"display_name": "Drax"},
                        "location": {"display_name": "London, UK"},
                        "redirect_url": "https://adzuna/1",
                        "description": "Power markets",
                        "created": "2026-08-19T00:00:00Z",
                    }
                ]
            },
        )
    )
    jobs = await _fetch(
        SourceEntry(
            kind="adzuna", slug="gb", app_id="x", app_key="y", queries=["energy"]
        )
    )
    assert jobs[0].company == "Drax"
    assert jobs[0].source == "adzuna:gb"


# --- fetcher behaviour -----------------------------------------------------


@respx.mock
async def test_fetcher_retries_5xx_then_succeeds() -> None:
    route = respx.get("https://x.test/j").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})]
    )
    async with Fetcher(HTTPConfig(max_retries=2)) as f:
        assert await f.fetch_json("https://x.test/j") == {"ok": True}
    assert route.call_count == 2


@respx.mock
async def test_fetcher_does_not_retry_404() -> None:
    """A wrong slug should fail fast and loudly, not burn three retries."""
    route = respx.get("https://x.test/j").mock(return_value=httpx.Response(404))
    async with Fetcher(HTTPConfig(max_retries=3)) as f:
        with pytest.raises(FetchError, match="404"):
            await f.fetch_json("https://x.test/j")
    assert route.call_count == 1


@respx.mock
async def test_fetcher_raises_on_bad_json() -> None:
    respx.get("https://x.test/j").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    async with Fetcher(HTTPConfig(max_retries=0)) as f:
        with pytest.raises(FetchError, match="bad JSON"):
            await f.fetch_json("https://x.test/j")


async def test_fetcher_requires_context_manager() -> None:
    with pytest.raises(RuntimeError, match="context manager"):
        _ = Fetcher().client


# --- raw text fetching (the structured source needs HTML, not JSON) --------


@respx.mock
async def test_fetch_text_returns_the_body_and_response_headers() -> None:
    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(200, html="<html><body>hi</body></html>")
    )
    async with Fetcher(HTTPConfig(max_retries=0)) as f:
        body = await f.fetch_text("https://example.com/page")
    assert "hi" in body


@respx.mock
async def test_fetch_text_raises_fetcherror_on_404_without_retrying() -> None:
    route = respx.get("https://example.com/gone").mock(return_value=httpx.Response(404))
    async with Fetcher(HTTPConfig(max_retries=3)) as f:
        with pytest.raises(FetchError):
            await f.fetch_text("https://example.com/gone")
    assert route.call_count == 1, "a 404 is a wrong URL, not a transient failure"


# --- model_copy skips validation -------------------------------------------


@respx.mock
async def test_workday_enrichment_keeps_posted_a_real_date() -> None:
    """_enrich used model_copy(update=...), which bypasses pydantic entirely,
    so Workday's raw startDate string landed in a `date` field. Nothing failed
    until the digest called .isoformat() on it and the whole scan crashed."""
    base = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/Careers"
    respx.post(f"{base}/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 1,
                "jobPostings": [
                    {
                        "title": "Power Trader",
                        "externalPath": "/job/London/Power-Trader_R1",
                        "locationsText": "London",
                    }
                ],
            },
        )
    )
    respx.get(f"{base}/job/London/Power-Trader_R1").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobPostingInfo": {
                    "jobDescription": "<p>Trade power.</p>",
                    "startDate": "2026-08-23T00:00:00Z",
                }
            },
        )
    )
    jobs = await _fetch(
        SourceEntry(kind="workday", slug="acme", site="Careers", host="wd3")
    )
    assert len(jobs) == 1
    posted = jobs[0].posted
    assert posted is not None
    assert hasattr(posted, "isoformat"), f"posted is a {type(posted).__name__}"
    assert posted.isoformat() == "2026-08-23"


@respx.mock
async def test_lever_falls_back_to_the_eu_host() -> None:
    """A board on Lever's EU domain 404s on the default host.

    Without the fallback the source fails with a bare 404, which is
    indistinguishable from a mistyped slug and tells the reader nothing.
    """
    respx.get("https://api.lever.co/v0/postings/prima").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.eu.lever.co/v0/postings/prima").mock(
        return_value=httpx.Response(
            200,
            json=[{"id": "1", "text": "Data Engineer", "hostedUrl": "https://e/1"}],
        )
    )
    async with Fetcher() as f:
        jobs = await get_source(SourceEntry(kind="lever", slug="prima"), f).fetch()
    assert [j.title for j in jobs] == ["Data Engineer"]


@respx.mock
async def test_lever_non_eu_board_never_touches_the_eu_host() -> None:
    respx.get("https://api.lever.co/v0/postings/vitol").mock(
        return_value=httpx.Response(200, json=[])
    )
    eu = respx.get("https://api.eu.lever.co/v0/postings/vitol")
    async with Fetcher() as f:
        jobs = await get_source(SourceEntry(kind="lever", slug="vitol"), f).fetch()
    assert jobs == []
    assert not eu.called, "an empty board must not be retried on the EU host"


@respx.mock
async def test_lever_wrong_slug_still_fails_loudly() -> None:
    """404 on both hosts is a bad slug, and must not read as an empty board."""
    respx.get("https://api.lever.co/v0/postings/nope").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://api.eu.lever.co/v0/postings/nope").mock(
        return_value=httpx.Response(404)
    )
    async with Fetcher() as f:
        with pytest.raises(FetchError):
            await get_source(SourceEntry(kind="lever", slug="nope"), f).fetch()
