from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.pipeline import _fetch_one
from rolescan.sources import available, get_source
from rolescan.sources.base import (
    _REGISTRY,
    PostingCache,
    Source,
    SourceSkipped,
    strip_html,
)


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


@respx.mock
async def test_adzuna_raises_when_every_query_fails() -> None:
    """A revoked key answers 401 on every query; returning [] reported `ok`
    with 0 jobs and the quiet alarm never named it (it had never returned
    rows under that key). All-failed is a failure."""
    respx.get(url__regex=r"https://api\.adzuna\.com/.*").mock(
        return_value=httpx.Response(401)
    )
    with pytest.raises(FetchError, match="every Adzuna query failed"):
        await _fetch(
            SourceEntry(
                kind="adzuna",
                slug="gb",
                app_id="x",
                app_key="y",
                queries=["energy", "python"],
            )
        )


@respx.mock
async def test_adzuna_keeps_going_when_one_query_fails() -> None:
    respx.get(
        "https://api.adzuna.com/v1/api/jobs/gb/search/1",
        params__contains={"what": "energy"},
    ).mock(return_value=httpx.Response(401))
    respx.get(
        "https://api.adzuna.com/v1/api/jobs/gb/search/1",
        params__contains={"what": "python"},
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "1",
                        "title": "Python Analyst",
                        "company": {"display_name": "Acme"},
                        "location": {"display_name": "London"},
                        "redirect_url": "https://adzuna/1",
                        "description": "python",
                        "created": "2026-10-01T00:00:00Z",
                    }
                ]
            },
        )
    )
    jobs = await _fetch(
        SourceEntry(
            kind="adzuna",
            slug="gb",
            app_id="x",
            app_key="y",
            queries=["energy", "python"],
        )
    )
    assert [j.title for j in jobs] == ["Python Analyst"]


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


# --- 2.5.8: how much of the board a source read ----------------------------

WD = "https://acme.wd3.myworkdayjobs.com/wday/cxs/acme/Careers"


async def _read(entry: SourceEntry) -> tuple[Source, list[Job]]:
    """The source as well as its jobs: `fetch` reports on the source."""
    async with Fetcher(HTTPConfig(max_retries=0)) as f:
        source = get_source(entry, f)
        return source, await source.fetch()


def _acme(**options: Any) -> SourceEntry:
    """acme's Workday board; no detail calls unless `details=True`."""
    board: dict[str, Any] = {"kind": "workday", "slug": "acme", "label": "Acme"}
    board |= {"site": "Careers", "host": "wd3", "details": False} | options
    return SourceEntry.model_validate(board)


def _workday_board(
    ids: range, *, by_text: dict[str, range] | None = None
) -> list[dict[str, Any]]:
    """Serve acme's listing as Workday does: `limit` postings a page, and the
    board's `total` on the first page only (0 on later ones, as in
    `test_workday_pages_past_the_second_page`). `by_text` serves a different
    board per `searchText`. Returns every request body, in order."""
    bodies: list[dict[str, Any]] = []

    def page(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        board = (by_text or {}).get(body["searchText"], ids)
        offset = body["offset"]
        rows = [
            {"title": f"Job {i}", "externalPath": f"/job/X/J_{i}"}
            for i in board[offset : offset + body["limit"]]
        ]
        return httpx.Response(
            200,
            json={"total": len(board) if offset == 0 else 0, "jobPostings": rows},
        )

    respx.post(f"{WD}/jobs").mock(side_effect=page)
    return bodies


def test_a_source_starts_with_nothing_to_report() -> None:
    source = get_source(SourceEntry(kind="greenhouse", slug="acme"), Fetcher())
    assert (source.total, source.truncated, source.note) == (None, "", "")


@respx.mock
async def test_workday_asks_for_the_whole_board_by_default() -> None:
    bodies = _workday_board(range(3))
    source, jobs = await _read(_acme())
    assert bodies == [{"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}]
    assert len(jobs) == 3
    assert (source.total, source.truncated, source.note) == (3, "", "")


@respx.mock
async def test_workday_reads_to_max_rows_and_says_it_stopped_short() -> None:
    """Four large boards were each read as exactly 500 postings on 2026-10-06,
    against live totals of 2,000 to 3,891, and every run read ok: 2.5.7
    stopped at 25 pages of 20 and said nothing. The default cap is still
    500; the read now says where it stopped."""
    _workday_board(range(2000))
    source, jobs = await _read(_acme())
    assert len(jobs) == 500
    assert source.total == 2000
    assert source.truncated == "stopped at 500 of 2000 postings (max_rows 500)"


@respx.mock
async def test_a_board_exactly_max_rows_long_is_not_cut_short() -> None:
    """A false "cut short" on every run would teach the reader to ignore it."""
    _workday_board(range(500))
    source, jobs = await _read(_acme())
    assert len(jobs) == 500
    assert (source.total, source.truncated) == (500, "")


@pytest.mark.parametrize(
    ("max_rows", "read", "truncated"),
    [
        ("30", 30, "stopped at 30 of 45 postings (max_rows 30)"),
        (45, 45, ""),
        ("2000", 45, ""),
    ],
)
@respx.mock
async def test_workday_max_rows_sets_the_cap(
    max_rows: object, read: int, truncated: str
) -> None:
    """A number or a string: a plugin passes options through from its own
    config, where every value is a string. 30 is not a multiple of the page
    size, so the second page is trimmed to fit."""
    _workday_board(range(45))
    source, jobs = await _read(_acme(max_rows=max_rows))
    assert [j.raw_id for j in jobs] == [f"/job/X/J_{i}" for i in range(read)]
    assert source.truncated == truncated


@pytest.mark.parametrize("max_rows", ["lots", 0, "-5"])
@respx.mock
async def test_workday_refuses_a_max_rows_that_is_not_a_positive_number(
    max_rows: object,
) -> None:
    with pytest.raises(ValueError, match="max_rows"):
        await _read(_acme(max_rows=max_rows))


@pytest.mark.parametrize(
    "facets",
    [
        {"Country_and_Jurisdiction": ["a1b2", "c3d4"], "jobFamilyGroup": "e5f6"},
        '{"Country_and_Jurisdiction": ["a1b2", "c3d4"], "jobFamilyGroup": "e5f6"}',
    ],
)
@respx.mock
async def test_workday_sends_applied_facets(facets: object) -> None:
    """A mapping in YAML, or the same as a JSON string; one id is a list."""
    bodies = _workday_board(range(3))
    await _read(_acme(applied_facets=facets))
    assert bodies[0]["appliedFacets"] == {
        "Country_and_Jurisdiction": ["a1b2", "c3d4"],
        "jobFamilyGroup": ["e5f6"],
    }


@pytest.mark.parametrize("facets", ['["a", "list"]', "{not json", {"Country": [1, 2]}])
@respx.mock
async def test_workday_refuses_applied_facets_it_cannot_send(facets: object) -> None:
    with pytest.raises(ValueError, match="applied_facets"):
        await _read(_acme(applied_facets=facets))


@respx.mock
async def test_one_search_text_is_one_pass_with_the_boards_total() -> None:
    bodies = _workday_board(range(0), by_text={"data": range(7)})
    source, jobs = await _read(_acme(search_text="data"))
    assert [b["searchText"] for b in bodies] == ["data"]
    assert (len(jobs), source.total) == (7, 7)


@respx.mock
async def test_workday_reads_a_pass_per_search_text_and_keeps_each_posting_once() -> (
    None
):
    bodies = _workday_board(
        range(0), by_text={"data": range(30), "analyst": range(20, 45)}
    )
    source, jobs = await _read(_acme(search_text='["data", "analyst"]'))
    assert sorted({b["searchText"] for b in bodies}) == ["analyst", "data"]
    assert sorted(int(j.raw_id.rsplit("_", 1)[1]) for j in jobs) == list(range(45))
    assert source.total is None, "overlapping passes state no one board total"
    assert source.truncated == ""


@respx.mock
async def test_workday_names_the_search_pass_that_was_cut_short() -> None:
    _workday_board(range(0), by_text={"data": range(45), "analyst": range(100, 110)})
    source, jobs = await _read(_acme(search_text=["data", "analyst"], max_rows=20))
    assert len(jobs) == 30
    assert source.truncated == (
        'stopped at 20 of 45 postings for search "data" (max_rows 20)'
    )


@respx.mock
async def test_a_board_whose_pages_end_before_its_total_is_cut_short() -> None:
    """The board states 45 and serves 20; the other 25 were never read."""

    def page(request: httpx.Request) -> httpx.Response:
        offset = json.loads(request.content)["offset"]
        rows = [
            {"title": f"Job {i}", "externalPath": f"/job/X/J_{i}"}
            for i in range(20 if offset == 0 else 0)
        ]
        return httpx.Response(
            200, json={"total": 45 if offset == 0 else 0, "jobPostings": rows}
        )

    respx.post(f"{WD}/jobs").mock(side_effect=page)
    source, jobs = await _read(_acme())
    assert len(jobs) == 20
    assert source.truncated == "the board lists 45 postings but its pages ended at 20"


@respx.mock
async def test_workday_skips_a_posting_that_is_not_a_job_and_reads_the_rest() -> None:
    """On 2026-10-07 a live board failed whole with "1 validation error for
    Job": one posting had no title, and the exception took every other
    posting on the board with it. The detail call proves the survivors stay
    paired with their own listing rows."""
    respx.post(f"{WD}/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 2,
                "jobPostings": [
                    {"title": "", "externalPath": "/job/X/Blank_0"},
                    {
                        "title": "Data Scientist",
                        "externalPath": "/job/X/DS_1",
                        "locationsText": "London",
                    },
                ],
            },
        )
    )
    respx.get(f"{WD}/job/X/DS_1").mock(
        return_value=httpx.Response(
            200, json={"jobPostingInfo": {"jobDescription": "<p>Python and SQL</p>"}}
        )
    )
    source, jobs = await _read(_acme(details=True))
    assert [(j.title, j.description) for j in jobs] == [
        ("Data Scientist", "Python and SQL")
    ]
    assert source.note.startswith(
        "skipped 1 posting(s) that are not valid jobs (first: /job/X/Blank_0: "
    )
    assert source.truncated == "", "a posting read and refused is not a cut"


class _SaysHowItWent(Source):
    """A plugin that reports its read, as Workday now does."""

    name = "says_how_it_went"

    async def fetch(self) -> list[Job]:
        self.total = 3891
        self.truncated = "stopped at 2000 of 3891 postings"
        self.note = "1 posting skipped"
        return [
            Job(source=self.name, company="Acme", title="Analyst", url="https://a/1")
        ]


class _OlderPlugin(Source):
    """A plugin written before 2.5.8 whose __init__ skips Source.__init__."""

    name = "older_plugin"

    def __init__(
        self, entry: SourceEntry, fetcher: Fetcher, cache: PostingCache | None = None
    ) -> None:
        self.entry = entry

    async def fetch(self) -> list[Job]:
        return []


async def test_the_report_carries_what_the_source_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_REGISTRY, "says_how_it_went", _SaysHowItWent)
    monkeypatch.setitem(_REGISTRY, "older_plugin", _OlderPlugin)
    async with Fetcher(HTTPConfig()) as f:
        said, _ = await _fetch_one(SourceEntry(kind="says_how_it_went", slug="a"), f)
        older, _ = await _fetch_one(SourceEntry(kind="older_plugin", slug="b"), f)

    assert (said.count, said.total, said.truncated, said.note) == (
        1,
        3891,
        "stopped at 2000 of 3891 postings",
        "1 posting skipped",
    )
    assert (older.ok, older.total, older.truncated, older.note) == (True, None, "", "")
