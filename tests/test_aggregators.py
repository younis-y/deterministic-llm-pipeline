"""The `reed` and `jooble` aggregator sources (2.6.0).

Both are built from the providers' public API documentation and have not been
run against the live services (no key was available): the fixtures below are
in the documented shapes, and a first real run is what would show a field the
documentation did not mention.

  Reed    GET https://www.reed.co.uk/api/1.0/search, HTTP Basic with the key as
          the user name; {"results": [...], "totalResults": N}; `resultsToTake`
          up to 100 and `resultsToSkip` page through it; dates are dd/mm/yyyy.
  Jooble  POST https://jooble.org/api/<key> with a JSON body; {"totalCount": N,
          "jobs": [...]}; `page` pages through it.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import date
from typing import Any

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.pipeline import _fetch_one
from rolescan.sources import get_source
from rolescan.sources.aggregators import Jooble, Reed
from rolescan.sources.base import ProbeStatus, SourceSkipped

REED = "https://www.reed.co.uk/api/1.0/search"
KEY = "k3y-0000-example"
JOOBLE = f"https://jooble.org/api/{KEY}"


async def _read(entry: SourceEntry) -> tuple[list[Job], Any]:
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        source = get_source(entry, fetcher)
        return await source.fetch(), source


# --- Reed ----------------------------------------------------------------------


def _reed_row(n: int, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "jobId": 51000000 + n,
        "employerId": 1234,
        "employerName": "Acme Energy",
        "employerProfileId": None,
        "employerProfileName": None,
        "jobTitle": f"Energy Data Analyst {n}",
        "locationName": "London",
        "minimumSalary": 45000.0,
        "maximumSalary": 55000.0,
        "currency": "GBP",
        "expirationDate": "30/10/2026",
        "date": "09/10/2026",
        "jobDescription": "Analyse power markets with Python &amp; SQL. <b>Hybrid</b>.",
        "applications": 3,
        "jobUrl": f"https://www.reed.co.uk/jobs/energy-data-analyst/{51000000 + n}",
    }
    row.update(over)
    return row


def _reed_entry(**options: Any) -> SourceEntry:
    base: dict[str, Any] = {
        "kind": "reed",
        "slug": "uk",
        "label": "Reed UK",
        "api_key": KEY,
        "queries": ["energy analyst"],
    }
    base.update(options)
    return SourceEntry(**base)


def _reed_pages(total: int, per_page: int = 100) -> list[respx.Route]:
    """One answer per page over `total` results, matched on `resultsToSkip`."""
    routes = []
    for page in range(9):
        skip = page * per_page
        rows = [_reed_row(n) for n in range(skip, min(skip + per_page, total))]
        routes.append(
            respx.get(REED, params={"resultsToSkip": str(skip)}).mock(
                return_value=httpx.Response(
                    200, json={"results": rows, "totalResults": total}
                )
            )
        )
    return routes


@respx.mock
async def test_reed_reads_every_page_the_total_says_there_are() -> None:
    routes = _reed_pages(230)

    jobs, source = await _read(_reed_entry())

    assert len(jobs) == 230
    assert [r.called for r in routes[:4]] == [True, True, True, False]
    assert source.total == 230
    assert source.truncated == ""


@respx.mock
async def test_reed_asks_for_a_page_and_where_to_start() -> None:
    routes = _reed_pages(130, per_page=50)

    await _read(_reed_entry(results_per_page=50))

    first = routes[0].calls[0].request.url.params
    assert first["keywords"] == "energy analyst"
    assert first["resultsToTake"] == "50"
    assert first["resultsToSkip"] == "0"
    assert routes[1].calls[0].request.url.params["resultsToSkip"] == "50"


@respx.mock
async def test_reed_stops_at_max_pages_and_says_so() -> None:
    _reed_pages(1000)

    jobs, source = await _read(_reed_entry(max_pages=2))

    assert len(jobs) == 200
    assert source.total == 1000
    assert "200 of 1000" in source.truncated
    assert "max_pages 2" in source.truncated
    assert 'query "energy analyst"' in source.truncated


@respx.mock
async def test_reed_stops_on_a_short_page() -> None:
    routes = _reed_pages(130)

    jobs, source = await _read(_reed_entry())

    assert len(jobs) == 130
    assert routes[2].called is False
    assert source.truncated == ""


@respx.mock
async def test_reed_sends_the_key_as_basic_auth_and_never_in_the_url() -> None:
    routes = _reed_pages(3)

    await _read(_reed_entry())

    request = routes[0].calls[0].request
    expected = "Basic " + base64.b64encode(f"{KEY}:".encode()).decode()
    assert request.headers["Authorization"] == expected
    assert KEY not in str(request.url)


@respx.mock
async def test_reed_maps_its_fields_onto_a_job() -> None:
    _reed_pages(1)

    jobs, _ = await _read(_reed_entry())

    job = jobs[0]
    assert job.source == "reed:uk"
    assert job.company == "Acme Energy"
    assert job.title == "Energy Data Analyst 0"
    assert job.location == "London"
    assert job.url == "https://www.reed.co.uk/jobs/energy-data-analyst/51000000"
    assert job.raw_id == "51000000"
    assert job.posted == date(2026, 10, 9), "Reed dates are day first"
    assert job.description == "Analyse power markets with Python & SQL. Hybrid."
    assert job.remote is False


@respx.mock
async def test_reed_sees_a_remote_role_in_the_title_or_the_place() -> None:
    respx.get(REED).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    _reed_row(1, locationName="Remote, UK"),
                    _reed_row(2, jobTitle="Remote Energy Analyst"),
                    _reed_row(3),
                ],
                "totalResults": 3,
            },
        )
    )

    jobs, _ = await _read(_reed_entry())

    assert [j.remote for j in jobs] == [True, True, False]


@respx.mock
async def test_reed_skips_a_row_it_cannot_use_and_says_how_many() -> None:
    respx.get(REED).mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    _reed_row(1),
                    _reed_row(2, jobUrl=""),
                    _reed_row(3, jobTitle=""),
                    _reed_row(4, date="not a date"),
                ],
                "totalResults": 4,
            },
        )
    )

    jobs, source = await _read(_reed_entry())

    assert [j.raw_id for j in jobs] == ["51000001", "51000004"]
    assert jobs[1].posted is None
    assert "2" in source.note and "title or link" in source.note


@respx.mock
async def test_reed_passes_its_search_options_through() -> None:
    route = respx.get(REED).mock(
        return_value=httpx.Response(200, json={"results": [], "totalResults": 0})
    )

    await _read(
        _reed_entry(
            where="Leeds",
            distance=20,
            graduate=True,
            direct_employer_only=True,
        )
    )

    params = route.calls[0].request.url.params
    assert params["locationName"] == "Leeds"
    assert params["distanceFromLocation"] == "20"
    assert params["graduate"] == "true"
    assert params["postedByDirectEmployer"] == "true"


@respx.mock
async def test_reed_sends_no_option_it_was_not_given() -> None:
    route = respx.get(REED).mock(
        return_value=httpx.Response(200, json={"results": [], "totalResults": 0})
    )

    await _read(_reed_entry())

    params = route.calls[0].request.url.params
    assert set(params) == {"keywords", "resultsToTake", "resultsToSkip"}


@respx.mock
async def test_reed_runs_each_query_and_keeps_all_of_them() -> None:
    for query, base in (("energy analyst", 0), ("data scientist", 500)):
        respx.get(REED, params={"keywords": query}).mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [_reed_row(base + i) for i in range(3)],
                    "totalResults": 3,
                },
            )
        )

    jobs, source = await _read(
        _reed_entry(queries=["energy analyst", "data scientist"])
    )

    assert len(jobs) == 6
    assert source.total is None, "two queries overlap, so none is claimed"


@respx.mock
async def test_reed_one_failed_query_keeps_the_other() -> None:
    respx.get(REED, params={"keywords": "energy analyst"}).mock(
        return_value=httpx.Response(
            200, json={"results": [_reed_row(1)], "totalResults": 1}
        )
    )
    respx.get(REED, params={"keywords": "data scientist"}).mock(
        return_value=httpx.Response(500)
    )

    jobs, _ = await _read(_reed_entry(queries=["energy analyst", "data scientist"]))

    assert len(jobs) == 1


@respx.mock
async def test_reed_every_query_failing_is_an_error_not_an_empty_source() -> None:
    respx.get(REED).mock(return_value=httpx.Response(401))

    with pytest.raises(FetchError, match="every Reed query failed"):
        await _read(_reed_entry())


@respx.mock
async def test_reed_a_later_page_that_fails_keeps_what_was_read() -> None:
    respx.get(REED, params={"resultsToSkip": "0"}).mock(
        return_value=httpx.Response(
            200,
            json={"results": [_reed_row(n) for n in range(100)], "totalResults": 300},
        )
    )
    respx.get(REED, params={"resultsToSkip": "100"}).mock(
        return_value=httpx.Response(503)
    )

    jobs, source = await _read(_reed_entry())

    assert len(jobs) == 100
    assert "page 2 failed" in source.truncated


async def test_reed_without_a_key_is_skipped_with_one_clear_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REED_API_KEY", raising=False)

    with pytest.raises(SourceSkipped, match="REED_API_KEY"):
        await _read(_reed_entry(api_key=""))


@respx.mock
async def test_reed_reads_its_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REED_API_KEY", "env-key")
    route = respx.get(REED).mock(
        return_value=httpx.Response(200, json={"results": [], "totalResults": 0})
    )

    await _read(_reed_entry(api_key=""))

    sent = route.calls[0].request.headers["Authorization"]
    assert sent == "Basic " + base64.b64encode(b"env-key:").decode()


async def test_reed_with_no_key_is_a_skipped_source_in_a_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REED_API_KEY", raising=False)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, jobs = await _fetch_one(_reed_entry(api_key=""), fetcher)

    assert report.skipped and not report.ok
    assert jobs == []
    assert "REED_API_KEY" in report.error
    assert "\n" not in report.error


async def test_reed_probe_says_skipped_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REED_API_KEY", raising=False)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        result = await get_source(_reed_entry(api_key=""), fetcher).probe()

    assert result.status is ProbeStatus.SKIPPED


async def test_reed_with_no_queries_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="queries"):
        await _read(_reed_entry(queries=[]))


@pytest.mark.parametrize(
    ("option", "bad"),
    [
        ("max_pages", 0),
        ("max_pages", "five"),
        ("max_pages", True),
        ("results_per_page", 101),
        ("results_per_page", 0),
        ("distance", 0),
        ("graduate", "yes"),
        ("direct_employer_only", 1),
    ],
)
async def test_reed_a_bad_option_is_an_error_that_names_it(
    option: str, bad: Any
) -> None:
    with pytest.raises(ValueError, match=option):
        await _read(_reed_entry(**{option: bad}))


def test_reed_declares_its_dates_are_freshness() -> None:
    assert Reed.dates_are_freshness is True
    assert Reed.name == "reed"


# --- Jooble --------------------------------------------------------------------


def _jooble_row(n: int, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "title": f"Energy <b>Analyst</b> {n}",
        "location": "London, UK",
        "snippet": "Work on &nbsp;power markets with <b>Python</b>...",
        "salary": "",
        "source": "example-board.test",
        "type": "Full-time",
        "link": f"https://jooble.org/desc/{7000 + n}",
        "company": "Acme Energy",
        "updated": "2026-10-08T00:00:00.0000000",
        "id": 7000 + n,
    }
    row.update(over)
    return row


def _jooble_entry(**options: Any) -> SourceEntry:
    base: dict[str, Any] = {
        "kind": "jooble",
        "slug": "gb",
        "label": "Jooble UK",
        "api_key": KEY,
        "queries": ["energy analyst"],
    }
    base.update(options)
    return SourceEntry(**base)


def _jooble_pages(total: int, per_page: int = 20) -> respx.Route:
    """One route answering by the `page` in the request body."""

    def answer(request: httpx.Request) -> httpx.Response:
        page = int(json.loads(request.content)["page"])
        start = (page - 1) * per_page
        rows = [_jooble_row(n) for n in range(start, min(start + per_page, total))]
        return httpx.Response(200, json={"totalCount": total, "jobs": rows})

    return respx.post(JOOBLE).mock(side_effect=answer)


@respx.mock
async def test_jooble_posts_the_search_as_json_with_the_key_in_the_path() -> None:
    route = _jooble_pages(3)

    await _read(_jooble_entry(where="Leeds", radius=25, salary=40000))

    body = json.loads(route.calls[0].request.content)
    assert body == {
        "keywords": "energy analyst",
        "page": 1,
        "location": "Leeds",
        "radius": "25",
        "salary": "40000",
    }


@respx.mock
async def test_jooble_sends_no_option_it_was_not_given() -> None:
    route = _jooble_pages(3)

    await _read(_jooble_entry())

    assert json.loads(route.calls[0].request.content) == {
        "keywords": "energy analyst",
        "page": 1,
    }


@respx.mock
async def test_jooble_reads_every_page_the_total_count_says_there_are() -> None:
    route = _jooble_pages(45)

    jobs, source = await _read(_jooble_entry())

    assert len(jobs) == 45
    assert route.call_count == 3
    assert source.total == 45
    assert source.truncated == ""


@respx.mock
async def test_jooble_stops_on_an_empty_page_whatever_the_total_says() -> None:
    """The documentation gives no page size and a count that drifts, so a page
    with nothing on it is the end."""

    def answer(request: httpx.Request) -> httpx.Response:
        page = int(json.loads(request.content)["page"])
        rows = [_jooble_row(n + 20 * page) for n in range(20)] if page <= 2 else []
        return httpx.Response(200, json={"totalCount": 9999, "jobs": rows})

    route = respx.post(JOOBLE).mock(side_effect=answer)

    jobs, source = await _read(_jooble_entry(max_pages=9))

    assert len(jobs) == 40
    assert route.call_count == 3
    assert source.truncated == ""


@respx.mock
async def test_jooble_stops_at_max_pages_and_says_so() -> None:
    _jooble_pages(500)

    jobs, source = await _read(_jooble_entry(max_pages=2))

    assert len(jobs) == 40
    assert source.total == 500
    assert "40 of 500" in source.truncated and "max_pages 2" in source.truncated


@respx.mock
async def test_jooble_maps_its_fields_onto_a_job() -> None:
    _jooble_pages(1)

    jobs, _ = await _read(_jooble_entry())

    job = jobs[0]
    assert job.source == "jooble:gb"
    assert job.company == "Acme Energy"
    assert job.title == "Energy Analyst 0"
    assert job.location == "London, UK"
    assert job.url == "https://jooble.org/desc/7000"
    assert job.raw_id == "7000"
    assert job.posted == date(2026, 10, 8)
    assert job.description == "Work on power markets with Python..."


@respx.mock
async def test_jooble_names_a_company_it_was_not_given_unknown() -> None:
    respx.post(JOOBLE).mock(
        return_value=httpx.Response(
            200,
            json={
                "totalCount": 2,
                "jobs": [_jooble_row(1, company=""), _jooble_row(2, link="")],
            },
        )
    )

    jobs, source = await _read(_jooble_entry())

    assert [j.company for j in jobs] == ["unknown"]
    assert "title or link" in source.note


@respx.mock
async def test_jooble_error_text_never_carries_the_key() -> None:
    """The key is in the URL path, and a failed request's error is shown in
    the digest and the log."""
    respx.post(JOOBLE).mock(return_value=httpx.Response(403))

    with pytest.raises(FetchError) as caught:
        await _read(_jooble_entry())

    assert KEY not in str(caught.value)
    assert KEY not in caught.value.url
    assert KEY not in repr(caught.value.__cause__)
    assert "jooble.org/api/" in caught.value.url


@respx.mock
async def test_jooble_a_failed_scan_report_and_log_never_carry_the_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(JOOBLE).mock(return_value=httpx.Response(500))
    caplog.set_level(logging.DEBUG, logger="rolescan")

    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, _ = await _fetch_one(_jooble_entry(), fetcher)

    assert report.error and KEY not in report.error
    assert KEY not in caplog.text


@respx.mock
async def test_jooble_a_later_page_that_fails_keeps_what_was_read() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if int(json.loads(request.content)["page"]) == 1:
            return httpx.Response(
                200,
                json={
                    "totalCount": 100,
                    "jobs": [_jooble_row(n) for n in range(20)],
                },
            )
        return httpx.Response(503)

    respx.post(JOOBLE).mock(side_effect=answer)

    jobs, source = await _read(_jooble_entry())

    assert len(jobs) == 20
    assert "page 2 failed" in source.truncated
    assert KEY not in source.truncated


@respx.mock
async def test_jooble_every_query_failing_is_an_error_not_an_empty_source() -> None:
    respx.post(JOOBLE).mock(return_value=httpx.Response(401))

    with pytest.raises(FetchError, match="every Jooble query failed"):
        await _read(_jooble_entry())


async def test_jooble_without_a_key_is_skipped_with_one_clear_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOOBLE_API_KEY", raising=False)

    with pytest.raises(SourceSkipped, match="JOOBLE_API_KEY"):
        await _read(_jooble_entry(api_key=""))


@respx.mock
async def test_jooble_reads_its_key_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOOBLE_API_KEY", "env-key")
    route = respx.post("https://jooble.org/api/env-key").mock(
        return_value=httpx.Response(200, json={"totalCount": 0, "jobs": []})
    )

    await _read(_jooble_entry(api_key=""))

    assert route.called


async def test_jooble_with_no_key_is_a_skipped_source_in_a_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JOOBLE_API_KEY", raising=False)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, jobs = await _fetch_one(_jooble_entry(api_key=""), fetcher)

    assert report.skipped and jobs == []
    assert "JOOBLE_API_KEY" in report.error and "\n" not in report.error


async def test_jooble_with_no_queries_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="queries"):
        await _read(_jooble_entry(queries=[]))


@pytest.mark.parametrize(
    ("option", "bad"),
    [("max_pages", 0), ("radius", 0), ("radius", "wide"), ("salary", -5)],
)
async def test_jooble_a_bad_option_is_an_error_that_names_it(
    option: str, bad: Any
) -> None:
    with pytest.raises(ValueError, match=option):
        await _read(_jooble_entry(**{option: bad}))


def test_jooble_declares_its_dates_are_freshness() -> None:
    assert Jooble.dates_are_freshness is True
    assert Jooble.name == "jooble"
