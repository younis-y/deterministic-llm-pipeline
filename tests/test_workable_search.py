"""The `workable_search` source (2.6.0): Workable's cross-employer job search.

`https://jobs.workable.com/api/v1/jobs` takes `query` and `location` and pages
with a token. The shape below is what one live request showed on 2026-10-09
(status 200; top-level keys `autoAppliedFilters`, `jobs`, `nextPageToken`,
`title`, `totalSize`), with every value invented. That one request did not show
the name of the paging parameter, which is taken from the service's own site
(`pageToken`) and is the first thing to check against a second page of a real
answer.
"""

from __future__ import annotations

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
from rolescan.sources.aggregators import Jooble, Reed, WorkableSearch
from rolescan.sources.base import ProbeStatus

URL = "https://jobs.workable.com/api/v1/jobs"


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every pause the source asks for, recorded and not slept."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("rolescan.sources.aggregators.asyncio.sleep", fake_sleep)
    return slept


def _job(n: int, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "department": "Data",
        "id": f"1A2B3C4D5E{n:02d}",
        "title": f"Data Analyst {n}",
        "state": "published",
        "description": "<p>Analyse <b>energy</b> markets &amp; build dashboards.</p>",
        "socialSharingDescription": "Analyse energy markets.",
        "employmentType": "Full time",
        "benefitsSection": "<ul><li>Pension</li></ul>",
        "requirementsSection": "<p>Python and SQL.</p>",
        "url": f"https://jobs.workable.com/view/1A2B3C4D5E{n:02d}/data-analyst-{n}",
        "language": "en",
        "locations": ["London, England, United Kingdom"],
        "location": {
            "city": "London",
            "subregion": "England",
            "countryName": "United Kingdom",
        },
        "created": "2026-10-08T09:30:00.000Z",
        "updated": "2026-10-09T07:00:00.000Z",
        "company": {
            "id": "c-0001",
            "title": "Acme Energy",
            "website": "https://acme.example.test",
            "image": "https://img.example.test/acme.png",
            "description": "<p>An energy company.</p>",
            "url": "https://apply.workable.com/acme-energy",
            "socialSharingImage": "https://img.example.test/acme-share.png",
            "socialSharingDescription": "Acme.",
        },
        "isFeatured": False,
        "workplace": "hybrid",
    }
    row.update(over)
    return row


def _answer(rows: list[dict[str, Any]], total: int, token: str = "") -> httpx.Response:
    body: dict[str, Any] = {
        "title": "Data Analyst Jobs",
        "totalSize": total,
        "jobs": rows,
        "autoAppliedFilters": {},
    }
    if token:
        body["nextPageToken"] = token
    return httpx.Response(200, json=body)


def _serve(*pages: tuple[list[dict[str, Any]], int, str]) -> respx.Route:
    """Answer page n of a search: the first on no token, page n on `tok-n`.

    A page is (jobs, totalSize, the next page's token or "")."""

    def answer(request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("pageToken")
        index = 0 if token is None else int(token.split("-")[1]) - 1
        if index >= len(pages):
            return httpx.Response(500)
        rows, total, following = pages[index]
        return _answer(rows, total, following)

    return respx.get(URL).mock(side_effect=answer)


def _entry(**options: Any) -> SourceEntry:
    base: dict[str, Any] = {
        "kind": "workable_search",
        "slug": "gb",
        "label": "Workable search",
        "queries": ["data analyst"],
    }
    base.update(options)
    return SourceEntry(**base)


async def _read(entry: SourceEntry) -> tuple[list[Job], Any]:
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        source = get_source(entry, fetcher)
        return await source.fetch(), source


@respx.mock
async def test_it_asks_with_the_query_and_the_place_and_no_key() -> None:
    route = respx.get(URL).mock(return_value=_answer([_job(1)], 1))

    await _read(_entry(where="London"))

    request = route.calls[0].request
    assert dict(request.url.params) == {"query": "data analyst", "location": "London"}
    assert "Authorization" not in request.headers


@respx.mock
async def test_it_sends_no_place_it_was_not_given() -> None:
    route = respx.get(URL).mock(return_value=_answer([_job(1)], 1))

    await _read(_entry())

    assert dict(route.calls[0].request.url.params) == {"query": "data analyst"}


@respx.mock
async def test_it_maps_a_job_onto_the_posting_model() -> None:
    respx.get(URL).mock(return_value=_answer([_job(7)], 1))

    jobs, _ = await _read(_entry())

    job = jobs[0]
    assert job.source == "workable_search:gb"
    assert job.company == "Acme Energy"
    assert job.title == "Data Analyst 7"
    assert job.location == "London, England, United Kingdom"
    assert job.url == "https://jobs.workable.com/view/1A2B3C4D5E07/data-analyst-7"
    assert job.raw_id == "1A2B3C4D5E07"
    assert job.posted == date(2026, 10, 8)
    assert job.remote is False
    assert "Analyse energy markets & build dashboards." in job.description
    assert "Python and SQL." in job.description
    assert "<" not in job.description


@respx.mock
async def test_a_remote_workplace_is_remote() -> None:
    respx.get(URL).mock(
        return_value=_answer(
            [
                _job(1, workplace="remote"),
                _job(2, workplace="on_site"),
                _job(3, workplace="on_site", title="Remote Data Analyst"),
            ],
            3,
        )
    )

    jobs, _ = await _read(_entry())

    assert [j.remote for j in jobs] == [True, False, True]


@respx.mock
async def test_a_job_with_a_partial_location_still_has_one() -> None:
    respx.get(URL).mock(
        return_value=_answer(
            [
                _job(1, location={"countryName": "Ireland"}),
                _job(2, location={}, locations=["Remote, Germany"]),
                _job(3, location=None, locations=[]),
            ],
            3,
        )
    )

    jobs, _ = await _read(_entry())

    assert [j.location for j in jobs] == ["Ireland", "Remote, Germany", ""]


@respx.mock
async def test_it_follows_the_page_token_until_there_is_none() -> None:
    route = _serve(
        ([_job(n) for n in range(1, 4)], 5, "tok-2"),
        ([_job(n) for n in range(4, 6)], 5, ""),
    )

    jobs, source = await _read(_entry())

    assert [j.raw_id for j in jobs] == [f"1A2B3C4D5E{n:02d}" for n in range(1, 6)]
    first, second = (c.request.url.params for c in route.calls)
    assert "pageToken" not in first
    assert second["pageToken"] == "tok-2"
    assert second["query"] == "data analyst", (
        "the search is asked again, with the token"
    )
    assert source.total == 5
    assert source.truncated == ""


@respx.mock
async def test_it_stops_at_max_pages_and_says_so() -> None:
    route = _serve(
        ([_job(n) for n in range(1, 4)], 90, "tok-2"),
        ([_job(n) for n in range(4, 7)], 90, "tok-3"),
        ([_job(n) for n in range(7, 10)], 90, "tok-4"),
    )

    jobs, source = await _read(_entry(max_pages=2))

    assert len(jobs) == 6
    assert route.call_count == 2
    assert source.total == 90
    assert "6 of 90" in source.truncated and "max_pages 2" in source.truncated


@respx.mock
async def test_an_empty_page_ends_the_read_whatever_the_token_says() -> None:
    route = _serve(([_job(1)], 9, "tok-2"), ([], 9, "tok-3"))

    jobs, source = await _read(_entry(max_pages=9))

    assert len(jobs) == 1
    assert route.call_count == 2
    assert source.truncated == ""


@respx.mock
async def test_it_stops_when_the_stated_total_is_reached() -> None:
    route = _serve(([_job(1)], 1, "tok-2"))

    jobs, _ = await _read(_entry())

    assert len(jobs) == 1
    assert route.call_count == 1


@respx.mock
async def test_it_stops_when_there_is_no_token_even_if_the_total_is_larger() -> None:
    route = _serve(([_job(1), _job(2)], 40, ""))

    jobs, source = await _read(_entry())

    assert len(jobs) == 2
    assert route.call_count == 1
    assert source.truncated == "", "the service said that was the last page"


@respx.mock
async def test_a_later_page_that_fails_keeps_what_was_read() -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if "pageToken" in request.url.params:
            return httpx.Response(503)
        return _answer([_job(1), _job(2)], 40, token="tok-2")

    respx.get(URL).mock(side_effect=answer)

    jobs, source = await _read(_entry())

    assert len(jobs) == 2
    assert "page 2 failed" in source.truncated


@respx.mock
async def test_every_query_failing_is_an_error_not_an_empty_source() -> None:
    respx.get(URL).mock(return_value=httpx.Response(500))

    with pytest.raises(FetchError, match="every Workable search query failed"):
        await _read(_entry())


@respx.mock
async def test_it_skips_a_row_it_cannot_use_and_says_so() -> None:
    respx.get(URL).mock(
        return_value=_answer([_job(1), _job(2, url=""), _job(3, title="")], 3)
    )

    jobs, source = await _read(_entry())

    assert len(jobs) == 1
    assert "2" in source.note and "title or link" in source.note


@respx.mock
async def test_two_queries_are_two_searches() -> None:
    for query in ("data analyst", "energy trader"):
        respx.get(URL, params={"query": query}).mock(
            return_value=_answer([_job(len(query))], 1)
        )

    jobs, source = await _read(_entry(queries=["data analyst", "energy trader"]))

    assert len(jobs) == 2
    assert source.total is None, "two queries overlap, so none is claimed"


@respx.mock
async def test_it_needs_no_key_and_runs_in_a_scan() -> None:
    respx.get(URL).mock(return_value=_answer([_job(1)], 1))

    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, jobs = await _fetch_one(_entry(), fetcher)

    assert report.ok and not report.skipped
    assert report.count == 1 and report.total == 1
    assert len(jobs) == 1


@respx.mock
async def test_probe_reports_the_count_it_read() -> None:
    respx.get(URL).mock(return_value=_answer([_job(1), _job(2)], 2))

    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        result = await get_source(_entry(), fetcher).probe()

    assert result.status is ProbeStatus.OK and result.count == 2


async def test_no_queries_is_an_error_not_a_silent_zero() -> None:
    with pytest.raises(ValueError, match="queries"):
        await _read(_entry(queries=[]))


@pytest.mark.parametrize("bad", [0, -1, "five", True, 2.5])
async def test_a_bad_max_pages_is_an_error_that_names_it(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_pages"):
        await _read(_entry(max_pages=bad))


def test_it_declares_its_dates_are_freshness() -> None:
    assert WorkableSearch.dates_are_freshness is True
    assert WorkableSearch.name == "workable_search"


# --- a paging parameter that does not page --------------------------------------


@respx.mock
async def test_it_stops_when_the_service_gives_back_the_token_it_was_sent() -> None:
    """A wrong parameter name, or a service that does not move on, answers the
    same token again; following it would repeat until max_pages."""
    calls = 0

    def answer(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        rows = [_job(calls * 10 + 1), _job(calls * 10 + 2)]
        return _answer(rows, 90, "tok-stuck")

    respx.get(URL).mock(side_effect=answer)

    jobs, source = await _read(_entry(max_pages=9))

    assert calls == 2, "page 1, then page 2 under tok-stuck, then no more"
    assert len(jobs) == 4, "both pages were new, so both are kept"
    assert "same page token" in source.truncated
    assert "4" in source.truncated


@respx.mock
async def test_it_stops_when_a_page_starts_with_a_posting_already_read() -> None:
    """The parameter is ignored and the token changes each time: every answer
    is the first page again. The repeat is dropped, not listed twice."""
    calls = 0

    def answer(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _answer([_job(1), _job(2)], 90, f"tok-{calls + 1}")

    respx.get(URL).mock(side_effect=answer)

    jobs, source = await _read(_entry(max_pages=9))

    assert calls == 2
    assert [j.raw_id for j in jobs] == ["1A2B3C4D5E01", "1A2B3C4D5E02"]
    assert "repeated" in source.truncated and "paging parameter" in source.truncated


@respx.mock
async def test_a_genuine_second_page_is_not_mistaken_for_a_repeat() -> None:
    route = _serve(
        ([_job(1), _job(2)], 4, "tok-2"),
        ([_job(3), _job(4)], 4, ""),
    )

    jobs, source = await _read(_entry())

    assert len(jobs) == 4 and route.call_count == 2
    assert source.truncated == ""


# --- one request at a time, with a pause ------------------------------------------


@respx.mock
async def test_queries_run_one_after_another_with_a_two_second_pause(
    sleeps: list[float],
) -> None:
    order: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        order.append(f"{query}:{request.url.params.get('pageToken', '-')}")
        n = len(order)
        return _answer([_job(n)], 2, "" if "pageToken" in request.url.params else "t")

    respx.get(URL).mock(side_effect=answer)

    jobs, _ = await _read(_entry(queries=["alpha", "beta"]))

    assert len(jobs) == 4
    assert order == ["alpha:-", "alpha:t", "beta:-", "beta:t"], "not interleaved"
    assert sleeps == [2.0, 2.0, 2.0], "between pages and between queries"


@respx.mock
async def test_the_pause_is_the_delay_option_and_zero_turns_it_off(
    sleeps: list[float],
) -> None:
    respx.get(URL).mock(return_value=_answer([_job(1)], 1))

    await _read(_entry(queries=["alpha", "beta"], delay=0.5))
    assert sleeps == [0.5]

    sleeps.clear()
    await _read(_entry(queries=["alpha", "beta"], delay=0))
    assert sleeps == []


@respx.mock
async def test_queries_written_as_one_plain_value_is_one_search() -> None:
    """Read as a list of characters it was one search per letter, each a
    request."""
    route = respx.get(URL).mock(return_value=_answer([_job(1)], 1))

    await _read(_entry(queries="data analyst"))

    assert route.call_count == 1
    assert route.calls[0].request.url.params["query"] == "data analyst"


@respx.mock
async def test_a_failed_query_does_not_stop_the_next_one() -> None:
    respx.get(URL, params={"query": "alpha"}).mock(return_value=httpx.Response(500))
    respx.get(URL, params={"query": "beta"}).mock(return_value=_answer([_job(2)], 1))

    jobs, _ = await _read(_entry(queries=["alpha", "beta"]))

    assert [j.raw_id for j in jobs] == ["1A2B3C4D5E02"]


@pytest.mark.parametrize("bad", [-1, "slow", True, float("nan"), float("inf")])
async def test_a_bad_delay_is_an_error_that_names_it(bad: Any) -> None:
    with pytest.raises(ValueError, match="delay"):
        await _read(_entry(delay=bad))


def test_the_keyed_aggregators_still_run_their_queries_together() -> None:
    assert WorkableSearch.default_delay == 2.0
    assert Reed.default_delay is None and Jooble.default_delay is None
