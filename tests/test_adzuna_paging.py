"""Adzuna paging.

`search/1` was hard-coded, so one entry with six queries read at most 300
rows however many results the API said it held, and the digest said nothing:
a read cut at a page and a small result set were the same number. The source
now reads the response's `count`, fetches further pages up to `max_pages`
(default 5, at `results_per_page` 50), stops when a page comes back short,
and reports `total` and `truncated` like the other sources (2.5.8)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.pipeline import _check_coverage, _fetch_one
from rolescan.sources import get_source
from rolescan.sources.adzuna import Adzuna
from rolescan.store import Store

BASE = "https://api.adzuna.com/v1/api/jobs/gb/search"


def _row(n: int, query: str = "energy") -> dict[str, Any]:
    return {
        "id": f"{query}-{n}",
        "title": f"Energy Analyst {n}",
        "company": {"display_name": "Acme"},
        "location": {"display_name": "London, UK"},
        "redirect_url": f"https://jobs.example.test/{query}/{n}",
        "description": "Power markets",
        "created": "2026-08-19T00:00:00Z",
    }


def _page(
    start: int, size: int, count: int | None, query: str = "energy"
) -> httpx.Response:
    body: dict[str, Any] = {"results": [_row(start + i, query) for i in range(size)]}
    if count is not None:
        body["count"] = count
    return httpx.Response(200, json=body)


def _entry(**options: Any) -> SourceEntry:
    base: dict[str, Any] = {
        "kind": "adzuna",
        "slug": "gb",
        "label": "Adzuna UK",
        "app_id": "id",
        "app_key": "key",
        "queries": ["energy"],
    }
    base.update(options)
    return SourceEntry(**base)


async def _read(entry: SourceEntry) -> tuple[list[Job], Adzuna]:
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        source = get_source(entry, fetcher)
        assert isinstance(source, Adzuna)
        return await source.fetch(), source


def _mock_pages(count: int, per_page: int = 50, query: str = "energy") -> list[Any]:
    """One route per page 1..9 over `count` results, `per_page` rows a page."""
    routes = []
    for page in range(1, 10):
        start = (page - 1) * per_page
        size = max(0, min(per_page, count - start))
        routes.append(
            respx.get(f"{BASE}/{page}").mock(return_value=_page(start, size, count))
        )
    return routes


@respx.mock
async def test_it_reads_every_page_the_count_says_there_are() -> None:
    routes = _mock_pages(120)
    jobs, source = await _read(_entry())
    assert len(jobs) == 120
    assert [r.called for r in routes[:4]] == [True, True, True, False]
    assert source.total == 120
    assert source.truncated == ""


@respx.mock
async def test_a_short_first_page_is_the_only_request() -> None:
    routes = _mock_pages(10)
    jobs, source = await _read(_entry())
    assert len(jobs) == 10
    assert routes[0].called and not routes[1].called
    assert source.total == 10 and source.truncated == ""


@respx.mock
async def test_it_stops_at_max_pages_and_says_how_much_it_left() -> None:
    routes = _mock_pages(1000)
    jobs, source = await _read(_entry(max_pages=2))
    assert len(jobs) == 100
    assert [r.called for r in routes[:3]] == [True, True, False]
    assert source.total == 1000
    assert "energy" in source.truncated
    assert "100 of 1000" in source.truncated
    assert "max_pages 2" in source.truncated


@respx.mock
async def test_max_pages_defaults_to_five() -> None:
    routes = _mock_pages(1000)
    jobs, _ = await _read(_entry())
    assert len(jobs) == 250
    assert [r.called for r in routes[:6]] == [True] * 5 + [False]


@respx.mock
async def test_results_per_page_is_sent_and_decides_when_a_page_is_short() -> None:
    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.params["results_per_page"])
        page = int(request.url.path.rsplit("/", 1)[1])
        # 20 + 20 + 7 rows: the third page is short, so it is the last.
        sizes = {1: 20, 2: 20, 3: 7}
        return _page((page - 1) * 20, sizes.get(page, 0), 47)

    respx.get(url__regex=rf"{BASE}/\d+").mock(side_effect=answer)
    jobs, source = await _read(_entry(results_per_page=20))
    assert len(jobs) == 47
    assert sent == ["20", "20", "20"]
    assert source.truncated == ""


@respx.mock
async def test_the_default_page_size_is_fifty() -> None:
    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.params["results_per_page"])
        return _page(0, 3, 3)

    respx.get(f"{BASE}/1").mock(side_effect=answer)
    await _read(_entry())
    assert sent == ["50"]


@respx.mock
async def test_a_full_page_with_no_count_is_read_to_the_cap_and_flagged() -> None:
    for page in range(1, 4):
        respx.get(f"{BASE}/{page}").mock(
            return_value=_page((page - 1) * 50, 50, count=None)
        )
    jobs, source = await _read(_entry(max_pages=3))
    assert len(jobs) == 150
    assert source.total is None
    assert "max_pages 3" in source.truncated


@respx.mock
async def test_a_page_that_fails_after_the_first_keeps_what_was_read() -> None:
    respx.get(f"{BASE}/1").mock(return_value=_page(0, 50, 500))
    respx.get(f"{BASE}/2").mock(return_value=httpx.Response(500))
    jobs, source = await _read(_entry())
    assert len(jobs) == 50
    assert "page 2" in source.truncated and "energy" in source.truncated
    assert source.total == 500


@respx.mock
async def test_a_failing_first_page_still_fails_the_source() -> None:
    respx.get(f"{BASE}/1").mock(return_value=httpx.Response(500))
    with pytest.raises(FetchError, match="every Adzuna query failed"):
        await _read(_entry())


@respx.mock
async def test_several_queries_claim_no_single_total_and_name_the_cut_one() -> None:
    """Queries overlap, so no one count is the board's: as Workday does with
    several search texts, the source claims none."""

    def answer(request: httpx.Request) -> httpx.Response:
        query = request.url.params["what"]
        page = int(request.url.path.rsplit("/", 1)[1])
        count = 400 if query == "big" else 30
        start = (page - 1) * 50
        return _page(start, max(0, min(50, count - start)), count, query)

    respx.get(url__regex=rf"{BASE}/\d+").mock(side_effect=answer)
    jobs, source = await _read(_entry(queries=["big", "small"], max_pages=2))
    assert len(jobs) == 100 + 30
    assert source.total is None
    assert '"big"' in source.truncated and '"small"' not in source.truncated


@respx.mock
async def test_one_query_one_total() -> None:
    _mock_pages(75)
    _, source = await _read(_entry())
    assert source.total == 75


@respx.mock
async def test_a_second_read_does_not_carry_the_first_reads_report() -> None:
    _mock_pages(1000)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        source = get_source(_entry(max_pages=1), fetcher)
        await source.fetch()
        assert source.truncated
    respx.clear()
    _mock_pages(5)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        again = get_source(_entry(max_pages=1), fetcher)
        await again.fetch()
        assert again.truncated == "" and again.total == 5


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_pages": 0}, "max_pages"),
        ({"max_pages": "many"}, "max_pages"),
        ({"max_pages": True}, "max_pages"),
        ({"results_per_page": 0}, "results_per_page"),
        ({"results_per_page": 51}, "results_per_page"),
        ({"results_per_page": "fifty"}, "results_per_page"),
    ],
)
async def test_a_bad_paging_option_is_an_error_that_names_it(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await _read(_entry(**options))


@respx.mock
async def test_the_pipeline_carries_total_and_truncated_onto_the_report() -> None:
    _mock_pages(1000)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, jobs = await _fetch_one(_entry(max_pages=2), fetcher)
    assert report.ok and report.count == len(jobs) == 100
    assert report.total == 1000
    assert "100 of 1000" in report.truncated


@respx.mock
async def test_a_cut_adzuna_read_raises_the_cut_short_alarm(tmp_path: Any) -> None:
    _mock_pages(1000)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        report, _ = await _fetch_one(_entry(max_pages=2), fetcher)
    async with Store(tmp_path / "s.db") as store:
        found = await _check_coverage([report], store)
    assert found.truncated
    ((label, read, total, why),) = found.truncated
    assert (label, read, total) == ("Adzuna UK", 100, 1000) and "max_pages" in why


@respx.mock
async def test_the_parsed_rows_are_unchanged() -> None:
    respx.get(f"{BASE}/1").mock(return_value=_page(7, 1, 1))
    jobs, _ = await _read(_entry())
    (job,) = jobs
    assert job.source == "adzuna:gb"
    assert job.company == "Acme" and job.title == "Energy Analyst 7"
    assert job.url == "https://jobs.example.test/energy/7"
