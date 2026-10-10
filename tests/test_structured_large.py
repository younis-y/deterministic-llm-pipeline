"""`structured` on a large sitemap.

A job board's latest-jobs sitemap lists thousands of URLs. The source already
had `url_pattern`, `exclude_pattern`, `max_pages` and `delay`; what a big
sitemap still needed:

  * robots.txt's `Crawl-delay` as a floor under `delay`;
  * `max_age_days`, which drops a URL whose `lastmod` is too old;
  * `incremental`, which reads only the URLs whose `lastmod` is newer than the
    last scan that read the whole board;
  * `max_sitemap_urls`, which refuses a sitemap too large to be a careers board.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx

from rolescan.config import HTTPConfig, SourceEntry
from rolescan.http import Fetcher, FetchError
from rolescan.models import Job
from rolescan.sources import get_source
from rolescan.sources.base import ProbeStatus
from rolescan.sources.structured import Structured
from rolescan.store import Store

HOST = "https://jobs.example.test"
SITEMAP = f"{HOST}/sitemap.xml"
ROBOTS = f"{HOST}/robots.txt"


def _url(n: int) -> str:
    return f"{HOST}/job/{n}/analyst"


def _page(n: int) -> str:
    return (
        '<html><head><script type="application/ld+json">'
        '{"@context":"http://schema.org","@type":"JobPosting",'
        f'"title":"Analyst {n}","datePosted":"2026-10-01",'
        f'"identifier":{{"@type":"PropertyValue","value":"{n}"}},'
        '"hiringOrganization":{"@type":"Organization","name":"Acme Energy"},'
        '"jobLocation":{"@type":"Place","address":{"@type":"PostalAddress",'
        '"addressLocality":"London","addressCountry":"GB"}},'
        '"description":"Power markets and Python."}'
        "</script></head><body>page</body></html>"
    )


def _sitemap(*entries: tuple[str, str]) -> str:
    urls = "".join(
        f"<url><loc>{loc}</loc>" + (f"<lastmod>{lm}</lastmod>" if lm else "") + "</url>"
        for loc, lm in entries
    )
    return f'<?xml version="1.0"?><urlset>{urls}</urlset>'


def _days_ago(days: int) -> str:
    return (datetime.now(UTC).date() - timedelta(days=days)).isoformat()


def _entry(**over: Any) -> SourceEntry:
    base: dict[str, Any] = {
        "kind": "structured",
        "slug": "acme",
        "label": "Acme Energy",
        "sitemap": SITEMAP,
        "url_pattern": "/job/",
        "delay": 0,
    }
    base.update(over)
    return SourceEntry(**base)


def _mock_pages(*numbers: int) -> dict[int, respx.Route]:
    return {
        n: respx.get(_url(n)).mock(return_value=httpx.Response(200, text=_page(n)))
        for n in numbers
    }


async def _read(
    entry: SourceEntry, *, since: str = "", cache: Store | None = None
) -> tuple[list[Job], Structured]:
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        source = get_source(entry, fetcher, cache)
        assert isinstance(source, Structured)
        if since:
            source.since = since
        return await source.fetch(), source


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("rolescan.sources.structured.asyncio.sleep", fake_sleep)
    return slept


# --- robots.txt Crawl-delay ------------------------------------------------


@respx.mock
async def test_a_crawl_delay_in_robots_txt_is_the_floor_under_delay(
    sleeps: list[float],
) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), "x"), (_url(2), "x"), (_url(3), "x"))
        )
    )
    respx.get(ROBOTS).mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 3\n")
    )
    _mock_pages(1, 2, 3)

    jobs, _ = await _read(_entry(delay=0.5))

    assert len(jobs) == 3
    assert sleeps == [3.0, 3.0]


@respx.mock
async def test_a_longer_configured_delay_still_wins(sleeps: list[float]) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), "x"), (_url(2), "x")))
    )
    respx.get(ROBOTS).mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 1\n")
    )
    _mock_pages(1, 2)

    await _read(_entry(delay=4))

    assert sleeps == [4.0]


@respx.mock
async def test_the_group_for_rolescan_beats_the_wildcard_group(
    sleeps: list[float],
) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), "x"), (_url(2), "x")))
    )
    respx.get(ROBOTS).mock(
        return_value=httpx.Response(
            200,
            text=(
                "User-agent: *\nCrawl-delay: 20\n\n"
                "User-agent: rolescan\nCrawl-delay: 2\n"
            ),
        )
    )
    _mock_pages(1, 2)

    await _read(_entry(delay=0))

    assert sleeps == [2.0]


@respx.mock
@pytest.mark.parametrize(
    "robots",
    [
        httpx.Response(404),
        httpx.Response(200, text="User-agent: *\nDisallow: /admin\n"),
        httpx.Response(200, text="<html>not a robots file</html>"),
    ],
    ids=["missing", "no-crawl-delay", "garbage"],
)
async def test_no_crawl_delay_leaves_delay_alone(
    robots: httpx.Response, sleeps: list[float]
) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), "x"), (_url(2), "x")))
    )
    respx.get(ROBOTS).mock(return_value=robots)
    _mock_pages(1, 2)

    jobs, _ = await _read(_entry(delay=0.5))

    assert len(jobs) == 2
    assert sleeps == [0.5]


@respx.mock
async def test_robots_txt_is_not_asked_for_when_every_page_is_cached(
    tmp_path: Any, sleeps: list[float]
) -> None:
    """A run that goes nowhere near the host costs it nothing, not even a
    robots.txt; and one request needs no gap, so it needs no robots.txt."""
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), "2026-09-01"), (_url(2), "2026-09-01"))
        )
    )
    robots = respx.get(ROBOTS).mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 3\n")
    )
    _mock_pages(1, 2)
    async with Store(tmp_path / "s.db") as store:
        await _read(_entry(), cache=store)
        assert robots.call_count == 1
        await _read(_entry(), cache=store)

    assert robots.call_count == 1, "the second run fetched nothing, so asked nothing"


@respx.mock
async def test_a_crawl_delay_is_asked_once_per_host(sleeps: list[float]) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), "x"), (_url(2), "x"), (_url(3), "x"))
        )
    )
    robots = respx.get(ROBOTS).mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 1\n")
    )
    _mock_pages(1, 2, 3)

    await _read(_entry())

    assert robots.call_count == 1


@respx.mock
async def test_an_absurd_crawl_delay_is_limited_and_says_so(
    sleeps: list[float],
) -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), "x"), (_url(2), "x")))
    )
    respx.get(ROBOTS).mock(
        return_value=httpx.Response(200, text="User-agent: *\nCrawl-delay: 3600\n")
    )
    _mock_pages(1, 2)

    _, source = await _read(_entry())

    assert sleeps == [30.0]
    assert "3600" in source.note and "30" in source.note


# --- max_sitemap_urls --------------------------------------------------------


@respx.mock
async def test_a_sitemap_over_max_sitemap_urls_is_refused_with_a_clear_error() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap(*[(_url(n), "x") for n in range(1, 8)])
        )
    )
    pages = _mock_pages(1)

    with pytest.raises(FetchError) as caught:
        await _read(_entry(max_sitemap_urls=5))

    message = caught.value.detail
    assert "7" in message and "max_sitemap_urls" in message and "5" in message
    assert "url_pattern" in message, "it says how to narrow the read"
    assert pages[1].call_count == 0


@respx.mock
async def test_a_sitemap_at_the_limit_is_read() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap(*[(_url(n), "x") for n in range(1, 4)])
        )
    )
    _mock_pages(1, 2, 3)

    jobs, _ = await _read(_entry(max_sitemap_urls=3))

    assert len(jobs) == 3


def test_the_default_limit_is_fifty_thousand() -> None:
    assert Structured(_entry(), None).max_sitemap_urls == 50_000  # type: ignore[arg-type]


@respx.mock
async def test_probe_reports_an_oversized_sitemap_as_a_failure() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap(*[(_url(n), "x") for n in range(1, 8)])
        )
    )
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        result = await get_source(_entry(max_sitemap_urls=5), fetcher).probe()

    assert result.status is ProbeStatus.FAIL
    assert "max_sitemap_urls" in result.detail


@pytest.mark.parametrize("bad", [0, -1, 2.5, "many", True])
def test_max_sitemap_urls_must_be_a_whole_number_of_at_least_one(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_sitemap_urls"):
        _ = Structured(_entry(max_sitemap_urls=bad), None).max_sitemap_urls  # type: ignore[arg-type]


# --- max_age_days --------------------------------------------------------------


@respx.mock
async def test_urls_with_an_old_lastmod_are_skipped_before_any_fetch() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap(
                (_url(1), _days_ago(2)),
                (_url(2), _days_ago(30)),
                (_url(3), _days_ago(9)),
                (_url(4), ""),
                (_url(5), "not a date"),
            ),
        )
    )
    pages = _mock_pages(1, 2, 3, 4, 5)

    jobs, source = await _read(_entry(max_age_days=10))

    assert sorted(j.raw_id for j in jobs) == ["1", "3", "4", "5"]
    assert pages[2].call_count == 0, "a month-old lastmod is never fetched"
    assert source.total == 4, "the board's count is what is in scope"


@respx.mock
async def test_the_age_limit_includes_the_boundary_day() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), _days_ago(10))))
    )
    _mock_pages(1)

    jobs, _ = await _read(_entry(max_age_days=10))

    assert len(jobs) == 1


@respx.mock
async def test_a_full_timestamp_lastmod_is_read_by_its_date() -> None:
    old = (datetime.now(UTC) - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), old)))
    )
    pages = _mock_pages(1)

    jobs, _ = await _read(_entry(max_age_days=10))

    assert jobs == [] and pages[1].call_count == 0


@pytest.mark.parametrize("bad", [0, -3, 1.5, "ten", False])
def test_max_age_days_must_be_a_whole_number_of_at_least_one(bad: Any) -> None:
    with pytest.raises(ValueError, match="max_age_days"):
        _ = Structured(_entry(max_age_days=bad), None).max_age_days  # type: ignore[arg-type]


@respx.mock
async def test_probe_counts_only_the_urls_inside_max_age_days() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap((_url(1), _days_ago(1)), (_url(2), _days_ago(100))),
        )
    )
    _mock_pages(1, 2)
    async with Fetcher(HTTPConfig(max_retries=0)) as fetcher:
        result = await get_source(_entry(max_age_days=7), fetcher).probe()

    assert result.status is ProbeStatus.OK
    assert result.count == 1


# --- incremental ------------------------------------------------------------


def _since(days_ago: int, fingerprint: str = "") -> str:
    """A stored mark from `days_ago` days back, as the source writes one."""
    when = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    return f"{when}|{fingerprint}" if fingerprint else when


@respx.mock
async def test_incremental_reads_only_urls_newer_than_the_last_whole_run() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap(
                (_url(1), _days_ago(0)),
                (_url(2), _days_ago(1)),
                (_url(3), _days_ago(9)),
                (_url(4), _days_ago(400)),
            ),
        )
    )
    pages = _mock_pages(1, 2, 3, 4)
    entry = _entry(incremental=True)
    source_for_mark = Structured(entry, None)  # type: ignore[arg-type]
    since = f"{_since(5)}|{source_for_mark._fingerprint()}"

    jobs, source = await _read(entry, since=since)

    assert sorted(j.raw_id for j in jobs) == ["1", "2"]
    assert pages[3].call_count == 0 and pages[4].call_count == 0
    assert source.incremental is True
    assert source.total == 4, "the board's own count, not what was new"
    assert source.truncated == ""


@respx.mock
async def test_incremental_keeps_a_days_slack_for_a_run_that_straddled_midnight() -> (
    None
):
    """The mark is when the sitemap was read, and a page can be edited after it
    was read and before the day turned, so the window opens a day early."""
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap((_url(1), _days_ago(6)), (_url(2), _days_ago(7))),
        )
    )
    pages = _mock_pages(1, 2)
    entry = _entry(incremental=True)
    since = f"{_since(5)}|{Structured(entry, None)._fingerprint()}"  # type: ignore[arg-type]

    jobs, _ = await _read(entry, since=since)

    assert [j.raw_id for j in jobs] == ["1"]
    assert pages[2].call_count == 0


@respx.mock
async def test_the_first_incremental_run_reads_everything_in_scope() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap((_url(1), _days_ago(0)), (_url(2), _days_ago(800))),
        )
    )
    _mock_pages(1, 2)

    jobs, source = await _read(_entry(incremental=True))

    assert len(jobs) == 2
    assert source.next_mark, "a whole read leaves a mark for the next run"


@respx.mock
async def test_a_mark_from_other_sitemap_settings_is_ignored() -> None:
    """Changing `url_pattern` brings in URLs the old mark never covered."""
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), _days_ago(0)), (_url(2), _days_ago(300)))
        )
    )
    _mock_pages(1, 2)

    jobs, _ = await _read(_entry(incremental=True), since=_since(1, "deadbeef"))

    assert len(jobs) == 2


@respx.mock
async def test_entries_with_no_lastmod_are_read_and_the_note_says_why() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), ""), (_url(2), _days_ago(300)))
        )
    )
    pages = _mock_pages(1, 2)
    entry = _entry(incremental=True)
    since = f"{_since(2)}|{Structured(entry, None)._fingerprint()}"  # type: ignore[arg-type]

    jobs, source = await _read(entry, since=since)

    assert [j.raw_id for j in jobs] == ["1"]
    assert pages[2].call_count == 0
    assert "no lastmod" in source.note


@respx.mock
async def test_the_mark_is_the_time_the_sitemap_was_read() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(200, text=_sitemap((_url(1), _days_ago(0))))
    )
    _mock_pages(1)
    before = datetime.now(UTC).replace(microsecond=0)

    _, source = await _read(_entry(incremental=True))

    stamp, _, fingerprint = source.next_mark.partition("|")
    assert before <= datetime.fromisoformat(stamp) <= datetime.now(UTC)
    assert fingerprint == source._fingerprint()


@respx.mock
async def test_a_page_that_failed_for_now_holds_the_mark_back() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), _days_ago(0)), (_url(2), _days_ago(0)))
        )
    )
    respx.get(_url(1)).mock(return_value=httpx.Response(200, text=_page(1)))
    respx.get(_url(2)).mock(return_value=httpx.Response(503))

    jobs, source = await _read(_entry(incremental=True))

    assert [j.raw_id for j in jobs] == ["1"]
    assert source.next_mark == "", "page 2 must be tried again next run"
    assert "1" in source.note and "again" in source.note


@respx.mock
async def test_a_page_that_is_gone_does_not_hold_the_mark_back() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200, text=_sitemap((_url(1), _days_ago(0)), (_url(2), _days_ago(0)))
        )
    )
    respx.get(_url(1)).mock(return_value=httpx.Response(200, text=_page(1)))
    respx.get(_url(2)).mock(return_value=httpx.Response(404))

    _, source = await _read(_entry(incremental=True))

    assert source.next_mark != ""


@respx.mock
async def test_a_read_cut_at_max_pages_takes_the_newest_and_says_so() -> None:
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap(
                (_url(1), _days_ago(5)),
                (_url(2), _days_ago(1)),
                (_url(3), _days_ago(3)),
                (_url(4), ""),
            ),
        )
    )
    pages = _mock_pages(1, 2, 3, 4)

    jobs, source = await _read(_entry(incremental=True, max_pages=2))

    assert sorted(j.raw_id for j in jobs) == ["2", "3"]
    assert pages[1].call_count == 0 and pages[4].call_count == 0
    assert source.total == 4
    assert "2 of 4" in source.truncated and "max_pages" in source.truncated


@respx.mock
async def test_a_read_cut_at_max_pages_leaves_no_new_mark() -> None:
    """Postings in the window were not read, and a mark would put them behind
    it for good: the old one stays until a read covers the window."""
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap(
                (_url(1), _days_ago(1)),
                (_url(2), _days_ago(2)),
                (_url(3), _days_ago(3)),
            ),
        )
    )
    _mock_pages(1, 2, 3)

    _, cut = await _read(_entry(incremental=True, max_pages=2))
    _, whole = await _read(_entry(incremental=True, max_pages=3))

    assert cut.truncated and cut.next_mark == ""
    assert not whole.truncated and whole.next_mark != ""


@respx.mock
async def test_without_incremental_the_sitemap_order_and_every_url_are_kept() -> None:
    """The default read is what it was: nothing is dropped for being old and
    nothing is reordered, but a cut at max_pages is now said."""
    respx.get(SITEMAP).mock(
        return_value=httpx.Response(
            200,
            text=_sitemap(
                (_url(1), _days_ago(400)),
                (_url(2), _days_ago(0)),
                (_url(3), _days_ago(200)),
            ),
        )
    )
    pages = _mock_pages(1, 2, 3)

    jobs, source = await _read(_entry(max_pages=2), since=_since(1))

    assert [j.raw_id for j in jobs] == ["1", "2"]
    assert pages[3].call_count == 0
    assert source.next_mark == ""
    assert source.incremental is False
    assert source.total == 3
    assert "2 of 3" in source.truncated


# "true" and "false" as text are read since 2.8.1 (a plugin's string config).
@pytest.mark.parametrize("bad", ["yes", "1", 1, None])
def test_incremental_must_be_true_or_false(bad: Any) -> None:
    with pytest.raises(ValueError, match="incremental"):
        _ = Structured(_entry(incremental=bad), None).incremental  # type: ignore[arg-type]


def test_a_mark_is_a_date_when_it_is_read_back() -> None:
    source = Structured(_entry(incremental=True), None)  # type: ignore[arg-type]
    source.since = f"2026-10-01T09:30:00+00:00|{source._fingerprint()}"
    assert source._since_date() == date(2026, 10, 1)
    source.since = "garbage"
    assert source._since_date() is None
    source.since = ""
    assert source._since_date() is None


# --- options passed as text (2.8.1) ---------------------------------------------
#
# A plugin passes a source's options through from its own config, where every
# value is a string (the workday source has accepted "1000" since 2.5.8). The
# structured options added in 2.6.0 refused "30", so a plugin could not set them.


def _with(**options: Any) -> Structured:
    entry = SourceEntry(
        kind="structured",
        slug="acme",
        label="Acme",
        sitemap="https://jobs.example.test/sitemap.xml",
        **options,
    )
    source = get_source(entry, Fetcher(HTTPConfig()))
    assert isinstance(source, Structured)
    return source


@pytest.mark.parametrize(
    ("name", "text", "number"),
    [("max_age_days", "30", 30), ("max_sitemap_urls", "60000", 60000)],
)
def test_a_whole_number_given_as_text_is_read(
    name: str, text: str, number: int
) -> None:
    assert getattr(_with(**{name: text}), name) == number


@pytest.mark.parametrize("text", ["0", "-3", "thirty", "2.5", ""])
def test_text_that_is_not_a_whole_number_of_at_least_one_is_refused(
    text: str,
) -> None:
    with pytest.raises(ValueError, match="max_age_days"):
        _ = _with(max_age_days=text).max_age_days


@pytest.mark.parametrize(
    ("text", "value"), [("true", True), ("False", False), (" TRUE ", True)]
)
def test_incremental_given_as_text_is_read(text: str, value: bool) -> None:
    assert _with(incremental=text).incremental is value


def test_incremental_text_other_than_true_or_false_is_refused() -> None:
    with pytest.raises(ValueError, match="incremental"):
        _ = _with(incremental="yes").incremental
