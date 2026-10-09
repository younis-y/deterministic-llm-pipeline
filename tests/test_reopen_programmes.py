"""Annual programmes come back (2.7.0): `output.reopen_programme_days`.

`Job.uid` is company, title and location, and `seen` is never trimmed. A
yearly scheme whose title names no year ("Summer Internship", "Graduate
Programme") was therefore listed once and never again, whichever year the
board opened it for. Since 2.5.8 a scan refreshes `seen.last_seen` on every
posting it finds already seen, so a long gap in `last_seen` means the board
stopped listing it. After `reopen_programme_days` days of that gap a posting
whose TITLE names a programme counts as new again.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from conftest import scan_and_record
from rolescan.config import Config, OutputConfig
from rolescan.digest import render_html, render_markdown
from rolescan.models import Job, ScoredJob, is_programme_title
from rolescan.pipeline import ScanResult, run_scan
from rolescan.store import Store

# --- the title pattern -------------------------------------------------------


@pytest.mark.parametrize(
    "title",
    [
        "Summer Internship",
        "Summer Internship 2027",
        "SUMMER INTERNSHIP",
        "Internship, Energy Trading",
        "Data Science Intern",
        "Technology Interns",
        "Graduate Programme",
        "Graduate Program",
        "Graduate Scheme",
        "Graduate Trainee",
        "Graduate Energy Trading Programme",
        "Data Analyst - Graduate Scheme",
        "Off-Cycle Internship",
        "Off Cycle Analyst",
        "Offcycle Placement",
        "Industrial Placement",
        "Placement Year",
        "Summer Analyst",
        "Spring Associate",
        "Winter Internship",
        "Summer Week",
        "Insight Day",
        "Insight Week",
        "Spring Insight Programme",
        "Vacation Scheme",
        "Summer Vacation Scheme 2027",
        "Winter Vacation Programme",
        "Summer Student",
    ],
)
def test_a_programme_title_matches(title: str) -> None:
    assert is_programme_title(title)


@pytest.mark.parametrize(
    "title",
    [
        "International Sales Manager",
        "Internal Audit Manager",
        "Internet Platform Engineer",
        "Data Analyst",
        "Data Analyst (graduate-level degree)",
        "Graduate Data Scientist",
        "Graduate Software Engineer",
        "Displacement Engineer",
        "Springfield Account Manager",
        "Winter Maintenance Operative",
        "Summer Manager",
        "Insight Analyst",
        "Senior Cycle Engineer",
        "Programme Manager",
        "Placement Manager",
        "Graduate Placement Officer",
        "Placement Co-ordinator",
        "",
    ],
)
def test_a_non_programme_title_does_not_match(title: str) -> None:
    assert not is_programme_title(title)


# --- the config key ----------------------------------------------------------


def test_the_default_is_120_days_and_zero_disables() -> None:
    assert OutputConfig().reopen_programme_days == 120
    assert OutputConfig(reopen_programme_days=0).reopen_programme_days == 0


def test_a_negative_gap_is_refused() -> None:
    with pytest.raises(ValidationError):
        OutputConfig(reopen_programme_days=-1)


# --- the store ---------------------------------------------------------------


def _days_ago(n: int) -> str:
    return (datetime.now(UTC) - timedelta(days=n)).isoformat(timespec="seconds")


def _posting(title: str, company: str = "Acme") -> ScoredJob:
    return ScoredJob(
        job=Job(
            source="t",
            company=company,
            title=title,
            location="London",
            url=f"https://acme.example/{company}/{title}".replace(" ", "-"),
        )
    )


async def _age(store: Store, scored: ScoredJob, days: int) -> None:
    await store.db.execute(
        "UPDATE seen SET first_seen = ?, last_seen = ? WHERE uid = ?",
        (_days_ago(days), _days_ago(days), scored.job.uid),
    )
    await store.db.commit()


async def _row(store: Store, scored: ScoredJob) -> tuple[str, str]:
    [(first, last)] = await store.db.execute_fetchall(
        "SELECT first_seen, last_seen FROM seen WHERE uid = ?", (scored.job.uid,)
    )
    return str(first), str(last)


async def test_a_programme_last_seen_200_days_ago_comes_back(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 200)

        fresh = await store.filter_new([programme], reopen_programme_days=120)

    assert [s.job.uid for s in fresh] == [programme.job.uid]
    assert fresh[0].reopened_after_days == 200


async def test_a_programme_listed_30_days_ago_does_not(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 30)

        assert await store.filter_new([programme], reopen_programme_days=120) == []


async def test_the_gap_must_exceed_the_setting(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 119)
        assert await store.filter_new([programme], reopen_programme_days=120) == []
        await _age(store, programme, 121)
        assert len(await store.filter_new([programme], reopen_programme_days=120)) == 1


async def test_a_non_programme_200_days_ago_does_not(tmp_path: Path) -> None:
    ordinary = _posting("Data Analyst")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([ordinary])
        await _age(store, ordinary, 200)

        assert await store.filter_new([ordinary], reopen_programme_days=120) == []


async def test_zero_disables_it_and_so_does_leaving_it_out(tmp_path: Path) -> None:
    programme = _posting("Graduate Programme")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 400)

        assert await store.filter_new([programme], reopen_programme_days=0) == []
        assert await store.filter_new([programme]) == []


async def test_an_unseen_posting_is_new_as_before(tmp_path: Path) -> None:
    programme = _posting("Graduate Programme")
    async with Store(tmp_path / "s.db") as store:
        fresh = await store.filter_new([programme], reopen_programme_days=120)

    assert [s.job.uid for s in fresh] == [programme.job.uid]
    assert fresh[0].reopened_after_days == 0


async def test_a_posting_listed_every_scan_never_comes_back(tmp_path: Path) -> None:
    """The touch is what keeps it off: a board that lists the posting on every
    scan keeps `last_seen` fresh however old `first_seen` is."""
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await store.db.execute("UPDATE seen SET first_seen = ?", (_days_ago(900),))
        await store.db.commit()

        for _ in range(3):
            assert (
                await store.filter_new(
                    [programme], touch=True, reopen_programme_days=120
                )
                == []
            )


async def test_a_reopened_posting_is_not_touched_so_a_deferral_keeps_it_coming(
    tmp_path: Path,
) -> None:
    """Touching it would erase the gap before the posting was recorded: a scan
    that deferred it (model ceiling, digest cap) would lose it."""
    programme, other = _posting("Summer Internship"), _posting("Data Analyst")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme, other])
        await _age(store, programme, 200)
        await _age(store, other, 200)

        fresh = await store.filter_new(
            [programme, other], touch=True, reopen_programme_days=120
        )
        _, programme_last = await _row(store, programme)
        _, other_last = await _row(store, other)

        assert [s.job.title for s in fresh] == ["Summer Internship"]
        assert programme_last < _days_ago(150), "left alone"
        assert other_last > _days_ago(1), "an ordinary posting is touched"
        again = await store.filter_new([programme], reopen_programme_days=120)
        assert len(again) == 1


async def test_recording_it_again_gives_a_fresh_first_seen_and_it_stays_off(
    tmp_path: Path,
) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 200)

        [back] = await store.filter_new([programme], reopen_programme_days=120)
        await store.record_all([(back, "judged")])

        first, last = await _row(store, programme)
        assert first > _days_ago(1)
        assert last > _days_ago(1)
        assert (
            await store.filter_new([programme], touch=True, reopen_programme_days=120)
            == []
        )


async def test_an_ordinary_record_keeps_first_seen(tmp_path: Path) -> None:
    posting = _posting("Data Analyst")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([posting])
        await _age(store, posting, 200)

        await store.record_all([posting])

        first, last = await _row(store, posting)
        assert first < _days_ago(150)
        assert last > _days_ago(1)


async def test_the_application_row_is_untouched(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await store.mark(programme.job.url, "applied", "Acme", "Summer Internship")
        before = await store.db.execute_fetchall("SELECT * FROM applications")
        await _age(store, programme, 200)

        [back] = await store.filter_new([programme], reopen_programme_days=120)
        await store.record_all([(back, "judged")])

        after = await store.db.execute_fetchall("SELECT * FROM applications")
        assert [tuple(r) for r in after] == [tuple(r) for r in before]
        assert await store.application_state(programme.job.url) == "applied"


async def test_a_dry_filter_writes_nothing(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await _age(store, programme, 200)
        before = await _row(store, programme)

        fresh = await store.filter_new([programme], reopen_programme_days=120)

        assert len(fresh) == 1
        assert await _row(store, programme) == before


async def test_order_is_kept_and_only_the_returning_postings_are_marked(
    tmp_path: Path,
) -> None:
    old, ordinary, new = (
        _posting("Summer Internship"),
        _posting("Data Analyst"),
        _posting("Graduate Programme"),
    )
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([old, ordinary])
        await _age(store, old, 200)

        fresh = await store.filter_new([new, old, ordinary], reopen_programme_days=120)

    assert [s.job.title for s in fresh] == ["Graduate Programme", "Summer Internship"]
    assert [s.reopened_after_days for s in fresh] == [0, 200]


async def test_a_row_with_an_unreadable_date_is_left_alone(tmp_path: Path) -> None:
    programme = _posting("Summer Internship")
    async with Store(tmp_path / "s.db") as store:
        await store.record_all([programme])
        await store.db.execute("UPDATE seen SET last_seen = 'not a date'")
        await store.db.commit()

        assert await store.filter_new([programme], reopen_programme_days=120) == []


# --- end to end: a scan ------------------------------------------------------

BOARD_URL = "https://boards-api.greenhouse.io/v1/boards/acme/jobs"


def _cfg(tmp_path: Path, **output: object) -> Config:
    return Config.model_validate(
        {
            "profile": {
                "keywords": {"energy": 6, "python": 4},
                "min_keyword_score": 10,
                "min_report_score": 10,
            },
            "llm": {"enabled": False},
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                **output,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


def _board() -> dict[str, object]:
    rows = [
        (1, "Summer Internship", "energy python energy python"),
        (2, "Energy Python Analyst", "energy python energy python"),
    ]
    return {
        "jobs": [
            {
                "id": jid,
                "title": title,
                "location": {"name": "London, UK"},
                "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                "content": content,
                "updated_at": "2026-10-01T10:00:00Z",
            }
            for jid, title, content in rows
        ]
    }


async def _age_all(tmp_path: Path, days: int) -> None:
    async with Store(tmp_path / "seen.db") as store:
        await store.db.execute(
            "UPDATE seen SET first_seen = ?, last_seen = ?",
            (_days_ago(days), _days_ago(days)),
        )
        await store.db.commit()


def _titles(result: ScanResult) -> list[str]:
    return [s.job.title for s in result.reportable]


@respx.mock
async def test_a_scan_lists_a_returning_programme_once_and_says_so(
    tmp_path: Path,
) -> None:
    respx.get(BOARD_URL).mock(return_value=httpx.Response(200, json=_board()))
    cfg = _cfg(tmp_path)

    first = await scan_and_record(cfg)
    assert sorted(_titles(first)) == ["Energy Python Analyst", "Summer Internship"]

    await _age_all(tmp_path, 200)
    second = await scan_and_record(cfg)

    assert _titles(second) == ["Summer Internship"]
    assert second.already_seen == 1
    assert second.reportable[0].reopened_after_days == 200
    markdown = render_markdown(second)
    assert "listed again after 200 days off the board" in markdown
    assert "listed again after 200 days off the board" in render_html(second)

    third = await scan_and_record(cfg)
    assert _titles(third) == []
    assert third.already_seen == 2
    assert "off the board" not in render_markdown(third)


@respx.mock
async def test_a_dismissed_programme_still_comes_back_with_its_mark_kept(
    tmp_path: Path,
) -> None:
    respx.get(BOARD_URL).mock(return_value=httpx.Response(200, json=_board()))
    cfg = _cfg(tmp_path)
    await scan_and_record(cfg)
    async with Store(tmp_path / "seen.db") as store:
        await store.mark(
            "https://boards.greenhouse.io/acme/jobs/1", "dismissed", "Acme", "x"
        )
    await _age_all(tmp_path, 200)

    result = await scan_and_record(cfg)

    assert _titles(result) == ["Summer Internship"]
    assert "listed again after 200 days off the board" in render_markdown(result)
    async with Store(tmp_path / "seen.db") as store:
        assert (
            await store.application_state("https://boards.greenhouse.io/acme/jobs/1")
            == "dismissed"
        )


@respx.mock
async def test_zero_turns_it_off_for_a_scan(tmp_path: Path) -> None:
    respx.get(BOARD_URL).mock(return_value=httpx.Response(200, json=_board()))
    cfg = _cfg(tmp_path, reopen_programme_days=0)
    await scan_and_record(cfg)
    await _age_all(tmp_path, 400)

    result = await scan_and_record(cfg)

    assert _titles(result) == []
    assert result.already_seen == 2


@respx.mock
async def test_a_dry_scan_lists_it_and_writes_nothing(tmp_path: Path) -> None:
    respx.get(BOARD_URL).mock(return_value=httpx.Response(200, json=_board()))
    cfg = _cfg(tmp_path)
    await scan_and_record(cfg)
    await _age_all(tmp_path, 200)
    async with Store(tmp_path / "seen.db") as store:
        before = await store.db.execute_fetchall(
            "SELECT uid, first_seen, last_seen FROM seen ORDER BY uid"
        )

    result = await run_scan(cfg, dry_run=True)

    assert _titles(result) == ["Summer Internship"]
    assert result.to_record == []
    async with Store(tmp_path / "seen.db") as store:
        after = await store.db.execute_fetchall(
            "SELECT uid, first_seen, last_seen FROM seen ORDER BY uid"
        )
    assert [tuple(r) for r in after] == [tuple(r) for r in before]


@respx.mock
async def test_a_returning_programme_the_digest_cap_deferred_comes_back_next_scan(
    tmp_path: Path,
) -> None:
    """The gap is only closed when the posting is recorded. One the digest cap
    held back keeps its old `last_seen`, so it is still returning next scan."""
    board = {
        "jobs": [
            {
                "id": jid,
                "title": title,
                "location": {"name": "London, UK"},
                "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{jid}",
                "content": "energy python energy python",
                "updated_at": "2026-10-01T10:00:00Z",
            }
            for jid, title in ((1, "Summer Internship"), (2, "Graduate Programme"))
        ]
    }
    respx.get(BOARD_URL).mock(return_value=httpx.Response(200, json=board))
    cfg = _cfg(tmp_path, max_roles=1)
    for _ in range(2):  # the cap defers one of the two, so it takes two scans
        await scan_and_record(cfg)
    await _age_all(tmp_path, 200)

    first = await scan_and_record(cfg)
    second = await scan_and_record(cfg)
    third = await scan_and_record(cfg)

    assert len(_titles(first)) == 1
    assert [s.deferred for s in first.deferred] == ["digest_cap"]
    assert _titles(second) == [s.job.title for s in first.deferred]
    assert set(_titles(first)) | set(_titles(second)) == {
        "Summer Internship",
        "Graduate Programme",
    }
    assert _titles(third) == []
