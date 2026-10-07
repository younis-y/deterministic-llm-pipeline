"""A posting the judge never reached is `deferred`, never `seen` (2.5.7).

On 2026-09-26 a run had 982 candidates against a 600-call ceiling: 382 were
left on their keyword score, then recorded as seen with everything else, so
they could never surface again. The ceiling now marks them."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from conftest import OLLAMA_MODEL, mock_ollama, scan_and_record
from rolescan.config import Config, OutputConfig
from rolescan.digest import render_html, render_markdown
from rolescan.models import Confidence, FitVerdict, Job, ScoredJob, Verdict
from rolescan.pipeline import ScanResult, _prefilter, _rank, assessed
from rolescan.scoring import FitScorer
from rolescan.store import Store


def _cfg(tmp_path: Path, max_calls: int) -> Config:
    return Config.model_validate(
        {
            "profile": {"keywords": {"energy": 6}, "min_keyword_score": 0},
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "mode": "judge",
                "model": OLLAMA_MODEL,
                "max_calls_per_run": max_calls,
            },
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


def _scored(i: int, score: int) -> ScoredJob:
    job = Job(
        source="t",
        company="Acme",
        title=f"Energy Analyst {i}",
        url=f"https://acme.example/{i}",
        description="energy markets",
    )
    return ScoredJob(job=job, keyword_score=score)


@respx.mock
async def test_the_ceiling_marks_the_weakest_candidate_deferred(tmp_path: Path) -> None:
    mock_ollama()
    cfg = _cfg(tmp_path, max_calls=1)
    scorer = FitScorer(cfg.llm, cfg.profile, None)
    out = await scorer.score_all([_scored(1, 30), _scored(2, 90)])
    by_title = {s.job.title: s for s in out}
    assert by_title["Energy Analyst 2"].fit is not None
    assert by_title["Energy Analyst 2"].deferred == ""
    assert by_title["Energy Analyst 1"].fit is None
    assert by_title["Energy Analyst 1"].deferred == "llm_ceiling"


@respx.mock
async def test_facts_mode_ceiling_defers_without_calling_backend(
    tmp_path: Path,
) -> None:
    """Production default mode (facts) also marks ceiling hits as deferred (2.5.7).

    With max_calls_per_run=0, the ceiling fires before any backend call, so no
    HTTP request is made. The posting is marked deferred and never judged.
    """
    mock_ollama()
    cfg = _cfg(tmp_path, max_calls=0)
    # Reset mode to production default (facts), not overridden to judge.
    cfg.llm.mode = "facts"
    scorer = FitScorer(cfg.llm, cfg.profile, None)
    out = await scorer.score_all([_scored(1, 50)])
    assert out[0].fit is None
    assert out[0].deferred == "llm_ceiling"
    # No backend call was made because the ceiling fired first.
    assert respx.post("http://localhost:11434/api/chat").call_count == 0


def test_deferred_defaults_to_empty() -> None:
    assert _scored(1, 1).deferred == ""


def _pipeline_cfg(tmp_path: Path, **llm: object) -> Config:
    base: dict[str, object] = {"enabled": False}
    base.update(llm)
    return Config.model_validate(
        {
            "profile": {
                "keywords": {"energy": 6, "python": 4},
                "hard_blockers": ["security clearance"],
                "min_keyword_score": 18,
                "min_report_score": 10,
            },
            "llm": base,
            "output": {
                "dir": str(tmp_path),
                "db_path": str(tmp_path / "seen.db"),
                "max_roles": 1,
            },
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


def _board(*jobs: tuple[int, str, str]) -> dict[str, object]:
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
            for jid, title, content in jobs
        ]
    }


async def _seen(tmp_path: Path) -> set[str]:
    async with Store(tmp_path / "seen.db") as store:
        rows = await store.db.execute_fetchall("SELECT title FROM seen")
    return {r[0] for r in rows}


def test_prefilter_defers_a_reject_with_no_description() -> None:
    thin = _scored(1, 0).model_copy(
        update={"job": _scored(1, 0).job.model_copy(update={"description": ""})}
    )
    low = _scored(2, 5)
    high = _scored(3, 50)
    candidates, rejects, deferred = _prefilter([thin, low, high], 18)
    assert [s.job.title for s in candidates] == ["Energy Analyst 3"]
    assert [s.job.title for s in rejects] == ["Energy Analyst 2"]
    assert [s.deferred for s in deferred] == ["thin"]


def test_assessed_leaves_out_every_deferred_posting() -> None:
    judged = [
        _scored(1, 50),
        _scored(2, 40).model_copy(update={"deferred": "llm_ceiling"}),
    ]
    rejects = [_scored(3, 5)]
    kept = assessed(rejects, judged, backend_broke=False)
    assert {s.job.title for s, _ in kept} == {"Energy Analyst 1", "Energy Analyst 3"}


def test_assessed_attaches_the_reason_each_posting_was_recorded_for() -> None:
    """`prefilter` for a keyword reject; for a judged posting the rule that
    fired, else `blocked` for a hard-blocker hit, else plain `judged`."""

    def verdict(rule: str | None) -> FitVerdict:
        return FitVerdict(
            fit_score=60,
            verdict=Verdict.CONSIDER,
            confidence=Confidence.MEDIUM,
            reason="Some overlap.",
            rule=rule,
        )

    ruled = _scored(1, 50).model_copy(update={"fit": verdict("level")})
    blocked = _scored(2, 50).model_copy(
        update={"fit": verdict(None), "blocker_hits": ["security clearance"]}
    )
    plain_judged = _scored(3, 50).model_copy(update={"fit": verdict(None)})
    reject = _scored(4, 5)
    kept = assessed([reject], [ruled, blocked, plain_judged], backend_broke=False)
    assert {s.job.title: reason for s, reason in kept} == {
        "Energy Analyst 4": "prefilter",
        "Energy Analyst 1": "level",
        "Energy Analyst 2": "blocked",
        "Energy Analyst 3": "judged",
    }


def test_rank_returns_the_overflow_marked_digest_cap() -> None:
    cfg = _pipeline_cfg(Path("/tmp"))
    keep, _, overflow = _rank([_scored(1, 60), _scored(2, 50)], cfg)
    assert [s.job.title for s in keep] == ["Energy Analyst 1"]
    assert [s.deferred for s in overflow] == ["digest_cap"]


def test_rank_keeps_the_reason_a_role_was_already_deferred_for() -> None:
    cfg = _pipeline_cfg(Path("/tmp"))
    ceiling = _scored(2, 50).model_copy(update={"deferred": "llm_ceiling"})
    _, _, overflow = _rank([_scored(1, 60), ceiling], cfg)
    assert [s.deferred for s in overflow] == ["llm_ceiling"]


@respx.mock
async def test_roles_past_the_digest_cap_come_back_next_run(tmp_path: Path) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_board(
                (1, "Energy Python Analyst", "energy python"),
                (2, "Energy Analyst", "energy python"),
            ),
        )
    )
    cfg = _pipeline_cfg(tmp_path)
    first = await scan_and_record(cfg)
    assert len(first.reportable) == 1
    assert [s.deferred for s in first.deferred] == ["digest_cap"]
    assert await _seen(tmp_path) == {"Energy Python Analyst"}
    second = await scan_and_record(cfg)
    assert [s.job.title for s in second.reportable] == ["Energy Analyst"]


@respx.mock
async def test_deferred_posting_with_a_term_is_listed_not_recorded(
    tmp_path: Path,
) -> None:
    """The ceiling skipped it AND a hard term matched. It is
    listed under the terms group (so the term can be checked) and still not
    recorded, because the term is checked again next run at no cost."""
    mock_ollama()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_board(
                (1, "Energy Python Analyst", "energy python"),
                (2, "Energy Analyst", "energy python, security clearance needed"),
            ),
        )
    )
    cfg = _pipeline_cfg(
        tmp_path,
        enabled=True,
        backend="ollama",
        mode="judge",
        model=OLLAMA_MODEL,
        max_calls_per_run=1,
    )
    result = await scan_and_record(cfg)
    assert [(s.job.title, s.deferred) for s in result.deferred] == [
        ("Energy Analyst", "llm_ceiling")
    ]
    text = render_markdown(result)
    assert "Energy Analyst" in text and "security clearance" in text
    assert "Energy Analyst" not in await _seen(tmp_path)


@respx.mock
async def test_a_posting_with_no_text_is_deferred_not_recorded(tmp_path: Path) -> None:
    """End to end: a posting whose description never arrived
    and whose title scores nothing is held back, not buried. It is counted in
    the digest and absent from `seen`, so the next run (which may have the
    text) sees it again; the on-topic posting beside it is recorded."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json=_board(
                (1, "Marketing Manager", ""),
                (2, "Energy Python Analyst", "energy python"),
            ),
        )
    )
    result = await scan_and_record(_pipeline_cfg(tmp_path))
    assert "Marketing Manager" not in await _seen(tmp_path)
    assert await _seen(tmp_path) == {"Energy Python Analyst"}
    assert [s.deferred for s in result.deferred] == ["thin"]
    assert "1 without text yet" in render_markdown(result)


def test_stats_line_counts_deferred() -> None:
    result = ScanResult(
        unique=3,
        deferred=[_scored(1, 1).model_copy(update={"deferred": "llm_ceiling"})],
    )
    assert "1 deferred to the next run (1 over the LLM budget)" in render_markdown(
        result
    )


# --- a posting with no text is held back, then listed as unread (2.5.7) ------
#
# "Deferred to the next run" is permanent for a source that never sends text
# (a Workday board with `details: false`, a structured page with no body): the
# same postings were held back on every run and never listed. After
# `output.thin_unread_after` runs a posting is listed once, with its link, and
# recorded.


def _thin_cfg(tmp_path: Path, after: int = 2) -> Config:
    cfg = _pipeline_cfg(tmp_path)
    cfg.output.thin_unread_after = after
    return cfg


def _mock_board(*jobs: tuple[int, str, str]) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_board(*jobs))
    )


async def _deferred_times(tmp_path: Path) -> dict[str, int]:
    async with Store(tmp_path / "seen.db") as store:
        rows = await store.db.execute_fetchall("SELECT uid, times FROM deferred")
    return {r[0]: r[1] for r in rows}


async def _seen_reasons(tmp_path: Path) -> dict[str, str]:
    async with Store(tmp_path / "seen.db") as store:
        rows = await store.db.execute_fetchall("SELECT title, reason FROM seen")
    return {r[0]: r[1] for r in rows}


def test_thin_unread_after_defaults_to_three_and_must_be_positive() -> None:
    assert OutputConfig().thin_unread_after == 3
    with pytest.raises(ValidationError):
        OutputConfig(thin_unread_after=0)


@respx.mock
async def test_a_posting_with_no_text_is_listed_as_unread_on_the_nth_run(
    tmp_path: Path,
) -> None:
    _mock_board((1, "Marketing Manager", ""))
    cfg = _thin_cfg(tmp_path, after=2)

    first = await scan_and_record(cfg)
    assert [s.deferred for s in first.deferred] == ["thin"]
    assert first.unread == []
    assert await _seen_reasons(tmp_path) == {}
    assert list((await _deferred_times(tmp_path)).values()) == [1]
    assert "Unread" not in render_markdown(first)

    second = await scan_and_record(cfg)
    assert [s.job.title for s in second.unread] == ["Marketing Manager"]
    assert second.deferred == [], "an unread posting is no longer deferred"
    assert await _seen_reasons(tmp_path) == {"Marketing Manager": "thin_unread"}
    assert await _deferred_times(tmp_path) == {}, "its count goes once it is recorded"
    text = render_markdown(second)
    assert "## Unread (no text after 2 runs)" in text
    assert "[Marketing Manager](https://boards.greenhouse.io/acme/jobs/1)" in text
    assert "1 listed as unread" in text

    third = await scan_and_record(cfg)
    assert third.unread == [] and third.deferred == []
    assert third.already_seen == 1
    assert "Unread" not in render_markdown(third)


@respx.mock
async def test_the_unread_section_is_in_the_html_digest_too(tmp_path: Path) -> None:
    _mock_board((1, "Marketing Manager", ""))
    cfg = _thin_cfg(tmp_path, after=1)

    result = await scan_and_record(cfg)

    html_text = render_html(result)
    assert "Unread (no text after 1 run)" in html_text
    assert 'href="https://boards.greenhouse.io/acme/jobs/1"' in html_text
    assert "Marketing Manager" in html_text


@respx.mock
async def test_the_unread_section_comes_before_hidden_by_your_rules(
    tmp_path: Path,
) -> None:
    """Both sections can appear in one digest; unread is the one nothing has
    judged, so it comes first."""
    _mock_board(
        (1, "Marketing Manager", ""),
        (2, "Energy Analyst", "energy python, security clearance needed"),
    )
    cfg = _thin_cfg(tmp_path, after=1)
    cfg.output.show_blocked = False

    result = await scan_and_record(cfg)

    text = render_markdown(result)
    assert "## Unread" in text and "## Hidden by your rules" in text
    assert text.index("## Unread") < text.index("## Hidden by your rules")
    html_text = render_html(result)
    assert html_text.index(">Unread") < html_text.index(">Hidden by your rules")


@respx.mock
async def test_a_dry_run_does_not_count_a_text_less_posting(tmp_path: Path) -> None:
    _mock_board((1, "Marketing Manager", ""))
    cfg = _thin_cfg(tmp_path, after=2)

    for _ in range(3):
        result = await scan_and_record(cfg, dry_run=True)
        assert result.unread == []
        assert [s.deferred for s in result.deferred] == ["thin"]

    assert await _deferred_times(tmp_path) == {}
    assert await _seen_reasons(tmp_path) == {}


@respx.mock
async def test_a_posting_that_gains_text_is_judged_and_its_count_is_dropped(
    tmp_path: Path,
) -> None:
    cfg = _thin_cfg(tmp_path, after=3)
    _mock_board((1, "Python Analyst", ""))
    held = await scan_and_record(cfg)
    assert [s.deferred for s in held.deferred] == ["thin"]
    assert list((await _deferred_times(tmp_path)).values()) == [1]

    respx.clear()
    _mock_board((1, "Python Analyst", "energy python"))
    judged = await scan_and_record(cfg)

    assert [s.job.title for s in judged.reportable] == ["Python Analyst"]
    assert judged.unread == []
    assert await _deferred_times(tmp_path) == {}
    assert await _seen_reasons(tmp_path) == {"Python Analyst": "judged"}


@respx.mock
async def test_a_failed_digest_leaves_the_count_in_place(tmp_path: Path) -> None:
    """The count is dropped by `record_scan`, after the digest exists. A run
    whose digest is never written must not lose it, or the posting would
    start again from 1 on every failed run."""
    _mock_board((1, "Marketing Manager", ""))
    cfg = _thin_cfg(tmp_path, after=2)
    await scan_and_record(cfg)

    from rolescan.pipeline import run_scan

    unread = await run_scan(cfg)  # the digest write "fails": no record_scan
    assert [s.job.title for s in unread.unread] == ["Marketing Manager"]
    assert list((await _deferred_times(tmp_path)).values()) == [2]
    again = await run_scan(cfg)
    assert [s.job.title for s in again.unread] == ["Marketing Manager"]


def test_an_unread_posting_that_matched_a_term_says_so() -> None:
    """It is listed under Unread, not the terms group; the term it matched is
    shown so the reader can still check it."""
    hit = _scored(1, 0).model_copy(update={"blocker_hits": ["security clearance"]})
    result = ScanResult(unread=[hit], unread_after=3)
    text = render_markdown(result)
    assert 'blocked by "security clearance"' in text
