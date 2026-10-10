"""`llm.max_minutes`: a wall-clock budget for the model, beside the call count (2.7.0).

On a local model one call takes 10 to 40 seconds, so `max_calls_per_run` says
little about how long a scan runs. The time budget counts from the run's first
model call; once it is spent no new call starts, calls already in flight
finish, and every posting not yet scored takes the path the call ceiling
takes: deferred, never written to `seen`, come round again next run.
"""

from __future__ import annotations

import asyncio
import functools
import json
from pathlib import Path

import httpx
import pytest
import respx
from pydantic import ValidationError

from conftest import LLM_VERDICT, OLLAMA_MODEL, mock_ollama_show, scan_and_record
from rolescan import pipeline
from rolescan.config import Config, LLMConfig, ProfileConfig
from rolescan.digest import render_html, render_markdown
from rolescan.models import FitVerdict, Job, ScoredJob
from rolescan.pipeline import ScanResult
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import PostingFacts
from rolescan.scoring.judges import Judge
from rolescan.store import Store

MINUTE = 60.0


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _SlowJudge(Judge):
    """A judge whose every call takes `seconds` of the fake clock."""

    name = "slow-test-judge"

    def __init__(self, clock: _Clock, seconds: float) -> None:
        super().__init__(LLMConfig())
        self.clock = clock
        self.seconds = seconds
        self.calls = 0

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise NotImplementedError

    async def facts(self, system: str, user: str) -> PostingFacts:
        self.calls += 1
        self.clock.now += self.seconds
        return PostingFacts.model_validate({"fit_score": 60, "reason": "fits"})


def _postings(n: int) -> list[ScoredJob]:
    return [
        ScoredJob(
            job=Job(
                source="t",
                company=f"Acme {i}",
                title="Data Analyst",
                url=f"https://acme.example/{i}",
                description="SQL and Python.",
            ),
            keyword_score=100 - i,
        )
        for i in range(n)
    ]


def _scorer(
    *, minutes: float, call_minutes: float, max_calls: int = 60
) -> tuple[FitScorer, _SlowJudge]:
    clock = _Clock()
    cfg = LLMConfig(
        enabled=True,
        backend="ollama",
        max_concurrent=1,
        max_minutes=minutes,
        max_calls_per_run=max_calls,
    )
    scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), clock=clock)
    judge = _SlowJudge(clock, call_minutes * MINUTE)
    scorer._judge = judge
    return scorer, judge


# --- the config key ----------------------------------------------------------


def test_the_budget_defaults_to_no_limit() -> None:
    assert LLMConfig().max_minutes == 0


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(max_minutes=-1)


def test_a_fractional_budget_is_accepted() -> None:
    assert LLMConfig(max_minutes=7.5).max_minutes == 7.5


# --- the scorer --------------------------------------------------------------


async def test_the_call_that_passes_the_budget_scores_and_the_rest_are_deferred() -> (
    None
):
    """Four-minute calls against a ten-minute budget: calls 1 and 2 end at 4
    and 8 minutes, call 3 starts inside the budget and ends at 12, and no
    fourth call starts."""
    scorer, judge = _scorer(minutes=10, call_minutes=4)

    out = await scorer.score_all(_postings(6))

    assert judge.calls == 3
    assert scorer.calls_made == 3
    scored = [s for s in out if s.fit is not None]
    deferred = [s for s in out if s.deferred]
    assert [s.job.company for s in scored] == ["Acme 0", "Acme 1", "Acme 2"]
    assert [s.deferred for s in deferred] == ["llm_time"] * 3
    assert all(s.fit is None for s in deferred)
    record = scorer.run_record()
    assert record is not None
    assert record.calls == 3
    assert record.deferred == 3


async def test_calls_in_flight_when_the_budget_runs_out_finish_and_are_scored() -> None:
    """With several calls at once, the budget can run out while some are still
    running. Those finish and are scored; nothing waiting behind them starts.

    Two slots, four postings, a one-minute budget. The first two calls start
    together and are held open; the clock moves two minutes while they run, so
    the budget is spent with both in flight. Only then are they let go."""
    clock = _Clock()
    cfg = LLMConfig(enabled=True, backend="ollama", max_concurrent=2, max_minutes=1)
    scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), clock=clock)
    started = asyncio.Event()
    release = asyncio.Event()
    in_flight = 0
    calls = 0

    class _Held(Judge):
        name = "held-test-judge"

        def __init__(self) -> None:
            super().__init__(LLMConfig())

        async def verdict(self, system: str, user: str) -> FitVerdict:
            raise NotImplementedError

        async def facts(self, system: str, user: str) -> PostingFacts:
            nonlocal in_flight, calls
            calls += 1
            in_flight += 1
            if in_flight == 2:
                started.set()
            await release.wait()
            in_flight -= 1
            return PostingFacts.model_validate({"fit_score": 60, "reason": "fits"})

    scorer._judge = _Held()

    run = asyncio.ensure_future(scorer.score_all(_postings(4)))
    await asyncio.wait_for(started.wait(), timeout=5)
    assert calls == 2
    clock.now += 2 * MINUTE  # the budget is spent while both calls are running
    release.set()
    out = await asyncio.wait_for(run, timeout=5)

    assert calls == 2, "no call started once the budget was spent"
    assert scorer.calls_made == 2
    assert [s.job.company for s in out if s.fit is not None] == [
        "Acme 0",
        "Acme 1",
    ]
    assert all(s.fit.fit_score == 60 for s in out if s.fit is not None)
    deferred = [s for s in out if s.deferred]
    assert [s.job.company for s in deferred] == ["Acme 2", "Acme 3"]
    assert {s.deferred for s in deferred} == {"llm_time"}
    record = scorer.run_record()
    assert record is not None
    assert (record.calls, record.deferred) == (2, 2)


async def test_a_zero_budget_never_defers_on_time() -> None:
    scorer, judge = _scorer(minutes=0, call_minutes=10_000)

    out = await scorer.score_all(_postings(6))

    assert judge.calls == 6
    assert not any(s.deferred for s in out)


async def test_the_clock_starts_at_the_first_call() -> None:
    """Time spent before the first model call (fetching, cache reads) is not
    the model's: a run whose clock is far from zero still gets its first call."""
    scorer, judge = _scorer(minutes=1, call_minutes=5)
    scorer._clock.now = 1_000_000.0  # type: ignore[attr-defined]

    out = await scorer.score_all(_postings(3))

    assert judge.calls == 1
    assert [s.deferred for s in out] == ["", "llm_time", "llm_time"]


async def test_the_budget_is_spent_when_elapsed_reaches_it() -> None:
    scorer, judge = _scorer(minutes=8, call_minutes=4)

    out = await scorer.score_all(_postings(5))

    assert judge.calls == 2
    assert [s.deferred for s in out].count("llm_time") == 3


async def test_when_the_time_budget_is_hit_first_it_is_the_one_named() -> None:
    scorer, judge = _scorer(minutes=10, call_minutes=4, max_calls=5)

    out = await scorer.score_all(_postings(8))

    assert judge.calls == 3
    assert {s.deferred for s in out if s.deferred} == {"llm_time"}


async def test_when_the_call_ceiling_is_hit_first_it_is_the_one_named() -> None:
    """Call 2 is the last the ceiling allows and starts at minute 6. It ends at
    minute 12, so by the next posting both limits are spent, but the ceiling
    was reached first."""
    scorer, judge = _scorer(minutes=10, call_minutes=6, max_calls=2)

    out = await scorer.score_all(_postings(5))

    assert judge.calls == 2
    assert {s.deferred for s in out if s.deferred} == {"llm_ceiling"}


async def test_the_budget_also_stops_enrichment() -> None:
    """An enricher is a second model call, so it is held to the same budget:
    once the time is spent the plain verdict stands."""
    scorer, _ = _scorer(minutes=1, call_minutes=2)
    started: list[str] = []

    class _Enrich:
        verdict_model = FitVerdict

        async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
            started.append(job.title)
            return verdict.model_copy(update={"reason": "enriched"})

    scorer._enricher = _Enrich()  # type: ignore[assignment]
    scorer._enricher_built = True
    verdict = FitVerdict.model_validate(LLM_VERDICT)
    job = _postings(1)[0].job

    # Control: inside the budget the enricher runs.
    assert (await scorer._maybe_enrich(job, verdict)).reason == "enriched"
    assert started == [job.title]

    started.clear()
    scorer._clock.now += 2 * MINUTE  # type: ignore[attr-defined]

    assert await scorer._maybe_enrich(job, verdict) is verdict
    assert started == []


# --- the digest --------------------------------------------------------------


def _deferred(reason: str, n: int) -> list[ScoredJob]:
    return [s.model_copy(update={"deferred": reason}) for s in _postings(n)]


def test_the_digest_names_the_time_budget() -> None:
    result = ScanResult(
        unique=9, llm_max_minutes=10.0, deferred=_deferred("llm_time", 3)
    )

    text = render_markdown(result)

    assert (
        "model time budget of 10 minutes spent; 3 postings deferred to the next run"
        in text
    )
    assert "over the LLM budget" not in text
    assert "3 deferred to the next run" not in text


def test_the_digest_keeps_the_call_count_wording_for_the_ceiling() -> None:
    result = ScanResult(
        unique=9, llm_max_minutes=10.0, deferred=_deferred("llm_ceiling", 3)
    )

    text = render_markdown(result)

    assert "3 deferred to the next run (3 over the LLM budget)" in text
    assert "model time budget" not in text


def test_the_digest_says_one_minute_and_one_posting() -> None:
    result = ScanResult(
        unique=9, llm_max_minutes=1.0, deferred=_deferred("llm_time", 1)
    )

    assert (
        "model time budget of 1 minute spent; 1 posting deferred to the next run"
        in render_markdown(result)
    )


def test_the_digest_prints_a_fractional_budget_plainly() -> None:
    result = ScanResult(
        unique=9, llm_max_minutes=7.5, deferred=_deferred("llm_time", 2)
    )

    assert "model time budget of 7.5 minutes spent" in render_markdown(result)


def test_the_time_clause_sits_beside_other_deferrals_without_double_counting() -> None:
    result = ScanResult(
        unique=9,
        llm_max_minutes=10.0,
        deferred=_deferred("llm_time", 2) + _deferred("digest_cap", 1),
    )

    text = render_markdown(result)

    assert "model time budget of 10 minutes spent; 2 postings deferred" in text
    assert "1 deferred to the next run (1 over the digest cap)" in text


def test_the_html_digest_carries_the_same_wording() -> None:
    result = ScanResult(
        unique=9, llm_max_minutes=10.0, deferred=_deferred("llm_time", 3)
    )

    assert (
        "model time budget of 10 minutes spent; 3 postings deferred to the next run"
        in render_html(result)
    )


# --- end to end: a deferred posting is never written to `seen` ----------------


def _cfg(tmp_path: Path, minutes: float) -> Config:
    return Config.model_validate(
        {
            "profile": {
                "keywords": {"energy": 6, "python": 4},
                "min_keyword_score": 0,
                "min_report_score": 10,
            },
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "mode": "judge",
                "model": OLLAMA_MODEL,
                "max_concurrent": 1,
                "cascade": False,
                "max_minutes": minutes,
            },
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


@respx.mock
async def test_postings_deferred_on_time_are_not_recorded_and_come_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = _Clock()
    monkeypatch.setattr(
        pipeline, "FitScorer", functools.partial(FitScorer, clock=clock)
    )
    titles = [f"Energy Python Analyst {i}" for i in range(5)]
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 100 + i,
                        "title": t,
                        "location": {"name": "London, UK"},
                        "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{100 + i}",
                        "content": "energy python",
                    }
                    for i, t in enumerate(titles)
                ]
            },
        )
    )

    def _answer(request: httpx.Request) -> httpx.Response:
        clock.now += 4 * MINUTE
        return httpx.Response(
            200, json={"message": {"content": json.dumps(LLM_VERDICT)}}
        )

    respx.post("http://localhost:11434/api/chat").mock(side_effect=_answer)
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": OLLAMA_MODEL}]})
    )
    mock_ollama_show()

    result = await scan_and_record(_cfg(tmp_path, minutes=10))

    assert result.llm_calls == 3
    assert [s.deferred for s in result.deferred] == ["llm_time"] * 2
    assert "model time budget of 10 minutes spent; 2 postings deferred" in (
        render_markdown(result)
    )
    async with Store(tmp_path / "seen.db") as store:
        rows = await store.db.execute_fetchall("SELECT title FROM seen")
    recorded = {str(r[0]) for r in rows}
    assert len(recorded) == 3
    assert recorded.isdisjoint({s.job.title for s in result.deferred})

    # The next run, with the budget gone, scores exactly the two it left.
    clock.now = 0.0
    second = await scan_and_record(_cfg(tmp_path, minutes=0))
    assert second.already_seen == 3
    assert second.llm_calls == 2
    assert not second.deferred
