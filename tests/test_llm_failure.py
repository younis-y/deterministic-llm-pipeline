"""A run whose LLM scoring failed says so in its exit status (2.5.8).

Measured 2026-10-07: with Ollama down, `rolescan scan` printed "LLM scoring
did not run" and exited 0, so launchd or any scheduler saw a success;
with a runner that accepted the connection and never answered, each posting
waited out `llm.timeout`, so 250 candidates at 120 s took 8.3 hours to end
with every one errored. Now: five failures in a row stop the calls and defer
the rest, and a failed run exits 3."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from conftest import mock_ollama_show, plain
from rolescan import cli
from rolescan.cli import app
from rolescan.config import LLMConfig, ProfileConfig
from rolescan.digest import render_markdown
from rolescan.models import FitVerdict, Job, ScoredJob
from rolescan.pipeline import ScanResult
from rolescan.scoring import FitScorer
from rolescan.scoring import enrich as enrich_mod
from rolescan.scoring.enrich import Enricher, register_enricher
from rolescan.scoring.facts import PostingFacts
from rolescan.scoring.judges import Judge
from rolescan.scoring.llm import BREAKER_AFTER
from rolescan.store import Store


class _Scripted(Judge):
    """A judge whose calls succeed or fail in a given order, then fail."""

    name = "scripted-test-judge"

    def __init__(self, outcomes: list[bool]) -> None:
        super().__init__(LLMConfig())
        self.outcomes = outcomes
        self.calls = 0

    async def verdict(self, system: str, user: str) -> FitVerdict:
        raise NotImplementedError

    async def facts(self, system: str, user: str) -> PostingFacts:
        ok = self.calls < len(self.outcomes) and self.outcomes[self.calls]
        self.calls += 1
        if not ok:
            msg = "model runner crashed"
            raise RuntimeError(msg)
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
            keyword_score=20,
        )
        for i in range(n)
    ]


def _scorer(outcomes: list[bool]) -> tuple[FitScorer, _Scripted]:
    cfg = LLMConfig(enabled=True, backend="ollama", max_concurrent=1)
    scorer = FitScorer(cfg, ProfileConfig(min_report_score=55))
    judge = _Scripted(outcomes)
    scorer._judge = judge
    return scorer, judge


async def test_five_failures_in_a_row_stop_the_calls() -> None:
    scorer, judge = _scorer([])

    out = await scorer.score_all(_postings(12))

    assert judge.calls == BREAKER_AFTER == 5
    assert scorer.errors == 5
    assert scorer.tripped
    assert [s.deferred for s in out].count("llm_breaker") == 7


async def test_a_success_resets_the_count() -> None:
    scorer, judge = _scorer([False] * 4 + [True] + [False] * 4 + [True] * 3)

    out = await scorer.score_all(_postings(12))

    assert judge.calls == 12
    assert not scorer.tripped
    assert not any(s.deferred for s in out)


async def test_after_the_breaker_trips_a_cached_posting_is_still_judged(
    tmp_path: Path,
) -> None:
    """The breaker stops model CALLS; a posting whose facts are cached needs
    none, so it is judged from the cache, not deferred with the rest."""
    postings = [
        s.model_copy(
            update={
                "job": s.job.model_copy(update={"description": f"SQL, desk {i}."}),
                "keyword_score": 20 - i,
            }
        )
        for i, s in enumerate(_postings(8))
    ]
    cached = postings[-1]
    async with Store(tmp_path / "s.db") as store:
        cfg = LLMConfig(enabled=True, backend="ollama", max_concurrent=1)
        scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), store)
        judge = _Scripted([])
        scorer._judge = judge
        await store.put_verdict(
            scorer._facts_cache_key(cached.job),
            PostingFacts.model_validate({"fit_score": 70, "reason": "fits"}),
        )

        out = await scorer.score_all(postings)

    assert scorer.tripped and judge.calls == BREAKER_AFTER
    [last] = [s for s in out if s.job.url == cached.job.url]
    assert last.fit is not None and last.llm_cached and not last.deferred


class _CountingEnricher(Enricher):
    name = "counting-test-enricher"

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.calls = 0

    async def enrich(self, job: Job, verdict: FitVerdict) -> FitVerdict:
        self.calls += 1
        return verdict


async def _fail() -> None:
    msg = "model runner crashed"
    raise RuntimeError(msg)


async def test_a_tripped_breaker_also_stops_enrichment_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enrichment is a second model call. A scorer that has stopped calling the
    model does not start one for a posting it judged from the cache; the
    control run, with no trip, enriches the same posting."""
    monkeypatch.setattr(enrich_mod, "_REGISTRY", {})
    register_enricher(_CountingEnricher)
    cached = _postings(1)[0]
    facts = PostingFacts.model_validate({"fit_score": 70, "reason": "fits"})

    async def run(*, trip: bool) -> tuple[_CountingEnricher, ScoredJob]:
        async with Store(tmp_path / f"s-{trip}.db") as store:
            cfg = LLMConfig(
                enabled=True,
                backend="ollama",
                mode="facts",
                enricher=_CountingEnricher.name,
            )
            scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), store)
            scorer._judge = _Scripted([])
            await store.put_verdict(scorer._facts_cache_key(cached.job), facts)
            if trip:
                for _ in range(BREAKER_AFTER):
                    with pytest.raises(RuntimeError):
                        await scorer._counted(_fail())
                assert scorer.tripped
            [out] = await scorer.score_all([cached])
            enricher = scorer._get_enricher()
            assert isinstance(enricher, _CountingEnricher)
            return enricher, out

    enricher, out = await run(trip=False)
    assert enricher.calls == 1, "control: an untripped scorer enriches"

    enricher, out = await run(trip=True)
    assert out.fit is not None and out.llm_cached and not out.deferred
    assert enricher.calls == 0


class _SlowFailing(_Scripted):
    """Fails every call, but only after yielding once, so postings queue on
    the scorer's semaphore behind it instead of failing back to back."""

    async def facts(self, system: str, user: str) -> PostingFacts:
        await asyncio.sleep(0)
        return await super().facts(system, user)


async def test_an_enrichment_waiting_its_turn_stops_when_the_breaker_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard at the top of `_maybe_enrich` is not enough: this posting
    clears it (nothing has failed yet), then queues for the semaphore behind
    five failing calls, and gets its turn after the breaker has tripped."""
    monkeypatch.setattr(enrich_mod, "_REGISTRY", {})
    register_enricher(_CountingEnricher)
    postings = [
        s.model_copy(
            update={
                "job": s.job.model_copy(update={"description": f"SQL, desk {i}."}),
                "keyword_score": 40 - i,
            }
        )
        for i, s in enumerate(_postings(BREAKER_AFTER + 1))
    ]
    cached = postings[-1]
    async with Store(tmp_path / "s.db") as store:
        cfg = LLMConfig(
            enabled=True,
            backend="ollama",
            mode="facts",
            enricher=_CountingEnricher.name,
            max_concurrent=1,
        )
        scorer = FitScorer(cfg, ProfileConfig(min_report_score=55), store)
        scorer._judge = _SlowFailing([])
        await store.put_verdict(
            scorer._facts_cache_key(cached.job),
            PostingFacts.model_validate({"fit_score": 70, "reason": "fits"}),
        )

        out = await scorer.score_all(postings)

    enricher = scorer._get_enricher()
    assert isinstance(enricher, _CountingEnricher)
    assert scorer.tripped and scorer.errors == BREAKER_AFTER
    assert enricher.calls == 0
    [last] = [s for s in out if s.job.url == cached.job.url]
    assert last.fit is not None and last.llm_cached


@pytest.mark.parametrize(
    ("result", "fails"),
    [
        (ScanResult(), False),
        (ScanResult(llm_unusable="ollama is not running"), True),
        (ScanResult(llm_unusable="probe timed out", llm_scored=4), False),
        (ScanResult(llm_scored=4, llm_errors=1), False),
        (ScanResult(llm_scored=3, llm_errors=2), True),
        (ScanResult(llm_scored=40, llm_errors=5, llm_breaker=True), True),
    ],
    ids=[
        "keyword-only",
        "judge-never-ran",
        "probe-failed-but-scored",
        "one-in-five-failed",
        "two-in-five-failed",
        "breaker",
    ],
)
def test_what_counts_as_a_failed_llm_run(result: ScanResult, fails: bool) -> None:
    assert bool(result.llm_failure) is fails


def test_a_deferral_reason_with_no_label_is_still_named() -> None:
    posting = _postings(1)[0].model_copy(update={"deferred": "something_new"})
    breaker = _postings(1)[0].model_copy(update={"deferred": "llm_breaker"})

    text = render_markdown(ScanResult(deferred=[posting, breaker]))

    assert (
        "2 deferred to the next run (1 after the LLM stopped answering, "
        "1 something_new)"
    ) in text
    assert "()" not in text


_CONFIG = """
profile:
  keywords: {energy: 6, python: 4}
  min_keyword_score: 4
llm:
  enabled: true
  backend: ollama
  model: m:1
sources:
  - {kind: greenhouse, slug: acme, label: Acme Energy}
output:
  dir: digests
  db_path: seen.db
"""
_BOARD = {
    "jobs": [
        {
            "id": i,
            "title": f"Energy Data Analyst {i}",
            "location": {"name": "London, UK"},
            "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{i}",
            "content": "<p>Python and energy markets.</p>",
            "updated_at": "2026-08-20T10:00:00Z",
        }
        for i in range(1, 8)
    ]
}


@respx.mock
def test_scan_exits_3_when_ollama_is_down_and_still_writes_the_digest(
    tmp_path: Path,
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)

    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert result.exit_code == 3, result.output
    assert "Exit status 3" in " ".join(plain(result.output).split())
    assert (tmp_path / "digests" / "latest.md").is_file()


@respx.mock
def test_the_exit_status_comes_after_the_digest_is_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper that sees exit 3 must still have been sent the digest: the
    status is the last thing `scan` does, not a reason to skip the email."""
    delivered: list[Path] = []
    monkeypatch.setattr(
        cli, "_deliver", lambda text, html, cfg, path, **kw: delivered.append(path)
    )
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG)

    result = CliRunner().invoke(app, ["scan", "-c", str(config)])

    assert result.exit_code == 3, result.output
    assert len(delivered) == 1 and delivered[0].is_file()


@respx.mock
def test_scan_exits_3_when_the_breaker_trips_and_the_digest_names_the_deferred(
    tmp_path: Path,
) -> None:
    """A server that answers HTTP 500 to every call: five calls fail, the
    other two postings are held for the next run, and the run fails."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "m:1"}]})
    )
    mock_ollama_show()
    chat = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(500, text="model runner has unexpectedly stopped")
    )
    config = tmp_path / "config.yaml"
    config.write_text(_CONFIG.replace("model: m:1", "model: m:1\n  max_concurrent: 1"))

    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])

    assert result.exit_code == 3, result.output
    assert "stopped after repeated failures" in " ".join(plain(result.output).split())
    assert chat.call_count == 2 * BREAKER_AFTER, "each failed call was retried once"
    digest = (tmp_path / "digests" / "latest.md").read_text()
    assert "2 deferred to the next run (2 after the LLM stopped answering)" in digest
