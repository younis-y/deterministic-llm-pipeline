"""A posting the judge never reached is `deferred`, never `seen` (2.5.7).

On 2026-09-26 a run had 982 candidates against a 600-call ceiling: 382 were
left on their keyword score, then recorded as seen with everything else, so
they could never surface again. The ceiling now marks them."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from conftest import OLLAMA_MODEL, mock_ollama
from rolescan.config import Config
from rolescan.digest import render_markdown
from rolescan.models import Job, ScoredJob
from rolescan.pipeline import ScanResult, _prefilter, _rank, assessed, run_scan
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
    assert {s.job.title for s in kept} == {"Energy Analyst 1", "Energy Analyst 3"}


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
    first = await run_scan(cfg)
    assert len(first.reportable) == 1
    assert [s.deferred for s in first.deferred] == ["digest_cap"]
    assert await _seen(tmp_path) == {"Energy Python Analyst"}
    second = await run_scan(cfg)
    assert [s.job.title for s in second.reportable] == ["Energy Analyst"]


@respx.mock
async def test_deferred_posting_with_a_term_is_listed_not_recorded(
    tmp_path: Path,
) -> None:
    """Review focus 1: the ceiling skipped it AND a hard term matched. It is
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
    result = await run_scan(cfg)
    assert [(s.job.title, s.deferred) for s in result.deferred] == [
        ("Energy Analyst", "llm_ceiling")
    ]
    text = render_markdown(result)
    assert "Energy Analyst" in text and "security clearance" in text
    assert "Energy Analyst" not in await _seen(tmp_path)


def test_stats_line_counts_deferred() -> None:
    result = ScanResult(
        unique=3,
        deferred=[_scored(1, 1).model_copy(update={"deferred": "llm_ceiling"})],
    )
    assert "1 deferred to the next run (1 over the LLM budget)" in render_markdown(
        result
    )
