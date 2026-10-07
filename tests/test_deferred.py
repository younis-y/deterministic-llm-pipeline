"""A posting the judge never reached is `deferred`, never `seen` (2.5.7).

On 2026-09-26 a run had 982 candidates against a 600-call ceiling: 382 were
left on their keyword score, then recorded as seen with everything else, so
they could never surface again. The ceiling now marks them."""

from __future__ import annotations

from pathlib import Path

import respx

from conftest import OLLAMA_MODEL, mock_ollama
from rolescan.config import Config
from rolescan.models import Job, ScoredJob
from rolescan.scoring import FitScorer


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
