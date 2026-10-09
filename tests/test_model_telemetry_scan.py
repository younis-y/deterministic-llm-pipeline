"""A scan records one `llm_runs` row, and says when the model's rates jump (2.6.0).

The model here is a local server mocked at the HTTP layer that answers every
posting with "no facts stated", so what the resolvers then change is exactly
what the postings' own titles say: a title naming a level or a field makes the
resolver override the model's blank answer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import respx

from conftest import OLLAMA_MODEL, mock_ollama_show, scan_and_record
from rolescan.config import Config
from rolescan.digest import render_html, render_markdown
from rolescan.pipeline import ScanResult, run_scan
from rolescan.store import LlmRun, Store

BASE = "http://localhost:11434"
DIGEST = "0123456789abcdef0123456789abcdef"

#: The model's answer for every posting: nothing stated.
BLANK = {
    "level": {"value": "not_stated", "quote": ""},
    "years_required": {"value": None, "quote": ""},
    "student_only": {"value": None, "quote": ""},
    "graduation_year": {"value": None, "quote": ""},
    "hard_bars": [],
    "field": {"value": None, "quote": ""},
    "fit_score": 70,
    "reason": "Python and trading overlap.",
    "keywords_missing": [],
}

_ROLES = ("Gas", "Power", "Solar", "Wind", "Hydro", "Carbon", "Nuclear", "Coal")


def _board(n: int) -> dict[str, Any]:
    """`n` senior postings whose titles differ by a word, so none merges."""
    return {
        "jobs": [
            {
                "id": i,
                "title": f"Senior Energy Data Scientist, {_ROLES[i]} Desk",
                "location": {"name": "London, UK"},
                "absolute_url": f"https://boards.greenhouse.io/acme/jobs/{i}",
                "content": "Python, trading, energy, day-ahead forecasting.",
                "updated_at": "2026-08-20T10:00:00Z",
            }
            for i in range(n)
        ]
    }


def _serve(n: int) -> respx.Route:
    """A board of `n` postings and a local model that answers `BLANK`."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_board(n))
    )
    respx.get(f"{BASE}/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": OLLAMA_MODEL, "digest": DIGEST}]}
        )
    )
    mock_ollama_show()
    return respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": json.dumps(BLANK)}}
        )
    )


def _cfg(tmp_path: Path, *, enabled: bool = True) -> Config:
    return Config.model_validate(
        {
            "profile": {
                "keywords": {
                    "energy": 6,
                    "data scientist": 7,
                    "python": 4,
                    "trading": 6,
                },
                "min_keyword_score": 18,
                "min_report_score": 55,
            },
            "llm": {"enabled": enabled, "backend": "ollama", "model": OLLAMA_MODEL},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


async def _runs(cfg: Config) -> list[LlmRun]:
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        return await store.recent_llm_runs(10)


async def _seed(cfg: Config, n: int, **kw: Any) -> None:
    """`n` earlier runs, each of ten postings, of `model` (this scan's by
    default)."""
    model = kw.pop("model", OLLAMA_MODEL)
    async with Store(cfg.resolve(cfg.output.db_path)) as store:
        for day in range(n):
            await store.record_llm_run(
                LlmRun(
                    ran=f"2026-01-0{day + 1}T00:00:00+00:00",
                    backend="ollama",
                    model=model,
                    postings=10,
                    **kw,
                )
            )


@respx.mock
async def test_a_scan_records_one_row_for_the_model_it_used(tmp_path: Path) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)

    await run_scan(cfg)

    [run] = await _runs(cfg)
    assert (run.backend, run.model) == ("ollama", OLLAMA_MODEL)
    assert run.model_digest == DIGEST[:12]
    assert (run.calls, run.cached, run.errors, run.deferred) == (6, 0, 0, 0)
    assert run.breaker is False
    assert run.postings == 6
    assert run.level_overridden == 6, "the titles say Senior, the model said nothing"
    assert run.field_overridden == 6


@respx.mock
async def test_a_dry_scan_records_nothing(tmp_path: Path) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)

    result = await run_scan(cfg, dry_run=True)

    assert await _runs(cfg) == []
    assert result.llm_run is not None, "the run is still measured, just not kept"
    assert result.llm_run.postings == 6


@respx.mock
async def test_a_keyword_only_scan_records_nothing(tmp_path: Path) -> None:
    _serve(6)
    cfg = _cfg(tmp_path, enabled=False)

    result = await run_scan(cfg, check_llm=False)

    assert await _runs(cfg) == []
    assert result.llm_run is None
    assert result.model_health == []


@respx.mock
async def test_a_scan_with_nothing_new_to_score_records_nothing(
    tmp_path: Path,
) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await scan_and_record(cfg)
    assert len(await _runs(cfg)) == 1

    again = await scan_and_record(cfg)

    assert again.llm_calls == 0
    assert len(await _runs(cfg)) == 1


@respx.mock
async def test_the_scan_names_a_rate_that_jumped_over_the_last_runs(
    tmp_path: Path,
) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 4)

    result = await run_scan(cfg)

    assert [f.label for f in result.model_health] == [
        "level overridden",
        "field overridden",
    ]
    assert result.model_health[0].affected == 6
    assert result.model_health[0].postings == 6
    assert result.model_health[0].median == 0.0


@respx.mock
async def test_the_scan_is_silent_with_fewer_than_three_earlier_runs(
    tmp_path: Path,
) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 2)

    result = await run_scan(cfg)

    assert result.model_health == []


@respx.mock
async def test_runs_of_another_model_are_no_baseline(tmp_path: Path) -> None:
    """Switching model starts a fresh baseline: rates on another model say
    nothing about this one."""
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 4, model="some-other-model:1")

    result = await run_scan(cfg)

    assert result.model_health == []


@respx.mock
async def test_a_dry_scan_still_compares_with_the_kept_runs(tmp_path: Path) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 4)

    result = await run_scan(cfg, dry_run=True)

    assert result.model_health, "a dry run reads the history, it only writes nothing"
    assert len(await _runs(cfg)) == 4


@respx.mock
async def test_the_run_is_compared_with_the_runs_before_it_not_with_itself(
    tmp_path: Path,
) -> None:
    """Four scans that override every posting alike (nothing is marked seen
    between them, so each scores the same six, from the cache after the
    first): the fourth has three earlier runs that look just like it, so it is
    not news."""
    _serve(6)
    cfg = _cfg(tmp_path)
    results: list[ScanResult] = [await run_scan(cfg) for _ in range(4)]

    assert [r.model_health for r in results] == [[], [], [], []]
    runs = await _runs(cfg)
    assert [(r.calls, r.cached) for r in reversed(runs)] == [
        (6, 0),
        (0, 6),
        (0, 6),
        (0, 6),
    ]


@respx.mock
async def test_the_digest_of_a_scan_that_jumped_opens_with_the_line(
    tmp_path: Path,
) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 4)

    result = await run_scan(cfg)

    for text in (render_markdown(result), render_html(result)):
        assert "Model health" in text
        assert text.index("Needs attention") < text.index("Model health")
        assert text.index("Model health") < text.index("Senior Energy Data Scientist")
        assert "level overridden 100% (median 0%)" in text


@respx.mock
async def test_the_digest_of_a_steady_scan_says_nothing(tmp_path: Path) -> None:
    _serve(6)
    cfg = _cfg(tmp_path)
    await _seed(cfg, 4, level_overridden=10, field_overridden=10)

    result = await run_scan(cfg)

    assert result.model_health == []
    assert "Model health" not in render_markdown(result)
