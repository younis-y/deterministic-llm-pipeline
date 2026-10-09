"""`llm.mode: judge`, the single-call judge, is a supported mode and not a
leftover: it is what a comparison against facts mode runs, and what a plugin
judge that decides everything itself relies on.

One end-to-end test pins what makes it different from facts mode, so a change
to either mode cannot quietly turn the other into it."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

from conftest import LLM_VERDICT, OLLAMA_MODEL, mock_ollama, scan_and_record
from rolescan.config import Config
from rolescan.models import FitVerdict
from rolescan.store import Store

ADVERT = (
    "Python, trading, energy, day-ahead forecasting. "
    "You will have 5+ years of experience in power markets."
)


def _config(tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "profile": {
                "keywords": {"energy": 6, "python": 4, "trading": 6},
                "min_keyword_score": 18,
                "min_report_score": 55,
                # Facts mode would skip this advert for asking 5+ years.
                "rules": {"max_years_required": 2},
            },
            "llm": {
                "enabled": True,
                "backend": "ollama",
                "model": OLLAMA_MODEL,
                "mode": "judge",
                "extra_prompt": "Prefer roles on a trading desk.",
            },
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "seen.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )


@respx.mock
async def test_judge_mode_lets_the_model_decide(tmp_path: Path) -> None:
    mock_ollama()
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Energy Trading Analyst",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://example.test/jobs/1",
                        "content": f"<p>{ADVERT}</p>",
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                ]
            },
        )
    )
    cfg = _config(tmp_path)

    result = await scan_and_record(cfg)

    # The model's own verdict stands: `profile.rules` is facts mode's.
    assert len(result.reportable) == 1
    scored = result.reportable[0]
    assert scored.fit is not None
    assert scored.fit.verdict.value == LLM_VERDICT["verdict"]
    assert scored.fit.fit_score == LLM_VERDICT["fit_score"]
    assert scored.fit.rule is None, "no rule ran: decide() is facts mode's"

    # `llm.extra_prompt` reaches the model in judge mode, and only here.
    chats = [c for c in respx.calls if c.request.url.path == "/api/chat"]
    assert chats, "the model was called"
    sent = json.loads(chats[0].request.content)
    assert "Prefer roles on a trading desk." in json.dumps(sent["messages"])

    # The verdict is cached under the bare content hash, which facts mode never
    # uses, so one posting can hold both a verdict and a set of facts.
    async with Store(tmp_path / "seen.db") as store:
        cached = await store.get_verdict(scored.job.content_hash, 30, FitVerdict)
    assert cached is not None
    assert cached.fit_score == LLM_VERDICT["fit_score"]
