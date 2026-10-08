"""How every Ollama call is made (2.5.8): `num_ctx` always, one retry on a
timeout or a 5xx, and a cut prompt is an error, never a set of facts.

Measured 2026-10-07: rolescan sent no `num_ctx`, so Ollama sized the window
from the machine's memory (32,768 on a 36 GB Mac, 4,096 on a smaller
one). Forced to 4,096 it evaluated 2,050 tokens of a 7,290-token facts prompt,
dropping the instructions and examples from the middle, and still returned
valid JSON: `fit_score` moved from 70 to 50 and nothing said why."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from rolescan.config import Config, LLMConfig
from rolescan.pipeline import run_scan
from rolescan.scoring.judges import (
    OllamaJudge,
    OllamaUsage,
    PromptTruncatedError,
    ollama_options,
)

CHAT = "http://localhost:11434/api/chat"
FACTS = {
    "level": {"value": "not_stated", "quote": ""},
    "years_required": {"value": None, "quote": ""},
    "student_only": {"value": None, "quote": ""},
    "graduation_year": {"value": None, "quote": ""},
    "field": {"value": None, "quote": ""},
    "hard_bars": [],
    "fit_score": 70,
    "reason": "Strong SQL overlap.",
    "keywords_missing": [],
}


def _answer(prompt: int | None = None, output: int | None = None) -> httpx.Response:
    body: dict[str, object] = {"message": {"content": json.dumps(FACTS)}}
    if prompt is not None:
        body["prompt_eval_count"] = prompt
    if output is not None:
        body["eval_count"] = output
    return httpx.Response(200, json=body)


def test_every_call_names_its_context_window() -> None:
    assert ollama_options(LLMConfig()) == {
        "num_ctx": 12288,
        "num_predict": 1500,
        "temperature": 0.0,
    }
    custom = LLMConfig(num_ctx=16384, max_tokens=800)
    assert ollama_options(custom)["num_ctx"] == 16384
    assert ollama_options(custom, num_predict=400)["num_predict"] == 400


def test_a_context_window_too_small_for_any_prompt_is_refused() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 2048"):
        LLMConfig(num_ctx=1024)


@respx.mock
async def test_the_window_reaches_the_server() -> None:
    route = respx.post(CHAT).mock(return_value=_answer())
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama", num_ctx=16384))

    await judge.facts("system", "user")

    assert json.loads(route.calls[0].request.content)["options"]["num_ctx"] == 16384


@respx.mock
async def test_usage_is_kept_when_the_prompt_arrived_whole() -> None:
    respx.post(CHAT).mock(return_value=_answer(prompt=2000, output=120))
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    await judge.facts("s" * 6000, "u" * 2000)  # at least 1,333 tokens

    assert judge.last_usage == OllamaUsage(prompt_tokens=2000, output_tokens=120)


@respx.mock
async def test_a_prompt_cut_to_fit_is_an_error_not_facts() -> None:
    """The 2026-10-07 reproduction: 2,050 tokens evaluated of a prompt that
    cannot be shorter than 5,000."""
    respx.post(CHAT).mock(return_value=_answer(prompt=2050, output=73))
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    with pytest.raises(PromptTruncatedError, match="cut the prompt"):
        await judge.facts("s" * 28000, "u" * 2000)


@respx.mock
async def test_a_full_context_window_is_an_error() -> None:
    respx.post(CHAT).mock(return_value=_answer(prompt=11000, output=1000))
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    with pytest.raises(PromptTruncatedError, match="filled 12000 of the 12288"):
        await judge.facts("s" * 6000, "u")


@respx.mock
async def test_the_truncation_check_can_be_switched_off() -> None:
    """The check assumes `prompt_eval_count` counts the whole prompt even when
    Ollama reuses its cache (measured on one server and version). A server
    that counts only the uncached part would fail every call; the flag is the
    way out, and the counts are still recorded."""
    respx.post(CHAT).mock(return_value=_answer(prompt=2050, output=73))
    judge = OllamaJudge(
        LLMConfig(enabled=True, backend="ollama", check_truncation=False)
    )

    assert (await judge.facts("s" * 28000, "u" * 2000)).fit_score == 70
    assert judge.last_usage == OllamaUsage(prompt_tokens=2050, output_tokens=73)


@respx.mock
async def test_a_server_that_reports_no_counts_is_not_second_guessed() -> None:
    respx.post(CHAT).mock(return_value=_answer())
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    assert (await judge.facts("s" * 28000, "u")).fit_score == 70
    assert judge.last_usage is None


@pytest.mark.parametrize(
    "first",
    [httpx.Response(500, text="runner restarted"), httpx.ReadTimeout("slow")],
    ids=["5xx", "timeout"],
)
@respx.mock
async def test_one_transient_failure_is_retried(
    first: httpx.Response | Exception,
) -> None:
    route = respx.post(CHAT).mock(side_effect=[first, _answer()])
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    assert (await judge.facts("s", "u")).fit_score == 70
    assert route.call_count == 2


@respx.mock
async def test_a_second_failure_is_reported_and_not_retried_again() -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(500, text="down"))
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await judge.facts("s", "u")
    assert route.call_count == 2


@respx.mock
async def test_a_client_error_is_not_retried() -> None:
    route = respx.post(CHAT).mock(return_value=httpx.Response(404, text="no model"))
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama"))

    with pytest.raises(RuntimeError, match="HTTP 404"):
        await judge.facts("s", "u")
    assert route.call_count == 1


@respx.mock
async def test_a_cut_prompt_is_counted_and_its_posting_is_not_recorded(
    tmp_path: Path,
) -> None:
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "id": 1,
                        "title": "Energy Data Analyst",
                        "location": {"name": "London, UK"},
                        "absolute_url": "https://boards.greenhouse.io/acme/jobs/1",
                        "content": "<p>Python and energy markets.</p>",
                        "updated_at": "2026-08-20T10:00:00Z",
                    }
                ]
            },
        )
    )
    respx.post(CHAT).mock(return_value=_answer(prompt=10, output=50))
    cfg = Config.model_validate(
        {
            "profile": {"keywords": {"energy": 6}, "min_keyword_score": 1},
            "llm": {"enabled": True, "backend": "ollama", "model": "m:1"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
            "sources": [{"kind": "greenhouse", "slug": "acme", "label": "Acme"}],
        }
    )

    result = await run_scan(cfg, check_llm=False)

    assert result.llm_errors == 1
    assert "PromptTruncatedError" in result.llm_error_detail
    assert result.to_record == [], "a posting judged on a cut prompt stays unseen"
