"""The pre-scan check knows the model's window and which weights ran (2.5.8).

`num_ctx` above the model's own context window is a window Ollama will not
provide, so the preflight reads `/api/show` and says so before any posting is
scored. And a tag (`qwen2.5:14b`) can be re-pushed upstream and pulled again
with nothing recorded; the digest Ollama already reports in `/api/tags` now
travels with the run and is printed in the digest's stats line."""

from __future__ import annotations

from pathlib import Path

import httpx
import respx

from conftest import mock_ollama_show
from rolescan.config import Config, LLMConfig
from rolescan.digest import render_markdown
from rolescan.pipeline import ScanResult, run_scan
from rolescan.scoring.judges import BackendStatus, OllamaJudge, backend_status

TAGS = "http://localhost:11434/api/tags"
SHOW = "http://localhost:11434/api/show"
DIGEST = "0123456789abcdef0123456789abcdef"


def _tags(name: str = "m:1") -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": name, "digest": DIGEST}]})


@respx.mock
async def test_a_model_window_smaller_than_num_ctx_is_reported() -> None:
    respx.get(TAGS).mock(return_value=_tags())
    mock_ollama_show(context_length=8192)
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama", model="m:1"))

    reason = await judge.preflight()

    assert "8192-token context window" in reason
    assert "llm.num_ctx (12288)" in reason


@respx.mock
async def test_a_model_window_that_holds_num_ctx_passes() -> None:
    respx.get(TAGS).mock(return_value=_tags())
    show = mock_ollama_show(context_length=32768)
    cfg = LLMConfig(enabled=True, backend="ollama", model="m:1")

    assert await OllamaJudge(cfg).preflight() == ""
    assert show.called


@respx.mock
async def test_a_server_that_cannot_answer_show_is_not_held_against_it() -> None:
    respx.get(TAGS).mock(return_value=_tags())
    respx.post(SHOW).mock(return_value=httpx.Response(404, text="not found"))
    cfg = LLMConfig(enabled=True, backend="ollama", model="m:1")

    assert await OllamaJudge(cfg).preflight() == ""


@respx.mock
async def test_the_model_digest_comes_back_with_the_status() -> None:
    respx.get(TAGS).mock(return_value=_tags("m:latest"))
    mock_ollama_show()
    cfg = LLMConfig(enabled=True, backend="ollama", model="m")

    assert await backend_status(cfg) == BackendStatus(
        reason="", model_digest="0123456789ab"
    )


@respx.mock
async def test_an_unusable_backend_reports_no_digest() -> None:
    respx.get(TAGS).mock(return_value=_tags("other:1"))
    cfg = LLMConfig(enabled=True, backend="ollama", model="m:1")

    status = await backend_status(cfg)

    assert "not pulled" in status.reason
    assert status.model_digest == ""


@respx.mock
async def test_the_run_carries_the_model_and_its_digest(tmp_path: Path) -> None:
    respx.get(TAGS).mock(return_value=_tags())
    mock_ollama_show()
    cfg = Config.model_validate(
        {
            "llm": {"enabled": True, "backend": "ollama", "model": "m:1"},
            "output": {"dir": str(tmp_path), "db_path": str(tmp_path / "s.db")},
        }
    )

    result = await run_scan(cfg)

    assert (result.llm_model, result.llm_model_digest) == ("m:1", "0123456789ab")


def test_the_stats_line_names_the_model_that_scored() -> None:
    result = ScanResult(
        llm_calls=3, llm_cached=1, llm_model="m:1", llm_model_digest="0123456789ab"
    )

    assert "3 scored, 1 from cache by m:1 (0123456789ab)" in render_markdown(result)
