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
async def test_the_window_is_the_models_own_architecture_not_the_first_key() -> None:
    """`model_info` can carry a second `*.context_length` (a vision tower's,
    say) ahead of the language model's: the architecture names which one is."""
    respx.get(TAGS).mock(return_value=_tags())
    respx.post(SHOW).mock(
        return_value=httpx.Response(
            200,
            json={
                "model_info": {
                    "clip.context_length": 77,
                    "general.architecture": "llama",
                    "llama.context_length": 4096,
                }
            },
        )
    )
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama", model="m:1"))

    reason = await judge.preflight()

    assert "4096-token context window" in reason
    assert "77" not in reason


@respx.mock
async def test_without_an_architecture_the_first_window_is_read() -> None:
    respx.get(TAGS).mock(return_value=_tags())
    respx.post(SHOW).mock(
        return_value=httpx.Response(
            200,
            json={
                "model_info": {
                    "general.parameter_count": 14770033664,
                    "qwen2.context_length": 8192,
                }
            },
        )
    )
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama", model="m:1"))

    assert "8192-token context window" in await judge.preflight()


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
async def test_a_digest_with_its_algorithm_prefix_is_not_cut_short() -> None:
    """Some servers print `sha256:<hex>`: twelve characters of that would be
    `sha256:012345`, six hex digits that say nothing about the weights."""
    respx.get(TAGS).mock(
        return_value=httpx.Response(
            200, json={"models": [{"name": "m:1", "digest": f"sha256:{DIGEST}"}]}
        )
    )
    mock_ollama_show()

    status = await backend_status(
        LLMConfig(enabled=True, backend="ollama", model="m:1")
    )

    assert status.model_digest == "0123456789ab"


@respx.mock
async def test_an_unusable_backend_reports_no_digest() -> None:
    """The model is pulled and its digest is read, but its window is too small
    for `num_ctx`: the status names the reason and carries no digest, because
    a run that cannot score has no weights to record."""
    respx.get(TAGS).mock(return_value=_tags())
    mock_ollama_show(context_length=8192)
    cfg = LLMConfig(enabled=True, backend="ollama", model="m:1")

    status = await backend_status(cfg)

    assert "8192-token context window" in status.reason
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
        llm_calls=3,
        llm_cached=1,
        llm_scored=4,
        llm_model="m:1",
        llm_model_digest="0123456789ab",
    )

    assert "3 scored, 1 from cache by m:1 (0123456789ab)" in render_markdown(result)
