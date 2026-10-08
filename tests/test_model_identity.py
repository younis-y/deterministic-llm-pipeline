"""The pre-scan check knows the model's window and which weights ran (2.5.8).

`num_ctx` above the model's own context window is a window Ollama will not
provide, so the preflight reads `/api/show` and says so before any posting is
scored. And a tag (`qwen2.5:14b`) can be re-pushed upstream and pulled again
with nothing recorded; the digest Ollama already reports in `/api/tags` now
travels with the run and is printed in the digest's stats line."""

from __future__ import annotations

import html
import json
from pathlib import Path

import httpx
import respx
from typer.testing import CliRunner, Result

from conftest import mock_ollama_show, plain
from rolescan.cli import app
from rolescan.config import Config, LLMConfig
from rolescan.digest import render_html, render_markdown
from rolescan.pipeline import ScanResult, run_scan
from rolescan.scoring.judges import BackendStatus, OllamaJudge, backend_status

TAGS = "http://localhost:11434/api/tags"
SHOW = "http://localhost:11434/api/show"
DIGEST = "0123456789abcdef0123456789abcdef"
WINDOW_REASON = (
    "the model's context window (8192) is below llm.num_ctx (12288); "
    "lower llm.num_ctx or pick a larger model"
)


def _tags(name: str = "m:1") -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": name, "digest": DIGEST}]})


@respx.mock
async def test_a_model_window_smaller_than_num_ctx_is_reported() -> None:
    respx.get(TAGS).mock(return_value=_tags())
    mock_ollama_show(context_length=8192)
    judge = OllamaJudge(LLMConfig(enabled=True, backend="ollama", model="m:1"))

    reason = await judge.preflight()

    assert reason == WINDOW_REASON
    assert judge.context_window == 8192


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

    assert "context window (4096)" in reason
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

    assert "context window (8192)" in await judge.preflight()


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
        reason="", model_digest="0123456789ab", context_window=32768
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

    assert status.reason == WINDOW_REASON
    assert status.model_digest == ""
    assert status.context_window == 8192, "the window travels with the reason"


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


# --- a window below num_ctx stops the LLM stage -----------------------------

_SCAN_CONFIG = """
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
        for i in range(1, 4)
    ]
}
#: What a server that capped the window at 8192 answers: about 8,000 prompt
#: tokens evaluated, which is above the character floor and below 97% of a
#: 12,288 `num_ctx`, so the token counts alone cannot show the cut.
_CUT_ANSWER = {
    "message": {"content": json.dumps({"fit_score": 70, "reason": "fits"})},
    "prompt_eval_count": 8000,
    "eval_count": 100,
}


def _window_scan(
    tmp_path: Path, window: int, extra: str = ""
) -> tuple[Result, respx.Route, respx.Route]:
    """`rolescan scan` against a healthy server whose model holds `window`
    tokens; returns the CLI result, the chat route and the tags route."""
    respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
        return_value=httpx.Response(200, json=_BOARD)
    )
    tags = respx.get(TAGS).mock(return_value=_tags())
    respx.post(SHOW).mock(
        return_value=httpx.Response(
            200,
            json={
                "model_info": {
                    "general.architecture": "llama",
                    "llama.context_length": window,
                }
            },
        )
    )
    chat = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=_CUT_ANSWER)
    )
    config = tmp_path / "config.yaml"
    config.write_text(_SCAN_CONFIG.replace("model: m:1", f"model: m:1{extra}"))
    result = CliRunner().invoke(app, ["scan", "-c", str(config), "--no-email"])
    return result, chat, tags


@respx.mock
def test_a_window_below_num_ctx_stops_scoring_and_exits_3(tmp_path: Path) -> None:
    """An 8k model under the default `num_ctx`, with a healthy chat endpoint.
    Scoring it would cut every facts prompt of 7.3k-8.3k tokens with nothing
    in the counts to show it, so no posting is sent to the model, the run
    falls back to keyword scores and fails."""
    result, chat, tags = _window_scan(tmp_path, 8192)

    assert result.exit_code == 3, result.output
    assert chat.call_count == 0, "no posting reaches a model that would cut it"
    assert tags.call_count == 1, "no identity re-read for a run that cannot score"
    message = f"LLM scoring did not run: {WINDOW_REASON}"
    out = " ".join(plain(result.output).split())
    assert f"{message}. Exit status 3." in out
    assert "window.." not in out
    assert "liveness probe" not in out
    assert "scoring ran anyway" not in out
    digest = (tmp_path / "digests" / "latest.md").read_text()
    assert f"{message}." in digest
    assert "window.." not in digest
    assert "liveness probe" not in digest
    assert "scoring ran anyway" not in digest
    assert "scored," not in digest, "nothing was scored"


@respx.mock
def test_switching_the_truncation_check_off_does_not_lift_the_window_stop(
    tmp_path: Path,
) -> None:
    """`llm.check_truncation` is the way out for a server that miscounts its
    tokens. A window smaller than `num_ctx` is not a count: it stops the run
    whatever that flag says."""
    result, chat, _ = _window_scan(tmp_path, 8192, extra="\n  check_truncation: false")

    assert result.exit_code == 3, result.output
    assert chat.call_count == 0
    assert f"LLM scoring did not run: {WINDOW_REASON}" in " ".join(
        plain(result.output).split()
    )


@respx.mock
def test_a_window_that_holds_num_ctx_scores_normally(tmp_path: Path) -> None:
    result, chat, _ = _window_scan(tmp_path, 32768)

    assert result.exit_code == 0, result.output
    assert chat.call_count == 3
    digest = (tmp_path / "digests" / "latest.md").read_text()
    assert "3 scored, 0 from cache by m:1 (0123456789ab)" in digest
    assert "did not run" not in digest


def test_both_digests_say_why_the_window_stopped_scoring() -> None:
    result = ScanResult(
        llm_backend="ollama",
        llm_model="m:1",
        llm_unusable=WINDOW_REASON,
        llm_window_stop=True,
    )
    message = f"LLM scoring did not run: {WINDOW_REASON}."

    assert result.llm_failure == message.removesuffix(".")
    assert f"**{message}**" in render_markdown(result)
    assert message in html.unescape(render_html(result))
    for text in (render_markdown(result), html.unescape(render_html(result))):
        assert "did not run at all" not in text
        assert "liveness probe" not in text
