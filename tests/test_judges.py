"""Pluggable LLM backends.

The tool is useful with no credentials at all: every source except Adzuna is a
public endpoint, and keyword scoring is pure Python. The LLM stage is the only
part that ever needed a key, so it is a plugin — `anthropic` for people who
have a key, `ollama` for a local model, and neither when `llm.enabled` is off.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from rolescan.config import LLMConfig, ProfileConfig
from rolescan.models import FitVerdict, Job, ScoredJob
from rolescan.scoring import CVLibrary, FitScorer
from rolescan.scoring.judges import (
    available_judges,
    get_judge,
    unusable_backend_reason,
)

VERDICT_JSON = {
    "fit_score": 72,
    "verdict": "consider",
    "confidence": "medium",
    "reason": "Strong power-market overlap, but the role wants five years.",
    "cv_variant": "CV_EnergySystems-Modelling",
    "tailoring": ["Lead with the day-ahead forecasting project."],
    "blockers": [],
    "keywords_missing": ["Kubernetes"],
}


def _job() -> ScoredJob:
    job = Job(
        source="test",
        company="Acme",
        title="Power Market Analyst",
        location="London",
        url="https://x/1",
        description="Day-ahead price forecasting with Python.",
    )
    return ScoredJob(job=job, keyword_score=40)


# --- the registry ----------------------------------------------------------


def test_both_backends_are_registered() -> None:
    assert set(available_judges()) >= {"anthropic", "ollama"}


def test_unknown_backend_names_the_ones_that_exist() -> None:
    with pytest.raises(KeyError) as e:
        get_judge("gpt4", LLMConfig())
    assert "anthropic" in str(e.value) and "ollama" in str(e.value)


def test_only_the_anthropic_backend_declares_it_needs_a_key() -> None:
    assert available_judges()["anthropic"].needs_api_key is True
    assert available_judges()["ollama"].needs_api_key is False


# --- config: a keyless backend must not be disabled for want of a key ------


def test_ollama_backend_stays_enabled_without_an_api_key() -> None:
    """LLMConfig disabled itself whenever api_key was empty. That is correct
    for a hosted API and wrong for a local model, which never has one."""
    cfg = LLMConfig(enabled=True, backend="ollama", api_key="")
    assert cfg.enabled is True


def test_anthropic_backend_still_disables_itself_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = LLMConfig(enabled=True, backend="anthropic", api_key="")
    assert cfg.enabled is False, "no key means no hosted scoring, silently or not"


def test_default_backend_is_anthropic() -> None:
    assert LLMConfig().backend == "anthropic"


# --- the ollama backend ----------------------------------------------------


@respx.mock
async def test_ollama_judge_returns_a_validated_verdict() -> None:
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(VERDICT_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    judge = get_judge("ollama", cfg)
    verdict = await judge.verdict("system text", "user text")
    assert isinstance(verdict, FitVerdict)
    assert verdict.fit_score == 72
    assert route.called
    body = __import__("json").loads(route.calls[0].request.content)
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False
    assert body["format"]["type"] == "object", "schema-constrained, not free JSON"
    assert "fit_score" in body["format"]["properties"]


@respx.mock
async def test_ollama_connection_refused_is_a_useful_message() -> None:
    """The most likely failure is that ollama simply is not running."""
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    judge = get_judge("ollama", LLMConfig(enabled=True, backend="ollama"))
    with pytest.raises(RuntimeError) as e:
        await judge.verdict("s", "u")
    assert "ollama" in str(e.value).casefold()
    assert "11434" in str(e.value)


@respx.mock
async def test_ollama_base_url_is_configurable() -> None:
    respx.post("http://box.local:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(VERDICT_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", base_url="http://box.local:11434")
    assert (await get_judge("ollama", cfg).verdict("s", "u")).fit_score == 72


# --- the scorer uses whichever backend is configured -----------------------


@respx.mock
async def test_fit_scorer_routes_through_the_configured_backend() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(VERDICT_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    scorer = FitScorer(cfg, ProfileConfig(), CVLibrary({}), None)
    scored = await scorer.score_all([_job()])
    assert scored[0].fit is not None
    assert scored[0].fit.fit_score == 72
    assert scorer.errors == 0


@respx.mock
async def test_scorer_is_enabled_for_a_local_backend_with_no_key_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FitScorer.enabled read `cfg.api_key` directly, so on a machine with no
    ANTHROPIC_API_KEY the ollama backend scored nothing — the precise failure
    the plugin exists to prevent. It passes on a developer's machine because
    the env var happens to be set there."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(VERDICT_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    assert cfg.api_key == "", "no key on this machine"
    scorer = FitScorer(cfg, ProfileConfig(), CVLibrary({}), None)
    assert scorer.enabled is True, "a local backend needs no key"
    scored = await scorer.score_all([_job()])
    assert scored[0].fit is not None and scored[0].fit.fit_score == 72


def test_backends_command_lists_both_and_flags_which_need_a_key() -> None:
    from typer.testing import CliRunner

    from rolescan.cli import app

    result = CliRunner().invoke(app, ["backends"])
    assert result.exit_code == 0, result.output
    assert "anthropic" in result.output and "ollama" in result.output
    assert "ANTHROPIC_API_KEY" in result.output


def test_the_unverified_ollama_backend_says_so_where_it_is_chosen() -> None:
    """The backend was written to Ollama's documented API and tested against a
    mock, which proves the request shape but not the contract. That caveat is
    only useful where someone picks a backend, so it lives in the description
    `rolescan backends` prints — not just in a note somewhere. Delete this test
    when the backend has been run against a live server, and the caveat with it.
    """
    from typer.testing import CliRunner

    from rolescan.cli import app

    out = CliRunner().invoke(app, ["backends"]).output.casefold()
    assert "not yet verified" in out


# --- a judge that cannot run must say so, loudly, before the scan ----------


def _unregister(name: str) -> None:
    from rolescan.scoring import judges

    judges._REGISTRY.pop(name, None)


async def test_an_unknown_backend_name_is_reported_with_the_real_ones() -> None:
    cfg = LLMConfig(enabled=True, backend="gpt4", api_key="present")
    reason = await unusable_backend_reason(cfg)
    assert "gpt4" in reason
    assert "anthropic" in reason and "ollama" in reason


async def test_a_hosted_backend_with_no_key_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact production failure: the key was revoked, LLMConfig switched
    scoring off by itself, and every digest afterwards was keyword-only and
    empty with nothing anywhere saying why."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = LLMConfig(enabled=True, backend="anthropic", api_key="")
    assert cfg.enabled is False, "the silent self-disable still happens"
    reason = await unusable_backend_reason(cfg)
    assert "anthropic" in reason
    assert "api key" in reason.casefold()
    assert "ANTHROPIC_API_KEY" in reason


async def test_a_backend_whose_sdk_is_absent_is_reported() -> None:
    """`anthropic` is imported lazily, so a missing SDK used to surface as N
    identical scoring errors after the fetch budget had already been spent."""
    from rolescan.scoring.judges import Judge, register

    @register
    class _NoSdkJudge(Judge):
        name = "nosdk"
        needs_api_key = False
        requires_module = "a_package_that_is_not_installed_anywhere"
        description = "test only"

        async def verdict(self, system: str, user: str) -> FitVerdict:
            raise NotImplementedError

    try:
        reason = await unusable_backend_reason(LLMConfig(enabled=True, backend="nosdk"))
    finally:
        _unregister("nosdk")
    assert "a_package_that_is_not_installed_anywhere" in reason
    assert "not installed" in reason


@respx.mock
async def test_a_usable_backend_reports_nothing() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="llama3.1:8b")
    assert await unusable_backend_reason(cfg) == ""


async def test_a_deliberate_keyword_only_run_is_not_nagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`llm.enabled: false` is a choice, not a fault. Warning about a backend
    nobody asked to use would train the reader to ignore the one line that
    matters when the key really does go missing."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = LLMConfig(enabled=False, backend="anthropic")
    assert await unusable_backend_reason(cfg) == ""


# --- ollama preflight: a reachable server is not the same as a usable one --


@respx.mock
async def test_ollama_preflight_passes_when_the_model_is_pulled() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]},
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    judge = get_judge("ollama", cfg)
    assert await judge.preflight() == ""


@respx.mock
async def test_ollama_preflight_reports_an_unreachable_server() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    reason = await get_judge("ollama", cfg).preflight()
    assert "localhost:11434" in reason
    assert "ollama serve" in reason


@respx.mock
async def test_ollama_preflight_names_the_address_it_tried() -> None:
    respx.get("http://box.local:11434/api/tags").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    cfg = LLMConfig(enabled=True, backend="ollama", base_url="http://box.local:11434")
    reason = await get_judge("ollama", cfg).preflight()
    assert "box.local:11434" in reason


@respx.mock
async def test_ollama_preflight_reports_a_reachable_server_missing_the_model() -> None:
    """The server is up, but the configured model was never pulled — a
    reachable-but-unusable state that is at least as common in practice as
    the server being down outright, and just as silent without this check."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    reason = await get_judge("ollama", cfg).preflight()
    assert "qwen2.5:7b" in reason
    assert "not pulled" in reason.casefold()
    assert "llama3.1:8b" in reason, "names what IS available, not just what isn't"


@respx.mock
async def test_ollama_preflight_end_to_end_via_unusable_backend_reason() -> None:
    """The seam this whole feature depends on: a preflight failure must reach
    `unusable_backend_reason`, the function the scan actually calls, not just
    be reachable in isolation on the judge."""
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    cfg = LLMConfig(enabled=True, backend="ollama")
    reason = await unusable_backend_reason(cfg)
    assert "localhost:11434" in reason
    assert "ollama serve" in reason


async def test_a_backend_with_no_preflight_override_is_unaffected() -> None:
    """The default `Judge.preflight()` is "nothing to check" — a backend
    that never overrides it must report itself usable without touching the
    network at all. No respx mock is installed here on purpose: a stray
    network call would fail loudly rather than silently pass."""
    from rolescan.scoring.judges import Judge, register

    @register
    class _NoPreflightJudge(Judge):
        name = "nopreflight"
        needs_api_key = False
        description = "test only"

        async def verdict(self, system: str, user: str) -> FitVerdict:
            raise NotImplementedError

    try:
        judge = get_judge("nopreflight", LLMConfig(enabled=True, backend="nopreflight"))
        assert await judge.preflight() == ""
        reason = await unusable_backend_reason(
            LLMConfig(enabled=True, backend="nopreflight")
        )
        assert reason == ""
    finally:
        _unregister("nopreflight")
