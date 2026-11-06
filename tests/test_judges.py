"""Pluggable LLM backends.

The tool is useful with no credentials at all: every source except Adzuna is a
public endpoint, and keyword scoring is pure Python. The LLM stage is the only
part that ever needed a key, so it is a plugin — `anthropic` for people who
have a key, `ollama` for a local model, and neither when `llm.enabled` is off.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx

from conftest import mock_ollama_show
from rolescan.config import LLMConfig, ProfileConfig
from rolescan.models import Job, ScoredJob
from rolescan.scoring import FitScorer
from rolescan.scoring.facts import (
    FieldFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
)
from rolescan.scoring.judges import (
    AnthropicJudge,
    available_judges,
    get_judge,
    unusable_backend_reason,
)

FACTS_JSON = {
    "level": {"value": "junior", "quote": ""},
    "years_required": {"value": None, "quote": ""},
    "student_only": {"value": None, "quote": ""},
    "hard_bars": [],
    "field": {"value": None, "quote": ""},
    "fit_score": 72,
    "reason": "Strong power-market overlap.",
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
async def test_ollama_connection_refused_is_a_useful_message() -> None:
    """The most likely failure is that ollama simply is not running."""
    route = respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    judge = get_judge("ollama", LLMConfig(enabled=True, backend="ollama"))
    with pytest.raises(RuntimeError) as e:
        await judge.facts("s", "u")
    assert "ollama" in str(e.value).casefold()
    assert "11434" in str(e.value)
    assert route.call_count == 1, "a server that is not running is not retried"


@respx.mock
async def test_ollama_base_url_is_configurable() -> None:
    respx.post("http://box.local:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(FACTS_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", base_url="http://box.local:11434")
    assert (await get_judge("ollama", cfg).facts("s", "u")).fit_score == 72


@respx.mock
async def test_ollama_scores_deterministically_unless_told_otherwise() -> None:
    """Temperature reaches the server, and defaults to zero.

    It was absent from the request entirely, so the local backend inherited
    Ollama's default of 0.8 while every accuracy benchmark this project ran
    used 0 - the measured numbers described a configuration that never
    shipped, and the same posting could score differently on a re-run.
    """
    seen: dict[str, object] = {}

    def capture(request: httpx.Request) -> httpx.Response:
        seen.update(__import__("json").loads(request.content))
        return httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(FACTS_JSON)}}
        )

    respx.post("http://localhost:11434/api/chat").mock(side_effect=capture)

    cfg = LLMConfig(enabled=True, backend="ollama")
    await get_judge("ollama", cfg).facts("s", "u")
    assert seen["options"] == {
        "num_ctx": 12288,
        "num_predict": cfg.max_tokens,
        "temperature": 0.0,
    }

    seen.clear()
    cfg = LLMConfig(enabled=True, backend="ollama", temperature=0.7)
    await get_judge("ollama", cfg).facts("s", "u")
    assert seen["options"]["temperature"] == 0.7


# --- the scorer uses whichever backend is configured -----------------------


@respx.mock
async def test_fit_scorer_routes_through_the_configured_backend() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(FACTS_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    scorer = FitScorer(cfg, ProfileConfig(), None)
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
            200, json={"message": {"content": __import__("json").dumps(FACTS_JSON)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    assert cfg.api_key == "", "no key on this machine"
    scorer = FitScorer(cfg, ProfileConfig(), None)
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

        async def facts(self, system: str, user: str) -> PostingFacts:
            raise NotImplementedError

    try:
        reason = await unusable_backend_reason(LLMConfig(enabled=True, backend="nosdk"))
    finally:
        _unregister("nosdk")
    assert "a_package_that_is_not_installed_anywhere" in reason
    assert "not installed" in reason
    # The package is not on an index: the hint installs from a checkout.
    assert 'pip install -e ".[nosdk]"' in reason
    assert "rolescan[" not in reason


@respx.mock
async def test_a_usable_backend_reports_nothing() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
    )
    mock_ollama_show()
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


@respx.mock
async def test_ollama_facts_survive_a_message_that_is_not_an_object() -> None:
    """The outer layer was guarded and the inner one was not, so a body of
    {"message": "hello"} - JSON, and a dict - raised AttributeError on .get.
    That is precisely the raw exception in the digest those guards exist to
    prevent: FitScorer catches it, but what the reader is then shown is a
    Python attribute error rather than a sentence about the backend."""
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": "hello"})
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    with pytest.raises(RuntimeError) as e:
        await get_judge("ollama", cfg).facts("system", "user")
    assert "usable posting facts" in str(e.value)


@respx.mock
async def test_ollama_facts_survive_a_body_that_is_not_an_object() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json=["not", "an", "object"])
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    with pytest.raises(RuntimeError) as e:
        await get_judge("ollama", cfg).facts("system", "user")
    assert "usable posting facts" in str(e.value)


# --- ollama preflight: a reachable server is not the same as a usable one --


@respx.mock
async def test_ollama_preflight_passes_when_the_model_is_pulled() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:7b"}]},
        )
    )
    mock_ollama_show()
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

        async def facts(self, system: str, user: str) -> PostingFacts:
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


# --- the check that prevents silent failure must not cause total failure ---


@respx.mock
async def test_preflight_survives_a_200_that_is_not_json() -> None:
    """Something else bound to 11434, a proxy, a captive portal.
    `json.JSONDecodeError` is a `ValueError`, not an `httpx.HTTPError`, so
    before this it escaped preflight, escaped `run_scan`, and killed the scan
    before a single source was fetched: no digest, no email, a traceback in a
    launchd log nobody reads."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, text="<html>Login required</html>")
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    reason = await unusable_backend_reason(cfg)
    assert reason, "an unreadable answer is a reason, not an exception"
    assert "not with json" in reason.casefold()
    assert "base_url" in reason


@respx.mock
async def test_preflight_distinguishes_a_broken_server_from_a_stopped_one() -> None:
    """`httpx.HTTPStatusError` subclasses `HTTPError`, so a server answering
    500 on /api/tags was told to run `ollama serve` - starting a process that
    is demonstrably already running. The two causes need different actions."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(500, text="internal error")
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    reason = await get_judge("ollama", cfg).preflight()
    assert "500" in reason
    assert "ollama serve" not in reason, "the server is up; do not say to start it"
    assert "running" in reason.casefold()


@respx.mock
async def test_preflight_ignores_junk_entries_in_the_model_list() -> None:
    """A malformed entry is a reason to report, or to skip, never to raise."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200, json={"models": ["qwen2.5:7b", None, {"name": "qwen2.5:7b"}]}
        )
    )
    mock_ollama_show()
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    assert await get_judge("ollama", cfg).preflight() == ""


@respx.mock
async def test_preflight_handles_a_json_body_of_the_wrong_shape() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=["not", "a", "dict"])
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    reason = await get_judge("ollama", cfg).preflight()
    assert "not pulled" in reason.casefold()


@respx.mock
async def test_an_untagged_model_is_not_satisfied_by_a_different_tag() -> None:
    """`llama3.1` means `llama3.1:latest` to ollama, so a server holding only
    `llama3.1:70b` cannot serve it. Comparing against bare prefixes said it
    could, and the scan then failed one posting at a time - which is the
    failure mode this preflight exists to replace."""
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:70b"}]})
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="llama3.1")
    reason = await get_judge("ollama", cfg).preflight()
    assert "not pulled" in reason.casefold()
    assert "llama3.1:70b" in reason


@respx.mock
async def test_an_untagged_model_matches_the_latest_tag() -> None:
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:latest"}]})
    )
    mock_ollama_show()
    cfg = LLMConfig(enabled=True, backend="ollama", model="llama3.1")
    assert await get_judge("ollama", cfg).preflight() == ""


async def test_a_judge_whose_constructor_raises_does_not_kill_the_scan() -> None:
    """`cls(cfg)` now runs at startup, before anything is fetched. Judges are
    a public plugin API, so a third-party `__init__` that raises would take
    the whole run down - the same total failure, from the other end."""
    from rolescan.scoring.judges import Judge, register

    @register
    class _ExplodingJudge(Judge):
        name = "exploding"
        needs_api_key = False
        description = "test only"

        def __init__(self, cfg: LLMConfig) -> None:
            msg = "no config for you"
            raise RuntimeError(msg)

        async def facts(self, system: str, user: str) -> PostingFacts:
            raise NotImplementedError

    try:
        reason = await unusable_backend_reason(
            LLMConfig(enabled=True, backend="exploding")
        )
        assert "exploding" in reason
        assert "RuntimeError" in reason
        assert "no config for you" in reason
    finally:
        _unregister("exploding")


# --- a plugin's key comes from the variable the plugin names ---------------


async def test_a_third_party_judge_key_is_read_from_its_own_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`judges.py` tells the user to export `cls.api_key_env`, so the config
    has to read that variable. Reading ANTHROPIC_API_KEY unconditionally made
    a public plugin API that names a variable it never looks at: the user
    exports OPENAI_API_KEY, the config still self-disables for want of a key,
    and the message tells them to export what they just exported."""
    from rolescan.scoring.judges import Judge, register

    @register
    class _OtherHostedJudge(Judge):
        name = "otherhosted"
        needs_api_key = True
        api_key_env = "OTHER_PROVIDER_KEY"
        description = "test only"

        async def facts(self, system: str, user: str) -> PostingFacts:
            raise NotImplementedError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_PROVIDER_KEY", "sk-other-123")
    try:
        cfg = LLMConfig(enabled=True, backend="otherhosted")
        assert cfg.api_key == "sk-other-123"
        assert cfg.enabled, "a backend with its key exported must not self-disable"
        assert await unusable_backend_reason(cfg) == ""
    finally:
        _unregister("otherhosted")


async def test_a_third_party_judge_without_its_key_still_self_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rolescan.scoring.judges import Judge, register

    @register
    class _OtherHostedJudge2(Judge):
        name = "otherhosted2"
        needs_api_key = True
        api_key_env = "OTHER_PROVIDER_KEY_2"
        description = "test only"

        async def facts(self, system: str, user: str) -> PostingFacts:
            raise NotImplementedError

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-for-this-backend")
    monkeypatch.delenv("OTHER_PROVIDER_KEY_2", raising=False)
    try:
        cfg = LLMConfig(enabled=True, backend="otherhosted2")
        assert cfg.api_key == "", "another provider's key is not this one's"
        assert not cfg.enabled
        assert "OTHER_PROVIDER_KEY_2" in await unusable_backend_reason(cfg)
    finally:
        _unregister("otherhosted2")


# --- the anthropic backend -------------------------------------------------


def _fake_facts() -> PostingFacts:
    return PostingFacts(
        level=LevelFact(),
        years_required=YearsFact(),
        student_only=StudentFact(),
        hard_bars=[],
        field=FieldFact(),
        fit_score=70,
        reason="Matches the core skills.",
        keywords_missing=[],
    )


class _FakeMessages:
    """Stands in for `AsyncAnthropic().messages`; records every call.

    `parse` takes the keywords the SDK's `AsyncMessages.parse` takes and no
    others, so a wrong keyword raises `TypeError` here exactly as it does
    there. It used to take `**kw`, which is how a `temperature=` keyword that
    the SDK rejects could pass a whole suite. The SDK's `create` and `parse`
    have no `temperature` parameter in current versions: sampling is set
    through `extra_body`.
    """

    def __init__(self, *, empty: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.empty = empty

    async def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: Any,
        system: Any = None,
        output_format: Any = None,
        tools: Any = None,
        extra_body: Any = None,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
                "system": system,
                "output_format": output_format,
                "tools": tools,
                "extra_body": extra_body,
            }
        )
        return self._answer(output_format)

    def _answer(self, fmt: Any) -> SimpleNamespace:
        if self.empty:
            return SimpleNamespace(parsed_output=None, stop_reason="max_tokens")
        return SimpleNamespace(parsed_output=_fake_facts())


def _anthropic(fake: _FakeMessages, **llm: Any) -> AnthropicJudge:
    judge = AnthropicJudge(LLMConfig(backend="anthropic", api_key="k", **llm))
    judge._client = SimpleNamespace(messages=fake)
    return judge


# --- prompt caching: every anthropic call sends one cached system block ----


def _cached(system: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


async def test_anthropic_facts_sends_a_cached_system_block() -> None:
    fake = _FakeMessages()
    facts = await _anthropic(fake).facts("sys", "user")
    assert fake.calls[0]["system"] == _cached("sys")
    assert fake.calls[0]["output_format"] is PostingFacts
    assert isinstance(facts, PostingFacts)


async def test_anthropic_facts_with_no_output_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError, match="no posting facts"):
        await _anthropic(_FakeMessages(empty=True)).facts("s", "u")


# --- temperature: the SDK takes no keyword for it, so it goes in extra_body ---


def test_the_fake_client_takes_only_what_the_real_sdk_takes() -> None:
    """The strict fake is only worth having while it is strict AND faithful:
    every keyword it accepts must exist on the installed SDK's `parse`, and
    `temperature` must be refused by it (the SDK refuses it with TypeError)."""
    import inspect

    from anthropic.resources.messages import AsyncMessages

    real = set(inspect.signature(AsyncMessages.parse).parameters)
    fake = set(inspect.signature(_FakeMessages.parse).parameters) - {"self"}
    assert fake <= real, f"not on the real parse(): {sorted(fake - real)}"
    assert "extra_body" in real
    assert "temperature" not in real, "the SDK has a keyword now: drop extra_body"
    assert not {"kw", "kwargs"} & fake, "a catch-all would accept any keyword"


async def test_the_fake_client_refuses_a_temperature_keyword_like_the_sdk() -> None:
    kw: Any = {"temperature": 0.0, "model": "m", "max_tokens": 1, "messages": []}
    with pytest.raises(TypeError, match="temperature"):
        await _FakeMessages().parse(**kw)


async def test_anthropic_facts_sends_the_configured_temperature() -> None:
    fake = _FakeMessages()
    await _anthropic(fake).facts("sys", "user")
    assert fake.calls[0]["extra_body"] == {"temperature": 0.0}


async def test_anthropic_passes_a_changed_temperature_through() -> None:
    fake = _FakeMessages()
    judge = _anthropic(fake, temperature=0.3)
    await judge.facts("sys", "user")
    assert fake.calls[0]["extra_body"]["temperature"] == 0.3


async def test_anthropic_still_sends_the_rest_of_the_request_with_it() -> None:
    """Moving to a helper must not drop what each call already sent."""
    fake = _FakeMessages()
    judge = _anthropic(fake, model="claude-haiku-4-5-20251001", max_tokens=777)
    await judge.facts("sys", "user")
    [call] = fake.calls
    assert call["model"] == "claude-haiku-4-5-20251001"
    assert call["max_tokens"] == 777
    assert call["messages"] == [{"role": "user", "content": "user"}]
    assert call["system"] == _cached("sys")


def _rejecting_400(message: str) -> Exception:
    import httpx2
    from anthropic import BadRequestError

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return BadRequestError(
        message,
        response=httpx2.Response(400, request=request),
        body={"type": "error", "error": {"message": message}},
    )


class _RejectsTemperature(_FakeMessages):
    """A model that refuses the parameter, as claude-sonnet-5 does (a 400)."""

    def __init__(self, error: Exception | None = None) -> None:
        super().__init__()
        self.attempts = 0
        self.error = error or _rejecting_400(
            "`temperature` is deprecated for this model."
        )

    async def parse(self, **kw: Any) -> SimpleNamespace:
        self.attempts += 1
        await asyncio.sleep(0)  # a real call yields: let the others start
        if kw.get("extra_body") and "temperature" in kw["extra_body"]:
            raise self.error
        return await super().parse(**kw)


async def test_a_model_that_rejects_temperature_is_called_without_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """claude-sonnet-5 (the default `llm.model`) answers 400 to any
    `temperature`. Sending it unconditionally would fail every call of a
    default install, so the call is made again without it, once, and the judge
    stops sending it: the scan is no worse off than before the setting was
    sent."""
    import logging

    fake = _RejectsTemperature()
    judge = _anthropic(fake)
    with caplog.at_level(logging.WARNING, logger="rolescan.scoring.judges"):
        for _ in range(3):
            await judge.facts("sys", "user")
    assert [c["extra_body"] for c in fake.calls] == [None, None, None]
    assert fake.attempts == 4, "one rejected attempt, then no more probing"
    warned = [r for r in caplog.records if "temperature" in r.getMessage()]
    assert len(warned) == 1, "said once, not on every posting"


async def test_concurrent_first_calls_warn_once_and_later_ones_omit_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`llm.max_concurrent` calls are in flight before the first 400 comes
    back, and each sees it. They all retry, but only the one that finds the
    flag still set clears it and says so."""
    import logging

    fake = _RejectsTemperature()
    judge = _anthropic(fake)
    with caplog.at_level(logging.WARNING, logger="rolescan.scoring.judges"):
        results = await asyncio.gather(*(judge.facts("sys", "user") for _ in range(5)))
        warned = [r for r in caplog.records if "temperature" in r.getMessage()]
        assert len(warned) == 1, [r.getMessage()[:60] for r in warned]
        before = fake.attempts
        await judge.facts("sys", "user")
    assert all(isinstance(r, PostingFacts) for r in results)
    assert fake.attempts == before + 1, "a later call is sent once, without it"
    assert [c["extra_body"] for c in fake.calls] == [None] * 6
    assert len([r for r in caplog.records if "temperature" in r.getMessage()]) == 1


async def test_a_400_about_something_else_is_not_swallowed() -> None:
    fake = _RejectsTemperature(_rejecting_400("max_tokens: too large"))
    with pytest.raises(Exception, match="max_tokens"):
        await _anthropic(fake).facts("sys", "user")
    assert fake.attempts == 1


async def test_an_error_that_is_not_a_400_is_not_retried() -> None:
    fake = _RejectsTemperature(RuntimeError("temperature, but a server fault"))
    with pytest.raises(RuntimeError, match="server fault"):
        await _anthropic(fake).facts("sys", "user")
    assert fake.attempts == 1


async def test_temperature_reaches_the_request_body_through_the_real_sdk() -> None:
    """The fake pins what we hand the SDK; this pins what the SDK puts on the
    wire: `extra_body` is merged into the JSON body, at the top level."""
    import json

    import httpx2
    from anthropic import AsyncAnthropic

    bodies: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        text = json.dumps(FACTS_JSON)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    judge = AnthropicJudge(LLMConfig(backend="anthropic", api_key="k", temperature=0.3))
    judge._client = AsyncAnthropic(
        api_key="k",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(answer)),
        max_retries=0,
    )
    facts = await judge.facts("sys", "user")
    assert facts.fit_score == 72
    assert [b["temperature"] for b in bodies] == [0.3]


@respx.mock
async def test_ollama_judge_returns_validated_facts() -> None:
    facts_json = {
        "level": {"value": "junior", "quote": ""},
        "years_required": {"value": None, "quote": ""},
        "student_only": {"value": None, "quote": ""},
        "hard_bars": [],
        "field": {"value": None, "quote": ""},
        "fit_score": 55,
        "reason": "Solid Python overlap.",
        "keywords_missing": ["Kubernetes"],
    }
    route = respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200, json={"message": {"content": __import__("json").dumps(facts_json)}}
        )
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    judge = get_judge("ollama", cfg)
    facts = await judge.facts("system text", "user text")
    assert isinstance(facts, PostingFacts)
    assert facts.fit_score == 55
    assert route.called
    body = __import__("json").loads(route.calls[0].request.content)
    assert body["format"] == PostingFacts.model_json_schema()


@respx.mock
async def test_ollama_facts_survives_a_body_that_is_not_usable() -> None:
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "not json"}})
    )
    cfg = LLMConfig(enabled=True, backend="ollama", model="qwen2.5:7b")
    with pytest.raises(RuntimeError):
        await get_judge("ollama", cfg).facts("system text", "user text")


