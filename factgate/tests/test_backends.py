import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from factgate.backends import (
    AnthropicBackend,
    OpenAICompatibleBackend,
    ReplayBackend,
    get_backend,
    render_user,
)
from factgate.config import ModelConfig
from factgate.items import Item
from factgate.schema import FactSpec, build_model

MODEL = build_model({"amount": FactSpec(type="number")})
RAW = {"amount": {"value": 12.5, "quote": "12.50"}}
ITEM = Item(id="i1", fields={"subject": "Hi", "body": "Pay 12.50"})


def test_render_user_keeps_fields_apart_and_in_order() -> None:
    assert render_user(ITEM) == "## subject\nHi\n\n## body\nPay 12.50"


def _chat(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


@respx.mock
async def test_openai_compatible_sends_the_schema_and_reads_the_answer() -> None:
    route = respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=_chat(json.dumps(RAW))
    )
    cfg = ModelConfig(backend="openai-compatible", name="m", base_url="http://llm.test/v1")
    answer = await OpenAICompatibleBackend(cfg).extract("sys", "user", MODEL, "i1")
    body = json.loads(route.calls[0].request.content)
    assert body["response_format"]["json_schema"]["schema"] == MODEL.model_json_schema()
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert "temperature" not in body
    assert answer.raw == RAW
    assert (answer.input_tokens, answer.output_tokens) == (11, 7)


@respx.mock
async def test_openai_compatible_retries_once_on_a_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("factgate.backends.RETRY_DELAY", 0)
    respx.post("http://llm.test/v1/chat/completions").mock(
        side_effect=[httpx.Response(503), _chat(json.dumps(RAW))]
    )
    cfg = ModelConfig(backend="openai-compatible", name="m", base_url="http://llm.test/v1")
    answer = await OpenAICompatibleBackend(cfg).extract("s", "u", MODEL, "i1")
    assert answer.raw == RAW


@respx.mock
async def test_openai_compatible_sends_the_key_from_its_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_KEY", "sk-test")
    route = respx.post("http://llm.test/v1/chat/completions").mock(
        return_value=_chat(json.dumps(RAW))
    )
    cfg = ModelConfig(
        backend="openai-compatible", name="m", base_url="http://llm.test/v1", api_key_env="MY_KEY"
    )
    await OpenAICompatibleBackend(cfg).extract("s", "u", MODEL, "i1")
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"


async def test_anthropic_parses_the_structured_answer() -> None:
    import httpx2
    from anthropic import AsyncAnthropic

    sent: list[Any] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": json.dumps(RAW)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 9,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 5,
                },
            },
        )

    backend = AnthropicBackend(ModelConfig(name="claude-haiku-5-5"))
    backend._client = AsyncAnthropic(
        api_key="sk-test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(answer))
    )
    got = await backend.extract("sys", "user", MODEL, "i1")
    assert sent[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in sent[0]
    assert got.raw == RAW
    assert got.output_tokens == 9
    assert got.input_tokens == 925, "cached prompt tokens are input too"


def test_replay_answers_by_item_id(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"id": "i1", "raw": RAW}) + "\n")
    backend = get_backend(ModelConfig(name="m"), replay=path)
    assert isinstance(backend, ReplayBackend)


async def test_replay_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"id": "i1", "raw": RAW}) + "\n")
    backend = ReplayBackend.from_file(path)
    assert (await backend.extract("s", "u", MODEL, "i1")).raw == RAW
    with pytest.raises(KeyError, match="no recorded answer for item 'i2'"):
        await backend.extract("s", "u", MODEL, "i2")


def test_unknown_backend_is_named() -> None:
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        get_backend(ModelConfig(backend="nope", name="m"), replay=None)
