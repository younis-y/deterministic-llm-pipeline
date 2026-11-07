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


class _BatchServer:
    """The Messages and Message Batches endpoints, faked behind the real SDK.

    The first prompt is answered as a direct call; the batch answers the rest
    (in reverse order, as a batch may), unless `ends` is False, in which case
    it stays in progress until cancelled and then reports the second request
    canceled.
    """

    def __init__(self, *, ends: bool = True, errored: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        self.ends, self.errored = ends, errored
        self.direct: list[Any] = []
        self.created: list[Any] = []
        self.cancelled = False

    @staticmethod
    def _message(raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": json.dumps(raw)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }

    def _batch(self, status: str) -> dict[str, Any]:
        url = "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"
        return {
            "id": "msgbatch_1",
            "type": "message_batch",
            "processing_status": status,
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": "2026-11-07T12:00:00Z",
            "expires_at": "2026-11-08T12:00:00Z",
            "ended_at": "2026-11-07T12:01:00Z" if status == "ended" else None,
            "cancel_initiated_at": None,
            "archived_at": None,
            "results_url": url if status == "ended" else None,
        }

    def _results(self) -> str:
        rows = []
        for request in reversed(self.created[0]["requests"]):
            cid = request["custom_id"]
            if cid in self.errored:
                result: dict[str, Any] = {
                    "type": "errored",
                    "error": {"type": "error", "error": {"type": "api_error", "message": "boom"}},
                }
            elif not self.ends and cid != "r1":
                result = {"type": "canceled"}
            else:
                n = int(cid[1:])
                result = {
                    "type": "succeeded",
                    "message": self._message({"amount": {"value": n, "quote": f"Pay {n}"}}),
                }
            rows.append(json.dumps({"custom_id": cid, "result": result}))
        return "\n".join(rows) + "\n"

    def __call__(self, request: Any) -> Any:
        import httpx2

        path = request.url.path
        if path == "/v1/messages":
            self.direct.append(json.loads(request.content))
            return httpx2.Response(
                200, json=self._message({"amount": {"value": 0, "quote": "Pay 0"}})
            )
        if path == "/v1/messages/batches" and request.method == "POST":
            self.created.append(json.loads(request.content))
            return httpx2.Response(200, json=self._batch("in_progress"))
        if path.endswith("/cancel"):
            self.cancelled = True
            return httpx2.Response(200, json=self._batch("canceling"))
        if path.endswith("/results"):
            return httpx2.Response(200, text=self._results())
        ended = self.ends or self.cancelled
        return httpx2.Response(200, json=self._batch("ended" if ended else "in_progress"))


def _batching(server: _BatchServer) -> AnthropicBackend:
    import httpx2
    from anthropic import AsyncAnthropic

    backend = AnthropicBackend(ModelConfig(name="claude-haiku-5-5", batch=True))
    backend._client = AsyncAnthropic(
        api_key="sk-test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(server)),
        max_retries=0,
    )
    backend.poll_seconds = 0
    return backend


async def test_a_batch_warms_the_cache_with_one_direct_call_then_sends_the_rest() -> None:
    import time

    server = _BatchServer(errored={"r2"})
    users = ["Pay 0", "Pay 1", "Pay 2", "Pay 3"]
    answers = await _batching(server).extract_many("sys", users, MODEL, time.monotonic() + 60)
    hour = {"type": "ephemeral", "ttl": "1h"}
    assert server.direct[0]["system"][0]["cache_control"] == hour
    [created] = server.created
    assert [r["custom_id"] for r in created["requests"]] == ["r1", "r2", "r3"]
    params = created["requests"][0]["params"]
    assert params["system"][0]["cache_control"] == hour
    assert params["messages"] == [{"role": "user", "content": "Pay 1"}]
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert [getattr(a, "raw", {}).get("amount", {}).get("value") for a in answers[:2]] == [0, 1]
    assert isinstance(answers[2], Exception) and "boom" in str(answers[2])
    assert answers[3].raw == {"amount": {"value": 3, "quote": "Pay 3"}}  # type: ignore[union-attr]
    assert answers[1].input_tokens == 10 and answers[1].output_tokens == 4  # type: ignore[union-attr]


async def test_a_batch_past_its_deadline_is_cancelled_and_keeps_what_it_answered() -> None:
    import time

    server = _BatchServer(ends=False)
    answers = await _batching(server).extract_many(
        "sys", ["Pay 0", "Pay 1", "Pay 2"], MODEL, time.monotonic()
    )
    assert server.cancelled
    assert answers[0].raw["amount"]["value"] == 0  # type: ignore[union-attr]
    assert answers[1].raw["amount"]["value"] == 1  # type: ignore[union-attr]
    assert answers[2] is None, "cancelled before it was answered: not an error"


def test_a_backend_that_cannot_batch_says_so() -> None:
    cfg = ModelConfig(backend="openai-compatible", name="m", base_url="http://x/v1", batch=True)
    with pytest.raises(ValueError, match="openai-compatible backend cannot batch"):
        get_backend(cfg, replay=None)


def test_replay_ignores_batch(tmp_path: Path) -> None:
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps({"id": "i1", "raw": RAW}) + "\n")
    backend = get_backend(ModelConfig(name="m", batch=True), replay=path)
    assert isinstance(backend, ReplayBackend) and not backend.batches
