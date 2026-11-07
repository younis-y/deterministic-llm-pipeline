"""Backends: how a model is asked for the facts. Each returns the raw answer;
checking and deciding happen in the pipeline, the same for every backend."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar

import httpx
from pydantic import BaseModel

from factgate.config import ModelConfig
from factgate.items import Item

log = logging.getLogger(__name__)

#: Seconds before the one retry of a call that timed out or got a 5xx.
RETRY_DELAY = 2.0
_TIMEOUT = 120.0
#: Checks a cancelled batch gets to report that it has ended.
CANCEL_POLLS = 20
#: Tries at reading an ended batch's results.
RESULT_TRIES = 3


class MissingAnswer(KeyError):
    """A replay file has no answer for this item: not a call, not a fault."""


class Answer(BaseModel):
    raw: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(ABC):
    name: ClassVar[str] = ""
    #: True when `extract_many` sends one batch at a lower price than a call each.
    batches: ClassVar[bool] = False

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        """The model's raw answer for one item, shaped by `model`'s schema."""

    async def extract_many(
        self, system: str, users: list[str], model: type[BaseModel], deadline: float
    ) -> list[Answer | Exception | None]:
        """One answer per prompt, in order: an `Answer`, the exception that
        request failed with, or None when it was not answered by `deadline`
        (a `time.monotonic()` value). Only a backend that `batches` has it."""
        raise NotImplementedError(f"the {self.name} backend cannot batch")


def render_user(item: Item) -> str:
    """The item as the model sees it: one `## field` block per field, in order."""
    return "\n\n".join(f"## {name}\n{text}" for name, text in item.fields.items())


def _rejects_temperature(e: Exception) -> bool:
    return "temperature" in str(e) and getattr(e, "status_code", None) == 400


def _input_tokens(usage: Any) -> int:
    """Input tokens, the prompt read from or written to the cache included."""
    cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    return int(usage.input_tokens) + cached


def _batch_answer(result: Any) -> Answer | Exception | None:
    """One batch result: an `Answer`, the error it failed with, or None when
    it was cancelled or expired before it was answered."""
    if result.type in {"canceled", "expired"}:
        return None
    if result.type != "succeeded":
        error = getattr(getattr(result, "error", None), "error", None)
        detail = f"{getattr(error, 'type', '')}: {getattr(error, 'message', '')}"
        return RuntimeError(f"batch request {result.type} ({detail})")
    message = result.message
    text = next((b.text for b in message.content if b.type == "text"), "")
    if message.stop_reason != "end_turn" or not text:
        return RuntimeError(f"no facts in the answer (stop_reason={message.stop_reason!r})")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return RuntimeError(f"the answer is not JSON ({e.msg})")
    if not isinstance(raw, dict):
        return RuntimeError("the answer is not a JSON object")
    usage = message.usage
    return Answer(raw=raw, input_tokens=_input_tokens(usage), output_tokens=usage.output_tokens)


class AnthropicBackend(Backend):
    """Claude, with structured output and the system prompt cached."""

    name = "anthropic"
    batches = True
    #: Seconds between checks on a running batch (a test sets 0).
    poll_seconds = 10.0

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self._client: Any = None
        self._send_temperature = cfg.temperature is not None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - depends on the install
                msg = 'the anthropic backend needs: pip install "factgate[anthropic]"'
                raise RuntimeError(msg) from e
            key = os.environ.get(self.cfg.api_key_env or "ANTHROPIC_API_KEY")
            self._client = anthropic.AsyncAnthropic(api_key=key)
        return self._client

    @staticmethod
    def _system(system: str, ttl: str = "") -> list[dict[str, Any]]:
        control = {"type": "ephemeral", "ttl": ttl} if ttl else {"type": "ephemeral"}
        return [{"type": "text", "text": system, "cache_control": control}]

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        return await self._direct(system, user, model)

    async def _direct(
        self, system: str, user: str, model: type[BaseModel], ttl: str = ""
    ) -> Answer:
        request: dict[str, Any] = {
            "model": self.cfg.name,
            "max_tokens": self.cfg.max_tokens,
            "system": self._system(system, ttl),
            "messages": [{"role": "user", "content": user}],
            "output_format": model,
        }
        client = self._get_client()
        response: Any = None
        if self._send_temperature:
            try:
                response = await client.messages.parse(
                    **request, extra_body={"temperature": self.cfg.temperature}
                )
            except Exception as e:
                if not _rejects_temperature(e):
                    raise
                self._send_temperature = False
                log.warning("model %s refuses temperature; sending none", self.cfg.name)
        if response is None:
            response = await client.messages.parse(**request)
        parsed = response.parsed_output
        if parsed is None:
            msg = f"no facts in the answer (stop_reason={response.stop_reason!r})"
            raise RuntimeError(msg)
        return Answer(
            raw=parsed.model_dump(mode="json"),
            input_tokens=_input_tokens(response.usage),
            output_tokens=response.usage.output_tokens,
        )

    async def extract_many(
        self, system: str, users: list[str], model: type[BaseModel], deadline: float
    ) -> list[Answer | Exception | None]:
        """The first prompt by a direct call that writes the shared prompt to a
        one-hour cache entry, the rest as one Message Batch, at half the price,
        that reads it. Requests in a batch run side by side, so without that
        first call each would write the prompt again, and a write costs more
        than the batch saves (rolescan 2.9.0)."""
        if not users:
            return []
        try:
            first = await self._direct(system, users[0], model, ttl="1h")
        except Exception as e:  # the API is not answering: a batch would fail too, later
            return [e] * len(users)
        answers: list[Answer | Exception | None] = [first] + [None] * (len(users) - 1)
        if len(users) == 1:
            return answers
        batches = self._get_client().messages.batches
        requests = [
            {"custom_id": f"r{n}", "params": self._params(system, user, model)}
            for n, user in enumerate(users[1:], 1)
        ]
        try:
            batch = await batches.create(requests=requests)
        except Exception as e:
            return [first] + [e] * (len(users) - 1)
        try:
            ended = await self._wait(batch.id, deadline)
        except asyncio.CancelledError:  # the run was stopped: stop paying for the batch
            with contextlib.suppress(Exception):
                await asyncio.shield(batches.cancel(batch.id))
            raise
        if ended:
            for row in await self._rows(batch.id):
                answers[int(row.custom_id[1:])] = _batch_answer(row.result)
        return answers

    def _params(self, system: str, user: str, model: type[BaseModel]) -> dict[str, Any]:
        """A direct call's request as a batch entry: the same cached system
        block, the schema through the SDK's own `transform_schema`."""
        from anthropic import transform_schema

        params: dict[str, Any] = {
            "model": self.cfg.name,
            "max_tokens": self.cfg.max_tokens,
            "system": self._system(system, "1h"),
            "messages": [{"role": "user", "content": user}],
            "output_config": {"format": {"type": "json_schema", "schema": transform_schema(model)}},
        }
        if self._send_temperature:
            params["temperature"] = self.cfg.temperature
        return params

    async def _ended(self, batch_id: str) -> bool:
        """A failed check counts as "not yet": the batch runs on regardless,
        and giving up on it would lose answers already paid for."""
        try:
            batch = await self._get_client().messages.batches.retrieve(batch_id)
        except Exception as e:
            log.warning("could not check batch %s (%s); trying again", batch_id, e)
            return False
        return bool(batch.processing_status == "ended")

    async def _wait(self, batch_id: str, deadline: float) -> bool:
        """Whether the batch ended by `deadline`. Past it the batch is
        cancelled and given a little longer: what it answered is kept."""
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll_seconds)
            if await self._ended(batch_id):
                return True
        with contextlib.suppress(Exception):
            await self._get_client().messages.batches.cancel(batch_id)
        for _ in range(CANCEL_POLLS):
            if await self._ended(batch_id):
                return True
            await asyncio.sleep(self.poll_seconds)
        return False

    async def _rows(self, batch_id: str) -> list[Any]:
        batches = self._get_client().messages.batches
        for attempt in range(1, RESULT_TRIES + 1):
            try:
                return [row async for row in await batches.results(batch_id)]
            except Exception as e:
                log.warning("could not read batch %s results (%s)", batch_id, e)
                if attempt < RESULT_TRIES:
                    await asyncio.sleep(self.poll_seconds)
        return []


class OpenAICompatibleBackend(Backend):
    """Any server with the OpenAI chat API: Ollama, vLLM, LM Studio, OpenAI."""

    name = "openai-compatible"

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        if not self.cfg.base_url:
            raise ValueError("the openai-compatible backend needs model.base_url")
        body: dict[str, Any] = {
            "model": self.cfg.name,
            "max_tokens": self.cfg.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "facts", "schema": model.model_json_schema()},
            },
        }
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        headers = {}
        key = os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else None
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        data = await self._post(url, body, headers)
        usage = data.get("usage") or {}
        content = data["choices"][0]["message"]["content"]
        return Answer(
            raw=json.loads(content),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def _post(self, url: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for attempt in (1, 2):
                try:
                    response = await client.post(url, json=body, headers=headers)
                    if response.status_code < 500 or attempt == 2:
                        response.raise_for_status()
                        return response.json()
                except httpx.TimeoutException:
                    if attempt == 2:
                        raise
                await asyncio.sleep(RETRY_DELAY)
        raise AssertionError("unreachable")  # pragma: no cover


class ReplayBackend(Backend):
    """Answers recorded earlier (`factgate run --record`), keyed by item id."""

    name = "replay"

    def __init__(self, cfg: ModelConfig, answers: dict[str, dict[str, Any]]) -> None:
        super().__init__(cfg)
        self.answers = answers

    @classmethod
    def from_file(cls, path: Path, cfg: ModelConfig | None = None) -> ReplayBackend:
        answers = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                answers[str(row["id"])] = row["raw"]
        return cls(cfg or ModelConfig(backend="replay", name="replay"), answers)

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        if item_id not in self.answers:
            raise MissingAnswer(f"no recorded answer for item {item_id!r}")
        return Answer(raw=self.answers[item_id])


_BUILTIN: dict[str, type[Backend]] = {
    AnthropicBackend.name: AnthropicBackend,
    OpenAICompatibleBackend.name: OpenAICompatibleBackend,
}


def get_backend(cfg: ModelConfig, replay: Path | None) -> Backend:
    """The configured backend, or replay answers from a file when given (a
    replay answers at once, so it ignores `batch`)."""
    if replay is not None:
        return ReplayBackend.from_file(replay, cfg)
    if cfg.backend in _BUILTIN:
        backend = _BUILTIN[cfg.backend](cfg)
    else:
        found = entry_points(group="factgate.backends", name=cfg.backend)
        if not found:
            raise ValueError(f"unknown backend {cfg.backend!r}")
        backend = next(iter(found)).load()(cfg)
    if cfg.batch and not backend.batches:
        raise ValueError(f"the {cfg.backend} backend cannot batch: set model.batch to false")
    return backend
