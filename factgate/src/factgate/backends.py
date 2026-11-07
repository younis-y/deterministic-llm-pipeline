"""Backends: how a model is asked for the facts. Each returns the raw answer;
checking and deciding happen in the pipeline, the same for every backend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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


class MissingAnswer(KeyError):
    """A replay file has no answer for this item: not a call, not a fault."""


class Answer(BaseModel):
    raw: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


class Backend(ABC):
    name: ClassVar[str] = ""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        """The model's raw answer for one item, shaped by `model`'s schema."""


def render_user(item: Item) -> str:
    """The item as the model sees it: one `## field` block per field, in order."""
    return "\n\n".join(f"## {name}\n{text}" for name, text in item.fields.items())


def _rejects_temperature(e: Exception) -> bool:
    return "temperature" in str(e) and getattr(e, "status_code", None) == 400


class AnthropicBackend(Backend):
    """Claude, with structured output and the system prompt cached."""

    name = "anthropic"

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

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        request: dict[str, Any] = {
            "model": self.cfg.name,
            "max_tokens": self.cfg.max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
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
        usage = response.usage
        cached = (getattr(usage, "cache_read_input_tokens", 0) or 0) + (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        return Answer(
            raw=parsed.model_dump(mode="json"),
            input_tokens=usage.input_tokens + cached,
            output_tokens=usage.output_tokens,
        )


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
    """The configured backend, or replay answers from a file when given."""
    if replay is not None:
        return ReplayBackend.from_file(replay, cfg)
    if cfg.backend in _BUILTIN:
        return _BUILTIN[cfg.backend](cfg)
    found = entry_points(group="factgate.backends", name=cfg.backend)
    if not found:
        raise ValueError(f"unknown backend {cfg.backend!r}")
    backend: type[Backend] = next(iter(found)).load()
    return backend(cfg)
