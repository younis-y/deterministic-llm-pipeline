"""One run: prefilter, cache, model call, quote check, resolvers, rules.

Every item ends as exactly one of `decided`, `filtered`, `deferred` or
`error`, and the model never sets the outcome: `decide` does.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from factgate.backends import Backend, MissingAnswer, render_user
from factgate.cache import Cache, cache_key, fingerprint
from factgate.config import Config, Prefilter
from factgate.items import Item
from factgate.rules import Decision, decide
from factgate.schema import build_model, system_prompt
from factgate.verify import check_quotes, normalise

Resolver = Callable[[Item, BaseModel], BaseModel]
Status = Literal["decided", "filtered", "deferred", "error"]
#: Errors in a row after which the run stops calling and defers the rest.
BREAKER = 3


class Result(BaseModel):
    id: str
    status: Status
    outcome: str | None = None
    rule: str | None = None
    #: fact -> {"value", "quote"} (a list of those for a list fact).
    facts: dict[str, Any] = {}
    #: The quotes of the facts the fired rule read; empty for the default.
    quotes: dict[str, Any] = {}
    reason: str = ""
    meta: dict[str, Any] = {}


class RunStats(BaseModel):
    #: The backend that answered (`replay` when answers came from a file).
    backend: str = ""
    items: int = 0
    calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    deferred: int = 0
    filtered: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def _found(term: str, texts: list[str]) -> bool:
    """`term` as whole words in one of `texts`, folded as the quote check folds."""
    pattern = re.compile(rf"(?<!\w){re.escape(normalise(term))}(?!\w)")
    return any(pattern.search(text) for text in texts)


def prefilter_reason(item: Item, prefilter: Prefilter | None) -> str | None:
    """Why the prefilter drops `item` (whole words, any field), or None."""
    if prefilter is None:
        return None
    texts = [normalise(text) for text in item.fields.values()]
    for term in prefilter.exclude_any:
        if _found(term, texts):
            return f"excluded by {term!r}"
    if prefilter.require_any and not any(_found(t, texts) for t in prefilter.require_any):
        return "none of require_any found"
    return None


def _module(name: str, base: Path | None) -> ModuleType:
    """`name` from a `.py` file in `base` if there is one, loaded under a name
    of its own so two configs' `fixes.py` never share a module; else installed."""
    path = base.joinpath(*name.split(".")).with_suffix(".py") if base is not None else None
    if path is None or not path.is_file():
        return importlib.import_module(name)
    unique = "factgate_resolvers_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:  # pragma: no cover - a .py file always has one
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_resolvers(names: list[str], base: Path | None = None) -> list[Resolver]:
    """Each `module:function` in `names`. A module in `base` (the config's
    folder) loads as well as an installed one."""
    resolvers = []
    for name in names:
        module, _, attr = name.partition(":")
        try:
            resolvers.append(getattr(_module(module, base), attr))
        except Exception as e:
            raise ValueError(f"resolver {name!r} cannot be loaded ({e})") from e
    return resolvers


def finish(
    raw: dict[str, Any],
    item: Item,
    cfg: Config,
    model: type[BaseModel],
    resolvers: list[Resolver],
    as_of: dt.date,
) -> tuple[BaseModel, Decision]:
    """The model's raw answer, checked, corrected, checked again, and decided.

    The same step for a fresh answer, a cached one, an eval case's recorded
    one and a regrade, so they can never disagree. Raises `ValidationError`
    when `raw` does not fit the facts schema.
    """
    facts = check_quotes(model.model_validate(raw), item)
    for resolver in resolvers:
        facts = resolver(item, facts)
    if resolvers:
        facts = check_quotes(model.model_validate(facts.model_dump()), item)
    return facts, decide(facts, cfg.rules, cfg.default_outcome, as_of)


def _decided(item: Item, facts: BaseModel, decision: Decision) -> Result:
    return Result(
        id=item.id,
        status="decided",
        outcome=decision.outcome,
        rule=decision.rule,
        facts=facts.model_dump(mode="json"),
        quotes=decision.quotes,
        meta=item.meta,
    )


class _Run:
    """One run's state: its limits, its results and how an answer is settled."""

    def __init__(
        self,
        cfg: Config,
        backend: Backend,
        as_of: dt.date,
        cache: Cache | None,
        record: Path | None,
        resolvers: list[Resolver],
    ) -> None:
        self.cfg, self.backend, self.as_of = cfg, backend, as_of
        self.cache, self.record, self.resolvers = cache, record, resolvers
        self.model = build_model(cfg.facts)
        self.system = system_prompt(cfg.facts)
        schema = json.dumps(self.model.model_json_schema(), sort_keys=True)
        self.fp = fingerprint(
            self.system, schema, cfg.model.max_tokens, cfg.model.temperature, cfg.model.base_url
        )
        self.stats = RunStats(backend=backend.name)
        self.results: dict[str, Result] = {}
        self.in_a_row = 0
        self.deadline = time.monotonic() + cfg.model.max_minutes * 60
        self.gate = asyncio.Semaphore(cfg.model.max_concurrent)

    def key(self, item: Item) -> str:
        return cache_key(item, self.fp, self.backend.name, self.cfg.model.name)

    def stop_reason(self) -> str | None:
        if self.in_a_row >= BREAKER:
            return f"stopped after {BREAKER} errors in a row"
        if self.stats.calls >= self.cfg.model.max_calls:
            return f"call limit reached (max_calls {self.cfg.model.max_calls})"
        if time.monotonic() > self.deadline:
            return f"time limit reached (max_minutes {self.cfg.model.max_minutes:g})"
        return None

    def end(self, item: Item, status: Status, reason: str) -> None:
        self.results[item.id] = Result(id=item.id, status=status, reason=reason, meta=item.meta)

    def settle(self, item: Item, raw: dict[str, Any]) -> bool:
        """Record, finish and decide `raw`; False when it does not fit the schema."""
        if self.record is not None:
            with self.record.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"id": item.id, "raw": raw}) + "\n")
        try:
            facts, decision = finish(raw, item, self.cfg, self.model, self.resolvers, self.as_of)
        except ValidationError as e:
            self.end(item, "error", f"invalid answer: {e.errors()[0]['msg']}")
            return False
        except Exception as e:  # a resolver's bug costs this item, not the run
            self.end(item, "error", f"{type(e).__name__} after the answer: {e}")
            return True
        self.results[item.id] = _decided(item, facts, decision)
        return True

    async def batch(self, items: list[Item]) -> None:
        """Every item in one batch, as many as `max_calls` allows. The batch is
        waited on until `max_minutes`; what it had not answered by then is
        deferred, as an item past the time limit always is."""
        limit = self.cfg.model.max_calls
        for item in items[max(0, limit - self.stats.calls) :]:
            self.end(item, "deferred", f"call limit reached (max_calls {limit})")
        sent = items[: max(0, limit - self.stats.calls)]
        if not sent:
            return
        self.stats.calls += len(sent)
        prompts = [render_user(item) for item in sent]
        try:
            answers = await self.backend.extract_many(
                self.system, prompts, self.model, self.deadline
            )
        except Exception as e:
            for item in sent:
                self.end(item, "error", f"the batch failed: {type(e).__name__}: {e}")
            return
        minutes = self.cfg.model.max_minutes
        for item, answer in zip(sent, answers, strict=True):
            if answer is None:
                self.end(item, "deferred", f"not answered within the time limit ({minutes:g} min)")
            elif isinstance(answer, Exception):
                self.end(item, "error", f"{type(answer).__name__}: {answer}")
            else:
                self.stats.input_tokens += answer.input_tokens
                self.stats.output_tokens += answer.output_tokens
                if self.settle(item, answer.raw) and self.cache is not None:
                    await self.cache.put(self.key(item), answer.raw)

    async def call(self, item: Item) -> None:
        async with self.gate:
            stop = self.stop_reason()
            if stop:
                self.end(item, "deferred", stop)
                return
            self.stats.calls += 1
            try:
                answer = await self.backend.extract(
                    self.system, render_user(item), self.model, item.id
                )
            except MissingAnswer as e:
                self.stats.calls -= 1  # nothing was asked
                self.end(item, "error", str(e.args[0]))
                return
            except Exception as e:
                self.in_a_row += 1
                self.end(item, "error", f"{type(e).__name__}: {e}")
                return
            self.in_a_row = 0
            self.stats.input_tokens += answer.input_tokens
            self.stats.output_tokens += answer.output_tokens
            if self.settle(item, answer.raw) and self.cache is not None:
                await self.cache.put(self.key(item), answer.raw)


async def run(
    cfg: Config,
    items: list[Item],
    backend: Backend,
    *,
    as_of: dt.date,
    cache: Cache | None = None,
    record: Path | None = None,
    resolvers: list[Resolver] | None = None,
) -> tuple[list[Result], RunStats]:
    """Decide every item that fits the call and time limits; defer the rest.

    `record` gets every answer as it arrives, the cache's included, so a run
    that stops half way keeps what it paid for.
    """
    started = time.monotonic()
    if record is not None:
        record.parent.mkdir(parents=True, exist_ok=True)
    state = _Run(cfg, backend, as_of, cache, record, resolvers or [])
    pending = []
    for item in items:
        reason = prefilter_reason(item, cfg.prefilter)
        if reason:
            state.end(item, "filtered", reason)
            continue
        raw = await cache.get(state.key(item)) if cache is not None else None
        if raw is not None:
            state.stats.cache_hits += 1
            state.settle(item, raw)
            continue
        pending.append(item)
    if cfg.model.batch and backend.batches:
        await state.batch(pending)
    else:
        await asyncio.gather(*(state.call(item) for item in pending))

    ordered = [state.results[item.id] for item in items]
    stats = state.stats
    stats.items = len(items)
    stats.errors = sum(r.status == "error" for r in ordered)
    stats.deferred = sum(r.status == "deferred" for r in ordered)
    stats.filtered = sum(r.status == "filtered" for r in ordered)
    stats.seconds = round(time.monotonic() - started, 1)
    return ordered, stats
