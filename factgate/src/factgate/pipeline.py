"""One run: prefilter, cache, model call, quote check, resolvers, rules.

Every item ends as exactly one of `decided`, `filtered`, `deferred` or
`error`, and the model never sets the outcome: `decide` does.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from factgate.backends import Backend, render_user
from factgate.cache import Cache, cache_key, fingerprint
from factgate.config import Config, Prefilter
from factgate.items import Item
from factgate.rules import Decision, decide
from factgate.schema import build_model, system_prompt
from factgate.verify import check_quotes

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


def _found(term: str, text: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE) is not None


def prefilter_reason(item: Item, prefilter: Prefilter | None) -> str | None:
    """Why the prefilter drops `item` (whole words, any field), or None."""
    if prefilter is None:
        return None
    text = "\n".join(item.fields.values())
    for term in prefilter.exclude_any:
        if _found(term, text):
            return f"excluded by {term!r}"
    if prefilter.require_any and not any(_found(t, text) for t in prefilter.require_any):
        return "none of require_any found"
    return None


def load_resolvers(names: list[str], base: Path | None = None) -> list[Resolver]:
    """Each `module:function` in `names`. A module in `base` (the config's
    folder) loads as well as an installed one."""
    folder = str(base) if base is not None else ""
    added = bool(folder) and folder not in sys.path
    if added:
        sys.path.insert(0, folder)
    resolvers = []
    try:
        for name in names:
            module, _, attr = name.partition(":")
            try:
                resolvers.append(getattr(importlib.import_module(module), attr))
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"resolver {name!r} cannot be loaded ({e})") from e
    finally:
        if added:
            sys.path.remove(folder)
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
        meta=item.meta,
    )


class _Run:
    """The state of one run's model calls."""

    def __init__(self, cfg: Config, backend: Backend, as_of: dt.date) -> None:
        self.cfg, self.backend, self.as_of = cfg, backend, as_of
        self.model = build_model(cfg.facts)
        self.system = system_prompt(cfg.facts)
        schema = json.dumps(self.model.model_json_schema(), sort_keys=True)
        self.fp = fingerprint(self.system, schema, cfg.model.max_tokens, cfg.model.temperature)
        self.stats = RunStats()
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
    """Decide every item that fits the call and time limits; defer the rest."""
    started = time.monotonic()
    state = _Run(cfg, backend, as_of)
    fixes = resolvers or []
    results: dict[str, Result] = {}
    recorded: list[str] = []

    def settle(item: Item, raw: dict[str, Any]) -> bool:
        try:
            facts, decision = finish(raw, item, cfg, state.model, fixes, as_of)
        except ValidationError as e:
            reason = f"invalid answer: {e.errors()[0]['msg']}"
            results[item.id] = Result(id=item.id, status="error", reason=reason, meta=item.meta)
            return False
        results[item.id] = _decided(item, facts, decision)
        return True

    async def call(item: Item) -> None:
        async with state.gate:
            stop = state.stop_reason()
            if stop:
                results[item.id] = Result(
                    id=item.id, status="deferred", reason=stop, meta=item.meta
                )
                return
            state.stats.calls += 1
            try:
                answer = await backend.extract(
                    state.system, render_user(item), state.model, item.id
                )
            except Exception as e:
                state.in_a_row += 1
                results[item.id] = Result(
                    id=item.id, status="error", reason=f"{type(e).__name__}: {e}", meta=item.meta
                )
                return
            state.in_a_row = 0
            state.stats.input_tokens += answer.input_tokens
            state.stats.output_tokens += answer.output_tokens
            if settle(item, answer.raw):
                if cache is not None:
                    await cache.put(state.key(item), answer.raw)
                recorded.append(json.dumps({"id": item.id, "raw": answer.raw}))

    pending = []
    for item in items:
        reason = prefilter_reason(item, cfg.prefilter)
        if reason:
            results[item.id] = Result(id=item.id, status="filtered", reason=reason, meta=item.meta)
            continue
        raw = await cache.get(state.key(item)) if cache is not None else None
        if raw is not None and settle(item, raw):
            state.stats.cache_hits += 1
            continue
        pending.append(item)
    await asyncio.gather(*(call(item) for item in pending))

    if record is not None and recorded:
        record.parent.mkdir(parents=True, exist_ok=True)
        with record.open("a", encoding="utf-8") as f:
            f.write("\n".join(recorded) + "\n")
    ordered = [results[item.id] for item in items]
    stats = state.stats
    stats.backend = backend.name
    stats.items = len(items)
    stats.errors = sum(r.status == "error" for r in ordered)
    stats.deferred = sum(r.status == "deferred" for r in ordered)
    stats.filtered = sum(r.status == "filtered" for r in ordered)
    stats.seconds = round(time.monotonic() - started, 1)
    return ordered, stats
