import datetime as dt
import json
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from factgate.backends import Answer, Backend, ReplayBackend
from factgate.cache import Cache
from factgate.config import Config, ModelConfig
from factgate.items import Item
from factgate.pipeline import load_resolvers, run

AS_OF = dt.date(2026, 11, 7)


def _cfg(**extra: Any) -> Config:
    return Config.model_validate(
        {
            "name": "t",
            "items": {"jsonl": "x"},
            "facts": {
                "amount": {"type": "number"},
                "kind": {"type": "enum", "values": ["a", "b"]},
            },
            "rules": [{"name": "big", "when": {"amount": {"gt": 100}}, "then": "review"}],
            "default_outcome": "ok",
            "model": {"name": "m", "max_calls": 10},
            **extra,
        }
    )


def _items(n: int) -> list[Item]:
    return [Item(id=str(i), fields={"text": f"Total {i * 100} due"}) for i in range(n)]


def _raw(i: int) -> dict[str, Any]:
    return {"amount": {"value": i * 100, "quote": f"Total {i * 100}"}}


class Counting(Backend):
    name = "counting"

    def __init__(self, cfg: ModelConfig, fail: set[str] = frozenset()) -> None:  # type: ignore[assignment]
        super().__init__(cfg)
        self.calls = 0
        self.fail = fail

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        self.calls += 1
        if item_id in self.fail:
            raise RuntimeError("boom")
        return Answer(raw=_raw(int(item_id)), output_tokens=5)


async def test_decides_and_quotes() -> None:
    cfg = _cfg()
    results, stats = await run(cfg, _items(3), Counting(cfg.model), as_of=AS_OF)
    assert [(r.status, r.outcome, r.rule) for r in results] == [
        ("decided", "ok", None),
        ("decided", "ok", None),
        ("decided", "review", "big"),
    ]
    assert results[2].facts["amount"] == {"value": 200.0, "quote": "Total 200"}
    assert stats.calls == 3 and stats.output_tokens == 15


async def test_prefilter_filters_before_any_call() -> None:
    cfg = _cfg(prefilter={"exclude_any": ["total 100"]})
    backend = Counting(cfg.model)
    results, _ = await run(cfg, _items(3), backend, as_of=AS_OF)
    assert results[1].status == "filtered" and "total 100" in results[1].reason
    assert backend.calls == 2


async def test_require_any_is_whole_word() -> None:
    cfg = _cfg(prefilter={"require_any": ["tot"]})
    results, _ = await run(cfg, _items(1), Counting(cfg.model), as_of=AS_OF)
    assert results[0].status == "filtered"


async def test_the_call_ceiling_defers_the_rest() -> None:
    cfg = _cfg()
    cfg.model.max_calls = 2
    cfg.model.max_concurrent = 1
    results, stats = await run(cfg, _items(5), Counting(cfg.model), as_of=AS_OF)
    assert [r.status for r in results] == ["decided"] * 2 + ["deferred"] * 3
    assert "max_calls" in results[4].reason and stats.deferred == 3


async def test_an_invalid_answer_is_an_error_and_the_run_goes_on() -> None:
    cfg = _cfg()
    backend = ReplayBackend(
        cfg.model, {"0": {"kind": {"value": "zzz", "quote": "x"}}, "1": _raw(1)}
    )
    results, stats = await run(cfg, _items(2), backend, as_of=AS_OF)
    assert [r.status for r in results] == ["error", "decided"]
    assert "invalid answer" in results[0].reason and stats.errors == 1


async def test_three_errors_in_a_row_defer_the_rest() -> None:
    cfg = _cfg()
    cfg.model.max_concurrent = 1
    backend = Counting(cfg.model, fail={"0", "1", "2"})
    results, _ = await run(cfg, _items(5), backend, as_of=AS_OF)
    assert [r.status for r in results] == ["error"] * 3 + ["deferred"] * 2
    assert "3 errors in a row" in results[4].reason and backend.calls == 3


async def test_a_second_run_is_served_from_the_cache(tmp_path: Path) -> None:
    cfg = _cfg()
    async with Cache(tmp_path / "c.sqlite") as cache:
        await run(cfg, _items(2), Counting(cfg.model), as_of=AS_OF, cache=cache)
        backend = Counting(cfg.model)
        results, stats = await run(cfg, _items(2), backend, as_of=AS_OF, cache=cache)
    assert backend.calls == 0 and stats.cache_hits == 2
    assert [r.status for r in results] == ["decided", "decided"]


async def test_record_writes_the_replay_format(tmp_path: Path) -> None:
    cfg = _cfg()
    path = tmp_path / "answers.jsonl"
    await run(cfg, _items(2), Counting(cfg.model), as_of=AS_OF, record=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert sorted(r["id"] for r in rows) == ["0", "1"]
    assert ReplayBackend.from_file(path).answers["1"] == _raw(1)


async def test_a_resolver_cannot_slip_in_an_unquoted_value(monkeypatch: Any) -> None:
    module = types.ModuleType("fixes")

    def invent(item: Item, facts: Any) -> Any:
        return facts.model_copy(update={"amount": type(facts.amount)(value=999, quote="made up")})

    module.invent = invent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fixes", module)
    cfg = _cfg(resolvers=["fixes:invent"])
    results, _ = await run(
        cfg, _items(1), Counting(cfg.model), as_of=AS_OF, resolvers=load_resolvers(cfg.resolvers)
    )
    assert results[0].facts["amount"]["value"] is None


def test_a_bad_resolver_name_is_named() -> None:
    import pytest

    with pytest.raises(ValueError, match="resolver 'nope:fn'"):
        load_resolvers(["nope:fn"])
