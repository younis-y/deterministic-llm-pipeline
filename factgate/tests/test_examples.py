"""The shipped examples: they run with no key, their cases hold, and the README
quotes the live results they recorded."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

from factgate.backends import ReplayBackend
from factgate.config import load_config
from factgate.eval.cases import load_cases
from factgate.eval.live import EvalRun, offline_run
from factgate.eval.report import readme_matches
from factgate.items import read_items
from factgate.pipeline import load_resolvers, run

ROOT = Path(__file__).parent.parent
EXAMPLES = ROOT / "examples"
NAMES = ["invoices", "support-tickets", "tenders"]
AS_OF = dt.date(2026, 11, 7)


@pytest.mark.parametrize("name", NAMES)
async def test_the_recorded_answers_decide_every_item(name: str) -> None:
    folder = EXAMPLES / name
    cfg = load_config(folder / "config.yaml")
    backend = ReplayBackend.from_file(folder / "answers.jsonl", cfg.model)
    results, stats = await run(
        cfg,
        read_items(cfg.items, cfg.base),
        backend,
        as_of=AS_OF,
        resolvers=load_resolvers(cfg.resolvers, cfg.base),
    )
    assert {r.status for r in results} <= {"decided", "filtered"}, results
    assert stats.errors == 0 and stats.deferred == 0


@pytest.mark.parametrize("name", NAMES)
def test_the_offline_cases_hold(name: str) -> None:
    folder = EXAMPLES / name
    cfg = load_config(folder / "config.yaml")
    cases = load_cases([folder / "cases.jsonl"], cfg)
    ran = offline_run(cases, cfg, "ci", AS_OF)
    assert ran.cases, "every visible case carries a model answer"
    assert ran.unexpected() == []


def test_the_heldout_invoices_are_what_the_oracle_writes() -> None:
    path = EXAMPLES / "invoices" / "gen_heldout.py"
    spec = importlib.util.spec_from_file_location("gen_heldout", path)
    assert spec is not None and spec.loader is not None
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    written, _ = oracle.cases()
    on_disk = (EXAMPLES / "invoices" / "heldout.jsonl").read_text(encoding="utf-8")
    assert [json.loads(line) for line in on_disk.splitlines()] == written


@pytest.mark.parametrize("name", NAMES)
def test_the_readme_quotes_the_latest_live_run(name: str) -> None:
    runs = [_load(p) for p in (EXAMPLES / name / "results").glob("*.json")]
    live = [r for r in runs if r.mode == "live"]
    assert live, f"no live run recorded for {name}"
    latest = max(live, key=lambda r: r.date)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme_matches(readme, latest) == []
    [row] = [line for line in readme.splitlines() if f"(examples/{name})" in line]
    assert f"pass@1 {latest.pass_at_1:.0%}" in row, row
    assert f"| {len(latest.cases)} (" in row, row
    cfg = load_config(EXAMPLES / name / "config.yaml")
    offline = offline_run(load_cases([EXAMPLES / name / "cases.jsonl"], cfg), cfg, "ci", AS_OF)
    passing = sum(c.state == "pass" for c in offline.cases)
    assert f"{passing} of {len(offline.cases)} pass" in row, row


def _load(path: Path) -> EvalRun:
    return EvalRun.model_validate_json(path.read_text(encoding="utf-8"))
