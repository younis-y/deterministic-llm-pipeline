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
from factgate.eval.live import EvalRun, offline_run, wilson
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
    written = oracle.cases()
    on_disk = (EXAMPLES / "invoices" / "heldout.jsonl").read_text(encoding="utf-8")
    assert [json.loads(line) for line in on_disk.splitlines()] == written


def _latest_live(name: str) -> EvalRun:
    runs = [_load(p) for p in (EXAMPLES / name / "results").glob("*.json")]
    live = [r for r in runs if r.mode == "live"]
    assert live, f"no live run recorded for {name}"
    return max(live, key=lambda r: r.date)


def measured_row(name: str) -> str:
    """The figures the README's Measured table must show for `name`, cell by cell."""
    run = _latest_live(name)
    cfg = load_config(EXAMPLES / name / "config.yaml")
    offline = offline_run(load_cases([EXAMPLES / name / "cases.jsonl"], cfg), cfg, "ci", AS_OF)
    held = sum(c.heldout for c in run.cases)
    conclusive = [c for c in run.cases if c.state != "inconclusive"]
    lo, hi = wilson(sum(c.state == "pass" for c in conclusive), len(conclusive))
    passing = sum(c.state == "pass" for c in offline.cases)
    defects = sum(bool(c.open_defect) for c in offline.cases)
    offline_cell = f"{passing} of {len(offline.cases)} pass" + (
        f", {defects} open defect" if defects else ""
    )
    return (
        f"| {len(run.cases)} ({len(run.cases) - held} + {held}) "
        f"| pass@1 {run.pass_at_1:.0%}, pass^{run.repeats} {run.pass_hat_k:.0%}, "
        f"Wilson 95% {lo:.0%}-{hi:.0%} | {offline_cell} |"
    )


@pytest.mark.parametrize("name", NAMES)
def test_the_readme_row_is_the_latest_live_run(name: str) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme_matches(readme, _latest_live(name)) == []
    [row] = [line for line in readme.splitlines() if f"(examples/{name})" in line]
    assert row.endswith(measured_row(name)), f"{row}\nshould end {measured_row(name)}"


def test_the_readme_totals_are_the_latest_live_runs() -> None:
    runs = [_latest_live(name) for name in NAMES]
    calls = sum(len(a) for r in runs for c in r.cases for a in [c.attempts])
    seconds = round(sum(r.seconds for r in runs))
    tokens = sum(r.tokens.get("output", 0) for r in runs)
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert f"The {calls} live calls took {seconds} seconds and {tokens:,} output tokens" in readme


def _load(path: Path) -> EvalRun:
    return EvalRun.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
def test_the_batch_check_matches_the_direct_run(name: str) -> None:
    """model.batch was checked live on the same cases: no case may have moved
    between pass and fail, or become inconclusive (a move into or out of
    flaky is model noise)."""
    from factgate.eval.compare import compare

    [path] = (EXAMPLES / name / "results" / "batch").glob("*.json")
    batch = _load(path)
    assert batch.mode == "live" and batch.repeats == 3
    for moves in compare(_latest_live(name), batch):
        assert (moves.fixed, moves.broke, moves.inconclusive) == ([], [], []), moves
        assert (moves.only_before, moves.only_after) == ([], []), moves
