import datetime as dt
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from factgate.backends import Answer, Backend
from factgate.cli import app
from factgate.config import Config, ModelConfig
from factgate.eval.cases import load_cases
from factgate.eval.compare import compare
from factgate.eval.live import EvalRun, case_state, live, offline_run, regrade, wilson
from factgate.eval.report import readme_matches, report_markdown

AS_OF = dt.date(2026, 11, 7)
CFG_DATA: dict[str, Any] = {
    "name": "t",
    "items": {"jsonl": "x"},
    "facts": {"amount": {"type": "number"}},
    "rules": [{"name": "big", "when": {"amount": {"gt": 100}}, "then": "review"}],
    "default_outcome": "ok",
    "model": {"name": "m", "max_concurrent": 1},
}
CFG = Config.model_validate(CFG_DATA)


def _cases(tmp_path: Path) -> list[Any]:
    rows = [
        {"id": "big", "text": "Total 500", "expect": {"outcome": "review"}, "tags": ["money"]},
        {"id": "small", "text": "Total 50", "expect": {"outcome": "ok"}},
    ]
    held = [{"id": "h1", "text": "SECRET Total 900", "expect": {"outcome": "review"}}]
    a, h = tmp_path / "a.jsonl", tmp_path / "h.jsonl"
    a.write_text("\n".join(json.dumps(r) for r in rows))
    h.write_text("\n".join(json.dumps(r) for r in held))
    return load_cases([a], CFG, heldout=[h])


@pytest.mark.parametrize(
    ("outcomes", "state"),
    [
        (["pass", "pass"], "pass"),
        (["pass", "fail"], "flaky"),
        (["fail", "fail"], "fail"),
        (["pass", "inconclusive"], "inconclusive"),
        (["fail", "inconclusive"], "fail"),
        (["inconclusive"], "inconclusive"),
    ],
)
def test_case_state(outcomes: list[str], state: str) -> None:
    assert case_state(outcomes) == state


def test_wilson() -> None:
    lo, hi = wilson(7, 8)
    assert (round(lo, 3), round(hi, 3)) == (0.529, 0.978)
    assert wilson(0, 0) == (0.0, 0.0)


class Scripted(Backend):
    """Answers each case from a script: a number, or an exception to raise."""

    name = "scripted"

    def __init__(self, script: dict[str, list[Any]]) -> None:
        super().__init__(ModelConfig(name="m"))
        self.script = script
        self.calls = 0

    async def extract(self, system: str, user: str, model: type[BaseModel], item_id: str) -> Answer:
        self.calls += 1
        value = self.script[item_id].pop(0)
        if isinstance(value, Exception):
            raise value
        text = user.split("\n", 1)[1]
        return Answer(raw={"amount": {"value": value, "quote": text}}, output_tokens=3)


async def test_live_states_and_inconclusive(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    backend = Scripted({"big": [500, 500], "small": [50, RuntimeError("down")], "h1": [900, 1]})
    run = await live(cases, CFG, backend, repeats=2, label="base", as_of=AS_OF)
    states = {c.id: c.state for c in run.cases}
    assert states == {"big": "pass", "small": "inconclusive", "h1": "flaky"}
    # 1 inconclusive attempt is out of pass@1's denominator: 4 passes of 5.
    assert run.pass_at_1 == pytest.approx(4 / 5)
    assert run.pass_hat_k == pytest.approx(1 / 2)  # 'small' is out of it
    flaky = next(c for c in run.cases if c.id == "h1")
    assert flaky.facts_changed
    assert run.tokens["output"] == 15


async def test_live_leaves_out_offline_only_cases(tmp_path: Path) -> None:
    """An offline_only case grades what the code does with ITS model answer
    (a misread number the quote check clears, say): a live model answers
    differently, so asking it would grade nothing real."""
    path = tmp_path / "a.jsonl"
    rows = [
        {"id": "big", "text": "Total 500", "expect": {"outcome": "review"}},
        {
            "id": "misread",
            "text": "Total 500",
            "expect": {"facts": {"amount": {"stated": False}}},
            "model_facts": {"amount": {"value": 5000, "quote": "Total 500"}},
            "offline_only": True,
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    cases = load_cases([path], CFG)
    backend = Scripted({"big": [500]})
    run = await live(cases, CFG, backend, repeats=1, label="l", as_of=AS_OF)
    assert [c.id for c in run.cases] == ["big"] and backend.calls == 1
    assert run.offline_only == 1
    assert "1 offline-only case" in report_markdown(run, CFG)
    assert [c.id for c in offline_run(cases, CFG, "o", AS_OF).cases] == ["misread"]


async def test_the_report_hides_heldout_text(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    run = await live(
        cases,
        CFG,
        Scripted({"big": [5], "small": [50], "h1": [1]}),
        repeats=1,
        label="base",
        as_of=AS_OF,
    )
    md = report_markdown(run, CFG)
    assert "h1" in md and "SECRET" not in md and "amount=" not in md.split("Held-out")[1]
    assert "Wilson 95%" in md


def _run(label: str, states: dict[str, str], heldout: set[str] = frozenset()) -> EvalRun:  # type: ignore[assignment]
    return EvalRun.model_validate(
        {
            "label": label,
            "mode": "live",
            "cases": [
                {"id": i, "state": s, "heldout": i in heldout, "attempts": []}
                for i, s in states.items()
            ],
        }
    )


def test_compare_counts_only_stable_moves() -> None:
    before = _run("a", {"f": "fail", "p": "pass", "x": "flaky", "i": "pass", "h": "fail"}, {"h"})
    after = _run(
        "b", {"f": "pass", "p": "fail", "x": "pass", "i": "inconclusive", "h": "pass"}, {"h"}
    )
    visible, held = compare(before, after)
    assert (visible.fixed, visible.broke, visible.noise, visible.inconclusive) == (
        ["f"],
        ["p"],
        ["x"],
        ["i"],
    )
    assert held.fixed == ["h"] and held.split == "held-out"


async def test_regrade_flips_a_case_after_a_rule_change_with_no_calls(tmp_path: Path) -> None:
    cases = _cases(tmp_path)
    backend = Scripted({"big": [500], "small": [50], "h1": [900]})
    run = await live(cases, CFG, backend, repeats=1, label="base", as_of=AS_OF)
    stricter = Config.model_validate(
        {**CFG_DATA, "rules": [{"name": "big", "when": {"amount": {"gt": 1000}}, "then": "review"}]}
    )
    again = regrade(run, cases, stricter, AS_OF)
    assert backend.calls == 3
    assert again.label == "base-regrade"
    assert {c.id: c.state for c in again.cases}["big"] == "fail"


def test_offline_run_marks_open_defects(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "d",
                "text": "Total 500",
                "open_defect": "known",
                "expect": {"outcome": "review"},
                "model_facts": {},
            }
        )
    )
    run = offline_run(load_cases([path], CFG), CFG, "ci", AS_OF)
    assert run.cases[0].state == "fail" and run.cases[0].open_defect == "known"
    assert run.unexpected() == []


def test_offline_leaves_out_cases_with_no_model_answer(tmp_path: Path) -> None:
    """A case with no `model_facts` (a held-out one, say) has nothing to feed
    the code offline: it runs live only, and never fails CI for it."""
    run = offline_run(_cases(tmp_path), CFG, "ci", AS_OF)
    assert run.cases == [] and run.unexpected() == []


def test_readme_matches_names_missing_figures(tmp_path: Path) -> None:
    run = _run("a", {"p": "pass"})
    run.pass_at_1, run.model, run.repeats = 1.0, "m", 3
    assert readme_matches("`m`, 3 runs: pass@1 100%", run) == []
    assert readme_matches("nothing", run) == ["`m`, 3 runs", "pass@1 100%"]


def test_cli_eval_run_offline_and_compare(tmp_path: Path) -> None:
    runner = CliRunner()
    folder = tmp_path / "s"
    runner.invoke(app, ["init", str(folder)])
    results = tmp_path / "results"
    args = [
        "eval",
        "run",
        str(folder / "config.yaml"),
        str(folder / "cases.jsonl"),
        "--results",
        str(results),
        "--as-of",
        "2026-11-07",
    ]
    first = runner.invoke(app, [*args, "--label", "a"])
    assert first.exit_code == 0, first.output
    runner.invoke(app, [*args, "--label", "b"])
    assert (results / "a.json").exists() and (results / "a.md").exists()
    cmp = runner.invoke(app, ["eval", "compare", str(results / "a.json"), str(results / "b.json")])
    assert cmp.exit_code == 0 and "Fixed" in cmp.output
