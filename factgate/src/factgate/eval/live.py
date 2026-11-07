"""Eval runs: each case several times, ONE state per case, never an average.

From the owner's `intensive.py` (Judge the Agent hackathon, 4 November 2026):
a case passes only if every attempt passed, fails if none did, and is flaky
if both happened. An attempt that could not be graded (the backend raised,
the answer did not fit the schema) is inconclusive, and inconclusive is never
a pass: it is counted outside the pass rate.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from factgate.backends import Backend, render_user
from factgate.config import Config
from factgate.eval.cases import Case
from factgate.eval.grade import grade
from factgate.pipeline import Resolver, finish, load_resolvers
from factgate.schema import build_model, system_prompt

Outcome = Literal["pass", "fail", "inconclusive"]


class Attempt(BaseModel):
    outcome: Outcome
    #: The model's raw answer, kept so `regrade` can re-score it for free.
    raw: dict[str, Any] | None = None
    fact_failures: list[str] = []
    outcome_failure: str | None = None
    dangerous: bool = False
    error: str = ""


class CaseRun(BaseModel):
    id: str
    heldout: bool = False
    tags: list[str] = []
    open_defect: str = ""
    state: str
    attempts: list[Attempt]
    #: The model's facts differed between conclusive attempts: model noise,
    #: not a grading or rule problem.
    facts_changed: bool = False


class EvalRun(BaseModel):
    label: str
    mode: Literal["offline", "live", "regrade"]
    date: str = ""
    backend: str = ""
    model: str = ""
    repeats: int = 1
    seconds: float = 0
    tokens: dict[str, int] = {}
    pass_at_1: float = 0.0
    pass_hat_k: float = 0.0
    #: Offline-only cases a live run left out.
    offline_only: int = 0
    cases: list[CaseRun]

    def unexpected(self) -> list[str]:
        """Cases whose result breaks CI: a new failure, or an open defect that passed."""
        out = []
        for c in self.cases:
            if c.open_defect and c.state == "pass":
                out.append(f"{c.id} passes: remove its open_defect")
            elif not c.open_defect and c.state in {"fail", "flaky"}:
                out.append(f"{c.id} {c.state}")
        return out


def case_state(outcomes: list[str]) -> str:
    seen = set(outcomes)
    if "pass" in seen and "fail" in seen:
        return "flaky"
    if "fail" in seen:
        return "fail"
    if "inconclusive" in seen:
        return "inconclusive"
    return "pass"


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score 95% interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


class _Grader:
    def __init__(self, cfg: Config, as_of: dt.date) -> None:
        self.cfg, self.as_of = cfg, as_of
        self.model = build_model(cfg.facts)
        self.resolvers: list[Resolver] = load_resolvers(cfg.resolvers, cfg.base)

    def attempt(self, case: Case, raw: dict[str, Any] | None, error: str = "") -> Attempt:
        if raw is None:
            return Attempt(outcome="inconclusive", error=error)
        when = case.as_of or self.as_of
        try:
            facts, decision = finish(raw, case.item, self.cfg, self.model, self.resolvers, when)
        except ValidationError as e:
            msg = f"invalid answer: {e.errors()[0]['msg']}"
            return Attempt(outcome="inconclusive", raw=raw, error=msg)
        got = grade(case, facts, decision, self.cfg, when)
        return Attempt(
            outcome="pass" if got.passed else "fail",
            raw=raw,
            fact_failures=got.fact_failures,
            outcome_failure=got.outcome_failure,
            dangerous=got.dangerous,
        )


def _case_run(case: Case, attempts: list[Attempt]) -> CaseRun:
    answers = {json.dumps(a.raw, sort_keys=True) for a in attempts if a.outcome != "inconclusive"}
    return CaseRun(
        id=case.id,
        heldout=case.heldout,
        tags=case.tags,
        open_defect=case.open_defect,
        state=case_state([a.outcome for a in attempts]),
        attempts=attempts,
        facts_changed=len(answers) > 1,
    )


def _finish_run(run: EvalRun) -> EvalRun:
    attempts = [a for c in run.cases for a in c.attempts if a.outcome != "inconclusive"]
    conclusive = [c for c in run.cases if c.state != "inconclusive"]
    run.pass_at_1 = sum(a.outcome == "pass" for a in attempts) / len(attempts) if attempts else 0
    run.pass_hat_k = (
        sum(c.state == "pass" for c in conclusive) / len(conclusive) if conclusive else 0
    )
    run.date = run.date or dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return run


def offline_run(cases: list[Case], cfg: Config, label: str, as_of: dt.date) -> EvalRun:
    """One attempt per case, the case's `model_facts` standing in for the model.

    A case with no `model_facts` (a held-out one, say) runs live only.
    """
    grader = _Grader(cfg, as_of)
    runs = [
        _case_run(c, [grader.attempt(c, c.model_facts)]) for c in cases if c.model_facts is not None
    ]
    return _finish_run(EvalRun(label=label, mode="offline", cases=runs))


async def live(
    cases: list[Case],
    cfg: Config,
    backend: Backend,
    *,
    repeats: int,
    label: str,
    as_of: dt.date,
) -> EvalRun:
    """Every case `repeats` times through the model; no cache. An
    offline_only case is left out: its expectation needs its own answer."""
    started = time.monotonic()
    skipped = sum(c.offline_only for c in cases)
    cases = [c for c in cases if not c.offline_only]
    grader = _Grader(cfg, as_of)
    system = system_prompt(cfg.facts)
    gate = asyncio.Semaphore(cfg.model.max_concurrent)
    tokens = {"input": 0, "output": 0}

    async def one(case: Case) -> Attempt:
        async with gate:
            try:
                answer = await backend.extract(
                    system, render_user(case.item), grader.model, case.id
                )
            except Exception as e:
                return grader.attempt(case, None, f"{type(e).__name__}: {e}")
        tokens["input"] += answer.input_tokens
        tokens["output"] += answer.output_tokens
        return grader.attempt(case, answer.raw)

    attempts = [await asyncio.gather(*(one(c) for c in cases)) for _ in range(repeats)]
    runs = [_case_run(c, [a[i] for a in attempts]) for i, c in enumerate(cases)]
    return _finish_run(
        EvalRun(
            label=label,
            mode="live",
            backend=backend.name,
            model=cfg.model.name,
            repeats=repeats,
            seconds=round(time.monotonic() - started, 1),
            tokens=tokens,
            offline_only=skipped,
            cases=runs,
        )
    )


def regrade(run: EvalRun, cases: list[Case], cfg: Config, as_of: dt.date) -> EvalRun:
    """`run`'s saved answers re-scored with today's config and cases; no calls.

    Measures a rule, resolver or case change with zero model noise and zero
    cost. Cases no longer in `cases` are left out.
    """
    grader = _Grader(cfg, as_of)
    by_id = {c.id: c for c in cases}
    runs = [
        _case_run(by_id[c.id], [grader.attempt(by_id[c.id], a.raw, a.error) for a in c.attempts])
        for c in run.cases
        if c.id in by_id
    ]
    again = run.model_copy(update={"label": f"{run.label}-regrade", "mode": "regrade", "date": ""})
    again.cases = runs
    return _finish_run(again)
