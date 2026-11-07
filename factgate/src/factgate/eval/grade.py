"""Grading: does a case's finished facts and outcome meet what it expects?"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel

from factgate.config import Config, Dangerous
from factgate.eval.cases import Case
from factgate.pipeline import Resolver, finish
from factgate.rules import Decision, matches, value_of


class Grade(BaseModel):
    passed: bool
    #: `fact=value` for every expected fact that missed.
    fact_failures: list[str] = []
    #: `outcome=<got>` when the outcome missed.
    outcome_failure: str | None = None
    #: The miss is one the config names as dangerous.
    dangerous: bool = False


def grade(case: Case, facts: BaseModel, decision: Decision, cfg: Config, as_of: dt.date) -> Grade:
    when = case.as_of or as_of
    fact_failures = [
        f"{name}={value_of(facts, name)!r}"
        for name, cond in case.expect_facts.items()
        if not matches(value_of(facts, name), cond, when)
    ]
    expected = case.expect_outcome
    missed = expected is not None and decision.outcome != expected
    dangerous = missed and Dangerous(expected=str(expected), got=decision.outcome) in cfg.dangerous
    return Grade(
        passed=not fact_failures and not missed,
        fact_failures=fact_failures,
        outcome_failure=f"outcome={decision.outcome}" if missed else None,
        dangerous=dangerous,
    )


def offline(
    case: Case,
    cfg: Config,
    model: type[BaseModel],
    resolvers: list[Resolver],
    as_of: dt.date,
) -> Grade:
    """`grade` with the case's own `model_facts` standing in for the model."""
    when = case.as_of or as_of
    facts, decision = finish(case.model_facts or {}, case.item, cfg, model, resolvers, when)
    return grade(case, facts, decision, cfg, when)
