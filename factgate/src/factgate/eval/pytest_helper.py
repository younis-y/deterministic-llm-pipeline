"""Run a config's cases inside your own pytest suite.

    from factgate.eval.pytest_helper import assert_case, case_params

    @pytest.mark.parametrize("bound", case_params("config.yaml", "cases.jsonl"))
    def test_case(bound):
        assert_case(bound)

A case with `open_defect` runs as a strict expected failure, so fixing it
fails the suite until the mark is removed, and the fix is then held. A case
with no `model_facts` runs live only and is left out.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, NamedTuple

from factgate.config import Config, load_config
from factgate.eval.cases import Case, load_cases
from factgate.eval.grade import offline
from factgate.pipeline import load_resolvers
from factgate.schema import build_model


class Bound(NamedTuple):
    cfg: Config
    case: Case


def case_params(config: str | Path, *paths: str | Path) -> list[Any]:
    import pytest

    cfg = load_config(Path(config))
    cases = load_cases([Path(p) for p in paths], cfg)
    return [
        pytest.param(
            Bound(cfg, c),
            id=c.id,
            marks=[pytest.mark.xfail(strict=True, reason=c.open_defect)] if c.open_defect else [],
        )
        for c in cases
        if c.model_facts is not None
    ]


def assert_case(bound: Bound, as_of: dt.date | None = None) -> None:
    cfg, case = bound
    model = build_model(cfg.facts)
    today = as_of or dt.datetime.now(dt.UTC).date()
    got = offline(case, cfg, model, load_resolvers(cfg.resolvers, cfg.base), today)
    assert got.passed, f"{case.id}: {got.fact_failures} {got.outcome_failure or ''} {case.note}"
