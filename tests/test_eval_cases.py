"""`evals/cases.json` through the code alone: no model, no network.

A case with an `open_defect` fails today and runs as a strict xfail, so a fix
fails this test until the case's `open_defect` is deleted, and CI then holds
the fix. A new failure fails CI outright.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals.harness import CASES, offline


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            c,
            id=c["id"],
            marks=[pytest.mark.xfail(strict=True, reason=c["open_defect"])]
            if "open_defect" in c
            else [],
        )
        for c in CASES
    ],
)
def test_case(case: dict[str, Any]) -> None:
    assert offline(case) == [], case.get("note", "")
