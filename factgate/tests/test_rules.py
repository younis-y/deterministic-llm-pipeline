import datetime as dt
from typing import Any

import pytest

from factgate.rules import Rule, decide, matches
from factgate.schema import FactSpec, build_model

AS_OF = dt.date(2026, 11, 7)


@pytest.mark.parametrize(
    ("value", "cond", "expected"),
    [
        (True, {"is": True}, True),
        (False, {"is": True}, False),
        ("GBP", {"eq": "GBP"}, True),
        ("GBP", {"ne": "GBP"}, False),
        (5, {"lt": 6}, True),
        (6, {"lt": 6}, False),
        (6, {"le": 6}, True),
        (7, {"gt": 6}, True),
        (6, {"ge": 7}, False),
        ("EUR", {"in": ["GBP", "EUR"]}, True),
        ("USD", {"in": ["GBP", "EUR"]}, False),
        ("Final Notice", {"contains": "notice"}, True),
        ("Reminder", {"contains": "notice"}, False),
        ("PO-1234", {"matches": r"^po-\d+$"}, True),
        ("PO-12x", {"matches": r"^po-\d+$"}, False),
        (None, {"stated": False}, True),
        (None, {"gt": 0}, False),
        (None, {"ne": "GBP"}, False),
        (3, {"stated": True}, True),
        (5, {"gt": 1, "lt": 4}, False),
        (dt.date(2026, 11, 20), {"within_days": 14}, True),
        (dt.date(2026, 11, 22), {"within_days": 14}, False),
        (dt.date(2026, 11, 1), {"within_days": 14}, False),
        (dt.date(2026, 11, 6), {"before_days": 0}, True),
        (dt.date(2026, 11, 7), {"before_days": 0}, False),
        (dt.date(2026, 11, 20), {"gt": "2026-11-19"}, True),
        (dt.date(2026, 11, 1), {"in": ["2026-11-01"]}, True),
        (dt.date(2026, 11, 1), {"in": [dt.date(2026, 11, 2)]}, False),
        (["A", "B"], {"contains": "B"}, True),
        (["A", "B"], {"contains": "C"}, False),
        ([], {"stated": False}, True),
        ([3, 9], {"gt": 8}, True),
    ],
)
def test_operators(value: Any, cond: dict[str, Any], expected: bool) -> None:
    assert matches(value, cond, AS_OF) is expected


MODEL = build_model({"amount": FactSpec(type="number"), "disputed": FactSpec(type="bool")})
RULES = [
    Rule(name="disputed", when={"disputed": {"is": True}}, then="hold"),
    Rule(name="large", when={"amount": {"gt": 10000}}, then="review"),
]


def test_first_match_wins_and_quotes_the_fired_facts() -> None:
    facts = MODEL.model_validate(
        {
            "amount": {"value": 20000, "quote": "20,000"},
            "disputed": {"value": True, "quote": "on hold"},
        }
    )
    decision = decide(facts, RULES, "approve", AS_OF)
    assert (decision.outcome, decision.rule) == ("hold", "disputed")
    assert decision.quotes == {"disputed": "on hold"}


def test_default_when_nothing_matches() -> None:
    decision = decide(MODEL.model_validate({}), RULES, "approve", AS_OF)
    assert (decision.outcome, decision.rule, decision.quotes) == ("approve", None, {})
