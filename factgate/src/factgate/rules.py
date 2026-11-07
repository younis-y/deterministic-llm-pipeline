"""Rules: fixed conditions on the checked facts. Only code decides the outcome."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

Condition = dict[str, Any]


def _date(arg: Any) -> Any:
    return dt.date.fromisoformat(arg) if isinstance(arg, str) else arg


def _cmp(op: Callable[[Any, Any], bool]) -> Callable[[Any, Any, dt.date], bool]:
    def check(value: Any, arg: Any, _as_of: dt.date) -> bool:
        if isinstance(value, dt.date):
            arg = _date(arg)
        return op(value, arg)

    return check


#: Operator -> test on one stated value. `stated` is handled before these.
_SCALAR: dict[str, Callable[[Any, Any, dt.date], bool]] = {
    "is": _cmp(lambda v, a: v == a),
    "eq": _cmp(lambda v, a: v == a),
    "ne": _cmp(lambda v, a: v != a),
    "lt": _cmp(lambda v, a: v < a),
    "le": _cmp(lambda v, a: v <= a),
    "gt": _cmp(lambda v, a: v > a),
    "ge": _cmp(lambda v, a: v >= a),
    "in": lambda v, a, _: v in ([_date(x) for x in a] if isinstance(v, dt.date) else a),
    "contains": lambda v, a, _: str(a).casefold() in str(v).casefold(),
    "matches": lambda v, a, _: re.search(a, str(v), re.IGNORECASE) is not None,
    "within_days": lambda v, a, d: d <= v <= d + dt.timedelta(days=a),
    "before_days": lambda v, a, d: v < d + dt.timedelta(days=a),
}
OPERATORS = frozenset({*_SCALAR, "stated"})
#: Which operators make sense for which fact type (checked when a config loads).
ALLOWED: dict[str, frozenset[str]] = {
    "number": frozenset({"is", "eq", "ne", "lt", "le", "gt", "ge", "in", "stated"}),
    "integer": frozenset({"is", "eq", "ne", "lt", "le", "gt", "ge", "in", "stated"}),
    "date": frozenset(OPERATORS - {"contains", "matches"}),
    "text": frozenset({"is", "eq", "ne", "in", "contains", "matches", "stated"}),
    "enum": frozenset({"is", "eq", "ne", "in", "stated"}),
    "bool": frozenset({"is", "eq", "ne", "stated"}),
}


def _one(op: str, value: Any, arg: Any, as_of: dt.date) -> bool:
    if isinstance(value, list):
        if op == "stated":
            return bool(value) is bool(arg)
        if op == "contains":
            return arg in value
        return any(_one(op, v, arg, as_of) for v in value)
    if op == "stated":
        return (value is not None) is bool(arg)
    return value is not None and _SCALAR[op](value, arg, as_of)


def matches(value: Any, cond: Condition, as_of: dt.date) -> bool:
    """Whether a fact's value meets every operator in `cond`.

    An unstated value (None) meets nothing but `stated: false`. On a list an
    operator holds if any entry meets it; `contains` means an entry equals the
    argument and `stated` means the list is not empty.
    """
    return all(_one(op, value, arg, as_of) for op, arg in cond.items())


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    #: fact name -> condition; every fact's condition must hold.
    when: dict[str, Condition]
    then: str


class Decision(BaseModel):
    outcome: str
    rule: str | None = None
    #: The quotes of the facts the fired rule read.
    quotes: dict[str, str | list[str]] = {}


def value_of(facts: BaseModel, name: str) -> Any:
    fact = getattr(facts, name)
    return [f.value for f in fact] if isinstance(fact, list) else fact.value


def quote_of(facts: BaseModel, name: str) -> str | list[str]:
    fact = getattr(facts, name)
    return [f.quote for f in fact] if isinstance(fact, list) else str(fact.quote)


def decide(facts: BaseModel, rules: list[Rule], default: str, as_of: dt.date) -> Decision:
    """The first rule whose conditions all hold, else `default`."""
    for rule in rules:
        if all(matches(value_of(facts, f), c, as_of) for f, c in rule.when.items()):
            quotes = {f: quote_of(facts, f) for f in rule.when}
            return Decision(outcome=rule.then, rule=rule.name, quotes=quotes)
    return Decision(outcome=default)
