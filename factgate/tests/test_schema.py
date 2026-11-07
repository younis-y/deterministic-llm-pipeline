import datetime as dt
from typing import Any

import pytest
from pydantic import ValidationError

from factgate.schema import Fact, FactSpec, build_model, system_prompt

SPECS = {
    "amount": FactSpec(type="number", describe="Total due"),
    "currency": FactSpec(type="enum", values=["GBP", "EUR"]),
    "due": FactSpec(type="date"),
    "codes": FactSpec(type="list", of="text"),
}


def test_build_model_types_and_defaults() -> None:
    model = build_model(SPECS)
    facts: Any = model.model_validate(
        {
            "amount": {"value": 12.5, "quote": "12.50"},
            "due": {"value": "2026-11-20", "quote": "20 Nov"},
            "codes": [{"value": "A1", "quote": "A1"}],
        }
    )
    assert facts.amount == Fact[float](value=12.5, quote="12.50")
    assert facts.due.value == dt.date(2026, 11, 20)
    assert facts.currency.value is None
    assert facts.codes[0].value == "A1"


def test_enum_outside_values_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_model(SPECS).model_validate({"currency": {"value": "JPY", "quote": "JPY"}})


def test_enum_needs_values_and_list_needs_of() -> None:
    with pytest.raises(ValidationError, match="values"):
        FactSpec(type="enum")
    with pytest.raises(ValidationError, match="of"):
        FactSpec(type="list")


def test_prompt_names_every_fact_and_value() -> None:
    prompt = system_prompt(SPECS)
    for word in ("amount", "Total due", "currency", "GBP", "EUR", "due", "codes", "verbatim"):
        assert word in prompt
