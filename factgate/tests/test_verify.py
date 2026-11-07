from typing import Any

from factgate.items import Item
from factgate.schema import FactSpec, build_model
from factgate.verify import check_quotes

MODEL = build_model(
    {
        "amount": FactSpec(type="number"),
        "count": FactSpec(type="integer"),
        "note": FactSpec(type="text"),
        "tags": FactSpec(type="list", of="text"),
    }
)
ITEM = Item(
    id="1",
    fields={"subject": "Invoice 42 overdue", "body": "Total due: £12,500.00 — “final”"},
)


def _check(raw: dict[str, object]) -> Any:
    return check_quotes(MODEL.model_validate(raw), ITEM)


def test_a_quote_across_two_fields_is_cleared() -> None:
    facts = _check({"note": {"value": "x", "quote": "overdue Total due"}})
    assert facts.note.value is None and facts.note.quote == ""


def test_typographic_folding_verifies() -> None:
    facts = _check({"note": {"value": "final", "quote": '-  "FINAL"'}})
    assert facts.note.value == "final"


def test_a_number_must_be_in_its_own_quote() -> None:
    wrong = _check({"amount": {"value": 125000, "quote": "Total due: £12,500"}})
    right = _check({"amount": {"value": 12500, "quote": "Total due: £12,500"}})
    assert wrong.amount.value is None
    assert right.amount.value == 12500


def test_an_integer_with_no_number_in_its_quote_is_cleared() -> None:
    facts = _check({"count": {"value": 3, "quote": "Invoice"}})
    assert facts.count.value is None


def test_list_elements_are_checked_one_by_one() -> None:
    facts = _check(
        {"tags": [{"value": "a", "quote": "Invoice 42"}, {"value": "b", "quote": "made up"}]}
    )
    assert [t.value for t in facts.tags] == ["a"]


def test_an_unstated_fact_stays_unstated() -> None:
    facts = _check({})
    assert facts.note.value is None
