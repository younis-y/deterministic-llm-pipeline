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


DATED = build_model({"due": FactSpec(type="date")})


def test_a_date_must_carry_its_quotes_year() -> None:
    item = Item(id="1", fields={"text": "Tenders close on 12 November 2026."})
    wrong = check_quotes(
        DATED.model_validate({"due": {"value": "2027-11-12", "quote": "12 November 2026"}}), item
    )
    right = check_quotes(
        DATED.model_validate({"due": {"value": "2026-11-12", "quote": "12 November 2026"}}), item
    )
    no_year = check_quotes(
        DATED.model_validate({"due": {"value": "2026-11-12", "quote": "Tenders close"}}), item
    )
    assert wrong.due.value is None  # type: ignore[attr-defined]
    assert str(right.due.value) == "2026-11-12"  # type: ignore[attr-defined]
    assert no_year.due.value is not None  # type: ignore[attr-defined]


def test_a_blank_quote_is_no_quote() -> None:
    for blank in (" ", "\n", chr(0x00A0)):
        facts = _check({"note": {"value": "x", "quote": blank}})
        assert facts.note.value is None, repr(blank)


def test_a_piece_of_a_grouped_number_is_not_the_number() -> None:
    for piece in (540, 12, 0):
        got = _check({"amount": {"value": piece, "quote": "Total due: £12,500.00"}})
        assert got.amount.value is None, piece
    assert _check({"amount": {"value": 12500, "quote": "Total due: £12,500.00"}}).amount.value


def test_a_number_keeps_its_sign_and_a_range_is_not_negative() -> None:
    item = Item(id="2", fields={"text": "Total due: £12,500. Balance: -£1,240.00, range 0-2."})

    def kept(value: float, quote: str) -> object:
        facts = MODEL.model_validate({"amount": {"value": value, "quote": quote}})
        return check_quotes(facts, item).amount.value  # type: ignore[attr-defined]

    assert kept(-12500, "Total due: £12,500") is None
    assert kept(-1240, "Balance: -£1,240.00") == -1240
    assert kept(1240, "Balance: -£1,240.00") is None
    assert kept(2, "range 0-2") == 2


def test_a_scale_after_a_number_counts() -> None:
    item = Item(id="3", fields={"text": "Budget £40k, or £2.5m, or 1.2 million; 5 minutes."})
    for value, quote in [(40000, "Budget £40k"), (2500000, "£2.5m"), (1200000, "1.2 million")]:
        facts = MODEL.model_validate({"amount": {"value": value, "quote": quote}})
        assert check_quotes(facts, item).amount.value == value, quote  # type: ignore[attr-defined]
    facts = MODEL.model_validate({"amount": {"value": 5000000, "quote": "5 minutes"}})
    assert check_quotes(facts, item).amount.value is None  # type: ignore[attr-defined]
