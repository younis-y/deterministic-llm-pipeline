"""Write heldout.jsonl: invoices whose right answer is computed, not guessed.

    python examples/invoices/gen_heldout.py

A fix tuned on the visible cases can pass them and still be wrong, so these
cases test the same rules with new values and wordings: three for each rule
that fires, three for the default, and three approved on a threshold (an
amount of exactly 10,000, a due date of today). Each answer is the config's own rules
applied to the facts the generator wrote into the text. These rules settle
every combination; an oracle for rules that leave a case open must skip it
rather than guess, or it grades the model on the guess. Adapted from the
Judge the Agent hackathon's `gen_orders.py`.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import random
from pathlib import Path

from factgate.config import load_config
from factgate.rules import Decision, decide
from factgate.schema import build_model

HERE = Path(__file__).parent
CFG = load_config(HERE / "config.yaml")
FACTS = build_model(CFG.facts)
AS_OF = dt.date(2026, 11, 7)
#: Cases kept per rule that fires (and for the default, and on a threshold).
PER_GROUP = 3

SUPPLIERS = ["Alder Print", "Brook Couriers", "Cobalt Labs", "Dunmore Electrical"]
AMOUNTS = [420.0, 3150.5, 9800.0, 10000.0, 14200.0]
CURRENCIES = {"GBP": "£{:,.2f}", "EUR": "EUR {:,.2f}", "USD": "USD {:,.2f}"}
DUE_IN_DAYS = [-12, 0, 9, 26]
AMOUNT_LINES = [
    "Amount due: {money}.",
    "Subtotal {sub}, VAT {vat}, total due {money}.",
    "Please pay {money} in full.",
    "Net {sub} plus VAT {vat}. Total payable: {money}.",
]
DUE_LINES = ["Payment due by {date}.", "Due date: {date}.", "Please pay by {date}."]
#: (wording, disputed?) - one says the word and means the opposite.
STATUS_LINES = [
    ("", False),
    ("This invoice is on hold until the delivery query is resolved.", True),
    ("Nothing on this invoice is disputed.", False),
]


def money(fmt: str, value: float) -> str:
    return fmt.format(value)


def human(day: dt.date) -> str:
    return f"{day.day} {day:%B %Y}"


def decision(
    amount: float, currency: str, due: dt.date, po: str | None, disputed: bool
) -> Decision:
    """The config's rules on the true facts."""
    facts = FACTS.model_validate(
        {
            "amount_due": {"value": amount, "quote": "x"},
            "currency": {"value": currency, "quote": "x"},
            "due_date": {"value": due.isoformat(), "quote": "x"},
            "disputed": {"value": disputed, "quote": "x"} if disputed else {},
            "po_number": {"value": po, "quote": "x"} if po else {},
        }
    )
    return decide(facts, CFG.rules, CFG.default_outcome, AS_OF)


def outcome(amount: float, currency: str, due: dt.date, po: str | None, disputed: bool) -> str:
    return decision(amount, currency, due, po, disputed).outcome


# Self-check against answers worked out by hand before trusting the oracle.
assert outcome(860, "GBP", dt.date(2026, 12, 12), "PO-1", False) == "approve"
assert outcome(12540, "GBP", dt.date(2026, 12, 15), "PO-1", False) == "review"
assert outcome(640, "GBP", dt.date(2026, 11, 20), "PO-1", True) == "hold"
assert outcome(980, "GBP", dt.date(2026, 10, 1), "PO-1", False) == "review"
assert outcome(3800, "EUR", dt.date(2026, 12, 10), "PO-1", False) == "review"


def cases() -> list[dict[str, object]]:
    combos = list(itertools.product(AMOUNTS, CURRENCIES, DUE_IN_DAYS, (True, False), STATUS_LINES))
    random.Random(7).shuffle(combos)  # a spread of wordings and values, the same every run
    kept: dict[str, list[dict[str, object]]] = {}
    for n, (amount, currency, days, has_po, (status, disputed)) in enumerate(combos):
        due = AS_OF + dt.timedelta(days=days)
        po = f"PO-{90000 + n}" if has_po else None
        decided = decision(amount, currency, due, po, disputed)
        # On a threshold and decided by it: 10,000 is not over 10,000, today is not overdue.
        on_threshold = (amount == 10000 or days == 0) and decided.rule is None
        group = kept.setdefault("threshold" if on_threshold else decided.rule or "default", [])
        if len(group) == PER_GROUP:
            continue
        fmt = CURRENCIES[currency]
        sub = round(amount / 1.2, 2)
        line = AMOUNT_LINES[n % len(AMOUNT_LINES)].format(
            money=money(fmt, amount), sub=money(fmt, sub), vat=money(fmt, round(amount - sub, 2))
        )
        number = 4000 + n
        supplier = SUPPLIERS[n % len(SUPPLIERS)]
        body = " ".join(
            part
            for part in (
                f"Invoice {number} from {supplier}" + (f", quoting {po}." if po else "."),
                line,
                DUE_LINES[n % len(DUE_LINES)].format(date=human(due)),
                status,
            )
            if part
        )
        group.append(
            {
                "fields": {"subject": f"Invoice {number} - {supplier}", "body": body},
                "expect": {"facts": {"amount_due": {"eq": amount}}, "outcome": decided.outcome},
                "tags": [decided.rule or "default"] + (["threshold"] if on_threshold else []),
                "as_of": AS_OF.isoformat(),
            }
        )
    written = [case for group in kept.values() for case in group]
    return [{"id": f"h-inv-{i:03d}", **c} for i, c in enumerate(written, 1)]


def main() -> None:
    written = cases()
    path = HERE / "heldout.jsonl"
    path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in written), encoding="utf-8"
    )
    print(f"wrote {len(written)} cases to {path}")


if __name__ == "__main__":
    main()
