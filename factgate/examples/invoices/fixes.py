"""Resolvers for the invoice example: code that corrects the model.

A resolver takes the item and the checked facts and returns corrected facts.
factgate checks the quotes again afterwards, so a resolver quotes the item
verbatim, as the model must.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from factgate.items import Item

#: Words that mark an amount as something other than the total now due.
_NOT_THE_TOTAL = re.compile(r"\b(?:subtotal|vat|previous balance|paid)\b", re.IGNORECASE)
#: "total due £12,540.00", "Amount now due: £860.00", "Amount due EUR 3,800.00":
#: a money amount, never a bare number ("Amount due 30 days from ...").
_TOTAL = re.compile(
    r"(?:total|amount)(?: now)? due:?\s*(?:£|€|\$|EUR |USD )\s?(\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def total_not_subtotal(item: Item, facts: BaseModel) -> BaseModel:
    """Replace an amount the model read from a subtotal, a VAT line or an
    earlier balance with the item's own "total due" figure, when it has one."""
    fact = facts.amount_due  # type: ignore[attr-defined]
    if fact.value is None or not _NOT_THE_TOTAL.search(fact.quote):
        return facts
    for text in item.fields.values():
        found = _TOTAL.search(text)
        if found:
            amount = float(found.group(1).replace(",", ""))
            return facts.model_copy(
                update={"amount_due": type(fact)(value=amount, quote=found.group(0))}
            )
    return facts
