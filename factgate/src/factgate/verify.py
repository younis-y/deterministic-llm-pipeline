"""The quote check: a fact counts only if its quote is really in the item."""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel

from factgate.items import Item
from factgate.schema import Fact

#: Typographic look-alikes folded to their plain form on both sides, so a real
#: quote does not fail only on which quote mark or dash was used (rolescan).
_TYPO_FOLD = str.maketrans(
    {
        chr(0x2018): "'",
        chr(0x2019): "'",
        chr(0x201C): '"',
        chr(0x201D): '"',
        chr(0x2013): "-",
        chr(0x2014): "-",
        chr(0x2212): "-",
        chr(0x00A0): " ",
        chr(0x202F): " ",
    }
)
#: A number as written: plain ("42", "12.5") or grouped ("12,500.00", "12 500").
_NUMBER = re.compile(r"\d{1,3}(?:[, ]\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def normalise(text: str) -> str:
    return " ".join(text.translate(_TYPO_FOLD).split()).casefold()


def numbers_in(quote: str) -> list[float]:
    """Every number written in `quote`, separators and currency symbols ignored."""
    folded = quote.translate(_TYPO_FOLD)
    plain = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", folded)]
    grouped = [float(re.sub(r"[, ]", "", m)) for m in _NUMBER.findall(folded)]
    return plain + grouped


def _number_in_quote(value: float, quote: str) -> bool:
    return any(math.isclose(abs(value), n, abs_tol=1e-9) for n in numbers_in(quote))


def _verified(fact: Fact[Any], haystacks: list[str]) -> bool:
    if fact.value is None or not fact.quote:
        return False
    q = normalise(fact.quote)
    if not any(q in h for h in haystacks):
        return False
    if isinstance(fact.value, int | float) and not isinstance(fact.value, bool):
        return _number_in_quote(float(fact.value), fact.quote)
    return True


def check_quotes(facts: BaseModel, item: Item) -> BaseModel:
    """`facts` with every fact whose quote is not in the item cleared.

    Each field of the item is a separate haystack. A number must also be in
    its own quote. An unverified fact becomes unstated; an unverified list
    entry is dropped. Never mutates `facts`.
    """
    haystacks = [normalise(text) for text in item.fields.values()]
    update: dict[str, Any] = {}
    for name in type(facts).model_fields:
        fact = getattr(facts, name)
        if isinstance(fact, list):
            update[name] = [f for f in fact if _verified(f, haystacks)]
        elif not _verified(fact, haystacks):
            update[name] = type(fact)()
    return facts.model_copy(update=update)
