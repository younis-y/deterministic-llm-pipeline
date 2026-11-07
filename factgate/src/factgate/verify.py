"""The quote check: a fact counts only if its quote is really in the item."""

from __future__ import annotations

import datetime as dt
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
#: A number as written, whole: grouped with commas ("12,500.00"), grouped with
#: spaces ("12 500", which may also be two numbers), or plain ("42", "12.5").
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?: \d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
#: A scale written straight after a number: "40k", "£2.5m", "1.2 million".
_SCALE = re.compile(r" ?(k|m|bn|thousand|million|billion)\b", re.IGNORECASE)
_SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
#: A year written in a quote ("2026"), for the date check.
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def normalise(text: str) -> str:
    return " ".join(text.translate(_TYPO_FOLD).split()).casefold()


def _negative(text: str, start: int) -> bool:
    """A minus sign just before the number, ahead of any currency sign, and
    not the dash of a range ("0-2")."""
    before = text[:start].rstrip(" £$€")
    return before.endswith("-") and not before[:-1].rstrip()[-1:].isdigit()


def numbers_in(quote: str) -> list[float]:
    """Every number written in `quote`, whole: separators and currency signs
    ignored, a minus sign kept, and a scale ("40k") applied as well as not."""
    text = quote.translate(_TYPO_FOLD)
    found: list[float] = []
    for m in _NUMBER.finditer(text):
        token = m.group()
        values = [float(token.replace(",", "").replace(" ", ""))]
        if " " in token:
            values += [float(part) for part in token.split(" ")]
        scale = _SCALE.match(text, m.end())
        if scale:
            values.append(values[0] * _SCALES[scale.group(1).lower()])
        sign = -1.0 if _negative(text, m.start()) else 1.0
        found += [sign * v for v in values]
    return found


def _number_in_quote(value: float, quote: str) -> bool:
    return any(math.isclose(value, n, abs_tol=1e-9) for n in numbers_in(quote))


def _verified(fact: Fact[Any], haystacks: list[str]) -> bool:
    q = normalise(fact.quote)
    if fact.value is None or not q or not any(q in h for h in haystacks):
        return False
    if isinstance(fact.value, int | float) and not isinstance(fact.value, bool):
        return _number_in_quote(float(fact.value), fact.quote)
    if isinstance(fact.value, dt.date):
        years = _YEAR.findall(fact.quote)
        return not years or str(fact.value.year) in years
    return True


def check_quotes(facts: BaseModel, item: Item) -> BaseModel:
    """`facts` with every fact whose quote is not in the item cleared.

    Each field of the item is a separate haystack. A number must also be in
    its own quote, and a date's year must be one its quote states, when the
    quote states a year. An unverified fact becomes unstated; an unverified
    list entry is dropped. Never mutates `facts`.
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
