"""The facts cache key's hand-bumped version has to move with the extraction.

`FACTS_KEY_VERSION` (see `rolescan.scoring.llm.cache_key`) names what a cached row
holds. The prompt fingerprint moves the key by itself when the prompt or the schema
changes, but the version is the one part a person has to remember, and the
resolver pattern tables decide what a cached fact means once it is read back. So
this pins a digest of the three things that define the extraction: `SYSTEM_FACTS`,
the facts JSON schema and every regex or word table in `rolescan.scoring.facts`. Change
one without changing the version and this fails.

To change one on purpose: bump `FACTS_KEY_VERSION` in `rolescan/scoring/llm.py`
(and the shape pins in `tests/test_facts_mode.py` and `tests/test_facts_cache.py`),
then add the new version and the digest this test prints to `PINNED`.
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import json
import re
from typing import Any

import pytest

from rolescan.scoring import facts, llm

#: version -> digest, for every version whose digest is known.
PINNED: dict[str, str] = {
    "facts-v14": "6fa9547845ac303c",
}


def _canon(value: object) -> Any:
    """`value` as plain JSON data, in a form that does not depend on Python's
    hash order. Raises for a type it does not know, so a new kind of table fails
    here and is handled on purpose."""
    if isinstance(value, re.Pattern):
        return ["re", value.pattern, value.flags]
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return [[_canon(k), _canon(v)] for k, v in sorted(value.items(), key=repr)]
    if isinstance(value, list | tuple):
        return [_canon(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((_canon(v) for v in value), key=lambda c: json.dumps(c))
    msg = f"no canonical form for {type(value).__name__}"
    raise TypeError(msg)


def _tables() -> dict[str, Any]:
    """Every data constant in `rolescan.scoring.facts`: patterns, word tables,
    limits. Code, classes, modules and typing aliases are not data."""
    out: dict[str, Any] = {}
    for name, value in sorted(vars(facts).items()):
        if name.startswith("__"):
            continue
        if (
            inspect.isroutine(value)
            or inspect.isclass(value)
            or inspect.ismodule(value)
        ):
            continue
        if type(value).__module__ in {"typing", "__future__"}:
            continue
        out[name] = _canon(value)
    return out


def extraction_digest() -> str:
    parts = {
        "system_facts": llm.SYSTEM_FACTS,
        "schema": facts.PostingFacts.model_json_schema(),
        "tables": _tables(),
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def test_the_extraction_is_pinned_to_the_facts_key_version() -> None:
    digest = extraction_digest()
    pinned = PINNED.get(llm.FACTS_KEY_VERSION)
    assert pinned is not None, (
        f"{llm.FACTS_KEY_VERSION} has no pin: add `{llm.FACTS_KEY_VERSION!r}: "
        f"{digest!r}` to PINNED"
    )
    assert digest == pinned, (
        "SYSTEM_FACTS, the facts JSON schema or a resolver pattern table changed "
        f"without a facts-vN bump: bump FACTS_KEY_VERSION (now {llm.FACTS_KEY_VERSION}), "
        f"then pin {digest!r} for the new version"
    )


def test_the_digest_sees_the_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    before = extraction_digest()
    monkeypatch.setattr(llm, "SYSTEM_FACTS", llm.SYSTEM_FACTS + " One more rule.")
    assert extraction_digest() != before


def test_the_digest_sees_a_resolver_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    before = extraction_digest()
    monkeypatch.setattr(facts, "_YEARS_MAX", facts._YEARS_MAX + 1)
    assert extraction_digest() != before
    monkeypatch.undo()
    monkeypatch.setattr(facts, "_HR", re.compile(r"\bhr\b"))
    assert extraction_digest() != before


def test_the_digest_sees_a_word_table(monkeypatch: pytest.MonkeyPatch) -> None:
    before = extraction_digest()
    monkeypatch.setattr(facts, "_LEVEL_WORDS", facts._LEVEL_WORDS[:-1])
    assert extraction_digest() != before


def test_the_digest_sees_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    before = extraction_digest()
    original = facts.PostingFacts.model_json_schema

    def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {**original(*args, **kwargs), "x-added": True}

    monkeypatch.setattr(facts.PostingFacts, "model_json_schema", changed)
    assert extraction_digest() != before


def test_the_digest_is_not_a_function_of_the_clock_or_the_environment() -> None:
    assert extraction_digest() == extraction_digest()
