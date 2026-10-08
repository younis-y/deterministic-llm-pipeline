"""`_term_hit` checks for the term as a substring before it runs the
whole-word pattern (2.5.8). The substring test is a necessary condition for
the pattern, so it may change speed and nothing else; these rows pin that the
answers did not move."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rolescan.config import Config
from rolescan.scoring import keyword
from rolescan.scoring.keyword import _term_hit

ROOT = Path(__file__).resolve().parents[1]


def _reference(term: str, blob: str) -> bool:
    """2.5.7's `_term_hit`, verbatim."""
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", blob) is not None


@pytest.mark.parametrize(
    ("term", "blob", "hit"),
    [
        ("crypto", "cryptographic hashing", False),
        ("crypto", "a crypto trading desk", True),
        ("head of", "based at head office", False),
        ("head of", "reports to the head of data", True),
        ("director", "the board of directors", False),
        ("director", "a director of analytics", True),
        ("c++", "python and c++ required", True),
        ("c++", "c++17 preferred", False),
        ("uae national", "uae nationals only", False),
        ("10+ years", "10+ years of experience", True),
        ("military", "", False),
        ("sql", "SQL and Python", False),
        ("café", "CAFÉ latte", False),
    ],
)
def test_the_precheck_changes_no_answer(term: str, blob: str, hit: bool) -> None:
    assert _term_hit(term, blob) is hit
    assert _reference(term, blob) is hit


def _shipped_terms() -> list[str]:
    """Every blocker term in the shipped configs, weighted and hard."""
    terms: set[str] = set()
    for path in [ROOT / "config.example.yaml", *(ROOT / "examples").glob("*.yaml")]:
        profile = Config.load(path).profile
        terms.update(profile.blockers)
        terms.update(profile.hard_blockers)
        terms.update(profile.title_only_blockers)
    return sorted(terms)


def test_the_shipped_blocker_terms_are_a_real_corpus() -> None:
    assert len(_shipped_terms()) >= 5


@pytest.mark.parametrize("term", _shipped_terms())
def test_every_shipped_term_answers_as_it_did(term: str) -> None:
    """Each shipped blocker, against the shapes a posting gives it: standing
    alone, inside a longer word, beside punctuation, in another case, and
    absent."""
    blobs = [
        "",
        "graduate data analyst, london",
        term,
        f"the {term}.",
        f"({term})",
        f"{term}s and more",
        f"pre{term}",
        f"{term}ed",
        f"x{term}y",
        f"a {term} role, then another {term}",
        term.upper(),
        term.title(),
    ]
    for blob in blobs:
        assert _term_hit(term, blob) is _reference(term, blob), (term, blob)


def test_an_absent_term_never_reaches_the_pattern() -> None:
    keyword._term_pattern.cache_clear()
    assert not _term_hit("blockchain", "graduate data analyst, london")
    assert keyword._term_pattern.cache_info().misses == 0
    assert _term_hit("london", "graduate data analyst, london")
    assert keyword._term_pattern.cache_info().misses == 1


def test_an_absent_term_compiles_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same property without the cache's own counters: `re.compile` is not
    called for a term the substring test rules out, and is called for one it
    does not."""
    keyword._term_pattern.cache_clear()
    real_compile = re.compile
    compiled: list[str] = []

    def spy(pattern: str, flags: int = 0) -> re.Pattern[str]:
        compiled.append(pattern)
        return real_compile(pattern, flags)

    monkeypatch.setattr(re, "compile", spy)
    assert not _term_hit("blockchain", "graduate data analyst, london")
    assert compiled == []
    assert _term_hit("london", "graduate data analyst, london")
    assert len(compiled) == 1
    assert "london" in compiled[0]


def test_a_pattern_is_compiled_once_per_term() -> None:
    keyword._term_pattern.cache_clear()
    for _ in range(3):
        assert _term_hit("london", "graduate data analyst, london")
    info = keyword._term_pattern.cache_info()
    assert (info.misses, info.hits) == (1, 2)
