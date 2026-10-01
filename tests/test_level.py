"""`resolve_level`: the deterministic level pass applied after `verify_facts`.

Cases 1-3 are live postings the local model got wrong on the first facts-mode
night: it quoted a title that starts with a level word and still answered
`not_stated`, and it skipped a two-years role on a one-word "Senior" quote.
"""

from __future__ import annotations

import pytest

from rolescan.models import Job, JobField, Level
from rolescan.scoring.facts import (
    FieldFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
    resolve_level,
    verify_facts,
)


def _job(title: str, description: str = "We build data pipelines.") -> Job:
    return Job(source="t", company="Acme", title=title, url="https://x/1",
               description=description)


def _facts(level: LevelFact | None = None, **kw: object) -> PostingFacts:
    base: dict[str, object] = {
        "level": level or LevelFact(),
        "years_required": YearsFact(),
        "student_only": StudentFact(),
        "hard_bars": [],
        "field": FieldFact(value=JobField.data_engineering, quote="data pipelines"),
        "fit_score": 75,
        "reason": "Pipelines.",
        "keywords_missing": [],
    }
    base.update(kw)
    return PostingFacts(**base)


def _resolve(job: Job, level: LevelFact) -> LevelFact:
    return resolve_level(verify_facts(_facts(level), job), job).level


def test_live_senior_title_quoted_but_left_not_stated_becomes_senior() -> None:
    job = _job("Senior Data & BI Engineer")
    got = _resolve(job, LevelFact(value=Level.not_stated, quote="Senior Data & BI Engineer"))
    assert got.value == Level.senior
    assert got.quote == "Senior Data & BI Engineer"


def test_live_lead_title_quoted_but_left_not_stated_becomes_lead() -> None:
    job = _job("Lead Data Engineer")
    got = _resolve(job, LevelFact(value=Level.not_stated, quote="Lead Data Engineer"))
    assert got.value == Level.lead_principal
    assert got.quote == "Lead Data Engineer"


def test_live_one_word_senior_in_the_description_does_not_skip_a_two_years_role() -> None:
    job = _job(
        "Data Engineer",
        "A minimum of 2+ years' experience. You will work with Senior stakeholders.",
    )
    got = _resolve(job, LevelFact(value=Level.senior, quote="Senior"))
    assert got.value == Level.not_stated
    assert got.quote == ""


def test_an_intern_posting_is_an_intern_posting_whatever_else_the_title_says() -> None:
    job = _job("Senior Analyst Internship")
    got = _resolve(job, LevelFact())
    assert got.value == Level.graduate_entry
    assert got.quote == "Senior Analyst Internship"


def test_the_title_overrides_a_conflicting_model_level() -> None:
    job = _job("Senior AI Engineer", "Junior-friendly team of five people.")
    got = _resolve(job, LevelFact(value=Level.junior, quote="Junior-friendly team"))
    assert got.value == Level.senior and got.quote == "Senior AI Engineer"


def test_a_title_without_keywords_keeps_a_valid_two_word_model_quote() -> None:
    job = _job("Data Engineer", "This is a mid-level role in a small team.")
    got = _resolve(job, LevelFact(value=Level.mid, quote="mid-level role"))
    assert got.value == Level.mid and got.quote == "mid-level role"


def test_a_not_stated_value_with_a_keyword_quote_is_derived_from_the_quote() -> None:
    job = _job("Data Engineer", "Hiring a senior engineer to own the platform.")
    got = _resolve(job, LevelFact(value=Level.not_stated, quote="a senior engineer"))
    assert got.value == Level.senior and got.quote == "a senior engineer"


def test_a_not_stated_value_with_a_one_word_keyword_quote_stays_not_stated() -> None:
    job = _job("Data Engineer", "Reporting to the Senior team.")
    got = _resolve(job, LevelFact(value=Level.not_stated, quote="Senior"))
    assert got.value == Level.not_stated


def test_nothing_anywhere_is_not_stated() -> None:
    got = _resolve(_job("Data Engineer"), LevelFact())
    assert got.value == Level.not_stated and got.quote == ""


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Sr. Data Engineer", Level.senior),
        ("Principal ML Engineer", Level.lead_principal),
        ("Staff Engineer", Level.lead_principal),
        ("Head of Data", Level.lead_principal),
        ("Engineering Manager, Data", Level.lead_principal),
        ("Senior Engineering Manager", Level.lead_principal),
        ("Jr Data Analyst", Level.junior),
        ("Graduate AI Engineer", Level.graduate_entry),
        ("Junior Graduate Data Engineer", Level.graduate_entry),
        ("Entry-Level Data Analyst", Level.graduate_entry),
        ("Entry Level Data Analyst", Level.graduate_entry),
        ("Data Science Placement", Level.graduate_entry),
        ("Fullstack developer & DevOps (Internship)", Level.graduate_entry),
        ("Trainee Data Engineer", Level.graduate_entry),
        ("Data Apprentice", Level.graduate_entry),
        ("Grad Data Engineer", Level.graduate_entry),
    ],
)
def test_title_keyword_table(title: str, level: Level) -> None:
    assert _resolve(_job(title), LevelFact()).value == level


@pytest.mark.parametrize(
    "title",
    ["Internal Tools Engineer", "Headcount Planning Analyst", "Staffing Data Analyst",
     "Leadership Data Analyst", "Analytics Engineer"],
)
def test_title_keywords_match_whole_words_only(title: str) -> None:
    assert _resolve(_job(title), LevelFact()).value == Level.not_stated


def test_resolve_level_is_idempotent_and_never_mutates() -> None:
    job = _job("Data Engineer", "Hiring a senior engineer to own the platform.")
    once = resolve_level(
        verify_facts(_facts(LevelFact(value=Level.not_stated, quote="a senior engineer")), job),
        job,
    )
    twice = resolve_level(once, job)
    assert once == twice
