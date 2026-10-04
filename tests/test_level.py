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
    level_from_text,
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
        # 2.4.2: "manager" is no longer a level word, so these two changed
        # (were lead_principal in 2.4.1).
        ("Engineering Manager, Data", Level.not_stated),
        ("Senior Engineering Manager", Level.senior),
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


# --- 2.4.2 title level table ---------------------------------------------
# "manager" is gone; "staff" and "lead" count only before a role word; any
# graduate/junior word beats any senior/lead word in the same title.


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Junior Product Manager", Level.junior),
        ("Jr Staff Engineer", Level.junior),
        ("Assistant Manager", Level.junior),
        ("Senior Analyst Internship", Level.graduate_entry),
        ("Head of Data", Level.lead_principal),
        ("Principal Data Scientist", Level.lead_principal),
        ("Senior Data Engineer", Level.senior),
        ("Lead Data Engineer", Level.lead_principal),
        ("Lead Data & AI Engineer", Level.lead_principal),
        ("Senior Lead Engineer", Level.lead_principal),
        ("Staff Scientist, Data", Level.lead_principal),
        ("Senior Director of Data", Level.lead_principal),
        ("Product Manager", None),
        ("Staff Accountant", None),
        ("Lead Generation Executive", None),
        ("Headquarters Analyst", None),
    ],
)
def test_level_words_2_4_2(title: str, level: Level | None) -> None:
    assert level_from_text(title) == level
    expected = Level.not_stated if level is None else level
    assert _resolve(_job(title), LevelFact()).value == expected


# --- 2.5.1 mid row ------------------------------------------------------------
# "mid-level" and "intermediate" are mid; "mid-senior" is senior. Precedence is
# unchanged: graduate/junior words beat everything, lead beats senior, and
# senior beats mid.


@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Mid-Level Data Engineer", Level.mid),
        ("Mid Level Data Engineer", Level.mid),
        ("Mid\u2013Level Data Engineer", Level.mid),
        ("Data Engineer (Mid-Level)", Level.mid),
        ("Intermediate Data Analyst", Level.mid),
        ("Mid-Senior Data Scientist", Level.senior),
        ("Mid Senior Data Scientist", Level.senior),
        ("Senior / Mid-Level Data Engineer", Level.senior),
        ("Lead Data Engineer (Mid-Level)", Level.lead_principal),
        ("Mid-Level Graduate Engineer", Level.graduate_entry),
        ("Junior to Mid-Level Analyst", Level.junior),
        ("Midlands Data Engineer", None),
        ("Mid Market Account Executive", None),
        ("Mid-Office Analyst", None),
    ],
)
def test_level_words_2_5_1_mid_row(title: str, level: Level | None) -> None:
    assert level_from_text(title) == level
    expected = Level.not_stated if level is None else level
    assert _resolve(_job(title), LevelFact()).value == expected


# 2.5.3: titles from Tem's live board (2026-10-03) that named a lead or staff
# level the table did not know, so `level_from_title_only` let them through -
# "Staff Data Analyst" came out APPLY 75 for a graduate. The negatives are the
# reasons the staff and lead rows were narrow in the first place.
@pytest.mark.parametrize(
    ("title", "level"),
    [
        ("Staff Data Analyst", Level.lead_principal),
        ("Senior Staff Machine Learning Engineer - Pricing", Level.lead_principal),
        ("Staff Full Stack Engineer - Sell Side Engine", Level.lead_principal),
        ("Staff QA Automation Engineer", Level.lead_principal),
        ("Tech Lead - Payments and Billing", Level.lead_principal),
        ("Team Lead, Data Platform", Level.lead_principal),
        ("Brand & Creative Design Lead", Level.lead_principal),
        ("Video Lead", Level.lead_principal),
        ("Data Lead, EMEA", Level.lead_principal),
        ("Junior Staff Engineer", Level.junior),
        ("Staff Accountant", None),
        ("Staffing Coordinator", None),
        ("Staff Nurse", None),
        ("Lead Generation Executive", None),
        ("Leading Edge Data Engineer", None),
        ("Market Analyst", None),
    ],
)
def test_level_words_2_5_3_staff_and_lead_titles(title: str, level: Level | None) -> None:
    assert level_from_text(title) == level
