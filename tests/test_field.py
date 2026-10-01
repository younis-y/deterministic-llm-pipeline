"""`resolve_field`: the deterministic field pass applied after `verify_facts`.

The local model left `field` null on 82 of 91 cached postings, so
`profile.rules.allowed_fields` almost never fired - the same failure `level`
had before `resolve_level` read the title. These pin the title table, its
precedence (first matching field wins), and the fallbacks to the model's own
verified quote.
"""

from __future__ import annotations

import pytest

from rolescan.models import Job, JobField
from rolescan.scoring.facts import (
    FieldFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    YearsFact,
    field_from_text,
    resolve_field,
    verify_facts,
)


def _job(title: str, description: str = "We build data pipelines.") -> Job:
    return Job(source="t", company="Acme", title=title, url="https://x/1",
               description=description)


def _facts(field: FieldFact | None = None) -> PostingFacts:
    return PostingFacts(
        level=LevelFact(),
        years_required=YearsFact(),
        student_only=StudentFact(),
        hard_bars=[],
        field=field or FieldFact(),
        fit_score=75,
        reason="Pipelines.",
        keywords_missing=[],
    )


def _resolve(job: Job, field: FieldFact | None = None) -> FieldFact:
    return resolve_field(verify_facts(_facts(field), job), job).field


@pytest.mark.parametrize(
    ("title", "field"),
    [
        ("Data Engineer", JobField.data_engineering),
        ("Analytics Engineer", JobField.data_engineering),
        ("Machine Learning Engineer", JobField.ai_llm),
        ("Senior AI Engineer", JobField.ai_llm),
        ("Data Scientist - Fraud", JobField.data_science),
        ("Graduate Data Analyst", JobField.analytics_bi),
        ("Fullstack developer & DevOps (Internship)", JobField.software),
        # Data words beat software words.
        ("Software Engineer, Data Platform", JobField.data_engineering),
        # Never guess a field (least of all `other`) from a title.
        ("Commercial Strategy & Performance Analyst", None),
        ("Payroll Administrator Intern", None),
    ],
)
def test_title_field_table(title: str, field: JobField | None) -> None:
    assert field_from_text(title) == field


@pytest.mark.parametrize(
    ("title", "field"),
    [
        ("ML Engineer, Data Platform", JobField.ai_llm),
        ("AI/ML Engineer", JobField.ai_llm),
        ("MLOps Engineer", JobField.ai_llm),
        ("LLM Engineer", JobField.ai_llm),
        ("NLP Research Scientist", JobField.ai_llm),
        ("Applied Scientist, Generative AI", JobField.ai_llm),
        ("Computer Vision Engineer", JobField.ai_llm),
        ("ETL Developer", JobField.data_engineering),
        ("Big Data Developer", JobField.data_engineering),
        ("Data Architect", JobField.data_engineering),
        ("Quantitative Researcher", JobField.data_science),
        ("Quant Research Analyst", JobField.data_science),
        ("Statistician", JobField.data_science),
        ("Data Scientist / Data Analyst", JobField.data_science),
        ("Power BI Developer", JobField.analytics_bi),
        ("Business Intelligence Analyst", JobField.analytics_bi),
        ("Insights Analyst", JobField.analytics_bi),
        ("BI Developer / Software Engineer", JobField.analytics_bi),
        ("Back-End Engineer", JobField.software),
        ("Front-end Developer", JobField.software),
        ("Full Stack Engineer", JobField.software),
        ("Site Reliability Engineer", JobField.software),
        ("SRE", JobField.software),
        ("iOS Developer", JobField.software),
        ("DATA ENGINEER", JobField.data_engineering),
    ],
)
def test_title_field_table_precedence_and_variants(
    title: str, field: JobField
) -> None:
    assert field_from_text(title) == field


@pytest.mark.parametrize(
    ("title", "field"),
    [
        # 2.4.4: natural inflections on the ai_llm/data_engineering/
        # data_science phrases. A software word must not win just because
        # the data/AI phrase was spelled as a gerund or a plural.
        ("Software Engineer - Data Engineering", JobField.data_engineering),
        ("Software Engineer Intern (Data Engineering)", JobField.data_engineering),
        ("AI Engineering Intern", JobField.ai_llm),
        ("AI Software Engineer", JobField.ai_llm),
        ("Data Platforms Engineer", JobField.data_engineering),
    ],
)
def test_title_field_inflections(title: str, field: JobField) -> None:
    assert field_from_text(title) == field


@pytest.mark.parametrize(
    "title",
    ["Hallmark Brand Analyst", "Dubai Engineer", "Data Entry Clerk",
     "Data Protection Officer", "AI Sales Executive", "Engineer"],
)
def test_title_field_words_match_whole_phrases_only(title: str) -> None:
    assert field_from_text(title) is None
    assert _resolve(_job(title, "Nothing relevant here.")).value is None


def test_a_title_field_quotes_the_title() -> None:
    got = _resolve(_job("Graduate Data Analyst"))
    assert got.value == JobField.analytics_bi
    assert got.quote == "Graduate Data Analyst"


def test_the_title_overrides_a_conflicting_model_field() -> None:
    job = _job("Data Scientist - Fraud", "We run Python services at scale.")
    got = _resolve(job, FieldFact(value=JobField.software, quote="Python services"))
    assert got.value == JobField.data_science
    assert got.quote == "Data Scientist - Fraud"


def test_a_title_without_field_words_keeps_a_valid_two_word_model_quote() -> None:
    job = _job("Commercial Strategy & Performance Analyst",
               "You will build data pipelines in Python.")
    got = _resolve(job, FieldFact(value=JobField.data_engineering, quote="data pipelines"))
    assert got.value == JobField.data_engineering and got.quote == "data pipelines"


def test_the_models_own_other_survives_with_a_two_word_quote() -> None:
    """`other` is never guessed from a title, but the model may still say it."""
    job = _job("Payroll Administrator Intern", "Run the monthly payroll cycle.")
    got = _resolve(job, FieldFact(value=JobField.other, quote="monthly payroll"))
    assert got.value == JobField.other and got.quote == "monthly payroll"


def test_a_one_word_model_quote_is_dropped() -> None:
    job = _job("Commercial Strategy & Performance Analyst", "Python preferred.")
    got = _resolve(job, FieldFact(value=JobField.software, quote="Python"))
    assert got.value is None and got.quote == ""


def test_a_null_value_with_a_field_quote_is_derived_from_the_quote() -> None:
    job = _job("Commercial Strategy & Performance Analyst",
               "Partner with a data engineer on reporting.")
    got = _resolve(job, FieldFact(value=None, quote="a data engineer"))
    assert got.value == JobField.data_engineering and got.quote == "a data engineer"


def test_an_unverified_model_quote_never_survives() -> None:
    job = _job("Commercial Strategy & Performance Analyst", "Own the forecast.")
    got = _resolve(job, FieldFact(value=JobField.ai_llm, quote="machine learning models"))
    assert got.value is None and got.quote == ""


def test_nothing_anywhere_is_none() -> None:
    got = _resolve(_job("Payroll Administrator Intern", "Run payroll."))
    assert got.value is None and got.quote == ""


def test_resolve_field_is_idempotent_and_never_mutates() -> None:
    job = _job("Commercial Strategy & Performance Analyst",
               "Partner with a data engineer on reporting.")
    facts = verify_facts(_facts(FieldFact(value=None, quote="a data engineer")), job)
    before = facts.model_copy(deep=True)
    once = resolve_field(facts, job)
    assert facts == before
    assert resolve_field(once, job) == once


def test_an_unchanged_field_returns_the_same_object() -> None:
    job = _job("Data Engineer")
    facts = resolve_field(verify_facts(_facts(), job), job)
    assert resolve_field(facts, job) is facts
