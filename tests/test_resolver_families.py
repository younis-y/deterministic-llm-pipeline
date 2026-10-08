"""Advert shapes that must not set a fact, from the 2026-10-07 fuzz (2.5.8).

A property-based fuzz run generated advert-like sentences, family by family,
that must NOT produce a years requirement, a students-only flag, a
nationality or clearance bar, a senior level or a software field, and found
every family failing. These are its shrunk examples and their close
relatives as fixed rows, run through `Job` and the resolver chain exactly
as `FitScorer` runs it.

The `FIXED` families were closed in 2.5.8 by fixed phrases added to an
existing guard, checked against 3,695 cached adverts (two outputs changed,
both soft requirements correctly cleared). The `OPEN` families need a guard
that reads more than a fixed phrase (a benefit, a company's own programme, a
customer), so they are strict xfails: each row is a known false positive,
and the fix that clears one must also remove its mark.
"""

from __future__ import annotations

import pytest

from rolescan.models import Job, JobField, Level
from rolescan.scoring.facts import (
    PostingFacts,
    resolve_field,
    resolve_hard_bars,
    resolve_level,
    resolve_student,
    resolve_years,
)

OPEN = pytest.mark.xfail(strict=True, reason="open fuzz family: known false positive")


def _resolved(description: str, title: str = "Graduate Data Analyst") -> PostingFacts:
    job = Job(
        source="t",
        company="Acme",
        title=title,
        location="Dubai",
        url="https://acme.example/1",
        description=description,
    )
    facts = PostingFacts.model_validate(
        {"fit_score": 85, "reason": "Strong SQL overlap."}
    )
    facts = resolve_hard_bars(resolve_years(facts, job), job)
    return resolve_field(resolve_level(resolve_student(facts, job), job), job)


def _rows(families: dict[str, list[str]], *marks: pytest.MarkDecorator) -> list[object]:
    return [
        pytest.param(text, id=f"{family}-{i}", marks=marks)
        for family, texts in families.items()
        for i, text in enumerate(texts, 1)
    ]


YEARS_FIXED = {
    "soft_after": [
        "2 years of experience in SQL would be an asset.",
        "2 years of experience in SQL is an asset.",
        "2 years of experience in SQL would be great.",
        "2 years of experience in SQL is ideal.",
        "2 years of experience in SQL would set you apart.",
        "2 years of experience in SQL is not mandatory.",
        "2 years of experience in SQL is not necessary.",
        "2 years of experience in SQL would be useful.",
        "2 years of experience in SQL would help.",
    ],
    "soft_before": [
        "Even better if you have 2 years of experience with Spark.",
        "We'd love you to have 2 years of experience with Spark.",
        "Extra points for 2 years of experience with Spark.",
        "What would make you stand out: 2 years of experience with Spark.",
        "It would be great if you had 2 years of experience with Spark.",
        "Pluses: 2 years of experience with Spark.",
    ],
    "cap_wording": [
        "Candidates should have at most 2 years of experience.",
        "Graduates with 2 years of experience or less are encouraged to apply.",
        "You will have 2 years of experience or fewer.",
    ],
    "cap_exclusion": [
        "Applicants with more than 2 years of experience are not eligible for this programme.",
        "Candidates with over 2 years of experience will not be considered.",
        "Graduates with 2 years of experience are ineligible.",
        "Those with more than 2 years of experience need not apply.",
        "Applicants with 2 years of experience should apply to experienced roles.",
    ],
    "programme_length": [
        "Across 2 years as a Graduate Analyst, you will rotate across three teams.",
        "During 2 years as an analyst on the desk, with three rotations.",
        "Throughout 2 years in the role, you will rotate across desks.",
        "Over the course of 2 years as a Graduate Analyst, you will rotate across three teams.",
        "The programme runs for 2 years as an analyst on the desk, with three rotations.",
        "It lasts 2 years in the role, you will rotate across desks.",
    ],
}

YEARS_OPEN = {
    "third_party": [
        "The team has 2 years of experience in power markets.",
        "Our data team have 2 years of experience in power markets.",
        "Your manager has 2 years of experience in power markets.",
    ],
    "peers": [
        "Work alongside engineers with 2 years of experience.",
        "Learn from traders with 2 years of experience.",
    ],
    "degree_alternative": [
        "A PhD in Computer Science or 2 years of experience.",
        "A Master's degree in a relevant field or 2 years of experience.",
    ],
    "soft_heading_list": [
        "Preferred qualifications: Experience with Spark. 2 years of experience with Airflow.",
    ],
    "company": [
        "With 2 years of experience in the region, zenko is the market leader.",
    ],
    "hr_duty": ["Screen candidates with 2 years of experience."],
    "benefits": [
        "Employees with 2+ years of experience receive an extra day of leave."
    ],
    "range_wording": [
        "You have between 1 and 2 years of experience.",
        "1 or 2 years of experience in analytics.",
    ],
    "over_as_length": [
        "Over 2 years as a Graduate Analyst, you will rotate across three teams.",
    ],
}


@pytest.mark.parametrize("text", _rows(YEARS_FIXED) + _rows(YEARS_OPEN, OPEN))
def test_no_years_requirement_is_read(text: str) -> None:
    years = _resolved(text).years_required
    assert years.value is None or years.value <= 1, years


STUDENT_OPEN = {
    "boilerplate": [
        "We are currently pursuing growth across the GCC.",
        "Our analysts are currently studying for the CFA.",
    ],
    "duty": ["Build dashboards tracking currently enrolled students."],
    "cross_reference": [
        "This graduate scheme is for graduates. If you are a penultimate year "
        "student, please apply to our Summer Internship instead.",
        "Penultimate-year students should see our Spring Programme.",
    ],
    "benefit": [
        "We support colleagues returning to education through our sponsorship scheme."
    ],
    "professional_qual": ["Qualified or currently studying towards ACCA."],
    "mentor": ["You will mentor first-year MBA students."],
}
STUDENT_PASSING = {
    "cross_reference": [
        "Open to penultimate year students, final year students and graduates.",
    ],
}


@pytest.mark.parametrize("text", _rows(STUDENT_PASSING) + _rows(STUDENT_OPEN, OPEN))
def test_not_students_only(text: str) -> None:
    facts = _resolved(text, title="Graduate Analyst Programme 2027")
    assert facts.student_only.value is not True, facts.student_only


BARS_FIXED = {
    "clearance_duty": [
        "Ensure that contractors hold security clearance.",
        "Verify that employees hold valid security clearance.",
        "Check that suppliers hold SC clearance.",
        "Confirm that new joiners hold security clearance.",
    ],
}
BARS_PASSING = {
    "holiday_or_company": [
        "Celebrate UAE National Day with the team.",
        "Saudi National Day is a paid holiday.",
        "Clients include Saudi National Bank and Qatar National Bank.",
        "We partner with the UK National Grid.",
    ],
}
BARS_OPEN = {
    "nationals_only_benefit": [
        "Housing allowance (UAE nationals only).",
        "Pension (GPSSA) contributions applies only to UAE nationals.",
    ],
    "company_programme": [
        "Through our UAE National Development Programme we have trained 500 people since 2015.",
        "Our academy, aimed at developing UAE nationals, has won awards.",
    ],
    "hr_duty": ["You will support HR in recruiting UAE nationals."],
    "documents": [
        "Required documents: CV - Family book (for UAE Nationals) - Passport copy"
    ],
    "customers": [
        "Our product helps UAE passport holders renew their documents.",
        "Only US citizens can open an account with our app.",
    ],
    "waived_next_clause": [
        "Hiring UAE nationals is part of our commitment; this role is open to everyone.",
    ],
    "clearance_as_data": [
        "Build dashboards on security clearance data for the MoD.",
        "We build software that automates security vetting for government agencies.",
    ],
    "cleared_staff": [
        "Our security-cleared consultants support government clients.",
        "We are a List X company with SC cleared facilities.",
    ],
}


@pytest.mark.parametrize(
    "text", _rows(BARS_FIXED) + _rows(BARS_PASSING) + _rows(BARS_OPEN, OPEN)
)
def test_no_hard_bar_is_read(text: str) -> None:
    assert _resolved(text, title="Data Analyst").hard_bars == []


@pytest.mark.parametrize(
    "title",
    [
        pytest.param(t, id=f"business_line-{i}", marks=OPEN)
        for i, t in enumerate(
            [
                "Analyst, Principal Investments",
                "Associate - Director's Office",
                "Data Analyst - Lead-to-Cash",
            ],
            1,
        )
    ],
)
def test_a_business_line_in_the_title_is_not_a_level(title: str) -> None:
    level = _resolved("Requirements: SQL.", title=title).level.value
    assert level in (Level.not_stated, Level.graduate_entry, Level.junior)


@pytest.mark.parametrize(
    "title",
    [
        pytest.param(t, id=f"business_context-{i}", marks=OPEN)
        for i, t in enumerate(
            [
                "Operations Analyst - Back-End Operations",
                "Business Analyst - DevOps Transformation",
            ],
            1,
        )
    ],
)
def test_a_business_context_in_the_title_is_not_software(title: str) -> None:
    assert _resolved("Requirements: SQL.", title=title).field.value != JobField.software
