"""2.5.5: `resolve_years`, the deterministic years-of-experience pass.

The local model was observed leaving `years_required` null on adverts that
state it in so many words. In the 2026-10-05/06 digests, 12 of 13 adverts
with a plain "N+ years of experience" line came back null, while the model's
own reason text said "requires 3+ years of experience"; the owner's
`max_years_required: 1` rule therefore never fired and senior roles were
rated APPLY. After `verify_facts`, code reads the advert for an unambiguous
requirement and fills the fact only when the model left it empty, quoting
the advert's own words. These pin the phrasings seen in real adverts, the
company boilerplate and soft preferences that must NOT fire, and the rule.
"""

from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import Job, Verdict
from rolescan.scoring.facts import (
    PostingFacts,
    YearsFact,
    resolve_years,
    verify_facts,
    years_required_stated,
)
from rolescan.scoring.rules import decide

# Real advert lines from the evaluation set and two digests (companies left out).
STATED = [
    (
        "Required Qualifications: 5+ years of hands-on data engineering experience building pipelines",
        5,
    ),
    ("you will bring a minimum of 10 years of overall IT experience, backed by", 10),
    (
        "Qualifications 2\u20134 years experience in data engineering or a related field",
        2,
    ),
    (
        "Essential Skills: Minimum 2+ years\u2019 hands-on experience designing and developing",
        2,
    ),
    ("At least 3 years of experience in a similar data analytics engineering role", 3),
    (
        "REQUIREMENTS Bachelor\u2019s degree with 1\u20133 years of relevant experience",
        1,
    ),
    ("Data Analyst with 3\u20137 years of experience in construction or EPC", 3),
    ("5 Years + experience in integration work", 5),
    ("Proven experience (3+ years) in designing and implementing AI systems", 3),
    (
        "You have 4+ years in a data role (Analytics Engineering and/or Data Engineering)",
        4,
    ),
    ("Seniority: 5+ years, hands-on across data, controls and Power BI", 5),
    ("2 years working in a similar role.", 2),
    # a soft word in the PREVIOUS clause does not make this one a preference
    (
        "Computer Science or equivalent qualification (preferred) 2+ years of solid experience in software development",
        2,
    ),
    (
        "or a related field preferred. Minimum of 4 years of experience in GIS, data visualization",
        4,
    ),
    (
        "We\u2019re looking for someone with over 1 year of commercial experience who has",
        1,
    ),
]

NOT_A_REQUIREMENT = [
    # preferences
    "Strong Python development experience, ideally with 6+ years of professional software engineering experience.",
    "Preferably 3+ years of experience with dbt; a master's degree is a plus",
    # the company, or a person, describing itself
    "Halian Group: With over 28 years of experience, we have come to understand that innovation is the only way",
    "an award-winning trading provider, possessing more than 25 years of experience with multiple offices around the world",
    "With more than 15 years of global experience, we have supported programs across 150+ countries",
    "The pod is led by a Senior Portfolio Manager with 15+ years of experience and a proven track record",
    "For over 50 years, we have been at the forefront of innovation and sustainability",
    "For more than 200 years, we've transformed knowledge into discoveries",
    # a number of years that is not experience at all
    "Over the next 5 years we will double the size of the platform team",
    "a 2 year fixed-term contract with a view to extension",
    "0 years of experience required - we train you",
    "",
]


@pytest.mark.parametrize(("text", "years"), STATED)
def test_a_stated_requirement_is_read_with_its_own_words(text: str, years: int) -> None:
    found = years_required_stated(text)
    assert found is not None
    value, quote = found
    assert value == years
    assert quote and quote in text  # verbatim, so the quote guard would pass it


@pytest.mark.parametrize("text", NOT_A_REQUIREMENT)
def test_preferences_and_company_boilerplate_do_not_count(text: str) -> None:
    assert years_required_stated(text) is None


def test_the_first_requirement_wins_over_later_boilerplate_and_vice_versa() -> None:
    text = "About us: for over 50 years, we have led the market. Requirements: 3+ years of experience in SQL."
    assert years_required_stated(text) == (3, "3+ years of experience")


def _job(description: str, title: str = "Data Engineer") -> Job:
    return Job(
        source="t",
        company="Acme",
        title=title,
        location="London",
        url="https://acme.test/1",
        description=description,
    )


def _facts(
    years: int | None = None, quote: str = "", fit_score: int = 80
) -> PostingFacts:
    return PostingFacts.model_validate(
        {
            "level": {"value": "not_stated", "quote": ""},
            "years_required": {"value": years, "quote": quote},
            "student_only": {"value": None, "quote": ""},
            "graduation_year": {"value": None, "quote": ""},
            "hard_bars": [],
            "field": {"value": None, "quote": ""},
            "fit_score": fit_score,
            "reason": "Strong SQL overlap.",
            "keywords_missing": [],
        }
    )


ADVERT = "What we need: 3+ years of experience in data engineering. Strong SQL."


def test_a_null_fact_is_filled_from_the_advert() -> None:
    facts = _facts()
    out = resolve_years(facts, _job(ADVERT))
    assert out.years_required == YearsFact(value=3, quote="3+ years of experience")
    assert facts.years_required.value is None  # never mutates its input
    assert resolve_years(out, _job(ADVERT)) == out  # idempotent


def test_the_models_own_verified_value_is_kept() -> None:
    facts = verify_facts(
        _facts(years=1, quote="1 year of experience"),
        _job("Needs 1 year of experience. Senior staff have 3+ years of experience."),
    )
    assert resolve_years(facts, _job(ADVERT)).years_required.value == 1


def test_the_title_is_read_before_the_description() -> None:
    job = _job("Nothing stated here.", title="Data Engineer (5+ years experience)")
    assert resolve_years(_facts(), job).years_required == YearsFact(
        value=5, quote="5+ years experience"
    )


def test_an_advert_without_a_requirement_is_unchanged() -> None:
    facts = _facts()
    assert (
        resolve_years(
            facts,
            _job("Fresh graduates welcome. Ideally with 2+ years of SQL experience."),
        )
        == facts
    )


def test_the_filled_fact_fires_the_owners_years_rule() -> None:
    rules = RulesConfig(max_years_required=1)
    facts = resolve_years(_facts(), _job(ADVERT))
    verdict = decide(facts, rules, 40)
    assert verdict.verdict == Verdict.SKIP
    assert "3+ years of experience" in verdict.reason


def test_one_year_stays_within_the_owners_cap() -> None:
    rules = RulesConfig(max_years_required=1)
    facts = resolve_years(
        _facts(), _job("Requirements: 1 year of experience with Python. Great team.")
    )
    assert facts.years_required.value == 1
    assert decide(facts, rules, 40).verdict == Verdict.APPLY
