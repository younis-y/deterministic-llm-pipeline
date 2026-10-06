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
    # review round: decimals floor, phrasings without "of experience", a newline
    # is a clause break, "US" is not "us", and a candidate described as "has"
    ("At least 1.5+ years' experience in communications or PR", 1),
    ("Requirements: 3+ years as a Data Engineer or Analytics Engineer", 3),
    ("3+ years working with SQL and Python in production", 3),
    ("Experience of at least 3 years in analytics engineering", 3),
    ("Years of experience: 3-5", 3),
    ("Ideally a CS degree\n3+ years of experience in SQL", 3),
    ("3+ years of experience in the US market", 3),
    ("The ideal candidate has 3+ years of experience in Python", 3),
    # second review: requirement verbs and "you" keep the candidate as subject
    ("Must have 3+ years of experience", 3),
    ("Should have 4+ years of experience in analytics", 4),
    ("Skills\nMust have 3+ years of experience", 3),
    ("Our team is looking for a Data Engineer with 3+ years of experience", 3),
    ("Our engineers need 4 years of experience with Spark", 4),
    ("We want someone who has 5 years of experience", 5),
    ("candidates who have 3+ years of experience", 3),
    ("This role reports to the CFO and requires 5+ years of experience", 5),
    ("You will report to the Head of Data and have 4+ years of experience", 4),
    ("Reporting to the CTO, you have 6 years of experience in data", 6),
    # an "or" that is not the alternative to the years, and a domain qualifier
    ("3+ years of experience with Python or SQL, and a degree in CS", 3),
    ("3+ years of experience in banking or consulting, and a Bachelor's degree", 3),
    ("3+ years of experience, preferably in banking", 3),
    ("3+ years of experience, ideally within fintech", 3),
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
    # review round: an upper limit is not a minimum
    "Fresh graduates or candidates with up to 2 years of experience are encouraged to apply",
    "Candidates must have less than 2 years of professional experience",
    "Maximum 2 years of experience; this is an early-career trainee role",
    "no more than 3 years of work experience",
    "Experience: Up to 2 years of professional experience. Graduated within the last 2 years",
    # graduate-scheme wording: a programme length is not a requirement
    "Our 2 year rotational programme gives you experience across the business",
    "The programme lasts 2 years and gives you hands-on experience",
    "Over 2 years you will gain experience in three teams",
    "graduated within the last 2 years with some experience of Python",
    "a 2 year fixed-term contract offering experience of live systems",
    "Over the next 5 years our experienced team will double",
    # a preference stated after the count, or in a heading
    "3+ years of experience with Spark is preferred",
    "2+ years of experience (preferred)",
    "2-5 years financial planning experience preferred",
    "Preferred Qualifications: 3+ years of experience with dbt",
    "Nice to have: 2+ years of experience with Airflow",
    # someone other than the candidate
    "You will be mentored by a senior engineer with 8+ years of experience",
    "Reporting to the Head of Data, who has 12 years of experience in analytics",
    "Our team members average 7 years of experience",
    "With 12 years of experience in the region, Acme is the market leader",
    "Acme has 10+ years of experience delivering analytics platforms",
    # a decimal below one, and an alternative that waives the years
    "0.5+ years of experience is enough for this role",
    "2+ years of post-Bachelor's machine learning experience, or a Master's degree in a relevant field",
    # second review: a career path is not a requirement
    "You will spend 2 years as an Analyst before being promoted to Associate",
    "After 2 years as an Analyst you will progress to Associate",
    "Join as an Analyst; 3 years as an Associate leads to VP",
    # a preference heading on its own line
    "Preferred Qualifications:\n- 3+ years of experience in SQL",
    "Nice to have\n- 2+ years of experience with Airflow",
    "Bonus points\n5+ years of experience with Kafka",
    "Desired:\n3+ years of experience in dbt",
    # more preference words after the count, and caps after the word
    "3+ years of experience with Spark is beneficial",
    "3+ years of experience in Kafka is helpful but not required",
    "2 years maximum experience; this is a trainee role",
    "Max. 2 years experience",
    # programme wording with other verbs and nouns
    "Join our 2 year graduate programme, gaining experience across three teams",
    "A 2 year programme providing experience across four desks",
    "Our 2 year programme covers experience in sales and trading",
    "Our 3 year apprenticeship provides hands-on experience",
    "We'll give you 2 years of structured experience",
    "Gain 2 years of experience in 12 months",
    "Our scheme offers 2 years of experience",
    # probes: a degree offered instead of the years, "or equivalent", and "upto"
    "Bachelor's degree or 3+ years of experience in lieu of a degree",
    "A degree, or 3+ years' experience in a similar role",
    "3+ years of experience as an analyst, or equivalent",
    "Upto 2 years experience",
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


@pytest.mark.parametrize(
    "text",
    [
        "About us: for over 50 years, we have led the market. Requirements: 3+ years of experience in SQL.",
        "Requirements: 3+ years of experience in SQL. About us: for over 50 years, we have led the market.",
    ],
)
def test_the_requirement_is_read_whichever_side_of_the_boilerplate_it_sits(
    text: str,
) -> None:
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
    job = _job("Needs 1 year of experience. Senior staff have 3+ years of experience.")
    facts = verify_facts(_facts(years=1, quote="1 year of experience"), job)
    assert resolve_years(facts, job).years_required.value == 1


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
