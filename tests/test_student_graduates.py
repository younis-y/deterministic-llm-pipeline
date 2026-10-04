"""2.5.2: `resolve_student`, the recent-graduates guard on `student_only`.

A model reading "internship" and "students" in the same advert sets
`student_only` even when the advert, a line later, also accepts recent
graduates. Three adverts in one evaluation run were skipped this way on a
graduate's profile, each with an explicit graduate clause the model ignored.
After `verify_facts`, code reads the advert for that clause and, when it is
there, overrules the model, quoting the advert's own words. These pin the
clause on the real advert lines, the student-only wording that must NOT
trigger it, and its independence from the graduation-year rule.
"""

from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import Job, Level, Verdict
from rolescan.scoring.facts import (
    GraduationYearFact,
    LevelFact,
    PostingFacts,
    StudentFact,
    graduates_eligible,
    resolve_student,
    verify_facts,
)
from rolescan.scoring.rules import decide

# Real advert lines from the evaluation run (company names left out).
AIRLINE_INTERNSHIP = (
    "Elevate is our highly selective six-month internship programme for "
    "exceptional university students and recent graduates who have already "
    "demonstrated strong academic achievement and the potential to succeed in "
    "a demanding, international environment. To Be Eligible, You Must Be in "
    "the final year of a Bachelor\u2019s or Master\u2019s degree, or have "
    "graduated less than a year ago. Demonstrate a strong academic track record."
)
CONSULTING_GRADUATE = (
    "Model development, validation and audit of financial, risk, marketing and "
    "business-oriented models R&D projects Requirements: Recent graduates or "
    "final year students. Should desirably have knowledge of Artificial "
    "Intelligence modeling techniques."
)
ECONOMICS_INTERN = (
    "Participating in client meetings and conference calls Skills/profile "
    "Sought Currently undertaking or recently completed Bachelor\u2019s, "
    "Master\u2019s, or Ph.D. in economics, ideally with a focus on "
    "microeconomics, industrial organisation and/or econometrics"
)
PRODUCT_INTERN = (
    "What you\u2019ll need Currently studying towards, or recently graduated "
    "with, a degree in business, product, design, engineering, or a related field"
)
RESEARCH_ANALYSTS = (
    "Requirements Graduates / students in their final year of study. If this "
    "is not you, please look at our other opportunities listed on our website."
)


def _job(description: str, title: str = "Summer Internship") -> Job:
    return Job(source="t", company="Acme", title=title, url="https://x/1",
               description=description)


def _facts(student: StudentFact, **kw: object) -> PostingFacts:
    base: dict[str, object] = {
        "level": LevelFact(),
        "student_only": student,
        "fit_score": 70,
        "reason": "Python and SQL match.",
    }
    base.update(kw)
    return PostingFacts(**base)


def _student(job: Job, quote: str, **kw: object) -> StudentFact:
    facts = _facts(StudentFact(value=True, quote=quote), **kw)
    return resolve_student(verify_facts(facts, job), job).student_only


@pytest.mark.parametrize(
    ("description", "model_quote", "says"),
    [
        (AIRLINE_INTERNSHIP, "in the final year of a Bachelor\u2019s or Master\u2019s degree",
         "recent graduates"),
        (CONSULTING_GRADUATE, "final year students", "Recent graduates or final year students."),
        (ECONOMICS_INTERN, "Currently undertaking", "recently completed Bachelor"),
        (PRODUCT_INTERN, "Currently studying towards", "recently graduated with"),
        (RESEARCH_ANALYSTS, "students in their final year of study",
         "Graduates / students in their final year of study."),
    ],
    ids=["airline-internship", "consulting-graduate", "economics-intern", "product-intern", "research-analysts"],
)
def test_a_student_only_advert_that_accepts_graduates_is_not_student_only(
    description: str, model_quote: str, says: str
) -> None:
    got = _student(_job(description), model_quote)
    assert got.value is False
    assert says.translate(str.maketrans("\u2019", "'")) in got.quote.translate(
        str.maketrans("\u2019", "'")
    )


def test_the_override_quote_is_the_adverts_own_words_and_verifies() -> None:
    job = _job(AIRLINE_INTERNSHIP)
    facts = resolve_student(
        verify_facts(_facts(StudentFact(value=True, quote="final year")), job), job
    )
    assert facts.student_only.quote
    assert len(facts.student_only.quote) <= 200
    # The quote survives the quote guard it would meet on a cache replay.
    assert verify_facts(facts, job).student_only == facts.student_only


@pytest.mark.parametrize(
    "line",
    [
        "Open to recent graduates and final year students.",
        "We welcome recently graduated candidates.",
        "Applicants must have graduated within the last 12 months.",
        "You graduated in 2025 or 2026.",
        "or have graduated less than a year ago",
        "Candidates who have recently completed a degree in maths.",
        "You have recently completed your studies.",
        "Open to students or graduates.",
        "Open to students or recent graduates.",
        "Final-year students and recent graduates.",
        "final year students or graduates of any discipline",
        "Graduates are also welcome to apply.",
        "Graduates are eligible.",
        "Graduates or students may apply.",
        "Students or fresh graduates are welcome to apply.",
        "Fresh graduate or graduated within the last 2 years.",
        "Graduates or current students.",
        "Graduates / students in their final year of study.",
        "final-year students & graduates",
    ],
)
def test_graduate_eligibility_wording(line: str) -> None:
    assert graduates_eligible(line) is not None


@pytest.mark.parametrize(
    "line",
    [
        "This internship is open to current students only.",
        "You must be enrolled in a degree and returning to study after the placement.",
        "A summer programme for penultimate-year students.",
        "Recent graduates are not eligible for this programme.",
        "This programme is not open to recent graduates.",
        "Open to undergraduate and graduate students only.",
        "Our graduate scheme opens in spring; this internship is for current students.",
        "A ten-week internship for students graduating in 2028.",
        "Postgraduates studying for a PhD.",
        # Employer boilerplate on placement adverts, not an eligibility line.
        "Each year we recruit hundreds of graduates and students who help us.",
        "Career Area Students and Graduates Job Description",
        "Every year, we welcome over 20,000 students and graduates into our business.",
        # Exclusions whose negation sits too far away, and peers not applicants.
        "If you have already graduated with a bachelor's degree and are currently "
        "studying a postgraduate Masters, you are not eligible for this programme.",
        "Share ideas with other recent graduates across our business units.",
        "Candidates who graduated in 2024 or before will not be considered.",
        "You recently completed a project. Degree in maths preferred.",
    ],
)
def test_student_only_wording_does_not_count_as_graduate_eligibility(line: str) -> None:
    assert graduates_eligible(line) is None


@pytest.mark.parametrize(
    "description",
    [
        "This internship is open to current students only.",
        "You must be enrolled in a degree and returning to study after the placement.",
        "A summer programme for penultimate-year students.",
    ],
)
def test_a_genuinely_student_only_advert_stays_student_only(description: str) -> None:
    quote = description.rstrip(".")
    got = _student(_job(description), quote)
    assert got.value is True
    assert got.quote == quote


def test_a_title_that_admits_graduates_counts() -> None:
    job = _job("Join our team for 12 weeks.", title="Research Analysts - GRADUATES ONLY")
    got = _student(job, "Join our team for 12 weeks")
    assert got.value is False
    assert got.quote == "Research Analysts - GRADUATES ONLY"


@pytest.mark.parametrize("value", [None, False])
def test_the_guard_only_ever_clears_a_true(value: bool | None) -> None:
    job = _job(CONSULTING_GRADUATE)
    quote = "final year students" if value is not None else ""
    facts = verify_facts(_facts(StudentFact(value=value, quote=quote)), job)
    assert resolve_student(facts, job) == facts


def test_resolve_student_is_pure_and_idempotent() -> None:
    job = _job(CONSULTING_GRADUATE)
    facts = verify_facts(_facts(StudentFact(value=True, quote="final year students")), job)
    once = resolve_student(facts, job)
    assert facts.student_only.value is True  # never mutated
    assert resolve_student(once, job) == once


RULES = RulesConfig(
    student_only="skip",
    allowed_levels=[Level.graduate_entry, Level.junior, Level.not_stated],
)


@pytest.mark.parametrize(
    ("description", "model_quote"),
    [
        (AIRLINE_INTERNSHIP, "exceptional university students"),
        (CONSULTING_GRADUATE, "final year students"),
        (ECONOMICS_INTERN, "Currently undertaking"),
    ],
    ids=["airline-internship", "consulting-graduate", "economics-intern"],
)
def test_the_audit_cases_no_longer_skip_as_student_only(
    description: str, model_quote: str
) -> None:
    job = _job(description)
    verified = verify_facts(_facts(StudentFact(value=True, quote=model_quote)), job)
    assert verified.student_only.value is True  # the model's flag verified
    assert decide(verified, RULES, 50).verdict == Verdict.SKIP  # 2.5.1 outcome

    verdict = decide(resolve_student(verified, job), RULES, 50)

    assert verdict.verdict == Verdict.APPLY


def test_the_graduation_year_rule_is_unchanged_by_the_guard() -> None:
    """A "2028 graduates" advert is still caught by max_graduation_year, even
    when the student flag is overruled."""
    job = _job("Open to students or recent graduates graduating in 2028.")
    facts = _facts(
        StudentFact(value=True, quote="Open to students"),
        graduation_year=GraduationYearFact(value=2028, quote="graduating in 2028"),
    )
    resolved = resolve_student(verify_facts(facts, job), job)
    assert resolved.student_only.value is False
    verdict = decide(resolved, RULES.model_copy(update={"max_graduation_year": 2027}), 50)
    assert verdict.verdict == Verdict.SKIP
    assert verdict.reason == 'Skip: advert says "graduating in 2028"'


# 2.5.3: the other direction. The local model left `student_only` unset on
# adverts that restrict eligibility to current students in so many words, so
# a graduate's profile scored them APPLY (Brevan Howard's 2027 Abu Dhabi
# internships, 68). Real advert lines, 2026-10-03/04.
BREVAN_INTERNSHIP = (
    "The goal of our summer internship program is to convert top performing "
    "interns to our 2028 Graduate Program. Qualifications & Requirements A "
    "penultimate year undergraduate/junior or 1st year master's or PhD student "
    "at a recognized University."
)
BLACKROCK_SUMMER = (
    "Who can apply: Candidates should be in their penultimate year of studies "
    "and graduating from an undergraduate or a master\u2019s degree program in 2028."
)
LLM_LAB_INTERN = (
    "Qualifications Currently pursuing a Bachelor\u2019s, Master\u2019s, or PhD "
    "degree in Computer Science, Artificial Intelligence, Machine Learning, Data "
    "Science, or a related technical field."
)
FLEET_INTERN = (
    "We are seeking a motivated student / fresh graduate for a 6 Months "
    "internship. Qualifications Fresh graduate or final year student in Computer "
    "Engineering, Computer Science, Artificial Intelligence."
)
ADVISORY_INTERN = (
    "You will Need To Have The final stages of studies or a recent graduate. "
    "Top academic performance in Economics, Finance, or a related discipline."
)


def _student_from(job: Job, student: StudentFact) -> StudentFact:
    return resolve_student(verify_facts(_facts(student), job), job).student_only


@pytest.mark.parametrize(
    ("description", "says"),
    [
        (BREVAN_INTERNSHIP, "penultimate year"),
        (BLACKROCK_SUMMER, "penultimate year"),
        (LLM_LAB_INTERN, "Currently pursuing"),
    ],
    ids=["brevan-internship", "blackrock-summer", "llm-lab-intern"],
)
@pytest.mark.parametrize("model", [StudentFact(), StudentFact(value=False, quote="")])
def test_a_students_only_advert_the_model_missed_becomes_student_only(
    description: str, says: str, model: StudentFact
) -> None:
    job = _job(description)
    got = _student_from(job, model)
    assert got.value is True
    assert says in got.quote
    # The quote survives the quote guard it would meet on a cache replay.
    assert verify_facts(_facts(got), job).student_only == got


@pytest.mark.parametrize(
    "description",
    [FLEET_INTERN, ADVISORY_INTERN, ECONOMICS_INTERN,
     "You do not need to be currently enrolled at a university.",
     "We build data pipelines for the trading desk."],
    ids=["fleet-intern", "advisory-intern", "economics-intern", "negated", "no-wording"],
)
def test_graduate_friendly_or_silent_adverts_are_not_made_student_only(description: str) -> None:
    assert _student_from(_job(description), StudentFact()).value is not True


def test_a_brevan_style_advert_is_skipped_under_student_only_skip() -> None:
    job = _job(BREVAN_INTERNSHIP, title="2027 Summer Internship Program - AI & Quantitative Analyst")
    facts = resolve_student(verify_facts(_facts(StudentFact()), job), job)
    assert decide(facts, RULES, 50).verdict == Verdict.SKIP
