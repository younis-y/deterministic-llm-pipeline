"""2.5.6: `resolve_hard_bars`, the deterministic nationality and clearance pass.

On the owner's 200-posting evaluation the local model named a bar on only 7 of
the 26 adverts that state one. The keyword blockers catch most of the rest,
but adverts phrased like "Data Analyst - UAE National, ICQA/LND Analytics
Team", "aimed at preparing UAE Nationals for successful careers", "Officer
Regulatory Reporting - Emirati Talent" and "we require all candidates to hold
an active eDV clearance" reached the digest as APPLY. After `verify_facts`,
code reads the advert for an eligibility-shaped bar and adds one only for a
kind the model left out, quoting the advert's own words. A block hides the
role for good, so precision matters more than recall: these pin the real
lines that are bars and the real lines that only look like one (a
preference, a residency alternative, customers, a duty, a hedge).
"""

from __future__ import annotations

import pytest

from rolescan.config import RulesConfig
from rolescan.models import BarKind, Job, Verdict
from rolescan.scoring.facts import (
    HardBar,
    PostingFacts,
    _bar_supported,
    hard_bars_stated,
    resolve_hard_bars,
    verify_facts,
)
from rolescan.scoring.rules import decide

# Real titles and advert lines from the owner's cached adverts (companies left
# out). Each is (title, description).
NATIONALITY = [
    ("Data Analyst - UAE National, ICQA/LND Analytics Team", ""),
    ("Officer Regulatory Reporting - Emirati Talent", ""),
    ("Intern - Business Operations Analyst (UAE Nationals Only)", ""),
    ("Graduate Trainee - Technology (UAE Nationals)", ""),
    ("Graduate Stakeholder Engineer_(Emirati National)", ""),
    ("Opportunities for UAE Nationals - RU'YA 2026", ""),
    ("Associate Sales Performance - Emirati", ""),
    ("Business Insights Analyst - UAE National Graduate Programme", ""),
    ("Buying Trainee (Emiratisation)", ""),
    ("Sales Analytics Graduate Emiratization", ""),
    ("Specialist, Group Strategy (Emiratized)", ""),
    (
        "Analyst",
        "The Program is a full-time development initiative aimed at preparing "
        "UAE Nationals for successful careers in the banking sector.",
    ),
    (
        "Analyst",
        "Motivated, dedicated and hardworking. This role is open to UAE Nationals only",
    ),
    ("Analyst", "Only Saudi nationals will be considered for this vacancy."),
    (
        "Analyst",
        "Programme Duration: 24 months Eligibility: UAE Nationals only About us",
    ),
    ("Analyst", "Ability to work in a fast-paced environment Emirati nationals ONLY."),
    ("Analyst", "Please note: This position is open to Jordanian nationals only."),
    ("Analyst", "This position is open to UAE National fresh graduates."),
    ("Analyst", "This program is reserved for UAE Nationals; Medical fitness"),
    (
        "Analyst",
        "To qualify for this, you must be a British citizen and have lived "
        "permanently in the UK for the last 5 years.",
    ),
    (
        "Analyst",
        "Due to the nature of the work, candidates must be a sole British National.",
    ),
    ("Analyst", "Qualifications UAE national with family book. Degree in Data Science"),
    (
        "Analyst",
        "In line with our commitment to Emiratisation, only UAE Nationals with a "
        "family book will be considered for this role.",
    ),
    (
        "Analyst",
        "Must hold a UAE National Family Book. Must be willing to work on-site.",
    ),
    ("Analyst", "Requirements: UAE National (with Family Book) Currently a student"),
    (
        "Analyst",
        "We are pleased to offer internship opportunities for UAE National students.",
    ),
    (
        "Analyst",
        "Location: Saudi Arabia Nationality: Saudi National Experience: 3-5 Years",
    ),
    ("Analyst", "Distributed engineering team. Saudi passport required."),
    (
        "Analyst",
        "Master\u2019s degree required Saudi National required At least two years",
    ),
    ("Analyst", "This is an Emiratisation role within the finance team."),
    # corpus review: a plural people-word, a recruiting line, a programme name
    (
        "Analyst",
        "Requirements: Out of work for more than 3 months. Nationality: Saudis "
        "only Experience with SQL, other relational databases and other types of "
        "databases like NoSQL Hands on experience in Data Warehousing",
    ),
    (
        "Analyst",
        "Dicetek LLC in Dubai is seeking a UAE National fresh graduate for a Data "
        "Analyst / Business Analyst role.",
    ),
    ("Analyst", "We are looking for UAE Nationals with the following:"),
    (
        "Analyst",
        "The Emirates Group is offering a UAE National Graduate Programme in Dubai.",
    ),
    (
        "Graduate Analyst",
        "JLL UAE invites UAE Nationals to join the UAE National Graduate Analyst "
        "Programme in Dubai.",
    ),
    # A welcome to students and graduates after the bar is not a preference
    # about the bar itself.
    (
        "Analyst",
        "Eligibility Saudi nationals only We welcome both current students and "
        "fresh graduates",
    ),
]

NOT_A_NATIONALITY_BAR = [
    # a preference
    ("Analyst", "Additional Program Information UAE nationals are encouraged to apply"),
    ("Analyst", "UAE nationals preferred, with a focus on developing local talent."),
    ("Sales Analyst Intern - Dubai, UAE (UAE Nationals Preferred)", ""),
    ("Analyst", "Eligibility: UAE Nationals are prioritized for this internship."),
    (
        "Analyst",
        "Preference will be given to UAE National talents. UAE Nationals holding "
        "a Family Book are encouraged to apply",
    ),
    ("Analyst", "We welcome applications from all qualified UAE National candidates."),
    ("Analyst", "Nationality All Nationalities (Priority for UAE National) Salary"),
    # an alternative the owner satisfies
    ("Analyst", "Valid UAE residence visa, or be a UAE or GCC national."),
    (
        "Analyst",
        "Discover is our internship programme for UAE National students and "
        "students who hold a valid UAE residence visa.",
    ),
    ("Analyst", "Be a UAE National or hold a valid UAE residence visa."),
    # a duty, customers, beneficiaries, the workforce
    ("Analyst", "You will support Emiratization objectives across the business."),
    (
        "Analyst",
        "Develop employees with emphasis on UAE Nationals to meet Emiratization targets.",
    ),
    ("Analyst", "We deliver government services for UAE nationals and residents."),
    (
        "Analyst",
        "The programme's beneficiaries are Jordanian refugees and host communities.",
    ),
    ("Analyst", "Today 40% of our staff are UAE nationals."),
    ("Analyst", "Mentoring UAE National colleagues is part of the role."),
    (
        "Analyst",
        "We are seeking a candidate to support UAE nationals in their careers.",
    ),
    ("Analyst", "We are seeking UAE nationals and expatriates for our Dubai office."),
    ("Analyst", "25 days annual leave plus all UAE national holidays."),
    # a country, a place, a programme name, the job board's own label
    (
        "Analyst",
        "Job Location Dubai, UAE Nationality Any Nationality Salary Not Specified",
    ),
    ("Analyst", "Making a real difference to UK national security."),
    ("Analyst", "Agency programs, Summer campaigns and Saudi National Day events."),
    ("Analyst", "Based in Riyadh, Saudi Arabia"),
    ("Analyst", "You must have the right to work in the UK"),
    (
        "Analyst",
        "Experience Requirements: UAE Nationals: Minimum 2 years Other Nationalities: 4 years",
    ),
    ("Analyst", "A minimum of ten years (8 years for UAE nationals) of experience."),
    # a negation
    ("Analyst", "This role is not restricted to UAE nationals."),
    ("Analyst", "You do not need to be a British citizen to apply."),
    # the HR role that runs the programme, not a hire under it
    ("Emiratisation Manager", ""),
    # corpus review: the national programme named in an equal-opportunity line
    (
        "Data Analyst",
        "All applicants will be considered, with the understanding that "
        "preference will be given to the designated groups in accordance with "
        "the United Arab Emirates Emiratization Program.",
    ),
    ("", ""),
]

CLEARANCE = [
    ("DV Cleared Data Engineers", ""),
    (
        "Systems Engineer",
        "Due to the nature of the programme this team is working in, we require "
        "all candidates to hold an active eDV clearance for this role.",
    ),
    ("Engineer", "Collaborate with stakeholders. Active DV clearance is essential."),
    ("Engineer", "Must hold valid SC Clearance"),
    ("Engineer", "Clearance: Active SC + UKSV confirmation required"),
    ("Engineer", "Candidates must be eligible to attain SC clearance."),
    ("Engineer", "Eligible to obtain UK Security Clearance (SC)."),
    (
        "Engineer",
        "Any offer of employment is subject to satisfactory BPSS and SC security clearance.",
    ),
    ("Engineer", "NPPV3 required."),
    ("Engineer", "Developed Vetting required."),
    ("Engineer", "All candidates must be eligible for Security Clearance."),
    (
        "Engineer",
        "Our Data Engineering roles require the eligibility for SC clearance.",
    ),
    ("Data Engineer - SC Cleared / SC Eligible", ""),
    ("Engineer", "Candidates must be eligible for DV."),
    ("Engineer", "Must have active eDV."),
]

NOT_A_CLEARANCE_BAR = [
    # a negation
    ("Engineer", "No security clearance required."),
    ("Engineer", "Security clearance is not required for this role."),
    ("Engineer", "Clearance is not needed."),
    ("Engineer", "You do not need clearance to apply."),
    ("Engineer", "You don't need SC clearance for this team."),
    (
        "Engineer",
        "If you do not hold an active DV clearance, please read the guidance.",
    ),
    # a hedge about other roles
    (
        "Engineer",
        "Please be aware that some of our UK roles require a UK security clearance.",
    ),
    (
        "Engineer",
        "You may also need to gain UK SC-level Security Clearance or Export "
        "Control, depending on the role.",
    ),
    (
        "Engineer",
        "Depending on the project, UK Developed Vetting (DV) eligibility may be required.",
    ),
    (
        "Engineer",
        "Many roles also require higher levels of National Security Vetting "
        "where applicants must typically have 5 to 10 years of continuous "
        "residency in the UK depending on the vetting level required for the "
        "role, to allow for meaningful security vetting checks.",
    ),
    # corpus review: a residency rule given as the reason, in boilerplate
    (
        "Engineer",
        "We therefore ask that you only apply if you meet the residency "
        "requirements (i.e. you are a British citizen or have been resident in "
        "the UK for the past 5 years), as this is the prerequisite for a "
        "security clearance.",
    ),
    # a preference
    ("Engineer", "Current SC clearance is highly desirable."),
    ("Engineer", "Candidates should ideally hold active SC clearance."),
    (
        "Engineer",
        "Candidates who already hold UK security clearance are particularly encouraged to apply.",
    ),
    ("Engineer", "Desired Skills: Must be eligible for Security Clearance"),
    # other clearances, the company, a certificate, a degree
    (
        "Analyst",
        "The life-cycle of a trade from execution to clearance, settlement and cash.",
    ),
    ("Analyst", "Offers are subject to satisfactory references and police clearance."),
    ("Analyst", "Validate documentation for UAE shipment clearance."),
    ("Analyst", "An enhanced Disclosure and Barring Service (DBS) clearance."),
    (
        "Software Engineer Intern - Summer 2027 (DV Commodities)",
        "DV Trading has scaled fast.",
    ),
    ("SC&E - Capital Projects Transformation Analyst", ""),
    ("Analyst", "Microsoft certifications such as SC-200 or AZ-500."),
    ("Analyst", "MSc in Computer Science"),
]


def _bar(title: str, description: str, kind: BarKind) -> HardBar | None:
    return next(
        (b for b in hard_bars_stated(title, description) if b.kind == kind), None
    )


def _assert_verbatim_and_supported(bar: HardBar, title: str, description: str) -> None:
    assert bar.quote
    assert bar.quote in title or bar.quote in description  # the quote guard passes it
    assert _bar_supported(bar)


@pytest.mark.parametrize(("title", "description"), NATIONALITY)
def test_a_nationality_bar_is_read_with_its_own_words(
    title: str, description: str
) -> None:
    bar = _bar(title, description, BarKind.nationality)
    assert bar is not None
    _assert_verbatim_and_supported(bar, title, description)


@pytest.mark.parametrize(("title", "description"), NOT_A_NATIONALITY_BAR)
def test_lines_that_only_mention_a_nationality_are_not_bars(
    title: str, description: str
) -> None:
    assert _bar(title, description, BarKind.nationality) is None


@pytest.mark.parametrize(("title", "description"), CLEARANCE)
def test_a_clearance_bar_is_read_with_its_own_words(
    title: str, description: str
) -> None:
    bar = _bar(title, description, BarKind.clearance)
    assert bar is not None
    _assert_verbatim_and_supported(bar, title, description)


@pytest.mark.parametrize(("title", "description"), NOT_A_CLEARANCE_BAR)
def test_lines_that_only_mention_clearance_are_not_bars(
    title: str, description: str
) -> None:
    assert _bar(title, description, BarKind.clearance) is None


def test_curly_punctuation_still_quotes_the_advert_verbatim() -> None:
    text = "Eligibility \u2013 UAE Nationals only \u2013 apply via the portal"
    bar = _bar("Analyst", text, BarKind.nationality)
    assert bar is not None
    assert bar.quote in text


def test_a_short_sentence_is_quoted_whole() -> None:
    text = "Great team. This role is open to UAE Nationals only. Apply now."
    bar = _bar("Analyst", text, BarKind.nationality)
    assert bar == HardBar(
        kind=BarKind.nationality, quote="This role is open to UAE Nationals only."
    )


def _job(description: str, title: str = "Data Analyst") -> Job:
    return Job(
        source="t",
        company="Acme",
        title=title,
        location="Dubai",
        url="https://acme.test/1",
        description=description,
    )


def _facts(*bars: HardBar, fit_score: int = 80) -> PostingFacts:
    return PostingFacts.model_validate(
        {
            "level": {"value": "not_stated", "quote": ""},
            "hard_bars": [b.model_dump() for b in bars],
            "fit_score": fit_score,
            "reason": "Strong SQL overlap.",
        }
    )


EDV = (
    "Due to the nature of the programme this team is working in, we require "
    "all candidates to hold an active eDV clearance for this role."
)


def test_a_bar_the_model_left_out_is_added_from_the_advert() -> None:
    facts = _facts()
    job = _job(EDV)
    out = resolve_hard_bars(facts, job)
    assert [b.kind for b in out.hard_bars] == [BarKind.clearance]
    assert facts.hard_bars == []  # never mutates its input
    # verbatim and naming its kind, so the quote guard keeps it as it is
    assert verify_facts(out, job).hard_bars == out.hard_bars


def test_the_title_is_read_as_well_as_the_description() -> None:
    job = _job(EDV, title="Data Analyst - UAE National, ICQA/LND Analytics Team")
    out = resolve_hard_bars(_facts(), job)
    assert {b.kind for b in out.hard_bars} == {BarKind.nationality, BarKind.clearance}
    nationality = next(b for b in out.hard_bars if b.kind == BarKind.nationality)
    assert nationality.quote == job.title


def test_a_kind_the_model_already_named_is_left_alone() -> None:
    job = _job("Saudi nationals only. " + EDV, title="Data Analyst (UAE Nationals)")
    model_bar = HardBar(kind=BarKind.nationality, quote="Saudi nationals only")
    out = resolve_hard_bars(_facts(model_bar), job)
    assert [b for b in out.hard_bars if b.kind == BarKind.nationality] == [model_bar]
    assert [b.kind for b in out.hard_bars] == [BarKind.nationality, BarKind.clearance]


def test_a_work_auth_bar_does_not_stand_in_for_nationality() -> None:
    job = _job("No visa sponsorship. This role is open to UAE Nationals only.")
    model_bar = HardBar(kind=BarKind.work_auth, quote="No visa sponsorship")
    out = resolve_hard_bars(_facts(model_bar), job)
    assert [b.kind for b in out.hard_bars] == [BarKind.work_auth, BarKind.nationality]


def test_an_advert_without_a_bar_is_unchanged() -> None:
    facts = _facts()
    job = _job("UAE nationals are encouraged to apply. No security clearance required.")
    assert resolve_hard_bars(facts, job) is facts


def test_it_is_idempotent() -> None:
    job = _job(EDV, title="Officer Regulatory Reporting - Emirati Talent")
    once = resolve_hard_bars(_facts(), job)
    assert resolve_hard_bars(once, job) == once


def test_a_full_list_keeps_the_model_bars_that_block() -> None:
    others = [
        HardBar(kind=BarKind.other, quote=f"Driving licence {n}") for n in range(4)
    ]
    model_bar = HardBar(kind=BarKind.nationality, quote="Saudi nationals only")
    job = _job("Saudi nationals only. Driving licence 0 1 2 3. " + EDV)
    out = resolve_hard_bars(_facts(*others, model_bar), job)
    assert len(out.hard_bars) == 5  # the schema's cap, so a cached copy reloads intact
    assert {BarKind.nationality, BarKind.clearance} <= {b.kind for b in out.hard_bars}
    assert PostingFacts.model_validate(out.model_dump()) == out


def test_the_added_bar_blocks_the_role() -> None:
    facts = resolve_hard_bars(_facts(fit_score=85), _job(EDV, title="Systems Engineer"))
    verdict = decide(facts, RulesConfig(), 40)
    assert verdict.verdict == Verdict.BLOCKED
    assert "active eDV clearance" in verdict.reason
