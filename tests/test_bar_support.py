"""2.5.2: a nationality or clearance bar must be supported by its own quote.

The quote guard proves a quote is in the posting, not that it says what the
bar claims. One evaluation posting was blocked as a structural bar - hidden,
recorded as seen, never prepared - on a verbatim quote that had nothing to do
with nationality or clearance: the advert never mentions either. A block is
a one-way door, so `verify_facts` now also requires a nationality bar's quote
to name nationality and a clearance bar's quote to name clearance. A bar
whose quote does not is dropped, exactly like a bar whose quote is not in the
posting.
"""

from __future__ import annotations

import pytest

from rolescan.models import BarKind, Job, Verdict
from rolescan.scoring.facts import HardBar, LevelFact, PostingFacts, verify_facts
from rolescan.scoring.rules import decide

# The real advert from the evaluation run (company name left out). Its full
# 2,688-character description says nothing about nationality or clearance.
TELECOM_ANALYST = Job(
    source="t",
    company="Acme",
    title="Data Enablement & Platforms Analyst",
    url="https://x/2",
    description=(
        "The role holder shall carry out his duties in accordance with the "
        "stipulated business policies and procedure Job Responsibility Supports "
        "managing the coordination with corporate IT teams to ensure the "
        "security of the data and adheres to the established internal security "
        "practices. Years Of Experience 0 - 2 Years Nature Of Experience Prior "
        "experience in Data Enablement & Platforms and specifically in the "
        "telecommunication function is strongly preferred"
    ),
)


def _facts(*bars: HardBar) -> PostingFacts:
    return PostingFacts(level=LevelFact(), hard_bars=list(bars), fit_score=60,
                        reason="Data platform overlap.")


def _kept(job: Job, kind: BarKind, quote: str) -> bool:
    return bool(verify_facts(_facts(HardBar(kind=kind, quote=quote)), job).hard_bars)


def _job(description: str) -> Job:
    return Job(source="t", company="Acme", title="Data Analyst", url="https://x/1",
               description=description)


@pytest.mark.parametrize(
    ("kind", "quote"),
    [
        (BarKind.nationality,
         "The role holder shall carry out his duties in accordance with the "
         "stipulated business policies and procedure"),
        (BarKind.clearance, "adheres to the established internal security practices"),
        (BarKind.clearance, "ensure the security of the data"),
    ],
)
def test_the_audit_bar_with_no_nationality_or_clearance_words_is_dropped(
    kind: BarKind, quote: str
) -> None:
    facts = _facts(HardBar(kind=kind, quote=quote))
    assert not verify_facts(facts, TELECOM_ANALYST).hard_bars
    # ... so the posting is decided on fit, not blocked at 20.
    verdict = decide(verify_facts(facts, TELECOM_ANALYST), None, 50)
    assert verdict.verdict != Verdict.BLOCKED
    assert verdict.fit_score == 60


@pytest.mark.parametrize(
    "quote",
    [
        "UAE Nationals only",
        "Open to Saudi nationals",
        "British citizenship required",
        "Must be a US citizen",
        "Applicants must hold a valid UK passport",
        "This is an Emiratisation role",
        "Emiratization programme",
        "Part of our Saudization drive",
        "Emirati candidates only",
        "Nationality: Qatari",
        "Branch Sales Officer (UAEN Only )",
        "Applicants holding a Family Book are encouraged to apply",
    ],
)
def test_a_genuine_nationality_bar_survives(quote: str) -> None:
    job = _job(f"About the role. {quote}. Apply now.")
    assert _kept(job, BarKind.nationality, quote)
    assert decide(verify_facts(_facts(HardBar(kind=BarKind.nationality, quote=quote)), job),
                  None, 50).verdict == Verdict.BLOCKED


@pytest.mark.parametrize(
    "quote",
    [
        "Must hold active SC clearance",
        "SC cleared",
        "DV",
        "Active SC + UKSV confirmation required",
        "Must pass BPSS checks",
        "NPPV3 required",
        "Subject to a security check",
        "Developed Vetting required",
        "Active TS/SCI with polygraph",
        "Security clearance required",
    ],
)
def test_a_genuine_clearance_bar_survives(quote: str) -> None:
    job = _job(f"About the role. {quote}. Apply now.")
    assert _kept(job, BarKind.clearance, quote)


@pytest.mark.parametrize(
    "quote",
    [
        # Right to work is work authorisation, never nationality.
        "You must have the right to work in the UK",
        "No visa sponsorship available",
        # A country named as a place is not a nationality requirement.
        "Based in Riyadh, Saudi Arabia",
        "Join our international team",
    ],
)
def test_a_nationality_bar_without_nationality_words_is_dropped(quote: str) -> None:
    job = _job(f"About the role. {quote}. Apply now.")
    assert not _kept(job, BarKind.nationality, quote)


@pytest.mark.parametrize(
    "quote",
    [
        "Strong attention to security best practices",
        "MSc in Computer Science",  # "Sc" in a degree is not SC clearance
        "Background in ESG reporting",
    ],
)
def test_a_clearance_bar_without_clearance_words_is_dropped(quote: str) -> None:
    job = _job(f"About the role. {quote}. Apply now.")
    assert not _kept(job, BarKind.clearance, quote)


@pytest.mark.parametrize(
    ("kind", "quote"),
    [
        (BarKind.work_auth, "No visa sponsorship available"),
        (BarKind.other, "A full UK driving licence is required"),
    ],
)
def test_other_bar_kinds_need_only_a_verbatim_quote(kind: BarKind, quote: str) -> None:
    job = _job(f"About the role. {quote}. Apply now.")
    assert _kept(job, kind, quote)


def test_only_the_unsupported_bar_is_dropped() -> None:
    job = _job("UAE Nationals only. We value teamwork and integrity.")
    facts = _facts(
        HardBar(kind=BarKind.clearance, quote="We value teamwork and integrity"),
        HardBar(kind=BarKind.nationality, quote="UAE Nationals only"),
    )
    assert verify_facts(facts, job).hard_bars == [
        HardBar(kind=BarKind.nationality, quote="UAE Nationals only")
    ]
