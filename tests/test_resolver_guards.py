"""One row per guard alternative of the years and hard-bar resolvers (2.5.8).

On 2026-10-07 a mutation pass removed each alternative of every precision
guard in `years_required_stated` and `hard_bars_stated`, one at a time: 477
of 616 mutants survived all 1,140 tests (a mutation score of 22.6%), and four
guards could be deleted outright with the suite still green. Each row is a
sentence that changes when its alternative is removed (the probe sentences
written for that purpose), so a guard word can no longer be dropped or
loosened without a red test. Every sentence goes through `Job`, so it is
whitespace-normalised exactly as production text is.

The ids name the guard and the alternative a row protects; one row, for
`_TITLE_SEPARATOR`, is a title from a real advert, the only input found that
needs that guard. Controls are probe sentences whose guards an older test
already pins; they stay as a net.

Whole guards: 48 of 48 killed (2026-10-08). The last survivor,
`_BAR_CLAUSE_START`, changes no probe sentence but is load-bearing: it keeps a
soft word in a neighbouring clause, across a comma or bracket, from waiving a
bar ("Saudi nationals only, Python is a plus." is still a nationality bar), and
the two `_BAR_CLAUSE_START` rows pin it.

Expected values are 2.5.7's output, read by hand, except two rows 2.5.7 got
wrong and 2.5.8 fixed: "Our people bring 9 years" (a third party's years,
read as a requirement) and "Security clearance: never required." (a bar).
"""

from __future__ import annotations

import pytest

from rolescan.models import Job
from rolescan.scoring.facts import hard_bars_stated, years_required_stated

Years = tuple[int, str] | None
Bars = list[tuple[str, str]]


def _years(text: str) -> Years:
    job = Job(
        source="t",
        company="Acme",
        title="Analyst",
        url="https://acme.example/1",
        description=text,
    )
    return years_required_stated(job.description)


def _bars(title: str, description: str = "") -> Bars:
    job = Job(
        source="t",
        company="Acme",
        title=title,
        url="https://acme.example/1",
        description=description,
    )
    return [
        (b.kind.value, b.quote) for b in hard_bars_stated(job.title, job.description)
    ]


# fmt: off
WHOLE_GUARDS = [
    pytest.param("Data Analyst", "Security clearance may be required.", [], id="_BAR_HEDGED_AFTER:may-be-required"),
    pytest.param("Data Analyst", "Security clearance may be needed for some projects.", [], id="_BAR_HEDGED_AFTER:may-be-needed"),
    pytest.param("Data Analyst", "SC clearance where applicable.", [], id="_BAR_HEDGED_AFTER:where-applicable"),
    pytest.param("Data Analyst", "SC clearance depending on the project.", [], id="_BAR_HEDGED_AFTER:depending-on"),
    pytest.param("Data Analyst", "Design and run our Emiratisation programme for UAE nationals.", [], id="_LOCALISATION_DUTY:design-and-run"),
    pytest.param("Data Analyst", "Manage the Emiratisation programme for UAE nationals.", [], id="_LOCALISATION_DUTY:manage"),
    pytest.param("Data Analyst - Office of Emiratisation", "", [], id="_OF_BEFORE:office-of"),
    pytest.param("SC Cleared Analyst", "", [("clearance", "SC Cleared Analyst")], id="_CLEARED:cleared-analyst"),
]

YEARS_GUARDS = [
    pytest.param("In your first 2 years as an Analyst you will rotate across desks.", None, id="_YEARS_CAP:first"),
    pytest.param("After 2 years as an Analyst, you may be promoted to Associate.", None, id="_YEARS_CAP:after"),
    pytest.param("Over the next 2 years as a graduate, you will rotate.", None, id="_YEARS_CAP:next"),
    pytest.param("Your previous 3 years of experience will not count.", None, id="_YEARS_CAP:previous"),
    pytest.param("The scheme provides 2 years of experience in trading.", None, id="_YEARS_CAP:provides"),
    pytest.param("The programme covers 2 years of experience across desks.", None, id="_YEARS_CAP:covers"),
    pytest.param("You must have served 3 years as an officer.", None, id="_YEARS_CAP:served"),
    pytest.param("Desirable: 3+ years of experience in SQL.", None, id="_YEARS_SOFT:desirable"),
    pytest.param("Good to have: 3+ years of experience in SQL.", None, id="_YEARS_SOFT:good-to-have"),
    pytest.param("A plus: 3+ years of experience in SQL.", None, id="_YEARS_SOFT:a-plus"),
    pytest.param("An advantage: 3+ years of experience in SQL.", None, id="_YEARS_SOFT:advantage"),
    pytest.param("3+ years of experience in SQL is desirable.", None, id="_YEARS_SOFT_AFTER:desirable"),
    pytest.param("3+ years of experience in SQL is a bonus.", None, id="_YEARS_SOFT_AFTER:bonus"),
    pytest.param("3+ years of experience in SQL is optional.", None, id="_YEARS_SOFT_AFTER:optional"),
    pytest.param("3+ years of experience in SQL would be helpful.", None, id="_YEARS_SOFT_AFTER:helpful"),
    pytest.param("3+ years of experience in SQL is valued.", None, id="_YEARS_SOFT_AFTER:valued"),
    pytest.param("3+ years of experience in SQL is welcome.", None, id="_YEARS_SOFT_AFTER:welcome"),
    pytest.param("3+ years of experience in SQL is not required.", None, id="_YEARS_SOFT_AFTER:not-required"),
    pytest.param("3+ years of experience in SQL is not essential.", None, id="_YEARS_SOFT_AFTER:not-essential"),
    pytest.param("3+ years of experience in SQL is preferable.", None, id="_YEARS_SOFT_AFTER:preferable"),
    pytest.param("3+ years of experience in SQL is desired.", None, id="_YEARS_SOFT_AFTER:desired"),
    pytest.param("3+ years of experience in SQL is nice to have.", None, id="_YEARS_SOFT_AFTER:nice-to-have"),
    pytest.param("3+ years of experience in SQL is advantageous.", None, id="_YEARS_SOFT_AFTER:advantageous"),
    pytest.param("You will join a team led by an engineer with 12 years of experience.", None, id="_YEARS_OTHERS:led-by"),
    pytest.param("You will be managed by a director with 12 years of experience.", None, id="_YEARS_OTHERS:managed-by"),
    pytest.param("The desk is headed by a trader with 12 years of experience.", None, id="_YEARS_OTHERS:headed-by"),
    pytest.param("You will report to the Head of Data, with 12 years of experience.", None, id="_YEARS_OTHERS:reports-to"),
    pytest.param("Our analysts average 7 years of experience.", None, id="_YEARS_OTHERS:average"),
    pytest.param("Join a team of experts with 9 years of experience.", None, id="_YEARS_OTHERS:team-of"),
    pytest.param("Our founders have 12 years of experience at a large bank.", None, id="_YEARS_OTHERS:our-founders"),
    pytest.param("Our consultants have 12 years of experience.", None, id="_YEARS_OTHERS:our-consultants"),
    pytest.param("Our people bring 9 years of experience.", None, id="_YEARS_REQ_VERB:brings"),
    pytest.param("Our staff have 9 years of experience.", None, id="_YEARS_OTHERS:our-staff"),
    pytest.param("Our leadership has 12 years of experience.", None, id="_YEARS_OTHERS:our-leadership"),
    pytest.param("3+ years of experience, or a PhD in a related field.", None, id="_YEARS_WAIVED:phd"),
    pytest.param("3+ years of experience, or an MSc in Statistics.", None, id="_YEARS_WAIVED:msc"),
    pytest.param("3+ years of experience, or a doctorate.", None, id="_YEARS_WAIVED:doctorate"),
    pytest.param("A PhD, or 3+ years of experience.", None, id="_YEARS_WAIVED_BEFORE:phd"),
    pytest.param("An MSc, or 3+ years of experience.", None, id="_YEARS_WAIVED_BEFORE:msc"),
    pytest.param("A relevant qualification, or 3+ years of experience.", None, id="_YEARS_WAIVED_BEFORE:qualification"),
    pytest.param("A Master's, or 3+ years of experience.", None, id="_YEARS_WAIVED_BEFORE:masters"),
    pytest.param("3 years as an Associate before promotion to VP.", None, id="_YEARS_CAREER_PATH:promotion-to"),
    pytest.param("With 10 years of experience, we deliver for clients.", None, id="_YEARS_COMPANY_AFTER:we"),
    pytest.param("With 10 years of experience, our team delivers.", None, id="_YEARS_COMPANY_AFTER:our"),
    pytest.param("With 10 years of experience in the region, Acme Group has grown.", None, id="_YEARS_COMPANY_AFTER:has"),
    pytest.param("With 10 years of experience in the region, Acme Group was founded.", None, id="_YEARS_COMPANY_AFTER:was"),
    pytest.param("With 10 years of experience in the region, Acme Group serves clients.", None, id="_YEARS_COMPANY_AFTER:serves"),
    pytest.param("Acme Capital boasts 10 years of experience.", None, id="_YEARS_COMPANY_SUBJECT:boasts"),
    pytest.param("Acme Capital brings 10 years of experience.", None, id="_YEARS_COMPANY_SUBJECT:brings"),
    pytest.param("Applicants who have 5 years of experience.", (5, "5 years of experience"), id="_YEARS_CANDIDATE:applicants"),
    pytest.param("We want individuals who have 5 years of experience.", (5, "5 years of experience"), id="_YEARS_CANDIDATE:individuals"),
    pytest.param("Our team is seeking someone with 5+ years of experience.", (5, "5+ years of experience"), id="_YEARS_REQ_VERB:seeking"),
    pytest.param("Our team wants someone with 5+ years of experience.", (5, "5+ years of experience"), id="_YEARS_REQ_VERB:wants"),
    pytest.param("Our team must have 5+ years of experience.", (5, "5+ years of experience"), id="_YEARS_REQ_VERB:must-have"),
    pytest.param("Reporting to the CTO, you will bring 5+ years of experience.", (5, "5+ years of experience"), id="_YEARS_REQ_VERB:you-will-bring"),
    pytest.param("2 years rotation experience", None, id="_YEARS_STATED:gap-word-rotation"),
]

BAR_GUARDS = [
    pytest.param("Data Analyst", "Open to UAE nationals and residents.", [], id="_BAR_ALTERNATIVE:residents"),
    pytest.param("Data Analyst", "Open to UAE nationals and visa holders.", [], id="_BAR_ALTERNATIVE:visas"),
    pytest.param("Data Analyst", "Open to UAE nationals and non-nationals.", [], id="_BAR_ALTERNATIVE:non-nationals"),
    pytest.param("Data Analyst", "Open to UAE nationals and all nationalities.", [], id="_BAR_ALTERNATIVE:all-nationalities"),
    pytest.param("Data Analyst", "Nationality: UAE or any.", [], id="_BAR_ALTERNATIVE:or-any"),
    pytest.param("Data Analyst", "Open to UAE nationals and international applicants.", [], id="_BAR_ALTERNATIVE:international-applicants"),
    pytest.param("Data Analyst", "Must be a British citizen or eligible to work in the UK.", [], id="_BAR_OR_ALTERNATIVE:eligible-to-work"),
    pytest.param("Data Analyst", "Must be a British citizen or hold ILR.", [], id="_BAR_OR_ALTERNATIVE:ilr"),
    pytest.param("Data Analyst", "Coordinate security vetting for new starters.", [], id="_CLEARANCE_DUTY:coordinate"),
    pytest.param("Data Analyst", "Administer security vetting for new starters.", [], id="_CLEARANCE_DUTY:administer"),
    pytest.param("Data Analyst", "Handle security vetting for new starters.", [], id="_CLEARANCE_DUTY:handle"),
    pytest.param("Data Analyst", "Run security vetting for new starters.", [], id="_CLEARANCE_DUTY:run"),
    pytest.param("Data Analyst", "Support security vetting for new starters.", [], id="_CLEARANCE_DUTY:support"),
    pytest.param("Data Analyst", "Manage security vetting for new starters.", [], id="_CLEARANCE_DUTY:manage"),
    pytest.param("Data Analyst", "Process security vetting for new starters.", [], id="_CLEARANCE_DUTY:process"),
    pytest.param("Data Analyst", "Maintain security clearance logs.", [], id="_CLEARANCE_DUTY:maintain"),
    pytest.param("Data Analyst", "Sponsor security clearance renewals.", [], id="_CLEARANCE_DUTY:sponsor"),
    pytest.param("Data Analyst", "Help candidates to obtain security clearance.", [], id="_CLEARANCE_DUTY:help-to-obtain"),
    pytest.param("Data Analyst", "Assist staff to obtain security clearance.", [], id="_CLEARANCE_DUTY:assist-to-obtain"),
    pytest.param("Data Analyst", "Clients obtain security clearance through us.", [], id="_CLEARANCE_DUTY:clients-obtain"),
    pytest.param("Data Analyst", "Our clients hold active SC clearance.", [], id="_CLEARANCE_HOLDERS:clients"),
    pytest.param("Data Analyst", "Our users have valid SC clearance.", [], id="_CLEARANCE_HOLDERS:users"),
    pytest.param("Data Analyst", "Our colleagues who hold SC clearance.", [], id="_CLEARANCE_HOLDERS:colleagues"),
    pytest.param("Data Analyst", "Our partners with current security clearance.", [], id="_CLEARANCE_HOLDERS:partners"),
    pytest.param("Data Analyst", "Join the security clearance unit.", [], id="_CLEARANCE_WORK_AFTER:unit"),
    pytest.param("Data Analyst", "Join the security vetting department.", [], id="_CLEARANCE_WORK_AFTER:department"),
    pytest.param("Data Analyst", "Join the security vetting office.", [], id="_CLEARANCE_WORK_AFTER:office"),
    pytest.param("Data Analyst", "Submit security clearance requests.", [], id="_CLEARANCE_WORK_AFTER:requests"),
    pytest.param("Data Analyst", "Keep the security clearance register.", [], id="_CLEARANCE_WORK_AFTER:registers"),
    pytest.param("Data Analyst", "You will sit alongside DV cleared analysts.", [], id="_CLEARANCE_PEERS_BEFORE:alongside"),
    pytest.param("Data Analyst", "Among SC cleared engineers.", [], id="_CLEARANCE_PEERS_BEFORE:among"),
    pytest.param("Data Analyst", "Join fellow SC cleared engineers.", [], id="_CLEARANCE_PEERS_BEFORE:fellow"),
    pytest.param("Data Analyst", "Join other SC cleared engineers.", [], id="_CLEARANCE_PEERS_BEFORE:other"),
    pytest.param("Data Analyst", "BPSS and DV security clearance required.", [("clearance", "BPSS and DV security clearance required.")], id="_ABOVE_BPSS:DV"),
    pytest.param("Data Analyst", "BPSS and eDV clearance required.", [("clearance", "BPSS and eDV clearance required.")], id="_ABOVE_BPSS:eDV"),
    pytest.param("Data Analyst", "BPSS and developed vetting required.", [("clearance", "BPSS and developed vetting required.")], id="_ABOVE_BPSS:developed-vetting"),
    pytest.param("Data Analyst", "BPSS and NPPV3 required.", [("clearance", "BPSS and NPPV3 required.")], id="_ABOVE_BPSS:nppv"),
    pytest.param("Data Analyst", "BPSS and UKSV security clearance required.", [("clearance", "BPSS and UKSV security clearance required.")], id="_ABOVE_BPSS:uksv"),
    pytest.param("Data Analyst", "Work without security clearance.", [], id="_BAR_NEGATED_BEFORE:without"),
    pytest.param("Data Analyst", "Neither UAE nationality nor security clearance is required.", [], id="_BAR_NEGATED_BEFORE:nor"),
    pytest.param("Data Analyst", "Security clearance: none.", [], id="_BAR_NEGATED_AFTER:none"),
    pytest.param("Data Analyst", "Security clearance: n/a.", [], id="_BAR_NEGATED_AFTER:n/a"),
    pytest.param("Data Analyst", "Security clearance isn't required.", [], id="_BAR_NEGATED_AFTER:isnt"),
    pytest.param("Data Analyst", "SC clearance is particularly useful.", [], id="_BAR_SOFT_AFTER:particularly"),
    pytest.param("Data Analyst", "Preferably hold active SC clearance.", [], id="_BAR_SOFT_BEFORE:preferably"),
    pytest.param("Data Analyst", "Desirable: active SC clearance.", [], id="_BAR_SOFT_BEFORE:desirable"),
    pytest.param("Data Analyst", "Bonus: active SC clearance.", [], id="_BAR_SOFT_BEFORE:bonus"),
    pytest.param("Data Analyst", "Priority given to UAE nationals only.", [], id="_BAR_SOFT_BEFORE:priority"),
    pytest.param("Data Analyst", "Particularly UAE nationals only.", [], id="_BAR_SOFT_BEFORE:particularly"),
    pytest.param("Data Analyst", "Current SC clearance is beneficial.", [], id="_BAR_SOFT_IN_CLAUSE:beneficial"),
    pytest.param("Data Analyst", "Certain roles require security clearance.", [], id="_BAR_HEDGED_BEFORE:certain"),
    pytest.param("Data Analyst", "Most of our roles require security clearance.", [], id="_BAR_HEDGED_BEFORE:most"),
    pytest.param("Data Analyst", "You might need security clearance.", [], id="_BAR_HEDGED_BEFORE:might-need"),
    pytest.param("Data Analyst", "You could be asked for security clearance.", [], id="_BAR_HEDGED_BEFORE:could-be-asked"),
    pytest.param("Data Analyst", "If required, security clearance.", [], id="_BAR_HEDGED_BEFORE:if-required"),
    pytest.param("Emiratisation Director", "", [], id="_TITLE_ROLE_NOUN:director"),
    pytest.param("Emiratisation Advisor", "", [], id="_TITLE_ROLE_NOUN:advisor"),
    pytest.param("Emiratisation Partner", "", [], id="_TITLE_ROLE_NOUN:partner"),
    pytest.param("Emiratisation Consultant", "", [], id="_TITLE_ROLE_NOUN:consultant"),
    pytest.param("Emiratisation Coordinator", "", [], id="_TITLE_ROLE_NOUN:coordinator"),
    pytest.param("Human Resources Analyst - Emiratisation", "", [], id="_HR:human-resources"),
    pytest.param("Data Analyst", "Saudi nationals only • Python • SQL", [("nationality", "Saudi nationals only")], id="_BAR_SENTENCE_END:bullet"),
    pytest.param("Data Analyst", "Saudi nationals only, Python is a plus.", [("nationality", "Saudi nationals only, Python is a plus.")], id="_BAR_CLAUSE_START:soft-word-after-comma"),
    pytest.param("Data Analyst", "Preferably based in Dubai, security clearance is required.", [("clearance", "Preferably based in Dubai, security clearance is required.")], id="_BAR_CLAUSE_START:soft-word-before-comma"),
    pytest.param("Officer, Customer Support - Emiratized Role", "", [("nationality", "Officer, Customer Support - Emiratized Role")], id="_TITLE_SEPARATOR:dash-segment"),
]

YEARS_CONTROLS = [
    pytest.param("Graduates with under 3 years of experience are welcome to apply.", None, id="years-control-01"),
    pytest.param("Candidates with fewer than 3 years of experience are welcome to apply.", None, id="years-control-02"),
    pytest.param("Max 3 years of experience.", None, id="years-control-03"),
    pytest.param("within the past 3 years of experience in data", None, id="years-control-04"),
    pytest.param("You will report to the Head of Data, who has 12 years of experience.", None, id="years-control-05"),
    pytest.param("Our team members have 9 years of experience.", None, id="years-control-06"),
    pytest.param("3+ years of experience, or a Master's degree.", None, id="years-control-07"),
    pytest.param("3+ years of experience as an analyst, or equivalent.", None, id="years-control-08"),
    pytest.param("3 years as an Associate before being promoted to VP.", None, id="years-control-09"),
    pytest.param("3 years as an Associate, then you will progress to VP.", None, id="years-control-10"),
    pytest.param("3 years as an Associate leads to VP.", None, id="years-control-11"),
    pytest.param("Acme Capital offers 10 years of experience.", None, id="years-control-12"),
    pytest.param("We need someone who has 5 years of experience.", (5, "5 years of experience"), id="years-control-13"),
    pytest.param("The role reports to the CFO and requires 5+ years of experience.", (5, "5+ years of experience"), id="years-control-14"),
    pytest.param("2 years and gives you hands-on experience.", None, id="years-control-15"),
    pytest.param("2 years with our experience team.", None, id="years-control-16"),
    pytest.param("2 years in an experience role.", (2, "2 years in an experience role"), id="years-control-17"),
    pytest.param("2 years graduate scheme experience", None, id="years-control-18"),
]

BAR_CONTROLS = [
    pytest.param("Data Analyst", "Open to UAE nationals and expats.", [], id="bar-control-01"),
    pytest.param("Data Analyst", "Must be a British citizen or have permanent residence.", [], id="bar-control-02"),
    pytest.param("Data Analyst", "Must be a British citizen, or anyone with the right to work in the UK.", [], id="bar-control-03"),
    pytest.param("Data Analyst", "UK nationals and EU citizens with settled status.", [], id="bar-control-04"),
    pytest.param("Data Analyst", "UK nationals and candidates with settled status.", [], id="bar-control-05"),
    pytest.param("Data Analyst", "We sponsor your SC clearance.", [("clearance", "We sponsor your SC clearance.")], id="bar-control-06"),
    pytest.param("Data Analyst", "Report to the security vetting manager.", [], id="bar-control-07"),
    pytest.param("Data Analyst", "Baseline Personnel Security Standard checks apply; security clearance is part of onboarding.", [("clearance", "security clearance is part of onboarding.")], id="bar-control-08"),
    pytest.param("Data Analyst", "Security clearance: never required.", [], id="bar-control-09"),
    pytest.param("Data Analyst", "Security clearance - not required.", [], id="bar-control-10"),
    pytest.param("Data Analyst", "Security clearance will be not needed.", [], id="bar-control-11"),
    pytest.param("Data Analyst", "Security clearance is not mandatory for this role.", [], id="bar-control-12"),
    pytest.param("Data Analyst", "UAE nationals preferred.", [], id="bar-control-13"),
    pytest.param("Data Analyst", "UAE nationals desirable.", [], id="bar-control-14"),
    pytest.param("Data Analyst", "UAE nationals are a bonus.", [], id="bar-control-15"),
    pytest.param("Data Analyst", "UAE nationals are a plus.", [], id="bar-control-16"),
    pytest.param("Data Analyst", "UAE nationals are an advantage.", [], id="bar-control-17"),
    pytest.param("Data Analyst", "UAE nationals are welcome.", [], id="bar-control-18"),
    pytest.param("Data Analyst", "UAE nationals have priority.", [], id="bar-control-19"),
    pytest.param("Data Analyst", "UAE nationals will be prioritised.", [], id="bar-control-20"),
    pytest.param("Data Analyst", "Ideally hold active SC clearance.", [], id="bar-control-21"),
    pytest.param("Data Analyst", "Current SC clearance is not essential.", [], id="bar-control-22"),
    pytest.param("Data Analyst", "Preference will be given to UAE nationals only.", [], id="bar-control-23"),
    pytest.param("HR Graduate - Emiratisation Manager", "", [], id="bar-control-24"),
    pytest.param("Emiratisation Intern", "", [("nationality", "Emiratisation Intern")], id="bar-control-25"),
    pytest.param("Senior Manager - Emiratisation", "", [], id="bar-control-26"),
    pytest.param("Senior Manager_Emiratisation", "", [], id="bar-control-27"),
    pytest.param("Senior Manager | Emiratisation", "", [], id="bar-control-28"),
]
# fmt: on


@pytest.mark.parametrize(("title", "description", "expected"), WHOLE_GUARDS)
def test_a_guard_no_test_protected(
    title: str, description: str, expected: Bars
) -> None:
    """`_BAR_HEDGED_AFTER`, `_LOCALISATION_DUTY` (with `_localisation_waived`),
    `_OF_BEFORE` and `_CLEARED` could each be deleted with every test green."""
    assert _bars(title, description) == expected


@pytest.mark.parametrize(("text", "expected"), YEARS_GUARDS)
def test_a_years_guard_alternative(text: str, expected: Years) -> None:
    assert _years(text) == expected


@pytest.mark.parametrize(("title", "description", "expected"), BAR_GUARDS)
def test_a_bar_guard_alternative(title: str, description: str, expected: Bars) -> None:
    assert _bars(title, description) == expected


@pytest.mark.parametrize(("text", "expected"), YEARS_CONTROLS)
def test_a_years_control(text: str, expected: Years) -> None:
    assert _years(text) == expected


@pytest.mark.parametrize(("title", "description", "expected"), BAR_CONTROLS)
def test_a_bar_control(title: str, description: str, expected: Bars) -> None:
    assert _bars(title, description) == expected
