"""2.5.7: every nationality bar used to block, including the ones the
candidate meets. A bar naming the candidate's own nationality is not a bar for
that candidate, so "Freedonian nationals only" no longer hides a role from a
Freedonian, while a bar naming some other nationality still does."""

from __future__ import annotations

from rolescan.config import ProfileConfig
from rolescan.models import BarKind, Verdict
from rolescan.scoring.facts import HardBar, PostingFacts
from rolescan.scoring.rules import decide

NATS = ["freedonian", "freedonia", "sylvanian", "sylvania"]


def _facts(quote: str, kind: BarKind = BarKind.nationality) -> PostingFacts:
    return PostingFacts(
        fit_score=80, reason="fits", hard_bars=[HardBar(kind=kind, quote=quote)]
    )


def test_a_bar_the_candidate_meets_is_dropped() -> None:
    verdict = decide(_facts("Freedonian nationals only"), None, 40, nationalities=NATS)
    assert verdict.verdict is Verdict.APPLY and verdict.rule is None


def test_a_bar_naming_an_own_nationality_among_others_is_dropped() -> None:
    """One alternative the candidate holds is enough: the bar is not a bar."""
    verdict = decide(
        _facts("Ruritanian or Freedonian nationals only"),
        None,
        40,
        nationalities=NATS,
    )
    assert verdict.rule is None


def test_a_bar_the_candidate_does_not_meet_still_blocks() -> None:
    verdict = decide(_facts("Ruritanian nationals only"), None, 40, nationalities=NATS)
    assert verdict.verdict is Verdict.BLOCKED


def test_a_clearance_bar_is_unaffected() -> None:
    verdict = decide(
        _facts("must hold SC clearance", BarKind.clearance),
        None,
        40,
        nationalities=["british"],
    )
    assert verdict.verdict is Verdict.BLOCKED


def test_nationalities_are_normalised_at_load() -> None:
    profile = ProfileConfig.model_validate(
        {"nationalities": [" Freedonian ", "SYLVANIA"]}
    )
    assert profile.nationalities == ["freedonian", "sylvania"]


def test_a_country_name_does_not_match_its_demonym_longer_form() -> None:
    """Word boundary: a 'sylvania' candidate is still blocked by 'Sylvanian nationals only'."""
    verdict = decide(
        _facts("Sylvanian nationals only"), None, 40, nationalities=["sylvania"]
    )
    assert verdict.verdict is Verdict.BLOCKED


def test_freedonia_does_not_match_freedonian() -> None:
    """Word boundary: a 'freedonia' candidate is still blocked by 'Freedonian nationals only'."""
    verdict = decide(
        _facts("Freedonian nationals only"), None, 40, nationalities=["freedonia"]
    )
    assert verdict.verdict is Verdict.BLOCKED
