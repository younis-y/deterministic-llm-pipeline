"""2.5.7: every nationality bar used to block, including the ones the
candidate meets. "Jordanian nationals only" in Amman and "must be a British
citizen" in London were blocked for good for a Jordanian candidate with the
right to work in the UK (eval audit H1, 2026-10-07)."""

from __future__ import annotations

from rolescan.config import ProfileConfig
from rolescan.models import BarKind, Verdict
from rolescan.scoring.facts import HardBar, PostingFacts
from rolescan.scoring.rules import decide

NATS = ["jordanian", "jordan", "dominica", "dominican"]


def _facts(quote: str, kind: BarKind = BarKind.nationality) -> PostingFacts:
    return PostingFacts(
        fit_score=80, reason="fits", hard_bars=[HardBar(kind=kind, quote=quote)]
    )


def test_a_bar_the_candidate_meets_is_dropped() -> None:
    verdict = decide(_facts("Jordanian nationals only"), None, 40, nationalities=NATS)
    assert verdict.verdict is Verdict.APPLY and verdict.rule is None


def test_a_bar_naming_an_own_nationality_among_others_is_dropped() -> None:
    """Review focus 5."""
    verdict = decide(
        _facts("UAE or Jordanian nationals only"), None, 40, nationalities=NATS
    )
    assert verdict.rule is None


def test_a_bar_the_candidate_does_not_meet_still_blocks() -> None:
    verdict = decide(_facts("UAE nationals only"), None, 40, nationalities=NATS)
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
        {"nationalities": [" Jordanian ", "DOMINICA"]}
    )
    assert profile.nationalities == ["jordanian", "dominica"]


def test_dominica_does_not_match_dominican() -> None:
    """Word boundary: 'dominica' candidate is still blocked by 'Dominican nationals only'."""
    verdict = decide(
        _facts("Dominican nationals only"), None, 40, nationalities=["dominica"]
    )
    assert verdict.verdict is Verdict.BLOCKED


def test_jordan_does_not_match_jordanian() -> None:
    """Word boundary: 'jordan' candidate is still blocked by 'Jordanian nationals only'."""
    verdict = decide(
        _facts("Jordanian nationals only"), None, 40, nationalities=["jordan"]
    )
    assert verdict.verdict is Verdict.BLOCKED
