"""Stage one: the cheap deterministic prefilter.

This is not trying to be clever. Its whole job is to throw away the eighty
percent of postings that are obviously irrelevant, so the LLM only ever reads
plausible ones. That is what keeps a daily scan across forty employers costing
pennies rather than pounds.

It also runs standalone when no API key is configured, so the tool degrades to
something useful rather than to nothing.
"""

from __future__ import annotations

from rolescan.config import ProfileConfig
from rolescan.models import Job, ScoredJob

__all__ = ["TITLE_MULTIPLIER", "score_keywords"]

TITLE_MULTIPLIER = 3
"""A term in the title says what the role *is*. The same term buried in the
body often just describes the team. Weighting them equally is the classic way
these filters go wrong."""


def score_keywords(job: Job, profile: ProfileConfig) -> ScoredJob:
    """Score one posting against the configured keyword and blocker weights."""
    title = job.title.casefold()
    blob = job.blob
    total = 0
    hits: list[str] = []
    penalties: list[str] = []
    blockers: list[str] = []

    for term, weight in profile.keywords.items():
        needle = term.casefold()
        if needle in title:
            total += weight * TITLE_MULTIPLIER
            hits.append(f"{term} (title)")
        elif needle in blob:
            total += weight
            hits.append(term)

    for term, penalty in profile.blockers.items():
        if term.casefold() in blob:
            total -= penalty
            penalties.append(term)
            if penalty >= profile.hard_blocker_score:
                blockers.append(term)

    if not _location_ok(job, profile):
        total -= profile.location_penalty
        # Not a hard bar: the candidate allows remote and has several target
        # cities, so this goes in keyword_penalties (for display) but never
        # in blocker_hits (which forces a `blocked` verdict).
        penalties.append("location mismatch")

    return ScoredJob(
        job=job,
        keyword_score=total,
        keyword_hits=hits,
        keyword_penalties=penalties,
        blocker_hits=blockers,
    )


def _location_ok(job: Job, profile: ProfileConfig) -> bool:
    if not profile.locations:
        return True
    if job.remote and profile.allow_remote:
        return True
    loc = job.location.casefold()
    if not loc:
        # No stated location is not evidence of a bad one. Let the LLM decide
        # rather than penalising a posting for being terse.
        return True
    return any(w.casefold() in loc for w in profile.locations)
