"""Stage one: the cheap deterministic prefilter.

This is not trying to be clever. Its whole job is to throw away the eighty
percent of postings that are obviously irrelevant, so the LLM only ever reads
plausible ones. That is what keeps a daily scan across forty employers costing
pennies rather than pounds.

It also runs standalone when no API key is configured, so the tool degrades to
something useful rather than to nothing.
"""

from __future__ import annotations

import re

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
        if _term_hit(term, blob):
            total -= penalty
            penalties.append(term)

    # Hardness is a list, not a threshold on the weights above. A weight says
    # what a term costs; this says what no application can get past. A term
    # can be in both (it costs its weight AND blocks), in `blockers` alone (a
    # preference, however heavy), or here alone (a bar that costs nothing).
    # Both fields are normalised at load and matched identically, so the two
    # can never disagree about whether a term is present.
    for term in profile.hard_blockers:
        if _term_hit(term, blob):
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


def _term_hit(term: str, blob: str) -> bool:
    r"""Whether a configured blocker term appears in the posting as a whole word.

    Used for every blocker, weighted and hard alike. The loose substring test
    this replaces was kept for the weighted ones on the grounds that a wrong
    points deduction is recoverable - the LLM still reads the posting and the
    reader still sees the flag. It is not: the deduction is taken before
    `min_keyword_score`, and a posting that falls under that gate is
    prefiltered, written to `seen` and never shown again. A 50-point term
    charged to a posting that does not contain it deletes a modest role
    exactly as thoroughly as a hard bar does, and more quietly, because
    nothing records that a blocker was involved.

    Unanchored matching gets that wrong in practice, not in theory. `crypto`
    as a hard bar (no interest in crypto trading) also matches
    "cryptographic", so a security-adjacent quant role mentioning
    cryptographic hashing once was force-blocked over the model's objection.
    The weighted side of the same list carries the same defect at a smaller
    blast radius and a higher frequency: `head of` matches "head office" and
    `director` matches "directorate" and "board of directors", all of them
    constant in UK and Gulf adverts.

    `\w` lookarounds rather than `\b`, because `\b` is defined against the
    adjacent character in the PATTERN as well as the text: for a term ending
    in punctuation, such as `c++`, a trailing `\b` would then demand a word
    character immediately after it and the bar would silently stop matching.
    Multi-word terms are unaffected either way.
    """
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", blob) is not None


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
