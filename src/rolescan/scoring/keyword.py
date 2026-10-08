"""Stage one: the cheap deterministic prefilter.

This is not trying to be clever. Its whole job is to throw away the eighty
percent of postings that are obviously irrelevant, so the LLM only ever reads
plausible ones. That is what keeps a daily scan across forty employers costing
pennies rather than pounds.

It also runs standalone when no API key is configured, so the tool degrades to
something useful rather than to nothing.
"""

from __future__ import annotations

import functools
import re

from rolescan.config import ProfileConfig
from rolescan.models import Job, ScoredJob

__all__ = [
    "AGENCY_PENALTY",
    "LOCATION_PENALTY",
    "SYNTHETIC_PENALTIES",
    "TITLE_MULTIPLIER",
    "score_keywords",
]

TITLE_MULTIPLIER = 3
"""A term in the title says what the role *is*. The same term buried in the
body often just describes the team. Weighting them equally is the classic way
these filters go wrong."""

AGENCY_PENALTY = "posted by an agency"
LOCATION_PENALTY = "location mismatch"
SYNTHETIC_PENALTIES = frozenset({AGENCY_PENALTY, LOCATION_PENALTY})
"""The two `keyword_penalties` labels that are not configured `blockers`
terms, so a reader of that list can tell them apart."""


def score_keywords(job: Job, profile: ProfileConfig) -> ScoredJob:
    """Score one posting against the configured keyword and blocker weights."""
    title = job.title.casefold()
    blob = job.blob
    total = 0
    hits: list[str] = []
    penalties: list[str] = []

    for term, weight in profile.keywords.items():
        needle = term.casefold()
        if needle in title:
            total += weight * TITLE_MULTIPLIER
            hits.append(f"{term} (title)")
        elif needle in blob:
            total += weight
            hits.append(term)

    title_only = set(profile.title_only_blockers)
    for term, penalty in profile.blockers.items():
        haystack = title if term in title_only else blob
        if _term_hit(term, haystack):
            total -= penalty
            penalties.append(term)

    # Hardness is a list, not a threshold on the weights above. A weight says
    # what a term costs; this says what no application can get past. A term
    # can be in both (it costs its weight AND blocks), in `blockers` alone (a
    # preference, however heavy), or here alone (a bar that costs nothing).
    # Both fields are normalised at load, so the same spelling of a term is
    # in both. A term that is also in `title_only_blockers` is the exception
    # to "matched identically": it costs its weight only from the title, but
    # still bars from the whole text here.
    blockers = [t for t in profile.hard_blockers if _term_hit(t, blob)]
    blockers += [
        f"location: {t}"
        for t in profile.excluded_locations
        if _agency_hit(job.location, [t])
    ]

    if profile.agency_penalty and _agency_hit(job.company, profile.agencies):
        total -= profile.agency_penalty
        penalties.append(AGENCY_PENALTY)

    if not _location_ok(job, profile):
        total -= profile.location_penalty
        # Not a hard bar: the candidate allows remote and has several target
        # cities, so this goes in keyword_penalties (for display) but never
        # in blocker_hits (which forces a `blocked` verdict).
        penalties.append(LOCATION_PENALTY)

    return ScoredJob(
        job=job,
        keyword_score=total,
        keyword_hits=hits,
        keyword_penalties=penalties,
        blocker_hits=blockers,
    )


_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise_company(name: str) -> str:
    """Casefold, reduce runs of punctuation to single spaces, pad with spaces.

    The padding is what makes a plain `in` test a whole-word-run test, so
    "Owen Thomas" matches "Owen Thomas | B Corp(tm)" without "Data Idols"
    matching a company called "Data Idolsmith".
    """
    return f" {_NON_WORD.sub(' ', name.casefold()).strip()} "


def _agency_hit(company: str, agencies: list[str]) -> bool:
    """Whether this company is one of the configured agencies.

    Company only. Matching the whole posting instead would fire on any advert
    that names a recruiter in its text, including an employer explaining that
    it does not use them.
    """
    haystack = _normalise_company(company)
    return any(_normalise_company(a) in haystack for a in agencies)


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

    A plain substring test runs first (2.5.8): the pattern matches the term
    literally and case-sensitively, so `term in blob` is necessary for a hit,
    and most of 100-odd terms are absent from most postings. Measured
    2026-10-07 on 2,730 postings with a real profile: 4.0 ms to 0.27 ms
    per posting, `keyword_score` and `blocker_hits` identical on every one.
    """
    return term in blob and _term_pattern(term).search(blob) is not None


@functools.lru_cache(maxsize=1024)
def _term_pattern(term: str) -> re.Pattern[str]:
    """The whole-word pattern for `term`, compiled once per term."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")


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
