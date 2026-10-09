"""Is the model behaving as it did on the last runs? (2.6.0)

The quote guard and the resolvers correct what the model says, and a lot of
what reaches the digest is that correction: on one 40-posting sample they
changed the model's level on 30% of postings and its field on 20%. Nothing
watched those rates, so a model re-pulled under the same tag, an edited prompt
or a resolver bug moved them and the digest still read as normal.

`model_health` compares one run's rates (`LlmRun`, kept in `llm_runs`) with
the median of the runs before it, and names each rate that is more than twice
that median. It is written to stay quiet: it says nothing until three earlier
runs exist to compare with, nothing for a run of fewer than five postings, and
a median of zero needs five postings, not one, to count as twice. The digest
prints what it returns in its opening alarm block.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from fractions import Fraction
from typing import NamedTuple

from rolescan.store import LlmRun

__all__ = [
    "MIN_EARLIER_RUNS",
    "MIN_POSTINGS",
    "SPIKE_FACTOR",
    "TRAILING_RUNS",
    "HealthFlag",
    "model_health",
]

#: How many earlier runs set the baseline, newest first.
TRAILING_RUNS = 5

#: Earlier runs needed before anything is said: fewer is no baseline.
MIN_EARLIER_RUNS = 3

#: A rate is flagged above this multiple of the median.
SPIKE_FACTOR = 2

#: A run needs this many postings to be judged at all, and against a median of
#: zero this many of them must show the effect (any one posting is "twice"
#: nothing, which is not a change in the model).
MIN_POSTINGS = 5

#: The rates compared, as (`LlmRun` field, the words the digest uses).
_RATES: tuple[tuple[str, str], ...] = (
    ("quotes_rejected", "quotes rejected"),
    ("level_overridden", "level overridden"),
    ("field_overridden", "field overridden"),
    ("years_set", "years filled in"),
    ("years_cleared", "years cleared"),
    ("bars_added", "bars added"),
)


class HealthFlag(NamedTuple):
    """One rate over twice its median: the digest's words for it, how many of
    this run's `postings` it `affected`, and the median share of the earlier
    runs (0 to 1), taken over `runs` earlier runs (3 to 5)."""

    label: str
    affected: int
    postings: int
    median: float
    runs: int = TRAILING_RUNS


def model_health(run: LlmRun, earlier: Sequence[LlmRun]) -> list[HealthFlag]:
    """The rates of `run` that are more than twice the median of `earlier`.

    `earlier` is the runs before this one, NEWEST FIRST (what
    `Store.recent_llm_runs` returns): the first `TRAILING_RUNS` that finished
    at least one posting's facts set the median. An empty list means nothing
    to say: healthy, or too little history to tell.

    Rates are shares of a run's postings, compared exactly (as fractions), so
    a rate of exactly twice the median is not flagged whatever floating point
    would make of it.
    """
    if run.postings < MIN_POSTINGS:
        return []
    trailing = [r for r in earlier if r.postings > 0][:TRAILING_RUNS]
    if len(trailing) < MIN_EARLIER_RUNS:
        return []
    flags: list[HealthFlag] = []
    for attr, label in _RATES:
        count = int(getattr(run, attr))
        median = statistics.median(
            Fraction(int(getattr(r, attr)), r.postings) for r in trailing
        )
        if Fraction(count, run.postings) <= SPIKE_FACTOR * median:
            continue
        if median == 0 and count < MIN_POSTINGS:
            continue
        flags.append(
            HealthFlag(label, count, run.postings, float(median), len(trailing))
        )
    return flags
