"""Merging near-duplicate postings within one run.

`pipeline.deduplicate` collapses exact repeats by `Job.uid` (company, title,
location). That misses the same role written two ways: on 2026-09-30 one
Sanderson "Data Engineer" arrived as "South East London, London" and again as
"London, UK", and the digest listed it twice.

This pass compares normalised company, normalised city and title similarity.
It never changes `Job.uid`, which is also the `seen` key: altering it would
make every stored posting look new.

Near-duplicates across runs are out of scope; catching them needs a store
change.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from rolescan.models import Job

#: Starting point, pinned by tests: "Data Engineer" vs "Senior Data Engineer"
#: scores 0.79 and must stay separate.
TITLE_SIMILARITY = 0.9

_LEGAL_SUFFIXES = frozenset(
    {
        "ltd",
        "limited",
        "plc",
        "llc",
        "llp",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "gmbh",
        "ag",
        "sa",
        "bv",
    }
)
_NOT_A_CITY = frozenset(
    {
        "uk",
        "united kingdom",
        "england",
        "scotland",
        "wales",
        "gb",
        "great britain",
        "united arab emirates",
        "uae",
        "remote",
        "hybrid",
    }
)
_COMPASS = re.compile(r"^(?:(?:north|south|east|west|central|greater)\s+)+")
_AREA = re.compile(r"\s+(?:metropolitan area|area)$")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.casefold())).strip()


def company_key(company: str) -> str:
    """Casefolded company with legal suffixes and any `| tagline` removed."""
    head = company.split("|", 1)[0]
    return " ".join(w for w in _clean(head).split() if w not in _LEGAL_SUFFIXES)


def city_key(location: str) -> str:
    """The first comma-separated part that names a place, not a country."""
    for part in location.split(","):
        token = _AREA.sub("", _COMPASS.sub("", _clean(part)))
        if token and token not in _NOT_A_CITY:
            return token
    return ""


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _clean(a), _clean(b)).ratio()


def merge_near_duplicates(jobs: list[Job]) -> list[Job]:
    """Drop near-duplicates, keeping the copy with the longest description.

    The result keeps the input order of the survivors, so nothing downstream
    sees a reordering it did not ask for. A posting with no company is never
    merged: aggregators leave it blank, and two blanks are not one employer.
    """
    buckets: dict[tuple[str, str], list[Job]] = {}
    keep: set[int] = set()
    for job in sorted(jobs, key=lambda j: (-len(j.description), j.uid, j.url)):
        company = company_key(job.company)
        if not company:
            keep.add(id(job))
            continue
        bucket = buckets.setdefault((company, city_key(job.location)), [])
        if any(
            title_similarity(k.title, job.title) >= TITLE_SIMILARITY for k in bucket
        ):
            continue
        bucket.append(job)
        keep.add(id(job))
    return [j for j in jobs if id(j) in keep]
