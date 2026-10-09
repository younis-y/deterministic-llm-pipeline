# ADR-0022: Posting identity is company|title|location; near-duplicates merge only on equal title word sets

**Date**: 2026-09-01 (identity), 2026-09-30 (near-duplicate merge)
**Status**: accepted
**Deciders**: maintainers

## Context

`Job.uid` is also the `seen` key (ADR-0013), so any change to it makes every stored
posting look new. The same role arrives written two ways: one recruiter's "Data
Engineer" came as "South East London, London" and as "London, UK", and the digest
listed it twice. Merging too eagerly is worse, because the posting that loses the merge
is recorded as seen and hidden for good.

## Decision

- `Job.uid` is the hash of casefolded `company|title|location`, first 16 hex characters.
  It deliberately excludes the URL and the description, and the merge never alters it.
- `deduplicate` first collapses exact repeats by `uid`, keeping the longest description.
- `merge_near_duplicates` then runs, after the age filter, within one run only. It
  groups by normalised company and normalised city. Two titles are one role only when
  their cleaned word sets are equal (a plural "s" is stripped on words longer than three
  characters) and `SequenceMatcher` similarity reaches `TITLE_SIMILARITY` (0.9). It keeps
  the copy with the longest description. A posting with no company is never merged, since
  two blanks are not one employer. Legal suffixes and a `| tagline` are stripped from
  company names, and country words and compass prefixes are not cities.

## Alternatives Considered

### Alternative 1: Similarity alone (0.9 or more)
- **Pros**: simple.
- **Cons**: it merged "Data Engineer I" with "Data Engineer II", and "Senior Data
  Engineer" with "Senior Data Engineer II". The loser was then hidden for good.
- **Why not**: the word-set rule was added on top.

### Alternative 2: Key identity on the URL
- **Pros**: exact.
- **Cons**: the same role on two boards is two postings, and a repost with a new
  requisition id reads as new.
- **Why not**: the URL is excluded by design.

## Consequences

### Positive
- One listing per role per run, with a conservative near-duplicate rule.

### Negative
- Across runs, a role re-listed under a slightly different title or location is a new
  `uid` and appears again. The same title, company and location reposted months later is
  the same `uid` and never appears (ADR-0013).
- "Data Engineer" and "Data Engineer II" intentionally stay separate, so a duplicate pair
  can still show twice.

### Risks
- `city_key` takes the first non-country part of the location, so a district-first
  location ("Canary Wharf, London") and diacritics produce different cities.
- Another package that reuses `company_key`, `city_key` or `same_title` depends on their
  behaviour, so a change here changes what it emits.

## Evidence
- `rolescan/models.py`: `Job.uid`, `Job.content_hash`.
- `rolescan/dedup.py`: the module docstring, `merge_near_duplicates`, `TITLE_SIMILARITY`.
- `rolescan/pipeline.py`: `deduplicate`, and the order of the age filter before the merge
  in `run_scan`.
- `tests/test_dedup.py`.
