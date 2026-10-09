# ADR-0006: Only nationality and clearance bars block; other bars skip; work authorisation decides nothing

**Date**: 2026-10-01
**Status**: accepted
**Deciders**: maintainers

## Context

A `blocked` verdict is a one-way door. With `output.show_blocked: false` the
posting is dropped from the digest and written to `seen`, so it never resurfaces.
The local model abused the open-ended `other` bar kind (it filed "have not worked
in a financial services environment" under it) and extracted `work_auth` bars from
a free-text guess about which countries the candidate can already work in. Neither
was reliable enough to delete a role.

## Decision

In `decide`:
- A verified `nationality` or `clearance` bar means `blocked` (`_STRUCTURAL_BARS`),
  with the score capped at 20.
- Any `other` bar is a rule skip that quotes it, capped under `min_report_score`,
  and never a block.
- `work_auth` bars are extracted but ignored entirely (`_IGNORED_BARS`). Work
  authorisation is handled by configured `hard_blockers` (ADR-0010).
- A bar must name its own kind in its quote (`_BAR_WORDS`), or it is dropped.
- Wrong seniority is a skip, never a block.
- A nationality bar whose quote names a nationality the candidate holds
  (`profile.nationalities`) is dropped before any of this.

## Alternatives Considered

### Alternative 1: Any verified hard bar blocks
- **Pros**: the simplest rule, and the first design.
- **Cons**: `other` and `work_auth` produced wrong, permanent deletions.
- **Why not**: superseded within a day of shipping.

### Alternative 2: Let the model block on work authorisation
- **Pros**: it is what judge mode does.
- **Cons**: it needs the model to know the candidate's right to work from prose.
- **Why not**: that judgement was not reliable enough to hide a role on.

## Consequences

### Positive
- Two block categories only, both with vocabulary the code can verify.
- A model's invented bar can downgrade a role to a visible skip, not delete it.

### Negative
- Visa wording ("no sponsorship", "right to work in the UK") is not read by code.
  It relies on the user's `hard_blockers`. A role that requires an existing right
  to work is not blocked for a candidate who lacks it, unless the user lists the
  phrase. That omission is deliberate, because the right to work is a fact about
  the candidate that a posting cannot settle, but it is a gap for a user who
  expects it to be checked.
- `judge` mode still lets the model block on work authorisation, so the two modes
  disagree on policy.

### Risks
- A resolver false positive on a structural bar is a silent deletion. The only
  mitigation is "Hidden by your rules" (ADR-0009), and a hidden posting is listed
  once.
- The skip for an `other` bar is still a hide: the posting is capped below the
  digest.

## Evidence
- `rolescan/scoring/rules.py`: `_STRUCTURAL_BARS`, `_IGNORED_BARS` and their
  comments, `decide`.
- `rolescan/scoring/facts.py`: `_BAR_WORDS`, `_bar_supported`.
- `rolescan/scoring/llm.py`: `SYSTEM_FACTS` instructs the model on the four kinds
  and says work authorisation "is checked by configured keywords".
- `tests/test_rules.py`, `tests/test_bar_support.py`, `tests/test_nationalities.py`.
