# ADR-0007: BPSS alone is not a structural bar by default

**Date**: 2026-10-07
**Status**: accepted for the code resolver; the model path still disagrees (see Consequences)
**Deciders**: maintainers

## Context

`resolve_hard_bars` reads clearance wording in code. BPSS (Baseline Personnel
Security Standard) is the UK's baseline pre-employment screen. It has no
nationality or residency rule, and most non-citizens with the right to work pass
it. Counting it as a clearance bar made a small but real share of the clearance
bars the resolver found on a corpus of cached adverts BPSS-only: a role anyone
with the right to work could take would have been blocked, and a block hides a
role permanently.

## Decision

The code resolver does not treat BPSS as a bar. BPSS is deliberately not a pattern
in `_CLEARANCE_STATED`, and a sentence that names only BPSS is waived, even under
a "Security clearance:" label (`_clearance_waived`). A sentence that names BPSS
together with SC, DV, eDV, CTC, NPPV, UKSV, "developed vetting", "top secret" or
counter-terrorist checks is still a bar (`_ABOVE_BPSS`). A user who treats BPSS as
a wall lists `bpss` in `profile.hard_blockers`.

## Alternatives Considered

### Alternative 1: BPSS is a clearance bar
- **Pros**: consistent with the rest of the clearance vocabulary.
- **Cons**: it deletes roles an applicant with the right to work can take.
- **Why not**: the BPSS-only adverts were exactly the roles it would have hidden.

### Alternative 2: BPSS is a soft penalty
- **Pros**: visible, not deleted.
- **Cons**: there is no mechanism for a resolver-level soft bar.
- **Why not**: out of scope. The keyword `blockers` weight is the route.

## Consequences

### Positive
- Roles that merely require BPSS reach the model and the digest.
- Higher clearance wording in the same sentence still blocks.

### Negative: the decision is not applied everywhere
- **The model path still blocks on BPSS.** `_BAR_WORDS` for clearance contains
  `bpss`, so a model-named clearance bar quoting "BPSS clearance" survives
  `verify_facts` and `decide` returns `blocked`. The code resolver on the same
  advert finds no bar. The comment beside `_CLEARANCE_STATED` records this as
  intended ("a model-named BPSS bar still verifies"), but it means the default
  depends on whether the model happened to name the bar.
- **The README lists BPSS among the clearance words** a clearance bar's quote may
  contain, which reads as BPSS being a clearance.
- **A user's own `hard_blockers` or `blockers` can list `bpss`**, and then BPSS-only
  adverts are blocked by the keyword layer on the whole posting, before any model
  runs. That is a choice the config makes, and it overrides this default.

### Risks
- A user who copies an example config may not know BPSS is deliberately absent from
  the resolver.

## Evidence
- `rolescan/scoring/facts.py`: the comment above `_CLEARANCE_STATED`, `_BPSS`,
  `_ABOVE_BPSS`, `_clearance_waived`, `resolve_hard_bars`, `_BAR_WORDS`.
- `tests/test_hard_bars_stated.py`, `tests/test_bar_support.py`.
