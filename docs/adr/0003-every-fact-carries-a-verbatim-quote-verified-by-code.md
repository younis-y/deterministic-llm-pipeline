# ADR-0003: Every fact carries a verbatim quote, and code verifies it

**Date**: 2026-09-30
**Status**: accepted
**Deciders**: maintainers

## Context

A model asked what a role requires will invent a plausible answer when the
posting is silent, and nothing downstream can tell a fabricated "5 years
minimum" from a real one. The local model also quoted field labels from the
prompt template ("Title:"), quoted the title's tail spliced onto the
description's head, and named a nationality or clearance bar on a verbatim
sentence about "business policies and procedure". A verbatim quote proves that
the text exists, not that it says what the fact claims.

## Decision

Every extracted fact (level, years, student-only, graduation year, field, each
hard bar) must carry the exact text it was read from. `verify_facts` is the one
place that checks. The quote must be a substring of the normalised title or of
the normalised description, checked as two separate haystacks and never joined,
with typographic look-alikes folded (curly quotes, dashes, non-breaking space).
A failing fact is downgraded to "not stated", and a failing bar is dropped
outright. Two further guards apply: a graduation year must appear as a whole
number inside its own quote, and a nationality or clearance bar's quote must
name its kind (`_BAR_WORDS`). Quotes are capped at 200 characters.

## Alternatives Considered

### Alternative 1: Trust the model's value without a quote
- **Pros**: a simpler schema.
- **Cons**: fabricated values decide verdicts invisibly.
- **Why not**: the point of ADR-0002 is that an unverifiable fact must never
  decide a verdict.

### Alternative 2: Fuzzy or semantic quote matching
- **Pros**: tolerates paraphrase.
- **Cons**: a fuzzy match can verify invented text.
- **Why not**: matching may only get stricter. The fold is characters only, so it
  cannot make invented text verify.

## Consequences

### Positive
- Every skip in the digest quotes the advert, so the reader can check it.
- A hallucinated bar cannot block. One block in a labelled evaluation came from a
  verbatim but irrelevant quote, and the bar-word check now rejects that shape.

### Negative
- Verbatim-only means a correct fact with a paraphrased quote is lost. The
  resolvers (ADR-0004) exist partly to recover those cases, each with its own
  verbatim quote.
- Worked examples in `llm.facts_examples_file` must pass the same guard when the
  config loads, so a bad example stops the run before a single posting is scored.

### Risks
- The guard proves existence, not relevance. A `student_only` quote can be
  verbatim and incomplete ("in the final year of a Bachelor's" while the advert
  also welcomes graduates); `resolve_student` exists because of this.
- `verify_facts` does not check that a `value` agrees with its `quote`, except for
  the graduation year. A mis-valued level with a real quote survives, and
  `resolve_level` then overrides it from the title.

## Evidence
- `rolescan/scoring/facts.py`: the module docstring (the two-haystack rule and
  the fold), `verify_facts`, `_year_in_quote`, `_bar_supported`, `_BAR_WORDS`.
- `rolescan/scoring/llm.py`: `USER_FACTS` ends by telling the model never to quote
  field labels, because the label would verify and say nothing.
- `tests/test_facts.py`, `tests/test_bar_support.py`, `tests/test_graduation_year.py`,
  `tests/test_facts_examples.py`.
