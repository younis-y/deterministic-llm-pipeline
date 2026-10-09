# ADR-0004: Deterministic resolvers override or fill the model's facts

**Date**: 2026-10-01
**Status**: accepted
**Deciders**: maintainers

## Context

Even with quotes verified (ADR-0003), the local model left facts null or wrong
where the advert is explicit. On a corpus of cached adverts it left `field` null
on most postings, left `years_required` null on almost every advert that stated
"3+ years", named a nationality or clearance bar on only a minority of the
adverts that stated one, set `student_only` on internships that also accept
recent graduates, and left it unset on "penultimate year" ones. Each miss let a
wrong role reach the digest as APPLY, or hid a good one. Tightening the prompt
did not fix it, because the model is the unreliable part.

## Decision

After `verify_facts`, five pure, idempotent resolvers read the advert in code and
settle the fact, each quoting the advert's own words so the result would pass the
quote guard itself. `finish_facts` applies them in this order: `resolve_years`,
`resolve_hard_bars`, `resolve_student`, `resolve_level`, `resolve_field`.

| Resolver | What it does to the model's fact |
|---|---|
| `resolve_level` | A level word in the title overrides the model. Otherwise the model's level survives only with a verified quote of two words or more. Records `source` (title, text or none). |
| `resolve_field` | A field phrase in the title overrides the model. Otherwise the model's field survives with a quote of two words or more. Never returns `other` from a title. |
| `resolve_years` | Fills a null, and corrects the top of a range to its low end. A value the model stated and verified is otherwise kept. |
| `resolve_student` | Overrides both ways: clears `student_only` when the advert accepts recent graduates, and sets it from enrolment wording when there is no graduate clause. |
| `resolve_hard_bars` | Adds a nationality or clearance bar for a kind the model did not name. Never removes a verified bar. |

The resolvers err toward not firing. Negation, preference, "or equivalent", duty
wording, third parties and caps ("up to 2 years") all waive, because a rule that
hides a role for good should only read eligibility-shaped wording.

Since 2.5.8 the cache holds the model's raw facts and `finish_facts` runs on
every read, so a resolver fix reaches cached postings on the next run with no
cache bump and no model call (ADR-0008).

## Alternatives Considered

### Alternative 1: Better prompts and worked examples
- **Pros**: no code.
- **Cons**: worked examples (`llm.facts_examples_file`) help, but the model still
  returned null for stated years.
- **Why not**: kept as a complement, not sufficient alone.

### Alternative 2: Use a hosted model for extraction
- **Pros**: a hosted model extracted facts noticeably better than the local one
  before the resolvers existed.
- **Cons**: it needs a key and sends postings off the machine.
- **Why not**: it stays available as `llm.backend: anthropic`. The resolvers close
  most of the gap, so the keyless path stays worth using.

### Alternative 3: Re-run the resolvers on cache read instead of caching their output
- **Pros**: a resolver fix applies to every cached posting without a cache bump.
- **Cons**: the cache then has to hold the pre-resolve facts.
- **Why not**: it was not done at first, and was adopted in 2.5.8 once a cache
  bump and the re-scoring it forced proved too slow.

## Consequences

### Positive
- On a labelled corpus the years resolver read every years-driven "too senior"
  case and flagged none of the good roles, and the bar resolver found many more
  nationality and clearance bars than the keyword blockers had.
- The same model scores markedly better against a hand-labelled set with the
  resolvers on. The labelled set is not published.

### Negative
- A large regex surface with many waiver rules. Each new advert wording needs a
  test, and `tests/test_resolver_guards.py` pins every guard so that disabling
  one fails a test.
- Resolver precision is hand-tuned on the corpus it was built from. Wording a
  public default meets can still hit an edge case.

### Risks
- A resolver false positive is a silent deletion, because a block is a one-way
  door (ADR-0006). The mitigation is the "Hidden by your rules" list (ADR-0009).
- A title override means a model that is right loses to a misleading title
  ("Senior Analyst Internship" reads graduate-entry, because graduate and intern
  words beat senior by design).

## Evidence
- `rolescan/scoring/facts.py`: `resolve_years`, `resolve_student`,
  `resolve_hard_bars`, `resolve_level`, `resolve_field` and their docstrings.
- `rolescan/scoring/llm.py`: `finish_facts` (the order).
- `tests/test_years_required.py`, `tests/test_student_graduates.py`,
  `tests/test_hard_bars_stated.py`, `tests/test_level.py`, `tests/test_field.py`,
  `tests/test_resolver_precision.py`, `tests/test_resolver_families.py`,
  `tests/test_resolver_guards.py`.
