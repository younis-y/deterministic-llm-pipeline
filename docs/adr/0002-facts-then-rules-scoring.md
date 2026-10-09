# ADR-0002: The model extracts facts; code applies the rules

**Date**: 2026-09-30
**Status**: accepted
**Deciders**: maintainers

## Context

In the single-call "judge" mode the model was asked to apply several rules at
once: skip if the role is senior, unless the years match, unless the field is
out of scope. On a small hand-labelled set the local model gave a noticeable
share of bad recommendations, and most of those broke a rule the profile stated
outright (an advert asking for 3+ years, for example). A model blends rules into
one fuzzy judgement, so a rule that should be absolute (a nationality bar) ends
up negotiable against a preference (a fit score). The prompt also carried
context that had nothing to do with fit.

## Decision

`llm.mode: facts` (the default) asks the model only for quoted facts (level,
years required, student-only, graduation year, field, hard bars) plus a
skills-and-domain `fit_score`. `rolescan.scoring.rules.decide` is pure (no I/O,
no logging) and turns the verified facts into the verdict in a fixed order: hard
bar, graduation year, student-only, level, years, field, then the fit-score
bands (65 and above applies, 40 and above is worth considering). A rule skip
caps the score just under `min_report_score`, and a hard bar caps it at 20, so a
rule-skipped posting can never outrank one that reached the digest on fit. A
fact the advert does not state never fires a rule.

## Alternatives Considered

### Alternative 1: Keep the judge and tighten the prompt
- **Pros**: one call, no new code.
- **Cons**: a local model does not apply nested rules reliably, and Ollama takes
  the output schema as a grammar and ignores field descriptions, so a rule that
  lives only in a schema description never reaches it.
- **Why not**: the failure was measured, and the fix cannot live in the prompt.

### Alternative 2: Keep the judge and add a deterministic filter on its verdict
- **Pros**: cheap.
- **Cons**: the filter would have no verified facts to act on.
- **Why not**: it needs the facts anyway.

## Consequences

### Positive
- Every skip names its rule and quotes the advert (`FitVerdict.rule`), which
  makes a wrong skip auditable (ADR-0009).
- The cache holds the model's facts, not a verdict, so a change to `rules` or to
  `min_report_score` re-decides every cached posting for free (ADR-0008).
- A rule change is a config or code edit, never a prompt edit.

### Negative
- Facts have to be verified and resolved in code (ADR-0003, ADR-0004). The
  design trades prompt fragility for a large deterministic resolver surface.
- `judge` mode still exists with a different policy: there the model itself can
  still block on work authorisation, which facts mode ignores (ADR-0006).

### Risks
- The model's `fit_score` is the one fuzzy input left. It must not be asked to
  weigh level or eligibility; `SYSTEM_FACTS` tells it to ignore both, and the
  prompt is the only instruction both backends read.
- The order in `decide` is load-bearing. `RULE_ORDER` drives the digest's
  grouping, so a new rule has to be added in both places.

## Evidence
- `rolescan/scoring/rules.py`: the module docstring (why rules live in code),
  `decide`, `RULE_ORDER`, `APPLY_AT`, `CONSIDER_AT`, `BLOCKED_CAP`.
- `rolescan/scoring/llm.py`: `SYSTEM_FACTS`, and `FitScorer`'s facts path
  (`finish_facts`, then `decide`, on every cache hit).
- `tests/test_rules.py`, `tests/test_rules_config.py`, `tests/test_facts_mode.py`.
