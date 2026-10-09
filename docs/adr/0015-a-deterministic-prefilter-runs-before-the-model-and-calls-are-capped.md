# ADR-0015: A deterministic keyword prefilter runs before any model call, and model calls have a hard ceiling

**Date**: 2026-09-01
**Status**: accepted
**Deciders**: maintainers

## Context

Cost and time are set by how many postings reach the model. On a recorded run the
keyword stage discarded the great majority of the unique postings it saw (the README's
"Measured quality" has the figures), and a local model is the slow stage of a scan. A
hosted backend would also bill per call. At realistic volumes a scan is bound by how
many new postings the sources supply, not by the model.

## Decision

The pipeline order is fixed: fetch (all sources concurrently), age filter, exact and
near-duplicate merge, keyword score, drop already seen or dismissed, prefilter at
`min_keyword_score`, model, record, rank. Keywords count tripled in the title,
blockers subtract their weight, a location mismatch costs `location_penalty`, and an
agency listing costs `agency_penalty`. Postings under the gate are never judged and are
recorded as seen. `llm.max_calls_per_run` is a hard ceiling. Postings past it are
deferred to the next run and keep their keyword score meanwhile. An enricher, if one is
configured, runs only for postings that clear `min_report_score`.

## Alternatives Considered

### Alternative 1: Send everything to the model
- **Pros**: no prefilter misses.
- **Cons**: cost and wall-clock time.
- **Why not**: the reason the design exists.

### Alternative 2: An embedding or classifier prefilter
- **Pros**: better recall than keywords.
- **Cons**: another model, and another class of miss.
- **Why not**: not tried. The keyword stage is reproducible and free, and it works with
  no key (`--no-llm`).

## Consequences

### Positive
- Cheap, reproducible and free to re-run, and the tool is useful with no key.

### Negative
- **A prefilter miss is invisible and permanent**, because the posting is recorded as
  seen (ADR-0013). A role whose title names an allowed field but shares none of the
  user's keywords is rejected, and a title that matches a keyword by accident
  ("Internal Audit Analyst" scoring on `intern`) passes. The gate is a cliff, and it
  was set by hand from a small sample.
- Since 2.5.7 a reject within `profile.hidden_gate_margin` of the gate is listed under
  "Hidden by your rules", which turns the cliff into something a reader can see. A
  reject further under is only counted in the stats line.
- `keywords` match as substrings (ADR-0010), so the gate is noisier than the blockers.
- A new source scales the prefilter's input but not its recall. Every board added
  multiplies the postings that meet the gate, and at scale the gate, not the model,
  decides what the reader sees.

### Risks
- A low `max_calls_per_run` bounds a run but defers the rest, and a posting that stays
  past the ceiling for many runs is never judged.
- The README's measured discard rate comes from one dated run with one profile, and
  nothing re-measures it.

## Evidence
- `rolescan/pipeline.py`: `run_scan` (the stage order), `_prefilter`, `_judge`.
- `rolescan/scoring/keyword.py`: the module docstring and `score_keywords`.
- `rolescan/config.py`: `ProfileConfig.min_keyword_score`, `LLMConfig.max_calls_per_run`.
- `README.md`: "The two decisions worth reading the code for" and "Measured quality".
- `tests/test_scoring.py`, `tests/test_pipeline.py`, `tests/test_deferred.py`,
  `tests/test_hidden_rejects.py`.
