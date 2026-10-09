# ADR-0008: The facts cache key carries a hand-bumped payload version, the backend, the model and a prompt fingerprint

**Date**: 2026-09-30 (a hand-bumped version); revised in 2.5.8 (raw facts and a fingerprint)
**Status**: accepted; a test now guards the hand-bumped part
**Deciders**: maintainers

## Context

The verdict cache saves model calls (`llm.cache_days`, 30 by default). A cache is
only as good as its key: anything that changes what the model is asked, or what a
stored row means, makes old rows wrong. In facts mode the first design stored the
verified and resolved facts. Because the resolvers (ADR-0004) ran before caching,
every resolver or word-table change made old rows replay a stale answer: a row from
before the years resolver existed could replay a null that the user's years rule
never saw, and a row from before the bar resolver could replay an APPLY for a role
the candidate could not hold. Facts extracted by one model must also never be
served after the user switches models, which is the switch an evaluation of two
backends exists to inform. And editing the system prompt or the candidate summary
replayed facts made for the old prompt until the cache expired.

## Decision

Facts-mode key: `<content_hash>:<FACTS_KEY_VERSION>:<backend>:<model>:<fingerprint>`.

- The row holds the model's **raw** facts. `finish_facts` (verification, then the
  five resolvers) runs on every read, so a resolver or word-table change reaches
  every cached posting on the next run with no model call and no bump.
- `prompt_fingerprint` is 12 hex characters of a hash of everything that decides
  the model's answer: the rendered system prompt (instructions, candidate summary,
  worked examples), the user template, the facts JSON schema, the served model's
  digest, `llm.num_ctx`, `llm.description_chars` and `llm.temperature`. Editing any
  of them moves the key by itself.
- `FACTS_KEY_VERSION` (`facts-v14` at 2.5.10) is bumped by hand, and only when what
  a stored row **holds** changes meaning: a new field in the cached payload, or a
  change to the raw-fact shape. v2 moved the cache from verdicts to facts, v7 added
  the graduation year and the level source, and v14 moved it from resolved facts to
  raw facts. Every earlier bump was a resolver or table change, which no longer
  needs one.
- Judge-mode keys stay the bare content hash.
- `tests/test_facts_key_pin.py` pins a hash of `SYSTEM_FACTS`, the facts JSON schema
  and the resolver pattern tables. It fails with "bump FACTS_KEY_VERSION" when any
  of them changes while `FACTS_KEY_VERSION` stays the same, so a change to what the
  extraction means cannot ship by accident. The fingerprint moves the key on its own
  for the first two; the pin makes the decision conscious, and covers the tables.

## Alternatives Considered

### Alternative 1: Cache the resolved facts and bump on every resolver change
- **Pros**: nothing to re-resolve on read.
- **Cons**: a forgotten bump replays stale facts for up to `cache_days`, and a bump
  re-scores the whole backlog on a slow local model.
- **Why not**: this was the first design, and both failure modes happened.

### Alternative 2: Hash the resolver source into the key
- **Pros**: automatic.
- **Cons**: every refactor invalidates the cache.
- **Why not**: with the resolvers run on read it is also unnecessary.

### Alternative 3: Key on the content hash only and let `cache_days` expire rows
- **Pros**: the simplest key.
- **Cons**: stale facts are served for up to 30 days after a fix.
- **Why not**: that is what the first version effectively was.

## Consequences

### Positive
- A resolver fix is free: it applies to the whole cache on the next run.
- A model switch, a prompt edit, an examples edit and a summary edit are all safe by
  construction.

### Negative
- A change to the cached payload shape still costs one re-extraction per posting on
  next sight.
- The fingerprint hashes the whole prompt, so a one-word edit to the summary asks
  the model again for every posting.
- If the served model's digest cannot be read, the facts cache is skipped for that
  run, neither read nor written, and the digest says so.

### Risks
- The pin test cannot know whether a table change alters what a stored row means. It
  forces the question, and the person bumping decides.
- The pin hashes the facts schema, which pydantic generates. The hash was identical
  across the pydantic 2.11 to 2.13 releases tried, but a future pydantic release
  that reshapes the schema would move it without any change here; bump the pin, not
  the version, if the facts shape is unchanged.

## Evidence
- `rolescan/scoring/llm.py`: `FACTS_KEY_VERSION`, `prompt_fingerprint`, `cache_key`
  (its docstring carries the version history), `finish_facts`,
  `FitScorer._facts_cache_key`.
- `tests/test_facts_cache.py`, `tests/test_facts_mode.py` (the key's shape),
  `tests/test_facts_key_pin.py` (the guard).
