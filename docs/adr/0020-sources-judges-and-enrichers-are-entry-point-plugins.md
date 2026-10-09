# ADR-0020: Sources, judges and enrichers load through entry-point plugins; the core has no command plugin group

**Date**: 2026-09-01 (sources, judges), 2026-09-30 (enrichers)
**Status**: accepted
**Deciders**: maintainers

## Context

The core is public and meant to stay small. Anything that does not belong in it (a
source for a site that needs a login, a model backend that is not Claude or Ollama,
extra work done per posting) must be able to attach without being edited into it. The
scoring backend also had to be swappable, a hosted Claude or a local Ollama model,
without touching pipeline code.

## Decision

Three `importlib.metadata` entry-point groups:

- `rolescan.sources` holds source kinds. Each declares `dates_are_freshness` and
  `ambiguous_when_empty`.
- `rolescan.judges` holds model backends. Each declares `needs_api_key`,
  `cheap_triage`, `preflight()` and `facts()`.
- `rolescan.enrichers` holds extra work done after the gate, only for `apply` and
  `consider` postings at or above `min_report_score`. A failing enricher keeps the plain
  verdict and never drops a posting. At most one is used, named by `llm.enricher`, and
  the core ships none.

A caller adds fields by returning a subclass of `FitVerdict`, which pydantic validates
wherever the parent is annotated. `llm.extra_prompt` is the one free-text seam into the
judge-mode prompt.

There is deliberately **no entry-point group for CLI commands**. A package that wants its
own commands ships its own console script rather than a hook that would need a public
edit.

## Alternatives Considered

### Alternative 1: Fork the core
- **Pros**: no plugin API to maintain.
- **Cons**: drift, and the public repository ends up carrying code that does not belong.
- **Why not**: it defeats the point of a small public core.

### Alternative 2: Fixed extra fields on `FitVerdict`
- **Pros**: the simplest option, and the state of an earlier release.
- **Cons**: it teaches the public prompt a vocabulary it has no business knowing.
- **Why not**: removed, and replaced with the enricher seam plus `llm.extra_prompt`.
  `test_the_public_prompt_names_no_documents` pins that the prompts name no documents.

## Consequences

### Positive
- A new ATS reader or model backend is a package, not a patch.
- The plugin contracts are tested with fake judges and fake enrichers.

### Negative
- A plugin API is a compatibility surface. Each core release changes it only by adding
  fields and keyword parameters with defaults, and the changelog lists what was added.
- A misspelled `llm.enricher` is a pre-scan warning (`enricher_unusable`), not a failure,
  and the digest gets its own one-line note.

### Risks
- Entry points are discovered when `rolescan.sources` is imported. A plugin that fails to
  import can disable its own group, and an unknown source kind fails per source (an error
  line), not the run.
- The core treats an unregistered source kind as an aggregator when filtering by age
  (`_ages_meaningfully`), so a typo in a kind is aged out.

## Evidence
- `pyproject.toml`: the `[project.entry-points."rolescan.judges"]` and
  `[project.entry-points."rolescan.sources"]` tables.
- `rolescan/sources/base.py`: `Source`, `register`, `load_plugins`;
  `rolescan/scoring/judges.py`: `Judge`, `register`, `available_judges`;
  `rolescan/scoring/enrich.py`: `Enricher`, `get_enricher`.
- `tests/test_enrich.py`, `tests/test_judges.py`, `tests/test_scoring.py`,
  `tests/test_sources.py`.
