# ADR-0011: An employer board's posted date is not a freshness signal

**Date**: 2026-09-28
**Status**: accepted; every employer-board source declares it since 2.6.0 (see Consequences)
**Deciders**: maintainers

## Context

`profile.max_age_days` (default 90) drops stale postings. That is right on an
aggregator, where `posted` is when the advert went up. On an employer's own
applicant-tracking board, `posted` is when the requisition was opened, and the
listing's presence on the board is the freshness signal. Measured on live boards,
a large share of an active employer's openings were older than 90 days, some by
years. A uniform cutoff discarded hundreds of live postings and replaced them in
the digest with recruitment-agency reposts, which an aggregator always dates as
yesterday: strictly worse than having no filter.

## Decision

`Source.dates_are_freshness` is a class variable. Every reader of an employer's own
board (`greenhouse`, `lever`, `ashby`, `workable`, `smartrecruiters`, and since
2.6.0 `structured` and `workday`) sets it to False, and the age filter skips their
rows. Only an aggregator (`adzuna`, `reed`, `jooble`, `workable_search`) keeps the
default of True. An unknown source kind is aged out, so the safe default is to
filter. A posting with no date at all is kept, because unknown
is not old. The age filter runs before near-duplicate merging, so a stale
longest-description survivor cannot take its fresh twin down with it (ADR-0022).

## Alternatives Considered

### Alternative 1: One cutoff for every source
- **Pros**: trivial.
- **Cons**: deleted live requisitions at scale, measured.
- **Why not**: it shipped briefly and was reverted.

### Alternative 2: A per-source `max_age_days`
- **Pros**: flexible.
- **Cons**: a config burden, and a source already knows what its own date means.
- **Why not**: the source declares its own semantics instead.

## Consequences

### Positive
- Evergreen employer requisitions stay visible, and the digest favours direct
  employers over reposts.

### Negative
- A requisition that is filled but not yet taken down stays in the digest. The only
  freshness test for an employer row is that it is still on the board.
- **The rule missed two readers for five releases.** Until 2.6.0, `structured`
  (schema.org `datePosted`) and `workday` (`startDate`, read from each posting's
  detail; the listing carries only "Posted Today") read employer boards but
  inherited the base default of `dates_are_freshness = True`, so an evergreen
  requisition on those platforms was aged out at `max_age_days`, exactly as the
  uniform cutoff did before this decision. Both now declare False. A posting that
  was aged out was never scored or recorded as seen, so the first scan after the
  upgrade reports those live requisitions as new: expect one larger digest from a
  `structured` or `workday` source.

### Risks
- A new reader for an employer board that forgets to set the flag inherits True and
  starts dropping live postings. `tests/test_source_registry.py` lists every
  registered source and the value it must declare, so a new source fails there until
  its author states one.
- An aggregator that sets the flag False by mistake lets stale reposts through.

## Evidence
- `rolescan/sources/base.py`: `Source.dates_are_freshness` and its docstring.
- `rolescan/pipeline.py`: `_ages_meaningfully`, `_drop_stale`, and the order of the
  age filter before `merge_near_duplicates` in `run_scan`.
- `tests/test_pipeline.py`, and `tests/test_source_registry.py` for the per-source values.
