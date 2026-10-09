# ADR-0013: A posting is recorded as seen once it has been assessed, and `seen` never expires

**Date**: 2026-09-01 (design), 2026-09-24 (the backend-broke gate)
**Status**: accepted; the permanence is implied by the code more than argued in it
**Deciders**: maintainers

## Context

A digest must not repeat roles: a role is reported once and never again. Identity is
`Job.uid`, the hash of `company|title|location` (casefolded), which deliberately
excludes the URL so that the same role reposted with a new requisition id does not
read as new (ADR-0022). A scan could, however, record a posting it never really
judged. A run whose Ollama or hosted backend could not start once recorded dozens of
unjudged postings that could then never surface again.

## Decision

- `seen` is keyed on `uid`. `Store.filter_new` drops any posting already in it before
  scoring. Rows in `seen` are never removed or expired; `prune` trims caches only.
- A posting is written to `seen` when it was assessed. Prefilter rejects always are
  (assessed on keywords, rejected on merit). Judged postings are when the judge
  worked, or when scoring was switched off on purpose. They are not when the
  intended judge was supposed to work and did not (`llm_unusable` or any scoring
  error), so those stay reachable on the next run (`assessed`).
- A posting the run did not get to (over the model budget, over the digest cap, no
  description text yet) is deferred, not recorded, so it comes round again.
- The write happens after the digest is on disk (`record_scan`), so a crash between
  the two loses nothing.
- `--dry` records nothing.
- A `dismissed` mark also matches on URL, because the same role arrives under a new
  `uid` from the next board.
- `rolescan unsee URL` removes a posting from `seen` so the next scan reports it
  again. It is the way back from a wrong hide.

## Alternatives Considered

### Alternative 1: Expire `seen` after N days
- **Pros**: recurring roles, such as an annual programme with a fixed title, return.
- **Cons**: repeats in every digest.
- **Why not**: never chosen, and no code path exists for it.

### Alternative 2: Key on the URL
- **Pros**: exact.
- **Cons**: the same role on two boards reads as two, and a repost reads as new.
- **Why not**: identity is company, title and location, plus a near-duplicate merge
  (ADR-0022).

## Consequences

### Positive
- Digests only ever carry new roles, and `rolescan stats` counts a monotonic set.
- A broken backend cannot bury a day's postings.

### Negative
- **Every hide is permanent unless undone.** A prefilter reject, a rule skip, a block
  and a below-gate score are all recorded, and a rule or keyword fix does not
  resurrect them by itself. Fixing a resolver bug does not recover the roles it
  wrongly hid last week; `unsee` does, one posting at a time.
- **Annual re-posts are invisible.** A programme re-opened with the same company,
  title and location has the same `uid` and never resurfaces. Titles that carry the
  year ("2027 Graduate Programme") are protected by accident. For a search that
  targets recurring programmes this is the highest-value implied risk in the system.
- `seen` does not record the facts or the basis of the decision, only the company,
  title, location, URL, source, score, verdict, reason and first and last seen. A
  hidden posting can only be re-examined if its advert is fetched again.

### Risks
- Deleting `seen.db` is the only bulk reset, and it also deletes the `applications`
  table (shortlist, applied, dismissed). The README says so.

## Evidence
- `rolescan/store.py`: the module docstring, `Store.filter_new`, `Store.unsee`,
  `Store.record_all`, `Store.prune_all`.
- `rolescan/models.py`: `Job.uid`, `Job.content_hash`.
- `rolescan/pipeline.py`: `assessed`, `record_scan`, `_drop_already_handled`.
- `tests/test_store_seen.py`, `tests/test_store.py`, `tests/test_pipeline.py`,
  `tests/test_deferred.py`.
