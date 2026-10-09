# ADR-0014: Silence must never be ambiguous: a failing component must be visible in the digest, not only in a log

**Date**: 2026-09-23, accumulated since
**Status**: accepted; the project's defining design principle, enforced component by component
**Deciders**: maintainers

## Context

Every serious defect this project has had was a component that stopped working
without raising: a hardcoded first page, a concurrency default nobody chose, an
empty query list, options nested one level too deep. Several were found in a single
day. Later instances followed the same shape: a blocking list left silently empty by
a migration, Workday paging capped by the total on page one, a scan started before
the network was up, a reader that emitted navigation links as postings. The person
who has to be told is the reader of the digest, and nobody reads a log.

## Decision

1. **Raise or surface, never swallow.** A source that detects its own breakage raises.
   `pipeline._fetch_one` records it as a failed source, and the digest names it.
   One dead board must not take down the run, so the constraint is "must never fail
   the run", not "must never raise". Zero results is not breakage.
2. **Coverage memory.** `source_counts` records what each source returned. A source
   that returned nothing, did not raise, and returned rows in its recent runs is
   named under "Sources that went quiet". A source that never worked is unconfigured,
   not quiet. Since 2.5.8 "Sources that shrank" and "Sources that were cut short"
   say the same for a board that returned far fewer rows than before, or fewer than
   it holds. Each alarm is shown even when it is the only thing to report, in both
   the Markdown and the HTML digest. Since 2.6.0 the alarms open the digest, ahead
   of the roles and of the hidden list, so a long list below cannot push them past
   the point where an email client clips the message (ADR-0009). Since 2.6.0 the
   same holds for the model: `llm_runs` records, per scan, how many postings the
   quote guard or a resolver changed, and a rate more than twice the median of the
   last three to five runs (as many as exist) of the same model is a "Model health"
   line in that block. A model
   re-pulled under the same tag, or an edited prompt, moves those rates without
   raising anything.
3. **Preflight.** The configured judge, and the enricher, are probed before any
   fetch. An unusable backend is a digest banner, not a keyword-only digest that
   looks normal. A preflight must itself never raise.
4. **Record only what was assessed** (ADR-0013), so a broken judge cannot bury roles.
5. **The exit status says it too.** `rolescan scan` exits 0 ok; 1 email failed, the
   day's backup failed its check, or the store is from a newer rolescan; 2 config
   error; 3 LLM scoring failed as a whole; 4 another run holds the store's lock. The
   digest is written and sent first.
6. **Statuses distinguish `EMPTY` from `UNKNOWN`.** `discover` only reports an empty
   board as healthy for APIs verified to answer 404 for an unknown slug. An API that
   answers 200 with zero rows for a nonsense slug is `UNKNOWN`, because a zero count
   there proves nothing.

## Alternatives Considered

### Alternative 1: Fail the whole run on any source failure
- **Pros**: loud.
- **Cons**: forty sources are forty chances for a 404, and one dead board must not
  take down the digest.
- **Why not**: isolation plus a named line in the digest is the compromise.

### Alternative 2: Log warnings and rely on an operator
- **Pros**: simple.
- **Cons**: the reader reads the email, and a scheduler's log is read by nobody.
- **Why not**: tried first, and it produced the defect class.

## Consequences

### Positive
- Many instances are now caught on the first run after they appear.
- The principle is testable: each defect arrives with a regression test.

### Negative
- Enforcement is per component. No test enumerates every source kind and asserts that
  it can fail visibly.
- Email delivery shares one SMTP path with the digest, so a failed send is reported
  by the exit status and a line on the terminal, not by the digest it failed to
  deliver.

### Risks
- The prefilter is the one place that is silent by design: a posting under
  `min_keyword_score` is counted in the stats line, recorded and not listed. Rule and
  term hides are listed instead (ADR-0009).

## Evidence
- `rolescan/pipeline.py`: `_fetch_one`, `_check_coverage`, `_preflight`,
  `ScanResult.llm_failure`.
- `rolescan/digest.py`: the failure, quiet, shrunk and cut-short sections.
- `rolescan/cli.py`: `_SCAN_EXIT_STATUS`.
- `rolescan/sources/base.py`: `ProbeStatus`.
- `tests/test_quiet_alarm.py`, `tests/test_coverage_alarms.py`,
  `tests/test_llm_failure.py`, `tests/test_probe.py`, `tests/test_e2e_cli.py`.
