# ADR-0010: Hardness is an explicit list, and each configured list has its own match scope

**Date**: 2026-09-24 (`hard_blockers`), 2026-09-29 (`excluded_locations`)
**Status**: accepted
**Deciders**: maintainers

## Context

Keyword `blockers` started as one weight dictionary where a high weight meant
"hard". That encoded two independent things in one integer: what a term costs
(retuned whenever the prefilter is calibrated) and whether it is a wall (a fact
about the candidate that changes once a decade). It let a 60-point preference
block a role, and it made a 10-point fatal bar inexpressible. A whole-posting
match on a geography term was measured to delete a large number of real roles in
the cities the candidate wanted, whose adverts merely mentioned a foreign parent
company, while blocking none that were actually in the excluded country.
Substring matching deleted roles too: `crypto` hit "cryptographic", `head of` hit
"head office", `director` hit "directorate". Every blocker hit, weighted or hard,
can end in permanent deletion: a posting that falls under `min_keyword_score` is
prefiltered, recorded as seen and never shown again.

## Decision

- **`profile.hard_blockers` is the sole authority on hardness**: a list, not a
  weight threshold. A term in both lists keeps its weight; in `hard_blockers` alone
  it blocks and costs nothing; in `blockers` alone it only costs points. A
  `hard_blocker_score` weight threshold was designed and removed before release.
- **The match scope differs by list, on measured grounds:**
  - `hard_blockers` and `blockers`: the whole title and description (`Job.blob`),
    as whole words or phrases (`(?<!\w)term(?!\w)`), normalised at load.
  - `title_only_blockers`: a `blockers` term whose weight counts only in the title.
  - `excluded_locations`: the posting's location field only, as a hard bar.
  - `agencies`: the company name only, as a score penalty (`agency_penalty`),
    never a bar.
  - `keywords` (the positive terms): plain substrings, tripled in the title.
- **Load-time refusals**: an empty or one-character `hard_blockers` entry fails the
  config, because it would block everything. Two `blockers` keys that normalise to
  one term resolve to the heavier weight.
- **Preferences are not walls**: words such as `crypto`, `matlab`, seniority words
  and "phd required" belong in `blockers` as weights, not in `hard_blockers`.

## Alternatives Considered

### Alternative 1: A weight threshold for "hard"
- **Pros**: one dictionary.
- **Cons**: both shipped example configs would have put structural bars below the
  line.
- **Why not**: removed before it was ever released.

### Alternative 2: Match everything against the whole posting
- **Pros**: one mechanism.
- **Cons**: a geography term matched against the whole posting blocks every advert
  that mentions a foreign office, and none of the ones actually located there.
- **Why not**: geography belongs in the location field and employer identity in the
  company field.

### Alternative 3: Loose substring matching on weighted terms ("a points deduction is recoverable")
- **Pros**: catches inflections.
- **Cons**: the deduction is taken before `min_keyword_score`, so it deletes
  silently, as thoroughly as a hard bar.
- **Why not**: tried, then reversed.

## Consequences

### Positive
- A bar and a preference can be expressed independently, and geography no longer
  false-blocks.
- The load-time checks stop the two "block everything" configurations.

### Negative
- **Whole-posting `hard_blockers` are negation-blind, while the code resolvers are
  deliberately conservative** (ADR-0004: negation, preference, duty and third-party
  wording all waive). `hard_blockers: [security clearance]` matches "No security
  clearance required", and `emiratisation` matches an HR "Emiratisation Programme
  Manager" title that the resolver returns no bar for. A user who lists a term is
  opting into that. The hidden-by-rules section (ADR-0009) is how a term that
  matches the wrong thing shows up.
- **Weighted blockers are charged against the whole posting.** A `head of` weight
  fires on "you will report to the Head of Data", and a `military` weight fires on
  equal-opportunity boilerplate. `title_only_blockers` (2.5.7) is the remedy for
  words that name a level or sector in a title but are boilerplate in a body, and
  postings the weights push under the gate are now listed (ADR-0009).
- **`excluded_locations` matches a token run inside the location**, so a bare place
  name also bars a different place of the same name ("Washington, England" under a
  bar on `Washington`). Write the qualified form.
- **`keywords` match as substrings**, so `intern` scores on "international" and
  `ai` on "maintain". The direction of the error is extra model calls and a
  noisier gate, which is why the asymmetry with blockers was tolerated. It is now
  stated in the config reference rather than left to be discovered.

### Risks
- A configured term is the most powerful hide in the system, and it runs before the
  model. ADR-0009 lists term hides, but only once.

## Evidence
- `rolescan/config.py`: `ProfileConfig` (`blockers`, `hard_blockers`,
  `title_only_blockers`, `excluded_locations`, `agencies`), `_clean_blocker_terms`.
- `rolescan/scoring/keyword.py`: `score_keywords`, `_term_hit`, `_agency_hit`.
- `docs/config.md`: the "How terms are matched" section.
- `tests/test_scoring.py`, `tests/test_term_hit.py`, `tests/test_rules_config.py`.
