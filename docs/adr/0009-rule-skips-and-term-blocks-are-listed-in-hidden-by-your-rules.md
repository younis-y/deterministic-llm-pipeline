# ADR-0009: Nothing a rule or a blocking term hides is hidden without a trace ("Hidden by your rules")

**Date**: 2026-10-07
**Status**: accepted; capped at `output.hidden_max` since 2.6.0 (see Decision)
**Deciders**: maintainers

## Context

A rule skip is capped below `min_report_score`, and a block is dropped by
`show_blocked: false`. On the day this was measured, a large share of the postings
a model had scored were hidden by `profile.rules` with no trace. A mis-read advert
or a resolver bug that skipped a good role was invisible, and a hidden posting is
recorded in `seen` and never returns (ADR-0013). A configured `hard_blockers`
term does the same, usually at the prefilter, before any model sees the posting.
For a job search the worst outcome is a wrongly hidden role.

## Decision

The digest (Markdown and HTML) renders a "Hidden by your rules" section whenever
something was hidden, one line each (company, linked title, location, reason),
grouped in `decide` order (`RULE_ORDER`: hard bar, graduation year, student-only,
level, years, field), then a group for configured terms (`hard_blockers`,
`excluded_locations`), then the weighted `blockers` group and a "just under your
keyword gate" group. `ScanResult.rule_hidden` holds them. Scope:

- judged postings whose `fit.rule` is set, or that the model blocked, or that carry
  `blocker_hits`, and are not in the digest;
- prefilter rejects when a blocking term's `blockers` weight is what put them under
  `min_keyword_score`, and those within `profile.hidden_gate_margin` of the gate
  (default 10, `0` lists none), whatever put them there. A posting that fails the
  gate by more is low relevance, not a rule hide;
- `excluded_locations` carries no weight, so an irrelevant posting in an excluded
  location is never listed, while a relevant one clears the gate, is judged and is
  listed from there;
- each posting appears once, and the rule that fired wins over a term.

The stats line says "N hidden by your rules (listed below)". `FitVerdict.rule` is
kept out of the model's JSON schema and cleared if a judge echoes it, so a model
cannot name a rule.

**Length (2.6.0).** The section first shipped with no cap, at about a third of a
kilobyte of HTML a line, and sat before the alarms. A long list pushed the alarms
past the point where Gmail clips a message (about 102 KB), which is the one place
they must not be. Now:

- `output.hidden_max` (default 60, `0` lists none) caps the lines of the section,
  counted across every rule. The cut keeps the highest-scoring postings: a hide with
  a high score is the likeliest wrong one, which is what the section is for.
- The section opens, when capped, with a count per rule, and each group heading
  carries its count ("Level (2 of 40 listed)"). A rule whose rows were all cut is in
  the counts and has no group.
- The whole list is written to `<digest stem>-hidden.md` beside the digest, only when
  the cap cut it. The digest and the email name that file, and the stats line reads
  "N hidden by your rules (M listed below, all N in the file)". The file is written
  before the digest, so a digest never names a file that is not there.
- The alarms moved ahead of the roles and of this section (see ADR-0014), and the
  HTML part is held under a size budget by shrinking role cards, never by cutting
  the alarms. A caller of `render_markdown` or `render_html` that passes no
  `hidden_max` gets no cap.
- In the email the roles spend the budget before this section does. The section is
  laid out last, with whatever the roles leave: its heading, every rule's count and
  where the rest is, always, and as many rows as fit, highest scores first. So
  raising `hidden_max` can never push a role out of the email. When the email lists
  fewer rows than the cap allows, the section says so; if the cap did not cut the
  list there is no file, and it points to the Markdown digest, which lists up to
  `hidden_max` rows.

## Alternatives Considered

### Alternative 1: Show hidden roles inline with a badge
- **Pros**: nothing leaves the main list.
- **Cons**: on a heavy day the hidden roles would bury the real list.
- **Why not**: a separate section, one line each.

### Alternative 2: Log only
- **Pros**: no digest churn.
- **Cons**: the reader of a digest reads the digest, not a log.
- **Why not**: tried first. The block was logged, and it was the only record that
  a specific role had existed.

### Alternative 3: Do not record rule-hidden postings as seen
- **Pros**: a rule fix would resurface them.
- **Cons**: the same hides would repeat every run.
- **Why not**: hidden postings are still recorded (ADR-0013), and `rolescan unsee`
  brings one back on request.

## Consequences

### Positive
- A wrong term, rule or resolver shows up as visible lines rather than as silence.
- It makes the resolver-precision work (ADR-0004) auditable.

### Negative
- A hidden posting is listed once, on the run that first sees it, and is then
  `seen`. A reader who skips that digest never learns of it. A `--dry` run records
  nothing, so it lists them every time.
- Rows are Markdown links built from unescaped titles.

### Risks
- A new rule added to `decide` but not to `RULE_ORDER` and `_RULE_LABELS` renders
  under its raw name after the known rules. It does not vanish, by design.
- The section exists to be read. Nothing checks that anyone reads it. Capped, the
  lines below the cut are in a file nobody is made to open, so a wrongly hidden
  posting with a low score is the one most likely to stay hidden. The per-rule counts
  are what a reader sees of it, and a count that jumps is the cue to open the file.
- A tie at the cut falls to company and title, which is stable but arbitrary.

## Evidence
- `rolescan/digest.py`: `_RULE_LABELS`, `_hidden_group`, `_rule_hidden_rows`,
  `_hidden_view`, `_rule_hidden_section`, `render_hidden_list`.
- `rolescan/pipeline.py`: `ScanResult.rule_hidden` (its docstring gives the scope),
  `_rule_hidden`.
- `rolescan/scoring/rules.py`: `RULE_ORDER`; `rolescan/scoring/judges.py`:
  `_without_rule`.
- `tests/test_digest.py`, `tests/test_digest_volume.py` (the cap, the file and the
  size budget), `tests/test_hidden_rejects.py`, `tests/test_pipeline.py`,
  `tests/test_rules.py`.
