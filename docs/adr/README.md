# Architecture decision records

The decisions that shape how rolescan scores, hides and remembers postings, each
with the reasoning and the alternatives that were turned down. They are written
after the fact from the code and its history, so the dates are when each decision
was taken, and where the code has since moved on the record says so.

The numbers are not contiguous. The gaps are decisions about code that lives
outside this repository.

| ADR | Decision | Status | Date |
|-----|----------|--------|------|
| [0002](0002-facts-then-rules-scoring.md) | The model extracts facts; code applies the rules | accepted | 2026-09-30 |
| [0003](0003-every-fact-carries-a-verbatim-quote-verified-by-code.md) | Every fact carries a verbatim quote, and code verifies it | accepted | 2026-09-30 |
| [0004](0004-deterministic-resolvers-override-or-fill-the-model.md) | Deterministic resolvers override or fill the model's facts | accepted | 2026-10-01 |
| [0005](0005-level-and-field-come-from-the-title-and-only-a-title-level-fires-the-level-rule.md) | Level and field are read from the title; only a title level can skip | accepted | 2026-10-01 |
| [0006](0006-only-nationality-and-clearance-bars-block.md) | Only nationality and clearance bars block; other bars skip; work authorisation decides nothing | accepted | 2026-10-01 |
| [0007](0007-bpss-is-not-a-structural-bar-by-default.md) | BPSS alone is not a structural bar by default | accepted, model path differs | 2026-10-07 |
| [0008](0008-facts-cache-key-is-versioned-by-hand-and-names-backend-model-and-examples.md) | The facts cache key carries a hand-bumped version, the backend, the model and a prompt fingerprint | accepted, guarded by a test | 2026-09-30 |
| [0009](0009-rule-skips-and-term-blocks-are-listed-in-hidden-by-your-rules.md) | Nothing a rule or a blocking term hides is hidden without a trace | accepted, capped at `output.hidden_max` since 2.6.0 | 2026-10-07 |
| [0010](0010-hardness-is-an-explicit-list-and-each-term-list-has-its-own-match-scope.md) | Hardness is an explicit list; each configured list has its own match scope | accepted | 2026-09-24 |
| [0011](0011-employer-board-dates-are-not-freshness.md) | An employer board's posted date is not a freshness signal | accepted, covers every employer-board source since 2.6.0 | 2026-09-28 |
| [0013](0013-a-posting-is-recorded-as-seen-once-assessed-and-never-expires.md) | A posting is recorded as seen once assessed, and `seen` never expires | accepted, annual programmes return after a gap since 2.7.0 | 2026-09-01 |
| [0014](0014-a-run-must-never-succeed-silently.md) | Silence must never be ambiguous | accepted | 2026-09-23 |
| [0015](0015-a-deterministic-prefilter-runs-before-the-model-and-calls-are-capped.md) | A deterministic prefilter runs before the model; calls are capped | accepted | 2026-09-01 |
| [0020](0020-sources-judges-and-enrichers-are-entry-point-plugins.md) | Sources, judges and enrichers are entry-point plugins | accepted | 2026-09-01 |
| [0022](0022-posting-identity-and-near-duplicate-merging.md) | Posting identity and near-duplicate merging | accepted | 2026-09-01 |

## Adding one

Copy the shape of an existing record: context, decision, alternatives considered,
consequences (positive, negative, risks), evidence. Cite code by module and
function name, not by line number, and name the tests that guard the decision. A
record states what was decided and why. It does not name the people involved, and
it quotes no figure that came from someone's own data.

When a later change contradicts a record, say so in that record's status line and
in its consequences, and link the newer one. A record is not edited to hide that
the world moved on.
