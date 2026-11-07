# Moving rolescan onto factgate: a plan, not started

Decided on 2026-11-07: **not now.** rolescan runs live every day, and moving it
is a rewrite of its scoring core, so the move waits until a release can be
proven to decide every posting the way the current one does. This file is the
plan for when it starts.

## What would change

rolescan keeps everything about jobs; factgate takes over the method.

| Part | Today, in rolescan | After |
|---|---|---|
| Quote check | `scoring/facts.py` `verify_facts`, job-aware | factgate's `check_quotes`, with the job-aware checks moved into resolvers |
| Resolvers | `resolve_years`, `resolve_hard_bars`, `resolve_student`, `resolve_level`, `resolve_field` | the same functions, as factgate resolvers named in a config |
| Rules | `scoring/rules.py` `decide` and `RULE_ORDER` | factgate rules in YAML where a condition can say it; a resolver where it cannot |
| Model calls, limits, cache | `FitScorer`: call ceiling, time budget, breaker, facts cache | factgate's pipeline and cache |
| Backends | `scoring/judges.py`: Anthropic (with batch) and Ollama | factgate's `anthropic` and `openai-compatible` |
| Eval | `evals/cases.json`, `evals/harness.py` | factgate cases, offline and live |
| Sources, dedup, store, digest | rolescan | unchanged: rolescan |

## Risks

1. **Verdicts drift.** rolescan's quote check knows jobs. A graduation year
   must be in its quote, and a nationality or clearance bar must name its kind.
   factgate's generic check knows none of that. Every such check has to move
   into a resolver, and any one missed changes verdicts with no error.
2. **Rules YAML cannot say.** `level_from_title_only`, `nationalities` and
   `field_exempt_companies` depend on where a fact was read, who the candidate
   is, or which company is hiring. factgate conditions test a fact's value
   only, so these become resolvers, or factgate's conditions grow. Each option
   is a design choice to make before any code moves.
3. **The local model loses its safety checks.** rolescan talks to Ollama's
   own `/api/chat`. It sets `num_ctx` on every call, detects a prompt that was
   cut, and keys the cache on the served model's digest. factgate's
   `openai-compatible` backend does none of these. Moving without them would
   bring back the silent truncation that rolescan 2.5.8 fixed.
4. **Every cached posting is asked again once.** The two caches use different
   keys. The cost is bounded by `max_calls`, but the first runs after the
   move would defer postings.
5. **The private companion breaks.** rolescan-extra registers an enricher,
   judges and sources against rolescan's own interfaces. The move changes
   those interfaces, so rolescan-extra has to move in step, under a new major
   version of rolescan.
6. **Batch semantics differ.** In rolescan a request a cancelled batch did not
   answer is a scoring error. In factgate it is deferred. The digest's counts
   would change meaning.

## Test strategy

- **Golden comparison, with no model calls.** Record model answers once for
  rolescan's 26 eval cases and a frozen sample of real postings, then replay
  them through both pipelines. Every posting must get the same verdict and
  the same rule. Each difference is either fixed or written down as intended
  before the move goes further.
- **Both test suites stay green throughout.** rolescan's 2,298 tests, and its
  26 eval cases ported to factgate's case format and run by both harnesses.
- **A shadow run.** One scheduled scan through both paths, with the two
  digests compared before anyone reads the new one.
- **A way back.** rolescan-extra keeps its current rolescan pin until the
  factgate-based release has passed the shadow run.

## Order of work

1. Port the eval cases and the golden replay set. No production code
   changes.
2. Close risk 3 in factgate: an Ollama option that sets `num_ctx`, detects a
   cut prompt and keys the cache on the model digest.
3. Move the resolvers and rules behind factgate's interfaces, inside rolescan,
   until the golden comparison matches.
4. Swap `FitScorer` for factgate's pipeline; run the shadow scan.
5. Release a new major version of rolescan; move rolescan-extra; drop the old
   code.
