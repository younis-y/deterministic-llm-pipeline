# rolescan

[![ci](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml)

A job scanner that reads employers' own career sites, scores each posting
against a profile you write, and writes a ranked digest. A model reads only the
postings that pass a free keyword prefilter, and it never makes the decision:
it reports facts, each with a quote that code checks against the advert, and
fixed rules decide. **The model reports; code decides.**

- **Cheap by design.** Most postings are dropped by deterministic keyword rules
  before any model call. A call ceiling and a time budget cap each run;
  postings past either are deferred to the next run, never dropped, and a
  cached verdict is never paid for twice. One measured run cost about USD 0.003
  a posting ([Models, keys and cost](#models-keys-and-cost)).
- **Auditable.** The same facts always give the same verdict, and every hidden
  role is listed with the rule that hid it.
- **Pluggable.** Sources and scoring backends load through entry points, so a
  new ATS adapter, or a local Ollama model in place of Claude, is a plugin, not
  a fork.
- **Tested.** Over two thousand tests run offline against `respx`-mocked
  transport, so the real HTTP clients and parsers run against recorded
  response shapes. `mypy` runs strict, and an eval runs in CI
  ([Measured quality](#measured-quality)).

**The same method, for any documents: [factgate](factgate/).** One YAML file
points it at invoices, support tickets, tenders or anything else made of text,
and its three examples run with no API key.

## Quickstart

rolescan needs Python 3.11 or newer. No account, key or model is needed to try
it.

```bash
git clone <url-of-this-repository> rolescan
cd rolescan
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                       # add ".[anthropic]" later for Claude scoring

cp examples/quickstart.yaml config.yaml
rolescan discover                      # checks that the board in config.yaml answers
rolescan scan --dry --no-llm --no-email
```

`examples/quickstart.yaml` reads one public job board with a few generic
keywords and turns every optional part off: no model, no email, no key. The
scan prints a digest and writes it to `digests/digest-dry.md`. A test runs these
commands against a mocked board, so the example cannot drift away from the code.

**Keyword-only scoring works with no model.** `--no-llm` (or `llm.enabled:
false`) scores every posting on weighted keywords in plain Python, then ranks,
deduplicates and tracks what it has seen. Those scores run lower than a model's,
so the quickstart sets `min_report_score` to 20, where the shipped
`config.example.yaml` uses 55, a number for model scores. Raise it as your
keywords sharpen.

**Use `--dry` while you tune.** A normal `rolescan scan` records every posting it
assessed as seen, the ones the prefilter dropped included, and does not show
them again. `--dry` fetches and scores the same way but records nothing, so you
can edit `config.yaml` and re-run until the digest looks right. It writes
`digest-dry.md` and leaves `latest.md` as the last real run's digest. It still
makes model calls when a model is enabled. It still drops postings an earlier
real scan recorded as seen, so tune before the first real scan, or against a new
`output.db_path`. If you scanned for real too early, `rolescan unsee URL` makes
one posting reportable again.

When you are happy with the digest, edit `profile` to describe yourself, add the
employers you care about under `sources` (`rolescan slugs "Employer Name"` and
`rolescan discover` help find the right board), and run `rolescan scan` without
`--dry`. To score with a model, see [What you must supply](#what-you-must-supply).

## The three decisions worth reading the code for

**A deterministic prefilter runs before anything expensive.** Scoring is two
stages. The first is pure Python over the posting text: weighted keyword terms,
tripled when they appear in the title, minus blocker terms and a location
penalty. Postings below `min_keyword_score` never reach the second stage. Only
the survivors are sent to an LLM. The ordering is the point: the cheap stage is
not a nicety, it is what decides the cost of the tool.

**Sources and LLM backends are both entry-point plugins.** They register
through `importlib.metadata` under the `rolescan.sources` and `rolescan.judges`
groups, declared in `pyproject.toml`. A third-party package can ship a new ATS
adapter or a new model backend without touching pipeline code, and the pipeline
holds no knowledge of either. Adding an employer on an already-supported
platform is a line of config and no code at all.

**The model never makes the decision.** The model
returns structured facts with quotes; a quote the advert does not contain is
dropped, resolver passes settle the level, field and years required, and
`decide` applies your `profile.rules` in plain code. A model that misreads one
fact costs one fact, not a silent verdict, and a rule change re-runs on cached
facts without paying for the model again.

## Why it exists

Aggregators are late, partial, and full of recruiters. Postings arrive first on
the employer's own site, which is backed by an applicant-tracking system with a
public JSON API. rolescan reads those directly, so a role shows up the day it is
posted, from the source, with the real apply link.

The hard part is not fetching. It is knowing that Octopus Energy's board is
`octoenergy` on Lever and not `octopus` on Greenhouse. `rolescan slugs` and
`rolescan discover` exist for exactly that.

## What it reads

| Source | Needs a key | Notes |
|---|---|---|
| `greenhouse` `lever` `ashby` `workable` | no | public JSON, one request per board |
| `smartrecruiters` | no | paginated. Returns 200 with zero rows for a slug that does not exist, so `discover` reports `UNKNOWN` rather than lying |
| `workday` | no | needs tenant, site and host, all three readable from the careers URL. Reads up to 500 postings a board (`max_rows` raises it; `applied_facets` and `search_text` narrow a big board) and says when a board held more |
| `structured` | no | any site publishing schema.org `JobPosting` plus a sitemap. Covers Phenom People, SAP SuccessFactors, Teamtailor and Harbour today, by configuration rather than by adapter. A big sitemap stays polite and bounded: robots.txt's `Crawl-delay` is a floor under `delay`, `max_age_days` skips old URLs, `incremental` reads only what changed since the last scan that read its whole window (it compares sitemap `lastmod` only, so a URL first listed under an older date is missed until one scan with `incremental: false`; a read cut short at `max_pages` keeps the old mark, and repeats until the window fits), and `max_sitemap_urls` refuses an oversized file |
| `adzuna` | yes | an aggregator. Free tier, 20 countries |
| `workable_search` | no | Workable's search across the employers that post on it, by query and place. Checked with one live request on 2026-10-09; the paging parameter's name is still to be confirmed against a second real page, and a read stops and says so if a page repeats. Queries run one at a time, `delay` (default 2 s) apart |
| `reed` | yes | Reed.co.uk, the UK (`REED_API_KEY`). Search results carry a shortened description. Built from the public API documentation and not yet run against the live service |
| `jooble` | yes | Jooble, 60-odd countries (`JOOBLE_API_KEY`). Its key is part of the request URL, which a `--verbose` run logs with the key replaced by `<redacted>`. Built from the public API documentation and not yet run against the live service |

## How scoring works

Two stages. The first is keywords: weighted terms, tripled when they appear in
the title, minus `blockers` and a location penalty. A `blockers` term costs
points; a `hard_blockers` term forces `blocked` whatever the model says, so keep
that list short. The second, for the postings that pass, is a model
(`anthropic`, or a local one through `ollama`) asked only what the advert says:
the level, the minimum years, whether it is open only to students, the earliest
graduation year it accepts, the field and any hard bar, each with the words it
read them from. Code drops a fact whose quote is not in the posting and applies
your `profile.rules` in a fixed order. A fact the advert does not state never
fires a rule, and the model's 0-100 fit score decides the rest. Facts are
cached and the rules run on every read, so a rule change costs no model calls.

[docs/scoring.md](docs/scoring.md) has the rule order, the `profile.rules` keys,
worked examples, the local model's context window and the cache.

## What gets hidden, and where to look

rolescan reports most of what it leaves out of your digest. The stats line
counts what a threshold dropped. A "Hidden by your rules" section lists each
posting a rule skipped or a weighted term pushed under the gate, with the
advert's own words, so a rule that matches the wrong thing shows up rather than
staying silent. A `hard_blockers` or `excluded_locations` match stays at the
bottom of the digest, marked BLOCKED, unless `output.show_blocked: false` moves
it into that section.

After a real scan, a `--dry` re-run finds nothing new, because it drops seen
postings too. Run `rolescan unsee URL` for one posting, or tune against a
scratch `output.db_path`. [docs/hidden.md](docs/hidden.md) has where to look for
each kind of hidden posting, the section's cap and when a posting is listed
again.

## What you must supply

- **Your own `config.yaml`.** Nothing is shipped. `examples/quickstart.yaml` is
  the smallest working file, `config.example.yaml` is a starting point with the
  common keys commented, and [docs/config.md](docs/config.md) lists every key
  with its type, default and meaning (generated from the config models, so it
  cannot drift). `examples/energy-trading.yaml` is a fully resolved, working
  source list for UK and Gulf energy, with a placeholder profile.
- **Your own API key.** Every secret resolves from the environment when the
  corresponding config field is left blank: `ANTHROPIC_API_KEY`, and optionally
  `ADZUNA_APP_ID` / `ADZUNA_APP_KEY`, `REED_API_KEY`, `JOOBLE_API_KEY` and
  `ROLESCAN_SMTP_PASS`. Nothing is
  bundled, and `config.yaml` is gitignored so a key pasted there by hand does
  not follow you into a commit.
- **Your own ATS slug dataset**, if you want `rolescan slugs`. See
  `ats-data/README.md`. It is CC BY-NC 4.0 and is not bundled here.

Relative paths resolve against the config file, not the working directory, so a
cron job writes to the same database as an interactive run.

## Models, keys and cost

Scoring has three levels. Each is optional, and each says so when it does not
run.

| Level | Needs | Cost | Notes |
|---|---|---|---|
| Keywords only (`--no-llm`, or `llm.enabled: false`) | nothing | free | Pure Python. Scores run lower than a model's: set `min_report_score` to about 20 to 30. |
| A local model (`llm.backend: ollama`) | [Ollama](https://ollama.com) running, and a model you have pulled | free; the time depends on your hardware | No key, and nothing leaves your machine. rolescan asks for `llm.num_ctx` tokens of context (12,288 by default) and refuses a model whose window is smaller. |
| Claude (`llm.backend: anthropic`) | the optional SDK (`pip install -e ".[anthropic]"`) and `ANTHROPIC_API_KEY` | your provider's price per token, times the calls | See below. |

`rolescan backends` lists what is registered and which backends need a key.

Only postings that pass the prefilter reach a model, a cached answer is not
paid for twice, and `llm.max_calls_per_run` and `llm.max_minutes` cap each
scan. [docs/models.md](docs/models.md) has those limits, batch calls at half
price, what a call sends and the model-health check.

One cost has been measured. With `claude-haiku-4-5` in facts mode, two
evaluation runs of 97 postings each, on 8 October 2026, cost USD 0.29 and
USD 0.30: about USD 0.003 a posting. That is one model on one set of adverts.
Yours depends on the model you choose and how long your postings are, so read
your provider's usage after one `scan`.

If a model is configured but cannot run (no key, the SDK missing, the server
unreachable, a context window too small), the scan says so at the top of its
output and in the digest, ranks on keywords instead, and exits 3. It does not
pretend.

## Privacy and storage

Only with `llm.backend: anthropic` does anything go to a model provider: your
`profile.summary`, any worked examples, and the title, company, location, date
and the start of the description of each posting that passes the prefilter.
What rolescan keeps (`seen.db`, `digests/`, `backups/`) is local and describes
your job search, so treat it as private. It reads public endpoints only, with
no logins or cookies. [docs/privacy.md](docs/privacy.md) has where keys go, what
is stored, how email is sent and what rolescan requests.

## Measured quality

Two measurements are recorded, each stated here once.

**The prefilter, 25 August 2026.** One run against a private config covering the
same employers as `examples/energy-trading.yaml`:

| | |
|---|---|
| Unique postings fetched | 526 |
| Sources contributing | 9 |
| Discarded by the keyword prefilter | 490 (93%) |
| Reached the scoring stage | 36, about one in fifteen |
| Scored by a model | 0: no backend was configured on that run, so all 36 were scored on keywords alone |
| Scored from cache | 0 (first run) |

The counters and the per-source outcome footer are in `examples/run-summary.md`.
The ranked postings are not: a digest records one person's job search, and that
is not something to publish. The discard rate is a function of how tightly you
write your keywords.

**The facts pipeline, 7 November 2026.** `evals/cases.json` holds 26 synthetic
adverts, each written around a trap that hides a good role or lets a bad one
through: a range from zero ("0-2 years"), a cap ("up to 3 years"), an age
limit, "student" naming who the role serves, a programme year read as a
graduation year, an equal-opportunity "national origin", visa sponsorship as a
benefit, level words that are not levels. Each case states what the finished
facts and the rule that fires must be.

| Run | What answers | Passed |
|---|---|---|
| Offline, in CI (`tests/test_eval_cases.py`) | each case's own model answer, often a wrong one, through the quote check, resolvers and rules | 26 of 26 |
| Live, `claude-haiku-5-5`, 3 runs (`python -m evals.harness`) | the model, through the same `FitScorer` a scan uses | 26 of 26 in every run: pass@1 100%, pass^3 100% |

The offline run is the stricter one: it feeds the code the wrong answers a
model has been seen to give and checks that the code corrects them. A case the
code gets wrong can be marked `open_defect` and runs as a strict expected
failure, so fixing it fails CI until the mark is removed; 3.0.1 fixed the seven
that 3.0.0 shipped with, and none is open. The 78 live calls took 47 seconds
and 40,720 output tokens, about USD 0.02 of output; the run's full record is in
`evals/results/`. 26 short adverts is a regression set, not a
benchmark: it says which traps the pipeline handles, not how often real
postings contain them.

## Commands

```
rolescan discover     probe every configured source; report OK / EMPTY / UNKNOWN / SKIPPED / FAIL
rolescan slugs NAME   find a real board slug in a harvested ATS dataset
rolescan sources      every registered source kind and its slug format
rolescan backends     every registered LLM backend, and which need a key
rolescan scan         fetch, score, write the digest   (--dry, --no-llm, --no-email)
rolescan mark URL S   record what you did with a posting: shortlist, applied, dismissed
rolescan unsee URL    forget a posting (url or uid) so the next scan can report it again
rolescan show         reprint the latest digest
rolescan stats        how many postings the store has seen
rolescan backup       copy the store to backups/ and check the copy
rolescan prune        trim old caches (output.retention_days); never seen or applications
```

[docs/commands.md](docs/commands.md) covers `mark`'s three states, the exit
statuses a scheduler sees, the settings for a VPN that blocks mail or job
sites, `discover`'s five outcomes, and finding slugs.

## Scope

What this does not claim:

- **A model can miss structural bars.** The eval under
  [Measured quality](#measured-quality) covers 26 synthetic adverts, not your
  market. Configured `hard_blockers` do not depend on the model, so put anything
  you cannot apply your way past there.
- **Public endpoints only.** Nothing behind a login, no session cookies, no
  captcha solving, no scraping of anything a careers page does not serve to an
  anonymous browser.
- **Five ATS platforms have first-class adapters**: Greenhouse, Lever, Ashby,
  Workable and SmartRecruiters, plus Workday, which needs three values rather
  than a slug. Everything else is reached through the generic schema.org source
  or not at all.
- **The slug dataset is third-party, incomplete and not redistributable.** It
  ships no SmartRecruiters file. Slug resolution remains partly manual.
- **Not a tracker.** It records what it has seen so it does not show you the
  same posting twice. It does not manage applications, and it does not apply to
  anything on your behalf.
- **The scores are a triage heuristic**, not a prediction of whether you will be
  interviewed.

## Public core and private plugins

This repository is the core: the scoring, the rules, the digest, the store and
the source adapters for public career sites. It reads public endpoints and
sitemaps only. It does not include a scraper for sites that need a login or
forbid automated access, a bulk board crawler, tooling that tailors documents or
applies to roles, or a scheduler, and none is planned here.

The core is built to be extended without a fork. Sources, model backends and
enrichers load through the entry-point groups `rolescan.sources`,
`rolescan.judges` and `rolescan.enrichers`, so a separate package can ship its
own and register it by installing alongside this one. Nothing in the core
requires such a package, and the core imports none. `ScanResult`, `ScoredJob`,
`SourceReport`, `Source`, `Store` and `FitScorer` are the library surface a
plugin builds on; the changelog lists what each release added to them.

## Reading the source

The module docstrings are written to record negative results, not just to
describe what the code does. Why length ratio is the wrong discriminator for
slug matching; why conditional GETs are useless against employers who answer
`If-Modified-Since` with 200 and the full body, leaving sitemap `lastmod` as
the only workable cache-invalidation signal; why Workday's 422 and 404 mean
opposite things, one a wrong tenant and one a wrong site; why SmartRecruiters'
zero result count is never verification. Approaches that were tried and
rejected are written down where the next person will hit them, which is the
main reason the docstrings are long. The decisions behind the design, and the
alternatives turned down, are collected in [docs/adr](docs/adr/README.md).

## Development

```bash
pip install -e ".[dev]"
pytest && mypy src tests && ruff check .
```

`ruff check` is the quality gate, not just a formatter: alongside the usual
rules it enforces `max-complexity = 12` (mccabe), naive-datetime detection,
unused arguments and exception-message hygiene. `pyproject.toml` records which
rule families are deliberately **not** selected, and why, so the next person
does not re-add bandit to read 477 "assert used" findings out of a test suite.

Three more checks are not worth a pre-commit hook but are worth running when
the dependency list, the module layout or a hot function changes:

```bash
uvx deptry .                                       # unused/missing/transitive deps
uvx radon cc src -n C -s                           # complexity regressions
uvx vulture src --min-confidence 80 \
    --ignore-names exc_type,exc,tb,attrs           # dead code
```

All three are clean. The `--ignore-names` are parameters a protocol requires
and the implementation does not use: `__aexit__`'s three, and `attrs` on
`HTMLParser.handle_starttag`. `radon` is advisory - the enforced ceiling is
ruff's `C901`, which counts differently and more conservatively than radon
does, so expect radon to report grade C on functions ruff passes.

The suite is offline by construction: HTTP is mocked with `respx`, so tests
exercise the real client and the real parsers against recorded payloads rather
than stubs, and it runs in seconds. mypy runs in strict mode over `src` and
`tests`. CI runs all three checks on Python 3.11 and 3.12. See
`.github/workflows/ci.yml`.

## Licence

MIT. See `LICENSE`.

The harvested ATS directories that `rolescan slugs` reads are not covered by it:
they come from a third party under CC BY-NC 4.0, are never bundled, and are
gitignored. See `ats-data/README.md`.
