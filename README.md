# rolescan

[![ci](https://github.com/younis-y/rolescan/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/rolescan/actions/workflows/ci.yml)

A job scanner that reads employers' own career sites, scores postings against
your CV, and writes a ranked digest, spending an LLM call only on the small
minority that survive a deterministic prefilter.

**That prefilter is the design point.** On one real run it scanned 526 unique
postings across 9 employer sources and discarded 490 of them (**93%**) on
deterministic keyword rules before any model was invoked, leaving 36 to score.
The expensive stage sees a fortieth of the traffic, and the cheap stage is
reproducible and free to re-run.

Both the sources and the scoring backends load through **entry points**, so
adding an ATS adapter or swapping Claude for a local Ollama model is a plugin,
not a fork. **264 tests run offline** against `respx`-mocked transport: the real
HTTP clients and real parsers are exercised against recorded response shapes
rather than stubbed out. `mypy` runs strict across `src` and `tests`.


## Quickstart

```bash
pip install -e ".[dev]"
pytest                                    # 264 tests, no network
cp config.example.yaml config.yaml        # then edit sources and profile
rolescan discover                          # probe the sources you configured
rolescan scan --no-llm                     # keyword-only run, no API key needed
```

`--no-llm` gets you a ranked digest on the deterministic prefilter alone, so the
tool is useful before you supply a key. See [What you must supply](#what-you-must-supply).

## The two decisions worth reading the code for

**A deterministic prefilter runs before anything expensive.** Scoring is two
stages. The first is pure Python over the posting text: weighted keyword terms,
tripled when they appear in the title, minus blocker terms and a location
penalty. Postings below `min_keyword_score` never reach the second stage. Only
the survivors are sent to an LLM. In the run recorded in
`examples/run-summary.md` that stage discarded 490 of 526 postings before a
single model call. The ordering is the point: the cheap stage is not a
nicety, it is what decides the cost of the tool.

**Sources and LLM backends are both entry-point plugins.** They register
through `importlib.metadata` under the `rolescan.sources` and `rolescan.judges`
groups, declared in `pyproject.toml`. A third-party package can ship a new ATS
adapter or a new model backend without touching pipeline code, and the pipeline
holds no knowledge of either. Adding an employer on an already-supported
platform is a line of config and no code at all.

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
| `workday` | no | needs tenant, site and host, all three readable from the careers URL |
| `structured` | no | any site publishing schema.org `JobPosting` plus a sitemap. Covers Phenom People, SAP SuccessFactors, Teamtailor and Harbour today, by configuration rather than by adapter |
| `adzuna` | yes | the one aggregator. Free tier, 20 countries |

## How scoring works

**1. Keywords.** Pure Python, no network, no credentials. Described above. You
can stop here: `--no-llm` still gives a ranked, deduplicated, seen-tracked
digest.

**Blockers, and which of them are walls.** Two separate fields, because they
answer different questions:

```yaml
blockers:            # a severity gradient. Points off the keyword score.
  security clearance: 60
  matlab: 15
hard_blockers:       # structural bars. Nothing you can apply your way past.
  - security clearance
  - uae national
```

`blockers` says what a term *costs*. However heavy, it only moves a posting
down the ranking, and the LLM can still say apply — until the deduction takes
it under `min_keyword_score`, at which point it is prefiltered, recorded as
seen, and gone. That is why the two fields are matched identically rather than
the weights being matched loosely.

`hard_blockers` says what no application can get past. A term here forces
`blocked` whatever the model decides, so the posting is pushed to the bottom of
the digest — or, with `output.show_blocked: false`, removed from it entirely
and recorded as seen, which means you never see that role again. **That is a
real deletion, on a term you typed into a YAML file**, so keep the list short
and specific. Terms in both fields are normalised at load and matched
case-insensitively on word boundaries (`crypto` does not match
"cryptographic", `head of` does not match "head office"), a term in both
lists keeps its weight, and a term in `hard_blockers` alone blocks and costs
nothing. Whenever
a posting is deleted this way the digest's stats line says how many, so a
blocker matching the wrong thing shows up as a number rather than as silence.

Hardness is deliberately not a weight threshold. A weight is retuned whenever
you calibrate the prefilter; whether a clearance is a wall is a fact about you
that changes once a decade. One number could not encode both without one of
them silently changing the other.

**2. An LLM,** optionally. It picks which of your CV variants to send, scores
fit, flags hard eligibility bars, and names concrete edits to make. Verdicts
are cached on a content hash of the posting text, so re-running costs nothing
for unchanged postings.

The backend is a plugin. `rolescan backends` lists what is registered:

| Name | Needs a key | What it is |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude API. Best quality. Needs ANTHROPIC_API_KEY. |
| `ollama` | no | Local model via Ollama. Free, offline, no key. Run against a live server: 72% verdict accuracy over 25 postings, 50% blocker recall. |

Both backends have now run against a live server.

The Anthropic backend scored 36 postings in the 25 August 2026 run recorded in
`examples/run-summary.md`, with no failures. That key has since been revoked,
so the run is a record rather than something you can re-execute.

The Ollama backend has been measured rather than merely exercised: a
25-posting benchmark against hand-labelled expectations, plus one live
34-posting scan. **Verdict accuracy 72%. Score/verdict violations 0%** — the
band table in the scoring prompt and the verdict returned never disagreed.
**Model-level blocker recall 50%:** it named half the structural bars in the
benchmark set.

That last number is the one worth reading. The tool does not rely on the model
to catch structural bars — `hard_blockers` are matched in Python before the
model is called and force `blocked` whatever it says, and `min_report_score`
gates the rest — but a bar that appears only in the posting text, and that no
configured term names, is one this backend will miss about half the time. The
hosted backend is materially better at it. Choose accordingly.

With `llm.enabled: false`, or with no key and no local model, you get stage one
alone. If a configured backend fails, the digest says so rather than quietly
serving keyword scores that look like a normal run. There is a test that
asserts exactly this.

## What you must supply

- **Your own `config.yaml`.** Nothing is shipped. `config.example.yaml`
  documents every knob; `examples/energy-trading.yaml` is a fully resolved,
  working source list for UK and Gulf energy, with a placeholder profile.
- **Your own API key.** Every secret resolves from the environment when the
  corresponding config field is left blank: `ANTHROPIC_API_KEY`, and optionally
  `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` and `ROLESCAN_SMTP_PASS`. Nothing is
  bundled, and `config.yaml` is gitignored so a key pasted there by hand does
  not follow you into a commit.
- **Your own CV files**, if you want tailoring advice rather than generic
  advice. See `cvs/README.md`.
- **Your own ATS slug dataset**, if you want `rolescan slugs`. See
  `ats-data/README.md`. It is CC BY-NC 4.0 and is not bundled here.

Relative paths resolve against the config file, not the working directory, so a
cron job writes to the same database as an interactive run.

## Results

One run, 25 August 2026, against a private config covering the same employers
as `examples/energy-trading.yaml`:

| | |
|---|---|
| Unique postings fetched | 526 |
| Sources contributing | 9 |
| Discarded by the keyword prefilter | 490 (93%) |
| Sent to stage two | 36 |
| Scored from cache | 0 (first run) |

The counters and the per-source outcome footer from that run are reproduced in
`examples/run-summary.md`. The ranked postings are not: a digest records one
person's job search, and that is not something to publish. This is one
measurement against one source list, not a benchmark. The discard rate is a
function of how tightly you write your keywords.

## Commands

```
rolescan discover     probe every configured source; report OK / EMPTY / UNKNOWN / SKIPPED / FAIL
rolescan slugs NAME   find a real board slug in a harvested ATS dataset
rolescan sources      every registered source kind and its slug format
rolescan backends     every registered LLM backend, and which need a key
rolescan cvs          which CV variants were found and how they parse
rolescan scan         fetch, score, write the digest   (--dry, --no-llm, --no-email)
rolescan mark URL S   record what you did with a posting: shortlist, applied, dismissed
rolescan show         reprint the latest digest
rolescan stats        how many postings the store has seen
rolescan prune        drop stale cached verdicts
```

`mark` takes one of three states. `shortlist` keeps a posting in the digest's
**Shortlist** section, which is repeated under the new roles on every run so an
open application stays in front of you rather than scrolling away with
yesterday's email. `applied` clears it from that section. `dismissed` clears it
too, and additionally stops the posting ever being reported again, however many
boards go on listing it. Every shortlist row in the digest carries the exact
`mark` command to clear it, with the config path already filled in.

`discover` distinguishes five outcomes on purpose. `EMPTY` means a real board
with no openings; `UNKNOWN` means an API that cannot tell an empty board from a
wrong slug. Collapsing those into a single "OK" hides which of them actually returned postings.

## Finding slugs

`rolescan slugs` searches a harvested ATS directory you download yourself. It
matches on `difflib.SequenceMatcher` with a length-aware cap and a directional
containment bonus, because absolute name length rather than length ratio is
what separates `octoenergy` from `octopusenergy` without also collapsing `vitl`
into `vitol`.

The dataset has no SmartRecruiters file and covers no bespoke platforms, so
expect to resolve some employers by hand: open the careers page, follow where it
redirects, and read the ATS and slug out of the final URL. `discover` then
confirms or refutes it.

## Scope

What this does not claim:

- **The local backend catches about half the structural bars the hosted one
  does.** Measured, not estimated: see above. Configured `hard_blockers` do not
  depend on the model, so put anything you cannot apply your way past there.
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

## Reading the source

The module docstrings are written to record negative results, not just to
describe what the code does. Why length ratio is the wrong discriminator for
slug matching; why conditional GETs are useless against employers who answer
`If-Modified-Since` with 200 and the full body, leaving sitemap `lastmod` as
the only workable cache-invalidation signal; why Workday's 422 and 404 mean
opposite things, one a wrong tenant and one a wrong site; why SmartRecruiters'
zero result count is never verification. Approaches that were tried and
rejected are written down where the next person will hit them, which is the
main reason the docstrings are long.

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

The suite is 264 tests and runs in about five seconds with no network: `respx`
mocks the transport, so the real HTTP clients and the real parsers are exercised
against recorded response shapes rather than being stubbed out. `mypy` runs
strict over `src` and `tests`.

The suite is offline by construction: HTTP is mocked with `respx`, so tests
exercise the real client and the real parsers against recorded payloads rather
than stubs. mypy runs in strict mode over `src` and `tests`. CI runs all three
on Python 3.11 and 3.12. See `.github/workflows/ci.yml`.

## Licence

MIT. See `LICENSE`.

The harvested ATS directories that `rolescan slugs` reads are not covered by it:
they come from a third party under CC BY-NC 4.0, are never bundled, and are
gitignored. See `ats-data/README.md`.
