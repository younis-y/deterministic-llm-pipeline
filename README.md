# rolescan

[![ci](https://github.com/younis-y/rolescan/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/rolescan/actions/workflows/ci.yml)

A job scanner that reads employers' own career sites, scores postings against
the profile you write, and writes a ranked digest, spending an LLM call only on
the small minority that survive a deterministic prefilter.

**That prefilter is the design point.** Most postings are discarded on
deterministic keyword rules before any model is invoked, so the expensive stage
sees a small fraction of the traffic, and the cheap stage is reproducible and
free to re-run. [Measured quality](#measured-quality) has one real run.

Both the sources and the scoring backends load through **entry points**, so
adding an ATS adapter or swapping Claude for a local Ollama model is a plugin,
not a fork. Over a thousand tests run offline against `respx`-mocked transport:
the real HTTP clients and real parsers are exercised against recorded response
shapes rather than stubbed out. `mypy` runs strict across `src` and `tests`.


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

## The two decisions worth reading the code for

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
a posting is deleted this way the digest's stats line says how many, and its
"Hidden by your rules" section lists the posting with the term that matched,
so a blocker matching the wrong thing shows up rather than as silence. That
includes a posting the term's `blockers` weight took under the prefilter; one
that would have failed the prefilter without the term is not listed, and an
`excluded_locations` entry, which has no weight, never lists a prefiltered
posting. It is listed once, on the run that first sees it.

Hardness is deliberately not a weight threshold. A weight is retuned whenever
you calibrate the prefilter; whether a clearance is a wall is a fact about you
that changes once a decade. One number could not encode both without one of
them silently changing the other.

**2. An LLM,** optionally. It scores fit against your profile summary and
flags hard eligibility bars. Verdicts are cached on a content hash of the
posting text, so re-running costs nothing for unchanged postings.

The backend is a plugin. `rolescan backends` lists what is registered:

| Name | Needs a key | What it is |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude API. Best quality. Needs ANTHROPIC_API_KEY. |
| `ollama` | no | Local model via Ollama. Free, offline, no key. See [Measured quality](#measured-quality). |

The tool does not rely on a model to catch structural bars: `hard_blockers` are
matched in Python before the model is called and force `blocked` whatever it
says, and `min_report_score` gates the rest. A bar that appears only in the
posting text, and that no configured term names, is the one a model can miss.

**Facts first, then rules.** By default (`llm.mode: facts`) the model is not
asked whether you should apply. It is asked what the advert says: the level,
the minimum years, whether it is open only to current students, the earliest
graduation year it accepts, the field, and any hard eligibility bar, each with
the advert text it read it from. Code checks every quote is really in the
posting (a fact whose quote is not there is dropped), then applies your
`profile.rules` in a fixed order:

1. A nationality or security-clearance bar: `blocked`. The bar's own quote
   must name it - nationality, national(s), citizen(ship), passport,
   Emirati, Emiratisation and the like for nationality ("right to work" is
   not one); clearance, cleared, vetting, SC, DV, BPSS, NPPV for clearance -
   or the bar is dropped like an unverified quote. Any other stated
   mandatory requirement (a licence, a residency) is a `skip` that quotes it,
   never a block. A work-authorisation bar the model extracts decides
   nothing: put the visa wording you cannot get past in `hard_blockers`.
2. With `max_graduation_year` set, a stated graduation year above it: `skip`.
   The year is the earliest the advert accepts, so "graduating 2027 or 2028"
   is 2027 and passes a limit of 2027.
3. `student_only: skip` and the advert is students-only: `skip`. While
   `max_graduation_year` is set, this applies only to adverts that state no
   graduation year: one that states a year within the limit passes, however
   student-only it is, because the year is the sharper test. An advert that
   also accepts graduates ("students or recent graduates", "recently
   completed a degree", "graduated within the last year") is not
   students-only, whatever the model says: code finds that wording in the
   title or description and clears the flag, quoting it. Wording with a
   negation close by ("recent graduates are not eligible") does not count.
4. The level is not in `allowed_levels`: `skip`. A level word in the title
   decides the level: graduate, grad, intern, internship, placement,
   entry-level, trainee, apprentice; junior, jr, assistant; senior, sr,
   mid-senior; principal, head of, director, "staff" before
   engineer/scientist/developer, and "lead" before a role word (Lead Data
   Engineer, not Lead Generation); mid-level, intermediate. Graduate and
   junior words beat the rest; lead beats senior; senior beats mid.
   "manager" is not a level word. Otherwise the model's level counts only
   with a quote of two words or more. With `level_from_title_only: true`,
   only a level word in the title can fire this rule; a level the model
   states is ignored, even when its quote is copied from the title (the
   title "Data Engineer" names no level, so a model answering "mid" from it
   has guessed).
5. The stated minimum years exceed `max_years_required`: `skip`.
6. The field is not in `allowed_fields`: `skip`. Like level, the field is
   read from the title first (Data Engineer, Data Analyst, ML Engineer,
   Quantitative Analyst, Product Manager, Consultant and so on; a title is
   never read as `other`); otherwise the model's field counts only with a
   quote of two words or more. Data and AI words win over the rest ("Data
   Science Consultant" is `data_science`), quant and product words beat
   software words, and software beats a plain "consultant".
7. Otherwise the model's 0-100 skills/domain score decides.

A fact the advert does not state never fires a rule, so an advert that says
nothing about level or years is decided on fit alone. Every rule skip names
the rule and quotes the advert, and the digest lists every posting a rule
kept out of it under "Hidden by your rules", one line each grouped by rule,
so a wrong skip shows up rather than vanishing. The `rules` keys and their
defaults:

| Key | Default | Meaning |
|---|---|---|
| `max_years_required` | `null` | Skip a stated minimum above this; `null` disables the check. |
| `allowed_levels` | `[graduate_entry, junior, mid, not_stated]` | Levels that pass. Also `senior`, `lead_principal`. Keep `not_stated`. |
| `student_only` | `allow` | `skip` drops roles open only to current students. |
| `max_graduation_year` | `null` | Skip a stated graduation year above this; `null` is no limit, and a stated year then changes nothing. |
| `level_from_title_only` | `false` | `true`: only a level word in the job title can skip; a level the model states is ignored. |
| `allowed_fields` | `null` (any) | From `data_engineering`, `ai_llm`, `data_science`, `analytics_bi`, `quant`, `product`, `software`, `consulting`, `finance`, `other`. |

Since 2.5.7 the digest's stats line also reports a posting a run could not
get to: "N deferred to the next run", with the reason (over the LLM budget,
over the digest cap, or no description text yet). A deferred posting is not
recorded as seen, so it comes round again. A posting with no text is held back
only `output.thin_unread_after` times (default 3): after that many runs without
text it is listed once under "Unread (no text after N runs)", with its link,
and recorded, so a source that never sends descriptions cannot hold it back for
ever. To bring back a posting a wrong rule or term hid, run `rolescan unsee
URL` (or its uid); the next scan reports it again. Four keys tune what gets
hidden. On `profile`: `nationalities` (the demonyms and countries you hold, so
a nationality bar that names one is not a bar for you; it acts in facts mode,
through the rules, and does nothing in judge mode or against a `hard_blockers`
term), `title_only_blockers` (`blockers` terms such as
`head of`, `director` or `military` whose weight counts only in the job title,
not in boilerplate in the body) and `hidden_gate_margin` (how far under
`min_keyword_score` a reject may fall and still be listed under "Hidden by
your rules"; default 10, 0 lists none). On `profile.rules`:
`field_exempt_companies` (employers whose postings pass the field rule
whatever their field). All four are in `config.example.yaml`.

`llm.facts_examples_file` points at a YAML list of worked examples, each a
posting excerpt and the facts it should yield:

```yaml
- title: Data Science Intern
  company: Example Co
  description: A ten-week internship for students graduating in 2028.
  facts:                      # the shape the model returns
    level: {value: graduate_entry, quote: Data Science Intern}
    graduation_year: {value: 2028, quote: graduating in 2028}
    fit_score: 70
    reason: Python and SQL match.
```

They are appended to the facts-mode system prompt after its instructions, so
they sit inside the cached prefix. The path is relative to the config file,
and every example is validated when the config loads: a bad one stops the run
and names the example. That includes the quote check every real posting gets:
each quote must appear in the example's own title or description, a
graduation year must appear in its quote, a nationality or clearance bar must
name its kind, and an example must not mark students-only an advert that
also accepts graduates. No file, no change to the prompt.

With examples, a facts-mode call needs about 8k tokens of context (the
prompt, the examples and one posting). rolescan asks Ollama for
`llm.num_ctx` tokens on every call (12,288 by default), checks before the
scan that the model can hold that many, and treats an answer whose token
counts show the prompt was cut as an error rather than as facts. Raise
`llm.num_ctx` if you add many more examples. A model whose own window is
smaller than `llm.num_ctx` (an 8k one, under the default) is refused: no
posting is sent to it, the run falls back to keyword scoring and exits 3;
lower `llm.num_ctx` to its window, or choose a longer one. If your Ollama
counts only the uncached part of a prompt, every call would look cut: set
`llm.check_truncation: false`.

Facts are cached as the model gave them, per posting text and per
fingerprint of everything that decides the answer (the prompt, your summary,
the examples, the model's digest, `num_ctx`). Verification, the resolvers
and your rules run on every read, so a rules or resolver change applies to
cached postings at no model cost, and a prompt or model change asks the
model again.

**Judge mode.** `llm.mode: judge` is the other supported mode, and it is not
going away. It makes one call per posting and lets the model give the score,
the verdict and the blockers itself, so there is no quote check and no
resolver. Use it to compare against facts mode, or when you want the model's
own reading of a posting. What differs from facts mode:

- `profile.rules` do not apply: they act on facts, and judge mode extracts none.
  `hard_blockers` still force `blocked`, as in every mode.
- The model may block on a bar it names, a work-authorisation bar included.
  Facts mode ignores work authorisation and blocks only on a nationality or
  clearance bar that passes the quote check.
- `llm.extra_prompt` is added to the prompt, which facts mode does not send.
- Nothing names a rule, so "Hidden by your rules" lists hard-term hits under
  "Your blocking terms" and the model's own blocks under "Nationality,
  clearance or other hard bar".
- Verdicts are cached on the posting's content hash alone for `llm.cache_days`.
  A cached judge verdict is never served as facts, or facts as a verdict.

`llm.enricher` names a plugin registered under the `rolescan.enrichers`
entry-point group that does extra work only for postings that clear
`min_report_score`; a failing enricher keeps the plain verdict.
`llm.extra_prompt` is appended to the judge-mode prompt only. In facts mode it
is not sent to the scoring call (an enricher may use it), and rolescan logs a
note at startup when it is set with no enricher configured.

With `llm.enabled: false`, or with no key and no local model, you get stage one
alone. If a configured backend fails, the digest says so rather than quietly
serving keyword scores that look like a normal run. There is a test that
asserts exactly this.

## What gets hidden, and where to look

rolescan leaves postings out of your digest, and it reports most of what it
leaves out. Where to look for each kind:

| Why a posting is not in your digest | Where to look |
|---|---|
| Its keyword score was under `min_keyword_score`, so no model saw it | The stats line counts it ("N filtered before scoring"). It is not listed. To see more, lower the number and re-assess (see below). |
| Its score was under `min_report_score` | The stats line counts it ("N below min_report_score"). It is not listed. Lower the number and re-assess (see below). |
| A `profile.rules` rule skipped it, or the advert states a nationality or clearance bar | **Hidden by your rules**, one line each: the rule, and the advert's own words. |
| A `hard_blockers` term or an `excluded_locations` entry matched | Shown at the bottom, marked BLOCKED, while `output.show_blocked` is `true` (the default). With `false` it is removed and listed under **Hidden by your rules**, and the stats line says "N blocked and hidden". |
| Weighted `blockers` pushed it under the keyword gate, or it fell within `profile.hidden_gate_margin` of the gate | **Hidden by your rules**, under "Your weighted terms" or "Just under your keyword gate". |
| It is waiting its turn (model budget, digest cap, no text yet) | "N deferred to the next run" in the stats line. A deferred posting is not recorded as seen, so it comes round again. |

The **Hidden by your rules** section appears only when something was hidden, and
the stats line then says "N hidden by your rules (listed below)". It is how you
catch a rule or a term that matches the wrong thing. Each group heading carries
its count. It looks like this:

```markdown
## Hidden by your rules

Skipped or blocked by one of your rules, so not listed above. One line each, so a wrong skip can be spotted.

**Years of experience** (1)

- **Example Co** · [Senior Data Engineer](https://example.test/jobs/12) · London: advert asks for "5+ years"

**Your blocking terms (hard_blockers, excluded_locations)** (1)

- **Example Plc** · [Data Analyst](https://example.test/jobs/31) · Leeds: blocked by "security clearance"
```

**At volume, the section is capped and the alarms come first.** The digest lists
at most `output.hidden_max` hidden postings (60 by default), the highest-scoring
first, since a hide with a high score is the likeliest wrong one. Every rule's
count is still shown. When the cap cuts the list, the whole of it is written to
`<digest name>-hidden.md` beside the digest, the digest names that file, and the
stats line reads "N hidden by your rules (M listed below, all N in the file)".
Gmail clips an email past about 102 KB, so the digest opens with what needs
attention (a source that failed, went quiet, shrank or was cut short, a model that
did not run or whose rates jumped), then the roles, then this section. The roles
get the email's room first: if an email holds more role cards than fit, the later
ones become one line each, and the digest on disk keeps them all. This section
gets what the roles leave, so a high `hidden_max` never pushes a role out; the
email always keeps its heading, every rule's count and where the rest is, and as
many rows as fit, which the section says when it is fewer than the cap.

**Re-assessing needs more than `--dry`.** Every posting a real scan assessed is
recorded as seen, the ones under a threshold included, and a `--dry` run drops
seen postings like any other. So after a real scan, lowering a threshold and
re-running `--dry` finds nothing new. Either run `rolescan unsee URL` for a
posting you know of, or tune against a scratch store: point `output.db_path` at a
new file (and `output.dir` if you like), run `--dry` as often as you want, then
point it back.

**A hidden posting is listed once.** Everything a scan assesses is recorded as
seen, hidden postings included, and is not reported again; a `--dry` run records
nothing, so it lists them every time. Read the section on the day you scan: the
digest is also saved under `digests/`, and `rolescan show` reprints the latest
one. If a rule or term was wrong, fix it in `config.yaml`, then
`rolescan unsee URL` (the URL as the digest prints it) so the next scan reports
that posting again. Deleting `seen.db` re-assesses everything, but it also
clears your shortlist, applied and dismissed marks.

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

What keeps a model's cost down:

- Only postings that clear the keyword prefilter reach a model.
- A verdict is cached for `llm.cache_days` (30), so an unchanged posting is not
  paid for twice.
- `llm.max_calls_per_run` (60) is a hard ceiling per scan. Postings past it are
  deferred to the next run, not dropped.
- A call sends the fixed instructions, your `profile.summary`, any worked
  examples, and one posting's title, company, location, date and the first
  `llm.description_chars` characters of its description (6,000 by default): a
  few thousand tokens in, a few hundred out.

One cost has been measured. With `claude-haiku-4-5` in facts mode, two
evaluation runs of 97 postings each, on 8 October 2026, cost USD 0.29 and
USD 0.30: about USD 0.003 a posting. That is one model on one set of adverts.
Yours depends on the model you choose and how long your postings are, so read
your provider's usage after one `scan`.

If a model is configured but cannot run (no key, the SDK missing, the server
unreachable, a context window too small), the scan says so at the top of its
output and in the digest, ranks on keywords instead, and exits 3. It does not
pretend.

**Model health.** In facts mode the quote guard drops what a posting does not
say, and code passes then set the level and the field, fill the years required
and add a hard bar the model missed. A lot of what reaches the digest is that
correction, so each scan that used a model keeps one row of counts (`llm_runs`
in `seen.db`: calls, cache hits, errors, and how many postings the guard or a
pass changed) and compares them with the median of the last three to five runs
(as many as exist) of the same backend and model. When a rate is more than twice
that median, over at least five postings, the digest's "Needs attention" block
adds a "Model health" line: the model, its prompt or a pass may have changed. It
says nothing until three earlier runs exist, and a `--dry` run compares but keeps
nothing.

## Privacy and storage

- **What leaves your machine.** Only with `llm.backend: anthropic`: your
  `profile.summary`, the optional `llm.facts_examples_file`, and for each
  posting that passes the prefilter its title, company, location, date and the
  first `llm.description_chars` characters of its description. `profile.name` is
  not sent. With `ollama` and with keyword-only scoring, nothing is sent to a
  model provider. The sources are fetched over HTTPS from the employers' own
  public endpoints.
- **Where keys go.** Environment variables: `ANTHROPIC_API_KEY`,
  `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`, `REED_API_KEY`, `JOOBLE_API_KEY`,
  `ROLESCAN_SMTP_PASS`. You can instead write `llm.api_key`,
  `output.email.password`, an Adzuna `app_id` and `app_key`, or a Reed or
  Jooble `api_key` in `config.yaml`; that file is gitignored, so keep it out of version control and
  out of any backup you share. rolescan does not read a `.env` file: export the
  variable in your shell or scheduler.
- **What is stored.** `seen.db` (SQLite, next to `config.yaml` unless
  `output.db_path` says otherwise) holds every posting a scan assessed (URL,
  title, company, location, source), cached model answers, cached posting pages
  from sitemap sources, per-source counts, where each incremental source left
  off (`source_marks`), per-run model counts (`llm_runs`, no posting text), and
  your `mark` decisions.
  `digests/` holds each scan's digest, and `backups/` beside the store holds
  daily copies of it (`output.backup_keep`). All of it describes your job
  search, so treat it as private; `seen.db`, `digests/` and `config.yaml` are
  gitignored by default. Delete them to forget everything.
  `output.retention_days` trims the caches, never `seen` or your marks.
- **Email.** If enabled, the digest is sent over SMTP with the host and
  credentials you give: STARTTLS on port 587 by default, implicit TLS on 465.
  The server's certificate must verify against your system's trusted
  authorities, on both; a relay with a self-signed certificate is refused.
- **What rolescan requests.** Public job-board JSON endpoints and the public
  sitemap and job pages you list, with the User-Agent
  `rolescan/<version> (+<http.contact_url>)`. No logins, no cookies, nothing
  behind a form. `http.max_concurrent` and a per-source `delay` pace the
  requests. `robots.txt` is read in one place: a `structured` source asks the
  host for its `Crawl-delay` and keeps `delay` at least that long. The core does
  not check any `Disallow` rule, for any source, so add only sites whose terms
  let you read them this way, and keep `delay` generous.

## Measured quality

Two measurements are recorded, each stated here once. Neither is a benchmark.

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

**The local backend, recorded 24 September 2026, in judge mode.** A 25-posting
benchmark against hand-labelled expectations, plus one live 34-posting scan, in
the older `llm.mode: judge`, before facts mode existed: **verdict accuracy 72%**,
no score/verdict disagreements, and **model-level blocker recall 50%**, so the
model named half the structural bars in the benchmark set. The record does not
name the model or the hardware, and the labelled set is private. Treat blocker
recall as the number to design around: it is why `hard_blockers` exist.

Nothing is recorded here for the default facts mode, or for the Anthropic
backend, which is exercised by mocked tests only and has not been benchmarked.

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

`mark` takes one of three states. `shortlist` keeps a posting in the digest's
**Shortlist** section, which is repeated under the new roles on every run so an
open application stays in front of you rather than scrolling away with
yesterday's email. `applied` clears it from that section. `dismissed` clears it
too, and additionally stops the posting ever being reported again, however many
boards go on listing it. Every shortlist row in the digest carries the exact
`mark` command to clear it, with the config path already filled in.

`scan --dry` marks nothing seen and writes `digest-dry.md`, leaving `latest.md`
as the last real run's digest. `unsee` takes a posting's URL (as the digest
prints it) or its uid and removes it from the seen list, so the next scan can
report it again.

`rolescan scan` tells a scheduler how the run went. Exit status: 0 ok; 1
email failed, the day's backup failed its check, or the store was written by
a newer rolescan (takes precedence over 3); 2 config error; 3 LLM scoring
failed as a whole (after the digest was written and sent); 4 another run
holds the store's lock. The digest says why. The store gets a checked copy in
`backups/` beside it once a day (`output.backup_keep`, default 7), and its
caches are trimmed after each real scan (`output.retention_days`).

If the digest is written but the email times out, a VPN may be blocking the mail
ports: set `output.email.bind_interface` to the interface that reaches the
internet directly (for example `en0`) and the mail socket binds to that
interface's address, looked up at every send (port 465 uses implicit TLS, any
other port STARTTLS).

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

- **A local model can miss structural bars.** The one recorded measurement is
  under [Measured quality](#measured-quality). Configured `hard_blockers` do not
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
