# factgate: the rolescan method as a plug-and-play library

Date: 2026-11-07. Status: draft for the owner's approval.

## Intent

rolescan proves a method: a model reads a document and reports facts, each with
the exact words it read them from; code checks every quote is really in the
document, then fixed rules decide the outcome. The model never decides. A cheap
keyword prefilter runs first, calls and spend are capped, nothing is paid for
twice, and a graded eval runs in CI.

Today the method only exists welded to job adverts: the `Job` model, the facts
(level, years, student-only), the rules and the digest are all job-specific, and
someone triaging invoices, tenders or support tickets would have to remove most
of the 14,000 lines. The owner wants the project to be a great tool anyone can
plug in for their own purpose: "feeding it anything that suits the
infrastructure".

factgate is that tool: the method, with the job domain taken out, configured in
YAML.

## Who it serves and what success looks like

A developer or analyst who has a pile of text documents and wants a reliable,
auditable yes/no/label on each one, cheaper and more predictable than "ask the
model".

Success:
1. A newcomer clones the repo and runs a shipped example end to end in under a
   minute, with no API key and no model (recorded answers, below).
2. They point it at their own documents by writing one YAML file (the facts to
   pull out and the rules), with no Python.
3. Every decision says which rule fired and quotes the document's own words.
4. They can write graded cases for their config and run them offline in CI and
   live against a model, with the same harness rolescan uses.

## Goals

- A config-driven pipeline: items in, prefilter, model extracts quoted facts,
  quote check, optional resolvers, rules, decision out.
- Facts declared in YAML. Python users may pass their own Pydantic model instead.
- Rules declared in YAML, evaluated by code, with no `eval` of user text.
- The rolescan guarantees, generalised: quote verification, prefilter before any
  call, call ceiling and wall-clock budget, deferral of what does not fit (never
  dropped), a cache keyed so a prompt, schema or model change is never served
  stale answers, a run log of calls, errors and token counts.
- An eval harness: cases with expected facts and decisions, an offline mode that
  feeds recorded or hand-written model answers through the code (for CI), a live
  mode reporting pass@1 and pass^k, and results written to a file.
- Three small shipped examples, each with sample items, a config, recorded model
  answers and eval cases.
- A README a stranger can follow, whose figures are checked by a test (the
  rolescan convention).

## Non-goals (v1)

- Migrating rolescan onto factgate. rolescan stays as it is and keeps working;
  the README points to it as the full worked example. Moving rolescan's generic
  parts (cache, spend cap, quote check) onto factgate is a later, separate
  piece of work.
- Publishing to PyPI. Install is from the repo (below). Publishing is
  outward-facing and needs the owner's say-so later.
- A web UI, a hosted service, an agent framework, fetching from websites.
- Images, PDFs or other non-text input. Items are text; converting a PDF to
  text is the user's step.
- Providers beyond the two backends below.

## Anti-goals (failure even if it technically works)

- The model decides the outcome, anywhere, by any path.
- A user expression is executed as code.
- An item disappears silently: every item ends as decided, deferred or errored,
  and the output says which.
- rolescan breaks, or its CI goes red.
- Personal data or the owner's private config lands in the public repo.

## How it works

### Items

An item is `{id, fields, meta}`: `fields` is one or more named text fields
(`subject`, `body`; a single field is called `text` by default), and `meta` is
free-form and passed through to the output. Each field is a separate haystack
for the quote check, never joined, so a quote cannot splice the end of one field
onto the start of the next (rolescan checks title and description this way). Built-in readers: a JSONL file (one object per line, field names
configurable), a CSV (id and text columns configurable), a folder of `.txt` or
`.md` files (id = file name), and stdin as JSONL. More readers load as
entry-point plugins (`factgate.sources`), the same pattern rolescan uses.

### Config (one YAML file)

```yaml
name: invoice-triage
items: {jsonl: invoices.jsonl, id: invoice_no, fields: [subject, body]}

prefilter:            # optional; plain keyword rules, no model
  require_any: [invoice, "amount due"]
  exclude_any: [credit note]

facts:                # what the model must report, each with its quote
  amount_due: {type: number, describe: "Total amount due, in the invoice currency"}
  due_date:   {type: date,   describe: "Payment due date"}
  currency:   {type: enum, values: [GBP, EUR, USD, other]}
  disputed:   {type: bool,   describe: "The invoice says it is disputed or on hold"}

rules:                # first match wins; conditions in one rule are ANDed
  - name: disputed
    when: {disputed: {is: true}}
    then: hold
  - name: large
    when: {amount_due: {gt: 10000}}
    then: review
  - name: overdue
    when: {due_date: {before_days: 0}}
    then: review
default_outcome: approve

model: {backend: anthropic, name: claude-haiku-5-5, max_calls: 200, max_minutes: 20}
cache: .factgate/cache.sqlite
```

Python users may pass their own Pydantic model instead of `facts:`; every
field must be a `Fact[T]` (value plus quote) from factgate, with `T` one of the
types below, so the quote check, the prompt and the rule conditions all apply.

Fact types: `number`, `integer`, `text`, `bool`, `date` (ISO), `enum`, and
`list` of any of these. Every fact is optional: the model may report "not
stated", and a fact whose quote is missing from the item is cleared to "not
stated" by the quote check, never guessed.

Rule conditions per fact: `is`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`,
`contains`, `matches` (a regex, checked when the config loads), `stated`
(true/false), and for dates `within_days: N` (between today and N days ahead)
and `before_days: N` (earlier than N days from today; `0` is "in the past").
"Today" is the run's `as_of` date: today by default, fixed by `--as-of` or in an
eval case, so replays and offline evals do not depend on the clock. A rule's `then` is any outcome label the user chooses. A
condition on a fact that is not stated is false, except `stated: false`.

### Runs

Every run reads the whole input and outputs every item. The cache makes a
repeat free: an item answered before is not sent again. There is no
seen-tracking in v1. An item `deferred` by the call or time limit is simply
answered on a later run, when it is no longer competing with the items the
cache now serves. A run is named by its start time in the summary.

### Pipeline

1. Read items. Hash each item's text.
2. Prefilter: items failing it end as `filtered` with the term that did it. No
   model call.
3. Cache lookup by key `hash : schema+prompt fingerprint : backend : model`.
4. Model call for the rest, within `max_calls` and `max_minutes`; what does not
   fit ends as `deferred` and is first in line next run.
5. Quote check: each fact's quote must appear in the item text (after the same
   whitespace, case and typographic-quote folding rolescan uses), else the fact
   is cleared. A quote is a list when the fact is a list, one per element.
   For `number` and `integer` facts the value must also be in its own quote:
   the quote's numbers, read with thousands separators and currency symbols
   removed, must include the value, so "Total due: £12,500" answered as
   125000 is cleared (rolescan does the same for graduation years). `text`,
   `bool`, `enum` and `date` values are not value-checked in v1; a resolver can
   add a check, and the README says so plainly. The cache stores the model's
   raw answer, and the check runs on every read, as in rolescan (ADR-0008).
6. Resolvers (optional): Python functions named in the config
   (`resolvers: [mypkg.fixes:fix_amounts]`) that take the item and the checked
   facts and return corrected facts. This is rolescan's "code corrects the
   model" step, opened up. The quote check runs again on what resolvers
   return, so a resolver must quote the item verbatim like the model does,
   and every decision still quotes the document.
7. Rules: first match wins, else `default`.
8. Output: one JSONL line per item with id, outcome, the rule that fired, the
   facts with their quotes, `meta`, and status (`decided`, `filtered`,
   `deferred`, `error`). A short Markdown summary: counts per outcome, per rule,
   calls, cache hits, errors, tokens.

### Backends

- `anthropic`: structured output against the facts schema, prompt caching,
  optional batch mode, as rolescan does today.
- `openai-compatible`: one HTTP backend for any server speaking the OpenAI chat
  API with JSON-schema output. This covers Ollama, vLLM, LM Studio and OpenAI
  itself with one piece of code, configured by `base_url`.
- `replay`: answers from a recorded file. Used by the examples, the offline
  eval and the no-key quickstart.

More load as entry-point plugins (`factgate.backends`).

### CLI

- `factgate run config.yaml` (flags: `--replay answers.jsonl`, `--dry` to stop
  before any model call and show what would be sent, `--out`).
- `factgate eval config.yaml cases.yaml [--live --runs 3]`.
- `factgate init <name>`: writes a starter config, three sample items and two
  cases into a new folder.
- `factgate check config.yaml`: validates the config and prints the generated
  prompt and schema.

### Eval

The eval combines rolescan's `evals/` with the harness the owner's team built
at the Judge the Agent hackathon (Friendly Data Meetup, 4 November 2026,
github.com/computator1200/parcel-pigeon-evals). That repo has no LICENSE file,
so only the owner's own code from it (`evals/intensive.py`, `evals/gen_orders.py`,
the ladder) may be copied; everything else is reused as ideas and credited in
the README. Its LLM judge, rubric and judge-consistency tools do not apply:
factgate grades by code, and the README says that is the point.

**Cases.** One YAML or JSONL file per area (`cases-*.jsonl`), merged on load.
A case is an item, plus what the finished facts and the outcome must be,
written with the same operators as rules (`amount_due: {gt: 10000}`,
`outcome: hold`), plus optional `model_facts` (a recorded or hand-written model
answer for the offline run), `tags` and `open_defect`. Loading fails loudly,
with `file:line`, on invalid JSON or YAML, a duplicate id across files, an
unknown fact or operator, and a case with nothing to grade (the hackathon's
`merge_cases` rules).

**Offline (CI).** Every case with `model_facts` goes through quote check,
resolvers and rules; no model. A pytest helper runs a user's cases inside their
own suite; `open_defect` cases run as strict expected failures.

**Live.** `factgate eval --live --repeats 3` runs each case k times and gives
each ONE state, never an average (the hackathon's `intensive` rules):
- `pass`: every attempt passed; `fail`: none did; `flaky`: both.
- `inconclusive`: an attempt errored (API error, timeout, cut-off answer).
  Inconclusive is never a pass and is counted outside the pass rate.
- For every failure: whether it failed on a fact or only on the outcome, and
  for a flaky case whether the model's facts changed between attempts (model
  noise) or stayed the same (a grading or rule problem).
- pass@1 and pass^k as in rolescan, with a Wilson 95% interval on the pass
  rate, stating the method.

**Held-out.** Cases in a `heldout/` file are run and scored but shown as ids
and totals only, so the person tuning the prompt, resolvers or rules cannot
tune to them. The README states independence is by honour.

**Generated cases.** A documented pattern (and the invoices example uses it):
a small Python oracle writes held-out cases whose right answer is computed from
the rules, with new values and wordings, and writes no case where the rules do
not decide (from `gen_orders.py`).

**Compare.** `factgate eval compare <run-a> <run-b>`: per case, before and
after state. Only stable fail to stable pass counts as fixed and stable pass to
stable fail as broken; moves into or out of flaky are listed separately as
noise, and the held-out transitions are reported apart from the visible ones.

**Regrade.** `factgate eval regrade <run>` re-scores the model answers saved in
a run with the current quote check, resolvers and rules: a rule or resolver
change is measured with zero model noise and zero cost (the ladder's
`regrade`). Every live run therefore saves each attempt's raw model answer.

**Costly errors.** A config may name dangerous mistakes:
`dangerous: [{expected: hold, got: approve}]`. Reports split wrong outcomes
into dangerous and other, and a dangerous error is listed first and named in
the summary (the hackathon's DANGEROUS / CONSERVATIVE split).

**Reports.** Each run writes `results/<label>.json` (every attempt) and
`results/<label>.md` (states per tag, the visible and held-out totals,
inconclusive, dangerous errors, seconds, calls, tokens). A test helper checks
that figures quoted in a README match the latest result file (rolescan's
`test_readme_claims`, and the hackathon's claims check).

**Writing cases.** `docs/writing-cases.md`: one behaviour per case; take the
truth from the document, not memory; make a case fail for the right reason;
and a checklist of traps generalised from rolescan and the hackathon (missing
values, values that look like the fact but are not, caps and ranges,
negations, boilerplate, injection text inside the item, off-topic items).

### Examples (shipped)

- `examples/invoices`: approve / review / hold, with a disputed invoice and a
  credit note the prefilter drops.
- `examples/support-tickets`: route to billing / bug / account / other, with an
  "urgent" flag only when the ticket says so in words.
- `examples/tenders`: bid / no-bid on public-sector tenders by deadline, value
  and a mandatory certification.

All sample text is synthetic. Each ships recorded answers so it runs with no
key, and a live result recorded on `claude-haiku-5-5` for the README.

## Where it lives

In this repo, as a separate package in `factgate/` with its own `pyproject.toml`
and tests, so `rolescan` is untouched. Install:
`pip install "git+https://github.com/younis-y/deterministic-llm-pipeline#subdirectory=factgate"`.
CI gains a job for it with the same gates rolescan has (ruff, strict mypy,
pytest, vulture, deptry). The top-level README opens with factgate as the
library and rolescan as the flagship application built on the method.

Dependencies: pydantic, pyyaml, httpx, typer, aiosqlite; `anthropic` as an
extra. Python 3.11+.

## Decided by the owner

- Make the project plug-and-play for any purpose; proceed with this shape: a
  library, YAML facts and rules, rolescan as the flagship example.
- Use everything that applies from the Judge the Agent hackathon repo.

## My calls (the owner can overturn any)

- The name `factgate`.
- rolescan is not migrated in v1.
- No PyPI in v1; install from the repo subdirectory.
- `openai-compatible` instead of a dedicated Ollama backend.
- YAML condition operators instead of an expression language.
- The three example domains.
- Which hackathon pieces apply, and that the LLM-judge tooling does not.

Reviewer questions settled as my calls: named text fields per item; numeric
values checked against their own quote; relative dates through an `as_of`
date; resolver output re-checked; a Python schema must use `Fact[T]`; no state
between runs beyond the cache. The live recordings use the owner's existing
Anthropic key, about USD 0.02 per example run.

## Left to the builder

Module layout, internal class names, the exact prompt wording (it must state
the facts, their types and descriptions, and require a verbatim quote for each),
the Markdown summary layout, retry and timeout values (take rolescan's), and how
the replay file is keyed.
