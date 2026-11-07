# factgate

[![ci](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml)

Decisions on any pile of text documents that you can audit line by line. A
model reads each document and reports the facts you asked for, each with the
exact words it read them from. Code checks that every quote is really in the
document, then rules you write in YAML decide the outcome. **The model reports;
code decides.**

Asking a model "approve or review?" gives an answer you cannot check, cannot
hold steady and cannot test. factgate splits that question in two. The model
does the reading, which is what it is good at. A few lines of YAML do the
deciding, and they always give the same answer for the same facts:

```yaml
facts:
  amount_due: {type: number, describe: "The total amount due on the invoice"}
  disputed:   {type: bool,   describe: "The email says the invoice is disputed or on hold"}
rules:
  - name: disputed
    when: {disputed: {is: true}}
    then: hold
  - name: large
    when: {amount_due: {gt: 10000}}
    then: review
default_outcome: approve
```

What you get from every run:

- **Every decision names the rule that fired (or says the default decided) and
  quotes the document.**
- **A fact whose quote is not in the document is cleared, never guessed.** A
  number must also appear in its own quote, whole and with its sign, so "total
  due £12,540.00" read as 125400, or as 540, is cleared ("£40k" counts as
  40,000), and a date's year must match the year its quote states, when it
  states one.
- **Spend has hard limits.** A keyword prefilter runs before any model call.
  Once a run has made `max_calls` calls or spent `max_minutes`, no new call
  starts (one already in flight finishes). Every item ends as exactly one of
  `decided`, `filtered`, `deferred` or `error`, and the output says which:
  nothing is dropped.
- **Nothing is paid for twice.** Answers are cached on the item's text and a
  fingerprint of the prompt, the schema, the model and its server, so a repeat
  run is free and a changed prompt never gets an old answer. The quote check and the rules
  run again on every read, so changing a rule costs no model calls.
- **Graded cases** run offline in CI and live against a model, with repeats,
  held-out cases and a before-and-after comparison.

factgate is the method behind [rolescan](../README.md), the job scanner in this
repository, with the job adverts taken out.

## Quickstart: no key, no model, under a minute

```bash
git clone https://github.com/younis-y/deterministic-llm-pipeline
cd deterministic-llm-pipeline/factgate
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
factgate run examples/invoices/config.yaml --replay examples/invoices/answers.jsonl --as-of 2026-11-07
```

`--replay` reads model answers recorded earlier, so nothing is sent anywhere.
`--as-of` fixes the date the rules treat as today, so an invoice due on 30
November 2026 does not turn overdue when you run this next year. It prints:

```
# invoice-triage

8 items: 7 decided, 1 filtered, 0 deferred, 0 errors.
7 answers replayed from a file, no model called; 0 from cache.

| Outcome | Items |
|---|---|
| review | 4 |
| approve | 2 |
| hold | 1 |

| Rule | Items |
|---|---|
| (default: approve) | 2 |
| large | 1 |
| disputed | 1 |
| no-po | 1 |
| not-sterling | 1 |
| overdue | 1 |
```

and writes one JSON line per invoice to `examples/invoices/out/results.jsonl`
(shortened here):

```json
{"id": "inv-02", "status": "decided", "outcome": "review", "rule": "large",
 "quotes": {"amount_due": "total due £12,540.00"},
 "facts": {"amount_due": {"value": 12540.0, "quote": "total due £12,540.00"}, ...}}
```

That invoice says "Subtotal £10,450.00, VAT £2,090.00, total due £12,540.00",
and the decision quotes the total its rule read. The credit note among the eight never reached
a model: the prefilter dropped it.

## Your own documents

```bash
factgate init my-triage          # a starter config, 3 items, recorded answers, 2 cases
factgate check my-triage/config.yaml   # the exact prompt and JSON schema the model gets
factgate run my-triage/config.yaml --dry   # how many items would be sent; no model call
factgate run my-triage/config.yaml         # needs the anthropic extra and a key: see Providers
```

A config is one YAML file. The invoice example, shortened (the full file is
[`examples/invoices/config.yaml`](examples/invoices/config.yaml)):

```yaml
name: invoice-triage
items: {jsonl: items.jsonl, id: id, fields: [subject, body]}

prefilter:                  # optional; whole words, any field, no model
  require_any: [invoice]
  exclude_any: [credit note]

facts:                      # what the model reports, each with a quote
  amount_due: {type: number, describe: "The total amount due (not a subtotal or VAT line)"}
  currency:   {type: enum, values: [GBP, EUR, USD, other]}
  due_date:   {type: date, describe: "The date payment is due"}
  disputed:   {type: bool, describe: "The email says the invoice is disputed or on hold"}
  po_number:  {type: text, describe: "The purchase order number the invoice quotes"}

rules:                      # first match wins; conditions in one rule are ANDed
  - {name: disputed,   when: {disputed: {is: true}},       then: hold}
  - {name: large,      when: {amount_due: {gt: 10000}},    then: review}
  - {name: no-po,      when: {po_number: {stated: false}}, then: review}
  - {name: no-amount,  when: {amount_due: {stated: false}}, then: review}
  - {name: overdue,    when: {due_date: {before_days: 0}}, then: review}
default_outcome: approve

dangerous:                  # mistakes the eval reports first
  - {expected: hold, got: approve}

model: {backend: anthropic, name: claude-haiku-5-5, max_calls: 50, max_minutes: 10}
```

**Items** come from one reader: `jsonl: file`, `csv: file`, `folder: dir` (each
`.txt` or `.md` file is an item, its name the id) or `stdin: true` (JSONL), or a
plugin under the `factgate.sources` entry-point group. `fields` names the text
fields; each one is checked for quotes on its own, so a quote cannot join the
end of the subject to the start of the body. `meta` names fields passed
through to the output untouched.

**Fact types**: `number`, `integer`, `text`, `bool`, `date` (ISO), `enum` (with
`values`), and `list` of any of them (with `of`). Every fact may be "not
stated": the model says so with a null value, and the quote check says so for
any value it cannot find.

**Rule conditions**:

| Operator | Applies to | True when |
|---|---|---|
| `is`, `eq`, `ne` | every type | equal / not equal |
| `lt`, `le`, `gt`, `ge` | number, integer, date | the comparison holds |
| `in` | every type but bool | the value is one of a list |
| `contains`, `matches` | text, and lists of text | a substring is there / a regex finds a match |
| `stated` | every type | `true`: the fact was found; `false`: it was not |
| `within_days: N` | date | between today and N days ahead |
| `before_days: N` | date | earlier than N days from today; `0` is "in the past" |

A condition on a fact that is not stated is false, except `stated: false`. On a
list, an operator holds when any entry meets it, `contains` means an entry
equals the argument and `stated` means the list is not empty. A config with a
rule on an unknown fact, an unknown operator, an operator or argument that does
not fit the fact's type (`gt: "10,000"`, `eq: GPB` for an enum of GBP and EUR)
or a bad regex fails to load, naming the rule.

**Make "not stated" safe.** A fact the quote check clears becomes "not stated",
and a rule on it is then false. If nothing else catches it, the item falls
through to `default_outcome`. That is why the invoice example sends an
unstated amount, due date or currency to review, and the tender example an
unstated deadline or value: otherwise a misread amount would be cleared and the
invoice approved.

## How a run works

1. Read the items and hash each one's text.
2. Prefilter: an item that fails ends as `filtered`, with the term that did it.
3. Cache: an item answered before, under the same prompt, schema, model and
   server, is served from `.factgate/cache.sqlite`.
4. Model: the rest are sent, `max_concurrent` at a time, until `max_calls` or
   `max_minutes` runs out. What does not fit ends as `deferred` and is answered
   on a later run. Three errors in a row stop the run and defer the rest.
   `--record answers.jsonl` keeps every answer as it arrives.
5. Quote check: each fact's quote must appear in one field of the item (spaces,
   case and typographic quotes and dashes folded), or the fact is cleared.
6. Resolvers (optional): your Python functions correct the checked facts, and
   the quote check runs again on what they return.
7. Rules: the first rule whose conditions all hold decides; else the default.
8. Output: `out/results.jsonl` (id, status, outcome, rule, the quotes the rule
   read, every fact with its quote, meta, and why for anything not decided)
   and a Markdown summary.

An answer that does not fit the schema (a bad enum value, broken JSON), or a
resolver that raises on one item, makes that item `error` and the run goes on.
`factgate run` exits 1 when any item errored. Two items with one id are an
error before anything runs.

## What the quote check does not catch

- **A value that is in the document but is the wrong one.** "Previous balance
  £4,000.00 (paid)" is a real quote with a real number in it. A resolver can
  catch it: the invoice example ships one,
  [`fixes.py`](examples/invoices/fixes.py), that swaps an amount read from a
  subtotal, a VAT line or an earlier balance for the email's own "total due"
  figure.
- **An enum, text or bool value read wrongly from a real quote.** GBP read from
  "Amount due EUR 3,800.00" gets through. The invoice example keeps this as an
  open defect so that CI holds it in view (see Evals).
- **A negation.** "SC clearance is not required" contains "SC clearance". The
  model has to read it as `false`, and the tender example's cases check that it
  does.

Resolvers are named in the config as `module:function`, and a module next to
the config loads as well as an installed one. They take the item and the
checked facts and return corrected facts. They run your Python, so only run a
config you trust, as with any script.

## Evals

A case is an item plus what its finished facts and outcome must be, written
with the same operators as rules:

```json
{"id": "inv-earlier-balance", "fields": {"subject": "Invoice 3319 from Kestrel IT", "body": "... Previous balance £4,000.00 (paid, thank you). Amount now due: £860.00 by 12 December 2026."},
 "expect": {"facts": {"amount_due": {"eq": 860}}, "outcome": "approve"},
 "model_facts": {"amount_due": {"value": 4000, "quote": "Previous balance £4,000.00"}, "...": "..."},
 "tags": ["amounts", "resolver"], "as_of": "2026-11-07"}
```

`model_facts` is a model's answer, recorded or written by hand and often wrong
on purpose. Case files are JSONL or YAML, one area per file, and are merged on
load. A bad case fails to load with its `file:line`: invalid JSON, a duplicate
id across files, an unknown fact, operator or outcome, or a case with nothing
to grade. [Writing cases](docs/writing-cases.md) covers how to write good ones.

**Offline, in CI.** Each case's `model_facts` goes through the quote check,
the resolvers and the rules, with no model:

```bash
factgate eval run examples/invoices/config.yaml examples/invoices/cases.jsonl
```

Or inside your own pytest suite:

```python
import pytest
from factgate.eval.pytest_helper import assert_case, case_params


@pytest.mark.parametrize("bound", case_params("config.yaml", "cases.jsonl"))
def test_case(bound):
    assert_case(bound)
```

A case that cannot be graded (an answer that does not fit the schema, a
resolver that raises) fails CI. A case marked `open_defect` is a known
failure. It runs as a strict expected failure, so fixing it fails CI until the
mark is removed, and the fix is held from then on. A case marked
`offline_only` grades what the code does with its own answer (a misread number
the quote check must clear, say), so a live run leaves it out. A misspelt key
in a case fails to load.

**Live, with repeats.** `--live --repeats 3` asks the model three times per
case and gives each case one state, not an average. A live eval makes cases ×
repeats calls and is not held to `max_calls`, so count before you run it.

| State | Means |
|---|---|
| `pass` | every attempt passed |
| `fail` | no attempt passed |
| `flaky` | some passed and some failed |
| `inconclusive` | an attempt could not be graded: the backend raised or the answer did not fit the schema |

Inconclusive is never a pass. It is counted outside the pass rate, never in
it. A report gives pass@1 (passed attempts over conclusive attempts) and
pass^k (cases that passed every time), with a Wilson 95% interval. For each
failure it says whether a fact or only the outcome missed, and for a flaky case
whether the model's answers changed between attempts (model noise) or stayed
the same (a rule or grading problem). Dangerous mistakes are listed first.

**Held-out cases** go in a separate file passed with `--heldout`. They are run
and scored, but the report shows only their ids and states, so whoever tunes
the prompt, resolvers or rules cannot tune to them. That independence is by
honour: the file is right there, and a run's result file keeps the model's
answers, quotes included, for held-out cases too. The invoice example's
held-out set is written by an oracle,
[`gen_heldout.py`](examples/invoices/gen_heldout.py), which computes each
right answer from the rules on the values it put into the text, three per rule
and three exactly on a threshold.

**Compare and regrade.** Every live attempt saves the model's raw answer:

```bash
factgate eval compare results/before.json results/after.json
factgate eval regrade results/before.json config.yaml cases.jsonl --heldout heldout.jsonl
```

`compare` lists the cases that moved. Only fail to pass counts as fixed and only
pass to fail as broken; a move into or out of `flaky` is noise, held-out moves
are counted apart from visible ones, and a case in only one of the runs is
listed as such. `regrade` re-scores a run's saved answers with today's config
and cases, on the date the run used, so a rule or resolver change is measured
with no model calls and no model noise.

### Measured

Recorded on 7 November 2026 with `claude-haiku-5-5`, 3 runs of every case,
through `factgate eval run ... --live --repeats 3`. The results are in each
example's `results/` folder.

| Example | Decides | Cases run live (visible + held-out) | Live result | Offline, in CI |
|---|---|---|---|---|
| [invoices](examples/invoices) | approve, review or hold | 32 (11 + 21) | pass@1 100%, pass^3 100%, Wilson 95% 89%-100% | 12 of 13 pass, 1 open defect |
| [support-tickets](examples/support-tickets) | which team, or page on-call | 14 (9 + 5) | pass@1 100%, pass^3 100%, Wilson 95% 78%-100% | 9 of 9 pass |
| [tenders](examples/tenders) | bid, no-bid or review | 12 (7 + 5) | pass@1 97%, pass^3 92%, Wilson 95% 65%-99% | 9 of 9 pass |

The 174 live calls took 69 seconds and 34,795 output tokens, with 279,987
input tokens counting the prompt read back from the cache: at most USD 0.05.
The four offline-only cases did not run live.

One case is flaky, and it shows what "make not stated safe" is for. The tender
"Budget up to £18,000. Responses due 30 December 2026." is a no-bid: under
£50,000. In one attempt of three the model reported no value, and in none the
deadline, so the `no-deadline` rule sent it to review. Without that rule it
would have fallen through to `bid`, the config's own dangerous mistake.

Near-perfect scores on 58 short synthetic cases mean the model handles these
traps (a subtotal next to the total, a paid balance, "Annoying!!!" that is not
urgency, an injected instruction, a clearance "not required"). They do not
mean that on your documents: the intervals show how little 12 to 32 cases can
say. A test rebuilds every figure in this table from the result files.

## Providers

| Backend | For | Needs |
|---|---|---|
| `anthropic` | Claude, with structured output and the prompt cached | `pip install -e ".[anthropic]"`, `ANTHROPIC_API_KEY` |
| `openai-compatible` | any server with the OpenAI chat API and JSON-schema output: Ollama (checked), vLLM, LM Studio, OpenAI (not tested) | `model.base_url`, and `model.api_key_env` if the server wants a key |
| `replay` | answers recorded with `factgate run --record answers.jsonl` | the file, passed as `--replay` |

A local model through Ollama, with no key and no bill:

```yaml
model: {backend: openai-compatible, name: "qwen2.5:14b", base_url: "http://localhost:11434/v1"}
```

[Providers](docs/providers.md) has the details, and how to add a backend as a
plugin under the `factgate.backends` entry-point group.

## From Python

```python
import asyncio
import datetime as dt
from pathlib import Path

from factgate.backends import get_backend
from factgate.config import load_config
from factgate.items import read_items
from factgate.pipeline import load_resolvers, run

cfg = load_config(Path("examples/invoices/config.yaml"))
backend = get_backend(cfg.model, replay=Path("examples/invoices/answers.jsonl"))
results, stats = asyncio.run(
    run(
        cfg,
        read_items(cfg.items, cfg.base),
        backend,
        as_of=dt.date(2026, 11, 7),
        resolvers=load_resolvers(cfg.resolvers, cfg.base),
    )
)
for r in results:
    print(r.id, r.status, r.outcome, r.rule)
```

## Not in this version

- Documents other than text. Converting a PDF to text is your step.
- A facts model written in Python instead of `facts:` in YAML.
- Batching calls through a provider's batch API.
- Installing from PyPI: install from this repository, as above, or with
  `pip install "git+https://github.com/younis-y/deterministic-llm-pipeline#subdirectory=factgate"`.

## Credits

The eval borrows from the harness built by the team at **Judge the Agent**, The
Friendly Data Meetup London, 4 November 2026
([computator1200/parcel-pigeon-evals](https://github.com/computator1200/parcel-pigeon-evals),
on the challenge material from
[dl-lim/friendly-meetup-hackathon](https://github.com/dl-lim/friendly-meetup-hackathon)):
one state per case across repeats, inconclusive kept out of the pass rate,
held-out cases, compare by transitions, regrade of saved answers, the split of
dangerous mistakes from the rest, and checking a README's claims against the
result files. Only the parts Younis Younis wrote there are adapted here as code (the
repeat runner's states and the held-out oracle); the rest are reused as ideas.
That harness graded a chatbot with an LLM judge. factgate grades with code,
which is the point: a rule either holds or it does not.

## Licence

MIT.
