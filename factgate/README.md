# factgate

[![ci](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/younis-y/deterministic-llm-pipeline/actions/workflows/ci.yml)

factgate turns text documents into decisions you can audit: approve or review
an invoice, route a support ticket, bid on a tender. A model reads each
document and reports the facts you ask for, each with the exact words it read
them from. Code checks that every quote is really in the document, then rules
you write in YAML decide. **The model reports; code decides.**

Ask a model "approve or review?" and you get an answer you cannot check, hold
steady or test. With factgate the model only reads. A few lines of YAML decide,
and the same facts always get the same decision:

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

- **Every decision names its rule**, or says the default decided, and quotes
  the document.
- **A fact whose quote is not in the document is cleared, never guessed.** A
  number must also appear whole in its own quote, sign included, and a date's
  year must match any year its quote states.
- **Spend has hard limits.** A keyword prefilter runs before any model call,
  and `max_calls` and `max_minutes` cap each run. Every item ends `decided`,
  `filtered`, `deferred` or `error`: none is dropped.
- **Nothing is paid for twice.** Answers are cached against the item's text,
  the prompt, the schema and the model. A repeat run is free, a changed prompt
  never gets an old answer, and a rule change makes no model calls.
- **Graded cases** run offline in CI and live against a model, with repeats,
  held-out cases and a before-and-after compare.

factgate is the method behind [rolescan](../README.md), the job scanner in this
repository, with the job adverts taken out.

## Quickstart: no key, no model

You need Python 3.11 or newer.

```bash
git clone https://github.com/younis-y/deterministic-llm-pipeline
cd deterministic-llm-pipeline/factgate
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
factgate run examples/invoices/config.yaml --replay examples/invoices/answers.jsonl --as-of 2026-11-07
```

`--replay` uses model answers recorded earlier, so nothing is sent anywhere,
and `--as-of` fixes the date the rules treat as today. It prints:

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

It also writes one JSON line per invoice to
`examples/invoices/out/results.jsonl`:

```json
{"id": "inv-02", "status": "decided", "outcome": "review", "rule": "large",
 "quotes": {"amount_due": "total due £12,540.00"}, "facts": {...}}
```

That invoice says "Subtotal £10,450.00, VAT £2,090.00, total due £12,540.00",
and the decision quotes the total its rule read. The credit note among the
eight never reached a model: the prefilter dropped it.

## Your own documents

```bash
factgate init my-triage                    # a starter config, 3 items, recorded answers, 2 cases
factgate check my-triage/config.yaml       # the exact prompt and JSON schema the model gets
factgate run my-triage/config.yaml --dry   # how many items would be sent; no model call
factgate run my-triage/config.yaml         # needs a model: see Providers
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

Rules compare facts with `is`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `in`,
`contains`, `matches`, `stated`, `within_days` and `before_days`. A config that
names an unknown fact or operator, or gives an argument that does not fit the
fact's type, fails to load and names the rule. The
[config reference](docs/config.md) covers every key: item readers, fact types,
operators, resolvers.

**Make "not stated" safe.** A fact the quote check clears becomes "not stated",
so every rule on it is false and the item falls through to `default_outcome`.
Give such facts a rule of their own. The invoice example sends an unstated
amount, due date or currency to review; otherwise a misread amount would be
cleared and the invoice approved.

## How a run works

1. **Read** the items and hash each one's text.
2. **Prefilter**: an item that fails ends `filtered`.
3. **Cache**: an item answered before under the same prompt, schema, model and
   server is served from `.factgate/cache.sqlite`.
4. **Ask the model**, `max_concurrent` at a time, until `max_calls` or
   `max_minutes` runs out. The rest end `deferred` and are answered on a later
   run. Three errors in a row stop the run and defer the rest.
5. **Check quotes**: each quote must appear in one field of the item (spaces,
   case, typographic quotes and dashes folded), or its fact is cleared.
6. **Resolvers** (optional): your Python functions correct the checked facts,
   and the quote check runs again on what they return.
7. **Decide**: the first rule whose conditions all hold, else the default.
8. **Write** `out/results.jsonl` (each item's status, outcome, rule, quotes and
   facts) and a Markdown summary.

An item whose answer does not fit the schema, or whose resolver raises, ends
`error` and the run goes on; `factgate run` then exits 1.

The quote check cannot catch three things. One is a wrong value that is really
in the document, such as "Previous balance £4,000.00 (paid)". A resolver can
catch that, as the invoice example's [`fixes.py`](examples/invoices/fixes.py)
does. Another is an enum, text or bool read wrongly from a real quote (GBP from
"Amount due EUR 3,800.00"), which the invoice example keeps as an open defect.
The last is a negation: "SC clearance is not required" contains "SC
clearance", and the tender cases check the model reads it as `false`.

## Evals

A case is an item plus what its facts and outcome must be, written with the
rule operators:

```json
{"id": "inv-earlier-balance", "fields": {"subject": "Invoice 3319 from Kestrel IT", "body": "... Previous balance £4,000.00 (paid, thank you). Amount now due: £860.00 by 12 December 2026."},
 "expect": {"facts": {"amount_due": {"eq": 860}}, "outcome": "approve"},
 "model_facts": {"amount_due": {"value": 4000, "quote": "Previous balance £4,000.00"}, "...": "..."},
 "tags": ["amounts", "resolver"], "as_of": "2026-11-07"}
```

`model_facts` is a model's answer, recorded or written by hand and often wrong
on purpose. Case files are JSONL or YAML and are merged on load; a bad case
fails to load with its `file:line`.

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

**Live, with repeats.** `--live --repeats 3` asks the model three times per
case and gives each case one state, not an average:

| State | Means |
|---|---|
| `pass` | every attempt passed |
| `fail` | no attempt passed |
| `flaky` | some passed and some failed |
| `inconclusive` | an attempt could not be graded: the backend raised or the answer did not fit the schema |

Inconclusive is never a pass: it is kept out of the pass rate. The report gives
pass@1 (passed attempts over conclusive ones), pass^k (cases that passed every
time) and a Wilson 95% interval, and lists dangerous mistakes first. A live
eval makes cases × repeats calls and is not held to `max_calls`, so count
before you run it.

- **Held-out cases**, passed with `--heldout`, are scored, but the report shows
  only their ids and states, so whoever tunes the prompt or rules does not see
  them. That holds by honour: the file is right there.
- **`factgate eval compare before.json after.json`** lists the cases that
  moved. Only fail to pass counts as fixed and only pass to fail as broken; a
  move into or out of `flaky` is noise.
- **`factgate eval regrade`** re-scores a run's saved answers with today's
  config and cases, on the run's own date, so a rule change is measured with no
  model calls.

[Writing cases](docs/writing-cases.md) covers what makes a good case, held-out
sets, known defects (`open_defect`) and cases that only test the code
(`offline_only`).

### Measured

Recorded on 7 November 2026 with `claude-haiku-5-5`, 3 runs of every case,
through `factgate eval run ... --live --repeats 3`. The results are in each
example's `results/` folder, and a test rebuilds the table and the totals under
it from those files.

| Example | Decides | Cases run live (visible + held-out) | Live result | Offline, in CI |
|---|---|---|---|---|
| [invoices](examples/invoices) | approve, review or hold | 32 (11 + 21) | pass@1 100%, pass^3 100%, Wilson 95% 89%-100% | 12 of 13 pass, 1 open defect |
| [support-tickets](examples/support-tickets) | which team, or page on-call | 14 (9 + 5) | pass@1 100%, pass^3 100%, Wilson 95% 78%-100% | 9 of 9 pass |
| [tenders](examples/tenders) | bid, no-bid or review | 12 (7 + 5) | pass@1 97%, pass^3 92%, Wilson 95% 65%-99% | 9 of 9 pass |

The 174 live calls took 69 seconds and 34,795 output tokens, at most USD 0.05.
The four offline-only cases did not run live.

The one flaky case shows why "not stated" needs rules. "Budget up to £18,000.
Responses due 30 December 2026." is a no-bid, being under £50,000. In one
attempt of three the model reported neither the value nor the deadline, so the
`no-deadline` rule sent it to review. Without the two "not stated" rules it
would have fallen through to `bid`, the config's dangerous mistake.

The 58 cases are short, synthetic and built around traps: a subtotal next to
the total, a paid balance, "Annoying!!!" that is not urgency, an injected
instruction, a clearance "not required". Scores on them do not promise the same
on your documents, and the intervals show how little 12 to 32 cases can say.

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

**Batch calls, at half the price.** With `model.batch: true` (anthropic only),
a run sends its items, and a live eval its attempts, as one Message Batch. The
run waits up to `max_minutes`, then cancels the batch and defers what it did not
answer (an eval marks it inconclusive), so batch the runs nobody is waiting on.
On the 58 example cases it gave the same results as direct calls
([details](docs/providers.md#batch-calls)).

[Providers](docs/providers.md) also covers the cache, and how to add a backend
or an item reader as a plugin.

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

## Roadmap

- **Not on PyPI yet.** Install from a clone, as above, or with
  `pip install "git+https://github.com/younis-y/deterministic-llm-pipeline#subdirectory=factgate"`.
- **A facts model written in Python**, instead of `facts:` in YAML: not planned
  until someone needs it.
- **rolescan on factgate.** rolescan is in daily use, so it keeps its own copy
  of the method for now.
  [The plan](../docs/superpowers/plans/2026-11-07-rolescan-onto-factgate.md)
  covers the move, its risks and how it would be tested.
- **Out of scope:** documents other than text. Converting a PDF to text is your
  step.

## Credits

The eval borrows from the harness built by the **Judge the Agent** team at The
Friendly Data Meetup London, 4 November 2026
([computator1200/parcel-pigeon-evals](https://github.com/computator1200/parcel-pigeon-evals),
on the challenge material from
[dl-lim/friendly-meetup-hackathon](https://github.com/dl-lim/friendly-meetup-hackathon)):
one state per case across repeats, inconclusive kept out of the pass rate,
held-out cases, compare by transitions, regrade of saved answers, dangerous
mistakes split from the rest, and a README checked against its result files.
Only the parts Younis Younis wrote there are adapted here as code (the repeat
runner's states and the held-out oracle); the rest are reused as ideas. That
harness graded a chatbot with an LLM judge; factgate grades with code.

## Licence

MIT.
