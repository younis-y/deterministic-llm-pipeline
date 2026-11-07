# Config reference

A config is one YAML file. Paths in it start from the folder it is in, and an
unknown key anywhere fails to load. The README's
[invoice example](../README.md#your-own-documents) shows a whole config; the
`model:` block is in [Providers](providers.md).

## Items

`items:` names exactly one reader: `jsonl: file`, `csv: file`, `folder: dir`
(each `.txt` or `.md` file is an item, its name the id), `stdin: true` (JSONL),
or `plugin: name` for a reader registered under the `factgate.sources`
entry-point group ([Writing a reader](providers.md#writing-a-reader)). `id`
names the id field (default `id`). `fields` names the text fields; each one is
checked for quotes on its own, so a quote cannot join the end of the subject to
the start of the body. `meta` names fields passed through to the output
untouched. Two items with one id are an error before anything runs.

## Prefilter

`prefilter:` drops items before any model call. It matches whole words in any
field, folded as the quote check folds. An item with any `exclude_any` term,
or with none of the `require_any` terms, ends `filtered`, and the output says
which.

## Facts

Fact types: `number`, `integer`, `text`, `bool`, `date` (ISO), `enum` (with
`values`), and `list` of any of them (with `of`). `describe` tells the model
what to look for. Every fact may be "not stated": the model says so with a null
value, and the quote check says so for any value it cannot find.

## The quote check

Each fact's quote must appear in one field of the item (spaces, case and
typographic quotes and dashes folded), or the fact is cleared. A number must
also appear in its own quote, whole and with its sign, so "total due
£12,540.00" read as 125400, or as 540, is cleared ("£40k" counts as 40,000). A
date's year must match the year its quote states, when it states one.

## Rules

Rules are tried in order. The first whose conditions all hold decides, and the
conditions in one rule are ANDed. When no rule holds, `default_outcome` decides.

| Operator | Applies to | True when |
|---|---|---|
| `is`, `eq`, `ne` | every type | equal / not equal |
| `lt`, `le`, `gt`, `ge` | number, integer, date | the comparison holds |
| `in` | every type but bool | the value is one of a list |
| `contains`, `matches` | text, and lists of text | a substring is there / a regex finds a match |
| `stated` | every type | `true`: the fact was found; `false`: it was not |
| `within_days: N` | date | between today and N days ahead |
| `before_days: N` | date | earlier than N days from today; `0` is "in the past" |

A condition on a fact that is not stated is false, except `stated: false`, so
a fact the quote check can clear needs a rule that catches it (the README's
"Make not stated safe"). On a list, an operator holds when any entry meets it,
`contains` means an entry equals the argument and `stated` means the list is
not empty. A config with a rule on an unknown fact, an unknown operator, an
operator or argument that does not fit the fact's type (`gt: "10,000"`,
`eq: GPB` for an enum of GBP and EUR) or a bad regex fails to load, naming the
rule.

## Resolvers

`resolvers:` lists `module:function` names, run in order on the checked facts,
and a module next to the config loads as well as an installed one. Each takes
the item and the checked facts and returns corrected facts, and the quote check
runs again on what it returns. A resolver that raises on one item makes that
item `error` and the run goes on. Resolvers run your Python, so only run a
config you trust, as with any script.

The invoice example ships one, [`fixes.py`](../examples/invoices/fixes.py),
that swaps an amount read from a subtotal, a VAT line or an earlier balance for
the email's own "total due" figure.

## Dangerous mistakes

`dangerous:` lists `{expected, got}` pairs, the mistakes that cost more than
others. The eval lists them first: see
[Writing cases](writing-cases.md#dangerous-mistakes).

## Cache

`cache:` is where answers are kept, by default `.factgate/cache.sqlite`, and
`null` turns it off. [The cache](providers.md#the-cache) says what its key
covers.
