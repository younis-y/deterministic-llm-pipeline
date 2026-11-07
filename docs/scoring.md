# How scoring works

**1. Keywords.** Pure Python, no network, no credentials: weighted keyword
terms, tripled when they appear in the title, minus blocker terms and a
location penalty. Postings below `min_keyword_score` never reach the second
stage. You can stop here: `--no-llm` still gives a ranked, deduplicated,
seen-tracked digest.

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
posting. It is listed once, on the run that first sees it (an annual
programme can be listed again after a gap: `output.reopen_programme_days`).

Hardness is deliberately not a weight threshold. A weight is retuned whenever
you calibrate the prefilter; whether a clearance is a wall is a fact about you
that changes once a decade. One number could not encode both without one of
them silently changing the other.

**2. An LLM,** optionally. It scores fit against your profile summary and
flags hard eligibility bars. Its answers are cached (see the end of this
page), so re-running costs nothing for unchanged postings.

The backend is a plugin. `rolescan backends` lists what is registered:

| Name | Needs a key | What it is |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude API. Best quality. Needs ANTHROPIC_API_KEY. |
| `ollama` | no | Local model via Ollama. Free, offline, no key. See [Measured quality](../README.md#measured-quality). |

The tool does not rely on a model to catch structural bars: `hard_blockers` are
matched in Python before the model is called and force `blocked` whatever it
says, and `min_report_score` gates the rest. A bar that appears only in the
posting text, and that no configured term names, is the one a model can miss.

**Facts first, then rules.** The model is not
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
over the digest cap, or no description text yet), or, when `llm.max_minutes`
ran out, "model time budget of N minutes spent; M postings deferred to the next
run". A deferred posting is not
recorded as seen, so it comes round again. A posting with no text is held back
only `output.thin_unread_after` times (default 3): after that many runs without
text it is listed once under "Unread (no text after N runs)", with its link,
and recorded, so a source that never sends descriptions cannot hold it back for
ever. To bring back a posting a wrong rule or term hid, run `rolescan unsee
URL` (or its uid); the next scan reports it again. Four keys tune what gets
hidden. On `profile`: `nationalities` (the demonyms and countries you hold, so
a nationality bar that names one is not a bar for you; it acts through the
rules, and does nothing against a `hard_blockers` term), `title_only_blockers` (`blockers` terms such as
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

`llm.enricher` names a plugin registered under the `rolescan.enrichers`
entry-point group that does extra work only for postings that clear
`min_report_score`; a failing enricher keeps the plain verdict.
`llm.extra_prompt` is never sent to the scoring call (an enricher may use it),
and rolescan logs a note at startup when it is set with no enricher configured.

With `llm.enabled: false`, or with no key and no local model, you get stage one
alone. If a configured backend fails, the digest says so rather than quietly
serving keyword scores that look like a normal run. There is a test that
asserts exactly this.
